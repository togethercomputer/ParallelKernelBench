"""
Optimized GraphBolt feature exchange using symmetric memory and custom CUDA P2P.

Strategy:
- Persistent Symmetric Memory: We maintain persistent symmetric memory buffers for metadata 
  (`meta_buf`) and the output features (`out_buf`). Sizes are dynamically managed, communicating 
  reallocation requests via device-side `meta_buf` flags to minimize CPU/NCCL overhead.
- Fused Gather & Push: Instead of gathering features locally into an intermediate buffer and 
  running an all-to-all collective, a custom CUDA kernel directly gathers rows from `local_features` 
  using `seed_inverse_ids` and pushes them to the correct remote `out_buf`s via NVLink P2P stores.
- Compute-Comm Overlap: P2P stores are issued concurrently with index calculations and memory loads 
  from local HBM, fully utilizing memory/NVLink parallelism without blocking on bulk collectives.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void prefetch_and_check_kernel(
    const int64_t* const* meta_ptrs,
    int n,
    int rank,
    int64_t* local_dst_offsets,
    int* global_needs_realloc
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int needs = 0;
        for (int r = 0; r < n; ++r) {
            if (meta_ptrs[r][n] == 1) {
                needs = 1;
            }
        }
        *global_needs_realloc = needs;
        
        // Prefetch base offsets for chunks we will push to remote peers
        for (int j = 0; j < n; ++j) {
            int dst = (j + rank) % n;
            int k = (n - dst + rank) % n;
            local_dst_offsets[j] = meta_ptrs[dst][k];
        }
    }
}

void launch_prefetch_and_check(
    torch::Tensor meta_ptrs,
    int n,
    int rank,
    torch::Tensor local_dst_offsets,
    torch::Tensor global_needs_realloc
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    prefetch_and_check_kernel<<<1, 1, 0, stream>>>(
        (const int64_t* const*)meta_ptrs.data_ptr<int64_t>(),
        n, rank,
        local_dst_offsets.data_ptr<int64_t>(),
        global_needs_realloc.data_ptr<int32_t>()
    );
}

template <typename T>
__global__ void fused_gather_push_kernel(
    const T* __restrict__ local_features,
    const int64_t* __restrict__ seed_inverse_ids,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ local_dst_offsets,
    T* const* out_ptrs,
    int rank,
    int n,
    int64_t H_vec,
    int64_t total_elements_vec
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < total_elements_vec; i += gridDim.x * blockDim.x) {
        int64_t row = i / H_vec;
        int64_t col = i % H_vec;

        int j = 0;
        #pragma unroll
        for (int step = 0; step < 8; ++step) {
            if (step < n && row >= src_offsets[step+1]) {
                j = step + 1;
            }
        }

        int dst = (j + rank) % n;
        int64_t offset_in_chunk = row - src_offsets[j];
        int64_t dst_base = local_dst_offsets[j];
        int64_t out_row = dst_base + offset_in_chunk;

        int64_t src_row = seed_inverse_ids[row];
        T val = local_features[src_row * H_vec + col];
        
        // P2P direct write to the destination rank's symmetric buffer
        out_ptrs[dst][out_row * H_vec + col] = val;
    }
}

void launch_fused_gather_push(
    torch::Tensor local_features,
    torch::Tensor seed_inverse_ids,
    torch::Tensor src_offsets,
    torch::Tensor local_dst_offsets,
    torch::Tensor out_ptrs,
    int rank,
    int n
) {
    int64_t N_send = seed_inverse_ids.size(0);
    int64_t H = local_features.size(1);
    
    if (N_send == 0) return;
    
    int64_t element_size = local_features.element_size();
    int64_t H_bytes = H * element_size;
    TORCH_CHECK(H_bytes % 2 == 0, "Feature row size in bytes must be multiple of 2");
    int64_t H_units = H_bytes / 2;
    
    // Choose optimal vectorized load/store alignment based on inner dimension
    int vec_size = 1;
    if (H_units % 8 == 0) vec_size = 8;
    else if (H_units % 4 == 0) vec_size = 4;
    else if (H_units % 2 == 0) vec_size = 2;
    
    int64_t H_vec = H_units / vec_size;
    int64_t total_elements_vec = N_send * H_vec;
    
    int threads = 256;
    int blocks = (total_elements_vec + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const int64_t* seed_inverse_ids_ptr = seed_inverse_ids.data_ptr<int64_t>();
    const int64_t* src_offsets_ptr = src_offsets.data_ptr<int64_t>();
    const int64_t* local_dst_offsets_ptr = local_dst_offsets.data_ptr<int64_t>();
    const void* out_ptrs_ptr = out_ptrs.data_ptr<int64_t>();
    const void* local_features_ptr = local_features.data_ptr();
    
    if (vec_size == 8) {
        fused_gather_push_kernel<uint4><<<blocks, threads, 0, stream>>>(
            (const uint4*)local_features_ptr,
            seed_inverse_ids_ptr,
            src_offsets_ptr,
            local_dst_offsets_ptr,
            (uint4* const*)out_ptrs_ptr,
            rank, n, H_vec, total_elements_vec
        );
    } else if (vec_size == 4) {
        fused_gather_push_kernel<uint2><<<blocks, threads, 0, stream>>>(
            (const uint2*)local_features_ptr,
            seed_inverse_ids_ptr,
            src_offsets_ptr,
            local_dst_offsets_ptr,
            (uint2* const*)out_ptrs_ptr,
            rank, n, H_vec, total_elements_vec
        );
    } else if (vec_size == 2) {
        fused_gather_push_kernel<uint32_t><<<blocks, threads, 0, stream>>>(
            (const uint32_t*)local_features_ptr,
            seed_inverse_ids_ptr,
            src_offsets_ptr,
            local_dst_offsets_ptr,
            (uint32_t* const*)out_ptrs_ptr,
            rank, n, H_vec, total_elements_vec
        );
    } else {
        fused_gather_push_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
            (const uint16_t*)local_features_ptr,
            seed_inverse_ids_ptr,
            src_offsets_ptr,
            local_dst_offsets_ptr,
            (uint16_t* const*)out_ptrs_ptr,
            rank, n, H_vec, total_elements_vec
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_prefetch_and_check", &launch_prefetch_and_check, "Prefetch metadata and check reallocation");
    m.def("launch_fused_gather_push", &launch_fused_gather_push, "Fused gather and push over symmetric memory");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_feat_exchange_ext", CUDA_SRC)
    return _ext

class SymmMemState:
    def __init__(self, n: int, device: torch.device):
        self.n = n
        self.device = device
        self.meta_buf = symm_mem.empty(n + 1, dtype=torch.int64, device=device)
        self.meta_hdl = symm_mem.rendezvous(self.meta_buf)
        self.meta_ptrs = torch.tensor(self.meta_hdl.buffer_ptrs, dtype=torch.int64, device=device)
        self.out_buf = None
        self.out_hdl = None
        self.out_ptrs = None
        self.my_max_rows = 0

_state = None

@torch.no_grad()
def solution(
    local_features: torch.Tensor,
    seed_inverse_ids: torch.Tensor,
    counts_sent: List[int],
    counts_received: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    n = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = local_features.device
    
    global _state
    if _state is None:
        _state = SymmMemState(n, device)
        _get_ext()
        
    my_rows = sum(counts_sent)
    my_needs_realloc = 1 if (my_rows > _state.my_max_rows or _state.out_buf is None) else 0
    
    dst_offsets = [0] * n
    curr = 0
    for i in range(n):
        dst_offsets[i] = curr
        curr += counts_sent[i]
        
    meta_local = dst_offsets + [my_needs_realloc]
    meta_cpu = torch.tensor(meta_local, dtype=torch.int64, pin_memory=True)
    _state.meta_buf.copy_(meta_cpu, non_blocking=True)
    
    # Wait for all ranks to expose their destination offsets and dynamic allocation needs
    _state.meta_hdl.barrier(channel=0)
    
    global_needs_realloc_dev = torch.empty(1, dtype=torch.int32, device=device)
    local_dst_offsets = torch.empty(n, dtype=torch.int64, device=device)
    
    _get_ext().launch_prefetch_and_check(
        _state.meta_ptrs, n, rank, local_dst_offsets, global_needs_realloc_dev
    )
    
    # Conditionally expand symmetric output buffer capacity without extra NCCL blocking logic
    if global_needs_realloc_dev.item() == 1:
        new_max = max(my_rows, int(_state.my_max_rows * 1.5))
        if new_max < 1024:
            new_max = 1024
        if _state.out_buf is None or new_max > _state.my_max_rows:
            _state.out_buf = symm_mem.empty((new_max, local_features.size(1)), dtype=local_features.dtype, device=device)
            _state.my_max_rows = new_max
            
        _state.out_hdl = symm_mem.rendezvous(_state.out_buf, group=group)
        _state.out_ptrs = torch.tensor(_state.out_hdl.buffer_ptrs, dtype=torch.int64, device=device)
        
    src_offsets_list = [0] * (n + 1)
    curr = 0
    for i in range(n):
        src_offsets_list[i] = curr
        curr += counts_received[i]
    src_offsets_list[n] = curr
    src_offsets = torch.tensor(src_offsets_list, dtype=torch.int64, device=device)
    
    _get_ext().launch_fused_gather_push(
        local_features,
        seed_inverse_ids.contiguous(),
        src_offsets,
        local_dst_offsets,
        _state.out_ptrs,
        rank,
        n
    )
    
    # Wait for all remote pushes to our local symmetric out_buf to complete
    _state.out_hdl.barrier(channel=0)
    
    # Return cloned sub-tensor matching GraphBolt output/mutability semantics
    return _state.out_buf[:my_rows, :].clone()