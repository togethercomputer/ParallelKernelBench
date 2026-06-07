"""
Strategy:
1. **Algorithmic Reduction**: The reference globally broadcasts `local_neg_scores` (`P x K` elements), computing identical row-wise rankings redundantly across all GPUs. We replace this with local rank computation, shrinking network traffic by > `K`x.
2. **Device-Side Gather & UVA**: We use symmetric memory (`symm_mem`) and direct UVA load instructions for `gather_rankings`. This skips NCCL all-gather and avoids maintaining multiple large buffers.
3. **Compute-Comm Overlap**: While the CPU computes buffer offsets synchronously (a ~5us operation), the GPU pipelines the custom warp-level reduction kernel directly into the symmetric memory buffer.
4. **Fused Math**: We evaluate PyTorch's elementwise `sigmoid` and sort sequence seamlessly within the custom CUDA kernel using warp intrinsics, dropping multiple intermediate memory roundtrips.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Warp-level kernel: efficiently fuses sigmoid float casting and warp-stride local ranking counting
__global__ void compute_local_rankings_warp_kernel(
    const __nv_bfloat16* __restrict__ pos_scores,
    const __nv_bfloat16* __restrict__ neg_scores,
    int64_t* __restrict__ rankings,
    int P,
    int K
) {
    int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (i < P) {
        float pos_val = __bfloat162float(pos_scores[i]);
        float pos_sig = 1.0f / (1.0f + expf(-pos_val));
        float pos_sig_cmp = __bfloat162float(__float2bfloat16(pos_sig));

        int local_count = 0;
        for (int j = lane; j < K; j += 32) {
            float neg_val = __bfloat162float(neg_scores[i * K + j]);
            float neg_sig = 1.0f / (1.0f + expf(-neg_val));
            float neg_sig_cmp = __bfloat162float(__float2bfloat16(neg_sig));
            
            // Replicates stable descent sorting logic exactly
            if (neg_sig_cmp > pos_sig_cmp) {
                local_count++;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_count += __shfl_down_sync(0xffffffff, local_count, offset);
        }

        if (lane == 0) {
            rankings[i] = 1 + (int64_t)local_count;
        }
    }
}

__global__ void gather_sizes_kernel(
    const uint64_t* ptrs,
    int64_t* sizes_out,
    int world_size
) {
    int r = threadIdx.x;
    if (r < world_size) {
        const int64_t* peer_buf = reinterpret_cast<const int64_t*>(ptrs[r]);
        sizes_out[r] = peer_buf[0];
    }
}

__global__ void gather_rankings_kernel(
    const uint64_t* ptrs,
    const int64_t* sizes,
    const int64_t* offsets,
    int64_t* global_rankings,
    int world_size
) {
    int r = blockIdx.y;
    int64_t size = sizes[r];
    int64_t offset = offsets[r];
    const int64_t* peer_buf = reinterpret_cast<const int64_t*>(ptrs[r]);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
        global_rankings[offset + i] = peer_buf[i];
    }
}

void compute_local_rankings(
    torch::Tensor pos_scores,
    torch::Tensor neg_scores,
    torch::Tensor rankings,
    int P,
    int K
) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (P + warps_per_block - 1) / warps_per_block;
    if (blocks == 0) return;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    compute_local_rankings_warp_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(pos_scores.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(neg_scores.data_ptr<at::BFloat16>()),
        rankings.data_ptr<int64_t>(),
        P,
        K
    );
}

void gather_sizes(
    torch::Tensor ptrs,
    torch::Tensor sizes_out,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_sizes_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(ptrs.data_ptr<int64_t>()),
        sizes_out.data_ptr<int64_t>(),
        world_size
    );
}

void gather_rankings(
    torch::Tensor ptrs,
    torch::Tensor sizes,
    torch::Tensor offsets,
    torch::Tensor global_rankings,
    int world_size
) {
    int threads = 256;
    int blocks_x = 256; 
    dim3 blocks(blocks_x, world_size);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_rankings_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(ptrs.data_ptr<int64_t>()),
        sizes.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        global_rankings.data_ptr<int64_t>(),
        world_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_local_rankings", &compute_local_rankings, "Compute local rankings");
    m.def("gather_sizes", &gather_sizes, "Gather sizes via UVA");
    m.def("gather_rankings", &gather_rankings, "Gather rankings via UVA");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_ranking_opt", CUDA_SRC)
    return _ext


_meta_cache = {}
def get_meta_cache(device, group):
    group = group or dist.group.WORLD
    if group not in _meta_cache:
        meta_buf = symm_mem.empty(1, dtype=torch.int64, device=device)
        meta_hdl = symm_mem.rendezvous(meta_buf, group)
        ptrs_tensor = torch.tensor(meta_hdl.buffer_ptrs, dtype=torch.int64, device=device)
        world_size = dist.get_world_size(group)
        sizes_out = torch.empty(world_size, dtype=torch.int64, device=device)
        offsets_dev = torch.empty(world_size, dtype=torch.int64, device=device)
        _meta_cache[group] = (meta_buf, meta_hdl, ptrs_tensor, sizes_out, offsets_dev)
    return _meta_cache[group]


_comm_cache = {}
def get_comm_cache(min_capacity, device, group):
    group = group or dist.group.WORLD
    if group not in _comm_cache:
        _comm_cache[group] = {"capacity": 0, "buf": None, "hdl": None, "ptrs": None}
    
    cache = _comm_cache[group]
    if min_capacity > cache["capacity"]:
        new_cap = max(min_capacity * 2, 1024)
        buf = symm_mem.empty(new_cap, dtype=torch.int64, device=device)
        hdl = symm_mem.rendezvous(buf, group)
        ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
        cache["capacity"] = new_cap
        cache["buf"] = buf
        cache["hdl"] = hdl
        cache["ptrs"] = ptrs
        
    return cache["buf"], cache["hdl"], cache["ptrs"]


@torch.no_grad()
def solution(
    local_pos_scores: torch.Tensor,
    local_neg_scores: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = local_pos_scores.device

    P_r = local_pos_scores.shape[0]
    K = local_neg_scores.shape[1] if local_neg_scores.ndim > 1 else 0

    local_pos_scores = local_pos_scores.contiguous()
    local_neg_scores = local_neg_scores.contiguous()

    if world_size == 1:
        global_rankings = torch.empty(P_r, dtype=torch.int64, device=device)
        if P_r > 0:
            _get_ext().compute_local_rankings(
                local_pos_scores,
                local_neg_scores,
                global_rankings,
                P_r,
                K
            )
        return global_rankings

    ext = _get_ext()

    # 1. Share sizes via symmetric memory to avoid NCCL syncs
    meta_buf, meta_hdl, meta_ptrs, sizes_out, offsets_dev = get_meta_cache(device, group)
    
    meta_buf[0] = P_r
    meta_hdl.barrier(channel=0)
    ext.gather_sizes(meta_ptrs, sizes_out, world_size)
    
    # Implicitly syncs CPU strictly to calculate buffer sizes & total elements safely
    all_sizes = sizes_out.tolist()
    total_P = sum(all_sizes)
    max_P = max(all_sizes) if all_sizes else 0
    
    offsets = [0] * world_size
    for i in range(1, world_size):
        offsets[i] = offsets[i-1] + all_sizes[i-1]
        
    offsets_dev.copy_(torch.tensor(offsets, dtype=torch.int64, device=device), non_blocking=True)
    
    # 2. Extract shared symmetric rankings buffer with dynamic capacity checking
    comm_buf, comm_hdl, comm_ptrs = get_comm_cache(max_P, device, group)
    
    # 3. Queue local computations directly into peer-readable comm_buf 
    if P_r > 0:
        ext.compute_local_rankings(
            local_pos_scores,
            local_neg_scores,
            comm_buf,
            P_r,
            K
        )
        
    # 4. Enforce writes locally prior to peer retrieval
    comm_hdl.barrier(channel=0)
    
    # 5. Overlap final continuous gather output
    global_rankings = torch.empty(total_P, dtype=torch.int64, device=device)
    if total_P > 0:
        ext.gather_rankings(
            comm_ptrs,
            sizes_out,
            offsets_dev,
            global_rankings,
            world_size
        )
        
    return global_rankings