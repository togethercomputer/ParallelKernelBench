import itertools
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void gather_bf16_kernel(
    const nv_bfloat16* __restrict__ features,
    const int64_t* __restrict__ indices,
    nv_bfloat16* __restrict__ out,
    int num_rows,
    int H
) {
    // 2D Block: blockDim.x handles elements per row, blockDim.y handles rows
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;
    
    int64_t src_row = indices[row];
    const nv_bfloat16* src = features + src_row * H;
    nv_bfloat16* dst = out + row * H;
    
    if (H % 8 == 0) {
        int num_vecs = H / 8;
        int tid = threadIdx.x;
        const int4* src_vec = reinterpret_cast<const int4*>(src);
        int4* dst_vec = reinterpret_cast<int4*>(dst);
        for (int i = tid; i < num_vecs; i += blockDim.x) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        int tid = threadIdx.x;
        for (int i = tid; i < H; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

__global__ void uva_exchange_bf16_kernel(
    const int64_t* __restrict__ remote_meta_ptrs, // Size W
    const int64_t* __restrict__ remote_data_ptrs, // Size W
    const int32_t* __restrict__ local_start_rows, // Size W
    const int32_t* __restrict__ num_rows,         // Size W
    nv_bfloat16* __restrict__ out,
    int W,
    int rank,
    int H
) {
    // blockIdx.y assigns a set of blocks to a specific peer `p`
    int p = blockIdx.y;
    if (p >= W) return;
    
    int rows_to_copy = num_rows[p];
    if (rows_to_copy == 0) return;
    
    // Chunk index sent from `p` to `rank` in the original unshifted list
    int i_p = (rank - p + W) % W;
    
    const int32_t* remote_meta = reinterpret_cast<const int32_t*>(remote_meta_ptrs[p]);
    int remote_start_row = remote_meta[i_p];
    
    const nv_bfloat16* remote_data = reinterpret_cast<const nv_bfloat16*>(remote_data_ptrs[p]);
    const nv_bfloat16* src = remote_data + remote_start_row * H;
    
    nv_bfloat16* dst = out + local_start_rows[p] * H;
    
    int total_elements = rows_to_copy * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    if (H % 8 == 0) {
        int total_vecs = total_elements / 8;
        int vec_idx = idx;
        int vec_stride = stride;
        const int4* src_vec = reinterpret_cast<const int4*>(src);
        int4* dst_vec = reinterpret_cast<int4*>(dst);
        for (int i = vec_idx; i < total_vecs; i += vec_stride) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        for (int i = idx; i < total_elements; i += stride) {
            dst[i] = src[i];
        }
    }
}

void gather_bf16(
    torch::Tensor local_features,
    torch::Tensor seed_inverse_ids,
    torch::Tensor data_buf
) {
    int num_gather_rows = seed_inverse_ids.size(0);
    int H = local_features.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (num_gather_rows > 0) {
        dim3 block(32, 8);
        dim3 grid((num_gather_rows + 7) / 8);
        gather_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const nv_bfloat16*>(local_features.data_ptr<at::BFloat16>()),
            seed_inverse_ids.data_ptr<int64_t>(),
            reinterpret_cast<nv_bfloat16*>(data_buf.data_ptr<at::BFloat16>()),
            num_gather_rows,
            H
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void uva_exchange_bf16(
    torch::Tensor remote_meta_ptrs,
    torch::Tensor remote_data_ptrs,
    torch::Tensor local_start_rows,
    torch::Tensor num_rows,
    torch::Tensor out,
    int W,
    int rank,
    int H
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    // 256 threads per block, 256 blocks per peer chunk to easily saturate H100 with UVA traffic
    dim3 block(256);
    dim3 grid(256, W);
    
    uva_exchange_bf16_kernel<<<grid, block, 0, stream>>>(
        remote_meta_ptrs.data_ptr<int64_t>(),
        remote_data_ptrs.data_ptr<int64_t>(),
        local_start_rows.data_ptr<int32_t>(),
        num_rows.data_ptr<int32_t>(),
        reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        W, rank, H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_bf16", &gather_bf16, "Custom Gather BF16 features");
    m.def("uva_exchange_bf16", &uva_exchange_bf16, "UVA Exchange BF16 features");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_uva_exchange_ext", CUDA_SRC)
    return _ext


class SymmCache:
    def __init__(self):
        self.data_buf = None
        self.data_hdl = None
        self.data_capacity = 0
        
        self.meta_buf = None
        self.meta_hdl = None
        
        self.remote_meta_ptrs = None
        self.remote_data_ptrs = None


_cache = SymmCache()


@torch.no_grad()
def solution(
    local_features: torch.Tensor,
    seed_inverse_ids: torch.Tensor,
    counts_sent: List[int],
    counts_received: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    W = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = local_features.device
    H = local_features.size(1)
    
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    ext = _get_ext()
    
    # Check max allocation needed across all ranks and grow cached symm_mem safely
    local_req = sum(counts_received)
    if _cache.data_capacity < local_req:
        req_t = torch.tensor([local_req], dtype=torch.int64, device=device)
        dist.all_reduce(req_t, op=dist.ReduceOp.MAX, group=group)
        new_capacity = int(req_t.item() * 1.2)  # Amortize reallocations with 20% pad
        if new_capacity < 1024:
            new_capacity = 1024
        
        _cache.data_buf = symm_mem.empty((new_capacity, H), dtype=torch.bfloat16, device=device)
        _cache.data_hdl = symm_mem.rendezvous(_cache.data_buf, group)
        _cache.data_capacity = new_capacity
        
        if _cache.meta_buf is None:
            _cache.meta_buf = symm_mem.empty((W,), dtype=torch.int32, device=device)
            _cache.meta_hdl = symm_mem.rendezvous(_cache.meta_buf, group)
            _cache.remote_meta_ptrs = torch.tensor(_cache.meta_hdl.buffer_ptrs, dtype=torch.int64, device=device)
            
        _cache.remote_data_ptrs = torch.tensor(_cache.data_hdl.buffer_ptrs, dtype=torch.int64, device=device)
        
    # 1. Asynchronously gather required features locally onto symmetric data buffer
    ext.gather_bf16(local_features, seed_inverse_ids, _cache.data_buf)
    
    # 2. Concurrently compute metadata structures and chunk alignments on Host
    # Calculate starting row offsets for chunks we send
    meta_offsets = [0] * W
    curr = 0
    for i in range(W):
        meta_offsets[i] = curr
        curr += counts_received[i]
        
    # Calculate starting row offsets for chunks we receive 
    out_offsets = [0] * W
    curr_out = 0
    for i in range(W):
        out_offsets[i] = curr_out
        curr_out += counts_sent[i]
        
    local_start_rows = [0] * W
    peer_num_rows = [0] * W
    for p in range(W):
        i_r = (p - rank + W) % W
        local_start_rows[p] = out_offsets[i_r]
        peer_num_rows[p] = counts_sent[i_r]
        
    meta_t = torch.tensor(meta_offsets, dtype=torch.int32, device=device)
    local_start_t = torch.tensor(local_start_rows, dtype=torch.int32, device=device)
    num_rows_t = torch.tensor(peer_num_rows, dtype=torch.int32, device=device)
    
    # Write metadata into symmetric meta buffer (enables zero NCCL offsets exchange via UVA)
    _cache.meta_buf.copy_(meta_t, non_blocking=True)
    
    out = local_features.new_empty((sum(counts_sent), H))
    
    # 3. Synchronize memory globally; guarantee reads are valid via barrier on device stream
    _cache.data_hdl.barrier(channel=0)
    
    # 4. Exchange kernel dynamically pulls over NVLink exploiting unrolled pointers
    ext.uva_exchange_bf16(
        _cache.remote_meta_ptrs,
        _cache.remote_data_ptrs,
        local_start_t,
        num_rows_t,
        out,
        W, rank, H
    )
    
    return out