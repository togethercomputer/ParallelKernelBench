"""
Strategy:
1. **Device-side Peer Pulls (UVA)**: Allocate a symmetric memory buffer per rank to hold its input. A custom CUDA kernel uses vectorized reads (up to 128-bit) to pull directly from remote peers' symmetric buffers over NVLink into the local output tensor, sidestepping NCCL overhead entirely.
2. **Compute–Communication Overlap**: The local chunk copy (`tensor` -> `out[rank]`) is scheduled asynchronously on the stream before the inter-GPU synchronization (`hdl.barrier`), allowing it to overlap with peer pulls and barrier waits.
3. **Dynamic Alignment**: The pull kernel dynamically inspects pointer alignment to utilize 128-bit, 64-bit, or 32-bit memory instructions, ensuring max NVLink bandwidth regardless of the arbitrary input shape.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

__global__ void allgather_pull_kernel(
    const uint64_t* __restrict__ peer_ptrs,
    void* __restrict__ out,
    int64_t bytes_per_rank,
    int world_size,
    int my_rank
) {
    int rank_to_read = blockIdx.y; 
    if (rank_to_read == my_rank) return;
    
    const char* src = reinterpret_cast<const char*>(static_cast<uintptr_t>(peer_ptrs[rank_to_read]));
    char* dst = reinterpret_cast<char*>(out) + rank_to_read * bytes_per_rank;
    
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    // Dynamically fallback to smaller vectorized loads if shapes enforce unaligned pointers
    if (((uintptr_t)src % 16 == 0) && ((uintptr_t)dst % 16 == 0)) {
        int64_t numel_v = bytes_per_rank / 16;
        const uint4* src_v = reinterpret_cast<const uint4*>(src);
        uint4* dst_v = reinterpret_cast<uint4*>(dst);
        
        for (int64_t i = tid; i < numel_v; i += stride) {
            dst_v[i] = src_v[i];
        }
        
        int64_t rem_start = numel_v * 16;
        for (int64_t i = rem_start + tid; i < bytes_per_rank; i += stride) {
            dst[i] = src[i];
        }
    } else if (((uintptr_t)src % 8 == 0) && ((uintptr_t)dst % 8 == 0)) {
        int64_t numel_v = bytes_per_rank / 8;
        const uint2* src_v = reinterpret_cast<const uint2*>(src);
        uint2* dst_v = reinterpret_cast<uint2*>(dst);
        
        for (int64_t i = tid; i < numel_v; i += stride) {
            dst_v[i] = src_v[i];
        }
        
        int64_t rem_start = numel_v * 8;
        for (int64_t i = rem_start + tid; i < bytes_per_rank; i += stride) {
            dst[i] = src[i];
        }
    } else if (((uintptr_t)src % 4 == 0) && ((uintptr_t)dst % 4 == 0)) {
        int64_t numel_v = bytes_per_rank / 4;
        const uint32_t* src_v = reinterpret_cast<const uint32_t*>(src);
        uint32_t* dst_v = reinterpret_cast<uint32_t*>(dst);
        
        for (int64_t i = tid; i < numel_v; i += stride) {
            dst_v[i] = src_v[i];
        }
        
        int64_t rem_start = numel_v * 4;
        for (int64_t i = rem_start + tid; i < bytes_per_rank; i += stride) {
            dst[i] = src[i];
        }
    } else {
        // Safe scalar fallback
        for (int64_t i = tid; i < bytes_per_rank; i += stride) {
            dst[i] = src[i];
        }
    }
}

void launch_allgather(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor out,
    int64_t bytes_per_rank,
    int world_size,
    int my_rank
) {
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs_tensor.data_ptr<int64_t>());
    
    int threads = 256;
    int64_t numel_v = bytes_per_rank / 16;
    int blocks_x = std::min((int64_t)256, (numel_v + threads - 1) / threads);
    if (blocks_x <= 0) blocks_x = 1;
    
    // gridDim.y handles each peer rank's memory pull mapping smoothly
    dim3 blocks(blocks_x, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    allgather_pull_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, out.data_ptr(), bytes_per_rank, world_size, my_rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_allgather", &launch_allgather, "UVA Pull AllGather");
}
'''

_ext = None
_ext_compiled = False

def _get_ext():
    global _ext, _ext_compiled
    if not _ext_compiled:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            _ext = compile_cuda_extension("allgather_pull_ext", CUDA_SRC)
        if dist.is_initialized():
            dist.barrier()
        if rank != 0:
            _ext = compile_cuda_extension("allgather_pull_ext", CUDA_SRC)
        _ext_compiled = True
    return _ext

_symm_cache = {}

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    key = (n, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, ptrs_tensor)
    _symm_cache[key] = res
    return res

@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor.unsqueeze(0).clone()
        
    tensor = tensor.contiguous()
    n = tensor.numel()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    _get_ext()
    
    out_shape = (world_size,) + tensor.shape
    out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
    
    if n == 0:
        return out
        
    buf, hdl, ptrs_tensor = _get_symm_state(n, tensor.dtype, tensor.device)
    
    # Start Device-To-Device copy into the local symmetric buffer
    buf.copy_(tensor.view(-1))
    
    # Overlap local chunk's placement into output while coordinating peers
    out[rank].copy_(tensor)
    
    # Block local streams until all peers' symmetric buffers are fully visible
    hdl.barrier(channel=0)
    
    # Direct UVA pull of remote chunks via NVLink into the final allocation
    bytes_per_rank = n * tensor.element_size()
    _get_ext().launch_allgather(ptrs_tensor, out, bytes_per_rank, world_size, rank)
    
    return out