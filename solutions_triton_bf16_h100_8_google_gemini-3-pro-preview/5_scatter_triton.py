"""
Strategy:
- **Device-Side Direct Push**: Instead of relying on host-driven NCCL collectives to orchestrate the scatter, we use symmetric memory (UVA) to expose all receiving ranks' destination buffers directly to the source rank.
- **Maximized P2P Bandwidth**: A custom CUDA kernel on the source rank pushes the tensor chunks to all remote peers concurrently in a single launch. The 2D grid assigns independent blocks to different remote ranks, inherently overlapping the outgoing NVLink transfers and fully saturating the source's egress bandwidth.
- **Minimal Memory Overhead**: We avoid allocating full-size staging buffers on receivers. Every rank allocates only its exact output chunk size in symmetric memory. The source reads straight from the contiguous input tensor and writes remotely without intermediate local staging.
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

__global__ void push_scatter_kernel(
    const char* __restrict__ src,
    const uintptr_t* __restrict__ dst_ptrs,
    size_t chunk_bytes
) {
    // blockIdx.y selects the destination rank
    int rank = blockIdx.y;
    char* dst = reinterpret_cast<char*>(dst_ptrs[rank]);
    
    // Each rank gets a consecutive slice of the source tensor
    const char* src_chunk = src + rank * chunk_bytes;

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    // Fast vectorised paths for aligned chunks
    if (((uintptr_t)src_chunk % 16 == 0) && ((uintptr_t)dst % 16 == 0) && (chunk_bytes % 16 == 0)) {
        size_t n = chunk_bytes / 16;
        const uint4* src_vec = reinterpret_cast<const uint4*>(src_chunk);
        uint4* dst_vec = reinterpret_cast<uint4*>(dst);
        for (size_t i = idx; i < n; i += stride) {
            dst_vec[i] = src_vec[i];
        }
    } else if (((uintptr_t)src_chunk % 8 == 0) && ((uintptr_t)dst % 8 == 0) && (chunk_bytes % 8 == 0)) {
        size_t n = chunk_bytes / 8;
        const uint2* src_vec = reinterpret_cast<const uint2*>(src_chunk);
        uint2* dst_vec = reinterpret_cast<uint2*>(dst);
        for (size_t i = idx; i < n; i += stride) {
            dst_vec[i] = src_vec[i];
        }
    } else if (((uintptr_t)src_chunk % 4 == 0) && ((uintptr_t)dst % 4 == 0) && (chunk_bytes % 4 == 0)) {
        size_t n = chunk_bytes / 4;
        const uint32_t* src_vec = reinterpret_cast<const uint32_t*>(src_chunk);
        uint32_t* dst_vec = reinterpret_cast<uint32_t*>(dst);
        for (size_t i = idx; i < n; i += stride) {
            dst_vec[i] = src_vec[i];
        }
    } else if (((uintptr_t)src_chunk % 2 == 0) && ((uintptr_t)dst % 2 == 0) && (chunk_bytes % 2 == 0)) {
        size_t n = chunk_bytes / 2;
        const uint16_t* src_vec = reinterpret_cast<const uint16_t*>(src_chunk);
        uint16_t* dst_vec = reinterpret_cast<uint16_t*>(dst);
        for (size_t i = idx; i < n; i += stride) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        // Fallback for unaligned or odd-sized chunks
        for (size_t i = idx; i < chunk_bytes; i += stride) {
            dst[i] = src_chunk[i];
        }
    }
}

void uva_push_scatter(
    torch::Tensor src_tensor,
    torch::Tensor dst_ptrs_tensor,
    int64_t chunk_bytes,
    int world_size
) {
    TORCH_CHECK(src_tensor.is_cuda(), "src must be CUDA");
    TORCH_CHECK(src_tensor.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst_ptrs_tensor.is_cuda(), "dst_ptrs must be CUDA");

    int threads = 256;
    int blocks_per_rank = 512; // Sufficient to saturate Hopper NVLink
    dim3 grid(blocks_per_rank, world_size);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const char* src = reinterpret_cast<const char*>(src_tensor.data_ptr());
    const uintptr_t* dst_ptrs = reinterpret_cast<const uintptr_t*>(dst_ptrs_tensor.data_ptr());

    push_scatter_kernel<<<grid, threads, 0, stream>>>(src, dst_ptrs, (size_t)chunk_bytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_push_scatter", &uva_push_scatter, "UVA push for direct scatter");
}
'''

_ext = None
_ext_initialized = False

def _get_ext():
    global _ext, _ext_initialized
    if not _ext_initialized:
        # Prevent race condition on extension compilation
        if dist.get_rank() == 0:
            _ext = compile_cuda_extension("uva_push_scatter_ext", CUDA_SRC)
        dist.barrier()
        if dist.get_rank() != 0:
            _ext = compile_cuda_extension("uva_push_scatter_ext", CUDA_SRC)
        _ext_initialized = True
    return _ext

_symm_cache = {}

def _get_symm_state(chunk_shape: tuple, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    cache_key = (chunk_shape, dtype, device)
    if cache_key in _symm_cache:
        return _symm_cache[cache_key]

    # Allocate symmetric memory exactly matching the chunk size per rank
    buf = symm_mem.empty(*chunk_shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    
    # Pre-compute an array of destination pointers for the source rank's kernel
    ptrs = [int(p) for p in hdl.buffer_ptrs]
    ptrs_tensor = torch.tensor(ptrs, dtype=torch.int64, device=device)
    
    _symm_cache[cache_key] = (buf, hdl, ptrs_tensor)
    return buf, hdl, ptrs_tensor

@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == src:
        assert tensor.shape[0] == world_size, f"Source tensor must have {world_size} chunks"
        chunk_shape = tuple(tensor.shape[1:])
    else:
        chunk_shape = tuple(tensor.shape)
        
    ext = _get_ext()
    buf, hdl, ptrs_tensor = _get_symm_state(chunk_shape, tensor.dtype, tensor.device)
    
    # Barrier 1: Ensure all ranks have initialized and stabilized their symmetric buffers
    hdl.barrier(channel=0)
    
    if rank == src:
        chunk_bytes = (tensor.numel() // world_size) * tensor.element_size()
        ext.uva_push_scatter(tensor, ptrs_tensor, chunk_bytes, world_size)
        
    # Barrier 2: Ensure source rank has finished pushing its chunks to all destination buffers
    hdl.barrier(channel=0)
    
    # Fast local asynchronous copy from the symmetric staging buffer to the final output
    out = torch.empty(chunk_shape, dtype=tensor.dtype, device=tensor.device)
    out.copy_(buf)
    
    return out