"""
Scatter via symmetric memory: src rank publishes all chunks into its symmetric
buffer; non-src ranks pull their chunk directly via UVA peer pointer with a
custom CUDA kernel. Single device-side barrier, then a peer-load copy.
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

template <typename T>
__global__ void copy_kernel(const T* __restrict__ src, T* __restrict__ dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        dst[idx] = src[idx];
    }
}

__global__ void copy_vec16_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, int64_t n_vec) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n_vec; idx += stride) {
        dst[idx] = src[idx];
    }
}

void peer_copy(int64_t src_ptr, torch::Tensor dst, int64_t nbytes) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const void* sptr = reinterpret_cast<const void*>(static_cast<uintptr_t>(src_ptr));
    void* dptr = dst.data_ptr();

    if ((nbytes % 16 == 0) && ((uintptr_t)sptr % 16 == 0) && ((uintptr_t)dptr % 16 == 0)) {
        int64_t n_vec = nbytes / 16;
        int threads = 256;
        int blocks = (int)((n_vec + threads - 1) / threads);
        if (blocks > 1024) blocks = 1024;
        copy_vec16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(sptr),
            reinterpret_cast<uint4*>(dptr),
            n_vec);
    } else {
        int64_t n = nbytes;
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        if (blocks > 1024) blocks = 1024;
        copy_kernel<char><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const char*>(sptr),
            reinterpret_cast<char*>(dptr),
            n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("peer_copy", &peer_copy, "Peer copy via UVA pointer");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("scatter_symm_ext", CUDA_SRC)
    return _ext

_cache = {}

def _get_buf(total_numel: int, dtype: torch.dtype, device: torch.device):
    key = (total_numel, dtype, device.index)
    if key in _cache:
        return _cache[key]
    buf = symm_mem.empty(total_numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    assert dist.is_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == src:
        assert tensor.shape[0] == world_size
        chunk_shape = tensor.shape[1:]
        chunk_numel = 1
        for s in chunk_shape:
            chunk_numel *= s
    else:
        chunk_shape = tensor.shape
        chunk_numel = tensor.numel()

    total_numel = chunk_numel * world_size
    dtype = tensor.dtype
    device = tensor.device

    # Ensure extension compiled on all ranks
    _get_ext()

    buf, hdl = _get_buf(total_numel, dtype, device)

    if rank == src:
        # Publish all chunks into symmetric buffer
        buf.copy_(tensor.reshape(-1).contiguous())

    # Device-side barrier so non-src ranks see src's writes
    hdl.barrier(channel=0)

    out = torch.empty(chunk_shape, dtype=dtype, device=device)

    # Each rank reads its chunk from src's symmetric buffer
    src_base_ptr = int(hdl.buffer_ptrs[src])
    elem_size = tensor.element_size()
    chunk_offset_bytes = rank * chunk_numel * elem_size
    src_chunk_ptr = src_base_ptr + chunk_offset_bytes
    nbytes = chunk_numel * elem_size

    _get_ext().peer_copy(src_chunk_ptr, out, nbytes)

    hdl.barrier(channel=1)

    return out