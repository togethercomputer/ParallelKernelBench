"""
Broadcast via symmetric memory: source rank writes into symm buffer, peers
read the source's UVA pointer directly via a custom CUDA kernel using
vectorized 16-byte loads. No NCCL on the hot path.
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

__global__ void broadcast_copy_kernel(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    int64_t n_vec,
    const char* __restrict__ src_tail,
    char* __restrict__ dst_tail,
    int64_t tail_bytes
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n_vec; i += stride) {
        dst[i] = src[i];
    }
    if (blockIdx.x == 0 && threadIdx.x < tail_bytes) {
        dst_tail[threadIdx.x] = src_tail[threadIdx.x];
    }
}

void launch_broadcast_copy(
    int64_t src_ptr,
    torch::Tensor dst,
    int64_t total_bytes
) {
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");

    int64_t n_vec = total_bytes / 16;
    int64_t tail_bytes = total_bytes - n_vec * 16;

    const uint4* src_v = reinterpret_cast<const uint4*>(static_cast<uintptr_t>(src_ptr));
    uint4* dst_v = reinterpret_cast<uint4*>(dst.data_ptr());
    const char* src_tail = reinterpret_cast<const char*>(static_cast<uintptr_t>(src_ptr) + n_vec * 16);
    char* dst_tail = reinterpret_cast<char*>(dst.data_ptr()) + n_vec * 16;

    int threads = 256;
    int blocks = (int)((n_vec + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 2048) blocks = 2048;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    broadcast_copy_kernel<<<blocks, threads, 0, stream>>>(
        src_v, dst_v, n_vec, src_tail, dst_tail, tail_bytes
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_broadcast_copy", &launch_broadcast_copy,
          "Vectorized device-side broadcast copy from peer UVA ptr");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_broadcast_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_symm(nbytes: int, device: torch.device):
    key = (nbytes, device)
    if key in _cache:
        return _cache[key]
    buf = symm_mem.empty(nbytes, device=device, dtype=torch.uint8)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    assert dist.is_initialized()
    assert tensor.is_cuda and tensor.is_contiguous()

    # Warm compile uniformly
    _get_ext()

    rank = dist.get_rank()
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes == 0:
        return tensor.clone()

    buf, hdl = _get_symm(nbytes, tensor.device)

    # Source writes its tensor bytes into the symmetric buffer
    if rank == src:
        buf.copy_(tensor.view(torch.uint8).reshape(-1))

    # Ensure src write is visible to all peers
    hdl.barrier(channel=0)

    out = torch.empty_like(tensor)
    src_ptr = int(hdl.buffer_ptrs[src])
    _get_ext().launch_broadcast_copy(src_ptr, out.view(torch.uint8).reshape(-1), nbytes)

    # Make sure all peers finish reading before next call mutates buf
    hdl.barrier(channel=1)

    return out