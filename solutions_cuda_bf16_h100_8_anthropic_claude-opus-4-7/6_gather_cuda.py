"""
Custom CUDA gather using symmetric memory: all ranks write their chunk into
a symmetric buffer; dst rank reads all peer chunks via UVA pointers in a single
kernel that stacks them into the output tensor.
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
__global__ void gather_stack_kernel(
    const long long* __restrict__ peer_ptrs,
    T* __restrict__ out,
    int world_size,
    int64_t chunk_numel
) {
    int r = blockIdx.y;
    if (r >= world_size) return;
    const T* src = (const T*)peer_ptrs[r];
    T* dst = out + (int64_t)r * chunk_numel;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < chunk_numel; idx += stride) {
        dst[idx] = src[idx];
    }
}

void launch_gather_stack(
    torch::Tensor peer_ptrs,  // int64 [world_size]
    torch::Tensor out,        // [world_size, *chunk_shape]
    int64_t chunk_numel,
    int world_size,
    int element_size
) {
    const long long* d_ptrs = (const long long*)peer_ptrs.data_ptr<int64_t>();

    int threads = 256;
    int blocks_x = (int)((chunk_numel + threads - 1) / threads);
    if (blocks_x > 1024) blocks_x = 1024;
    if (blocks_x < 1) blocks_x = 1;
    dim3 blocks(blocks_x, world_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (element_size == 2) {
        gather_stack_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, (uint16_t*)out.data_ptr(), world_size, chunk_numel);
    } else if (element_size == 4) {
        gather_stack_kernel<uint32_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, (uint32_t*)out.data_ptr(), world_size, chunk_numel);
    } else if (element_size == 8) {
        gather_stack_kernel<uint64_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, (uint64_t*)out.data_ptr(), world_size, chunk_numel);
    } else if (element_size == 1) {
        gather_stack_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, (uint8_t*)out.data_ptr(), world_size, chunk_numel);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_stack", &launch_gather_stack, "Gather + stack via UVA peer pointers");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_gather_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_resources(shape, dtype, device):
    key = (tuple(shape), dtype, device)
    if key in _cache:
        return _cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    _cache[key] = (buf, hdl, ptrs_tensor)
    return _cache[key]


@torch.no_grad()
def solution(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    assert dist.is_initialized()
    assert tensor.is_cuda

    inp = tensor.contiguous()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Compile on all ranks (cached)
    _get_ext()

    buf, hdl, ptrs_tensor = _get_resources(inp.shape, inp.dtype, inp.device)
    buf.copy_(inp)

    # Synchronize: ensure all ranks have written before dst reads
    hdl.barrier(channel=0)

    if rank == dst:
        out_shape = (world_size,) + tuple(inp.shape)
        out = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)
        chunk_numel = inp.numel()
        _get_ext().launch_gather_stack(
            ptrs_tensor, out, chunk_numel, world_size, inp.element_size()
        )
        # Ensure dst is done reading before any rank reuses buffer
        hdl.barrier(channel=1)
        return out
    else:
        hdl.barrier(channel=1)
        return tensor