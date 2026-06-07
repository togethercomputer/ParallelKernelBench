"""
Ulysses all_gather_into_tensor via symmetric memory + custom CUDA kernel.
Each rank writes its shard into a symmetric buffer; a CUDA kernel reads
peer shards directly via UVA peer pointers and stitches the gathered tensor.
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
#include <cstdint>

// Vectorized copy: each thread copies 16 bytes (uint4)
__global__ void gather_peers_kernel(
    const long long* __restrict__ peer_ptrs,  // [world_size]
    char* __restrict__ out,                    // gathered output
    int64_t shard_bytes,
    int world_size
) {
    int rank = blockIdx.y;
    if (rank >= world_size) return;

    const char* src = (const char*)peer_ptrs[rank];
    char* dst = out + (int64_t)rank * shard_bytes;

    int64_t n16 = shard_bytes / 16;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const uint4* src4 = (const uint4*)src;
    uint4* dst4 = (uint4*)dst;

    for (int64_t i = tid; i < n16; i += stride) {
        dst4[i] = src4[i];
    }

    // Tail bytes
    int64_t tail_start = n16 * 16;
    int64_t tail = shard_bytes - tail_start;
    if (tail > 0 && blockIdx.x == 0) {
        for (int64_t i = threadIdx.x; i < tail; i += blockDim.x) {
            dst[tail_start + i] = src[tail_start + i];
        }
    }
}

void launch_gather_peers(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor out,
    int64_t shard_bytes,
    int world_size
) {
    const long long* d_ptrs = (const long long*)peer_ptrs_tensor.data_ptr<int64_t>();
    char* d_out = (char*)out.data_ptr();

    int threads = 256;
    int64_t n16 = shard_bytes / 16;
    int blocks_x = (int)((n16 + threads - 1) / threads);
    if (blocks_x < 1) blocks_x = 1;
    if (blocks_x > 512) blocks_x = 512;

    dim3 grid(blocks_x, world_size, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_peers_kernel<<<grid, threads, 0, stream>>>(
        d_ptrs, d_out, shard_bytes, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_peers", &launch_gather_peers, "Gather peer shards via UVA");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_allgather_uva_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_resources(shard_shape, dtype, device, group):
    key = (tuple(shard_shape), dtype, device, id(group))
    if key in _cache:
        return _cache[key]

    buf = symm_mem.empty(shard_shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)

    _cache[key] = (buf, hdl, ptrs_tensor)
    return _cache[key]


# Warmup the extension once
_ext_warmed = False


def solution(
    x: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    x = x.contiguous()
    global _ext_warmed
    if not _ext_warmed:
        _get_ext()
        _ext_warmed = True

    buf, hdl, ptrs_tensor = _get_resources(tuple(x.shape), x.dtype, x.device, group)

    # Stage local shard into symmetric buffer
    buf.copy_(x)

    # Synchronize so all peers' writes to their symmetric buffers are visible
    hdl.barrier(channel=0)

    # Allocate output
    dim_size = list(x.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=x.dtype, device=x.device)

    shard_bytes = x.numel() * x.element_size()
    _get_ext().launch_gather_peers(ptrs_tensor, output, shard_bytes, world_size)

    # Ensure no peer overwrites its buffer until all reads complete
    hdl.barrier(channel=1)

    return output