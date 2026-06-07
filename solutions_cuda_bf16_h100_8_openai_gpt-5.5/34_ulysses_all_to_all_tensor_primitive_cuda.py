from typing import Optional
import math

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
__global__ void alltoall_pull_scalar_kernel(
    const long long* __restrict__ ptrs,
    T* __restrict__ out,
    int world_size,
    int rank,
    int64_t chunk_numel,
    int64_t scatter_period,
    int64_t gather_period,
    int64_t total_numel
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < total_numel; idx += stride) {
        int64_t src = idx / chunk_numel;
        int64_t elem = idx - src * chunk_numel;

        // elem is linear in chunk_shape, where scatter_dim has size S/world.
        // Embed that chunk into source rank's full input at scatter segment = local rank.
        int64_t before_s = elem / scatter_period;
        int64_t rem_s = elem - before_s * scatter_period;
        int64_t in_off = before_s * scatter_period * (int64_t)world_size
                       + (int64_t)rank * scatter_period
                       + rem_s;

        // Write chunk from source rank into output gather segment = src.
        int64_t before_g = elem / gather_period;
        int64_t rem_g = elem - before_g * gather_period;
        int64_t out_off = before_g * gather_period * (int64_t)world_size
                        + src * gather_period
                        + rem_g;

        const T* __restrict__ src_ptr = reinterpret_cast<const T*>((uintptr_t)ptrs[src]);
        out[out_off] = src_ptr[in_off];
    }
}

__global__ void alltoall_pull_bf16_vec8_kernel(
    const long long* __restrict__ ptrs,
    uint4* __restrict__ out,
    int world_size,
    int rank,
    int64_t chunk_vecs,
    int64_t scatter_period_vecs,
    int64_t gather_period_vecs,
    int64_t total_vecs
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < total_vecs; idx += stride) {
        int64_t src = idx / chunk_vecs;
        int64_t elem = idx - src * chunk_vecs;

        int64_t before_s = elem / scatter_period_vecs;
        int64_t rem_s = elem - before_s * scatter_period_vecs;
        int64_t in_off = before_s * scatter_period_vecs * (int64_t)world_size
                       + (int64_t)rank * scatter_period_vecs
                       + rem_s;

        int64_t before_g = elem / gather_period_vecs;
        int64_t rem_g = elem - before_g * gather_period_vecs;
        int64_t out_off = before_g * gather_period_vecs * (int64_t)world_size
                        + src * gather_period_vecs
                        + rem_g;

        const uint4* __restrict__ src_ptr =
            reinterpret_cast<const uint4*>((uintptr_t)ptrs[src]);
        out[out_off] = src_ptr[in_off];
    }
}

void launch_alltoall_pull(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int world_size,
    int rank,
    int64_t chunk_numel,
    int64_t scatter_period,
    int64_t gather_period,
    int elem_size,
    bool use_vec8
) {
    TORCH_CHECK(ptrs_tensor.is_cuda(), "ptrs_tensor must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(ptrs_tensor.dtype() == torch::kInt64, "ptrs_tensor must be int64");

    const long long* ptrs =
        reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>());

    int64_t total_numel = chunk_numel * (int64_t)world_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int threads = 256;

    if (use_vec8) {
        int64_t total_vecs = total_numel / 8;
        int64_t chunk_vecs = chunk_numel / 8;
        int64_t scatter_period_vecs = scatter_period / 8;
        int64_t gather_period_vecs = gather_period / 8;
        int blocks = (int)((total_vecs + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        if (blocks > 65535) blocks = 65535;

        alltoall_pull_bf16_vec8_kernel<<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<uint4*>(out.data_ptr()),
            world_size,
            rank,
            chunk_vecs,
            scatter_period_vecs,
            gather_period_vecs,
            total_vecs
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    int blocks = (int)((total_numel + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    if (elem_size == 1) {
        alltoall_pull_scalar_kernel<unsigned char><<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<unsigned char*>(out.data_ptr()), world_size, rank,
            chunk_numel, scatter_period, gather_period, total_numel);
    } else if (elem_size == 2) {
        alltoall_pull_scalar_kernel<unsigned short><<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<unsigned short*>(out.data_ptr()), world_size, rank,
            chunk_numel, scatter_period, gather_period, total_numel);
    } else if (elem_size == 4) {
        alltoall_pull_scalar_kernel<unsigned int><<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<unsigned int*>(out.data_ptr()), world_size, rank,
            chunk_numel, scatter_period, gather_period, total_numel);
    } else if (elem_size == 8) {
        alltoall_pull_scalar_kernel<unsigned long long><<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<unsigned long long*>(out.data_ptr()), world_size, rank,
            chunk_numel, scatter_period, gather_period, total_numel);
    } else {
        TORCH_CHECK(false, "unsupported element size");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_alltoall_pull", &launch_alltoall_pull,
          "Symmetric-memory UVA all-to-all tensor pull/cat kernel");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_symm_alltoall_pull_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _prod(xs):
    v = 1
    for x in xs:
        v *= int(x)
    return int(v)


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError("dimension out of range")
    return dim


def _group_key(group):
    # ProcessGroup objects are stable for the lifetime of the benchmark.
    return id(group)


def _get_resources(x_shape, out_shape, dtype, device, group, scatter_dim, gather_dim):
    key = (
        tuple(int(s) for s in x_shape),
        tuple(int(s) for s in out_shape),
        dtype,
        int(device.index if device.index is not None else torch.cuda.current_device()),
        _group_key(group),
        int(scatter_dim),
        int(gather_dim),
    )
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty(tuple(int(s) for s in x_shape), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)

    out = torch.empty(tuple(int(s) for s in out_shape), device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (buf, hdl, out, ptrs_tensor)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return x.contiguous()

    assert x.is_cuda, "x must be a CUDA tensor"
    assert dist.is_initialized(), "torch.distributed must be initialized"

    x_contig = x.contiguous()
    ndim = x_contig.dim()
    scatter_dim = _normalize_dim(scatter_dim, ndim)
    gather_dim = _normalize_dim(gather_dim, ndim)

    shape = [int(s) for s in x_contig.shape]
    assert shape[scatter_dim] % world_size == 0, "scatter_dim must be divisible by world_size"

    scatter_chunk = shape[scatter_dim] // world_size

    chunk_shape = list(shape)
    chunk_shape[scatter_dim] = scatter_chunk

    out_shape = list(chunk_shape)
    out_shape[gather_dim] *= world_size

    # Linear periods inside one received chunk.
    # scatter_period = scatter_chunk * prod(dims after scatter_dim in full/chunk layout)
    # gather_period  = chunk_shape[gather_dim] * prod(dims after gather_dim in chunk layout)
    scatter_period = scatter_chunk * _prod(shape[scatter_dim + 1:])
    gather_period = chunk_shape[gather_dim] * _prod(chunk_shape[gather_dim + 1:])

    chunk_numel = x_contig.numel() // world_size
    rank = dist.get_rank(group)

    buf, hdl, out, ptrs_tensor = _get_resources(
        tuple(shape),
        tuple(out_shape),
        x_contig.dtype,
        x_contig.device,
        group,
        scatter_dim,
        gather_dim,
    )

    # Publish this rank's full contiguous input in symmetric memory.
    buf.copy_(x_contig)

    # Make peer writes visible before any rank starts UVA pulls.
    hdl.barrier(channel=0)

    elem_size = x_contig.element_size()

    # BF16/FP16 raw 16-byte vector path: 8 x 2-byte elements per transaction.
    # Requires periods and chunk size aligned so vectors never cross logical row boundaries.
    use_vec8 = (
        elem_size == 2
        and chunk_numel % 8 == 0
        and scatter_period % 8 == 0
        and gather_period % 8 == 0
    )

    _get_ext().launch_alltoall_pull(
        ptrs_tensor,
        out,
        int(world_size),
        int(rank),
        int(chunk_numel),
        int(scatter_period),
        int(gather_period),
        int(elem_size),
        bool(use_vec8),
    )

    # Prevent symmetric input buffer reuse until every rank has completed peer reads.
    hdl.barrier(channel=1)

    return out