from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <pybind11/stl.h>

static inline int64_t tensor_nbytes(const torch::Tensor& t) {
    return t.numel() * t.element_size();
}

void stage_d2d(torch::Tensor src, torch::Tensor dst, int64_t nbytes) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA tensors");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");
    TORCH_CHECK(nbytes >= 0, "nbytes must be non-negative");
    TORCH_CHECK(tensor_nbytes(src) >= nbytes, "src too small");
    TORCH_CHECK(tensor_nbytes(dst) >= nbytes, "dst too small");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        static_cast<size_t>(nbytes),
        cudaMemcpyDeviceToDevice,
        stream));
}

void pack_meta_host(std::vector<int64_t> splits, torch::Tensor meta, int world_size) {
    TORCH_CHECK(meta.is_cuda(), "meta must be CUDA");
    TORCH_CHECK(meta.dtype() == torch::kInt32, "meta must be int32");
    TORCH_CHECK(meta.is_contiguous(), "meta must be contiguous");
    TORCH_CHECK((int)splits.size() == world_size, "splits length must equal world_size");
    TORCH_CHECK(meta.numel() >= 2 * world_size, "meta too small");

    std::vector<int32_t> h(2 * world_size);
    int64_t prefix = 0;
    for (int i = 0; i < world_size; ++i) {
        TORCH_CHECK(splits[i] >= 0 && splits[i] <= INT32_MAX, "split out of int32 range");
        h[i] = static_cast<int32_t>(splits[i]);
        h[world_size + i] = static_cast<int32_t>(prefix);
        prefix += splits[i];
        TORCH_CHECK(prefix <= INT32_MAX, "prefix out of int32 range");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        meta.data_ptr<int32_t>(),
        h.data(),
        static_cast<size_t>(2 * world_size * sizeof(int32_t)),
        cudaMemcpyHostToDevice,
        stream));
}

__device__ __forceinline__ bool aligned16(const void* p) {
    return ((reinterpret_cast<uintptr_t>(p) & 15ull) == 0ull);
}

__global__ void alltoall_gather_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ meta_ptrs,
    char* __restrict__ out,
    int64_t hidden_dim,
    int elem_size,
    int rank,
    int world_size
) {
    const int src_rank = blockIdx.y;

    const int32_t* __restrict__ src_meta =
        reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(meta_ptrs[src_rank]));

    const int rows = src_meta[rank];
    if (rows <= 0) {
        return;
    }

    const int in_row_offset = src_meta[world_size + rank];

    int out_row_offset = 0;
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r >= src_rank || r >= world_size) break;
        const int32_t* __restrict__ m =
            reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(meta_ptrs[r]));
        out_row_offset += m[rank];
    }

    const int64_t row_bytes = hidden_dim * static_cast<int64_t>(elem_size);
    const int64_t nbytes = static_cast<int64_t>(rows) * row_bytes;

    const char* __restrict__ src =
        reinterpret_cast<const char*>(static_cast<uintptr_t>(data_ptrs[src_rank])) +
        static_cast<int64_t>(in_row_offset) * row_bytes;
    char* __restrict__ dst =
        out + static_cast<int64_t>(out_row_offset) * row_bytes;

    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    if (aligned16(src) && aligned16(dst)) {
        const int64_t n16 = nbytes >> 4;
        const uint4* __restrict__ src4 = reinterpret_cast<const uint4*>(src);
        uint4* __restrict__ dst4 = reinterpret_cast<uint4*>(dst);

        for (int64_t i = tid; i < n16; i += stride) {
            dst4[i] = src4[i];
        }

        const int tail = static_cast<int>(nbytes & 15);
        if (tail && blockIdx.x == 0) {
            const int64_t base = n16 << 4;
            for (int t = threadIdx.x; t < tail; t += blockDim.x) {
                dst[base + t] = src[base + t];
            }
        }
    } else {
        for (int64_t i = tid; i < nbytes; i += stride) {
            dst[i] = src[i];
        }
    }
}

void launch_alltoall_gather(
    torch::Tensor data_ptrs,
    torch::Tensor meta_ptrs,
    torch::Tensor out,
    int64_t hidden_dim,
    int elem_size,
    int rank,
    int world_size,
    int64_t max_rows_per_peer
) {
    TORCH_CHECK(data_ptrs.is_cuda() && meta_ptrs.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(data_ptrs.dtype() == torch::kInt64, "data_ptrs must be int64");
    TORCH_CHECK(meta_ptrs.dtype() == torch::kInt64, "meta_ptrs must be int64");
    TORCH_CHECK(data_ptrs.is_contiguous() && meta_ptrs.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(data_ptrs.numel() >= world_size && meta_ptrs.numel() >= world_size, "ptr arrays too small");
    TORCH_CHECK(world_size > 0 && world_size <= 8, "optimized for world_size in [1, 8]");
    TORCH_CHECK(rank >= 0 && rank < world_size, "bad rank");
    TORCH_CHECK(hidden_dim >= 0 && elem_size > 0, "bad shape");

    const int threads = 256;
    const int64_t max_bytes = max_rows_per_peer * hidden_dim * static_cast<int64_t>(elem_size);
    int blocks_x = static_cast<int>((max_bytes + (int64_t)threads * 16 - 1) / ((int64_t)threads * 16));
    if (blocks_x < 1) blocks_x = 1;
    if (blocks_x > 65535) blocks_x = 65535;

    dim3 grid(blocks_x, world_size, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    alltoall_gather_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(meta_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<char*>(out.data_ptr()),
        hidden_dim,
        elem_size,
        rank,
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stage_d2d", &stage_d2d, "Stage contiguous tensor bytes into symmetric memory");
    m.def("pack_meta_host", &pack_meta_host, "Pack all_to_all split metadata into symmetric memory");
    m.def("launch_alltoall_gather", &launch_alltoall_gather, "UVA peer all_to_all gather");
}
'''


_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_alltoall_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


def _as_int_list(x, world_size: int):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.device.type == "cpu":
            vals = x.to(dtype=torch.int64).tolist()
        else:
            vals = x.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        vals = list(x)
    vals = [int(v) for v in vals]
    assert len(vals) == world_size
    return vals


def _equal_splits(total_rows: int, world_size: int):
    assert total_rows % world_size == 0
    q = total_rows // world_size
    return [q for _ in range(world_size)]


def _get_resources(numel: int, dtype: torch.dtype, device: torch.device, group, world_size: int):
    key = (id(group), device.index, str(device), dtype, int(numel), int(world_size))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    data_buf = symm_mem.empty((numel,), device=device, dtype=dtype)
    data_hdl = symm_mem.rendezvous(data_buf, group)

    meta_buf = symm_mem.empty((2 * world_size,), device=device, dtype=torch.int32)
    meta_hdl = symm_mem.rendezvous(meta_buf, group)

    data_ptrs = torch.tensor(data_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    meta_ptrs = torch.tensor(meta_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "data_buf": data_buf,
        "data_hdl": data_hdl,
        "meta_buf": meta_buf,
        "meta_hdl": meta_hdl,
        "data_ptrs": data_ptrs,
        "meta_ptrs": meta_ptrs,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    local_tensor: torch.Tensor,
    input_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    output_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return local_tensor.contiguous()

    assert local_tensor.is_cuda
    assert local_tensor.dim() == 2
    assert world_size <= 8

    ext = _get_ext()

    if not local_tensor.is_contiguous():
        local_tensor = local_tensor.contiguous()

    rank = dist.get_rank(group)
    local_rows = int(local_tensor.size(0))
    hidden_dim = int(local_tensor.size(1))
    elem_size = int(local_tensor.element_size())

    in_splits = _as_int_list(input_split_sizes, world_size)
    if in_splits is None:
        in_splits = _equal_splits(local_rows, world_size)

    out_splits = _as_int_list(output_split_sizes, world_size)
    if out_splits is None:
        out_rows = local_rows
        out_splits = _equal_splits(out_rows, world_size)
    else:
        out_rows = int(sum(out_splits))

    output = torch.empty(
        (out_rows, hidden_dim),
        dtype=local_tensor.dtype,
        device=local_tensor.device,
    )

    res = _get_resources(
        local_tensor.numel(),
        local_tensor.dtype,
        local_tensor.device,
        group,
        world_size,
    )

    data_buf = res["data_buf"]
    data_hdl = res["data_hdl"]
    meta_buf = res["meta_buf"]

    nbytes = int(local_tensor.numel() * elem_size)

    # Local staging + metadata publish.  The following symmetric-memory barrier
    # makes both visible to peer UVA loads before the device-side gather.
    ext.stage_d2d(local_tensor, data_buf, nbytes)
    ext.pack_meta_host(in_splits, meta_buf, world_size)
    data_hdl.barrier(channel=0)

    max_rows_per_peer = max(out_splits) if out_splits else 0
    ext.launch_alltoall_gather(
        res["data_ptrs"],
        res["meta_ptrs"],
        output,
        hidden_dim,
        elem_size,
        rank,
        world_size,
        int(max_rows_per_peer),
    )

    # Collective completion / safe symmetric-buffer reuse without NCCL.
    data_hdl.barrier(channel=1)
    return output