"""
Device-side GraphStorm link-prediction ranking.

Strategy:
- Replace all_reduce + all_to_all broadcast with symmetric-memory rendezvous buffers.
- Each rank writes only its local positive/negative scores once; all ranks rank directly from peer UVA pointers.
- Fuse sigmoid + ranking into one CUDA kernel: O(P*K) count instead of sigmoid + sort O(P*K log K).
- Keep BF16 behavior by comparing BF16-rounded sigmoid values.
"""

from typing import Optional, Dict, Tuple, Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_I64(x) TORCH_CHECK((x).dtype() == torch::kInt64, #x " must be int64")

// dtype_enum: 0=bf16, 1=float32, 2=float16

__device__ __forceinline__ float sigmoid_round_for_dtype(float x, int dtype_enum) {
    float y = 1.0f / (1.0f + expf(-x));
    if (dtype_enum == 0) {
        return __bfloat162float(__float2bfloat16(y));
    } else if (dtype_enum == 2) {
        return __half2float(__float2half(y));
    }
    return y;
}

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_f32(scalar_t x) {
    return static_cast<float>(x);
}

__global__ void fill_meta_kernel(
    int64_t* __restrict__ meta,
    int64_t p,
    int64_t k
) {
    if (threadIdx.x == 0) {
        meta[0] = p;
        meta[1] = k;
    }
}

__global__ void compute_meta_kernel(
    const int64_t* __restrict__ meta_ptrs,
    int64_t* __restrict__ sizes_offsets,
    int64_t* __restrict__ summary,
    int world_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t sum_p = 0;
        int64_t max_p = 0;
        int64_t k_ref = -1;

        for (int r = 0; r < world_size; ++r) {
            const int64_t* m = reinterpret_cast<const int64_t*>(
                static_cast<uintptr_t>(meta_ptrs[r])
            );
            int64_t p = m[0];
            int64_t k = m[1];

            sizes_offsets[r] = p;
            sizes_offsets[world_size + r] = sum_p;

            sum_p += p;
            if (p > max_p) max_p = p;
            if (k_ref < 0) k_ref = k;
        }

        sizes_offsets[2 * world_size] = sum_p;
        summary[0] = sum_p;
        summary[1] = max_p;
        summary[2] = k_ref < 0 ? 0 : k_ref;
    }
}

template <typename scalar_t>
__global__ void pack_scores_kernel(
    const scalar_t* __restrict__ pos,
    const scalar_t* __restrict__ neg,
    scalar_t* __restrict__ dst,
    int64_t p,
    int64_t k
) {
    int64_t cols = k + 1;
    int64_t n = p * cols;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t row = idx / cols;
        int64_t col = idx - row * cols;

        if (col == 0) {
            dst[idx] = pos[row];
        } else {
            dst[idx] = neg[row * k + (col - 1)];
        }
    }
}

template <typename scalar_t>
__global__ void local_rank_kernel(
    const scalar_t* __restrict__ pos,
    const scalar_t* __restrict__ neg,
    int64_t* __restrict__ out,
    int64_t p,
    int64_t k,
    int dtype_enum
) {
    int64_t row = (int64_t)blockIdx.x;
    if (row >= p) return;

    int tid = threadIdx.x;
    float ps = sigmoid_round_for_dtype(scalar_to_f32(pos[row]), dtype_enum);

    int cnt = 0;
    for (int64_t j = tid; j < k; j += blockDim.x) {
        float ns = sigmoid_round_for_dtype(scalar_to_f32(neg[row * k + j]), dtype_enum);
        cnt += (ns > ps);
    }

    extern __shared__ int smem[];
    smem[tid] = cnt;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = (int64_t)smem[0] + 1;
    }
}

template <typename scalar_t>
__global__ void remote_rank_kernel(
    const int64_t* __restrict__ data_ptrs,
    const int64_t* __restrict__ sizes_offsets,
    int64_t* __restrict__ out,
    int world_size,
    int64_t k,
    int dtype_enum
) {
    int64_t global_row = (int64_t)blockIdx.x;
    const int64_t* offsets = sizes_offsets + world_size;
    int64_t total = offsets[world_size];
    if (global_row >= total) return;

    int owner = 0;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r >= world_size) break;
        if (global_row >= offsets[r] && global_row < offsets[r + 1]) {
            owner = r;
            break;
        }
    }

    int64_t local_row = global_row - offsets[owner];
    const scalar_t* base = reinterpret_cast<const scalar_t*>(
        static_cast<uintptr_t>(data_ptrs[owner])
    ) + local_row * (k + 1);

    int tid = threadIdx.x;
    float ps = sigmoid_round_for_dtype(scalar_to_f32(base[0]), dtype_enum);

    int cnt = 0;
    for (int64_t j = tid; j < k; j += blockDim.x) {
        float ns = sigmoid_round_for_dtype(scalar_to_f32(base[j + 1]), dtype_enum);
        cnt += (ns > ps);
    }

    extern __shared__ int smem[];
    smem[tid] = cnt;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[global_row] = (int64_t)smem[0] + 1;
    }
}

void fill_meta(torch::Tensor meta, int64_t p, int64_t k) {
    CHECK_CUDA(meta);
    CHECK_CONTIGUOUS(meta);
    CHECK_I64(meta);
    TORCH_CHECK(meta.numel() >= 2, "meta must have >=2 elements");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fill_meta_kernel<<<1, 32, 0, stream>>>(meta.data_ptr<int64_t>(), p, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

py::tuple compute_meta_sync(
    torch::Tensor meta_ptrs,
    torch::Tensor sizes_offsets,
    torch::Tensor summary,
    int world_size
) {
    CHECK_CUDA(meta_ptrs);
    CHECK_CUDA(sizes_offsets);
    CHECK_CUDA(summary);
    CHECK_CONTIGUOUS(meta_ptrs);
    CHECK_CONTIGUOUS(sizes_offsets);
    CHECK_CONTIGUOUS(summary);
    CHECK_I64(meta_ptrs);
    CHECK_I64(sizes_offsets);
    CHECK_I64(summary);

    TORCH_CHECK(meta_ptrs.numel() >= world_size, "meta_ptrs too small");
    TORCH_CHECK(sizes_offsets.numel() >= 2 * world_size + 1, "sizes_offsets too small");
    TORCH_CHECK(summary.numel() >= 3, "summary too small");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    compute_meta_kernel<<<1, 32, 0, stream>>>(
        meta_ptrs.data_ptr<int64_t>(),
        sizes_offsets.data_ptr<int64_t>(),
        summary.data_ptr<int64_t>(),
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int64_t h[3];
    C10_CUDA_CHECK(cudaMemcpyAsync(
        h,
        summary.data_ptr<int64_t>(),
        sizeof(int64_t) * 3,
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return py::make_tuple(h[0], h[1], h[2]);
}

void pack_scores(
    torch::Tensor pos,
    torch::Tensor neg,
    torch::Tensor dst,
    int64_t p,
    int64_t k,
    int dtype_enum
) {
    CHECK_CUDA(pos);
    CHECK_CUDA(neg);
    CHECK_CUDA(dst);
    CHECK_CONTIGUOUS(pos);
    CHECK_CONTIGUOUS(neg);
    CHECK_CONTIGUOUS(dst);

    if (p == 0) return;

    int64_t n = p * (k + 1);
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        pack_scores_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
            pos.data_ptr<at::BFloat16>(),
            neg.data_ptr<at::BFloat16>(),
            dst.data_ptr<at::BFloat16>(),
            p,
            k
        );
    } else if (dtype_enum == 1) {
        pack_scores_kernel<float><<<blocks, threads, 0, stream>>>(
            pos.data_ptr<float>(),
            neg.data_ptr<float>(),
            dst.data_ptr<float>(),
            p,
            k
        );
    } else if (dtype_enum == 2) {
        pack_scores_kernel<at::Half><<<blocks, threads, 0, stream>>>(
            pos.data_ptr<at::Half>(),
            neg.data_ptr<at::Half>(),
            dst.data_ptr<at::Half>(),
            p,
            k
        );
    } else {
        TORCH_CHECK(false, "unsupported dtype enum");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rank_local(
    torch::Tensor pos,
    torch::Tensor neg,
    torch::Tensor out,
    int64_t p,
    int64_t k,
    int dtype_enum,
    int threads
) {
    CHECK_CUDA(pos);
    CHECK_CUDA(neg);
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(pos);
    CHECK_CONTIGUOUS(neg);
    CHECK_CONTIGUOUS(out);
    CHECK_I64(out);

    if (p == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t smem = (size_t)threads * sizeof(int);

    if (dtype_enum == 0) {
        local_rank_kernel<at::BFloat16><<<p, threads, smem, stream>>>(
            pos.data_ptr<at::BFloat16>(),
            neg.data_ptr<at::BFloat16>(),
            out.data_ptr<int64_t>(),
            p,
            k,
            dtype_enum
        );
    } else if (dtype_enum == 1) {
        local_rank_kernel<float><<<p, threads, smem, stream>>>(
            pos.data_ptr<float>(),
            neg.data_ptr<float>(),
            out.data_ptr<int64_t>(),
            p,
            k,
            dtype_enum
        );
    } else if (dtype_enum == 2) {
        local_rank_kernel<at::Half><<<p, threads, smem, stream>>>(
            pos.data_ptr<at::Half>(),
            neg.data_ptr<at::Half>(),
            out.data_ptr<int64_t>(),
            p,
            k,
            dtype_enum
        );
    } else {
        TORCH_CHECK(false, "unsupported dtype enum");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rank_remote(
    torch::Tensor data_ptrs,
    torch::Tensor sizes_offsets,
    torch::Tensor out,
    int world_size,
    int64_t k,
    int dtype_enum,
    int threads
) {
    CHECK_CUDA(data_ptrs);
    CHECK_CUDA(sizes_offsets);
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(data_ptrs);
    CHECK_CONTIGUOUS(sizes_offsets);
    CHECK_CONTIGUOUS(out);
    CHECK_I64(data_ptrs);
    CHECK_I64(sizes_offsets);
    CHECK_I64(out);

    int64_t total = out.numel();
    if (total == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t smem = (size_t)threads * sizeof(int);

    if (dtype_enum == 0) {
        remote_rank_kernel<at::BFloat16><<<total, threads, smem, stream>>>(
            data_ptrs.data_ptr<int64_t>(),
            sizes_offsets.data_ptr<int64_t>(),
            out.data_ptr<int64_t>(),
            world_size,
            k,
            dtype_enum
        );
    } else if (dtype_enum == 1) {
        remote_rank_kernel<float><<<total, threads, smem, stream>>>(
            data_ptrs.data_ptr<int64_t>(),
            sizes_offsets.data_ptr<int64_t>(),
            out.data_ptr<int64_t>(),
            world_size,
            k,
            dtype_enum
        );
    } else if (dtype_enum == 2) {
        remote_rank_kernel<at::Half><<<total, threads, smem, stream>>>(
            data_ptrs.data_ptr<int64_t>(),
            sizes_offsets.data_ptr<int64_t>(),
            out.data_ptr<int64_t>(),
            world_size,
            k,
            dtype_enum
        );
    } else {
        TORCH_CHECK(false, "unsupported dtype enum");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fill_meta", &fill_meta, "write local P/K metadata");
    m.def("compute_meta_sync", &compute_meta_sync, "gather symmetric metadata and return sum/max/K");
    m.def("pack_scores", &pack_scores, "pack pos and neg scores into symmetric row-major buffer");
    m.def("rank_local", &rank_local, "single-rank fused sigmoid ranking");
    m.def("rank_remote", &rank_remote, "multi-rank UVA fused sigmoid ranking");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gs_linkpred_rank_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


_META_SLOTS = 8
_meta_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
_data_cache: Dict[Tuple[int, int, torch.dtype, int, int], Dict[str, Any]] = {}


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"unsupported dtype for custom CUDA ranking: {dtype}")


def _rank_threads(k: int) -> int:
    if k <= 32:
        return 32
    if k <= 64:
        return 64
    if k <= 128:
        return 128
    return 256


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _group_key(group: dist.ProcessGroup, device: torch.device, world_size: int) -> Tuple[int, int, int]:
    return (id(group), int(device.index if device.index is not None else torch.cuda.current_device()), world_size)


def _get_meta_resource(group: dist.ProcessGroup, device: torch.device, world_size: int):
    key = _group_key(group, device, world_size)
    cached = _meta_cache.get(key)
    if cached is not None:
        return cached

    meta = symm_mem.empty((_META_SLOTS,), device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(meta, group)

    meta_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    sizes_offsets = torch.empty((2 * world_size + 1,), device=device, dtype=torch.int64)
    summary = torch.empty((3,), device=device, dtype=torch.int64)

    cached = {
        "meta": meta,
        "hdl": hdl,
        "meta_ptrs": meta_ptrs,
        "sizes_offsets": sizes_offsets,
        "summary": summary,
    }
    _meta_cache[key] = cached
    return cached


def _get_data_resource(
    group: dist.ProcessGroup,
    device: torch.device,
    world_size: int,
    max_p: int,
    k: int,
    dtype: torch.dtype,
):
    cap_p = _next_pow2(max_p)
    key = (
        id(group),
        int(device.index if device.index is not None else torch.cuda.current_device()),
        dtype,
        world_size,
        k,
    )

    cached = _data_cache.get(key)
    if cached is not None and cached["cap_p"] >= max_p:
        return cached

    total_elems = max(1, cap_p * (k + 1))
    buf = symm_mem.empty((total_elems,), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    data_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "cap_p": cap_p,
        "buf": buf,
        "hdl": hdl,
        "data_ptrs": data_ptrs,
    }
    _data_cache[key] = cached
    return cached


@torch.no_grad()
def _local_only_rank(local_pos_scores: torch.Tensor, local_neg_scores: torch.Tensor) -> torch.Tensor:
    ext = _get_ext()

    pos = local_pos_scores.reshape(-1).contiguous()
    neg = local_neg_scores.contiguous()

    p = int(pos.numel())
    k = int(neg.shape[1]) if neg.ndim == 2 else 0
    out = torch.empty((p,), device=pos.device, dtype=torch.long)

    if p == 0:
        return out

    dtype_enum = _dtype_code(pos.dtype)
    ext.rank_local(pos, neg, out, p, k, dtype_enum, _rank_threads(k))
    return out


@torch.no_grad()
def solution(
    local_pos_scores: torch.Tensor,
    local_neg_scores: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or (dist.group.WORLD if dist.is_initialized() else None)

    if not dist.is_initialized() or group is None:
        return _local_only_rank(local_pos_scores, local_neg_scores)

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return _local_only_rank(local_pos_scores, local_neg_scores)

    assert local_pos_scores.is_cuda, "local_pos_scores must be CUDA"
    assert local_neg_scores.is_cuda, "local_neg_scores must be CUDA"
    assert local_pos_scores.dtype == local_neg_scores.dtype, "pos/neg dtypes must match"
    assert local_neg_scores.ndim == 2, "local_neg_scores must have shape [P, K]"

    ext = _get_ext()

    pos = local_pos_scores.reshape(-1).contiguous()
    neg = local_neg_scores.contiguous()

    p = int(pos.numel())
    k = int(neg.shape[1])
    dtype = pos.dtype
    dtype_enum = _dtype_code(dtype)
    device = pos.device

    meta_res = _get_meta_resource(group, device, world_size)
    meta = meta_res["meta"]
    meta_hdl = meta_res["hdl"]

    ext.fill_meta(meta, p, k)
    meta_hdl.barrier(channel=0)

    sum_p, max_p, global_k = ext.compute_meta_sync(
        meta_res["meta_ptrs"],
        meta_res["sizes_offsets"],
        meta_res["summary"],
        world_size,
    )
    sum_p = int(sum_p)
    max_p = int(max_p)
    global_k = int(global_k)

    # The reference requires compatible negative-score width across ranks.
    # We use local K for layout and validate against gathered metadata.
    assert global_k == k, "all ranks must use the same negative-score width K"

    out = torch.empty((sum_p,), device=device, dtype=torch.long)
    if sum_p == 0:
        return out

    data_res = _get_data_resource(group, device, world_size, max_p, k, dtype)
    buf = data_res["buf"]
    data_hdl = data_res["hdl"]

    ext.pack_scores(pos, neg, buf, p, k, dtype_enum)
    data_hdl.barrier(channel=1)

    ext.rank_remote(
        data_res["data_ptrs"],
        meta_res["sizes_offsets"],
        out,
        world_size,
        k,
        dtype_enum,
        _rank_threads(k),
    )

    return out