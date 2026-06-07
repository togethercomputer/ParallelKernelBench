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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

template <typename T>
__device__ __forceinline__ float cvt_to_float(T x);

template <>
__device__ __forceinline__ float cvt_to_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float cvt_to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float cvt_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T cvt_from_float(float x);

template <>
__device__ __forceinline__ float cvt_from_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ __half cvt_from_float<__half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 cvt_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float block_sum(float v, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

template <typename scalar_t>
__global__ void cp_attention_uva_kernel(
    const scalar_t* __restrict__ q,
    const int64_t* __restrict__ k_ptrs,
    const int64_t* __restrict__ v_ptrs,
    scalar_t* __restrict__ out,
    int B,
    int S,
    int H,
    int D,
    int world_size,
    int rank,
    float scale,
    bool causal
) {
    const int64_t row = (int64_t)blockIdx.x;
    const int tid = threadIdx.x;

    const int qi = row % S;
    const int tmp0 = row / S;
    const int h = tmp0 % H;
    const int b = tmp0 / H;

    const int64_t q_base = (((int64_t)b * S + qi) * H + h) * D;

    __shared__ float smem[1024];
    __shared__ float s_m;
    __shared__ float s_l;
    __shared__ float s_w;

    float m = -INFINITY;
    float l = 0.0f;

    // Pass 1: numerically stable row max / denominator over all visible CP shards.
    for (int rr = 0; rr < world_size; ++rr) {
        if (causal && rr > rank) {
            continue;
        }

        const int max_j = (causal && rr == rank) ? (qi + 1) : S;
        const scalar_t* __restrict__ k_base_ptr =
            reinterpret_cast<const scalar_t*>((uintptr_t)k_ptrs[rr]);

        for (int kj = 0; kj < max_j; ++kj) {
            const int64_t k_base = (((int64_t)b * S + kj) * H + h) * D;

            float partial = 0.0f;
            for (int d = tid; d < D; d += blockDim.x) {
                const float qv = cvt_to_float<scalar_t>(q[q_base + d]);
                const float kv = cvt_to_float<scalar_t>(k_base_ptr[k_base + d]);
                partial = fmaf(qv, kv, partial);
            }

            const float dot = block_sum(partial, smem) * scale;

            if (tid == 0) {
                const float new_m = fmaxf(m, dot);
                l = l * __expf(m - new_m) + __expf(dot - new_m);
                m = new_m;
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        s_m = m;
        s_l = l;
    }
    __syncthreads();

    // Pass 2: recompute scores, normalize, accumulate V.
    float acc = 0.0f;

    for (int rr = 0; rr < world_size; ++rr) {
        if (causal && rr > rank) {
            continue;
        }

        const int max_j = (causal && rr == rank) ? (qi + 1) : S;
        const scalar_t* __restrict__ k_base_ptr =
            reinterpret_cast<const scalar_t*>((uintptr_t)k_ptrs[rr]);
        const scalar_t* __restrict__ v_base_ptr =
            reinterpret_cast<const scalar_t*>((uintptr_t)v_ptrs[rr]);

        for (int kj = 0; kj < max_j; ++kj) {
            const int64_t kv_base = (((int64_t)b * S + kj) * H + h) * D;

            float partial = 0.0f;
            for (int d = tid; d < D; d += blockDim.x) {
                const float qv = cvt_to_float<scalar_t>(q[q_base + d]);
                const float kv = cvt_to_float<scalar_t>(k_base_ptr[kv_base + d]);
                partial = fmaf(qv, kv, partial);
            }

            const float dot = block_sum(partial, smem) * scale;

            if (tid == 0) {
                s_w = __expf(dot - s_m) / s_l;
            }
            __syncthreads();

            if (tid < D) {
                const float vv = cvt_to_float<scalar_t>(v_base_ptr[kv_base + tid]);
                acc = fmaf(s_w, vv, acc);
            }
            __syncthreads();
        }
    }

    if (tid < D) {
        out[q_base + tid] = cvt_from_float<scalar_t>(acc);
    }
}

static int pick_threads(int D) {
    int threads = 32;
    while (threads < D) {
        threads <<= 1;
    }
    if (threads > 1024) {
        threads = 1024;
    }
    return threads;
}

void launch_cp_attention_uva(
    torch::Tensor q,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    torch::Tensor out,
    int64_t B,
    int64_t S,
    int64_t H,
    int64_t D,
    int64_t world_size,
    int64_t rank,
    double scale,
    bool causal
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(k_ptrs.is_cuda() && v_ptrs.is_cuda(), "pointer tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(k_ptrs.scalar_type() == torch::kInt64, "k_ptrs must be int64");
    TORCH_CHECK(v_ptrs.scalar_type() == torch::kInt64, "v_ptrs must be int64");
    TORCH_CHECK(D <= 1024, "head dimension D > 1024 is not supported by this kernel");

    const int threads = pick_threads((int)D);
    const int64_t rows64 = B * S * H;
    TORCH_CHECK(rows64 <= INT_MAX, "too many attention rows for this launch");
    const dim3 grid((unsigned int)rows64);
    const dim3 block((unsigned int)threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int64_t* kp = k_ptrs.data_ptr<int64_t>();
    const int64_t* vp = v_ptrs.data_ptr<int64_t>();

    if (q.scalar_type() == torch::kBFloat16) {
        const __nv_bfloat16* qptr =
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
        __nv_bfloat16* optr =
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
        cp_attention_uva_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            qptr, kp, vp, optr,
            (int)B, (int)S, (int)H, (int)D,
            (int)world_size, (int)rank, (float)scale, causal);
    } else if (q.scalar_type() == torch::kFloat16) {
        const __half* qptr =
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
        __half* optr =
            reinterpret_cast<__half*>(out.data_ptr<at::Half>());
        cp_attention_uva_kernel<__half><<<grid, block, 0, stream>>>(
            qptr, kp, vp, optr,
            (int)B, (int)S, (int)H, (int)D,
            (int)world_size, (int)rank, (float)scale, causal);
    } else if (q.scalar_type() == torch::kFloat32) {
        cp_attention_uva_kernel<float><<<grid, block, 0, stream>>>(
            q.data_ptr<float>(), kp, vp, out.data_ptr<float>(),
            (int)B, (int)S, (int)H, (int)D,
            (int)world_size, (int)rank, (float)scale, causal);
    } else {
        TORCH_CHECK(false, "supported dtypes: bfloat16, float16, float32");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_cp_attention_uva", &launch_cp_attention_uva,
          "Context-parallel ring attention via symmetric-memory UVA peer loads");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attention_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _group_key(group: dist.ProcessGroup):
    return id(group)


def _get_symm_resources(
    shape,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
):
    key = (tuple(shape), dtype, device.index, _group_key(group))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    k_buf = symm_mem.empty(shape, device=device, dtype=dtype)
    v_buf = symm_mem.empty(shape, device=device, dtype=dtype)

    k_hdl = symm_mem.rendezvous(k_buf, group)
    v_hdl = symm_mem.rendezvous(v_buf, group)

    out = torch.empty(shape, device=device, dtype=dtype)
    k_ptrs = torch.tensor(k_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    v_ptrs = torch.tensor(v_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = {
        "k_buf": k_buf,
        "v_buf": v_buf,
        "k_hdl": k_hdl,
        "v_hdl": v_hdl,
        "out": out,
        "k_ptrs": k_ptrs,
        "v_ptrs": v_ptrs,
    }
    _resource_cache[key] = res
    return res


_single_rank_ptr_cache = {}


def _get_single_rank_resources(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor):
    key = (tuple(q.shape), q.dtype, q.device.index, int(k.data_ptr()), int(v.data_ptr()))
    cached = _single_rank_ptr_cache.get(key)
    if cached is not None:
        cached["out"] = torch.empty_like(q)
        return cached

    k_ptrs = torch.tensor([int(k.data_ptr())], device=q.device, dtype=torch.int64)
    v_ptrs = torch.tensor([int(v.data_ptr())], device=q.device, dtype=torch.int64)
    out = torch.empty_like(q)

    res = {"k_ptrs": k_ptrs, "v_ptrs": v_ptrs, "out": out}
    _single_rank_ptr_cache[key] = res
    return res


@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Per-rank context-parallel attention forward.

    q, k, v: [B, S_local, H, D], normally BF16 CUDA contiguous or strided tensors.
    Returns: [B, S_local, H, D], same dtype as q.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v must be CUDA tensors"
    assert q.dim() == 4 and k.shape == q.shape and v.shape == q.shape
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in (torch.bfloat16, torch.float16, torch.float32)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    B, S, H, D = q_c.shape
    assert D <= 1024, "head dimension larger than 1024 is not supported"

    ext = _get_ext()

    if not dist.is_initialized():
        res = _get_single_rank_resources(k_c, v_c, q_c)
        ext.launch_cp_attention_uva(
            q_c,
            res["k_ptrs"],
            res["v_ptrs"],
            res["out"],
            int(B),
            int(S),
            int(H),
            int(D),
            1,
            0,
            float(softmax_scale),
            bool(causal),
        )
        return res["out"]

    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        res = _get_single_rank_resources(k_c, v_c, q_c)
        ext.launch_cp_attention_uva(
            q_c,
            res["k_ptrs"],
            res["v_ptrs"],
            res["out"],
            int(B),
            int(S),
            int(H),
            int(D),
            1,
            0,
            float(softmax_scale),
            bool(causal),
        )
        return res["out"]

    res = _get_symm_resources(tuple(q_c.shape), q_c.dtype, q_c.device, group)

    # Publish local K/V into symmetric buffers; peer GPUs read these through UVA.
    res["k_buf"].copy_(k_c)
    res["v_buf"].copy_(v_c)

    # Device-side symmetric barriers protect visibility before peer reads.
    res["k_hdl"].barrier(channel=0)
    res["v_hdl"].barrier(channel=1)

    ext.launch_cp_attention_uva(
        q_c,
        res["k_ptrs"],
        res["v_ptrs"],
        res["out"],
        int(B),
        int(S),
        int(H),
        int(D),
        int(world_size),
        int(rank),
        float(softmax_scale),
        bool(causal),
    )

    # Prevent a faster rank from overwriting its symmetric K/V while peers may
    # still be reading this iteration's buffers.
    res["k_hdl"].barrier(channel=2)

    return res["out"]