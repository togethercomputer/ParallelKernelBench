"""
Fused AdamW on flat shards — single BF16 CUDA kernel, no collectives needed.
Local elementwise op; we just fuse everything into one launch with vectorized
BF16 loads/stores to minimize memory traffic and launch overhead.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

template<typename T>
__device__ __forceinline__ float to_float(T x);

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template<>
__device__ __forceinline__ float to_float<float>(float x) { return x; }

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
template<>
__device__ __forceinline__ float from_float<float>(float x) { return x; }

// Vectorized BF16 fused AdamW: 8 elements per thread via float4 loads on bf16x8
__global__ void adamw_bf16_kernel(
    const __nv_bfloat16* __restrict__ p_in,
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ m_in,
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ p_out,
    __nv_bfloat16* __restrict__ m_out,
    __nv_bfloat16* __restrict__ v_out,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float eps,
    float weight_decay,
    float inv_bc1,
    float inv_bc2_sqrt,
    int64_t n
) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Process 8 bf16 elements at a time using float4 (= 16 bytes = 8 bf16)
    const int64_t n_vec = n / 8;
    const float4* p_in_v  = reinterpret_cast<const float4*>(p_in);
    const float4* g_v     = reinterpret_cast<const float4*>(g);
    const float4* m_in_v  = reinterpret_cast<const float4*>(m_in);
    const float4* v_in_v  = reinterpret_cast<const float4*>(v_in);
    float4* p_out_v       = reinterpret_cast<float4*>(p_out);
    float4* m_out_v       = reinterpret_cast<float4*>(m_out);
    float4* v_out_v       = reinterpret_cast<float4*>(v_out);

    const float lr_wd = lr * weight_decay;

    for (int64_t i = tid; i < n_vec; i += stride) {
        float4 pv = p_in_v[i];
        float4 gv = g_v[i];
        float4 mv = m_in_v[i];
        float4 vv = v_in_v[i];

        __nv_bfloat16* pb = reinterpret_cast<__nv_bfloat16*>(&pv);
        __nv_bfloat16* gb = reinterpret_cast<__nv_bfloat16*>(&gv);
        __nv_bfloat16* mb = reinterpret_cast<__nv_bfloat16*>(&mv);
        __nv_bfloat16* vb = reinterpret_cast<__nv_bfloat16*>(&vv);

        float4 op, om, ov;
        __nv_bfloat16* opb = reinterpret_cast<__nv_bfloat16*>(&op);
        __nv_bfloat16* omb = reinterpret_cast<__nv_bfloat16*>(&om);
        __nv_bfloat16* ovb = reinterpret_cast<__nv_bfloat16*>(&ov);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float p = __bfloat162float(pb[k]);
            float gr = __bfloat162float(gb[k]);
            float mm = __bfloat162float(mb[k]);
            float vvv = __bfloat162float(vb[k]);

            mm = mm * beta1 + gr * one_minus_beta1;
            vvv = vvv * beta2 + gr * gr * one_minus_beta2;

            float m_hat = mm * inv_bc1;
            float v_hat_sqrt = sqrtf(vvv) * inv_bc2_sqrt;
            float denom = v_hat_sqrt + eps;
            float upd = m_hat / denom;

            float p_new = p - lr * upd - lr_wd * p;

            opb[k] = __float2bfloat16(p_new);
            omb[k] = __float2bfloat16(mm);
            ovb[k] = __float2bfloat16(vvv);
        }

        p_out_v[i] = op;
        m_out_v[i] = om;
        v_out_v[i] = ov;
    }

    // Tail
    const int64_t tail_start = n_vec * 8;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        float p = __bfloat162float(p_in[i]);
        float gr = __bfloat162float(g[i]);
        float mm = __bfloat162float(m_in[i]);
        float vvv = __bfloat162float(v_in[i]);

        mm = mm * beta1 + gr * one_minus_beta1;
        vvv = vvv * beta2 + gr * gr * one_minus_beta2;

        float m_hat = mm * inv_bc1;
        float v_hat_sqrt = sqrtf(vvv) * inv_bc2_sqrt;
        float denom = v_hat_sqrt + eps;
        float upd = m_hat / denom;

        float p_new = p - lr * upd - lr_wd * p;

        p_out[i] = __float2bfloat16(p_new);
        m_out[i] = __float2bfloat16(mm);
        v_out[i] = __float2bfloat16(vvv);
    }
}

__global__ void adamw_f32_kernel(
    const float* __restrict__ p_in,
    const float* __restrict__ g,
    const float* __restrict__ m_in,
    const float* __restrict__ v_in,
    float* __restrict__ p_out,
    float* __restrict__ m_out,
    float* __restrict__ v_out,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float eps,
    float weight_decay,
    float inv_bc1,
    float inv_bc2_sqrt,
    int64_t n
) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const float lr_wd = lr * weight_decay;

    for (int64_t i = tid; i < n; i += stride) {
        float p = p_in[i];
        float gr = g[i];
        float mm = m_in[i];
        float vvv = v_in[i];

        mm = mm * beta1 + gr * one_minus_beta1;
        vvv = vvv * beta2 + gr * gr * one_minus_beta2;

        float m_hat = mm * inv_bc1;
        float v_hat_sqrt = sqrtf(vvv) * inv_bc2_sqrt;
        float denom = v_hat_sqrt + eps;
        float upd = m_hat / denom;

        float p_new = p - lr * upd - lr_wd * p;

        p_out[i] = p_new;
        m_out[i] = mm;
        v_out[i] = vvv;
    }
}

void launch_adamw(
    torch::Tensor p_in, torch::Tensor g, torch::Tensor m_in, torch::Tensor v_in,
    torch::Tensor p_out, torch::Tensor m_out, torch::Tensor v_out,
    double lr, double beta1, double beta2, double eps, double weight_decay,
    double inv_bc1, double inv_bc2_sqrt
) {
    int64_t n = p_in.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int blocks;

    if (p_in.dtype() == torch::kBFloat16) {
        int64_t n_vec = (n + 7) / 8;
        blocks = (int)((n_vec + threads - 1) / threads);
        if (blocks > 2048) blocks = 2048;
        if (blocks < 1) blocks = 1;
        adamw_bf16_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)p_in.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)g.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)m_in.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)v_in.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)p_out.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)m_out.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)v_out.data_ptr<at::BFloat16>(),
            (float)lr, (float)beta1, (float)beta2,
            (float)(1.0 - beta1), (float)(1.0 - beta2),
            (float)eps, (float)weight_decay,
            (float)inv_bc1, (float)inv_bc2_sqrt, n);
    } else {
        blocks = (int)((n + threads - 1) / threads);
        if (blocks > 2048) blocks = 2048;
        if (blocks < 1) blocks = 1;
        adamw_f32_kernel<<<blocks, threads, 0, stream>>>(
            p_in.data_ptr<float>(), g.data_ptr<float>(),
            m_in.data_ptr<float>(), v_in.data_ptr<float>(),
            p_out.data_ptr<float>(), m_out.data_ptr<float>(),
            v_out.data_ptr<float>(),
            (float)lr, (float)beta1, (float)beta2,
            (float)(1.0 - beta1), (float)(1.0 - beta2),
            (float)eps, (float)weight_decay,
            (float)inv_bc1, (float)inv_bc2_sqrt, n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_adamw", &launch_adamw, "Fused AdamW kernel");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_adamw_bf16_ext", CUDA_SRC)
    return _ext


@torch.no_grad()
def solution(
    flat_param_shard: Tensor,
    flat_grad_shard: Tensor,
    exp_avg_shard: Tensor,
    exp_avg_sq_shard: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    assert step >= 1
    assert (
        flat_param_shard.shape == flat_grad_shard.shape
        == exp_avg_shard.shape == exp_avg_sq_shard.shape
    )

    p = flat_param_shard.contiguous()
    g = flat_grad_shard.contiguous()
    m_in = exp_avg_shard.contiguous()
    v_in = exp_avg_sq_shard.contiguous()

    p_out = torch.empty_like(p)
    m_out = torch.empty_like(m_in)
    v_out = torch.empty_like(v_in)

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    inv_bc1 = 1.0 / bc1
    inv_bc2_sqrt = 1.0 / math.sqrt(bc2)

    _get_ext().launch_adamw(
        p, g, m_in, v_in, p_out, m_out, v_out,
        float(lr), float(beta1), float(beta2),
        float(eps), float(weight_decay),
        float(inv_bc1), float(inv_bc2_sqrt),
    )

    return p_out, m_out, v_out


__all__ = ["solution"]