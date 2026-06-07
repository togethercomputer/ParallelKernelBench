"""
Strategy:
- **Single Fused CUDA Kernel**: We replace the sequence of native PyTorch element-wise operations (clone, mul, add, addcmul, div, sqrt) with one fused C++ extension kernel. This eliminates multiple memory roundtrips to HBM and slashes CPU kernel launch overhead.
- **Precision Management**: Following PyTorch's mixed-precision optimizer standards, the kernel dynamically casts memory-bound types (like bfloat16 or float16) to float32 for all intermediate calculations (moments, bias correction, weight decay), writing the accurate results back out in the requested tensor dtype.
- **Maximal H100 Bandwidth**: Uses a simple grid-stride loop mapped onto the H100 SMs, leaning on the massive L2 cache for perfect memory coalescing on flat FSDP parameter shards without requiring restrictive vector alignment.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename scalar_t_p, typename scalar_t_m>
__global__ void adamw_kernel(
    const scalar_t_p* __restrict__ p_in,
    const scalar_t_p* __restrict__ g_in,
    const scalar_t_m* __restrict__ m_in,
    const scalar_t_m* __restrict__ v_in,
    scalar_t_p* __restrict__ p_out,
    scalar_t_m* __restrict__ m_out,
    scalar_t_m* __restrict__ v_out,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    #pragma unroll 4
    for (; idx < n; idx += stride) {
        float p = static_cast<float>(p_in[idx]);
        float g = static_cast<float>(g_in[idx]);
        float m = static_cast<float>(m_in[idx]);
        float v = static_cast<float>(v_in[idx]);

        // Update biased first moment estimate
        m = m * beta1 + g * (1.0f - beta1);
        
        // Update biased second raw moment estimate
        v = v * beta2 + g * g * (1.0f - beta2);

        // Compute bias-corrected moments
        float m_hat = m / bc1;
        float v_hat = v / bc2;

        float denom = sqrtf(v_hat) + eps;

        // Decoupled weight decay and Adam step:
        // theta_new = theta - lr * (m_hat / denom) - lr * weight_decay * theta
        float p_new = p - lr * (m_hat / denom);
        p_new = p_new - (lr * weight_decay) * p;

        p_out[idx] = static_cast<scalar_t_p>(p_new);
        m_out[idx] = static_cast<scalar_t_m>(m);
        v_out[idx] = static_cast<scalar_t_m>(v);
    }
}

void launch_adamw(
    torch::Tensor p_in,
    torch::Tensor g_in,
    torch::Tensor m_in,
    torch::Tensor v_in,
    torch::Tensor p_out,
    torch::Tensor m_out,
    torch::Tensor v_out,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2
) {
    int64_t n = p_in.numel();
    if (n == 0) return;

    // Use 512 threads per block and enough blocks to saturate H100 SMs,
    // relying on the grid-stride loop for larger element counts.
    const int threads = 512;
    const int blocks = std::min((int)((n + threads - 1) / threads), 2048);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Dispatch across the common precision configurations
    if (p_in.scalar_type() == torch::kBFloat16 && m_in.scalar_type() == torch::kBFloat16) {
        adamw_kernel<at::BFloat16, at::BFloat16><<<blocks, threads, 0, stream>>>(
            p_in.data_ptr<at::BFloat16>(), g_in.data_ptr<at::BFloat16>(),
            m_in.data_ptr<at::BFloat16>(), v_in.data_ptr<at::BFloat16>(),
            p_out.data_ptr<at::BFloat16>(), m_out.data_ptr<at::BFloat16>(), v_out.data_ptr<at::BFloat16>(),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, n
        );
    } else if (p_in.scalar_type() == torch::kFloat32 && m_in.scalar_type() == torch::kFloat32) {
        adamw_kernel<float, float><<<blocks, threads, 0, stream>>>(
            p_in.data_ptr<float>(), g_in.data_ptr<float>(),
            m_in.data_ptr<float>(), v_in.data_ptr<float>(),
            p_out.data_ptr<float>(), m_out.data_ptr<float>(), v_out.data_ptr<float>(),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, n
        );
    } else if (p_in.scalar_type() == torch::kBFloat16 && m_in.scalar_type() == torch::kFloat32) {
        adamw_kernel<at::BFloat16, float><<<blocks, threads, 0, stream>>>(
            p_in.data_ptr<at::BFloat16>(), g_in.data_ptr<at::BFloat16>(),
            m_in.data_ptr<float>(), v_in.data_ptr<float>(),
            p_out.data_ptr<at::BFloat16>(), m_out.data_ptr<float>(), v_out.data_ptr<float>(),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, n
        );
    } else if (p_in.scalar_type() == torch::kHalf && m_in.scalar_type() == torch::kFloat32) {
        adamw_kernel<at::Half, float><<<blocks, threads, 0, stream>>>(
            p_in.data_ptr<at::Half>(), g_in.data_ptr<at::Half>(),
            m_in.data_ptr<float>(), v_in.data_ptr<float>(),
            p_out.data_ptr<at::Half>(), m_out.data_ptr<float>(), v_out.data_ptr<float>(),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, n
        );
    } else if (p_in.scalar_type() == torch::kHalf && m_in.scalar_type() == torch::kHalf) {
        adamw_kernel<at::Half, at::Half><<<blocks, threads, 0, stream>>>(
            p_in.data_ptr<at::Half>(), g_in.data_ptr<at::Half>(),
            m_in.data_ptr<at::Half>(), v_in.data_ptr<at::Half>(),
            p_out.data_ptr<at::Half>(), m_out.data_ptr<at::Half>(), v_out.data_ptr<at::Half>(),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2, n
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination for Fused AdamW");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_adamw", &launch_adamw, "Fused AdamW C++ kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_adamw_sharded_ext", CUDA_SRC)
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
    """
    Decoupled AdamW (Loshchilov & Hutter) on one rank's shards.
    """
    assert step >= 1
    assert (
        flat_param_shard.shape == flat_grad_shard.shape == exp_avg_shard.shape == exp_avg_sq_shard.shape
    )

    # Ensure tensors are contiguous and valid for the CUDA kernel pointers
    flat_param_shard = flat_param_shard.contiguous()
    flat_grad_shard = flat_grad_shard.contiguous()
    exp_avg_shard = exp_avg_shard.contiguous()
    exp_avg_sq_shard = exp_avg_sq_shard.contiguous()

    # Allocate outputs matching the out-of-place signature
    out_param = torch.empty_like(flat_param_shard)
    out_m = torch.empty_like(exp_avg_shard)
    out_v = torch.empty_like(exp_avg_sq_shard)

    # Pre-calculate bias correction factors on the host
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    # Dispatch to customized fused kernel
    _get_ext().launch_adamw(
        flat_param_shard,
        flat_grad_shard,
        exp_avg_shard,
        exp_avg_sq_shard,
        out_param,
        out_m,
        out_v,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bc1,
        bc2
    )

    return out_param, out_m, out_v

__all__ = ["solution"]