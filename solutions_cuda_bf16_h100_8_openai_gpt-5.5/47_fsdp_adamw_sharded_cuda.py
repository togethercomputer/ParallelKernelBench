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
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

constexpr int DTYPE_F32  = 0;
constexpr int DTYPE_BF16 = 1;

int dtype_code(const torch::Tensor& t) {
    if (t.scalar_type() == torch::kFloat32) {
        return DTYPE_F32;
    }
    if (t.scalar_type() == torch::kBFloat16) {
        return DTYPE_BF16;
    }
    TORCH_CHECK(false, "Only float32 and bfloat16 tensors are supported");
    return -1;
}

__device__ __forceinline__ float load_as_f32(const void* __restrict__ p, int dtype, int64_t i) {
    if (dtype == DTYPE_F32) {
        return static_cast<const float*>(p)[i];
    } else {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    }
}

__device__ __forceinline__ float round_to_dtype(float x, int dtype) {
    if (dtype == DTYPE_BF16) {
        return __bfloat162float(__float2bfloat16_rn(x));
    }
    return x;
}

__device__ __forceinline__ void store_from_f32(void* __restrict__ p, int dtype, int64_t i, float x) {
    if (dtype == DTYPE_F32) {
        static_cast<float*>(p)[i] = x;
    } else {
        static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(x);
    }
}

template<int VEC>
__global__ void adamw_shard_kernel(
    const void* __restrict__ param,
    const void* __restrict__ grad,
    const void* __restrict__ exp_avg,
    const void* __restrict__ exp_avg_sq,
    void* __restrict__ out_param,
    void* __restrict__ out_exp_avg,
    void* __restrict__ out_exp_avg_sq,
    int64_t n,
    int param_dtype,
    int grad_dtype,
    int m_dtype,
    int v_dtype,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float inv_bc1,
    float inv_sqrt_bc2
) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x * VEC;
    int64_t base = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC;

    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;
    const float step_size = lr * inv_bc1;
    const float decay_alpha = lr * weight_decay;

    for (; base < n; base += stride) {
        #pragma unroll
        for (int lane = 0; lane < VEC; ++lane) {
            const int64_t i = base + lane;
            if (i >= n) {
                break;
            }

            const float p = load_as_f32(param, param_dtype, i);
            const float g = load_as_f32(grad, grad_dtype, i);
            const float m_old = load_as_f32(exp_avg, m_dtype, i);
            const float v_old = load_as_f32(exp_avg_sq, v_dtype, i);

            float m_new = beta1 * m_old + one_minus_beta1 * g;
            float v_new = beta2 * v_old + one_minus_beta2 * g * g;

            // PyTorch reference uses in-place moment tensors, so if moments are bf16
            // the rounded values are then used by subsequent math.
            const float m_for_update = round_to_dtype(m_new, m_dtype);
            const float v_for_update = round_to_dtype(v_new, v_dtype);

            const float denom = sqrtf(fmaxf(v_for_update, 0.0f)) * inv_sqrt_bc2 + eps;
            float theta = p - step_size * (m_for_update / denom);

            // Reference performs two in-place adds on theta.  For bf16 params,
            // the first add rounds before the decoupled weight-decay add.
            theta = round_to_dtype(theta, param_dtype);
            theta = theta - decay_alpha * p;

            store_from_f32(out_param, param_dtype, i, theta);
            store_from_f32(out_exp_avg, m_dtype, i, m_new);
            store_from_f32(out_exp_avg_sq, v_dtype, i, v_new);
        }
    }
}

} // namespace

void adamw_shard_update(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor out_param,
    torch::Tensor out_exp_avg,
    torch::Tensor out_exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double inv_bc1,
    double inv_sqrt_bc2
) {
    TORCH_CHECK(param.is_cuda(), "param must be CUDA");
    TORCH_CHECK(grad.is_cuda(), "grad must be CUDA");
    TORCH_CHECK(exp_avg.is_cuda(), "exp_avg must be CUDA");
    TORCH_CHECK(exp_avg_sq.is_cuda(), "exp_avg_sq must be CUDA");
    TORCH_CHECK(out_param.is_cuda(), "out_param must be CUDA");
    TORCH_CHECK(out_exp_avg.is_cuda(), "out_exp_avg must be CUDA");
    TORCH_CHECK(out_exp_avg_sq.is_cuda(), "out_exp_avg_sq must be CUDA");

    TORCH_CHECK(param.is_contiguous(), "param must be contiguous");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(), "exp_avg must be contiguous");
    TORCH_CHECK(exp_avg_sq.is_contiguous(), "exp_avg_sq must be contiguous");
    TORCH_CHECK(out_param.is_contiguous(), "out_param must be contiguous");
    TORCH_CHECK(out_exp_avg.is_contiguous(), "out_exp_avg must be contiguous");
    TORCH_CHECK(out_exp_avg_sq.is_contiguous(), "out_exp_avg_sq must be contiguous");

    TORCH_CHECK(param.numel() == grad.numel(), "param/grad numel mismatch");
    TORCH_CHECK(param.numel() == exp_avg.numel(), "param/exp_avg numel mismatch");
    TORCH_CHECK(param.numel() == exp_avg_sq.numel(), "param/exp_avg_sq numel mismatch");
    TORCH_CHECK(param.numel() == out_param.numel(), "param/out_param numel mismatch");
    TORCH_CHECK(exp_avg.numel() == out_exp_avg.numel(), "exp_avg/out_exp_avg numel mismatch");
    TORCH_CHECK(exp_avg_sq.numel() == out_exp_avg_sq.numel(), "exp_avg_sq/out_exp_avg_sq numel mismatch");

    TORCH_CHECK(out_param.scalar_type() == param.scalar_type(), "out_param dtype mismatch");
    TORCH_CHECK(out_exp_avg.scalar_type() == exp_avg.scalar_type(), "out_exp_avg dtype mismatch");
    TORCH_CHECK(out_exp_avg_sq.scalar_type() == exp_avg_sq.scalar_type(), "out_exp_avg_sq dtype mismatch");

    const int64_t n = param.numel();
    if (n == 0) {
        return;
    }

    const int param_dtype = dtype_code(param);
    const int grad_dtype = dtype_code(grad);
    const int m_dtype = dtype_code(exp_avg);
    const int v_dtype = dtype_code(exp_avg_sq);

    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    constexpr int threads = 256;
    constexpr int vec = 4;
    const int64_t elems_per_block = static_cast<int64_t>(threads) * vec;
    int blocks_for_n = static_cast<int>((n + elems_per_block - 1) / elems_per_block);
    int blocks = blocks_for_n;
    const int target_blocks = prop.multiProcessorCount * 8;
    if (blocks > target_blocks) {
        blocks = target_blocks;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    adamw_shard_kernel<vec><<<blocks, threads, 0, stream>>>(
        param.data_ptr(),
        grad.data_ptr(),
        exp_avg.data_ptr(),
        exp_avg_sq.data_ptr(),
        out_param.data_ptr(),
        out_exp_avg.data_ptr(),
        out_exp_avg_sq.data_ptr(),
        n,
        param_dtype,
        grad_dtype,
        m_dtype,
        v_dtype,
        static_cast<float>(lr),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(eps),
        static_cast<float>(weight_decay),
        static_cast<float>(inv_bc1),
        static_cast<float>(inv_sqrt_bc2)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adamw_shard_update", &adamw_shard_update,
          "Fused AdamW update for flat FSDP/ZeRO parameter shards");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_adamw_sharded_bf16_h100_ext", CUDA_SRC)
    return _ext


def _symm_empty_like(x: Tensor) -> Tensor:
    # Symmetric allocation keeps optimizer outputs UVA-addressable for downstream
    # custom all-gather/reduce paths.  No distributed collective is needed here
    # because this AdamW shard update is purely local after reduce-scatter.
    if (
        x.is_cuda
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return symm_mem.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
    return torch.empty_like(x)


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
    Decoupled AdamW on one rank's flat shard, fused into a single CUDA kernel.
    """
    assert step >= 1
    assert (
        flat_param_shard.shape
        == flat_grad_shard.shape
        == exp_avg_shard.shape
        == exp_avg_sq_shard.shape
    )
    assert flat_param_shard.is_cuda
    assert flat_grad_shard.is_cuda
    assert exp_avg_shard.is_cuda
    assert exp_avg_sq_shard.is_cuda
    assert flat_param_shard.is_contiguous()
    assert flat_grad_shard.is_contiguous()
    assert exp_avg_shard.is_contiguous()
    assert exp_avg_sq_shard.is_contiguous()

    updated_param = _symm_empty_like(flat_param_shard)
    updated_exp_avg = _symm_empty_like(exp_avg_shard)
    updated_exp_avg_sq = _symm_empty_like(exp_avg_sq_shard)

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    inv_bc1 = 1.0 / bc1
    inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)

    _get_ext().adamw_shard_update(
        flat_param_shard,
        flat_grad_shard,
        exp_avg_shard,
        exp_avg_sq_shard,
        updated_param,
        updated_exp_avg,
        updated_exp_avg_sq,
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
        float(inv_bc1),
        float(inv_sqrt_bc2),
    )

    return updated_param, updated_exp_avg, updated_exp_avg_sq


__all__ = ["solution"]