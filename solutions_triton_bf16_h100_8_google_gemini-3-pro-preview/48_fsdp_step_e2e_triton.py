import math
from typing import Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _unflatten_dense_tensors

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

struct PeerPtrs {
    const void* ptrs[8];
};

template<typename T>
__global__ void all_gather_kernel_vec8(
    PeerPtrs peer_ptrs,
    T* __restrict__ full_flat,
    int64_t p_vec,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p_vec * world_size) {
        int rank = idx / p_vec;
        int64_t offset = idx % p_vec;
        const T* src = reinterpret_cast<const T*>(peer_ptrs.ptrs[rank]);
        full_flat[idx] = src[offset];
    }
}

__global__ void all_gather_kernel_scalar(
    PeerPtrs peer_ptrs,
    __nv_bfloat16* __restrict__ full_flat,
    int64_t p,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p * world_size) {
        int rank = idx / p;
        int64_t offset = idx % p;
        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs.ptrs[rank]);
        full_flat[idx] = src[offset];
    }
}

__global__ void fused_rs_adamw_kernel_vec2(
    PeerPtrs peer_g_ptrs,
    const __nv_bfloat16* __restrict__ local_param,
    const __nv_bfloat16* __restrict__ local_m,
    const __nv_bfloat16* __restrict__ local_v,
    __nv_bfloat16* __restrict__ out_param,
    __nv_bfloat16* __restrict__ out_m,
    __nv_bfloat16* __restrict__ out_v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2,
    int64_t p, int world_size, int rank
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = idx * 2;
    if (i < p) {
        float2 g_sum = make_float2(0.0f, 0.0f);
        int64_t g_offset = rank * p + i;
        
        #pragma unroll
        for (int k = 0; k < world_size; ++k) {
            const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_g_ptrs.ptrs[k]);
            const __nv_bfloat162* g_ptr = reinterpret_cast<const __nv_bfloat162*>(&src[g_offset]);
            __nv_bfloat162 g_val = *g_ptr;
            float2 g_f = __bfloat1622float2(g_val);
            g_sum.x += g_f.x;
            g_sum.y += g_f.y;
        }
        
        float2 g;
        g.x = g_sum.x / world_size;
        g.y = g_sum.y / world_size;

        __nv_bfloat162 p_val2 = *reinterpret_cast<const __nv_bfloat162*>(&local_param[i]);
        __nv_bfloat162 m_val2 = *reinterpret_cast<const __nv_bfloat162*>(&local_m[i]);
        __nv_bfloat162 v_val2 = *reinterpret_cast<const __nv_bfloat162*>(&local_v[i]);

        float2 p_val = __bfloat1622float2(p_val2);
        float2 m_val = __bfloat1622float2(m_val2);
        float2 v_val = __bfloat1622float2(v_val2);

        m_val.x = m_val.x * beta1 + g.x * (1.0f - beta1);
        m_val.y = m_val.y * beta1 + g.y * (1.0f - beta1);

        v_val.x = v_val.x * beta2 + g.x * g.x * (1.0f - beta2);
        v_val.y = v_val.y * beta2 + g.y * g.y * (1.0f - beta2);

        float m_hat_x = m_val.x / bc1;
        float m_hat_y = m_val.y / bc1;

        float v_hat_x = v_val.x / bc2;
        float v_hat_y = v_val.y / bc2;

        float denom_x = sqrtf(v_hat_x) + eps;
        float denom_y = sqrtf(v_hat_y) + eps;

        float new_p_x = p_val.x - lr * ((m_hat_x / denom_x) + p_val.x * weight_decay);
        float new_p_y = p_val.y - lr * ((m_hat_y / denom_y) + p_val.y * weight_decay);

        __nv_bfloat162 out_p2 = __floats2bfloat162_rn(new_p_x, new_p_y);
        __nv_bfloat162 out_m2 = __floats2bfloat162_rn(m_val.x, m_val.y);
        __nv_bfloat162 out_v2 = __floats2bfloat162_rn(v_val.x, v_val.y);

        *reinterpret_cast<__nv_bfloat162*>(&out_param[i]) = out_p2;
        *reinterpret_cast<__nv_bfloat162*>(&out_m[i]) = out_m2;
        *reinterpret_cast<__nv_bfloat162*>(&out_v[i]) = out_v2;
    }
}

__global__ void fused_rs_adamw_kernel_scalar(
    PeerPtrs peer_g_ptrs,
    const __nv_bfloat16* __restrict__ local_param,
    const __nv_bfloat16* __restrict__ local_m,
    const __nv_bfloat16* __restrict__ local_v,
    __nv_bfloat16* __restrict__ out_param,
    __nv_bfloat16* __restrict__ out_m,
    __nv_bfloat16* __restrict__ out_v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2,
    int64_t p, int world_size, int rank
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < p) {
        float g_sum = 0.0f;
        int64_t g_offset = rank * p + idx;
        
        #pragma unroll
        for (int k = 0; k < world_size; ++k) {
            const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_g_ptrs.ptrs[k]);
            g_sum += __bfloat162float(src[g_offset]);
        }
        float g = g_sum / world_size;

        float p_val = __bfloat162float(local_param[idx]);
        float m_val = __bfloat162float(local_m[idx]);
        float v_val = __bfloat162float(local_v[idx]);

        m_val = m_val * beta1 + g * (1.0f - beta1);
        v_val = v_val * beta2 + g * g * (1.0f - beta2);

        float m_hat = m_val / bc1;
        float v_hat = v_val / bc2;

        float denom = sqrtf(v_hat) + eps;
        
        float update = (m_hat / denom) + p_val * weight_decay;
        float new_p = p_val - lr * update;

        out_param[idx] = __float2bfloat16(new_p);
        out_m[idx]     = __float2bfloat16(m_val);
        out_v[idx]     = __float2bfloat16(v_val);
    }
}

void run_all_gather(
    std::vector<int64_t> peer_ptrs_int,
    torch::Tensor full_flat,
    int64_t p,
    int world_size
) {
    TORCH_CHECK(world_size <= 8, "World size > 8 not supported by this optimized path");
    PeerPtrs ptrs;
    for (int i = 0; i < world_size; ++i) {
        ptrs.ptrs[i] = reinterpret_cast<const void*>(peer_ptrs_int[i]);
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (p % 8 == 0) {
        int64_t p_vec = p / 8;
        int threads = 256;
        int blocks = (p_vec * world_size + threads - 1) / threads;
        all_gather_kernel_vec8<uint4><<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<uint4*>(full_flat.data_ptr()),
            p_vec,
            world_size
        );
    } else {
        int threads = 256;
        int blocks = (p * world_size + threads - 1) / threads;
        all_gather_kernel_scalar<<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<__nv_bfloat16*>(full_flat.data_ptr()),
            p,
            world_size
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_fused_rs_adamw(
    std::vector<int64_t> peer_g_ptrs_int,
    torch::Tensor local_param,
    torch::Tensor local_m,
    torch::Tensor local_v,
    torch::Tensor out_param,
    torch::Tensor out_m,
    torch::Tensor out_v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int step,
    int64_t p, int world_size, int rank
) {
    PeerPtrs ptrs;
    for (int i = 0; i < world_size; ++i) {
        ptrs.ptrs[i] = reinterpret_cast<const void*>(peer_g_ptrs_int[i]);
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    float bc1 = 1.0f - std::pow(beta1, (float)step);
    float bc2 = 1.0f - std::pow(beta2, (float)step);

    if (p % 2 == 0) {
        int64_t p_vec = p / 2;
        int threads = 256;
        int blocks = (p_vec + threads - 1) / threads;
        fused_rs_adamw_kernel_vec2<<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<const __nv_bfloat16*>(local_param.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(local_m.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(local_v.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_param.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_m.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_v.data_ptr()),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2,
            p, world_size, rank
        );
    } else {
        int threads = 256;
        int blocks = (p + threads - 1) / threads;
        fused_rs_adamw_kernel_scalar<<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<const __nv_bfloat16*>(local_param.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(local_m.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(local_v.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_param.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_m.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out_v.data_ptr()),
            lr, beta1, beta2, eps, weight_decay, bc1, bc2,
            p, world_size, rank
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_all_gather", &run_all_gather, "Custom all-gather using UVA");
    m.def("run_fused_rs_adamw", &run_fused_rs_adamw, "Fused Reduce-Scatter and AdamW");
}
"""

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_custom_e2e_bf16", CUDA_SRC)
    return _ext


class _Workspace:
    def __init__(self, p: int, world_size: int, dtype: torch.dtype, device: torch.device, param_shapes):
        self.p = p
        self.world_size = world_size
        
        # Buffer for input parameter shard to distribute globally 
        self.symm_param_shard = symm_mem.empty(p, dtype=dtype, device=device)
        self.hdl_param = symm_mem.rendezvous(self.symm_param_shard, dist.group.WORLD)
        self.peer_ptrs = [int(self.hdl_param.buffer_ptrs[i]) for i in range(world_size)]
        
        # Pre-allocated gathered tensor 
        self.full_flat = torch.empty(world_size * p, dtype=dtype, device=device)
        
        # Buffer holding locally computed backward gradients (will be reduce-scattered)
        self.symm_full_g = symm_mem.empty(world_size * p, dtype=dtype, device=device)
        self.hdl_g = symm_mem.rendezvous(self.symm_full_g, dist.group.WORLD)
        self.peer_g_ptrs = [int(self.hdl_g.buffer_ptrs[i]) for i in range(world_size)]
        
        # Cached dummy templates for structural unflattening bounds checks
        self.templates = [torch.zeros(shape, dtype=dtype, device=device) for shape in param_shapes]

_workspace = None

def _get_workspace(p: int, world_size: int, dtype: torch.dtype, device: torch.device, param_shapes):
    global _workspace
    if _workspace is None:
        _workspace = _Workspace(p, world_size, dtype, device, param_shapes)
    return _workspace


def solution(
    X_local: Tensor,
    y_local: Tensor,
    flat_param_shard: Tensor,
    param_shapes: Sequence[tuple[int, ...]],
    exp_avg_shard: Tensor,
    exp_avg_sq_shard: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    p = flat_param_shard.numel()
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    ws = _get_workspace(p, world_size, flat_param_shard.dtype, flat_param_shard.device, param_shapes)

    # 1. Provide local parameter state to peer UVA memory space
    ws.symm_param_shard.copy_(flat_param_shard)
    ws.hdl_param.barrier(channel=0)

    # 2. Fast Custom All-Gather
    ext.run_all_gather(ws.peer_ptrs, ws.full_flat, p, world_size)

    # 3. Unflatten, Forward, Backward
    params_f = _unflatten_dense_tensors(ws.full_flat, ws.templates)
    params = [t.detach().requires_grad_(True) for t in params_f]

    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    # 4. Flatten Gradients out to peer-accessible Symmetric Memory Layout
    torch.cat([x.grad.reshape(-1) for x in params], out=ws.symm_full_g)
    ws.hdl_g.barrier(channel=0)

    # 5. Fused Reduce-Scatter and AdamW Output Tensors
    out_param = torch.empty_like(flat_param_shard)
    out_m = torch.empty_like(exp_avg_shard)
    out_v = torch.empty_like(exp_avg_sq_shard)

    ext.run_fused_rs_adamw(
        ws.peer_g_ptrs,
        flat_param_shard,
        exp_avg_shard,
        exp_avg_sq_shard,
        out_param, out_m, out_v,
        lr, beta1, beta2, eps, weight_decay,
        step, p, world_size, rank
    )

    return out_param, out_m, out_v

__all__ = ["solution"]