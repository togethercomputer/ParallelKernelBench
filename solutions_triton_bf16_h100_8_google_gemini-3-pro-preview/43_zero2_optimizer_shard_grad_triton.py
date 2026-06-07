"""
Strategy:
- **Device-Side Communication:** Replaced PyTorch's `dist.broadcast`, `dist.reduce_scatter_tensor`, and `dist.all_gather_into_tensor` with direct UVA loads and stores over NVLink using `torch.distributed._symmetric_memory`.
- **Compute–Communication Fusion:** Developed a unified C++ CUDA kernel that pulls gradients directly from peer symmetric buffers (fusing reduce-scatter), applies the Adam optimizer math on local state, and immediately pushes updated parameter slices to all peers' symmetric buffers (fusing all-gather). This completely eliminates multiple intermediate gradient/parameter buffers and collective kernel overhead.
- **Synchronized P2P:** Utilizes direct device-side barriers (`hdl.barrier(channel=0)`) to cleanly sequence operations on the default stream without CPU launch bottlenecks, maximizing compute-communication overlap.
"""

from __future__ import annotations

import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

template<typename T>
__global__ void bcast_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

void broadcast_symm(
    int64_t src_ptr,
    torch::Tensor dst,
    int64_t n
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (dst.scalar_type() == torch::kBFloat16) {
        bcast_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(src_ptr)),
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
            n
        );
    } else {
        bcast_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(static_cast<uintptr_t>(src_ptr)),
            reinterpret_cast<float*>(dst.data_ptr()),
            n
        );
    }
}

struct PtrArray {
    const void* ptrs[8];
};
struct MutablePtrArray {
    void* ptrs[8];
};

template<typename T>
__global__ void fused_reduce_scatter_adam_push_kernel(
    PtrArray g_ptrs,
    MutablePtrArray p_ptrs,
    const T* __restrict__ w_part_in,
    float* __restrict__ m_part,
    float* __restrict__ v_part,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2,
    int64_t offset,
    int64_t part_size,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < part_size) {
        float g_sum = 0.0f;
        
        for (int r = 0; r < world_size; ++r) {
            const T* g_ptr = reinterpret_cast<const T*>(g_ptrs.ptrs[r]);
            float g;
            if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                g = __bfloat162float(g_ptr[offset + idx]);
            } else {
                g = g_ptr[offset + idx];
            }
            g_sum += g;
        }
        
        float g_avg = g_sum / world_size;
        
        float m = m_part[idx];
        float v = v_part[idx];
        
        m = beta1 * m + (1.0f - beta1) * g_avg;
        v = beta2 * v + (1.0f - beta2) * g_avg * g_avg;
        
        m_part[idx] = m;
        v_part[idx] = v;
        
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        
        float w;
        if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            w = __bfloat162float(w_part_in[idx]);
        } else {
            w = w_part_in[idx];
        }
        
        w = w - lr * m_hat / (sqrtf(v_hat) + eps);
        
        T w_out;
        if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            w_out = __float2bfloat16(w);
        } else {
            w_out = w;
        }
        
        for (int r = 0; r < world_size; ++r) {
            T* p_ptr = reinterpret_cast<T*>(p_ptrs.ptrs[r]);
            p_ptr[offset + idx] = w_out;
        }
    }
}

void fused_zero2_step(
    std::vector<int64_t> g_ptrs_int,
    std::vector<int64_t> p_ptrs_int,
    torch::Tensor w_part,
    torch::Tensor m_part,
    torch::Tensor v_part,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2,
    int64_t offset,
    int64_t part_size,
    int world_size
) {
    PtrArray g_ptrs;
    MutablePtrArray p_ptrs;
    for (int i = 0; i < world_size; ++i) {
        g_ptrs.ptrs[i] = reinterpret_cast<const void*>(static_cast<uintptr_t>(g_ptrs_int[i]));
        p_ptrs.ptrs[i] = reinterpret_cast<void*>(static_cast<uintptr_t>(p_ptrs_int[i]));
    }
    
    const int threads = 256;
    const int blocks = (part_size + threads - 1) / threads;
    if (blocks == 0) return;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (w_part.scalar_type() == torch::kBFloat16) {
        fused_reduce_scatter_adam_push_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            g_ptrs, p_ptrs,
            reinterpret_cast<const __nv_bfloat16*>(w_part.data_ptr()),
            m_part.data_ptr<float>(),
            v_part.data_ptr<float>(),
            lr, beta1, beta2, eps, bc1, bc2,
            offset, part_size, world_size
        );
    } else {
        fused_reduce_scatter_adam_push_kernel<float><<<blocks, threads, 0, stream>>>(
            g_ptrs, p_ptrs,
            reinterpret_cast<const float*>(w_part.data_ptr()),
            m_part.data_ptr<float>(),
            v_part.data_ptr<float>(),
            lr, beta1, beta2, eps, bc1, bc2,
            offset, part_size, world_size
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_symm", &broadcast_symm, "UVA broadcast symmetric memory");
    m.def("fused_zero2_step", &fused_zero2_step, "Fused reduce-scatter, Adam, and push");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("zero2_fused_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(name: str, n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if name in _symm_cache:
        c = _symm_cache[name]
        if c["n"] == n and c["dtype"] == dtype:
            return c["buf"], c["hdl"]
    
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[name] = {"n": n, "dtype": dtype, "buf": buf, "hdl": hdl}
    return buf, hdl


def solution(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_part: Tensor,
    exp_avg_sq_part: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        _get_ext()
    dist.barrier()
    
    templates = [W1, b1, W2, b2]
    flat_p_temp = _flatten_dense_tensors(templates)
    
    # Persistent symmetric parameter buffer
    symm_p, hdl_p = _get_symm_state("p", flat_p_temp.numel(), flat_p_temp.dtype, flat_p_temp.device)
    
    # Broadcast replaced by Rank 0 init + UVA peer read
    if rank == 0:
        symm_p.copy_(flat_p_temp)
    hdl_p.barrier(channel=0)
    
    if rank != 0:
        _get_ext().broadcast_symm(int(hdl_p.buffer_ptrs[0]), symm_p, symm_p.numel())
    hdl_p.barrier(channel=0)
    
    param_views = _unflatten_dense_tensors(symm_p, templates)
    params = [t.detach().requires_grad_(True) for t in param_views]

    part = exp_avg_part.numel()
    assert symm_p.numel() == part * world_size
    assert exp_avg_part.dtype == torch.float32 and exp_avg_sq_part.dtype == torch.float32

    m_part = exp_avg_part.clone()
    v_part = exp_avg_sq_part.clone()

    # Forward + Backward
    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    flat_g = _flatten_dense_tensors([p.grad for p in params]).contiguous()
    
    # Persistent symmetric gradient buffer
    symm_g, hdl_g = _get_symm_state("g", flat_g.numel(), flat_g.dtype, flat_g.device)
    symm_g.copy_(flat_g)
    hdl_g.barrier(channel=0)

    start = rank * part
    w_part = symm_p[start : start + part]

    assert step >= 1
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    g_ptrs = [int(ptr) for ptr in hdl_g.buffer_ptrs][:world_size]
    p_ptrs = [int(ptr) for ptr in hdl_p.buffer_ptrs][:world_size]

    # Fused Reduce-Scatter, Adam Optimizer, and All-Gather directly manipulating peer buffers
    _get_ext().fused_zero2_step(
        g_ptrs, p_ptrs,
        w_part, m_part, v_part,
        lr, beta1, beta2, eps, bc1, bc2,
        start, part, world_size
    )

    # Sync to ensure all ranks have finished writing updated parameters to our symmetric buffer
    hdl_p.barrier(channel=0)

    out_params = _unflatten_dense_tensors(symm_p, templates)
    return (*out_params, m_part, v_part)

__all__ = ["solution"]