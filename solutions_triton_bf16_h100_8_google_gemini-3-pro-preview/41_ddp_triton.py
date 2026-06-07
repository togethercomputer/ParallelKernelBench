"""
Strategy:
- **Device-side communication**: Replaced high-level collectives with direct peer-to-peer memory access over NVLink using `torch.distributed._symmetric_memory` and UVA pointers.
- **Compute-communication fusion**: The gradient average and Adam optimizer step are fused into a single custom CUDA kernel. Each GPU fetches remote gradients directly from peers on-the-fly, hiding interconnect latency behind Adam's math operations.
- **Zero-allocation hot path**: Model parameters and Adam states are maintained as views into pre-allocated symmetric memory buffers. This eliminates PyTorch's flattened buffer allocations during training.
"""

from __future__ import annotations

import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _unflatten_dense_tensors
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

__all__ = ["solution"]

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

struct PtrArray {
    const void* ptrs[8];
};

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template<typename T>
__global__ void uva_copy_kernel(T* dst, const T* src, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

template<typename T>
__global__ void fused_allreduce_adam_kernel(
    PtrArray grad_ptrs,
    T* flat_grad,
    T* flat_params,
    T* flat_m,
    T* flat_v,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float bc1,
    float bc2,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum_g = 0.0f;
        #pragma unroll
        for (int i = 0; i < world_size; i++) {
            sum_g += to_float(reinterpret_cast<const T*>(grad_ptrs.ptrs[i])[idx]);
        }
        float avg_g = sum_g / world_size;
        
        flat_grad[idx] = from_float<T>(avg_g);
        
        float m = to_float(flat_m[idx]);
        float v = to_float(flat_v[idx]);
        
        m = m * beta1 + avg_g * (1.0f - beta1);
        v = v * beta2 + avg_g * avg_g * (1.0f - beta2);
        
        flat_m[idx] = from_float<T>(m);
        flat_v[idx] = from_float<T>(v);
        
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float denom = sqrtf(v_hat) + eps;
        
        float p = to_float(flat_params[idx]);
        p -= lr * (m_hat / denom);
        flat_params[idx] = from_float<T>(p);
    }
}

void uva_copy(torch::Tensor dst, int64_t src_ptr, int64_t n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (dst.scalar_type() == torch::kBFloat16) {
        uva_copy_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(src_ptr),
            n
        );
    } else {
        uva_copy_kernel<float><<<blocks, threads, 0, stream>>>(
            dst.data_ptr<float>(),
            reinterpret_cast<const float*>(src_ptr),
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_allreduce_adam(
    std::vector<int64_t> grad_ptr_ints,
    torch::Tensor flat_grad,
    torch::Tensor flat_params,
    torch::Tensor flat_m,
    torch::Tensor flat_v,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float bc1,
    float bc2,
    int world_size,
    int64_t n
) {
    TORCH_CHECK(world_size <= 8, "world_size must be <= 8 to fit in PtrArray");
    PtrArray grad_ptrs;
    for (int i = 0; i < world_size; i++) {
        grad_ptrs.ptrs[i] = reinterpret_cast<const void*>(grad_ptr_ints[i]);
    }
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (flat_params.scalar_type() == torch::kBFloat16) {
        fused_allreduce_adam_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            grad_ptrs,
            reinterpret_cast<__nv_bfloat16*>(flat_grad.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(flat_params.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(flat_m.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(flat_v.data_ptr()),
            beta1, beta2, lr, eps, bc1, bc2, world_size, n
        );
    } else {
        fused_allreduce_adam_kernel<float><<<blocks, threads, 0, stream>>>(
            grad_ptrs,
            flat_grad.data_ptr<float>(),
            flat_params.data_ptr<float>(),
            flat_m.data_ptr<float>(),
            flat_v.data_ptr<float>(),
            beta1, beta2, lr, eps, bc1, bc2, world_size, n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_copy", &uva_copy, "UVA remote copy kernel");
    m.def("fused_allreduce_adam", &fused_allreduce_adam, "Fused peer All-Reduce and Adam kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_ddp_ext", CUDA_SRC)
    return _ext


_symm_cache = None

def _get_symm_state(n_params: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n_params and c["dtype"] == dtype:
            return c["symm_bcast"], c["hdl_bcast"], c["symm_grad"], c["hdl_grad"]

    # One big contiguous buffer for params, m, v broadcasts
    symm_bcast = symm_mem.empty(3 * n_params, device=device, dtype=dtype)
    hdl_bcast = symm_mem.rendezvous(symm_bcast, dist.group.WORLD)
    
    # Symmetrical buffer specifically for local gradients
    symm_grad = symm_mem.empty(n_params, device=device, dtype=dtype)
    hdl_grad = symm_mem.rendezvous(symm_grad, dist.group.WORLD)

    _symm_cache = {
        "n": n_params,
        "dtype": dtype,
        "symm_bcast": symm_bcast,
        "hdl_bcast": hdl_bcast,
        "symm_grad": symm_grad,
        "hdl_grad": hdl_grad
    }
    return symm_bcast, hdl_bcast, symm_grad, hdl_grad


def solution(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_W1: Tensor,
    exp_avg_b1: Tensor,
    exp_avg_W2: Tensor,
    exp_avg_b2: Tensor,
    exp_avg_sq_W1: Tensor,
    exp_avg_sq_b1: Tensor,
    exp_avg_sq_W2: Tensor,
    exp_avg_sq_b2: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, ...]:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    
    params = [W1, b1, W2, b2]
    exp_avg = [exp_avg_W1, exp_avg_b1, exp_avg_W2, exp_avg_b2]
    exp_avg_sq = [exp_avg_sq_W1, exp_avg_sq_b1, exp_avg_sq_W2, exp_avg_sq_b2]
    
    total_params = sum(p.numel() for p in params)
    dtype = W1.dtype
    device = W1.device
    
    symm_bcast, hdl_bcast, symm_grad, hdl_grad = _get_symm_state(total_params, dtype, device)
    
    with torch.no_grad():
        # Pack inputs on rank 0 for broadcast into the shared symmetric buffer
        if rank == 0:
            offset = 0
            for t in params + exp_avg + exp_avg_sq:
                symm_bcast[offset:offset+t.numel()].copy_(t.view(-1))
                offset += t.numel()
                
        hdl_bcast.barrier(channel=0)
        
        # Read the entire combined state from rank 0's NVLink exposed UVA pointer
        if rank != 0:
            rank0_ptr = int(hdl_bcast.buffer_ptrs[0])
            ext.uva_copy(symm_bcast, rank0_ptr, 3 * total_params)
            
        hdl_bcast.barrier(channel=0)
        
        flat_params = symm_bcast[:total_params]
        flat_m = symm_bcast[total_params:2*total_params]
        flat_v = symm_bcast[2*total_params:]
        
        broadcast_params = _unflatten_dense_tensors(flat_params, params)
        out_exp_avg = list(_unflatten_dense_tensors(flat_m, exp_avg))
        out_exp_avg_sq = list(_unflatten_dense_tensors(flat_v, exp_avg_sq))
    
    # Detach into standard graph leaves that reference our shared memory states
    out_params = [t.detach().requires_grad_(True) for t in broadcast_params]
    
    # Forward and backward directly updating out_params leaves
    h = F.relu(F.linear(X_local, out_params[0], out_params[1]))
    out = F.linear(h, out_params[2], out_params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()
    
    with torch.no_grad():
        # Flatten gradient segments into the symmetric gradient buffer
        offset = 0
        for p in out_params:
            g = p.grad
            symm_grad[offset:offset+g.numel()].copy_(g.view(-1))
            offset += g.numel()
            
        # Ensure all local gradient buffers are ready for remote reading
        hdl_grad.barrier(channel=0)
        
        bc1 = 1.0 - math.pow(beta1, step)
        bc2 = 1.0 - math.pow(beta2, step)
        grad_ptrs = [int(hdl_grad.buffer_ptrs[i]) for i in range(world_size)]
        
        # Fuse All-Reduce with Adam: SMs pull peer gradients synchronously over NVLink
        # while locally streaming computations avoiding any further intermediate allocations.
        ext.fused_allreduce_adam(
            grad_ptrs,
            symm_grad,  # Written out with true averaged gradient
            flat_params,
            flat_m,
            flat_v,
            beta1, beta2, lr, eps, bc1, bc2,
            world_size, total_params
        )
        
        # Prevent any rank from overwriting remote states before others finish reading
        hdl_grad.barrier(channel=0)
        
        # Hydrate p.grad tensors with actual aggregated gradients to respect API expectations
        avg_grads = _unflatten_dense_tensors(symm_grad, out_params)
        for p, g in zip(out_params, avg_grads):
            if p.grad is not None:
                p.grad.copy_(g)
        
    return tuple(list(out_params) + out_exp_avg + out_exp_avg_sq)