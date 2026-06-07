"""
Strategy:
1. P2P All-Gather: We allocate a symmetric buffer for the full model parameters. Each rank writes its param shard, then we use a single fast custom CUDA kernel to directly fetch peers' shards over NVLink.
2. Forward/Backward Pass: Executed using PyTorch native ops directly on the unflattened views of the symmetric buffer.
3. Fused Reduce-Scatter & AdamW: We allocate a symmetric buffer for the full gradients. After PyTorch computes the full flat gradient, each rank writes it to symmetric memory. A second custom CUDA kernel reads each rank's assigned gradient slice directly from all peers, performs the reduction, and natively applies the AdamW update directly into the newly allocated output param/momentum shards in a single pass. This avoids multiple read/writes and intermediate gradient clones.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <algorithm>

// Exactly emulate PyTorch's elementwise eager bfloat16 truncations
__device__ __forceinline__ float trunc_bf16(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__global__ void all_gather_kernel(
    const long long* __restrict__ peer_full_flats,
    int64_t p,
    int world_size,
    int rank
) {
    int64_t total = p * world_size;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total; 
         idx += (int64_t)gridDim.x * blockDim.x) {
        
        int target_rank = idx / p;
        if (target_rank != rank) {
            __nv_bfloat16* local_ptr = (__nv_bfloat16*)peer_full_flats[rank];
            const __nv_bfloat16* remote_ptr = (const __nv_bfloat16*)peer_full_flats[target_rank];
            local_ptr[idx] = remote_ptr[idx];
        }
    }
}

__global__ void rs_adamw_kernel(
    const long long* __restrict__ peer_grads,
    __nv_bfloat16* __restrict__ theta_out,
    __nv_bfloat16* __restrict__ m_out,
    __nv_bfloat16* __restrict__ v_out,
    const __nv_bfloat16* __restrict__ theta_in,
    const __nv_bfloat16* __restrict__ m_in,
    const __nv_bfloat16* __restrict__ v_in,
    int64_t p,
    int world_size,
    int rank,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; 
         i < p; 
         i += (int64_t)gridDim.x * blockDim.x) {
        
        int64_t offset = rank * p + i;
        float g_sum = 0.0f;
        
        #pragma unroll
        for (int w = 0; w < world_size; ++w) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)peer_grads[w];
            g_sum += __bfloat162float(src[offset]);
        }
        
        float g = g_sum / world_size;
        g = trunc_bf16(g);

        float m_val = __bfloat162float(m_in[i]);
        float v_val = __bfloat162float(v_in[i]);
        float orig_theta = __bfloat162float(theta_in[i]);
        float theta_val = orig_theta;

        m_val = trunc_bf16(m_val * beta1);
        m_val = trunc_bf16(m_val + g * (1.0f - beta1));

        v_val = trunc_bf16(v_val * beta2);
        v_val = trunc_bf16(v_val + g * g * (1.0f - beta2));

        float m_hat = trunc_bf16(m_val / bc1);
        float v_hat = trunc_bf16(v_val / bc2);
        float denom = trunc_bf16(trunc_bf16(sqrtf(v_hat)) + eps);

        float step_term = trunc_bf16(m_hat / denom);
        theta_val = trunc_bf16(theta_val - lr * step_term);
        theta_val = trunc_bf16(theta_val - lr * weight_decay * orig_theta);

        m_out[i] = __float2bfloat16(m_val);
        v_out[i] = __float2bfloat16(v_val);
        theta_out[i] = __float2bfloat16(theta_val);
    }
}

void launch_all_gather(
    torch::Tensor peer_ptrs,
    int64_t p,
    int world_size,
    int rank
) {
    int64_t total = p * world_size;
    if (total == 0) return;
    int threads = 512;
    int blocks = std::max(1, std::min((int)((total + threads - 1) / threads), 65535));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    all_gather_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        p,
        world_size,
        rank
    );
}

void launch_rs_adamw(
    torch::Tensor peer_grads_ptrs,
    torch::Tensor theta_out,
    torch::Tensor m_out,
    torch::Tensor v_out,
    torch::Tensor theta_in,
    torch::Tensor m_in,
    torch::Tensor v_in,
    int64_t p,
    int world_size,
    int rank,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2
) {
    if (p == 0) return;
    int threads = 512;
    int blocks = std::max(1, std::min((int)((p + threads - 1) / threads), 65535));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    rs_adamw_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)peer_grads_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)theta_out.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)m_out.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)v_out.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)theta_in.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)m_in.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)v_in.data_ptr<at::BFloat16>(),
        p,
        world_size,
        rank,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bc1,
        bc2
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_all_gather", &launch_all_gather, "Custom P2P all gather");
    m.def("launch_rs_adamw", &launch_rs_adamw, "Fused ReduceScatter and AdamW");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_step_e2e_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(p: int, world_size: int, dtype: torch.dtype, device: torch.device):
    key = (p, world_size, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]

    full_flat_buf = symm_mem.empty(world_size * p, device=device, dtype=dtype)
    full_flat_hdl = symm_mem.rendezvous(full_flat_buf, dist.group.WORLD)
    full_flat_ptrs = torch.tensor(full_flat_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    sym_grad_buf = symm_mem.empty(world_size * p, device=device, dtype=dtype)
    sym_grad_hdl = symm_mem.rendezvous(sym_grad_buf, dist.group.WORLD)
    sym_grad_ptrs = torch.tensor(sym_grad_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (full_flat_buf, full_flat_hdl, full_flat_ptrs, sym_grad_buf, sym_grad_hdl, sym_grad_ptrs)
    _symm_cache[key] = res
    return res

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
    
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert step >= 1

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    flat_param_shard = flat_param_shard.contiguous()
    exp_avg_shard = exp_avg_shard.contiguous()
    exp_avg_sq_shard = exp_avg_sq_shard.contiguous()
    
    p = flat_param_shard.numel()
    dtype = flat_param_shard.dtype
    device = flat_param_shard.device
    
    full_flat, full_flat_hdl, full_flat_ptrs, sym_grad, sym_grad_hdl, sym_grad_ptrs = \
        _get_symm_state(p, world_size, dtype, device)
        
    if rank == 0:
        _get_ext()
    dist.barrier()
    
    # 1. P2P All-Gather parameter chunks
    full_flat[rank * p : (rank + 1) * p].copy_(flat_param_shard)
    full_flat_hdl.barrier(channel=0)
    
    _get_ext().launch_all_gather(full_flat_ptrs, p, world_size, rank)
    
    # 2. PyTorch Forward & Backward Pass on unflattened views
    templates = [torch.empty(shape, dtype=dtype, device=device) for shape in param_shapes]
    params_f = _unflatten_dense_tensors(full_flat, templates)
    params = [t.detach().requires_grad_(True) for t in params_f]

    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()
    
    # 3. Share gradients across peers
    flat_g = _flatten_dense_tensors([x.grad for x in params])
    sym_grad.copy_(flat_g)
    sym_grad_hdl.barrier(channel=0)
    
    theta_out = torch.empty_like(flat_param_shard)
    m_out = torch.empty_like(exp_avg_shard)
    v_out = torch.empty_like(exp_avg_sq_shard)
    
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    
    # 4. Read gradient chunks, Reduce, and Step AdamW seamlessly 
    _get_ext().launch_rs_adamw(
        sym_grad_ptrs,
        theta_out, m_out, v_out,
        flat_param_shard, exp_avg_shard, exp_avg_sq_shard,
        p, world_size, rank,
        float(lr), float(beta1), float(beta2), float(eps), float(weight_decay), float(bc1), float(bc2)
    )
    
    sym_grad_hdl.barrier(channel=1)
    
    return theta_out, m_out, v_out

__all__ = ["solution"]