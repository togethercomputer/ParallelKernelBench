"""
Strategy:
- **Device-Side Communication & UVA**: Replaced `all_reduce`, `broadcast`, and `all_gather` with custom direct-memory access kernels using `torch.distributed._symmetric_memory`. Each GPU directly reads peer gradients and weights over NVLink, bypassing NCCL launch and buffer overheads.
- **Fused Reduce-Scatter and Adam**: Instead of a full `all_reduce` followed by slicing and math, a single custom kernel pulls gradient partitions from peers, averages them, applies the Adam step, and updates the local weight partition directly in one pass.
- **Compute-Communication Overlap & Alignment**: By switching from push-based collectives to pull-based UVA kernels, we remove intermediate buffer allocations (`gathered`, `w_part`, `flat_g` chunks) and implicitly align memory reads with arithmetic. Symmetric memory barriers strictly separate the read/write phases to ensure consistency.
"""

from __future__ import annotations

import math

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

struct PtrArray {
    const void* ptrs[8];
};

template <typename scalar_t>
__global__ void broadcast_kernel(const scalar_t* __restrict__ root_w, scalar_t* __restrict__ local_w, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        local_w[idx] = root_w[idx];
    }
}

template <typename scalar_t>
__global__ void all_gather_kernel(
    PtrArray peer_w_ptrs,
    scalar_t* __restrict__ local_w,
    int world_size,
    int64_t part_size,
    int rank
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = world_size * part_size;
    if (idx < total_elements) {
        int peer = idx / part_size;
        if (peer != rank) {
            const scalar_t* peer_w = reinterpret_cast<const scalar_t*>(peer_w_ptrs.ptrs[peer]);
            local_w[idx] = peer_w[idx];
        }
    }
}

template <typename scalar_t, typename mom_t>
__global__ void reduce_scatter_adam_kernel(
    PtrArray peer_g_ptrs,
    scalar_t* __restrict__ local_w,
    mom_t* __restrict__ m_part,
    mom_t* __restrict__ v_part,
    int world_size,
    int64_t part_size,
    int64_t offset,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < part_size) {
        float g_sum = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < world_size; i++) {
            const scalar_t* peer_g = reinterpret_cast<const scalar_t*>(peer_g_ptrs.ptrs[i]);
            g_sum += static_cast<float>(peer_g[offset + idx]);
        }
        float g = g_sum / world_size;
        
        float m = static_cast<float>(m_part[idx]);
        float v = static_cast<float>(v_part[idx]);
        
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
        
        m_part[idx] = static_cast<mom_t>(m);
        v_part[idx] = static_cast<mom_t>(v);
        
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        
        float w = static_cast<float>(local_w[offset + idx]);
        w -= lr * m_hat / (sqrtf(v_hat) + eps);
        local_w[offset + idx] = static_cast<scalar_t>(w);
    }
}

void zero1_step(
    int rank,
    int world_size,
    std::vector<int64_t> g_ptrs,
    std::vector<int64_t> w_ptrs,
    torch::Tensor local_w,
    torch::Tensor m_part,
    torch::Tensor v_part,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2
) {
    int64_t total_elements = local_w.numel();
    if (total_elements == 0) return;
    
    int64_t part_size = total_elements / world_size;
    int64_t offset = rank * part_size;

    PtrArray p_g, p_w;
    for (int i = 0; i < world_size; i++) {
        p_g.ptrs[i] = reinterpret_cast<const void*>(g_ptrs[i]);
        p_w.ptrs[i] = reinterpret_cast<const void*>(w_ptrs[i]);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    int blocks_rs = (part_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, local_w.scalar_type(), "reduce_scatter_adam", ([&] {
        using scalar_t_w = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, m_part.scalar_type(), "reduce_scatter_adam_mom", ([&] {
            reduce_scatter_adam_kernel<scalar_t_w, scalar_t><<<blocks_rs, threads, 0, stream>>>(
                p_g,
                local_w.data_ptr<scalar_t_w>(),
                m_part.data_ptr<scalar_t>(),
                v_part.data_ptr<scalar_t>(),
                world_size,
                part_size,
                offset,
                lr, beta1, beta2, eps, bc1, bc2
            );
        }));
    }));
}

void all_gather_step(
    int rank,
    int world_size,
    std::vector<int64_t> w_ptrs,
    torch::Tensor local_w
) {
    int64_t total_elements = local_w.numel();
    if (total_elements == 0) return;
    
    int64_t part_size = total_elements / world_size;

    PtrArray p_w;
    for (int i = 0; i < world_size; i++) {
        p_w.ptrs[i] = reinterpret_cast<const void*>(w_ptrs[i]);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    int blocks_ag = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, local_w.scalar_type(), "all_gather", ([&] {
        all_gather_kernel<scalar_t><<<blocks_ag, threads, 0, stream>>>(
            p_w,
            local_w.data_ptr<scalar_t>(),
            world_size,
            part_size,
            rank
        );
    }));
}

void bcast_step(
    int64_t root_ptr,
    torch::Tensor local_w
) {
    int64_t numel = local_w.numel();
    if (numel == 0) return;
    
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, local_w.scalar_type(), "broadcast", ([&] {
        broadcast_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const scalar_t*>(root_ptr),
            local_w.data_ptr<scalar_t>(),
            numel
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("zero1_step", &zero1_step, "ZeRO-1 reduce-scatter + Adam");
    m.def("all_gather_step", &all_gather_step, "ZeRO-1 all-gather");
    m.def("bcast_step", &bcast_step, "Broadcast step");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("zero1_fused_uva_ext", CUDA_SRC)
    return _ext

_symm_cache = None
def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype:
            return c["w"], c["g"], c["hdl_w"], c["hdl_g"]
        else:
            _symm_cache = None

    symm_w = symm_mem.empty(n, device=device, dtype=dtype)
    symm_g = symm_mem.empty(n, device=device, dtype=dtype)
    
    hdl_w = symm_mem.rendezvous(symm_w, dist.group.WORLD)
    hdl_g = symm_mem.rendezvous(symm_g, dist.group.WORLD)
    
    _symm_cache = {
        "n": n, "dtype": dtype, 
        "w": symm_w, "g": symm_g,
        "hdl_w": hdl_w, "hdl_g": hdl_g
    }
    return symm_w, symm_g, hdl_w, hdl_g


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

    # Pre-compile on rank 0 to prevent compile races
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    templates = [W1, b1, W2, b2]
    flat_p = _flatten_dense_tensors(templates)
    numel = flat_p.numel()
    
    part = exp_avg_part.numel()
    assert numel == part * world_size

    symm_w, symm_g, hdl_w, hdl_g = _get_symm_state(numel, flat_p.dtype, flat_p.device)
    
    # Broadcast weights from Rank 0 using UVA symmetric memory
    if rank == 0:
        symm_w.copy_(flat_p)
    hdl_w.barrier(channel=0)
    
    if rank != 0:
        ext.bcast_step(hdl_w.buffer_ptrs[0], symm_w)
    hdl_w.barrier(channel=1)

    # Reconstruct required-grad parameters directly from symmetric memory to save allocation
    param_views = _unflatten_dense_tensors(symm_w, templates)
    params = [t.detach().requires_grad_(True) for t in param_views]

    m_part = exp_avg_part.clone()
    v_part = exp_avg_sq_part.clone()

    # Forward & backward pass
    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    # Flatten computed gradients and write to symmetric gradient buffer
    flat_g = _flatten_dense_tensors([p.grad for p in params])
    symm_g.copy_(flat_g)
    hdl_g.barrier(channel=0)

    # Bias correction
    assert step >= 1
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    # Fused Reduce-Scatter + Adam: directly pulls peer gradients via UVA and updates the local weight partition
    ext.zero1_step(
        rank, world_size, 
        list(hdl_g.buffer_ptrs), list(hdl_w.buffer_ptrs), 
        symm_w, m_part, v_part,
        lr, beta1, beta2, eps, bc1, bc2
    )
    
    hdl_w.barrier(channel=2)

    # All-Gather: pull updated peer partitions directly into the local symm_w buffer
    ext.all_gather_step(rank, world_size, list(hdl_w.buffer_ptrs), symm_w)

    hdl_w.barrier(channel=3)

    out_params = _unflatten_dense_tensors(symm_w, templates)
    return (*out_params, m_part, v_part)

__all__ = ["solution"]