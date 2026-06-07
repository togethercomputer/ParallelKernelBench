"""
FSDP-style one step using symmetric memory all-gather + reduce-scatter with
a fused AdamW kernel. BF16 hot path on H100 with NVLink P2P.

Strategy:
- All-gather: each rank writes its shard into a symmetric buffer; peers read
  directly via UVA pointers in a custom CUDA kernel (one kernel, no NCCL).
- Forward/backward: keep using torch (cuBLAS GEMMs hit tensor cores) — small MLP.
- Reduce-scatter: each rank reads its slice from every peer's symmetric buffer
  and sums in-kernel; fused with AdamW update in a single kernel.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// All-gather: copy from all peers' shard buffer into the full output.
// full_out[r * p + i] = peer_buf[r][i]
__global__ void allgather_bf16_kernel(
    const long long* __restrict__ peer_ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t p
) {
    int r = blockIdx.y;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const __nv_bfloat16* src = (const __nv_bfloat16*)peer_ptrs[r];
    __nv_bfloat16* dst = out + (int64_t)r * p;
    for (; idx < p; idx += stride) {
        dst[idx] = src[idx];
    }
}

void launch_allgather_bf16(
    torch::Tensor peer_ptrs,
    torch::Tensor out,
    int64_t p,
    int world_size
) {
    const long long* d_ptrs = (const long long*)peer_ptrs.data_ptr<int64_t>();
    int threads = 256;
    int blocks_x = (int)((p + threads - 1) / threads);
    if (blocks_x > 512) blocks_x = 512;
    dim3 blocks(blocks_x, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allgather_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs,
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        world_size,
        p
    );
}

// Fused reduce-scatter + AdamW.
// peer_grad_ptrs: world_size pointers into per-rank flat-grad symmetric buffers
//   each of length world_size * p (bf16). This rank reads slice [rank*p:(rank+1)*p]
//   from every peer and sums (then div world_size).
// Updates m, v, theta in place (all bf16).
__global__ void fused_rs_adamw_bf16_kernel(
    const long long* __restrict__ peer_grad_ptrs,
    __nv_bfloat16* __restrict__ theta,
    __nv_bfloat16* __restrict__ m,
    __nv_bfloat16* __restrict__ v,
    int world_size,
    int rank,
    int64_t p,
    float inv_world_size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t off = (int64_t)rank * p;

    for (; idx < p; idx += stride) {
        float g = 0.0f;
        #pragma unroll 1
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)peer_grad_ptrs[r];
            g += __bfloat162float(src[off + idx]);
        }
        g *= inv_world_size;

        float th = __bfloat162float(theta[idx]);
        float mv = __bfloat162float(m[idx]);
        float vv = __bfloat162float(v[idx]);

        mv = beta1 * mv + (1.0f - beta1) * g;
        vv = beta2 * vv + (1.0f - beta2) * g * g;

        float m_hat = mv / bc1;
        float v_hat = vv / bc2;
        float denom = sqrtf(v_hat) + eps;

        float th_orig = th;
        th = th - lr * (m_hat / denom);
        th = th - lr * weight_decay * th_orig;

        theta[idx] = __float2bfloat16(th);
        m[idx]     = __float2bfloat16(mv);
        v[idx]     = __float2bfloat16(vv);
    }
}

void launch_fused_rs_adamw_bf16(
    torch::Tensor peer_grad_ptrs,
    torch::Tensor theta,
    torch::Tensor m,
    torch::Tensor v,
    int world_size,
    int rank,
    int64_t p,
    double inv_world_size,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    double bc1,
    double bc2
) {
    const long long* d_ptrs = (const long long*)peer_grad_ptrs.data_ptr<int64_t>();
    int threads = 256;
    int blocks = (int)((p + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_rs_adamw_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs,
        (__nv_bfloat16*)theta.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)m.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)v.data_ptr<at::BFloat16>(),
        world_size,
        rank,
        p,
        (float)inv_world_size,
        (float)lr,
        (float)beta1,
        (float)beta2,
        (float)eps,
        (float)weight_decay,
        (float)bc1,
        (float)bc2
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_allgather_bf16", &launch_allgather_bf16, "AG bf16");
    m.def("launch_fused_rs_adamw_bf16", &launch_fused_rs_adamw_bf16, "Fused RS+AdamW bf16");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_step_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_resources(p: int, world_size: int, dtype: torch.dtype, device: torch.device):
    key = (p, world_size, dtype, str(device))
    if key in _cache:
        return _cache[key]

    # Symmetric buffer for parameter shards (size p, this rank writes its shard).
    param_buf = symm_mem.empty(p, device=device, dtype=dtype)
    param_hdl = symm_mem.rendezvous(param_buf, dist.group.WORLD)
    param_ptrs = torch.tensor(param_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    # Symmetric buffer for full flat gradients (size world_size * p).
    grad_buf = symm_mem.empty(world_size * p, device=device, dtype=dtype)
    grad_hdl = symm_mem.rendezvous(grad_buf, dist.group.WORLD)
    grad_ptrs = torch.tensor(grad_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    full_flat = torch.empty(world_size * p, dtype=dtype, device=device)

    res = (param_buf, param_hdl, param_ptrs, grad_buf, grad_hdl, grad_ptrs, full_flat)
    _cache[key] = res
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
    assert dist.is_initialized()
    assert step >= 1

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    p = flat_param_shard.numel()
    device = flat_param_shard.device
    dtype = flat_param_shard.dtype

    ext = _get_ext()

    (param_buf, param_hdl, param_ptrs,
     grad_buf, grad_hdl, grad_ptrs,
     full_flat) = _get_resources(p, world_size, dtype, device)

    # ---- All-gather via symm_mem ----
    with torch.no_grad():
        param_buf.copy_(flat_param_shard.contiguous())
    param_hdl.barrier(channel=0)
    ext.launch_allgather_bf16(param_ptrs, full_flat, p, world_size)

    # ---- Forward / backward (PyTorch / cuBLAS tensor cores) ----
    templates = [torch.empty(shape, dtype=dtype, device=device) for shape in param_shapes]
    params_f = _unflatten_dense_tensors(full_flat, templates)
    params = [t.detach().requires_grad_(True) for t in params_f]

    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    flat_g = _flatten_dense_tensors([x.grad for x in params]).contiguous()

    # ---- Write our full grad into symmetric buffer; peers will read their slice ----
    with torch.no_grad():
        grad_buf.copy_(flat_g)
    grad_hdl.barrier(channel=0)

    # ---- Fused reduce-scatter + AdamW ----
    theta = flat_param_shard.clone().contiguous()
    m = exp_avg_shard.clone().contiguous()
    v = exp_avg_sq_shard.clone().contiguous()

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    ext.launch_fused_rs_adamw_bf16(
        grad_ptrs,
        theta, m, v,
        world_size, rank, p,
        1.0 / world_size,
        lr, beta1, beta2, eps, weight_decay,
        bc1, bc2,
    )

    grad_hdl.barrier(channel=1)

    return theta, m, v


__all__ = ["solution"]