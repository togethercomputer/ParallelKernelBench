"""
Fused Adam + AllGather via symmetric memory.

Strategy:
- Each rank writes its updated shard directly into its slot of a symmetric
  output buffer of size [world_size * P]. The Adam math is fused with the
  store, so there's no full-model temporary.
- After local Adam+store, every rank pulls peer shards directly through UVA
  device pointers from symm_mem rendezvous (NVLink P2P), into the local
  full output. This replaces dist.all_gather_into_tensor.
- A symm_mem barrier provides the required publish/visibility ordering.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Fused Adam: read state shards, compute updated weight, write into local slot
// of the symmetric all-gather buffer (and update master_shard, exp_avg, exp_avg_sq).
__global__ void fused_adam_pack_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad,
    __nv_bfloat16* __restrict__ master,
    __nv_bfloat16* __restrict__ m_state,
    __nv_bfloat16* __restrict__ v_state,
    __nv_bfloat16* __restrict__ out_slot,  // points into symm buffer at rank slot
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float g = __bfloat162float(grad[idx]);
        float m = __bfloat162float(m_state[idx]);
        float v = __bfloat162float(v_state[idx]);
        float w = __bfloat162float(master[idx]);

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        w = w - lr * (m_hat / (sqrtf(v_hat) + eps));

        m_state[idx] = __float2bfloat16(m);
        v_state[idx] = __float2bfloat16(v);
        master[idx] = __float2bfloat16(w);
        out_slot[idx] = __float2bfloat16(w);
    }
}

__global__ void fused_adam_pack_f32_kernel(
    const float* __restrict__ grad,
    float* __restrict__ master,
    float* __restrict__ m_state,
    float* __restrict__ v_state,
    float* __restrict__ out_slot,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float g = grad[idx];
        float m = m_state[idx];
        float v = v_state[idx];
        float w = master[idx];

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        w = w - lr * (m_hat / (sqrtf(v_hat) + eps));

        m_state[idx] = m;
        v_state[idx] = v;
        master[idx] = w;
        out_slot[idx] = w;
    }
}

// Pull peer shards from remote symm buffers into local full output via UVA.
// peer_ptrs[r] points to the start of rank r's symm buffer (size world_size*P),
// but we only need the slot at offset r*P for rank r.
__global__ void gather_peers_bf16_kernel(
    const long long* __restrict__ peer_buf_ptrs,
    __nv_bfloat16* __restrict__ out_full,
    int world_size,
    int my_rank,
    int64_t p
) {
    int r = blockIdx.y;
    if (r == my_rank) return;  // already written locally
    const __nv_bfloat16* src = (const __nv_bfloat16*)peer_buf_ptrs[r];
    src += (int64_t)r * p;  // peer's own slot
    __nv_bfloat16* dst = out_full + (int64_t)r * p;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // vectorized 4xbf16 (8 bytes) loads
    int64_t n4 = p / 4;
    const uint64_t* src4 = reinterpret_cast<const uint64_t*>(src);
    uint64_t* dst4 = reinterpret_cast<uint64_t*>(dst);
    for (int64_t i = idx; i < n4; i += stride) {
        dst4[i] = src4[i];
    }
    // tail
    int64_t tail_start = n4 * 4;
    for (int64_t i = tail_start + idx; i < p; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void gather_peers_f32_kernel(
    const long long* __restrict__ peer_buf_ptrs,
    float* __restrict__ out_full,
    int world_size,
    int my_rank,
    int64_t p
) {
    int r = blockIdx.y;
    if (r == my_rank) return;
    const float* src = (const float*)peer_buf_ptrs[r];
    src += (int64_t)r * p;
    float* dst = out_full + (int64_t)r * p;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t n4 = p / 4;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    for (int64_t i = idx; i < n4; i += stride) {
        dst4[i] = src4[i];
    }
    int64_t tail_start = n4 * 4;
    for (int64_t i = tail_start + idx; i < p; i += stride) {
        dst[i] = src[i];
    }
}

void launch_fused_adam_pack(
    torch::Tensor grad,
    torch::Tensor master,
    torch::Tensor m_state,
    torch::Tensor v_state,
    int64_t out_slot_ptr,
    double lr, double beta1, double beta2, double eps,
    double bc1, double bc2,
    int64_t n,
    int dtype_enum
) {
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        fused_adam_pack_bf16_kernel<<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)grad.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)master.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)m_state.data_ptr<at::BFloat16>(),
            (__nv_bfloat16*)v_state.data_ptr<at::BFloat16>(),
            reinterpret_cast<__nv_bfloat16*>((uintptr_t)out_slot_ptr),
            (float)lr, (float)beta1, (float)beta2, (float)eps,
            (float)bc1, (float)bc2, n);
    } else {
        fused_adam_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
            grad.data_ptr<float>(),
            master.data_ptr<float>(),
            m_state.data_ptr<float>(),
            v_state.data_ptr<float>(),
            reinterpret_cast<float*>((uintptr_t)out_slot_ptr),
            (float)lr, (float)beta1, (float)beta2, (float)eps,
            (float)bc1, (float)bc2, n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_peers(
    torch::Tensor peer_ptrs,
    torch::Tensor out_full,
    int world_size,
    int my_rank,
    int64_t p,
    int dtype_enum
) {
    int threads = 256;
    int x_blocks = (int)((p / 4 + threads - 1) / threads);
    if (x_blocks < 1) x_blocks = 1;
    if (x_blocks > 1024) x_blocks = 1024;
    dim3 grid(x_blocks, world_size, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* d_ptrs = (const long long*)peer_ptrs.data_ptr<int64_t>();

    if (dtype_enum == 0) {
        gather_peers_bf16_kernel<<<grid, threads, 0, stream>>>(
            d_ptrs,
            (__nv_bfloat16*)out_full.data_ptr<at::BFloat16>(),
            world_size, my_rank, p);
    } else {
        gather_peers_f32_kernel<<<grid, threads, 0, stream>>>(
            d_ptrs,
            out_full.data_ptr<float>(),
            world_size, my_rank, p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_adam_pack", &launch_fused_adam_pack, "Fused Adam + pack into symm slot");
    m.def("launch_gather_peers", &launch_gather_peers, "Gather peer shards via UVA");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_adam_unshard_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_resources(p: int, dtype: torch.dtype, device: torch.device, world_size: int):
    key = (p, dtype, device, world_size)
    if key in _cache:
        return _cache[key]

    total = world_size * p
    symm_buf = symm_mem.empty(total, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(symm_buf, dist.group.WORLD)
    peer_ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    res = (symm_buf, hdl, peer_ptrs)
    _cache[key] = res
    return res


@torch.no_grad()
def solution(
    grad_shard: Tensor,
    master_shard: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> Tensor:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    p = grad_shard.numel()
    dtype = master_shard.dtype
    device = master_shard.device

    # Make local working copies (so reference state isn't mutated)
    m = exp_avg.clone().contiguous()
    v = exp_avg_sq.clone().contiguous()
    w = master_shard.clone().contiguous()
    g = grad_shard.contiguous()

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    ext = _get_ext()
    symm_buf, hdl, peer_ptrs = _get_resources(p, dtype, device, world_size)

    dtype_enum = 0 if dtype == torch.bfloat16 else 1
    if dtype not in (torch.bfloat16, torch.float32):
        # fallback path
        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        m_hat = m / bc1
        v_hat = v / bc2
        w.add_(m_hat.div(v_hat.sqrt().add(eps)).mul(-lr))
        gathered = torch.empty(world_size * p, dtype=w.dtype, device=w.device)
        dist.all_gather_into_tensor(gathered, w.contiguous())
        return gathered

    # Compute address of this rank's slot in the symmetric buffer
    slot_ptr = int(symm_buf.data_ptr()) + rank * p * symm_buf.element_size()

    # Fused Adam + write directly into symm slot
    ext.launch_fused_adam_pack(
        g, w, m, v,
        slot_ptr,
        float(lr), float(beta1), float(beta2), float(eps),
        float(bc1), float(bc2),
        p, dtype_enum,
    )

    # Publish slot to peers and ensure all peers have written theirs.
    hdl.barrier(channel=0)

    # Pull peer shards via UVA into a local full output.
    out_full = torch.empty(world_size * p, dtype=dtype, device=device)
    # Copy our local slot into out_full
    out_full.narrow(0, rank * p, p).copy_(symm_buf.narrow(0, rank * p, p))
    # Gather all other peers' slots
    ext.launch_gather_peers(peer_ptrs, out_full, world_size, rank, p, dtype_enum)

    # Ensure no peer reuses the symm buffer before everyone has read.
    hdl.barrier(channel=1)

    return out_full


__all__ = ["solution"]