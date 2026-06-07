"""
FSDP2 + EP clip_grad_norm using custom CUDA kernels and symmetric memory all-reduce.

Strategy:
- Fused BF16 squared-norm kernel: per-tensor block reduction directly in FP32, summed into a single accumulator.
- All-reduce the scalar via symmetric memory + multimem.ld_reduce/st on bf16x2 (one 8-byte slot for two FP32 lanes packed via float).
- Use a tiny FP32 scalar all-reduce kernel over peer pointers (1 element); world size <= 8, so unrolled load+sum is dominated by NVLink latency.
- Fused in-place scale kernel for clipping.
"""

import os
import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------- Squared-norm kernel (BF16 -> FP32 accumulation) ----------------

template <int BLOCK>
__global__ void bf16_sqnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    int64_t n,
    float* __restrict__ partial   // [gridDim.x]
) {
    int64_t tid = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * BLOCK;
    float acc = 0.f;

    // Vectorized load: 8 bf16 = 16 bytes
    int64_t n_vec = n / 8;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    for (int64_t i = tid; i < n_vec; i += stride) {
        uint4 v = xv[i];
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float f = __bfloat162float(h[j]);
            acc += f * f;
        }
    }
    // Tail
    int64_t tail_start = n_vec * 8;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        float f = __bfloat162float(x[i]);
        acc += f * f;
    }

    __shared__ float smem[BLOCK];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = smem[0];
}

template <int BLOCK>
__global__ void fp32_reduce_kernel(
    const float* __restrict__ partial,
    int n,
    float* __restrict__ out,
    int out_idx,
    float scale
) {
    __shared__ float smem[BLOCK];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += BLOCK) acc += partial[i];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Add into out[out_idx] (initialized to 0 by host)
        atomicAdd(out + out_idx, smem[0] * scale);
    }
}

void launch_sqnorm_bf16(torch::Tensor x, torch::Tensor out, int64_t out_idx, double scale) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(out.is_cuda() && out.dtype() == torch::kFloat32);
    int64_t n = x.numel();
    if (n == 0) return;

    constexpr int BLOCK = 256;
    int blocks = (int)std::min<int64_t>((n + BLOCK * 8 - 1) / (BLOCK * 8), 1024);
    if (blocks < 1) blocks = 1;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto partial = torch::empty({blocks}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    bf16_sqnorm_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        n,
        partial.data_ptr<float>()
    );
    fp32_reduce_kernel<BLOCK><<<1, BLOCK, 0, stream>>>(
        partial.data_ptr<float>(), blocks, out.data_ptr<float>(), (int)out_idx, (float)scale
    );
}

// ---------------- In-place scale (BF16) ----------------

__global__ void bf16_scale_inplace_kernel(
    __nv_bfloat16* __restrict__ x,
    int64_t n,
    const float* __restrict__ total_norm,  // FP32 scalar
    float max_norm
) {
    float tn = *total_norm;
    if (!(tn > max_norm)) return;
    float coef = max_norm / tn;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t n_vec = n / 8;
    uint4* xv = reinterpret_cast<uint4*>(x);
    for (int64_t i = tid; i < n_vec; i += stride) {
        uint4 v = xv[i];
        __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float f = __bfloat162float(h[j]) * coef;
            h[j] = __float2bfloat16(f);
        }
        xv[i] = v;
    }
    int64_t tail_start = n_vec * 8;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        float f = __bfloat162float(x[i]) * coef;
        x[i] = __float2bfloat16(f);
    }
}

void launch_scale_inplace_bf16(torch::Tensor x, torch::Tensor total_norm, double max_norm) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    int64_t n = x.numel();
    if (n == 0) return;
    int threads = 256;
    int blocks = (int)std::min<int64_t>((n + threads * 8 - 1) / (threads * 8), 1024);
    if (blocks < 1) blocks = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    bf16_scale_inplace_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        n, total_norm.data_ptr<float>(), (float)max_norm
    );
}

// ---------------- FP32 scalar all-reduce via peer pointers ----------------
// Each rank writes its scalar to its symm buffer slot; barrier; each rank
// loads from all peers and writes sum back. Designed for tiny (<=128) numel.

__global__ void fp32_peer_allreduce_kernel(
    const long long* __restrict__ ptrs,  // world_size peer device pointers (FP32 buffer)
    int world_size,
    int n,
    float* __restrict__ out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float s = 0.f;
    #pragma unroll 8
    for (int r = 0; r < world_size; ++r) {
        const float* p = (const float*)ptrs[r];
        s += p[idx];
    }
    out[idx] = s;
}

void launch_fp32_peer_allreduce(
    torch::Tensor ptrs_tensor,
    int64_t world_size,
    torch::Tensor out,
    int64_t n
) {
    int threads = 32;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fp32_peer_allreduce_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)ptrs_tensor.data_ptr<int64_t>(),
        (int)world_size,
        (int)n,
        out.data_ptr<float>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_sqnorm_bf16", &launch_sqnorm_bf16, "BF16 squared-norm accumulator");
    m.def("launch_scale_inplace_bf16", &launch_scale_inplace_bf16, "BF16 in-place scale by clip coef");
    m.def("launch_fp32_peer_allreduce", &launch_fp32_peer_allreduce, "Tiny FP32 peer all-reduce");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("clip_grad_norm_ep_ext", CUDA_SRC)
    return _ext


# Cache symm-mem state per (group_id, slot_count, dtype, device)
_symm_cache = {}


def _get_symm_state(group: dist.ProcessGroup, n_slots: int, device: torch.device):
    key = (id(group), n_slots, device.index)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(n_slots, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    out = torch.empty(n_slots, device=device, dtype=torch.float32)
    state = (buf, hdl, ptrs_tensor, out)
    _symm_cache[key] = state
    return state


def _scalar_allreduce_symm(val: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Reduce a 1-element FP32 tensor across `group` using symm-mem peer pointers.
    Falls back to dist.all_reduce on errors."""
    if group is None:
        return val
    try:
        ws = dist.get_world_size(group)
        if ws == 1:
            return val
        buf, hdl, ptrs_tensor, out = _get_symm_state(group, 1, val.device)
        buf.copy_(val.view(-1))
        hdl.barrier(channel=0)
        _get_ext().launch_fp32_peer_allreduce(ptrs_tensor, ws, out, 1)
        hdl.barrier(channel=1)
        return out
    except Exception:
        v = val.clone()
        dist.all_reduce(v, op=dist.ReduceOp.SUM, group=group)
        return v


def _local_sqnorm_acc(grad_tensors: List[torch.Tensor], device: torch.device, scale: float = 1.0) -> torch.Tensor:
    """Compute sum of squared norms (with optional per-tensor scale^2 effectively applied
    via passing scale here means we scale BEFORE squaring -> caller passes scale=1 unless
    pre-scaled)."""
    out = torch.zeros(1, device=device, dtype=torch.float32)
    ext = _get_ext()
    for g in grad_tensors:
        if g is None or g.numel() == 0:
            continue
        gc = g.detach()
        if not gc.is_contiguous():
            gc = gc.contiguous()
        if gc.dtype == torch.bfloat16:
            ext.launch_sqnorm_bf16(gc, out, 0, float(scale * scale))
        else:
            # Fallback for non-bf16
            gn = torch.norm(gc.to(torch.float32), p=2.0)
            out = out + (gn * gn) * (scale * scale)
    return out


@torch.no_grad()
def solution(
    non_ep_grad_tensors: List[torch.Tensor],
    ep_grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    ep_size: int = 1,
    fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    # Determine device
    dev = None
    for t in list(non_ep_grad_tensors) + list(ep_grad_tensors):
        if t is not None:
            dev = t.device
            break
    if dev is None:
        dev = torch.device("cuda", torch.cuda.current_device())

    ext = _get_ext()

    # In-place pre-scale EP grads by 1/ep_size
    if ep_size > 1 and ep_grad_tensors:
        scale = 1.0 / float(ep_size)
        for t in ep_grad_tensors:
            if t is not None and t.numel() > 0:
                t.detach().mul_(scale)

    # Local squared norms
    non_ep_local = _local_sqnorm_acc(non_ep_grad_tensors, dev)
    ep_local = _local_sqnorm_acc(ep_grad_tensors, dev)

    # Reduce non-EP over fsdp_group
    non_ep_total = _scalar_allreduce_symm(non_ep_local, fsdp_group) if fsdp_group is not None else non_ep_local

    # Reduce EP over ep_fsdp then ep
    ep_total = ep_local
    if ep_fsdp_group is not None:
        ep_total = _scalar_allreduce_symm(ep_total, ep_fsdp_group)
    if ep_group is not None:
        ep_total = _scalar_allreduce_symm(ep_total, ep_group)

    inv_p = 1.0 / float(norm_type)
    total_sumsq = (non_ep_total + ep_total).view(())
    total_norm = total_sumsq.pow(inv_p)

    # Decide on host whether to clip (single sync), then fused scale
    tn_host = float(total_norm.item())
    if tn_host > max_norm and tn_host > 0.0:
        coef = max_norm / tn_host
        # Use the device tensor as scale source for the kernel (kernel reads ptr).
        # Easier: just multiply in-place via custom kernel using a fixed coef.
        for t in non_ep_grad_tensors:
            if t is not None and t.numel() > 0:
                if t.dtype == torch.bfloat16 and t.is_contiguous():
                    # Provide total_norm tensor; kernel computes coef internally.
                    ext.launch_scale_inplace_bf16(t, total_norm.contiguous(), float(max_norm))
                else:
                    t.mul_(coef)
        for t in ep_grad_tensors:
            if t is not None and t.numel() > 0:
                if t.dtype == torch.bfloat16 and t.is_contiguous():
                    ext.launch_scale_inplace_bf16(t, total_norm.contiguous(), float(max_norm))
                else:
                    t.mul_(coef)

    return total_norm