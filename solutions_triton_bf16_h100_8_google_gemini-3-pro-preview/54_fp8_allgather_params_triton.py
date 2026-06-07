"""
Strategy:
- Use symmetric memory (`symm_mem`) to enable direct cross-GPU data movement via UVA.
- Compute the rolling absolute-max and derive the scaling factor on the device without CPU synchronization.
- **Kernel 1 (Push & Quantize):** Each rank scales and quantizes its local BF16 shard directly into its local FP8 symmetric memory buffer. The scale factor is also written to a symmetric memory float buffer.
- A single barrier ensures all peers have materialized their FP8 representations and scale factors.
- **Kernel 2 (Pull & Dequantize):** Each rank acts as a receiver, launching a 2D grid that simultaneously pulls FP8 chunks and scale factors from all peers via UVA pointers, dequantizing directly into the final contiguous BF16 full-gather tensor.
- This fully fused push-pull architecture drastically reduces memory bandwidth compared to a standard all-gather by transporting only 8-bit payloads over NVLink, entirely bypassing opaque collective overhead.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor
from utils.cuda_helpers import compile_cuda_extension

_FP8_E4M3_MAX = 448.0

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

__global__ void quantize_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ scale_ptr,
    __nv_fp8_e4m3* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    int64_t P
) {
    float scale = *scale_ptr;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread 0 writes the scalar scale for this shard
    if (idx == 0) {
        *out_scale = scale;
    }
    
    // Scale and convert to FP8 natively
    if (idx < P) {
        float val = __bfloat162float(input[idx]);
        out_fp8[idx] = __nv_fp8_e4m3(val / scale);
    }
}

__global__ void pull_and_dequantize_kernel(
    const uint64_t* __restrict__ symm_fp8_ptrs,
    const uint64_t* __restrict__ symm_scale_ptrs,
    __nv_bfloat16* __restrict__ out_full,
    int64_t P
) {
    int peer = blockIdx.y;
    int64_t local_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (local_idx < P) {
        const __nv_fp8_e4m3* peer_fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(static_cast<uintptr_t>(symm_fp8_ptrs[peer]));
        float scale = *reinterpret_cast<const float*>(static_cast<uintptr_t>(symm_scale_ptrs[peer]));
        
        float val = float(peer_fp8[local_idx]);
        out_full[(int64_t)peer * P + local_idx] = __float2bfloat16(val * scale);
    }
}

void quantize_to_symm(
    torch::Tensor local_shard,
    torch::Tensor scale,
    torch::Tensor local_symm_fp8,
    torch::Tensor local_symm_scale,
    int64_t P
) {
    TORCH_CHECK(local_shard.is_contiguous());
    TORCH_CHECK(local_symm_fp8.is_contiguous());

    const int threads = 256;
    const int blocks = (int)((P + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    quantize_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(local_shard.data_ptr()),
        reinterpret_cast<const float*>(scale.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(local_symm_fp8.data_ptr()),
        reinterpret_cast<float*>(local_symm_scale.data_ptr()),
        P
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pull_from_symm(
    torch::Tensor symm_fp8_ptrs,
    torch::Tensor symm_scale_ptrs,
    torch::Tensor out_full,
    int world_size,
    int64_t P
) {
    const int threads = 256;
    dim3 blocks((int)((P + threads - 1) / threads), world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    pull_and_dequantize_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(symm_fp8_ptrs.data_ptr()),
        reinterpret_cast<const uint64_t*>(symm_scale_ptrs.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out_full.data_ptr()),
        P
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_to_symm", &quantize_to_symm, "Quantize to local symmetric memory buffer");
    m.def("pull_from_symm", &pull_from_symm, "Pull & dequantize from peers' symmetric memory buffers");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fp8_gather_fused", CUDA_SRC)
    return _ext


_symm_cache = {}


def _get_symm_state(p: int, device: torch.device):
    """
    Rendezvous two symmetric memory buffers on first encounter for a given shape:
    One for the FP8 payload and one for the scalar float32 scale. 
    Also caches a GPU tensor holding the list of all valid UVA peer pointers.
    """
    global _symm_cache
    key = (p, device)
    if key in _symm_cache:
        return _symm_cache[key]

    buf_fp8 = symm_mem.empty(p, device=device, dtype=torch.float8_e4m3fn)
    hdl_fp8 = symm_mem.rendezvous(buf_fp8, dist.group.WORLD)

    buf_scale = symm_mem.empty(1, device=device, dtype=torch.float32)
    hdl_scale = symm_mem.rendezvous(buf_scale, dist.group.WORLD)

    fp8_ptrs = torch.tensor(hdl_fp8.buffer_ptrs, dtype=torch.int64, device=device)
    scale_ptrs = torch.tensor(hdl_scale.buffer_ptrs, dtype=torch.int64, device=device)

    _symm_cache[key] = (buf_fp8, hdl_fp8, buf_scale, hdl_scale, fp8_ptrs, scale_ptrs)
    return _symm_cache[key]


@torch.no_grad()
def _fp8_round_trip_bf16(x: Tensor, scale: Tensor) -> Tensor:
    xf = x.float()
    qs = xf / scale
    q = qs.to(torch.float8_e4m3fn)
    return (q.float() * scale).to(dtype=x.dtype)


@torch.no_grad()
def solution(flat_param_shard: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    """
    Args:
        flat_param_shard: Local parameter shard ``[P]`` (BF16 or other float dtype).
        amax_history: Rolling absolute-max buffer for dynamic FP8 scaling.

    Returns:
        ``(flat_full_bf16, updated_amax_history)`` — concatenation of all ranks'
        reconstructed shards (identical on every rank) and the updated history tensor.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    p = flat_param_shard.numel()

    # Pre-emptively cast to optimized format if FP32 or otherwise
    orig_dtype = flat_param_shard.dtype
    if orig_dtype != torch.bfloat16:
        flat_param_shard_bf16 = flat_param_shard.to(torch.bfloat16)
    else:
        flat_param_shard_bf16 = flat_param_shard

    # Compute history and scale cleanly on the device without syncing the CPU
    cur_abs_max = flat_param_shard.abs().max().to(torch.float32)
    updated_hist = torch.roll(amax_history, shifts=-1, dims=0)
    updated_hist[-1] = cur_abs_max.to(dtype=updated_hist.dtype)
    scale = updated_hist.max().clamp(min=1e-12).to(torch.float32) / _FP8_E4M3_MAX

    if p == 0:
        return torch.empty(0, dtype=orig_dtype, device=flat_param_shard.device), updated_hist

    if world_size == 1:
        recon = _fp8_round_trip_bf16(flat_param_shard_bf16, scale)
        if orig_dtype != torch.bfloat16:
            recon = recon.to(orig_dtype)
        return recon, updated_hist

    # Avoid race conditions dynamically compiling kernel
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    buf_fp8, hdl_fp8, buf_scale, hdl_scale, fp8_ptrs, scale_ptrs = _get_symm_state(p, flat_param_shard.device)

    # 1) Fused scaling + quantizing local shard directly to symmetric mem buffer
    ext.quantize_to_symm(
        flat_param_shard_bf16.contiguous(),
        scale,
        buf_fp8,
        buf_scale,
        p
    )

    # 2) Synchronize state across all peers
    hdl_fp8.barrier(channel=0)

    # 3) Allocate full gather block, simultaneously pull and dequantize all partitions
    full_bf16 = torch.empty(world_size * p, dtype=torch.bfloat16, device=flat_param_shard.device)
    ext.pull_from_symm(
        fp8_ptrs,
        scale_ptrs,
        full_bf16,
        world_size,
        p
    )

    if orig_dtype != torch.bfloat16:
        full = full_bf16.to(orig_dtype)
    else:
        full = full_bf16

    return full, updated_hist


__all__ = ["solution"]