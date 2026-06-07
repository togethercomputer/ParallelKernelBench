"""
FP8 reduce-scatter via symmetric memory + custom CUDA kernels.

Strategy:
- Compute amax via a custom CUDA reduction kernel (BF16 -> FP32).
- Use a single all-reduce (max) on the scalar amax across ranks to keep history
  consistent — actually we just need this rank's amax for history; scale uses
  history max which is local. So no extra collective needed for amax.
- Fused FP8 round-trip quant/dequant kernel writes directly into a symmetric
  memory buffer.
- Reduce-scatter implemented as: each rank reads its shard slice from all peers
  via UVA peer pointers and sums them into the output shard, divided by world_size.
- Barriers via symm_mem signal pad on device.
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
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// ---------------- amax reduction (BF16 -> FP32 scalar) ----------------
__global__ void amax_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ out,
    int64_t n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    float local = 0.0f;
    for (int64_t i = idx; i < n; i += stride) {
        float v = fabsf(__bfloat162float(x[i]));
        if (v > local) local = v;
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sdata[tid], b = sdata[tid + s];
            sdata[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax((int*)out, __float_as_int(sdata[0]));
    }
}

void launch_amax_bf16(torch::Tensor x, torch::Tensor out) {
    int64_t n = x.numel();
    int threads = 512;
    int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 1024);
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(out.data_ptr<float>(), 0, sizeof(float), s);
    amax_bf16_kernel<<<blocks, threads, threads * sizeof(float), s>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        out.data_ptr<float>(), n);
}

// ---------------- FP8 round-trip into symmetric buffer ----------------
__global__ void fp8_round_trip_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ scale_ptr,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    for (int64_t i = idx; i < n; i += stride) {
        float xf = __bfloat162float(x[i]);
        float qs = xf * inv_scale;
        __nv_fp8_e4m3 q = __nv_fp8_e4m3(qs);
        float deq = float(q) * scale;
        out[i] = __float2bfloat16(deq);
    }
}

void launch_fp8_round_trip(
    torch::Tensor x, torch::Tensor out, torch::Tensor scale, int64_t n
) {
    int threads = 512;
    int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 2048);
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    fp8_round_trip_kernel<<<blocks, threads, 0, s>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        scale.data_ptr<float>(),
        n);
}

// Compute scale = max(history) / FP8_MAX, clamped, on device
__global__ void compute_scale_kernel(
    const float* __restrict__ hist, float* __restrict__ scale_out,
    int hist_len, float fp8_max
) {
    float m = 0.0f;
    for (int i = threadIdx.x; i < hist_len; i += blockDim.x) {
        float v = hist[i];
        if (v > m) m = v;
    }
    __shared__ float sm[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        float t = __shfl_down_sync(0xffffffff, m, o);
        if (t > m) m = t;
    }
    if (lane == 0) sm[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (threadIdx.x < (blockDim.x + 31) / 32) ? sm[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            float t = __shfl_down_sync(0xffffffff, m, o);
            if (t > m) m = t;
        }
        if (threadIdx.x == 0) {
            float c = m < 1e-12f ? 1e-12f : m;
            *scale_out = c / fp8_max;
        }
    }
}

void launch_compute_scale(torch::Tensor hist, torch::Tensor scale_out) {
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    compute_scale_kernel<<<1, 128, 0, s>>>(
        hist.data_ptr<float>(), scale_out.data_ptr<float>(),
        (int)hist.numel(), 448.0f);
}

// ---------------- Reduce-scatter via peer pointers ----------------
// Each rank reads its shard slice (offset = rank * shard_elems) from all peers,
// sums them, divides by world_size, writes to out_shard.

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__global__ void barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank, int world_size, uint64_t block_id
) {
    if (blockIdx.x != 0) return;
    int t = threadIdx.x;
    if (t >= world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[t];
    uint32_t* send_addr = (uint32_t*)(remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = (uint32_t*)(local_base + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

void launch_barrier(torch::Tensor signal_pad_ptrs, int rank, int world_size, int64_t block_id) {
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    barrier_kernel<<<1, 32, 0, s>>>(
        (const uint64_t*)signal_pad_ptrs.data_ptr<int64_t>(),
        rank, world_size, (uint64_t)block_id);
}

__global__ void reduce_scatter_bf16_kernel(
    const long long* __restrict__ peer_ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size, int rank, int64_t shard_elems, float inv_ws
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t base = (int64_t)rank * shard_elems;
    for (int64_t i = idx; i < shard_elems; i += stride) {
        float sum = 0.0f;
        #pragma unroll 1
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)peer_ptrs[r];
            sum += __bfloat162float(src[base + i]);
        }
        out[i] = __float2bfloat16(sum * inv_ws);
    }
}

void launch_reduce_scatter_bf16(
    torch::Tensor peer_ptrs, torch::Tensor out,
    int world_size, int rank, int64_t shard_elems
) {
    int threads = 512;
    int blocks = (int)std::min<int64_t>((shard_elems + threads - 1) / threads, 2048);
    float inv_ws = 1.0f / (float)world_size;
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    reduce_scatter_bf16_kernel<<<blocks, threads, 0, s>>>(
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        world_size, rank, shard_elems, inv_ws);
}

// Update amax history: shift left, append cur amax. hist is fp32.
__global__ void update_history_kernel(
    float* __restrict__ hist, const float* __restrict__ cur, int len
) {
    int t = threadIdx.x;
    extern __shared__ float buf[];
    if (t < len) buf[t] = hist[t];
    __syncthreads();
    if (t < len - 1) hist[t] = buf[t + 1];
    if (t == 0) hist[len - 1] = *cur;
}

void launch_update_history(torch::Tensor hist, torch::Tensor cur) {
    int len = (int)hist.numel();
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    update_history_kernel<<<1, ((len + 31) / 32) * 32, len * sizeof(float), s>>>(
        hist.data_ptr<float>(), cur.data_ptr<float>(), len);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("amax_bf16", &launch_amax_bf16);
    m.def("fp8_round_trip", &launch_fp8_round_trip);
    m.def("compute_scale", &launch_compute_scale);
    m.def("barrier", &launch_barrier);
    m.def("reduce_scatter_bf16", &launch_reduce_scatter_bf16);
    m.def("update_history", &launch_update_history);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fp8_rs_ext_v1", CUDA_SRC)
    return _ext


_cache = {}

def _get_resources(n: int, dtype: torch.dtype, device: torch.device):
    key = (n, dtype, device)
    if key in _cache:
        return _cache[key]
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    peer_ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    res = (buf, hdl, peer_ptrs)
    _cache[key] = res
    return res


@torch.no_grad()
def solution(flat_grads: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n = flat_grads.numel()
    assert n % world_size == 0
    shard_elems = n // world_size
    device = flat_grads.device
    dtype = flat_grads.dtype

    # Fast path requires bf16; fall back to reference otherwise.
    if dtype != torch.bfloat16:
        cur_abs_max = flat_grads.abs().max().to(torch.float32)
        out_hist = torch.roll(amax_history, shifts=-1, dims=0)
        out_hist[-1] = cur_abs_max.to(dtype=out_hist.dtype)
        scale = out_hist.max().clamp(min=1e-12).to(torch.float32) / _FP8_E4M3_MAX
        xf = flat_grads.float()
        q = (xf / scale).to(torch.float8_e4m3fn)
        recon = (q.float() * scale).to(dtype=dtype)
        out_shard = torch.empty(shard_elems, dtype=dtype, device=device)
        dist.reduce_scatter_tensor(out_shard, recon.contiguous(), op=dist.ReduceOp.SUM)
        out_shard.div_(world_size)
        return out_shard, out_hist

    ext = _get_ext()
    flat_grads = flat_grads.contiguous()

    buf, hdl, peer_ptrs = _get_resources(n, dtype, device)

    # 1) Compute current amax (BF16 -> FP32 scalar).
    cur_amax = torch.zeros(1, dtype=torch.float32, device=device)
    ext.amax_bf16(flat_grads, cur_amax)

    # 2) Update history on device.
    out_hist = amax_history.clone()
    if out_hist.dtype != torch.float32:
        hist_f32 = out_hist.to(torch.float32)
        ext.update_history(hist_f32, cur_amax)
        out_hist = hist_f32.to(amax_history.dtype)
        hist_for_scale = hist_f32
    else:
        ext.update_history(out_hist, cur_amax)
        hist_for_scale = out_hist

    # 3) Compute scale on device.
    scale = torch.empty(1, dtype=torch.float32, device=device)
    ext.compute_scale(hist_for_scale, scale)

    # 4) FP8 round-trip directly into symmetric buffer.
    ext.fp8_round_trip(flat_grads, buf, scale, n)

    # 5) Barrier across ranks (device-side).
    ext.barrier(hdl.signal_pad_ptrs_dev, rank, world_size, 0)

    # 6) Reduce-scatter via peer pointers.
    out_shard = torch.empty(shard_elems, dtype=dtype, device=device)
    ext.reduce_scatter_bf16(peer_ptrs, out_shard, world_size, rank, shard_elems)

    # 7) Trailing barrier so peers don't overwrite buf before we finish reading.
    ext.barrier(hdl.signal_pad_ptrs_dev, rank, world_size, 1)

    return out_shard, out_hist


__all__ = ["solution"]