"""
FP8 all-gather: fused BF16->FP8 quant + write into rank's slot of a symmetric
buffer, then fused FP8->BF16 dequant via direct peer reads (UVA).

Strategy:
- Each rank computes scale on-device, quantizes its shard to FP8 directly into
  its slot of a symmetric FP8 buffer (size world_size * P, fp8).
- Also writes its scale into a symmetric scale buffer (size world_size, fp32).
- A single device-side barrier (hdl.barrier) syncs all writers.
- A fused dequant kernel reads peer FP8 slots via UVA pointers and writes BF16
  output for all ranks. Each rank produces the full BF16 vector.
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

// --------- amax + scale + quantize to FP8 (writes to local slot) ----------

extern "C" __global__ void amax_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ amax_out,
    int64_t n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local = 0.0f;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float v = fabsf(__bfloat162float(x[idx]));
        if (v > local) local = v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float o = sdata[tid + s];
            if (o > sdata[tid]) sdata[tid] = o;
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax((int*)amax_out, __float_as_int(sdata[0]));
    }
}

// Compute scale = max(amax_history) / FP8_MAX, write to scale_out[rank].
// amax_history has been rolled and last slot replaced with current amax.
// We just take max over the history.
extern "C" __global__ void compute_scale_kernel(
    const float* __restrict__ amax_history,
    float* __restrict__ scale_local,  // single float
    int hist_len,
    float fp8_max
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float m = 0.0f;
        for (int i = 0; i < hist_len; ++i) {
            float v = amax_history[i];
            if (v > m) m = v;
        }
        if (m < 1e-12f) m = 1e-12f;
        scale_local[0] = m / fp8_max;
    }
}

// Roll history left by 1 and place new amax at end.
extern "C" __global__ void roll_and_set_kernel(
    const float* __restrict__ in_hist,
    const float* __restrict__ new_amax,
    float* __restrict__ out_hist,
    int hist_len
) {
    int tid = threadIdx.x;
    if (tid < hist_len - 1) {
        out_hist[tid] = in_hist[tid + 1];
    } else if (tid == hist_len - 1) {
        out_hist[tid] = new_amax[0];
    }
}

extern "C" __global__ void quantize_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ out_slot,  // pointer into symmetric buffer at our rank slot
    const float* __restrict__ scale_ptr,    // single float on device
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float inv_scale = 1.0f / scale_ptr[0];
    for (; idx < n; idx += stride) {
        float v = __bfloat162float(x[idx]) * inv_scale;
        out_slot[idx] = __nv_fp8_e4m3(v);
    }
}

// Dequant: for each rank r, read fp8 from peer_ptrs[r] (slot of size P starts at offset r*P
// within each rank's local buffer; but symmetric buffer is sized world_size*P and each rank
// writes into its OWN rank slot. So peer r's data is at peer_ptrs[r] + r*P).
// We produce out[r*P + i] = peer_data * peer_scale[r].
extern "C" __global__ void dequant_gather_kernel(
    const uint64_t* __restrict__ fp8_peer_ptrs,    // [world_size]
    const uint64_t* __restrict__ scale_peer_ptrs,  // [world_size]  (each peer's scale buf, size 1)
    __nv_bfloat16* __restrict__ out,               // [world_size * P]
    int64_t P,
    int world_size
) {
    int rank_id = blockIdx.y;
    if (rank_id >= world_size) return;

    const __nv_fp8_e4m3* peer_buf = reinterpret_cast<const __nv_fp8_e4m3*>(fp8_peer_ptrs[rank_id]);
    const float* peer_scale_ptr = reinterpret_cast<const float*>(scale_peer_ptrs[rank_id]);
    // Peer rank r writes its data into slot r of its own buffer => offset r*P
    const __nv_fp8_e4m3* peer_slot = peer_buf + (int64_t)rank_id * P;
    float s = peer_scale_ptr[0];

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    __nv_bfloat16* out_slot = out + (int64_t)rank_id * P;
    for (; idx < P; idx += stride) {
        float v = float(peer_slot[idx]) * s;
        out_slot[idx] = __float2bfloat16(v);
    }
}

void launch_amax(torch::Tensor x, torch::Tensor amax_out) {
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 512);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(amax_out.data_ptr<float>(), 0, sizeof(float), stream);
    amax_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        amax_out.data_ptr<float>(),
        n);
}

void launch_roll(torch::Tensor in_hist, torch::Tensor new_amax, torch::Tensor out_hist) {
    int hist_len = in_hist.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    roll_and_set_kernel<<<1, hist_len, 0, stream>>>(
        in_hist.data_ptr<float>(),
        new_amax.data_ptr<float>(),
        out_hist.data_ptr<float>(),
        hist_len);
}

void launch_compute_scale(torch::Tensor hist, torch::Tensor scale_out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    compute_scale_kernel<<<1, 1, 0, stream>>>(
        hist.data_ptr<float>(),
        scale_out.data_ptr<float>(),
        (int)hist.numel(),
        448.0f);
}

void launch_quantize(torch::Tensor x, int64_t out_slot_ptr, torch::Tensor scale) {
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 2048);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_to_fp8_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        reinterpret_cast<__nv_fp8_e4m3*>(out_slot_ptr),
        scale.data_ptr<float>(),
        n);
}

void launch_dequant_gather(
    torch::Tensor fp8_peer_ptrs,
    torch::Tensor scale_peer_ptrs,
    torch::Tensor out,
    int64_t P,
    int world_size
) {
    int threads = 256;
    int blocks_x = (int)std::min<int64_t>((P + threads - 1) / threads, 1024);
    dim3 grid(blocks_x, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_gather_kernel<<<grid, threads, 0, stream>>>(
        (const uint64_t*)fp8_peer_ptrs.data_ptr<int64_t>(),
        (const uint64_t*)scale_peer_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        P,
        world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_amax", &launch_amax);
    m.def("launch_roll", &launch_roll);
    m.def("launch_compute_scale", &launch_compute_scale);
    m.def("launch_quantize", &launch_quantize);
    m.def("launch_dequant_gather", &launch_dequant_gather);
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fp8_allgather_ext_p54", CUDA_SRC)
    return _ext


_cache = {}


def _get_resources(P: int, world_size: int, device, dtype):
    key = (P, world_size, device, dtype)
    if key in _cache:
        return _cache[key]

    # Symmetric FP8 buffer of size world_size * P
    fp8_buf = symm_mem.empty(world_size * P, device=device, dtype=torch.float8_e4m3fn)
    fp8_hdl = symm_mem.rendezvous(fp8_buf, dist.group.WORLD)

    # Symmetric scale buffer (1 float per rank's own buffer)
    scale_buf = symm_mem.empty(1, device=device, dtype=torch.float32)
    scale_hdl = symm_mem.rendezvous(scale_buf, dist.group.WORLD)

    fp8_ptrs = torch.tensor(fp8_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    scale_ptrs = torch.tensor(scale_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    out = torch.empty(world_size * P, device=device, dtype=dtype)
    amax_scratch = torch.zeros(1, device=device, dtype=torch.float32)

    res = {
        "fp8_buf": fp8_buf,
        "fp8_hdl": fp8_hdl,
        "scale_buf": scale_buf,
        "scale_hdl": scale_hdl,
        "fp8_ptrs": fp8_ptrs,
        "scale_ptrs": scale_ptrs,
        "out": out,
        "amax_scratch": amax_scratch,
    }
    _cache[key] = res
    return res


@torch.no_grad()
def solution(flat_param_shard: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    assert dist.is_initialized()
    ext = _get_ext()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    P = flat_param_shard.numel()
    device = flat_param_shard.device
    dtype = flat_param_shard.dtype

    x = flat_param_shard.contiguous()
    if dtype != torch.bfloat16:
        x_bf16 = x.to(torch.bfloat16)
    else:
        x_bf16 = x

    res = _get_resources(P, world_size, device, torch.bfloat16)

    # 1. amax over local shard
    ext.launch_amax(x_bf16, res["amax_scratch"])

    # 2. roll history + insert
    updated_hist = torch.empty_like(amax_history, dtype=torch.float32)
    if amax_history.dtype != torch.float32:
        in_hist = amax_history.to(torch.float32)
    else:
        in_hist = amax_history
    ext.launch_roll(in_hist, res["amax_scratch"], updated_hist)

    # 3. compute scale into local symmetric scale slot
    ext.launch_compute_scale(updated_hist, res["scale_buf"])

    # 4. quantize bf16 -> fp8 directly into our slot of fp8 symmetric buffer
    fp8_local_ptr = int(res["fp8_hdl"].buffer_ptrs[rank]) + rank * P  # bytes (fp8 = 1 byte)
    ext.launch_quantize(x_bf16, fp8_local_ptr, res["scale_buf"])

    # 5. device-side barrier across ranks
    res["fp8_hdl"].barrier(channel=0)
    res["scale_hdl"].barrier(channel=1)

    # 6. fused dequant + gather: each rank reads peer fp8 + peer scale via UVA
    ext.launch_dequant_gather(
        res["fp8_ptrs"],
        res["scale_ptrs"],
        res["out"],
        P,
        world_size,
    )

    # ensure dequant kernel completes before next round reuses buffers
    res["fp8_hdl"].barrier(channel=2)

    out = res["out"]
    if dtype != torch.bfloat16:
        out = out.to(dtype)

    # Cast updated_hist back to original dtype
    if amax_history.dtype != torch.float32:
        updated_hist = updated_hist.to(amax_history.dtype)

    return out, updated_hist


__all__ = ["solution"]