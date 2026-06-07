from __future__ import annotations

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
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

static constexpr int DTYPE_BF16 = 0;
static constexpr int DTYPE_F32  = 1;
static constexpr int DTYPE_F16  = 2;

// -----------------------------------------------------------------------------
// Device-side symmetric-memory signal-pad barrier.
// One reusable slot per resident CTA. Each slot stores world_size uint32 signals.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void send_signal_release(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_acquire(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1u);
}

__device__ __forceinline__ void cta_peer_barrier(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int slot,
    int rank,
    int world_size
) {
    int t = threadIdx.x;
    if (t < world_size) {
        uint64_t local_base  = signal_pad_ptrs[rank];
        uint64_t remote_base = signal_pad_ptrs[t];

        uint64_t send_off = ((uint64_t)slot * (uint64_t)world_size + (uint64_t)rank) * 4ull;
        uint64_t wait_off = ((uint64_t)slot * (uint64_t)world_size + (uint64_t)t) * 4ull;

        uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base + send_off);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base  + wait_off);

        send_signal_release(send_addr);
        wait_signal_acquire(wait_addr);
    }
}

__device__ __forceinline__ float read_as_f32(
    const void* __restrict__ x,
    int64_t idx,
    int dtype_enum
) {
    if (dtype_enum == DTYPE_BF16) {
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(x);
        return __bfloat162float(p[idx]);
    } else if (dtype_enum == DTYPE_F16) {
        const __half* p = reinterpret_cast<const __half*>(x);
        return __half2float(p[idx]);
    } else {
        const float* p = reinterpret_cast<const float*>(x);
        return p[idx];
    }
}

__device__ __forceinline__ void write_from_f32(
    void* __restrict__ y,
    int64_t idx,
    float v,
    int dtype_enum
) {
    if (dtype_enum == DTYPE_BF16) {
        __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(y);
        p[idx] = __float2bfloat16_rn(v);
    } else if (dtype_enum == DTYPE_F16) {
        __half* p = reinterpret_cast<__half*>(y);
        p[idx] = __float2half_rn(v);
    } else {
        float* p = reinterpret_cast<float*>(y);
        p[idx] = v;
    }
}

__device__ __forceinline__ int8_t quantize_nearest_even(float v, float scale) {
    float r = nearbyintf(v / scale);  // round-to-nearest-even, matching torch.round
    r = fminf(127.0f, fmaxf(-127.0f, r));
    return static_cast<int8_t>(r);
}

// Symmetric byte layout per rank:
//   [0, n)                         int8 q values
//   [scale_offset, scale_offset+4*nb) FP32 scales
//
// Persistent CTA loop:
//   1. compute one block's absmax/scale and local q into symmetric memory
//   2. system fence + device-side peer barrier for that CTA slot
//   3. read every rank's q/scale through UVA, dequantize to FP32 sum, average,
//      cast to original dtype
__global__ void quant_int8_avg_kernel(
    const void* __restrict__ input,
    uint8_t* __restrict__ local_symm,
    const int64_t* __restrict__ symm_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    void* __restrict__ output,
    int64_t n,
    int64_t nb,
    int64_t block_size,
    int64_t scale_offset,
    int world_size,
    int rank,
    int dtype_enum
) {
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int slot = blockIdx.x;

    int8_t* __restrict__ local_q = reinterpret_cast<int8_t*>(local_symm);
    float* __restrict__ local_scales =
        reinterpret_cast<float*>(local_symm + scale_offset);

    for (int64_t b = blockIdx.x; b < nb; b += gridDim.x) {
        const int64_t base = b * block_size;

        float local_absmax = 0.0f;
        for (int64_t j = tid; j < block_size; j += blockDim.x) {
            int64_t idx = base + j;
            float v = (idx < n) ? read_as_f32(input, idx, dtype_enum) : 0.0f;
            local_absmax = fmaxf(local_absmax, fabsf(v));
        }

        smem[tid] = local_absmax;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
            }
            __syncthreads();
        }

        float scale = fmaxf(smem[0], 1.0e-8f) / 127.0f;
        if (tid == 0) {
            local_scales[b] = scale;
        }
        __syncthreads();

        for (int64_t j = tid; j < block_size; j += blockDim.x) {
            int64_t idx = base + j;
            if (idx < n) {
                float v = read_as_f32(input, idx, dtype_enum);
                local_q[idx] = quantize_nearest_even(v, scale);
            }
        }

        // Make q/scales visible to peer GPUs before signaling this block ready.
        __threadfence_system();
        __syncthreads();

        cta_peer_barrier(signal_pad_ptrs, slot, rank, world_size);
        __syncthreads();

        for (int64_t j = tid; j < block_size; j += blockDim.x) {
            int64_t idx = base + j;
            if (idx < n) {
                float sum = 0.0f;

                #pragma unroll
                for (int r = 0; r < 8; ++r) {
                    if (r < world_size) {
                        uint8_t* remote_base =
                            reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(symm_ptrs[r]));
                        const int8_t* remote_q =
                            reinterpret_cast<const int8_t*>(remote_base);
                        const float* remote_scales =
                            reinterpret_cast<const float*>(remote_base + scale_offset);

                        sum += static_cast<float>(remote_q[idx]) * remote_scales[b];
                    }
                }

                write_from_f32(output, idx, sum / static_cast<float>(world_size), dtype_enum);
            }
        }

        __syncthreads();
    }
}

void launch_quant_int8_avg(
    torch::Tensor input,
    torch::Tensor symm_buf,
    torch::Tensor symm_ptrs,
    torch::Tensor signal_pad_ptrs,
    torch::Tensor output,
    int64_t n,
    int64_t nb,
    int64_t block_size,
    int64_t scale_offset,
    int world_size,
    int rank,
    int dtype_enum,
    int num_ctas,
    int threads
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(symm_buf.is_cuda(), "symm_buf must be CUDA");
    TORCH_CHECK(symm_ptrs.is_cuda(), "symm_ptrs must be CUDA");
    TORCH_CHECK(signal_pad_ptrs.is_cuda(), "signal_pad_ptrs must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(symm_buf.dtype() == torch::kUInt8, "symm_buf must be uint8");
    TORCH_CHECK(symm_ptrs.dtype() == torch::kInt64, "symm_ptrs must be int64");
    TORCH_CHECK(signal_pad_ptrs.dtype() == torch::kInt64, "signal_pad_ptrs must be int64");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t shmem = static_cast<size_t>(threads) * sizeof(float);

    quant_int8_avg_kernel<<<num_ctas, threads, shmem, stream>>>(
        input.data_ptr(),
        symm_buf.data_ptr<uint8_t>(),
        symm_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>()),
        output.data_ptr(),
        n,
        nb,
        block_size,
        scale_offset,
        world_size,
        rank,
        dtype_enum
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_quant_int8_avg", &launch_quant_int8_avg,
          "Block INT8 quant/dequant + symmetric-memory peer average");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("quantized_grad_avg_symm_int8_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _ceil_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.float32:
        return 1
    if dtype is torch.float16:
        return 2
    raise TypeError(f"optimized CUDA path supports bf16/fp16/fp32 gradients, got {dtype}")


def _get_resources(n: int, block_size: int, device: torch.device):
    nb = (n + block_size - 1) // block_size
    scale_offset = (n + 3) & ~3
    total_bytes = scale_offset + nb * 4

    key = (device.index, n, block_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    symm_buf = symm_mem.empty((total_bytes,), device=device, dtype=torch.uint8)
    hdl = symm_mem.rendezvous(symm_buf, dist.group.WORLD)
    symm_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "symm_buf": symm_buf,
        "hdl": hdl,
        "symm_ptrs": symm_ptrs,
        "scale_offset": scale_offset,
        "nb": nb,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    flat_grad: Tensor,
    block_size: int,
) -> Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert block_size >= 1
    assert flat_grad.is_cuda, "flat_grad must be CUDA"

    dtype_enum = _dtype_enum(flat_grad.dtype)

    orig_shape = flat_grad.shape
    x = flat_grad.contiguous().reshape(-1)
    n = x.numel()

    if n == 0:
        return torch.empty_like(flat_grad.contiguous())

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size <= 8, "this H100/NVLink kernel is specialized for <= 8 ranks"

    res = _get_resources(n, int(block_size), x.device)
    out = torch.empty_like(x)

    nb = res["nb"]

    threads = min(1024, max(32, _ceil_pow2(min(int(block_size), 1024))))
    # Keep signal-pad footprint small and use persistent CTAs. 128 slots = 4 KiB for 8 ranks.
    num_ctas = min(max(1, nb), 128)

    _get_ext().launch_quant_int8_avg(
        x,
        res["symm_buf"],
        res["symm_ptrs"],
        res["hdl"].signal_pad_ptrs_dev,
        out,
        n,
        nb,
        int(block_size),
        res["scale_offset"],
        world_size,
        rank,
        dtype_enum,
        num_ctas,
        threads,
    )

    return out.reshape(orig_shape)


__all__ = ["solution"]