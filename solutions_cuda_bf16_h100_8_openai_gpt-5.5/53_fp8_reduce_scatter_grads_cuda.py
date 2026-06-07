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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cmath>

#define FP8_E4M3_MAX 448.0f

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ float abs_f32(float x) {
    return fabsf(x);
}

// Emulate torch.float8_e4m3fn round-trip value in float:
//   q = round_to_e4m3fn(x / scale)
//   return float(q) * scale
//
// This path is finite/saturating for the range used here.  Scale is chosen from
// amax / 448, so normal inputs satisfy abs(x/scale) <= 448.
__device__ __forceinline__ float fp8_e4m3fn_roundtrip_f32(float x, float scale) {
    if (!(scale > 0.0f)) {
        return 0.0f;
    }

    float y = x / scale;

    if (isnan(y)) {
        return y;
    }

    float sign = copysignf(1.0f, y);
    float a = fabsf(y);

    if (a == 0.0f) {
        return copysignf(0.0f, y);
    }

    // E4M3FN:
    // exponent bias = 7
    // min normal = 2^-6
    // subnormal quantum = 2^-9
    // max finite = 448
    constexpr float MIN_NORMAL = 0.015625f;       // 2^-6
    constexpr float SUB_QUANT  = 0.001953125f;    // 2^-9

    float qv;

    if (a < MIN_NORMAL) {
        // Subnormal / zero. RNE to multiples of 2^-9.
        float m = nearbyintf(a * 512.0f);
        if (m <= 0.0f) {
            qv = 0.0f;
        } else {
            if (m > 8.0f) m = 8.0f;  // m==8 is numerically min-normal.
            qv = m * SUB_QUANT;
        }
    } else {
        int e2;
        frexpf(a, &e2);          // a = frac * 2^e2, frac in [0.5, 1)
        int exp_unbiased = e2 - 1;

        float unit = ldexpf(1.0f, exp_unbiased);
        float norm = a / unit;   // [1, 2)

        float mf = nearbyintf((norm - 1.0f) * 8.0f);
        int mant = (int)mf;
        int exp_field = exp_unbiased + 7;

        if (mant >= 8) {
            mant = 0;
            exp_unbiased += 1;
            exp_field += 1;
        }

        // Avoid NaN code 0x7f; max finite is 0x7e = 448.
        if (exp_field > 15 || (exp_field == 15 && mant > 6)) {
            qv = FP8_E4M3_MAX;
        } else if (exp_field <= 0) {
            // Should only happen near boundary; value-wise fallback.
            float m = nearbyintf(a * 512.0f);
            if (m <= 0.0f) qv = 0.0f;
            else qv = fminf(m, 8.0f) * SUB_QUANT;
        } else {
            qv = ldexpf(1.0f + ((float)mant) * 0.125f, exp_unbiased);
        }
    }

    return sign * qv * scale;
}

__device__ __forceinline__ float load_bf16_as_f32(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

__device__ __forceinline__ float load_f16_as_f32(const __half* p) {
    return __half2float(*p);
}

__device__ __forceinline__ float load_f32_as_f32(const float* p) {
    return *p;
}

__device__ __forceinline__ void store_bf16_from_f32(__nv_bfloat16* p, float x) {
    *p = __float2bfloat16(x);
}

__device__ __forceinline__ void store_f16_from_f32(__half* p, float x) {
    *p = __float2half(x);
}

__device__ __forceinline__ void store_f32_from_f32(float* p, float x) {
    *p = x;
}

// -----------------------------------------------------------------------------
// Stage 1: copy local full gradients into symmetric memory and reduce absmax.
// -----------------------------------------------------------------------------

template <typename T, typename Loader>
__global__ void prepare_copy_absmax_kernel(
    const T* __restrict__ x,
    T* __restrict__ symm_x,
    float* __restrict__ block_max,
    int64_t n,
    Loader loader
) {
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + tid;

    float local_max = 0.0f;

    for (; i < n; i += stride) {
        T v = x[i];
        symm_x[i] = v;
        float vf = loader(&x[i]);
        local_max = fmaxf(local_max, fabsf(vf));
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = smem[0];
    }
}

struct Bf16Loader {
    __device__ __forceinline__ float operator()(const __nv_bfloat16* p) const {
        return __bfloat162float(*p);
    }
};
struct F16Loader {
    __device__ __forceinline__ float operator()(const __half* p) const {
        return __half2float(*p);
    }
};
struct F32Loader {
    __device__ __forceinline__ float operator()(const float* p) const {
        return *p;
    }
};

// -----------------------------------------------------------------------------
// Stage 2: update rolling amax history and publish local scale in symm memory.
// -----------------------------------------------------------------------------

template <typename HistT>
__device__ __forceinline__ float hist_load_as_f32(const HistT* p);

template <>
__device__ __forceinline__ float hist_load_as_f32<float>(const float* p) {
    return *p;
}

template <>
__device__ __forceinline__ float hist_load_as_f32<__half>(const __half* p) {
    return __half2float(*p);
}

template <>
__device__ __forceinline__ float hist_load_as_f32<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename HistT>
__device__ __forceinline__ void hist_store_from_f32(HistT* p, float x);

template <>
__device__ __forceinline__ void hist_store_from_f32<float>(float* p, float x) {
    *p = x;
}

template <>
__device__ __forceinline__ void hist_store_from_f32<__half>(__half* p, float x) {
    *p = __float2half(x);
}

template <>
__device__ __forceinline__ void hist_store_from_f32<__nv_bfloat16>(__nv_bfloat16* p, float x) {
    *p = __float2bfloat16(x);
}

template <typename HistT>
__global__ void finalize_history_scale_kernel(
    const float* __restrict__ block_max,
    int num_blocks,
    const HistT* __restrict__ old_hist,
    HistT* __restrict__ new_hist,
    int64_t hist_len,
    float* __restrict__ symm_scale
) {
    __shared__ float smem[1024];

    const int tid = threadIdx.x;

    float cur = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        cur = fmaxf(cur, block_max[i]);
    }

    smem[tid] = cur;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    float cur_abs_max = smem[0];

    // Roll left and append cur_abs_max converted to history dtype, matching:
    // out = torch.roll(hist, -1); out[-1] = cur_abs_max.to(out.dtype)
    for (int64_t i = tid; i < hist_len; i += blockDim.x) {
        float v;
        if (i == hist_len - 1) {
            hist_store_from_f32<HistT>(&new_hist[i], cur_abs_max);
        } else {
            v = hist_load_as_f32<HistT>(&old_hist[i + 1]);
            hist_store_from_f32<HistT>(&new_hist[i], v);
        }
    }

    __syncthreads();

    float local_hist_max = 0.0f;
    for (int64_t i = tid; i < hist_len; i += blockDim.x) {
        float v;
        if (i == hist_len - 1) {
            // Important for fp16/bf16 histories: max uses the stored rounded value.
            HistT tmp;
            hist_store_from_f32<HistT>(&tmp, cur_abs_max);
            v = hist_load_as_f32<HistT>(&tmp);
        } else {
            v = hist_load_as_f32<HistT>(&old_hist[i + 1]);
        }
        local_hist_max = fmaxf(local_hist_max, v);
    }

    smem[tid] = local_hist_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = fmaxf(smem[0], 1.0e-12f);
        symm_scale[0] = m / FP8_E4M3_MAX;
    }
}

// -----------------------------------------------------------------------------
// Stage 3: fused FP8 round-trip + reduce-scatter average.
// Each rank reads only its own shard from every peer via UVA peer pointers.
// -----------------------------------------------------------------------------

__global__ void rs_fp8_bf16_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ scale_ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int rank,
    int64_t shard_elems
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t off = (int64_t)rank * shard_elems;

    const float inv_w = 1.0f / (float)world_size;

    for (; idx < shard_elems; idx += stride) {
        float sum = 0.0f;
        int64_t g = off + idx;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= world_size) break;

            const __nv_bfloat16* peer =
                reinterpret_cast<const __nv_bfloat16*>((uintptr_t)data_ptrs[r]);
            const float* sp =
                reinterpret_cast<const float*>((uintptr_t)scale_ptrs[r]);

            float scale = sp[0];
            float x = __bfloat162float(peer[g]);
            float recon_f = fp8_e4m3fn_roundtrip_f32(x, scale);

            // Reference materializes recon as BF16 before reduce-scatter.
            float recon_bf16 = __bfloat162float(__float2bfloat16(recon_f));
            sum += recon_bf16;
        }

        out[idx] = __float2bfloat16(sum * inv_w);
    }
}

__global__ void rs_fp8_f16_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ scale_ptrs,
    __half* __restrict__ out,
    int world_size,
    int rank,
    int64_t shard_elems
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t off = (int64_t)rank * shard_elems;

    const float inv_w = 1.0f / (float)world_size;

    for (; idx < shard_elems; idx += stride) {
        float sum = 0.0f;
        int64_t g = off + idx;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= world_size) break;

            const __half* peer =
                reinterpret_cast<const __half*>((uintptr_t)data_ptrs[r]);
            const float* sp =
                reinterpret_cast<const float*>((uintptr_t)scale_ptrs[r]);

            float scale = sp[0];
            float x = __half2float(peer[g]);
            float recon_f = fp8_e4m3fn_roundtrip_f32(x, scale);

            // Reference materializes recon as FP16 before reduce-scatter.
            float recon_f16 = __half2float(__float2half(recon_f));
            sum += recon_f16;
        }

        out[idx] = __float2half(sum * inv_w);
    }
}

__global__ void rs_fp8_f32_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ scale_ptrs,
    float* __restrict__ out,
    int world_size,
    int rank,
    int64_t shard_elems
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t off = (int64_t)rank * shard_elems;

    const float inv_w = 1.0f / (float)world_size;

    for (; idx < shard_elems; idx += stride) {
        float sum = 0.0f;
        int64_t g = off + idx;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= world_size) break;

            const float* peer =
                reinterpret_cast<const float*>((uintptr_t)data_ptrs[r]);
            const float* sp =
                reinterpret_cast<const float*>((uintptr_t)scale_ptrs[r]);

            float scale = sp[0];
            float x = peer[g];
            float recon_f = fp8_e4m3fn_roundtrip_f32(x, scale);
            sum += recon_f;
        }

        out[idx] = sum * inv_w;
    }
}

// -----------------------------------------------------------------------------
// Launchers
// dtype_enum: 0=bf16, 1=f16, 2=f32
// hist_enum : 0=f32, 1=f16, 2=bf16
// -----------------------------------------------------------------------------

int launch_prepare_copy_absmax(
    torch::Tensor x,
    torch::Tensor symm_x,
    torch::Tensor block_max,
    int64_t n,
    int dtype_enum
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(symm_x.is_cuda(), "symm_x must be CUDA");
    TORCH_CHECK(block_max.is_cuda(), "block_max must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(symm_x.is_contiguous(), "symm_x must be contiguous");

    constexpr int threads = 256;
    int64_t need_blocks = (n + threads - 1) / threads;
    int max_blocks = (int)block_max.numel();
    int blocks = (int)min<int64_t>(max<int64_t>(need_blocks, 1), max_blocks);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t shmem = threads * sizeof(float);

    if (dtype_enum == 0) {
        prepare_copy_absmax_kernel<<<blocks, threads, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(symm_x.data_ptr<at::BFloat16>()),
            block_max.data_ptr<float>(),
            n,
            Bf16Loader{}
        );
    } else if (dtype_enum == 1) {
        prepare_copy_absmax_kernel<<<blocks, threads, shmem, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(symm_x.data_ptr<at::Half>()),
            block_max.data_ptr<float>(),
            n,
            F16Loader{}
        );
    } else {
        prepare_copy_absmax_kernel<<<blocks, threads, shmem, stream>>>(
            x.data_ptr<float>(),
            symm_x.data_ptr<float>(),
            block_max.data_ptr<float>(),
            n,
            F32Loader{}
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return blocks;
}

void launch_finalize_history_scale(
    torch::Tensor block_max,
    int num_blocks,
    torch::Tensor old_hist,
    torch::Tensor new_hist,
    torch::Tensor symm_scale,
    int hist_enum
) {
    TORCH_CHECK(block_max.is_cuda(), "block_max must be CUDA");
    TORCH_CHECK(old_hist.is_cuda(), "old_hist must be CUDA");
    TORCH_CHECK(new_hist.is_cuda(), "new_hist must be CUDA");
    TORCH_CHECK(symm_scale.is_cuda(), "symm_scale must be CUDA");
    TORCH_CHECK(old_hist.is_contiguous(), "old_hist must be contiguous");
    TORCH_CHECK(new_hist.is_contiguous(), "new_hist must be contiguous");

    constexpr int threads = 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t hist_len = old_hist.numel();

    if (hist_enum == 0) {
        finalize_history_scale_kernel<<<1, threads, 0, stream>>>(
            block_max.data_ptr<float>(),
            num_blocks,
            old_hist.data_ptr<float>(),
            new_hist.data_ptr<float>(),
            hist_len,
            symm_scale.data_ptr<float>()
        );
    } else if (hist_enum == 1) {
        finalize_history_scale_kernel<<<1, threads, 0, stream>>>(
            block_max.data_ptr<float>(),
            num_blocks,
            reinterpret_cast<const __half*>(old_hist.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(new_hist.data_ptr<at::Half>()),
            hist_len,
            symm_scale.data_ptr<float>()
        );
    } else {
        finalize_history_scale_kernel<<<1, threads, 0, stream>>>(
            block_max.data_ptr<float>(),
            num_blocks,
            reinterpret_cast<const __nv_bfloat16*>(old_hist.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(new_hist.data_ptr<at::BFloat16>()),
            hist_len,
            symm_scale.data_ptr<float>()
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_rs_fp8_avg(
    torch::Tensor data_ptrs,
    torch::Tensor scale_ptrs,
    torch::Tensor out,
    int world_size,
    int rank,
    int64_t shard_elems,
    int dtype_enum
) {
    TORCH_CHECK(data_ptrs.is_cuda(), "data_ptrs must be CUDA");
    TORCH_CHECK(scale_ptrs.is_cuda(), "scale_ptrs must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");

    constexpr int threads = 256;
    int blocks = (int)((shard_elems + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* dptrs =
        reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>());
    const long long* sptrs =
        reinterpret_cast<const long long*>(scale_ptrs.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        rs_fp8_bf16_kernel<<<blocks, threads, 0, stream>>>(
            dptrs,
            sptrs,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            world_size,
            rank,
            shard_elems
        );
    } else if (dtype_enum == 1) {
        rs_fp8_f16_kernel<<<blocks, threads, 0, stream>>>(
            dptrs,
            sptrs,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            world_size,
            rank,
            shard_elems
        );
    } else {
        rs_fp8_f32_kernel<<<blocks, threads, 0, stream>>>(
            dptrs,
            sptrs,
            out.data_ptr<float>(),
            world_size,
            rank,
            shard_elems
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_prepare_copy_absmax", &launch_prepare_copy_absmax,
          "copy local gradients to symmetric memory and compute block absmax");
    m.def("launch_finalize_history_scale", &launch_finalize_history_scale,
          "update rolling amax history and publish fp8 scale");
    m.def("launch_rs_fp8_avg", &launch_rs_fp8_avg,
          "fused fp8 roundtrip + UVA reduce-scatter average");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fp8_reduce_scatter_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


_MAX_REDUCE_BLOCKS = 4096
_resource_cache: dict[tuple, tuple] = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.float16:
        return 1
    if dtype is torch.float32:
        return 2
    raise TypeError(f"unsupported flat_grads dtype: {dtype}")


def _hist_enum(dtype: torch.dtype) -> int:
    if dtype is torch.float32:
        return 0
    if dtype is torch.float16:
        return 1
    if dtype is torch.bfloat16:
        return 2
    raise TypeError(f"unsupported amax_history dtype: {dtype}")


def _get_resources(
    n: int,
    shard_elems: int,
    grad_dtype: torch.dtype,
    hist_shape: tuple[int, ...],
    hist_dtype: torch.dtype,
    device: torch.device,
    world_size: int,
):
    key = (n, shard_elems, grad_dtype, hist_shape, hist_dtype, device, world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    symm_grads = symm_mem.empty(n, device=device, dtype=grad_dtype)
    grad_hdl = symm_mem.rendezvous(symm_grads, dist.group.WORLD)

    symm_scale = symm_mem.empty(1, device=device, dtype=torch.float32)
    scale_hdl = symm_mem.rendezvous(symm_scale, dist.group.WORLD)

    out_shard = torch.empty(shard_elems, device=device, dtype=grad_dtype)
    updated_hist = torch.empty(hist_shape, device=device, dtype=hist_dtype)

    block_max = torch.empty(_MAX_REDUCE_BLOCKS, device=device, dtype=torch.float32)

    grad_ptrs = torch.tensor(grad_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    scale_ptrs = torch.tensor(scale_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (
        symm_grads,
        grad_hdl,
        symm_scale,
        scale_hdl,
        out_shard,
        updated_hist,
        block_max,
        grad_ptrs,
        scale_ptrs,
    )
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(flat_grads: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    """
    FP8 E4M3 simulated-wire reduce-scatter average over flattened gradients.

    This implementation avoids NCCL reduce_scatter_tensor.  Each rank publishes its
    full local flattened gradient and local FP8 scale in symmetric memory, then each
    rank directly loads only its destination shard from every peer via UVA and fuses
    FP8 round-trip reconstruction with the reduce-scatter average.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert flat_grads.is_cuda, "flat_grads must be CUDA"
    assert amax_history.is_cuda, "amax_history must be CUDA"
    assert flat_grads.is_contiguous(), "flat_grads must be contiguous"
    assert amax_history.is_contiguous(), "amax_history must be contiguous"
    assert amax_history.dim() == 1, "amax_history must be 1D"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    n = flat_grads.numel()
    assert n % world_size == 0, (
        f"flat_grads numel {n} must be divisible by world_size {world_size}"
    )
    shard_elems = n // world_size

    grad_dtype_id = _dtype_enum(flat_grads.dtype)
    hist_dtype_id = _hist_enum(amax_history.dtype)

    (
        symm_grads,
        grad_hdl,
        symm_scale,
        _scale_hdl,
        out_shard,
        updated_hist,
        block_max,
        grad_ptrs,
        scale_ptrs,
    ) = _get_resources(
        n=n,
        shard_elems=shard_elems,
        grad_dtype=flat_grads.dtype,
        hist_shape=tuple(amax_history.shape),
        hist_dtype=amax_history.dtype,
        device=flat_grads.device,
        world_size=world_size,
    )

    ext = _get_ext()

    # Local fused copy -> symmetric memory plus block absmax.
    num_blocks = ext.launch_prepare_copy_absmax(
        flat_grads,
        symm_grads,
        block_max,
        n,
        grad_dtype_id,
    )

    # Roll/update amax history and publish this rank's scalar scale into symmetric memory.
    ext.launch_finalize_history_scale(
        block_max,
        int(num_blocks),
        amax_history,
        updated_hist,
        symm_scale,
        hist_dtype_id,
    )

    # Ensure every rank's symmetric gradient buffer and scale are visible before
    # direct peer loads in the fused reduce-scatter kernel.
    grad_hdl.barrier(channel=0)

    ext.launch_rs_fp8_avg(
        grad_ptrs,
        scale_ptrs,
        out_shard,
        world_size,
        rank,
        shard_elems,
        grad_dtype_id,
    )

    return out_shard, updated_hist


__all__ = ["solution"]