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
#include <cuda_fp16.h>

#include <cstdint>

#define FP8_E4M3_MAX 448.0f

// -----------------------------------------------------------------------------
// dtype helpers
// -----------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float load_as_float(T v);

template <>
__device__ __forceinline__ float load_as_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float load_as_float<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T float_to_dtype(float v);

template <>
__device__ __forceinline__ float float_to_dtype<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_dtype<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <>
__device__ __forceinline__ __half float_to_dtype<__half>(float v) {
    return __float2half_rn(v);
}

template <typename T>
__device__ __forceinline__ float store_hist_value(T* out, int64_t idx, float v) {
    T q = float_to_dtype<T>(v);
    out[idx] = q;
    return load_as_float<T>(q);
}

// -----------------------------------------------------------------------------
// Software E4M3FN round-trip.  Inputs are finite and scaled so |x| <= 448.
// Rounds to nearest-even by using __float2int_rn on the appropriate E4M3 grid.
// -----------------------------------------------------------------------------

__device__ __forceinline__ float fp8_e4m3fn_roundtrip_float(float x) {
    if (x == 0.0f) {
        return x;
    }

    float ax = fabsf(x);
    float q;

    // E4M3FN positive levels:
    //   subnormal + smallest normal grid: step 2^-9 up to 2^-5
    //   normal binade with exponent e: step 2^(e-3)
    if (ax < 0.03125f) {  // 2^-5
        int k = __float2int_rn(ax * 512.0f);
        q = ((float)k) * 0.001953125f;  // 2^-9
    } else {
        union {
            float f;
            uint32_t u;
        } u;
        u.f = ax;
        int e = (int)((u.u >> 23) & 0xff) - 127;

        if (e > 8) {
            q = FP8_E4M3_MAX;
        } else {
            float inv_step = ldexpf(1.0f, 3 - e);
            int k = __float2int_rn(ax * inv_step);
            q = ldexpf((float)k, e - 3);
            if (q > FP8_E4M3_MAX) {
                q = FP8_E4M3_MAX;
            }
        }
    }

    return copysignf(q, x);
}

// -----------------------------------------------------------------------------
// Stage 1: shard absmax reduction
// -----------------------------------------------------------------------------

template <typename T>
__global__ void reduce_absmax_kernel(
    const T* __restrict__ x,
    float* __restrict__ partials,
    int64_t n
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + tid;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    float local = 0.0f;
    for (; idx < n; idx += stride) {
        float v = fabsf(load_as_float<T>(x[idx]));
        local = fmaxf(local, v);
    }

    smem[tid] = local;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = smem[0];
    }
}

// -----------------------------------------------------------------------------
// Stage 2: update rolling history and compute scale
// updated_hist = roll(old_hist, -1); updated_hist[-1] = cur_abs_max
// scale = max(updated_hist).clamp_min(1e-12) / 448
// -----------------------------------------------------------------------------

template <typename H>
__global__ void update_history_scale_kernel(
    const H* __restrict__ old_hist,
    H* __restrict__ updated_hist,
    const float* __restrict__ partials,
    float* __restrict__ scale,
    int64_t hist_n,
    int num_partials
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    float cur = 0.0f;
    for (int i = tid; i < num_partials; i += blockDim.x) {
        cur = fmaxf(cur, partials[i]);
    }

    smem[tid] = cur;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    cur = smem[0];
    __syncthreads();

    float hist_max = 0.0f;

    for (int64_t i = tid; i < hist_n; i += blockDim.x) {
        float stored_v;
        if (i == hist_n - 1) {
            stored_v = store_hist_value<H>(updated_hist, i, cur);
        } else {
            H v = old_hist[i + 1];
            updated_hist[i] = v;
            stored_v = load_as_float<H>(v);
        }
        hist_max = fmaxf(hist_max, stored_v);
    }

    smem[tid] = hist_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = fmaxf(smem[0], 1.0e-12f);
        scale[0] = m / FP8_E4M3_MAX;
    }
}

// -----------------------------------------------------------------------------
// Stage 3: local BF16/FP32/FP16 -> FP8 E4M3FN -> original dtype reconstruction,
// written directly to symmetric-memory gather buffer.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void quant_roundtrip_to_symm_kernel(
    const T* __restrict__ x,
    const float* __restrict__ scale,
    T* __restrict__ symm_out,
    int64_t n
) {
    float s = scale[0];
    float inv_s = 1.0f / s;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float xf = load_as_float<T>(x[idx]);
        float qf = fp8_e4m3fn_roundtrip_float(xf * inv_s);
        float recon = qf * s;
        symm_out[idx] = float_to_dtype<T>(recon);
    }
}

// -----------------------------------------------------------------------------
// Stage 4: all-gather by peer UVA loads from symmetric buffers.
// full[r * P + i] = peer_buffer[r][i]
// -----------------------------------------------------------------------------

template <typename T>
__global__ void allgather_peer_load_kernel(
    const int64_t* __restrict__ ptrs,
    T* __restrict__ full,
    int world_size,
    int64_t p
) {
    int64_t total = (int64_t)world_size * p;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int r = (int)(idx / p);
        int64_t off = idx - (int64_t)r * p;
        const T* src = reinterpret_cast<const T*>((uintptr_t)ptrs[r]);
        full[idx] = src[off];
    }
}

// -----------------------------------------------------------------------------
// C++ launchers
// -----------------------------------------------------------------------------

static inline int ceil_div_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

void launch_local_fp8_pack(
    torch::Tensor shard,
    torch::Tensor old_hist,
    torch::Tensor updated_hist,
    torch::Tensor scale,
    torch::Tensor partials,
    torch::Tensor symm_buf
) {
    TORCH_CHECK(shard.is_cuda(), "shard must be CUDA");
    TORCH_CHECK(old_hist.is_cuda() && updated_hist.is_cuda(), "history tensors must be CUDA");
    TORCH_CHECK(scale.is_cuda() && partials.is_cuda() && symm_buf.is_cuda(), "buffers must be CUDA");
    TORCH_CHECK(shard.is_contiguous(), "shard must be contiguous");
    TORCH_CHECK(old_hist.is_contiguous() && updated_hist.is_contiguous(), "history must be contiguous");
    TORCH_CHECK(symm_buf.is_contiguous(), "symm_buf must be contiguous");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(partials.scalar_type() == torch::kFloat32, "partials must be float32");
    TORCH_CHECK(old_hist.scalar_type() == updated_hist.scalar_type(), "history dtypes must match");
    TORCH_CHECK(shard.scalar_type() == symm_buf.scalar_type(), "shard/symm dtype mismatch");

    int64_t n = shard.numel();
    int64_t hist_n = old_hist.numel();

    TORCH_CHECK(hist_n > 0, "amax_history must be non-empty");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int blocks = ceil_div_i64(n, threads);
    if (blocks < 1) blocks = 1;
    if (blocks > (int)partials.numel()) blocks = (int)partials.numel();

    size_t shmem_reduce = threads * sizeof(float);

    if (shard.scalar_type() == torch::kBFloat16) {
        const __nv_bfloat16* x =
            reinterpret_cast<const __nv_bfloat16*>(shard.data_ptr<at::BFloat16>());
        reduce_absmax_kernel<__nv_bfloat16><<<blocks, threads, shmem_reduce, stream>>>(
            x, partials.data_ptr<float>(), n);
    } else if (shard.scalar_type() == torch::kFloat32) {
        reduce_absmax_kernel<float><<<blocks, threads, shmem_reduce, stream>>>(
            shard.data_ptr<float>(), partials.data_ptr<float>(), n);
    } else if (shard.scalar_type() == torch::kFloat16) {
        const __half* x =
            reinterpret_cast<const __half*>(shard.data_ptr<at::Half>());
        reduce_absmax_kernel<__half><<<blocks, threads, shmem_reduce, stream>>>(
            x, partials.data_ptr<float>(), n);
    } else {
        TORCH_CHECK(false, "supported shard dtypes: bfloat16, float32, float16");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int hist_threads = 1024;
    size_t shmem_hist = hist_threads * sizeof(float);

    if (old_hist.scalar_type() == torch::kFloat32) {
        update_history_scale_kernel<float><<<1, hist_threads, shmem_hist, stream>>>(
            old_hist.data_ptr<float>(),
            updated_hist.data_ptr<float>(),
            partials.data_ptr<float>(),
            scale.data_ptr<float>(),
            hist_n,
            blocks);
    } else if (old_hist.scalar_type() == torch::kBFloat16) {
        const __nv_bfloat16* oldp =
            reinterpret_cast<const __nv_bfloat16*>(old_hist.data_ptr<at::BFloat16>());
        __nv_bfloat16* outp =
            reinterpret_cast<__nv_bfloat16*>(updated_hist.data_ptr<at::BFloat16>());
        update_history_scale_kernel<__nv_bfloat16><<<1, hist_threads, shmem_hist, stream>>>(
            oldp, outp, partials.data_ptr<float>(), scale.data_ptr<float>(), hist_n, blocks);
    } else if (old_hist.scalar_type() == torch::kFloat16) {
        const __half* oldp =
            reinterpret_cast<const __half*>(old_hist.data_ptr<at::Half>());
        __half* outp =
            reinterpret_cast<__half*>(updated_hist.data_ptr<at::Half>());
        update_history_scale_kernel<__half><<<1, hist_threads, shmem_hist, stream>>>(
            oldp, outp, partials.data_ptr<float>(), scale.data_ptr<float>(), hist_n, blocks);
    } else {
        TORCH_CHECK(false, "supported history dtypes: float32, bfloat16, float16");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int q_threads = 256;
    int q_blocks = ceil_div_i64(n, q_threads);
    if (q_blocks < 1) q_blocks = 1;
    if (q_blocks > 65535) q_blocks = 65535;

    if (shard.scalar_type() == torch::kBFloat16) {
        const __nv_bfloat16* x =
            reinterpret_cast<const __nv_bfloat16*>(shard.data_ptr<at::BFloat16>());
        __nv_bfloat16* out =
            reinterpret_cast<__nv_bfloat16*>(symm_buf.data_ptr<at::BFloat16>());
        quant_roundtrip_to_symm_kernel<__nv_bfloat16><<<q_blocks, q_threads, 0, stream>>>(
            x, scale.data_ptr<float>(), out, n);
    } else if (shard.scalar_type() == torch::kFloat32) {
        quant_roundtrip_to_symm_kernel<float><<<q_blocks, q_threads, 0, stream>>>(
            shard.data_ptr<float>(), scale.data_ptr<float>(), symm_buf.data_ptr<float>(), n);
    } else if (shard.scalar_type() == torch::kFloat16) {
        const __half* x =
            reinterpret_cast<const __half*>(shard.data_ptr<at::Half>());
        __half* out =
            reinterpret_cast<__half*>(symm_buf.data_ptr<at::Half>());
        quant_roundtrip_to_symm_kernel<__half><<<q_blocks, q_threads, 0, stream>>>(
            x, scale.data_ptr<float>(), out, n);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_allgather_peer_load(
    torch::Tensor ptrs,
    torch::Tensor full,
    int64_t p
) {
    TORCH_CHECK(ptrs.is_cuda(), "ptrs must be CUDA");
    TORCH_CHECK(full.is_cuda(), "full must be CUDA");
    TORCH_CHECK(ptrs.scalar_type() == torch::kInt64, "ptrs must be int64");
    TORCH_CHECK(ptrs.is_contiguous() && full.is_contiguous(), "tensors must be contiguous");

    int world_size = (int)ptrs.numel();
    int64_t total = (int64_t)world_size * p;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int blocks = ceil_div_i64(total, threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    const int64_t* d_ptrs = ptrs.data_ptr<int64_t>();

    if (full.scalar_type() == torch::kBFloat16) {
        __nv_bfloat16* out =
            reinterpret_cast<__nv_bfloat16*>(full.data_ptr<at::BFloat16>());
        allgather_peer_load_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            d_ptrs, out, world_size, p);
    } else if (full.scalar_type() == torch::kFloat32) {
        allgather_peer_load_kernel<float><<<blocks, threads, 0, stream>>>(
            d_ptrs, full.data_ptr<float>(), world_size, p);
    } else if (full.scalar_type() == torch::kFloat16) {
        __half* out =
            reinterpret_cast<__half*>(full.data_ptr<at::Half>());
        allgather_peer_load_kernel<__half><<<blocks, threads, 0, stream>>>(
            d_ptrs, out, world_size, p);
    } else {
        TORCH_CHECK(false, "supported full dtypes: bfloat16, float32, float16");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_local_fp8_pack", &launch_local_fp8_pack,
          "Update amax history, compute scale, FP8 round-trip, pack into symmetric buffer");
    m.def("launch_allgather_peer_load", &launch_allgather_peer_load,
          "All-gather from symmetric peer buffers via UVA loads");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fp8_param_allgather_symm_uva_ext", CUDA_SRC)
    return _ext


_resource_cache: dict[tuple, dict] = {}


def _device_key(device: torch.device) -> tuple[str, int | None]:
    d = torch.device(device)
    return (d.type, d.index)


def _get_resources(
    p: int,
    shard_dtype: torch.dtype,
    hist_shape: tuple[int, ...],
    hist_dtype: torch.dtype,
    device: torch.device,
    world_size: int,
) -> dict:
    key = (p, shard_dtype, hist_shape, hist_dtype, _device_key(device), world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    symm_buf = symm_mem.empty((p,), device=device, dtype=shard_dtype)
    hdl = symm_mem.rendezvous(symm_buf, dist.group.WORLD)

    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    state = {
        "symm_buf": symm_buf,
        "hdl": hdl,
        "ptrs": ptrs,
        "partials": torch.empty((1024,), device=device, dtype=torch.float32),
        "scale": torch.empty((1,), device=device, dtype=torch.float32),
        "hist_outs": [
            torch.empty(hist_shape, device=device, dtype=hist_dtype),
            torch.empty(hist_shape, device=device, dtype=hist_dtype),
        ],
        "full_outs": [
            torch.empty((world_size * p,), device=device, dtype=shard_dtype),
            torch.empty((world_size * p,), device=device, dtype=shard_dtype),
        ],
        "toggle": 0,
    }
    _resource_cache[key] = state
    return state


@torch.no_grad()
def solution(flat_param_shard: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    """
    FP8 all-gather for parameter unshard.

    Custom CUDA path:
      1. local absmax reduction + rolling amax_history update + scale computation
      2. local dtype -> FP8 E4M3FN -> dtype reconstruction into symmetric memory
      3. symmetric-memory barrier
      4. all-gather by CUDA peer UVA loads from symmetric buffers
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert flat_param_shard.is_cuda, "flat_param_shard must be CUDA"
    assert amax_history.is_cuda, "amax_history must be CUDA"
    assert flat_param_shard.dtype in (torch.bfloat16, torch.float32, torch.float16)
    assert amax_history.dtype in (torch.float32, torch.bfloat16, torch.float16)

    world_size = dist.get_world_size()
    p = flat_param_shard.numel()

    shard = flat_param_shard if flat_param_shard.is_contiguous() else flat_param_shard.contiguous()
    hist = amax_history if amax_history.is_contiguous() else amax_history.contiguous()

    ext = _get_ext()

    state = _get_resources(
        p=p,
        shard_dtype=shard.dtype,
        hist_shape=tuple(hist.shape),
        hist_dtype=hist.dtype,
        device=shard.device,
        world_size=world_size,
    )

    state["toggle"] ^= 1
    buf_idx = state["toggle"]

    updated_hist = state["hist_outs"][buf_idx]
    if updated_hist.data_ptr() == hist.data_ptr():
        buf_idx ^= 1
        updated_hist = state["hist_outs"][buf_idx]

    full = state["full_outs"][buf_idx]

    ext.launch_local_fp8_pack(
        shard,
        hist,
        updated_hist,
        state["scale"],
        state["partials"],
        state["symm_buf"],
    )

    # Symmetric-memory synchronization: publishes this rank's reconstructed shard
    # before peer-load all-gather reads it.
    state["hdl"].barrier(channel=0)

    ext.launch_allgather_peer_load(state["ptrs"], full, p)

    return full, updated_hist


__all__ = ["solution"]