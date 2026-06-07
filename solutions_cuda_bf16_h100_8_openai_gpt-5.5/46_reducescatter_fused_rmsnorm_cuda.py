from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>

#ifndef MAX_OPTIN_SMEM
#define MAX_OPTIN_SMEM 98304
#endif

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        scratch[wid] = v;
    }
    __syncthreads();

    v = 0.0f;
    if (wid == 0) {
        v = (lane < nwarp) ? scratch[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            scratch[0] = v;
        }
    }
    __syncthreads();
    return scratch[0];
}

__device__ __forceinline__ float load_gamma_val(const void* gamma, int idx, int gamma_dtype) {
    // gamma_dtype: 0 = bf16, 1 = fp32
    if (gamma_dtype == 0) {
        const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(gamma);
        return __bfloat162float(g[idx]);
    } else {
        const float* g = reinterpret_cast<const float*>(gamma);
        return g[idx];
    }
}

__global__ void rs_rmsnorm_bf16_shared_kernel(
    const long long* __restrict__ ptrs,
    const void* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out,
    int64_t rows,
    int64_t chunk,
    int hidden,
    int rank,
    int world_size,
    float eps,
    int gamma_dtype
) {
    const int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float smem[];
    float* xbuf = smem;
    float* red = smem + hidden;

    const int64_t base = (int64_t)rank * chunk + row * (int64_t)hidden;
    float ss = 0.0f;
    const float inv_world = 1.0f / (float)world_size;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* src =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[r]);
                s += __bfloat162float(src[base + h]);
            }
        }

        // Match the reference ordering closely: BF16 reduce-scatter average is
        // materialized before RMSNorm's float() cast.
        __nv_bfloat16 xb = __float2bfloat16(s * inv_world);
        float x = __bfloat162float(xb);
        xbuf[h] = x;
        ss += x * x;
    }

    float total_ss = block_reduce_sum(ss, red);
    float inv_rms = rsqrtf(total_ss / (float)hidden + eps);

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float y = xbuf[h] * inv_rms * load_gamma_val(gamma, h, gamma_dtype);
        out[row * (int64_t)hidden + h] = __float2bfloat16(y);
    }
}

__global__ void rs_rmsnorm_bf16_twopass_kernel(
    const long long* __restrict__ ptrs,
    const void* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out,
    int64_t rows,
    int64_t chunk,
    int hidden,
    int rank,
    int world_size,
    float eps,
    int gamma_dtype
) {
    const int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float red[];
    const int64_t base = (int64_t)rank * chunk + row * (int64_t)hidden;
    const float inv_world = 1.0f / (float)world_size;

    float ss = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* src =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[r]);
                s += __bfloat162float(src[base + h]);
            }
        }
        __nv_bfloat16 xb = __float2bfloat16(s * inv_world);
        float x = __bfloat162float(xb);
        ss += x * x;
    }

    float total_ss = block_reduce_sum(ss, red);
    float inv_rms = rsqrtf(total_ss / (float)hidden + eps);

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* src =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[r]);
                s += __bfloat162float(src[base + h]);
            }
        }
        __nv_bfloat16 xb = __float2bfloat16(s * inv_world);
        float x = __bfloat162float(xb);
        float y = x * inv_rms * load_gamma_val(gamma, h, gamma_dtype);
        out[row * (int64_t)hidden + h] = __float2bfloat16(y);
    }
}

__global__ void rs_rmsnorm_f32_shared_kernel(
    const long long* __restrict__ ptrs,
    const void* __restrict__ gamma,
    float* __restrict__ out,
    int64_t rows,
    int64_t chunk,
    int hidden,
    int rank,
    int world_size,
    float eps,
    int gamma_dtype
) {
    const int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float smem[];
    float* xbuf = smem;
    float* red = smem + hidden;

    const int64_t base = (int64_t)rank * chunk + row * (int64_t)hidden;
    const float inv_world = 1.0f / (float)world_size;

    float ss = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const float* src = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
                s += src[base + h];
            }
        }
        float x = s * inv_world;
        xbuf[h] = x;
        ss += x * x;
    }

    float total_ss = block_reduce_sum(ss, red);
    float inv_rms = rsqrtf(total_ss / (float)hidden + eps);

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[row * (int64_t)hidden + h] =
            xbuf[h] * inv_rms * load_gamma_val(gamma, h, gamma_dtype);
    }
}

__global__ void rs_rmsnorm_f32_twopass_kernel(
    const long long* __restrict__ ptrs,
    const void* __restrict__ gamma,
    float* __restrict__ out,
    int64_t rows,
    int64_t chunk,
    int hidden,
    int rank,
    int world_size,
    float eps,
    int gamma_dtype
) {
    const int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float red[];
    const int64_t base = (int64_t)rank * chunk + row * (int64_t)hidden;
    const float inv_world = 1.0f / (float)world_size;

    float ss = 0.0f;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const float* src = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
                s += src[base + h];
            }
        }
        float x = s * inv_world;
        ss += x * x;
    }

    float total_ss = block_reduce_sum(ss, red);
    float inv_rms = rsqrtf(total_ss / (float)hidden + eps);

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const float* src = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
                s += src[base + h];
            }
        }
        float x = s * inv_world;
        out[row * (int64_t)hidden + h] =
            x * inv_rms * load_gamma_val(gamma, h, gamma_dtype);
    }
}

void copy_into_symm(torch::Tensor src, torch::Tensor dst, int64_t nbytes) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA tensors");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        (size_t)nbytes,
        cudaMemcpyDeviceToDevice,
        stream));
}

void launch_rs_rmsnorm(
    torch::Tensor ptrs_tensor,
    torch::Tensor gamma,
    torch::Tensor out,
    int64_t rows,
    int64_t chunk,
    int64_t hidden64,
    int rank,
    int world_size,
    double eps,
    int input_dtype,
    int gamma_dtype
) {
    TORCH_CHECK(ptrs_tensor.is_cuda(), "ptrs_tensor must be CUDA");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(), "gamma must be contiguous CUDA");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out must be contiguous CUDA");
    TORCH_CHECK(world_size > 0 && world_size <= 8, "this H100/NVLink kernel expects world_size in [1, 8]");
    TORCH_CHECK(hidden64 > 0 && hidden64 <= INT_MAX, "invalid hidden size");
    TORCH_CHECK(rows >= 0, "invalid rows");

    if (rows == 0) {
        return;
    }

    const int hidden = (int)hidden64;
    int threads = 256;
    if (hidden <= 64) {
        threads = 64;
    } else if (hidden <= 128) {
        threads = 128;
    }

    const dim3 grid((unsigned int)rows);
    const dim3 block(threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* d_ptrs =
        reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>());
    const void* gptr = gamma.data_ptr();

    const size_t shared_smem = ((size_t)hidden + 32u) * sizeof(float);
    const size_t reduce_smem = 32u * sizeof(float);

    if (input_dtype == 0) {
        __nv_bfloat16* optr =
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());

        if (shared_smem <= (size_t)MAX_OPTIN_SMEM) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                rs_rmsnorm_bf16_shared_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                MAX_OPTIN_SMEM));
            rs_rmsnorm_bf16_shared_kernel<<<grid, block, shared_smem, stream>>>(
                d_ptrs, gptr, optr, rows, chunk, hidden, rank, world_size,
                (float)eps, gamma_dtype);
        } else {
            rs_rmsnorm_bf16_twopass_kernel<<<grid, block, reduce_smem, stream>>>(
                d_ptrs, gptr, optr, rows, chunk, hidden, rank, world_size,
                (float)eps, gamma_dtype);
        }
    } else {
        float* optr = out.data_ptr<float>();

        if (shared_smem <= (size_t)MAX_OPTIN_SMEM) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                rs_rmsnorm_f32_shared_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                MAX_OPTIN_SMEM));
            rs_rmsnorm_f32_shared_kernel<<<grid, block, shared_smem, stream>>>(
                d_ptrs, gptr, optr, rows, chunk, hidden, rank, world_size,
                (float)eps, gamma_dtype);
        } else {
            rs_rmsnorm_f32_twopass_kernel<<<grid, block, reduce_smem, stream>>>(
                d_ptrs, gptr, optr, rows, chunk, hidden, rank, world_size,
                (float)eps, gamma_dtype);
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_into_symm", &copy_into_symm,
          "Async D2D copy into symmetric memory buffer");
    m.def("launch_rs_rmsnorm", &launch_rs_rmsnorm,
          "Fused symmetric-memory reduce-scatter average + RMSNorm");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "symm_rs_rmsnorm_bf16_h100_ext",
            CUDA_SRC,
        )
    return _ext


_resource_cache: dict[tuple, tuple[Tensor, object, Tensor, Tensor]] = {}


def _get_resources(
    n: int,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
):
    key = (n, rows, hidden, dtype, device)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty((n,), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    out = torch.empty((rows, hidden), device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (buf, hdl, out, ptrs_tensor)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    rs_input_1d: Tensor,
    gamma: Tensor,
    eps: float,
) -> Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert rs_input_1d.is_cuda, "rs_input_1d must be CUDA"
    assert gamma.is_cuda, "gamma must be CUDA"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    n = rs_input_1d.numel()
    assert n % world_size == 0
    chunk = n // world_size

    hidden = gamma.numel()
    assert hidden > 0
    assert chunk % hidden == 0, f"chunk ({chunk}) must divide hidden ({hidden})"
    rows = chunk // hidden

    if rows == 0:
        return torch.empty((rows, hidden), dtype=rs_input_1d.dtype, device=rs_input_1d.device)

    if rs_input_1d.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("optimized CUDA path supports bfloat16 and float32 inputs")

    ext = _get_ext()

    inp = rs_input_1d
    if not inp.is_contiguous():
        inp = inp.contiguous()

    if gamma.dtype == torch.bfloat16:
        gamma_c = gamma if gamma.is_contiguous() else gamma.contiguous()
        gamma_dtype = 0
    elif gamma.dtype == torch.float32:
        gamma_c = gamma if gamma.is_contiguous() else gamma.contiguous()
        gamma_dtype = 1
    else:
        gamma_c = gamma.float().contiguous()
        gamma_dtype = 1

    buf, hdl, out, ptrs_tensor = _get_resources(
        n,
        rows,
        hidden,
        inp.dtype,
        inp.device,
    )

    ext.copy_into_symm(inp, buf, n * inp.element_size())

    # Symmetric-memory device-side barrier: all ranks have published their
    # full RS input before peer UVA loads begin.
    hdl.barrier(channel=0)

    input_dtype = 0 if inp.dtype == torch.bfloat16 else 1
    ext.launch_rs_rmsnorm(
        ptrs_tensor,
        gamma_c,
        out,
        rows,
        chunk,
        hidden,
        rank,
        world_size,
        float(eps),
        input_dtype,
        gamma_dtype,
    )

    # Prevent the next invocation from overwriting this rank's symmetric input
    # while a slower peer is still pulling it.
    hdl.barrier(channel=1)

    return out


__all__ = ["solution"]