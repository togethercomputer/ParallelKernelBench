import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __forceinline__ float load_as_float(const void* x, int64_t idx, int dtype_enum) {
    if (dtype_enum == 0) {
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(x);
        return __bfloat162float(p[idx]);
    } else if (dtype_enum == 1) {
        const float* p = reinterpret_cast<const float*>(x);
        return p[idx];
    } else {
        const __half* p = reinterpret_cast<const __half*>(x);
        return __half2float(p[idx]);
    }
}

__global__ void block_fp8_quant_kernel(
    const void* __restrict__ x,
    unsigned char* __restrict__ y,
    float* __restrict__ s,
    int64_t num_blocks,
    int block_size,
    int dtype_enum
) {
    extern __shared__ float smem[];

    const int64_t bid = (int64_t)blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int64_t base = bid * (int64_t)block_size;

    float local_max = 0.0f;

    for (int i = tid; i < block_size; i += nthreads) {
        float v = load_as_float(x, base + i, dtype_enum);
        local_max = fmaxf(local_max, fabsf(v));
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    const float maxv = smem[0];
    const float scale = maxv / 448.0f;
    const float inv_scale = (scale == 0.0f) ? 1.0f : (1.0f / scale);

    if (tid == 0) {
        s[bid] = scale;
    }

    for (int i = tid; i < block_size; i += nthreads) {
        float v = load_as_float(x, base + i, dtype_enum) * inv_scale;
        y[base + i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}

__global__ void gather_quantized_and_scales_kernel(
    const long long* __restrict__ y_ptrs,
    const long long* __restrict__ s_ptrs,
    unsigned char* __restrict__ y_out,
    float* __restrict__ s_out,
    int world_size,
    int64_t n_y,
    int64_t n_s,
    bool y_vec16,
    bool s_vec4
) {
    const int64_t y_work = y_vec16 ? ((int64_t)world_size * (n_y >> 4))
                                   : ((int64_t)world_size * n_y);
    const int64_t s_work = s_vec4 ? ((int64_t)world_size * (n_s >> 2))
                                  : ((int64_t)world_size * n_s);
    const int64_t total = y_work > s_work ? y_work : s_work;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        if (idx < y_work) {
            if (y_vec16) {
                const int64_t vecs_per_rank = n_y >> 4;
                const int r = (int)(idx / vecs_per_rank);
                const int64_t j = idx - (int64_t)r * vecs_per_rank;

                const uint4* src = reinterpret_cast<const uint4*>(
                    (const unsigned char*)reinterpret_cast<const void*>(y_ptrs[r]));
                uint4* dst = reinterpret_cast<uint4*>(y_out + (int64_t)r * n_y);
                dst[j] = src[j];
            } else {
                const int r = (int)(idx / n_y);
                const int64_t j = idx - (int64_t)r * n_y;
                const unsigned char* src =
                    (const unsigned char*)reinterpret_cast<const void*>(y_ptrs[r]);
                y_out[(int64_t)r * n_y + j] = src[j];
            }
        }

        if (idx < s_work) {
            if (s_vec4) {
                const int64_t vecs_per_rank = n_s >> 2;
                const int r = (int)(idx / vecs_per_rank);
                const int64_t j = idx - (int64_t)r * vecs_per_rank;

                const float4* src = reinterpret_cast<const float4*>(
                    (const float*)reinterpret_cast<const void*>(s_ptrs[r]));
                float4* dst = reinterpret_cast<float4*>(s_out + (int64_t)r * n_s);
                dst[j] = src[j];
            } else {
                const int r = (int)(idx / n_s);
                const int64_t j = idx - (int64_t)r * n_s;
                const float* src =
                    (const float*)reinterpret_cast<const void*>(s_ptrs[r]);
                s_out[(int64_t)r * n_s + j] = src[j];
            }
        }
    }
}

void quantize_fp8(
    torch::Tensor x,
    torch::Tensor y_raw,
    torch::Tensor s,
    int64_t n,
    int block_size,
    int dtype_enum
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(y_raw.is_cuda(), "y must be CUDA");
    TORCH_CHECK(s.is_cuda(), "s must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y_raw.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(s.is_contiguous(), "s must be contiguous");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(n % block_size == 0, "n must be divisible by block_size");

    const int64_t num_blocks = n / block_size;
    if (num_blocks == 0) {
        return;
    }

    int threads = 256;
    if (block_size <= 64) {
        threads = 64;
    } else if (block_size <= 128) {
        threads = 128;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    block_fp8_quant_kernel<<<(unsigned int)num_blocks, threads, threads * sizeof(float), stream>>>(
        x.data_ptr(),
        reinterpret_cast<unsigned char*>(y_raw.data_ptr()),
        s.data_ptr<float>(),
        num_blocks,
        block_size,
        dtype_enum
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_quantized_and_scales(
    torch::Tensor y_ptrs,
    torch::Tensor s_ptrs,
    torch::Tensor y_out,
    torch::Tensor s_out,
    int64_t n_y,
    int64_t n_s
) {
    TORCH_CHECK(y_ptrs.is_cuda() && s_ptrs.is_cuda(), "ptr tensors must be CUDA");
    TORCH_CHECK(y_out.is_cuda() && s_out.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(y_ptrs.dtype() == torch::kInt64, "y_ptrs must be int64");
    TORCH_CHECK(s_ptrs.dtype() == torch::kInt64, "s_ptrs must be int64");
    TORCH_CHECK(y_out.is_contiguous() && s_out.is_contiguous(), "outputs must be contiguous");

    const int world_size = (int)y_ptrs.size(0);
    if (world_size <= 0 || n_y == 0 || n_s == 0) {
        return;
    }

    const uintptr_t y_addr = (uintptr_t)y_out.data_ptr();
    const uintptr_t s_addr = (uintptr_t)s_out.data_ptr<float>();
    const bool y_vec16 = ((n_y & 15LL) == 0) && ((y_addr & 15ULL) == 0);
    const bool s_vec4 = ((n_s & 3LL) == 0) && ((s_addr & 15ULL) == 0);

    const int64_t y_work = y_vec16 ? ((int64_t)world_size * (n_y >> 4))
                                   : ((int64_t)world_size * n_y);
    const int64_t s_work = s_vec4 ? ((int64_t)world_size * (n_s >> 2))
                                  : ((int64_t)world_size * n_s);
    const int64_t total = y_work > s_work ? y_work : s_work;
    if (total == 0) {
        return;
    }

    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 131072) {
        blocks = 131072;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_quantized_and_scales_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(y_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(s_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<unsigned char*>(y_out.data_ptr()),
        s_out.data_ptr<float>(),
        world_size,
        n_y,
        n_s,
        y_vec16,
        s_vec4
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_fp8", &quantize_fp8, "BF16/FP32/FP16 block FP8 E4M3 quantization");
    m.def("gather_quantized_and_scales", &gather_quantized_and_scales,
          "UVA symmetric-memory all-gather for FP8 bytes and FP32 scales");
}
'''


_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("blocked_fp8_quantize_symm_uva_ext", CUDA_SRC)
    return _ext


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError("solution supports bfloat16, float32, and float16 inputs")


def _scale_shape(local_shape, block_size: int):
    return tuple(local_shape[:-1]) + (local_shape[-1] // block_size,)


def _cat0_shape(local_shape, world_size: int):
    return (local_shape[0] * world_size,) + tuple(local_shape[1:])


def _get_resources(local_tensor: torch.Tensor, block_size: int, n_y: int, n_s: int):
    world_size = dist.get_world_size()
    device = local_tensor.device
    key = (
        tuple(local_tensor.shape),
        local_tensor.dtype,
        int(device.index if device.index is not None else torch.cuda.current_device()),
        int(block_size),
        int(world_size),
    )

    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    y_sym = symm_mem.empty((n_y,), device=device, dtype=torch.uint8)
    s_sym = symm_mem.empty((n_s,), device=device, dtype=torch.float32)

    y_hdl = symm_mem.rendezvous(y_sym, dist.group.WORLD)
    s_hdl = symm_mem.rendezvous(s_sym, dist.group.WORLD)

    y_ptrs = torch.tensor(y_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    s_ptrs = torch.tensor(s_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "y_sym": y_sym,
        "s_sym": s_sym,
        "y_hdl": y_hdl,
        "s_hdl": s_hdl,
        "y_ptrs": y_ptrs,
        "s_ptrs": s_ptrs,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(local_tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert local_tensor.is_cuda, "Input tensor must be CUDA"
    assert local_tensor.is_contiguous(), "Input tensor must be contiguous"
    assert local_tensor.dim() >= 1, "Input tensor must have at least one dimension"
    assert local_tensor.size(-1) % block_size == 0, "Last dimension must be divisible by block_size"

    ext = _get_ext()
    dtype_enum = _dtype_enum(local_tensor.dtype)

    n_y = local_tensor.numel()
    n_s = n_y // block_size
    s_local_shape = _scale_shape(tuple(local_tensor.shape), block_size)

    if not dist.is_initialized() or dist.get_world_size() == 1:
        y_local = torch.empty(local_tensor.shape, device=local_tensor.device, dtype=torch.float8_e4m3fn)
        s_local = torch.empty(s_local_shape, device=local_tensor.device, dtype=torch.float32)
        ext.quantize_fp8(local_tensor, y_local, s_local, n_y, int(block_size), dtype_enum)
        return y_local, s_local

    world_size = dist.get_world_size()
    res = _get_resources(local_tensor, int(block_size), n_y, n_s)

    y_sym = res["y_sym"]
    s_sym = res["s_sym"]

    ext.quantize_fp8(local_tensor, y_sym, s_sym, n_y, int(block_size), dtype_enum)

    # Device-visible symmetric-memory synchronization; avoids NCCL all_gather.
    res["y_hdl"].barrier(channel=0)

    y_global_shape = _cat0_shape(tuple(local_tensor.shape), world_size)
    s_global_shape = _cat0_shape(s_local_shape, world_size)

    y_global_u8 = torch.empty(y_global_shape, device=local_tensor.device, dtype=torch.uint8)
    s_global = torch.empty(s_global_shape, device=local_tensor.device, dtype=torch.float32)

    ext.gather_quantized_and_scales(
        res["y_ptrs"],
        res["s_ptrs"],
        y_global_u8,
        s_global,
        n_y,
        n_s,
    )

    return y_global_u8.view(torch.float8_e4m3fn), s_global