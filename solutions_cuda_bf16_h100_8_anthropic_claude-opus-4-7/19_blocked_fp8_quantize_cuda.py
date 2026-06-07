import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define FP8_E4M3_MAX 448.0f

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_float<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__global__ void block_fp8_quant_kernel(
    const T* __restrict__ x,
    __nv_fp8_storage_t* __restrict__ y,
    float* __restrict__ s,
    int64_t num_blocks,
    int block_size
) {
    int64_t pid = blockIdx.x;
    if (pid >= num_blocks) return;

    int tid = threadIdx.x;
    int64_t base = pid * block_size;

    extern __shared__ float sdata[];

    // Pass 1: load and compute max abs
    float local_max = 0.0f;
    for (int i = tid; i < block_size; i += blockDim.x) {
        float v = to_float<T>(x[base + i]);
        float a = fabsf(v);
        if (a > local_max) local_max = a;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Block reduction
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float other = sdata[tid + off];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }
    float maxv = sdata[0];
    float scale = maxv / FP8_E4M3_MAX;
    float scale_safe = (scale == 0.0f) ? 1.0f : scale;

    if (tid == 0) {
        s[pid] = scale;
    }

    // Pass 2: quantize
    float inv = 1.0f / scale_safe;
    for (int i = tid; i < block_size; i += blockDim.x) {
        float v = to_float<T>(x[base + i]) * inv;
        // Convert to fp8 e4m3
        __nv_fp8_storage_t out = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        y[base + i] = out;
    }
}

void launch_block_fp8_quant(
    torch::Tensor x,
    torch::Tensor y,         // uint8 view of fp8 buffer
    torch::Tensor s,
    int64_t num_blocks,
    int block_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = block_size < 256 ? block_size : 256;
    // round up to power of 2 for reduction
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    size_t shm = threads * sizeof(float);

    if (x.scalar_type() == at::kBFloat16) {
        block_fp8_quant_kernel<__nv_bfloat16><<<num_blocks, threads, shm, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_fp8_storage_t*>(y.data_ptr<uint8_t>()),
            s.data_ptr<float>(),
            num_blocks, block_size);
    } else if (x.scalar_type() == at::kFloat) {
        block_fp8_quant_kernel<float><<<num_blocks, threads, shm, stream>>>(
            x.data_ptr<float>(),
            reinterpret_cast<__nv_fp8_storage_t*>(y.data_ptr<uint8_t>()),
            s.data_ptr<float>(),
            num_blocks, block_size);
    } else if (x.scalar_type() == at::kHalf) {
        block_fp8_quant_kernel<__half><<<num_blocks, threads, shm, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__nv_fp8_storage_t*>(y.data_ptr<uint8_t>()),
            s.data_ptr<float>(),
            num_blocks, block_size);
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Gather from peer symmetric buffers via UVA into contiguous output.
// y_ptrs[r] points to rank r's fp8 buffer (size = local_bytes per rank)
// Output layout: concatenate along dim 0 -> rank r's slice goes to offset r*local_bytes
__global__ void gather_uint8_kernel(
    const uint64_t* __restrict__ y_ptrs,
    uint8_t* __restrict__ y_global,
    int world_size,
    int64_t local_numel
) {
    int r = blockIdx.y;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(y_ptrs[r]);
    uint8_t* dst = y_global + (int64_t)r * local_numel;
    // Vectorized copy via uint4
    int64_t vec_n = local_numel / 16;
    const uint4* vsrc = reinterpret_cast<const uint4*>(src);
    uint4* vdst = reinterpret_cast<uint4*>(dst);
    for (int64_t i = idx; i < vec_n; i += stride) {
        vdst[i] = vsrc[i];
    }
    int64_t tail_start = vec_n * 16;
    for (int64_t i = tail_start + idx; i < local_numel; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void gather_float_kernel(
    const uint64_t* __restrict__ s_ptrs,
    float* __restrict__ s_global,
    int world_size,
    int64_t local_numel
) {
    int r = blockIdx.y;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const float* src = reinterpret_cast<const float*>(s_ptrs[r]);
    float* dst = s_global + (int64_t)r * local_numel;
    for (int64_t i = idx; i < local_numel; i += stride) {
        dst[i] = src[i];
    }
}

void launch_gather(
    torch::Tensor y_ptrs,    // int64 tensor [world_size] device
    torch::Tensor s_ptrs,    // int64 tensor [world_size] device
    torch::Tensor y_global,  // uint8
    torch::Tensor s_global,  // float
    int world_size,
    int64_t y_local_numel,
    int64_t s_local_numel
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    {
        int threads = 256;
        int64_t vec_n = y_local_numel / 16;
        int64_t work = vec_n > 0 ? vec_n : y_local_numel;
        int blocks_x = (int)std::min<int64_t>((work + threads - 1) / threads, 1024);
        if (blocks_x < 1) blocks_x = 1;
        dim3 grid(blocks_x, world_size, 1);
        gather_uint8_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(y_ptrs.data_ptr<int64_t>()),
            y_global.data_ptr<uint8_t>(),
            world_size,
            y_local_numel);
    }
    {
        int threads = 256;
        int blocks_x = (int)std::min<int64_t>((s_local_numel + threads - 1) / threads, 1024);
        if (blocks_x < 1) blocks_x = 1;
        dim3 grid(blocks_x, world_size, 1);
        gather_float_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(s_ptrs.data_ptr<int64_t>()),
            s_global.data_ptr<float>(),
            world_size,
            s_local_numel);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_block_fp8_quant", &launch_block_fp8_quant, "Block FP8 E4M3 quant");
    m.def("launch_gather", &launch_gather, "Gather from peers via UVA");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("block_fp8_quant_symm_ext", CUDA_SRC)
    return _ext


_cache = {}

def _get_resources(local_shape, dtype, block_size, device, world_size):
    key = (tuple(local_shape), dtype, block_size, device.index, world_size)
    if key in _cache:
        return _cache[key]

    last = local_shape[-1]
    leading = 1
    for d in local_shape[:-1]:
        leading *= d
    s_shape = (*local_shape[:-1], last // block_size)

    # Symmetric buffers (uint8 for fp8 storage, float32 for scales)
    y_buf = symm_mem.empty(local_shape, device=device, dtype=torch.uint8)
    s_buf = symm_mem.empty(s_shape, device=device, dtype=torch.float32)
    y_hdl = symm_mem.rendezvous(y_buf, dist.group.WORLD)
    s_hdl = symm_mem.rendezvous(s_buf, dist.group.WORLD)

    y_ptrs = torch.tensor(list(y_hdl.buffer_ptrs), device=device, dtype=torch.int64)
    s_ptrs = torch.tensor(list(s_hdl.buffer_ptrs), device=device, dtype=torch.int64)

    # Global output buffers (concat along dim 0)
    y_global_shape = (local_shape[0] * world_size, *local_shape[1:]) if len(local_shape) > 1 else (local_shape[0] * world_size,)
    s_global_shape = (s_shape[0] * world_size, *s_shape[1:]) if len(s_shape) > 1 else (s_shape[0] * world_size,)

    y_global = torch.empty(y_global_shape, device=device, dtype=torch.uint8)
    s_global = torch.empty(s_global_shape, device=device, dtype=torch.float32)

    res = {
        'y_buf': y_buf, 's_buf': s_buf,
        'y_hdl': y_hdl, 's_hdl': s_hdl,
        'y_ptrs': y_ptrs, 's_ptrs': s_ptrs,
        'y_global': y_global, 's_global': s_global,
        's_shape': s_shape,
        'y_local_numel': y_buf.numel(),
        's_local_numel': s_buf.numel(),
    }
    _cache[key] = res
    return res


@torch.no_grad()
def solution(local_tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert local_tensor.is_contiguous()
    assert local_tensor.size(-1) % block_size == 0

    ext = _get_ext()
    device = local_tensor.device

    if not dist.is_initialized():
        # Single GPU fallback
        y_local = torch.empty_like(local_tensor, dtype=torch.float8_e4m3fn)
        s_shape = (*local_tensor.size()[:-1], local_tensor.size(-1) // block_size)
        s_local = torch.empty(s_shape, device=device, dtype=torch.float32)
        num_blocks = local_tensor.numel() // block_size
        ext.launch_block_fp8_quant(local_tensor, y_local.view(torch.uint8), s_local, num_blocks, block_size)
        return y_local, s_local

    world_size = dist.get_world_size()
    res = _get_resources(local_tensor.shape, local_tensor.dtype, block_size, device, world_size)

    num_blocks = local_tensor.numel() // block_size

    # 1. Quantize directly into symmetric buffers
    ext.launch_block_fp8_quant(
        local_tensor,
        res['y_buf'],   # uint8 symm buffer
        res['s_buf'],
        num_blocks,
        block_size,
    )

    # 2. Device-side barrier to ensure all peers have produced their data
    res['y_hdl'].barrier(channel=0)
    res['s_hdl'].barrier(channel=1)

    # 3. Gather from peer UVA pointers into global output
    ext.launch_gather(
        res['y_ptrs'],
        res['s_ptrs'],
        res['y_global'],
        res['s_global'],
        world_size,
        res['y_local_numel'],
        res['s_local_numel'],
    )

    # Final barrier to ensure all reads are complete before next call mutates buffers
    res['y_hdl'].barrier(channel=2)

    y_global = res['y_global'].view(torch.float8_e4m3fn)
    return y_global, res['s_global']