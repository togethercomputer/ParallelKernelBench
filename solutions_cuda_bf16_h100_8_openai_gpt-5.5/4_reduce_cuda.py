import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>


// -----------------------------------------------------------------------------
// Symmetric-buffer staging copy
// -----------------------------------------------------------------------------

template <typename T>
__global__ void copy_kernel(const T* __restrict__ src, T* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        dst[i] = src[i];
    }
}


// -----------------------------------------------------------------------------
// Hopper/NVSwitch multicast BF16 reduce: dst only, no broadcast.
// Each 16B lane is 8 BF16 values represented as v4.bf16x2.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr)
        : "memory");
}

__global__ void multimem_reduce_bf16_kernel(
    uint64_t multicast_base,
    __nv_bfloat16* __restrict__ out,
    int64_t chunks_16b
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    uint4* out4 = reinterpret_cast<uint4*>(out);
    uint64_t* base = reinterpret_cast<uint64_t*>(multicast_base);

    for (; idx < chunks_16b; idx += stride) {
        uint32_t x, y, z, w;
        const uint64_t* mptr = base + idx * 2;  // 16B = two uint64 slots
        multimem_ld_reduce_bf16x4(mptr, x, y, z, w);
        out4[idx] = make_uint4(x, y, z, w);
    }
}


// -----------------------------------------------------------------------------
// UVA peer-pointer dst-only fallback reductions.
// -----------------------------------------------------------------------------

__global__ void reduce_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t n,
    int64_t start
) {
    int64_t i = start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src =
                reinterpret_cast<const __nv_bfloat16*>(ptrs[r]);
            sum += __bfloat162float(src[i]);
        }
        out[i] = __float2bfloat16(sum);
    }
}

__global__ void reduce_f16_kernel(
    const long long* __restrict__ ptrs,
    __half* __restrict__ out,
    int world_size,
    int64_t n,
    int64_t start
) {
    int64_t i = start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const __half* src = reinterpret_cast<const __half*>(ptrs[r]);
            sum += __half2float(src[i]);
        }
        out[i] = __float2half(sum);
    }
}

__global__ void reduce_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size,
    int64_t n,
    int64_t start
) {
    int64_t i = start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const float* src = reinterpret_cast<const float*>(ptrs[r]);
            sum += src[i];
        }
        out[i] = sum;
    }
}

__global__ void reduce_f64_kernel(
    const long long* __restrict__ ptrs,
    double* __restrict__ out,
    int world_size,
    int64_t n,
    int64_t start
) {
    int64_t i = start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        double sum = 0.0;
        for (int r = 0; r < world_size; ++r) {
            const double* src = reinterpret_cast<const double*>(ptrs[r]);
            sum += src[i];
        }
        out[i] = sum;
    }
}

template <typename T, typename ACC>
__global__ void reduce_int_kernel(
    const long long* __restrict__ ptrs,
    T* __restrict__ out,
    int world_size,
    int64_t n,
    int64_t start
) {
    int64_t i = start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        ACC sum = 0;
        for (int r = 0; r < world_size; ++r) {
            const T* src = reinterpret_cast<const T*>(ptrs[r]);
            sum += (ACC)src[i];
        }
        out[i] = (T)sum;
    }
}


// dtype enum:
// 0 bf16, 1 f32, 2 f16, 3 f64, 4 i64, 5 i32, 6 i16, 7 i8, 8 u8

void launch_copy(torch::Tensor src, torch::Tensor dst, int64_t n, int dtype_enum) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "copy tensors must be CUDA");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "copy tensors must be contiguous");

    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    switch (dtype_enum) {
        case 0:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
                n);
            break;
        case 1:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                src.data_ptr<float>(), dst.data_ptr<float>(), n);
            break;
        case 2:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
                n);
            break;
        case 3:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                src.data_ptr<double>(), dst.data_ptr<double>(), n);
            break;
        case 4:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(), n);
            break;
        case 5:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(), n);
            break;
        case 6:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                src.data_ptr<int16_t>(), dst.data_ptr<int16_t>(), n);
            break;
        case 7:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                src.data_ptr<int8_t>(), dst.data_ptr<int8_t>(), n);
            break;
        case 8:
            copy_kernel<<<blocks, threads, 0, stream>>>(
                src.data_ptr<uint8_t>(), dst.data_ptr<uint8_t>(), n);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype enum");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_multimem_reduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor out,
    int64_t chunks_16b
) {
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be BF16");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    if (chunks_16b <= 0) {
        return;
    }

    const int threads = 256;
    int blocks = (int)((chunks_16b + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_reduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        multicast_ptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        chunks_16b);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n,
    int64_t start,
    int dtype_enum
) {
    TORCH_CHECK(ptrs_tensor.is_cuda(), "ptrs_tensor must be CUDA");
    TORCH_CHECK(ptrs_tensor.dtype() == torch::kInt64, "ptrs_tensor must be int64");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    if (start >= n) {
        return;
    }

    int world_size = (int)ptrs_tensor.size(0);
    const long long* d_ptrs =
        reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>());

    int64_t work = n - start;
    const int threads = 256;
    int blocks = (int)((work + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    switch (dtype_enum) {
        case 0:
            reduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
                d_ptrs,
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                world_size,
                n,
                start);
            break;
        case 1:
            reduce_f32_kernel<<<blocks, threads, 0, stream>>>(
                d_ptrs, out.data_ptr<float>(), world_size, n, start);
            break;
        case 2:
            reduce_f16_kernel<<<blocks, threads, 0, stream>>>(
                d_ptrs,
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                world_size,
                n,
                start);
            break;
        case 3:
            reduce_f64_kernel<<<blocks, threads, 0, stream>>>(
                d_ptrs, out.data_ptr<double>(), world_size, n, start);
            break;
        case 4:
            reduce_int_kernel<int64_t, long long><<<blocks, threads, 0, stream>>>(
                d_ptrs, out.data_ptr<int64_t>(), world_size, n, start);
            break;
        case 5:
            reduce_int_kernel<int32_t, long long><<<blocks, threads, 0, stream>>>(
                d_ptrs, out.data_ptr<int32_t>(), world_size, n, start);
            break;
        case 6:
            reduce_int_kernel<int16_t, int><<<blocks, threads, 0, stream>>>(
                d_ptrs, out.data_ptr<int16_t>(), world_size, n, start);
            break;
        case 7:
            reduce_int_kernel<int8_t, int><<<blocks, threads, 0, stream>>>(
                d_ptrs, out.data_ptr<int8_t>(), world_size, n, start);
            break;
        case 8:
            reduce_int_kernel<uint8_t, unsigned int><<<blocks, threads, 0, stream>>>(
                d_ptrs, out.data_ptr<uint8_t>(), world_size, n, start);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype enum");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_copy", &launch_copy, "stage tensor into symmetric memory");
    m.def("launch_multimem_reduce_bf16", &launch_multimem_reduce_bf16,
          "dst-only BF16 reduce using Hopper multimem.ld_reduce");
    m.def("launch_reduce", &launch_reduce,
          "dst-only UVA peer-pointer reduce fallback");
}
'''


_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("reduce_symm_mem_multimem_bf16_ext", CUDA_SRC)
    return _ext


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.float32:
        return 1
    if dtype is torch.float16:
        return 2
    if dtype is torch.float64:
        return 3
    if dtype is torch.int64:
        return 4
    if dtype is torch.int32:
        return 5
    if dtype is torch.int16:
        return 6
    if dtype is torch.int8:
        return 7
    if dtype is torch.uint8:
        return 8
    raise TypeError(f"unsupported dtype for custom reduce: {dtype}")


def _get_resources(shape, dtype, device, world_size):
    key = (tuple(shape), dtype, device, world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty(tuple(shape), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = torch.empty(tuple(shape), device=device, dtype=dtype)

    ptrs = torch.tensor(
        [int(p) for p in hdl.buffer_ptrs],
        device=device,
        dtype=torch.int64,
    )

    cached = (buf, hdl, out, ptrs)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda, "input must be a CUDA tensor"
    assert tensor.is_contiguous(), "input must be contiguous"

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert 0 <= dst < world_size, "invalid dst rank"

    ext = _get_ext()
    dtype_enum = _dtype_enum(tensor.dtype)
    n = tensor.numel()

    buf, hdl, out, ptrs = _get_resources(
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        world_size,
    )

    # Stage this rank's input into symmetric memory on the current stream.
    ext.launch_copy(tensor, buf, n, dtype_enum)

    # Make all staged symmetric writes visible before dst reads peers.
    hdl.barrier(channel=0)

    if rank == dst:
        if tensor.dtype is torch.bfloat16:
            # Fast path: 8 BF16 values per 16B multimem reduction.
            chunks_16b = n // 8
            if chunks_16b > 0:
                ext.launch_multimem_reduce_bf16(
                    int(hdl.multicast_ptr),
                    out,
                    chunks_16b,
                )

            # Exact-size tail, if any, via direct UVA peer loads.
            tail_start = chunks_16b * 8
            if tail_start < n:
                ext.launch_reduce(ptrs, out, n, tail_start, dtype_enum)
        else:
            ext.launch_reduce(ptrs, out, n, 0, dtype_enum)

    # Prevent non-dst ranks from reusing symmetric buffers before dst finishes.
    hdl.barrier(channel=1)

    if rank == dst:
        return out.reshape_as(tensor)
    return tensor