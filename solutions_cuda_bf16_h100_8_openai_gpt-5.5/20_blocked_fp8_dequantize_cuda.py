import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

__device__ __forceinline__ float make_qnan() {
    return __uint_as_float(0x7fffffffU);
}

__device__ __forceinline__ float make_inf(bool neg) {
    return __uint_as_float((neg ? 0xff800000U : 0x7f800000U));
}

// PyTorch torch.float8_e4m3fn: sign:1 exp:4 mant:3, bias=7.
// No infinities. 0x7f/0xff are NaN; max finite is +/-448.
__device__ __forceinline__ float fp8_e4m3fn_to_f32(uint8_t x) {
    const uint32_t sign = (uint32_t)(x >> 7);
    const uint32_t ax = (uint32_t)(x & 0x7f);
    const uint32_t e = (uint32_t)((x >> 3) & 0x0f);
    const uint32_t m = (uint32_t)(x & 0x07);

    if (ax == 0) {
        return sign ? -0.0f : 0.0f;
    }
    if (e == 0) {
        // subnormal: (-1)^s * mantissa/8 * 2^(1-bias) = m * 2^-9
        float v = (float)m * 0.001953125f;
        return sign ? -v : v;
    }
    if (e == 15 && m == 7) {
        return make_qnan();
    }

    const uint32_t exp_f32 = e - 7 + 127;
    const uint32_t bits = (sign << 31) | (exp_f32 << 23) | (m << 20);
    return __uint_as_float(bits);
}

// PyTorch torch.float8_e5m2: sign:1 exp:5 mant:2, bias=15.
__device__ __forceinline__ float fp8_e5m2_to_f32(uint8_t x) {
    const uint32_t sign = (uint32_t)(x >> 7);
    const uint32_t ax = (uint32_t)(x & 0x7f);
    const uint32_t e = (uint32_t)((x >> 2) & 0x1f);
    const uint32_t m = (uint32_t)(x & 0x03);

    if (ax == 0) {
        return sign ? -0.0f : 0.0f;
    }
    if (e == 0) {
        // subnormal: m/4 * 2^(1-15) = m * 2^-16
        float v = (float)m * 0.0000152587890625f;
        return sign ? -v : v;
    }
    if (e == 31) {
        if (m == 0) return make_inf(sign != 0);
        return make_qnan();
    }

    const uint32_t exp_f32 = e - 15 + 127;
    const uint32_t bits = (sign << 31) | (exp_f32 << 23) | (m << 21);
    return __uint_as_float(bits);
}

// PyTorch torch.float8_e4m3fnuz: sign:1 exp:4 mant:3, bias=8, unsigned zero.
// 0x80 is NaN.
__device__ __forceinline__ float fp8_e4m3fnuz_to_f32(uint8_t x) {
    if (x == 0) return 0.0f;
    if (x == 0x80) return make_qnan();

    const uint32_t sign = (uint32_t)(x >> 7);
    const uint32_t e = (uint32_t)((x >> 3) & 0x0f);
    const uint32_t m = (uint32_t)(x & 0x07);

    if (e == 0) {
        // m/8 * 2^(1-8) = m * 2^-10
        float v = (float)m * 0.0009765625f;
        return sign ? -v : v;
    }

    const uint32_t exp_f32 = e - 8 + 127;
    const uint32_t bits = (sign << 31) | (exp_f32 << 23) | (m << 20);
    return __uint_as_float(bits);
}

// PyTorch torch.float8_e5m2fnuz: sign:1 exp:5 mant:2, bias=16, unsigned zero.
// 0x80 is NaN.
__device__ __forceinline__ float fp8_e5m2fnuz_to_f32(uint8_t x) {
    if (x == 0) return 0.0f;
    if (x == 0x80) return make_qnan();

    const uint32_t sign = (uint32_t)(x >> 7);
    const uint32_t e = (uint32_t)((x >> 2) & 0x1f);
    const uint32_t m = (uint32_t)(x & 0x03);

    if (e == 0) {
        // m/4 * 2^(1-16) = m * 2^-17
        float v = (float)m * 0.00000762939453125f;
        return sign ? -v : v;
    }

    const uint32_t exp_f32 = e - 16 + 127;
    const uint32_t bits = (sign << 31) | (exp_f32 << 23) | (m << 21);
    return __uint_as_float(bits);
}

template<int FP8_KIND>
__device__ __forceinline__ float fp8_to_f32(uint8_t x) {
    if constexpr (FP8_KIND == 0) {
        return fp8_e4m3fn_to_f32(x);
    } else if constexpr (FP8_KIND == 1) {
        return fp8_e5m2_to_f32(x);
    } else if constexpr (FP8_KIND == 2) {
        return fp8_e4m3fnuz_to_f32(x);
    } else {
        return fp8_e5m2fnuz_to_f32(x);
    }
}

void publish_inputs(
    torch::Tensor local_y,
    torch::Tensor local_s,
    torch::Tensor y_symm_u8,
    torch::Tensor s_symm_f32
) {
    TORCH_CHECK(local_y.is_cuda(), "local_y must be CUDA");
    TORCH_CHECK(local_s.is_cuda(), "local_s must be CUDA");
    TORCH_CHECK(y_symm_u8.is_cuda(), "y_symm_u8 must be CUDA");
    TORCH_CHECK(s_symm_f32.is_cuda(), "s_symm_f32 must be CUDA");
    TORCH_CHECK(local_y.is_contiguous(), "local_y must be contiguous");
    TORCH_CHECK(local_s.is_contiguous(), "local_s must be contiguous");
    TORCH_CHECK(y_symm_u8.is_contiguous(), "y_symm_u8 must be contiguous");
    TORCH_CHECK(s_symm_f32.is_contiguous(), "s_symm_f32 must be contiguous");
    TORCH_CHECK(local_y.element_size() == 1, "local_y must be an 8-bit FP8 tensor");
    TORCH_CHECK(local_s.dtype() == torch::kFloat32, "local_s must be float32");
    TORCH_CHECK(y_symm_u8.dtype() == torch::kUInt8, "y_symm_u8 must be uint8");
    TORCH_CHECK(s_symm_f32.dtype() == torch::kFloat32, "s_symm_f32 must be float32");
    TORCH_CHECK(y_symm_u8.numel() == local_y.numel(), "bad y symmetric buffer size");
    TORCH_CHECK(s_symm_f32.numel() == local_s.numel(), "bad scale symmetric buffer size");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const size_t y_bytes = (size_t)local_y.numel();
    const size_t s_bytes = (size_t)local_s.numel() * sizeof(float);

    if (y_bytes) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            y_symm_u8.data_ptr<uint8_t>(),
            local_y.data_ptr(),
            y_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
    }

    if (s_bytes) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            s_symm_f32.data_ptr<float>(),
            local_s.data_ptr<float>(),
            s_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
    }
}

template<int FP8_KIND>
__global__ void dequant_alltoall_from_symm_kernel(
    const unsigned long long* __restrict__ y_ptrs,
    const unsigned long long* __restrict__ s_ptrs,
    float* __restrict__ out,
    int world_size,
    int rank,
    int64_t chunk_numel,
    int64_t blocks_per_chunk,
    int block_size
) {
    const int64_t global_block = (int64_t)blockIdx.x;
    const int src_rank = (int)(global_block / blocks_per_chunk);
    const int64_t block_in_chunk = global_block - (int64_t)src_rank * blocks_per_chunk;

    if (src_rank >= world_size) return;

    __shared__ const uint8_t* y_base;
    __shared__ const float* s_base;
    __shared__ float scale;

    if (threadIdx.x == 0) {
        y_base = reinterpret_cast<const uint8_t*>((uintptr_t)y_ptrs[src_rank]);
        s_base = reinterpret_cast<const float*>((uintptr_t)s_ptrs[src_rank]);
        scale = s_base[(int64_t)rank * blocks_per_chunk + block_in_chunk];
    }

    __syncthreads();

    const int64_t src_chunk_offset =
        (int64_t)rank * chunk_numel + block_in_chunk * (int64_t)block_size;
    const int64_t dst_chunk_offset =
        (int64_t)src_rank * chunk_numel + block_in_chunk * (int64_t)block_size;

    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
        const uint8_t q = y_base[src_chunk_offset + i];
        out[dst_chunk_offset + i] = fp8_to_f32<FP8_KIND>(q) * scale;
    }
}

void launch_dequant_alltoall(
    torch::Tensor y_ptrs_tensor,
    torch::Tensor s_ptrs_tensor,
    torch::Tensor out,
    int64_t chunk_numel,
    int64_t blocks_per_chunk,
    int block_size,
    int rank,
    int fp8_kind
) {
    TORCH_CHECK(y_ptrs_tensor.is_cuda(), "y_ptrs_tensor must be CUDA");
    TORCH_CHECK(s_ptrs_tensor.is_cuda(), "s_ptrs_tensor must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(y_ptrs_tensor.dtype() == torch::kInt64, "y_ptrs_tensor must be int64");
    TORCH_CHECK(s_ptrs_tensor.dtype() == torch::kInt64, "s_ptrs_tensor must be int64");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(block_size > 0 && block_size <= 4096, "unsupported block_size");
    TORCH_CHECK(blocks_per_chunk >= 0, "bad blocks_per_chunk");

    const int world_size = (int)y_ptrs_tensor.numel();
    if (chunk_numel == 0 || blocks_per_chunk == 0 || world_size == 0) {
        return;
    }

    int threads = 1;
    while (threads < block_size && threads < 256) {
        threads <<= 1;
    }

    const int64_t total_blocks_i64 = (int64_t)world_size * blocks_per_chunk;
    TORCH_CHECK(total_blocks_i64 <= 2147483647LL, "too many dequant blocks for grid.x");

    dim3 grid((unsigned int)total_blocks_i64);
    dim3 block((unsigned int)threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const unsigned long long* y_ptrs =
        reinterpret_cast<const unsigned long long*>(y_ptrs_tensor.data_ptr<int64_t>());
    const unsigned long long* s_ptrs =
        reinterpret_cast<const unsigned long long*>(s_ptrs_tensor.data_ptr<int64_t>());

    if (fp8_kind == 0) {
        dequant_alltoall_from_symm_kernel<0><<<grid, block, 0, stream>>>(
            y_ptrs, s_ptrs, out.data_ptr<float>(),
            world_size, rank, chunk_numel, blocks_per_chunk, block_size);
    } else if (fp8_kind == 1) {
        dequant_alltoall_from_symm_kernel<1><<<grid, block, 0, stream>>>(
            y_ptrs, s_ptrs, out.data_ptr<float>(),
            world_size, rank, chunk_numel, blocks_per_chunk, block_size);
    } else if (fp8_kind == 2) {
        dequant_alltoall_from_symm_kernel<2><<<grid, block, 0, stream>>>(
            y_ptrs, s_ptrs, out.data_ptr<float>(),
            world_size, rank, chunk_numel, blocks_per_chunk, block_size);
    } else {
        dequant_alltoall_from_symm_kernel<3><<<grid, block, 0, stream>>>(
            y_ptrs, s_ptrs, out.data_ptr<float>(),
            world_size, rank, chunk_numel, blocks_per_chunk, block_size);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("publish_inputs", &publish_inputs,
          "Copy local FP8 payload/scales into symmetric buffers");
    m.def("launch_dequant_alltoall", &launch_dequant_alltoall,
          "Fused UVA all-to-all read + FP8 dequantization");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "blocked_fp8_dequant_alltoall_symm_uva_ext",
            CUDA_SRC,
        )
    return _ext


_resource_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return 0
    if hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2:
        return 1
    if hasattr(torch, "float8_e4m3fnuz") and dtype == torch.float8_e4m3fnuz:
        return 2
    if hasattr(torch, "float8_e5m2fnuz") and dtype == torch.float8_e5m2fnuz:
        return 3
    raise TypeError(f"local_y must be a torch FP8 dtype, got {dtype}")


def _get_resources(
    y_numel: int,
    s_numel: int,
    out_shape: tuple,
    y_dtype: torch.dtype,
    device: torch.device,
    world_size: int,
):
    key = (y_numel, s_numel, out_shape, y_dtype, device, world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    # Use uint8 symmetric storage for FP8 payload so rendezvous support does not
    # depend on float8 allocator support.
    y_symm = symm_mem.empty((y_numel,), device=device, dtype=torch.uint8)
    y_hdl = symm_mem.rendezvous(y_symm, dist.group.WORLD)

    s_symm = symm_mem.empty((s_numel,), device=device, dtype=torch.float32)
    s_hdl = symm_mem.rendezvous(s_symm, dist.group.WORLD)

    out = torch.empty(out_shape, device=device, dtype=torch.float32)

    y_ptrs = torch.tensor(y_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    s_ptrs = torch.tensor(s_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "y_symm": y_symm,
        "s_symm": s_symm,
        "y_hdl": y_hdl,
        "s_hdl": s_hdl,
        "out": out,
        "y_ptrs": y_ptrs,
        "s_ptrs": s_ptrs,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    local_y: torch.Tensor,
    local_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert local_y.is_cuda, "local_y must be CUDA"
    assert local_s.is_cuda, "local_s must be CUDA"
    assert local_y.is_contiguous(), "Input tensor local_y must be contiguous"
    assert local_s.is_contiguous(), "Scale tensor local_s must be contiguous"
    assert local_s.dtype == torch.float32, "Scale tensor local_s must be float32"
    assert local_y.dim() >= 1 and local_y.shape[0] == world_size, (
        f"local_y first dimension must equal world_size ({world_size}), got {local_y.shape[0]}"
    )

    fp8_kind = _dtype_enum(local_y.dtype)

    chunk_shape = tuple(local_y.shape[1:])
    chunk_numel = local_y.numel() // world_size
    num_elements = local_y.numel()

    assert block_size > 0, "block_size must be positive"
    assert chunk_numel % block_size == 0, (
        f"Chunk size {chunk_numel} must be divisible by block_size ({block_size})"
    )

    blocks_per_chunk = chunk_numel // block_size
    expected_s_numel = world_size * blocks_per_chunk
    assert local_s.numel() == expected_s_numel, (
        f"local_s.numel() must be {expected_s_numel}, got {local_s.numel()}"
    )

    out_shape = (world_size, *chunk_shape)

    if num_elements == 0:
        return torch.empty(out_shape, device=local_y.device, dtype=torch.float32)

    ext = _get_ext()
    res = _get_resources(
        y_numel=num_elements,
        s_numel=local_s.numel(),
        out_shape=out_shape,
        y_dtype=local_y.dtype,
        device=local_y.device,
        world_size=world_size,
    )

    # Publish compressed payload and scales to symmetric memory.
    ext.publish_inputs(
        local_y,
        local_s,
        res["y_symm"],
        res["s_symm"],
    )

    # Symmetric-memory rank barrier; no NCCL all_to_all/all_gather/all_reduce.
    # It orders publication before remote UVA reads in the fused kernel.
    res["y_hdl"].barrier(channel=0)

    ext.launch_dequant_alltoall(
        res["y_ptrs"],
        res["s_ptrs"],
        res["out"],
        int(chunk_numel),
        int(blocks_per_chunk),
        int(block_size),
        int(rank),
        int(fp8_kind),
    )

    return res["out"]