# Strategy:
# - Use torch.distributed._symmetric_memory rendezvous once per shape/dtype and exchange UVA pointers.
# - Copy each rank input into a symmetric buffer, then perform BF16 all-reduce on-device.
# - Fast path uses Hopper/NVSwitch multimem.ld_reduce + multimem.st on the multicast pointer.
# - Non-BF16 or non-128b-aligned BF16 falls back to a custom UVA peer-load reduction kernel.
# - No NCCL all_reduce/all_gather is used on the hot path.

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
#include <type_traits>

// -----------------------------------------------------------------------------
// Utility: async device-to-device copy into symmetric memory.
// -----------------------------------------------------------------------------

void copy_bytes(torch::Tensor src, torch::Tensor dst, int64_t nbytes) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA tensors");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), (size_t)nbytes,
                    cudaMemcpyDeviceToDevice, stream);
}

// -----------------------------------------------------------------------------
// Device-side signal-pad blockwise barriers for multimem path.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1u);
}

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
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1u);
}

__device__ __forceinline__ void blockwise_barrier_relaxed(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    const int t = threadIdx.x;
    if (t >= world_size) return;

    const uint64_t local_base = signal_pad_ptrs[rank];
    const uint64_t remote_base = signal_pad_ptrs[t];

    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)t);

    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

__device__ __forceinline__ void blockwise_barrier_acq_rel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    const int t = threadIdx.x;
    if (t >= world_size) return;

    const uint64_t local_base = signal_pad_ptrs[rank];
    const uint64_t remote_base = signal_pad_ptrs[t];

    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)t);

    send_signal_release(send_addr);
    wait_signal_acquire(wait_addr);
}

// -----------------------------------------------------------------------------
// Hopper NVSwitch multimem BF16 all-reduce.
//
// Each thread reduces one 128-bit slot = 8 BF16 values packed as four bf16x2
// lanes. Work is partitioned by rank; multimem.st broadcasts each reduced slot
// back to every rank's symmetric buffer.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& x,
    uint32_t& y,
    uint32_t& z,
    uint32_t& w
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
        : "l"(addr)
        : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(
    uint64_t* addr,
    uint32_t x,
    uint32_t y,
    uint32_t z,
    uint32_t w
) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w)
        : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t num_128b_slots,
    int world_size,
    int rank,
    int block_stride
) {
    const uint64_t block_id = (uint64_t)blockIdx.x;

    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t slots_per_rank =
        (num_128b_slots + (int64_t)world_size - 1) / (int64_t)world_size;

    const int tid = threadIdx.x;
    const int nblocks = gridDim.x;

    for (int64_t local_slot = (int64_t)block_id * (int64_t)block_stride + tid;
         local_slot < slots_per_rank;
         local_slot += (int64_t)nblocks * (int64_t)block_stride) {
        const int64_t global_slot = (int64_t)rank * slots_per_rank + local_slot;
        if (global_slot >= num_128b_slots) continue;

        uint64_t* mm_ptr = reinterpret_cast<uint64_t*>(multicast_base) + global_slot * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(mm_ptr, x, y, z, w);
        multimem_st_bf16x4(mm_ptr, x, y, z, w);
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// -----------------------------------------------------------------------------
// UVA peer-pointer fallback kernels.
// -----------------------------------------------------------------------------

__global__ void allreduce_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* p =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[r]);
                sum += __bfloat162float(p[idx]);
            }
        }
        out[idx] = __float2bfloat16(sum);
    }
}

__global__ void allreduce_f16_kernel(
    const long long* __restrict__ ptrs,
    __half* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __half* p = reinterpret_cast<const __half*>((uintptr_t)ptrs[r]);
                sum += __half2float(p[idx]);
            }
        }
        out[idx] = __float2half(sum);
    }
}

__global__ void allreduce_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const float* p = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
                sum += p[idx];
            }
        }
        out[idx] = sum;
    }
}

__global__ void allreduce_f64_kernel(
    const long long* __restrict__ ptrs,
    double* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        double sum = 0.0;
        for (int r = 0; r < world_size; ++r) {
            const double* p = reinterpret_cast<const double*>((uintptr_t)ptrs[r]);
            sum += p[idx];
        }
        out[idx] = sum;
    }
}

template <typename T, typename AccT>
__global__ void allreduce_int_kernel(
    const long long* __restrict__ ptrs,
    T* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        AccT sum = 0;
        for (int r = 0; r < world_size; ++r) {
            const T* p = reinterpret_cast<const T*>((uintptr_t)ptrs[r]);
            sum += (AccT)p[idx];
        }
        out[idx] = (T)sum;
    }
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t num_128b_slots,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride
) {
    TORCH_CHECK(signal_pad_ptrs_tensor.is_cuda(), "signal_pad_ptrs_tensor must be CUDA");
    const uint64_t* signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr,
        signal,
        num_128b_slots,
        world_size,
        rank,
        block_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_allreduce_uva(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n,
    int dtype_enum
) {
    TORCH_CHECK(ptrs_tensor.is_cuda(), "ptrs_tensor must be CUDA");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out must be contiguous CUDA tensor");

    const long long* ptrs =
        reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>());
    const int world_size = (int)ptrs_tensor.size(0);

    if (n == 0) return;

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            world_size, n);
    } else if (dtype_enum == 1) {
        allreduce_f32_kernel<<<blocks, threads, 0, stream>>>(
            ptrs, out.data_ptr<float>(), world_size, n);
    } else if (dtype_enum == 2) {
        allreduce_f16_kernel<<<blocks, threads, 0, stream>>>(
            ptrs, reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            world_size, n);
    } else if (dtype_enum == 3) {
        allreduce_f64_kernel<<<blocks, threads, 0, stream>>>(
            ptrs, out.data_ptr<double>(), world_size, n);
    } else if (dtype_enum == 4) {
        allreduce_int_kernel<int32_t, int64_t><<<blocks, threads, 0, stream>>>(
            ptrs, out.data_ptr<int32_t>(), world_size, n);
    } else if (dtype_enum == 5) {
        allreduce_int_kernel<int64_t, int64_t><<<blocks, threads, 0, stream>>>(
            ptrs, out.data_ptr<int64_t>(), world_size, n);
    } else if (dtype_enum == 6) {
        allreduce_int_kernel<int16_t, int32_t><<<blocks, threads, 0, stream>>>(
            ptrs, out.data_ptr<int16_t>(), world_size, n);
    } else if (dtype_enum == 7) {
        allreduce_int_kernel<int8_t, int32_t><<<blocks, threads, 0, stream>>>(
            ptrs, out.data_ptr<int8_t>(), world_size, n);
    } else if (dtype_enum == 8) {
        allreduce_int_kernel<uint8_t, uint32_t><<<blocks, threads, 0, stream>>>(
            ptrs, out.data_ptr<uint8_t>(), world_size, n);
    } else {
        TORCH_CHECK(false, "unsupported dtype_enum");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_bytes", &copy_bytes, "Async D2D byte copy");
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16,
          "Hopper/NVSwitch multimem BF16 all-reduce");
    m.def("launch_allreduce_uva", &launch_allreduce_uva,
          "UVA peer-pointer all-reduce fallback");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("allreduce_bf16_h100_symm_mem_ext", CUDA_SRC)
    return _ext


# Multimem tuning: one 128-bit slot per thread iteration.
MAX_NUM_BLOCKS = 4
MAX_BLOCK_SIZE = 1024
BF16_PER_128B = 8


def _multimem_launch_config(numel: int, world_size: int):
    slots = numel // BF16_PER_128B
    slots_per_rank = (slots + world_size - 1) // world_size

    if slots_per_rank <= MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < max(1, slots_per_rank):
            block_size <<= 1
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min(MAX_NUM_BLOCKS, (slots_per_rank + block_size - 1) // block_size)

    return num_blocks, block_size, block_size


_resource_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.float32:
        return 1
    if dtype is torch.float16:
        return 2
    if dtype is torch.float64:
        return 3
    if dtype is torch.int32:
        return 4
    if dtype is torch.int64:
        return 5
    if dtype is torch.int16:
        return 6
    if dtype is torch.int8:
        return 7
    if dtype is torch.uint8:
        return 8
    raise TypeError(f"unsupported dtype for custom all-reduce: {dtype}")


def _get_resources(tensor: torch.Tensor):
    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    device = tensor.device
    world_size = dist.get_world_size()
    key = (shape, dtype, device.index, world_size)

    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    out = torch.empty(shape, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (buf, hdl, out, ptrs_tensor)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda, "input must be a CUDA tensor"
    assert tensor.is_contiguous(), "input must be contiguous"

    n = tensor.numel()
    if n == 0:
        return torch.empty_like(tensor)

    ext = _get_ext()
    buf, hdl, out, ptrs_tensor = _get_resources(tensor)

    # Place this rank's payload in symmetric memory; all following communication is device-side.
    ext.copy_bytes(tensor, buf, tensor.nbytes)

    dtype = tensor.dtype
    world_size = hdl.world_size
    rank = hdl.rank

    # Hopper/NVSwitch BF16 fast path. Requires exact 128-bit slot alignment.
    if dtype is torch.bfloat16 and (n % BF16_PER_128B) == 0:
        # Make producer copy visible to peers before kernels enter the device-side barrier.
        hdl.barrier(channel=0)

        num_blocks, block_size, block_stride = _multimem_launch_config(n, world_size)
        ext.launch_multimem_allreduce_bf16(
            int(hdl.multicast_ptr),
            hdl.signal_pad_ptrs_dev,
            n // BF16_PER_128B,
            world_size,
            rank,
            num_blocks,
            block_size,
            block_stride,
        )

        # The symmetric buffer now contains the full reduced tensor on every rank.
        return buf.reshape_as(tensor)

    # Generic UVA fallback, still avoiding NCCL collectives.
    hdl.barrier(channel=0)
    ext.launch_allreduce_uva(ptrs_tensor, out, n, _dtype_enum(dtype))
    return out.reshape_as(tensor)