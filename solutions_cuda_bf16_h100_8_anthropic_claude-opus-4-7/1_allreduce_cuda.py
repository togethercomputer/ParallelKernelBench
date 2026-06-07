"""
All-reduce (SUM) using torch symmetric memory + NVSwitch multimem PTX for BF16.
Falls back to peer-pointer CUDA reduction for other dtypes.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void blockwise_barrier_relaxed(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id, int rank, int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

__device__ void blockwise_barrier_acq_rel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id, int rank, int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr) : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w
) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank,
    int block_stride
) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank =
        (numel_128 + (int64_t)world_size - 1) / (int64_t)world_size;

    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = (int64_t)block_id * (int64_t)block_stride;
         block_start < numel_per_rank;
         block_start += (int64_t)num_programs * (int64_t)block_stride)
    {
        const int64_t offsets = block_start + (int64_t)tid;
        if (offsets >= numel_per_rank) continue;
        const int64_t idx = (int64_t)rank * numel_per_rank + offsets;
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

__global__ void allreduce_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

__global__ void allreduce_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const float* src = (const float*)ptrs[r];
            sum += src[idx];
        }
        out[idx] = sum;
    }
}

__global__ void allreduce_f16_kernel(
    const long long* __restrict__ ptrs,
    __half* __restrict__ out,
    int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __half* src = (const __half*)ptrs[r];
            sum += __half2float(src[idx]);
        }
        out[idx] = __float2half(sum);
    }
}

__global__ void allreduce_i32_kernel(
    const long long* __restrict__ ptrs,
    int* __restrict__ out,
    int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int sum = 0;
        for (int r = 0; r < world_size; ++r) {
            const int* src = (const int*)ptrs[r];
            sum += src[idx];
        }
        out[idx] = sum;
    }
}

__global__ void allreduce_i64_kernel(
    const long long* __restrict__ ptrs,
    long long* __restrict__ out,
    int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        long long sum = 0;
        for (int r = 0; r < world_size; ++r) {
            const long long* src = (const long long*)ptrs[r];
            sum += src[idx];
        }
        out[idx] = sum;
    }
}

__global__ void allreduce_f64_kernel(
    const long long* __restrict__ ptrs,
    double* __restrict__ out,
    int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        double sum = 0.0;
        for (int r = 0; r < world_size; ++r) {
            const double* src = (const double*)ptrs[r];
            sum += src[idx];
        }
        out[idx] = sum;
    }
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel,
    int world_size, int rank,
    int num_blocks, int block_size, int block_stride
) {
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel, world_size, rank, block_stride);
}

void launch_allreduce(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n,
    int dtype_enum
) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();

    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
    } else if (dtype_enum == 1) {
        allreduce_f32_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<float>(), world_size, n);
    } else if (dtype_enum == 2) {
        allreduce_f16_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, (__half*)out.data_ptr<at::Half>(), world_size, n);
    } else if (dtype_enum == 3) {
        allreduce_i32_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<int>(), world_size, n);
    } else if (dtype_enum == 4) {
        allreduce_i64_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, (long long*)out.data_ptr<int64_t>(), world_size, n);
    } else if (dtype_enum == 5) {
        allreduce_f64_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<double>(), world_size, n);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16,
          "Multimem all-reduce on symmetric multicast pointer");
    m.def("launch_allreduce", &launch_allreduce, "Custom P2P all-reduce kernel");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("p2p_allreduce_multimem_ext_v2", CUDA_SRC)
    return _ext


WARP_SIZE = 32
MAX_NUM_BLOCKS = 8
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16


def _multimem_launch_config(numel: int, world_size: int) -> tuple[int, int, int]:
    numel_per_thread = BYTES_PER_THREAD // 2  # bf16
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < max(num_threads, 1):
            block_size *= 2
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min(
            (num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE,
            MAX_NUM_BLOCKS,
        )
    return num_blocks, max(block_size, 1), max(block_size, 1)


_DTYPE_ENUM = {
    torch.bfloat16: 0,
    torch.float32: 1,
    torch.float16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float64: 5,
}


_resource_cache = {}


def _get_resources(shape, dtype, device):
    key = (tuple(shape), dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]

    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    out = torch.empty(shape, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, out, ptrs_tensor)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor.clone()

    input_tensor = tensor.contiguous()
    n = input_tensor.numel()
    dtype = input_tensor.dtype

    # Trigger compile on rank 0 first to avoid races
    _get_ext()

    buf, hdl, out, ptrs_tensor = _get_resources(input_tensor.shape, dtype, input_tensor.device)
    buf.copy_(input_tensor)

    if dtype == torch.bfloat16:
        numel_per_thread = BYTES_PER_THREAD // input_tensor.element_size()
        if n % numel_per_thread == 0 and n > 0:
            numel_128 = n // numel_per_thread
            num_blocks, block_size, block_stride = _multimem_launch_config(n, hdl.world_size)

            hdl.barrier(channel=0)

            multicast_ptr = int(hdl.multicast_ptr)
            signal_dev = hdl.signal_pad_ptrs_dev
            _get_ext().launch_multimem_allreduce_bf16(
                multicast_ptr,
                signal_dev,
                numel_128,
                hdl.world_size,
                hdl.rank,
                num_blocks,
                block_size,
                block_stride,
            )
            return buf.clone()

        # Fallback for non-aligned bf16
        hdl.barrier(channel=0)
        _get_ext().launch_allreduce(ptrs_tensor, out, n, 0)
        return out

    if dtype not in _DTYPE_ENUM:
        # Fallback to NCCL for unsupported dtypes
        out_t = input_tensor.clone()
        dist.all_reduce(out_t, op=dist.ReduceOp.SUM)
        return out_t

    hdl.barrier(channel=0)
    _get_ext().launch_allreduce(ptrs_tensor, out, n, _DTYPE_ENUM[dtype])
    return out