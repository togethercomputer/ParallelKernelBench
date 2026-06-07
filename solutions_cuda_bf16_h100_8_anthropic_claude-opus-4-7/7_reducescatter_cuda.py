"""
Reduce-scatter via symmetric memory + NVSwitch multimem PTX.

Each rank reads its assigned chunk through the multicast pointer using
multimem.ld_reduce (in-switch SUM). No broadcast needed - only this rank
needs its chunk. Single barrier before, single barrier after.
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

// Reduce-scatter kernel: each rank reads its own chunk via multimem.ld_reduce,
// stores to local output. Chunk offset is (rank * chunk_numel_128) elements.
__global__ void multimem_reduce_scatter_bf16_kernel(
    uint64_t multicast_base,           // multicast pointer to symm buffer
    uint4* __restrict__ out,           // local output (chunk_numel_128 v4 elements)
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t chunk_numel_128,           // # of 128-bit elements per chunk
    int world_size,
    int rank
) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const uint64_t* mc_chunk_base =
        reinterpret_cast<const uint64_t*>(multicast_base) +
        (uint64_t)rank * (uint64_t)chunk_numel_128 * 2ULL;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < chunk_numel_128; i += stride) {
        const uint64_t* addr = mc_chunk_base + i * 2ULL;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(addr, x, y, z, w);
        uint4 v = make_uint4(x, y, z, w);
        out[i] = v;
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// Fallback: peer-pointer reduce-scatter for non-bf16 / unaligned cases.
__global__ void rs_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size, int rank,
    int64_t chunk_n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t base = (int64_t)rank * chunk_n;
    for (; idx < chunk_n; idx += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[base + idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

__global__ void rs_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size, int rank,
    int64_t chunk_n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t base = (int64_t)rank * chunk_n;
    for (; idx < chunk_n; idx += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const float* src = (const float*)ptrs[r];
            sum += src[base + idx];
        }
        out[idx] = sum;
    }
}

void launch_multimem_rs_bf16(
    uint64_t multicast_ptr,
    torch::Tensor out,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t chunk_numel_128,
    int world_size,
    int rank,
    int num_blocks,
    int block_size
) {
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_reduce_scatter_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr,
        reinterpret_cast<uint4*>(out.data_ptr()),
        d_signal,
        chunk_numel_128,
        world_size,
        rank);
}

void launch_rs_fallback(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int rank,
    int64_t chunk_n,
    int dtype_enum
) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (chunk_n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (dtype_enum == 0) {
        rs_bf16_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
            world_size, rank, chunk_n);
    } else {
        rs_f32_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<float>(),
            world_size, rank, chunk_n);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_rs_bf16", &launch_multimem_rs_bf16, "Multimem reduce-scatter bf16");
    m.def("launch_rs_fallback", &launch_rs_fallback, "Peer-pointer reduce-scatter fallback");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("rs_multimem_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _get_resources(shape, dtype, device):
    key = (tuple(shape), dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]

    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, ptrs_tensor)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    input_tensor = tensor.contiguous()
    n = input_tensor.numel()
    chunk_n = n // world_size
    out_shape = (input_tensor.shape[0] // world_size,) + tuple(input_tensor.shape[1:])
    out = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    buf, hdl, ptrs_tensor = _get_resources(tuple(input_tensor.shape), input_tensor.dtype, input_tensor.device)
    buf.copy_(input_tensor)

    ext = _get_ext()

    # Multimem path: bf16, chunk size aligned to 8 elements (128-bit)
    if input_tensor.dtype == torch.bfloat16 and (chunk_n % 8 == 0):
        chunk_numel_128 = chunk_n // 8

        # Sync writes to symmetric buffer across all ranks
        hdl.barrier(channel=0)

        block_size = 512
        num_blocks = min((chunk_numel_128 + block_size - 1) // block_size, 16)
        if num_blocks < 1:
            num_blocks = 1

        ext.launch_multimem_rs_bf16(
            int(hdl.multicast_ptr),
            out.view(-1).view(torch.bfloat16),
            hdl.signal_pad_ptrs_dev,
            chunk_numel_128,
            hdl.world_size,
            hdl.rank,
            num_blocks,
            block_size,
        )
        return out

    # Fallback path
    hdl.barrier(channel=0)
    dtype_enum = 0 if input_tensor.dtype == torch.bfloat16 else 1
    ext.launch_rs_fallback(ptrs_tensor, out, rank, chunk_n, dtype_enum)
    hdl.barrier(channel=0)
    return out