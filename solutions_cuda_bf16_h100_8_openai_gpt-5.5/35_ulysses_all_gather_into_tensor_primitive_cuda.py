from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

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

__device__ __forceinline__ void blockwise_barrier(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t slot,
    int rank,
    int world_size
) {
    int peer = (int)threadIdx.x;
    if (peer >= world_size) {
        return;
    }

    const uint64_t elem_off = (slot * (uint64_t)world_size + (uint64_t)rank) * sizeof(uint32_t);
    const uint64_t wait_off = (slot * (uint64_t)world_size + (uint64_t)peer) * sizeof(uint32_t);

    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        (uintptr_t)signal_pad_ptrs[peer] + elem_off);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        (uintptr_t)signal_pad_ptrs[rank] + wait_off);

    send_signal_release(send_addr);
    wait_signal_acquire(wait_addr);
}

template<int ITEMS>
__global__ void allgather_broadcast_vec16_kernel(
    const char* __restrict__ x_bytes,
    const long long* __restrict__ out_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t nvec16,
    int64_t shard_nbytes,
    int world_size,
    int rank
) {
    const uint4* __restrict__ x_vec = reinterpret_cast<const uint4*>(x_bytes);
    const int tid = (int)threadIdx.x;
    const int64_t tile_vecs = (int64_t)blockDim.x * ITEMS;

    for (int64_t tile = (int64_t)blockIdx.x * tile_vecs;
         tile < nvec16;
         tile += (int64_t)gridDim.x * tile_vecs) {

        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            int64_t v = tile + (int64_t)tid + (int64_t)j * blockDim.x;
            if (v < nvec16) {
                uint4 val = x_vec[v];

                for (int peer = 0; peer < world_size; ++peer) {
                    char* peer_out = reinterpret_cast<char*>(
                        (uintptr_t)out_ptrs[peer] + (int64_t)rank * shard_nbytes);
                    uint4* __restrict__ dst_vec = reinterpret_cast<uint4*>(peer_out);
                    dst_vec[v] = val;
                }
            }
        }

        __threadfence_system();
        __syncthreads();
        blockwise_barrier(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
        __syncthreads();
    }
}

template<int ITEMS>
__global__ void allgather_broadcast_byte_kernel(
    const unsigned char* __restrict__ x_bytes,
    const long long* __restrict__ out_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t shard_nbytes,
    int world_size,
    int rank
) {
    const int tid = (int)threadIdx.x;
    const int64_t tile_bytes = (int64_t)blockDim.x * ITEMS;

    for (int64_t tile = (int64_t)blockIdx.x * tile_bytes;
         tile < shard_nbytes;
         tile += (int64_t)gridDim.x * tile_bytes) {

        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            int64_t b = tile + (int64_t)tid + (int64_t)j * blockDim.x;
            if (b < shard_nbytes) {
                unsigned char val = x_bytes[b];

                for (int peer = 0; peer < world_size; ++peer) {
                    unsigned char* peer_out = reinterpret_cast<unsigned char*>(
                        (uintptr_t)out_ptrs[peer] + (int64_t)rank * shard_nbytes);
                    peer_out[b] = val;
                }
            }
        }

        __threadfence_system();
        __syncthreads();
        blockwise_barrier(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
        __syncthreads();
    }
}

void launch_ulysses_allgather_broadcast(
    torch::Tensor x,
    torch::Tensor out,
    torch::Tensor out_ptrs_tensor,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t shard_nbytes,
    int world_size,
    int rank,
    int num_blocks,
    int num_threads
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out_ptrs_tensor.is_cuda(), "out_ptrs_tensor must be CUDA");
    TORCH_CHECK(signal_pad_ptrs_tensor.is_cuda(), "signal_pad_ptrs_tensor must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out_ptrs_tensor.dtype() == torch::kInt64, "out_ptrs_tensor must be int64");
    TORCH_CHECK(signal_pad_ptrs_tensor.dtype() == torch::kInt64, "signal_pad_ptrs_tensor must be int64");

    if (shard_nbytes <= 0) {
        return;
    }

    const long long* out_ptrs =
        reinterpret_cast<const long long*>(out_ptrs_tensor.data_ptr<int64_t>());
    const uint64_t* signal_ptrs =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if ((shard_nbytes & 15LL) == 0) {
        constexpr int ITEMS = 8;
        int64_t nvec16 = shard_nbytes >> 4;
        allgather_broadcast_vec16_kernel<ITEMS>
            <<<num_blocks, num_threads, 0, stream>>>(
                reinterpret_cast<const char*>(x.data_ptr()),
                out_ptrs,
                signal_ptrs,
                nvec16,
                shard_nbytes,
                world_size,
                rank);
    } else {
        constexpr int ITEMS = 16;
        allgather_broadcast_byte_kernel<ITEMS>
            <<<num_blocks, num_threads, 0, stream>>>(
                reinterpret_cast<const unsigned char*>(x.data_ptr()),
                out_ptrs,
                signal_ptrs,
                shard_nbytes,
                world_size,
                rank);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "launch_ulysses_allgather_broadcast",
        &launch_ulysses_allgather_broadcast,
        "Ulysses all_gather_into_tensor via symmetric-memory UVA peer stores");
}
'''


_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_allgather_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


def _device_key(device: torch.device):
    if device.index is not None:
        return (device.type, device.index)
    return (device.type, torch.cuda.current_device())


def _get_resources(x: torch.Tensor, group, world_size: int):
    out_shape = list(x.shape)
    out_shape[0] *= world_size
    out_shape = tuple(out_shape)

    key = (
        tuple(x.shape),
        out_shape,
        x.dtype,
        _device_key(x.device),
        id(group),
        world_size,
    )

    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    out = symm_mem.empty(out_shape, dtype=x.dtype, device=x.device)
    hdl = symm_mem.rendezvous(out, group)

    ptrs = torch.tensor(
        [int(p) for p in hdl.buffer_ptrs],
        dtype=torch.int64,
        device=x.device,
    )

    cached = (out, hdl, ptrs)
    _resource_cache[key] = cached
    return cached


def _launch_config(shard_nbytes: int, device: torch.device):
    threads = 256
    if shard_nbytes <= 0:
        return 1, threads

    if (shard_nbytes & 15) == 0:
        units = shard_nbytes // 16
        items = 8
    else:
        units = shard_nbytes
        items = 16

    blocks_needed = (units + threads * items - 1) // (threads * items)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    # Keep every block resident to avoid producer/consumer deadlock in the
    # cross-rank device-side block barriers.
    blocks = max(1, min(int(blocks_needed), int(sm_count)))
    return blocks, threads


@torch.no_grad()
def solution(
    x: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return x.contiguous()

    x = x.contiguous()

    dim_size = list(x.size())
    dim_size[0] = dim_size[0] * world_size

    if x.numel() == 0:
        return torch.empty(dim_size, dtype=x.dtype, device=x.device)

    out, hdl, ptrs = _get_resources(x, group, world_size)

    shard_nbytes = x.numel() * x.element_size()
    blocks, threads = _launch_config(shard_nbytes, x.device)

    rank = getattr(hdl, "rank", dist.get_rank(group))

    _get_ext().launch_ulysses_allgather_broadcast(
        x,
        out,
        ptrs,
        hdl.signal_pad_ptrs_dev,
        int(shard_nbytes),
        int(world_size),
        int(rank),
        int(blocks),
        int(threads),
    )

    return out