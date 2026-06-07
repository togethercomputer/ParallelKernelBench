import math
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

__device__ __forceinline__ void blockwise_barrier_acq_rel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    const int tid = threadIdx.x;
    if (tid < world_size) {
        uint32_t* local_base  = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank]);
        uint32_t* remote_base = reinterpret_cast<uint32_t*>(signal_pad_ptrs[tid]);

        uint32_t* send_addr = remote_base + block_id * (uint64_t)world_size + rank;
        uint32_t* wait_addr = local_base  + block_id * (uint64_t)world_size + tid;

        send_signal_release(send_addr);
        wait_signal_acquire(wait_addr);
    }
}

__global__ void allgather_push_vec16_kernel(
    const uint4* __restrict__ src,
    const int64_t* __restrict__ out_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t n_vec16,
    int64_t nbytes_per_rank,
    int world_size,
    int rank
) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    #pragma unroll
    for (int dst_rank = 0; dst_rank < 8; ++dst_rank) {
        if (dst_rank >= world_size) break;

        char* dst_bytes = reinterpret_cast<char*>(out_ptrs[dst_rank]) +
                          (int64_t)rank * nbytes_per_rank;
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(dst_bytes);

        for (int64_t i = tid; i < n_vec16; i += stride) {
            dst[i] = src[i];
        }
    }

    __threadfence_system();
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
}

__global__ void allgather_push_bytes_kernel(
    const uint8_t* __restrict__ src,
    const int64_t* __restrict__ out_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t nbytes_per_rank,
    int world_size,
    int rank
) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    #pragma unroll
    for (int dst_rank = 0; dst_rank < 8; ++dst_rank) {
        if (dst_rank >= world_size) break;

        uint8_t* __restrict__ dst =
            reinterpret_cast<uint8_t*>(out_ptrs[dst_rank]) +
            (int64_t)rank * nbytes_per_rank;

        for (int64_t i = tid; i < nbytes_per_rank; i += stride) {
            dst[i] = src[i];
        }
    }

    __threadfence_system();
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
}

void launch_allgather_push(
    torch::Tensor input,
    torch::Tensor out_ptrs_tensor,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t nbytes_per_rank,
    int world_size,
    int rank,
    int num_blocks,
    int num_threads
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(out_ptrs_tensor.is_cuda(), "out_ptrs_tensor must be CUDA");
    TORCH_CHECK(signal_pad_ptrs_tensor.is_cuda(), "signal_pad_ptrs_tensor must be CUDA");
    TORCH_CHECK(out_ptrs_tensor.dtype() == torch::kInt64, "out_ptrs_tensor must be int64");
    TORCH_CHECK(signal_pad_ptrs_tensor.dtype() == torch::kInt64,
                "signal_pad_ptrs_tensor must be int64/uint64 storage");
    TORCH_CHECK(world_size >= 1 && world_size <= 8, "optimized path expects world_size in [1, 8]");

    const uintptr_t src_addr = reinterpret_cast<uintptr_t>(input.data_ptr());
    const bool vec16 =
        ((src_addr & 0xFULL) == 0) && ((nbytes_per_rank & 0xFULL) == 0);

    const int64_t* out_ptrs =
        reinterpret_cast<const int64_t*>(out_ptrs_tensor.data_ptr<int64_t>());
    const uint64_t* signal_ptrs =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (vec16) {
        const int64_t n_vec16 = nbytes_per_rank >> 4;
        allgather_push_vec16_kernel<<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(input.data_ptr()),
            out_ptrs,
            signal_ptrs,
            n_vec16,
            nbytes_per_rank,
            world_size,
            rank
        );
    } else {
        allgather_push_bytes_kernel<<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(input.data_ptr()),
            out_ptrs,
            signal_ptrs,
            nbytes_per_rank,
            world_size,
            rank
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_allgather_push", &launch_allgather_push,
          "Symmetric-memory UVA push all-gather");
}
'''


_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_uva_push_allgather_bf16_h100_ext", CUDA_SRC)
    return _ext


def _launch_config(nbytes: int) -> tuple[int, int]:
    threads = 256
    if nbytes <= 0:
        return 1, threads

    # Keep all blocks resident on H100 to avoid device-barrier scheduling hazards,
    # while still exposing enough parallelism to saturate NVLink for large BF16 payloads.
    vec_items = (nbytes + 15) // 16
    blocks = (vec_items + threads * 8 - 1) // (threads * 8)

    if nbytes < 256 * 1024:
        max_blocks = 4
    elif nbytes < 4 * 1024 * 1024:
        max_blocks = 16
    else:
        max_blocks = 32

    blocks = max(1, min(max_blocks, blocks))
    return blocks, threads


def _get_resources(input_shape, dtype, device, world_size):
    key = (tuple(input_shape), dtype, int(device.index if device.index is not None else torch.cuda.current_device()), world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    out_shape = (world_size,) + tuple(input_shape)
    out = symm_mem.empty(out_shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(out, dist.group.WORLD)

    # Device array of UVA output base pointers, one per rank.
    out_ptrs_dev = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (out, hdl, out_ptrs_dev)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda, "input must be CUDA"
    assert tensor.is_contiguous(), "input must be contiguous"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    out, hdl, out_ptrs_dev = _get_resources(
        tensor.shape,
        tensor.dtype,
        tensor.device,
        world_size,
    )

    nbytes = tensor.numel() * tensor.element_size()
    blocks, threads = _launch_config(nbytes)

    _get_ext().launch_allgather_push(
        tensor,
        out_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
        nbytes,
        world_size,
        rank,
        blocks,
        threads,
    )

    return out