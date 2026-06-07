from typing import Optional, Tuple

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
#include <algorithm>

// ---------------------------------------------------------------------------
// Blockwise barrier definitions
// ---------------------------------------------------------------------------

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp)
            : "l"(addr)
            : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp)
            : "l"(addr)
            : "memory");
    } while (tmp != 1u);
}

__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp)
            : "l"(addr)
            : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp)
            : "l"(addr)
            : "memory");
    } while (tmp != 1u);
}

__device__ void blockwise_barrier_relaxed(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) {
        return;
    }
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
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) {
        return;
    }
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

// ---------------------------------------------------------------------------
// NVSwitch Multimem ALLREDUCE
// ---------------------------------------------------------------------------

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr)
        : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr,
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
        if (offsets >= numel_per_rank) {
            continue;
        }
        const int64_t idx = (int64_t)rank * numel_per_rank + offsets;
        uint64_t* ptrs =
            reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// ---------------------------------------------------------------------------
// Fallback Peer-Pointer ALLREDUCE
// ---------------------------------------------------------------------------

template<typename T>
__global__ void allreduce_sum_kernel(
    const long long* __restrict__ ptrs,
    T* __restrict__ out,
    int world_size,
    int64_t n
);

template<>
__global__ void allreduce_sum_kernel<at::BFloat16>(
    const long long* __restrict__ ptrs,
    at::BFloat16* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

template<>
__global__ void allreduce_sum_kernel<float>(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const float* src = (const float*)ptrs[r];
            sum += src[idx];
        }
        out[idx] = sum;
    }
}

// ---------------------------------------------------------------------------
// Pack and Unpack Kernels
// ---------------------------------------------------------------------------

template<typename T>
__global__ void pack_3_kernel(
    const T* __restrict__ in1, int n1,
    const T* __restrict__ in2, int n2,
    const T* __restrict__ in3, int n3,
    T* __restrict__ out
) {
    if (blockIdx.y == 0) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n1) out[idx] = in1[idx];
    } else if (blockIdx.y == 1) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n2) out[n1 + idx] = in2[idx];
    } else if (blockIdx.y == 2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n3) out[n1 + n2 + idx] = in3[idx];
    }
}

template<typename T>
__global__ void unpack_3_kernel(
    const T* __restrict__ in,
    int n1, T* __restrict__ out1,
    int n2, T* __restrict__ out2,
    int n3, T* __restrict__ out3
) {
    if (blockIdx.y == 0) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n1) out1[idx] = in[idx];
    } else if (blockIdx.y == 1) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n2) out2[idx] = in[n1 + idx];
    } else if (blockIdx.y == 2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n3) out3[idx] = in[n1 + n2 + idx];
    }
}

// ---------------------------------------------------------------------------
// Extension Bindings
// ---------------------------------------------------------------------------

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel_128,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride
) {
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr,
        d_signal,
        numel_128,
        world_size,
        rank,
        block_stride);
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
    if (blocks == 0) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        allreduce_sum_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<at::BFloat16>(), world_size, n);
    } else if (dtype_enum == 1) {
        allreduce_sum_kernel<float><<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<float>(), world_size, n);
    }
}

void launch_pack_3(
    torch::Tensor t1, torch::Tensor t2, torch::Tensor t3, torch::Tensor out) {
    int n1 = t1.numel();
    int n2 = t2.numel();
    int n3 = t3.numel();
    int max_n = std::max(n1, std::max(n2, n3));
    int threads = 256;
    int blocks_x = (max_n + threads - 1) / threads;
    if (blocks_x == 0) blocks_x = 1;
    dim3 blocks(blocks_x, 3);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (t1.dtype() == torch::kBFloat16) {
        pack_3_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
            t1.data_ptr<at::BFloat16>(), n1,
            t2.data_ptr<at::BFloat16>(), n2,
            t3.data_ptr<at::BFloat16>(), n3,
            out.data_ptr<at::BFloat16>()
        );
    } else if (t1.dtype() == torch::kFloat32) {
        pack_3_kernel<float><<<blocks, threads, 0, stream>>>(
            t1.data_ptr<float>(), n1,
            t2.data_ptr<float>(), n2,
            t3.data_ptr<float>(), n3,
            out.data_ptr<float>()
        );
    }
}

void launch_unpack_3(
    torch::Tensor in, torch::Tensor t1, torch::Tensor t2, torch::Tensor t3) {
    int n1 = t1.numel();
    int n2 = t2.numel();
    int n3 = t3.numel();
    int max_n = std::max(n1, std::max(n2, n3));
    int threads = 256;
    int blocks_x = (max_n + threads - 1) / threads;
    if (blocks_x == 0) blocks_x = 1;
    dim3 blocks(blocks_x, 3);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (t1.dtype() == torch::kBFloat16) {
        unpack_3_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
            in.data_ptr<at::BFloat16>(),
            n1, t1.data_ptr<at::BFloat16>(),
            n2, t2.data_ptr<at::BFloat16>(),
            n3, t3.data_ptr<at::BFloat16>()
        );
    } else if (t1.dtype() == torch::kFloat32) {
        unpack_3_kernel<float><<<blocks, threads, 0, stream>>>(
            in.data_ptr<float>(),
            n1, t1.data_ptr<float>(),
            n2, t2.data_ptr<float>(),
            n3, t3.data_ptr<float>()
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce", &launch_allreduce);
    m.def("launch_pack_3", &launch_pack_3);
    m.def("launch_unpack_3", &launch_unpack_3);
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_ep_lora_allreduce", CUDA_SRC)
    return _ext


WARP_SIZE = 32
MAX_NUM_BLOCKS = 4
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16


def _multimem_launch_config(numel: int, world_size: int) -> tuple[int, int, int]:
    numel_per_thread = BYTES_PER_THREAD // 2  # bf16 elements per thread
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 32  # Minimum bounds to prevent deadlock on blockwise barrier subsets
        while block_size < num_threads:
            block_size *= 2
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min(
            (num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE,
            MAX_NUM_BLOCKS,
        )
    return num_blocks, block_size, block_size


_resource_cache = {}


def _get_resources(padded_n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = (padded_n, dtype, device, id(group))
    if key in _resource_cache:
        return _resource_cache[key]

    # Initialize symmetrically mapping tensors with empty defaults; explicit zero guarantees 
    # out-of-bounds padded values won't corrupt the eventual reduce result.
    buf = symm_mem.empty(padded_n, device=device, dtype=dtype)
    buf.zero_()
    hdl = symm_mem.rendezvous(buf, group=group)

    out_buf = torch.empty(padded_n, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, ptrs_tensor, out_buf)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    grad_fc1_1_lora_A: torch.Tensor,
    grad_fc1_2_lora_A: torch.Tensor,
    grad_fc2_lora_B: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B

    n1 = grad_fc1_1_lora_A.numel()
    n2 = grad_fc1_2_lora_A.numel()
    n3 = grad_fc2_lora_B.numel()
    total_n = n1 + n2 + n3
    
    if total_n == 0:
        return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B

    dtype = grad_fc1_1_lora_A.dtype
    device = grad_fc1_1_lora_A.device

    # Pad payload perfectly onto world_size * 16-byte boundaries so the fallback 
    # memory instructions and multimem accesses do not fault.
    chunk_size = world_size * 8
    padded_n = ((total_n + chunk_size - 1) // chunk_size) * chunk_size

    buf, hdl, ptrs_tensor, out_buf = _get_resources(padded_n, dtype, device, group)

    c1 = grad_fc1_1_lora_A.is_contiguous()
    c2 = grad_fc1_2_lora_A.is_contiguous()
    c3 = grad_fc2_lora_B.is_contiguous()

    t1 = grad_fc1_1_lora_A if c1 else grad_fc1_1_lora_A.contiguous()
    t2 = grad_fc1_2_lora_A if c2 else grad_fc1_2_lora_A.contiguous()
    t3 = grad_fc2_lora_B if c3 else grad_fc2_lora_B.contiguous()

    ext = _get_ext()
    ext.launch_pack_3(t1, t2, t3, buf)

    rank = dist.get_rank(group)
    multicast_ptr = getattr(hdl, 'multicast_ptr', 0)
    use_multimem = (multicast_ptr != 0 and dtype == torch.bfloat16)

    if use_multimem:
        numel_128 = padded_n // 8
        num_blocks, block_size, block_stride = _multimem_launch_config(padded_n, world_size)

        dist.barrier(group=group)
        signal_dev = hdl.signal_pad_ptrs_dev
        
        ext.launch_multimem_allreduce_bf16(
            multicast_ptr,
            signal_dev,
            numel_128,
            world_size,
            rank,
            num_blocks,
            block_size,
            block_stride,
        )
        unpack_buf = buf
    else:
        hdl.barrier(channel=0)
        dtype_enum = 0 if dtype == torch.bfloat16 else 1
        ext.launch_allreduce(ptrs_tensor, out_buf, total_n, dtype_enum)
        hdl.barrier(channel=0)
        unpack_buf = out_buf

    ext.launch_unpack_3(unpack_buf, t1, t2, t3)

    if not c1: grad_fc1_1_lora_A.copy_(t1)
    if not c2: grad_fc1_2_lora_A.copy_(t2)
    if not c3: grad_fc2_lora_B.copy_(t3)

    return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B