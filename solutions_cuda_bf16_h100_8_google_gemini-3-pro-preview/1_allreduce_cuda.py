"""
Strategy:
1. We use `torch.distributed._symmetric_memory` to allocate identical device buffers across ranks.
2. The buffer is safely padded to a multiple of 16 bytes (8 bf16 elements). This ensures we can unconditionally use the Hopper NVSwitch `multimem` (hardware-accelerated multicast and in-switch reduction) for all BF16 inputs, avoiding slow fallback paths.
3. The custom CUDA multimem kernel perfectly distributes the reduction workload across all GPUs. Each GPU loads a disjoint slice of the mapped multicast window using `multimem.ld_reduce.v4.bf16x2`, letting the NVSwitch execute the actual reductions automatically. The result is written back via `multimem.st.v4.f32` (multicast store).
4. A device-side grid barrier (`blockwise_barrier_acq_rel`) synchronizes execution globally, ensuring no rank exits the kernel or overwrites the symmetric buffer before the collective completely finishes.
5. A custom CUDA fallback with native template dispatch handles arbitrary numeric types (fp32, fp16, int32) using direct P2P memory access. This completely replaces standard PyTorch NCCL collectives on the hot path.
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

// ---------------------------------------------------------------------------
// Blockwise barrier across symmetric signal pads
// ---------------------------------------------------------------------------

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
    uint64_t block_id, int rank, int world_size)
{
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
    uint64_t block_id, int rank, int world_size)
{
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

// ---------------------------------------------------------------------------
// NVSwitch Multimem operations
// ---------------------------------------------------------------------------

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr) : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
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
    int block_stride)
{
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank = (numel_128 + world_size - 1) / world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = (int64_t)block_id * block_stride;
         block_start < numel_per_rank;
         block_start += (int64_t)num_programs * block_stride)
    {
        const int64_t offset = block_start + tid;
        if (offset >= numel_per_rank) continue;
        
        const int64_t idx = rank * numel_per_rank + offset;
        if (idx < numel_128) {
            uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
            uint32_t x, y, z, w;
            multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
            multimem_st_bf16x4(ptrs, x, y, z, w);
        }
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// ---------------------------------------------------------------------------
// Scalar fallback for any dtype
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void allreduce_fallback_kernel(
    const long long* __restrict__ ptrs,
    scalar_t* __restrict__ out,
    int world_size, int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        double sum = 0.0;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const scalar_t* src = (const scalar_t*)ptrs[r];
            sum += static_cast<double>(src[idx]);
        }
        out[idx] = static_cast<scalar_t>(sum);
    }
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel_128,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride)
{
    const uint64_t* d_signal = reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel_128, world_size, rank, block_stride);
}

void launch_allreduce_fallback(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n)
{
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, out.scalar_type(), "allreduce_fallback", [&] {
        allreduce_fallback_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<scalar_t>(), world_size, n);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce_fallback", &launch_allreduce_fallback);
}
'''

_ext = None
_ext_compiled = False

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("allreduce_cuda_bf16_h100_ext", CUDA_SRC)
    return _ext

def _compile_ext():
    global _ext_compiled
    if not _ext_compiled:
        if dist.get_rank() == 0:
            _get_ext()
        dist.barrier()
        _get_ext()
        _ext_compiled = True

def _multimem_launch_config(numel: int, world_size: int) -> tuple[int, int, int]:
    numel_per_thread = 16 // 2  # 8 bf16 elements per 128-bit chunk
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    
    block_size = 32 # Minimum 32 threads required for the kernel's blockwise barrier to support world_size <= 32
    while block_size < num_threads and block_size < 1024:
        block_size *= 2
        
    if num_threads <= 1024:
        num_blocks = 1
    else:
        num_blocks = min((num_threads + 1024 - 1) // 1024, 4)
        
    return num_blocks, block_size, block_size

_resource_cache = {}

def _get_resources(numel: int, dtype: torch.dtype, device: torch.device):
    key = (numel, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]
    
    pad_numel = (numel + 7) // 8 * 8
    buf = symm_mem.empty(pad_numel, device=device, dtype=dtype)
    buf.zero_()
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    
    out = torch.empty(numel, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    res = (buf, hdl, out, ptrs_tensor, pad_numel)
    _resource_cache[key] = res
    return res

@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor.clone()
        
    input_tensor = tensor.contiguous()
    n = input_tensor.numel()
    if n == 0:
        return input_tensor.clone()
        
    dtype = input_tensor.dtype
    
    if not _ext_compiled:
        _compile_ext()
        
    buf, hdl, out, ptrs_tensor, pad_numel = _get_resources(n, dtype, input_tensor.device)
    
    buf[:n].copy_(input_tensor.view(-1))
    if pad_numel > n:
        buf[n:].zero_()
        
    # Stream-ordered device barrier ensures safe visibility of all copies to peers before kernel starts
    hdl.barrier(channel=0)
    
    if dtype == torch.bfloat16:
        numel_128 = pad_numel // 8
        num_blocks, block_size, block_stride = _multimem_launch_config(pad_numel, hdl.world_size)
        
        multicast_ptr = int(hdl.multicast_ptr)
        signal_dev = hdl.signal_pad_ptrs_dev
        
        _get_ext().launch_multimem_allreduce_bf16(
            multicast_ptr, signal_dev, numel_128, hdl.world_size,
            hdl.rank, num_blocks, block_size, block_stride
        )
        # The blockwise_barrier in the kernel natively guarantees completion, no extra barrier required
        return buf[:n].view(input_tensor.shape).clone()
    else:
        _get_ext().launch_allreduce_fallback(ptrs_tensor, out, n)
        # Add post-kernel barrier to avoid immediate next-iteration overwrites of `buf`
        hdl.barrier(channel=0)
        return out.view(input_tensor.shape)