"""
Row-parallel Conv3d with custom multimem all-reduce.

Strategy:
- Run local Conv3d via cuDNN (PyTorch F.conv3d) — hand-rolled Conv3d won't beat cuDNN.
- Replace dist.all_reduce with NVSwitch multimem.ld_reduce + multimem.st on bf16 symmetric buffer.
- Add bias as a fused epilogue kernel after reduction (saves a full pass over the tensor).
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension

_CONV3D_NUMEL_LIMIT = 2**31


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

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr) : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr,
    uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

// All-reduce via multimem; output written to local symmetric buffer (per rank).
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
        if (idx >= numel_128) continue;
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// Peer-pointer fallback: sums local symmetric buffers across ranks into out.
__global__ void allreduce_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size, int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

// Bias add epilogue. Reads from sym buffer (already reduced), writes to out, fused bias add.
__global__ void bias_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ inp,
    const __nv_bfloat16* __restrict__ bias,  // [C]
    __nv_bfloat16* __restrict__ out,
    int64_t total,
    int64_t per_channel,  // T*H*W
    int channels)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int64_t c = (idx / per_channel) % channels;
        float v = __bfloat162float(inp[idx]) + __bfloat162float(bias[c]);
        out[idx] = __float2bfloat16(v);
    }
}

__global__ void copy_bf16_kernel(
    const __nv_bfloat16* __restrict__ inp,
    __nv_bfloat16* __restrict__ out,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        out[idx] = inp[idx];
    }
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel_128,
    int world_size, int rank,
    int num_blocks, int block_size, int block_stride)
{
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel_128, world_size, rank, block_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_allreduce_bf16(
    torch::Tensor ptrs_tensor,
    torch::Tensor out, int64_t n)
{
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_bias_add_bf16(
    torch::Tensor inp, torch::Tensor bias, torch::Tensor out,
    int64_t total, int64_t per_channel, int channels)
{
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    bias_add_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)inp.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)bias.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        total, per_channel, channels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_copy_bf16(torch::Tensor inp, torch::Tensor out, int64_t n) {
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)inp.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce_bf16", &launch_allreduce_bf16);
    m.def("launch_bias_add_bf16", &launch_bias_add_bf16);
    m.def("launch_copy_bf16", &launch_copy_bf16);
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("conv3d_ar_ext", CUDA_SRC)
    return _ext


_symm_cache = {}


def _get_symm(numel: int, dtype: torch.dtype, device: torch.device, group):
    key = (numel, dtype, device.index)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]


WARP_SIZE = 32
MAX_NUM_BLOCKS = 24
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16


def _multimem_launch_config(numel: int, world_size: int):
    numel_per_thread = BYTES_PER_THREAD // 2  # bf16
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < max(num_threads, 1):
            block_size *= 2
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min((num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE, MAX_NUM_BLOCKS)
    return num_blocks, max(block_size, 1), max(block_size, 1)


def _to_3tuple(value):
    return (value, value, value) if isinstance(value, int) else value


def _ceil_to_divisible(n: int, dividend: int) -> int:
    return math.ceil(dividend / (dividend // n))


def _output_shape(input_shape, out_channels, kernel_size, stride, padding, dilation):
    shape = [input_shape[0], out_channels]
    for idx, size in enumerate(input_shape[-3:]):
        out = (size + 2 * padding[idx] - dilation[idx] * (kernel_size[idx] - 1) - 1)
        shape.append(math.floor(out / stride[idx] + 1))
    return shape


def _chunk_count(numel: int, channels: int, limit: int) -> int:
    chunks = math.ceil(numel / limit)
    return _ceil_to_divisible(chunks, channels)


def _channel_chunk_conv3d(x, weight, bias, stride, padding, dilation, groups, numel_limit):
    out_channels, in_channels = weight.shape[:2]
    output_shape = _output_shape(x.shape, out_channels, tuple(weight.shape[2:]),
                                  stride, padding, dilation)
    in_chunks = _chunk_count(x.numel(), in_channels, numel_limit)
    out_chunks = _chunk_count(math.prod(output_shape), out_channels, numel_limit)
    if in_chunks == 1 and out_chunks == 1:
        return F.conv3d(x, weight, bias, stride, padding, dilation, groups)

    x_chunks = x.chunk(in_chunks, dim=1)
    weight_out_chunks = weight.chunk(out_chunks, dim=0)
    bias_chunks = bias.chunk(out_chunks) if bias is not None else [None] * out_chunks
    outputs = []
    for weight_chunk, bias_chunk in zip(weight_out_chunks, bias_chunks):
        partial_sum = None
        for x_chunk, w_chunk in zip(x_chunks, weight_chunk.chunk(in_chunks, dim=1)):
            partial = F.conv3d(x_chunk, w_chunk, None, stride, padding, dilation, groups).float()
            partial_sum = partial if partial_sum is None else partial_sum + partial
        out = partial_sum.to(dtype=x.dtype)
        if bias_chunk is not None:
            out = out + bias_chunk.view(1, -1, 1, 1, 1)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def solution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Union[int, Tuple[int, int, int]],
    padding: Union[int, Tuple[int, int, int]],
    dilation: Union[int, Tuple[int, int, int]],
    groups: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD

    # Local conv (no bias), in original dtype.
    local = _channel_chunk_conv3d(
        input, weight, None,
        _to_3tuple(stride), _to_3tuple(padding), _to_3tuple(dilation),
        groups, _CONV3D_NUMEL_LIMIT,
    )

    # Fallback: non-bf16 → standard all_reduce path
    if local.dtype != torch.bfloat16 or not dist.is_initialized():
        if dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM, group=group)
        if bias is not None:
            local = local + bias.view(1, -1, 1, 1, 1)
        return local

    ext = _get_ext()
    n = local.numel()
    device = local.device

    # Round up symmetric buffer to multiple of (world_size * 8) for clean multimem chunks
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    align = world_size * 8  # 8 bf16 elements per 128-bit chunk
    n_pad = ((n + align - 1) // align) * align

    buf, hdl, ptrs_tensor = _get_symm(n_pad, torch.bfloat16, device, group)

    # Copy local conv result into symmetric buffer (zero pad implicit if pad region untouched;
    # we must zero pad tail because peers OR-add it).
    if n_pad > n:
        buf[n:].zero_()
    ext.launch_copy_bf16(local.view(-1), buf[:n], n)

    numel_per_thread = BYTES_PER_THREAD // 2  # 8
    use_multimem = (n_pad % numel_per_thread == 0) and hasattr(hdl, "multicast_ptr")

    if use_multimem:
        numel_128 = n_pad // numel_per_thread
        num_blocks, block_size, block_stride = _multimem_launch_config(n_pad, world_size)
        multicast_ptr = int(hdl.multicast_ptr)
        signal_dev = hdl.signal_pad_ptrs_dev
        ext.launch_multimem_allreduce_bf16(
            multicast_ptr, signal_dev, numel_128,
            world_size, rank, num_blocks, block_size, block_stride,
        )
        reduced = buf[:n]
    else:
        hdl.barrier(channel=0)
        out_buf = torch.empty(n, device=device, dtype=torch.bfloat16)
        ext.launch_allreduce_bf16(ptrs_tensor, out_buf, n)
        reduced = out_buf
        hdl.barrier(channel=0)

    # Fused bias-add epilogue
    out = torch.empty_like(local)
    if bias is not None:
        B, C, T, H, W = local.shape
        per_channel = T * H * W
        ext.launch_bias_add_bf16(
            reduced.view(-1), bias.contiguous().view(-1),
            out.view(-1), n, per_channel, C,
        )
    else:
        ext.launch_copy_bf16(reduced.view(-1), out.view(-1), n)

    return out