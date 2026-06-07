"""
Block-wise INT8 quantize/dequantize + all-reduce average using a fused CUDA kernel
and symmetric memory multimem all-reduce on bfloat16.

Strategy:
- Fuse block-wise INT8 quant -> dequant directly into the symmetric memory buffer
  in bf16, eliminating intermediate fp32 tensors.
- Use NVSwitch multimem.ld_reduce/st on bf16 to perform the all-reduce in-switch.
- Divide by world_size in a fused post-pass.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------- Signal-pad barrier ----------------
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

// ---------------- Fused block INT8 quant->dequant (bf16 input/output) ----------------
// One CUDA block handles one quantization-block of `block_size` elements.
// Padding: indices out of range are treated as 0.
extern "C" __global__ void block_int8_quant_dequant_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,   // length n (input)
    __nv_bfloat16* __restrict__ out,       // length nb*block_size (output, padded)
    int64_t n,
    int block_size,
    int64_t nb)
{
    int64_t bid = blockIdx.x;
    if (bid >= nb) return;
    int tid = threadIdx.x;

    int64_t base = bid * (int64_t)block_size;

    extern __shared__ float smem[];
    // Pass 1: load + compute |x|, reduce max
    float local_max = 0.0f;
    // Each thread handles multiple elements if block_size > blockDim.x
    for (int i = tid; i < block_size; i += blockDim.x) {
        int64_t idx = base + i;
        float v = 0.0f;
        if (idx < n) v = __bfloat162float(x[idx]);
        float av = fabsf(v);
        if (av > local_max) local_max = av;
        smem[i] = v; // stash value
    }

    // Block reduction of local_max
    __shared__ float block_max_arr[32];
    // warp reduce
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(mask, local_max, off);
        if (other > local_max) local_max = other;
    }
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) block_max_arr[warp_id] = local_max;
    __syncthreads();

    float absmax = 0.0f;
    int num_warps = (blockDim.x + 31) >> 5;
    if (tid < num_warps) {
        absmax = block_max_arr[tid];
    }
    if (warp_id == 0) {
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(mask, absmax, off);
            if (other > absmax) absmax = other;
        }
        if (tid == 0) block_max_arr[0] = absmax;
    }
    __syncthreads();
    absmax = block_max_arr[0];
    float scale = fmaxf(absmax, 1e-8f) / 127.0f;
    float inv_scale = 1.0f / scale;

    // Pass 2: quant -> dequant -> bf16 store
    for (int i = tid; i < block_size; i += blockDim.x) {
        float v = smem[i];
        float q = rintf(v * inv_scale);
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;
        float d = q * scale;
        int64_t idx = base + i;
        out[idx] = __float2bfloat16(d);
    }
}

void launch_block_int8_qd_bf16(
    torch::Tensor x,        // bf16 input (n)
    torch::Tensor out,      // bf16 output (nb*block_size)
    int64_t n,
    int block_size,
    int64_t nb)
{
    int threads = block_size < 1024 ? block_size : 1024;
    if (threads < 32) threads = 32;
    // round up to multiple of 32
    threads = ((threads + 31) / 32) * 32;
    int blocks = (int)nb;
    size_t smem_bytes = (size_t)block_size * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    block_int8_quant_dequant_bf16_kernel<<<blocks, threads, smem_bytes, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        n, block_size, nb);
}

// ---------------- Multimem all-reduce + scale (bf16) ----------------
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

// Scales each bf16x2 by 1/world_size after reduction.
__device__ __forceinline__ uint32_t scale_bf16x2(uint32_t packed, float inv_ws) {
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&packed);
    float a = __bfloat162float(v.x) * inv_ws;
    float b = __bfloat162float(v.y) * inv_ws;
    __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    uint32_t out;
    *reinterpret_cast<__nv_bfloat162*>(&out) = r;
    return out;
}

extern "C" __global__ void multimem_allreduce_avg_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128,   // total number of 128-bit (8 bf16) chunks
    int world_size,
    int rank,
    int block_stride,
    float inv_ws)
{
    const uint64_t block_id = (uint64_t)blockIdx.x;
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    int64_t numel_per_rank = (numel_128 + (int64_t)world_size - 1) / (int64_t)world_size;
    int num_programs = gridDim.x;
    int tid = threadIdx.x;

    for (int64_t block_start = (int64_t)block_id * (int64_t)block_stride;
         block_start < numel_per_rank;
         block_start += (int64_t)num_programs * (int64_t)block_stride)
    {
        int64_t offsets = block_start + (int64_t)tid;
        if (offsets >= numel_per_rank) continue;
        int64_t idx = (int64_t)rank * numel_per_rank + offsets;
        if (idx >= numel_128) continue;
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        x = scale_bf16x2(x, inv_ws);
        y = scale_bf16x2(y, inv_ws);
        z = scale_bf16x2(z, inv_ws);
        w = scale_bf16x2(w, inv_ws);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

void launch_multimem_allreduce_avg_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride,
    double inv_ws)
{
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_avg_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel, world_size, rank, block_stride, (float)inv_ws);
}

// ---------------- Peer-pointer fallback all-reduce + scale ----------------
extern "C" __global__ void allreduce_avg_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t n,
    float inv_ws)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(sum * inv_ws);
    }
}

void launch_allreduce_avg_bf16(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n,
    double inv_ws)
{
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = blocks64 > 65535 ? 65535 : (int)blocks64;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_avg_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        world_size, n, (float)inv_ws);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_block_int8_qd_bf16", &launch_block_int8_qd_bf16,
          "Fused block INT8 quant/dequant in bf16");
    m.def("launch_multimem_allreduce_avg_bf16", &launch_multimem_allreduce_avg_bf16,
          "Multimem all-reduce + average for bf16");
    m.def("launch_allreduce_avg_bf16", &launch_allreduce_avg_bf16,
          "Peer-pointer all-reduce + average for bf16");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("quant_grad_avg_ext", CUDA_SRC)
    return _ext


WARP_SIZE = 32
MAX_NUM_BLOCKS = 8
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16  # 8 bf16

def _multimem_launch_config(numel: int, world_size: int):
    numel_per_thread = BYTES_PER_THREAD // 2
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < max(num_threads, 1):
            block_size *= 2
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min((num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE, MAX_NUM_BLOCKS)
    return num_blocks, block_size, block_size


_resource_cache = {}

def _get_resources(padded_numel: int, dtype, device):
    key = (padded_numel, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]
    buf = symm_mem.empty(padded_numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    out_fallback = torch.empty(padded_numel, device=device, dtype=dtype)
    res = (buf, hdl, ptrs_tensor, out_fallback)
    _resource_cache[key] = res
    return res


_compiled_once = False

def _ensure_ext():
    global _compiled_once
    if not _compiled_once:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            _get_ext()
        if dist.is_initialized():
            dist.barrier()
        _get_ext()
        _compiled_once = True


@torch.no_grad()
def solution(flat_grad: Tensor, block_size: int) -> Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert block_size >= 1

    world_size = dist.get_world_size()
    orig_shape = flat_grad.shape
    orig_dtype = flat_grad.dtype
    device = flat_grad.device

    x = flat_grad.reshape(-1).contiguous()
    n = x.numel()

    if n == 0:
        return flat_grad.clone()

    _ensure_ext()
    ext = _get_ext()

    # Round padded length up to multiple of block_size and also multiple of 8 (bf16x8 chunk).
    pad_to = block_size
    # ensure padded_numel % 8 == 0 for multimem path
    nb = (n + block_size - 1) // block_size
    padded = nb * block_size
    # If padded not multiple of 8, expand (cheap; padded zeros)
    if padded % 8 != 0:
        padded = ((padded + 7) // 8) * 8

    # Cast input to bf16 if not already
    if x.dtype != torch.bfloat16:
        x_bf16 = x.to(torch.bfloat16)
    else:
        x_bf16 = x

    buf, hdl, ptrs_tensor, _out_fallback = _get_resources(padded, torch.bfloat16, device)

    # Zero tail padding in symmetric buffer (cheap; only the trailing slice needs zeroing).
    if padded > n:
        buf[n:].zero_()

    # Fused quant->dequant directly into symm buffer (writes nb*block_size elements).
    written = nb * block_size
    if written < padded:
        # zero region between written and padded just in case
        buf[written:padded].zero_()

    # Use a slice view as the destination
    ext.launch_block_int8_qd_bf16(x_bf16, buf, n, int(block_size), int(nb))

    # Barrier across ranks before multimem reduction (writes must be visible).
    hdl.barrier(channel=0)

    inv_ws = 1.0 / float(world_size)

    use_multimem = (padded % 8 == 0) and hasattr(hdl, "multicast_ptr")
    if use_multimem:
        try:
            multicast_ptr = int(hdl.multicast_ptr)
            if multicast_ptr == 0:
                use_multimem = False
        except Exception:
            use_multimem = False

    if use_multimem:
        numel_128 = padded // 8
        num_blocks, block_sz, block_stride = _multimem_launch_config(padded, hdl.world_size)
        ext.launch_multimem_allreduce_avg_bf16(
            int(hdl.multicast_ptr),
            hdl.signal_pad_ptrs_dev,
            numel_128,
            hdl.world_size,
            hdl.rank,
            num_blocks,
            block_sz,
            block_stride,
            inv_ws,
        )
        # After multimem, buf holds averaged result on all ranks.
        hdl.barrier(channel=1)
        result_bf16 = buf[:n]
    else:
        # Fallback peer-pointer reduction
        out = torch.empty(padded, device=device, dtype=torch.bfloat16)
        ext.launch_allreduce_avg_bf16(ptrs_tensor, out, padded, inv_ws)
        hdl.barrier(channel=1)
        result_bf16 = out[:n]

    if orig_dtype != torch.bfloat16:
        return result_bf16.to(orig_dtype).reshape(orig_shape)
    return result_bf16.clone().reshape(orig_shape)


__all__ = ["solution"]