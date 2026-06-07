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

__device__ void global_barrier_relaxed(
    const uint64_t* signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

__device__ void global_barrier_acq_rel(
    const uint64_t* signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr) : "memory");
}

// Fused RS-multimem + RMSNorm.
// Each block handles one row of the rank's chunk.
// chunk_base_in_buf: byte offset in symmetric buffer where this rank's chunk starts (in elements)
// hidden must be multiple of 8 (bf16x8 = 16 bytes)
extern "C" __global__ void fused_rs_rmsnorm_bf16_kernel(
    uint64_t multicast_base_ptr,        // multicast pointer to buf data
    const uint64_t* __restrict__ signal_pad_ptrs,
    __nv_bfloat16* __restrict__ out,    // [rows, hidden]
    const __nv_bfloat16* __restrict__ gamma, // [hidden]
    int rows,
    int hidden,
    int64_t chunk_offset_elems,         // rank * chunk in elements
    float inv_world,
    float eps,
    int world_size,
    int rank,
    int barrier_blocks                   // number of leading blocks doing barrier
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // Initial barrier: all blocks participate (using block_id = blockIdx.x)
    global_barrier_relaxed(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
    __syncthreads();

    if (row >= rows) {
        // still must do the trailing barrier
        __syncthreads();
        global_barrier_acq_rel(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
        return;
    }

    // Pointer to row in multicast buffer (elements -> 16-byte chunks of 8 bf16)
    const int64_t row_elem_offset = chunk_offset_elems + (int64_t)row * (int64_t)hidden;
    // Each 16-byte vector = 8 bf16 elements
    const int vec_per_row = hidden / 8;
    const uint64_t* mc_row_vec = reinterpret_cast<const uint64_t*>(
        multicast_base_ptr + row_elem_offset * sizeof(__nv_bfloat16));

    // Output and gamma
    __nv_bfloat16* out_row = out + (int64_t)row * (int64_t)hidden;

    // Pass 1: load via multimem reduce, compute sum of squares, store reduced bf16 into shared (or registers)
    // Use a temporary local buffer in shared memory of size hidden bf16 values.
    extern __shared__ __nv_bfloat16 smem_x[];

    float local_sumsq = 0.0f;

    // Process 8 bf16 (= 4 bf16x2 = 4 uint32) per vector
    for (int v = tid; v < vec_per_row; v += bdim) {
        uint32_t r0, r1, r2, r3;
        multimem_ld_reduce_bf16x4(mc_row_vec + (int64_t)v * 2, r0, r1, r2, r3);
        // Each rN packs two bf16 values
        uint32_t rs[4] = {r0, r1, r2, r3};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __nv_bfloat162 b2 = *reinterpret_cast<__nv_bfloat162*>(&rs[k]);
            float a = __bfloat162float(b2.x);
            float b = __bfloat162float(b2.y);
            // multiply by inv_world
            a *= inv_world;
            b *= inv_world;
            local_sumsq += a * a + b * b;
            // store back as bf16
            int idx = v * 8 + k * 2;
            smem_x[idx] = __float2bfloat16(a);
            smem_x[idx + 1] = __float2bfloat16(b);
        }
    }

    // Block reduction of local_sumsq
    __shared__ float ssum[32];
    // warp reduce
    unsigned mask = 0xffffffff;
    float v = local_sumsq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(mask, v, offset);
    }
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) ssum[warp_id] = v;
    __syncthreads();
    int num_warps = (bdim + 31) / 32;
    float total = 0.0f;
    if (warp_id == 0) {
        v = (tid < num_warps) ? ssum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_xor_sync(mask, v, offset);
        }
        if (lane == 0) ssum[0] = v;
    }
    __syncthreads();
    total = ssum[0];

    float mean_sq = total / (float)hidden;
    float rrms = rsqrtf(mean_sq + eps);

    // Pass 2: write out = x * rrms * gamma
    for (int i = tid; i < hidden; i += bdim) {
        float x = __bfloat162float(smem_x[i]);
        float g = __bfloat162float(gamma[i]);
        float y = x * rrms * g;
        out_row[i] = __float2bfloat16(y);
    }

    __syncthreads();
    global_barrier_acq_rel(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
}

void launch_fused_rs_rmsnorm_bf16(
    uint64_t multicast_base_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    torch::Tensor out,
    torch::Tensor gamma,
    int64_t rows,
    int64_t hidden,
    int64_t chunk_offset_elems,
    double inv_world,
    double eps,
    int64_t world_size,
    int64_t rank
) {
    TORCH_CHECK(out.is_cuda() && gamma.is_cuda());
    TORCH_CHECK(out.dtype() == torch::kBFloat16);
    TORCH_CHECK(gamma.dtype() == torch::kBFloat16);
    TORCH_CHECK(hidden % 8 == 0, "hidden must be multiple of 8");

    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());

    int block_size = 256;
    if (hidden < 256) {
        block_size = 128;
    }
    if (hidden >= 2048) block_size = 512;
    // Ensure block_size >= world_size for barrier
    if (block_size < (int)world_size) block_size = (int)world_size;

    int grid = (int)rows;
    if (grid < (int)world_size) grid = (int)world_size; // ensure barrier validity

    size_t smem = hidden * sizeof(__nv_bfloat16);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_rs_rmsnorm_bf16_kernel<<<grid, block_size, smem, stream>>>(
        multicast_base_ptr,
        d_signal,
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)gamma.data_ptr<at::BFloat16>(),
        (int)rows,
        (int)hidden,
        (int64_t)chunk_offset_elems,
        (float)inv_world,
        (float)eps,
        (int)world_size,
        (int)rank,
        grid
    );
}

// Fallback: peer-pointer reduce-scatter + RMSNorm for non-bf16 or unaligned cases
__global__ void rs_rmsnorm_peer_bf16_kernel(
    const long long* __restrict__ peer_ptrs,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gamma,
    int rows,
    int hidden,
    int64_t chunk_offset_elems,
    int world_size,
    float inv_world,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ __nv_bfloat16 smem_x2[];

    int64_t row_elem_offset = chunk_offset_elems + (int64_t)row * (int64_t)hidden;
    float local_sumsq = 0.0f;

    for (int i = tid; i < hidden; i += bdim) {
        float s = 0.0f;
        #pragma unroll 1
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)peer_ptrs[r];
            s += __bfloat162float(src[row_elem_offset + i]);
        }
        s *= inv_world;
        smem_x2[i] = __float2bfloat16(s);
        local_sumsq += s * s;
    }

    __shared__ float ssum[32];
    unsigned mask = 0xffffffff;
    float v = local_sumsq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_xor_sync(mask, v, offset);
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) ssum[warp_id] = v;
    __syncthreads();
    int num_warps = (bdim + 31) / 32;
    if (warp_id == 0) {
        v = (tid < num_warps) ? ssum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            v += __shfl_xor_sync(mask, v, offset);
        if (lane == 0) ssum[0] = v;
    }
    __syncthreads();
    float total = ssum[0];
    float rrms = rsqrtf(total / (float)hidden + eps);

    __nv_bfloat16* out_row = out + (int64_t)row * (int64_t)hidden;
    for (int i = tid; i < hidden; i += bdim) {
        float x = __bfloat162float(smem_x2[i]);
        float g = __bfloat162float(gamma[i]);
        out_row[i] = __float2bfloat16(x * rrms * g);
    }
}

void launch_rs_rmsnorm_peer_bf16(
    torch::Tensor peer_ptrs,
    torch::Tensor out,
    torch::Tensor gamma,
    int64_t rows,
    int64_t hidden,
    int64_t chunk_offset_elems,
    int64_t world_size,
    double inv_world,
    double eps
) {
    const long long* d_ptrs = (const long long*)peer_ptrs.data_ptr<int64_t>();
    int block_size = 256;
    if (hidden >= 2048) block_size = 512;
    int grid = (int)rows;
    size_t smem = hidden * sizeof(__nv_bfloat16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    rs_rmsnorm_peer_bf16_kernel<<<grid, block_size, smem, stream>>>(
        d_ptrs,
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)gamma.data_ptr<at::BFloat16>(),
        (int)rows, (int)hidden, chunk_offset_elems, (int)world_size,
        (float)inv_world, (float)eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_rs_rmsnorm_bf16", &launch_fused_rs_rmsnorm_bf16,
          "Fused multimem RS + RMSNorm (bf16)");
    m.def("launch_rs_rmsnorm_peer_bf16", &launch_rs_rmsnorm_peer_bf16,
          "Peer-pointer RS + RMSNorm (bf16)");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_rs_rmsnorm_ext", CUDA_SRC)
    return _ext


_buf_cache = {}


def _get_symm_buf(numel: int, dtype: torch.dtype, device: torch.device):
    key = (numel, dtype, device)
    if key in _buf_cache:
        return _buf_cache[key]
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _buf_cache[key] = (buf, hdl, ptrs_tensor)
    return _buf_cache[key]


@torch.no_grad()
def solution(
    rs_input_1d: Tensor,
    gamma: Tensor,
    eps: float,
) -> Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n = rs_input_1d.numel()
    assert n % world_size == 0
    chunk = n // world_size
    hidden = gamma.numel()
    assert chunk % hidden == 0
    rows = chunk // hidden

    device = rs_input_1d.device
    dtype = rs_input_1d.dtype

    out = torch.empty((rows, hidden), dtype=dtype, device=device)

    ext = _get_ext()

    # BF16 multimem fast path
    if dtype == torch.bfloat16 and hidden % 8 == 0:
        buf, hdl, ptrs_tensor = _get_symm_buf(n, dtype, device)
        buf.copy_(rs_input_1d.contiguous())

        # Make symm buffer writes visible across peers before multimem load_reduce
        dist.barrier()

        chunk_offset_elems = rank * chunk
        ext.launch_fused_rs_rmsnorm_bf16(
            int(hdl.multicast_ptr),
            hdl.signal_pad_ptrs_dev,
            out,
            gamma.contiguous(),
            rows,
            hidden,
            chunk_offset_elems,
            1.0 / world_size,
            float(eps),
            world_size,
            rank,
        )
        return out

    # Fallback: peer-pointer path (still custom CUDA, no NCCL)
    if dtype == torch.bfloat16:
        buf, hdl, ptrs_tensor = _get_symm_buf(n, dtype, device)
        buf.copy_(rs_input_1d.contiguous())
        hdl.barrier(channel=0)
        chunk_offset_elems = rank * chunk
        ext.launch_rs_rmsnorm_peer_bf16(
            ptrs_tensor, out, gamma.contiguous(),
            rows, hidden, chunk_offset_elems, world_size,
            1.0 / world_size, float(eps),
        )
        return out

    # Generic dtype fallback (rare): use reference path
    out_flat = torch.empty(chunk, dtype=dtype, device=device)
    dist.reduce_scatter_tensor(out_flat, rs_input_1d.contiguous(), op=dist.ReduceOp.SUM)
    out_flat.div_(world_size)
    x = out_flat.view(rows, hidden).float()
    gn = gamma.float()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).add(eps))
    y = x * rms * gn
    return y.to(dtype=dtype)


__all__ = ["solution"]