"""
Optimized implementation of Fused Reduce-Scatter + RMSNorm using symmetric memory and custom CUDA kernels.

Strategy:
- **Device-Side Communication**: Bypasses NCCL collective overhead by caching `rs_input_1d` in symmetric memory, enabling direct one-shot NVLink reads from all peers.
- **Kernel Fusion**: Fuses the reduction over all ranks, division by `world_size`, and RMSNorm into a single device kernel. This perfectly hides the latency of individual operations and drops intermediate HBM writes.
- **Compute-Communication Overlap**: Each block processes one row. It issues vectorized `uint4` (16 bytes = 8x bfloat16) reads across NVLink from all peers, computes the partial block sum for the RMSNorm stats, and applies the final normalization via L1/L2 cache, maximizing overlap between peer loads and dense scalar math.
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
#include <cub/cub.cuh>
#include <vector>

#define MAX_RANKS 16

struct PtrArray {
    const void* ptrs[MAX_RANKS];
};

template <int BLOCK_THREADS>
__global__ void fused_rs_rmsnorm_kernel_vec8(
    PtrArray ptrs,
    uint4* __restrict__ out,
    const uint4* __restrict__ gamma,
    float eps,
    int64_t rows,
    int64_t hidden_vecs,
    int64_t chunk_start_vec,
    int world_size,
    int64_t hidden
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    int64_t row_offset = chunk_start_vec + row * hidden_vecs;
    int64_t out_row_offset = row * hidden_vecs;

    float sum_sq = 0.0f;

    // Pass 1: Reduce across ranks, compute local squared sums, and temporarily store reduced output in HBM (L2 Cache)
    for (int64_t i = tid; i < hidden_vecs; i += stride) {
        float sums[8] = {0};
        for (int r = 0; r < world_size; r++) {
            const uint4* rank_ptr = (const uint4*)ptrs.ptrs[r];
            uint4 val4 = rank_ptr[row_offset + i];
            nv_bfloat16* vals = (nv_bfloat16*)&val4;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                sums[k] += __bfloat162float(vals[k]);
            }
        }
        
        uint4 out_val4;
        nv_bfloat16* out_vals = (nv_bfloat16*)&out_val4;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // Emulate the stock operation: float division -> bf16 intermediate -> float for sum_sq
            nv_bfloat16 reduced = __float2bfloat16(sums[k] / world_size);
            out_vals[k] = reduced;
            float x = __bfloat162float(reduced);
            sum_sq += x * x;
        }
        out[out_row_offset + i] = out_val4;
    }

    // Block-wide sum for RMSNorm variance
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);

    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(block_sum_sq / hidden + eps);
    }
    __syncthreads();

    float rms = s_rms;

    // Pass 2: Apply RMSNorm and weight vector
    for (int64_t i = tid; i < hidden_vecs; i += stride) {
        uint4 in_val4 = out[out_row_offset + i];
        uint4 gamma_val4 = gamma[i];
        
        nv_bfloat16* in_vals = (nv_bfloat16*)&in_val4;
        nv_bfloat16* gamma_vals = (nv_bfloat16*)&gamma_val4;
        
        uint4 out_val4;
        nv_bfloat16* out_vals = (nv_bfloat16*)&out_val4;
        
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float x = __bfloat162float(in_vals[k]);
            float g = __bfloat162float(gamma_vals[k]);
            out_vals[k] = __float2bfloat16(x * rms * g);
        }
        
        out[out_row_offset + i] = out_val4;
    }
}

template <int BLOCK_THREADS>
__global__ void fused_rs_rmsnorm_kernel_scalar(
    PtrArray ptrs,
    nv_bfloat16* __restrict__ out,
    const nv_bfloat16* __restrict__ gamma,
    float eps,
    int64_t rows,
    int64_t hidden,
    int64_t chunk_start_idx,
    int world_size
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    int64_t row_offset = chunk_start_idx + row * hidden;
    int64_t out_row_offset = row * hidden;

    float sum_sq = 0.0f;

    for (int64_t i = tid; i < hidden; i += stride) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; r++) {
            const nv_bfloat16* rank_ptr = (const nv_bfloat16*)ptrs.ptrs[r];
            sum += __bfloat162float(rank_ptr[row_offset + i]);
        }
        
        nv_bfloat16 reduced = __float2bfloat16(sum / world_size);
        out[out_row_offset + i] = reduced;
        float x = __bfloat162float(reduced);
        sum_sq += x * x;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum_sq = BlockReduce(temp_storage).Sum(sum_sq);

    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(block_sum_sq / hidden + eps);
    }
    __syncthreads();

    float rms = s_rms;

    for (int64_t i = tid; i < hidden; i += stride) {
        float x = __bfloat162float(out[out_row_offset + i]);
        float g = __bfloat162float(gamma[i]);
        out[out_row_offset + i] = __float2bfloat16(x * rms * g);
    }
}

void fused_rs_rmsnorm_cuda(
    std::vector<int64_t> ptr_list,
    torch::Tensor out,
    torch::Tensor gamma,
    float eps,
    int64_t chunk_start_idx,
    int world_size
) {
    TORCH_CHECK(world_size <= MAX_RANKS, "world_size exceeds maximum supported ranks");
    
    int64_t rows = out.size(0);
    int64_t hidden = out.size(1);

    PtrArray ptrs;
    bool all_aligned = true;
    for (int i = 0; i < world_size; i++) {
        ptrs.ptrs[i] = reinterpret_cast<const void*>(ptr_list[i]);
        if (reinterpret_cast<uintptr_t>(ptrs.ptrs[i]) % 16 != 0) all_aligned = false;
    }

    bool out_aligned = reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0;
    bool gamma_aligned = reinterpret_cast<uintptr_t>(gamma.data_ptr()) % 16 == 0;

    int threads = 512;
    if (hidden <= 1024) threads = 128;
    else if (hidden <= 2048) threads = 256;
    else if (hidden <= 4096) threads = 512;
    else threads = 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_VEC8(THREADS) \
        fused_rs_rmsnorm_kernel_vec8<THREADS><<<rows, THREADS, 0, stream>>>( \
            ptrs, \
            reinterpret_cast<uint4*>(out.data_ptr<at::BFloat16>()), \
            reinterpret_cast<const uint4*>(gamma.data_ptr<at::BFloat16>()), \
            eps, \
            rows, \
            hidden_vecs, \
            chunk_start_vec, \
            world_size, \
            hidden \
        )

    #define LAUNCH_SCALAR(THREADS) \
        fused_rs_rmsnorm_kernel_scalar<THREADS><<<rows, THREADS, 0, stream>>>( \
            ptrs, \
            reinterpret_cast<nv_bfloat16*>(out.data_ptr<at::BFloat16>()), \
            reinterpret_cast<const nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()), \
            eps, \
            rows, \
            hidden, \
            chunk_start_idx, \
            world_size \
        )

    if (hidden % 8 == 0 && all_aligned && out_aligned && gamma_aligned) {
        int64_t hidden_vecs = hidden / 8;
        int64_t chunk_start_vec = chunk_start_idx / 8;
        if (threads == 128) { LAUNCH_VEC8(128); }
        else if (threads == 256) { LAUNCH_VEC8(256); }
        else if (threads == 512) { LAUNCH_VEC8(512); }
        else { LAUNCH_VEC8(1024); }
    } else {
        if (threads == 128) { LAUNCH_SCALAR(128); }
        else if (threads == 256) { LAUNCH_SCALAR(256); }
        else if (threads == 512) { LAUNCH_SCALAR(512); }
        else { LAUNCH_SCALAR(1024); }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rs_rmsnorm", &fused_rs_rmsnorm_cuda, "Fused Symmetric ReduceScatter and RMSNorm");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_rs_rmsnorm_ext", CUDA_SRC)
    return _ext

_symm_cache = None

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache = {"n": n, "dtype": dtype, "device": device, "buf": buf, "hdl": hdl}
    return buf, hdl

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
    chunk = n // world_size
    hidden = gamma.numel()
    rows = chunk // hidden

    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    # Grab symmetric memory handle
    buf, hdl = _get_symm_state(n, rs_input_1d.dtype, rs_input_1d.device)
    
    # Load contiguous target buffer to Symmetric memory space
    buf.copy_(rs_input_1d.contiguous())
    
    # Barrier: Wait for all ranks to complete uploading to their symmetric buffer
    hdl.barrier(channel=0)

    # Allocate final normalized tensor buffer
    out = torch.empty((rows, hidden), dtype=rs_input_1d.dtype, device=rs_input_1d.device)

    # Extract peer device pointers
    ptrs = [int(p) for p in hdl.buffer_ptrs]
    chunk_start_idx = rank * chunk

    ext.fused_rs_rmsnorm(
        ptrs,
        out,
        gamma,
        eps,
        chunk_start_idx,
        world_size
    )

    # Barrier: Prevent looping overwrites to `buf` in training context before peers finish loads
    hdl.barrier(channel=0)

    return out

__all__ = ["solution"]