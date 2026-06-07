from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
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

// Multimem all-reduce on bf16 in 128-bit chunks
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
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// Peer-pointer fallback all-reduce for TP (bf16)
__global__ void allreduce_peer_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t n)
{
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

// Copy from peer symm buffers into a contiguous gathered tensor.
// Layout for "rows": peer p holds rows [p*rows_per : (p+1)*rows_per] of width W.
// Layout for "cols": peer p holds cols [p*cols_per : (p+1)*cols_per] of height H, width cols_per.
//   Output is [H, W_total] with W_total = cols_per * world.

__global__ void gather_rows_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t rows_per,
    int64_t cols)
{
    // out shape [world*rows_per, cols], peer p contributes rows [p*rows_per..]
    int64_t total = (int64_t)world_size * rows_per * cols;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t per_peer = rows_per * cols;
    for (; idx < total; idx += stride) {
        int64_t p = idx / per_peer;
        int64_t off = idx - p * per_peer;
        const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[p];
        out[idx] = src[off];
    }
}

__global__ void gather_cols_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t H,
    int64_t cols_per)
{
    // out shape [H, world*cols_per]
    int64_t W = (int64_t)world_size * cols_per;
    int64_t total = H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int64_t row = idx / W;
        int64_t col = idx - row * W;
        int64_t p = col / cols_per;
        int64_t cc = col - p * cols_per;
        const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[p];
        out[idx] = src[row * cols_per + cc];
    }
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride)
{
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel, world_size, rank, block_stride);
}

void launch_allreduce_peer_bf16(
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
    allreduce_peer_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
}

void launch_gather_rows_bf16(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t rows_per,
    int64_t cols)
{
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int64_t total = (int64_t)world_size * rows_per * cols;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_rows_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        world_size, rows_per, cols);
}

void launch_gather_cols_bf16(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t H,
    int64_t cols_per)
{
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int64_t W = (int64_t)world_size * cols_per;
    int64_t total = H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_cols_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        world_size, H, cols_per);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce_peer_bf16", &launch_allreduce_peer_bf16);
    m.def("launch_gather_rows_bf16", &launch_gather_rows_bf16);
    m.def("launch_gather_cols_bf16", &launch_gather_cols_bf16);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_tp_cuda_ext", CUDA_SRC)
    return _ext


# ---------------------- caches ----------------------

_groups_cache = {}  # (n_tp, n_fsdp) -> (tp_group, fsdp_group, tp_ranks, fsdp_ranks)
_symm_cache = {}    # key -> dict of buffers/handles


def _make_groups(n_tp: int, n_fsdp: int):
    key = (n_tp, n_fsdp)
    if key in _groups_cache:
        return _groups_cache[key]
    rank = dist.get_rank()
    tp_group = None
    fsdp_group = None
    my_tp_ranks = None
    my_fsdp_ranks = None
    # TP groups: for each j (fsdp index), ranks {j*n_tp + i : i}
    for j in range(n_fsdp):
        ranks = [j * n_tp + ii for ii in range(n_tp)]
        g = dist.new_group(ranks)
        if rank in ranks:
            tp_group = g
            my_tp_ranks = ranks
    # FSDP groups: for each i (tp index), ranks {j*n_tp + i : j}
    for i in range(n_tp):
        ranks = [jj * n_tp + i for jj in range(n_fsdp)]
        g = dist.new_group(ranks)
        if rank in ranks:
            fsdp_group = g
            my_fsdp_ranks = ranks
    res = (tp_group, fsdp_group, my_tp_ranks, my_fsdp_ranks)
    _groups_cache[key] = res
    return res


def _get_symm_buf(name: str, shape, dtype, device, group):
    """Get or create a symmetric memory buffer for the given group."""
    key = (name, tuple(shape), dtype, device, id(group))
    entry = _symm_cache.get(key)
    if entry is not None:
        return entry
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    entry = (buf, hdl, ptrs_tensor)
    _symm_cache[key] = entry
    return entry


WARP_SIZE = 32
MAX_NUM_BLOCKS = 8
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
        num_blocks = min(
            (num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE,
            MAX_NUM_BLOCKS,
        )
    return num_blocks, block_size, block_size


@torch.no_grad()
def solution(
    x_local: Tensor,
    W1_shard: Tensor,
    W2_shard: Tensor,
    W3_shard: Tensor,
    n_tp: int,
    n_fsdp: int,
) -> Tensor:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size == n_tp * n_fsdp
    device = x_local.device

    ext = _get_ext()
    tp_group, fsdp_group, _, _ = _make_groups(n_tp, n_fsdp)

    # ---- FSDP all-gather W1, W2 (concat along dim 0), W3 (concat along dim 1) via symm_mem ----
    # W1_shard: [D/N_FSDP, D_FF/N_TP] -> gathered [D, D_FF/N_TP]
    # W2_shard: same
    # W3_shard: [D_FF/N_TP, D/N_FSDP] -> gathered [D_FF/N_TP, D]
    W1_shard_c = W1_shard.contiguous()
    W2_shard_c = W2_shard.contiguous()
    W3_shard_c = W3_shard.contiguous()

    rows_per_w1, cols_w1 = W1_shard_c.shape
    rows_per_w2, cols_w2 = W2_shard_c.shape
    H_w3, cols_per_w3 = W3_shard_c.shape

    buf_w1, hdl_w1, ptrs_w1 = _get_symm_buf("w1", W1_shard_c.shape, W1_shard_c.dtype, device, fsdp_group)
    buf_w2, hdl_w2, ptrs_w2 = _get_symm_buf("w2", W2_shard_c.shape, W2_shard_c.dtype, device, fsdp_group)
    buf_w3, hdl_w3, ptrs_w3 = _get_symm_buf("w3", W3_shard_c.shape, W3_shard_c.dtype, device, fsdp_group)

    buf_w1.copy_(W1_shard_c)
    buf_w2.copy_(W2_shard_c)
    buf_w3.copy_(W3_shard_c)

    # Barrier across FSDP group so all peers have published shards
    hdl_w1.barrier(channel=0)
    hdl_w2.barrier(channel=1)
    hdl_w3.barrier(channel=2)

    W1 = torch.empty((n_fsdp * rows_per_w1, cols_w1), dtype=W1_shard_c.dtype, device=device)
    W2 = torch.empty((n_fsdp * rows_per_w2, cols_w2), dtype=W2_shard_c.dtype, device=device)
    W3 = torch.empty((H_w3, n_fsdp * cols_per_w3), dtype=W3_shard_c.dtype, device=device)

    ext.launch_gather_rows_bf16(ptrs_w1, W1, rows_per_w1, cols_w1)
    ext.launch_gather_rows_bf16(ptrs_w2, W2, rows_per_w2, cols_w2)
    ext.launch_gather_cols_bf16(ptrs_w3, W3, H_w3, cols_per_w3)

    # ---- Local SwiGLU MLP ----
    x1 = x_local @ W1
    x2 = x_local @ W2
    z = F.silu(x1) * x2
    y_partial = z @ W3  # [B/N_FSDP, D]

    # ---- TP all-reduce SUM via symm_mem ----
    y_partial = y_partial.contiguous()
    n = y_partial.numel()
    dtype = y_partial.dtype

    buf_y, hdl_y, ptrs_y = _get_symm_buf("y", y_partial.shape, dtype, device, tp_group)
    buf_y.copy_(y_partial)

    if dtype == torch.bfloat16 and (n % (BYTES_PER_THREAD // 2) == 0) and hasattr(hdl_y, "multicast_ptr"):
        try:
            multicast_ptr = int(hdl_y.multicast_ptr)
            have_multicast = multicast_ptr != 0
        except Exception:
            have_multicast = False

        if have_multicast:
            numel_per_thread = BYTES_PER_THREAD // 2
            numel_128 = n // numel_per_thread
            num_blocks, block_size, block_stride = _multimem_launch_config(n, hdl_y.world_size)

            hdl_y.barrier(channel=3)
            ext.launch_multimem_allreduce_bf16(
                multicast_ptr,
                hdl_y.signal_pad_ptrs_dev,
                numel_128,
                hdl_y.world_size,
                hdl_y.rank,
                num_blocks,
                block_size,
                block_stride,
            )
            return buf_y.reshape_as(y_partial).clone()

    # Fallback peer-pointer reduction
    hdl_y.barrier(channel=3)
    out = torch.empty_like(y_partial)
    ext.launch_allreduce_peer_bf16(ptrs_y, out, n)
    return out


__all__ = ["solution"]