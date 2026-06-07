"""
DDP training step using symmetric memory + custom CUDA kernels.
- Param/moment broadcast: skipped (assume already identical across ranks since rank 0 is authoritative;
  we copy rank 0's values via symm_mem broadcast in one fused kernel).
- Gradient all-reduce: multimem.ld_reduce on bf16 via NVSwitch, fused with /world_size.
- Forward/backward kept in PyTorch (uses cuBLAS tensor cores already).
- Adam step: fused custom CUDA kernel.
"""

from __future__ import annotations

import math
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---- signal pad barrier ----
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
    const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size)
{
    unsigned int t = threadIdx.x;
    if (t >= (unsigned)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[t];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}
__device__ void blockwise_barrier_acq_rel(
    const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size)
{
    unsigned int t = threadIdx.x;
    if (t >= (unsigned)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[t];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(addr) : "memory");
}
__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

// All-reduce SUM bf16 via multimem (in-place on symmetric buffer)
__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank,
    int block_stride)
{
    const uint64_t block_id = blockIdx.x;
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank = (numel_128 + world_size - 1) / world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t bs = (int64_t)block_id * block_stride;
         bs < numel_per_rank;
         bs += (int64_t)num_programs * block_stride)
    {
        const int64_t off = bs + tid;
        if (off >= numel_per_rank) continue;
        const int64_t idx = (int64_t)rank * numel_per_rank + off;
        uint64_t* p = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(p, x, y, z, w);
        multimem_st_bf16x4(p, x, y, z, w);
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// Fused Adam (bf16 params, bf16 moments, bf16 grad) with /world_size built into grad
__global__ void fused_adam_bf16_kernel(
    __nv_bfloat16* __restrict__ p,
    __nv_bfloat16* __restrict__ m,
    __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ g,
    float inv_world,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2,
    float lr,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float gv = __bfloat162float(g[idx]) * inv_world;
        float mv = __bfloat162float(m[idx]);
        float vv = __bfloat162float(v[idx]);
        mv = beta1 * mv + (1.0f - beta1) * gv;
        vv = beta2 * vv + (1.0f - beta2) * gv * gv;
        float m_hat = mv / bc1;
        float v_hat = vv / bc2;
        float denom = sqrtf(v_hat) + eps;
        float pv = __bfloat162float(p[idx]);
        pv -= lr * m_hat / denom;
        p[idx] = __float2bfloat16(pv);
        m[idx] = __float2bfloat16(mv);
        v[idx] = __float2bfloat16(vv);
    }
}

// Broadcast: rank 0 writes its data to symmetric buffer; multimem.st replicates to all peers.
// Simpler: just copy from symmetric buffer to local on each rank after barrier.
// We use a plain copy kernel for non-rank-0 to read from rank0's UVA pointer.
__global__ void copy_from_peer_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        dst[idx] = src[idx];
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

void launch_fused_adam_bf16(
    torch::Tensor p, torch::Tensor m, torch::Tensor v, torch::Tensor g,
    double inv_world, double beta1, double beta2, double eps,
    double bc1, double bc2, double lr)
{
    int64_t n = p.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 2048) blocks = 2048;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_adam_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)p.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)m.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)v.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)g.data_ptr<at::BFloat16>(),
        (float)inv_world, (float)beta1, (float)beta2, (float)eps,
        (float)bc1, (float)bc2, (float)lr, n);
}

void launch_copy_from_peer_bf16(
    torch::Tensor dst, int64_t src_ptr, int64_t n)
{
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 2048) blocks = 2048;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    copy_from_peer_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)dst.data_ptr<at::BFloat16>(),
        reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(src_ptr)),
        n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_fused_adam_bf16", &launch_fused_adam_bf16);
    m.def("launch_copy_from_peer_bf16", &launch_copy_from_peer_bf16);
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ddp_symm_ext_v1", CUDA_SRC)
    return _ext


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
        num_blocks = min((num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE, MAX_NUM_BLOCKS)
    return num_blocks, max(block_size, 1), max(block_size, 1)


_grad_buf_cache = {}
_param_buf_cache = {}


def _get_grad_buf(numel, dtype, device):
    key = (numel, dtype, device)
    if key in _grad_buf_cache:
        return _grad_buf_cache[key]
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _grad_buf_cache[key] = (buf, hdl)
    return buf, hdl


def _get_param_buf(numel, dtype, device):
    key = (numel, dtype, device)
    if key in _param_buf_cache:
        return _param_buf_cache[key]
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _param_buf_cache[key] = (buf, hdl)
    return buf, hdl


def _broadcast_via_symm(tensors, device):
    """Broadcast list of tensors from rank 0 to all via symmetric memory + UVA copy."""
    flat = _flatten_dense_tensors(tensors)
    n = flat.numel()
    buf, hdl = _get_param_buf(n, flat.dtype, device)
    rank = dist.get_rank()
    if rank == 0:
        buf.copy_(flat)
    hdl.barrier(channel=0)
    if rank != 0:
        # Copy from rank 0's UVA pointer
        peer_ptr = int(hdl.buffer_ptrs[0])
        _get_ext().launch_copy_from_peer_bf16(buf, peer_ptr, n)
    hdl.barrier(channel=1)
    out_flat = buf[:n].clone()
    return list(_unflatten_dense_tensors(out_flat, tensors))


@torch.no_grad()
def _do_allreduce_mean(flat_grad, world_size):
    """In-place all-reduce SUM via multimem on bf16, then we'll fold /world into Adam."""
    n = flat_grad.numel()
    device = flat_grad.device
    buf, hdl = _get_grad_buf(n, flat_grad.dtype, device)
    buf.copy_(flat_grad)

    numel_per_thread = BYTES_PER_THREAD // flat_grad.element_size()
    if flat_grad.dtype == torch.bfloat16 and (n % numel_per_thread == 0):
        numel_128 = n // numel_per_thread
        num_blocks, block_size, block_stride = _multimem_launch_config(n, hdl.world_size)
        # ensure all ranks finished writing buf
        hdl.barrier(channel=0)
        multicast_ptr = int(hdl.multicast_ptr)
        signal_dev = hdl.signal_pad_ptrs_dev
        _get_ext().launch_multimem_allreduce_bf16(
            multicast_ptr, signal_dev, numel_128,
            hdl.world_size, hdl.rank,
            num_blocks, block_size, block_stride,
        )
        hdl.barrier(channel=1)
        flat_grad.copy_(buf)
    else:
        # Fallback to dist.all_reduce
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)


def solution(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_W1: Tensor,
    exp_avg_b1: Tensor,
    exp_avg_W2: Tensor,
    exp_avg_b2: Tensor,
    exp_avg_sq_W1: Tensor,
    exp_avg_sq_b1: Tensor,
    exp_avg_sq_W2: Tensor,
    exp_avg_sq_b2: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, ...]:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    device = X_local.device

    # Ensure ext compiled (rank 0 first to avoid races)
    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()
    _get_ext()

    params_in = [W1, b1, W2, b2]
    m_in = [exp_avg_W1, exp_avg_b1, exp_avg_W2, exp_avg_b2]
    v_in = [exp_avg_sq_W1, exp_avg_sq_b1, exp_avg_sq_W2, exp_avg_sq_b2]

    # Broadcast params + moments from rank 0 (single combined flatten for fewer barriers)
    bcast_list = params_in + m_in + v_in
    bcast_out = _broadcast_via_symm(bcast_list, device)
    params = [t.detach().requires_grad_(True) for t in bcast_out[:4]]
    exp_avg = list(bcast_out[4:8])
    exp_avg_sq = list(bcast_out[8:12])

    # Forward / backward (cuBLAS handles tensor cores for bf16 matmul)
    with torch.enable_grad():
        h = F.relu(F.linear(X_local, params[0], params[1]))
        out = F.linear(h, params[2], params[3])
        loss = F.mse_loss(out, y_local)
        loss.backward()

    grads = [p.grad for p in params]
    flat_grad = _flatten_dense_tensors(grads).contiguous()

    # Custom multimem all-reduce
    _do_allreduce_mean(flat_grad, world_size)

    # Fused Adam with /world_size baked in
    inv_world = 1.0 / world_size
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    avg_grads = list(_unflatten_dense_tensors(flat_grad, grads))

    ext = _get_ext()
    out_params = []
    for p, m_buf, v_buf, g in zip(params, exp_avg, exp_avg_sq, avg_grads):
        p_data = p.data.contiguous()
        m_c = m_buf.contiguous()
        v_c = v_buf.contiguous()
        g_c = g.contiguous()
        ext.launch_fused_adam_bf16(
            p_data, m_c, v_c, g_c,
            inv_world, beta1, beta2, eps, bc1, bc2, lr,
        )
        out_params.append(p_data)
        # write back to m_buf, v_buf views
        if m_buf.data_ptr() != m_c.data_ptr():
            m_buf.copy_(m_c)
        if v_buf.data_ptr() != v_c.data_ptr():
            v_buf.copy_(v_c)

    return tuple(out_params + exp_avg + exp_avg_sq)


__all__ = ["solution"]