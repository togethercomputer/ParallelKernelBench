"""
ZeRO-1 step using torch symmetric memory + custom CUDA kernels.
- Param broadcast: device-side memcpy from rank 0's symm_mem buffer (UVA).
- Grad all-reduce (SUM, /world_size): multimem.ld_reduce.add + multimem.st (bf16x2 v4).
- Fused Adam on local partition: custom bf16 CUDA kernel.
- All-gather of weight shards: each rank writes its partition to its slot in the
  symmetric flat buffer; barrier; all ranks then have the full replica.
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

// ---------------- signal-pad barrier ----------------
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
__device__ void blockwise_barrier_acq_rel(
    const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size)
{
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

// ---------------- multimem all-reduce (bf16x2 v4) with /world_size ----------------
__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr) : "memory");
}
__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__device__ __forceinline__ uint32_t scale_bf16x2(uint32_t packed, float scale) {
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&packed);
    float a = __bfloat162float(v.x) * scale;
    float b = __bfloat162float(v.y) * scale;
    __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    uint32_t out;
    *reinterpret_cast<__nv_bfloat162*>(&out) = r;
    return out;
}

__global__ void multimem_allreduce_scale_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank,
    int block_stride,
    float scale)
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
        const int64_t off = block_start + (int64_t)tid;
        if (off >= numel_per_rank) continue;
        const int64_t idx = (int64_t)rank * numel_per_rank + off;
        if (idx * 8 >= numel_128 * 8) continue; // bound check (in 16B units off numel_128)
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        x = scale_bf16x2(x, scale);
        y = scale_bf16x2(y, scale);
        z = scale_bf16x2(z, scale);
        w = scale_bf16x2(w, scale);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

void launch_multimem_allreduce_scale_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel_bf16,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride,
    double scale)
{
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t numel_128 = numel_bf16 / 8;
    multimem_allreduce_scale_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel_128, world_size, rank, block_stride, (float)scale);
}

// ---------------- fallback all-reduce (peer pointers), bf16 ----------------
__global__ void allreduce_scale_bf16_kernel(
    const long long* ptrs,
    __nv_bfloat16* out,
    int world_size,
    int64_t n,
    float scale)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float s = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            s += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(s * scale);
    }
}

void launch_allreduce_scale_bf16(
    torch::Tensor ptrs_tensor,
    torch::Tensor out_buf,
    int64_t n,
    double scale)
{
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_scale_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out_buf.data_ptr<at::BFloat16>(),
        world_size, n, (float)scale);
}

// ---------------- fused Adam (bf16 params/grads, fp32 moments unused; kept bf16 for moments) ----------------
__global__ void fused_adam_bf16_kernel(
    __nv_bfloat16* w_part,        // updated in-place
    const __nv_bfloat16* g_part,  // grad partition
    __nv_bfloat16* m_part,        // exp_avg partition (in/out)
    __nv_bfloat16* v_part,        // exp_avg_sq partition (in/out)
    int64_t n,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2,
    float lr)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float w = __bfloat162float(w_part[idx]);
        float g = __bfloat162float(g_part[idx]);
        float m = __bfloat162float(m_part[idx]);
        float v = __bfloat162float(v_part[idx]);
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float upd = m_hat / (sqrtf(v_hat) + eps);
        w = w - lr * upd;
        w_part[idx] = __float2bfloat16(w);
        m_part[idx] = __float2bfloat16(m);
        v_part[idx] = __float2bfloat16(v);
    }
}

void launch_fused_adam_bf16(
    torch::Tensor w_part,
    torch::Tensor g_part,
    torch::Tensor m_part,
    torch::Tensor v_part,
    double beta1, double beta2, double eps,
    double bc1, double bc2, double lr)
{
    int64_t n = w_part.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_adam_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)w_part.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)g_part.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)m_part.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)v_part.data_ptr<at::BFloat16>(),
        n, (float)beta1, (float)beta2, (float)eps,
        (float)bc1, (float)bc2, (float)lr);
}

// ---------------- device memcpy from a UVA source pointer ----------------
__global__ void memcpy_from_ptr_kernel(
    void* dst, const void* src, int64_t nbytes)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t n4 = nbytes / 16;
    const uint4* s4 = (const uint4*)src;
    uint4* d4 = (uint4*)dst;
    for (int64_t i = idx; i < n4; i += stride) {
        d4[i] = s4[i];
    }
    int64_t tail_start = n4 * 16;
    for (int64_t i = tail_start + idx; i < nbytes; i += stride) {
        ((char*)dst)[i] = ((const char*)src)[i];
    }
}

void launch_memcpy_from_ptr(
    torch::Tensor dst,
    int64_t src_ptr,
    int64_t nbytes)
{
    int threads = 256;
    int blocks = 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    memcpy_from_ptr_kernel<<<blocks, threads, 0, stream>>>(
        dst.data_ptr(), reinterpret_cast<const void*>((uintptr_t)src_ptr), nbytes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_scale_bf16", &launch_multimem_allreduce_scale_bf16);
    m.def("launch_allreduce_scale_bf16", &launch_allreduce_scale_bf16);
    m.def("launch_fused_adam_bf16", &launch_fused_adam_bf16);
    m.def("launch_memcpy_from_ptr", &launch_memcpy_from_ptr);
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("zero1_cuda_ext", CUDA_SRC)
    return _ext


_cache = {}

def _get_buffers(numel_padded: int, device: torch.device):
    key = (numel_padded, device)
    if key in _cache:
        return _cache[key]
    # Symmetric param buffer (also used for all-gather of partitions).
    param_buf = symm_mem.empty(numel_padded, device=device, dtype=torch.bfloat16)
    param_hdl = symm_mem.rendezvous(param_buf, dist.group.WORLD)
    # Symmetric grad buffer.
    grad_buf = symm_mem.empty(numel_padded, device=device, dtype=torch.bfloat16)
    grad_hdl = symm_mem.rendezvous(grad_buf, dist.group.WORLD)

    ptrs_param = torch.tensor(param_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    ptrs_grad = torch.tensor(grad_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (param_buf, param_hdl, grad_buf, grad_hdl, ptrs_param, ptrs_grad)
    _cache[key] = res
    return res


WARP_SIZE = 32
MAX_NUM_BLOCKS = 24
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16


def _multimem_launch_config(numel_bf16: int, world_size: int):
    numel_per_thread = BYTES_PER_THREAD // 2  # bf16 -> 8 elements per 16B
    num_threads = (numel_bf16 // numel_per_thread + world_size - 1) // world_size
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
    return num_blocks, max(block_size, 1), max(block_size, 1)


@torch.no_grad()
def _broadcast_from_rank0(param_buf: Tensor, param_hdl, rank: int):
    if rank == 0:
        return
    peer_ptr = int(param_hdl.buffer_ptrs[0])
    nbytes = param_buf.numel() * param_buf.element_size()
    _get_ext().launch_memcpy_from_ptr(param_buf, peer_ptr, nbytes)


def solution(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_part: Tensor,
    exp_avg_sq_part: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = W1.device

    templates = [W1, b1, W2, b2]
    flat_template = _flatten_dense_tensors(templates)
    numel = flat_template.numel()
    part = exp_avg_part.numel()
    assert numel == part * world_size

    ext = _get_ext()

    param_buf, param_hdl, grad_buf, grad_hdl, ptrs_param, ptrs_grad = _get_buffers(numel, device)

    # ---- 1) Broadcast params: rank 0 fills symm buffer; peers copy from it via UVA ----
    if rank == 0:
        param_buf.copy_(flat_template)
    # Barrier so non-zero ranks see rank 0's data.
    dist.barrier()
    if rank != 0:
        _broadcast_from_rank0(param_buf, param_hdl, rank)
        torch.cuda.synchronize()

    # Build param views from broadcast flat buffer (autograd-enabled leaves).
    param_views = _unflatten_dense_tensors(param_buf, templates)
    params = [t.detach().clone().requires_grad_(True) for t in param_views]

    # ---- 2) Forward + backward (stock PyTorch autograd; small MLP) ----
    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    # ---- 3) Flatten grads into symm grad buffer, then multimem all-reduce + scale ----
    grads = [p.grad for p in params]
    flat_g = _flatten_dense_tensors(grads)
    grad_buf.copy_(flat_g)

    dist.barrier()
    inv_ws = 1.0 / float(world_size)

    use_multimem = (numel % 8 == 0) and hasattr(grad_hdl, "multicast_ptr") and int(grad_hdl.multicast_ptr) != 0
    if use_multimem:
        nb, bs, bstride = _multimem_launch_config(numel, world_size)
        ext.launch_multimem_allreduce_scale_bf16(
            int(grad_hdl.multicast_ptr),
            grad_hdl.signal_pad_ptrs_dev,
            numel,
            world_size,
            rank,
            nb, bs, bstride,
            inv_ws,
        )
        # After multimem, each rank's local grad_buf holds the reduced+scaled values.
        flat_g_reduced = grad_buf
    else:
        out_g = torch.empty(numel, device=device, dtype=torch.bfloat16)
        ext.launch_allreduce_scale_bf16(ptrs_grad, out_g, numel, inv_ws)
        flat_g_reduced = out_g

    # ---- 4) Fused Adam on local partition (in-place on a partition slice of param_buf) ----
    start = rank * part
    g_part = flat_g_reduced.narrow(0, start, part).contiguous()

    # Update exp_avg / exp_avg_sq in-place on caller-provided tensors (return them).
    m_part = exp_avg_part.clone()
    v_part = exp_avg_sq_part.clone()

    # Work on a partition slice of the symmetric param buffer directly:
    w_part_view = param_buf.narrow(0, start, part)

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    ext.launch_fused_adam_bf16(
        w_part_view, g_part, m_part, v_part,
        float(beta1), float(beta2), float(eps),
        float(bc1), float(bc2), float(lr),
    )

    # ---- 5) All-gather: each rank already wrote its updated partition into its slot
    # of param_buf. Other ranks' slots still hold pre-step weights; we need their
    # post-step weights. Fetch each peer's partition via UVA into our param_buf.
    dist.barrier()
    # Pull peer partitions into our local param_buf at their respective offsets.
    for peer in range(world_size):
        if peer == rank:
            continue
        peer_ptr = int(param_hdl.buffer_ptrs[peer])
        offset_bytes = peer * part * param_buf.element_size()
        dst_view = param_buf.narrow(0, peer * part, part)
        nbytes = part * param_buf.element_size()
        ext.launch_memcpy_from_ptr(dst_view, peer_ptr + offset_bytes, nbytes)

    torch.cuda.synchronize()
    dist.barrier()

    out_params = _unflatten_dense_tensors(param_buf, templates)
    out_params = [t.clone() for t in out_params]
    return (*out_params, m_part, v_part)


__all__ = ["solution"]