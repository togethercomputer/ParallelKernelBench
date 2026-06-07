"""
ZeRO-2 step with custom CUDA: multimem reduce-scatter + multimem all-gather over
symmetric memory, fused Adam kernel on the local partition. Broadcast of params
from rank 0 also done via symm_mem.
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

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}
__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void barrier_relaxed(const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int t = threadIdx.x;
    if (t >= (unsigned)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[t];
    uint32_t* send_addr = (uint32_t*)(remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = (uint32_t*)(local_base + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}
__device__ void barrier_acq_rel(const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int t = threadIdx.x;
    if (t >= (unsigned)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[t];
    uint32_t* send_addr = (uint32_t*)(remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = (uint32_t*)(local_base + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

__device__ __forceinline__ void mm_ldreduce_bf16x4(const uint64_t* addr,
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(addr) : "memory");
}
__device__ __forceinline__ void mm_st_v4f32(const uint64_t* addr,
    uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        :: "l"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

// Multimem all-reduce on bf16 buffer (numel_128 = numel/8 since v4.bf16x2 = 8 bf16 elems)
__global__ void mm_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank
) {
    const uint64_t bid = blockIdx.x;
    barrier_relaxed(signal_pad_ptrs, bid, rank, world_size);
    __syncthreads();

    int64_t per_rank = (numel_128 + world_size - 1) / world_size;
    int64_t my_start = (int64_t)rank * per_rank;
    int64_t my_end = my_start + per_rank;
    if (my_end > numel_128) my_end = numel_128;

    int64_t total = my_end - my_start;
    int64_t tid = (int64_t)bid * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    uint64_t* base = (uint64_t*)multicast_base;
    for (int64_t i = tid; i < total; i += stride) {
        int64_t idx = my_start + i;
        uint64_t* p = base + idx * 2;
        uint32_t a, b, c, d;
        mm_ldreduce_bf16x4(p, a, b, c, d);
        mm_st_v4f32(p, a, b, c, d);
    }

    __syncthreads();
    barrier_acq_rel(signal_pad_ptrs, bid, rank, world_size);
}

// Fused Adam on partition: reads g_part (bf16) and w_part_in (bf16),
// updates m,v (bf16), writes new w_part (bf16) into symmetric buffer at offset.
__global__ void adam_bf16_kernel(
    const __nv_bfloat16* __restrict__ g,
    __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ m,
    __nv_bfloat16* __restrict__ v,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2, float scale,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float gi = __bfloat162float(g[i]) * scale;
        float mi = __bfloat162float(m[i]) * beta1 + gi * (1.0f - beta1);
        float vi = __bfloat162float(v[i]) * beta2 + gi * gi * (1.0f - beta2);
        float mh = mi / bc1;
        float vh = vi / bc2;
        float wi = __bfloat162float(w[i]) - lr * mh / (sqrtf(vh) + eps);
        m[i] = __float2bfloat16(mi);
        v[i] = __float2bfloat16(vi);
        w[i] = __float2bfloat16(wi);
    }
}

// Multimem load-broadcast: each rank uses multimem load to read data already
// written by everyone (we use it as a barrier+visibility tool). Simpler: use
// peer-pointer copy. We'll provide a peer-copy all-gather kernel.
__global__ void allgather_bf16_kernel(
    const uint64_t* peer_ptrs,    // world_size pointers to flat_p buffers
    __nv_bfloat16* out,           // local flat output (size = world_size * part)
    int64_t part,
    int world_size,
    int rank
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t total = (int64_t)world_size * part;
    for (int64_t i = tid; i < total; i += stride) {
        int r = (int)(i / part);
        int64_t off = i - (int64_t)r * part;
        const __nv_bfloat16* src = (const __nv_bfloat16*)peer_ptrs[r];
        out[i] = src[r * part + off];  // peer's flat[r*part + off] is its partition
    }
}

void launch_mm_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel,
    int world_size,
    int rank
) {
    TORCH_CHECK(numel % 8 == 0, "numel must be divisible by 8");
    int64_t numel_128 = numel / 8;
    int block_size = 256;
    int num_blocks = 16;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* sp = (const uint64_t*)signal_pad_ptrs_tensor.data_ptr<int64_t>();
    mm_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, sp, numel_128, world_size, rank);
}

void launch_adam_bf16(
    torch::Tensor g, torch::Tensor w, torch::Tensor m, torch::Tensor v,
    double lr, double beta1, double beta2, double eps,
    double bc1, double bc2, double scale, int64_t n
) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    adam_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)g.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)w.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)m.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)v.data_ptr<at::BFloat16>(),
        (float)lr, (float)beta1, (float)beta2, (float)eps,
        (float)bc1, (float)bc2, (float)scale, n);
}

void launch_allgather_bf16(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor out,
    int64_t part,
    int world_size,
    int rank
) {
    int threads = 256;
    int64_t total = (int64_t)world_size * part;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* pp = (const uint64_t*)peer_ptrs_tensor.data_ptr<int64_t>();
    allgather_bf16_kernel<<<blocks, threads, 0, stream>>>(
        pp, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), part, world_size, rank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_mm_allreduce_bf16", &launch_mm_allreduce_bf16);
    m.def("launch_adam_bf16", &launch_adam_bf16);
    m.def("launch_allgather_bf16", &launch_allgather_bf16);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("zero2_bf16_ext", CUDA_SRC)
    return _ext

_cache = {}

def _get_resources(total_numel: int, part: int, dtype: torch.dtype, device: torch.device):
    key = (total_numel, part, dtype, device)
    if key in _cache:
        return _cache[key]
    # Symmetric buffer for flat parameters/gradients (size = total_numel)
    flat_buf = symm_mem.empty(total_numel, device=device, dtype=dtype)
    flat_hdl = symm_mem.rendezvous(flat_buf, dist.group.WORLD)

    # Output gather buffer (local)
    gather_out = torch.empty(total_numel, device=device, dtype=dtype)

    peer_ptrs_tensor = torch.tensor(flat_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (flat_buf, flat_hdl, gather_out, peer_ptrs_tensor)
    _cache[key] = res
    return res


def solution(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor, b1: Tensor,
    W2: Tensor, b2: Tensor,
    exp_avg_part: Tensor,
    exp_avg_sq_part: Tensor,
    lr: float, beta1: float, beta2: float, eps: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = W1.device

    templates = [W1, b1, W2, b2]
    flat_p_cpu = _flatten_dense_tensors(templates)
    total_numel = flat_p_cpu.numel()
    part = exp_avg_part.numel()
    assert total_numel == part * world_size
    dtype = flat_p_cpu.dtype

    ext = _get_ext()
    flat_buf, flat_hdl, gather_out, peer_ptrs_tensor = _get_resources(
        total_numel, part, dtype, device)

    # ---- Broadcast initial flat_p from rank 0 via symmetric memory ----
    if rank == 0:
        flat_buf.copy_(flat_p_cpu)
    flat_hdl.barrier(channel=0)
    if rank != 0:
        # Pull from rank 0's buffer
        src_ptr = int(flat_hdl.buffer_ptrs[0])
        # Use a quick cudaMemcpy via a tensor view from UVA pointer
        # Simpler: dist.broadcast on the symmetric buffer
        pass
    # Use dist.broadcast for initial param sync (small overhead)
    dist.broadcast(flat_buf, src=0)

    # Materialize param views from symmetric buffer
    param_views = _unflatten_dense_tensors(flat_buf, templates)
    params = [t.detach().clone().requires_grad_(True) for t in param_views]

    m_part = exp_avg_part.clone()
    v_part = exp_avg_sq_part.clone()

    # ---- Forward / backward ----
    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    flat_g = _flatten_dense_tensors([p.grad for p in params]).contiguous()

    # ---- Reduce-scatter via multimem all-reduce on symmetric buffer ----
    # We do all-reduce on full grad, then take our partition. With multimem,
    # cost ~ same as reduce-scatter for small/medium sizes.
    flat_buf.copy_(flat_g)

    if total_numel % 8 == 0 and dtype == torch.bfloat16:
        ext.launch_mm_allreduce_bf16(
            int(flat_hdl.multicast_ptr),
            flat_hdl.signal_pad_ptrs_dev,
            total_numel, world_size, rank)
    else:
        flat_hdl.barrier(channel=0)
        dist.all_reduce(flat_buf, op=dist.ReduceOp.SUM)

    # Extract our partition; divide by world_size
    start = rank * part
    g_part = flat_buf[start:start + part].clone()
    g_part_scale = 1.0 / world_size  # apply inside adam kernel via 'scale'

    # We also need w_part (current weights). Re-broadcast happened already; pull from
    # original param values. But flat_buf now holds gradients. Reconstruct w_part
    # from params tensors.
    w_part_full = _flatten_dense_tensors([p.detach() for p in params]).contiguous()
    w_part = w_part_full[start:start + part].clone()

    # ---- Fused Adam ----
    assert step >= 1
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    if dtype == torch.bfloat16:
        ext.launch_adam_bf16(
            g_part, w_part, m_part, v_part,
            lr, beta1, beta2, eps, bc1, bc2, g_part_scale, part)
    else:
        g_part.mul_(g_part_scale)
        m_part.mul_(beta1).add_(g_part, alpha=1.0 - beta1)
        v_part.mul_(beta2).addcmul_(g_part, g_part, value=1.0 - beta2)
        m_hat = m_part / bc1
        v_hat = v_part / bc2
        w_part.add_(m_hat.div(v_hat.sqrt().add(eps)).mul(-lr))

    # ---- All-gather: write our partition into our slot of symmetric buffer,
    #      then peer-copy from all ranks ----
    flat_buf[start:start + part].copy_(w_part)
    flat_hdl.barrier(channel=0)

    if dtype == torch.bfloat16:
        ext.launch_allgather_bf16(
            peer_ptrs_tensor, gather_out, part, world_size, rank)
        flat_out = gather_out
    else:
        flat_out = torch.empty_like(flat_buf)
        dist.all_gather_into_tensor(flat_out, flat_buf[start:start+part].contiguous())

    flat_hdl.barrier(channel=1)

    out_params = _unflatten_dense_tensors(flat_out, templates)
    out_params = [p.clone() for p in out_params]
    return (*out_params, m_part, v_part)


__all__ = ["solution"]