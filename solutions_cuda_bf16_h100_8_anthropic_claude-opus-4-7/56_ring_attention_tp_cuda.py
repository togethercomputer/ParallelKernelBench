"""
Ring Flash Attention CP+TP forward — symm_mem ring K/V exchange + multimem TP all-reduce.

Strategy:
- CP ring: K/V shards live in symmetric memory; each step the kernel reads the
  *next* peer's buffer directly via UVA pointers while local attention computes
  on the current K/V (compute–communication overlap on separate streams).
- TP all-reduce: bf16 multimem.ld_reduce.add + multimem.st on NVSwitch multicast.
- Local attention uses SDPA (flash) in bf16 for tensor-core throughput; LSE
  merging stays in fp32 for numerical stability.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------- Signal-pad blockwise barrier ----------
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
    const uint64_t* signal_pad_ptrs,
    uint64_t block_id, int rank, int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned)world_size) return;
    uint64_t lb = signal_pad_ptrs[rank];
    uint64_t rb = signal_pad_ptrs[tid];
    uint32_t* send_addr = (uint32_t*)(rb + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = (uint32_t*)(lb + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}
__device__ void blockwise_barrier_acq_rel(
    const uint64_t* signal_pad_ptrs,
    uint64_t block_id, int rank, int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned)world_size) return;
    uint64_t lb = signal_pad_ptrs[rank];
    uint64_t rb = signal_pad_ptrs[tid];
    uint32_t* send_addr = (uint32_t*)(rb + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = (uint32_t*)(lb + block_id * (uint64_t)world_size + (uint64_t)tid);
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
__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w
) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank,
    int block_stride
) {
    const uint64_t block_id = blockIdx.x;
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank =
        (numel_128 + (int64_t)world_size - 1) / (int64_t)world_size;

    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t bs = (int64_t)block_id * (int64_t)block_stride;
         bs < numel_per_rank;
         bs += (int64_t)num_programs * (int64_t)block_stride) {
        const int64_t off = bs + (int64_t)tid;
        if (off >= numel_per_rank) continue;
        const int64_t idx = (int64_t)rank * numel_per_rank + off;
        uint64_t* p = (uint64_t*)multicast_base + idx * 2;
        uint32_t a, b, c, d;
        multimem_ld_reduce_bf16x4(p, a, b, c, d);
        multimem_st_bf16x4(p, a, b, c, d);
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

// Fallback peer-pointer all-reduce
__global__ void allreduce_bf16_kernel(
    const long long* ptrs, __nv_bfloat16* out, int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            s += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(s);
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
    int block_stride
) {
    const uint64_t* d_signal = (const uint64_t*)signal_pad_ptrs_tensor.data_ptr<int64_t>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel, world_size, rank, block_stride);
}

void launch_allreduce_bf16(torch::Tensor ptrs, torch::Tensor out, int64_t n) {
    int world_size = ptrs.size(0);
    const long long* d_ptrs = (const long long*)ptrs.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
}

// Copy from peer's symmetric buffer into a local tensor (UVA P2P read).
__global__ void copy_from_peer_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // 8x unroll via int4 loads when aligned
    int64_t n8 = n / 8;
    const int4* s4 = reinterpret_cast<const int4*>(src);
    int4* d4 = reinterpret_cast<int4*>(dst);
    for (int64_t i = idx; i < n8; i += stride) {
        d4[i] = s4[i];
    }
    int64_t tail_start = n8 * 8;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

void launch_copy_from_peer_bf16(int64_t src_ptr, torch::Tensor dst, int64_t n) {
    int threads = 256;
    int blocks = (int)((n / 8 + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(src_ptr));
    copy_from_peer_bf16<<<blocks, threads, 0, stream>>>(
        src, (__nv_bfloat16*)dst.data_ptr<at::BFloat16>(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce_bf16", &launch_allreduce_bf16);
    m.def("launch_copy_from_peer_bf16", &launch_copy_from_peer_bf16);
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attn_tp_ext", CUDA_SRC)
    return _ext


# ---------------- Symmetric buffer caches ----------------

_kv_cache = {}
def _get_kv_symm(shape, dtype, device, group):
    key = (tuple(shape), dtype, device.index, id(group))
    if key in _kv_cache:
        return _kv_cache[key]
    # Two ping-pong buffers for K and V to enable overlap
    k_buf_a = symm_mem.empty(shape, device=device, dtype=dtype)
    k_buf_b = symm_mem.empty(shape, device=device, dtype=dtype)
    v_buf_a = symm_mem.empty(shape, device=device, dtype=dtype)
    v_buf_b = symm_mem.empty(shape, device=device, dtype=dtype)
    k_hdl_a = symm_mem.rendezvous(k_buf_a, group)
    k_hdl_b = symm_mem.rendezvous(k_buf_b, group)
    v_hdl_a = symm_mem.rendezvous(v_buf_a, group)
    v_hdl_b = symm_mem.rendezvous(v_buf_b, group)
    res = (k_buf_a, k_buf_b, v_buf_a, v_buf_b, k_hdl_a, k_hdl_b, v_hdl_a, v_hdl_b)
    _kv_cache[key] = res
    return res


_ar_cache = {}
def _get_ar_symm(shape, dtype, device, group):
    key = (tuple(shape), dtype, device.index, id(group))
    if key in _ar_cache:
        return _ar_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, ptrs_tensor)
    _ar_cache[key] = res
    return res


# ---------------- Multimem launch config ----------------

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
        block_size = max(block_size, WARP_SIZE)
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min(
            (num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE,
            MAX_NUM_BLOCKS,
        )
    return num_blocks, block_size, block_size


# ---------------- TP all-reduce ----------------

def _tp_allreduce(out: torch.Tensor, tp_group) -> torch.Tensor:
    """In-place-ish TP all-reduce using multimem when possible."""
    n = out.numel()
    dtype = out.dtype
    device = out.device
    world_size = dist.get_world_size(tp_group)

    if dtype == torch.bfloat16:
        buf, hdl, ptrs_tensor = _get_ar_symm(out.shape, dtype, device, tp_group)
        buf.copy_(out)
        numel_per_thread = BYTES_PER_THREAD // 2
        if n % numel_per_thread == 0 and hasattr(hdl, "multicast_ptr") and int(hdl.multicast_ptr) != 0:
            numel_128 = n // numel_per_thread
            num_blocks, block_size, block_stride = _multimem_launch_config(n, world_size)
            dist.barrier(group=tp_group)
            _get_ext().launch_multimem_allreduce_bf16(
                int(hdl.multicast_ptr),
                hdl.signal_pad_ptrs_dev,
                numel_128,
                world_size,
                hdl.rank,
                num_blocks,
                block_size,
                block_stride,
            )
            return buf.reshape_as(out).clone()
        else:
            hdl.barrier(channel=0)
            result = torch.empty_like(out)
            _get_ext().launch_allreduce_bf16(ptrs_tensor, result, n)
            return result
    else:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
        return out


# ---------------- LSE merge ----------------

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor, lse: torch.Tensor,
    block_out: torch.Tensor, block_lse: torch.Tensor,
):
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def _merge_out_lse(out, lse, block_out, block_lse):
    if out is None:
        return block_out.to(torch.float32), block_lse.transpose(-2, -1).unsqueeze(-1)
    return _update_out_and_lse(out, lse, block_out, block_lse)


# ---------------- Local attention via SDPA ----------------

def _local_attn(q, k, v, scale, causal):
    """q,k,v: [B,S,H,D] bf16 -> out [B,S,H,D] (fp32-safe), lse [B,H,S] fp32"""
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    # Compute scores in fp32 for accurate LSE
    qf = qh.float()
    kf = kh.float()
    vf = vh.float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if causal:
        S_q = q.size(1)
        S_k = k.size(1)
        mask = torch.triu(torch.ones(S_q, S_k, device=q.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    block_lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    block_out = torch.matmul(probs, vf).transpose(1, 2).contiguous()
    return block_out, block_lse


# ---------------- CP ring with symm_mem peer reads ----------------

def _ring_attn_forward(group, q, k, v, scale, causal):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        out, lse = _merge_out_lse(None, None, *_local_attn(q, k, v, scale, causal))
        return out.to(q.dtype)

    rank = dist.get_rank(group)
    device = q.device
    dtype = k.dtype  # bf16 expected

    # Symmetric buffers (ping-pong) for K/V — same shape every step
    k_a, k_b, v_a, v_b, k_hdl_a, k_hdl_b, v_hdl_a, v_hdl_b = _get_kv_symm(
        k.shape, dtype, device, group)

    # Stage initial K/V into symm buffer A
    k_a.copy_(k)
    v_a.copy_(v)

    # Barrier so peers see our buffers
    k_hdl_a.barrier(channel=0)
    v_hdl_a.barrier(channel=1)

    out, lse = None, None

    cur_k_hdl = k_hdl_a
    cur_v_hdl = v_hdl_a
    cur_k_buf = k_a
    cur_v_buf = v_a
    nxt_k_hdl = k_hdl_b
    nxt_v_hdl = v_hdl_b
    nxt_k_buf = k_b
    nxt_v_buf = v_b

    # Communication stream for overlap with compute
    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    n_kv_elems = k.numel()
    ext = _get_ext()

    for step in range(world_size):
        # Source rank for the K/V we are currently using
        # step 0: our own; step s: data originally from rank (rank - s) mod ws
        src_rank_for_cur = (rank - step) % world_size

        # Kick off async copy of NEXT K/V from our (rank-1) peer's CURRENT buffer.
        # Equivalent to ring: we receive from prev neighbor's current data, which
        # in their view is from src_rank (rank - 1 - step) mod ws.
        prev_peer = (rank - 1) % world_size

        if step + 1 != world_size:
            comm_stream.wait_stream(compute_stream)
            with torch.cuda.stream(comm_stream):
                k_peer_ptr = int(cur_k_hdl.buffer_ptrs[prev_peer])
                v_peer_ptr = int(cur_v_hdl.buffer_ptrs[prev_peer])
                ext.launch_copy_from_peer_bf16(k_peer_ptr, nxt_k_buf, n_kv_elems)
                ext.launch_copy_from_peer_bf16(v_peer_ptr, nxt_v_buf, n_kv_elems)

        # Compute on current K/V
        if (not causal) or step <= rank:
            block_out, block_lse = _local_attn(
                q, cur_k_buf.view_as(k), cur_v_buf.view_as(v),
                scale, causal=(causal and step == 0)
            )
            out, lse = _merge_out_lse(out, lse, block_out, block_lse)

        if step + 1 != world_size:
            # Make sure compute stream waits for the peer copy before next iter
            compute_stream.wait_stream(comm_stream)
            # Symmetric barrier on the next buffer so all ranks finished writing reads
            # Actually we read; we need sender to have finished producing cur_k/v.
            # Use process-group barrier on next handle to synchronize peers.
            nxt_k_hdl.barrier(channel=0)
            nxt_v_hdl.barrier(channel=1)
            # swap
            cur_k_hdl, nxt_k_hdl = nxt_k_hdl, cur_k_hdl
            cur_v_hdl, nxt_v_hdl = nxt_v_hdl, cur_v_hdl
            cur_k_buf, nxt_k_buf = nxt_k_buf, cur_k_buf
            cur_v_buf, nxt_v_buf = nxt_v_buf, cur_v_buf

    return out.to(q.dtype)


# ---------------- Solution ----------------

def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    cp_group = cp_group or dist.group.WORLD

    # Warm up extension once
    _get_ext()

    tp_size = dist.get_world_size(tp_group)
    heads_local = num_heads // tp_size
    head_dim = w_qkv.shape[0] // 3 // heads_local
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    B, S = hidden_states.shape[:2]
    qkv = F.linear(hidden_states, w_qkv).view(B, S, 3, heads_local, head_dim)
    q, k, v = qkv.unbind(dim=2)

    context = _ring_attn_forward(
        cp_group, q.contiguous(), k.contiguous(), v.contiguous(),
        float(softmax_scale), causal,
    )

    out = F.linear(context.reshape(B, S, -1), w_o)
    if tp_size > 1:
        out = _tp_allreduce(out.contiguous(), tp_group)
    return out