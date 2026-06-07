import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional, Tuple
import triton
import triton.language as tl
from utils.cuda_helpers import compile_cuda_extension

# ---------------------------------------------------------------------------
# Custom CUDA Extension for Vectorized P2P Async Copy
# ---------------------------------------------------------------------------

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void p2p_copy_kernel_float4(const float4* __restrict__ src, float4* __restrict__ dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

__global__ void p2p_copy_kernel_bf16(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

void async_p2p_copy(
    int64_t src_ptr,
    torch::Tensor dst,
    int64_t n_bytes,
    int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (n_bytes % 16 == 0) {
        int64_t n = n_bytes / 16;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        p2p_copy_kernel_float4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(src_ptr),
            reinterpret_cast<float4*>(dst.data_ptr()),
            n
        );
    } else {
        int64_t n = n_bytes / 2;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        p2p_copy_kernel_bf16<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(src_ptr),
            reinterpret_cast<uint16_t*>(dst.data_ptr()),
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("async_p2p_copy", &async_p2p_copy, "Async P2P copy");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("p2p_copy_ext", CUDA_SRC)
    return _ext

# ---------------------------------------------------------------------------
# Fused Triton Kernel for Local Attention + LSE Merging
# ---------------------------------------------------------------------------

@triton.jit
def _attn_fwd_step_kernel(
    Q, K, V, Out, LSE,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    scale,
    seqlen_q, seqlen_k,
    is_first_step: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch = tl.program_id(2)

    q_offset = batch * stride_qb + off_hz * stride_qh
    k_offset = batch * stride_kb + off_hz * stride_kh
    v_offset = batch * stride_vb + off_hz * stride_vh
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, BLOCK_D),
        strides=(stride_qs, stride_qd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0)
    )
    q = tl.load(Q_block_ptr)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_offset = batch * stride_lseb + off_hz * stride_lseh + offs_m * stride_lses
    mask_m = offs_m < seqlen_q
    
    if is_first_step:
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    else:
        m_i = tl.load(LSE + lse_offset, mask=mask_m, other=0.0)
        l_i = tl.ones([BLOCK_M], dtype=tl.float32)
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(seqlen_q, BLOCK_D),
            strides=(stride_os, stride_od),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_D),
            order=(1, 0)
        )
        acc = tl.load(O_block_ptr).to(tl.float32)

    lo = 0
    hi = seqlen_k
    if IS_CAUSAL:
        hi = tl.minimum(seqlen_k, (start_m + 1) * BLOCK_M)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_D, seqlen_k),
        strides=(stride_kd, stride_ks),
        offsets=(0, lo),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, BLOCK_D),
        strides=(stride_vs, stride_vd),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0)
    )

    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        qk = tl.dot(q, k, out_dtype=tl.float32) * scale
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if IS_CAUSAL:
            mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < seqlen_k)
        else:
            mask = offs_n[None, :] < seqlen_k
        qk = tl.where(mask, qk, float("-inf"))
            
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    if is_first_step:
        l_i = tl.where(l_i == 0.0, 1e-6, l_i)

    acc = acc / l_i[:, None]
    lse_out = m_i + tl.log(l_i)

    tl.store(LSE + lse_offset, lse_out, mask=mask_m)
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(seqlen_q, BLOCK_D),
        strides=(stride_os, stride_od),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))


def run_triton_attn_step(q, k, v, out, lse, scale, is_first, is_causal):
    B, S, H, D = q.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)
    
    grid = (triton.cdiv(S, BLOCK_M), H, B)
    
    _attn_fwd_step_kernel[grid](
        q, k, v, out, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        scale,
        S, S,
        is_first, is_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )

# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------

@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    softmax_scale = float(softmax_scale)
    
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    
    B, S, H, D = q.shape
    N = B * S * H * D
    dtype = q.dtype
    device = q.device

    out = torch.empty((B, S, H, D), dtype=dtype, device=device)
    lse = torch.empty((B, H, S), dtype=torch.float32, device=device)
    
    if world_size == 1:
        run_triton_attn_step(q, k, v, out, lse, softmax_scale, True, causal)
        return out
        
    # Allocate unified K and V buffer across symmetric memory domain
    kv_symm = symm_mem.empty(2 * N, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(kv_symm, group)
    
    # Expose and locally materialize K/V layout locally [2, B, S, H, D]
    kv_symm_view = kv_symm.view(2, B, S, H, D)
    kv_symm_view[0].copy_(k)
    kv_symm_view[1].copy_(v)
    hdl.barrier(channel=0)
    
    # Double-buffering workspace
    local_kv_buf = torch.empty((2, 2, B, S, H, D), dtype=dtype, device=device)
    
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()
    
    for step in range(world_size):
        # In causal context parallelism, if step > rank, the entire peer sequence chunk is in the future.
        if causal and step > rank:
            break
            
        curr_buf_idx = step % 2
        next_buf_idx = (step + 1) % 2
        
        # 1. Provide active operand subset via double-buffering
        if step == 0:
            k_curr, v_curr = k, v
        else:
            compute_stream.wait_stream(copy_stream)
            k_curr = local_kv_buf[curr_buf_idx, 0]
            v_curr = local_kv_buf[curr_buf_idx, 1]
            
        # 2. Prefetch step+1 subset completely asynchronously
        if step + 1 < world_size and not (causal and step + 1 > rank):
            next_peer = (rank - (step + 1)) % world_size
            peer_ptr = hdl.buffer_ptrs[next_peer]
            
            with torch.cuda.stream(copy_stream):
                # Ensure main stream finishes referencing next_buf_idx
                copy_stream.wait_stream(compute_stream)
                _get_ext().async_p2p_copy(
                    int(peer_ptr),
                    local_kv_buf[next_buf_idx],
                    2 * N * q.element_size(),
                    copy_stream.cuda_stream
                )
                
        # 3. Compute running step iteration
        is_causal = causal and (step == 0)
        is_first = (step == 0)
        run_triton_attn_step(q, k_curr, v_curr, out, lse, softmax_scale, is_first, is_causal)

    # Protect buffers in scope while peers perform their async pulling reads
    hdl.barrier(channel=1)
    return out