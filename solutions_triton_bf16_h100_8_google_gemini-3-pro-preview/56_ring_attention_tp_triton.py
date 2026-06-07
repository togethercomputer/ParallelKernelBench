import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension
import triton
import triton.language as tl
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Custom CUDA Extension for UVA Double-buffering & TP AllReduce
# ---------------------------------------------------------------------------

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

// Fast async direct memory copy bypassing NCCL using DMA engine
void async_copy(int64_t src_ptr, torch::Tensor dst, int64_t n) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const void* src = reinterpret_cast<const void*>(src_ptr);
    void* dst_ptr = dst.data_ptr();
    cudaMemcpyAsync(dst_ptr, src, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
}

// Fused TP all-reduce kernel running directly over UVA peer pointers
__global__ void tp_allreduce_kernel(
    const void* p0, const void* p1, const void* p2, const void* p3,
    const void* p4, const void* p5, const void* p6, const void* p7,
    __nv_bfloat16* out, int64_t n, int tp_size
) {
    const void* ptrs[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < tp_size; ++i) {
            if (ptrs[i] != nullptr) {
                const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(ptrs[i]);
                sum += __bfloat162float(p[idx]);
            }
        }
        out[idx] = __float2bfloat16(sum);
    }
}

void tp_allreduce(std::vector<int64_t> ptrs, torch::Tensor out) {
    int tp_size = ptrs.size();
    TORCH_CHECK(tp_size <= 8, "Max TP size supported is 8");
    int64_t p[8] = {0};
    for(int i = 0; i < tp_size; ++i) p[i] = ptrs[i];
    
    int64_t n = out.numel();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    tp_allreduce_kernel<<<blocks, threads, 0, stream>>>(
        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, tp_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("async_copy", &async_copy, "Async copy from raw pointer to tensor");
    m.def("tp_allreduce", &tp_allreduce, "TP allreduce over UVA");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_ring_attn_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(key, shape, dtype, device, group):
    global _symm_cache
    cache_key = (key, tuple(shape), dtype)
    if cache_key in _symm_cache:
        return _symm_cache[cache_key]
    
    n = torch.prod(torch.tensor(shape)).item()
    buf = symm_mem.empty(n, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    buf_tensor = buf.view(shape)
    _symm_cache[cache_key] = (buf_tensor, hdl)
    return buf_tensor, hdl

_copy_stream = None

# ---------------------------------------------------------------------------
# Triton Flash Attention Kernel (Stateful Accumulation)
# ---------------------------------------------------------------------------

def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    Out, Lse,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S_q, S_k, H, D,
    is_first_step: tl.constexpr, causal: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    b_idx = off_hz // H
    h_idx = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    q_ptrs = Q + b_idx * stride_qb + h_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S_q) & mask_d[None, :], other=0.0)
    
    # Init vs Carry over
    if is_first_step:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    else:
        lse_ptrs = Lse + b_idx * stride_lseb + h_idx * stride_lseh + offs_m * stride_lses
        m_i = tl.load(lse_ptrs, mask=offs_m < S_q, other=-float("inf"))
        l_i = tl.where(offs_m < S_q, 1.0, 0.0)
        
        o_ptrs = Out + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        acc = tl.load(o_ptrs, mask=(offs_m[:, None] < S_q) & mask_d[None, :], other=0.0).to(tl.float32)

    num_k_blocks = (S_k + BLOCK_N - 1) // BLOCK_N
    for start_n in range(0, num_k_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        if causal:
            k_min = start_n * BLOCK_N
            q_max = start_m * BLOCK_M + BLOCK_M - 1
            if k_min > q_max:
                break
                
        k_ptrs = K + b_idx * stride_kb + h_idx * stride_kh + offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd
        v_ptrs = V + b_idx * stride_vb + h_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=(offs_n[None, :] < S_k) & mask_d[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < S_k) & mask_d[None, :], other=0.0)
        
        qk = tl.dot(q, k) * sm_scale
        
        if causal:
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, -float("inf"))
            
        mask_k = offs_n[None, :] < S_k
        qk = tl.where(mask_k, qk, -float("inf"))
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        m_i_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i_safe - m_new_safe))
        
        qk_safe = tl.where(qk == -float("inf"), -10000.0, qk)
        beta = tl.exp(qk_safe - m_new_safe[:, None])
        beta = tl.where(qk == -float("inf"), 0.0, beta)
        
        l_ij = tl.sum(beta, 1)
        l_new = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc += tl.dot(beta.to(tl.bfloat16), v)
        
        m_i = m_new
        l_i = l_new

    # Store bounds safely
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    lse = m_i + tl.math.log(l_i_safe)
    acc = acc / l_i_safe[:, None]
    
    lse_ptrs = Lse + b_idx * stride_lseb + h_idx * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S_q)
    
    o_ptrs = Out + b_idx * stride_ob + h_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < S_q) & mask_d[None, :])


def triton_flash_attention(q, k, v, out, lse, sm_scale, causal, is_first_step):
    B, S_q, H, D = q.shape
    _, S_k, _, _ = k.shape
    
    BLOCK_M, BLOCK_N = 128, 128
    BLOCK_D = next_power_of_2(D)
    grid = (triton.cdiv(S_q, BLOCK_M), B * H, 1)
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, out, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        S_q, S_k, H, D,
        is_first_step=is_first_step, causal=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3
    )

# ---------------------------------------------------------------------------
# Forward Call
# ---------------------------------------------------------------------------

@torch.no_grad()
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

    tp_size = dist.get_world_size(tp_group)
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)

    heads_local = num_heads // tp_size
    head_dim = w_qkv.shape[0] // 3 // heads_local
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    # 1. QKV projection locally
    B, S = hidden_states.shape[:2]
    qkv = F.linear(hidden_states, w_qkv).view(B, S, 3, heads_local, head_dim)
    q, k, v = qkv.unbind(dim=2)
    
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    lse = torch.empty((B, heads_local, S), dtype=torch.float32, device=q.device)

    # 2. Ring CP pass overlaid with Direct NVLink Loads
    k_symm, k_hdl = _get_symm_state("k", k.shape, k.dtype, k.device, cp_group)
    v_symm, v_hdl = _get_symm_state("v", v.shape, v.dtype, v.device, cp_group)
    k_symm.copy_(k)
    v_symm.copy_(v)
    k_hdl.barrier(channel=0)
    v_hdl.barrier(channel=0)

    steps_to_run = (cp_rank + 1) if causal else cp_size
    k_buf = [torch.empty_like(k) for _ in range(2)] if cp_size > 1 else []
    v_buf = [torch.empty_like(v) for _ in range(2)] if cp_size > 1 else []

    global _copy_stream
    if _copy_stream is None:
        _copy_stream = torch.cuda.Stream()
    copy_stream = _copy_stream
    
    buf_idx = 0
    for step in range(steps_to_run):
        is_last_step = (step == steps_to_run - 1)
        
        # Sync with DMA copies prepared in preceding loops
        if step > 0:
            torch.cuda.current_stream().wait_stream(copy_stream)
            
        # Dispatch DMA ops asynchronously for the NEXT step (double buffering)
        if not is_last_step:
            next_remote_rank = (cp_rank - step - 1) % cp_size
            next_k_ptr = k_hdl.buffer_ptrs[next_remote_rank]
            next_v_ptr = v_hdl.buffer_ptrs[next_remote_rank]
            with torch.cuda.stream(copy_stream):
                _get_ext().async_copy(next_k_ptr, k_buf[1 - buf_idx], k.numel())
                _get_ext().async_copy(next_v_ptr, v_buf[1 - buf_idx], v.numel())
                
        # Resolve target memory to compute against locally
        if step == 0:
            current_k, current_v = k_symm, v_symm
        else:
            current_k, current_v = k_buf[buf_idx], v_buf[buf_idx]
            
        triton_flash_attention(
            q, current_k, current_v, out, lse, 
            float(softmax_scale), 
            causal=(causal and step == 0), 
            is_first_step=(step == 0)
        )
        
        if step > 0:
            buf_idx = 1 - buf_idx

    # Ensure no one jumps to the next iteration before current pulls are finished
    k_hdl.barrier(channel=1)
    v_hdl.barrier(channel=1)

    # 3. Row-parallel output projection + UVA Tensor-Parallel All-reduce
    out_proj = F.linear(out.reshape(B, S, -1), w_o).contiguous()
    
    if tp_size > 1:
        tp_buf, tp_hdl = _get_symm_state("tp", out_proj.shape, out_proj.dtype, out_proj.device, tp_group)
        tp_buf.copy_(out_proj)
        tp_hdl.barrier(channel=0)
        
        ptrs = [int(p) for p in tp_hdl.buffer_ptrs]
        _get_ext().tp_allreduce(ptrs, out_proj)
        
        tp_hdl.barrier(channel=1)
        
    return out_proj