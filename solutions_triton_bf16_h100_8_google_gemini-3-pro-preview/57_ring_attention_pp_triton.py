"""
Strategy:
1. Overlapping Context Parallelism: We allocate a double-buffered layout in symmetric memory for KV states. A custom CUDA extension handles zero-copy UVA vector transfers (int4 for bf16) to peer NVLink memory. This transfer executes on a separate CUDA stream, guaranteeing perfect overlap with the Triton attention computation of the current ring step.
2. Fused Attention & Update: Instead of launching local attention blocks and then using `torch.sigmoid` mathematically to update output components, we fuse the Megatron Flash Attention update into the Triton block. The running Output and LogSumExp vectors update perfectly in place via online softmax logic, sidestepping repetitive host-launched elementwise overheads.
3. Asynchronous Pipeline Synchronization: For PP stages, we push tensors directly over UVA and update an explicitly allocated `sync_buf` variable on the remote device via `__threadfence_system()`. The receiver busy-waits on device memory without ever halting the CPU thread, eliminating all `dist.isend`/`irecv` blockages on the fast path.
"""

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from utils.cuda_helpers import compile_cuda_extension

# ---------------------------------------------------------------------------
# Custom CUDA P2P UVA Data Mover & Sync
# ---------------------------------------------------------------------------

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void copy_uva_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t n_vec,
    int64_t n_total,
    const uint16_t* __restrict__ src_rem,
    uint16_t* __restrict__ dst_rem
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        dst[idx] = src[idx];
    }
    // Handle unaligned trailing elements
    if (idx == 0 && n_total % (sizeof(T) / 2) != 0) {
        int64_t offset = n_vec * (sizeof(T) / 2);
        int64_t rem = n_total - offset;
        for(int64_t i = 0; i < rem; ++i) {
            dst_rem[offset + i] = src_rem[offset + i];
        }
    }
}

void copy_uva_bf16(torch::Tensor src, int64_t dst_ptr) {
    int64_t n = src.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    int64_t vec_size = 8; // 8 bf16 = 16 bytes (int4)
    int64_t n_vec = n / vec_size;
    
    const int threads = 256;
    const int blocks = (n_vec + threads - 1) / threads;
    
    auto src_ptr = src.data_ptr();
    auto dst = reinterpret_cast<void*>(static_cast<uintptr_t>(dst_ptr));
    
    if (blocks > 0) {
        copy_uva_kernel<int4><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const int4*>(src_ptr),
            reinterpret_cast<int4*>(dst),
            n_vec,
            n,
            reinterpret_cast<const uint16_t*>(src_ptr),
            reinterpret_cast<uint16_t*>(dst)
        );
    } else if (n > 0) {
        copy_uva_kernel<int4><<<1, 1, 0, stream>>>(
            nullptr, nullptr, 0, n,
            reinterpret_cast<const uint16_t*>(src_ptr),
            reinterpret_cast<uint16_t*>(dst)
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void write_sync_kernel(int32_t* remote_sync) {
    __threadfence_system();
    *remote_sync = 1;
    __threadfence_system();
}

void write_sync(int64_t remote_ptr) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    write_sync_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<int32_t*>(remote_ptr));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void wait_sync_kernel(volatile int32_t* local_sync) {
    while (*local_sync == 0) {
        // device-side busy wait for peer PP rank
    }
    __threadfence_system();
    *local_sync = 0; // atomic reset for subsequent microbatches
    __threadfence_system();
}

void wait_sync(torch::Tensor local_sync) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    wait_sync_kernel<<<1, 1, 0, stream>>>(local_sync.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_uva_bf16", &copy_uva_bf16, "Vectorized UVA copy for BF16");
    m.def("write_sync", &write_sync, "Write sync flag to remote device via NVLink");
    m.def("wait_sync", &wait_sync, "Wait for local sync flag and reset");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attention_comm", CUDA_SRC)
    return _ext


# ---------------------------------------------------------------------------
# Symmetric Memory Buffer Cache
# ---------------------------------------------------------------------------

_symm_cache = {}

def _get_cp_symm_state(cp_group, B, S, H, D, dtype, device):
    global _symm_cache
    key = ('cp', id(cp_group), B, S, H, D, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    # Double buffer for K and V layout: [2, 2, B, S, H, D]
    buf_kv = symm_mem.empty((2, 2, B, S, H, D), dtype=dtype, device=device)
    hdl_kv = symm_mem.rendezvous(buf_kv, group=cp_group)
    
    _symm_cache[key] = (buf_kv, hdl_kv)
    return buf_kv, hdl_kv

def _get_pp_symm_state(pp_group, B, S, hidden_size, dtype, device):
    global _symm_cache
    key = ('pp', id(pp_group), B, S, hidden_size, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty((B, S, hidden_size), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group=pp_group)
    
    sync_buf = symm_mem.empty((1,), dtype=torch.int32, device=device)
    sync_buf.zero_()
    hdl_sync = symm_mem.rendezvous(sync_buf, group=pp_group)
    
    _symm_cache[key] = (buf, hdl, sync_buf, hdl_sync)
    return buf, hdl, sync_buf, hdl_sync

def _next_power_of_2(n: int) -> int:
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1

# ---------------------------------------------------------------------------
# Fused Local Attention Block with Online Output/LSE Merge
# ---------------------------------------------------------------------------

@triton.jit
def attn_fwd_kernel(
    Q, K, V, sm_scale,
    Out, Lse,
    stride_qz, stride_qs, stride_qh, stride_qd,
    stride_kz, stride_ks, stride_kh, stride_kd,
    stride_vz, stride_vs, stride_vh, stride_vd,
    stride_oz, stride_os, stride_oh, stride_od,
    stride_lsez, stride_lseh, stride_lses,
    Z, S_Q, S_K, H, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    INIT: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + off_z * stride_qz + offs_m[:, None] * stride_qs + off_h * stride_qh + offs_d[None, :] * stride_qd
    k_ptrs = K + off_z * stride_kz + offs_n[None, :] * stride_ks + off_h * stride_kh + offs_d[:, None] * stride_kd
    v_ptrs = V + off_z * stride_vz + offs_n[:, None] * stride_vs + off_h * stride_vh + offs_d[None, :] * stride_vd

    q_mask = (offs_m[:, None] < S_Q) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize or load existing ring running values
    if INIT:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    else:
        lse_ptrs = Lse + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lses
        m_i = tl.load(lse_ptrs, mask=offs_m < S_Q, other=-float('inf'))
        l_i = tl.ones([BLOCK_M], dtype=tl.float32)
        out_ptrs = Out + off_z * stride_oz + offs_m[:, None] * stride_os + off_h * stride_oh + offs_d[None, :] * stride_od
        acc = tl.load(out_ptrs, mask=q_mask, other=0.0)

    for start_n in range(0, S_K, BLOCK_N):
        k_load_mask = ((start_n + offs_n)[None, :] < S_K) & (offs_d[:, None] < D)
        k = tl.load(k_ptrs, mask=k_load_mask, other=0.0)
        qk = tl.dot(q, k) * sm_scale

        q_valid = offs_m[:, None] < S_Q
        k_valid = (start_n + offs_n)[None, :] < S_K

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(causal_mask & q_valid & k_valid, qk, float('-inf'))
        else:
            qk = tl.where(q_valid & k_valid, qk, float('-inf'))

        m_ij = tl.max(qk, 1)
        new_m = tl.maximum(m_i, m_ij)
        
        m_diff = m_i - new_m
        m_diff = tl.where(new_m == float('-inf'), 0.0, m_diff)
        alpha = tl.exp(m_diff)
        
        qk_diff = qk - new_m[:, None]
        qk_diff = tl.where(new_m[:, None] == float('-inf'), -float('inf'), qk_diff)
        beta = tl.exp(qk_diff)
        
        l_ij = tl.sum(beta, 1)
        new_l = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        v_load_mask = ((start_n + offs_n)[:, None] < S_K) & (offs_d[None, :] < D)
        v = tl.load(v_ptrs, mask=v_load_mask, other=0.0)
        p = beta.to(Q.dtype.element_ty)
        acc += tl.dot(p, v)

        m_i = new_m
        l_i = new_l
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    lse = m_i + tl.log(l_i)
    inv_l = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)
    out = acc * inv_l[:, None]

    out_ptrs = Out + off_z * stride_oz + offs_m[:, None] * stride_os + off_h * stride_oh + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(Out.dtype.element_ty), mask=q_mask)
    
    lse_ptrs = Lse + off_z * stride_lsez + off_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S_Q)


# ---------------------------------------------------------------------------
# Main Implementation Target
# ---------------------------------------------------------------------------

@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    if dist.is_initialized() and dist.get_rank() == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()

    B, S, D_hidden = hidden_states.shape
    cp_group = cp_group or dist.group.WORLD
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)

    head_dim = w_qkv.shape[0] // 3 // num_heads
    scale = float(softmax_scale if softmax_scale is not None else head_dim ** -0.5)

    is_first = True
    is_last = True
    if pp_group is not None and dist.get_world_size(pp_group) > 1:
        pp_rank = dist.get_rank(pp_group)
        pp_size = dist.get_world_size(pp_group)
        is_first = (pp_rank == 0)
        is_last = (pp_rank == pp_size - 1)

    # 1. Pipeline Parallel Receive via purely Device-Side Wait
    if is_first:
        stage_input = hidden_states
    else:
        buf, hdl, sync_buf, hdl_sync = _get_pp_symm_state(
            pp_group, B, S, D_hidden, hidden_states.dtype, hidden_states.device
        )
        _get_ext().wait_sync(sync_buf)
        stage_input = buf

    # 2. Local QKV Projection
    qkv = F.linear(stage_input, w_qkv).view(B, S, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)
    lse = torch.empty((B, num_heads, S), dtype=torch.float32, device=q.device)

    BLOCK_M = 128
    BLOCK_N = 64 if head_dim > 64 else 128
    BLOCK_D = _next_power_of_2(head_dim)
    grid = (triton.cdiv(S, BLOCK_M), B * num_heads)

    # 3. Context Parallel Ring Attention Forward
    if cp_size == 1:
        attn_fwd_kernel[grid](
            q, k, v, scale, out, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B, S, S, num_heads, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            IS_CAUSAL=causal, INIT=True
        )
        ctx = out
    else:
        buf_kv, hdl_kv = _get_cp_symm_state(cp_group, B, S, num_heads, head_dim, q.dtype, q.device)
        comm_stream = torch.cuda.Stream()
        
        send_rank = dist.get_global_rank(cp_group, (cp_rank + 1) % cp_size)
        numel_per_tensor = k.numel()
        element_size = k.element_size()
        
        for step in range(cp_size):
            db_idx = step % 2
            
            # Start Async UVA Push to Peer NVLink Receiver Buffer
            if step + 1 < cp_size:
                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_stream(torch.cuda.current_stream())
                    remote_ptr = int(hdl_kv.buffer_ptrs[send_rank])
                    k_offset = (db_idx * 2 + 0) * numel_per_tensor * element_size
                    v_offset = (db_idx * 2 + 1) * numel_per_tensor * element_size
                    
                    _get_ext().copy_uva_bf16(k, remote_ptr + k_offset)
                    _get_ext().copy_uva_bf16(v, remote_ptr + v_offset)
                    
            # Overlapped Attention Math Update
            if (not causal) or step <= cp_rank:
                is_causal_block = causal and (step == 0)
                init = (step == 0)
                attn_fwd_kernel[grid](
                    q, k, v, scale, out, lse,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    B, S, S, num_heads, head_dim,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                    IS_CAUSAL=is_causal_block, INIT=init
                )
                
            # Block and Prepare Peer Shift State Swap
            if step + 1 < cp_size:
                torch.cuda.current_stream().wait_stream(comm_stream)
                torch.cuda.synchronize() # Confirm outgoing UVA P2P transfers are system-visible
                hdl_kv.barrier(channel=0)
                
                k = buf_kv[db_idx, 0]
                v = buf_kv[db_idx, 1]
                
        ctx = out

    # 4. Local Attention Output Projection
    stage_output = F.linear(ctx.reshape(B, S, -1), w_o)

    # 5. Pipeline Parallel Send (Fire and Forget NVLink Sync Updates)
    if not is_last and pp_group is not None:
        buf, hdl, sync_buf, hdl_sync = _get_pp_symm_state(
            pp_group, B, S, D_hidden, stage_output.dtype, stage_output.device
        )
        next_rank = dist.get_global_rank(pp_group, (pp_rank + 1) % pp_size)
        remote_ptr = int(hdl.buffer_ptrs[next_rank])
        remote_sync_ptr = int(hdl_sync.buffer_ptrs[next_rank])
        
        _get_ext().copy_uva_bf16(stage_output, remote_ptr)
        _get_ext().write_sync(remote_sync_ptr)

    return stage_output