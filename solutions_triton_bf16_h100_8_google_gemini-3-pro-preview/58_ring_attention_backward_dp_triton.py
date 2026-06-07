import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension
import triton
import triton.language as tl
from typing import Optional, Tuple

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

__global__ void dp_allreduce_kernel(
    const __nv_bfloat16** ptrs,
    __nv_bfloat16* out,
    int dp_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < dp_size; ++i) {
            sum += __bfloat162float(ptrs[i][idx]);
        }
        sum /= dp_size;
        out[idx] = __float2bfloat16(sum);
    }
}

void dp_allreduce_bf16(
    std::vector<int64_t> ptrs_int,
    torch::Tensor out,
    int dp_size,
    int64_t n
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    const __nv_bfloat16** d_ptrs;
    cudaMallocAsync(&d_ptrs, dp_size * sizeof(void*), stream);
    
    std::vector<const __nv_bfloat16*> h_ptrs(dp_size);
    for (int i = 0; i < dp_size; ++i) {
        h_ptrs[i] = reinterpret_cast<const __nv_bfloat16*>(ptrs_int[i]);
    }
    cudaMemcpyAsync(d_ptrs, h_ptrs.data(), dp_size * sizeof(void*), cudaMemcpyHostToDevice, stream);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dp_allreduce_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, 
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), 
        dp_size, 
        n
    );
    cudaFreeAsync(d_ptrs, stream);
}

torch::Tensor make_tensor_from_ptr(int64_t ptr, std::vector<int64_t> sizes, std::vector<int64_t> strides) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    return torch::from_blob(reinterpret_cast<void*>(ptr), sizes, strides, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dp_allreduce_bf16", &dp_allreduce_bf16, "UVA DP All-Reduce for bf16");
    m.def("make_tensor_from_ptr", &make_tensor_from_ptr, "Create tensor from UVA pointer");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attn_bwd_dp", CUDA_SRC)
    return _ext

@triton.jit
def bwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, dout_ptr, lse_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    seqlen_q, seqlen_k, H,
    CAUSAL_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    
    offs_m = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    q_ptrs = q_ptr + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    dout_ptrs = dout_ptr + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    out_ptrs = out_ptr + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    
    lse_ptrs = lse_ptr + b * (H * seqlen_q) + h * seqlen_q + offs_m
    
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    dout = tl.load(dout_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    out = tl.load(out_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    lse = tl.load(lse_ptrs, mask=offs_m < seqlen_q, other=0.0)
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    for start_k in range(0, seqlen_k, BLOCK_N):
        offs_n_curr = start_k + offs_n
        
        k_ptrs = k_ptr + k_offset + offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + k_offset + offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd
        
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < seqlen_k, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < seqlen_k, other=0.0)
        
        qk = tl.dot(q, tl.trans(k)) * scale
        if CAUSAL_MASK:
            mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(mask, qk, float("-inf"))
            
        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(dout, tl.trans(v))
        row_dot = tl.sum(dout * out, axis=1)
        ds = p * (dp - row_dot[:, None])
        
        dq += tl.dot(ds.to(k.dtype), k) * scale
        
        dk = tl.dot(tl.trans(ds).to(q.dtype), q) * scale
        dv = tl.dot(tl.trans(p).to(dout.dtype), dout)
        
        dk_ptrs = dk_ptr + k_offset + offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd
        dv_ptrs = dv_ptr + k_offset + offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd
        
        tl.atomic_add(dk_ptrs, dk.to(tl.bfloat16), mask=offs_n_curr[:, None] < seqlen_k)
        tl.atomic_add(dv_ptrs, dv.to(tl.bfloat16), mask=offs_n_curr[:, None] < seqlen_k)
        
    dq_ptrs = dq_ptr + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    dq_prev = tl.load(dq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    tl.store(dq_ptrs, (dq_prev + dq).to(tl.bfloat16), mask=offs_m[:, None] < seqlen_q)


@torch.no_grad()
def solution(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_group = cp_group or dist.group.WORLD
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    def get_uva_tensor(ptr_int, ref_t):
        return ext.make_tensor_from_ptr(int(ptr_int), list(ref_t.shape), list(ref_t.stride()))

    B, S, H, D = q.shape
    q = q.contiguous()
    out = out.contiguous()
    dout = dout.contiguous()

    # Allocate & zero Context-Parallel symmetric memory queues
    k_buf = symm_mem.empty_like(k)
    v_buf = symm_mem.empty_like(v)
    dk_buf = symm_mem.empty_like(k)
    dv_buf = symm_mem.empty_like(v)
    
    k_buf.copy_(k)
    v_buf.copy_(v)
    dk_buf.zero_()
    dv_buf.zero_()

    k_hdl_cp = symm_mem.rendezvous(k_buf, cp_group)
    v_hdl_cp = symm_mem.rendezvous(v_buf, cp_group)
    dk_hdl_cp = symm_mem.rendezvous(dk_buf, cp_group)
    dv_hdl_cp = symm_mem.rendezvous(dv_buf, cp_group)
    
    dq_buf = symm_mem.empty_like(q)
    dq_buf.zero_()

    cp_hdls = [k_hdl_cp, v_hdl_cp, dk_hdl_cp, dv_hdl_cp]
    for hdl in cp_hdls:
        hdl.barrier(channel=0)

    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    grid = (triton.cdiv(S, 64), B * H)

    # Staggered evaluation directly addressing Peer memory (no ring datapath copy)
    for step in range(cp_size):
        p = (cp_rank - step) % cp_size
        
        # In global context, chunks with p > cp_rank reside strictly in the future.
        if causal and p > cp_rank:
            continue
            
        k_remote = get_uva_tensor(k_hdl_cp.buffer_ptrs[p], k)
        v_remote = get_uva_tensor(v_hdl_cp.buffer_ptrs[p], v)
        dk_remote = get_uva_tensor(dk_hdl_cp.buffer_ptrs[p], k)
        dv_remote = get_uva_tensor(dv_hdl_cp.buffer_ptrs[p], v)
        
        is_causal_mask = (causal and p == cp_rank)
        
        bwd_kernel[grid](
            q, k_remote, v_remote, out, dout, softmax_lse,
            dq_buf, dk_remote, dv_remote,
            float(softmax_scale),
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            S, S, H,
            is_causal_mask,
            BLOCK_M=64, BLOCK_N=64, D=D
        )

    for hdl in cp_hdls:
        hdl.barrier(channel=0)

    # DP All-Reduce directly onto UVA symmetric queues
    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        dp_size = dist.get_world_size(dp_group)
        
        dq_hdl_dp = symm_mem.rendezvous(dq_buf, dp_group)
        dk_hdl_dp = symm_mem.rendezvous(dk_buf, dp_group)
        dv_hdl_dp = symm_mem.rendezvous(dv_buf, dp_group)
        
        dq_hdl_dp.barrier(channel=0)
        dk_hdl_dp.barrier(channel=0)
        dv_hdl_dp.barrier(channel=0)
        
        out_dq = torch.empty_like(dq_buf)
        out_dk = torch.empty_like(dk_buf)
        out_dv = torch.empty_like(dv_buf)
        
        ext.dp_allreduce_bf16(list(dq_hdl_dp.buffer_ptrs), out_dq, dp_size, dq_buf.numel())
        ext.dp_allreduce_bf16(list(dk_hdl_dp.buffer_ptrs), out_dk, dp_size, dk_buf.numel())
        ext.dp_allreduce_bf16(list(dv_hdl_dp.buffer_ptrs), out_dv, dp_size, dv_buf.numel())
        
        dq_hdl_dp.barrier(channel=0)
        dk_hdl_dp.barrier(channel=0)
        dv_hdl_dp.barrier(channel=0)
        
        return out_dq, out_dk, out_dv

    return dq_buf, dk_buf, dv_buf