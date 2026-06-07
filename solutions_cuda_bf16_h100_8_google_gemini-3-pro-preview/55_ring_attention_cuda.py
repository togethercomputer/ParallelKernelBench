import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// 1. Asynchronous P2P copy using UVA over NVLink
void uva_copy_async(int64_t dst_ptr, int64_t src_ptr, int64_t bytes, int64_t stream_ptr) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr),
                    reinterpret_cast<const void*>(src_ptr),
                    bytes,
                    cudaMemcpyDeviceToDevice,
                    stream);
}

// 2. Fused Scores -> P (Softmax) + LSE tracking
__global__ void fused_scores_to_p_lse_kernel(
    float* __restrict__ scores,    // [B, H, S_q, S_k]
    float* __restrict__ block_lse, // [B, S_q, H] 
    const int S_q,
    const int S_k,
    const int H,
    const int total_rows,
    const bool causal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_rows) {
        int bh_idx = idx / S_q;
        int sq_idx = idx % S_q;
        int b_idx  = bh_idx / H;
        int h_idx  = bh_idx % H;
        
        float* row = scores + bh_idx * S_q * S_k + sq_idx * S_k;
        
        float max_val = -1e20f;
        int limit = causal ? (sq_idx + 1) : S_k;
        
        for (int i = 0; i < limit; ++i) {
            float val = row[i];
            if (val > max_val) max_val = val;
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < limit; ++i) {
            float e = expf(row[i] - max_val);
            row[i] = e;
            sum_exp += e;
        }
        
        // Zero out masked components (equivalent to -inf prior to softmax)
        for (int i = limit; i < S_k; ++i) {
            row[i] = 0.0f;
        }
        
        int lse_idx = b_idx * (S_q * H) + sq_idx * H + h_idx;
        block_lse[lse_idx] = max_val + logf(sum_exp);
        
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < limit; ++i) {
            row[i] *= inv_sum;
        }
    }
}

void launch_fused_scores_to_p_lse(
    torch::Tensor scores,
    torch::Tensor block_lse,
    bool causal
) {
    int B_H = scores.size(0) * scores.size(1);
    int H = scores.size(1);
    int S_q = scores.size(2);
    int S_k = scores.size(3);
    int total_rows = B_H * S_q;
    
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_scores_to_p_lse_kernel<<<blocks, threads, 0, stream>>>(
        scores.data_ptr<float>(),
        block_lse.data_ptr<float>(),
        S_q,
        S_k,
        H,
        total_rows,
        causal
    );
}

// 3. Fused LogSumExp merge & numerically stable update
__global__ void fused_merge_out_lse_kernel(
    float* __restrict__ out,       
    float* __restrict__ lse,       
    const float* __restrict__ block_out, 
    const float* __restrict__ block_lse, 
    const int D,
    const int num_elements
) {
    int elem_idx = blockIdx.x; // maps to [B, S_q, H] structure natively
    int d_idx = threadIdx.x;
    
    if (elem_idx < num_elements) {
        float b_lse = block_lse[elem_idx];
        float c_lse = lse[elem_idx];
        
        float x_sig = b_lse - c_lse;
        float sig;
        if (x_sig >= 0.0f) {
            sig = 1.0f / (1.0f + expf(-x_sig));
        } else {
            float e = expf(x_sig);
            sig = e / (1.0f + e);
        }
        
        for (int d = d_idx; d < D; d += blockDim.x) {
            float current_out = out[elem_idx * D + d];
            float b_out = block_out[elem_idx * D + d];
            out[elem_idx * D + d] = current_out - sig * (current_out - b_out);
        }
        
        if (d_idx == 0) {
            float x = c_lse - b_lse;
            float log_sig;
            if (x >= 0.0f) {
                log_sig = -log1pf(expf(-x));
            } else {
                log_sig = x - log1pf(expf(x));
            }
            lse[elem_idx] = c_lse - log_sig;
        }
    }
}

void launch_fused_merge_out_lse(
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor block_out,
    torch::Tensor block_lse
) {
    int num_elements = out.size(0) * out.size(1) * out.size(2);
    int D = out.size(3);
    
    int threads = (D < 1024) ? D : 1024;
    threads = (threads + 31) / 32 * 32;
    
    dim3 blocks(num_elements);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_merge_out_lse_kernel<<<blocks, threads, 0, stream>>>(
        out.data_ptr<float>(),
        lse.data_ptr<float>(),
        block_out.data_ptr<float>(),
        block_lse.data_ptr<float>(),
        D,
        num_elements
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_copy_async", &uva_copy_async, "UVA async copy via NVLink");
    m.def("launch_fused_scores_to_p_lse", &launch_fused_scores_to_p_lse, "Fused max, exp, scale and sum for Attention P");
    m.def("launch_fused_merge_out_lse", &launch_fused_merge_out_lse, "Fused out/lse tracking");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attn_cp_fused", CUDA_SRC)
    return _ext

_symm_cache = {}
def get_symm_buffers(shape, dtype, device, group):
    key = (tuple(shape), dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf_k = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl_k = symm_mem.rendezvous(buf_k, group)
    buf_v = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl_v = symm_mem.rendezvous(buf_v, group)
    
    _symm_cache[key] = (buf_k, hdl_k, buf_v, hdl_v)
    return _symm_cache[key]


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
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    ext = _get_ext()
    qh = q.transpose(1, 2).float() 
    
    # Fast path: context parallel strictly restricted to single rank
    if world_size == 1:
        kh = k.transpose(1, 2).float()
        vh = v.transpose(1, 2).float()
        scores = (torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale).contiguous()
        block_lse = torch.empty(q.size(0), q.size(1), q.size(2), dtype=torch.float32, device=q.device)
        ext.launch_fused_scores_to_p_lse(scores, block_lse, causal)
        return torch.matmul(scores, vh).transpose(1, 2).contiguous().to(q.dtype)

    # Initialize device-side P2P layout and synchronicity mechanics
    buf_k, hdl_k, buf_v, hdl_v = get_symm_buffers(k.shape, k.dtype, k.device, group)
    buf_k.copy_(k)
    buf_v.copy_(v)
    hdl_k.barrier(channel=0)
    hdl_v.barrier(channel=0)
    
    # Double buffers dynamically tracking the rotation step
    local_k = [torch.empty_like(k), torch.empty_like(k)]
    local_v = [torch.empty_like(v), torch.empty_like(v)]
    local_k[0].copy_(k)
    local_v[0].copy_(v)
    
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()
    events = [torch.cuda.Event() for _ in range(world_size)]
    
    bytes_to_copy = k.numel() * k.element_size()
    out, lse = None, None
    
    for step in range(world_size):
        # 1) Prefetch step overlapping (fetch K and V from peer UVA directly to stream)
        if step + 1 < world_size:
            next_buf_idx = (step + 1) % 2
            src_rank = (rank - (step + 1)) % world_size
            
            # Ensures target memory buffer has been totally freed by prior loops' computing workload
            copy_stream.wait_stream(compute_stream)
            
            remote_k_ptr = int(hdl_k.buffer_ptrs[src_rank])
            remote_v_ptr = int(hdl_v.buffer_ptrs[src_rank])
            
            with torch.cuda.stream(copy_stream):
                ext.uva_copy_async(local_k[next_buf_idx].data_ptr(), remote_k_ptr, bytes_to_copy, copy_stream.cuda_stream)
                ext.uva_copy_async(local_v[next_buf_idx].data_ptr(), remote_v_ptr, bytes_to_copy, copy_stream.cuda_stream)
                events[step+1].record(copy_stream)
        
        # 2) Target step compute resolving
        if (not causal) or step <= rank:
            if step > 0:
                compute_stream.wait_event(events[step])
            
            curr_k = local_k[step % 2]
            curr_v = local_v[step % 2]
            
            kh = curr_k.transpose(1, 2).float()
            vh = curr_v.transpose(1, 2).float()
            
            # PyTorch `matmul` heavily utilizes float tensor-cores on Hopper.
            scores = (torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale).contiguous()
            block_lse = torch.empty(q.size(0), q.size(1), q.size(2), dtype=torch.float32, device=q.device)
            
            is_causal = causal and (step == 0)
            
            # Cuda Kernel 1: Apply mask, track max, exp scales natively on device. Overwrites scores pointer via softmax rules.
            ext.launch_fused_scores_to_p_lse(scores, block_lse, is_causal)
            
            block_out = torch.matmul(scores, vh).transpose(1, 2).contiguous()
            
            if out is None:
                out = block_out  # Pass-through avoids re-allocation
                lse = block_lse 
            else:
                # Cuda Kernel 2: Single block-pass over output space tracking LSE
                ext.launch_fused_merge_out_lse(out, lse, block_out, block_lse)
                
    return out.to(q.dtype)