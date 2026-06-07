"""
Strategy:
1. Replaced `torch.distributed.all_to_all_single` with direct peer-to-peer gathers and scatters using `torch.distributed._symmetric_memory` and NVLink.
2. Fused the gradient computation, target masking, and scatter phase into a single custom CUDA kernel (`fused_grad_scatter_kernel`). This avoids materializing and reading the large intermediate gradient sequence in device memory.
3. Implemented a chunked double-buffering pipeline with asynchronous CUDA streams and events. This hides the P2P communication and compute of the current chunk behind the D2D copying of the next/previous chunks.
4. Used vectorized memory accesses (`int4`) in the gather kernel when `local_vocab` is aligned, maximizing the Hopper architecture's memory bandwidth utilization.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional, Tuple

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__global__ void gather_vp_to_seq_kernel(
    const int64_t* __restrict__ ptrs,
    int64_t offset_elements,
    at::BFloat16* __restrict__ out,
    int local_tokens,
    int local_vocab,
    int world_size,
    int rank
) {
    int64_t total_elements = (int64_t)local_tokens * world_size * local_vocab;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int v = idx % local_vocab;
        int p = (idx / local_vocab) % world_size;
        int t = idx / (local_vocab * world_size);
        
        int peer_token_idx = rank * local_tokens + t;
        int64_t peer_offset = peer_token_idx * local_vocab + v;
        
        const at::BFloat16* peer_ptr = (const at::BFloat16*)ptrs[p] + offset_elements;
        out[idx] = peer_ptr[peer_offset];
    }
}

__global__ void gather_vp_to_seq_kernel_vec8(
    const int64_t* __restrict__ ptrs,
    int64_t offset_elements,
    at::BFloat16* __restrict__ out,
    int local_tokens,
    int local_vocab,
    int world_size,
    int rank
) {
    int64_t total_vecs = ((int64_t)local_tokens * world_size * local_vocab) / 8;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_vecs) {
        int vec_v = idx % (local_vocab / 8);
        int p = (idx / (local_vocab / 8)) % world_size;
        int t = idx / ((local_vocab / 8) * world_size);
        
        int peer_token_idx = rank * local_tokens + t;
        int64_t peer_offset = peer_token_idx * (local_vocab / 8) + vec_v;
        
        const int4* peer_ptr = (const int4*)( ((const at::BFloat16*)ptrs[p]) + offset_elements );
        int4* out_ptr = (int4*)out;
        
        out_ptr[idx] = peer_ptr[peer_offset];
    }
}

__global__ void fused_grad_scatter_kernel(
    const float* __restrict__ probs,
    const int64_t* __restrict__ target_local,
    const at::BFloat16* __restrict__ grad_local,
    const bool* __restrict__ keep_mask,
    const int64_t* __restrict__ ptrs,
    int64_t offset_elements,
    int local_tokens,
    int local_vocab,
    int world_size,
    int rank
) {
    int64_t total_elements = (int64_t)local_tokens * world_size * local_vocab;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int v = idx % local_vocab;
        int p = (idx / local_vocab) % world_size;
        int t = idx / (local_vocab * world_size);
        
        int global_v = p * local_vocab + v;
        
        float p_val = probs[idx];
        float g = -p_val;
        
        int64_t target_idx = target_local[t];
        if (target_idx < 0) {
            target_idx += (world_size * local_vocab);
        }
        
        if (global_v == target_idx) {
            g += 1.0f;
        }
        
        float grad_out_val = __bfloat162float(grad_local[t]);
        g *= grad_out_val;
        
        if (keep_mask != nullptr) {
            if (!keep_mask[idx]) {
                g = 0.0f;
            }
        }
        
        int peer_token_idx = rank * local_tokens + t;
        int64_t peer_offset = peer_token_idx * local_vocab + v;
        
        at::BFloat16* peer_ptr = (at::BFloat16*)ptrs[p] + offset_elements;
        peer_ptr[peer_offset] = __float2bfloat16(g);
    }
}

void launch_gather(
    torch::Tensor ptrs,
    int64_t offset_elements,
    torch::Tensor out,
    int local_tokens,
    int local_vocab,
    int world_size,
    int rank
) {
    int64_t total_elements = (int64_t)local_tokens * world_size * local_vocab;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (local_vocab % 8 == 0 && total_elements % 8 == 0) {
        int threads = 256;
        int blocks = (total_elements / 8 + threads - 1) / threads;
        gather_vp_to_seq_kernel_vec8<<<blocks, threads, 0, stream>>>(
            ptrs.data_ptr<int64_t>(),
            offset_elements,
            (at::BFloat16*)out.data_ptr<at::BFloat16>(),
            local_tokens, local_vocab, world_size, rank
        );
    } else {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        gather_vp_to_seq_kernel<<<blocks, threads, 0, stream>>>(
            ptrs.data_ptr<int64_t>(),
            offset_elements,
            (at::BFloat16*)out.data_ptr<at::BFloat16>(),
            local_tokens, local_vocab, world_size, rank
        );
    }
}

void launch_fused_scatter(
    torch::Tensor probs,
    torch::Tensor target_local,
    torch::Tensor grad_local,
    std::optional<torch::Tensor> keep_mask,
    torch::Tensor ptrs,
    int64_t offset_elements,
    int local_tokens,
    int local_vocab,
    int world_size,
    int rank
) {
    int64_t total_elements = (int64_t)local_tokens * world_size * local_vocab;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const bool* mask_ptr = nullptr;
    if (keep_mask.has_value()) {
        mask_ptr = keep_mask.value().data_ptr<bool>();
    }
    
    fused_grad_scatter_kernel<<<blocks, threads, 0, stream>>>(
        probs.data_ptr<float>(),
        target_local.data_ptr<int64_t>(),
        (const at::BFloat16*)grad_local.data_ptr<at::BFloat16>(),
        mask_ptr,
        ptrs.data_ptr<int64_t>(),
        offset_elements,
        local_tokens, local_vocab, world_size, rank
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather", &launch_gather);
    m.def("launch_fused_scatter", &launch_fused_scatter);
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("chunked_vp_backward_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_buffers(chunk_tokens, local_vocab, dtype, device, group):
    key = (chunk_tokens, local_vocab, dtype, device, group)
    if key in _symm_cache:
        return _symm_cache[key]
        
    symm_logits = symm_mem.empty((2, chunk_tokens, local_vocab), dtype=dtype, device=device)
    hdl_logits = symm_mem.rendezvous(symm_logits, group=group)
    logits_ptrs = torch.tensor(hdl_logits.buffer_ptrs, device=device, dtype=torch.int64)
    
    symm_grad = symm_mem.empty((2, chunk_tokens, local_vocab), dtype=dtype, device=device)
    hdl_grad = symm_mem.rendezvous(symm_grad, group=group)
    grad_ptrs = torch.tensor(hdl_grad.buffer_ptrs, device=device, dtype=torch.int64)
    
    res = (symm_logits, hdl_logits, logits_ptrs, symm_grad, hdl_grad, grad_ptrs)
    _symm_cache[key] = res
    return res

_stream_cache = {}

def _get_streams(device):
    if device not in _stream_cache:
        _stream_cache[device] = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
    return _stream_cache[device]


def _apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: Optional[int],
    top_p: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    need_k = top_k is not None and top_k > 0
    need_p = top_p is not None and top_p < 1.0
    if not need_k and not need_p:
        return logits, None

    original_shape = logits.shape
    vocab_size = logits.shape[-1]
    logits_2d = logits.reshape(-1, vocab_size)
    if need_k:
        top_k = min(int(top_k), vocab_size)

    if need_k and not need_p:
        top_k_values, _ = torch.topk(logits_2d, top_k, dim=-1)
        threshold = top_k_values[..., -1:].expand_as(logits_2d)
        keep_mask = logits_2d >= threshold
        filtered = logits_2d.masked_fill(~keep_mask, float("-inf"))
        return filtered.reshape(original_shape), keep_mask.reshape(original_shape)

    sorted_logits, sorted_idx = logits_2d.sort(dim=-1, descending=False)
    top_k_mask = None
    if need_k:
        top_k_index = sorted_logits.shape[-1] - top_k
        threshold = sorted_logits[..., top_k_index : top_k_index + 1]
        top_k_mask = sorted_logits >= threshold
        sorted_logits = sorted_logits.masked_fill(~top_k_mask, float("-inf"))

    sorted_probs = sorted_logits.softmax(dim=-1)
    top_p_mask = torch.cumsum(sorted_probs, dim=-1) > 1 - top_p
    top_p_mask[..., -1] = True
    sorted_logits = sorted_logits.masked_fill(~top_p_mask, float("-inf"))

    keep_sorted = top_p_mask if top_k_mask is None else top_p_mask & top_k_mask
    filtered = sorted_logits.scatter(dim=-1, index=sorted_idx, src=sorted_logits)
    keep_mask = keep_sorted.scatter(dim=-1, index=sorted_idx, src=keep_sorted)
    return filtered.reshape(original_shape), keep_mask.reshape(original_shape)


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    chunk_size: int = 1,
) -> torch.Tensor:
    
    tp_group = tp_group or dist.group.WORLD
    world_size = dist.get_world_size(group=tp_group)
    rank = dist.get_rank(group=tp_group)
    
    if rank == 0:
        _get_ext()
    dist.barrier(group=tp_group)
    
    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    num_tokens = batch * seq_len
    chunk_tokens = batch * max(1, int(chunk_size))

    if num_tokens % world_size != 0:
        raise ValueError(f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}")
    if chunk_tokens % world_size != 0:
        raise ValueError(f"B*chunk_size={chunk_tokens} must be divisible by tp size {world_size}")

    device = vocab_parallel_logits.device
    dtype = vocab_parallel_logits.dtype

    logits_2d = vocab_parallel_logits.contiguous().reshape(num_tokens, local_vocab)
    target_flat = target.contiguous().reshape(-1)
    grad_flat = grad_output.contiguous().reshape(-1)
    
    grad_logits_2d = torch.empty_like(logits_2d)

    symm_logits, hdl_logits, logits_ptrs, symm_grad, hdl_grad, grad_ptrs = _get_symm_buffers(
        chunk_tokens, local_vocab, dtype, device, tp_group
    )

    s_compute = torch.cuda.current_stream()
    s_copy, s_out = _get_streams(device)

    events_out = [torch.cuda.Event() for _ in range(2)]
    events_copy = [torch.cuda.Event() for _ in range(2)]
    events_gather = [torch.cuda.Event() for _ in range(2)]

    chunks = []
    for start in range(0, num_tokens, chunk_tokens):
        end = min(start + chunk_tokens, num_tokens)
        chunks.append((start, end))
    num_chunks = len(chunks)

    if num_chunks > 0:
        c0_start, c0_end = chunks[0]
        with torch.cuda.stream(s_copy):
            symm_logits[0][:c0_end - c0_start].copy_(logits_2d[c0_start:c0_end])
            events_copy[0].record(s_copy)

    for i in range(num_chunks):
        b = i % 2
        nxt_b = (i + 1) % 2
        start, end = chunks[i]
        current = end - start
        local_tokens = current // world_size
        
        events_out[b].wait(s_compute)
        events_copy[b].wait(s_compute)
        hdl_logits.barrier(channel=b)
        
        seq_logits = torch.empty((local_tokens, world_size * local_vocab), dtype=dtype, device=device)
        offset_elements = b * chunk_tokens * local_vocab
        _get_ext().launch_gather(
            logits_ptrs, offset_elements, seq_logits,
            local_tokens, local_vocab, world_size, rank
        )
        
        events_gather[b].record(s_compute)
        
        if i + 1 < num_chunks:
            nxt_start, nxt_end = chunks[i+1]
            with torch.cuda.stream(s_copy):
                events_gather[nxt_b].wait(s_copy)
                symm_logits[nxt_b][:nxt_end - nxt_start].copy_(logits_2d[nxt_start:nxt_end])
                events_copy[nxt_b].record(s_copy)
        
        filtered, keep_mask = _apply_top_k_top_p(seq_logits, top_k=top_k, top_p=top_p)
        filtered = filtered.contiguous()
        if keep_mask is not None:
            keep_mask = keep_mask.contiguous()
            
        probs = F.softmax(filtered.float(), dim=-1).contiguous()
        
        target_local = target_flat[start:end][rank * local_tokens : (rank + 1) * local_tokens]
        grad_local = grad_flat[start:end][rank * local_tokens : (rank + 1) * local_tokens]
        
        _get_ext().launch_fused_scatter(
            probs, target_local, grad_local, keep_mask,
            grad_ptrs, offset_elements,
            local_tokens, local_vocab, world_size, rank
        )
        
        hdl_grad.barrier(channel=b)
        
        event_scatter_done = torch.cuda.Event()
        event_scatter_done.record(s_compute)
        event_scatter_done.wait(s_out)
        
        with torch.cuda.stream(s_out):
            grad_logits_2d[start:end].copy_(symm_grad[b][:current])
            events_out[b].record(s_out)
            
    s_compute.wait_stream(s_out)
    return grad_logits_2d.reshape(batch, seq_len, local_vocab)