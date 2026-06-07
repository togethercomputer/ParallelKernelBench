import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Kernel 1: Push logits directly to peers' symmetric memory (All-to-All + Permute)
// ---------------------------------------------------------------------------
__global__ void push_logits_kernel_vec(
    const __nv_bfloat16* __restrict__ local_logits, 
    const uint64_t* __restrict__ recv_buf_ptrs,     
    int current_tokens,
    int local_vocab,
    int world_size,
    int my_rank
) {
    int local_tokens = current_tokens / world_size;
    int total_sends = world_size * local_tokens;
    
    int send_idx = blockIdx.x;
    if (send_idx >= total_sends) return;
    
    int dst_rank = send_idx / local_tokens;
    int t_local = send_idx % local_tokens;
    
    int t_chunk = dst_rank * local_tokens + t_local;
    
    const __nv_bfloat16* src = local_logits + t_chunk * local_vocab;
    // Destination offset inherently accomplishes the sequence interleave permute
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(recv_buf_ptrs[dst_rank]) 
                         + t_local * (world_size * local_vocab) 
                         + my_rank * local_vocab;
                         
    if (local_vocab % 8 == 0) {
        int num_vec = local_vocab / 8;
        const uint4* src_vec = reinterpret_cast<const uint4*>(src);
        uint4* dst_vec = reinterpret_cast<uint4*>(dst);
        for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
            dst_vec[i] = src_vec[i];
        }
    } else if (local_vocab % 4 == 0) {
        int num_vec = local_vocab / 4;
        const uint64_t* src_vec = reinterpret_cast<const uint64_t*>(src);
        uint64_t* dst_vec = reinterpret_cast<uint64_t*>(dst);
        for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        for (int i = threadIdx.x; i < local_vocab; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Fused log_softmax, target gather, and All-Gather push
// ---------------------------------------------------------------------------
__global__ void fused_log_softmax_gather_push_kernel(
    const __nv_bfloat16* __restrict__ filtered_logits,
    const int64_t* __restrict__ target_local,
    const uint64_t* __restrict__ logprobs_buf_ptrs,
    int local_tokens,
    int full_vocab,
    int world_size,
    int my_rank
) {
    int t_local = blockIdx.x;
    if (t_local >= local_tokens) return;
    
    int64_t target_idx = target_local[t_local];
    const __nv_bfloat16* logits = filtered_logits + t_local * full_vocab;
    
    // 1. Thread-local max
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < full_vocab; i += blockDim.x) {
        float val = __bfloat162float(logits[i]);
        if (val > thread_max) thread_max = val;
    }
    
    // 2. Block-wide max reduction
    static __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    
    float warp_max = thread_max;
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
    }
    if (lane == 0) shared_max[warp] = warp_max;
    __syncthreads();
    
    float block_max = -1e20f;
    if (threadIdx.x < (blockDim.x / 32)) {
        block_max = shared_max[threadIdx.x];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
    }
    if (threadIdx.x == 0) shared_max[0] = block_max;
    __syncthreads();
    block_max = shared_max[0];
    
    // 3. Thread-local sum and target extraction
    float thread_sum = 0.0f;
    float target_val = 0.0f;
    for (int i = threadIdx.x; i < full_vocab; i += blockDim.x) {
        float val = __bfloat162float(logits[i]);
        if (i == target_idx) target_val = val;
        thread_sum += expf(val - block_max);
    }
    
    // 4. Block-wide sum reduction
    static __shared__ float shared_sum[32];
    static __shared__ float shared_target[32];
    
    float warp_sum = thread_sum;
    float warp_target = target_val;
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        warp_target += __shfl_down_sync(0xffffffff, warp_target, offset);
    }
    if (lane == 0) {
        shared_sum[warp] = warp_sum;
        shared_target[warp] = warp_target;
    }
    __syncthreads();
    
    float block_sum = 0.0f;
    float block_target = 0.0f;
    if (threadIdx.x < (blockDim.x / 32)) {
        block_sum = shared_sum[threadIdx.x];
        block_target = shared_target[threadIdx.x];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        block_target += __shfl_down_sync(0xffffffff, block_target, offset);
    }
    if (threadIdx.x == 0) shared_sum[0] = block_sum;
    if (threadIdx.x == 0) shared_target[0] = block_target;
    __syncthreads();
    
    block_sum = shared_sum[0];
    block_target = shared_target[0];
    
    // 5. Final log-prob logic and P2P All-Gather
    if (threadIdx.x == 0) {
        float log_prob = block_target - (block_max + logf(block_sum));
        int global_token_idx = my_rank * local_tokens + t_local;
        
        for (int dst = 0; dst < world_size; ++dst) {
            float* dst_ptr = reinterpret_cast<float*>(logprobs_buf_ptrs[dst]);
            dst_ptr[global_token_idx] = log_prob;
        }
    }
}

// ---------------------------------------------------------------------------
// Python Bindings
// ---------------------------------------------------------------------------
void launch_push_logits(
    torch::Tensor local_logits,
    torch::Tensor recv_buf_ptrs,
    int current_tokens,
    int local_vocab,
    int world_size,
    int my_rank
) {
    int local_tokens = current_tokens / world_size;
    int total_sends = world_size * local_tokens;
    if (total_sends == 0) return;
    
    int threads = 256;
    int blocks = total_sends;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    push_logits_kernel_vec<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(local_logits.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint64_t*>(recv_buf_ptrs.data_ptr<int64_t>()),
        current_tokens,
        local_vocab,
        world_size,
        my_rank
    );
}

void launch_fused_log_softmax_gather_push(
    torch::Tensor filtered_logits,
    torch::Tensor target_local,
    torch::Tensor logprobs_buf_ptrs,
    int local_tokens,
    int full_vocab,
    int world_size,
    int my_rank
) {
    if (local_tokens == 0) return;
    
    int threads = 256;
    int blocks = local_tokens;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    fused_log_softmax_gather_push_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(filtered_logits.data_ptr<at::BFloat16>()),
        target_local.data_ptr<int64_t>(),
        reinterpret_cast<const uint64_t*>(logprobs_buf_ptrs.data_ptr<int64_t>()),
        local_tokens,
        full_vocab,
        world_size,
        my_rank
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_push_logits", &launch_push_logits);
    m.def("launch_fused_log_softmax_gather_push", &launch_fused_log_softmax_gather_push);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("vocab_parallel_logprob_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_buffers(chunk_tokens, local_vocab, world_size, dtype, device, group):
    key = (chunk_tokens, local_vocab, world_size, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    max_local_tokens = chunk_tokens // world_size
    full_vocab = local_vocab * world_size
    
    recv_buf = symm_mem.empty((max_local_tokens, full_vocab), dtype=dtype, device=device)
    recv_hdl = symm_mem.rendezvous(recv_buf, group=group)
    recv_ptrs = torch.tensor(recv_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    logprobs_buf = symm_mem.empty((chunk_tokens,), dtype=torch.float32, device=device)
    logprobs_hdl = symm_mem.rendezvous(logprobs_buf, group=group)
    logprobs_ptrs = torch.tensor(logprobs_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    res = (recv_buf, recv_hdl, recv_ptrs, logprobs_buf, logprobs_hdl, logprobs_ptrs)
    _symm_cache[key] = res
    return res

def _apply_top_k_top_p(logits: torch.Tensor, top_k: Optional[int], top_p: float) -> torch.Tensor:
    need_k = top_k is not None and top_k > 0
    need_p = top_p is not None and top_p < 1.0
    if not need_k and not need_p:
        return logits

    original_shape = logits.shape
    vocab_size = logits.shape[-1]
    logits_2d = logits.reshape(-1, vocab_size)
    
    if need_k:
        top_k = min(int(top_k), vocab_size)

    if need_k and not need_p:
        top_k_values, _ = torch.topk(logits_2d, top_k, dim=-1)
        threshold = top_k_values[..., -1:].expand_as(logits_2d)
        filtered = logits_2d.masked_fill(logits_2d < threshold, float("-inf"))
        return filtered.reshape(original_shape)

    sorted_logits, sorted_idx = logits_2d.sort(dim=-1, descending=False)
    
    if need_k:
        top_k_index = sorted_logits.shape[-1] - top_k
        threshold = sorted_logits[..., top_k_index : top_k_index + 1]
        sorted_logits = sorted_logits.masked_fill(sorted_logits < threshold, float("-inf"))

    sorted_probs = sorted_logits.softmax(dim=-1)
    top_p_mask = torch.cumsum(sorted_probs, dim=-1) > 1 - top_p
    top_p_mask[..., -1] = True
    sorted_logits = sorted_logits.masked_fill(~top_p_mask, float("-inf"))
    filtered = sorted_logits.scatter(dim=-1, index=sorted_idx, src=sorted_logits)
    return filtered.reshape(original_shape)


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    chunk_size: int = 1,
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    world_size = dist.get_world_size(group=tp_group)
    rank = dist.get_rank(group=tp_group)
    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    
    num_tokens = batch * seq_len
    chunk_tokens = batch * max(1, int(chunk_size))

    if num_tokens % world_size != 0:
        raise ValueError(f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}")
    if chunk_tokens % world_size != 0:
        raise ValueError(f"B*chunk_size={chunk_tokens} must be divisible by tp size {world_size}")

    ext = _get_ext()
    
    recv_buf, recv_hdl, recv_ptrs, logprobs_buf, logprobs_hdl, logprobs_ptrs = _get_symm_buffers(
        chunk_tokens, local_vocab, world_size, vocab_parallel_logits.dtype, vocab_parallel_logits.device, tp_group
    )

    logits_2d = vocab_parallel_logits.reshape(num_tokens, local_vocab)
    target_flat = target.reshape(-1)
    
    full_vocab = local_vocab * world_size
    pieces = []

    for start in range(0, num_tokens, chunk_tokens):
        end = min(start + chunk_tokens, num_tokens)
        current = end - start
        local_tokens = current // world_size
        
        logits_chunk = logits_2d[start:end].contiguous()
        target_chunk = target_flat[start:end]
        target_local = target_chunk[rank * local_tokens : (rank + 1) * local_tokens].contiguous().long()

        # Step 1: Push symmetric UVA vocabulary transpose
        ext.launch_push_logits(
            logits_chunk, recv_ptrs, current, local_vocab, world_size, rank
        )
        
        # Synchronize streams across TP-domain before relying on fetched symmetric data
        recv_hdl.barrier(channel=0)
        
        # Step 2: Extract locally active sequence chunk and apply PyTorch complex filtering
        seq_logits = recv_buf[:local_tokens]
        filtered = _apply_top_k_top_p(seq_logits, top_k=top_k, top_p=top_p)
        
        # Step 3: Fast fused reduction yielding target token prob -> broadcast directly to peer buffers
        ext.launch_fused_log_softmax_gather_push(
            filtered, target_local, logprobs_ptrs, local_tokens, full_vocab, world_size, rank
        )
        
        # Ensure block reductions are flushed globally safely
        logprobs_hdl.barrier(channel=0)
        
        pieces.append(logprobs_buf[:current].clone())

    return torch.cat(pieces, dim=0).reshape(batch, seq_len)