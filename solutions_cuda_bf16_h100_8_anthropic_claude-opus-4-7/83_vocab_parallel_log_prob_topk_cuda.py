"""
Vocab-parallel log-probability with top-k/top-p filtering.

Strategy:
- Replace all-to-all with symmetric-memory: each rank writes its local vocab shard
  into a symmetric buffer, then peers directly read the slice they need via UVA
  device pointers in a single fused kernel that also transposes layout.
- Replace all_gather of target log-probs with symmetric-memory write+barrier:
  each rank writes its slice into a shared output buffer at its own offset,
  then a barrier makes results visible to all.
- Keep top-k/top-p filtering in PyTorch (small, complex, not the bottleneck),
  but fuse log_softmax + gather into a custom kernel.
"""

from typing import Optional

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

// Gather-and-transpose from peers' symmetric buffers.
// Each peer r holds [num_tokens, local_vocab] in row-major.
// We want output[t, r*local_vocab + v] = peer_r[(rank*local_tokens + t), v]
// for t in [0, local_tokens), r in [0, world_size), v in [0, local_vocab).
__global__ void gather_transpose_bf16_kernel(
    const uint64_t* __restrict__ peer_ptrs,  // [world_size]
    __nv_bfloat16* __restrict__ out,         // [local_tokens, world_size * local_vocab]
    int rank,
    int world_size,
    int local_tokens,
    int local_vocab,
    int num_tokens
) {
    // grid: (local_tokens, world_size), block over local_vocab
    int t = blockIdx.x;
    int r = blockIdx.y;
    int tid = threadIdx.x;

    if (t >= local_tokens || r >= world_size) return;

    const __nv_bfloat16* peer_buf = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[r]);
    int src_token = rank * local_tokens + t;
    const __nv_bfloat16* src_row = peer_buf + (int64_t)src_token * local_vocab;
    __nv_bfloat16* dst_row = out + (int64_t)t * world_size * local_vocab + (int64_t)r * local_vocab;

    // Vectorized copy via float4 (8 bf16 per thread)
    int vec_count = local_vocab / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src_row);
    float4* dst4 = reinterpret_cast<float4*>(dst_row);
    for (int i = tid; i < vec_count; i += blockDim.x) {
        dst4[i] = src4[i];
    }
    int tail_start = vec_count * 8;
    for (int i = tail_start + tid; i < local_vocab; i += blockDim.x) {
        dst_row[i] = src_row[i];
    }
}

void launch_gather_transpose_bf16(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor out,
    int64_t rank,
    int64_t world_size,
    int64_t local_tokens,
    int64_t local_vocab,
    int64_t num_tokens
) {
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    dim3 grid((unsigned)local_tokens, (unsigned)world_size, 1);
    gather_transpose_bf16_kernel<<<grid, threads, 0, stream>>>(
        d_ptrs,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        (int)rank, (int)world_size,
        (int)local_tokens, (int)local_vocab, (int)num_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused log_softmax + gather of target.
// logits: [N, V] (any float type, here we use float input)
// target: [N] long
// out: [N] float
// One block per row, handles V cols.
template <typename scalar_t>
__global__ void log_softmax_gather_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ target,
    float* __restrict__ out,
    int N,
    int V
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    const scalar_t* row_ptr = logits + (int64_t)row * V;
    int64_t tgt = target[row];

    // 1) max
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + bsz;

    float local_max = -INFINITY;
    for (int i = tid; i < V; i += bsz) {
        float v = (float)row_ptr[i];
        if (v > local_max) local_max = v;
    }
    smax[tid] = local_max;
    __syncthreads();
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = smax[tid], b = smax[tid + s];
            smax[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float row_max = smax[0];

    // 2) sum exp
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += bsz) {
        float v = (float)row_ptr[i];
        local_sum += expf(v - row_max);
    }
    ssum[tid] = local_sum;
    __syncthreads();
    for (int s = bsz / 2; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    float row_sum = ssum[0];
    float log_z = row_max + logf(row_sum);

    // 3) write target log-prob
    if (tid == 0) {
        float tv = (float)row_ptr[tgt];
        out[row] = tv - log_z;
    }
}

void launch_log_softmax_gather(
    torch::Tensor logits,   // [N, V] float32
    torch::Tensor target,   // [N] int64
    torch::Tensor out       // [N] float32
) {
    TORCH_CHECK(logits.dim() == 2, "logits 2d");
    int N = logits.size(0);
    int V = logits.size(1);
    int threads = 256;
    if (V < 256) {
        threads = 128;
    }
    if (V >= 1024) threads = 512;
    size_t smem = 2 * threads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (logits.dtype() == torch::kFloat32) {
        log_softmax_gather_kernel<float><<<N, threads, smem, stream>>>(
            logits.data_ptr<float>(),
            target.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            N, V);
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Copy local result tensor into a peer's symmetric output buffer at given offset.
__global__ void write_to_symm_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        dst[idx] = src[idx];
    }
}

void launch_write_to_symm(
    torch::Tensor src,
    int64_t dst_ptr,
    int64_t n
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    write_to_symm_kernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<float>(),
        reinterpret_cast<float*>(static_cast<uintptr_t>(dst_ptr)),
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_transpose_bf16", &launch_gather_transpose_bf16,
          "Gather and transpose vocab-parallel logits");
    m.def("launch_log_softmax_gather", &launch_log_softmax_gather,
          "Fused log_softmax + target gather");
    m.def("launch_write_to_symm", &launch_write_to_symm,
          "Write tensor into peer symmetric buffer");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("vocab_parallel_logprob_ext", CUDA_SRC)
    return _ext


_logits_cache = {}
_logprob_cache = {}


def _get_logits_symm(num_tokens, local_vocab, dtype, device, world_size):
    key = (num_tokens, local_vocab, dtype, device, world_size)
    if key in _logits_cache:
        return _logits_cache[key]
    buf = symm_mem.empty(num_tokens, local_vocab, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _logits_cache[key] = (buf, hdl, ptrs)
    return buf, hdl, ptrs


def _get_logprob_symm(total_tokens, device, world_size):
    key = (total_tokens, device, world_size)
    if key in _logprob_cache:
        return _logprob_cache[key]
    buf = symm_mem.empty(total_tokens, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _logprob_cache[key] = (buf, hdl, ptrs)
    return buf, hdl, ptrs


def _apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: Optional[int],
    top_p: float,
) -> torch.Tensor:
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
        keep_mask = logits_2d >= threshold
        filtered = torch.where(
            keep_mask,
            logits_2d,
            torch.full_like(logits_2d, float("-inf")),
        )
        return filtered.reshape(original_shape)

    logits_sort, logits_idx = logits_2d.sort(dim=-1, descending=False)

    top_k_mask = None
    if need_k:
        top_k_index = logits_sort.size(-1) - top_k
        threshold = logits_sort.gather(
            -1,
            torch.full(
                logits_sort.shape[:-1],
                top_k_index,
                device=logits_2d.device,
                dtype=torch.long,
            ).unsqueeze(-1),
        )
        top_k_mask = logits_sort >= threshold
        logits_sort = logits_sort.masked_fill(~top_k_mask, float("-inf"))

    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_p_mask = probs_sum > 1 - top_p
    top_p_mask[..., -1] = True
    logits_sort = logits_sort.masked_fill(~top_p_mask, float("-inf"))

    filtered = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return filtered.reshape(original_shape)


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    world_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    num_tokens = batch * seq_len

    if num_tokens % world_size != 0:
        raise ValueError(
            f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}"
        )
    local_tokens = num_tokens // world_size
    device = vocab_parallel_logits.device
    dtype = vocab_parallel_logits.dtype

    ext = _get_ext()

    logits_2d = vocab_parallel_logits.reshape(num_tokens, local_vocab).contiguous()
    target_flat = target.reshape(-1)
    target_local = target_flat[rank * local_tokens : (rank + 1) * local_tokens].contiguous()

    # 1) Write local logits into symmetric buffer
    sym_buf, sym_hdl, peer_ptrs = _get_logits_symm(
        num_tokens, local_vocab, dtype, device, world_size
    )
    sym_buf.copy_(logits_2d)
    sym_hdl.barrier(channel=0)

    # 2) Gather + transpose: read each peer's slice for our token range
    full_logits = torch.empty(
        local_tokens, world_size * local_vocab, dtype=dtype, device=device
    )
    ext.launch_gather_transpose_bf16(
        peer_ptrs, full_logits,
        rank, world_size, local_tokens, local_vocab, num_tokens
    )

    # 3) Filter (top-k/top-p) — only when needed
    filtered = _apply_top_k_top_p(full_logits, top_k=top_k, top_p=top_p)

    # 4) Fused log_softmax + gather
    filtered_f32 = filtered.to(torch.float32)
    token_logprobs = torch.empty(local_tokens, dtype=torch.float32, device=device)
    ext.launch_log_softmax_gather(filtered_f32, target_local, token_logprobs)

    # 5) All-gather via symmetric memory: each rank writes its slice into
    #    every peer's buffer at its own offset.
    lp_buf, lp_hdl, lp_peer_ptrs = _get_logprob_symm(num_tokens, device, world_size)
    lp_hdl.barrier(channel=1)

    offset_bytes = rank * local_tokens * 4  # float32
    for r in range(world_size):
        dst_ptr = int(lp_hdl.buffer_ptrs[r]) + offset_bytes
        ext.launch_write_to_symm(token_logprobs, dst_ptr, local_tokens)

    lp_hdl.barrier(channel=2)

    return lp_buf.clone().reshape(batch, seq_len)