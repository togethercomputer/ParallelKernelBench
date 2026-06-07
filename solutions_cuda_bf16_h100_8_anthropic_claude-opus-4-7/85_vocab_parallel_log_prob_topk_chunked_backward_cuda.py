import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from typing import Optional, Tuple

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------- Fused softmax+grad kernel (BF16) ----------------
// One block per row. Computes:
//   p = softmax(logits)
//   grad = (-p + onehot(target)) * grad_out * keep_mask (if any)

template <int BLOCK>
__global__ void fused_softmax_grad_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,  // [N, V]
    const int64_t* __restrict__ targets,        // [N]
    const __nv_bfloat16* __restrict__ grad_out, // [N]
    const bool* __restrict__ keep_mask,         // [N, V] or nullptr
    __nv_bfloat16* __restrict__ grad_logits,    // [N, V]
    int N,
    int V
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_logits = logits + (size_t)row * V;
    __nv_bfloat16* row_grad = grad_logits + (size_t)row * V;
    const bool* row_keep = keep_mask ? (keep_mask + (size_t)row * V) : nullptr;

    int tid = threadIdx.x;

    // 1) max
    float local_max = -INFINITY;
    for (int i = tid; i < V; i += BLOCK) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) local_max = v;
    }
    __shared__ float smax;
    typedef float scratch_t;
    __shared__ scratch_t sdata[BLOCK];
    sdata[tid] = local_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float a = sdata[tid], b = sdata[tid + off];
            sdata[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) smax = sdata[0];
    __syncthreads();
    float row_max = smax;

    // 2) sum exp
    float local_sum = 0.0f;
    for (int i = tid; i < V; i += BLOCK) {
        float v = __bfloat162float(row_logits[i]);
        if (isfinite(v)) {
            local_sum += __expf(v - row_max);
        }
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sdata[tid] += sdata[tid + off];
        __syncthreads();
    }
    __shared__ float ssum;
    if (tid == 0) ssum = sdata[0];
    __syncthreads();
    float inv_sum = 1.0f / ssum;

    int64_t tgt = targets[row];
    float go = __bfloat162float(grad_out[row]);

    // 3) write grad
    for (int i = tid; i < V; i += BLOCK) {
        float v = __bfloat162float(row_logits[i]);
        float p = isfinite(v) ? __expf(v - row_max) * inv_sum : 0.0f;
        float g = -p;
        if ((int64_t)i == tgt) g += 1.0f;
        g *= go;
        if (row_keep) {
            if (!row_keep[i]) g = 0.0f;
        }
        row_grad[i] = __float2bfloat16(g);
    }
}

void launch_fused_softmax_grad_bf16(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor grad_out,
    torch::Tensor keep_mask, // bool or empty
    torch::Tensor grad_logits,
    bool has_keep
) {
    int N = logits.size(0);
    int V = logits.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const __nv_bfloat16* lp = (const __nv_bfloat16*)logits.data_ptr<at::BFloat16>();
    const int64_t* tp = targets.data_ptr<int64_t>();
    const __nv_bfloat16* gop = (const __nv_bfloat16*)grad_out.data_ptr<at::BFloat16>();
    __nv_bfloat16* glp = (__nv_bfloat16*)grad_logits.data_ptr<at::BFloat16>();
    const bool* kmp = has_keep ? keep_mask.data_ptr<bool>() : nullptr;

    constexpr int BLOCK = 512;
    fused_softmax_grad_bf16_kernel<BLOCK><<<N, BLOCK, 0, stream>>>(
        lp, tp, gop, kmp, glp, N, V);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------- All-to-all via symmetric memory ----------------
// vp_to_seq: input is [num_tokens, local_vocab] in symm buffer
//            (num_tokens = world_size * local_tokens).
// Each rank reads from peer p: rows [rank*local_tokens : (rank+1)*local_tokens]
// from peer p's input buffer, places at output cols [p*local_vocab : (p+1)*local_vocab].
// Output shape: [local_tokens, world_size * local_vocab]

__global__ void a2a_vp_to_seq_kernel(
    const uint64_t* __restrict__ peer_in_ptrs,  // [world_size]
    __nv_bfloat16* __restrict__ out,            // [local_tokens, V_global]
    int rank,
    int world_size,
    int local_tokens,
    int local_vocab
) {
    int peer = blockIdx.y;
    int row = blockIdx.x;
    if (row >= local_tokens) return;

    const __nv_bfloat16* peer_in = (const __nv_bfloat16*)peer_in_ptrs[peer];
    // src row in peer_in: rank * local_tokens + row
    const __nv_bfloat16* src = peer_in + (size_t)(rank * local_tokens + row) * local_vocab;
    __nv_bfloat16* dst = out + (size_t)row * (world_size * local_vocab) + peer * local_vocab;

    for (int i = threadIdx.x; i < local_vocab; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// seq_to_vp: input is [local_tokens, world_size*local_vocab] in symm buffer.
// Each rank's output [world_size*local_tokens, local_vocab]: row r*local_tokens + t
// comes from peer r's input row t, columns [rank*local_vocab : (rank+1)*local_vocab].

__global__ void a2a_seq_to_vp_kernel(
    const uint64_t* __restrict__ peer_in_ptrs,  // [world_size]
    __nv_bfloat16* __restrict__ out,            // [world_size*local_tokens, local_vocab]
    int rank,
    int world_size,
    int local_tokens,
    int local_vocab,
    int v_global
) {
    int peer = blockIdx.y;
    int row = blockIdx.x;
    if (row >= local_tokens) return;

    const __nv_bfloat16* peer_in = (const __nv_bfloat16*)peer_in_ptrs[peer];
    const __nv_bfloat16* src = peer_in + (size_t)row * v_global + rank * local_vocab;
    __nv_bfloat16* dst = out + (size_t)(peer * local_tokens + row) * local_vocab;

    for (int i = threadIdx.x; i < local_vocab; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void launch_a2a_vp_to_seq(
    torch::Tensor peer_ptrs,  // int64 [world_size]
    torch::Tensor out,
    int64_t rank,
    int64_t world_size,
    int64_t local_tokens,
    int64_t local_vocab
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* pp = (const uint64_t*)peer_ptrs.data_ptr<int64_t>();
    __nv_bfloat16* op = (__nv_bfloat16*)out.data_ptr<at::BFloat16>();

    int threads = 256;
    dim3 grid(local_tokens, world_size);
    a2a_vp_to_seq_kernel<<<grid, threads, 0, stream>>>(
        pp, op, (int)rank, (int)world_size, (int)local_tokens, (int)local_vocab);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_a2a_seq_to_vp(
    torch::Tensor peer_ptrs,
    torch::Tensor out,
    int64_t rank,
    int64_t world_size,
    int64_t local_tokens,
    int64_t local_vocab
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* pp = (const uint64_t*)peer_ptrs.data_ptr<int64_t>();
    __nv_bfloat16* op = (__nv_bfloat16*)out.data_ptr<at::BFloat16>();
    int v_global = world_size * local_vocab;

    int threads = 256;
    dim3 grid(local_tokens, world_size);
    a2a_seq_to_vp_kernel<<<grid, threads, 0, stream>>>(
        pp, op, (int)rank, (int)world_size, (int)local_tokens, (int)local_vocab, v_global);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_softmax_grad_bf16", &launch_fused_softmax_grad_bf16);
    m.def("launch_a2a_vp_to_seq", &launch_a2a_vp_to_seq);
    m.def("launch_a2a_seq_to_vp", &launch_a2a_seq_to_vp);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("vocab_parallel_logprob_bwd_ext", CUDA_SRC)
    return _ext


_symm_cache = {}

def _get_symm_buf(name, numel, dtype, device, group):
    key = (name, numel, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, ptrs)
    return buf, hdl, ptrs


def _apply_top_k_top_p(logits, top_k, top_p):
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
    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    num_tokens = batch * seq_len
    chunk_tokens = batch * max(1, int(chunk_size))

    if num_tokens % world_size != 0:
        raise ValueError(f"B*S={num_tokens} must be divisible by tp size {world_size}")
    if chunk_tokens % world_size != 0:
        raise ValueError(f"B*chunk={chunk_tokens} must be divisible by tp size {world_size}")

    device = vocab_parallel_logits.device
    dtype = vocab_parallel_logits.dtype
    ext = _get_ext()

    logits_2d = vocab_parallel_logits.reshape(num_tokens, local_vocab).contiguous()
    target_flat = target.reshape(-1).contiguous()
    grad_flat = grad_output.reshape(-1).contiguous()

    v_global = world_size * local_vocab
    local_tokens = chunk_tokens // world_size

    # Symmetric buffers
    # in_buf: holds vp-layout chunk for vp_to_seq, size chunk_tokens * local_vocab
    # out_buf: holds seq-layout grad for seq_to_vp, size local_tokens * v_global
    in_buf, in_hdl, in_ptrs = _get_symm_buf(
        "in", chunk_tokens * local_vocab, dtype, device, tp_group)
    out_buf, out_hdl, out_ptrs = _get_symm_buf(
        "out", local_tokens * v_global, dtype, device, tp_group)

    # Output grad buffer
    grad_logits_full = torch.empty(num_tokens, local_vocab, dtype=dtype, device=device)
    seq_logits = torch.empty(local_tokens, v_global, dtype=dtype, device=device)
    grad_seq = torch.empty(local_tokens, v_global, dtype=dtype, device=device)

    row_ids = torch.arange(local_tokens, device=device)

    for start in range(0, num_tokens, chunk_tokens):
        end = min(start + chunk_tokens, num_tokens)
        current = end - start
        ltok = current // world_size
        target_local = target_flat[start + rank * ltok : start + (rank + 1) * ltok]
        grad_local = grad_flat[start + rank * ltok : start + (rank + 1) * ltok]

        # Stage 1: copy chunk to symm in_buf
        in_buf.view(chunk_tokens, local_vocab)[:current].copy_(logits_2d[start:end])
        in_hdl.barrier(channel=0)

        # Stage 2: vp_to_seq via symm-mem P2P kernel
        if current == chunk_tokens:
            seq_out = seq_logits
        else:
            seq_out = torch.empty(ltok, v_global, dtype=dtype, device=device)
        ext.launch_a2a_vp_to_seq(
            in_ptrs, seq_out, rank, world_size, ltok, local_vocab)

        in_hdl.barrier(channel=1)

        # Stage 3: top-k/top-p filter (CPU-style fallback for unusual cases)
        filtered, keep_mask = _apply_top_k_top_p(seq_out, top_k=top_k, top_p=top_p)

        # Stage 4: fused softmax + grad
        if current == chunk_tokens:
            gs = grad_seq
        else:
            gs = torch.empty(ltok, v_global, dtype=dtype, device=device)

        if not filtered.is_contiguous():
            filtered = filtered.contiguous()
        grad_local_bf16 = grad_local.contiguous()
        if grad_local_bf16.dtype != dtype:
            grad_local_bf16 = grad_local_bf16.to(dtype)

        if keep_mask is not None:
            km = keep_mask.contiguous()
            ext.launch_fused_softmax_grad_bf16(
                filtered, target_local.contiguous(), grad_local_bf16, km, gs, True)
        else:
            empty_km = torch.empty(0, dtype=torch.bool, device=device)
            ext.launch_fused_softmax_grad_bf16(
                filtered, target_local.contiguous(), grad_local_bf16, empty_km, gs, False)

        # Stage 5: copy gs to symm out_buf, then seq_to_vp
        out_buf.view(ltok, v_global)[:].copy_(gs)
        out_hdl.barrier(channel=0)

        chunk_out = grad_logits_full[start:end]  # [chunk_tokens, local_vocab]
        ext.launch_a2a_seq_to_vp(
            out_ptrs, chunk_out, rank, world_size, ltok, local_vocab)

        out_hdl.barrier(channel=1)

    return grad_logits_full.reshape(batch, seq_len, local_vocab)