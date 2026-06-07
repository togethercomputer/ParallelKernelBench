"""
Chunked vocab-parallel target log-probability with symmetric-memory based
device-side all-to-all and all-gather (no NCCL on the hot path).
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

// Block-wise barrier using symm_mem signal pads
__device__ __forceinline__ void send_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acq_rel.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acq_rel.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void barrier_all_blocks(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t signal_slot,
    int rank,
    int world_size
) {
    if (threadIdx.x < (unsigned)world_size && blockIdx.x == 0) {
        uint64_t local_base = signal_pad_ptrs[rank];
        uint64_t remote_base = signal_pad_ptrs[threadIdx.x];
        uint32_t* send_addr = reinterpret_cast<uint32_t*>(
            remote_base + signal_slot * (uint64_t)world_size + (uint64_t)rank);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
            local_base + signal_slot * (uint64_t)world_size + (uint64_t)threadIdx.x);
        send_signal(send_addr);
        wait_signal(wait_addr);
    }
}

// All-to-all reshape kernel via peer pointers.
// Each rank has [num_tokens, local_vocab] in its symm buffer (input layout).
// World_size ranks. We read from each peer to build [local_tokens, world_size*local_vocab]
// in this rank's output.
// num_tokens = local_tokens * world_size.
//
// Output[t, r * local_vocab + v] = peer[r].input[(rank * local_tokens + t), v]
//
// We launch with grid over (local_tokens, world_size); each block copies one chunk of local_vocab.
__global__ void all_to_all_vp_to_seq_bf16_kernel(
    const uint64_t* __restrict__ peer_input_ptrs, // [world_size]
    __nv_bfloat16* __restrict__ out,              // [local_tokens, world_size * local_vocab]
    int local_tokens,
    int local_vocab,
    int world_size,
    int rank
) {
    int t = blockIdx.x;        // local_tokens
    int r = blockIdx.y;        // peer rank
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    if (t >= local_tokens || r >= world_size) return;

    const __nv_bfloat16* peer_in =
        reinterpret_cast<const __nv_bfloat16*>(peer_input_ptrs[r]);

    // Source row in peer r: row index = rank * local_tokens + t, columns = local_vocab
    int src_row = rank * local_tokens + t;
    const __nv_bfloat16* src = peer_in + (size_t)src_row * local_vocab;

    // Destination columns: r * local_vocab .. (r+1)*local_vocab
    __nv_bfloat16* dst = out + (size_t)t * (world_size * local_vocab) + r * local_vocab;

    // Vectorized copy via int4 (16B) when possible
    int v4 = local_vocab / 8;  // 8 bf16 per int4
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    for (int i = tid; i < v4; i += bsz) {
        dst4[i] = src4[i];
    }
    int rem_start = v4 * 8;
    for (int i = rem_start + tid; i < local_vocab; i += bsz) {
        dst[i] = src[i];
    }
}

void launch_all_to_all_vp_to_seq_bf16(
    torch::Tensor peer_ptrs,    // int64 [world_size]
    torch::Tensor out,          // bf16 [local_tokens, world_size*local_vocab]
    int64_t local_tokens,
    int64_t local_vocab,
    int64_t world_size,
    int64_t rank
) {
    dim3 grid((unsigned)local_tokens, (unsigned)world_size);
    int threads = 128;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>());
    all_to_all_vp_to_seq_bf16_kernel<<<grid, threads, 0, stream>>>(
        d_ptrs,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        (int)local_tokens, (int)local_vocab, (int)world_size, (int)rank
    );
}

// All-gather of float [local_tokens] from all ranks into [world_size * local_tokens]
// out[r * local_tokens + t] = peer[r].in[t]
__global__ void all_gather_f32_kernel(
    const uint64_t* __restrict__ peer_in_ptrs,  // [world_size]
    float* __restrict__ out,
    int local_tokens,
    int world_size
) {
    int r = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= world_size || tid >= local_tokens) return;
    const float* src = reinterpret_cast<const float*>(peer_in_ptrs[r]);
    out[r * local_tokens + tid] = src[tid];
}

void launch_all_gather_f32(
    torch::Tensor peer_ptrs,
    torch::Tensor out,
    int64_t local_tokens,
    int64_t world_size
) {
    int threads = 256;
    int bx = ((int)local_tokens + threads - 1) / threads;
    dim3 grid((unsigned)bx, (unsigned)world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>());
    all_gather_f32_kernel<<<grid, threads, 0, stream>>>(
        d_ptrs,
        out.data_ptr<float>(),
        (int)local_tokens, (int)world_size
    );
}

// Barrier kernel using symm_mem signal pads
__global__ void symm_barrier_kernel(
    const uint64_t* signal_pad_ptrs,
    uint64_t slot,
    int rank,
    int world_size
) {
    barrier_all_blocks(signal_pad_ptrs, slot, rank, world_size);
}

void launch_symm_barrier(
    torch::Tensor signal_ptrs,
    int64_t slot,
    int64_t rank,
    int64_t world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(signal_ptrs.data_ptr<int64_t>());
    symm_barrier_kernel<<<1, 32, 0, stream>>>(d_ptrs, (uint64_t)slot, (int)rank, (int)world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_all_to_all_vp_to_seq_bf16", &launch_all_to_all_vp_to_seq_bf16,
          "Symm-mem peer-pointer all-to-all (vocab-parallel -> seq-parallel) bf16");
    m.def("launch_all_gather_f32", &launch_all_gather_f32,
          "Symm-mem peer-pointer all-gather f32");
    m.def("launch_symm_barrier", &launch_symm_barrier,
          "Symm-mem signal-pad barrier");
}
'''


_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("vp_logprob_symm_ext", CUDA_SRC)
    return _ext


# --- symm_mem buffer cache ---------------------------------------------------

_buf_cache = {}


def _get_input_buf(num_tokens, local_vocab, device, dtype, group, slot):
    key = ("in", slot, num_tokens, local_vocab, device, dtype, id(group))
    if key in _buf_cache:
        return _buf_cache[key]
    buf = symm_mem.empty(num_tokens * local_vocab, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    sig_ptrs = torch.tensor(list(hdl.signal_pad_ptrs), device=device, dtype=torch.int64)
    _buf_cache[key] = (buf, hdl, ptrs, sig_ptrs)
    return _buf_cache[key]


def _get_lp_buf(local_tokens, device, group, slot):
    key = ("lp", slot, local_tokens, device, id(group))
    if key in _buf_cache:
        return _buf_cache[key]
    buf = symm_mem.empty(local_tokens, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    sig_ptrs = torch.tensor(list(hdl.signal_pad_ptrs), device=device, dtype=torch.int64)
    _buf_cache[key] = (buf, hdl, ptrs, sig_ptrs)
    return _buf_cache[key]


# --- top-k/top-p filtering (PyTorch, off the comm path) ----------------------

def _apply_top_k_top_p(logits, top_k, top_p):
    need_k = top_k is not None and top_k > 0
    need_p = top_p is not None and top_p < 1.0
    if not need_k and not need_p:
        return logits

    vocab_size = logits.shape[-1]
    if need_k:
        top_k = min(int(top_k), vocab_size)

    if need_k and not need_p:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = top_k_values[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))

    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=False)
    if need_k:
        top_k_index = sorted_logits.shape[-1] - top_k
        threshold = sorted_logits[..., top_k_index : top_k_index + 1]
        sorted_logits = sorted_logits.masked_fill(sorted_logits < threshold, float("-inf"))

    sorted_probs = sorted_logits.softmax(dim=-1)
    top_p_mask = torch.cumsum(sorted_probs, dim=-1) > 1 - top_p
    top_p_mask[..., -1] = True
    sorted_logits = sorted_logits.masked_fill(~top_p_mask, float("-inf"))
    filtered = sorted_logits.scatter(dim=-1, index=sorted_idx, src=sorted_logits)
    return filtered


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
        raise ValueError(
            f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}"
        )
    if chunk_tokens % world_size != 0:
        raise ValueError(
            f"B*chunk_size={chunk_tokens} must be divisible by tp size {world_size}"
        )

    device = vocab_parallel_logits.device
    dtype = vocab_parallel_logits.dtype

    # Compile/load the extension on rank 0 first, then everyone.
    if rank == 0:
        _get_ext()
    dist.barrier(group=tp_group)
    ext = _get_ext()

    logits_2d = vocab_parallel_logits.reshape(num_tokens, local_vocab).contiguous()
    target_flat = target.reshape(-1)

    # Output buffer (full sequence log-probs)
    out_full = torch.empty(num_tokens, device=device, dtype=torch.float32)

    # Two-slot pipelining (double-buffered comm staging)
    n_slots = 2
    slot = 0

    # Two streams: one for comm staging (peer-pointer copies), one default for compute
    comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    starts = list(range(0, num_tokens, chunk_tokens))

    # Kick off: each chunk needs (a) write logits into symm input buf,
    # (b) barrier so peers can read, (c) all_to_all kernel into local seq buf,
    # (d) compute filter+logsoftmax+gather, (e) write to symm lp buf,
    # (f) barrier, (g) all_gather kernel, (h) write into out_full.

    for i, start in enumerate(starts):
        end = min(start + chunk_tokens, num_tokens)
        current = end - start
        local_tokens = current // world_size
        target_local = target_flat[
            start + rank * local_tokens : start + (rank + 1) * local_tokens
        ]

        in_buf, in_hdl, in_ptrs, in_sig = _get_input_buf(
            current, local_vocab, device, dtype, tp_group, slot
        )
        lp_buf, lp_hdl, lp_ptrs, lp_sig = _get_lp_buf(
            local_tokens, device, tp_group, slot
        )

        # Stage logits into symmetric input buffer on compute stream
        in_buf_view = in_buf.view(current, local_vocab)
        in_buf_view.copy_(logits_2d[start:end])

        # Barrier across ranks (device-side via signal pads), serialized through compute stream
        ext.launch_symm_barrier(in_sig, 2 * i, rank, world_size)

        # Peer-direct all-to-all into a local seq-parallel tensor
        seq_logits = torch.empty(
            (local_tokens, world_size * local_vocab), device=device, dtype=dtype
        )
        ext.launch_all_to_all_vp_to_seq_bf16(
            in_ptrs, seq_logits, local_tokens, local_vocab, world_size, rank
        )

        # Compute (filter + log_softmax + target gather) — pure compute path
        filtered = _apply_top_k_top_p(seq_logits, top_k=top_k, top_p=top_p)
        log_probs = F.log_softmax(filtered.float(), dim=-1)
        local_logprobs = torch.gather(
            log_probs, -1, target_local.unsqueeze(-1).long()
        ).squeeze(-1)

        # Stage into symm lp buffer
        lp_buf.copy_(local_logprobs)

        # Barrier then peer-direct all-gather into out_full slice
        ext.launch_symm_barrier(lp_sig, 2 * i + 1, rank, world_size)

        out_slice = out_full[start:end]  # length = current = world_size * local_tokens
        ext.launch_all_gather_f32(lp_ptrs, out_slice, local_tokens, world_size)

        slot = (slot + 1) % n_slots

    return out_full.reshape(batch, seq_len)