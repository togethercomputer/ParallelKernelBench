"""
MoE forward with expert LoRA, using symmetric-memory backed all-to-all
and all-gather primitives. Replaces NCCL collectives on the hot path with
custom CUDA kernels that read/write peer buffers directly via UVA pointers.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---- signal pad barrier ----
__device__ __forceinline__ void send_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void barrier_block(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal(send_addr);
    wait_signal(wait_addr);
}

// ---- all-gather of int64 tokens ----
__global__ void allgather_int64_kernel(
    const uint64_t* __restrict__ buffer_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t* __restrict__ out,
    int64_t local_n,
    int rank,
    int world_size
) {
    barrier_block(signal_pad_ptrs, blockIdx.x, rank, world_size);
    __syncthreads();

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = local_n * world_size;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < total; i += stride) {
        int peer = (int)(i / local_n);
        int64_t off = i - (int64_t)peer * local_n;
        const int64_t* src = reinterpret_cast<const int64_t*>(buffer_ptrs[peer]);
        out[i] = src[off];
    }

    __syncthreads();
    barrier_block(signal_pad_ptrs, gridDim.x + blockIdx.x, rank, world_size);
}

// ---- all-to-all variable, BF16 rows of fixed hidden ----
// Each rank already wrote its full send buffer (all peers contiguous) into
// its own symmetric buffer; segment for peer p lies at [send_off[p], send_off[p]+send_cnt[p]).
// The symmetric layout: rank r's symbuf at row offset send_off_to_peer (computed from
// input_splits) contains the rows destined for peer p. We let each rank PULL its data
// from peers using their published offsets.
//
// Simpler: every rank places at its symbuf[ peer_send_offsets[p] : ... ] the rows for peer p.
// Receiver r reads: for each peer p, rows from peer p's symbuf[peer_p.send_off_for_r : ... ]
// We pass per-peer per-rank src offsets.

__global__ void all2all_pull_bf16_kernel(
    const uint64_t* __restrict__ buffer_ptrs,    // peer symbuf base (BF16 rows, hidden cols)
    const uint64_t* __restrict__ signal_pad_ptrs,
    __nv_bfloat16* __restrict__ out,             // [total_recv, hidden]
    const int64_t* __restrict__ recv_offsets,    // [world_size+1] rec write offsets
    const int64_t* __restrict__ src_offsets,     // [world_size] per-peer src row offset
    int hidden,
    int world_size,
    int rank
) {
    barrier_block(signal_pad_ptrs, blockIdx.x, rank, world_size);
    __syncthreads();

    int peer = blockIdx.y;
    int64_t recv_start = recv_offsets[peer];
    int64_t recv_end = recv_offsets[peer + 1];
    int64_t n_rows = recv_end - recv_start;
    if (n_rows <= 0) {
        __syncthreads();
        barrier_block(signal_pad_ptrs, gridDim.x + blockIdx.x, rank, world_size);
        return;
    }

    int64_t src_start = src_offsets[peer];
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(buffer_ptrs[peer]);
    __nv_bfloat16* dst = out + recv_start * hidden;

    int64_t total = n_rows * (int64_t)hidden;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // vectorized via int4 (8 bf16 per int4)
    int64_t total_v = total / 8;
    const int4* src_v = reinterpret_cast<const int4*>(src + src_start * hidden);
    int4* dst_v = reinterpret_cast<int4*>(dst);
    for (int64_t i = tid; i < total_v; i += stride) {
        dst_v[i] = src_v[i];
    }
    int64_t tail_start = total_v * 8;
    for (int64_t i = tail_start + tid; i < total; i += stride) {
        dst[i] = src[(int64_t)src_start * hidden + i];
    }

    __syncthreads();
    barrier_block(signal_pad_ptrs, gridDim.x + blockIdx.x, rank, world_size);
}

void launch_allgather_int64(
    uint64_t buffer_ptrs_dev,
    uint64_t signal_pad_ptrs_dev,
    torch::Tensor out,
    int64_t local_n,
    int rank,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 64;
    if (threads < world_size) threads = world_size;
    int blocks = 1;
    allgather_int64_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(buffer_ptrs_dev),
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_dev),
        out.data_ptr<int64_t>(),
        local_n, rank, world_size);
}

void launch_all2all_pull_bf16(
    uint64_t buffer_ptrs_dev,
    uint64_t signal_pad_ptrs_dev,
    torch::Tensor out,
    torch::Tensor recv_offsets,
    torch::Tensor src_offsets,
    int64_t hidden,
    int world_size,
    int rank
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    if (threads < world_size) threads = world_size;
    int blocks_x = 32;
    dim3 grid(blocks_x, world_size, 1);
    all2all_pull_bf16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(buffer_ptrs_dev),
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_dev),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        recv_offsets.data_ptr<int64_t>(),
        src_offsets.data_ptr<int64_t>(),
        (int)hidden, world_size, rank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_allgather_int64", &launch_allgather_int64);
    m.def("launch_all2all_pull_bf16", &launch_all2all_pull_bf16);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_lora_symm_ext", CUDA_SRC)
    return _ext


# Symmetric memory caches
_ag_cache = {}
def _get_ag_buf(local_n: int, device, dtype=torch.int64):
    key = (local_n, dtype, device)
    if key in _ag_cache:
        return _ag_cache[key]
    buf = symm_mem.empty(local_n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _ag_cache[key] = (buf, hdl)
    return buf, hdl


_a2a_cache = {}
def _get_a2a_buf(max_rows: int, hidden: int, device, dtype=torch.bfloat16):
    key = (hidden, dtype, device)
    if key in _a2a_cache:
        buf, hdl, cap = _a2a_cache[key]
        if cap >= max_rows:
            return buf, hdl, cap
    cap = max(max_rows, 1024)
    cap = max(cap, _a2a_cache.get(key, (None, None, 0))[2] * 2 if key in _a2a_cache else cap)
    buf = symm_mem.empty((cap, hidden), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _a2a_cache[key] = (buf, hdl, cap)
    return buf, hdl, cap


def _custom_allgather_into_tensor(local: torch.Tensor, group) -> torch.Tensor:
    ws = dist.get_world_size(group)
    n = local.numel()
    buf, hdl = _get_ag_buf(n, local.device, local.dtype)
    buf.copy_(local.view(-1))
    out = torch.empty(n * ws, dtype=local.dtype, device=local.device)
    _get_ext().launch_allgather_int64(
        int(hdl.buffer_ptrs_dev),
        int(hdl.signal_pad_ptrs_dev),
        out, n, hdl.rank, hdl.world_size,
    )
    return out


def _custom_all_to_all_bf16(
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group,
) -> torch.Tensor:
    """input: [sum(input_splits), hidden] bf16. Returns [sum(output_splits), hidden]."""
    ws = dist.get_world_size(group)
    rank = dist.get_rank(group)
    hidden = input.size(1)
    device = input.device

    total_send = int(sum(input_split_sizes))
    total_recv = int(sum(output_split_sizes))

    # Compute send offsets on this rank (for placing my chunks for each peer)
    send_offsets = [0]
    for s in input_split_sizes:
        send_offsets.append(send_offsets[-1] + int(s))
    # send_offsets[p] = where rank's data destined to peer p starts in symbuf

    # We need src_offsets[peer p] = offset in peer p's symbuf where peer p stored data destined to me (rank)
    # Each rank's send_offsets[rank] gives that.
    # We need to gather all ranks' send_offsets and pick column = rank.
    # send_offsets has ws+1 entries; per-peer we need send_offsets_of_peer[rank].
    # Use input_split_sizes table: ag(input_splits) -> matrix [ws, ws] where row=src, col=dst.
    # Then for receiver = rank, src_offset[peer p] = sum_{q<rank} mat[p, q].

    # Local input_splits as int64 tensor of length ws
    local_splits = torch.tensor(input_split_sizes, dtype=torch.int64, device=device)
    # all-gather to [ws*ws]
    gathered = _custom_allgather_into_tensor(local_splits, group).view(ws, ws)
    # gathered[p, q] = peer p sends to peer q
    # src_offsets[p] = sum_{q<rank} gathered[p, q]
    src_offsets = gathered[:, :rank].sum(dim=1).contiguous()

    # recv_offsets on this rank (output side)
    recv_offsets_list = [0]
    for s in output_split_sizes:
        recv_offsets_list.append(recv_offsets_list[-1] + int(s))
    recv_offsets = torch.tensor(recv_offsets_list, dtype=torch.int64, device=device)

    # Place my input into symbuf at positions [send_offsets[p] : send_offsets[p+1]] for peer p.
    # Since input is already in that order (concat per peer), just copy whole.
    buf, hdl, cap = _get_a2a_buf(total_send, hidden, device, torch.bfloat16)
    if total_send > 0:
        buf[:total_send].copy_(input)

    out = torch.empty((total_recv, hidden), dtype=torch.bfloat16, device=device)

    _get_ext().launch_all2all_pull_bf16(
        int(hdl.buffer_ptrs_dev),
        int(hdl.signal_pad_ptrs_dev),
        out,
        recv_offsets,
        src_offsets,
        hidden,
        ws,
        rank,
    )
    return out


# ---------- Reference helpers (rewritten to use custom comm) ----------

def _preprocess(expert_mask, num_experts, ep_group):
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))
    input_splits = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    )
    flat = num_local_tokens_per_expert.contiguous().view(-1).to(torch.int64)
    gathered = _custom_allgather_into_tensor(flat, ep_group)
    num_global_tokens_per_expert = gathered.view(ep_size, flat.numel())
    s, e = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, s:e].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    num_global_sum = num_global_tokens_per_local_expert.sum(dim=0).cpu()
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
        -1, num_local_experts
    ).cpu()
    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum


def _permute(tokens, routing_map):
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    permuted = tokens.index_select(0, sorted_indices)
    return permuted, sorted_indices


def _sort_chunks_by_idxs(input, split_sizes, sorted_idxs):
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()
    chunks = torch.split(input, split_sizes, dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)


def _generate_weights_idx(routing_weights, selected_experts, num_experts):
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros((num_tokens, num_experts),
                              dtype=routing_weights.dtype, device=routing_weights.device)
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def _unpermute(tokens, routing_weights, hidden_states_shape, permutation_mapping, routing_map):
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    unp = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    expanded = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unp.scatter_add_(0, expanded, tokens)
    return unp


def token_pre_all2all(hidden_states, expert_mask, num_experts,
                     input_splits, output_splits,
                     num_global_tokens_per_local_expert, group):
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)
    local_perm, local_map = _permute(hidden_states, routing_map)

    # custom a2a on bf16
    if local_perm.dtype == torch.bfloat16:
        global_perm = _custom_all_to_all_bf16(
            local_perm.contiguous(), output_splits, input_splits, group
        )
    else:
        # fallback
        out = torch.empty((sum(output_splits), hidden_dim),
                          dtype=local_perm.dtype, device=local_perm.device)
        dist.all_to_all_single(out, local_perm.contiguous(),
                               output_split_sizes=output_splits,
                               input_split_sizes=input_splits, group=group)
        global_perm = out

    num_local_experts = num_experts // dist.get_world_size(group)
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    split_sizes = num_global_tokens_per_local_expert.ravel().tolist()
    global_perm = _sort_chunks_by_idxs(global_perm, split_sizes, permute_order)
    return global_perm, routing_map, local_map, org_shape


def tokens_post_all2all(expert_outputs, routing_weights, selected_experts, num_experts,
                        input_splits, output_splits, num_global_tokens_per_local_expert,
                        routing_map, local_input_permutation_mapping, org_shape, group):
    num_local_experts = num_experts // dist.get_world_size(group)
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist()
    expert_outputs = _sort_chunks_by_idxs(expert_outputs, split_sizes, unpermute_order)

    if expert_outputs.dtype == torch.bfloat16:
        unp = _custom_all_to_all_bf16(
            expert_outputs.contiguous(), input_splits, output_splits, group
        )
    else:
        unp = torch.empty((sum(input_splits), expert_outputs.size(1)),
                          dtype=expert_outputs.dtype, device=expert_outputs.device)
        dist.all_to_all_single(unp, expert_outputs.contiguous(),
                               output_split_sizes=input_splits,
                               input_split_sizes=output_splits, group=group)

    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    return _unpermute(unp, weights_idx, org_shape, local_input_permutation_mapping, routing_map)


def expert_forward_lora(x, gate_proj, up_proj, down_proj,
                         lora_gate_A, lora_gate_B, lora_up_A, lora_up_B,
                         lora_down_A, lora_down_B):
    gate_proj.to(x.dtype)
    up_proj.to(x.dtype)
    down_proj.to(x.dtype)
    lora_gate_A = lora_gate_A.to(x.dtype)
    lora_gate_B = lora_gate_B.to(x.dtype)
    lora_up_A = lora_up_A.to(x.dtype)
    lora_up_B = lora_up_B.to(x.dtype)
    lora_down_A = lora_down_A.to(x.dtype)
    lora_down_B = lora_down_B.to(x.dtype)
    F = torch.nn.functional
    xa_g = F.linear(x, lora_gate_A)
    gate_x = gate_proj(x) + F.linear(xa_g, lora_gate_B)
    gate = F.silu(gate_x)
    xa_u = F.linear(x, lora_up_A)
    up = up_proj(x) + F.linear(xa_u, lora_up_B)
    y = gate * up
    xa_d = F.linear(y, lora_down_A)
    return down_proj(y) + F.linear(xa_d, lora_down_B)


def solution(
    hidden_states, gate_weight, gate_bias,
    gate_proj, up_proj, down_proj,
    lora_gate_A, lora_gate_B, lora_up_A, lora_up_B, lora_down_A, lora_down_B,
    num_experts, top_k, group=None,
):
    group = group or dist.group.WORLD
    # Pre-compile extension on rank 0 then sync
    if dist.get_rank(group) == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()

    hidden_dim = hidden_states.size(-1)
    flat = hidden_states.reshape(-1, hidden_dim)

    router_logits = torch.nn.functional.linear(flat, gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

    input_splits, output_splits, num_global_tokens_per_local_expert, _ = _preprocess(
        expert_mask, num_experts, group
    )

    (global_perm, routing_map, local_map, org_shape) = token_pre_all2all(
        hidden_states, expert_mask, num_experts,
        input_splits, output_splits, num_global_tokens_per_local_expert, group,
    )

    expert_outputs = expert_forward_lora(
        global_perm, gate_proj, up_proj, down_proj,
        lora_gate_A, lora_gate_B, lora_up_A, lora_up_B, lora_down_A, lora_down_B,
    )

    out = tokens_post_all2all(
        expert_outputs, routing_weights, selected_experts, num_experts,
        input_splits, output_splits, num_global_tokens_per_local_expert,
        routing_map, local_map, org_shape, group,
    )
    return out