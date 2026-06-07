"""
Problem 51: MoE EP-wide forward, with custom all-to-all over symmetric memory.

Strategy:
- Replace dist.all_to_all_single with a symm_mem-based all-to-all that uses
  device-side P2P writes through UVA peer pointers.
- Replace dist.all_gather_into_tensor (for split metadata) with a symm_mem
  broadcast: each rank writes its small split vector into a symmetric buffer
  and peers read it directly.
- Keep Python-level orchestration to preserve correctness (autograd and the
  reference op mix), but shove the hot collective path onto direct device
  pointer copies.
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
#include <cstdint>

// Copy local rows into per-peer slots of peers' symmetric buffers.
// For each peer p, write input[input_offset[p] : input_offset[p]+input_split[p]]
// to peer_bufs[p] at row offset slot_offset[p] (which is the offset where
// THIS rank's contribution lives on peer p).
__global__ void a2a_scatter_kernel(
    const __nv_bfloat16* __restrict__ input,  // local permuted input
    const long long* __restrict__ peer_bufs,  // [world_size] device ptrs
    const int* __restrict__ input_offsets,    // [world_size]
    const int* __restrict__ input_splits,     // [world_size]
    const int* __restrict__ peer_slot_offsets, // [world_size]: row offset on peer p
    int hidden_dim,
    int world_size
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;

    int rows = input_splits[peer];
    if (rows == 0) return;
    int in_row_off = input_offsets[peer];
    int peer_row_off = peer_slot_offsets[peer];

    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(peer_bufs[peer]);
    const __nv_bfloat16* src = input + (int64_t)in_row_off * hidden_dim;
    __nv_bfloat16* dst_base = dst + (int64_t)peer_row_off * hidden_dim;

    int64_t total = (int64_t)rows * hidden_dim;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // 8 bytes per thread = 4 bf16 at a time when aligned
    for (int64_t i = tid; i < total; i += stride) {
        dst_base[i] = src[i];
    }
}

// Float32 variant
__global__ void a2a_scatter_kernel_f32(
    const float* __restrict__ input,
    const long long* __restrict__ peer_bufs,
    const int* __restrict__ input_offsets,
    const int* __restrict__ input_splits,
    const int* __restrict__ peer_slot_offsets,
    int hidden_dim,
    int world_size
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;

    int rows = input_splits[peer];
    if (rows == 0) return;
    int in_row_off = input_offsets[peer];
    int peer_row_off = peer_slot_offsets[peer];

    float* dst = reinterpret_cast<float*>(peer_bufs[peer]);
    const float* src = input + (int64_t)in_row_off * hidden_dim;
    float* dst_base = dst + (int64_t)peer_row_off * hidden_dim;

    int64_t total = (int64_t)rows * hidden_dim;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < total; i += stride) {
        dst_base[i] = src[i];
    }
}

void launch_a2a_scatter_bf16(
    torch::Tensor input,
    torch::Tensor peer_bufs,        // int64 [world_size]
    torch::Tensor input_offsets,    // int32 [world_size]
    torch::Tensor input_splits,     // int32 [world_size]
    torch::Tensor peer_slot_offsets // int32 [world_size]
) {
    int world_size = (int)peer_bufs.numel();
    int hidden_dim = (int)input.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 block(256);
    dim3 grid(64, world_size);
    a2a_scatter_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const long long*>(peer_bufs.data_ptr<int64_t>()),
        input_offsets.data_ptr<int>(),
        input_splits.data_ptr<int>(),
        peer_slot_offsets.data_ptr<int>(),
        hidden_dim,
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_a2a_scatter_f32(
    torch::Tensor input,
    torch::Tensor peer_bufs,
    torch::Tensor input_offsets,
    torch::Tensor input_splits,
    torch::Tensor peer_slot_offsets
) {
    int world_size = (int)peer_bufs.numel();
    int hidden_dim = (int)input.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 block(256);
    dim3 grid(64, world_size);
    a2a_scatter_kernel_f32<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        reinterpret_cast<const long long*>(peer_bufs.data_ptr<int64_t>()),
        input_offsets.data_ptr<int>(),
        input_splits.data_ptr<int>(),
        peer_slot_offsets.data_ptr<int>(),
        hidden_dim,
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_a2a_scatter_bf16", &launch_a2a_scatter_bf16, "All-to-all scatter bf16");
    m.def("launch_a2a_scatter_f32", &launch_a2a_scatter_f32, "All-to-all scatter f32");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_a2a_symm_ext", CUDA_SRC)
    return _ext


# ---------------- symmetric memory caches ----------------

_a2a_cache = {}  # key: (max_rows, hidden_dim, dtype, world_size) -> dict

def _get_a2a_buffer(rows_capacity: int, hidden_dim: int, dtype: torch.dtype,
                    device: torch.device, group: dist.ProcessGroup):
    ws = dist.get_world_size(group)
    # round up capacity to reduce churn
    cap = 1
    while cap < max(rows_capacity, 1):
        cap *= 2
    key = (cap, hidden_dim, dtype, ws)
    if key in _a2a_cache:
        return _a2a_cache[key]

    buf = symm_mem.empty((cap, hidden_dim), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    peer_ptrs = torch.tensor([int(hdl.buffer_ptrs[p]) for p in range(ws)],
                             device=device, dtype=torch.int64)
    res = {"buf": buf, "hdl": hdl, "peer_ptrs": peer_ptrs, "cap": cap}
    _a2a_cache[key] = res
    return res


_meta_cache = {}  # for split-size all-gather
def _get_meta_buffer(num_experts: int, device: torch.device, group: dist.ProcessGroup):
    ws = dist.get_world_size(group)
    key = (num_experts, ws, device)
    if key in _meta_cache:
        return _meta_cache[key]
    # one slot per rank, each holding num_experts ints (we use int64)
    buf = symm_mem.empty((ws, num_experts), device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(buf, group)
    res = {"buf": buf, "hdl": hdl}
    _meta_cache[key] = res
    return res


# ---------------- custom all-to-all (forward only path here) ----------------

def _custom_all_to_all(
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Symm-mem-backed all-to-all. Returns a fresh torch tensor."""
    ws = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if ws == 1:
        return input.contiguous()

    input = input.contiguous()
    hidden_dim = input.size(1)
    total_in = int(sum(input_split_sizes))
    total_out = int(sum(output_split_sizes))

    # We need each rank to know, for each peer p, the row offset within peer p's
    # buffer where this rank's contribution lands. That equals
    # sum_{r<rank} (rows that rank r sends to peer p) = sum_{r<rank} send_splits[r->p].
    # send_splits[r->p] on rank r equals input_split_sizes[p] for that rank.
    # But peers don't directly know our input_split_sizes. We must exchange them.
    # Use a symmetric int64 [ws, num_experts_proxy] -> here we just need ws ints per rank.

    # Gather all input_split_sizes via symm_mem broadcast.
    meta = _get_meta_buffer(ws, input.device, group)
    meta_buf = meta["buf"]            # [ws, ws] int64 — per rank, its input_splits
    meta_hdl = meta["hdl"]

    # Write our row to slot rank
    splits_t = torch.tensor(input_split_sizes, device=input.device, dtype=torch.int64)
    meta_buf[rank].copy_(splits_t)
    meta_hdl.barrier(channel=0)

    # Read all rows: all_input_splits[r, p] = number of rows rank r sends to peer p
    all_input_splits = meta_buf.clone()  # [ws, ws]

    # For peer p, rows from rank r land at offset cumulative over r<rank. We need
    # peer_slot_offsets[p] = sum_{r<rank} all_input_splits[r, p]
    # On each rank, compute as prefix sum along dim 0 up to rank.
    if rank == 0:
        peer_slot_offsets = torch.zeros(ws, device=input.device, dtype=torch.int32)
    else:
        peer_slot_offsets = all_input_splits[:rank, :].sum(dim=0).to(torch.int32)
    # peer_slot_offsets[p] in element units of rows.

    # Compute input_offsets on this rank (cumsum exclusive of input_split_sizes)
    in_off = [0] * ws
    s = 0
    for i in range(ws):
        in_off[i] = s
        s += input_split_sizes[i]
    input_offsets = torch.tensor(in_off, device=input.device, dtype=torch.int32)
    input_splits_t = torch.tensor(input_split_sizes, device=input.device, dtype=torch.int32)

    # Get symmetric output buffer big enough for total_out rows.
    a2a = _get_a2a_buffer(total_out, hidden_dim, input.dtype, input.device, group)
    sym_buf = a2a["buf"]
    sym_hdl = a2a["hdl"]
    peer_ptrs = a2a["peer_ptrs"]

    # Barrier so peers' buffers are ready to be written into.
    sym_hdl.barrier(channel=1)

    # Launch scatter kernel: each block writes to one peer.
    if input.dtype == torch.bfloat16:
        _get_ext().launch_a2a_scatter_bf16(input, peer_ptrs, input_offsets,
                                           input_splits_t, peer_slot_offsets)
    elif input.dtype == torch.float32:
        _get_ext().launch_a2a_scatter_f32(input, peer_ptrs, input_offsets,
                                          input_splits_t, peer_slot_offsets)
    else:
        # Fallback to NCCL for unusual dtypes
        sym_hdl.barrier(channel=2)
        out = torch.empty((total_out, hidden_dim), dtype=input.dtype, device=input.device)
        dist.all_to_all_single(out, input, output_split_sizes=output_split_sizes,
                               input_split_sizes=input_split_sizes, group=group)
        return out

    # Wait for all peers to finish writing into our buffer.
    sym_hdl.barrier(channel=2)

    # Slice and clone out the valid rows.
    out = sym_buf[:total_out].clone()
    return out


# ---------- AllToAll autograd that uses our custom kernel on forward ----------

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        if dist.get_world_size(group=group) == 1:
            return input.contiguous()
        return _custom_all_to_all(input, output_split_sizes, input_split_sizes, group)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            _AllToAll.apply(ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def _all_to_all(group, input, output_split_sizes, input_split_sizes):
    return _AllToAll.apply(group, input, output_split_sizes, input_split_sizes)


# ---------------- preprocess: replace all_gather with symm_mem ----------------

def _preprocess(expert_mask, num_experts, ep_group):
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))  # [num_experts]
    input_splits = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    )

    # All-gather of num_local_tokens_per_expert across ranks via symm_mem.
    flat = num_local_tokens_per_expert.contiguous().view(-1).to(torch.int64)
    meta = _get_meta_buffer(num_experts, flat.device, ep_group)
    meta_buf = meta["buf"]   # [ep_size, num_experts] int64
    meta_hdl = meta["hdl"]
    meta_buf[rank].copy_(flat)
    meta_hdl.barrier(channel=3)
    num_global_tokens_per_expert = meta_buf.clone().to(num_local_tokens_per_expert.dtype)
    # shape: [ep_size, num_experts]

    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        :, start_idx:end_idx
    ].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(
        dim=0
    ).to(torch.device("cpu"), non_blocking=True)
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
        -1, num_local_experts
    ).to(torch.device("cpu"), non_blocking=True)
    return (
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        num_global_sum_tokens_per_local_expert,
    )


# ----------- the rest of the helpers (unchanged from reference) -----------

def _permute(tokens, routing_map):
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def _sort_chunks_by_idxs(input, split_sizes, sorted_idxs):
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()
    chunks = torch.split(input, split_sizes, dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)


def _generate_weights_idx(routing_weights, selected_experts, num_experts):
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros(
        (num_tokens, num_experts),
        dtype=routing_weights.dtype, device=routing_weights.device,
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def _unpermute(tokens, routing_weights, hidden_states_shape, permutation_mapping, routing_map):
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    return unpermuted_tokens


def token_pre_all2all(hidden_states, expert_mask, num_experts, input_splits,
                     output_splits, num_global_tokens_per_local_expert, group=None):
    group = group or dist.group.WORLD
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)

    local_permuted_hidden_states, local_input_permutation_mapping = _permute(
        hidden_states, routing_map
    )
    expected_tokens = sum(input_splits)
    actual_tokens = local_permuted_hidden_states.shape[0]
    if expected_tokens != actual_tokens:
        raise RuntimeError(
            f"EP split mismatch: input_splits sum ({expected_tokens}) != "
            f"permuted tokens ({actual_tokens})"
        )

    global_permuted_hidden_states = _all_to_all(
        group, local_permuted_hidden_states, output_splits, input_splits
    )
    num_local_experts = num_experts // dist.get_world_size(group)
    permute_order = (
        torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    )
    split_sizes = num_global_tokens_per_local_expert.ravel().tolist()
    global_permuted_hidden_states = _sort_chunks_by_idxs(
        global_permuted_hidden_states, split_sizes, permute_order
    )
    return (global_permuted_hidden_states, routing_map,
            local_input_permutation_mapping, org_hidden_states_shape)


def tokens_post_all2all(expert_outputs, routing_weights, selected_experts, num_experts,
                       input_splits, output_splits, num_global_tokens_per_local_expert,
                       routing_map, local_input_permutation_mapping,
                       org_hidden_states_shape, group=None):
    group = group or dist.group.WORLD
    num_local_experts = num_experts // dist.get_world_size(group)
    unpermute_order = (
        torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    )
    split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist()
    expert_outputs = _sort_chunks_by_idxs(expert_outputs, split_sizes, unpermute_order)
    unpermute_outputs = _all_to_all(group, expert_outputs, input_splits, output_splits)
    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    unpermute_outputs = _unpermute(
        unpermute_outputs, weights_idx, org_hidden_states_shape,
        local_input_permutation_mapping, routing_map,
    )
    return unpermute_outputs


def expert_forward(x, gate_proj, up_proj, down_proj):
    gate = torch.nn.functional.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    # Ensure JIT compiled on rank 0 first
    _get_ext()

    hidden_dim = hidden_states.size(-1)

    router_logits = torch.nn.functional.linear(
        hidden_states.reshape(-1, hidden_dim), gate_weight, gate_bias
    )
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)

    input_splits, output_splits, num_global_tokens_per_local_expert, _ = _preprocess(
        expert_mask, num_experts, group
    )

    (global_permuted_hidden_states, routing_map,
     local_input_permutation_mapping, org_hidden_states_shape) = token_pre_all2all(
        hidden_states, expert_mask, num_experts, input_splits, output_splits,
        num_global_tokens_per_local_expert, group,
    )

    expert_outputs = expert_forward(
        global_permuted_hidden_states, gate_proj, up_proj, down_proj
    )

    out = tokens_post_all2all(
        expert_outputs, routing_weights, selected_experts, num_experts,
        input_splits, output_splits, num_global_tokens_per_local_expert,
        routing_map, local_input_permutation_mapping,
        org_hidden_states_shape, group,
    )
    return out