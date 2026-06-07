"""
MoE forward+backward with custom CUDA all_to_all using symmetric memory + UVA peer copies.
Replaces dist.all_to_all_single with device-side P2P copies through symm_mem buffers.
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

// Copy from local source into peer's symmetric buffer at peer-specific offsets.
// Each block handles one (peer) chunk.
__global__ void p2p_scatter_kernel(
    const uint8_t* __restrict__ src,           // local source bytes
    const uint64_t* __restrict__ peer_bufs,    // [world_size] peer base ptrs
    const int64_t* __restrict__ src_offsets,   // [world_size] byte offsets in src
    const int64_t* __restrict__ dst_offsets,   // [world_size] byte offsets at peer
    const int64_t* __restrict__ sizes,         // [world_size] byte sizes
    int world_size
) {
    int peer = blockIdx.x;
    if (peer >= world_size) return;
    int64_t sz = sizes[peer];
    if (sz <= 0) return;

    const uint8_t* s = src + src_offsets[peer];
    uint8_t* d = reinterpret_cast<uint8_t*>(peer_bufs[peer]) + dst_offsets[peer];

    // Vectorized copy in 16-byte chunks
    int64_t n16 = sz / 16;
    int64_t rem = sz - n16 * 16;
    const int4* s4 = reinterpret_cast<const int4*>(s);
    int4* d4 = reinterpret_cast<int4*>(d);

    for (int64_t i = threadIdx.x; i < n16; i += blockDim.x) {
        d4[i] = s4[i];
    }
    // Tail bytes
    int64_t tail_start = n16 * 16;
    for (int64_t i = threadIdx.x; i < rem; i += blockDim.x) {
        d[tail_start + i] = s[tail_start + i];
    }
}

void launch_p2p_scatter(
    torch::Tensor src_buf,
    torch::Tensor peer_bufs,    // int64 [world_size]
    torch::Tensor src_offsets,  // int64 [world_size]  (bytes)
    torch::Tensor dst_offsets,  // int64 [world_size]  (bytes)
    torch::Tensor sizes,        // int64 [world_size]  (bytes)
    int64_t world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint8_t* src = reinterpret_cast<const uint8_t*>(src_buf.data_ptr());
    const uint64_t* peer_p = reinterpret_cast<const uint64_t*>(peer_bufs.data_ptr<int64_t>());
    const int64_t* so = src_offsets.data_ptr<int64_t>();
    const int64_t* dof = dst_offsets.data_ptr<int64_t>();
    const int64_t* sz = sizes.data_ptr<int64_t>();
    int threads = 256;
    p2p_scatter_kernel<<<(int)world_size, threads, 0, stream>>>(
        src, peer_p, so, dof, sz, (int)world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_p2p_scatter", &launch_p2p_scatter, "P2P scatter via UVA peer pointers");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_p2p_a2a_ext", CUDA_SRC)
    return _ext


# Symmetric memory pool: keep buffers keyed by (size_bytes)
_symm_pool = {}


def _get_symm_buf(size_bytes: int, device: torch.device):
    # Round up to power-of-two for reuse
    bucket = 1
    while bucket < max(size_bytes, 1024):
        bucket *= 2
    key = (bucket, device.index)
    if key not in _symm_pool:
        buf = symm_mem.empty(bucket, device=device, dtype=torch.uint8)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        peer_ptrs = torch.tensor(
            [int(p) for p in hdl.buffer_ptrs], device=device, dtype=torch.int64
        )
        _symm_pool[key] = (buf, hdl, peer_ptrs)
    return _symm_pool[key]


def _custom_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: dist.ProcessGroup,
):
    """All-to-all via symmetric memory P2P copies."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = input.device
    elem_size = input.element_size()
    row_bytes = input.stride(0) * elem_size if input.dim() > 1 else elem_size

    # Compute byte sizes/offsets for sending (input view)
    in_offsets_rows = [0]
    for s in input_split_sizes:
        in_offsets_rows.append(in_offsets_rows[-1] + s)
    # Compute byte sizes/offsets for receiving (output view)
    out_offsets_rows = [0]
    for s in output_split_sizes:
        out_offsets_rows.append(out_offsets_rows[-1] + s)

    total_in_bytes = in_offsets_rows[-1] * row_bytes
    total_out_bytes = out_offsets_rows[-1] * row_bytes

    # We need symmetric staging so peers know where to write.
    # Approach: each rank has a symmetric "recv" buffer big enough for total_out_bytes
    # across all ranks. We allgather max recv size... simpler: use dist.barrier for size negotiation.
    # For correctness, allocate a symmetric buffer sized to max across ranks.
    local_recv_bytes = total_out_bytes
    # Negotiate global max via small allreduce (one int)
    sz_t = torch.tensor([local_recv_bytes], device=device, dtype=torch.int64)
    dist.all_reduce(sz_t, op=dist.ReduceOp.MAX, group=group)
    max_recv_bytes = int(sz_t.item())

    # Also need symmetric send buffer (since src_buf must be on local memory; that's fine,
    # we read locally). But peers write into our recv buf. So recv buf must be symmetric.
    recv_buf, recv_hdl, recv_peer_ptrs = _get_symm_buf(max_recv_bytes, device)

    # Each rank i sends to rank j at offset = where on rank j's recv buffer rank i's chunk goes.
    # Rank j expects from rank i a chunk of size output_split_sizes[i] (on rank j).
    # So rank i needs to know, for each peer j, the offset on j where i's data goes.
    # That offset on rank j = sum_{k<i} output_split_sizes_on_j[k].
    # We need each rank's output_split_sizes vector. AllGather them.

    out_splits_t = torch.tensor(output_split_sizes, device=device, dtype=torch.int64)
    all_out_splits = torch.empty(world_size * world_size, device=device, dtype=torch.int64)
    dist.all_gather_into_tensor(all_out_splits, out_splits_t, group=group)
    all_out_splits = all_out_splits.view(world_size, world_size)  # [rank_j, rank_i]

    # For each peer j, dst row offset = sum over k<rank of all_out_splits[j, k]
    dst_row_offsets = []
    for j in range(world_size):
        prefix = int(all_out_splits[j, :rank].sum().item())
        dst_row_offsets.append(prefix)

    src_offsets_bytes = torch.tensor(
        [in_offsets_rows[j] * row_bytes for j in range(world_size)],
        device=device, dtype=torch.int64,
    )
    dst_offsets_bytes = torch.tensor(
        [dst_row_offsets[j] * row_bytes for j in range(world_size)],
        device=device, dtype=torch.int64,
    )
    sizes_bytes = torch.tensor(
        [input_split_sizes[j] * row_bytes for j in range(world_size)],
        device=device, dtype=torch.int64,
    )

    # Barrier so all peers have allocated/registered recv buffer
    recv_hdl.barrier(channel=0)

    # Launch P2P scatter: each block writes to one peer's recv buffer
    _get_ext().launch_p2p_scatter(
        input.contiguous().view(torch.uint8).view(-1),
        recv_peer_ptrs,
        src_offsets_bytes,
        dst_offsets_bytes,
        sizes_bytes,
        world_size,
    )

    # Barrier so all writes complete before we read
    recv_hdl.barrier(channel=1)

    # Copy from recv_buf into output
    if total_out_bytes > 0:
        out_bytes_view = output.contiguous().view(torch.uint8).view(-1)
        out_bytes_view.copy_(recv_buf[:total_out_bytes])


# ----- AllToAll autograd wrapper using custom P2P -----

class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        if dist.get_world_size(group=group) == 1:
            return input.contiguous()
        input = input.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(input)
            # fallback path
            dist.all_to_all_single(output, input, group=group)
            return output

        output = torch.empty(
            size=(sum(output_split_sizes), input.size(1)),
            dtype=input.dtype,
            device=input.device,
        )
        if output.numel() == 0 and input.numel() == 0:
            return output

        _custom_all_to_all_single(
            output, input,
            list(output_split_sizes), list(input_split_sizes),
            group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            _AllToAll.apply(
                ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes
            ),
            None,
            None,
        )


def _all_to_all(group, input, output_split_sizes, input_split_sizes):
    return _AllToAll.apply(group, input, output_split_sizes, input_split_sizes)


# ----- Preprocess -----

def _preprocess(expert_mask, num_experts, ep_group):
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))
    input_splits = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    )
    flat = num_local_tokens_per_expert.contiguous().view(-1)
    out_size = ep_size * flat.numel()
    gathered = torch.empty(out_size, dtype=flat.dtype, device=flat.device)
    dist.all_gather_into_tensor(gathered, flat, group=ep_group)
    num_global_tokens_per_expert = gathered.view(ep_size, flat.numel())
    s, e = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, s:e].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    num_global_sum = num_global_tokens_per_local_expert.sum(dim=0).to("cpu", non_blocking=True)
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
        -1, num_local_experts
    ).to("cpu", non_blocking=True)
    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum


def _permute(tokens, routing_map):
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
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
    w = torch.zeros((num_tokens, num_experts), dtype=routing_weights.dtype,
                    device=routing_weights.device)
    w.scatter_add_(1, selected_experts, routing_weights)
    return w


def _unpermute(tokens, routing_weights, hidden_states_shape, permutation_mapping, routing_map):
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    out = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    expanded = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    out.scatter_add_(0, expanded, tokens)
    return out


def token_pre_all2all(hidden_states, expert_mask, num_experts, input_splits,
                     output_splits, num_global_tokens_per_local_expert, group=None):
    group = group or dist.group.WORLD
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)
    local_perm, local_map = _permute(hidden_states, routing_map)
    if sum(input_splits) != local_perm.shape[0]:
        raise RuntimeError("EP split mismatch")
    global_perm = _all_to_all(group, local_perm, output_splits, input_splits)
    num_local_experts = num_experts // dist.get_world_size(group)
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    split_sizes = num_global_tokens_per_local_expert.ravel().tolist()
    global_perm = _sort_chunks_by_idxs(global_perm, split_sizes, permute_order)
    return global_perm, routing_map, local_map, org_shape


def tokens_post_all2all(expert_outputs, routing_weights, selected_experts, num_experts,
                       input_splits, output_splits, num_global_tokens_per_local_expert,
                       routing_map, local_input_permutation_mapping, org_hidden_states_shape,
                       group=None):
    group = group or dist.group.WORLD
    num_local_experts = num_experts // dist.get_world_size(group)
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist()
    expert_outputs = _sort_chunks_by_idxs(expert_outputs, split_sizes, unpermute_order)
    out = _all_to_all(group, expert_outputs, input_splits, output_splits)
    w = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    out = _unpermute(out, w, org_hidden_states_shape, local_input_permutation_mapping, routing_map)
    return out


def expert_forward(x, gate_proj, up_proj, down_proj):
    gate = torch.nn.functional.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


def solution(hidden_states, gate_weight, gate_bias, gate_proj, up_proj, down_proj,
             num_experts, top_k, group=None):
    group = group or dist.group.WORLD
    # Ensure ext compiled before any rank uses it
    if dist.is_initialized():
        if dist.get_rank(group) == 0:
            _get_ext()
        dist.barrier(group=group)
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

    (global_perm, routing_map, local_map, org_shape) = token_pre_all2all(
        hidden_states, expert_mask, num_experts, input_splits, output_splits,
        num_global_tokens_per_local_expert, group,
    )

    expert_outputs = expert_forward(global_perm, gate_proj, up_proj, down_proj)

    out = tokens_post_all2all(
        expert_outputs, routing_weights, selected_experts, num_experts,
        input_splits, output_splits, num_global_tokens_per_local_expert,
        routing_map, local_map, org_shape, group,
    )
    return out