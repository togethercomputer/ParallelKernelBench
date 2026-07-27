# Expert-parallel (EP) fused MoE forward — WIDE EP (multiple experts per rank).
#
# num_experts == world_size * 2, so each rank hosts num_local_experts = 2
# DISTINCT experts with their own weights (stacked as (E_local, ...) tensors).
# Dispatch uses dynamic variable splits like problem 31, but the local compute
# is a grouped GEMM over ragged per-expert token groups — received tokens are
# regrouped from (source rank, expert) order to expert-major order, each local
# expert runs its own SiLU MLP on its segment, and results are regrouped and
# returned. The ragged grouped GEMM + regrouping is the wide-EP-specific
# fusion opportunity that the single-expert problems (31 / 49 / 51) don't have.
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist


def _all_to_all(
    group: dist.ProcessGroup,
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
) -> torch.Tensor:
    if dist.get_world_size(group=group) == 1:
        return input.contiguous()
    input = input.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(input)
    else:
        output = torch.empty(
            size=(sum(output_split_sizes), input.size(1)),
            dtype=input.dtype,
            device=input.device,
        )
    dist.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def _preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> Tuple[List[int], List[int], torch.Tensor]:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))
    input_splits = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    )
    num_local_tokens_per_expert_flat = num_local_tokens_per_expert.contiguous().view(-1)
    output_size = ep_size * num_local_tokens_per_expert_flat.numel()
    num_global_tokens_per_expert_flat = torch.empty(
        output_size,
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(
        num_global_tokens_per_expert_flat, num_local_tokens_per_expert_flat, group=ep_group
    )
    num_global_tokens_per_expert = num_global_tokens_per_expert_flat.view(
        ep_size, num_local_tokens_per_expert.size(0)
    )
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        :, start_idx:end_idx
    ].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
        -1, num_local_experts
    ).to(torch.device("cpu"))
    return input_splits, output_splits, num_global_tokens_per_local_expert


def _permute(
    tokens: torch.Tensor, routing_map: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def _sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: Union[torch.Tensor, List[int]],
    sorted_idxs: List[int],
) -> torch.Tensor:
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()
    chunks = torch.split(input, split_sizes, dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)


def _generate_weights_idx(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros(
        (num_tokens, num_experts),
        dtype=routing_weights.dtype,
        device=routing_weights.device,
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def _unpermute(
    tokens: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
) -> torch.Tensor:
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    unpermuted_tokens = torch.zeros(
        hidden_states_shape, device=tokens.device, dtype=tokens.dtype
    )
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    return unpermuted_tokens


def _grouped_expert_forward(
    x: torch.Tensor,
    tokens_per_local_expert: List[int],
    w_gate: torch.Tensor,  # (E_local, inter_dim, hidden_dim)
    w_up: torch.Tensor,    # (E_local, inter_dim, hidden_dim)
    w_down: torch.Tensor,  # (E_local, hidden_dim, inter_dim)
) -> torch.Tensor:
    """Ragged grouped GEMM: expert e's SiLU MLP on its token segment."""
    outputs = []
    for e, seg in enumerate(torch.split(x, tokens_per_local_expert, dim=0)):
        gate = torch.nn.functional.silu(seg @ w_gate[e].T)
        up = seg @ w_up[e].T
        outputs.append((gate * up) @ w_down[e].T)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    num_local_experts = num_experts // world_size
    assert num_local_experts >= 2, "wide EP hosts multiple experts per rank"

    hidden_dim = hidden_states.size(-1)
    tokens = hidden_states.reshape(-1, hidden_dim)

    router_logits = torch.nn.functional.linear(tokens, gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)

    input_splits, output_splits, num_global_tokens_per_local_expert = _preprocess(
        expert_mask, num_experts, group
    )

    routing_map = expert_mask.sum(dim=1)
    permuted, permutation_mapping = _permute(tokens, routing_map)
    org_shape = tokens.shape

    global_tokens = _all_to_all(group, permuted, output_splits, input_splits)

    # Regroup from (source rank, expert)-order to expert-major order.
    permute_order = (
        torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    )
    split_sizes = num_global_tokens_per_local_expert.ravel().tolist()
    global_tokens = _sort_chunks_by_idxs(global_tokens, split_sizes, permute_order)

    tokens_per_local_expert = (
        num_global_tokens_per_local_expert.sum(dim=0).tolist()
    )
    expert_outputs = _grouped_expert_forward(
        global_tokens, tokens_per_local_expert, w_gate, w_up, w_down
    )

    # Regroup back to (source rank, expert)-order and return tokens.
    unpermute_order = (
        torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    )
    back_split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist()
    expert_outputs = _sort_chunks_by_idxs(expert_outputs, back_split_sizes, unpermute_order)

    returned = _all_to_all(group, expert_outputs, input_splits, output_splits)

    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    out = _unpermute(returned, weights_idx, org_shape, permutation_mapping, routing_map)
    return out.view_as(hidden_states)
