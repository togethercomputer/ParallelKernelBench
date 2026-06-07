from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist


def _permute(tokens: torch.Tensor, routing_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
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


def _all_to_all_forward(
    group: dist.ProcessGroup,
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
) -> torch.Tensor:
    if dist.get_world_size(group) == 1:
        return input.contiguous()
    input = input.contiguous()
    out_size = sum(output_split_sizes) if output_split_sizes else input.size(0)
    output = torch.empty((out_size, input.size(1)), dtype=input.dtype, device=input.device)
    dist.all_to_all_single(
        output, input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def solution(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: Union[List[int], torch.Tensor],
    output_splits: Union[List[int], torch.Tensor],
    num_global_tokens_per_local_expert: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    group = group or dist.group.WORLD
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)

    local_permuted_hidden_states, local_input_permutation_mapping = _permute(hidden_states, routing_map)

    expected_tokens = sum(input_splits) if isinstance(input_splits, list) else int(input_splits.sum().item())
    actual_tokens = local_permuted_hidden_states.shape[0]
    if expected_tokens != actual_tokens:
        raise RuntimeError(
            f"EP split mismatch: input_splits sum ({expected_tokens}) != permuted tokens ({actual_tokens})"
        )

    global_permuted_hidden_states = _all_to_all_forward(
        group, local_permuted_hidden_states, output_splits, input_splits
    )

    num_local_experts = num_experts // dist.get_world_size(group)
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    global_permuted_hidden_states = _sort_chunks_by_idxs(
        global_permuted_hidden_states,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )

    return global_permuted_hidden_states, routing_map, local_input_permutation_mapping, org_hidden_states_shape
