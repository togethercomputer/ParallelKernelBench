# MoE all-to-all split-metadata computation, kept entirely on-device.
#
# From the per-token expert mask, compute the send split sizes (tokens this
# rank routes to each peer), all-gather the per-expert token histogram, and
# derive the receive split sizes plus the per-(source rank, local expert)
# token counts. Everything stays on the GPU: reductions over the expert mask
# plus a single small all-gather — a fusion target for kernel-side metadata
# computation.
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def _preprocess_impl(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2)).to(torch.long)

    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1)

    num_global_tokens_per_expert = torch.empty(
        ep_size,
        num_local_tokens_per_expert.size(0),
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    output_splits = num_global_tokens_per_local_expert.sum(dim=1)

    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(-1, num_local_experts)

    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def solution(
    expert_mask: torch.Tensor,
    num_experts: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    return _preprocess_impl(expert_mask=expert_mask, num_experts=num_experts, ep_group=group)
