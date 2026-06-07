from typing import Optional, Tuple

import torch
import torch.distributed as dist


def _vocab_range(partition_vocab_size: int, rank: int) -> Tuple[int, int]:
    start = rank * partition_vocab_size
    return start, start + partition_vocab_size


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)

    logits_max = torch.max(vocab_parallel_logits, dim=-1).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
    vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

    partition_vocab_size = vocab_parallel_logits.shape[-1]
    vocab_start, vocab_end = _vocab_range(partition_vocab_size, rank)
    target_mask = (target < vocab_start) | (target >= vocab_end)
    masked_target = target - vocab_start
    masked_target = masked_target.masked_fill(target_mask, 0)

    logits_2d = vocab_parallel_logits.reshape(-1, partition_vocab_size)
    target_1d = masked_target.reshape(-1)
    row_ids = torch.arange(logits_2d.shape[0], device=logits_2d.device)
    predicted_logits = logits_2d[row_ids, target_1d].clone().reshape_as(target)
    predicted_logits = predicted_logits.masked_fill(target_mask, 0.0)
    dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

    exp_logits = torch.exp(vocab_parallel_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)
    dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

    return torch.log(sum_exp_logits) - predicted_logits