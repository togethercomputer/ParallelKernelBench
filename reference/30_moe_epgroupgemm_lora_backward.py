from typing import Optional, Tuple

import torch
import torch.distributed as dist


def solution(
    grad_fc1_1_lora_A: torch.Tensor,
    grad_fc1_2_lora_A: torch.Tensor,
    grad_fc2_lora_B: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    dist.all_reduce(grad_fc1_1_lora_A, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(grad_fc1_2_lora_A, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(grad_fc2_lora_B, op=dist.ReduceOp.SUM, group=group)
    return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B
