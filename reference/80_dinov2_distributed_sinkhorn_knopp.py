from typing import Optional

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    n_masked_patches_tensor: torch.Tensor,
    n_iterations: int = 3,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    q = torch.exp(teacher_output.float() / teacher_temp).T
    total_batch = n_masked_patches_tensor.to(device=q.device, dtype=q.dtype).clone()
    dist.all_reduce(total_batch, group=group)

    num_prototypes = q.shape[0]
    total_mass = q.sum()
    dist.all_reduce(total_mass, group=group)
    q /= total_mass

    for _ in range(n_iterations):
        row_sum = q.sum(dim=1, keepdim=True)
        dist.all_reduce(row_sum, group=group)
        q /= row_sum
        q /= num_prototypes

        q /= q.sum(dim=0, keepdim=True)
        q /= total_batch

    q *= total_batch
    return q.T.contiguous()
