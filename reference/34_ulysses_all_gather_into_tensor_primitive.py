from typing import Optional

import torch
import torch.distributed as dist


def solution(
    x: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    x = x.contiguous()
    dim_size = list(x.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(output, x, group=group)
    return output
