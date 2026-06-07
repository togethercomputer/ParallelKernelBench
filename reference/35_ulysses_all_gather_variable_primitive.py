from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


def solution(
    x: torch.Tensor,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    device = x.device
    dtype = x.dtype
    x = x.contiguous()
    x_size = torch.tensor(x.size(), dtype=torch.int64, device=device)
    size_list = [torch.zeros(x_size.size(), dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(size_list, x_size, group=group)
    tensor_list = [
        torch.empty(torch.Size(size_list[i].tolist()), dtype=dtype, device=device)
        for i in range(world_size)
    ]
    dist.all_gather(tensor_list, x, group=group)
    return torch.cat(tensor_list, dim=gather_dim).contiguous()
