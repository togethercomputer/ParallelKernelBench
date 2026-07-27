# Ulysses all-gather along an arbitrary dim.
#
# all_gather_into_tensor only concatenates along dim 0; the Ulysses sequence
# dim is usually not dim 0, so a rank-stacked gather must be followed by a
# reorder/copy to concatenate along `gather_dim`. A fused kernel can write
# gathered chunks directly into their strided destination.
from typing import Optional

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

    x = x.contiguous()
    stacked = torch.empty(
        (world_size,) + tuple(x.size()), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(stacked, x, group=group)
    return torch.cat(list(stacked.unbind(0)), dim=gather_dim).contiguous()
