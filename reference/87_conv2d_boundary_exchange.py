from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _gather_boundaries(
    x: torch.Tensor,
    boundary: int,
    group: dist.ProcessGroup,
) -> list[torch.Tensor]:
    if boundary == 0:
        empty = x[:, :, :0, :]
        return [torch.stack([empty, empty], dim=0)]

    local = torch.stack([x[:, :, :boundary, :], x[:, :, -boundary:, :]], dim=0)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local.contiguous(), group=group)
    return gathered


@torch.no_grad()
def solution(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int = 1,
    padding: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    boundary = int(padding)

    if boundary == 0 or world_size == 1:
        return F.conv2d(x, weight, bias, stride=stride, padding=padding)

    boundaries = _gather_boundaries(x, boundary, group)
    pieces = []
    if rank == 0:
        pieces.append(x.new_zeros(*x.shape[:2], boundary, x.shape[-1]))
    else:
        pieces.append(boundaries[rank - 1][1])
    pieces.append(x)
    if rank == world_size - 1:
        pieces.append(x.new_zeros(*x.shape[:2], boundary, x.shape[-1]))
    else:
        pieces.append(boundaries[rank + 1][0])

    padded_x = torch.cat(pieces, dim=2)
    return F.conv2d(padded_x, weight, bias, stride=stride, padding=(0, padding))
