from typing import Optional, Sequence

import torch
import torch.distributed as dist


def _all_to_all_transpose(
    tensor: torch.Tensor,
    split_dim: int,
    group: dist.ProcessGroup,
) -> list[torch.Tensor]:
    world_size = dist.get_world_size(group)
    chunk = tensor.shape[split_dim] // world_size
    send = [x.contiguous() for x in torch.split(tensor, chunk, dim=split_dim)]
    recv = [torch.empty_like(send[0]) for _ in range(world_size)]
    dist.all_to_all(recv, send, group=group)
    return recv


def _truncate(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    slices = [slice(None)] * tensor.ndim
    slices[dim % tensor.ndim] = slice(0, size)
    return tensor[tuple(slices)].contiguous()


@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Sequence[int],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    dim0, dim1 = int(dim[0]), int(dim[1])

    n0 = int(s[0]) if s[0] is not None else None
    n1 = int(s[1]) if s[1] is not None else None

    x1 = torch.fft.fft(x, n=n0, dim=dim0, norm=norm)

    x1_recv = _all_to_all_transpose(x1, split_dim=dim0, group=group)
    x1_tran = torch.cat(x1_recv, dim=dim1)

    x2 = torch.fft.fft(x1_tran, n=n1, dim=dim1, norm=norm)

    return _truncate(x2, dim1, x2.shape[dim1] // 2 + 1)