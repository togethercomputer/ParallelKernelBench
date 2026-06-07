from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


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


def _pad_zero(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    dim = dim % tensor.ndim
    pad = [0] * (2 * (tensor.ndim - dim))
    pad[1] = size - tensor.shape[dim]
    return F.pad(tensor, pad, mode="constant", value=0.0)


def _gather_dim(
    tensor: torch.Tensor,
    dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    chunks = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(chunks, tensor, group=group)
    return torch.cat(chunks, dim=dim).contiguous()


def _scatter_dim(
    tensor: torch.Tensor,
    dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    chunks = torch.split(tensor, tensor.shape[dim] // world_size, dim=dim)
    return chunks[rank].contiguous()


def _conj_pad_2d(
    tensor: torch.Tensor,
    pad_dim: int,
    other_dim: int,
    size: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    pad_dim = pad_dim % tensor.ndim
    other_dim = other_dim % tensor.ndim
    orig_size = tensor.shape[pad_dim]

    tensor_pad = _pad_zero(tensor, pad_dim, size)
    lhs_slice = [slice(0, s) for s in tensor.shape]
    lhs_slice[pad_dim] = slice(orig_size, size)
    rhs_slice = [slice(0, s) for s in tensor.shape]
    rhs_slice[pad_dim] = slice(1, size - orig_size + 1)
    tensor_pad[tuple(lhs_slice)] = torch.flip(torch.conj(tensor_pad[tuple(rhs_slice)]), dims=[pad_dim])

    tensor_pad = _gather_dim(tensor_pad, other_dim, group)
    flip_slice = [slice(0, s) for s in tensor_pad.shape]
    flip_slice[pad_dim] = slice(orig_size, size)
    flip_slice[other_dim] = slice(1, tensor_pad.shape[other_dim])
    tensor_pad[tuple(flip_slice)] = torch.flip(tensor_pad[tuple(flip_slice)], dims=[other_dim])
    return _scatter_dim(tensor_pad, other_dim, group)


@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Optional[Sequence[int]],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    dim0, dim1 = int(dim[0]), int(dim[1])
    if s is not None:
        first_dim_size = int(s[0])
        last_dim_size = int(s[1])
    else:
        first_dim_size = int(x.shape[dim0])
        last_dim_size = int(2 * (x.shape[dim1] - 1))

    x_pad = _conj_pad_2d(x, pad_dim=dim1, other_dim=dim0, size=last_dim_size, group=group)

    x1 = torch.fft.ifft(x_pad, n=last_dim_size, dim=dim1, norm=norm)

    x1_recv = _all_to_all_transpose(x1, split_dim=dim1, group=group)
    x1_tran = torch.cat(x1_recv, dim=dim0)

    x2 = torch.fft.ifft(x1_tran, n=first_dim_size, dim=dim0, norm=norm)
    return torch.real(x2).contiguous()