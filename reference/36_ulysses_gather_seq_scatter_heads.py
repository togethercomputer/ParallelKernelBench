from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def _all_to_all(
    local_input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_single(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    sp_world_size = dist.get_world_size(group)
    assert scatter_dim <= 1 and gather_dim <= 1
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (
            x.reshape(
                [gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:])
            )
            .transpose(0, 1)
            .reshape(
                [gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:])
            )
            .contiguous()
        )
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x.contiguous(), group=group)
    if scatter_dim == 0:
        output = torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
    return output


def _all_to_all_tensor(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    if scatter_dim <= 1 and gather_dim <= 1:
        return _all_to_all_single(x, scatter_dim, gather_dim, group)
    return _all_to_all(x, scatter_dim, gather_dim, group)


def solution(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: int = 0,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    sp_world = dist.get_world_size(group)
    x = _all_to_all_tensor(x, scatter_dim=head_dim, gather_dim=seq_dim, group=group)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        slc = [slice(None)] * x.dim()
        slc[seq_dim] = slice(0, -padding_size)
        x = x[tuple(slc)]
    return x
