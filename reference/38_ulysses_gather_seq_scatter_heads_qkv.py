from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


def _pad_tensor(x: Tensor, dim: int, padding_size: int, padding_value: int = 0) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.full(shape, padding_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[tuple(slc)]


def _all_to_all_single(
    x: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    group = group or dist.group.WORLD
    sp_world_size = dist.get_world_size(group)
    assert scatter_dim <= 1, "scatter_dim must be 0 or 1 when using all_to_all_single!"
    assert gather_dim <= 1, "gather_dim must be 0 or 1 when using all_to_all_single!"
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (
            x.reshape(
                [gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size]
                + list(x.shape[2:])
            )
            .transpose(0, 1)
            .reshape(
                [gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size]
                + list(x.shape[2:])
            )
            .contiguous()
        )

    output = torch.empty_like(x)
    comm = dist.all_to_all_single(output, x.contiguous(), group=group, async_op=async_op)

    if async_op:

        def wait():
            comm.wait()
            if scatter_dim == 0:
                return torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
            else:
                return output

        return wait

    if scatter_dim == 0:
        output = torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
    return output


def _all_to_all(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    group = group or dist.group.WORLD
    seq_world_size = dist.get_world_size(group)
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_tensor(
    x: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    if scatter_dim <= 1 and gather_dim <= 1:
        return _all_to_all_single(x, scatter_dim, gather_dim, group, async_op)
    return _all_to_all(x, scatter_dim, gather_dim, group, async_op)


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return _all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None, None]:
        if ctx.async_op:
            input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
        else:
            input_t = grad_output[0]
        return (
            None,
            _all_to_all_tensor(
                input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False
            ),
            None,
            None,
            None,
        )


def gather_seq_scatter_heads_qkv(
    qkv_tensor: Tensor,
    seq_dim: int,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
    async_op: bool = False,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    group = group or dist.group.WORLD
    if not group:
        return qkv_tensor
    sp_world = dist.get_world_size(group)
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim()
    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1]
    bef_all2all_shape = bef_all2all_shape[:-1] + [3, qkv_proj_dim // 3]
    qkv_tensor = qkv_tensor.view(bef_all2all_shape)
    if async_op:
        return _SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)
    qkv_tensor = _SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)

    if restore_shape:
        out_shape = list(orig_shape)
        out_shape[seq_dim] *= sp_world
        out_shape[-1] = qkv_proj_dim // sp_world
        qkv_tensor = qkv_tensor.view(out_shape)

    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = qkv_tensor.size(seq_dim) - unpadded_dim_size
        qkv_tensor = _unpad_tensor(qkv_tensor, seq_dim, padding_size)

    return qkv_tensor


def solution(
    qkv_tensor: torch.Tensor,
    seq_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    return gather_seq_scatter_heads_qkv(
        qkv_tensor,
        seq_dim=seq_dim,
        unpadded_dim_size=unpadded_dim_size or 0,
        restore_shape=restore_shape,
        async_op=False,
        group=group,
    )
