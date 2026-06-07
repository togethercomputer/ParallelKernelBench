from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
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


def gather_heads_scatter_seq(
    x: Tensor, head_dim: int, seq_dim: int, group: Optional[ProcessGroup] = None
) -> Tensor:
    group = group or dist.group.WORLD
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = dist.get_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return _SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    async_op: bool = False,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    group = group or dist.group.WORLD
    if not group:
        return x
    sp_world = dist.get_world_size(group)
    if async_op:
        return _SeqAllToAll.apply(group, x, head_dim, seq_dim, async_op)
    x = _SeqAllToAll.apply(group, x, head_dim, seq_dim, async_op)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = _unpad_tensor(x, seq_dim, padding_size)
    return x


def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = False,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal and q.size(1) > 1:
        S = scores.size(-1)
        causal_mask = torch.triu(
            torch.ones(S, S, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    num_heads: int = 8,
    causal: bool = False,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        B, S_local, H = hidden_states.shape
        head_dim = H // num_heads
        qkv = F.linear(hidden_states, w_qkv)
        qkv = qkv.view(B, S_local, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)
        scale = head_dim**-0.5
        attn_out = _local_attention(q, k, v, scale, causal=causal)
        out = attn_out.reshape(B, S_local, -1)
        return F.linear(out, w_o)

    B, S_local, H = hidden_states.shape
    head_dim = (w_qkv.shape[0] // 3) // num_heads
    assert (w_qkv.shape[0] // 3) == num_heads * head_dim
    assert num_heads % world_size == 0, "num_heads must be divisible by world_size"

    qkv = F.linear(hidden_states, w_qkv)
    qkv = qkv.view(B, S_local, 3, num_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    q = gather_seq_scatter_heads(q, seq_dim=1, head_dim=2, group=group)
    kv = torch.stack([k, v], dim=3)
    kv = kv.reshape(B, S_local, 2 * num_heads, head_dim)
    kv = gather_seq_scatter_heads(kv, seq_dim=1, head_dim=2, group=group)
    kv = kv.reshape(B, kv.size(1), num_heads // world_size, 2, head_dim)
    k = kv[:, :, :, 0, :]
    v = kv[:, :, :, 1, :]

    scale = head_dim**-0.5
    attn_out = _local_attention(q, k, v, scale, causal=causal)

    attn_out = gather_heads_scatter_seq(attn_out, seq_dim=1, head_dim=2, group=group)

    out = attn_out.reshape(B, attn_out.size(1), -1)
    return F.linear(out, w_o)
