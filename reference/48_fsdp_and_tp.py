from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def _make_tp_fsdp_groups(n_tp: int, n_fsdp: int, rank: int):
    tp_group = None
    fsdp_group = None
    for j in range(n_fsdp):
        ranks = [j * n_tp + ii for ii in range(n_tp)]
        g = dist.new_group(ranks)
        if rank in ranks:
            tp_group = g
    for i in range(n_tp):
        ranks = [jj * n_tp + i for jj in range(n_fsdp)]
        g = dist.new_group(ranks)
        if rank in ranks:
            fsdp_group = g
    assert tp_group is not None and fsdp_group is not None
    return tp_group, fsdp_group


def _gather_fsdp_concat_dim0(shard: Tensor, fsdp_group, parts: int) -> Tensor:
    lst = [torch.empty_like(shard) for _ in range(parts)]
    dist.all_gather(lst, shard.contiguous(), group=fsdp_group)
    return torch.cat(lst, dim=0)


def _gather_fsdp_concat_dim1(shard: Tensor, fsdp_group, parts: int) -> Tensor:
    lst = [torch.empty_like(shard) for _ in range(parts)]
    dist.all_gather(lst, shard.contiguous(), group=fsdp_group)
    return torch.cat(lst, dim=1)


@torch.no_grad()
def solution(
    x_local: Tensor,
    W1_shard: Tensor,
    W2_shard: Tensor,
    W3_shard: Tensor,
    n_tp: int,
    n_fsdp: int,
) -> Tensor:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size == n_tp * n_fsdp, (
        f"world_size ({world_size}) must equal n_tp * n_fsdp ({n_tp} * {n_fsdp})"
    )

    tp_group, fsdp_group = _make_tp_fsdp_groups(n_tp, n_fsdp, rank)

    W1 = _gather_fsdp_concat_dim0(W1_shard, fsdp_group, n_fsdp)
    W2 = _gather_fsdp_concat_dim0(W2_shard, fsdp_group, n_fsdp)
    W3 = _gather_fsdp_concat_dim1(W3_shard, fsdp_group, n_fsdp)

    x1 = x_local @ W1
    x2 = x_local @ W2
    z = F.silu(x1) * x2
    y_partial = z @ W3

    y = y_partial.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM, group=tp_group)
    return y
