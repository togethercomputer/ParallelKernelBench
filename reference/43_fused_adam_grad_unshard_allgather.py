from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import Tensor


@torch.no_grad()
def solution(
    grad_shard: Tensor,
    master_shard: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> Tensor:
    world_size = dist.get_world_size()

    assert step >= 1
    p = grad_shard.numel()
    assert p > 0

    m = exp_avg.clone()
    v = exp_avg_sq.clone()
    w = master_shard.clone()

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    g = grad_shard
    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    m_hat = m / bc1
    v_hat = v / bc2
    w.add_(m_hat.div(v_hat.sqrt().add(eps)).mul(-lr))

    gathered = torch.empty(world_size * p, dtype=w.dtype, device=w.device)
    dist.all_gather_into_tensor(gathered, w.contiguous())
    return gathered
