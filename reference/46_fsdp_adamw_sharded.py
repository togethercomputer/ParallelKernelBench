from __future__ import annotations

import math

import torch
from torch import Tensor


@torch.no_grad()
def solution(
    flat_param_shard: Tensor,
    flat_grad_shard: Tensor,
    exp_avg_shard: Tensor,
    exp_avg_sq_shard: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    assert step >= 1

    m = exp_avg_shard.clone()
    v = exp_avg_sq_shard.clone()
    g = flat_grad_shard
    theta = flat_param_shard.clone()

    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    m_hat = m / bc1
    v_hat = v / bc2
    denom = v_hat.sqrt().add(eps)

    theta.add_(m_hat.div(denom), alpha=-lr)
    theta.add_(flat_param_shard, alpha=-lr * weight_decay)

    return theta, m, v
