from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def solution(
    X_local: Tensor,
    y_local: Tensor,
    flat_param_shard: Tensor,
    param_shapes: Sequence[tuple[int, ...]],
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

    world_size = dist.get_world_size()
    p = flat_param_shard.numel()

    device = flat_param_shard.device
    dtype = flat_param_shard.dtype

    templates = [torch.zeros(shape, dtype=dtype, device=device) for shape in param_shapes]
    full_flat = torch.empty(world_size * p, dtype=dtype, device=device)
    dist.all_gather_into_tensor(full_flat, flat_param_shard.contiguous())

    params_f = _unflatten_dense_tensors(full_flat, templates)
    params = [t.detach().requires_grad_(True) for t in params_f]

    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    flat_g = _flatten_dense_tensors([x.grad for x in params])
    g_shard = torch.empty(p, dtype=flat_g.dtype, device=flat_g.device)
    dist.reduce_scatter_tensor(g_shard, flat_g.contiguous(), op=dist.ReduceOp.SUM)
    g_shard.div_(world_size)

    m = exp_avg_shard.clone()
    v = exp_avg_sq_shard.clone()
    theta = flat_param_shard.clone()

    m.mul_(beta1).add_(g_shard, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(g_shard, g_shard, value=1.0 - beta2)
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    m_hat = m / bc1
    v_hat = v / bc2
    denom = v_hat.sqrt().add(eps)

    theta.add_(m_hat.div(denom), alpha=-lr)
    theta.add_(flat_param_shard, alpha=-lr * weight_decay)

    return theta, m, v