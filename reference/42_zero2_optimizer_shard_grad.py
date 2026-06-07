from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def solution(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_part: Tensor,
    exp_avg_sq_part: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    templates = [W1, b1, W2, b2]
    flat_p = _flatten_dense_tensors(templates)
    dist.broadcast(flat_p, src=0)

    param_views = _unflatten_dense_tensors(flat_p, templates)
    params = [t.detach().requires_grad_(True) for t in param_views]

    part = exp_avg_part.numel()
    assert flat_p.numel() == part * world_size

    m_part = exp_avg_part.clone()
    v_part = exp_avg_sq_part.clone()

    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    flat_g = _flatten_dense_tensors([p.grad for p in params]).contiguous()
    g_part = torch.empty(part, dtype=flat_g.dtype, device=flat_g.device)
    dist.reduce_scatter_tensor(g_part, flat_g, op=dist.ReduceOp.SUM)
    g_part.div_(world_size)

    start = rank * part
    w_part = flat_p[start : start + part].clone()

    assert step >= 1
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    m_part.mul_(beta1).add_(g_part, alpha=1.0 - beta1)
    v_part.mul_(beta2).addcmul_(g_part, g_part, value=1.0 - beta2)
    m_hat = m_part / bc1
    v_hat = v_part / bc2
    w_part.add_(m_hat.div(v_hat.sqrt().add(eps)).mul(-lr))

    gathered = torch.empty_like(flat_p)
    dist.all_gather_into_tensor(gathered, w_part.contiguous())
    flat_p.copy_(gathered)

    out_params = _unflatten_dense_tensors(flat_p, templates)
    return (*out_params, m_part, v_part)
