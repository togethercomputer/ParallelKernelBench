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
    exp_avg_W1: Tensor,
    exp_avg_b1: Tensor,
    exp_avg_W2: Tensor,
    exp_avg_b2: Tensor,
    exp_avg_sq_W1: Tensor,
    exp_avg_sq_b1: Tensor,
    exp_avg_sq_W2: Tensor,
    exp_avg_sq_b2: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, ...]:
    world_size = dist.get_world_size()

    params = [W1, b1, W2, b2]
    exp_avg = [exp_avg_W1, exp_avg_b1, exp_avg_W2, exp_avg_b2]
    exp_avg_sq = [exp_avg_sq_W1, exp_avg_sq_b1, exp_avg_sq_W2, exp_avg_sq_b2]

    flat_params = _flatten_dense_tensors(params)
    dist.broadcast(flat_params, src=0)
    broadcast_params = _unflatten_dense_tensors(flat_params, params)
    params = [t.detach().requires_grad_(True) for t in broadcast_params]

    flat_m = _flatten_dense_tensors(exp_avg)
    dist.broadcast(flat_m, src=0)
    exp_avg = list(_unflatten_dense_tensors(flat_m, exp_avg))

    flat_v = _flatten_dense_tensors(exp_avg_sq)
    dist.broadcast(flat_v, src=0)
    exp_avg_sq = list(_unflatten_dense_tensors(flat_v, exp_avg_sq))

    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()

    grads = [p.grad for p in params]
    flat_grad = _flatten_dense_tensors(grads)
    dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
    flat_grad.div_(world_size)
    avg_grads = _unflatten_dense_tensors(flat_grad, grads)
    for p, g in zip(params, avg_grads):
        p.grad.copy_(g)

    assert step >= 1
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    for p, m_buf, v_buf in zip(params, exp_avg, exp_avg_sq):
        g = p.grad
        m_buf.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v_buf.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        m_hat = m_buf / bc1
        v_hat = v_buf / bc2
        denom = v_hat.sqrt().add(eps)
        p.data.add_(m_hat.div(denom).mul(-lr))

    out_tensors = tuple(list(params) + exp_avg + exp_avg_sq)
    return out_tensors
