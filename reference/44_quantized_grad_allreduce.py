from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def _block_int8_quant_dequant(x_flat: Tensor, block_size: int) -> Tensor:
    n = x_flat.numel()
    if n == 0:
        return x_flat.clone()
    flat = x_flat.contiguous().reshape(-1)
    pad = (-n) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    nb = flat.numel() // block_size
    xv = flat.view(nb, block_size)
    scales = xv.abs().amax(dim=1).float().clamp(min=1e-8) / 127.0
    q = (xv.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    out = (q.float() * scales.unsqueeze(1)).reshape(-1)
    return out[:n]


@torch.no_grad()
def solution(
    flat_grad: Tensor,
    block_size: int,
) -> Tensor:
    assert block_size >= 1

    world_size = dist.get_world_size()
    orig_shape = flat_grad.shape
    x = flat_grad.reshape(-1)

    rec = _block_int8_quant_dequant(x, block_size)
    acc = rec.float()
    dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    acc.div_(world_size)

    return acc.to(dtype=flat_grad.dtype).reshape(orig_shape)
