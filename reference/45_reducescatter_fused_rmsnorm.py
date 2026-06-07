from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


@torch.no_grad()
def solution(
    rs_input_1d: Tensor,
    gamma: Tensor,
    eps: float,
) -> Tensor:
    world_size = dist.get_world_size()
    n = rs_input_1d.numel()
    chunk = n // world_size

    hidden = gamma.numel()
    assert chunk % hidden == 0, f"chunk ({chunk}) must divide hidden ({hidden})"
    rows = chunk // hidden

    out_flat = torch.empty(chunk, dtype=rs_input_1d.dtype, device=rs_input_1d.device)
    dist.reduce_scatter_tensor(out_flat, rs_input_1d.contiguous(), op=dist.ReduceOp.SUM)
    out_flat.div_(world_size)

    x = out_flat.view(rows, hidden).float()
    gn = gamma.float()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).add(eps))
    y = x * rms * gn
    return y.to(dtype=rs_input_1d.dtype)
