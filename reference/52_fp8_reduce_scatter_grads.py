from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

_FP8_E4M3_MAX = 448.0


@torch.no_grad()
def _update_amax_history(amax_history: Tensor, cur_abs_max: Tensor) -> Tensor:
    out = torch.roll(amax_history, shifts=-1, dims=0)
    out[-1] = cur_abs_max.to(dtype=out.dtype)
    return out


@torch.no_grad()
def _fp8_round_trip_bf16(x: Tensor, scale: Tensor) -> Tensor:
    xf = x.float()
    qs = xf / scale
    q = qs.to(torch.float8_e4m3fn)
    return (q.float() * scale).to(dtype=x.dtype)


@torch.no_grad()
def solution(flat_grads: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    world_size = dist.get_world_size()
    n = flat_grads.numel()
    shard_elems = n // world_size

    cur_abs_max = flat_grads.abs().max().to(torch.float32)
    updated_hist = _update_amax_history(amax_history, cur_abs_max)

    scale = updated_hist.max().clamp(min=1e-12).to(torch.float32) / _FP8_E4M3_MAX
    recon = _fp8_round_trip_bf16(flat_grads, scale)

    out_shard = torch.empty(shard_elems, dtype=flat_grads.dtype, device=flat_grads.device)
    dist.reduce_scatter_tensor(out_shard, recon.contiguous(), op=dist.ReduceOp.SUM)
    out_shard.div_(world_size)

    return out_shard, updated_hist
