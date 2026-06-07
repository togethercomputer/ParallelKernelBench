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
def solution(flat_param_shard: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    world_size = dist.get_world_size()
    p = flat_param_shard.numel()

    cur_abs_max = flat_param_shard.abs().max().to(torch.float32)
    updated_hist = _update_amax_history(amax_history, cur_abs_max)

    scale = updated_hist.max().clamp(min=1e-12).to(torch.float32) / _FP8_E4M3_MAX
    recon = _fp8_round_trip_bf16(flat_param_shard, scale)

    full = torch.empty(world_size * p, dtype=flat_param_shard.dtype, device=flat_param_shard.device)
    dist.all_gather_into_tensor(full, recon.contiguous())

    return full, updated_hist
