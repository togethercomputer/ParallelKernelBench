# FP8 (E4M3) parameter all-gather with the fp8 payload on the wire.
#
# Each rank quantizes its parameter shard with a delayed-scaling factor (amax
# history), all-gathers the fp8 bytes plus each rank's scalar scale, and
# dequantizes the assembled full parameter with per-sender scales. The wire
# traffic is 1/2 of bf16; dequantization on the receiving side is the fusion
# opportunity.
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
def solution(flat_param_shard: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    world_size = dist.get_world_size()
    p = flat_param_shard.numel()

    cur_abs_max = flat_param_shard.abs().max().to(torch.float32)
    updated_hist = _update_amax_history(amax_history, cur_abs_max)

    scale = updated_hist.max().clamp(min=1e-12).to(torch.float32) / _FP8_E4M3_MAX

    q = (flat_param_shard.float() / scale).to(torch.float8_e4m3fn).contiguous()

    scales = torch.empty(world_size, dtype=torch.float32, device=flat_param_shard.device)
    dist.all_gather_into_tensor(scales, scale.reshape(1))

    q_full_u8 = torch.empty(world_size * p, dtype=torch.uint8, device=flat_param_shard.device)
    dist.all_gather_into_tensor(q_full_u8, q.view(torch.uint8))

    full = (
        q_full_u8.view(torch.float8_e4m3fn).float().reshape(world_size, p)
        * scales.unsqueeze(1)
    ).reshape(-1)

    return full.to(dtype=flat_param_shard.dtype), updated_hist
