# FP8 (E4M3) gradient reduce-scatter with the fp8 payload on the wire.
#
# Each rank quantizes its flat gradient with a delayed-scaling factor (amax
# history), exchanges the fp8 chunks (as raw bytes) plus each rank's scalar
# scale with an all-to-all / all-gather, then dequantizes the received chunks
# with the sender's scale and averages them into the local shard. NCCL cannot
# reduce fp8 directly, so dequant + reduce happens on the receiving rank —
# a natural target for a fused kernel.
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
def solution(flat_grads: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    world_size = dist.get_world_size()
    n = flat_grads.numel()
    shard_elems = n // world_size

    cur_abs_max = flat_grads.abs().max().to(torch.float32)
    updated_hist = _update_amax_history(amax_history, cur_abs_max)

    scale = updated_hist.max().clamp(min=1e-12).to(torch.float32) / _FP8_E4M3_MAX

    # Quantize the full local gradient with the rank-local delayed scale.
    q = (flat_grads.float() / scale).to(torch.float8_e4m3fn).contiguous()

    # Every peer needs the sender's scale to dequantize.
    scales = torch.empty(world_size, dtype=torch.float32, device=flat_grads.device)
    dist.all_gather_into_tensor(scales, scale.reshape(1))

    # Ship fp8 chunks as raw bytes: rank r receives chunk r from every peer.
    q_recv_u8 = torch.empty(n, dtype=torch.uint8, device=flat_grads.device)
    dist.all_to_all_single(q_recv_u8, q.view(torch.uint8))

    # Dequantize each received chunk with its sender's scale, then average.
    chunks = q_recv_u8.view(torch.float8_e4m3fn).float().reshape(world_size, shard_elems)
    out_shard = (chunks * scales.unsqueeze(1)).sum(dim=0).div_(world_size)

    return out_shard.to(dtype=flat_grads.dtype), updated_hist
