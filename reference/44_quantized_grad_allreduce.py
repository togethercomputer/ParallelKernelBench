# Block-wise int8-quantized gradient all-reduce (quantized payload on the wire).
#
# Two-phase quantized all-reduce: each rank block-quantizes its gradient to
# int8, exchanges int8 chunks + fp32 scales with an all-to-all
# (reduce-scatter phase), dequantizes and averages its shard, re-quantizes the
# result, and all-gathers int8 shards + scales (all-gather phase). Only int8
# payloads and per-block scales cross the wire.
from __future__ import annotations

from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def _block_int8_quant(x_flat: Tensor, block_size: int) -> Tuple[Tensor, Tensor]:
    """Quantize a flat fp tensor (numel divisible by block_size) to int8 + fp32 scales."""
    xv = x_flat.view(-1, block_size)
    scales = xv.abs().amax(dim=1).float().clamp(min=1e-8) / 127.0
    q = (xv.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(-1), scales


@torch.no_grad()
def _block_int8_dequant(q: Tensor, scales: Tensor, block_size: int) -> Tensor:
    return q.view(-1, block_size).float() * scales.float().unsqueeze(1)


@torch.no_grad()
def solution(
    flat_grad: Tensor,
    block_size: int,
) -> Tensor:
    assert block_size >= 1

    world_size = dist.get_world_size()
    orig_shape = flat_grad.shape
    x = flat_grad.reshape(-1)
    n = x.numel()

    # Pad so every rank shard is a whole number of blocks.
    shard = -(-n // (world_size * block_size)) * block_size
    padded = world_size * shard
    if padded != n:
        x = F.pad(x, (0, padded - n))

    # Phase 1: quantized reduce-scatter via all_to_all of int8 chunks + scales.
    q, scales = _block_int8_quant(x, block_size)
    q_recv = torch.empty_like(q)
    dist.all_to_all_single(q_recv, q)
    s_recv = torch.empty_like(scales)
    dist.all_to_all_single(s_recv, scales)

    # Dequantize the world_size received chunks and average this rank's shard.
    chunks = _block_int8_dequant(q_recv, s_recv, block_size).reshape(world_size, shard)
    shard_sum = chunks.sum(dim=0).div_(world_size)

    # Phase 2: quantized all-gather of the reduced shards.
    q2, scales2 = _block_int8_quant(shard_sum, block_size)
    q_full = torch.empty(padded, dtype=torch.int8, device=x.device)
    dist.all_gather_into_tensor(q_full, q2)
    s_full = torch.empty(padded // block_size, dtype=torch.float32, device=x.device)
    dist.all_gather_into_tensor(s_full, scales2)

    out = _block_int8_dequant(q_full, s_full, block_size).reshape(-1)[:n]
    return out.to(dtype=flat_grad.dtype).reshape(orig_shape)
