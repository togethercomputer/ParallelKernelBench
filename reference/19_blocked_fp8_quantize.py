# Blocked FP8 (E4M3) quantization + all-gather.
#
# Each rank quantizes its local tensor in contiguous blocks of `block_size`
# elements (per-block absmax scaling to the E4M3 range), then all ranks
# all-gather both the quantized payload and the per-block scales. The fp8
# payload crosses the wire as raw bytes (uint8 view).
from typing import Tuple

import torch
import torch.distributed as dist

_FP8_E4M3_MAX = 448.0


@torch.no_grad()
def solution(local_tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert local_tensor.size(-1) % block_size == 0, "Last dimension must be divisible by block_size"

    orig_shape = local_tensor.shape
    x = local_tensor.float().contiguous().reshape(-1, block_size)

    s = x.abs().amax(dim=1) / _FP8_E4M3_MAX
    # Prevent division by zero if all elements in the block are 0
    s_safe = torch.where(s == 0.0, torch.ones_like(s), s)

    y_local = (x / s_safe.unsqueeze(1)).to(torch.float8_e4m3fn).reshape(orig_shape)
    s_local = s.reshape(*orig_shape[:-1], orig_shape[-1] // block_size)

    if dist.is_initialized():
        world_size = dist.get_world_size()

        y_local_u8 = y_local.view(torch.uint8)
        y_gather_u8 = [torch.empty_like(y_local_u8) for _ in range(world_size)]
        dist.all_gather(y_gather_u8, y_local_u8)

        y_global_u8 = torch.cat(y_gather_u8, dim=0)
        y_global = y_global_u8.view(torch.float8_e4m3fn)

        s_gather = [torch.empty_like(s_local) for _ in range(world_size)]
        dist.all_gather(s_gather, s_local)

        s_global = torch.cat(s_gather, dim=0)
    else:
        y_global = y_local
        s_global = s_local

    return y_global, s_global
