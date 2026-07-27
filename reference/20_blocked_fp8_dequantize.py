# Blocked FP8 dequantize + all-to-all.
#
# Each rank holds `world_size` quantized chunks (one destined for each peer)
# plus their per-block scales. The reference dequantizes locally (per-block
# scale multiply) and exchanges the reconstructed chunks with a single
# all_to_all. A fused kernel can instead ship the compact fp8 payload and
# dequantize on the receiving side.
import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    local_y: torch.Tensor,
    local_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    world_size = dist.get_world_size()

    chunk_shape = local_y.shape[1:]
    chunk_numel = local_y.numel() // world_size
    assert chunk_numel % block_size == 0, (
        f"Chunk size {chunk_numel} must be divisible by block_size ({block_size})"
    )

    y_blocks = local_y.float().contiguous().reshape(-1, block_size)
    s_flat = local_s.float().contiguous().reshape(-1, 1)
    x = (y_blocks * s_flat).reshape(world_size, *chunk_shape)

    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)

    return out
