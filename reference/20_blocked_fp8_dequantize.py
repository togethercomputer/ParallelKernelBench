import torch
import torch.distributed as dist
import triton
import triton.language as tl

@triton.jit
def block_fp8_dequant_kernel(y_ptr, s_ptr, x_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    s = tl.load(s_ptr + pid)

    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)

    tl.store(x_ptr + offs, y * s, mask=mask)


def solution(
    local_y: torch.Tensor,
    local_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    world_size = dist.get_world_size()

    chunk_shape = local_y.shape[1:]
    chunk_numel = local_y.numel() // world_size
    num_elements = local_y.numel()
    assert chunk_numel % block_size == 0, (
        f"Chunk size {chunk_numel} must be divisible by block_size ({block_size})"
    )

    y_flat = local_y.view(-1)
    s_flat = local_s.view(-1)
    x_flat = torch.empty(num_elements, device=local_y.device, dtype=torch.float32)

    if num_elements > 0:
        grid = (triton.cdiv(num_elements, block_size),)
        block_fp8_dequant_kernel[grid](
            y_flat, s_flat, x_flat, num_elements, BLOCK_SIZE=block_size
        )

    x = x_flat.view(world_size, *chunk_shape)
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)

    return out
