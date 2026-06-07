import torch
import torch.distributed as dist
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def block_fp8_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs).to(tl.float32)
    
    # FP8 E4M3 max value is 448.0
    s = tl.max(tl.abs(x)) / 448.0
    
    # Prevent division by zero if all elements in the block are 0
    s_safe = tl.where(s == 0.0, 1.0, s)
    
    y = (x / s_safe).to(y_ptr.dtype.element_ty)
    
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)

def solution(local_tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert local_tensor.size(-1) % block_size == 0, "Last dimension must be divisible by block_size"
    
    y_local = torch.empty_like(local_tensor, dtype=torch.float8_e4m3fn)
    s_local = local_tensor.new_empty(
        *local_tensor.size()[:-1], local_tensor.size(-1) // block_size, dtype=torch.float32
    )
    
    grid = (triton.cdiv(local_tensor.numel(), block_size),)
    block_fp8_quant_kernel[grid](local_tensor, y_local, s_local, BLOCK_SIZE=block_size)
    
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
