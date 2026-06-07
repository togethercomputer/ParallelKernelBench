import torch
import torch.distributed as dist
from typing import Tuple

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)

def solution(
    q_local: torch.Tensor, 
    k_local: torch.Tensor, 
    cos_local: torch.Tensor, 
    sin_local: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape cos and sin to broadcast with q and k's head dimension (dim=2)
    cos = cos_local.unsqueeze(2)
    sin = sin_local.unsqueeze(2)
    
    q_embed_local = (q_local * cos) + (rotate_half(q_local) * sin)
    k_embed_local = (k_local * cos) + (rotate_half(k_local) * sin)
    
    if not dist.is_initialized():
        return q_embed_local, k_embed_local
        
    world_size = dist.get_world_size()
    
    q_gather_list = [torch.empty_like(q_embed_local) for _ in range(world_size)]
    k_gather_list = [torch.empty_like(k_embed_local) for _ in range(world_size)]
    
    dist.all_gather(q_gather_list, q_embed_local.contiguous())
    dist.all_gather(k_gather_list, k_embed_local.contiguous())
    
    q_embed_global = torch.cat(q_gather_list, dim=1)
    k_embed_global = torch.cat(k_gather_list, dim=1)
    
    return q_embed_global, k_embed_global
