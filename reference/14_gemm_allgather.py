import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    M, K = A.shape
    K_B, N_local = B.shape
    
    A = A.contiguous()
    B = B.contiguous()
    C_local = torch.matmul(A, B)
    
    C_gathered = [torch.zeros_like(C_local) for _ in range(world_size)]
    dist.all_gather(C_gathered, C_local)
    
    C = torch.cat(C_gathered, dim=1)
    
    return C
