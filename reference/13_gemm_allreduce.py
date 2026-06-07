import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B_local: torch.Tensor,
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    M, K = A_local.shape
    K_B, N = B_local.shape
    
    A_local = A_local.contiguous()
    B_local = B_local.contiguous()
    C_local = torch.matmul(A_local, B_local)
    
    C = C_local.clone()
    dist.all_reduce(C, op=dist.ReduceOp.SUM)
    
    return C
