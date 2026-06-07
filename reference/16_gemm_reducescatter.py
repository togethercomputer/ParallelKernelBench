import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B_local: torch.Tensor,
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    M, K_local = A_local.shape
    K_B, N = B_local.shape
    
    M_local = M // world_size
    
    A_local = A_local.contiguous()
    B_local = B_local.contiguous()
    C_partial = torch.matmul(A_local, B_local)
    
    C_local = torch.empty((M_local, N), dtype=C_partial.dtype, device=C_partial.device)
    dist.reduce_scatter_tensor(C_local, C_partial, op=dist.ReduceOp.SUM)
    
    return C_local
