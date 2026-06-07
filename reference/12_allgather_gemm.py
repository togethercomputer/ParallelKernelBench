import torch
import torch.distributed as dist


@torch.no_grad()
def solution(A_local: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    A_gathered = [torch.empty_like(A_local) for _ in range(world_size)]
    dist.all_gather(A_gathered, A_local)
    A_global = torch.cat(A_gathered, dim=1)
    return torch.matmul(A_global, B)
