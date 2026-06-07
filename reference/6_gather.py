import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    rank = dist.get_rank()

    if rank == dst:
        world_size = dist.get_world_size()
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None

    dist.gather(tensor, gather_list=gather_list, dst=dst)

    if rank == dst:
        return torch.stack(gather_list, dim=0)
    return tensor
