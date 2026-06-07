import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    rank = dist.get_rank()

    if rank == src:
        world_size = dist.get_world_size()
        scatter_list = [chunk.squeeze(0).contiguous() for chunk in tensor.chunk(world_size, dim=0)]
        out = torch.empty_like(scatter_list[0])
    else:
        scatter_list = None
        out = tensor.clone()

    dist.scatter(out, scatter_list=scatter_list, src=src)
    return out
