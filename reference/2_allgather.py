import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    out = tensor.new_empty((world_size,) + tensor.shape)
    dist.all_gather_into_tensor(out, tensor)
    return out
