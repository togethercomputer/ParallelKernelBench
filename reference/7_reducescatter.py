import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    chunk_size = tensor.shape[0] // world_size
    out = tensor.new_empty((chunk_size,) + tensor.shape[1:])
    dist.reduce_scatter_tensor(out, tensor, op=dist.ReduceOp.SUM)
    return out
