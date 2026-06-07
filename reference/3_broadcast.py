import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    out = tensor.clone()
    dist.broadcast(out, src=src)
    return out
