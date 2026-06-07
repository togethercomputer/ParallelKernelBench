import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out
