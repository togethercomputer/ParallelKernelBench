import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    out = tensor.clone()
    dist.reduce(out, dst=dst, op=dist.ReduceOp.SUM)
    return out
