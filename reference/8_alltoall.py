import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(tensor)
    dist.all_to_all_single(out, tensor)
    return out