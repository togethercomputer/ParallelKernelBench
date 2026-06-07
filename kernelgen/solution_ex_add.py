"""
Distributed element-wise vector add across two ranks. Each rank holds one local vector. The result is local + peer (same shape on every rank).
Implemented with torch.distributed.all_gather.
"""

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(2)]
    dist.all_gather(gathered, tensor.contiguous())
    out = gathered[0] + gathered[1]
    return out.reshape_as(tensor)
