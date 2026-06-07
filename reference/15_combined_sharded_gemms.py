import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.no_grad()
def solution(
    x_local: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M = x_local.shape[0]
    M_local = M // world_size

    x_local = x_local.contiguous()
    shards = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(shards, x_local)
    x_full = torch.cat(shards, dim=1)

    a = F.silu(torch.matmul(x_full, W1))
    a_loc = a[rank * M_local : (rank + 1) * M_local].contiguous()
    block = torch.matmul(a_loc, W2)

    H = block.shape[1]
    buf = block.new_zeros((M, H))
    buf[rank * M_local : (rank + 1) * M_local].copy_(block)

    y_local = block.new_empty((M_local, H))
    dist.reduce_scatter_tensor(y_local, buf, op=dist.ReduceOp.SUM)
    return y_local
