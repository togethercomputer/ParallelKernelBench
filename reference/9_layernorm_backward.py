import torch
import torch.distributed as dist


@torch.no_grad()
def solution(X_hat: torch.Tensor, dY: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d_beta = dY.sum(dim=0)
    d_gamma = (dY * X_hat).sum(dim=0)
    dist.all_reduce(d_beta, op=dist.ReduceOp.SUM)
    dist.all_reduce(d_gamma, op=dist.ReduceOp.SUM)
    return d_gamma, d_beta
