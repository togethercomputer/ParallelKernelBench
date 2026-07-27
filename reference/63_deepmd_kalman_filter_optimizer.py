# DeePMD-style layer-wise Kalman filter optimizer step (data-parallel).
#
# Each rank holds the same block structure (num_blocks blocks of identical
# size). The innovation denominator is all-reduced across ranks, each rank
# updates its local weight blocks and covariance blocks, and the updated
# weights are all-gathered so every rank sees every rank's blocks. Inputs are
# never mutated; per-block updates are independent (batched-GEMM / fusion
# opportunity) and the weight all-gather is a single fixed-size collective.
from typing import List, Tuple

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    H: List[torch.Tensor],
    error: torch.Tensor,
    weights: List[torch.Tensor],
    P: List[torch.Tensor],
    kalman_lambda: float,
    kalman_nue: float = 0.9987,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    weights_num = len(weights)
    dtype = weights[0].dtype
    device = weights[0].device
    lam = torch.as_tensor(kalman_lambda, dtype=dtype, device=device)
    err = error.to(device=device, dtype=dtype)

    tmp = torch.zeros((1, 1), dtype=dtype, device=device)
    for i in range(weights_num):
        tmp = tmp + (lam + torch.matmul(torch.matmul(H[i].T, P[i]), H[i]))

    if dist.is_initialized():
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)

    A = 1 / tmp

    new_weights: List[torch.Tensor] = []
    new_P: List[torch.Tensor] = []
    for i in range(weights_num):
        K = torch.matmul(P[i], H[i])
        new_weights.append(weights[i] + A * err * K)
        new_P.append((1 / lam) * (P[i] - A * torch.matmul(K, K.T)))

    out_weights = new_weights
    if dist.is_initialized():
        world_size = dist.get_world_size()
        block_sizes = [int(w.shape[0]) for w in new_weights]
        local_flat = torch.cat([w.reshape(-1) for w in new_weights], dim=0).contiguous()

        gathered = torch.empty(
            world_size * local_flat.numel(), dtype=dtype, device=device
        )
        dist.all_gather_into_tensor(gathered, local_flat)

        out_weights = []
        for r in range(world_size):
            rank_flat = gathered[r * local_flat.numel() : (r + 1) * local_flat.numel()]
            out_weights.extend(
                t.reshape(-1, 1) for t in torch.split(rank_flat, block_sizes)
            )

    kalman_lambda_next = (
        torch.as_tensor(kalman_nue, dtype=lam.dtype, device=lam.device) * lam
        + 1
        - torch.as_tensor(kalman_nue, dtype=lam.dtype, device=lam.device)
    )

    return out_weights, new_P, kalman_lambda_next
