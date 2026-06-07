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
    lam = torch.as_tensor(kalman_lambda, dtype=weights[0].dtype, device=weights[0].device)
    err = error.to(device=weights[0].device, dtype=weights[0].dtype)

    tmp = 0
    for i in range(weights_num):
        tmp = tmp + (lam + torch.matmul(torch.matmul(H[i].T, P[i]), H[i]))

    if dist.is_initialized():
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)

    A = 1 / tmp

    for i in range(weights_num):
        K = torch.matmul(P[i], H[i])
        weights[i] = weights[i] + A * err * K
        P[i] = (1 / lam) * (P[i] - A * torch.matmul(K, K.T))

    if dist.is_initialized():
        device = weights[0].device
        local_shape = [tensor.shape[0] for tensor in weights]

        shape_list = [
            torch.zeros_like(torch.empty(1), dtype=torch.float64, device=device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(shape_list, local_shape)

        weight_tensor = torch.cat([w.reshape(-1) for w in weights], dim=0).to(torch.float64)
        world_shape = [sum(inner_list) for inner_list in shape_list]

        weight_list = [
            torch.zeros(world_shape[i], dtype=torch.float64, device=device)
            for i in range(len(world_shape))
        ]
        dist.all_gather(weight_list, weight_tensor)

        result: List[torch.Tensor] = []
        for i in range(dist.get_world_size()):
            result = result + [
                t.reshape(-1, 1).to(dtype=weights[0].dtype)
                for t in torch.split(weight_list[i], shape_list[i])
            ]
        weights = result

    kalman_lambda_next = (
        torch.as_tensor(kalman_nue, dtype=lam.dtype, device=lam.device) * lam
        + 1
        - torch.as_tensor(kalman_nue, dtype=lam.dtype, device=lam.device)
    )

    return weights, P, kalman_lambda_next
