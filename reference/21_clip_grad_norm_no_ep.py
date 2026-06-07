from typing import List, Optional

import torch
import torch.distributed as dist


def _local_pth_sum(grad_tensors: List[torch.Tensor], p: float) -> torch.Tensor:
    dev = None
    acc = None
    for g in grad_tensors:
        if g is None:
            continue
        g_local = g
        if dev is None:
            dev = g_local.device
            acc = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.norm(g_local.detach().to(torch.float32), p=p)
        acc = acc + (gn ** p)
    if acc is None:
        acc = torch.tensor(0.0, device=next((t.device for t in grad_tensors if t is not None), torch.device("cuda", 0)), dtype=torch.float32)
    return acc


def _fsdp2_reduce_group(
    grad_tensors: List[torch.Tensor],
    norm_type: float,
    reduce_groups: List[tuple],
) -> torch.Tensor:
    p = float(norm_type)
    val = _local_pth_sum(grad_tensors, p)
    for _, group in reduce_groups:
        if group is not None:
            dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
    return val


def solution(
    grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    fsdp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    reduce_groups = [("fsdp", fsdp_group)]
    total_p = _fsdp2_reduce_group(grad_tensors, norm_type, reduce_groups)
    total_norm = total_p ** (1.0 / float(norm_type))

    if total_norm > max_norm:
        coef = (max_norm / total_norm)
        for t in grad_tensors:
            if t is not None:
                t.mul_(coef.to(t.device))

    return total_norm
