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
        acc = torch.tensor(
            0.0,
            device=next((t.device for t in grad_tensors if t is not None), torch.device("cuda", 0)),
            dtype=torch.float32,
        )
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
    non_ep_grad_tensors: List[torch.Tensor],
    ep_grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    ep_size: int = 1,
    fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    if ep_size > 1 and ep_grad_tensors:
        scale = 1.0 / float(ep_size)
        for t in ep_grad_tensors:
            if t is not None:
                t.detach().mul_(scale)

    non_ep_total = _fsdp2_reduce_group(
        non_ep_grad_tensors,
        norm_type=norm_type,
        reduce_groups=[("fsdp", fsdp_group)],
    )

    ep_total = _fsdp2_reduce_group(
        ep_grad_tensors,
        norm_type=norm_type,
        reduce_groups=[("ep_fsdp", ep_fsdp_group), ("ep", ep_group)],
    )

    total_norm = (non_ep_total + ep_total) ** (1.0 / float(norm_type))

    if total_norm > max_norm:
        coef = (max_norm / total_norm)
        for t in non_ep_grad_tensors:
            if t is not None:
                t.mul_(coef.to(t.device))
        for t in ep_grad_tensors:
            if t is not None:
                t.mul_(coef.to(t.device))

    return total_norm
