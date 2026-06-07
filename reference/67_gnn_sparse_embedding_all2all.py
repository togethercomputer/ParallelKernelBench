from typing import Optional, Tuple

import torch
import torch.distributed as dist


def _generate_permutation_remainder(
    idx: torch.Tensor,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    owner = (idx % world_size).long()
    send_splits = torch.bincount(owner, minlength=world_size)
    perm = torch.argsort(owner, stable=True).long()
    return perm, send_splits


@torch.no_grad()
def solution(
    idx: torch.Tensor,
    value: torch.Tensor,
    num_nodes: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return idx, value

    perm, send_splits = _generate_permutation_remainder(idx, world_size)

    recv_splits = torch.empty_like(send_splits)
    dist.all_to_all_single(recv_splits, send_splits, group=group)

    recv_splits = recv_splits.to("cpu", non_blocking=True)
    send_splits = send_splits.to("cpu", non_blocking=True)
    send_idx = idx[perm]
    send_value = value[perm]
    if idx.is_cuda:
        torch.cuda.current_stream().synchronize()

    recv_count = int(recv_splits.sum().item())
    recv_splits_list = recv_splits.tolist()
    send_splits_list = send_splits.tolist()

    recv_idx = torch.empty((recv_count,), dtype=idx.dtype, device=idx.device)
    dist.all_to_all_single(
        recv_idx,
        send_idx,
        output_split_sizes=recv_splits_list,
        input_split_sizes=send_splits_list,
        group=group,
    )

    recv_value = torch.empty(
        (recv_count, *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_to_all_single(
        recv_value,
        send_value,
        output_split_sizes=recv_splits_list,
        input_split_sizes=send_splits_list,
        group=group,
    )

    return recv_idx, recv_value
