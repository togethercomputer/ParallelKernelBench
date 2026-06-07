from typing import Optional

import torch
import torch.distributed as dist


def _broadcast_data(
    rank: int,
    world_size: int,
    data: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    if world_size == 1:
        return data

    sizes = torch.zeros(world_size, dtype=torch.long, device=data.device)
    sizes[rank] = data.shape[0]
    dist.all_reduce(sizes, op=dist.ReduceOp.SUM, group=group)

    send_splits = [data.shape[0] for _ in range(world_size)]
    recv_splits = sizes.to("cpu").tolist()
    send = data.repeat(*([world_size] + [1] * (data.ndim - 1))).contiguous()
    recv = torch.empty(
        (int(sizes.sum().item()), *data.shape[1:]),
        dtype=data.dtype,
        device=data.device,
    )
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=group,
    )
    return recv


def _calc_ranking(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    scores = torch.cat([pos_score.view(-1, 1), neg_score], dim=1)
    _, indices = torch.sort(torch.sigmoid(scores), dim=1, descending=True)
    return torch.nonzero(indices == 0)[:, 1].view(-1).detach() + 1


@torch.no_grad()
def solution(
    local_pos_scores: torch.Tensor,
    local_neg_scores: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    pos_scores = _broadcast_data(rank, world_size, local_pos_scores, group)
    neg_scores = _broadcast_data(rank, world_size, local_neg_scores, group)
    return _calc_ranking(pos_scores, neg_scores)