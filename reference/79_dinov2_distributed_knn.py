from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


def _topk_with_labels(
    similarity: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    topk_sims, indices = similarity.topk(k, dim=1, largest=True, sorted=True)
    topk_labels = torch.gather(labels.expand(similarity.shape[0], -1), 1, indices)
    return topk_sims, topk_labels


def _broadcast_queries(
    queries: torch.Tensor,
    source: int,
    rank: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    shape = torch.tensor(queries.shape, dtype=torch.long, device=queries.device)
    dist.broadcast(shape, src=source, group=group)
    if rank == source:
        out = queries.contiguous()
    else:
        out = queries.new_empty(tuple(int(v) for v in shape.tolist()))
    dist.broadcast(out, src=source, group=group)
    return out


def _local_candidates(
    queries: torch.Tensor,
    train_features_t: torch.Tensor,
    train_labels: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    similarity = queries @ train_features_t
    return _topk_with_labels(similarity, train_labels, k)


def _merge_on_owner(
    topk_sims: torch.Tensor,
    topk_labels: torch.Tensor,
    owner: int,
    rank: int,
    world_size: int,
    k: int,
    group: dist.ProcessGroup,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    gathered_sims: Optional[List[torch.Tensor]] = None
    gathered_labels: Optional[List[torch.Tensor]] = None
    if rank == owner:
        gathered_sims = [torch.empty_like(topk_sims) for _ in range(world_size)]
        gathered_labels = [torch.empty_like(topk_labels) for _ in range(world_size)]

    dist.gather(topk_sims, gather_list=gathered_sims, dst=owner, group=group)
    dist.gather(topk_labels, gather_list=gathered_labels, dst=owner, group=group)
    if rank != owner:
        return None

    all_sims = torch.cat(gathered_sims, dim=1)
    all_labels = torch.cat(gathered_labels, dim=1)
    return _topk_with_labels(all_sims, all_labels, k)


@torch.no_grad()
def solution(
    test_features_rank: torch.Tensor,
    train_features_rank_T: torch.Tensor,
    train_labels_rank: torch.Tensor,
    max_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    if max_k > train_features_rank_T.shape[1]:
        raise ValueError("max_k must not exceed the local train shard size")

    result: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    for owner in range(world_size):
        queries = _broadcast_queries(test_features_rank, owner, rank, group)
        topk_sims, topk_labels = _local_candidates(
            queries,
            train_features_rank_T,
            train_labels_rank,
            max_k,
        )
        merged = _merge_on_owner(
            topk_sims,
            topk_labels,
            owner,
            rank,
            world_size,
            max_k,
            group,
        )
        if merged is not None:
            result = merged

    if result is None:
        raise RuntimeError("k-NN ring did not produce local results")
    return result
