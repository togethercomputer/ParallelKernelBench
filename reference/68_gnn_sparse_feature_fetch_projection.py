from typing import Optional

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    local_embedding_shard: torch.Tensor,
    input_node_ids: torch.Tensor,
    proj_matrix: torch.Tensor,
    num_total_nodes: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    shard_size = (num_total_nodes + world_size - 1) // world_size
    embed_dim = local_embedding_shard.shape[1]
    num_queries = input_node_ids.shape[0]

    owner = (input_node_ids // shard_size).clamp(max=world_size - 1)
    sort_idx = torch.argsort(owner, stable=True)
    sorted_ids = input_node_ids[sort_idx]
    sorted_owner = owner[sort_idx]

    send_counts = torch.zeros(
        world_size, dtype=torch.long, device=input_node_ids.device
    )
    send_counts.scatter_add_(0, sorted_owner, torch.ones_like(sorted_owner))
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    send_splits = send_counts.to("cpu").tolist()
    recv_splits = recv_counts.to("cpu").tolist()
    recv_ids = torch.empty(
        int(recv_counts.sum().item()),
        dtype=input_node_ids.dtype,
        device=input_node_ids.device,
    )
    dist.all_to_all_single(
        recv_ids,
        sorted_ids,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=group,
    )

    local_ids = (recv_ids - rank * shard_size).long()
    fetched = local_embedding_shard[local_ids]

    gathered = torch.empty(
        (num_queries, embed_dim), dtype=fetched.dtype, device=fetched.device
    )
    dist.all_to_all_single(
        gathered,
        fetched,
        output_split_sizes=send_splits,
        input_split_sizes=recv_splits,
        group=group,
    )
    emb = gathered[torch.argsort(sort_idx, stable=True)]

    return emb @ proj_matrix
