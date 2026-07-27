# Distributed multi-hop GNN neighbor sampling over a partitioned CSC graph.
#
# Each hop: route frontier nodes to their owner rank (all_to_all), take the
# first `fanout` neighbors of each received node from the local CSC shard
# (deterministic selection so candidates can be checked bit-for-bit),
# return the sampled neighborhoods to the requesting ranks (all_to_all),
# dedup the new frontier, and finally relabel the sampled subgraph. All graph
# work is vectorized on-device; the host only touches the split-size metadata
# that all_to_all requires.
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


def _sample_one_hop_csc_dist(
    input_nodes: torch.Tensor,
    k: int,
    colptr: torch.Tensor,
    row: torch.Tensor,
    replace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministically take the first min(k, deg) neighbors of each input node.

    Returns (input_nodes ++ sampled_neighbors, sampled_edge_ids, per-node counts).
    """
    device = input_nodes.device
    n = input_nodes.numel()
    if n == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return input_nodes, empty, empty

    start = colptr[input_nodes]
    end = colptr[input_nodes + 1]
    deg = end - start
    take = deg.clamp(max=k) if k >= 0 else deg

    total = int(take.sum().item())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return input_nodes, empty, take

    seg = torch.repeat_interleave(torch.arange(n, device=device), take)
    out_ptr = torch.zeros(n + 1, dtype=torch.long, device=device)
    out_ptr[1:] = take.cumsum(0)
    local_off = torch.arange(total, device=device) - out_ptr[seg]
    eid = start[seg] + local_off
    nbr = row[eid]
    return torch.cat([input_nodes, nbr]), eid, take


def _remove_duplicates(out_node: torch.Tensor, node: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep first occurrences of cat([node, out_node]) in encounter order."""
    num_nodes = node.numel()
    combined = torch.cat([node, out_node])
    uniq, inverse = torch.unique(combined, return_inverse=True)
    first_idx = torch.full(
        (uniq.numel(),), combined.numel(), dtype=torch.long, device=combined.device
    )
    first_idx.scatter_reduce_(
        0,
        inverse,
        torch.arange(combined.numel(), device=combined.device),
        reduce="amin",
        include_self=True,
    )
    order = first_idx.sort().values
    node = combined[order]
    src = node[num_nodes:]
    return src, node


def _relabel_neighborhood(
    node: torch.Tensor,
    dst_with_dupl: torch.Tensor,
    node_with_dupl: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if node_with_dupl.numel() == 0:
        return node.new_empty(0), node.new_empty(0)

    assoc = torch.full(
        (int(node.max().item()) + 1,),
        -1,
        dtype=torch.long,
        device=node.device,
    )
    assoc[node] = torch.arange(node.numel(), device=node.device)
    row = assoc[node_with_dupl]
    col = assoc[dst_with_dupl]
    return row, col


def _exchange_nodes(
    send_nodes: torch.Tensor,
    send_counts: torch.Tensor,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = send_nodes.device
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    recv_nodes = torch.empty(int(recv_counts.sum().item()), dtype=torch.long, device=device)
    dist.all_to_all_single(
        recv_nodes,
        send_nodes,
        input_split_sizes=send_counts.cpu().tolist(),
        output_split_sizes=recv_counts.cpu().tolist(),
        group=group,
    )
    return recv_nodes, recv_counts


def _exchange_replies(
    sampled_nodes: torch.Tensor,
    sampled_edges: torch.Tensor,
    sampled_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = sampled_nodes.device
    # sampled_counts is grouped by requesting rank (recv_counts entries per rank).
    rank_bounds = torch.zeros(recv_counts.numel() + 1, dtype=torch.long, device=device)
    rank_bounds[1:] = recv_counts.cumsum(0)
    counts_cumsum = torch.zeros(sampled_counts.numel() + 1, dtype=torch.long, device=device)
    counts_cumsum[1:] = sampled_counts.cumsum(0)
    send_node_counts = counts_cumsum[rank_bounds[1:]] - counts_cumsum[rank_bounds[:-1]]

    reply_node_counts = torch.empty_like(send_node_counts)
    dist.all_to_all_single(reply_node_counts, send_node_counts, group=group)

    reply_count_counts = torch.empty_like(recv_counts)
    dist.all_to_all_single(reply_count_counts, recv_counts, group=group)

    reply_nodes = torch.empty(int(reply_node_counts.sum().item()), dtype=torch.long, device=device)
    reply_edges = torch.empty_like(reply_nodes)
    reply_counts = torch.empty(int(reply_count_counts.sum().item()), dtype=torch.long, device=device)

    dist.all_to_all_single(
        reply_nodes,
        sampled_nodes,
        input_split_sizes=send_node_counts.cpu().tolist(),
        output_split_sizes=reply_node_counts.cpu().tolist(),
        group=group,
    )
    dist.all_to_all_single(
        reply_edges,
        sampled_edges,
        input_split_sizes=send_node_counts.cpu().tolist(),
        output_split_sizes=reply_node_counts.cpu().tolist(),
        group=group,
    )
    dist.all_to_all_single(
        reply_counts,
        sampled_counts,
        input_split_sizes=recv_counts.cpu().tolist(),
        output_split_sizes=reply_count_counts.cpu().tolist(),
        group=group,
    )
    return reply_nodes, reply_edges, reply_counts


@torch.no_grad()
def solution(
    seed_nodes: torch.Tensor,
    fanouts: List[int],
    local_adj_row_ptr: torch.Tensor,
    local_adj_col: torch.Tensor,
    node_to_rank: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    replace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    device = seed_nodes.device

    seed = seed_nodes.to(dtype=torch.long, device=device)
    src = seed.clone()
    node = src.clone()
    node_with_dupl = [seed.new_empty(0)]
    dst_with_dupl = [seed.new_empty(0)]
    edge = [seed.new_empty(0)]

    for fanout in fanouts:
        if src.numel() == 0:
            break

        # Stable-partition the frontier by owner rank.
        partition_ids = node_to_rank[src].to(torch.long)
        order = torch.argsort(partition_ids, stable=True)
        send_nodes = src[order]
        send_counts = torch.bincount(partition_ids, minlength=world_size)

        recv_nodes, recv_counts = _exchange_nodes(send_nodes, send_counts, group)
        node_out, edge_out, sampled_counts = _sample_one_hop_csc_dist(
            recv_nodes, int(fanout), local_adj_row_ptr, local_adj_col, replace
        )
        sampled_nodes = node_out[recv_nodes.numel():]

        reply_nodes, reply_edges, reply_counts = _exchange_replies(
            sampled_nodes, edge_out, sampled_counts, recv_counts, group
        )

        # Reply chunks arrive in send order (frontier sorted by owner rank);
        # build gather indices that reorder them back to original frontier order.
        chunk_off = torch.zeros(reply_counts.numel() + 1, dtype=torch.long, device=device)
        chunk_off[1:] = reply_counts.cumsum(0)

        total = int(reply_counts.sum().item())
        if total == 0:
            break

        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=device)
        take = reply_counts[inv]  # counts per frontier node in original order
        seg = torch.repeat_interleave(torch.arange(src.numel(), device=device), take)
        out_ptr = torch.zeros(src.numel() + 1, dtype=torch.long, device=device)
        out_ptr[1:] = take.cumsum(0)
        local_off = torch.arange(total, device=device) - out_ptr[seg]
        gidx = chunk_off[inv][seg] + local_off

        out_node = reply_nodes[gidx]
        out_edge = reply_edges[gidx]
        out_dst = torch.repeat_interleave(src, take)

        src, node = _remove_duplicates(out_node, node)
        node_with_dupl.append(out_node)
        dst_with_dupl.append(out_dst)
        edge.append(out_edge)

    node_dupl = torch.cat(node_with_dupl)
    dst_dupl = torch.cat(dst_with_dupl)
    row, col = _relabel_neighborhood(node, dst_dupl, node_dupl)
    return node, row, col, torch.cat(edge)
