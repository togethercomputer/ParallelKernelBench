from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


def _sample_one_hop_csc_dist(
    input_nodes: torch.Tensor,
    k: int,
    colptr: torch.Tensor,
    row: torch.Tensor,
    replace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    n = input_nodes.numel()
    sampled_nodes = []
    sampled_edges = []
    cumsum = [n]

    for i in range(n):
        v = int(input_nodes[i].item())
        start = int(colptr[v].item())
        end = int(colptr[v + 1].item())
        deg = end - start
        take = min(k, deg) if k >= 0 else deg

        if take > 0:
            if replace:
                perm = torch.randint(deg, (take,), device=input_nodes.device)
            else:
                perm = torch.randperm(deg, device=input_nodes.device)[:take]
            sampled_nodes.append(row[start:end].index_select(0, perm))
            sampled_edges.append(torch.arange(start, end, device=input_nodes.device).index_select(0, perm))

        cumsum.append(cumsum[-1] + take)

    nbr_tensor = (
        torch.cat(sampled_nodes)
        if sampled_nodes
        else torch.empty(0, dtype=torch.long, device=input_nodes.device)
    )
    eid_tensor = (
        torch.cat(sampled_edges)
        if sampled_edges
        else torch.empty(0, dtype=torch.long, device=input_nodes.device)
    )
    return torch.cat([input_nodes, nbr_tensor]), eid_tensor, cumsum


def _remove_duplicates(out_node: torch.Tensor, node: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_nodes = node.numel()
    node_combined = torch.cat([node, out_node])
    _, idx = np.unique(node_combined.cpu().numpy(), return_index=True)
    idx = torch.from_numpy(idx).to(node.device).sort().values
    node = node_combined[idx]
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
    send_nodes_list: List[torch.Tensor],
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size(group)
    device = send_nodes_list[0].device
    send_counts = torch.tensor([x.numel() for x in send_nodes_list], dtype=torch.long, device=device)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    send_nodes = torch.cat(send_nodes_list) if send_nodes_list else torch.empty(0, dtype=torch.long, device=device)
    recv_nodes = torch.empty(int(recv_counts.sum().item()), dtype=torch.long, device=device)
    dist.all_to_all_single(
        recv_nodes,
        send_nodes,
        input_split_sizes=send_counts.cpu().tolist(),
        output_split_sizes=recv_counts.cpu().tolist(),
        group=group,
    )
    return recv_nodes, send_counts, recv_counts


def _exchange_replies(
    sampled_nodes: torch.Tensor,
    sampled_edges: torch.Tensor,
    sampled_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size(group)
    device = sampled_nodes.device
    recv_splits = recv_counts.cpu().tolist()
    send_node_counts = torch.empty(world_size, dtype=torch.long, device=device)
    offset = 0
    for r, count in enumerate(recv_splits):
        send_node_counts[r] = sampled_counts[offset : offset + count].sum()
        offset += count

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
        input_split_sizes=recv_splits,
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

        partition_ids = node_to_rank[src].to(torch.long)
        partition_orders = torch.empty_like(partition_ids)
        send_nodes_list = []
        send_pos_list = []
        for r in range(world_size):
            pos = (partition_ids == r).nonzero(as_tuple=False).flatten()
            partition_orders[pos] = torch.arange(pos.numel(), dtype=torch.long, device=device)
            send_nodes_list.append(src[pos])
            send_pos_list.append(pos)

        recv_nodes, send_counts, recv_counts = _exchange_nodes(send_nodes_list, group)
        node_out, edge_out, cumsum = _sample_one_hop_csc_dist(
            recv_nodes, int(fanout), local_adj_row_ptr, local_adj_col, replace
        )

        seed_size = recv_nodes.numel()
        sampled_nodes = node_out[seed_size:]
        sampled_counts = torch.tensor(
            np.subtract(np.array(cumsum[1:]), np.array(cumsum[:-1])),
            dtype=torch.long,
            device=device,
        )

        reply_nodes, reply_edges, reply_counts = _exchange_replies(
            sampled_nodes, edge_out, sampled_counts, recv_counts, group
        )

        rank_offsets = torch.cat(
            [send_counts.new_zeros(1), torch.cumsum(send_counts, dim=0)[:-1]]
        )
        grouped_index = rank_offsets[partition_ids] + partition_orders
        node_chunks = list(torch.split(reply_nodes, reply_counts.cpu().tolist()))
        edge_chunks = list(torch.split(reply_edges, reply_counts.cpu().tolist()))

        ordered_nodes = []
        ordered_edges = []
        ordered_dst = []
        for idx in grouped_index.tolist():
            ordered_nodes.append(node_chunks[idx])
            ordered_edges.append(edge_chunks[idx])
        for dst_node, count in zip(src, reply_counts[grouped_index]):
            ordered_dst.append(dst_node.repeat(int(count.item())))

        out_node = torch.cat(ordered_nodes) if ordered_nodes else seed.new_empty(0)
        out_edge = torch.cat(ordered_edges) if ordered_edges else seed.new_empty(0)
        out_dst = torch.cat(ordered_dst) if ordered_dst else seed.new_empty(0)
        if out_node.numel() == 0:
            break

        src, node = _remove_duplicates(out_node, node)
        node_with_dupl.append(out_node)
        dst_with_dupl.append(out_dst)
        edge.append(out_edge)

    node_dupl = torch.cat(node_with_dupl)
    dst_dupl = torch.cat(dst_with_dupl)
    row, col = _relabel_neighborhood(node, dst_dupl, node_dupl)
    return node, row, col, torch.cat(edge)