from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


def _local_sizes(group: dist.ProcessGroup, device: torch.device, local_n: int) -> List[int]:
    world_size = dist.get_world_size(group=group)
    size = torch.tensor([local_n], dtype=torch.long, device=device)
    gathered = [torch.empty_like(size) for _ in range(world_size)]
    dist.all_gather(gathered, size, group=group)
    return [int(item.item()) for item in gathered]


def _active_rank_info(rank: int, sizes: List[int]) -> Tuple[List[int], int]:
    active = [idx for idx, size in enumerate(sizes) if size > 0]
    sort_rank = active.index(rank) if rank in active else -1
    return active, sort_rank


def _extract_samples(
    sorted_local: torch.Tensor,
    sort_rank: int,
    n_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sort_rank < 0 or sorted_local.numel() == 0:
        values = sorted_local.new_full((n_samples,), float("inf"))
        ranks = torch.full((n_samples,), -1, dtype=torch.long, device=sorted_local.device)
        positions = torch.full_like(ranks, -1)
        return values, ranks, positions

    local_n = sorted_local.numel()
    sample_idx = torch.arange(n_samples, dtype=torch.long, device=sorted_local.device)
    valid_count = min(n_samples, local_n)
    values = sorted_local.new_full((n_samples,), float("inf"))
    ranks = torch.full((n_samples,), -1, dtype=torch.long, device=sorted_local.device)
    positions = torch.full_like(ranks, -1)
    if n_samples < local_n:
        valid_positions = ((sample_idx + 1) * local_n).div(n_samples, rounding_mode="floor") - 1
    else:
        valid_positions = sample_idx[:valid_count]
    values[:valid_count] = sorted_local[valid_positions[:valid_count]]
    ranks[:valid_count] = sort_rank
    positions[:valid_count] = valid_positions[:valid_count]
    return values, ranks, positions


def _gather_splitters(
    sample_values: torch.Tensor,
    sample_ranks: torch.Tensor,
    sample_positions: torch.Tensor,
    active_count: int,
    group: dist.ProcessGroup,
) -> List[Tuple[float, int, int]]:
    world_size = dist.get_world_size(group=group)
    value_parts = [torch.empty_like(sample_values) for _ in range(world_size)]
    rank_parts = [torch.empty_like(sample_ranks) for _ in range(world_size)]
    pos_parts = [torch.empty_like(sample_positions) for _ in range(world_size)]
    dist.all_gather(value_parts, sample_values, group=group)
    dist.all_gather(rank_parts, sample_ranks, group=group)
    dist.all_gather(pos_parts, sample_positions, group=group)

    values = torch.cat(value_parts).detach().cpu().tolist()
    ranks = torch.cat(rank_parts).detach().cpu().tolist()
    positions = torch.cat(pos_parts).detach().cpu().tolist()
    samples = [
        (float(value), int(sample_rank), int(position))
        for value, sample_rank, position in zip(values, ranks, positions)
        if int(sample_rank) >= 0
    ]
    samples.sort(key=lambda item: (item[0], item[1], item[2]))

    splitters: List[Tuple[float, int, int]] = []
    usable = len(samples)
    for sort_rank in range(active_count - 1):
        index = (sort_rank + 1) * usable // active_count - 1
        splitters.append(samples[max(0, min(index, usable - 1))])
    return splitters


def _split_positions(
    sorted_local: torch.Tensor,
    splitters: List[Tuple[float, int, int]],
    sort_rank: int,
) -> List[int]:
    if sort_rank < 0:
        return [0] * (len(splitters) + 2)

    boundaries = [0]
    for value, splitter_rank, splitter_position in splitters:
        probe = torch.tensor(value, dtype=sorted_local.dtype, device=sorted_local.device)
        if sort_rank > splitter_rank:
            end = int(torch.searchsorted(sorted_local, probe, right=False).item())
        elif sort_rank < splitter_rank:
            end = int(torch.searchsorted(sorted_local, probe, right=True).item())
        else:
            end = int(splitter_position) + 1
        boundaries.append(max(boundaries[-1], min(end, sorted_local.numel())))
    boundaries.append(sorted_local.numel())
    return boundaries


def _variable_all_to_all(
    send_chunks: List[torch.Tensor],
    group: dist.ProcessGroup,
) -> List[torch.Tensor]:
    device = send_chunks[0].device
    dtype = send_chunks[0].dtype
    send_counts = torch.tensor(
        [chunk.numel() for chunk in send_chunks], dtype=torch.long, device=device
    )
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    send = (
        torch.cat(send_chunks, dim=0)
        if int(send_counts.sum().item()) > 0
        else torch.empty(0, dtype=dtype, device=device)
    )
    recv = torch.empty(int(recv_counts.sum().item()), dtype=dtype, device=device)
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=recv_counts.cpu().tolist(),
        input_split_sizes=send_counts.cpu().tolist(),
        group=group,
    )

    outputs: List[torch.Tensor] = []
    offset = 0
    for count in recv_counts.cpu().tolist():
        next_offset = offset + int(count)
        outputs.append(recv[offset:next_offset])
        offset = next_offset
    return outputs


def _merge_sorted(chunks: List[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    chunks = [chunk for chunk in chunks if chunk.numel() > 0]
    if not chunks:
        return like.new_empty(0)
    return torch.cat(chunks, dim=0).sort().values


def _target_range(rank: int, world_size: int, total: int) -> Tuple[int, int]:
    base = total // world_size
    extra = total % world_size
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end


def _redistribute_exact(merged: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    sizes = _local_sizes(group, merged.device, merged.numel())
    total = sum(sizes)

    bucket_start = sum(sizes[:rank])
    bucket_end = bucket_start + merged.numel()
    send_chunks: List[torch.Tensor] = []
    for dest in range(world_size):
        target_start, target_end = _target_range(dest, world_size, total)
        start = max(bucket_start, target_start)
        end = min(bucket_end, target_end)
        if start < end:
            send_chunks.append(merged[start - bucket_start : end - bucket_start])
        else:
            send_chunks.append(merged.new_empty(0))
    return torch.cat(_variable_all_to_all(send_chunks, group), dim=0)


@torch.no_grad()
def solution(local_shard: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    sorted_local = local_shard.sort().values

    initial_sizes = _local_sizes(group, local_shard.device, local_shard.numel())
    active_ranks, sort_rank = _active_rank_info(rank, initial_sizes)
    active_count = len(active_ranks)
    if active_count == 0:
        return local_shard.new_empty(0)

    sample_values, sample_ranks, sample_positions = _extract_samples(
        sorted_local, sort_rank, active_count
    )
    splitters = _gather_splitters(
        sample_values, sample_ranks, sample_positions, active_count, group
    )
    boundaries = _split_positions(sorted_local, splitters, sort_rank)

    send_chunks = [sorted_local.new_empty(0) for _ in range(world_size)]
    for bucket, dest_rank in enumerate(active_ranks):
        send_chunks[dest_rank] = sorted_local[boundaries[bucket] : boundaries[bucket + 1]].contiguous()

    received = _variable_all_to_all(send_chunks, group)
    merged = _merge_sorted(received, sorted_local)
    return _redistribute_exact(merged, group)
