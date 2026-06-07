from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _zigzag_get_overlapping_patches(
    data: torch.Tensor,
    seq_dim: int,
    overlap_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = list(data.shape)
    shape[seq_dim : seq_dim + 1] = [2, data.shape[seq_dim] // 2]
    chunks = data.reshape(shape)

    order = list(range(chunks.dim()))
    order.insert(0, order.pop(seq_dim))
    chunks = chunks.permute(order)

    chunk_len = chunks.shape[seq_dim + 1]
    overlaps = chunks.narrow(seq_dim + 1, chunk_len - overlap_size, overlap_size)
    return overlaps[0], overlaps[1]


@torch.no_grad()
def solution(
    x: torch.Tensor,
    weight: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    group_ranks = dist.get_process_group_ranks(group)
    group_rank = dist.get_rank(group)
    group_world_size = len(group_ranks)

    batch, hidden, local_seq = x.shape
    chunk_len = local_seq // 2
    pad_size = weight.shape[-1] - 1
    chunk_a, chunk_b = _zigzag_get_overlapping_patches(x, 2, pad_size)

    ops = []
    recv_prev_a = None
    recv_next_b = None

    if group_rank > 0:
        recv_prev_a = torch.empty_like(chunk_a)
        ops.append(
            dist.P2POp(dist.irecv, recv_prev_a, group_ranks[group_rank - 1], group)
        )
    if group_rank < group_world_size - 1:
        ops.append(
            dist.P2POp(
                dist.isend,
                chunk_a.contiguous(),
                group_ranks[group_rank + 1],
                group,
            )
        )

    if group_rank < group_world_size - 1:
        recv_next_b = torch.empty_like(chunk_b)
        ops.append(
            dist.P2POp(dist.irecv, recv_next_b, group_ranks[group_rank + 1], group)
        )
    if group_rank > 0:
        ops.append(
            dist.P2POp(
                dist.isend,
                chunk_b.contiguous(),
                group_ranks[group_rank - 1],
                group,
            )
        )

    for request in dist.batch_isend_irecv(ops):
        request.wait()

    if recv_prev_a is None:
        recv_prev_a = torch.zeros_like(chunk_a)
    if recv_next_b is None:
        recv_next_b = chunk_a.clone().contiguous()

    # Move the two zigzag chunks into batch so both use the same grouped conv.
    x_chunks = x.reshape(batch, hidden, 2, chunk_len).permute(2, 0, 1, 3)
    x_chunks = x_chunks.reshape(2 * batch, hidden, chunk_len)
    padding = torch.cat([recv_prev_a, recv_next_b], dim=0)
    x_padded = torch.cat([padding, x_chunks], dim=-1)

    y = F.conv1d(x_padded, weight, bias=None, stride=1, padding=0, groups=hidden)
    return (
        y.reshape(2, batch, hidden, chunk_len)
        .permute(1, 2, 0, 3)
        .reshape(batch, hidden, local_seq)
    )