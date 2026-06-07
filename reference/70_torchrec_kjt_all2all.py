from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


def _sum_by_splits(values: List[int], splits: List[int]) -> List[int]:
    out: List[int] = []
    offset = 0
    for split in splits:
        out.append(sum(values[offset : offset + split]))
        offset += split
    return out


def _lengths_per_key(lengths: torch.Tensor, stride_per_key: List[int]) -> List[int]:
    out: List[int] = []
    offset = 0
    for stride in stride_per_key:
        out.append(int(lengths[offset : offset + stride].sum().item()))
        offset += stride
    return out


def _get_recat(
    local_split: int,
    num_splits: int,
    stagger: int = 1,
    device: Optional[torch.device] = None,
    batch_size_per_rank: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    if local_split == 0:
        return None

    feature_order = [
        x + num_splits // stagger * y
        for x in range(num_splits // stagger)
        for y in range(stagger)
    ]
    if batch_size_per_rank is None:
        recat = [
            feature_idx + rank_idx * local_split
            for feature_idx in range(local_split)
            for rank_idx in feature_order
        ]
    else:
        rank_offsets = [0]
        for batch_size in batch_size_per_rank[:-1]:
            rank_offsets.append(rank_offsets[-1] + local_split * batch_size)
        recat = [
            rank_offsets[rank_idx] + feature_idx * batch_size_per_rank[rank_idx] + b
            for feature_idx in range(local_split)
            for rank_idx in feature_order
            for b in range(batch_size_per_rank[rank_idx])
        ]
    return torch.tensor(recat, device=device, dtype=torch.int32)


def _permute_segments(
    data: torch.Tensor,
    segment_lengths: torch.Tensor,
    recat: torch.Tensor,
) -> torch.Tensor:
    segment_lengths = segment_lengths.to(device=data.device, dtype=torch.long)
    offsets = torch.zeros(
        segment_lengths.numel() + 1, dtype=torch.long, device=data.device
    )
    offsets[1:] = torch.cumsum(segment_lengths, dim=0)
    chunks = [
        data[int(offsets[idx].item()) : int(offsets[idx + 1].item())]
        for idx in recat.long().tolist()
    ]
    return torch.cat(chunks, dim=0) if chunks else data.new_empty((0,))


def _permute_2d_sparse_data(
    recat: torch.Tensor,
    lengths_2d: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    recat = recat.long()
    row_lengths = lengths_2d.sum(dim=1).to(torch.long)
    lengths_out = lengths_2d[recat]
    values_out = _permute_segments(values, row_lengths, recat)
    weights_out = None
    if weights is not None:
        weights_out = _permute_segments(weights, row_lengths, recat)
    return lengths_out, values_out, weights_out


def _all_to_all_tensor(
    tensor: torch.Tensor,
    input_splits: List[int],
    output_splits: List[int],
    pg: dist.ProcessGroup,
) -> Tuple[torch.Tensor, dist.Work]:
    output = torch.empty(
        (sum(output_splits),), dtype=tensor.dtype, device=tensor.device
    )
    work = dist.all_to_all_single(
        output,
        tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=pg,
        async_op=True,
    )
    return output, work


@torch.no_grad()
def solution(
    lengths: torch.Tensor,
    values: torch.Tensor,
    key_splits: List[int],
    batch_size: int,
    pg: Optional[dist.ProcessGroup] = None,
    weights: Optional[torch.Tensor] = None,
    stride_per_key: Optional[List[int]] = None,
    stagger: int = 1,
) -> Dict[str, torch.Tensor]:
    pg = pg or dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    device = lengths.device

    num_features = sum(key_splits)
    variable_stride = stride_per_key is not None
    if stride_per_key is None:
        stride_per_key = [batch_size] * num_features

    length_per_key = _lengths_per_key(lengths, stride_per_key)
    length_splits = _sum_by_splits(stride_per_key, key_splits)
    value_splits = _sum_by_splits(length_per_key, key_splits)

    input_splits = [length_splits, value_splits]
    input_tensors = [lengths, values]
    if variable_stride:
        input_splits.append(key_splits)
        input_tensors.append(
            torch.tensor(stride_per_key, dtype=torch.long, device=device)
        )
    if weights is not None:
        input_splits.append(value_splits)
        input_tensors.append(weights)

    split_tensors = [
        torch.tensor(splits, dtype=torch.long, device=device) for splits in input_splits
    ]
    if not variable_stride:
        split_tensors.append(
            torch.full((world_size,), batch_size, dtype=torch.long, device=device)
        )

    meta_input = torch.stack(split_tensors, dim=1).flatten()
    meta_output = torch.empty_like(meta_input)
    dist.all_to_all_single(meta_output, meta_input, group=pg)
    meta_rows = [
        [int(item) for item in row]
        for row in meta_output.view(world_size, -1).T.tolist()
    ]
    if variable_stride:
        output_splits = meta_rows
        stride_per_rank = None
    else:
        output_splits = meta_rows[:-1]
        stride_per_rank = meta_rows[-1]

    outputs: List[torch.Tensor] = []
    works: List[dist.Work] = []
    for tensor, in_splits, out_splits in zip(
        input_tensors, input_splits, output_splits
    ):
        output, work = _all_to_all_tensor(tensor, in_splits, out_splits, pg)
        outputs.append(output)
        works.append(work)
    for work in works:
        work.wait()

    recv_lengths = outputs[0]
    recv_values = outputs[1]
    recv_strides: Optional[torch.Tensor] = outputs[2] if variable_stride else None
    recv_weights: Optional[torch.Tensor] = None
    if weights is not None:
        recv_weights = outputs[-1]

    local_split = key_splits[rank]
    if variable_stride:
        assert recv_strides is not None
        recat = _get_recat(local_split, world_size, stagger, device=device)
        if recat is not None:
            value_segment_lengths = torch.tensor(
                _lengths_per_key(recv_lengths, recv_strides.to(torch.long).tolist()),
                dtype=torch.long,
                device=device,
            )
            recv_lengths = _permute_segments(recv_lengths, recv_strides, recat)
            recv_values = _permute_segments(recv_values, value_segment_lengths, recat)
            if recv_weights is not None:
                recv_weights = _permute_segments(
                    recv_weights, value_segment_lengths, recat
                )
        stride_per_key_per_rank = recv_strides.view(world_size, local_split).T
        if stagger > 1:
            order = (
                torch.arange(world_size, device=device)
                .view(stagger, -1)
                .T.reshape(-1)
            )
            stride_per_key_per_rank = stride_per_key_per_rank[:, order]
        result: Dict[str, torch.Tensor] = {
            "lengths": recv_lengths,
            "values": recv_values,
            "stride_per_key_per_rank": stride_per_key_per_rank,
        }
    else:
        assert stride_per_rank is not None
        single_batch_per_rank = all(
            stride == stride_per_rank[0] for stride in stride_per_rank
        )
        if single_batch_per_rank:
            recat = _get_recat(local_split, world_size, stagger, device=device)
            if recat is not None and stride_per_rank[0] > 0:
                lengths_2d, recv_values, recv_weights = _permute_2d_sparse_data(
                    recat,
                    recv_lengths.view(-1, stride_per_rank[0]),
                    recv_values,
                    recv_weights,
                )
                recv_lengths = lengths_2d.reshape(-1)
        else:
            recat = _get_recat(
                local_split,
                world_size,
                stagger,
                device=device,
                batch_size_per_rank=stride_per_rank,
            )
            if recat is not None:
                recv_values = _permute_segments(recv_values, recv_lengths, recat)
                if recv_weights is not None:
                    recv_weights = _permute_segments(recv_weights, recv_lengths, recat)
                recv_lengths = recv_lengths[recat.long()]
        result = {
            "lengths": recv_lengths,
            "values": recv_values,
            "stride": torch.tensor(sum(stride_per_rank), device=device),
            "stride_per_rank": torch.tensor(stride_per_rank, device=device),
        }

    if recv_weights is not None:
        result["weights"] = recv_weights
    return result