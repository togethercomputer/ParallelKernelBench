from typing import List, Optional

import torch
import torch.distributed as dist


def _shift(chunks: List[torch.Tensor], group: dist.ProcessGroup) -> List[torch.Tensor]:
    cutoff = len(chunks) - dist.get_rank(group)
    return chunks[cutoff:] + chunks[:cutoff]


def _all_to_all(
    outputs: List[torch.Tensor],
    inputs: List[torch.Tensor],
    group: dist.ProcessGroup,
) -> None:
    outputs = _shift(list(outputs), group)
    inputs = _shift(list(inputs), group)
    if outputs and outputs[0].is_cuda:
        dist.all_to_all(outputs, inputs, group=group)
        return

    output_splits = [out.size(0) for out in outputs]
    input_splits = [inp.size(0) for inp in inputs]
    flat_out = torch.cat(outputs) if outputs else torch.empty(0)
    flat_in = torch.cat(inputs) if inputs else torch.empty(0)
    dist.all_to_all_single(
        flat_out,
        flat_in,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    for out, temp in zip(outputs, flat_out.split(output_splits)):
        out.copy_(temp)


@torch.no_grad()
def solution(
    local_features: torch.Tensor,
    seed_inverse_ids: torch.Tensor,
    counts_sent: List[int],
    counts_received: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    gathered = local_features[seed_inverse_ids]
    out = local_features.new_empty((sum(counts_sent),) + local_features.shape[1:])
    _all_to_all(
        list(torch.split(out, counts_sent)),
        list(torch.split(gathered, counts_received)),
        group,
    )
    return out