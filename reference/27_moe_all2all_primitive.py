from typing import List, Optional, Union

import torch
import torch.distributed as dist


def solution(
    local_tensor: torch.Tensor,
    input_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    output_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return local_tensor.contiguous()

    local_tensor = local_tensor.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(local_tensor)
    else:
        out_size = sum(output_split_sizes) if isinstance(output_split_sizes, list) else int(output_split_sizes.sum().item())
        output = torch.empty(
            (out_size, local_tensor.size(1)),
            dtype=local_tensor.dtype,
            device=local_tensor.device,
        )
    dist.all_to_all_single(
        output,
        local_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output
