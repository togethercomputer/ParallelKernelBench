import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

_CONV3D_NUMEL_LIMIT = 2**31


def _to_3tuple(value: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    return (value, value, value) if isinstance(value, int) else value


def _ceil_to_divisible(n: int, dividend: int) -> int:
    return math.ceil(dividend / (dividend // n))


def _output_shape(
    input_shape: torch.Size,
    out_channels: int,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> List[int]:
    shape = [input_shape[0], out_channels]
    for idx, size in enumerate(input_shape[-3:]):
        out = (size + 2 * padding[idx] - dilation[idx] * (kernel_size[idx] - 1) - 1)
        shape.append(math.floor(out / stride[idx] + 1))
    return shape


def _chunk_count(numel: int, channels: int, limit: int) -> int:
    chunks = math.ceil(numel / limit)
    return _ceil_to_divisible(chunks, channels)


def _channel_chunk_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    groups: int,
    numel_limit: int,
) -> torch.Tensor:
    out_channels, in_channels = weight.shape[:2]
    output_shape = _output_shape(
        x.shape,
        out_channels,
        tuple(weight.shape[2:]),
        stride,
        padding,
        dilation,
    )
    in_chunks = _chunk_count(x.numel(), in_channels, numel_limit)
    out_chunks = _chunk_count(math.prod(output_shape), out_channels, numel_limit)
    if in_chunks == 1 and out_chunks == 1:
        return F.conv3d(x, weight, bias, stride, padding, dilation, groups)

    x_chunks = x.chunk(in_chunks, dim=1)
    weight_out_chunks = weight.chunk(out_chunks, dim=0)
    bias_chunks = bias.chunk(out_chunks) if bias is not None else [None] * out_chunks
    outputs: List[torch.Tensor] = []
    for weight_chunk, bias_chunk in zip(weight_out_chunks, bias_chunks):
        partial_sum: Optional[torch.Tensor] = None
        for x_chunk, w_chunk in zip(x_chunks, weight_chunk.chunk(in_chunks, dim=1)):
            partial = F.conv3d(
                x_chunk,
                w_chunk,
                None,
                stride,
                padding,
                dilation,
                groups,
            ).float()
            partial_sum = partial if partial_sum is None else partial_sum + partial
        if partial_sum is None:
            raise RuntimeError("conv3d chunking produced no partial outputs")
        out = partial_sum.to(dtype=x.dtype)
        if bias_chunk is not None:
            out = out + bias_chunk.view(1, -1, 1, 1, 1)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def solution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Union[int, Tuple[int, int, int]],
    padding: Union[int, Tuple[int, int, int]],
    dilation: Union[int, Tuple[int, int, int]],
    groups: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    out = _channel_chunk_conv3d(
        input,
        weight,
        None,
        _to_3tuple(stride),
        _to_3tuple(padding),
        _to_3tuple(dilation),
        groups,
        _CONV3D_NUMEL_LIMIT,
    )
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out
