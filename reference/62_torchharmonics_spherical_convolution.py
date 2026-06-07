from typing import List, Optional

import torch
import torch.distributed as dist


def _compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    if num_chunks == 1:
        return [size]
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)
    return [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]


def _transpose(
    tensor: torch.Tensor,
    dim0: int,
    dim1: int,
    dim1_split_sizes: List[int],
    group: dist.ProcessGroup,
) -> tuple[list[torch.Tensor], list[int]]:
    comm_size = dist.get_world_size(group=group)
    comm_rank = dist.get_rank(group=group)

    tsplit = torch.split(tensor, _compute_split_shapes(tensor.shape[dim0], comm_size), dim=dim0)
    x_send = [y.contiguous() for y in tsplit]
    x_send_shapes = [x.shape for x in x_send]
    x_recv = []
    x_shape = list(x_send_shapes[comm_rank])
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device))

    dist.all_to_all(x_recv, x_send, group=group)
    dim0_split_sizes = [x[dim0] for x in x_send_shapes]
    return x_recv, dim0_split_sizes


def _disco_s2_contraction_torch(
    x: torch.Tensor,
    psi: torch.Tensor,
    nlon_out: int,
) -> torch.Tensor:
    psi = psi.to(x.device)
    batch_size, n_chans, nlat_in, nlon_in = x.shape
    kernel_size, nlat_out, _ = psi.shape
    pscale = nlon_in // nlon_out

    x = x.reshape(1, batch_size * n_chans, nlat_in, nlon_in).permute(0, 2, 3, 1)
    x = x.expand(kernel_size, -1, -1, -1)

    y = torch.zeros(
        nlon_out,
        kernel_size,
        nlat_out,
        batch_size * n_chans,
        device=x.device,
        dtype=x.dtype,
    )

    for pout in range(nlon_out):
        y[pout] = torch.bmm(psi, x.reshape(kernel_size, nlat_in * nlon_in, -1))
        x = torch.roll(x, -pscale, dims=2)

    y = y.permute(3, 1, 2, 0).reshape(batch_size, n_chans, kernel_size, nlat_out, nlon_out)
    return y


@torch.no_grad()
def solution(
    x: torch.Tensor,
    psi: torch.Tensor,
    weight: torch.Tensor,
    groups: int,
    nlon_out: int,
    nlon_in: int,
    azimuth_group: Optional[dist.ProcessGroup] = None,
    polar_group: Optional[dist.ProcessGroup] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    azimuth_group = azimuth_group or dist.group.WORLD
    polar_group = polar_group or dist.group.WORLD
    azimuth_size = dist.get_world_size(group=azimuth_group)
    polar_size = dist.get_world_size(group=polar_group)
    polar_rank = dist.get_rank(group=polar_group)

    lon_in_shapes = _compute_split_shapes(nlon_in, azimuth_size)
    num_chans = x.shape[1]

    if azimuth_size > 1:
        xlist, _ = _transpose(x, dim0=1, dim1=-1, dim1_split_sizes=lon_in_shapes, group=azimuth_group)
        x = torch.cat(xlist, dim=-1)

    x = _disco_s2_contraction_torch(x, psi, nlon_out)

    if polar_size > 1:
        dtype = x.dtype
        xf = x.float().contiguous()
        dist.all_reduce(xf, group=polar_group)
        x = xf.to(dtype)

    if polar_size > 1:
        split_shapes = _compute_split_shapes(x.shape[-2], polar_size)
        x = list(torch.split(x, split_shapes, dim=-2))[polar_rank]

    if azimuth_size > 1:
        chan_shapes = _compute_split_shapes(num_chans, azimuth_size)
        xlist, _ = _transpose(x, dim0=-1, dim1=1, dim1_split_sizes=chan_shapes, group=azimuth_group)
        x = torch.cat(xlist, dim=1)

    B, C, K, H, W = x.shape
    groupsize = C // groups
    x = x.reshape(B, groups, groupsize, K, H, W)
    out = torch.einsum(
        "bgckxy,gock->bgoxy",
        x,
        weight.reshape(groups, -1, weight.shape[1], weight.shape[2]),
    ).contiguous()
    out = out.reshape(out.shape[0], -1, H, W)

    if bias is not None:
        out = out + bias.reshape(1, -1, 1, 1)

    return out
