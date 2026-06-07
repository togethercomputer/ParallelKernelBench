from typing import Optional

import torch
import torch.distributed as dist


def _zigzag_indices(num_chunks: int, device: torch.device) -> torch.Tensor:
    half = (num_chunks + 1) // 2
    left = torch.arange(half, device=device)
    right = torch.arange(num_chunks - 1, half - 1, -1, device=device)
    indices = torch.empty(num_chunks, dtype=torch.long, device=device)
    indices[0::2] = left
    indices[1::2] = right
    return indices


def _inverse_zigzag_indices(num_chunks: int, device: torch.device) -> torch.Tensor:
    half = num_chunks // 2
    left = torch.arange(half, device=device)
    right = torch.arange(num_chunks - 1, half - 1, -1, device=device)
    indices = torch.empty(num_chunks, dtype=torch.long, device=device)
    indices[0::2] = left
    indices[1::2] = right
    return torch.argsort(indices)


def _a2a_split_to_full(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    with_zigzag_splitting: bool,
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    batch, global_channels, local_seq = x.shape
    local_channels = global_channels // world_size
    seq_len = local_seq * world_size

    send = (
        x.reshape(batch, world_size, local_channels, local_seq)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    out = (
        recv.permute(1, 2, 0, 3)
        .reshape(batch, local_channels, seq_len)
        .contiguous()
    )

    if with_zigzag_splitting:
        num_chunks = 2 * world_size
        index = _inverse_zigzag_indices(num_chunks, out.device)
        out = (
            out.reshape(batch, local_channels, num_chunks, seq_len // num_chunks)
            .index_select(dim=2, index=index)
            .reshape(batch, local_channels, seq_len)
        )
    return out


def _a2a_full_to_split(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    with_zigzag_splitting: bool,
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    batch, local_channels, seq_len = x.shape
    local_seq = seq_len // world_size

    if with_zigzag_splitting:
        num_chunks = 2 * world_size
        index = _zigzag_indices(num_chunks, x.device)
        x = (
            x.reshape(batch, local_channels, num_chunks, seq_len // num_chunks)
            .index_select(dim=2, index=index)
            .reshape(batch, local_channels, seq_len)
        )

    send = (
        x.reshape(batch, local_channels, world_size, local_seq)
        .permute(2, 0, 1, 3)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.permute(1, 0, 2, 3)
        .reshape(batch, world_size * local_channels, local_seq)
        .contiguous()
    )


def _fftconv_ref(
    u: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    seq_len = u.shape[-1]
    fft_size = 2 * seq_len
    u_float = u.float()
    kernel_float = kernel.float()

    kernel_f = torch.fft.rfft(kernel_float, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u_float, n=fft_size)
    y = torch.fft.irfft(
        u_f * kernel_f.unsqueeze(0), n=fft_size, norm="forward"
    )[..., :seq_len]
    y = y + u_float * bias.float().unsqueeze(-1)
    return y.to(dtype=u.dtype)


@torch.no_grad()
def solution(
    x1_seq: torch.Tensor,
    x2_seq: torch.Tensor,
    v_seq: torch.Tensor,
    h: torch.Tensor,
    conv_bias: torch.Tensor,
    num_groups: int,
    group_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    with_zigzag_splitting: bool = True,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    x1 = _a2a_split_to_full(x1_seq, group, with_zigzag_splitting)
    x2 = _a2a_split_to_full(x2_seq, group, with_zigzag_splitting)
    v = _a2a_split_to_full(v_seq, group, with_zigzag_splitting)

    local_channels = x1.shape[1]
    local_groups = num_groups // world_size
    h_local = h[rank * local_groups : (rank + 1) * local_groups]
    h_local = h_local.repeat_interleave(group_dim, dim=0)
    bias_local = conv_bias[rank * local_channels : (rank + 1) * local_channels]

    z = x2 * v
    z = _fftconv_ref(z, h_local, bias_local)
    z = x1 * z
    z_seq = _a2a_full_to_split(z, group, with_zigzag_splitting)
    return z_seq.transpose(1, 2).contiguous()