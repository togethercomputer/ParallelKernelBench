from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _a2a_sequence_to_heads(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    batch, local_seq, heads, dim = x.shape
    local_heads = heads // world_size
    send = (
        x.reshape(batch, local_seq, world_size, local_heads, dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.permute(2, 0, 1, 3, 4)
        .reshape(batch, world_size * local_seq, local_heads, dim)
        .contiguous()
    )


def _a2a_heads_to_sequence(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    batch, seq_len, local_heads, dim = x.shape
    local_seq = seq_len // world_size
    send = (
        x.reshape(batch, world_size, local_seq, local_heads, dim)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.permute(2, 1, 0, 3, 4)
        .reshape(batch, local_seq, world_size * local_heads, dim)
        .contiguous()
    )


def _gated_delta_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    batch, seq_len, query_heads, key_dim = q.shape
    value_heads = v.shape[2]
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    scale = float(key_dim) ** -0.5

    assert value_heads % query_heads == 0
    repeat = value_heads // query_heads

    q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
    k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
    q = q.repeat_interleave(repeat, dim=2) * scale
    k = k.repeat_interleave(repeat, dim=2)

    a_scale = a_log.float().exp().view(1, 1, value_heads)
    dt_bias = dt_bias.float().view(1, 1, value_heads)
    decay = -a_scale * F.softplus(gate.float() + dt_bias)

    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.float().permute(0, 2, 1, 3).contiguous()
    decay = decay.exp().permute(0, 2, 1).contiguous()
    beta = beta.float().permute(0, 2, 1).contiguous()

    batch_heads = batch * value_heads
    q = q.reshape(batch_heads, seq_len, key_dim)
    k = k.reshape(batch_heads, seq_len, key_dim)
    v = v.reshape(batch_heads, seq_len, value_dim)
    decay = decay.reshape(batch_heads, seq_len)
    beta = beta.reshape(batch_heads, seq_len)

    state = torch.zeros(
        batch_heads, key_dim, value_dim, dtype=torch.float32, device=q.device
    )
    output = torch.empty(
        batch_heads, seq_len, value_dim, dtype=torch.float32, device=q.device
    )
    for step in range(seq_len):
        q_t = q[:, step]
        k_t = k[:, step]
        v_t = v[:, step]
        state = state * decay[:, step].view(batch_heads, 1, 1)
        projected = torch.bmm(k_t.unsqueeze(1), state).squeeze(1)
        update = (v_t - projected) * beta[:, step].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * update.unsqueeze(1)
        output[:, step] = torch.bmm(q_t.unsqueeze(1), state).squeeze(1)

    output = output.reshape(batch, value_heads, seq_len, value_dim)
    output = output.permute(0, 2, 1, 3).contiguous()
    return output.to(dtype=out_dtype)


@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    q_head = _a2a_sequence_to_heads(q, group)
    k_head = _a2a_sequence_to_heads(k, group)
    v_head = _a2a_sequence_to_heads(v, group)
    gate_head = _a2a_sequence_to_heads(gate.unsqueeze(-1), group).squeeze(-1)
    beta_head = _a2a_sequence_to_heads(beta.unsqueeze(-1), group).squeeze(-1)

    out = _gated_delta_recurrent(
        q_head,
        k_head,
        v_head,
        gate_head,
        beta_head,
        a_log,
        dt_bias,
    )
    return _a2a_heads_to_sequence(out, group)