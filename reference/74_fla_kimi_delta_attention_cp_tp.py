from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _all_gather_sequence(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    batch, local_seq = x.shape[:2]
    out = torch.empty(
        batch,
        world_size * local_seq,
        *x.shape[2:],
        dtype=x.dtype,
        device=x.device,
    )
    send = x.transpose(0, 1).contiguous()
    recv = torch.empty(
        world_size * local_seq,
        batch,
        *x.shape[2:],
        dtype=x.dtype,
        device=x.device,
    )
    dist.all_gather_into_tensor(recv, send, group=group)
    return recv.transpose(0, 1).contiguous().view_as(out)


def _kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype

    dt_bias = dt_bias.float().reshape(heads, key_dim)
    a_scale = a_log.float().exp().view(1, 1, heads, 1)
    decay = torch.exp(lower_bound * torch.sigmoid(a_scale * (g.float() + dt_bias)))
    beta = beta.float().sigmoid()

    scale = float(key_dim) ** -0.5
    q_float = F.normalize(q.float(), p=2, dim=-1) * scale
    k_float = F.normalize(k.float(), p=2, dim=-1)
    v_float = v.float()

    q_float = q_float.permute(0, 2, 1, 3).contiguous()
    k_float = k_float.permute(0, 2, 1, 3).contiguous()
    v_float = v_float.permute(0, 2, 1, 3).contiguous()
    decay = decay.permute(0, 2, 1, 3).contiguous()
    beta = beta.permute(0, 2, 1).contiguous()

    batch_heads = batch * heads
    q_float = q_float.reshape(batch_heads, seq_len, key_dim)
    k_float = k_float.reshape(batch_heads, seq_len, key_dim)
    v_float = v_float.reshape(batch_heads, seq_len, value_dim)
    decay = decay.reshape(batch_heads, seq_len, key_dim)
    beta = beta.reshape(batch_heads, seq_len)

    state = torch.zeros(
        batch_heads, key_dim, value_dim, dtype=torch.float32, device=q.device
    )
    output = torch.empty(
        batch_heads, seq_len, value_dim, dtype=torch.float32, device=q.device
    )
    for step in range(seq_len):
        q_t = q_float[:, step]
        k_t = k_float[:, step]
        v_t = v_float[:, step]
        state = decay[:, step].unsqueeze(-1) * state
        projected = torch.bmm(k_t.unsqueeze(1), state).squeeze(1)
        update = (v_t - projected) * beta[:, step].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * update.unsqueeze(1)
        output[:, step] = torch.bmm(q_t.unsqueeze(1), state).squeeze(1)

    output = output.reshape(batch, heads, seq_len, value_dim)
    output = output.permute(0, 2, 1, 3).contiguous()
    return output.to(dtype=out_dtype)


@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    cp_group = cp_group or dist.group.WORLD
    cp_size = dist.get_world_size(group=cp_group)
    cp_rank = dist.get_rank(group=cp_group)

    if cp_size > 1:
        q_full = _all_gather_sequence(q, cp_group)
        k_full = _all_gather_sequence(k, cp_group)
        v_full = _all_gather_sequence(v, cp_group)
        g_full = _all_gather_sequence(g, cp_group)
        beta_full = _all_gather_sequence(beta, cp_group)
    else:
        q_full, k_full, v_full, g_full, beta_full = q, k, v, g, beta

    out = _kda_forward(
        q_full,
        k_full,
        v_full,
        g_full,
        beta_full,
        a_log,
        dt_bias,
        lower_bound=-5.0,
    )
    if tp_group is not None and dist.get_world_size(group=tp_group) > 1:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)

    if cp_size == 1:
        return out
    local_seq = q.shape[1]
    start = cp_rank * local_seq
    return out[:, start : start + local_seq].contiguous()