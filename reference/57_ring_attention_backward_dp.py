from typing import Optional, Tuple

import torch
import torch.distributed as dist


class RingComm:
    def __init__(self, group: dist.ProcessGroup):
        self._group = group
        self._ops: list = []
        self._reqs = None
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.send_rank = dist.get_global_rank(group, (self.rank + 1) % self.world_size)
        self.recv_rank = dist.get_global_rank(group, (self.rank - 1) % self.world_size)

    def send_recv(self, to_send: torch.Tensor, recv_buf: Optional[torch.Tensor] = None) -> torch.Tensor:
        buf = recv_buf if recv_buf is not None else torch.empty_like(to_send)
        self._ops.append(dist.P2POp(dist.isend, to_send, self.send_rank, group=self._group))
        self._ops.append(dist.P2POp(dist.irecv, buf, self.recv_rank, group=self._group))
        return buf

    def commit(self):
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        for r in self._reqs:
            r.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(self, k: torch.Tensor, v: torch.Tensor):
        next_k = self.send_recv(k)
        next_v = self.send_recv(v)
        self.commit()
        return next_k, next_v


def _local_attn_backward(
    dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    out: torch.Tensor, softmax_lse: torch.Tensor,
    scale: float, causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    doh = dout.transpose(1, 2).float()
    outh = out.transpose(1, 2).float()

    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        sq, sk = q.size(1), k.size(1)
        mask = torch.triu(torch.ones(sq, sk, device=q.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = torch.exp(scores - softmax_lse)
    dP = torch.matmul(doh, vh.transpose(-2, -1))
    row_dot = (doh * outh).sum(dim=-1, keepdim=True)
    dS = probs * (dP - row_dot)

    dQ = torch.matmul(dS, kh) * scale
    dK = torch.matmul(dS.transpose(-2, -1), qh) * scale
    dV = torch.matmul(probs.transpose(-2, -1), doh)

    return (
        dQ.transpose(1, 2).contiguous(),
        dK.transpose(1, 2).contiguous(),
        dV.transpose(1, 2).contiguous(),
    )


def _ring_attn_backward(
    group: dist.ProcessGroup,
    dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    out: torch.Tensor, softmax_lse: torch.Tensor,
    scale: float, causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size(group)
    lse_4d = softmax_lse.unsqueeze(-1)

    if world_size == 1:
        dq, dk, dv = _local_attn_backward(dout, q, k, v, out, lse_4d, scale, causal)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)

    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step <= kv_comm.rank or not causal:
            block_dq, block_dk, block_dv = _local_attn_backward(
                dout, q, k, v, out, lse_4d, scale, causal=(causal and step == 0),
            )
            if dq is None:
                dq = block_dq.float()
                dk = block_dk.float()
                dv = block_dv.float()
            else:
                dq = dq + block_dq.float()
                d_kv_comm.wait()
                dk = block_dk.float() + next_dk
                dv = block_dv.float() + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


def solution(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_group = cp_group or dist.group.WORLD
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    dq, dk, dv = _ring_attn_backward(
        cp_group, dout, q.contiguous(), k.contiguous(), v.contiguous(),
        out, softmax_lse, float(softmax_scale), causal,
    )

    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        dp_size = dist.get_world_size(dp_group)
        for g in (dq, dk, dv):
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=dp_group)
            g.div_(dp_size)

    return dq, dk, dv
