from typing import Optional, Tuple
 
import torch
import torch.distributed as dist
import torch.nn.functional as F
 
 
@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor, lse: torch.Tensor,
    block_out: torch.Tensor, block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse
 
 
def _merge_out_lse(
    out: Optional[torch.Tensor], lse: Optional[torch.Tensor],
    block_out: torch.Tensor, block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        return block_out.to(torch.float32), block_lse.transpose(-2, -1).unsqueeze(-1)
    return _update_out_and_lse(out, lse, block_out, block_lse)
 
 
class RingComm:
    def __init__(self, group: dist.ProcessGroup):
        self._group = group
        self._ops = []
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
 
 
def _local_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: float, causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(q.size(1), k.size(1), device=q.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    block_lse = torch.logsumexp(scores, dim=-1)
    block_out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).contiguous()
    return block_out, block_lse
 
 
def _ring_attn_forward(
    group: dist.ProcessGroup,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: float, causal: bool,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        out, lse = _merge_out_lse(None, None, *_local_attn(q, k, v, scale, causal))
        return out.to(q.dtype)
 
    comm = RingComm(group)
    out, lse = None, None
 
    for step in range(world_size):
        if step + 1 != world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        if (not causal) or step <= comm.rank:
            block_out, block_lse = _local_attn(q, k, v, scale, causal=(causal and step == 0))
            out, lse = _merge_out_lse(out, lse, block_out, block_lse)
        if step + 1 != world_size:
            comm.wait()
            k, v = next_k, next_v
 
    return out.to(q.dtype)
 
 
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    cp_group = cp_group or dist.group.WORLD
 
    tp_size = dist.get_world_size(tp_group)
    heads_local = num_heads // tp_size
    head_dim = w_qkv.shape[0] // 3 // heads_local
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
 
    B, S = hidden_states.shape[:2]
    qkv = F.linear(hidden_states, w_qkv).view(B, S, 3, heads_local, head_dim)
    q, k, v = qkv.unbind(dim=2)
 
    context = _ring_attn_forward(cp_group, q.contiguous(), k.contiguous(), v.contiguous(),
                                 float(softmax_scale), causal)
 
    out = F.linear(context.reshape(B, S, -1), w_o)
    if tp_size > 1:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out
