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
 
 
def _pp_recv_forward(
    pp_group: dist.ProcessGroup,
    tensor_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    prev_rank = dist.get_global_rank(
        pp_group, (dist.get_rank(pp_group) - 1) % dist.get_world_size(pp_group)
    )
    buf = torch.empty(tensor_shape, dtype=dtype, device=device)
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, buf, prev_rank, group=pp_group)])
    for r in reqs:
        r.wait()
    return buf


def _pp_send_forward(
    pp_group: dist.ProcessGroup,
    tensor: torch.Tensor,
) -> None:
    next_rank = dist.get_global_rank(
        pp_group, (dist.get_rank(pp_group) + 1) % dist.get_world_size(pp_group)
    )
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor.contiguous(), next_rank, group=pp_group)])
    for r in reqs:
        r.wait()
 
 
def _attention_block(
    hidden: torch.Tensor, w_qkv: torch.Tensor, w_o: torch.Tensor,
    num_heads: int, scale: float, causal: bool,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    B, S, D = hidden.shape
    head_dim = w_qkv.shape[0] // 3 // num_heads
    qkv = F.linear(hidden, w_qkv).view(B, S, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    ctx = _ring_attn_forward(cp_group, q.contiguous(), k.contiguous(), v.contiguous(),
                             scale, causal)
    return F.linear(ctx.reshape(B, S, -1), w_o)
 
 
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    cp_group = cp_group or dist.group.WORLD
    head_dim = w_qkv.shape[0] // 3 // num_heads
    scale = float(softmax_scale if softmax_scale is not None else head_dim ** -0.5)
 
    is_first = True
    is_last = True
    if pp_group is not None and dist.get_world_size(pp_group) > 1:
        pp_rank = dist.get_rank(pp_group)
        pp_size = dist.get_world_size(pp_group)
        is_first = (pp_rank == 0)
        is_last = (pp_rank == pp_size - 1)
 
    if is_first:
        stage_input = hidden_states
    else:
        stage_input = _pp_recv_forward(
            pp_group, tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device,
        )
 
    stage_output = _attention_block(stage_input, w_qkv, w_o, num_heads, scale, causal, cp_group)
 
    if not is_last and pp_group is not None:
        _pp_send_forward(pp_group, stage_output)
 
    return stage_output
