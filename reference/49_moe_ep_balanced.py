# Expert-parallel (EP) fused MoE forward — BALANCED EP with STATIC dispatch.
#
# num_experts == world_size (one expert per rank). Unlike the dynamic-split
# MoE problems (31 / 50 / 51), balanced EP uses a fixed per-expert capacity:
# every rank sends exactly `capacity` token slots to every peer, so the
# all_to_all uses equal static splits and needs NO host-side size exchange
# (no .tolist() / no metadata round-trip). Tokens beyond capacity are dropped,
# unused slots are zero-padded — the standard capacity-factor trade-off. The
# whole pipeline (route -> scatter into slots -> a2a -> expert MLP -> a2a ->
# weighted combine) is device-side and communication is perfectly regular,
# which makes it an ideal target for fused compute/comm kernels.
from typing import Optional

import torch
import torch.distributed as dist


def expert_forward(
    x: torch.Tensor,
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
) -> torch.Tensor:
    gate = torch.nn.functional.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    assert num_experts == world_size, "balanced EP hosts exactly one expert per rank"

    hidden_dim = hidden_states.size(-1)
    tokens = hidden_states.reshape(-1, hidden_dim)
    num_tokens = tokens.size(0)
    device = tokens.device

    router_logits = torch.nn.functional.linear(tokens, gate_weight, gate_bias)
    probs = torch.softmax(router_logits.float(), dim=-1)
    routing_weights, selected_experts = torch.topk(probs, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    # Fixed per-expert capacity (capacity factor 1.0 over routed slots).
    capacity = (num_tokens * top_k + num_experts - 1) // num_experts

    # Position of each routed slot inside its expert queue (token-major order).
    flat_experts = selected_experts.reshape(-1)  # (num_tokens * top_k,)
    one_hot = torch.nn.functional.one_hot(flat_experts, num_experts)
    pos_in_expert = (one_hot.cumsum(dim=0) - 1).mul(one_hot).sum(dim=-1)
    keep = pos_in_expert < capacity

    # Scatter kept tokens into their (expert, slot) positions.
    send_buf = tokens.new_zeros(num_experts * capacity, hidden_dim)
    slot = flat_experts * capacity + pos_in_expert
    expanded_tokens = tokens.repeat_interleave(top_k, dim=0)
    send_buf[slot[keep]] = expanded_tokens[keep]

    # Static equal-split all_to_all: chunk e goes to rank e.
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=group)

    # This rank's expert processes all received slots (world_size * capacity).
    expert_out = expert_forward(recv_buf, gate_proj, up_proj, down_proj)

    # Return processed slots to their source ranks (same static splits).
    combine_buf = torch.empty_like(expert_out)
    dist.all_to_all_single(combine_buf, expert_out, group=group)

    # Weighted combine; dropped slots contribute zero.
    slot_out = tokens.new_zeros(num_tokens * top_k, hidden_dim)
    slot_out[keep] = combine_buf[slot[keep]]
    combined = (
        slot_out.view(num_tokens, top_k, hidden_dim)
        * routing_weights.unsqueeze(-1).to(slot_out.dtype)
    ).sum(dim=1)

    return combined.view_as(hidden_states)
