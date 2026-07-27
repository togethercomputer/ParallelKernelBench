# EP grouped-GEMM LoRA backward: per-expert LoRA adapter gradients + all-reduce.
#
# In expert-parallel MoE fine-tuning with shared LoRA adapters, each rank holds
# the activations / output-gradients of its local experts. The gradient of a
# shared LoRA-A (resp. LoRA-B) matrix is the sum over local experts of
# dY_e^T @ X_e — a grouped GEMM reduced over the expert dimension — followed by
# an all-reduce over the EP group (every rank contributes its local experts'
# partial sums). The GEMMs and the three all-reduces are independent per
# adapter, so a fused kernel can overlap compute with communication.
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def solution(
    expert_inputs: torch.Tensor,       # (E_local, T, d_in)  inputs to fc1 LoRA-A
    grad_fc1_1_out: torch.Tensor,      # (E_local, T, r)     grad wrt fc1 gate LoRA-A output
    grad_fc1_2_out: torch.Tensor,      # (E_local, T, r)     grad wrt fc1 up   LoRA-A output
    fc2_lora_acts: torch.Tensor,       # (E_local, T, r)     fc2 LoRA-A activations
    grad_expert_outputs: torch.Tensor, # (E_local, T, d_out) grad wrt fc2 LoRA-B output
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD

    # Grouped GEMMs: sum over the local-expert dimension.
    grad_fc1_1_lora_A = torch.bmm(
        grad_fc1_1_out.transpose(1, 2), expert_inputs
    ).sum(dim=0)  # (r, d_in)
    grad_fc1_2_lora_A = torch.bmm(
        grad_fc1_2_out.transpose(1, 2), expert_inputs
    ).sum(dim=0)  # (r, d_in)
    grad_fc2_lora_B = torch.bmm(
        grad_expert_outputs.transpose(1, 2), fc2_lora_acts
    ).sum(dim=0)  # (d_out, r)

    dist.all_reduce(grad_fc1_1_lora_A, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(grad_fc1_2_lora_A, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(grad_fc2_lora_B, op=dist.ReduceOp.SUM, group=group)
    return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B
