import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Any

def solution(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden_states, weight)
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    
    per_token_ce = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='none')
    
    new_logprobs_flat = -per_token_ce.detach()
    old_logprobs_flat = old_logprobs.view(-1)
    advantages_flat = advantages.view(-1)
    
    valid_mask = (labels_flat != ignore_index)
    n_valid_local = valid_mask.sum().float()
    
    n_valid_global = n_valid_local.clone()
    dist.all_reduce(n_valid_global, op=dist.ReduceOp.SUM)
    n_valid_global_clamped = n_valid_global.clamp(min=1.0)
    
    delta = (new_logprobs_flat - old_logprobs_flat).masked_fill(~valid_mask, 0.0).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(delta)
    
    per_token_pg = -(ratio * advantages_flat).masked_fill(~valid_mask, 0.0)
    local_pg_sum = per_token_pg.sum()
    
    global_pg_sum = local_pg_sum.clone()
    dist.all_reduce(global_pg_sum, op=dist.ReduceOp.SUM)
    true_pg = global_pg_sum / n_valid_global_clamped
    
    w = (ratio.detach() * advantages_flat).masked_fill(~valid_mask, 0.0)
    local_surrogate_sum = (w * per_token_ce).sum()
    
    surrogate = local_surrogate_sum / n_valid_global_clamped
    
    loss = true_pg.detach() + surrogate - surrogate.detach()
    
    ratio_valid = ratio.masked_fill(~valid_mask, 0.0)
    sum_ratio_local = ratio_valid.sum()
    dist.all_reduce(sum_ratio_local, op=dist.ReduceOp.SUM)
    ratio_mean = sum_ratio_local / n_valid_global_clamped
    
    ratio_for_min = ratio.masked_fill(~valid_mask, float('inf'))
    min_ratio_local = ratio_for_min.min() if n_valid_local > 0 else torch.tensor(float('inf'), device=ratio.device)
    dist.all_reduce(min_ratio_local, op=dist.ReduceOp.MIN)
    
    ratio_for_max = ratio.masked_fill(~valid_mask, float('-inf'))
    max_ratio_local = ratio_for_max.max() if n_valid_local > 0 else torch.tensor(float('-inf'), device=ratio.device)
    dist.all_reduce(max_ratio_local, op=dist.ReduceOp.MAX)
    
    k3_local = (ratio - delta - 1.0).masked_fill(~valid_mask, 0.0).sum()
    dist.all_reduce(k3_local, op=dist.ReduceOp.SUM)
    k3_mean = k3_local / n_valid_global_clamped
    
    entropy_local = per_token_ce.detach().masked_fill(~valid_mask, 0.0).sum()
    dist.all_reduce(entropy_local, op=dist.ReduceOp.SUM)
    entropy_mean = entropy_local / n_valid_global_clamped
    
    metrics = torch.stack([ratio_mean, min_ratio_local, max_ratio_local, k3_mean, entropy_mean])
    
    per_token_logprobs = new_logprobs_flat.view_as(labels)
    per_token_loss = per_token_pg.view_as(labels)
    
    return loss, None, per_token_logprobs, per_token_loss, metrics
