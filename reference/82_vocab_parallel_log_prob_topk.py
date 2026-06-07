from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: Optional[int],
    top_p: float,
) -> torch.Tensor:
    need_k = top_k is not None and top_k > 0  
    need_p = top_p is not None and top_p < 1.0  
  
    if not need_k and not need_p:  
        return logits  
  
    original_shape = logits.shape  
    vocab_size = logits.shape[-1]  
    logits_2d = logits.reshape(-1, vocab_size)
    if need_k:
        top_k = min(int(top_k), vocab_size)
  
    if need_k and not need_p:  
        top_k_values, _ = torch.topk(logits_2d, top_k, dim=-1)  
        threshold = top_k_values[..., -1:].expand_as(logits_2d)  
        keep_mask = logits_2d >= threshold  
        filtered = torch.where(  
            keep_mask,  
            logits_2d,  
            torch.full_like(logits_2d, float("-inf")),  
        )  
        return filtered.reshape(original_shape)
  
    logits_sort, logits_idx = logits_2d.sort(dim=-1, descending=False)  
  
    top_k_mask = None  
    if need_k:  
        top_k_index = logits_sort.size(-1) - top_k  
        threshold = logits_sort.gather(  
            -1,  
            torch.full(  
                logits_sort.shape[:-1],  
                top_k_index,  
                device=logits_2d.device,  
                dtype=torch.long,  
            ).unsqueeze(-1),  
        )  
        top_k_mask = logits_sort >= threshold  
        logits_sort = logits_sort.masked_fill(~top_k_mask, float("-inf"))  
  
    probs_sort = logits_sort.softmax(dim=-1)  
    probs_sum = torch.cumsum(probs_sort, dim=-1)  
    top_p_mask = probs_sum > 1 - top_p  
    top_p_mask[..., -1] = True  # always keep at least one token  
    logits_sort = logits_sort.masked_fill(~top_p_mask, float("-inf"))  
  
    filtered = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)  
    return filtered.reshape(original_shape)


def _all_to_all_vp_to_seq(
    vocab_parallel_logits: torch.Tensor,
    tp_group: dist.ProcessGroup,  
) -> torch.Tensor:
    world_size = dist.get_world_size(tp_group)  
    num_tokens, local_vocab = vocab_parallel_logits.shape
    local_tokens = num_tokens // world_size
  
    input_flat = vocab_parallel_logits.contiguous().flatten()
    output_flat = torch.empty_like(input_flat)  
    dist.all_to_all_single(output_flat, input_flat, group=tp_group)  
  
    output = output_flat.view(world_size, local_tokens, local_vocab)
    return output.permute(1, 0, 2).reshape(local_tokens, world_size * local_vocab)

  
@torch.no_grad()  
def solution(  
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    top_k: Optional[int] = None,  
    top_p: float = 1.0,  
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    world_size = dist.get_world_size(tp_group)  
    rank = dist.get_rank(tp_group)  
    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    num_tokens = batch * seq_len
  
    if num_tokens % world_size != 0:
        raise ValueError(  
            f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}"
        )  
    local_tokens = num_tokens // world_size
  
    logits_2d = vocab_parallel_logits.reshape(num_tokens, local_vocab)
    target_flat = target.reshape(-1)
    target_local = target_flat[rank * local_tokens : (rank + 1) * local_tokens]

    seq_parallel_logits = _all_to_all_vp_to_seq(logits_2d, tp_group)
    logits = _apply_top_k_top_p(seq_parallel_logits, top_k=top_k, top_p=top_p)
    log_probs = F.log_softmax(logits.to(dtype=torch.float32), dim=-1)  

    token_logprobs = torch.gather(log_probs, -1, target_local.unsqueeze(-1))
    token_logprobs = token_logprobs.squeeze(-1)

    gathered = [torch.empty_like(token_logprobs) for _ in range(world_size)]
    dist.all_gather(gathered, token_logprobs, group=tp_group)  
    return torch.cat(gathered, dim=0).reshape(batch, seq_len)