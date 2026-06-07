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
        filtered = logits_2d.masked_fill(logits_2d < threshold, float("-inf"))
        return filtered.reshape(original_shape)

    sorted_logits, sorted_idx = logits_2d.sort(dim=-1, descending=False)
    if need_k:
        top_k_index = sorted_logits.shape[-1] - top_k
        threshold = sorted_logits[..., top_k_index : top_k_index + 1]
        sorted_logits = sorted_logits.masked_fill(
            sorted_logits < threshold, float("-inf")
        )

    sorted_probs = sorted_logits.softmax(dim=-1)
    top_p_mask = torch.cumsum(sorted_probs, dim=-1) > 1 - top_p
    top_p_mask[..., -1] = True
    sorted_logits = sorted_logits.masked_fill(~top_p_mask, float("-inf"))
    filtered = sorted_logits.scatter(dim=-1, index=sorted_idx, src=sorted_logits)
    return filtered.reshape(original_shape)


def _all_to_all_vp_to_seq(
    logits: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    num_tokens, local_vocab = logits.shape
    local_tokens = num_tokens // world_size

    send = logits.contiguous().flatten()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    recv = recv.view(world_size, local_tokens, local_vocab)
    return recv.permute(1, 0, 2).reshape(local_tokens, world_size * local_vocab)


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    chunk_size: int = 1,
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    world_size = dist.get_world_size(group=tp_group)
    rank = dist.get_rank(group=tp_group)
    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    num_tokens = batch * seq_len
    chunk_tokens = batch * max(1, int(chunk_size))

    if num_tokens % world_size != 0:
        raise ValueError(
            f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}"
        )
    if chunk_tokens % world_size != 0:
        raise ValueError(
            f"B*chunk_size={chunk_tokens} must be divisible by tp size {world_size}"
        )

    logits_2d = vocab_parallel_logits.reshape(num_tokens, local_vocab)
    target_flat = target.reshape(-1)
    pieces = []

    for start in range(0, num_tokens, chunk_tokens):
        end = min(start + chunk_tokens, num_tokens)
        current = end - start
        local_tokens = current // world_size
        logits_chunk = logits_2d[start:end]
        target_chunk = target_flat[start:end]
        target_local = target_chunk[rank * local_tokens : (rank + 1) * local_tokens]

        seq_logits = _all_to_all_vp_to_seq(logits_chunk, tp_group)
        filtered = _apply_top_k_top_p(seq_logits, top_k=top_k, top_p=top_p)
        log_probs = F.log_softmax(filtered.float(), dim=-1)
        local_logprobs = torch.gather(
            log_probs, -1, target_local.unsqueeze(-1)
        ).squeeze(-1)

        gathered = [torch.empty_like(local_logprobs) for _ in range(world_size)]
        dist.all_gather(gathered, local_logprobs, group=tp_group)
        pieces.append(torch.cat(gathered, dim=0))

    return torch.cat(pieces, dim=0).reshape(batch, seq_len)
