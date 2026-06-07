from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: Optional[int],
    top_p: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    need_k = top_k is not None and top_k > 0
    need_p = top_p is not None and top_p < 1.0
    if not need_k and not need_p:
        return logits, None

    original_shape = logits.shape
    vocab_size = logits.shape[-1]
    logits_2d = logits.reshape(-1, vocab_size)
    if need_k:
        top_k = min(int(top_k), vocab_size)

    if need_k and not need_p:
        top_k_values, _ = torch.topk(logits_2d, top_k, dim=-1)
        threshold = top_k_values[..., -1:].expand_as(logits_2d)
        keep_mask = logits_2d >= threshold
        filtered = logits_2d.masked_fill(~keep_mask, float("-inf"))
        return filtered.reshape(original_shape), keep_mask.reshape(original_shape)

    sorted_logits, sorted_idx = logits_2d.sort(dim=-1, descending=False)
    top_k_mask = None
    if need_k:
        top_k_index = sorted_logits.shape[-1] - top_k
        threshold = sorted_logits[..., top_k_index : top_k_index + 1]
        top_k_mask = sorted_logits >= threshold
        sorted_logits = sorted_logits.masked_fill(~top_k_mask, float("-inf"))

    sorted_probs = sorted_logits.softmax(dim=-1)
    top_p_mask = torch.cumsum(sorted_probs, dim=-1) > 1 - top_p
    top_p_mask[..., -1] = True
    sorted_logits = sorted_logits.masked_fill(~top_p_mask, float("-inf"))

    keep_sorted = top_p_mask if top_k_mask is None else top_p_mask & top_k_mask
    filtered = sorted_logits.scatter(dim=-1, index=sorted_idx, src=sorted_logits)
    keep_mask = keep_sorted.scatter(dim=-1, index=sorted_idx, src=keep_sorted)
    return filtered.reshape(original_shape), keep_mask.reshape(original_shape)


def _all_to_all_vp_to_seq(
    logits: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    num_tokens, local_vocab = logits.shape
    local_tokens = num_tokens // world_size

    recv = torch.empty_like(logits.contiguous().flatten())
    dist.all_to_all_single(recv, logits.contiguous().flatten(), group=group)
    recv = recv.view(world_size, local_tokens, local_vocab)
    return recv.permute(1, 0, 2).reshape(local_tokens, world_size * local_vocab)


def _all_to_all_seq_to_vp(
    logits: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    local_tokens, vocab_size = logits.shape
    local_vocab = vocab_size // world_size

    send = logits.reshape(local_tokens, world_size, local_vocab)
    send = send.permute(1, 0, 2).contiguous().flatten()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.reshape(world_size * local_tokens, local_vocab)


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
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
    grad_flat = grad_output.reshape(-1)
    grad_chunks = []

    for start in range(0, num_tokens, chunk_tokens):
        end = min(start + chunk_tokens, num_tokens)
        current = end - start
        local_tokens = current // world_size
        target_chunk = target_flat[start:end]
        grad_chunk = grad_flat[start:end]
        target_local = target_chunk[rank * local_tokens : (rank + 1) * local_tokens]
        grad_local = grad_chunk[rank * local_tokens : (rank + 1) * local_tokens]

        seq_logits = _all_to_all_vp_to_seq(logits_2d[start:end], tp_group)
        filtered, keep_mask = _apply_top_k_top_p(seq_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered.float(), dim=-1)
        grad_seq = -probs
        row_ids = torch.arange(local_tokens, device=target_local.device)
        grad_seq[row_ids, target_local] += 1.0
        grad_seq.mul_(grad_local.unsqueeze(-1))
        if keep_mask is not None:
            grad_seq.mul_(keep_mask)

        grad_chunks.append(_all_to_all_seq_to_vp(grad_seq, tp_group))

    return torch.cat(grad_chunks, dim=0).reshape(batch, seq_len, local_vocab)
