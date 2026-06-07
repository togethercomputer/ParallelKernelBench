from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _a2a_rows(
    tensor: torch.Tensor,
    split_sizes: List[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    out = torch.empty_like(tensor)
    dist.all_to_all_single(
        out,
        tensor.contiguous(),
        output_split_sizes=split_sizes,
        input_split_sizes=split_sizes,
        group=group,
    )
    return out


def _a2a_async(
    tensor: torch.Tensor,
    split_sizes: List[int],
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, dist.Work]:
    out = torch.empty_like(tensor)
    handle = dist.all_to_all_single(
        out,
        tensor.contiguous(),
        output_split_sizes=split_sizes,
        input_split_sizes=split_sizes,
        group=group,
        async_op=True,
    )
    return out, handle


def _redistribute_kv(
    key_value: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    tokens, heads, width = key_value.shape
    if heads < world_size and world_size % heads == 0:
        key_value = key_value.repeat_interleave(world_size // heads, dim=1)
        heads = key_value.shape[1]
    if heads % world_size != 0:
        raise ValueError("KV heads must divide evenly across context ranks")

    local_heads = heads // world_size
    packed = key_value.reshape(tokens, world_size, local_heads, width)
    packed = packed.permute(1, 0, 2, 3).reshape(world_size * tokens, local_heads, width)
    return _a2a_rows(packed.contiguous(), [tokens] * world_size, group)


def _kv_by_range(
    kv: torch.Tensor,
    world_size: int,
    ranges: int,
    spb: int,
    clip_token_nums: int,
) -> torch.Tensor:
    _, heads, width = kv.shape
    kv = kv.reshape(world_size, ranges, spb, heads, width)
    kv = kv.permute(1, 0, 2, 3, 4).contiguous()
    kv = kv.reshape(ranges, world_size * spb, heads, width)
    return kv[:, :clip_token_nums].reshape(ranges * clip_token_nums, heads, width)


def _split_query(query: torch.Tensor, world_size: int, ranges: int) -> List[torch.Tensor]:
    tokens, heads, head_dim = query.shape
    if tokens % ranges != 0:
        raise ValueError("query token count must divide cp_shuffle_num")
    if heads % world_size != 0:
        raise ValueError("query heads must divide evenly across context ranks")

    spb = tokens // ranges
    local_heads = heads // world_size
    query = query.reshape(ranges, spb, world_size, local_heads, head_dim)
    query = query.permute(0, 2, 1, 3, 4).contiguous()
    query = query.reshape(ranges, world_size * spb, local_heads, head_dim)
    return [query[idx] for idx in range(ranges)]


def _restore_output(
    chunks: List[torch.Tensor],
    world_size: int,
    spb: int,
) -> torch.Tensor:
    out = torch.stack(chunks, dim=0)
    ranges, _, heads, head_dim = out.shape
    out = out.reshape(ranges, world_size, spb, heads, head_dim)
    out = out.permute(0, 2, 1, 3, 4).contiguous()
    return out.reshape(ranges * spb, world_size * heads, head_dim)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.unsqueeze(0).transpose(1, 2)
    k = k.unsqueeze(0).transpose(1, 2)
    v = v.unsqueeze(0).transpose(1, 2)
    if k.shape[1] < q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1).contiguous()


@torch.no_grad()
def solution(
    query: torch.Tensor,
    key_value: torch.Tensor,
    k_ranges: torch.Tensor,
    cp_shuffle_num: int,
    clip_token_nums: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    ranges = cp_shuffle_num
    tokens, _, head_dim = query.shape
    if tokens % ranges != 0:
        raise ValueError("query token count must divide cp_shuffle_num")
    spb = tokens // ranges
    clip_token_nums = int(clip_token_nums or world_size * spb)

    kv = _redistribute_kv(key_value, world_size, group)
    kv = _kv_by_range(kv, world_size, ranges, spb, clip_token_nums)
    key = kv[..., :head_dim]
    value = kv[..., head_dim:]

    q_chunks = _split_query(query, world_size, ranges)
    split_sizes = [spb] * world_size
    if ranges == 1:
        q_local, handle_q = _a2a_async(q_chunks[0], split_sizes, group)
        handle_q.wait()
        start = int(k_ranges[0, 0])
        end = int(k_ranges[0, 1])
        out = _sdpa(q_local, key[start:end], value[start:end])
        out, handle_o = _a2a_async(out, split_sizes, group)
        handle_o.wait()
        return _restore_output([out], world_size, spb)

    outputs: List[torch.Tensor] = []
    q_chunks[0], handle_q = _a2a_async(q_chunks[0], split_sizes, group)
    loop_var: Optional[torch.Tensor] = None
    loop_handle: Optional[dist.Work] = None
    prev_out: Optional[torch.Tensor] = None
    for idx in range(ranges):
        if idx == 0:
            handle_q.wait()
            q_local = q_chunks[0]
            loop_var, loop_handle = _a2a_async(q_chunks[1], split_sizes, group)
        else:
            assert loop_var is not None and loop_handle is not None
            loop_handle.wait()
            if loop_var.numel() == q_chunks[0].numel():
                q_local = loop_var
            else:
                q_local, ready_out = torch.chunk(loop_var, 2, dim=-1)
                outputs.append(ready_out)

            assert prev_out is not None
            send = (
                torch.cat([q_chunks[idx + 1], prev_out], dim=-1)
                if idx < ranges - 1
                else prev_out
            )
            loop_var, loop_handle = _a2a_async(send, split_sizes, group)

        start = int(k_ranges[idx, 0])
        end = int(k_ranges[idx, 1])
        prev_out = _sdpa(q_local, key[start:end], value[start:end])

        if idx == ranges - 1:
            assert loop_var is not None and loop_handle is not None
            loop_handle.wait()
            outputs.append(loop_var)
            last_out, handle_out = _a2a_async(prev_out, split_sizes, group)
            handle_out.wait()
            outputs.append(last_out)

    return _restore_output(outputs, world_size, spb)
