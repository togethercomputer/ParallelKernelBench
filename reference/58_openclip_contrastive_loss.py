from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _siglip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
    logit_bias: float,
    negative_only: bool,
) -> torch.Tensor:
    batch = image_features.size(0)
    logits = logit_scale * image_features @ text_features.T + logit_bias

    if negative_only:
        return -F.logsigmoid(-logits).sum() / batch

    labels = -torch.ones((batch, batch), device=logits.device, dtype=logits.dtype)
    labels.diagonal().fill_(1)
    return -F.logsigmoid(labels * logits).sum() / batch


def _exchange(
    group: dist.ProcessGroup,
    recv_from: int,
    send_to: int,
    tensor: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(tensor)
    ops = [
        dist.P2POp(dist.isend, tensor, dist.get_global_rank(group, send_to), group=group),
        dist.P2POp(dist.irecv, out, dist.get_global_rank(group, recv_from), group=group),
    ]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return out


def _exchange_bidir(
    group: dist.ProcessGroup,
    left: int,
    right: int,
    tensor_to_left: torch.Tensor,
    tensor_to_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from_left = torch.empty_like(tensor_to_right)
    from_right = torch.empty_like(tensor_to_left)
    left_global = dist.get_global_rank(group, left)
    right_global = dist.get_global_rank(group, right)
    ops = [
        dist.P2POp(dist.isend, tensor_to_right, right_global, group=group),
        dist.P2POp(dist.isend, tensor_to_left, left_global, group=group),
        dist.P2POp(dist.irecv, from_right, right_global, group=group),
        dist.P2POp(dist.irecv, from_left, left_global, group=group),
    ]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return from_right, from_left


@torch.no_grad()
def solution(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
    logit_bias: float = 0.0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    loss = _siglip_loss(image_features, text_features, logit_scale, logit_bias, False)

    left = (rank - 1) % world_size
    right = (rank + 1) % world_size
    text_to_left = text_features
    text_to_right = text_features
    num_bidir, remainder = divmod(world_size - 1, 2)

    for _ in range(num_bidir):
        from_right, from_left = _exchange_bidir(group, left, right, text_to_left, text_to_right)
        loss = loss + _siglip_loss(image_features, from_right, logit_scale, logit_bias, True)
        loss = loss + _siglip_loss(image_features, from_left, logit_scale, logit_bias, True)
        text_to_left, text_to_right = from_right, from_left

    if remainder:
        text_recv = _exchange(group, left, right, text_to_right)
        loss = loss + _siglip_loss(image_features, text_recv, logit_scale, logit_bias, True)

    return loss