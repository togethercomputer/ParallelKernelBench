from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

_NO_OBJ_LOGIT = -10.0


def _mask_iou(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_flat = lhs.flatten(1).float()
    rhs_flat = rhs.flatten(1).float()
    intersection = lhs_flat @ rhs_flat.T
    lhs_area = lhs_flat.sum(dim=1)
    rhs_area = rhs_flat.sum(dim=1)
    union = lhs_area[:, None] + rhs_area[None, :] - intersection
    return intersection / union.clamp_min(1.0)


def _all_gather_variable(
    tensor: torch.Tensor,
    counts: List[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    recv = [tensor.new_empty((count, *tensor.shape[1:])) for count in counts]
    dist.all_gather(recv, tensor.contiguous(), group=group)
    return torch.cat(recv, dim=0)


def _suppression_mask(
    masks: torch.Tensor,
    last_occluded: torch.Tensor,
    iou_threshold: float,
    reverse: bool,
) -> torch.Tensor:
    num_objects = masks.shape[0]
    suppress = torch.zeros(num_objects, dtype=torch.bool, device=masks.device)
    if num_objects <= 1:
        return suppress

    overlaps = torch.triu(_mask_iou(masks, masks) >= iou_threshold, diagonal=1)
    last_i = last_occluded.view(num_objects, 1)
    last_j = last_occluded.view(1, num_objects)
    cmp = torch.lt if reverse else torch.gt

    suppress_i = overlaps & cmp(last_i, last_j) & (last_j > -1)
    suppress_j = overlaps & cmp(last_j, last_i) & (last_i > -1)
    return suppress_i.any(dim=1) | suppress_j.any(dim=0)


@torch.no_grad()
def solution(
    low_res_masks_local: torch.Tensor,
    obj_scores_local: torch.Tensor,
    num_obj_per_gpu: List[int],
    last_occluded: torch.Tensor,
    iou_threshold: float = 0.7,
    reverse: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    expected = int(num_obj_per_gpu[rank])
    if low_res_masks_local.shape[0] != expected:
        raise ValueError("local mask count does not match num_obj_per_gpu")
    if obj_scores_local.shape[0] != expected:
        raise ValueError("local score count does not match num_obj_per_gpu")

    masks_local = low_res_masks_local.float().contiguous()
    scores_local = obj_scores_local.float().contiguous()
    masks_global = _all_gather_variable(masks_local, num_obj_per_gpu, group)
    scores_global = _all_gather_variable(scores_local, num_obj_per_gpu, group)

    last_occluded = last_occluded.to(device=masks_global.device, dtype=torch.long)
    binary_masks = masks_global > 0
    to_suppress = _suppression_mask(
        binary_masks, last_occluded, iou_threshold=iou_threshold, reverse=reverse
    )
    masks_global[to_suppress] = _NO_OBJ_LOGIT
    return masks_global, scores_global, to_suppress
