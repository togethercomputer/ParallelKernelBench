import torch
import torch.distributed as dist
from typing import Tuple, Optional


def forward(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if local_valid_tokens.item() == 0:
        loss = torch.nan_to_num(loss)

    loss_sum = loss * local_valid_tokens
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

    normalized_loss = loss_sum / global_valid_tokens
    return normalized_loss, loss_sum


def backward(
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    grad_normalized_loss: torch.Tensor,
    grad_loss_sum: Optional[torch.Tensor],
) -> torch.Tensor:
    grad_from_normalized = grad_normalized_loss * local_valid_tokens / global_valid_tokens

    if grad_loss_sum is not None:
        grad_from_sum = grad_loss_sum * local_valid_tokens
    else:
        grad_from_sum = torch.zeros_like(grad_normalized_loss, device=grad_normalized_loss.device)

    return grad_from_normalized + grad_from_sum


def solution(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    grad_normalized_loss: torch.Tensor,
    grad_loss_sum: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_loss, loss_sum = forward(loss, local_valid_tokens, global_valid_tokens)

    grad_loss = backward(
        local_valid_tokens,
        global_valid_tokens,
        grad_normalized_loss,
        grad_loss_sum,
    )

    return normalized_loss, loss_sum, grad_loss
