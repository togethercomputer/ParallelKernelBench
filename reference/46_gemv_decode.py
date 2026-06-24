import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    weight_shard: torch.Tensor,
    bias_shard: torch.Tensor,
) -> torch.Tensor:
    world_size = dist.get_world_size()

    local_logits = torch.matmul(hidden_states, weight_shard.t())
    local_logits = local_logits + bias_shard

    gathered = [torch.empty_like(local_logits) for _ in range(world_size)]
    dist.all_gather(gathered, local_logits.contiguous())
    logits = torch.cat(gathered, dim=1)

    return logits
