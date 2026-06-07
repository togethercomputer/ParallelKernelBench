import torch
import torch.distributed as dist

def solution(local_hidden_states: torch.Tensor, local_weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = local_hidden_states.dtype
    # Upcast to float32 for stable variance calculation
    local_hidden_states = local_hidden_states.to(torch.float32)
    
    local_sum_squares = local_hidden_states.pow(2).sum(dim=-1, keepdim=True)
    
    dist.all_reduce(local_sum_squares, op=dist.ReduceOp.SUM)
    
    world_size = dist.get_world_size()
    global_hidden_size = local_hidden_states.shape[-1] * world_size
    variance = local_sum_squares / global_hidden_size
    
    local_hidden_states = local_hidden_states * torch.rsqrt(variance + variance_epsilon)
    
    return local_weight * local_hidden_states.to(input_dtype)
