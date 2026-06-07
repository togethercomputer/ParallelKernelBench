import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    indices: torch.Tensor,
    local_shard: torch.Tensor,
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_size = local_shard.shape[0]
    embed_dim = local_shard.shape[1]
    
    indices = indices.contiguous().to(torch.cuda.current_device())
    
    target_ranks = indices // shard_size
    
    send_indices_list = [indices[target_ranks == r] for r in range(world_size)]
    send_counts = torch.tensor([len(idx) for idx in send_indices_list], dtype=torch.long, device='cuda')
    
    recv_counts = torch.zeros(world_size, dtype=torch.long, device='cuda')
    dist.all_to_all_single(recv_counts, send_counts)
    
    non_empty_lists = [idx_list for idx_list in send_indices_list if len(idx_list) > 0]
    if non_empty_lists:
        flat_send_indices = torch.cat(non_empty_lists)
    else:
        flat_send_indices = torch.empty(0, dtype=torch.long, device='cuda')
    
    total_recv = recv_counts.sum().item()
    total_send = send_counts.sum().item()
    received_indices = torch.empty(total_recv, dtype=torch.long, device='cuda')
    
    if total_recv > 0 or total_send > 0:
        dist.all_to_all_single(
            received_indices, 
            flat_send_indices,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
    
    if total_recv > 0:
        local_lookup_indices = received_indices - (rank * shard_size)
        local_lookup_indices = torch.clamp(local_lookup_indices, 0, shard_size - 1)
        retrieved_vectors = local_shard[local_lookup_indices]
    else:
        retrieved_vectors = torch.empty((0, embed_dim), dtype=local_shard.dtype, device='cuda')
    
    output_vectors = torch.empty((len(indices), embed_dim), dtype=local_shard.dtype, device='cuda')
    
    input_split_sizes = recv_counts.cpu().tolist()
    output_split_sizes = send_counts.cpu().tolist()
    
    if len(indices) > 0 or retrieved_vectors.numel() > 0:
        dist.all_to_all_single(
            output_vectors,
            retrieved_vectors,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes
        )
    
    return output_vectors
