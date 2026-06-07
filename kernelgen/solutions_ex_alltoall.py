import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: [world_size, *chunk_shape] - tensor[i] is the data to send to rank i
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank
                           # All ranks must provide tensors of identical shape and dtype.
) -> torch.Tensor:  # Returns: A tensor of shape [world_size, *chunk_shape] where output[i] contains
                    #          the data that rank i sent to this rank.
    """
    NCCL All-to-All: Each rank sends different data to every other rank.
    
    This is the most general collective operation. Each rank has world_size chunks,
    and chunk i is sent to rank i. After the operation, each rank has received
    one chunk from every other rank.
    
    This is also known as "transpose" or "personalized all-to-all" and is used
    in FFT, sorting, and transformer parallelism.
    
    Complexity: O(N) - each rank sends/receives N-1 messages
    
    Example with 4 ranks, input shape [4, 2]:
        Before: Rank 0: [[0→0, 0→0], [0→1, 0→1], [0→2, 0→2], [0→3, 0→3]]
                        (data to send to rank 0, 1, 2, 3 respectively)
                Rank 1: [[1→0, 1→0], [1→1, 1→1], [1→2, 1→2], [1→3, 1→3]]
                Rank 2: [[2→0, 2→0], [2→1, 2→1], [2→2, 2→2], [2→3, 2→3]]
                Rank 3: [[3→0, 3→0], [3→1, 3→1], [3→2, 3→2], [3→3, 3→3]]
        
        After:  Rank 0: [[0→0, 0→0], [1→0, 1→0], [2→0, 2→0], [3→0, 3→0]]
                        (data received from rank 0, 1, 2, 3 respectively)
                Rank 1: [[0→1, 0→1], [1→1, 1→1], [2→1, 2→1], [3→1, 3→1]]
                Rank 2: [[0→2, 0→2], [1→2, 1→2], [2→2, 2→2], [3→2, 3→2]]
                Rank 3: [[0→3, 0→3], [1→3, 1→3], [2→3, 2→3], [3→3, 3→3]]
    
    Think of it as transposing a distributed matrix where rows = source rank, cols = dest rank.
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - All ranks must call this function with tensors of the same shape and dtype
        - First dimension must equal world_size
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    world_size = dist.get_world_size()
    
    assert tensor.shape[0] == world_size, \
        f"First dimension ({tensor.shape[0]}) must equal world_size ({world_size})"
    
    out = torch.empty_like(tensor)
    dist.all_to_all_single(out, tensor)
    
    return out

