import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: Arbitrary (any number of dimensions, e.g., [M, N], [B, C, H, W], etc.)
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank (torch.cuda.current_device())
                           # All ranks must provide tensors of identical shape and dtype.
                           # The tensor values may differ across ranks.
) -> torch.Tensor:  # Returns: A new tensor of the same shape and dtype as input, containing the elementwise
                    #          sum across all ranks. Each element in the output is the sum of the corresponding
                    #          elements from all ranks' input tensors.
    """
    Simple NCCL all-reduce (sum) over all ranks.
    
    Performs an all-reduce collective operation that sums the input tensor across all ranks
    in the distributed process group. All ranks must participate and provide tensors of
    identical shape and dtype.
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensor must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must call this function with tensors of the same shape and dtype
    
    Returns:
        A new tensor (same shape/dtype as input) where each element is the sum of
        the corresponding elements from all ranks' input tensors.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out