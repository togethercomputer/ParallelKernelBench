import torch
import torch.distributed as dist


@torch.no_grad()
def solution(A_local: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    M, K_local = A_local.shape
    K = world_size * K_local

    A_local_t = A_local.transpose(0, 1).contiguous()
    A_t_buf = A_local_t.new_empty((world_size, K_local, M))
    dist.all_gather_into_tensor(A_t_buf, A_local_t)
    A_global_t = A_t_buf.reshape(K, M)

    C_t = torch.matmul(B.transpose(0, 1), A_global_t)
    return C_t.transpose(0, 1)
