import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple, Optional
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ void send_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__global__ void loss_allreduce_kernel(
    const float* __restrict__ local_scaled,        // [1] local loss * local_valid (sanitized)
    float* __restrict__ symm_buf,                  // [world] symm slot, this rank writes index rank
    const uint64_t* __restrict__ peer_buf_ptrs,    // world entries
    const uint64_t* __restrict__ signal_ptrs,      // world entries (signal pads)
    float* __restrict__ out_norm,                  // bf16/float scalar normalized
    float* __restrict__ out_sum,                   // float scalar loss_sum
    float* __restrict__ out_grad,                  // float scalar grad_loss
    float local_valid,
    float global_valid,
    float grad_norm_up,
    float grad_sum_up,
    int   has_grad_sum,
    int   rank,
    int   world_size
) {
    int tid = threadIdx.x;

    // Each rank publishes local_scaled into its OWN symm buffer slot 0,
    // peers will read it via peer_buf_ptrs[peer][0].
    if (tid == 0) {
        symm_buf[0] = *local_scaled;
        __threadfence_system();
    }
    __syncthreads();

    // Signal all peers, wait all peers (rank 0 of pad)
    if (tid < world_size) {
        uint32_t* send_addr = reinterpret_cast<uint32_t*>(signal_ptrs[tid]) + rank;
        send_signal(send_addr);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(signal_ptrs[rank]) + tid;
        wait_signal(wait_addr);
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.f;
        for (int r = 0; r < world_size; ++r) {
            const float* p = reinterpret_cast<const float*>(peer_buf_ptrs[r]);
            sum += p[0];
        }
        *out_sum  = sum;
        *out_norm = sum / global_valid;

        float g = grad_norm_up * local_valid / global_valid;
        if (has_grad_sum) g += grad_sum_up * local_valid;
        *out_grad = g;
    }
}

void launch_loss_allreduce(
    torch::Tensor local_scaled,   // float32 [1]
    torch::Tensor symm_buf,       // float32 [world]
    torch::Tensor peer_ptrs,      // int64 [world]
    torch::Tensor signal_ptrs,    // int64 [world]
    torch::Tensor out_norm,       // float32 [1]
    torch::Tensor out_sum,        // float32 [1]
    torch::Tensor out_grad,       // float32 [1]
    double local_valid,
    double global_valid,
    double grad_norm_up,
    double grad_sum_up,
    int64_t has_grad_sum,
    int64_t rank,
    int64_t world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = world_size < 32 ? 32 : world_size;
    loss_allreduce_kernel<<<1, threads, 0, stream>>>(
        local_scaled.data_ptr<float>(),
        symm_buf.data_ptr<float>(),
        reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(signal_ptrs.data_ptr<int64_t>()),
        out_norm.data_ptr<float>(),
        out_sum.data_ptr<float>(),
        out_grad.data_ptr<float>(),
        (float)local_valid, (float)global_valid,
        (float)grad_norm_up, (float)grad_sum_up,
        (int)has_grad_sum, (int)rank, (int)world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_loss_allreduce", &launch_loss_allreduce, "fused single-scalar allreduce");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("loss_allreduce_ext", CUDA_SRC)
    return _ext

_cache = None
def _get_state(device):
    global _cache
    if _cache is not None:
        return _cache
    world = dist.get_world_size()
    rank = dist.get_rank()
    buf = symm_mem.empty(world, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    peer_ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    signal_ptrs = torch.tensor(list(hdl.signal_pad_ptrs), device=device, dtype=torch.int64)
    _cache = {
        "buf": buf, "hdl": hdl,
        "peer_ptrs": peer_ptrs, "signal_ptrs": signal_ptrs,
        "rank": rank, "world": world,
    }
    return _cache


@torch.no_grad()
def solution(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    grad_normalized_loss: torch.Tensor,
    grad_loss_sum: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not dist.is_initialized():
        # Single-process fallback
        if local_valid_tokens.item() == 0:
            loss = torch.nan_to_num(loss)
        loss_sum = loss * local_valid_tokens
        normalized_loss = loss_sum / global_valid_tokens
        grad_loss = grad_normalized_loss * local_valid_tokens / global_valid_tokens
        if grad_loss_sum is not None:
            grad_loss = grad_loss + grad_loss_sum * local_valid_tokens
        return normalized_loss, loss_sum, grad_loss

    device = loss.device
    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    st = _get_state(device)

    in_dtype = loss.dtype

    # Sanitize loss (mirror reference: if local_valid == 0, nan_to_num the loss)
    local_valid_f = local_valid_tokens.detach().to(torch.float32).reshape(())
    global_valid_f = global_valid_tokens.detach().to(torch.float32).reshape(())
    loss_f = loss.detach().to(torch.float32).reshape(())

    # Device-side conditional sanitize: if local_valid == 0 -> nan_to_num
    zero_mask = (local_valid_f == 0)
    loss_safe = torch.where(zero_mask, torch.nan_to_num(loss_f), loss_f)
    local_scaled = (loss_safe * local_valid_f).reshape(1).contiguous()

    # Scalars to host (cheap; needed as kernel args)
    local_valid_h = float(local_valid_f.item())
    global_valid_h = float(global_valid_f.item())
    grad_norm_up_h = float(grad_normalized_loss.detach().to(torch.float32).reshape(()).item())
    if grad_loss_sum is not None:
        grad_sum_up_h = float(grad_loss_sum.detach().to(torch.float32).reshape(()).item())
        has_grad_sum = 1
    else:
        grad_sum_up_h = 0.0
        has_grad_sum = 0

    out_norm_f = torch.empty(1, device=device, dtype=torch.float32)
    out_sum_f  = torch.empty(1, device=device, dtype=torch.float32)
    out_grad_f = torch.empty(1, device=device, dtype=torch.float32)

    ext.launch_loss_allreduce(
        local_scaled,
        st["buf"],
        st["peer_ptrs"],
        st["signal_ptrs"],
        out_norm_f, out_sum_f, out_grad_f,
        local_valid_h, global_valid_h,
        grad_norm_up_h, grad_sum_up_h,
        has_grad_sum, st["rank"], st["world"],
    )

    normalized_loss = out_norm_f.to(in_dtype).reshape(loss.shape)
    loss_sum = out_sum_f.to(in_dtype).reshape(loss.shape)
    grad_loss = out_grad_f.to(grad_normalized_loss.dtype).reshape(grad_normalized_loss.shape)

    return normalized_loss, loss_sum, grad_loss