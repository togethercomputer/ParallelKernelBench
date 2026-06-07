import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple, Optional
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Helper structure to pass an array of pointers without allocating on device
struct Ptrs {
    void* ptrs[16];
};

template <typename T>
__global__ void step1_kernel(
    const T* loss,
    const T* local_valid_tokens,
    T* symm_buf
) {
    float lvt = static_cast<float>(*local_valid_tokens);
    float loss_sum = 0.0f;
    
    // Equivalent to PyTorch's nan_to_num handling when local_valid_tokens == 0
    if (lvt != 0.0f) {
        loss_sum = static_cast<float>(*loss) * lvt;
    }
    
    *symm_buf = static_cast<T>(loss_sum);
}

template <typename T>
__global__ void step2_kernel(
    Ptrs remote_ptrs,
    const T* local_valid_tokens,
    const T* global_valid_tokens,
    const T* grad_normalized_loss,
    const T* grad_loss_sum,
    T* out_normalized_loss,
    T* out_loss_sum,
    T* out_grad_loss,
    int world_size
) {
    float total_loss_sum = 0.0f;
    // Cross-peer NVLink reads using symmetric memory UVA pointers
    for (int i = 0; i < world_size; ++i) {
        total_loss_sum += static_cast<float>(*(reinterpret_cast<const T*>(remote_ptrs.ptrs[i])));
    }
    
    float gvt = static_cast<float>(*global_valid_tokens);
    float lvt = static_cast<float>(*local_valid_tokens);
    float gnl = static_cast<float>(*grad_normalized_loss);
    
    // Forward pass math
    float norm_loss = total_loss_sum / gvt;
    
    // Backward pass math
    float grad_from_norm = gnl * lvt / gvt;
    float grad_from_sum = 0.0f;
    
    if (grad_loss_sum != nullptr) {
        grad_from_sum = static_cast<float>(*grad_loss_sum) * lvt;
    }
    
    float grad_loss = grad_from_norm + grad_from_sum;
    
    *out_normalized_loss = static_cast<T>(norm_loss);
    *out_loss_sum = static_cast<T>(total_loss_sum);
    *out_grad_loss = static_cast<T>(grad_loss);
}

void run_step1(
    torch::Tensor loss,
    torch::Tensor local_valid_tokens,
    torch::Tensor symm_buf
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, loss.scalar_type(), "run_step1", [&] {
        step1_kernel<scalar_t><<<1, 1, 0, stream>>>(
            loss.data_ptr<scalar_t>(),
            local_valid_tokens.data_ptr<scalar_t>(),
            symm_buf.data_ptr<scalar_t>()
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_step2(
    std::vector<int64_t> remote_ptrs_int,
    torch::Tensor local_valid_tokens,
    torch::Tensor global_valid_tokens,
    torch::Tensor grad_normalized_loss,
    torch::Tensor grad_loss_sum,
    torch::Tensor out_normalized_loss,
    torch::Tensor out_loss_sum,
    torch::Tensor out_grad_loss
) {
    int world_size = remote_ptrs_int.size();
    TORCH_CHECK(world_size <= 16, "world_size > 16 not supported by fixed Ptrs struct");
    
    Ptrs remote_ptrs;
    for (int i = 0; i < world_size; ++i) {
        remote_ptrs.ptrs[i] = reinterpret_cast<void*>(remote_ptrs_int[i]);
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, local_valid_tokens.scalar_type(), "run_step2", [&] {
        const scalar_t* grad_loss_sum_ptr = nullptr;
        // Verify tensor is populated to handle Python passing an empty uninitialized Tensor() for None
        if (grad_loss_sum.defined() && grad_loss_sum.numel() > 0) {
            grad_loss_sum_ptr = grad_loss_sum.data_ptr<scalar_t>();
        }
        
        step2_kernel<scalar_t><<<1, 1, 0, stream>>>(
            remote_ptrs,
            local_valid_tokens.data_ptr<scalar_t>(),
            global_valid_tokens.data_ptr<scalar_t>(),
            grad_normalized_loss.data_ptr<scalar_t>(),
            grad_loss_sum_ptr,
            out_normalized_loss.data_ptr<scalar_t>(),
            out_loss_sum.data_ptr<scalar_t>(),
            out_grad_loss.data_ptr<scalar_t>(),
            world_size
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_step1", &run_step1, "Step 1: Compute local loss sum and store in symm_buf");
    m.def("run_step2", &run_step2, "Step 2: Reduce sum from symm_buf and compute outputs");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("loss_fused_ext", CUDA_SRC)
    return _ext


_symm_cache = None


def _get_symm_state(dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"], c["ptrs"]

    # Allocate symmetric scalar buffer
    buf = symm_mem.empty((1,), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    
    world_size = dist.get_world_size()
    ptrs = [int(hdl.buffer_ptrs[i]) for i in range(world_size)]
    
    _symm_cache = {"dtype": dtype, "device": device, "buf": buf, "hdl": hdl, "ptrs": ptrs}
    return buf, hdl, ptrs


@torch.no_grad()
def solution(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    grad_normalized_loss: torch.Tensor,
    grad_loss_sum: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    assert dist.is_initialized(), "solution() requires torch.distributed to be initialized"
    rank = dist.get_rank()

    # Isolate compilation execution to rank 0 
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    dtype = loss.dtype
    device = loss.device

    buf, hdl, ptrs = _get_symm_state(dtype, device)

    out_normalized_loss = torch.empty_like(loss)
    out_loss_sum = torch.empty_like(loss)
    out_grad_loss = torch.empty_like(loss)

    # Empty tensor fallback if None
    if grad_loss_sum is not None:
        grad_loss_sum_arg = grad_loss_sum.to(dtype)
    else:
        grad_loss_sum_arg = torch.Tensor()

    # Compute `loss_sum` using fast-path and load into symmetric memory
    ext.run_step1(loss, local_valid_tokens, buf)
    
    # Wait for all peers to write their chunk to symm_mem
    hdl.barrier(channel=0)
    
    # Read symm_mem buffers over NVLink, accumulate, and compute backward
    ext.run_step2(
        ptrs,
        local_valid_tokens,
        global_valid_tokens,
        grad_normalized_loss,
        grad_loss_sum_arg,
        out_normalized_loss,
        out_loss_sum,
        out_grad_loss
    )
    
    # Wait for reads to clear before starting next iteration (protects overwrite of `buf`)
    hdl.barrier(channel=0)

    return out_normalized_loss, out_loss_sum, out_grad_loss