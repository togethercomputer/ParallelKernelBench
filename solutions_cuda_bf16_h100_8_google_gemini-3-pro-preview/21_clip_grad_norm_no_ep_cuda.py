"""
Optimized L2 clip_grad_norm for FSDP using custom CUDA and Symmetric Memory.

Strategy:
1. Device-Side Communication: Replaced `dist.all_reduce` with a direct UVA read
   of peer memory. We accumulate each rank's local sum of squares into a 1-float
   symmetric memory buffer, then one kernel computes the global sum directly
   across NVLink via UVA pointers.
2. Complete Compute-Communication Overlap: The entire process (squaring,
   summation, global norm compute, and scaling) is pushed to the GPU stream.
   There are no CPU syncs (no `.item()` calls). The scaling kernel is launched
   asynchronously, conditionally executing only if `total_norm > max_norm`.
   Launch overhead is minimized by traversing and launching tensors inside C++.
"""

import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Kernel 1: Local norm squared accumulation
// ---------------------------------------------------------------------------
template <typename T>
__global__ void add_norm_sq_kernel(const T* __restrict__ data, float* acc, int64_t numel) {
    float local_sum = 0.0f;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < numel; idx += (int64_t)gridDim.x * blockDim.x) {
        float val = static_cast<float>(data[idx]);
        local_sum += val * val;
    }
    
    // Warp reduce
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    
    // Block reduce
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / 32) {
        local_sum = shared_sum[threadIdx.x];
    } else {
        local_sum = 0.0f;
    }
    
    // Final warp reduce on the shared sums
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(acc, local_sum);
        }
    }
}

void compute_local_norm_sq(std::vector<at::Tensor> tensors, int64_t buf_ptr) {
    float* acc = reinterpret_cast<float*>(buf_ptr);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Asynchronously zero the local symmetric memory accumulator
    cudaMemsetAsync(acc, 0, sizeof(float), stream);
    
    for (const auto& t : tensors) {
        if (!t.defined() || t.numel() == 0) continue;
        int64_t numel = t.numel();
        int threads = 256;
        int blocks = std::min((int)((numel + threads - 1) / threads), 1024);
        
        if (t.dtype() == torch::kBFloat16) {
            add_norm_sq_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<at::BFloat16>(), acc, numel);
        } else if (t.dtype() == torch::kFloat32) {
            add_norm_sq_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<float>(), acc, numel);
        } else if (t.dtype() == torch::kFloat16) {
            add_norm_sq_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<at::Half>(), acc, numel);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels 2 & 3: Read peers, global norm computation, and scaling
// ---------------------------------------------------------------------------
__global__ void compute_total_norm_kernel(
    const int64_t* __restrict__ peer_ptrs,
    float* __restrict__ out_total_norm,
    int group_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total_sq = 0.0f;
        for (int i = 0; i < group_size; ++i) {
            const float* peer_buf = reinterpret_cast<const float*>(peer_ptrs[i]);
            total_sq += *peer_buf;
        }
        *out_total_norm = sqrtf(total_sq);
    }
}

template <typename T>
__global__ void scale_gradients_kernel(
    T* __restrict__ data,
    const float* __restrict__ total_norm_ptr,
    float max_norm,
    int64_t numel
) {
    float total_norm = *total_norm_ptr;
    // Condition is purely device-side, avoiding CPU-GPU synchronization entirely
    if (total_norm > max_norm) {
        float coef = max_norm / total_norm;
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        for (; idx < numel; idx += (int64_t)gridDim.x * blockDim.x) {
            float val = static_cast<float>(data[idx]);
            data[idx] = static_cast<T>(val * coef);
        }
    }
}

void compute_global_norm_and_scale(
    std::vector<at::Tensor> tensors,
    at::Tensor peer_ptrs_tensor,
    float max_norm,
    int group_size,
    at::Tensor out_total_norm
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const int64_t* peer_ptrs = peer_ptrs_tensor.data_ptr<int64_t>();
    float* out_norm = out_total_norm.data_ptr<float>();
    
    // One thread accumulates all peers' partials over P2P UVA pointers
    compute_total_norm_kernel<<<1, 1, 0, stream>>>(peer_ptrs, out_norm, group_size);
    
    for (const auto& t : tensors) {
        if (!t.defined() || t.numel() == 0) continue;
        int64_t numel = t.numel();
        int threads = 256;
        int blocks = std::min((int)((numel + threads - 1) / threads), 1024);
        
        if (t.dtype() == torch::kBFloat16) {
            scale_gradients_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<at::BFloat16>(), out_norm, max_norm, numel);
        } else if (t.dtype() == torch::kFloat32) {
            scale_gradients_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<float>(), out_norm, max_norm, numel);
        } else if (t.dtype() == torch::kFloat16) {
            scale_gradients_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<at::Half>(), out_norm, max_norm, numel);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_local_norm_sq", &compute_local_norm_sq, "Compute local sum of squares");
    m.def("compute_global_norm_and_scale", &compute_global_norm_and_scale, "Compute global norm and scale in-place");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("clip_grad_norm_uva_ext", CUDA_SRC)
    return _ext


_symm_cache = {}


def _get_symm_state(device: torch.device, group: Optional[dist.ProcessGroup]):
    group_id = id(group) if group is not None else 0
    if group_id in _symm_cache:
        return _symm_cache[group_id]

    buf = symm_mem.empty((1,), dtype=torch.float32, device=device)
    hdl = symm_mem.rendezvous(buf, group=group if group is not None else dist.group.WORLD)
    peer_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    state = (buf, hdl, peer_ptrs)
    _symm_cache[group_id] = state
    return state


def fallback_solution(
    grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float,
    fsdp_group: Optional[dist.ProcessGroup]
) -> torch.Tensor:
    """Stock PyTorch fallback path."""
    p = float(norm_type)
    dev = None
    acc = None
    for g in grad_tensors:
        if g is None: 
            continue
        if dev is None:
            dev = g.device
            acc = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.norm(g.detach().to(torch.float32), p=p)
        acc = acc + (gn ** p)
        
    if acc is None:
        acc = torch.tensor(0.0, device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.float32)
    
    if fsdp_group is not None:
        dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=fsdp_group)
    elif dist.is_initialized():
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        
    total_norm = acc ** (1.0 / p)
    
    if total_norm > max_norm:
        coef = max_norm / total_norm
        for t in grad_tensors:
            if t is not None:
                t.mul_(coef.to(t.device))
                
    return total_norm


@torch.no_grad()
def solution(
    grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    fsdp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Computes global L2 norm across all ranks locally & distributed, then scales.
    Zero-overhead FSDP L2 norm clipping using symmetric memory & UVA buffers.
    """
    if not dist.is_initialized() or float(norm_type) != 2.0:
        return fallback_solution(grad_tensors, max_norm, norm_type, fsdp_group)
    
    valid_tensors = [t for t in grad_tensors if t is not None]
    
    device = valid_tensors[0].device if valid_tensors else torch.device("cuda", torch.cuda.current_device())
    
    buf, hdl, peer_ptrs = _get_symm_state(device, fsdp_group)
    out_total_norm = torch.empty((), dtype=torch.float32, device=device)
    
    ext = _get_ext()
    
    # Kernel 1: Calculate local sq norm asynchronously in 1 float
    ext.compute_local_norm_sq(valid_tensors, buf.data_ptr())
    
    # Memory wall 0: Ensures peers finished updating their local symmetric buffers
    hdl.barrier(channel=0)
    
    # Kernel 2 + 3: Sum the buffers over UVA, calculate root and scale asynchronously
    ext.compute_global_norm_and_scale(
        valid_tensors,
        peer_ptrs,
        float(max_norm),
        len(hdl.buffer_ptrs),
        out_total_norm
    )
    
    # Memory wall 1: Ensures peers read this rank's symmetric buffer 
    # before returning and allowing a new iteration to memset it.
    hdl.barrier(channel=1)
    
    return out_total_norm