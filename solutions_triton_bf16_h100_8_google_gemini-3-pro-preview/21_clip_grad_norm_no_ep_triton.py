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
#include <math.h>
#include <algorithm>

__global__ void local_norm_kernel(
    __nv_bfloat16** ptrs,
    int64_t* cum_sizes,
    int num_tensors,
    int64_t total_elements,
    float p,
    float* out_local_sq_norm
) {
    float local_sum = 0.0f;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    // Track tensor index across iterations for O(1) amortized lookup
    int tensor_idx = 0;
    
    for (int64_t i = tid; i < total_elements; i += stride) {
        while (tensor_idx < num_tensors && i >= cum_sizes[tensor_idx + 1]) {
            tensor_idx++;
        }
        if (tensor_idx < num_tensors) {
            int64_t offset = i - cum_sizes[tensor_idx];
            float val = __bfloat162float(ptrs[tensor_idx][offset]);
            if (p == 2.0f) {
                local_sum += val * val;
            } else {
                local_sum += powf(fabsf(val), p);
            }
        }
    }

    // Warp-level reduce
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        shared[wid] = local_sum;
    }
    __syncthreads();

    // Block-level reduce
    local_sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) {
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(out_local_sq_norm, local_sum);
    }
}

__global__ void scale_kernel(
    __nv_bfloat16** ptrs,
    int64_t* cum_sizes,
    int num_tensors,
    int64_t total_elements,
    const int64_t* peer_ptrs,
    int world_size,
    float max_norm,
    float p,
    float* out_total_norm
) {
    __shared__ float shared_scale;

    // Thread 0 collects symmetric memory bounds via UVA loads and determines the global scale
    if (threadIdx.x == 0) {
        float global_sq_norm = 0.0f;
        for (int i = 0; i < world_size; i++) {
            const float* peer_ptr = reinterpret_cast<const float*>(peer_ptrs[i]);
            global_sq_norm += *peer_ptr;
        }
        float total_norm = powf(global_sq_norm, 1.0f / p);
        if (total_norm > max_norm && total_norm > 0.0f) {
            shared_scale = max_norm / total_norm;
        } else {
            shared_scale = 1.0f;
        }
        // Write out the exact resulting L2 norm to Python boundary
        if (blockIdx.x == 0 && out_total_norm != nullptr) {
            *out_total_norm = total_norm;
        }
    }
    __syncthreads();

    float scale = shared_scale;
    if (scale == 1.0f) return; // Scale factor avoids unnecessary math

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int tensor_idx = 0;

    for (int64_t i = tid; i < total_elements; i += stride) {
        while (tensor_idx < num_tensors && i >= cum_sizes[tensor_idx + 1]) {
            tensor_idx++;
        }
        if (tensor_idx < num_tensors) {
            int64_t offset = i - cum_sizes[tensor_idx];
            float val = __bfloat162float(ptrs[tensor_idx][offset]);
            val *= scale;
            ptrs[tensor_idx][offset] = __float2bfloat16(val);
        }
    }
}

void compute_local_norm_bf16(
    int64_t ptrs_tensor_ptr,
    int64_t cum_sizes_ptr,
    int num_tensors,
    int64_t total_elements,
    float p,
    torch::Tensor out_local_sq_norm
) {
    out_local_sq_norm.zero_();

    if (total_elements > 0) {
        const int threads = 256;
        const int blocks = std::max(1, std::min((int)((total_elements + threads - 1) / threads), 1024 * 4));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        local_norm_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16**>(ptrs_tensor_ptr),
            reinterpret_cast<int64_t*>(cum_sizes_ptr),
            num_tensors,
            total_elements,
            p,
            out_local_sq_norm.data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void scale_tensors_bf16(
    int64_t ptrs_tensor_ptr,
    int64_t cum_sizes_ptr,
    int num_tensors,
    int64_t total_elements,
    int64_t peer_ptrs_ptr,
    int world_size,
    float max_norm,
    float p,
    torch::Tensor out_total_norm
) {
    const int threads = 256;
    const int blocks = std::max(1, std::min((int)((total_elements + threads - 1) / threads), 1024 * 4));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    scale_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16**>(ptrs_tensor_ptr),
        reinterpret_cast<int64_t*>(cum_sizes_ptr),
        num_tensors,
        total_elements,
        reinterpret_cast<const int64_t*>(peer_ptrs_ptr),
        world_size,
        max_norm,
        p,
        out_total_norm.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_local_norm_bf16", &compute_local_norm_bf16, "Compute local p-norm for BF16 tensors");
    m.def("scale_tensors_bf16", &scale_tensors_bf16, "Scale BF16 tensors based on global norm");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("clip_grad_norm_bf16_ext", CUDA_SRC)
    return _ext


_tensor_cache = {}

def _get_tensor_cache(valid_tensors: List[torch.Tensor], device: torch.device):
    global _tensor_cache
    # robust cache map bound to data storage address and numel mapping
    cache_key = tuple((t.data_ptr(), t.numel()) for t in valid_tensors)
    if cache_key in _tensor_cache:
        return _tensor_cache[cache_key]

    ptrs_list = []
    cum_sizes_list = [0]
    total_elements = 0
    for t in valid_tensors:
        ptrs_list.append(t.data_ptr())
        total_elements += t.numel()
        cum_sizes_list.append(total_elements)

    num_tensors = len(ptrs_list)
    ptrs_dev = torch.tensor(ptrs_list, dtype=torch.int64, device=device)
    cum_sizes_dev = torch.tensor(cum_sizes_list, dtype=torch.int64, device=device)
    
    # Cap cache to avoid host memory leaks from distinct model parameter group splits
    if len(_tensor_cache) >= 16:
        _tensor_cache.pop(next(iter(_tensor_cache)))
        
    _tensor_cache[cache_key] = (ptrs_dev, cum_sizes_dev, num_tensors, total_elements)
    return _tensor_cache[cache_key]


_symm_cache = {}

def _get_symm_state(fsdp_group: Optional[dist.ProcessGroup], device: torch.device):
    global _symm_cache
    group_id = id(fsdp_group)
    if group_id in _symm_cache:
        return _symm_cache[group_id]

    world_size = dist.get_world_size(fsdp_group)
    # The L2 intermediate local accumulation buffer
    buf = symm_mem.empty(1, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, fsdp_group)

    peer_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    out_total_norm = torch.empty(1, device=device, dtype=torch.float32)

    _symm_cache[group_id] = (buf, hdl, peer_ptrs, out_total_norm, world_size)
    return _symm_cache[group_id]


def _fallback_solution(grad_tensors: List[torch.Tensor], max_norm: float, norm_type: float):
    p = float(norm_type)
    device = next((t.device for t in grad_tensors if t is not None), torch.device("cuda"))
    acc = torch.tensor(0.0, device=device, dtype=torch.float32)
    for g in grad_tensors:
        if g is not None:
            acc += torch.norm(g.detach().to(torch.float32), p=p) ** p
            
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
    valid_tensors = [t for t in grad_tensors if t is not None]
    
    # Graceful fallback for non-bf16 or non-distributed regimes
    if not dist.is_initialized() or (valid_tensors and valid_tensors[0].dtype != torch.bfloat16):
        return _fallback_solution(grad_tensors, max_norm, norm_type)

    device = valid_tensors[0].device if valid_tensors else torch.device("cuda", torch.cuda.current_device())
    group = fsdp_group if fsdp_group is not None else dist.group.WORLD

    # Safely compile the extension on the lead rank and synchronize cluster
    if dist.get_rank(group) == 0:
        _get_ext()
    dist.barrier(group)
    ext = _get_ext()

    # Grab symmetric memory handles
    buf, hdl, peer_ptrs_dev, out_total_norm, world_size = _get_symm_state(group, device)

    # Empty inputs branch handles gracefully via 0-tensor dispatch to correctly broadcast total_norm scale
    if not valid_tensors:
        buf.zero_()
        torch.cuda.current_stream().synchronize()
        hdl.barrier(channel=0)
        ext.scale_tensors_bf16(
            0, 0, 0, 0, peer_ptrs_dev.data_ptr(), world_size, float(max_norm), float(norm_type), out_total_norm
        )
        return out_total_norm[0]

    # Gather descriptor mappings (sizes/addresses)
    ptrs_dev, cum_sizes_dev, num_tensors, total_elements = _get_tensor_cache(valid_tensors, device)

    # 1. Compute Local L2 Sub-Sum via 1D Thread Map Reduction
    ext.compute_local_norm_bf16(
        ptrs_dev.data_ptr(),
        cum_sizes_dev.data_ptr(),
        num_tensors,
        total_elements,
        float(norm_type),
        buf
    )

    # Rendezvous barrier blocks further traversal until all ranks have registered compute completion buffers.
    # Synchronization isolates global scaling sequence from stale UVA pointers.
    torch.cuda.current_stream().synchronize()
    hdl.barrier(channel=0)

    # 2. Scale In-Place Across Fully Distant Global L2 Sum Matrix 
    ext.scale_tensors_bf16(
        ptrs_dev.data_ptr(),
        cum_sizes_dev.data_ptr(),
        num_tensors,
        total_elements,
        peer_ptrs_dev.data_ptr(),
        world_size,
        float(max_norm),
        float(norm_type),
        out_total_norm
    )

    return out_total_norm[0]