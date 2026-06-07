"""
Strategy:
- We bypass the NCCL all_gather_into_tensor and multiple PyTorch reshape/sum ops.
- Kernel 1 (local_reduce): Reduces `expert_mask` locally to calculate `num_local_tokens_per_expert` and writes directly into a symmetric memory buffer, avoiding intermediate allocations.
- Device-side barrier: We use `hdl.barrier()` to asynchronously synchronize peers.
- Kernel 2 (gather_postprocess): A single thread block cooperatively loads all peers' symmetric buffers into shared memory over NVLink. It then computes `input_splits`, `output_splits`, `num_global_tokens_per_local_expert`, and `num_global_sum_tokens_per_local_expert` purely on-device from shared memory.
- Finally, the outputs are returned in the exact format required, with async CPU copies to overlap with subsequent host execution.
"""

from typing import List, Optional, Tuple

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

template <typename T>
__global__ void local_reduce_kernel(
    const T* __restrict__ mask,
    T* __restrict__ symm_buf,
    int num_experts,
    int N
) {
    int expert_idx = blockIdx.x;
    if (expert_idx >= num_experts) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += static_cast<float>(mask[expert_idx * N + i]);
    }

    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        shared[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        float warp_sum = (lane < (blockDim.x / 32)) ? shared[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        // Add a system-level memory fence to ensure visibility across NVLink before barrier
        if (lane == 0) {
            symm_buf[expert_idx] = static_cast<T>(warp_sum);
            __threadfence_system();
        }
    }
}

template <typename T>
__global__ void gather_postprocess_kernel(
    const uint64_t* __restrict__ peer_ptrs,
    T* __restrict__ global_tokens_local_expert,
    T* __restrict__ global_sum_tokens,
    float* __restrict__ input_splits,
    float* __restrict__ output_splits,
    int ep_size,
    int num_experts,
    int num_local_experts,
    int rank
) {
    extern __shared__ char smem[];
    T* s_all_bufs = reinterpret_cast<T*>(smem);

    int tid = threadIdx.x;
    int total_elements = ep_size * num_experts;

    // Cooperatively load all peers' symmetric buffers into shared memory
    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        int r = idx / num_experts;
        int e = idx % num_experts;
        const T* peer_buf = reinterpret_cast<const T*>(peer_ptrs[r]);
        s_all_bufs[r * num_experts + e] = peer_buf[e];
    }

    __syncthreads();

    // 1. global_tokens_local_expert
    int out_elements = ep_size * num_local_experts;
    for (int idx = tid; idx < out_elements; idx += blockDim.x) {
        int r = idx / num_local_experts;
        int i = idx % num_local_experts;
        global_tokens_local_expert[idx] = s_all_bufs[r * num_experts + rank * num_local_experts + i];
    }

    // 2. global_sum_tokens
    for (int i = tid; i < num_local_experts; i += blockDim.x) {
        float sum = 0.0f;
        for (int r = 0; r < ep_size; r++) {
            sum += static_cast<float>(s_all_bufs[r * num_experts + rank * num_local_experts + i]);
        }
        global_sum_tokens[i] = static_cast<T>(sum);
    }

    // 3. output_splits
    for (int r = tid; r < ep_size; r += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < num_local_experts; i++) {
            sum += static_cast<float>(s_all_bufs[r * num_experts + rank * num_local_experts + i]);
        }
        output_splits[r] = sum;
    }

    // 4. input_splits
    for (int r = tid; r < ep_size; r += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < num_local_experts; i++) {
            sum += static_cast<float>(s_all_bufs[rank * num_experts + r * num_local_experts + i]);
        }
        input_splits[r] = sum;
    }
}

void launch_local_reduce(
    torch::Tensor mask,
    torch::Tensor symm_buf,
    int num_experts,
    int N
) {
    int threads = 256;
    int blocks = num_experts;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mask.scalar_type(), "local_reduce", [&] {
        local_reduce_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            mask.data_ptr<scalar_t>(),
            symm_buf.data_ptr<scalar_t>(),
            num_experts,
            N
        );
    });
}

void launch_gather_postprocess(
    torch::Tensor peer_ptrs,
    torch::Tensor global_tokens_local_expert,
    torch::Tensor global_sum_tokens,
    torch::Tensor input_splits,
    torch::Tensor output_splits,
    int ep_size,
    int num_experts,
    int num_local_experts,
    int rank
) {
    int threads = 256;
    int blocks = 1;
    int shared_mem_bytes = ep_size * num_experts * global_tokens_local_expert.element_size();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, global_tokens_local_expert.scalar_type(), "gather_postprocess", [&] {
        gather_postprocess_kernel<scalar_t><<<blocks, threads, shared_mem_bytes, stream>>>(
            reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>()),
            global_tokens_local_expert.data_ptr<scalar_t>(),
            global_sum_tokens.data_ptr<scalar_t>(),
            input_splits.data_ptr<float>(),
            output_splits.data_ptr<float>(),
            ep_size,
            num_experts,
            num_local_experts,
            rank
        );
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_local_reduce", &launch_local_reduce, "Local reduction kernel");
    m.def("launch_gather_postprocess", &launch_gather_postprocess, "Gather and postprocess kernel");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_token_preprocess_ext", CUDA_SRC)
    return _ext


_symm_cache = {}


def _get_symm_resources(num_experts: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = (num_experts, dtype, device, group)
    if key in _symm_cache:
        return _symm_cache[key]

    buf = symm_mem.empty(num_experts, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, ptrs_tensor)
    _symm_cache[key] = res
    return res


def solution(
    expert_mask: torch.Tensor,
    num_experts: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    ep_size = group.size()
    rank = dist.get_rank(group)
    num_local_experts = num_experts // ep_size

    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()

    ext = _get_ext()
    
    # Cast boolean/integer masks to float32 to support Tensor core floating point math traits 
    # if standard MoE implementations feed non-FP inputs.
    if not expert_mask.is_floating_point():
        expert_mask = expert_mask.to(torch.float32)

    device = expert_mask.device
    dtype = expert_mask.dtype

    buf, hdl, ptrs_tensor = _get_symm_resources(num_experts, dtype, device, group)

    expert_mask_c = expert_mask.contiguous()
    N = expert_mask_c.numel() // num_experts

    # Kernel 1: Local reduction directly into symmetric memory
    ext.launch_local_reduce(expert_mask_c, buf, num_experts, N)

    # Barrier to ensure peer NVLink visibility
    hdl.barrier(channel=0)

    # Output allocation
    global_tokens_local_expert = torch.empty((ep_size, num_local_experts), dtype=dtype, device=device)
    global_sum_tokens = torch.empty((num_local_experts,), dtype=dtype, device=device)
    input_splits = torch.empty((ep_size,), dtype=torch.float32, device=device)
    output_splits = torch.empty((ep_size,), dtype=torch.float32, device=device)

    # Kernel 2: Gather sizes from peers' symmetric buffers and compute all outputs
    ext.launch_gather_postprocess(
        ptrs_tensor,
        global_tokens_local_expert,
        global_sum_tokens,
        input_splits,
        output_splits,
        ep_size,
        num_experts,
        num_local_experts,
        rank
    )

    # Move outputs to CPU to overlap with host-side logic
    input_splits_cpu = input_splits.to(torch.int).tolist()
    output_splits_cpu = output_splits.to(torch.int).tolist()
    out_tokens = global_tokens_local_expert.to(torch.device("cpu"), non_blocking=True)
    out_sum = global_sum_tokens.to(torch.device("cpu"), non_blocking=True)

    return input_splits_cpu, output_splits_cpu, out_tokens, out_sum