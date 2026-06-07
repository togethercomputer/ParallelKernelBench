"""
Strategy:
- Fuses local permutation, all-to-all communication, and the final chunk sorting into a single PULL-based direct memory access phase.
- Replaces NCCL `all_to_all_single` with custom device-side memory routing using `torch.distributed._symmetric_memory`, removing intermediate collective buffers.
- Pre-packs token blocks into a single reused symmetric buffer using a highly parallel fused gather kernel.
- Leverages UVA over NVLink: each rank computes exact destination offsets and directly pulls its required blocks of tokens from peer symmetric buffers into their exact final sorted positions, entirely skipping the CPU-bound `_sort_chunks_by_idxs` operation.
- Synchronization is strictly device-side via `hdl.barrier()`, ensuring full GPU compute-communication overlap without host stalling.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>

__global__ void gather_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int64_t* __restrict__ indices,
    __nv_bfloat16* __restrict__ dst,
    int64_t num_indices,
    int hidden_dim
) {
    int64_t token_idx = blockIdx.x;
    if (token_idx < num_indices) {
        int64_t src_row = indices[token_idx];
        const __nv_bfloat16* src_row_ptr = src + src_row * hidden_dim;
        __nv_bfloat16* dst_row_ptr = dst + token_idx * hidden_dim;
        
        if (hidden_dim % 8 == 0) {
            const float4* src_vec = reinterpret_cast<const float4*>(src_row_ptr);
            float4* dst_vec = reinterpret_cast<float4*>(dst_row_ptr);
            int num_vec = hidden_dim / 8;
            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                dst_vec[i] = src_vec[i];
            }
        } else {
            for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
                dst_row_ptr[i] = src_row_ptr[i];
            }
        }
    }
}

void gather_cuda(
    torch::Tensor src,
    torch::Tensor indices,
    torch::Tensor dst,
    int64_t num_indices,
    int hidden_dim
) {
    if (num_indices == 0) return;
    int threads = std::min(hidden_dim / 8, 256);
    if (hidden_dim % 8 != 0) threads = std::min(hidden_dim, 256);
    if (threads <= 0) threads = 1; 
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_kernel<<<num_indices, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        indices.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
        num_indices,
        hidden_dim
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void moe_pull_kernel(
    const uint8_t* const* __restrict__ symm_buf_ptrs,
    int64_t bytes_offsets,
    const int32_t* __restrict__ dst_offsets,
    const int32_t* __restrict__ counts,
    __nv_bfloat16* __restrict__ out,
    int W, int L, int hidden_dim, int my_rank
) {
    int chunk_idx = blockIdx.y;
    int src_rank = chunk_idx / L;
    int e_local = chunk_idx % L;
    
    int count = counts[src_rank * L + e_local];
    if (count == 0) return;
    
    int dst_offset = dst_offsets[src_rank * L + e_local];
    
    const uint8_t* remote_buf = symm_buf_ptrs[src_rank];
    const int32_t* remote_offsets = reinterpret_cast<const int32_t*>(remote_buf);
    const __nv_bfloat16* remote_in = reinterpret_cast<const __nv_bfloat16*>(remote_buf + bytes_offsets);
    
    __shared__ int src_offset;
    if (threadIdx.x == 0) {
        src_offset = remote_offsets[my_rank * L + e_local];
    }
    __syncthreads();
    
    int64_t total_elements = (int64_t)count * hidden_dim;
    const __nv_bfloat16* src = remote_in + (int64_t)src_offset * hidden_dim;
    __nv_bfloat16* dst = out + (int64_t)dst_offset * hidden_dim;
    
    int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    if (hidden_dim % 8 == 0) {
        int64_t total_vec = total_elements / 8;
        const float4* src_vec = reinterpret_cast<const float4*>(src);
        float4* dst_vec = reinterpret_cast<float4*>(dst);
        for (int64_t i = tid; i < total_vec; i += stride) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        for (int64_t i = tid; i < total_elements; i += stride) {
            dst[i] = src[i];
        }
    }
}

void moe_pull_cuda(
    torch::Tensor symm_buf_ptrs_tensor,
    int64_t bytes_offsets,
    torch::Tensor dst_offsets,
    torch::Tensor counts,
    torch::Tensor out,
    int W, int L, int hidden_dim, int my_rank
) {
    dim3 grid(16, W * L);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    moe_pull_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t* const*>(symm_buf_ptrs_tensor.data_ptr<int64_t>()),
        bytes_offsets,
        dst_offsets.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        W, L, hidden_dim, my_rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_cuda", &gather_cuda, "Gather tokens mapped by indices");
    m.def("moe_pull_cuda", &moe_pull_cuda, "MoE symmetric pull over NVLink");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_pre_all2all_pull_ext", CUDA_SRC)
    return _ext

_symm_cache = None

def _get_symm_state(max_tokens: int, hidden_dim: int, E: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    
    dtype_size = 2 if dtype == torch.bfloat16 else 4
    bytes_offsets = ((E * 4 + 255) // 256) * 256
    
    if _symm_cache is not None:
        c = _symm_cache
        if c["max_tokens"] >= max_tokens and c["dtype"] == dtype and c["E"] == E:
            symm_src_offsets = c["symm_buf"][:E*4].view(torch.int32)
            symm_in = c["symm_buf"][bytes_offsets:bytes_offsets + c["max_tokens"] * hidden_dim * dtype_size].view(dtype).view(-1, hidden_dim)
            return symm_src_offsets, symm_in, c["hdl"], c["buf_ptrs"], bytes_offsets

    alloc_tokens = int(max_tokens * 1.2)
    if alloc_tokens == 0:
        alloc_tokens = 1024

    bytes_in = alloc_tokens * hidden_dim * dtype_size
    total_bytes = bytes_offsets + bytes_in
    
    symm_buf = symm_mem.empty(total_bytes, device=device, dtype=torch.uint8)
    hdl = symm_mem.rendezvous(symm_buf, group)
    buf_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    _symm_cache = {
        "max_tokens": alloc_tokens,
        "dtype": dtype,
        "E": E,
        "symm_buf": symm_buf,
        "hdl": hdl,
        "buf_ptrs": buf_ptrs
    }
    
    symm_src_offsets = symm_buf[:E*4].view(torch.int32)
    symm_in = symm_buf[bytes_offsets:bytes_offsets + alloc_tokens * hidden_dim * dtype_size].view(dtype).view(-1, hidden_dim)
    
    return symm_src_offsets, symm_in, hdl, buf_ptrs, bytes_offsets

@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: Union[List[int], torch.Tensor],
    output_splits: Union[List[int], torch.Tensor],
    num_global_tokens_per_local_expert: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    group = group or dist.group.WORLD
    device = hidden_states.device
    W = dist.get_world_size(group)
    my_rank = dist.get_rank(group)
    L = num_experts // W
    
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    num_tokens = hidden_states.size(0)

    # Compile kernel centrally at runtime setup if needed
    if my_rank == 0:
        _get_ext()
    dist.barrier(group)
    ext = _get_ext()

    # Determine permutation indices matching _permute correctly
    routing_map = expert_mask.sum(dim=1)
    routing_map_bool = routing_map.bool()
    
    token_indices = torch.arange(num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map_bool)
    actual_tokens = sorted_indices.size(0)

    expected_tokens = sum(input_splits) if isinstance(input_splits, list) else int(input_splits.sum().item())
    if expected_tokens != actual_tokens:
        raise RuntimeError(
            f"EP split mismatch: input_splits sum ({expected_tokens}) != permuted tokens ({actual_tokens})"
        )

    # Compute source offsets
    local_expert_counts = routing_map.sum(dim=1).to(torch.int32)
    local_offsets = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32), 
        local_expert_counts.cumsum(0)[:-1].to(torch.int32)
    ])

    # Grab symmetric memory resources
    symm_src_offsets, symm_in, hdl, buf_ptrs, bytes_offsets = _get_symm_state(
        actual_tokens, hidden_dim, num_experts, hidden_states.dtype, device, group
    )

    # Scatter initial properties to symmetric buffers locally and gather peer pointers
    symm_src_offsets.copy_(local_offsets)
    ext.gather_cuda(hidden_states, sorted_indices, symm_in, actual_tokens, hidden_dim)

    # Barrier before cross-rank PULL
    hdl.barrier(channel=0)

    # Setup Native Output
    out_size = sum(output_splits) if isinstance(output_splits, list) else int(output_splits.sum().item())
    global_permuted_hidden_states = torch.empty((out_size, hidden_dim), device=device, dtype=hidden_states.dtype)

    # Reorder layout locally using exclusive prefix-sum logic exactly simulating chunk sorting
    counts_int32 = num_global_tokens_per_local_expert.to(torch.int32).contiguous()
    M_T = counts_int32.t()
    offsets_T = M_T.flatten().cumsum(0) - M_T.flatten()
    dst_offsets = offsets_T.reshape(L, W).t().contiguous()

    # Pull directly across UVA
    ext.moe_pull_cuda(
        buf_ptrs, 
        bytes_offsets, 
        dst_offsets, 
        counts_int32, 
        global_permuted_hidden_states, 
        W, 
        L, 
        hidden_dim, 
        my_rank
    )

    # Wait for completion, averting overlapping lifecycle states
    hdl.barrier(channel=0)

    return global_permuted_hidden_states, routing_map, sorted_indices, org_hidden_states_shape