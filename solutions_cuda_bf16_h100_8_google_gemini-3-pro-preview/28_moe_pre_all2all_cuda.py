from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename scalar_t>
__global__ void push_kernel_tokens(
    const scalar_t* __restrict__ hidden_states,
    const int64_t* __restrict__ sorted_indices,
    const int* __restrict__ send_offsets,
    const int* __restrict__ send_counts,
    const int64_t* __restrict__ peer_meta_bufs,
    const int64_t* __restrict__ peer_recv_bufs,
    int num_experts,
    int num_local_experts,
    int hidden_dim,
    int my_rank,
    int total_tokens
) {
    int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (token_idx >= total_tokens) return;

    // Binary search send_offsets to find the destination expert for this token
    int L = 0, R = num_experts - 1;
    int E = 0;
    while (L <= R) {
        int mid = (L + R) / 2;
        if (send_offsets[mid] <= token_idx) {
            E = mid;
            L = mid + 1;
        } else {
            R = mid - 1;
        }
    }

    int r = E / num_local_experts;
    int e = E % num_local_experts;

    int src_token = sorted_indices[token_idx];
    
    // Read remote target offset from the peer's meta_buf
    const int* remote_meta = reinterpret_cast<const int*>(peer_meta_bufs[r]);
    int dest_base = remote_meta[my_rank * num_local_experts + e];
    int dest_token = dest_base + (token_idx - send_offsets[E]);

    const scalar_t* src_row = hidden_states + src_token * hidden_dim;
    scalar_t* dest_row = reinterpret_cast<scalar_t*>(peer_recv_bufs[r]) + dest_token * hidden_dim;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    int bytes = hidden_dim * sizeof(scalar_t);
    // Use 128-bit, 64-bit, or 32-bit vectorized memory accesses natively
    if (bytes % 16 == 0) {
        const float4* src_vec = reinterpret_cast<const float4*>(src_row);
        float4* dest_vec = reinterpret_cast<float4*>(dest_row);
        int vec_dim = bytes / 16;
        for (int i = tid; i < vec_dim; i += stride) {
            dest_vec[i] = src_vec[i];
        }
    } else if (bytes % 8 == 0) {
        const float2* src_vec = reinterpret_cast<const float2*>(src_row);
        float2* dest_vec = reinterpret_cast<float2*>(dest_row);
        int vec_dim = bytes / 8;
        for (int i = tid; i < vec_dim; i += stride) {
            dest_vec[i] = src_vec[i];
        }
    } else if (bytes % 4 == 0) {
        const float* src_vec = reinterpret_cast<const float*>(src_row);
        float* dest_vec = reinterpret_cast<float*>(dest_row);
        int vec_dim = bytes / 4;
        for (int i = tid; i < vec_dim; i += stride) {
            dest_vec[i] = src_vec[i];
        }
    } else {
        for (int i = tid; i < hidden_dim; i += stride) {
            dest_row[i] = src_row[i];
        }
    }
}

void launch_push_tokens(
    torch::Tensor hidden_states,
    torch::Tensor sorted_indices,
    torch::Tensor send_offsets,
    torch::Tensor send_counts,
    torch::Tensor peer_meta_bufs,
    torch::Tensor peer_recv_bufs,
    int num_experts,
    int num_local_experts,
    int hidden_dim,
    int my_rank,
    int total_tokens
) {
    if (total_tokens == 0) return;

    // 2D Block mapping: 32 threads over feature dimension, 8 tokens per block
    int threads_x = 32;
    int threads_y = 8;
    dim3 block(threads_x, threads_y);
    int grid = (total_tokens + threads_y - 1) / threads_y;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto dtype = hidden_states.scalar_type();

    if (dtype == torch::kBFloat16) {
        push_kernel_tokens<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
            sorted_indices.data_ptr<int64_t>(),
            send_offsets.data_ptr<int>(),
            send_counts.data_ptr<int>(),
            peer_meta_bufs.data_ptr<int64_t>(),
            peer_recv_bufs.data_ptr<int64_t>(),
            num_experts, num_local_experts, hidden_dim, my_rank, total_tokens);
    } else if (dtype == torch::kFloat16) {
        push_kernel_tokens<half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(hidden_states.data_ptr()),
            sorted_indices.data_ptr<int64_t>(),
            send_offsets.data_ptr<int>(),
            send_counts.data_ptr<int>(),
            peer_meta_bufs.data_ptr<int64_t>(),
            peer_recv_bufs.data_ptr<int64_t>(),
            num_experts, num_local_experts, hidden_dim, my_rank, total_tokens);
    } else if (dtype == torch::kFloat32) {
        push_kernel_tokens<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(hidden_states.data_ptr()),
            sorted_indices.data_ptr<int64_t>(),
            send_offsets.data_ptr<int>(),
            send_counts.data_ptr<int>(),
            peer_meta_bufs.data_ptr<int64_t>(),
            peer_recv_bufs.data_ptr<int64_t>(),
            num_experts, num_local_experts, hidden_dim, my_rank, total_tokens);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for push_tokens kernel.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_push_tokens", &launch_push_tokens, "PUSH tokens directly to symmetric memory final offsets");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_push_tokens_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(max_tokens, hidden_dim, dtype, device, world_size, num_local_experts):
    key = (hidden_dim, dtype, device, world_size, num_local_experts)
    if key in _symm_cache:
        res = _symm_cache[key]
        if res['max_tokens'] >= max_tokens:
            return res

    # Re-allocate only if capacities exceed bounds
    recv_buf = symm_mem.empty((max_tokens, hidden_dim), dtype=dtype, device=device)
    hdl_recv = symm_mem.rendezvous(recv_buf, dist.group.WORLD)
    
    meta_buf = symm_mem.empty((world_size, num_local_experts), dtype=torch.int32, device=device)
    hdl_meta = symm_mem.rendezvous(meta_buf, dist.group.WORLD)
    
    peer_meta_ptrs = torch.tensor(hdl_meta.buffer_ptrs, dtype=torch.int64, device=device)
    peer_recv_ptrs = torch.tensor(hdl_recv.buffer_ptrs, dtype=torch.int64, device=device)
    
    res = {
        'max_tokens': max_tokens,
        'recv_buf': recv_buf,
        'hdl_recv': hdl_recv,
        'meta_buf': meta_buf,
        'hdl_meta': hdl_meta,
        'peer_meta_ptrs': peer_meta_ptrs,
        'peer_recv_ptrs': peer_recv_ptrs
    }
    _symm_cache[key] = res
    return res


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
    world_size = dist.get_world_size(group)
    my_rank = dist.get_rank(group)
    
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    org_hidden_states_shape = hidden_states.shape
    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    num_local_experts = num_experts // world_size

    routing_map = expert_mask.sum(dim=1)
    routing_map_bool = routing_map.bool()
    
    # ---------------------------------------------------------
    # Single GPU Short-circuit (No collective overhead)
    # ---------------------------------------------------------
    if world_size == 1:
        token_indices = torch.arange(num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
        sorted_indices = token_indices.masked_select(routing_map_bool)
        local_permuted = hidden_states.index_select(0, sorted_indices)
        
        expected_tokens = sum(input_splits) if isinstance(input_splits, list) else int(input_splits.sum().item())
        actual_tokens = sorted_indices.size(0)
        if expected_tokens != actual_tokens:
            raise RuntimeError(f"EP split mismatch: input_splits sum ({expected_tokens}) != permuted tokens ({actual_tokens})")
        return local_permuted, routing_map, sorted_indices, org_hidden_states_shape

    # Max bound ensuring enough space even during intense routing load variations
    max_tokens = world_size * num_tokens
    
    buf = _get_symm_state(max_tokens, hidden_dim, hidden_states.dtype, device, world_size, num_local_experts)
    recv_buf = buf['recv_buf']
    meta_buf = buf['meta_buf']
    hdl_recv = buf['hdl_recv']
    hdl_meta = buf['hdl_meta']

    # ---------------------------------------------------------
    # 1. Receiver Meta Computation & Symmetric Scatter
    # Compute the final destination offsets for peer chunks.
    # ---------------------------------------------------------
    N = num_global_tokens_per_local_expert
    expert_sizes = N.sum(dim=0)
    expert_base_offsets = expert_sizes.cumsum(dim=0) - expert_sizes
    rank_offsets_within_expert = N.cumsum(dim=0) - N
    dest_offsets = (expert_base_offsets.unsqueeze(0) + rank_offsets_within_expert).to(torch.int32)
    
    meta_buf.copy_(dest_offsets)
    hdl_meta.barrier(channel=0) # Sync independent metadata fast track
    
    # ---------------------------------------------------------
    # 2. Sender Local Prep
    # Overlapping execution locally while peers expose pointers.
    # ---------------------------------------------------------
    send_counts = routing_map_bool.sum(dim=1, dtype=torch.int32)
    send_offsets = send_counts.cumsum(dim=0) - send_counts
    
    token_indices = torch.arange(num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map_bool)
    
    total_send_tokens = sorted_indices.size(0)
    expected_tokens = sum(input_splits) if isinstance(input_splits, list) else int(input_splits.sum().item())
    if expected_tokens != total_send_tokens:
        raise RuntimeError(f"EP split mismatch: input_splits sum ({expected_tokens}) != permuted tokens ({total_send_tokens})")

    # ---------------------------------------------------------
    # 3. Custom Fused Push Scatter Operator
    # ---------------------------------------------------------
    _get_ext().launch_push_tokens(
        hidden_states,
        sorted_indices,
        send_offsets,
        send_counts,
        buf['peer_meta_ptrs'],
        buf['peer_recv_ptrs'],
        num_experts,
        num_local_experts,
        hidden_dim,
        my_rank,
        total_send_tokens
    )
    
    hdl_recv.barrier(channel=0)

    # ---------------------------------------------------------
    # 4. Final Output Construction
    # ---------------------------------------------------------
    total_recv_tokens = int(N.sum().item())
    global_permuted_hidden_states = recv_buf[:total_recv_tokens].clone()

    return global_permuted_hidden_states, routing_map, sorted_indices, org_hidden_states_shape