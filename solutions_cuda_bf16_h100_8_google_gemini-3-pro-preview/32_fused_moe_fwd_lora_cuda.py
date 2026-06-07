import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import List, Optional, Tuple, Union

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

__global__ void all_gather_counts_kernel(
    const int* __restrict__ local_counts,
    const long long* __restrict__ peer_ptrs,
    int rank,
    int num_experts,
    int world_size
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_experts) {
        int val = local_counts[tid];
        for (int p = 0; p < world_size; p++) {
            int* peer_counts = (int*)peer_ptrs[p];
            peer_counts[rank * num_experts + tid] = val;
        }
    }
}

template <typename scalar_t>
__global__ void dispatch_tokens_kernel(
    const scalar_t* __restrict__ hidden_states,
    const long long* __restrict__ peer_tokens_ptrs,
    const int* __restrict__ owner,
    const int* __restrict__ dest_offset,
    int hidden_dim,
    int top_k
) {
    int token_idx = blockIdx.x;
    int k_idx = blockIdx.y;

    int target_rank = owner[token_idx * top_k + k_idx];
    int target_offset = dest_offset[token_idx * top_k + k_idx];
    scalar_t* dest_ptr = (scalar_t*)peer_tokens_ptrs[target_rank];

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dest_ptr[target_offset * hidden_dim + i] = hidden_states[token_idx * hidden_dim + i];
    }
}

template <>
__global__ void dispatch_tokens_kernel<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__ hidden_states,
    const long long* __restrict__ peer_tokens_ptrs,
    const int* __restrict__ owner,
    const int* __restrict__ dest_offset,
    int hidden_dim,
    int top_k
) {
    int token_idx = blockIdx.x;
    int k_idx = blockIdx.y;

    int target_rank = owner[token_idx * top_k + k_idx];
    int target_offset = dest_offset[token_idx * top_k + k_idx];
    __nv_bfloat16* dest_ptr = (__nv_bfloat16*)peer_tokens_ptrs[target_rank];

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        dest_ptr[target_offset * hidden_dim + i] = hidden_states[token_idx * hidden_dim + i];
    }
}

template <typename scalar_t>
__global__ void pull_tokens_kernel(
    const long long* __restrict__ peer_expert_out_ptrs,
    const int* __restrict__ owner,
    const int* __restrict__ dest_offset,
    const float* __restrict__ routing_weights,
    scalar_t* __restrict__ out,
    int hidden_dim,
    int top_k
) {
    int token_idx = blockIdx.x;

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float accum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int target_rank = owner[token_idx * top_k + k];
            int target_offset = dest_offset[token_idx * top_k + k];
            float weight = routing_weights[token_idx * top_k + k];

            const float* src_ptr = (const float*)peer_expert_out_ptrs[target_rank];
            float val = src_ptr[target_offset * hidden_dim + i];
            accum += val * weight;
        }
        out[token_idx * hidden_dim + i] = accum;
    }
}

template <>
__global__ void pull_tokens_kernel<__nv_bfloat16>(
    const long long* __restrict__ peer_expert_out_ptrs,
    const int* __restrict__ owner,
    const int* __restrict__ dest_offset,
    const float* __restrict__ routing_weights,
    __nv_bfloat16* __restrict__ out,
    int hidden_dim,
    int top_k
) {
    int token_idx = blockIdx.x;

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float accum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int target_rank = owner[token_idx * top_k + k];
            int target_offset = dest_offset[token_idx * top_k + k];
            float weight = routing_weights[token_idx * top_k + k];

            const __nv_bfloat16* src_ptr = (const __nv_bfloat16*)peer_expert_out_ptrs[target_rank];
            float val = __bfloat162float(src_ptr[target_offset * hidden_dim + i]);
            accum += val * weight;
        }
        out[token_idx * hidden_dim + i] = __float2bfloat16(accum);
    }
}

void launch_all_gather_counts(
    torch::Tensor local_counts,
    torch::Tensor peer_ptrs,
    int rank,
    int num_experts,
    int world_size
) {
    int threads = std::min(num_experts, 1024);
    int blocks = (num_experts + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    all_gather_counts_kernel<<<blocks, threads, 0, stream>>>(
        local_counts.data_ptr<int>(),
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        rank,
        num_experts,
        world_size
    );
}

void launch_dispatch_tokens(
    torch::Tensor hidden_states,
    torch::Tensor peer_tokens_ptrs,
    torch::Tensor owner,
    torch::Tensor dest_offset,
    int top_k
) {
    int num_tokens = hidden_states.size(0);
    int hidden_dim = hidden_states.size(1);
    
    dim3 grid(num_tokens, top_k);
    int threads = std::min(hidden_dim, 1024);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (hidden_states.dtype() == torch::kBFloat16) {
        dispatch_tokens_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
            (__nv_bfloat16*)hidden_states.data_ptr<at::BFloat16>(),
            (const long long*)peer_tokens_ptrs.data_ptr<int64_t>(),
            owner.data_ptr<int>(),
            dest_offset.data_ptr<int>(),
            hidden_dim,
            top_k
        );
    } else {
        dispatch_tokens_kernel<float><<<grid, threads, 0, stream>>>(
            hidden_states.data_ptr<float>(),
            (const long long*)peer_tokens_ptrs.data_ptr<int64_t>(),
            owner.data_ptr<int>(),
            dest_offset.data_ptr<int>(),
            hidden_dim,
            top_k
        );
    }
}

void launch_pull_tokens(
    torch::Tensor peer_expert_out_ptrs,
    torch::Tensor owner,
    torch::Tensor dest_offset,
    torch::Tensor routing_weights,
    torch::Tensor out,
    int top_k
) {
    int num_tokens = out.size(0);
    int hidden_dim = out.size(1);
    
    int threads = std::min(hidden_dim, 1024);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (out.dtype() == torch::kBFloat16) {
        pull_tokens_kernel<__nv_bfloat16><<<num_tokens, threads, 0, stream>>>(
            (const long long*)peer_expert_out_ptrs.data_ptr<int64_t>(),
            owner.data_ptr<int>(),
            dest_offset.data_ptr<int>(),
            routing_weights.data_ptr<float>(),
            (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
            hidden_dim,
            top_k
        );
    } else {
        pull_tokens_kernel<float><<<num_tokens, threads, 0, stream>>>(
            (const long long*)peer_expert_out_ptrs.data_ptr<int64_t>(),
            owner.data_ptr<int>(),
            dest_offset.data_ptr<int>(),
            routing_weights.data_ptr<float>(),
            out.data_ptr<float>(),
            hidden_dim,
            top_k
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_all_gather_counts", &launch_all_gather_counts);
    m.def("launch_dispatch_tokens", &launch_dispatch_tokens);
    m.def("launch_pull_tokens", &launch_pull_tokens);
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_moe_lora_uva", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(world_size, num_experts, max_tokens_per_rank, hidden_dim, device, dtype):
    key = (world_size, num_experts, max_tokens_per_rank, hidden_dim, device, dtype)
    if key in _symm_cache:
        return _symm_cache[key]

    counts_buf = symm_mem.empty((world_size, num_experts), dtype=torch.int32, device=device)
    counts_hdl = symm_mem.rendezvous(counts_buf, dist.group.WORLD)
    counts_ptrs = torch.tensor(counts_hdl.buffer_ptrs, dtype=torch.int64, device=device)

    tokens_buf = symm_mem.empty((max_tokens_per_rank, hidden_dim), dtype=dtype, device=device)
    tokens_hdl = symm_mem.rendezvous(tokens_buf, dist.group.WORLD)
    tokens_ptrs = torch.tensor(tokens_hdl.buffer_ptrs, dtype=torch.int64, device=device)

    expert_out_buf = symm_mem.empty((max_tokens_per_rank, hidden_dim), dtype=dtype, device=device)
    expert_out_hdl = symm_mem.rendezvous(expert_out_buf, dist.group.WORLD)
    expert_out_ptrs = torch.tensor(expert_out_hdl.buffer_ptrs, dtype=torch.int64, device=device)

    res = (counts_buf, counts_hdl, counts_ptrs,
           tokens_buf, tokens_hdl, tokens_ptrs,
           expert_out_buf, expert_out_hdl, expert_out_ptrs)
    _symm_cache[key] = res
    return res

def expert_forward_lora(
    x: torch.Tensor,
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    lora_gate_A: torch.Tensor,
    lora_gate_B: torch.Tensor,
    lora_up_A: torch.Tensor,
    lora_up_B: torch.Tensor,
    lora_down_A: torch.Tensor,
    lora_down_B: torch.Tensor,
) -> torch.Tensor:
    F = torch.nn.functional
    xa_g = F.linear(x, lora_gate_A)
    gate_x = gate_proj(x) + F.linear(xa_g, lora_gate_B)
    gate = F.silu(gate_x)
    xa_u = F.linear(x, lora_up_A)
    up = up_proj(x) + F.linear(xa_u, lora_up_B)
    y = gate * up
    xa_d = F.linear(y, lora_down_A)
    return down_proj(y) + F.linear(xa_d, lora_down_B)

@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    lora_gate_A: torch.Tensor,
    lora_gate_B: torch.Tensor,
    lora_up_A: torch.Tensor,
    lora_up_B: torch.Tensor,
    lora_down_A: torch.Tensor,
    lora_down_B: torch.Tensor,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = hidden_states.device
    dtype = hidden_states.dtype
    ext = _get_ext()
    
    hidden_dim = hidden_states.size(-1)
    org_shape = hidden_states.shape
    hidden_states_flat = hidden_states.reshape(-1, hidden_dim).contiguous()
    num_tokens = hidden_states_flat.size(0)
    
    num_local_experts = num_experts // world_size
    max_tokens_per_rank = max(num_tokens * top_k * world_size, 65536)
    
    # Initialize P2P routing symmetric allocations
    (counts_buf, counts_hdl, counts_ptrs, 
     tokens_buf, tokens_hdl, tokens_ptrs, 
     expert_out_buf, expert_out_hdl, expert_out_ptrs) = _get_symm_state(
        world_size, num_experts, max_tokens_per_rank, hidden_dim, device, dtype
    )
    
    # 1. Routing
    router_logits = torch.nn.functional.linear(hidden_states_flat, gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1)
    
    # 2. Histogram & Global Scatter (Custom AllGather)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).sum(dim=1)
    local_counts = expert_mask.sum(dim=0).to(torch.int32)
    local_idx_matrix = expert_mask.cumsum(dim=0, dtype=torch.int32) - 1
    
    counts_hdl.barrier(channel=0)
    ext.launch_all_gather_counts(local_counts, counts_ptrs, rank, num_experts, world_size)
    counts_hdl.barrier(channel=0)
    
    # 3. Offsets Setup
    my_experts_counts = counts_buf[:, rank * num_local_experts : (rank + 1) * num_local_experts]
    total_tokens_per_my_expert = my_experts_counts.sum(dim=0)
    total_my_tokens = total_tokens_per_my_expert.sum().item()
    
    reshaped_total = counts_buf.sum(dim=0).view(world_size, num_local_experts)
    expert_base_global = torch.cat([
        torch.zeros((world_size, 1), dtype=torch.int32, device=device),
        reshaped_total[:, :-1].cumsum(dim=1)
    ], dim=1)
    
    sender_offset = torch.cat([
        torch.zeros((1, num_experts), dtype=torch.int32, device=device),
        counts_buf[:-1, :].cumsum(dim=0)
    ], dim=0)
    
    owner = (selected_experts // num_local_experts).to(torch.int32)
    le = (selected_experts % num_local_experts).to(torch.int32)
    
    expert_base_for_selected = expert_base_global[owner, le]
    sender_offset_for_selected = sender_offset[rank, selected_experts]
    local_idx_for_selected = local_idx_matrix.gather(1, selected_experts)
    dest_offset = expert_base_for_selected + sender_offset_for_selected + local_idx_for_selected
    
    # 4. UAV P2P Direct Dispatch 
    tokens_hdl.barrier(channel=0)
    ext.launch_dispatch_tokens(hidden_states_flat, tokens_ptrs, owner, dest_offset, top_k)
    tokens_hdl.barrier(channel=0)
    
    # 5. Shared Expert Computations (Fused LoRA block)
    expert_out_hdl.barrier(channel=0)
    if total_my_tokens > 0:
        expert_out = expert_forward_lora(
            tokens_buf[:total_my_tokens],
            gate_proj, up_proj, down_proj,
            lora_gate_A, lora_gate_B, lora_up_A, lora_up_B, lora_down_A, lora_down_B
        )
        expert_out_buf[:total_my_tokens].copy_(expert_out)
    expert_out_hdl.barrier(channel=0)
    
    # 6. P2P Direct Pull & Accumulate Result
    out_flat = torch.empty_like(hidden_states_flat)
    ext.launch_pull_tokens(expert_out_ptrs, owner, dest_offset, routing_weights.float().contiguous(), out_flat, top_k)
    expert_out_hdl.barrier(channel=0)
    
    return out_flat.reshape(org_shape)