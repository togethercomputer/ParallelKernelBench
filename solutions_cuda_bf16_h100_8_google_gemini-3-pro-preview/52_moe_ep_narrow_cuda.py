"""
Strategy:
1. Replace NCCL `all_to_all` and PyTorch chunk sorting with a single fused CUDA operator. 
2. We compute token routing distributions (`G`) using a fast symmetric memory `all_gather` instead of `dist.all_gather_into_tensor`.
3. We implement a custom autograd function (`FusedP2PAllToAll`) that calculates chunk offsets exactly mapping the local tokens to their final grouped-by-expert destination positions in remote symmetric memory. 
4. A single vectorized UVA push kernel simultaneously sends data over NVLink AND sorts the chunks, replacing both `_all_to_all` and `_sort_chunks_by_idxs` in one step.
5. In the backward pass, the exact inverse push operation efficiently returns gradients to the source buffers, minimizing host-device syncs and maximizing bandwidth utilization.
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

__global__ void push_chunks_kernel_vec(
    const float4* __restrict__ local_data,
    const long long* __restrict__ remote_ptrs,
    const int* __restrict__ src_offsets, 
    const int* __restrict__ dst_offsets, 
    const int* __restrict__ dst_ranks,   
    int num_chunks,
    int vec_hidden_dim,
    int total_vecs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vecs) return;
    
    int token_idx = idx / vec_hidden_dim;
    int dim_idx = idx % vec_hidden_dim;
    
    int chunk = 0;
    // Linear search is fast since num_chunks is very small (e.g., <= 64)
    for (int i = 0; i < num_chunks; ++i) {
        if (token_idx >= src_offsets[i] && token_idx < src_offsets[i+1]) {
            chunk = i;
            break;
        }
    }
    
    int token_offset_in_chunk = token_idx - src_offsets[chunk];
    int dst_rank = dst_ranks[chunk];
    int dst_off = dst_offsets[chunk];
    
    float4* dst = (float4*)remote_ptrs[dst_rank];
    dst[(dst_off + token_offset_in_chunk) * vec_hidden_dim + dim_idx] = local_data[idx];
}

void push_chunks_vec(
    torch::Tensor local_data,
    torch::Tensor remote_ptrs,
    int64_t src_offsets_ptr,
    int64_t dst_offsets_ptr,
    int64_t dst_ranks_ptr,
    int num_chunks,
    int vec_hidden_dim,
    int total_vecs
) {
    int threads = 256;
    int blocks = (total_vecs + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const long long* ptrs = (const long long*)remote_ptrs.data_ptr<int64_t>();
    const int* src_off = reinterpret_cast<const int*>(src_offsets_ptr);
    const int* dst_off = reinterpret_cast<const int*>(dst_offsets_ptr);
    const int* dst_r = reinterpret_cast<const int*>(dst_ranks_ptr);
    
    push_chunks_kernel_vec<<<blocks, threads, 0, stream>>>(
        (const float4*)local_data.data_ptr(),
        ptrs,
        src_off, dst_off, dst_r,
        num_chunks, vec_hidden_dim, total_vecs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void symm_all_gather_kernel(
    const int* __restrict__ local_data,
    const long long* __restrict__ remote_ptrs,
    int rank,
    int ep_size,
    int num_experts
) {
    int tid = threadIdx.x;
    if (tid < num_experts) {
        int val = local_data[tid];
        for (int dst = 0; dst < ep_size; ++dst) {
            int* dst_ptr = (int*)remote_ptrs[dst];
            dst_ptr[rank * num_experts + tid] = val;
        }
    }
}

void symm_all_gather(
    torch::Tensor local_data,
    torch::Tensor remote_ptrs,
    int rank,
    int ep_size,
    int num_experts
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const long long* ptrs = (const long long*)remote_ptrs.data_ptr<int64_t>();
    
    symm_all_gather_kernel<<<1, 32, 0, stream>>>(
        local_data.data_ptr<int>(),
        ptrs,
        rank, ep_size, num_experts
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("push_chunks_vec", &push_chunks_vec, "UVA chunk copy and sort");
    m.def("symm_all_gather", &symm_all_gather, "UVA all gather for G");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_moe_symm_p2p", CUDA_SRC)
    return _ext


_EP_SUBGROUP_CACHE: dict[tuple[int, int], Union[None, list]] = {}
def _resolve_ep_group_for_narrow_moe(num_experts: int) -> dist.ProcessGroup:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    ws = dist.get_world_size()
    rank = dist.get_rank()
    key = (ws, num_experts)
    if key not in _EP_SUBGROUP_CACHE:
        if num_experts >= ws:
            _EP_SUBGROUP_CACHE[key] = None
        elif ws % num_experts != 0:
            raise ValueError(f"narrow EP requires world_size ({ws}) % num_experts ({num_experts}) == 0")
        else:
            groups: list = []
            for r in range(ws // num_experts):
                ranks = list(range(r * num_experts, (r + 1) * num_experts))
                groups.append(dist.new_group(ranks))
            _EP_SUBGROUP_CACHE[key] = groups
    entry = _EP_SUBGROUP_CACHE[key]
    if entry is None:
        return dist.group.WORLD
    return entry[rank // num_experts]


_SYMM_BUFS = {}
def get_symm_buf(ep_group, buffer_id, max_elements, dtype, device):
    global _SYMM_BUFS
    key = (ep_group, buffer_id, dtype, device)
    if key not in _SYMM_BUFS:
        buf = symm_mem.empty((max_elements,), dtype=dtype, device=device)
        hdl = symm_mem.rendezvous(buf, group=ep_group)
        ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
        _SYMM_BUFS[key] = (buf, hdl, ptrs)
    return _SYMM_BUFS[key]


def _preprocess_symm(selected_experts: torch.Tensor, num_experts: int, ep_group: dist.ProcessGroup):
    ep_size = dist.get_world_size(ep_group)
    rank = dist.get_rank(ep_group)
    device = selected_experts.device
    
    routing_map = torch.zeros((num_experts, selected_experts.size(0)), dtype=torch.bool, device=device)
    routing_map.scatter_(0, selected_experts.T, 1)
    num_local_tokens_per_expert = routing_map.sum(dim=1, dtype=torch.int32)
    
    buf, hdl, remote_ptrs = get_symm_buf(ep_group, 'G', ep_size * num_experts, torch.int32, device)
    
    hdl.barrier()
    _get_ext().symm_all_gather(num_local_tokens_per_expert, remote_ptrs, rank, ep_size, num_experts)
    hdl.barrier()
    
    G = buf[:ep_size * num_experts].view(ep_size, num_experts)
    G_cpu = G.cpu().tolist()
    
    return routing_map, G_cpu


class FusedP2PAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permuted_input, op_type, G_cpu, ep_group, num_experts):
        ctx.op_type = op_type
        ctx.G_cpu = G_cpu
        ctx.ep_group = ep_group
        ctx.num_experts = num_experts
        
        ep_size = dist.get_world_size(ep_group)
        rank = dist.get_rank(ep_group)
        num_local_experts = num_experts // ep_size
        E_start = rank * num_local_experts
        E_end = E_start + num_local_experts
        
        device = permuted_input.device
        hidden_dim = permuted_input.size(-1)
        
        if op_type == 1:
            total_send = permuted_input.size(0)
            total_recv = sum(G_cpu[r][e] for e in range(E_start, E_end) for r in range(ep_size))
            num_chunks = num_experts
            
            src_offsets = [0] * (num_chunks + 1)
            dst_offsets = [0] * num_chunks
            dst_ranks = [0] * num_chunks
            
            for e in range(num_experts):
                dst = e // num_local_experts
                dst_ranks[e] = dst
                src_offsets[e] = sum(G_cpu[rank][k] for k in range(e))
                base_e = sum(G_cpu[r][k] for k in range(dst * num_local_experts, e) for r in range(ep_size))
                rank_off = sum(G_cpu[r][e] for r in range(rank))
                dst_offsets[e] = base_e + rank_off
            src_offsets[num_experts] = total_send
            buf_id = 'pre_fwd'
            
        else:
            total_send = permuted_input.size(0)
            total_recv = sum(G_cpu[rank][e] for e in range(num_experts))
            num_chunks = ep_size * num_local_experts
            
            src_offsets = [0] * (num_chunks + 1)
            dst_offsets = [0] * num_chunks
            dst_ranks = [0] * num_chunks
            
            chunk_idx = 0
            cur_off = 0
            for e in range(E_start, E_end):
                for s in range(ep_size):
                    dst_ranks[chunk_idx] = s
                    src_offsets[chunk_idx] = cur_off
                    dst_offsets[chunk_idx] = sum(G_cpu[s][k] for k in range(e))
                    cur_off += G_cpu[s][e]
                    chunk_idx += 1
            src_offsets[num_chunks] = cur_off
            buf_id = 'post_fwd'
            
        ctx.total_recv_fwd = total_recv
        ctx.total_send_fwd = total_send
        
        max_tokens = 65536 
        buf, hdl, remote_ptrs = get_symm_buf(ep_group, buf_id, max_tokens * hidden_dim, permuted_input.dtype, device)
        
        hdl.barrier()
        
        offsets_tensor = torch.tensor(src_offsets + dst_offsets + dst_ranks, dtype=torch.int32, device=device)
        d_src_offsets = offsets_tensor.data_ptr()
        d_dst_offsets = d_src_offsets + len(src_offsets) * 4
        d_dst_ranks = d_dst_offsets + len(dst_offsets) * 4
        
        vec_hidden_dim = hidden_dim
        if hidden_dim % 8 == 0 and permuted_input.dtype == torch.bfloat16:
            vec_hidden_dim = hidden_dim // 8
        elif hidden_dim % 4 == 0 and permuted_input.dtype == torch.float32:
            vec_hidden_dim = hidden_dim // 4
            
        total_vecs = total_send * vec_hidden_dim
        if total_vecs > 0:
            _get_ext().push_chunks_vec(
                permuted_input.contiguous(), remote_ptrs,
                d_src_offsets, d_dst_offsets, d_dst_ranks,
                num_chunks, vec_hidden_dim, total_vecs
            )
            
        hdl.barrier()
        
        out = torch.empty((total_recv, hidden_dim), dtype=permuted_input.dtype, device=device)
        if total_recv > 0:
            out.copy_(buf[:total_recv * hidden_dim].view(total_recv, hidden_dim))
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        op_type = ctx.op_type
        G_cpu = ctx.G_cpu
        ep_group = ctx.ep_group
        num_experts = ctx.num_experts
        
        ep_size = dist.get_world_size(ep_group)
        rank = dist.get_rank(ep_group)
        num_local_experts = num_experts // ep_size
        E_start = rank * num_local_experts
        E_end = E_start + num_local_experts
        
        device = grad_output.device
        hidden_dim = grad_output.size(-1)
        
        total_send = grad_output.size(0) 
        total_recv = ctx.total_send_fwd
        
        if op_type == 1:
            num_chunks = ep_size * num_local_experts
            src_offsets = [0] * (num_chunks + 1)
            dst_offsets = [0] * num_chunks
            dst_ranks = [0] * num_chunks
            
            chunk_idx = 0
            cur_off = 0
            for e in range(E_start, E_end):
                for s in range(ep_size):
                    dst_ranks[chunk_idx] = s
                    src_offsets[chunk_idx] = cur_off
                    dst_offsets[chunk_idx] = sum(G_cpu[s][k] for k in range(e))
                    cur_off += G_cpu[s][e]
                    chunk_idx += 1
            src_offsets[num_chunks] = cur_off
            buf_id = 'pre_bwd'
        else:
            num_chunks = num_experts
            src_offsets = [0] * (num_chunks + 1)
            dst_offsets = [0] * num_chunks
            dst_ranks = [0] * num_chunks
            
            for e in range(num_experts):
                dst = e // num_local_experts
                dst_ranks[e] = dst
                src_offsets[e] = sum(G_cpu[rank][k] for k in range(e))
                base_e = sum(G_cpu[r][k] for k in range(dst * num_local_experts, e) for r in range(ep_size))
                rank_off = sum(G_cpu[r][e] for r in range(rank))
                dst_offsets[e] = base_e + rank_off
            src_offsets[num_experts] = total_send
            buf_id = 'post_bwd'
            
        max_tokens = 65536
        buf, hdl, remote_ptrs = get_symm_buf(ep_group, buf_id, max_tokens * hidden_dim, grad_output.dtype, device)
        
        hdl.barrier()
        
        offsets_tensor = torch.tensor(src_offsets + dst_offsets + dst_ranks, dtype=torch.int32, device=device)
        d_src_offsets = offsets_tensor.data_ptr()
        d_dst_offsets = d_src_offsets + len(src_offsets) * 4
        d_dst_ranks = d_dst_offsets + len(dst_offsets) * 4
        
        vec_hidden_dim = hidden_dim
        if hidden_dim % 8 == 0 and grad_output.dtype == torch.bfloat16:
            vec_hidden_dim = hidden_dim // 8
        elif hidden_dim % 4 == 0 and grad_output.dtype == torch.float32:
            vec_hidden_dim = hidden_dim // 4
            
        total_vecs = total_send * vec_hidden_dim
        if total_vecs > 0:
            _get_ext().push_chunks_vec(
                grad_output.contiguous(), remote_ptrs,
                d_src_offsets, d_dst_offsets, d_dst_ranks,
                num_chunks, vec_hidden_dim, total_vecs
            )
            
        hdl.barrier()
        
        grad_input = torch.empty((total_recv, hidden_dim), dtype=grad_output.dtype, device=device)
        if total_recv > 0:
            grad_input.copy_(buf[:total_recv * hidden_dim].view(total_recv, hidden_dim))
            
        return grad_input, None, None, None, None


def _permute(tokens: torch.Tensor, routing_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def _generate_weights_idx(routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros((num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device)
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def _unpermute(
    tokens: torch.Tensor, routing_weights: torch.Tensor, hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor, routing_map: torch.Tensor
) -> torch.Tensor:
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map)
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    return unpermuted_tokens


def token_pre_all2all(
    hidden_states: torch.Tensor, routing_map: torch.Tensor, G_cpu: List[List[int]], 
    group: dist.ProcessGroup, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    
    local_permuted_hidden_states, local_input_permutation_mapping = _permute(hidden_states, routing_map)
    
    global_permuted_hidden_states = FusedP2PAllToAll.apply(
        local_permuted_hidden_states, 1, G_cpu, group, num_experts
    )
    return global_permuted_hidden_states, local_input_permutation_mapping, org_hidden_states_shape


def tokens_post_all2all(
    expert_outputs: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor,
    num_experts: int, routing_map: torch.Tensor, local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size, G_cpu: List[List[int]], group: dist.ProcessGroup
) -> torch.Tensor:
    unpermute_outputs = FusedP2PAllToAll.apply(
        expert_outputs, 2, G_cpu, group, num_experts
    )
    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    out = _unpermute(
        unpermute_outputs, weights_idx, org_hidden_states_shape,
        local_input_permutation_mapping, routing_map
    )
    return out


def expert_forward(
    x: torch.Tensor, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear, down_proj: torch.nn.Linear
) -> torch.Tensor:
    gate = torch.nn.functional.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    One MoE forward with completely custom Fused UVA backend replacing all collectives.
    """
    if group is None:
        group = _resolve_ep_group_for_narrow_moe(num_experts)
        
    hidden_dim = hidden_states.size(-1)

    router_logits = torch.nn.functional.linear(
        hidden_states.reshape(-1, hidden_dim), gate_weight, gate_bias
    )
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    
    _get_ext() # Init JIT extension
    
    routing_map, G_cpu = _preprocess_symm(selected_experts, num_experts, group)
    
    global_permuted, local_input_permutation_mapping, org_shape = token_pre_all2all(
        hidden_states, routing_map, G_cpu, group, num_experts
    )
    
    expert_outputs = expert_forward(
        global_permuted, gate_proj, up_proj, down_proj
    )
    
    out = tokens_post_all2all(
        expert_outputs, routing_weights, selected_experts, num_experts,
        routing_map, local_input_permutation_mapping, org_shape,
        G_cpu, group
    )
    return out