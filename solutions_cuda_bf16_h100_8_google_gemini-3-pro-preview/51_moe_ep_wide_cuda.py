"""
Strategy:
1. Replace NCCL AllToAll and AllGather with custom UVA memory movement kernels over symmetric memory buffers.
2. Directly compute the global layout offsets for each expert. This allows ranks to scatter tokens *directly* to their sorted, final destination in peers' memory, completely removing the need for intermediary sorting (e.g., `_sort_chunks_by_idxs`).
3. Encapsulate the scatter/gather data movements in custom `torch.autograd.Function`s (`MoEScatter`, `MoEGather`), ensuring smooth and direct gradient propagation using identical reverse UVA paths.
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
#include <cstdint>
#include <algorithm>

__global__ void push_chunks_kernel(
    const int8_t* __restrict__ src,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_ptrs,
    const int* __restrict__ chunk_sizes,
    int hidden_dim_bytes,
    int num_chunks,
    int blocks_per_chunk
) {
    int chunk_idx = blockIdx.x / blocks_per_chunk;
    int block_offset = blockIdx.x % blocks_per_chunk;
    if (chunk_idx >= num_chunks) return;
    
    int size = chunk_sizes[chunk_idx];
    if (size == 0) return;
    
    const int8_t* src_chunk = src + src_offsets[chunk_idx] * hidden_dim_bytes;
    int8_t* dst_chunk = reinterpret_cast<int8_t*>(dst_ptrs[chunk_idx]);
    
    int total_bytes = size * hidden_dim_bytes;
    int total_vec = total_bytes / 16;
    
    const float4* src_vec = reinterpret_cast<const float4*>(src_chunk);
    float4* dst_vec = reinterpret_cast<float4*>(dst_chunk);
    
    for (int i = block_offset * blockDim.x + threadIdx.x; i < total_vec; i += blocks_per_chunk * blockDim.x) {
        dst_vec[i] = src_vec[i];
    }
    
    if (block_offset == 0 && threadIdx.x == 0) {
        for(int i = total_vec * 16; i < total_bytes; ++i) {
            dst_chunk[i] = src_chunk[i];
        }
    }
}

__global__ void gather_counts_kernel(
    const int64_t* __restrict__ peer_ptrs,
    int* __restrict__ out,
    int world_size,
    int num_experts
) {
    int r = blockIdx.x;
    if (r >= world_size) return;
    const int* peer_count = reinterpret_cast<const int*>(peer_ptrs[r]);
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        out[r * num_experts + e] = peer_count[e];
    }
}

void launch_push_chunks(
    torch::Tensor src,
    torch::Tensor src_offsets,
    torch::Tensor dst_ptrs,
    torch::Tensor chunk_sizes,
    int hidden_dim_bytes
) {
    int num_chunks = chunk_sizes.size(0);
    int blocks_per_chunk = 16;
    int total_blocks = num_chunks * blocks_per_chunk;
    int threads = 256;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    push_chunks_kernel<<<total_blocks, threads, 0, stream>>>(
        reinterpret_cast<const int8_t*>(src.data_ptr()),
        src_offsets.data_ptr<int64_t>(),
        dst_ptrs.data_ptr<int64_t>(),
        chunk_sizes.data_ptr<int>(),
        hidden_dim_bytes,
        num_chunks,
        blocks_per_chunk
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_counts(
    torch::Tensor peer_ptrs,
    torch::Tensor out,
    int world_size,
    int num_experts
) {
    int threads = std::min(num_experts, 1024);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_counts_kernel<<<world_size, threads, 0, stream>>>(
        peer_ptrs.data_ptr<int64_t>(),
        out.data_ptr<int>(),
        world_size,
        num_experts
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_push_chunks", &launch_push_chunks, "Push chunks via UVA");
    m.def("launch_gather_counts", &launch_gather_counts, "Gather counts via UVA");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_uva_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_buffers(device, hidden_dim, world_size, num_experts, dtype):
    key = (device, dtype)
    if key in _symm_cache:
        return _symm_cache[key]
        
    MAX_TOKENS = 16384  # Abundant safety buffer for MoE tokens
    
    buf_counts = symm_mem.empty((num_experts,), dtype=torch.int32, device=device)
    hdl_counts = symm_mem.rendezvous(buf_counts, dist.group.WORLD)
    
    buf_global_permuted = symm_mem.empty((MAX_TOKENS, hidden_dim), dtype=dtype, device=device)
    hdl_global_permuted = symm_mem.rendezvous(buf_global_permuted, dist.group.WORLD)
    
    buf_grad_local = symm_mem.empty((MAX_TOKENS, hidden_dim), dtype=dtype, device=device)
    hdl_grad_local = symm_mem.rendezvous(buf_grad_local, dist.group.WORLD)
    
    buf_unpermute = symm_mem.empty((MAX_TOKENS, hidden_dim), dtype=dtype, device=device)
    hdl_unpermute = symm_mem.rendezvous(buf_unpermute, dist.group.WORLD)
    
    buf_grad_expert = symm_mem.empty((MAX_TOKENS, hidden_dim), dtype=dtype, device=device)
    hdl_grad_expert = symm_mem.rendezvous(buf_grad_expert, dist.group.WORLD)
    
    res = {
        "buf_counts": buf_counts,
        "hdl_counts": hdl_counts,
        "buf_global_permuted": buf_global_permuted,
        "hdl_global_permuted": hdl_global_permuted,
        "buf_grad_local": buf_grad_local,
        "hdl_grad_local": hdl_grad_local,
        "buf_unpermute": buf_unpermute,
        "hdl_unpermute": hdl_unpermute,
        "buf_grad_expert": buf_grad_expert,
        "hdl_grad_expert": hdl_grad_expert,
    }
    _symm_cache[key] = res
    return res


# ----- UVA Offset Calculation Utils -----

def compute_forward_scatter_args(global_counts_cpu, rank, world_size, num_experts, hidden_dim, element_size, symm_ptrs_cpu):
    num_local_experts = num_experts // world_size
    src_offsets = torch.zeros(num_experts, dtype=torch.int64)
    dst_ptrs = torch.zeros(num_experts, dtype=torch.int64)
    
    gc = global_counts_cpu.tolist()
    current_src_offset = 0
    
    for e in range(num_experts):
        src_offsets[e] = current_src_offset
        current_src_offset += gc[rank][e]
        
        dest_rank = e // num_local_experts
        expert_base = 0
        for e_prime in range(dest_rank * num_local_experts, e):
            expert_base += sum(gc[r][e_prime] for r in range(world_size))
            
        rank_offset = sum(gc[r][e] for r in range(rank))
        write_ptr = expert_base + rank_offset
        dst_ptrs[e] = symm_ptrs_cpu[dest_rank] + write_ptr * hidden_dim * element_size
        
    return src_offsets, dst_ptrs

def compute_backward_scatter_args(global_counts_cpu, rank, world_size, num_experts, hidden_dim, element_size, symm_ptrs_cpu):
    num_local_experts = num_experts // world_size
    num_chunks = num_local_experts * world_size
    src_offsets = torch.zeros(num_chunks, dtype=torch.int64)
    dst_ptrs = torch.zeros(num_chunks, dtype=torch.int64)
    chunk_sizes = torch.zeros(num_chunks, dtype=torch.int32)
    
    gc = global_counts_cpu.tolist()
    current_src_offset = 0
    
    for local_e in range(num_local_experts):
        e = rank * num_local_experts + local_e
        for dest_rank in range(world_size):
            c = local_e * world_size + dest_rank
            size = gc[dest_rank][e]
            chunk_sizes[c] = size
            src_offsets[c] = current_src_offset
            current_src_offset += size
            offset = sum(gc[dest_rank][ep] for ep in range(e))
            dst_ptrs[c] = symm_ptrs_cpu[dest_rank] + offset * hidden_dim * element_size
            
    return src_offsets, dst_ptrs, chunk_sizes

def compute_forward_gather_args(global_counts_cpu, rank, world_size, num_experts, hidden_dim, element_size, symm_ptrs_cpu):
    num_local_experts = num_experts // world_size
    num_chunks = num_local_experts * world_size
    src_offsets = torch.zeros(num_chunks, dtype=torch.int64)
    dst_ptrs = torch.zeros(num_chunks, dtype=torch.int64)
    chunk_sizes = torch.zeros(num_chunks, dtype=torch.int32)
    
    gc = global_counts_cpu.tolist()
    current_src_offset = 0
    
    for local_e in range(num_local_experts):
        e = rank * num_local_experts + local_e
        for dest_rank in range(world_size):
            c = local_e * world_size + dest_rank
            size = gc[dest_rank][e]
            chunk_sizes[c] = size
            src_offsets[c] = current_src_offset
            current_src_offset += size
            write_ptr = sum(gc[dest_rank][ep] for ep in range(e))
            dst_ptrs[c] = symm_ptrs_cpu[dest_rank] + write_ptr * hidden_dim * element_size
            
    return src_offsets, dst_ptrs, chunk_sizes

def compute_backward_gather_args(global_counts_cpu, rank, world_size, num_experts, hidden_dim, element_size, symm_ptrs_cpu):
    num_local_experts = num_experts // world_size
    src_offsets = torch.zeros(num_experts, dtype=torch.int64)
    dst_ptrs = torch.zeros(num_experts, dtype=torch.int64)
    chunk_sizes = torch.zeros(num_experts, dtype=torch.int32)
    
    gc = global_counts_cpu.tolist()
    current_src_offset = 0
    
    for e in range(num_experts):
        size = gc[rank][e]
        chunk_sizes[e] = size
        src_offsets[e] = current_src_offset
        current_src_offset += size
        
        dest_rank = e // num_local_experts
        local_e = e % num_local_experts
        
        offset = 0
        for le in range(local_e):
            ep = dest_rank * num_local_experts + le
            offset += sum(gc[r][ep] for r in range(world_size))
            
        offset += sum(gc[r][e] for r in range(rank))
        dst_ptrs[e] = symm_ptrs_cpu[dest_rank] + offset * hidden_dim * element_size
        
    return src_offsets, dst_ptrs, chunk_sizes


# ----- Autograd Collectives -----

class MoEScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_permuted, global_counts_cpu, rank, world_size, num_experts, hidden_dim, element_size, symm_ptrs_cpu, group, symm_grad_local_ptrs_cpu):
        src_offsets, dst_ptrs = compute_forward_scatter_args(
            global_counts_cpu, rank, world_size, num_experts, hidden_dim, element_size, symm_ptrs_cpu)
        chunk_sizes = global_counts_cpu[rank]
        
        dist.barrier(group=group)
        _get_ext().launch_push_chunks(
            local_permuted, 
            src_offsets.to(local_permuted.device),
            dst_ptrs.to(local_permuted.device),
            chunk_sizes.to(local_permuted.device),
            hidden_dim * element_size
        )
        dist.barrier(group=group)
        
        num_local_experts = num_experts // world_size
        total_received = global_counts_cpu[:, rank * num_local_experts : (rank + 1) * num_local_experts].sum().item()
        
        ctx.save_for_backward(global_counts_cpu)
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.num_experts = num_experts
        ctx.hidden_dim = hidden_dim
        ctx.element_size = element_size
        ctx.symm_grad_local_ptrs_cpu = symm_grad_local_ptrs_cpu
        ctx.group = group
        ctx.dtype = local_permuted.dtype
        
        symm_global_permuted = _get_symm_buffers(local_permuted.device, hidden_dim, world_size, num_experts, local_permuted.dtype)["buf_global_permuted"]
        return symm_global_permuted[:total_received].clone()

    @staticmethod
    def backward(ctx, grad_global_permuted):
        global_counts_cpu, = ctx.saved_tensors
        src_offsets, dst_ptrs, chunk_sizes = compute_backward_scatter_args(
            global_counts_cpu, ctx.rank, ctx.world_size, ctx.num_experts, ctx.hidden_dim, ctx.element_size, ctx.symm_grad_local_ptrs_cpu)
        
        dist.barrier(group=ctx.group)
        _get_ext().launch_push_chunks(
            grad_global_permuted.contiguous(),
            src_offsets.to(grad_global_permuted.device),
            dst_ptrs.to(grad_global_permuted.device),
            chunk_sizes.to(grad_global_permuted.device),
            ctx.hidden_dim * ctx.element_size
        )
        dist.barrier(group=ctx.group)
        
        symm_grad_local = _get_symm_buffers(grad_global_permuted.device, ctx.hidden_dim, ctx.world_size, ctx.num_experts, ctx.dtype)["buf_grad_local"]
        total_local = global_counts_cpu[ctx.rank].sum().item()
        return symm_grad_local[:total_local].clone(), None, None, None, None, None, None, None, None, None


class MoEGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, global_counts_cpu, rank, world_size, num_experts, hidden_dim, element_size, symm_ptrs_cpu, group, symm_grad_expert_ptrs_cpu):
        src_offsets, dst_ptrs, chunk_sizes = compute_forward_gather_args(
            global_counts_cpu, rank, world_size, num_experts, hidden_dim, element_size, symm_ptrs_cpu)
        
        dist.barrier(group=group)
        _get_ext().launch_push_chunks(
            expert_outputs.contiguous(),
            src_offsets.to(expert_outputs.device),
            dst_ptrs.to(expert_outputs.device),
            chunk_sizes.to(expert_outputs.device),
            hidden_dim * element_size
        )
        dist.barrier(group=group)
        
        total_received = global_counts_cpu[rank].sum().item()
        
        ctx.save_for_backward(global_counts_cpu)
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.num_experts = num_experts
        ctx.hidden_dim = hidden_dim
        ctx.element_size = element_size
        ctx.symm_grad_expert_ptrs_cpu = symm_grad_expert_ptrs_cpu
        ctx.group = group
        ctx.dtype = expert_outputs.dtype
        
        symm_unpermute = _get_symm_buffers(expert_outputs.device, hidden_dim, world_size, num_experts, expert_outputs.dtype)["buf_unpermute"]
        return symm_unpermute[:total_received].clone()

    @staticmethod
    def backward(ctx, grad_unpermute):
        global_counts_cpu, = ctx.saved_tensors
        src_offsets, dst_ptrs, chunk_sizes = compute_backward_gather_args(
            global_counts_cpu, ctx.rank, ctx.world_size, ctx.num_experts, ctx.hidden_dim, ctx.element_size, ctx.symm_grad_expert_ptrs_cpu)
        
        dist.barrier(group=ctx.group)
        _get_ext().launch_push_chunks(
            grad_unpermute.contiguous(),
            src_offsets.to(grad_unpermute.device),
            dst_ptrs.to(grad_unpermute.device),
            chunk_sizes.to(grad_unpermute.device),
            ctx.hidden_dim * ctx.element_size
        )
        dist.barrier(group=ctx.group)
        
        symm_grad_expert = _get_symm_buffers(grad_unpermute.device, ctx.hidden_dim, ctx.world_size, ctx.num_experts, ctx.dtype)["buf_grad_expert"]
        num_local_experts = ctx.num_experts // ctx.world_size
        total_local = global_counts_cpu[:, ctx.rank * num_local_experts : (ctx.rank + 1) * num_local_experts].sum().item()
        
        return symm_grad_expert[:total_local].clone(), None, None, None, None, None, None, None, None, None


def expert_forward(
    x: torch.Tensor,
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
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
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = hidden_states.device
    hidden_dim = hidden_states.size(-1)
    element_size = hidden_states.element_size()
    
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)

    # 1. Routing
    router_logits = torch.nn.functional.linear(
        hidden_states.reshape(-1, hidden_dim), gate_weight, gate_bias
    )
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)
    
    routing_map = expert_mask.sum(dim=1).bool()
    local_counts = routing_map.sum(dim=1).to(torch.int32)
    
    symm_res = _get_symm_buffers(device, hidden_dim, world_size, num_experts, hidden_states.dtype)
    
    # 2. Collect precise sizes with UVA
    symm_res["buf_counts"].copy_(local_counts)
    dist.barrier(group=group)
    
    global_counts = torch.empty((world_size, num_experts), dtype=torch.int32, device=device)
    peer_ptrs = torch.tensor(symm_res["hdl_counts"].buffer_ptrs, dtype=torch.int64, device=device)
    _get_ext().launch_gather_counts(peer_ptrs, global_counts, world_size, num_experts)
    global_counts_cpu = global_counts.cpu()
    
    # 3. Fast PyTorch-local permute
    num_tokens = hidden_states.size(0)
    token_indices = torch.arange(num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    local_permuted_hidden_states = hidden_states.index_select(0, sorted_indices)
    
    # 4. Forward Dispatch (UVA Scatter)
    global_permuted_hidden_states = MoEScatter.apply(
        local_permuted_hidden_states,
        global_counts_cpu,
        rank,
        world_size,
        num_experts,
        hidden_dim,
        element_size,
        symm_res["hdl_global_permuted"].buffer_ptrs,
        group,
        symm_res["hdl_grad_local"].buffer_ptrs
    )
    
    # 5. Local Expert
    expert_outputs = expert_forward(
        global_permuted_hidden_states, gate_proj, up_proj, down_proj
    )
    
    # 6. Gather (UVA Reverse Scatter)
    unpermute_outputs = MoEGather.apply(
        expert_outputs,
        global_counts_cpu,
        rank,
        world_size,
        num_experts,
        hidden_dim,
        element_size,
        symm_res["hdl_unpermute"].buffer_ptrs,
        group,
        symm_res["hdl_grad_expert"].buffer_ptrs
    )
    
    # 7. Unpermute via natively propagated Autograd hooks
    weights_idx = torch.zeros(
        (num_tokens, num_experts),
        dtype=routing_weights.dtype,
        device=device,
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    tokens_weight = weights_idx.T.contiguous().masked_select(routing_map)
    tokens = unpermute_outputs * tokens_weight.unsqueeze(-1)
    
    unpermuted_tokens = torch.zeros(
        hidden_states.shape, device=device, dtype=hidden_states.dtype
    )
    expanded_mapping = sorted_indices.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    
    return unpermuted_tokens.reshape(hidden_states.shape)