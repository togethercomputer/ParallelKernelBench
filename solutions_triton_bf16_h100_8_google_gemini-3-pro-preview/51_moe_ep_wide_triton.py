"""
Strategy:
1. Device-side Communication: All NCCL AllToAll and AllGather collectives are replaced with custom UVA pull-based kernels using symmetric memory (`torch.distributed._symmetric_memory`). Tokens are pulled directly from peer memory across NVLink without host-driven blockings.
2. Compute-Communication Overlap: The metadata AllGather for expert routing counts is dispatched asynchronously on a dedicated CUDA communication stream. It overlaps directly with the intensive local token indexing and permutation operations (`_permute`), fully masking collective latency.
3. Local Offset Determinism: Because the expert token distributions are aggregated asynchronously beforehand, all multi-rank AllToAll read/write scatter offsets are resolved locally via a deterministic matrix prefix-sum—completely eliminating synchronous dynamic size-exchanges.
4. Extensible UVA Extension: A JIT-compiled C++ CUDA extension handles multi-GPU memory transfers via cooperative grid tiling. It supports both BF16 and FP32, seamlessly wrapped within an Autograd function to guarantee accurate gradient scattering during the backward pass.
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

template<typename T>
__global__ void uva_all_to_all_pull_kernel(
    const int64_t* __restrict__ remote_ptrs,
    const int64_t* __restrict__ read_offsets,
    const int64_t* __restrict__ write_offsets,
    const int64_t* __restrict__ chunk_sizes,
    T* __restrict__ out,
    int64_t hidden_dim,
    int world_size
) {
    int peer = blockIdx.y;
    int64_t tokens_to_copy = chunk_sizes[peer];
    if (tokens_to_copy == 0) return;

    int64_t read_start = read_offsets[peer] * hidden_dim;
    int64_t write_start = write_offsets[peer] * hidden_dim;
    int64_t total_elements = tokens_to_copy * hidden_dim;

    const T* remote_data = reinterpret_cast<const T*>(remote_ptrs[peer]);

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;

    for (int64_t i = tid; i < total_elements; i += stride) {
        out[write_start + i] = remote_data[read_start + i];
    }
}

template<typename T>
__global__ void uva_all_gather_pull_kernel(
    const int64_t* __restrict__ remote_ptrs,
    T* __restrict__ out,
    int64_t chunk_size_elements,
    int world_size
) {
    int peer = blockIdx.y;
    int64_t total_elements = chunk_size_elements;
    int64_t write_start = peer * chunk_size_elements;

    const T* remote_data = reinterpret_cast<const T*>(remote_ptrs[peer]);

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;

    for (int64_t i = tid; i < total_elements; i += stride) {
        out[write_start + i] = remote_data[i];
    }
}

void uva_all_to_all_pull(
    torch::Tensor remote_ptrs,
    torch::Tensor read_offsets,
    torch::Tensor write_offsets,
    torch::Tensor chunk_sizes,
    torch::Tensor out,
    int64_t hidden_dim,
    int world_size
) {
    const int threads = 256;
    const int blocks = 256; 
    dim3 grid(blocks, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (out.dtype() == torch::kFloat32) {
        uva_all_to_all_pull_kernel<float><<<grid, threads, 0, stream>>>(
            remote_ptrs.data_ptr<int64_t>(),
            read_offsets.data_ptr<int64_t>(),
            write_offsets.data_ptr<int64_t>(),
            chunk_sizes.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            hidden_dim,
            world_size
        );
    } else if (out.dtype() == torch::kBFloat16) {
        uva_all_to_all_pull_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
            remote_ptrs.data_ptr<int64_t>(),
            read_offsets.data_ptr<int64_t>(),
            write_offsets.data_ptr<int64_t>(),
            chunk_sizes.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            hidden_dim,
            world_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for uva_all_to_all_pull");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void uva_all_gather_pull(
    torch::Tensor remote_ptrs,
    torch::Tensor out,
    int64_t chunk_size_elements,
    int world_size
) {
    const int threads = 256;
    const int blocks = 1; 
    dim3 grid(blocks, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (out.dtype() == torch::kInt64) {
        uva_all_gather_pull_kernel<int64_t><<<grid, threads, 0, stream>>>(
            remote_ptrs.data_ptr<int64_t>(),
            out.data_ptr<int64_t>(),
            chunk_size_elements,
            world_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for uva_all_gather_pull");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_all_to_all_pull", &uva_all_to_all_pull, "UVA all-to-all pull");
    m.def("uva_all_gather_pull", &uva_all_gather_pull, "UVA all-gather pull");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_moe_comm_bf16", CUDA_SRC)
    return _ext

_named_symm_buffers = {}
def get_named_symm_buffer(name: str, size: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    if name in _named_symm_buffers:
        buf, hdl = _named_symm_buffers[name]
        if buf.numel() >= size:
            return buf, hdl
    
    # Pre-allocate extra space for variable token capacities gracefully
    alloc_size = max(size, 1024 * 1024)
    buf = symm_mem.empty(alloc_size, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _named_symm_buffers[name] = (buf, hdl)
    return buf, hdl

_comm_stream = None
def get_comm_stream():
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.cuda.Stream()
    return _comm_stream


class _UVAAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, read_offsets, write_offsets, chunk_sizes, hidden_dim, world_size, bwd_read_offsets, bwd_write_offsets, bwd_chunk_sizes, name, group):
        ctx.bwd_read_offsets = bwd_read_offsets
        ctx.bwd_write_offsets = bwd_write_offsets
        ctx.bwd_chunk_sizes = bwd_chunk_sizes
        ctx.hidden_dim = hidden_dim
        ctx.world_size = world_size
        ctx.name = name
        ctx.group = group
        
        input = input.contiguous()
        out_tokens = sum(chunk_sizes)
        out = torch.empty((out_tokens, hidden_dim), dtype=input.dtype, device=input.device)
        
        buf, hdl = get_named_symm_buffer(name, input.numel(), input.dtype, input.device, group)
        buf[:input.numel()].copy_(input.view(-1))
        hdl.barrier(channel=0)
        
        remote_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=input.device)
        read_off_t = torch.tensor(read_offsets, dtype=torch.int64, device=input.device)
        write_off_t = torch.tensor(write_offsets, dtype=torch.int64, device=input.device)
        chunk_sizes_t = torch.tensor(chunk_sizes, dtype=torch.int64, device=input.device)
        
        _get_ext().uva_all_to_all_pull(
            remote_ptrs, read_off_t, write_off_t, chunk_sizes_t, out, hidden_dim, world_size
        )
        # Prevents overwriting local symm memory while peers are still reading from it
        hdl.barrier(channel=0)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        out_tokens = sum(ctx.bwd_chunk_sizes)
        grad_input = torch.empty((out_tokens, ctx.hidden_dim), dtype=grad_output.dtype, device=grad_output.device)
        
        buf, hdl = get_named_symm_buffer(ctx.name + "_bwd", grad_output.numel(), grad_output.dtype, grad_output.device, ctx.group)
        buf[:grad_output.numel()].copy_(grad_output.view(-1))
        hdl.barrier(channel=0)
        
        remote_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=grad_output.device)
        read_off_t = torch.tensor(ctx.bwd_read_offsets, dtype=torch.int64, device=grad_output.device)
        write_off_t = torch.tensor(ctx.bwd_write_offsets, dtype=torch.int64, device=grad_output.device)
        chunk_sizes_t = torch.tensor(ctx.bwd_chunk_sizes, dtype=torch.int64, device=grad_output.device)
        
        _get_ext().uva_all_to_all_pull(
            remote_ptrs, read_off_t, write_off_t, chunk_sizes_t, grad_input, ctx.hidden_dim, ctx.world_size
        )
        hdl.barrier(channel=0)
        
        return grad_input, None, None, None, None, None, None, None, None, None, None


def _preprocess_start(expert_mask: torch.Tensor, num_experts: int, ep_group: dist.ProcessGroup):
    ep_size = ep_group.size()
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))
    num_local_tokens_per_expert_flat = num_local_tokens_per_expert.contiguous().view(-1)
    chunk_size = num_local_tokens_per_expert_flat.numel()
    
    out = torch.empty(ep_size * chunk_size, dtype=num_local_tokens_per_expert_flat.dtype, device=num_local_tokens_per_expert_flat.device)
    buf, hdl = get_named_symm_buffer("preprocess_gather", chunk_size, num_local_tokens_per_expert_flat.dtype, num_local_tokens_per_expert_flat.device, ep_group)
    
    buf[:chunk_size].copy_(num_local_tokens_per_expert_flat)
    hdl.barrier(channel=0)
    
    # Overlap symmetric memory AllGather onto a dedicated stream
    comm_stream = get_comm_stream()
    with torch.cuda.stream(comm_stream):
        remote_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=num_local_tokens_per_expert_flat.device)
        _get_ext().uva_all_gather_pull(remote_ptrs, out, chunk_size, ep_size)
    
    return hdl, comm_stream, out, num_local_tokens_per_expert


def _preprocess_wait(hdl, comm_stream, out, num_local_tokens_per_expert, num_experts, ep_group):
    # Wait for overlapping communication stream to complete the AllGather
    torch.cuda.current_stream().wait_stream(comm_stream)
    hdl.barrier(channel=0)
    
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    
    num_global_tokens_per_expert = out.view(ep_size, num_local_tokens_per_expert.size(0))
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    
    num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert.view(-1, num_local_experts).to(torch.device("cpu"), non_blocking=True)
    
    return input_splits, output_splits, num_global_tokens_per_local_expert_cpu, num_global_tokens_per_expert


def compute_all2all_offsets(num_global_tokens_per_expert: torch.Tensor, rank: int, world_size: int):
    num_experts = num_global_tokens_per_expert.size(1)
    num_local_experts = num_experts // world_size
    
    # Send matrix mapping [source_rank, dest_rank] sizes
    send_matrix = torch.zeros((world_size, world_size), dtype=torch.int64)
    for i in range(world_size):
        for j in range(world_size):
            send_matrix[i, j] = num_global_tokens_per_expert[i, j*num_local_experts : (j+1)*num_local_experts].sum()
            
    # Forward Offsets (Pre All-to-All)
    read_offsets_pre = []
    chunk_sizes_pre = []
    for i in range(world_size):
        read_off = send_matrix[i, :rank].sum().item()
        read_offsets_pre.append(read_off)
        chunk_sizes_pre.append(send_matrix[i, rank].item())
        
    write_offsets_pre = [0] * world_size
    curr = 0
    for i in range(world_size):
        write_offsets_pre[i] = curr
        curr += send_matrix[i, rank].item()
        
    # Backward/Reverse Offsets (Post All-to-All)
    read_offsets_post = []
    chunk_sizes_post = []
    for i in range(world_size):
        read_off = 0
        for k in range(rank):
            read_off += send_matrix[k, i].item()
        read_offsets_post.append(read_off)
        chunk_sizes_post.append(send_matrix[rank, i].item())
        
    write_offsets_post = [0] * world_size
    curr = 0
    for i in range(world_size):
        write_offsets_post[i] = curr
        curr += send_matrix[rank, i].item()
        
    return {
        "pre": (read_offsets_pre, write_offsets_pre, chunk_sizes_pre),
        "post": (read_offsets_post, write_offsets_post, chunk_sizes_post)
    }


def _permute(tokens: torch.Tensor, routing_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def _sort_chunks_by_idxs(
    input: torch.Tensor, split_sizes: Union[torch.Tensor, List[int]], sorted_idxs: List[int]
) -> torch.Tensor:
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()
    chunks = torch.split(input, split_sizes, dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)


def _generate_weights_idx(
    routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts: int
) -> torch.Tensor:
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros(
        (num_tokens, num_experts),
        dtype=routing_weights.dtype,
        device=routing_weights.device,
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def _unpermute(
    tokens: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
) -> torch.Tensor:
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    unpermuted_tokens = torch.zeros(
        hidden_states_shape, device=tokens.device, dtype=tokens.dtype
    )
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    return unpermuted_tokens


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
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    if rank == 0:
        _get_ext()
    dist.barrier(group)
    
    hidden_dim = hidden_states.size(-1)
    
    router_logits = torch.nn.functional.linear(
        hidden_states.reshape(-1, hidden_dim), gate_weight, gate_bias
    )
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)

    # Overlap Preprocess AllGather with local routing maps and layout permutations
    hdl, comm_stream, out_gather, num_local_tokens = _preprocess_start(expert_mask, num_experts, group)

    routing_map = expert_mask.sum(dim=1)
    local_permuted_hidden_states, local_input_permutation_mapping = _permute(
        hidden_states.reshape(-1, hidden_dim), routing_map
    )

    # Re-sync local thread logic over the completed metadata AllGather results
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_tokens_per_expert = _preprocess_wait(
        hdl, comm_stream, out_gather, num_local_tokens, num_experts, group
    )
    
    # Deriving read/write scattering offsets deterministically with strict zero-communication guarantees
    offsets = compute_all2all_offsets(num_global_tokens_per_expert, rank, world_size)

    global_permuted_hidden_states = _UVAAllToAll.apply(
        local_permuted_hidden_states,
        *offsets["pre"],
        hidden_dim,
        world_size,
        *offsets["post"],
        "all2all_pre",
        group
    )

    num_local_experts = num_experts // world_size
    permute_order = (
        torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    )
    split_sizes = num_global_tokens_per_local_expert.ravel().tolist()
    global_permuted_hidden_states = _sort_chunks_by_idxs(
        global_permuted_hidden_states, split_sizes, permute_order
    )

    expert_outputs = expert_forward(
        global_permuted_hidden_states, gate_proj, up_proj, down_proj
    )

    unpermute_order = (
        torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    )
    split_sizes_post = num_global_tokens_per_local_expert.T.ravel().tolist()
    expert_outputs = _sort_chunks_by_idxs(
        expert_outputs, split_sizes_post, unpermute_order
    )
    
    unpermute_outputs = _UVAAllToAll.apply(
        expert_outputs,
        *offsets["post"],
        hidden_dim,
        world_size,
        *offsets["pre"],
        "all2all_post",
        group
    )

    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    out = _unpermute(
        unpermute_outputs,
        weights_idx,
        hidden_states.shape,
        local_input_permutation_mapping,
        routing_map,
    )
    return out