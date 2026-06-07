import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import List, Optional, Tuple, Union
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// 1. Gather counts kernel
__global__ void gather_counts_kernel(
    const uintptr_t* peer_ptrs,
    int32_t* gathered_counts,
    int world_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < world_size * world_size) {
        int r = idx / world_size;
        int c = idx % world_size;
        const int32_t* peer_counts = reinterpret_cast<const int32_t*>(peer_ptrs[r]);
        gathered_counts[idx] = peer_counts[c];
    }
}

void gather_counts_cuda(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor gathered_counts,
    int world_size
) {
    const uintptr_t* peer_ptrs = reinterpret_cast<const uintptr_t*>(peer_ptrs_tensor.data_ptr<int64_t>());
    int32_t* out = gathered_counts.data_ptr<int32_t>();
    int threads = 64;
    int blocks = (world_size * world_size + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_counts_kernel<<<blocks, threads, 0, stream>>>(peer_ptrs, out, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 2. Direct P2P Push Kernel
template <typename scalar_t>
__global__ void p2p_push_kernel(
    const scalar_t* __restrict__ send_buf,
    const int32_t* __restrict__ send_counts,
    const int32_t* __restrict__ send_offsets,
    const int32_t* __restrict__ dest_offsets,
    const uintptr_t* __restrict__ peer_recv_ptrs,
    int hidden_dim,
    int world_size
) {
    int dest_rank = blockIdx.y;
    int count = send_counts[dest_rank];
    if (count == 0) return;

    int send_offset_elem = send_offsets[dest_rank] * hidden_dim;
    int dest_offset_elem = dest_offsets[dest_rank] * hidden_dim;
    
    scalar_t* dest_buf = reinterpret_cast<scalar_t*>(peer_recv_ptrs[dest_rank]);
    
    int total_elements = count * hidden_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    constexpr int ElemsPerVec = 16 / sizeof(scalar_t);
    // Vectorize if alignment permits
    if (hidden_dim % ElemsPerVec == 0) {
        int total_vec = total_elements / ElemsPerVec;
        const ulong2* vec_send = reinterpret_cast<const ulong2*>(send_buf + send_offset_elem);
        ulong2* vec_dest = reinterpret_cast<ulong2*>(dest_buf + dest_offset_elem);
        for (int i = tid; i < total_vec; i += stride) {
            vec_dest[i] = vec_send[i];
        }
    } else {
        for (int i = tid; i < total_elements; i += stride) {
            dest_buf[dest_offset_elem + i] = send_buf[send_offset_elem + i];
        }
    }
}

void p2p_push_cuda(
    torch::Tensor send_buf,
    torch::Tensor send_counts,
    torch::Tensor send_offsets,
    torch::Tensor dest_offsets,
    torch::Tensor peer_recv_ptrs_tensor,
    int hidden_dim,
    int world_size
) {
    const int32_t* sc = send_counts.data_ptr<int32_t>();
    const int32_t* so = send_offsets.data_ptr<int32_t>();
    const int32_t* doff = dest_offsets.data_ptr<int32_t>();
    const uintptr_t* ptrs = reinterpret_cast<const uintptr_t*>(peer_recv_ptrs_tensor.data_ptr<int64_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 blocks(16, world_size);
    dim3 threads(256);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, send_buf.scalar_type(), "p2p_push", [&] {
        p2p_push_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            send_buf.data_ptr<scalar_t>(),
            sc, so, doff, ptrs,
            hidden_dim, world_size
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_counts_cuda", &gather_counts_cuda, "Gather local counts");
    m.def("p2p_push_cuda", &p2p_push_cuda, "P2P push via UVA");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_uva_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def get_symm_buffer(name: str, min_elements: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    if name in _symm_cache:
        buf, hdl = _symm_cache[name]
        if buf.numel() >= min_elements:
            return buf, hdl
            
    capacity = max(min_elements, 1024)
    capacity = 1 << (capacity - 1).bit_length()  # Round up to power of 2
    
    buf = symm_mem.empty(capacity, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group=group)
    _symm_cache[name] = (buf, hdl)
    return buf, hdl

class UVAAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, send_counts, send_offsets, dest_offsets, 
                recv_counts, recv_offsets, bwd_dest_offsets,
                fwd_ptrs, bwd_ptrs, fwd_hdl, bwd_hdl,
                fwd_buf, bwd_buf, total_recv_tokens, hidden_dim, world_size):
        
        ctx.save_for_backward(recv_counts, recv_offsets, bwd_dest_offsets, 
                              send_counts, send_offsets, dest_offsets)
        ctx.bwd_ptrs = bwd_ptrs
        ctx.bwd_hdl = bwd_hdl
        ctx.bwd_buf = bwd_buf
        ctx.hidden_dim = hidden_dim
        ctx.world_size = world_size
        ctx.total_send_tokens = input_tensor.size(0)

        input_tensor = input_tensor.contiguous()

        # Synchronize and direct UVA push
        fwd_hdl.barrier(channel=0)
        _get_ext().p2p_push_cuda(
            input_tensor, send_counts, send_offsets, dest_offsets, 
            fwd_ptrs, hidden_dim, world_size
        )
        fwd_hdl.barrier(channel=0)
        
        return fwd_buf[:total_recv_tokens * hidden_dim].view(total_recv_tokens, hidden_dim)

    @staticmethod
    def backward(ctx, grad_output):
        recv_counts, recv_offsets, bwd_dest_offsets, send_counts, send_offsets, dest_offsets = ctx.saved_tensors
        bwd_ptrs = ctx.bwd_ptrs
        bwd_hdl = ctx.bwd_hdl
        bwd_buf = ctx.bwd_buf
        hidden_dim = ctx.hidden_dim
        world_size = ctx.world_size
        total_send_tokens = ctx.total_send_tokens

        grad_output = grad_output.contiguous()

        # Execute identical reverse push logic for gradients!
        bwd_hdl.barrier(channel=0)
        _get_ext().p2p_push_cuda(
            grad_output, recv_counts, recv_offsets, bwd_dest_offsets,
            bwd_ptrs, hidden_dim, world_size
        )
        bwd_hdl.barrier(channel=0)

        return bwd_buf[:total_send_tokens * hidden_dim].view(total_send_tokens, hidden_dim), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


# ----- Support Utils -----

def _permute(tokens: torch.Tensor, routing_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices

def _generate_weights_idx(routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros((num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device)
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx

def _unpermute(tokens: torch.Tensor, routing_weights: torch.Tensor, hidden_states_shape: torch.Size, permutation_mapping: torch.Tensor, routing_map: torch.Tensor) -> torch.Tensor:
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    return unpermuted_tokens

def expert_forward(x: torch.Tensor, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear, down_proj: torch.nn.Linear) -> torch.Tensor:
    gate = torch.nn.functional.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


# ----- Primary Solution Module -----

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
    device = hidden_states.device
    hidden_dim = hidden_states.size(-1)
    
    if rank == 0:
        _get_ext()
    dist.barrier(group)
    _get_ext()
    
    # 1. Router
    router_logits = torch.nn.functional.linear(hidden_states.reshape(-1, hidden_dim), gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # 2. UVA Count Exchange
    local_counts = expert_mask.sum(dim=(1, 2)).to(torch.int32)
    counts_buf, counts_hdl = get_symm_buffer("counts", world_size, torch.int32, device, group)
    counts_buf[:world_size].copy_(local_counts)
    counts_hdl.barrier(channel=0)
    
    counts_ptrs_tensor = torch.tensor(counts_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    gathered_counts = torch.empty((world_size, world_size), dtype=torch.int32, device=device)
    _get_ext().gather_counts_cuda(counts_ptrs_tensor, gathered_counts, world_size)
    M = gathered_counts # M[r, c]: Tokens flowing from Rank r to Rank c
    
    # 3. Formulate offsets mapping
    send_counts_pre = M[rank, :]
    recv_counts_pre = M[:, rank]
    send_offsets_pre = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), send_counts_pre[:-1].cumsum(0, dtype=torch.int32)])
    recv_offsets_pre = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), recv_counts_pre[:-1].cumsum(0, dtype=torch.int32)])

    dest_offsets_pre = torch.zeros(world_size, dtype=torch.int32, device=device)
    bwd_dest_offsets_pre = torch.zeros(world_size, dtype=torch.int32, device=device)
    for j in range(world_size):
        dest_offsets_pre[j] = M[:rank, j].sum()
        bwd_dest_offsets_pre[j] = M[j, :rank].sum()
        
    send_counts_post = M[:, rank]
    recv_counts_post = M[rank, :]
    send_offsets_post = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), send_counts_post[:-1].cumsum(0, dtype=torch.int32)])
    recv_offsets_post = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), recv_counts_post[:-1].cumsum(0, dtype=torch.int32)])
    
    # Symmetry property of reverse route
    dest_offsets_post = bwd_dest_offsets_pre
    bwd_dest_offsets_post = dest_offsets_pre
    
    # 4. Prepare cached buffers scaling to dynamic capacity
    max_pre_recv = M.sum(dim=0).max().item()
    max_post_recv = M.sum(dim=1).max().item()
    
    buf_pre_fwd, hdl_pre_fwd = get_symm_buffer("pre_fwd", max_pre_recv * hidden_dim, hidden_states.dtype, device, group)
    buf_pre_bwd, hdl_pre_bwd = get_symm_buffer("pre_bwd", max_post_recv * hidden_dim, hidden_states.dtype, device, group)
    buf_post_fwd, hdl_post_fwd = get_symm_buffer("post_fwd", max_post_recv * hidden_dim, hidden_states.dtype, device, group)
    buf_post_bwd, hdl_post_bwd = get_symm_buffer("post_bwd", max_pre_recv * hidden_dim, hidden_states.dtype, device, group)
    
    ptrs_pre_fwd = torch.tensor(hdl_pre_fwd.buffer_ptrs, dtype=torch.int64, device=device)
    ptrs_pre_bwd = torch.tensor(hdl_pre_bwd.buffer_ptrs, dtype=torch.int64, device=device)
    ptrs_post_fwd = torch.tensor(hdl_post_fwd.buffer_ptrs, dtype=torch.int64, device=device)
    ptrs_post_bwd = torch.tensor(hdl_post_bwd.buffer_ptrs, dtype=torch.int64, device=device)
    
    # 5. Permute local tokens by destination
    routing_map = expert_mask.sum(dim=1)
    local_permuted_hidden_states, local_input_permutation_mapping = _permute(hidden_states.reshape(-1, hidden_dim), routing_map)
    
    # 6. Pre-MLP AllToAll (Forward scatter to peers)
    global_permuted_hidden_states = UVAAllToAll.apply(
        local_permuted_hidden_states, send_counts_pre, send_offsets_pre, dest_offsets_pre,
        recv_counts_pre, recv_offsets_pre, bwd_dest_offsets_pre,
        ptrs_pre_fwd, ptrs_pre_bwd, hdl_pre_fwd, hdl_pre_bwd,
        buf_pre_fwd, buf_pre_bwd, recv_counts_pre.sum().item(), hidden_dim, world_size
    )
    
    # 7. Expert Sub-Network Processing
    expert_outputs = expert_forward(global_permuted_hidden_states, gate_proj, up_proj, down_proj)
    
    # 8. Post-MLP AllToAll (Backward scatter outputs back to origins)
    unpermute_outputs = UVAAllToAll.apply(
        expert_outputs, send_counts_post, send_offsets_post, dest_offsets_post,
        recv_counts_post, recv_offsets_post, bwd_dest_offsets_post,
        ptrs_post_fwd, ptrs_post_bwd, hdl_post_fwd, hdl_post_bwd,
        buf_post_fwd, buf_post_bwd, recv_counts_post.sum().item(), hidden_dim, world_size
    )
    
    # 9. Local unpermute
    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    out = _unpermute(
        unpermute_outputs, weights_idx, hidden_states.shape, 
        local_input_permutation_mapping, routing_map
    )
    
    return out