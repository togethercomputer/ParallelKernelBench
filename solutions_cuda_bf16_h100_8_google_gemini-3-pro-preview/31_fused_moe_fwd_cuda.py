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

template <typename T>
__global__ void uva_push_kernel(
    const T* __restrict__ src_data,
    const int32_t* __restrict__ sorted_indices,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ src_offsets,
    const int32_t* __restrict__ dest_ranks,
    const int32_t* __restrict__ dest_offsets,
    const int64_t* __restrict__ dest_buf_ptrs,
    int num_chunks,
    int hidden_dim
) {
    int chunk_idx = blockIdx.y;
    if (chunk_idx >= num_chunks) return;
    
    int count = counts[chunk_idx];
    if (count == 0) return;
    
    int src_offset = src_offsets[chunk_idx];
    int dest_rank = dest_ranks[chunk_idx];
    int dest_offset = dest_offsets[chunk_idx];
    
    T* dest_buf = reinterpret_cast<T*>(dest_buf_ptrs[dest_rank]);
    
    int total_elements = count * hidden_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < total_elements; i += stride) {
        int token_idx = i / hidden_dim;
        int dim_idx = i % hidden_dim;
        
        int actual_src_token = sorted_indices ? sorted_indices[src_offset + token_idx] : (src_offset + token_idx);
        dest_buf[(dest_offset + token_idx) * hidden_dim + dim_idx] = src_data[actual_src_token * hidden_dim + dim_idx];
    }
}

void uva_push(
    torch::Tensor src_data,
    std::optional<torch::Tensor> sorted_indices,
    torch::Tensor counts,
    torch::Tensor src_offsets,
    torch::Tensor dest_ranks,
    torch::Tensor dest_offsets,
    torch::Tensor dest_buf_ptrs,
    int num_chunks,
    int hidden_dim
) {
    const int32_t* idxs = sorted_indices.has_value() ? sorted_indices.value().data_ptr<int32_t>() : nullptr;
    const int32_t* c = counts.data_ptr<int32_t>();
    const int32_t* s_off = src_offsets.data_ptr<int32_t>();
    const int32_t* d_ranks = dest_ranks.data_ptr<int32_t>();
    const int32_t* d_off = dest_offsets.data_ptr<int32_t>();
    const int64_t* d_ptrs = dest_buf_ptrs.data_ptr<int64_t>();
    
    int threads = 256;
    int blocks_x = 16; 
    dim3 blocks(blocks_x, num_chunks);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (src_data.dtype() == torch::kBFloat16) {
        uva_push_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src_data.data_ptr<at::BFloat16>()),
            idxs, c, s_off, d_ranks, d_off, d_ptrs, num_chunks, hidden_dim
        );
    } else if (src_data.dtype() == torch::kFloat32) {
        uva_push_kernel<float><<<blocks, threads, 0, stream>>>(
            src_data.data_ptr<float>(),
            idxs, c, s_off, d_ranks, d_off, d_ptrs, num_chunks, hidden_dim
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

__global__ void gather_N_kernel(
    int32_t* N_matrix,
    const int64_t* prep_ptrs,
    int num_experts,
    int world_size
) {
    int r = blockIdx.x;
    int e = threadIdx.x;
    if (r < world_size && e < num_experts) {
        int32_t* remote = reinterpret_cast<int32_t*>(prep_ptrs[r]);
        N_matrix[r * num_experts + e] = remote[e];
    }
}

void gather_N(torch::Tensor N_matrix, torch::Tensor prep_ptrs, int num_experts, int world_size) {
    gather_N_kernel<<<world_size, num_experts, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        N_matrix.data_ptr<int32_t>(), prep_ptrs.data_ptr<int64_t>(), num_experts, world_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_push", &uva_push, "UVA Fused Push");
    m.def("gather_N", &gather_N, "UVA SymmMem AllGather");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_moe_ext", CUDA_SRC)
    return _ext

_moe_symm_cache = None
def _get_buffers(max_tokens: int, hidden_dim: int, world_size: int, num_experts: int, device: torch.device, dtype: torch.dtype):
    global _moe_symm_cache
    key = (max_tokens, hidden_dim, world_size, num_experts, dtype)
    if _moe_symm_cache is not None and _moe_symm_cache.get('key') == key:
        return _moe_symm_cache
    
    prep_buf = symm_mem.empty((num_experts,), dtype=torch.int32, device=device)
    prep_hdl = symm_mem.rendezvous(prep_buf, dist.group.WORLD)
    
    fwd_recv = symm_mem.empty((max_tokens, hidden_dim), dtype=dtype, device=device)
    fwd_recv_hdl = symm_mem.rendezvous(fwd_recv, dist.group.WORLD)
    
    post_recv = symm_mem.empty((max_tokens, hidden_dim), dtype=dtype, device=device)
    post_recv_hdl = symm_mem.rendezvous(post_recv, dist.group.WORLD)
    
    bwd_recv = symm_mem.empty((max_tokens, hidden_dim), dtype=dtype, device=device)
    bwd_recv_hdl = symm_mem.rendezvous(bwd_recv, dist.group.WORLD)
    
    bwd_expert = symm_mem.empty((max_tokens, hidden_dim), dtype=dtype, device=device)
    bwd_expert_hdl = symm_mem.rendezvous(bwd_expert, dist.group.WORLD)
    
    def get_ptrs(hdl):
        return torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
        
    _moe_symm_cache = {
        'key': key,
        'prep': (prep_buf, prep_hdl, get_ptrs(prep_hdl)),
        'fwd': (fwd_recv, fwd_recv_hdl, get_ptrs(fwd_recv_hdl)),
        'post': (post_recv, post_recv_hdl, get_ptrs(post_recv_hdl)),
        'bwd': (bwd_recv, bwd_recv_hdl, get_ptrs(bwd_recv_hdl)),
        'bwd_exp': (bwd_expert, bwd_expert_hdl, get_ptrs(bwd_expert_hdl)),
    }
    return _moe_symm_cache

def compute_routing_tables(N_matrix: torch.Tensor, num_experts: int, rank: int, world_size: int):
    E_loc = num_experts // world_size
    N = N_matrix.cpu().tolist()
    
    fwd_counts, fwd_src_offsets, fwd_dest_ranks, fwd_dest_offsets = [], [], [], []
    for e_glob in range(num_experts):
        j, e_loc = e_glob // E_loc, e_glob % E_loc
        base_offset = sum(N[k][j * E_loc + e] for e in range(e_loc) for k in range(world_size))
        dest_offset = base_offset + sum(N[k][e_glob] for k in range(rank))
        src_offset = sum(N[rank][e] for e in range(e_glob))
        
        fwd_counts.append(N[rank][e_glob])
        fwd_src_offsets.append(src_offset)
        fwd_dest_ranks.append(j)
        fwd_dest_offsets.append(dest_offset)
        
    inv_counts, inv_src_offsets, inv_dest_ranks, inv_dest_offsets = [], [], [], []
    for e_loc in range(E_loc):
        e_glob = rank * E_loc + e_loc
        base_offset = sum(N[k][rank * E_loc + e] for e in range(e_loc) for k in range(world_size))
        for r in range(world_size):
            src_offset = base_offset + sum(N[k][e_glob] for k in range(r))
            dest_offset = sum(N[r][e] for e in range(e_glob))
            
            inv_counts.append(N[r][e_glob])
            inv_src_offsets.append(src_offset)
            inv_dest_ranks.append(r)
            inv_dest_offsets.append(dest_offset)
            
    total_recv = sum(N[k][rank * E_loc + e] for e in range(E_loc) for k in range(world_size))
    
    return (
        torch.tensor(fwd_counts, dtype=torch.int32, device='cuda'),
        torch.tensor(fwd_src_offsets, dtype=torch.int32, device='cuda'),
        torch.tensor(fwd_dest_ranks, dtype=torch.int32, device='cuda'),
        torch.tensor(fwd_dest_offsets, dtype=torch.int32, device='cuda'),
        torch.tensor(inv_counts, dtype=torch.int32, device='cuda'),
        torch.tensor(inv_src_offsets, dtype=torch.int32, device='cuda'),
        torch.tensor(inv_dest_ranks, dtype=torch.int32, device='cuda'),
        torch.tensor(inv_dest_offsets, dtype=torch.int32, device='cuda'),
        total_recv
    )

class PreAll2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, sorted_indices, fwd_tables, inv_tables, fwd_symm, bwd_symm, total_recv, hidden_dim):
        fwd_buf, fwd_hdl, fwd_ptrs = fwd_symm
        bwd_buf, bwd_hdl, bwd_ptrs = bwd_symm
        
        ctx.save_for_backward(sorted_indices)
        ctx.inv_tables = inv_tables
        ctx.bwd_buf, ctx.bwd_hdl, ctx.bwd_ptrs = bwd_buf, bwd_hdl, bwd_ptrs
        ctx.hidden_dim, ctx.num_tokens = hidden_dim, hidden_states.size(0)
        
        counts, src_offsets, dest_ranks, dest_offsets = fwd_tables
        if counts.size(0) > 0:
            _get_ext().uva_push(hidden_states, sorted_indices, counts, src_offsets, dest_ranks, dest_offsets, fwd_ptrs, counts.size(0), hidden_dim)
        fwd_hdl.barrier(channel=0)
        return fwd_buf[:total_recv].clone()

    @staticmethod
    def backward(ctx, grad_output):
        sorted_indices, = ctx.saved_tensors
        counts, src_offsets, dest_ranks, dest_offsets = ctx.inv_tables
        if counts.size(0) > 0:
            _get_ext().uva_push(grad_output.contiguous(), None, counts, src_offsets, dest_ranks, dest_offsets, ctx.bwd_ptrs, counts.size(0), ctx.hidden_dim)
        ctx.bwd_hdl.barrier(channel=0)
        
        grad_hidden_states = torch.zeros(ctx.num_tokens, ctx.hidden_dim, dtype=grad_output.dtype, device=grad_output.device)
        grad_hidden_states.index_put_((sorted_indices,), ctx.bwd_buf[:sorted_indices.size(0)], accumulate=True)
        return grad_hidden_states, None, None, None, None, None, None, None

class PostAll2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, inv_tables, fwd_tables, post_symm, bwd_exp_symm, total_sent, hidden_dim):
        post_buf, post_hdl, post_ptrs = post_symm
        bwd_exp_buf, bwd_exp_hdl, bwd_exp_ptrs = bwd_exp_symm
        
        ctx.fwd_tables = fwd_tables
        ctx.bwd_exp_buf, ctx.bwd_exp_hdl, ctx.bwd_exp_ptrs = bwd_exp_buf, bwd_exp_hdl, bwd_exp_ptrs
        ctx.hidden_dim, ctx.total_recv = hidden_dim, expert_outputs.size(0)
        
        counts, src_offsets, dest_ranks, dest_offsets = inv_tables
        if counts.size(0) > 0:
            _get_ext().uva_push(expert_outputs.contiguous(), None, counts, src_offsets, dest_ranks, dest_offsets, post_ptrs, counts.size(0), hidden_dim)
        post_hdl.barrier(channel=0)
        return post_buf[:total_sent].clone()

    @staticmethod
    def backward(ctx, grad_output):
        counts, src_offsets, dest_ranks, dest_offsets = ctx.fwd_tables
        if counts.size(0) > 0:
            _get_ext().uva_push(grad_output.contiguous(), None, counts, src_offsets, dest_ranks, dest_offsets, ctx.bwd_exp_ptrs, counts.size(0), ctx.hidden_dim)
        ctx.bwd_exp_hdl.barrier(channel=0)
        return ctx.bwd_exp_buf[:ctx.total_recv].clone(), None, None, None, None, None, None

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
    _get_ext()
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = hidden_states.device
    hidden_dim = hidden_states.size(-1)
    num_tokens = hidden_states.view(-1, hidden_dim).size(0)
    dtype = hidden_states.dtype
    
    max_tokens = world_size * num_tokens * top_k
    symm_cache = _get_buffers(max_tokens, hidden_dim, world_size, num_experts, device, dtype)
    
    # Router
    router_logits = torch.nn.functional.linear(hidden_states.view(-1, hidden_dim), gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    # Preprocess SymmMem AllGather
    prep_buf, prep_hdl, prep_ptrs = symm_cache['prep']
    prep_buf.copy_(expert_mask.sum(dim=(1, 2)).to(torch.int32))
    prep_hdl.barrier(channel=0)
    
    N_matrix = torch.empty((world_size, num_experts), dtype=torch.int32, device=device)
    _get_ext().gather_N(N_matrix, prep_ptrs, num_experts, world_size)
    
    tables = compute_routing_tables(N_matrix, num_experts, rank, world_size)
    fwd_tables, inv_tables, total_recv = tables[0:4], tables[4:8], tables[8]
    
    # Sorting config
    routing_map = expert_mask.sum(dim=1).bool()
    sorted_indices = torch.arange(num_tokens, device=device, dtype=torch.int32).unsqueeze(0).expand(num_experts, -1).masked_select(routing_map)
    
    # UVA Token Pre All2All
    recv_buf_fwd = PreAll2All.apply(
        hidden_states, sorted_indices, fwd_tables, inv_tables,
        symm_cache['fwd'], symm_cache['bwd'], total_recv, hidden_dim
    )
    
    # Expert execution
    expert_outputs = expert_forward(recv_buf_fwd, gate_proj, up_proj, down_proj)
    
    # UVA Tokens Post All2All
    post_recv_buf = PostAll2All.apply(
        expert_outputs, inv_tables, fwd_tables,
        symm_cache['post'], symm_cache['bwd_exp'], sorted_indices.size(0), hidden_dim
    )
    
    # Unpermute
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map)
    tokens = post_recv_buf * tokens_weight.unsqueeze(-1)
    unpermuted_tokens = torch.zeros_like(hidden_states)
    unpermuted_tokens.index_put_((sorted_indices,), tokens, accumulate=True)
    
    return unpermuted_tokens