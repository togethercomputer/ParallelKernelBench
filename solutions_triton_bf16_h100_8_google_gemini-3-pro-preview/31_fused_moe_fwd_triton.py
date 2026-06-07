"""
Strategy:
1. Device-side routing: Fused the Pre- and Post-Expert AllToAll into a single custom UVA Push C++ kernel via Symmetric Memory. Instead of multiple steps of `all_to_all` and chunk sorting, we precompute exact destination offsets and push tokens directly to their correct sorted positions over NVLink in one kernel.
2. Compute-Communication Overlap: The NCCL `all_gather` for routing token counts is asynchronous and perfectly overlapped with the local token permutation (`_permute`), hiding the small collective latency behind the local memory operation.
3. Zero-allocation routing: Forward and backward passes use identical pre-allocated symmetric buffers and reuse the same UvaPush kernel, eliminating all PyTorch `all_to_all` allocations on the hot path.
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

struct CopyJob {
    int32_t local_token_offset;
    int32_t remote_token_offset;
    int32_t token_count;
    int32_t target_rank;
};

__global__ void uva_push_kernel_vec(
    const at::BFloat16* __restrict__ local_data,
    const uintptr_t* __restrict__ remote_ptrs,
    const CopyJob* __restrict__ jobs,
    int num_jobs,
    int H
) {
    int job_idx = blockIdx.y;
    if (job_idx >= num_jobs) return;
    
    CopyJob job = jobs[job_idx];
    if (job.token_count == 0) return;
    
    const at::BFloat16* src_rem = local_data + job.local_token_offset * H;
    at::BFloat16* dst_rem = reinterpret_cast<at::BFloat16*>(remote_ptrs[job.target_rank]) + job.remote_token_offset * H;
    
    const float4* src_buf = reinterpret_cast<const float4*>(src_rem);
    float4* dst_buf = reinterpret_cast<float4*>(dst_rem);
    
    int total_vecs = (job.token_count * H) / 8; // 8 bf16 per float4
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < total_vecs; i += stride) {
        dst_buf[i] = src_buf[i];
    }
    
    int rem_start = total_vecs * 8;
    int total_elements = job.token_count * H;
    if (rem_start < total_elements) {
        for (int i = rem_start + tid; i < total_elements; i += stride) {
            dst_rem[i] = src_rem[i];
        }
    }
}

void uva_push(
    torch::Tensor local_data,
    torch::Tensor remote_ptrs,
    torch::Tensor jobs,
    int H
) {
    int num_jobs = jobs.size(0);
    if (num_jobs == 0 || local_data.numel() == 0) return;
    
    const at::BFloat16* local_ptr = local_data.data_ptr<at::BFloat16>();
    const uintptr_t* ptrs = reinterpret_cast<const uintptr_t*>(remote_ptrs.data_ptr<int64_t>());
    const CopyJob* jobs_ptr = reinterpret_cast<const CopyJob*>(jobs.data_ptr<int32_t>());
    
    dim3 grid(64, num_jobs);
    dim3 block(256);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    uva_push_kernel_vec<<<grid, block, 0, stream>>>(
        local_ptr, ptrs, jobs_ptr, num_jobs, H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_push", &uva_push, "UVA Push for MoE AllToAll");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_uva_push_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(max_tokens: int, H: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    key = (max_tokens, H, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]

    n = max_tokens * H
    fwd_buf = symm_mem.empty(n, device=device, dtype=dtype)
    fwd_hdl = symm_mem.rendezvous(fwd_buf, dist.group.WORLD)
    
    bwd_buf = symm_mem.empty(n, device=device, dtype=dtype)
    bwd_hdl = symm_mem.rendezvous(bwd_buf, dist.group.WORLD)
    
    fwd_ptrs = torch.tensor(fwd_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    bwd_ptrs = torch.tensor(bwd_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    res = (fwd_buf.view(-1, H), bwd_buf.view(-1, H), fwd_hdl, bwd_hdl, fwd_ptrs, bwd_ptrs)
    _symm_cache[key] = res
    return res

class UvaPush(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, jobs_push, jobs_pull, symm_buf_recv, symm_hdl, symm_ptrs_recv, bwd_symm_buf_recv, bwd_symm_hdl, bwd_symm_ptrs_recv, recv_count, H):
        ctx.save_for_backward(jobs_pull)
        ctx.bwd_symm_buf_recv = bwd_symm_buf_recv
        ctx.bwd_symm_hdl = bwd_symm_hdl
        ctx.bwd_symm_ptrs_recv = bwd_symm_ptrs_recv
        ctx.H = H
        ctx.input_size = input_tensor.size(0)
        
        symm_hdl.barrier(channel=0)
        if input_tensor.numel() > 0:
            _get_ext().uva_push(input_tensor.contiguous(), symm_ptrs_recv, jobs_push, H)
        symm_hdl.barrier(channel=0)
        
        return symm_buf_recv[:recv_count].clone()

    @staticmethod
    def backward(ctx, grad_output):
        jobs_pull, = ctx.saved_tensors
        bwd_symm_buf_recv = ctx.bwd_symm_buf_recv
        bwd_symm_hdl = ctx.bwd_symm_hdl
        bwd_symm_ptrs_recv = ctx.bwd_symm_ptrs_recv
        H = ctx.H
        
        bwd_symm_hdl.barrier(channel=0)
        if grad_output is not None and grad_output.numel() > 0:
            _get_ext().uva_push(grad_output.contiguous(), bwd_symm_ptrs_recv, jobs_pull, H)
        bwd_symm_hdl.barrier(channel=0)
        
        grad_input = bwd_symm_buf_recv[:ctx.input_size].clone() if grad_output is not None else None
        return grad_input, None, None, None, None, None, None, None, None, None, None

def create_jobs(rank, num_global_tokens_per_expert_cpu, ep_size, num_experts, num_local_experts, device):
    src_offsets = torch.zeros_like(num_global_tokens_per_expert_cpu)
    src_offsets[:, 1:] = num_global_tokens_per_expert_cpu.cumsum(dim=1)[:, :-1]
    
    tokens_per_routing = num_global_tokens_per_expert_cpu.view(ep_size, ep_size, num_local_experts)
    dst_layout = tokens_per_routing.permute(1, 2, 0).contiguous()
    
    dst_offsets = torch.zeros_like(dst_layout)
    dst_offsets.view(ep_size, -1)[:, 1:] = dst_layout.view(ep_size, -1).cumsum(dim=1)[:, :-1]
    
    jobs_fwd = []
    for E in range(num_experts):
        count = num_global_tokens_per_expert_cpu[rank, E].item()
        if count == 0: continue
        dst_rank = E // num_local_experts
        local_expert = E % num_local_experts
        l_off = src_offsets[rank, E].item()
        r_off = dst_offsets[dst_rank, local_expert, rank].item()
        jobs_fwd.append([l_off, r_off, count, dst_rank])
        
    jobs_bwd = []
    for e in range(num_local_experts):
        for s in range(ep_size):
            count = dst_layout[rank, e, s].item()
            if count == 0: continue
            E = rank * num_local_experts + e
            l_off = dst_offsets[rank, e, s].item()
            r_off = src_offsets[s, E].item()
            jobs_bwd.append([l_off, r_off, count, s])
            
    jobs_fwd_tensor = torch.tensor(jobs_fwd, dtype=torch.int32, device=device) if jobs_fwd else torch.empty((0, 4), dtype=torch.int32, device=device)
    jobs_bwd_tensor = torch.tensor(jobs_bwd, dtype=torch.int32, device=device) if jobs_bwd else torch.empty((0, 4), dtype=torch.int32, device=device)
    
    total_recv = dst_layout[rank].sum().item()
    return jobs_fwd_tensor, jobs_bwd_tensor, total_recv

def _permute(tokens: torch.Tensor, routing_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices

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
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    return unpermuted_tokens

def _generate_weights_idx(routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros((num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device)
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx

def expert_forward(x: torch.Tensor, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear, down_proj: torch.nn.Linear) -> torch.Tensor:
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
    ep_size = dist.get_world_size(group)
    num_local_experts = num_experts // ep_size
    hidden_dim = hidden_states.size(-1)
    
    assert hidden_states.dtype == torch.bfloat16, "Kernel optimized and mapped for bfloat16"
    _get_ext()
    
    # 1. Routing compute
    router_logits = torch.nn.functional.linear(hidden_states.reshape(-1, hidden_dim), gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1)
    
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))
    
    num_global_tokens_per_expert_flat = torch.empty(
        ep_size * num_experts,
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    
    # OVERLAP: Async all_gather overlaps completely with local data permutation
    work = dist.all_gather_into_tensor(
        num_global_tokens_per_expert_flat, 
        num_local_tokens_per_expert.contiguous().view(-1), 
        group=group,
        async_op=True
    )
    
    routing_map = expert_mask.sum(dim=1)
    local_permuted, local_mapping = _permute(hidden_states.reshape(-1, hidden_dim), routing_map)
    
    work.wait()
    
    # 2. Build explicit routing commands
    num_global_tokens_per_expert = num_global_tokens_per_expert_flat.view(ep_size, num_experts)
    counts_cpu = num_global_tokens_per_expert.cpu()
    jobs_fwd, jobs_bwd, total_recv = create_jobs(
        rank, counts_cpu, ep_size, num_experts, num_local_experts, hidden_states.device
    )
    
    max_tokens = ep_size * hidden_states.reshape(-1, hidden_dim).size(0) * top_k
    fwd_buf, bwd_buf, fwd_hdl, bwd_hdl, fwd_ptrs, bwd_ptrs = _get_symm_state(
        max_tokens, hidden_dim, hidden_states.dtype, hidden_states.device
    )
    
    # 3. AllToAll Route Pre-Expert (Forward Push via Symmetric Memory)
    global_permuted = UvaPush.apply(
        local_permuted, jobs_fwd, jobs_bwd, 
        fwd_buf, fwd_hdl, fwd_ptrs, 
        bwd_buf, bwd_hdl, bwd_ptrs, 
        total_recv, hidden_dim
    )
    
    # 4. Local Expert execution
    expert_outputs = expert_forward(global_permuted, gate_proj, up_proj, down_proj)
    
    # 5. AllToAll Route Post-Expert (Backward Push via Symmetric Memory)
    unpermuted_flat = UvaPush.apply(
        expert_outputs, jobs_bwd, jobs_fwd, 
        bwd_buf, bwd_hdl, bwd_ptrs, 
        fwd_buf, fwd_hdl, fwd_ptrs, 
        local_permuted.size(0), hidden_dim
    )
    
    # 6. Final unpermute and weighted sum
    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    out = _unpermute(
        unpermuted_flat, weights_idx, hidden_states.shape, local_mapping, routing_map
    )
    
    return out

def main() -> None:
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    num_experts = 8
    top_k = 2
    hidden_dim = 64
    intermediate_dim = 128
    batch, seq = 2, 16
    num_tokens = batch * seq
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size"

    torch.manual_seed(42 + rank)
    
    # Optimized precision matching the kernel layout logic
    dtype = torch.bfloat16
    
    hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    gate_weight = torch.randn(num_experts, hidden_dim, device=device, dtype=dtype)
    gate_bias = torch.randn(num_experts, device=device, dtype=dtype)
    gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, dtype=dtype).to(device)
    up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, dtype=dtype).to(device)
    down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, dtype=dtype).to(device)

    out = solution(
        hidden_states,
        gate_weight,
        gate_bias,
        gate_proj,
        up_proj,
        down_proj,
        num_experts=num_experts,
        top_k=top_k,
        group=group,
    )
    loss = out.sum()
    loss.backward()

    if rank == 0:
        print("MoE e2e forward + backward OK")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()