import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension
from typing import List, Optional, Tuple, Union
import triton
import triton.language as tl

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void uva_allgather_counts_kernel(
    const int64_t* local_counts,
    int64_t** peer_ptrs,
    int rank, int world_size, int num_experts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_experts) {
        int64_t val = local_counts[idx];
        for (int p = 0; p < world_size; ++p) {
            peer_ptrs[p][rank * num_experts + idx] = val;
        }
    }
}

__global__ void compute_offsets_kernel(
    const int64_t* counts, 
    int64_t* local_offsets, 
    int64_t* dest_offsets, 
    int64_t* local_res_offsets, 
    int64_t* dest_res_offsets, 
    int rank, int world_size, int num_experts)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int L = num_experts / world_size;

        // Forward push token offsets
        int64_t loc_off = 0;
        for (int e = 0; e < num_experts; ++e) {
            local_offsets[e] = loc_off;
            loc_off += counts[rank * num_experts + e];

            int j = e / L;
            int64_t dest_off = 0;
            // Sum all experts on rank j before e
            for (int x = j * L; x < e; ++x) {
                for (int p = 0; p < world_size; ++p) {
                    dest_off += counts[p * num_experts + x];
                }
            }
            // Sum all ranks before me for expert e
            for (int p = 0; p < rank; ++p) {
                dest_off += counts[p * num_experts + e];
            }
            dest_offsets[e] = dest_off;
        }

        // Backward pull/return offsets
        loc_off = 0;
        for (int el = 0; el < L; ++el) {
            int e = rank * L + el;
            for (int k = 0; k < world_size; ++k) {
                local_res_offsets[el * world_size + k] = loc_off;
                loc_off += counts[k * num_experts + e];

                int64_t dest_off = 0;
                // Dest rank k buffer is natively ordered by expert x
                for (int x = 0; x < e; ++x) {
                    dest_off += counts[k * num_experts + x];
                }
                dest_res_offsets[el * world_size + k] = dest_off;
            }
        }
    }
}

__global__ void uva_push_tokens_kernel_vec(
    const uint4* local_tokens,
    uint4** peer_ptrs,
    const int64_t* counts,
    const int64_t* local_offsets,
    const int64_t* dest_offsets,
    int rank, int world_size, int num_experts, int vec_H)
{
    int e = blockIdx.y;
    int64_t count = counts[rank * num_experts + e];
    int64_t total_elements = count * vec_H;
    int L = num_experts / world_size;
    int j = e / L;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        int64_t src_idx = local_offsets[e] * vec_H + idx;
        int64_t dst_idx = dest_offsets[e] * vec_H + idx;
        peer_ptrs[j][dst_idx] = local_tokens[src_idx];
    }
}

__global__ void uva_push_results_kernel_vec(
    const uint4* local_results_chunk,
    uint4** peer_ptrs,
    const int64_t* counts,
    const int64_t* local_res_offsets,
    const int64_t* dest_res_offsets,
    int rank, int world_size, int num_experts, int vec_H,
    int64_t chunk_start, int64_t chunk_end)
{
    int el = blockIdx.y / world_size;
    int k = blockIdx.y % world_size;
    int e = rank * (num_experts / world_size) + el;

    int64_t count = counts[k * num_experts + e];
    int64_t total_elements = count * vec_H;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        int64_t global_tok = local_res_offsets[el * world_size + k] + (idx / vec_H);
        if (global_tok >= chunk_start && global_tok < chunk_end) {
            int64_t src_idx = (global_tok - chunk_start) * vec_H + (idx % vec_H);
            int64_t dst_idx = dest_res_offsets[el * world_size + k] * vec_H + idx;
            peer_ptrs[k][dst_idx] = local_results_chunk[src_idx];
        }
    }
}

void run_allgather_counts(
    torch::Tensor local_counts, int64_t peer_ptrs_addr,
    int rank, int world_size, int num_experts)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (num_experts + threads - 1) / threads;
    int64_t** peer_ptrs = reinterpret_cast<int64_t**>(peer_ptrs_addr);
    uva_allgather_counts_kernel<<<blocks, threads, 0, stream>>>(
        local_counts.data_ptr<int64_t>(), peer_ptrs, rank, world_size, num_experts
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_compute_offsets(
    torch::Tensor counts, torch::Tensor local_offsets, torch::Tensor dest_offsets,
    torch::Tensor local_res_offsets, torch::Tensor dest_res_offsets,
    int rank, int world_size, int num_experts)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    compute_offsets_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int64_t>(), local_offsets.data_ptr<int64_t>(), dest_offsets.data_ptr<int64_t>(),
        local_res_offsets.data_ptr<int64_t>(), dest_res_offsets.data_ptr<int64_t>(),
        rank, world_size, num_experts
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_push_tokens(
    torch::Tensor local_tokens, int64_t peer_ptrs_addr, torch::Tensor counts,
    torch::Tensor local_offsets, torch::Tensor dest_offsets,
    int rank, int world_size, int num_experts, int H)
{
    TORCH_CHECK(H % 8 == 0, "H must be multiple of 8 for uint4 vec");
    int vec_H = H / 8;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(1024, num_experts);
    uint4** peer_ptrs = reinterpret_cast<uint4**>(peer_ptrs_addr);
    uva_push_tokens_kernel_vec<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(local_tokens.data_ptr()), peer_ptrs,
        counts.data_ptr<int64_t>(), local_offsets.data_ptr<int64_t>(), dest_offsets.data_ptr<int64_t>(),
        rank, world_size, num_experts, vec_H
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_push_results(
    torch::Tensor local_results_chunk, int64_t peer_ptrs_addr, torch::Tensor counts,
    torch::Tensor local_res_offsets, torch::Tensor dest_res_offsets,
    int rank, int world_size, int num_experts, int H,
    int64_t chunk_start, int64_t chunk_end)
{
    TORCH_CHECK(H % 8 == 0, "H must be multiple of 8 for uint4 vec");
    int vec_H = H / 8;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int L = num_experts / world_size;
    dim3 grid(1024, L * world_size);
    uint4** peer_ptrs = reinterpret_cast<uint4**>(peer_ptrs_addr);
    uva_push_results_kernel_vec<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(local_results_chunk.data_ptr()), peer_ptrs,
        counts.data_ptr<int64_t>(), local_res_offsets.data_ptr<int64_t>(), dest_res_offsets.data_ptr<int64_t>(),
        rank, world_size, num_experts, vec_H, chunk_start, chunk_end
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_allgather_counts", &run_allgather_counts);
    m.def("run_compute_offsets", &run_compute_offsets);
    m.def("run_push_tokens", &run_push_tokens);
    m.def("run_push_results", &run_push_results);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_uva_fused_lora", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(max_tokens: int, H: int, num_experts: int, device: torch.device):
    world_size = dist.get_world_size()
    key = (max_tokens, H, num_experts, world_size)
    if key in _symm_cache:
        return _symm_cache[key]

    counts_buf = symm_mem.empty((world_size, num_experts), dtype=torch.int64, device=device)
    counts_hdl = symm_mem.rendezvous(counts_buf)
    counts_ptrs = torch.tensor(counts_hdl.buffer_ptrs, dtype=torch.int64, device=device)

    tokens_buf = symm_mem.empty((max_tokens, H), dtype=torch.bfloat16, device=device)
    tokens_hdl = symm_mem.rendezvous(tokens_buf)
    tokens_ptrs = torch.tensor(tokens_hdl.buffer_ptrs, dtype=torch.int64, device=device)

    results_buf = symm_mem.empty((max_tokens, H), dtype=torch.bfloat16, device=device)
    results_hdl = symm_mem.rendezvous(results_buf)
    results_ptrs = torch.tensor(results_hdl.buffer_ptrs, dtype=torch.int64, device=device)

    L = num_experts // world_size
    state = {
        "counts_buf": counts_buf, "counts_hdl": counts_hdl, "counts_ptrs_ptr": counts_ptrs.data_ptr(),
        "tokens_buf": tokens_buf, "tokens_hdl": tokens_hdl, "tokens_ptrs_ptr": tokens_ptrs.data_ptr(),
        "results_buf": results_buf, "results_hdl": results_hdl, "results_ptrs_ptr": results_ptrs.data_ptr(),
        "local_offsets": torch.empty((num_experts,), dtype=torch.int64, device=device),
        "dest_offsets": torch.empty((num_experts,), dtype=torch.int64, device=device),
        "local_res_offsets": torch.empty((L * world_size,), dtype=torch.int64, device=device),
        "dest_res_offsets": torch.empty((L * world_size,), dtype=torch.int64, device=device),
    }
    _symm_cache[key] = state
    return state


@triton.jit
def fused_elementwise_kernel(
    gate_x_ptr, lora_g_ptr, up_x_ptr, lora_u_ptr, y_ptr, N_el, BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N_el

    gx = tl.load(gate_x_ptr + idx, mask=mask)
    lg = tl.load(lora_g_ptr + idx, mask=mask)
    ux = tl.load(up_x_ptr + idx, mask=mask)
    lu = tl.load(lora_u_ptr + idx, mask=mask)

    gx = gx + lg
    ux = ux + lu
    sig = 1.0 / (1.0 + tl.exp(-gx.to(tl.float32)))
    gate = gx.to(tl.float32) * sig
    y = gate * ux.to(tl.float32)

    tl.store(y_ptr + idx, y.to(tl.bfloat16), mask=mask)


def expert_forward_lora_fused(
    x: torch.Tensor, gate_proj, up_proj, down_proj,
    lora_gate_A, lora_gate_B, lora_up_A, lora_up_B, lora_down_A, lora_down_B
) -> torch.Tensor:
    xa_g = torch.nn.functional.linear(x, lora_gate_A)
    lora_g = torch.nn.functional.linear(xa_g, lora_gate_B).contiguous()
    gate_x = gate_proj(x).contiguous()

    xa_u = torch.nn.functional.linear(x, lora_up_A)
    lora_u = torch.nn.functional.linear(xa_u, lora_up_B).contiguous()
    up_x = up_proj(x).contiguous()

    y = torch.empty_like(gate_x)
    N_el = gate_x.numel()
    
    if N_el > 0:
        grid = (triton.cdiv(N_el, 1024),)
        fused_elementwise_kernel[grid](gate_x, lora_g, up_x, lora_u, y, N_el, BLOCK_SIZE=1024)

    xa_d = torch.nn.functional.linear(y, lora_down_A)
    lora_d = torch.nn.functional.linear(xa_d, lora_down_B)
    return down_proj(y) + lora_d


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
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_states_shape[-1])
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    return unpermuted_tokens


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    lora_gate_A: torch.Tensor, lora_gate_B: torch.Tensor,
    lora_up_A: torch.Tensor, lora_up_B: torch.Tensor,
    lora_down_A: torch.Tensor, lora_down_B: torch.Tensor,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    H = hidden_states.size(-1)

    router_logits = torch.nn.functional.linear(hidden_states.reshape(-1, H), gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))
    routing_map = expert_mask.sum(dim=1)

    ext = _get_ext()
    max_tokens = world_size * hidden_states.reshape(-1, H).size(0) * top_k
    symm = _get_symm_state(max_tokens, H, num_experts, hidden_states.device)

    ext.run_allgather_counts(
        num_local_tokens_per_expert, symm["counts_ptrs_ptr"], rank, world_size, num_experts
    )
    symm["counts_hdl"].barrier(channel=0)
    counts = symm["counts_buf"] 

    ext.run_compute_offsets(
        counts, symm["local_offsets"], symm["dest_offsets"], 
        symm["local_res_offsets"], symm["dest_res_offsets"], rank, world_size, num_experts
    )

    local_permuted, local_input_permutation_mapping = _permute(hidden_states.reshape(-1, H), routing_map)
    local_permuted = local_permuted.contiguous()

    ext.run_push_tokens(
        local_permuted, symm["tokens_ptrs_ptr"], counts,
        symm["local_offsets"], symm["dest_offsets"], rank, world_size, num_experts, H
    )
    symm["tokens_hdl"].barrier(channel=1)

    L = num_experts // world_size
    received_tokens_count = counts[:, rank * L : (rank + 1) * L].sum().item()
    global_permuted_hidden_states = symm["tokens_buf"][:received_tokens_count]

    # Pipeline Return Communication with Compute
    C = 2
    chunk_size = (received_tokens_count + C - 1) // C
    push_stream = torch.cuda.Stream()

    for c in range(C):
        start = c * chunk_size
        end = min((c + 1) * chunk_size, received_tokens_count)
        if start >= received_tokens_count:
            break

        chunk_x = global_permuted_hidden_states[start:end]
        chunk_y = expert_forward_lora_fused(
            chunk_x, gate_proj, up_proj, down_proj,
            lora_gate_A, lora_gate_B, lora_up_A, lora_up_B, lora_down_A, lora_down_B
        )

        push_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(push_stream):
            ext.run_push_results(
                chunk_y, symm["results_ptrs_ptr"], counts,
                symm["local_res_offsets"], symm["dest_res_offsets"],
                rank, world_size, num_experts, H, start, end
            )

    torch.cuda.current_stream().wait_stream(push_stream)
    symm["results_hdl"].barrier(channel=2)

    my_returned_count = counts[rank, :].sum().item()
    unpermute_outputs = symm["results_buf"][:my_returned_count]

    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    out = _unpermute(
        unpermute_outputs, weights_idx, hidden_states.shape,
        local_input_permutation_mapping, routing_map
    )
    symm["counts_hdl"].barrier(channel=3)
    return out