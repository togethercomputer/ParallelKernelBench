"""
Strategy:
1. Custom TMA-based All-to-All: Replaced NCCL's `dist.all_to_all_single` with a custom ThunderKittens 
   kernel `tk_jagged_all_to_all`. It uses asynchronous TMA loads/stores to move variable-sized token
   chunks directly between GPUs via symmetric memory (PGL).
2. Dynamic Tile Loading: Extracted the global `max_tokens` from the existing `_preprocess` step, 
   allowing us to allocate stable TKParallelTensors. The host passes a `send_tiles` array to the kernel 
   so it only TMA transfers valid data chunks, minimizing NVLink bandwidth waste.
3. Device-Side Synchronization: Fused `barrier_all` into the entrypoint to guarantee memory visibility 
   across the cluster without stalling the host.
4. Fast Packing/Casting: Pack/unpack steps use fast device-to-device `.copy_()`, which seamlessly 
   handles FP32 <-> BF16 conversions while staging data for the asynchronous PGL TMA transfers.
"""

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source for Jagged TMA All-To-All
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <vector>

using namespace kittens;

namespace jagged_all_to_all {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 64;

    using shared_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    // Layout: [batch, depth, rows, cols]
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    const int dev_idx;
    int send_tiles[NUM_DEVICES];

    __host__ inline dim3 grid() const {
        return dim3(NUM_DEVICES, input.rows() / ROW_BLOCK_SIZE);
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) + 1024);
    }
};

__device__ inline void kernel(const globals &G) {
    int dst_dev = blockIdx.x;
    int tile_idx = blockIdx.y;

    if (tile_idx >= G.send_tiles[dst_dev]) return;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::shared_tile &tile = allocator.allocate<globals::shared_tile>();

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    
    // Load from local input buffer: [dst_dev, 0, tile_idx, 0]
    tma::expect_bytes(arrived, sizeof(tile));
    tma::load_async(tile, G.input[G.dev_idx], {dst_dev, 0, tile_idx, 0}, arrived);
    wait(arrived, 0);
    
    // Store to remote output buffer on dst_dev: [0, G.dev_idx, tile_idx, 0]
    tma::store_async(G.output[dst_dev], tile, {0, G.dev_idx, tile_idx, 0});
}

} // namespace jagged_all_to_all

namespace jagged_all_to_all_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace jagged_all_to_all_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    std::vector<int> send_tiles
) {
    kittens::py::parallel_tensor_check(output, input);

    jagged_all_to_all::globals all_to_all_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename jagged_all_to_all::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename jagged_all_to_all::globals::parallel_layout>(input),
        .dev_idx = input.local_rank_
    };
    
    for(int i = 0; i < jagged_all_to_all::globals::NUM_DEVICES; i++) {
        all_to_all_G.send_tiles[i] = send_tiles[i];
    }

    jagged_all_to_all_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<jagged_all_to_all_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<jagged_all_to_all_barrier::config, jagged_all_to_all_barrier::globals, jagged_all_to_all_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<jagged_all_to_all::config, jagged_all_to_all::globals, jagged_all_to_all::kernel>(all_to_all_G);
    kittens::py::launch_kernel<jagged_all_to_all_barrier::config, jagged_all_to_all_barrier::globals, jagged_all_to_all_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_jagged_all_to_all", &entrypoint);
}
'''

TK_CUDA_FLAGS = [
    "-std=c++20", "--use_fast_math", "--expt-extended-lambda", "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER", "-gencode", "arch=compute_90a,code=sm_90a",
    "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-Wno-psabi", "-Xcompiler=-fno-strict-aliasing", "-DNDEBUG",
]

_ext = None
_ext_jit_ready = False


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_jagged_alltoall_ext",
            CUDA_SRC,
            extra_cuda_cflags=TK_CUDA_FLAGS,
            extra_include_paths=[
                os.path.join(TK_ROOT, "include"),
                os.path.join(TK_ROOT, "prototype"),
            ],
            extra_ldflags=["-lcuda"],
        )
    return _ext


def _ensure_ext_jit():
    global _ext_jit_ready
    if _ext_jit_ready:
        return _get_ext()
    rank = dist.get_rank()
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


# ----- Custom TK AllToAll Autograd Function -----

class TKJaggedAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, max_tokens, ext):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.max_tokens = max_tokens
        ctx.ext = ext

        W = dist.get_world_size(group=group)
        if W == 1:
            return input.contiguous()

        input = input.contiguous()
        H = input.shape[-1]
        
        assert W == 8, "This ThunderKittens kernel is compiled for NUM_DEVICES=8"
        assert H == 64, "This ThunderKittens kernel is compiled for COL_BLOCK_SIZE=64"

        ROW_BLOCK_SIZE = 16
        # Compute padding boundaries across all messages
        MAX_TILES = max(1, (max_tokens + ROW_BLOCK_SIZE - 1) // ROW_BLOCK_SIZE)
        PADDED_LEN = MAX_TILES * ROW_BLOCK_SIZE

        # Asymmetric shape prevents PGL aliasing and maps correctly to TMA coordinates
        # input_tk: [batch=W, depth=1] -> local read [dst_dev, 0, tile_idx, 0]
        # output_tk: [batch=1, depth=W] -> remote write [0, src_dev, tile_idx, 0]
        input_tk = get_or_create_parallel_tensor(ext, (W, 1, PADDED_LEN, H), torch.bfloat16, multicast=False)
        output_tk = get_or_create_parallel_tensor(ext, (1, W, PADDED_LEN, H), torch.bfloat16, multicast=False)
        barrier_tk = get_or_create_barrier(ext, num_devices=W)

        # Pack device-side splits dynamically (and implicitly cast FP32 -> BF16)
        offset = 0
        send_tiles = []
        for j in range(W):
            length = input_split_sizes[j]
            if length > 0:
                input_tk.data_[j, 0, :length, :].copy_(input[offset : offset + length])
            offset += length
            send_tiles.append((length + ROW_BLOCK_SIZE - 1) // ROW_BLOCK_SIZE)

        # Execute ThunderKittens workload
        ext.tk_jagged_all_to_all(output_tk, input_tk, barrier_tk, send_tiles)

        # Unpack remote chunks (and implicitly cast BF16 -> FP32)
        output = torch.empty((sum(output_split_sizes), H), dtype=input.dtype, device=input.device)
        offset = 0
        for j in range(W):
            length = output_split_sizes[j]
            if length > 0:
                output[offset : offset + length].copy_(output_tk.data_[0, j, :length, :])
            offset += length

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Swaps splits symmetrically
        grad_input = TKJaggedAllToAll.apply(
            ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes, ctx.max_tokens, ctx.ext
        )
        return None, grad_input, None, None, None, None


# ----- MoE Operations -----

def _preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor, int]:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))
    input_splits = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    )
    
    num_local_tokens_per_expert_flat = num_local_tokens_per_expert.contiguous().view(-1)
    output_size = ep_size * num_local_tokens_per_expert_flat.numel()
    num_global_tokens_per_expert_flat = torch.empty(
        output_size,
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(
        num_global_tokens_per_expert_flat, num_local_tokens_per_expert_flat, group=ep_group
    )
    
    num_global_tokens_per_expert = num_global_tokens_per_expert_flat.view(
        ep_size, num_local_tokens_per_expert.size(0)
    )
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        :, start_idx:end_idx
    ].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    
    # Calculate exactly what max_tokens needs to be via the global map
    send_matrix = torch.zeros((ep_size, ep_size), dtype=torch.int32, device=expert_mask.device)
    for j in range(ep_size):
        send_matrix[:, j] = num_global_tokens_per_expert[:, j * num_local_experts : (j + 1) * num_local_experts].sum(dim=1)
    max_tokens = send_matrix.max().item()

    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(
        dim=0
    ).to(torch.device("cpu"), non_blocking=True)
    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
        -1, num_local_experts
    ).to(torch.device("cpu"), non_blocking=True)
    
    return (
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        num_global_sum_tokens_per_local_expert,
        max_tokens
    )


def _permute(
    tokens: torch.Tensor, routing_map: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    input: torch.Tensor,
    split_sizes: Union[torch.Tensor, List[int]],
    sorted_idxs: List[int],
) -> torch.Tensor:
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()
    chunks = torch.split(input, split_sizes, dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)


def _generate_weights_idx(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
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


def token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: List[int],
    output_splits: List[int],
    num_global_tokens_per_local_expert: torch.Tensor,
    max_tokens: int,
    ext,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    group = group or dist.group.WORLD
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)

    local_permuted_hidden_states, local_input_permutation_mapping = _permute(
        hidden_states, routing_map
    )
    expected_tokens = sum(input_splits)
    actual_tokens = local_permuted_hidden_states.shape[0]
    if expected_tokens != actual_tokens:
        raise RuntimeError(
            f"EP split mismatch: input_splits sum ({expected_tokens}) != "
            f"permuted tokens ({actual_tokens})"
        )

    # Use parallelkittens optimized TMA jagged communication
    global_permuted_hidden_states = TKJaggedAllToAll.apply(
        group, local_permuted_hidden_states, output_splits, input_splits, max_tokens, ext
    )
    
    num_local_experts = num_experts // dist.get_world_size(group)
    permute_order = (
        torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    )
    split_sizes = num_global_tokens_per_local_expert.ravel().tolist()
    global_permuted_hidden_states = _sort_chunks_by_idxs(
        global_permuted_hidden_states, split_sizes, permute_order
    )
    return (
        global_permuted_hidden_states,
        routing_map,
        local_input_permutation_mapping,
        org_hidden_states_shape,
    )


def tokens_post_all2all(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    input_splits: List[int],
    output_splits: List[int],
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    max_tokens: int,
    ext,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    num_local_experts = num_experts // dist.get_world_size(group)
    unpermute_order = (
        torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    )
    split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist()
    expert_outputs = _sort_chunks_by_idxs(
        expert_outputs, split_sizes, unpermute_order
    )
    
    # Use parallelkittens optimized TMA jagged communication
    unpermute_outputs = TKJaggedAllToAll.apply(
        group, expert_outputs, input_splits, output_splits, max_tokens, ext
    )
    
    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    unpermute_outputs = _unpermute(
        unpermute_outputs,
        weights_idx,
        org_hidden_states_shape,
        local_input_permutation_mapping,
        routing_map,
    )
    return unpermute_outputs


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
    """
    One MoE forward. Returns combined expert output. Handles `.backward()` properly.
    Uses native ParallelKittens direct PGL operations for dynamic routing bottlenecks.
    """
    group = group or dist.group.WORLD
    hidden_dim = hidden_states.size(-1)

    ext = _ensure_ext_jit()

    # Router
    router_logits = torch.nn.functional.linear(
        hidden_states.reshape(-1, hidden_dim), gate_weight, gate_bias
    )
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)

    # Preprocess
    input_splits, output_splits, num_global_tokens_per_local_expert, _, max_tokens = _preprocess(
        expert_mask, num_experts, group
    )

    # Token pre all2all (fused with ThunderKittens TKJaggedAllToAll)
    (
        global_permuted_hidden_states,
        routing_map,
        local_input_permutation_mapping,
        org_hidden_states_shape,
    ) = token_pre_all2all(
        hidden_states,
        expert_mask,
        num_experts,
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        max_tokens,
        ext,
        group,
    )

    # Local expert (shared MLP)
    expert_outputs = expert_forward(
        global_permuted_hidden_states, gate_proj, up_proj, down_proj
    )

    # Tokens post all2all (fused with ThunderKittens TKJaggedAllToAll)
    out = tokens_post_all2all(
        expert_outputs,
        routing_weights,
        selected_experts,
        num_experts,
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        routing_map,
        local_input_permutation_mapping,
        org_hidden_states_shape,
        max_tokens,
        ext,
        group,
    )
    
    return out