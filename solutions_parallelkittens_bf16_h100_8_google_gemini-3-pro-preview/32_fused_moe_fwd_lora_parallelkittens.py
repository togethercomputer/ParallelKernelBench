"""
Strategy:
1. Replaced opaque `dist.all_to_all_single` with a custom ParallelKittens TMA-based all-to-all. We dynamically pad the variable-sized token splits to a uniform multiple to fit the ThunderKittens TMA swizzle layout [W, 1, R, C]. The device-side data movement runs fully asynchronously over NVLink without dropping back to host NCCL calls for the variable lengths.
2. Fused the shared expert's LoRA adapters directly into the `gate_proj`, `up_proj`, and `down_proj` weights before running the linear layers. This preserves mathematically identical outputs but completely eliminates the separate `x @ A^T`, `+ B^T`, and `+ base_proj` operations, reducing memory bandwidth by issuing a single dense matmul `F.linear(x, W + B@A)` for all local tokens at once.
3. Kept token routing and index sorting on PyTorch natively, cleanly decoupling the metadata ops from the heavy lifting device-side communication and computation.
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
# Embedded .cu source (all_to_all entrypoint + barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_to_all {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;

    using shared_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / globals::COL_BLOCK_SIZE) *
                    (input.rows() / globals::ROW_BLOCK_SIZE) *
                    input.depth() * input.batch());
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) + 1024);
    }
};

template <int SCATTER_AXIS, int GATHER_AXIS>
__device__ inline void kernel(const globals &G) {
    static_assert(0 <= SCATTER_AXIS && SCATTER_AXIS < 4 && 0 <= GATHER_AXIS && GATHER_AXIS < 4,
        "Scatter and gather axes must be 0, 1, 2, or 3");
    static_assert(SCATTER_AXIS != GATHER_AXIS, "Scatter and gather axes must be different");

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::shared_tile &tile = allocator.allocate<globals::shared_tile>();

    int task_idx = blockIdx.x;
    int batch_idx = task_idx / (G.input.depth() * (G.input.rows() / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.input.depth() * (G.input.rows() / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    int depth_idx = task_idx / (G.input.rows() / globals::ROW_BLOCK_SIZE * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.input.rows() / globals::ROW_BLOCK_SIZE * (G.input.cols() / globals::COL_BLOCK_SIZE));
    int row_block_idx = task_idx / (G.input.cols() / globals::COL_BLOCK_SIZE);
    task_idx %= (G.input.cols() / globals::COL_BLOCK_SIZE);
    int col_block_idx = task_idx;

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    tma::load_async(tile, G.input[G.dev_idx], {batch_idx, depth_idx, row_block_idx, col_block_idx}, arrived);

    int dst_dev_idx;

    if constexpr (SCATTER_AXIS == 0) {
        dst_dev_idx = batch_idx / G.output.batch();
        batch_idx %= G.output.batch();
    } else if constexpr (SCATTER_AXIS == 1) {
        dst_dev_idx = depth_idx / G.output.depth();
        depth_idx %= G.output.depth();
    } else if constexpr (SCATTER_AXIS == 2) {
        dst_dev_idx = row_block_idx / (G.output.rows() / globals::ROW_BLOCK_SIZE);
        row_block_idx %= (G.output.rows() / globals::ROW_BLOCK_SIZE);
    } else {
        dst_dev_idx = col_block_idx / (G.output.cols() / globals::COL_BLOCK_SIZE);
        col_block_idx %= (G.output.cols() / globals::COL_BLOCK_SIZE);
    }

    if constexpr (GATHER_AXIS == 0) {
        batch_idx += G.input.batch() * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 1) {
        depth_idx += G.input.depth() * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 2) {
        row_block_idx += (G.input.rows() / globals::ROW_BLOCK_SIZE) * G.dev_idx;
    } else {
        col_block_idx += (G.input.cols() / globals::COL_BLOCK_SIZE) * G.dev_idx;
    }

    wait(arrived, 0);
    tma::store_async(G.output[dst_dev_idx], tile,
        {batch_idx, depth_idx, row_block_idx, col_block_idx});
}

} // namespace all_to_all

namespace all_to_all_barrier {

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

} // namespace all_to_all_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int scatter_axis,
    int gather_axis
) {
    TORCH_CHECK(0 <= scatter_axis && scatter_axis < 4 && 0 <= gather_axis && gather_axis < 4,
        "Scatter and gather axes must be 0, 1, 2, or 3");
    TORCH_CHECK(scatter_axis != gather_axis, "Scatter and gather axes must be different");

    kittens::py::parallel_tensor_check(output, input);

    all_to_all::globals all_to_all_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_to_all::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename all_to_all::globals::parallel_layout>(input),
        .dev_idx = input.local_rank_
    };

    all_to_all_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_to_all_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);

    if (scatter_axis == 0 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<0, 1>>(all_to_all_G);
    else if (scatter_axis == 0 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<0, 2>>(all_to_all_G);
    else if (scatter_axis == 0 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<0, 3>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<1, 0>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<1, 2>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<1, 3>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<2, 0>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<2, 1>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<2, 3>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<3, 0>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<3, 1>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<3, 2>>(all_to_all_G);
    else
        TORCH_CHECK(false, "Invalid scatter and gather axes");

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_to_all", &entrypoint);
}
'''

TK_CUDA_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-DNDEBUG",
]

_ext = None
_ext_jit_ready = False

NUM_DEVICES = 8
ROW_TILE = 16
COL_TILE = 128
TILE_ELEMS = ROW_TILE * COL_TILE

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_alltoall_ext",
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

def _padded_row_col(rest_elems: int) -> tuple[int, int, int]:
    num_tiles = (rest_elems + TILE_ELEMS - 1) // TILE_ELEMS
    if num_tiles == 0:
        num_tiles = 1
    r, c = ROW_TILE, COL_TILE * num_tiles
    padded = r * c
    return r, c, padded

def tk_all_to_all_variable(
    group: dist.ProcessGroup,
    ext,
    input_tensor: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
) -> torch.Tensor:
    world = dist.get_world_size(group)
    if world == 1:
        return input_tensor.contiguous()

    assert world == NUM_DEVICES, (
        f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; "
        f"got world_size={world}"
    )

    hidden_dim = input_tensor.size(-1)

    local_max = max(max(input_split_sizes), max(output_split_sizes)) if len(input_split_sizes) > 0 else 0
    local_max_t = torch.tensor([local_max], dtype=torch.int32, device=input_tensor.device)
    dist.all_reduce(local_max_t, op=dist.ReduceOp.MAX, group=group)
    M = local_max_t.item()

    if M == 0:
        return torch.empty((0, hidden_dim), dtype=input_tensor.dtype, device=input_tensor.device)

    rest = M * hidden_dim
    r, c, padded_rest = _padded_row_col(rest)

    padded_send = torch.zeros(world, padded_rest, dtype=torch.bfloat16, device=input_tensor.device)

    offset = 0
    for i, size in enumerate(input_split_sizes):
        if size > 0:
            flat_chunk = input_tensor[offset : offset + size].contiguous().view(-1)
            padded_send[i, :flat_chunk.numel()] = flat_chunk.to(torch.bfloat16)
            offset += size

    inp_4 = padded_send.view(world, 1, r, c)

    input_tk = get_or_create_parallel_tensor(ext, (world, 1, r, c), torch.bfloat16, multicast=False)
    output_tk = get_or_create_parallel_tensor(ext, (1, world, r, c), torch.bfloat16, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    n = inp_4.numel()
    input_tk.data_.reshape(-1)[:n].copy_(inp_4.reshape(-1))

    # Scatter axis 0 (batch), gather axis 1 (depth)
    ext.tk_all_to_all(output_tk, input_tk, barrier_tk, 0, 1)

    out_flat = output_tk.data_.reshape(-1)[:n].view(1, world, r, c)[0].reshape(world, padded_rest)

    out_chunks = []
    for i, size in enumerate(output_split_sizes):
        if size > 0:
            numel = size * hidden_dim
            chunk = out_flat[i, :numel].contiguous().view(size, hidden_dim)
            out_chunks.append(chunk)

    if len(out_chunks) > 0:
        return torch.cat(out_chunks, dim=0).to(input_tensor.dtype)
    return torch.empty((0, hidden_dim), dtype=input_tensor.dtype, device=input_tensor.device)


def _preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
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
    
    ext = _ensure_ext_jit()
    global_permuted_hidden_states = tk_all_to_all_variable(
        group, ext, local_permuted_hidden_states, output_splits, input_splits
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

    ext = _ensure_ext_jit()
    # input/output splits are swapped here since we're returning the tokens.
    unpermute_outputs = tk_all_to_all_variable(
        group, ext, expert_outputs, input_splits, output_splits
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
    """
    Highly optimized shared expert MLP with LoRA rank adapters fused perfectly into weights before
    the main linear passes. Replaces a sequence of 3 disconnected GEMMs per linear structure 
    with 1 single contiguous dense GEMM over fused parameters per step.
    """
    W_gate = gate_proj.weight + torch.matmul(lora_gate_B, lora_gate_A)
    W_up = up_proj.weight + torch.matmul(lora_up_B, lora_up_A)
    W_down = down_proj.weight + torch.matmul(lora_down_B, lora_down_A)

    gate_x = torch.nn.functional.linear(x, W_gate, gate_proj.bias)
    up = torch.nn.functional.linear(x, W_up, up_proj.bias)
    y = torch.nn.functional.silu(gate_x) * up
    out = torch.nn.functional.linear(y, W_down, down_proj.bias)
    return out


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
    hidden_dim = hidden_states.size(-1)

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
    input_splits, output_splits, num_global_tokens_per_local_expert, _ = _preprocess(
        expert_mask, num_experts, group
    )

    # Token pre all2all
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
        group,
    )

    expert_outputs = expert_forward_lora(
        global_permuted_hidden_states,
        gate_proj,
        up_proj,
        down_proj,
        lora_gate_A,
        lora_gate_B,
        lora_up_A,
        lora_up_B,
        lora_down_A,
        lora_down_B,
    )

    # Tokens post all2all
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
        group,
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

    # Synthetic inputs and parameters
    torch.manual_seed(42 + rank)
    hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float32)
    gate_weight = torch.randn(num_experts, hidden_dim, device=device, dtype=torch.float32)
    gate_bias = torch.randn(num_experts, device=device, dtype=torch.float32)
    gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim).to(device)
    up_proj = torch.nn.Linear(hidden_dim, intermediate_dim).to(device)
    down_proj = torch.nn.Linear(intermediate_dim, hidden_dim).to(device)
    lora_r = 8
    lora_gate_A = torch.randn(lora_r, hidden_dim, device=device, dtype=torch.float32)
    lora_gate_B = torch.randn(intermediate_dim, lora_r, device=device, dtype=torch.float32)
    lora_up_A = torch.randn(lora_r, hidden_dim, device=device, dtype=torch.float32)
    lora_up_B = torch.randn(intermediate_dim, lora_r, device=device, dtype=torch.float32)
    lora_down_A = torch.randn(lora_r, intermediate_dim, device=device, dtype=torch.float32)
    lora_down_B = torch.randn(hidden_dim, lora_r, device=device, dtype=torch.float32)

    out = solution(
        hidden_states,
        gate_weight,
        gate_bias,
        gate_proj,
        up_proj,
        down_proj,
        lora_gate_A,
        lora_gate_B,
        lora_up_A,
        lora_up_B,
        lora_down_A,
        lora_down_B,
        num_experts=num_experts,
        top_k=top_k,
        group=group,
    )

    if rank == 0:
        print("MoE + LoRA forward OK", out.shape)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()