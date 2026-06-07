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
# Embedded .cu source for fused permute + P2P scatter
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <torch/extension.h>

using namespace kittens;

namespace fused_scatter {

using output_layout = pgl<gl<__nv_bfloat16, -1, -1, -1, -1>, 8, false>;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 128;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    output_layout output;
    const __nv_bfloat16* hidden_states;
    const int64_t* sorted_indices;
    const int64_t* sorted_expert_ids;
    const int32_t* my_dest_offsets;
    const int32_t* my_src_offsets;
    int num_local_experts;
    int hidden_dim;
    int total_routed;

    __host__ inline dim3 grid() const {
        return dim3(total_routed);
    }
};

__device__ inline void kernel(const globals &G) {
    int token_idx = blockIdx.x;
    if (token_idx >= G.total_routed) return;

    int64_t src_token_id = G.sorted_indices[token_idx];
    int64_t expert_id = G.sorted_expert_ids[token_idx];

    int owner_rank = expert_id / G.num_local_experts;
    int relative_idx = token_idx - G.my_src_offsets[expert_id];
    int dest_token_id = G.my_dest_offsets[expert_id] + relative_idx;

    const __nv_bfloat16* src_row = G.hidden_states + src_token_id * G.hidden_dim;
    __nv_bfloat16* dest_row = G.output[owner_rank].data + dest_token_id * G.hidden_dim;

    // Vectorized copy via 128-bit float4 (8 __nv_bfloat16 elements)
    int vec_len = G.hidden_dim / 8;
    const float4* src_vec = reinterpret_cast<const float4*>(src_row);
    float4* dest_vec = reinterpret_cast<float4*>(dest_row);

    for (int i = threadIdx.x; i < vec_len; i += blockDim.x) {
        dest_vec[i] = src_vec[i];
    }
    
    // Remainder
    int remainder_start = vec_len * 8;
    for (int i = remainder_start + threadIdx.x; i < G.hidden_dim; i += blockDim.x) {
        dest_row[i] = src_row[i];
    }
}

} // namespace fused_scatter

namespace barrier_ns {
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
}

void entrypoint(
    torch::Tensor hidden_states,
    torch::Tensor sorted_indices,
    torch::Tensor sorted_expert_ids,
    torch::Tensor my_dest_offsets,
    torch::Tensor my_src_offsets,
    kittens::py::TKParallelTensor &output_tk,
    kittens::py::TKParallelTensor &barrier_tk,
    int num_local_experts
) {
    int total_routed = sorted_indices.size(0);
    int hidden_dim = hidden_states.size(1);

    fused_scatter::globals G {
        .output = kittens::py::parallel_tensor_to_pgl<fused_scatter::output_layout>(output_tk),
        .hidden_states = reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        .sorted_indices = sorted_indices.data_ptr<int64_t>(),
        .sorted_expert_ids = sorted_expert_ids.data_ptr<int64_t>(),
        .my_dest_offsets = my_dest_offsets.data_ptr<int32_t>(),
        .my_src_offsets = my_src_offsets.data_ptr<int32_t>(),
        .num_local_experts = num_local_experts,
        .hidden_dim = hidden_dim,
        .total_routed = total_routed
    };

    barrier_ns::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<8>>(barrier_tk),
        .dev_idx = barrier_tk.local_rank_
    };

    kittens::py::launch_kernel<barrier_ns::config, barrier_ns::globals, barrier_ns::kernel>(barrier_G);
    if (total_routed > 0) {
        kittens::py::launch_kernel<fused_scatter::config, fused_scatter::globals, fused_scatter::kernel>(G);
    }
    kittens::py::launch_kernel<barrier_ns::config, barrier_ns::globals, barrier_ns::kernel>(barrier_G);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_fused_scatter", &entrypoint);
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


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_fused_moe_scatter_ext",
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


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: Union[List[int], torch.Tensor],
    output_splits: Union[List[int], torch.Tensor],
    num_global_tokens_per_local_expert: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    device = hidden_states.device
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    my_rank = dist.get_rank(group)

    assert world_size == 8, "This ThunderKittens parallel layout specifies NUM_DEVICES=8"

    ext = _ensure_ext_jit()

    hidden_dim = hidden_states.size(-1)
    org_hidden_states_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    
    hidden_states_2d = hidden_states.to(torch.bfloat16).view(-1, hidden_dim).contiguous()
    num_tokens = hidden_states_2d.size(0)

    # Fast token routing & mask extraction using built-ins
    routing_map = expert_mask.sum(dim=1)
    routing_map_bool = routing_map.bool()

    token_indices = torch.arange(num_tokens, device=device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map_bool)
    
    expert_indices = torch.arange(num_experts, device=device).unsqueeze(1).expand(-1, num_tokens)
    sorted_expert_ids = expert_indices.masked_select(routing_map_bool)

    expected_tokens = sum(input_splits) if isinstance(input_splits, list) else int(input_splits.sum().item())
    actual_tokens = sorted_indices.size(0)
    if expected_tokens != actual_tokens:
        raise RuntimeError(
            f"EP split mismatch: input_splits sum ({expected_tokens}) != permuted tokens ({actual_tokens})"
        )

    # 1. Exchange the tiny expert token counts to compute perfect destination buffer offsets natively
    my_tokens_per_expert = routing_map_bool.sum(dim=1).to(torch.int32)
    global_tokens_per_expert_flat = torch.empty(world_size * num_experts, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(global_tokens_per_expert_flat, my_tokens_per_expert, group=group)
    global_tokens_per_expert = global_tokens_per_expert_flat.view(world_size, num_experts)

    num_local_experts = num_experts // world_size

    # Calculate absolute offset per expert within the receiver's target buffer 
    expert_totals = global_tokens_per_expert.sum(dim=0)
    expert_totals_2d = expert_totals.view(world_size, num_local_experts)
    expert_base_offsets_2d = torch.zeros_like(expert_totals_2d)
    expert_base_offsets_2d[:, 1:] = expert_totals_2d[:, :-1].cumsum(dim=1)
    expert_base_offsets = expert_base_offsets_2d.view(num_experts)

    if my_rank > 0:
        my_rank_offset_within_expert = global_tokens_per_expert[:my_rank, :].sum(dim=0)
    else:
        my_rank_offset_within_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)

    # The exact placement offset our SM will scatter into the peer's SM symmetric buffer 
    my_dest_offsets = expert_base_offsets + my_rank_offset_within_expert

    my_src_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    my_src_offsets[1:] = my_tokens_per_expert[:-1].cumsum(dim=0)

    # 2. Setup symmetrical receive buffers (padding max size consistently across group)
    local_out_size = int(sum(output_splits) if isinstance(output_splits, list) else output_splits.sum().item())
    local_out_size_t = torch.tensor([local_out_size], dtype=torch.int32, device=device)
    max_out_size_t = local_out_size_t.clone()
    dist.all_reduce(max_out_size_t, op=dist.ReduceOp.MAX, group=group)
    max_out_size = max(1, max_out_size_t.item())

    # We use (1, 1, R, C) layout mapping for a generic pgl tensor match
    output_tk = get_or_create_parallel_tensor(
        ext, (1, 1, max_out_size, hidden_dim), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    # 3. Direct NVLink P2P DMA dispatch bypassing chunk assembly, serialization, and concatenation splits
    ext.tk_fused_scatter(
        hidden_states_2d,
        sorted_indices,
        sorted_expert_ids,
        my_dest_offsets,
        my_src_offsets,
        output_tk,
        barrier_tk,
        num_local_experts
    )

    global_permuted = output_tk.data_.view(-1, hidden_dim)[:local_out_size].to(original_dtype).clone()

    return global_permuted, routing_map, sorted_indices, org_hidden_states_shape