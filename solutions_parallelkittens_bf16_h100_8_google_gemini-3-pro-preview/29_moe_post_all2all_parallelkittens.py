"""
ThunderKittens MoE Post-All2All using P2P direct pull and Fused Unpermute.

Strategy:
1. Device-side P2P Communication: Instead of using `dist.all_to_all_single` and allocating intermediate receive buffers, ranks expose their sorted outgoing tokens in a symmetric `TKParallelTensor` send buffer.
2. Operator Fusion: We replace the memory-bound `scatter_add_` unpermute by having each rank directly pull its assigned tokens from peers over NVLink, multiply by the routing weights, and atomically add into the final output in a single fused Hopper kernel.
3. Compute-Communication Overlap: The small host-side metadata exchange (prefix sums of send chunks) is overlapped asynchronously with the local token sorting phase.
"""

import os
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source for Fused Pull and Unpermute
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>

using namespace kittens;

namespace fused_pull {

struct globals {
    static constexpr int NUM_DEVICES = 8;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, false>;
    
    parallel_layout send_buffer;
    int* peer_send_offsets;
    int* recv_offsets;
    __nv_bfloat16* tokens_weight;
    int64_t* permutation_mapping;
    __nv_bfloat16* unpermuted_tokens;
    
    int hidden_dim;
    int num_received_tokens;
};

__global__ void kernel(globals G) {
    int token_idx = blockIdx.x;
    if (token_idx >= G.num_received_tokens) return;

    // Find which rank this token comes from based on the received prefix sums
    int src_rank = 0;
    for (int r = 1; r < globals::NUM_DEVICES; ++r) {
        if (token_idx >= G.recv_offsets[r]) {
            src_rank = r;
        }
    }

    int local_offset = token_idx - G.recv_offsets[src_rank];
    int peer_offset = G.peer_send_offsets[src_rank] + local_offset;

    // Direct P2P read from the peer's symmetric send buffer
    kittens::bf16* base_ptr = G.send_buffer[src_rank].data;
    __nv_bfloat16* src_ptr = reinterpret_cast<__nv_bfloat16*>(base_ptr) + peer_offset * G.hidden_dim;
    
    int64_t dest_idx = G.permutation_mapping[token_idx];
    __nv_bfloat16* dst_ptr = &G.unpermuted_tokens[dest_idx * G.hidden_dim];

    float weight_f = __bfloat162float(G.tokens_weight[token_idx]);

    int tid = threadIdx.x;
    int num_float4 = G.hidden_dim / 8; // float4 reads 16 bytes = 8 x bfloat16
    float4* src_ptr_4 = reinterpret_cast<float4*>(src_ptr);

    for (int i = tid; i < num_float4; i += blockDim.x) {
        float4 val4 = src_ptr_4[i];
        __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&val4);
        
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = __bfloat162float(vals[j]);
            float res = v * weight_f;
            atomicAdd(&dst_ptr[8 * i + j], __float2bfloat16(res));
        }
    }
    
    // Handle remainder if hidden_dim is not a multiple of 8
    for (int idx = num_float4 * 8 + tid; idx < G.hidden_dim; idx += blockDim.x) {
        float v = __bfloat162float(src_ptr[idx]);
        float res = v * weight_f;
        atomicAdd(&dst_ptr[idx], __float2bfloat16(res));
    }
}

} // namespace fused_pull

namespace all_reduce_barrier {
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
} // namespace all_reduce_barrier

void entrypoint(
    kittens::py::TKParallelTensor &send_buffer,
    kittens::py::TKParallelTensor &barrier,
    torch::Tensor peer_send_offsets,
    torch::Tensor recv_offsets,
    torch::Tensor tokens_weight,
    torch::Tensor permutation_mapping,
    torch::Tensor unpermuted_tokens,
    int hidden_dim,
    int num_received_tokens
) {
    all_reduce_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_reduce_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // First barrier: wait for all ranks to populate their send buffers
    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);

    fused_pull::globals G;
    G.send_buffer = kittens::py::parallel_tensor_to_pgl<fused_pull::globals::parallel_layout>(send_buffer);
    G.peer_send_offsets = peer_send_offsets.data_ptr<int>();
    G.recv_offsets = recv_offsets.data_ptr<int>();
    G.tokens_weight = reinterpret_cast<__nv_bfloat16*>(tokens_weight.data_ptr<at::BFloat16>());
    G.permutation_mapping = permutation_mapping.data_ptr<int64_t>();
    G.unpermuted_tokens = reinterpret_cast<__nv_bfloat16*>(unpermuted_tokens.data_ptr<at::BFloat16>());
    G.hidden_dim = hidden_dim;
    G.num_received_tokens = num_received_tokens;

    if (num_received_tokens > 0) {
        // One block per received token
        fused_pull::kernel<<<num_received_tokens, 256>>>(G);
    }

    // Second barrier: wait until everyone has read from our send buffer before we safely exit
    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_fused_pull_unpermute", &entrypoint);
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
            "tk_fused_moe_pull_unpermute",
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
        (num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


@torch.no_grad()
def solution(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    input_splits: Union[List[int], torch.Tensor],
    output_splits: Union[List[int], torch.Tensor],
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    assert world_size == 8, "This ThunderKittens kernel is strictly built for 8 GPUs."
    
    rank = dist.get_rank(group)
    device = expert_outputs.device
    ext = _ensure_ext_jit()

    # 1. Asynchronously gather the chunk sizes to map out peer data structures
    if isinstance(output_splits, torch.Tensor):
        send_splits_tensor = output_splits.to(dtype=torch.int32, device=device)
    else:
        send_splits_tensor = torch.tensor(output_splits, dtype=torch.int32, device=device)

    all_send_splits = torch.empty((world_size, world_size), dtype=torch.int32, device=device)
    gather_list = [all_send_splits[i] for i in range(world_size)]
    handle = dist.all_gather(gather_list, send_splits_tensor, group=group, async_op=True)

    # 2. Sort the locally processed expert outputs while the host `all_gather` progresses
    num_local_experts = num_experts // world_size
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()

    sorted_expert_outputs = _sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )
    
    # 3. Secure Symmetric Buffer Allocation
    hidden_dim = org_hidden_states_shape[-1]
    topk = routing_weights.size(1)
    # The max tokens a rank could possibly process across ALL experts globally
    max_tokens = org_hidden_states_shape[0] * world_size * topk

    send_buffer_tk = get_or_create_parallel_tensor(
        ext, (1, 1, max_tokens, hidden_dim), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    num_elements = sorted_expert_outputs.size(0)
    if num_elements > 0:
        send_buffer_tk.data_[0, 0, :num_elements, :].copy_(sorted_expert_outputs.to(torch.bfloat16))

    handle.wait()
    
    # 4. Resolve Peer offsets mapping
    peer_send_offsets = torch.zeros((world_size, world_size), dtype=torch.int32, device=device)
    peer_send_offsets[:, 1:] = torch.cumsum(all_send_splits, dim=1)[:, :-1]

    my_recv_splits = all_send_splits[:, rank].contiguous()
    recv_offsets = torch.zeros(world_size, dtype=torch.int32, device=device)
    recv_offsets[1:] = torch.cumsum(my_recv_splits, dim=0)[:-1]
    
    num_received_tokens = my_recv_splits.sum().item()

    # 5. Extract strictly necessary weighting scalars mapping to the incoming stream
    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    tokens_weight = weights_idx.T.contiguous().masked_select(routing_map.bool())
    
    unpermuted_tokens = torch.zeros(org_hidden_states_shape, dtype=torch.bfloat16, device=device)
    local_input_permutation_mapping = local_input_permutation_mapping.to(torch.int64).contiguous()

    # 6. Fire Unified Fused Pull-and-Unpermute over NVLink
    ext.tk_fused_pull_unpermute(
        send_buffer_tk,
        barrier_tk,
        peer_send_offsets[:, rank].contiguous(),
        recv_offsets.contiguous(),
        tokens_weight.to(torch.bfloat16),
        local_input_permutation_mapping,
        unpermuted_tokens,
        hidden_dim,
        num_received_tokens
    )

    return unpermuted_tokens