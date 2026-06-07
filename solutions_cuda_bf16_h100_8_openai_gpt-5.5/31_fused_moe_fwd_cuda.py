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

__global__ void gather_counts_i64_kernel(
    const long long* __restrict__ ptrs,
    long long* __restrict__ out,
    int E,
    int world_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = E * world_size;
    if (idx >= total) return;
    int r = idx / E;
    int e = idx - r * E;
    const long long* src = reinterpret_cast<const long long*>(
        static_cast<uintptr_t>(ptrs[r])
    );
    out[idx] = src[e];
}

__global__ void alltoall_vec16_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ split_ptrs,
    const long long* __restrict__ out_splits,
    uint4* __restrict__ out,
    int rank,
    int world_size,
    int64_t row_units16,
    int64_t max_units_per_src
) {
    int src_rank = blockIdx.x;
    int64_t linear = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (linear >= max_units_per_src) return;

    int64_t nrows = out_splits[src_rank];
    int64_t total_units = nrows * row_units16;
    if (linear >= total_units) return;

    const long long* remote_splits = reinterpret_cast<const long long*>(
        static_cast<uintptr_t>(split_ptrs[src_rank])
    );

    int64_t remote_row_offset = 0;
    #pragma unroll
    for (int d = 0; d < 16; ++d) {
        if (d >= rank) break;
        remote_row_offset += remote_splits[d];
    }

    int64_t out_row_offset = 0;
    #pragma unroll
    for (int s = 0; s < 16; ++s) {
        if (s >= src_rank) break;
        out_row_offset += out_splits[s];
    }

    const uint4* remote = reinterpret_cast<const uint4*>(
        static_cast<uintptr_t>(data_ptrs[src_rank])
    );

    int64_t row = linear / row_units16;
    int64_t col = linear - row * row_units16;

    int64_t src_index = (remote_row_offset + row) * row_units16 + col;
    int64_t dst_index = (out_row_offset + row) * row_units16 + col;
    out[dst_index] = remote[src_index];
}

template <typename T>
__global__ void alltoall_scalar_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ split_ptrs,
    const long long* __restrict__ out_splits,
    T* __restrict__ out,
    int rank,
    int world_size,
    int64_t H,
    int64_t max_elems_per_src
) {
    int src_rank = blockIdx.x;
    int64_t linear = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (linear >= max_elems_per_src) return;

    int64_t nrows = out_splits[src_rank];
    int64_t total = nrows * H;
    if (linear >= total) return;

    const long long* remote_splits = reinterpret_cast<const long long*>(
        static_cast<uintptr_t>(split_ptrs[src_rank])
    );

    int64_t remote_row_offset = 0;
    #pragma unroll
    for (int d = 0; d < 16; ++d) {
        if (d >= rank) break;
        remote_row_offset += remote_splits[d];
    }

    int64_t out_row_offset = 0;
    #pragma unroll
    for (int s = 0; s < 16; ++s) {
        if (s >= src_rank) break;
        out_row_offset += out_splits[s];
    }

    const T* remote = reinterpret_cast<const T*>(
        static_cast<uintptr_t>(data_ptrs[src_rank])
    );

    int64_t row = linear / H;
    int64_t col = linear - row * H;

    out[(out_row_offset + row) * H + col] =
        remote[(remote_row_offset + row) * H + col];
}

void gather_counts_i64(
    torch::Tensor ptrs,
    torch::Tensor out,
    int E,
    int world_size
) {
    TORCH_CHECK(ptrs.is_cuda() && out.is_cuda(), "CUDA tensors required");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int total = E * world_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gather_counts_i64_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<long long*>(out.data_ptr<int64_t>()),
        E,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_alltoall_copy(
    torch::Tensor data_ptrs,
    torch::Tensor split_ptrs,
    torch::Tensor out_splits,
    torch::Tensor out,
    int rank,
    int world_size,
    int64_t H,
    int64_t max_out_rows,
    int dtype_enum
) {
    TORCH_CHECK(data_ptrs.is_cuda() && split_ptrs.is_cuda(), "ptr tensors must be CUDA");
    TORCH_CHECK(out_splits.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(out.is_contiguous(), "output must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int threads = 256;

    int elem_size = (dtype_enum == 0) ? 2 : 4;
    int64_t row_bytes = H * (int64_t)elem_size;

    const long long* dptrs = reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>());
    const long long* sptrs = reinterpret_cast<const long long*>(split_ptrs.data_ptr<int64_t>());
    const long long* osplits = reinterpret_cast<const long long*>(out_splits.data_ptr<int64_t>());

    if ((row_bytes % 16) == 0) {
        int64_t row_units16 = row_bytes / 16;
        int64_t max_units_per_src = max_out_rows * row_units16;
        int y = (int)((max_units_per_src + threads - 1) / threads);
        if (y < 1) y = 1;
        dim3 grid(world_size, y);
        alltoall_vec16_kernel<<<grid, threads, 0, stream>>>(
            dptrs,
            sptrs,
            osplits,
            reinterpret_cast<uint4*>(out.data_ptr()),
            rank,
            world_size,
            row_units16,
            max_units_per_src
        );
    } else if (dtype_enum == 0) {
        int64_t max_elems_per_src = max_out_rows * H;
        int y = (int)((max_elems_per_src + threads - 1) / threads);
        if (y < 1) y = 1;
        dim3 grid(world_size, y);
        alltoall_scalar_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            dptrs,
            sptrs,
            osplits,
            reinterpret_cast<uint16_t*>(out.data_ptr()),
            rank,
            world_size,
            H,
            max_elems_per_src
        );
    } else {
        int64_t max_elems_per_src = max_out_rows * H;
        int y = (int)((max_elems_per_src + threads - 1) / threads);
        if (y < 1) y = 1;
        dim3 grid(world_size, y);
        alltoall_scalar_kernel<float><<<grid, threads, 0, stream>>>(
            dptrs,
            sptrs,
            osplits,
            reinterpret_cast<float*>(out.data_ptr<float>()),
            rank,
            world_size,
            H,
            max_elems_per_src
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_counts_i64", &gather_counts_i64, "Gather int64 counts through UVA peer pointers");
    m.def("launch_alltoall_copy", &launch_alltoall_copy, "Symmetric-memory variable all-to-all copy");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_moe_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


_count_cache = {}
_a2a_cache = {}
_A2A_CAPACITY_ROWS = None


def _group_rank_world(group):
    return dist.get_rank(group), dist.get_world_size(group)


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16 or dtype == torch.float16:
        return 0
    if dtype == torch.float32:
        return 1
    raise TypeError(f"unsupported dtype for custom all-to-all: {dtype}")


def _get_count_resources(num_experts: int, device: torch.device, group: dist.ProcessGroup):
    rank, world = _group_rank_world(group)
    key = (num_experts, device, rank, world, id(group))
    if key in _count_cache:
        return _count_cache[key]

    buf = symm_mem.empty((num_experts,), device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(buf, group)
    gathered = torch.empty((world, num_experts), device=device, dtype=torch.int64)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _count_cache[key] = (buf, hdl, gathered, ptrs)
    return _count_cache[key]


def _get_a2a_resources(
    capacity_rows: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
):
    rank, world = _group_rank_world(group)
    key = (capacity_rows, hidden_dim, dtype, device, rank, world, id(group))
    if key in _a2a_cache:
        return _a2a_cache[key]

    data_buf = symm_mem.empty((capacity_rows, hidden_dim), device=device, dtype=dtype)
    data_hdl = symm_mem.rendezvous(data_buf, group)

    split_buf = symm_mem.empty((world,), device=device, dtype=torch.int64)
    split_hdl = symm_mem.rendezvous(split_buf, group)

    data_ptrs = torch.tensor(data_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    split_ptrs = torch.tensor(split_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    _a2a_cache[key] = (data_buf, data_hdl, split_buf, split_hdl, data_ptrs, split_ptrs)
    return _a2a_cache[key]


def _preprocess_symm(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
    rank, ep_size = _group_rank_world(ep_group)
    num_local_experts = num_experts // ep_size

    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2)).to(torch.int64).contiguous()

    input_splits = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts)
        .sum(dim=1)
        .tolist()
    )

    cnt_buf, cnt_hdl, gathered, ptrs = _get_count_resources(
        num_experts, expert_mask.device, ep_group
    )
    cnt_buf.copy_(num_local_tokens_per_expert)
    cnt_hdl.barrier(channel=0)

    _get_ext().gather_counts_i64(ptrs, gathered, num_experts, ep_size)
    cnt_hdl.barrier(channel=1)

    start_idx = rank * num_local_experts
    end_idx = (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = gathered[:, start_idx:end_idx].contiguous()

    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    num_global_sum_tokens_per_local_expert = (
        num_global_tokens_per_local_expert.sum(dim=0).to(torch.device("cpu"), non_blocking=False)
    )
    num_global_tokens_per_local_expert_cpu = (
        num_global_tokens_per_local_expert.view(-1, num_local_experts)
        .to(torch.device("cpu"), non_blocking=False)
    )

    return (
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert_cpu,
        num_global_sum_tokens_per_local_expert,
    )


def _symm_all_to_all_impl(
    group: dist.ProcessGroup,
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
) -> torch.Tensor:
    rank, world = _group_rank_world(group)
    if world == 1:
        return input.contiguous()

    assert input.is_cuda
    input = input.contiguous()
    H = input.size(1)

    if output_split_sizes is None:
        output_rows = input.size(0)
        output_split_sizes = [output_rows // world] * world
    else:
        output_rows = int(sum(output_split_sizes))

    if input_split_sizes is None:
        in_rows = input.size(0)
        input_split_sizes = [in_rows // world] * world

    capacity_rows = _A2A_CAPACITY_ROWS
    if capacity_rows is None:
        capacity_rows = max(int(input.size(0)), output_rows) * world
    capacity_rows = max(capacity_rows, int(input.size(0)), output_rows, 1)

    data_buf, data_hdl, split_buf, split_hdl, data_ptrs, split_ptrs = _get_a2a_resources(
        capacity_rows, H, input.dtype, input.device, group
    )

    data_buf[: input.size(0)].copy_(input)

    split_tensor = torch.tensor(input_split_sizes, device=input.device, dtype=torch.int64)
    split_buf.copy_(split_tensor)

    split_hdl.barrier(channel=0)
    data_hdl.barrier(channel=0)

    out = torch.empty((output_rows, H), device=input.device, dtype=input.dtype)
    out_splits_dev = torch.tensor(output_split_sizes, device=input.device, dtype=torch.int64)
    max_out_rows = max(output_split_sizes) if len(output_split_sizes) else 0

    if output_rows > 0 and max_out_rows > 0:
        _get_ext().launch_alltoall_copy(
            data_ptrs,
            split_ptrs,
            out_splits_dev,
            out,
            rank,
            world,
            H,
            int(max_out_rows),
            _dtype_enum(input.dtype),
        )

    data_hdl.barrier(channel=1)
    split_hdl.barrier(channel=1)
    return out


class _SymmAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        return _symm_all_to_all_impl(group, input, output_split_sizes, input_split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            _symm_all_to_all_impl(
                ctx.group,
                grad_output,
                ctx.input_split_sizes,
                ctx.output_split_sizes,
            ),
            None,
            None,
        )


def _all_to_all(
    group: dist.ProcessGroup,
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
) -> torch.Tensor:
    return _SymmAllToAll.apply(group, input, output_split_sizes, input_split_sizes)


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
    num_tokens, _ = routing_weights.shape
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

    expected_tokens = sum(input_splits)
    actual_tokens = local_permuted_hidden_states.shape[0]
    if expected_tokens != actual_tokens:
        raise RuntimeError(
            f"EP split mismatch: input_splits sum ({expected_tokens}) != "
            f"permuted tokens ({actual_tokens})"
        )

    global_permuted_hidden_states = _all_to_all(
        group, local_permuted_hidden_states, output_splits, input_splits
    )

    world = dist.get_world_size(group)
    num_local_experts = num_experts // world
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
    world = dist.get_world_size(group)
    num_local_experts = num_experts // world

    unpermute_order = (
        torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    )
    split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist()

    expert_outputs = _sort_chunks_by_idxs(
        expert_outputs, split_sizes, unpermute_order
    )

    unpermute_outputs = _all_to_all(group, expert_outputs, input_splits, output_splits)

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
    gate = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_proj.weight, gate_proj.bias))
    up = torch.nn.functional.linear(x, up_proj.weight, up_proj.bias)
    return torch.nn.functional.linear(gate * up, down_proj.weight, down_proj.bias)


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
    End-to-end MoE forward with custom symmetric-memory all-to-all in both
    forward and autograd backward. Dense expert math remains autograd-visible.
    """
    global _A2A_CAPACITY_ROWS

    group = group or dist.group.WORLD
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert hidden_states.is_cuda, "CUDA input required"

    _get_ext()

    world = dist.get_world_size(group)
    hidden_dim = hidden_states.size(-1)
    flat_hidden = hidden_states.reshape(-1, hidden_dim)
    num_tokens = flat_hidden.size(0)

    _A2A_CAPACITY_ROWS = max(1, world * num_tokens * top_k)

    router_logits = torch.nn.functional.linear(flat_hidden, gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=num_experts
    ).permute(2, 1, 0)

    input_splits, output_splits, num_global_tokens_per_local_expert, _ = _preprocess_symm(
        expert_mask, num_experts, group
    )

    (
        global_permuted_hidden_states,
        routing_map,
        local_input_permutation_mapping,
        org_hidden_states_shape,
    ) = token_pre_all2all(
        flat_hidden,
        expert_mask,
        num_experts,
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        group,
    )

    expert_outputs = expert_forward(
        global_permuted_hidden_states, gate_proj, up_proj, down_proj
    )

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