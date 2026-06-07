"""
MoE narrow EP forward, with custom CUDA + symmetric memory replacing NCCL collectives:
- Metadata all-gather: symm_mem buffer + device-side copy kernel.
- Token all-to-all (forward + backward): symm_mem buffer + device-side
  per-peer block copy kernel reading remote UVA pointers.
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
#include <cstdint>

// Gather flat int64 tensors from each rank's symmetric buffer into a contiguous output.
// Each peer contributes `n_per_rank` int64 elements at offset 0.
__global__ void gather_int64_kernel(
    const long long* __restrict__ peer_ptrs,
    long long* __restrict__ out,
    int world_size,
    int n_per_rank
) {
    int r = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= world_size || idx >= n_per_rank) return;
    const long long* src = (const long long*)peer_ptrs[r];
    out[r * n_per_rank + idx] = src[idx];
}

void launch_gather_int64(
    torch::Tensor peer_ptrs,   // [world_size] int64 (device pointers as int64)
    torch::Tensor out,         // [world_size * n_per_rank] int64
    int world_size,
    int n_per_rank
) {
    const long long* d_ptrs = (const long long*)peer_ptrs.data_ptr<int64_t>();
    int threads = 128;
    int blocks_x = (n_per_rank + threads - 1) / threads;
    dim3 grid(blocks_x, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_int64_kernel<<<grid, threads, 0, stream>>>(
        d_ptrs, out.data_ptr<int64_t>(), world_size, n_per_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// All-to-all of variable-length token rows in BF16.
// Each rank holds an input buffer of contiguous rows in symmetric memory.
// `input_splits[i]` = number of rows this rank sends to rank i (also = number of
// rows rank i pulls from this rank). Layout in input buf: rows for rank 0 first,
// then rank 1, etc. with cumulative offsets `input_offsets`.
//
// `output_splits[i]` = number of rows this rank receives from rank i.
// Output rows are placed contiguously: [from rank0 | from rank1 | ...]. The
// per-peer offsets in the *peer's* input buffer for our portion are computed
// from the gathered metadata (peer_input_offsets_for_me).
//
// We launch a 2D grid: (blocks per row chunk, world_size). Each y-block handles
// one peer; we copy `output_splits[peer]` rows of `hidden_dim` BF16 elements
// from peer input buffer to local output buffer.
__global__ void all_to_all_bf16_kernel(
    const long long* __restrict__ peer_input_ptrs, // [world_size] device ptrs to peers' input bufs
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ output_splits,           // [world_size]
    const int* __restrict__ output_offsets,          // [world_size] cum sum
    const int* __restrict__ peer_input_offsets_for_me, // [world_size] offset (in rows) inside peer's input buf
    int hidden_dim,
    int world_size
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;
    int rows = output_splits[peer];
    if (rows == 0) return;

    int out_row_off = output_offsets[peer];
    int in_row_off = peer_input_offsets_for_me[peer];

    const __nv_bfloat16* src = (const __nv_bfloat16*)peer_input_ptrs[peer];
    src += (size_t)in_row_off * hidden_dim;
    __nv_bfloat16* dst = out + (size_t)out_row_off * hidden_dim;

    int total = rows * hidden_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Vectorized copy via int4 (8 bf16 = 16 bytes) when aligned.
    if ((hidden_dim % 8) == 0 &&
        ((uintptr_t)src % 16 == 0) && ((uintptr_t)dst % 16 == 0)) {
        int total_v = total / 8;
        const int4* src4 = reinterpret_cast<const int4*>(src);
        int4* dst4 = reinterpret_cast<int4*>(dst);
        for (int i = tid; i < total_v; i += stride) {
            dst4[i] = src4[i];
        }
    } else {
        for (int i = tid; i < total; i += stride) {
            dst[i] = src[i];
        }
    }
}

void launch_all_to_all_bf16(
    torch::Tensor peer_input_ptrs,           // [world_size] int64
    torch::Tensor out,                       // [out_rows, hidden_dim] bf16
    torch::Tensor output_splits,             // [world_size] int32
    torch::Tensor output_offsets,            // [world_size] int32
    torch::Tensor peer_input_offsets_for_me, // [world_size] int32
    int hidden_dim,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks_x = 64;
    dim3 grid(blocks_x, world_size);
    all_to_all_bf16_kernel<<<grid, threads, 0, stream>>>(
        (const long long*)peer_input_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        output_splits.data_ptr<int>(),
        output_offsets.data_ptr<int>(),
        peer_input_offsets_for_me.data_ptr<int>(),
        hidden_dim,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// FP32 variant (for fallback / non-bf16 dtypes if needed).
__global__ void all_to_all_f32_kernel(
    const long long* __restrict__ peer_input_ptrs,
    float* __restrict__ out,
    const int* __restrict__ output_splits,
    const int* __restrict__ output_offsets,
    const int* __restrict__ peer_input_offsets_for_me,
    int hidden_dim,
    int world_size
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;
    int rows = output_splits[peer];
    if (rows == 0) return;

    int out_row_off = output_offsets[peer];
    int in_row_off = peer_input_offsets_for_me[peer];

    const float* src = (const float*)peer_input_ptrs[peer];
    src += (size_t)in_row_off * hidden_dim;
    float* dst = out + (size_t)out_row_off * hidden_dim;

    int total = rows * hidden_dim;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if ((hidden_dim % 4) == 0 &&
        ((uintptr_t)src % 16 == 0) && ((uintptr_t)dst % 16 == 0)) {
        int total_v = total / 4;
        const float4* src4 = reinterpret_cast<const float4*>(src);
        float4* dst4 = reinterpret_cast<float4*>(dst);
        for (int i = tid; i < total_v; i += stride) {
            dst4[i] = src4[i];
        }
    } else {
        for (int i = tid; i < total; i += stride) {
            dst[i] = src[i];
        }
    }
}

void launch_all_to_all_f32(
    torch::Tensor peer_input_ptrs,
    torch::Tensor out,
    torch::Tensor output_splits,
    torch::Tensor output_offsets,
    torch::Tensor peer_input_offsets_for_me,
    int hidden_dim,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks_x = 64;
    dim3 grid(blocks_x, world_size);
    all_to_all_f32_kernel<<<grid, threads, 0, stream>>>(
        (const long long*)peer_input_ptrs.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        output_splits.data_ptr<int>(),
        output_offsets.data_ptr<int>(),
        peer_input_offsets_for_me.data_ptr<int>(),
        hidden_dim,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_int64", &launch_gather_int64, "symm-mem int64 all-gather");
    m.def("launch_all_to_all_bf16", &launch_all_to_all_bf16, "symm-mem bf16 all-to-all");
    m.def("launch_all_to_all_f32", &launch_all_to_all_f32, "symm-mem f32 all-to-all");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_ep_narrow_symm_ext", CUDA_SRC)
    return _ext


# ---- EP subgroup resolution ----

_EP_SUBGROUP_CACHE: dict[tuple[int, int], None | list] = {}


def _resolve_ep_group_for_narrow_moe(num_experts: int) -> dist.ProcessGroup:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    ws = dist.get_world_size()
    rank = dist.get_rank()
    key = (ws, num_experts)
    if key not in _EP_SUBGROUP_CACHE:
        if num_experts >= ws:
            _EP_SUBGROUP_CACHE[key] = None
        elif ws % num_experts != 0:
            raise ValueError(
                f"narrow EP requires world_size ({ws}) % num_experts ({num_experts}) == 0"
            )
        else:
            groups: list = []
            for r in range(ws // num_experts):
                ranks = list(range(r * num_experts, (r + 1) * num_experts))
                groups.append(dist.new_group(ranks))
            _EP_SUBGROUP_CACHE[key] = groups
    entry = _EP_SUBGROUP_CACHE[key]
    if entry is None:
        return dist.group.WORLD
    return entry[rank // num_experts]


# ---- Symmetric memory caches ----
# We need a metadata symm buffer (int64) and a token symm buffer (bf16/f32).
# Token buffer is sized to the max rows seen so far. Subgroups have separate caches.

_META_CACHE: dict = {}      # group_key -> (buf, hdl, ptrs_tensor, n_slots)
_TOKEN_CACHE: dict = {}     # (group_key, dtype) -> (buf, hdl, ptrs_tensor, capacity_rows, hidden_dim)


def _group_key(group: dist.ProcessGroup) -> int:
    # ProcessGroup objects aren't hashable by content, but identity works for caching.
    return id(group)


def _get_meta_buf(group: dist.ProcessGroup, n_slots: int, device: torch.device):
    key = _group_key(group)
    entry = _META_CACHE.get(key)
    if entry is not None and entry[3] >= n_slots:
        return entry
    # (Re)allocate. Choose >= 256 slots and grow geometrically.
    cap = max(256, n_slots)
    if entry is not None:
        cap = max(cap, entry[3] * 2)
    buf = symm_mem.empty(cap, device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    entry = (buf, hdl, ptrs_tensor, cap)
    _META_CACHE[key] = entry
    return entry


def _get_token_buf(group: dist.ProcessGroup, rows: int, hidden_dim: int,
                   dtype: torch.dtype, device: torch.device):
    key = (_group_key(group), dtype, hidden_dim)
    entry = _TOKEN_CACHE.get(key)
    if entry is not None and entry[3] >= rows:
        return entry
    cap = max(rows, 64)
    if entry is not None:
        cap = max(cap, entry[3] * 2)
    buf = symm_mem.empty((cap, hidden_dim), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    entry = (buf, hdl, ptrs_tensor, cap, hidden_dim)
    _TOKEN_CACHE[key] = entry
    return entry


# ---- Symm-mem int64 all-gather (metadata) ----

def _symm_all_gather_int64(local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather a 1D int64 tensor from each rank in `group`. Returns [world_size * n] flat tensor."""
    n = local.numel()
    ws = group.size()
    device = local.device
    buf, hdl, ptrs_tensor, cap = _get_meta_buf(group, n, device)
    # Copy local into symm buf
    buf[:n].copy_(local.view(-1).to(torch.int64))
    hdl.barrier(channel=0)
    out = torch.empty(ws * n, device=device, dtype=torch.int64)
    _get_ext().launch_gather_int64(ptrs_tensor, out, ws, n)
    hdl.barrier(channel=1)
    return out


# ---- Symm-mem all-to-all with autograd ----

def _symm_all_to_all_forward(
    input_rows: torch.Tensor,        # [N_in, H]
    input_splits: List[int],
    output_splits: List[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Custom symm-mem all-to-all of variable-row chunks. Returns [N_out, H] tensor."""
    ws = group.size()
    rank = dist.get_rank(group)
    H = input_rows.size(-1)
    dtype = input_rows.dtype
    device = input_rows.device

    n_in = int(sum(input_splits))
    n_out = int(sum(output_splits))

    # Get / allocate token symm buffer big enough for n_in rows
    buf, hdl, ptrs_tensor, cap, _ = _get_token_buf(group, max(n_in, 1), H, dtype, device)

    # Compute input offsets (row-wise) for our buf layout.
    in_off = [0]
    for s in input_splits:
        in_off.append(in_off[-1] + int(s))

    # Copy input rows into symm buffer at offsets [0, in_off[1], in_off[2], ...].
    # Since input_rows is already laid out as [to rank0 | to rank1 | ...],
    # we can do a single copy of n_in rows.
    if n_in > 0:
        buf[:n_in].copy_(input_rows)

    # We also need each peer's input_splits so we know the offset from which we
    # pull our portion in their buffer. Gather input_splits across ranks via
    # symm-mem int64 all-gather.
    local_splits_t = torch.tensor(input_splits, device=device, dtype=torch.int64)
    gathered = _symm_all_gather_int64(local_splits_t, group)  # [ws*ws] flat
    gathered = gathered.view(ws, ws)  # gathered[r, j] = rank r's input_split for rank j
    # Our pull offset within peer r's buf: sum of gathered[r, :rank]
    pull_off = torch.zeros(ws, device=device, dtype=torch.int32)
    if ws > 0:
        cum = torch.cumsum(gathered, dim=1, dtype=torch.int64)  # [ws, ws]
        # For rank rk, peer r contributes gathered[r, rk] rows at offset cum[r, rk-1] (or 0 if rk=0)
        if rank == 0:
            pull_off.zero_()
        else:
            pull_off.copy_(cum[:, rank - 1].to(torch.int32))

    out_splits_t = torch.tensor(output_splits, device=device, dtype=torch.int32)
    out_off_t = torch.zeros(ws, device=device, dtype=torch.int32)
    out_off_t[1:] = torch.cumsum(out_splits_t[:-1], dim=0)

    out = torch.empty((n_out, H), device=device, dtype=dtype)

    # Wait for all peers to finish writing into their symm token bufs.
    hdl.barrier(channel=0)

    if dtype == torch.bfloat16:
        _get_ext().launch_all_to_all_bf16(
            ptrs_tensor, out, out_splits_t, out_off_t, pull_off, H, ws
        )
    elif dtype == torch.float32:
        _get_ext().launch_all_to_all_f32(
            ptrs_tensor, out, out_splits_t, out_off_t, pull_off, H, ws
        )
    else:
        # Fallback: cast to f32, run, cast back.
        out_f32 = torch.empty((n_out, H), device=device, dtype=torch.float32)
        # We can't reuse buf (wrong dtype). Use NCCL fallback.
        hdl.barrier(channel=1)
        out_native = torch.empty_like(out)
        dist.all_to_all_single(
            out_native, input_rows.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return out_native

    # Ensure all reads done before any rank reuses its buffer.
    hdl.barrier(channel=1)
    return out


class _SymmAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        if dist.get_world_size(group=group) == 1:
            return input.contiguous()
        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split: assume input.size(0) divisible by world_size.
            ws = dist.get_world_size(group)
            assert input.size(0) % ws == 0
            per = input.size(0) // ws
            input_split_sizes_eff = [per] * ws
            output_split_sizes_eff = [per] * ws
        else:
            input_split_sizes_eff = list(input_split_sizes)
            output_split_sizes_eff = list(output_split_sizes)
        return _symm_all_to_all_forward(
            input, input_split_sizes_eff, output_split_sizes_eff, group
        )

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            _SymmAllToAll.apply(
                ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes
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


# ----- Preprocess (symm-mem all-gather of metadata) -----

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
    num_local_tokens_per_expert_flat = num_local_tokens_per_expert.contiguous().view(-1).to(torch.int64)
    # symm-mem all-gather
    gathered_flat = _symm_all_gather_int64(num_local_tokens_per_expert_flat, ep_group)
    num_global_tokens_per_expert = gathered_flat.view(
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


# ----- Permute / sort / unpermute / weights -----

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


# ----- Token pre/post all2all -----

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
    if group is None:
        group = _resolve_ep_group_for_narrow_moe(num_experts)

    # Eager compile the extension (rank 0 first to avoid race), then barrier.
    if dist.is_initialized():
        if dist.get_rank() == 0:
            _get_ext()
        dist.barrier()
        _get_ext()

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

    input_splits, output_splits, num_global_tokens_per_local_expert, _ = _preprocess(
        expert_mask, num_experts, group
    )

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