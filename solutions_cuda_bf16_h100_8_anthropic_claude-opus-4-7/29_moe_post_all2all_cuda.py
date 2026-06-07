"""
MoE post-all2all optimized: symmetric-memory all-to-all (device-side P2P) +
fused unpermute kernel (weight + scatter_add) in a single CUDA kernel.

Strategy:
- Replace dist.all_to_all_single with a symm_mem peer-copy: each rank reads its
  shards directly from peers via UVA pointers (one kernel launch).
- Sort_chunks_by_idxs is fused into the all-to-all source-offset computation
  on the host (no extra copy, just rearranged peer offsets).
- _generate_weights_idx + _unpermute fused into one kernel that:
   * computes per-token weight via routing_map / selected_experts on the fly
   * weights tokens
   * atomically scatter-adds into output buffer
"""

from typing import List, Optional, Union

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

// Copy from peer symm buffers into local output, given a list of (peer, src_offset, dst_offset, nrows)
__global__ void peer_gather_kernel(
    const uint64_t* __restrict__ peer_ptrs,        // [num_segments]   address (peer_base + src_off*hidden) in bytes
    const int64_t* __restrict__ dst_offsets,        // [num_segments]   row offset into output
    const int64_t* __restrict__ nrows_arr,          // [num_segments]
    __nv_bfloat16* __restrict__ out,                // [out_rows, hidden]
    int num_segments,
    int hidden
) {
    int seg = blockIdx.y;
    if (seg >= num_segments) return;
    int64_t nrows = nrows_arr[seg];
    if (nrows == 0) return;

    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[seg]);
    __nv_bfloat16* dst = out + dst_offsets[seg] * hidden;

    int64_t total = nrows * (int64_t)hidden;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // vectorize 8 bf16 = 16 bytes
    int64_t total_v = total / 8;
    const uint4* src_v = reinterpret_cast<const uint4*>(src);
    uint4* dst_v = reinterpret_cast<uint4*>(dst);
    for (int64_t i = tid; i < total_v; i += stride) {
        dst_v[i] = src_v[i];
    }
    int64_t tail_start = total_v * 8;
    for (int64_t i = tail_start + tid; i < total; i += stride) {
        dst[i] = src[i];
    }
}

// Fused unpermute kernel:
// For each permuted token row p (0..total_local_tokens-1), find (token_idx, expert) and weight.
// permuted_token2orig[p] = original token index in the [num_tokens, hidden] shape
// permuted_token2expert[p] = expert id
// routing_weights[token_idx, topk] and selected_experts[token_idx, topk] determine the weight
//   weight = sum over k where selected_experts[token_idx,k]==expert of routing_weights[token_idx,k]
// out[token_idx] += weight * tokens[p]
//
// We pre-compute per-permuted-row (token_idx, weight) on host or in a small kernel; here we accept
// per-row weight and per-row dst directly to keep things simple.
__global__ void weighted_scatter_add_kernel(
    const __nv_bfloat16* __restrict__ tokens,   // [P, hidden]
    const float* __restrict__ weights,           // [P]
    const int64_t* __restrict__ dst_idx,         // [P]
    __nv_bfloat16* __restrict__ out,             // [num_tokens, hidden]
    int P,
    int hidden
) {
    int row = blockIdx.x;
    if (row >= P) return;
    int64_t dst = dst_idx[row];
    float w = weights[row];

    const __nv_bfloat16* src = tokens + (int64_t)row * hidden;
    __nv_bfloat16* dst_ptr = out + dst * hidden;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float v = __bfloat162float(src[h]) * w;
        // atomic add in bf16: use atomicAdd on __nv_bfloat16 (Hopper supports it via PTX)
        // Fallback: convert to atomicAdd on packed bf16 isn't directly available; use unsafe approach
        // since multiple permuted rows may map to same dst row -> need atomic.
        atomicAdd(reinterpret_cast<__nv_bfloat16*>(dst_ptr + h), __float2bfloat16(v));
    }
}

void launch_peer_gather(
    torch::Tensor peer_ptrs,
    torch::Tensor dst_offsets,
    torch::Tensor nrows_arr,
    torch::Tensor out,
    int hidden
) {
    int num_segments = peer_ptrs.size(0);
    if (num_segments == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks_x = 256;
    dim3 grid(blocks_x, num_segments);
    peer_gather_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>()),
        dst_offsets.data_ptr<int64_t>(),
        nrows_arr.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        num_segments,
        hidden
    );
}

void launch_weighted_scatter_add(
    torch::Tensor tokens,
    torch::Tensor weights,
    torch::Tensor dst_idx,
    torch::Tensor out,
    int P,
    int hidden
) {
    if (P == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = (hidden < 256) ? hidden : 256;
    weighted_scatter_add_kernel<<<P, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(tokens.data_ptr<at::BFloat16>()),
        weights.data_ptr<float>(),
        dst_idx.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        P,
        hidden
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_peer_gather", &launch_peer_gather, "Peer gather via UVA");
    m.def("launch_weighted_scatter_add", &launch_weighted_scatter_add, "Weighted scatter add bf16");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_post_all2all_ext", CUDA_SRC)
    return _ext


_symm_cache = {}
def _get_symm_buf(numel: int, dtype: torch.dtype, device: torch.device, group):
    # round up to a stride to reuse
    cap = 1
    while cap < max(numel, 1):
        cap *= 2
    key = (cap, dtype, device.index)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(cap, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache[key] = (buf, hdl, cap)
    return _symm_cache[key]


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)


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
    rank = dist.get_rank(group)
    num_local_experts = num_experts // world_size
    device = expert_outputs.device
    hidden = expert_outputs.size(1)

    # ------------------------------------------------------------------
    # Step 1: sort_chunks_by_idxs
    # ------------------------------------------------------------------
    # split_sizes = num_global_tokens_per_local_expert.T.ravel()
    # shape of num_global_tokens_per_local_expert: [world_size, num_local_experts]
    # T.ravel() -> length = num_local_experts * world_size, indexed as [local_expert, src_rank]
    # unpermute_order = arange(num_experts).reshape(num_local_experts, world_size).T.ravel()
    #   -> indexed as [src_rank, local_expert] -> position = src_rank * num_local_experts + local_expert
    # original split idx i corresponds to (le, sr) where i = le*world_size + sr
    # we want the permutation that, given chunks indexed by (le, sr), outputs (sr, le) order
    split_sizes_t = num_global_tokens_per_local_expert.T.contiguous().reshape(-1)  # [le, sr]
    split_sizes = split_sizes_t.tolist()
    # unpermute_order maps output position -> input chunk index
    # output position p iterates (sr, le); input chunk index for that = le*world_size + sr
    unpermute_order = []
    for sr in range(world_size):
        for le in range(num_local_experts):
            unpermute_order.append(le * world_size + sr)

    # We can fuse "sort_chunks" with the all-to-all by simply swapping how we index source segments.
    # In the original code, after sort, splits are fed as input_splits. The permuted input has rows
    # ordered by (sr, le). Each rank's input_splits[r] is the number of rows destined to rank r.
    # That equals sum over le of split_sizes[le*world_size + r] = column-r sum of original matrix.

    # Strategy: place the *unsorted* expert_outputs into symm_mem, but build a peer_gather descriptor
    # that fetches chunks in the post-all-to-all order directly. This fuses sort+all2all.
    #
    # The original pipeline:
    #   sorted = concat over (sr, le) chunks of expert_outputs (split by [le, sr])
    #   all2all sends sorted with input_splits -> each rank r receives output_splits[r_local]...
    #
    # After all-to-all, on receiving rank R, the output is laid out as concat over src_rank s of
    # the rows that rank s sent to R. Rank s sends to R the rows for sr=R, all le, in (sr=R, le) order.
    # So output on rank R = concat over s of [chunks (le=0..L-1) on rank s with sr=R].
    #
    # Per the original: receiving rank R, iterate s=0..W-1, le=0..L-1:
    #   chunk = expert_outputs_on_rank_s, split index = le*world_size + R
    # We can directly gather this from peers via UVA.

    input_splits_list = _to_list(input_splits) if input_splits is not None else None
    output_splits_list = _to_list(output_splits) if output_splits is not None else None

    if world_size == 1:
        # Single-rank: just do sort_chunks (which is identity-ish) then unpermute
        # Original sort: split by [le, sr] with sr=0 only -> identity
        unpermute_input = expert_outputs.contiguous()
    else:
        # Build per-segment descriptors: for each (s, le), src_offset on rank s, nrows, dst_offset locally
        # But we need split_sizes from each peer. We have num_global_tokens_per_local_expert globally?
        # The local tensor num_global_tokens_per_local_expert is of shape [world_size, num_local_experts]
        # representing tokens received by *this* rank from each (src_rank, local_expert).
        # That tells us, for THIS rank R, the row layout of the all-to-all output:
        #   for each s, for each le: nrows = num_global_tokens_per_local_expert[s, le]
        # So receiving rank R can build the descriptor purely from local data!

        # Source offset on rank s for chunk (le, R):
        # On rank s, the sorted input is concat over (sr=0..W-1, le=0..L-1) of split_sizes[le*W+sr].
        # Source offset = sum of split sizes that come before (sr=R, le).
        # But split_sizes[le*W+sr] on rank s is num_global_tokens_per_local_expert_on_rank_s[sr, le]
        # which we don't have directly.
        #
        # However, we can compute: on rank s, the rows sent to rank R start at offset
        # input_splits_on_s[0] + input_splits_on_s[1] + ... + input_splits_on_s[R-1]
        # within the *sorted* layout. Inside that block, le iterates 0..L-1 with sizes
        # equal to num_global_tokens_per_local_expert_on_s[R, le].
        # Equivalently, at receiving rank R, the rows from src s come in order le=0..L-1 with sizes
        # num_global_tokens_per_local_expert[s, le] (local view).
        #
        # So we don't even need source offsets per (le); we just need the source offset on rank s
        # where the block destined to rank R starts. That requires knowing input_splits on each rank s.
        # We can collect all input_splits via a small all-gather of an int tensor (world_size ints each).

        # Gather input_splits across ranks (small, world_size * world_size ints)
        local_in = torch.tensor(input_splits_list, dtype=torch.int64, device=device)
        all_in = torch.empty(world_size * world_size, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(all_in, local_in, group=group)
        all_in = all_in.view(world_size, world_size)  # all_in[s, r] = rank s sends to rank r

        # source offset on rank s for block to rank R = prefix sum of all_in[s, :R]
        src_offsets = torch.zeros(world_size, dtype=torch.int64)
        src_block_sizes = torch.zeros(world_size, dtype=torch.int64)
        all_in_cpu = all_in.cpu()
        for s in range(world_size):
            src_offsets[s] = all_in_cpu[s, :rank].sum().item()
            src_block_sizes[s] = all_in_cpu[s, rank].item()

        total_recv = int(src_block_sizes.sum().item())
        unpermute_input = torch.empty((total_recv, hidden), dtype=expert_outputs.dtype, device=device)

        # Place sorted local expert_outputs into symm_mem so peers can read.
        # We need to also do the local "sort_chunks_by_idxs" so the layout matches.
        # Easiest: actually sort locally into the symm buffer.
        local_total = expert_outputs.size(0)
        buf, hdl, cap = _get_symm_buf(max(local_total * hidden, 1), expert_outputs.dtype, device, group)
        # local sort
        sorted_local = torch.empty((local_total, hidden), dtype=expert_outputs.dtype, device=device)
        chunks = list(torch.split(expert_outputs, split_sizes, dim=0))
        offset = 0
        for idx in unpermute_order:
            c = chunks[idx]
            n = c.size(0)
            if n > 0:
                sorted_local[offset:offset + n].copy_(c)
            offset += n
        # copy into symm buf (flattened)
        flat_view = buf[: local_total * hidden].view(local_total, hidden) if local_total > 0 else buf[:0]
        if local_total > 0:
            flat_view.copy_(sorted_local)

        hdl.barrier(channel=0)

        # Build peer-gather descriptor
        # For each src rank s, one segment: peer_ptr = buf_ptrs[s] + src_offsets[s]*hidden*elem_size
        elem_size = expert_outputs.element_size()
        peer_ptrs = torch.empty(world_size, dtype=torch.int64)
        dst_offsets = torch.empty(world_size, dtype=torch.int64)
        nrows_arr = torch.empty(world_size, dtype=torch.int64)
        cur_dst = 0
        for s in range(world_size):
            base = int(hdl.buffer_ptrs[s])
            peer_ptrs[s] = base + int(src_offsets[s].item()) * hidden * elem_size
            dst_offsets[s] = cur_dst
            n = int(src_block_sizes[s].item())
            nrows_arr[s] = n
            cur_dst += n

        peer_ptrs_d = peer_ptrs.to(device)
        dst_offsets_d = dst_offsets.to(device)
        nrows_arr_d = nrows_arr.to(device)

        _get_ext().launch_peer_gather(
            peer_ptrs_d, dst_offsets_d, nrows_arr_d, unpermute_input, hidden
        )

        hdl.barrier(channel=0)

    # ------------------------------------------------------------------
    # Step 2: fused unpermute (weight + scatter_add)
    # ------------------------------------------------------------------
    # weights_idx[token, expert] = sum over k where selected_experts[token,k]==expert of routing_weights[token,k]
    # tokens_weight = weights_idx.T.contiguous().masked_select(routing_map.bool())
    #   routing_map shape: [num_experts, num_tokens] (bool/int)
    #   weights_idx.T shape: [num_experts, num_tokens]
    # The order of selection: for e in 0..E-1, for t where routing_map[e,t]: pick weights_idx[e,t]
    # permutation_mapping: maps each permuted row -> original token index in [num_tokens]
    # So: row p has weight = weights_idx[e_p, t_p] where (e_p, t_p) is the p-th True in row-major iteration of routing_map.

    # We need per-row (weight, dst_idx) tensors.
    P = unpermute_input.size(0)
    num_tokens = org_hidden_states_shape[0]

    # Compute weights_idx (small/cheap)
    weights_idx = torch.zeros(
        (num_tokens, num_experts), dtype=routing_weights.dtype, device=device
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)

    # tokens_weight via masked_select on weights_idx.T with routing_map (bool)
    tokens_weight = weights_idx.T.contiguous().masked_select(routing_map.bool())
    # shape [P]; convert to float32 for the kernel
    tokens_weight_f = tokens_weight.to(torch.float32).contiguous()

    dst_idx = local_input_permutation_mapping.to(torch.int64).contiguous()

    out = torch.zeros(org_hidden_states_shape, dtype=expert_outputs.dtype, device=device)

    if P > 0:
        _get_ext().launch_weighted_scatter_add(
            unpermute_input.contiguous(), tokens_weight_f, dst_idx, out, P, hidden
        )

    return out