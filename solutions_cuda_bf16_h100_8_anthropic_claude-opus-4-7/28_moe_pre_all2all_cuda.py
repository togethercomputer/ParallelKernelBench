"""
MoE EP token_pre_all2all using symmetric memory + custom CUDA kernels.
- Fused permute (gather by routing_map) into a symm_mem send buffer.
- Device-side all-to-all via UVA peer reads from symmetric memory.
- Fused chunk-reorder (sort_chunks_by_idxs) on device.
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

// ---------------------------------------------------------------
// Build sorted_indices and permuted tokens from routing_map
// routing_map: [num_experts, num_tokens] int8/bool (0/1)
// We compute for each expert e the list of token indices where mask=1,
// concatenated in expert-major order.
// ---------------------------------------------------------------

// Phase 1: per-expert exclusive prefix offsets (counts) - computed on host or device.
// Phase 2: scatter token indices into sorted_indices, then index_select.

// Single-kernel: each expert handled by a block; block-wide scan.
__global__ void permute_build_kernel(
    const uint8_t* __restrict__ routing_map, // [E, N]
    const __nv_bfloat16* __restrict__ tokens, // [N, H]
    __nv_bfloat16* __restrict__ out_tokens,   // [total, H]
    int* __restrict__ sorted_indices,          // [total]
    const int* __restrict__ expert_offsets,    // [E] start offsets
    int num_tokens,
    int hidden
) {
    int e = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    const uint8_t* row = routing_map + (size_t)e * num_tokens;
    int base = expert_offsets[e];

    // Iterate in tiles. Use block-wide scan for indices.
    extern __shared__ int smem[];
    int* s_scan = smem; // size bs+1

    int local_count_total = 0;
    for (int tile = 0; tile < num_tokens; tile += bs) {
        int idx = tile + tid;
        int v = (idx < num_tokens) ? (int)row[idx] : 0;
        // exclusive scan
        s_scan[tid] = v;
        __syncthreads();
        // simple Hillis-Steele scan
        for (int off = 1; off < bs; off <<= 1) {
            int x = (tid >= off) ? s_scan[tid - off] : 0;
            __syncthreads();
            s_scan[tid] += x;
            __syncthreads();
        }
        int incl = s_scan[tid];
        int excl = incl - v;
        int total = s_scan[bs - 1];
        if (v && idx < num_tokens) {
            sorted_indices[base + local_count_total + excl] = idx;
        }
        local_count_total += total;
        __syncthreads();
    }
}

__global__ void gather_tokens_kernel(
    const __nv_bfloat16* __restrict__ tokens, // [N, H]
    const int* __restrict__ sorted_indices,    // [M]
    __nv_bfloat16* __restrict__ out,           // [M, H]
    int M, int H
) {
    int row = blockIdx.x;
    if (row >= M) return;
    int src = sorted_indices[row];
    const __nv_bfloat16* sp = tokens + (size_t)src * H;
    __nv_bfloat16* dp = out + (size_t)row * H;
    // vector copy as int4
    int H4 = H / 8; // 8 bf16 per int4
    const int4* sp4 = reinterpret_cast<const int4*>(sp);
    int4* dp4 = reinterpret_cast<int4*>(dp);
    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        dp4[i] = sp4[i];
    }
    int tail_start = H4 * 8;
    for (int i = tail_start + threadIdx.x; i < H; i += blockDim.x) {
        dp[i] = sp[i];
    }
}

// All-to-all via UVA peer reads from a symmetric send buffer.
// Each rank reads its slice from each peer and writes contiguously into local out.
// in_offsets[r] = starting row in peer r's send buffer destined to this rank
// in_sizes[r]   = number of rows from peer r
// out_offsets[r]= starting row in local out for chunk from peer r
__global__ void a2a_read_kernel(
    const uint64_t* __restrict__ peer_send_ptrs, // [W] pointers to peers' send buffers
    __nv_bfloat16* __restrict__ out,              // [out_total, H]
    const int* __restrict__ in_offsets,            // [W]
    const int* __restrict__ in_sizes,              // [W]
    const int* __restrict__ out_offsets,           // [W]
    int world_size,
    int H
) {
    int r = blockIdx.y;
    int row_in_chunk = blockIdx.x;
    int sz = in_sizes[r];
    if (row_in_chunk >= sz) return;
    const __nv_bfloat16* peer_buf = reinterpret_cast<const __nv_bfloat16*>(peer_send_ptrs[r]);
    int src_row = in_offsets[r] + row_in_chunk;
    int dst_row = out_offsets[r] + row_in_chunk;
    const __nv_bfloat16* sp = peer_buf + (size_t)src_row * H;
    __nv_bfloat16* dp = out + (size_t)dst_row * H;
    int H4 = H / 8;
    const int4* sp4 = reinterpret_cast<const int4*>(sp);
    int4* dp4 = reinterpret_cast<int4*>(dp);
    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        dp4[i] = sp4[i];
    }
    int tail_start = H4 * 8;
    for (int i = tail_start + threadIdx.x; i < H; i += blockDim.x) {
        dp[i] = sp[i];
    }
}

// Reorder chunks: given chunks of sizes split_sizes laid out in `in`,
// produce `out` formed by concatenating chunks[order[i]] in order.
__global__ void reorder_chunks_kernel(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ src_starts,  // [K] start of each src chunk in `in`
    const int* __restrict__ dst_starts,  // [K] start of each dst chunk in `out`
    const int* __restrict__ chunk_sizes, // [K] in row order of dst (i.e., size of order[i])
    int K, int H
) {
    int k = blockIdx.y;
    int row_in_chunk = blockIdx.x;
    int sz = chunk_sizes[k];
    if (row_in_chunk >= sz) return;
    int src_row = src_starts[k] + row_in_chunk;
    int dst_row = dst_starts[k] + row_in_chunk;
    const __nv_bfloat16* sp = in + (size_t)src_row * H;
    __nv_bfloat16* dp = out + (size_t)dst_row * H;
    int H4 = H / 8;
    const int4* sp4 = reinterpret_cast<const int4*>(sp);
    int4* dp4 = reinterpret_cast<int4*>(dp);
    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        dp4[i] = sp4[i];
    }
    int tail_start = H4 * 8;
    for (int i = tail_start + threadIdx.x; i < H; i += blockDim.x) {
        dp[i] = sp[i];
    }
}

void launch_permute_build(
    torch::Tensor routing_map_u8, // [E, N]
    torch::Tensor expert_offsets, // [E] int32
    torch::Tensor sorted_indices, // [total] int32
    int num_tokens
) {
    int E = routing_map_u8.size(0);
    int bs = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    permute_build_kernel<<<E, bs, sizeof(int)*(bs+1), stream>>>(
        routing_map_u8.data_ptr<uint8_t>(),
        nullptr, nullptr,
        sorted_indices.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        num_tokens, 0
    );
}

void launch_gather_tokens(
    torch::Tensor tokens,          // [N, H] bf16
    torch::Tensor sorted_indices,  // [M] int32
    torch::Tensor out              // [M, H] bf16
) {
    int M = sorted_indices.size(0);
    int H = tokens.size(1);
    if (M == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_tokens_kernel<<<M, 128, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(tokens.data_ptr<at::BFloat16>()),
        sorted_indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        M, H
    );
}

void launch_a2a_read(
    torch::Tensor peer_send_ptrs,
    torch::Tensor out,
    torch::Tensor in_offsets,
    torch::Tensor in_sizes,
    torch::Tensor out_offsets,
    int world_size,
    int max_chunk_rows,
    int H
) {
    if (max_chunk_rows == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(max_chunk_rows, world_size);
    a2a_read_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(peer_send_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        in_offsets.data_ptr<int>(),
        in_sizes.data_ptr<int>(),
        out_offsets.data_ptr<int>(),
        world_size, H
    );
}

void launch_reorder_chunks(
    torch::Tensor in,
    torch::Tensor out,
    torch::Tensor src_starts,
    torch::Tensor dst_starts,
    torch::Tensor chunk_sizes,
    int max_chunk_rows,
    int H
) {
    int K = src_starts.size(0);
    if (K == 0 || max_chunk_rows == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(max_chunk_rows, K);
    reorder_chunks_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(in.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        src_starts.data_ptr<int>(),
        dst_starts.data_ptr<int>(),
        chunk_sizes.data_ptr<int>(),
        K, H
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_permute_build", &launch_permute_build, "");
    m.def("launch_gather_tokens", &launch_gather_tokens, "");
    m.def("launch_a2a_read", &launch_a2a_read, "");
    m.def("launch_reorder_chunks", &launch_reorder_chunks, "");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_pre_a2a_ext", CUDA_SRC)
    return _ext


_send_buf_cache = {}
_recv_buf_cache = {}


def _get_send_buf(rows: int, hidden: int, dtype, device):
    key = (rows, hidden, dtype, device)
    if key in _send_buf_cache:
        return _send_buf_cache[key]
    buf = symm_mem.empty((rows, hidden), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _send_buf_cache[key] = (buf, hdl, ptrs)
    return _send_buf_cache[key]


def _get_recv_buf(rows: int, hidden: int, dtype, device):
    key = (rows, hidden, dtype, device)
    if key in _recv_buf_cache:
        return _recv_buf_cache[key]
    out = torch.empty((rows, hidden), dtype=dtype, device=device)
    _recv_buf_cache[key] = out
    return out


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
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim).contiguous()
    org_hidden_states_shape = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # routing_map: [E, N]
    routing_map = expert_mask.sum(dim=1)
    routing_map_bool = routing_map.bool()
    E, N = routing_map_bool.shape

    # Normalize splits to lists (host) - small, CPU side
    if isinstance(input_splits, torch.Tensor):
        input_splits_list = input_splits.tolist()
    else:
        input_splits_list = list(input_splits)
    if isinstance(output_splits, torch.Tensor):
        output_splits_list = output_splits.tolist()
    else:
        output_splits_list = list(output_splits)

    total_in = sum(input_splits_list)
    total_out = sum(output_splits_list)

    ext = _get_ext()

    # ---- Permute: build sorted_indices ----
    # per-expert counts -> exclusive prefix sums (host computed; small E)
    counts = routing_map_bool.sum(dim=1)  # [E] on device
    counts_cpu = counts.to('cpu', non_blocking=False)
    counts_list = counts_cpu.tolist()
    expert_offsets_list = [0] * E
    s = 0
    for i in range(E):
        expert_offsets_list[i] = s
        s += counts_list[i]
    total_local = s

    if total_local != total_in:
        raise RuntimeError(
            f"EP split mismatch: input_splits sum ({total_in}) != permuted tokens ({total_local})"
        )

    routing_map_u8 = routing_map_bool.to(torch.uint8).contiguous()
    expert_offsets = torch.tensor(expert_offsets_list, device=device, dtype=torch.int32)
    sorted_indices_i32 = torch.empty((total_local,), device=device, dtype=torch.int32)

    if E > 0 and N > 0 and total_local > 0:
        ext.launch_permute_build(routing_map_u8, expert_offsets, sorted_indices_i32, N)

    # Gather permuted tokens directly into symm_mem send buffer.
    # We need to handle world_size==1 specially, but still permute.
    if world_size == 1:
        local_permuted = torch.empty((total_local, hidden_dim), dtype=dtype, device=device)
        if total_local > 0:
            ext.launch_gather_tokens(hidden_states, sorted_indices_i32, local_permuted)
        # No A2A; direct sort_chunks
        global_permuted = local_permuted
        # sort_chunks_by_idxs
        num_local_experts = num_experts // 1
        permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
        split_sizes = num_global_tokens_per_local_expert.ravel().tolist()
        # apply reorder
        if len(permute_order) > 0:
            # compute src_starts
            src_starts = [0] * len(split_sizes)
            acc = 0
            for i, sz in enumerate(split_sizes):
                src_starts[i] = acc
                acc += sz
            dst_chunk_sizes = [split_sizes[i] for i in permute_order]
            dst_starts = [0] * len(permute_order)
            acc = 0
            for i, sz in enumerate(dst_chunk_sizes):
                dst_starts[i] = acc
                acc += sz
            src_starts_reordered = [src_starts[i] for i in permute_order]
            out_total = acc
            out_tensor = torch.empty((out_total, hidden_dim), dtype=dtype, device=device)
            if dst_chunk_sizes:
                max_rows = max(dst_chunk_sizes) if dst_chunk_sizes else 0
                ss = torch.tensor(src_starts_reordered, device=device, dtype=torch.int32)
                ds = torch.tensor(dst_starts, device=device, dtype=torch.int32)
                cs = torch.tensor(dst_chunk_sizes, device=device, dtype=torch.int32)
                ext.launch_reorder_chunks(global_permuted, out_tensor, ss, ds, cs, max_rows, hidden_dim)
            global_permuted = out_tensor

        sorted_indices_long = sorted_indices_i32.to(torch.int64)
        return global_permuted, routing_map, sorted_indices_long, org_hidden_states_shape

    # World size > 1 path: use symm_mem send buffer.
    send_rows = max(total_local, 1)
    send_buf, send_hdl, peer_ptrs = _get_send_buf(send_rows, hidden_dim, dtype, device)

    if total_local > 0:
        ext.launch_gather_tokens(hidden_states, sorted_indices_i32, send_buf[:total_local])

    # Compute in_offsets (per-peer offsets in their send buffer destined to this rank)
    # peer p sends slice [sum(input_splits_p[:rank]) : sum(input_splits_p[:rank+1])] to this rank.
    # We don't have other peers' input_splits; but output_splits[r] is what we receive from rank r,
    # equal to input_splits_r[rank]. The offset within rank r's send buffer is sum_j<rank input_splits_r[j].
    # That requires all-gathering input_splits or computing via output_splits—but offsets depend on each peer's
    # own splits, which aren't known locally.
    #
    # Strategy: use an extra all-gather of input_splits across the group, cached by world_size.
    # Use one-time all_gather on small int tensor; this is small overhead.
    splits_dev = torch.tensor(input_splits_list, device=device, dtype=torch.int32)
    gathered = [torch.empty(world_size, device=device, dtype=torch.int32) for _ in range(world_size)]
    dist.all_gather(gathered, splits_dev, group=group)
    # gathered[r] = input_splits of rank r (length world_size)
    # in_offsets[r] = sum_{j<rank} gathered[r][j]
    in_offsets_list = []
    for r in range(world_size):
        gr = gathered[r]
        if rank == 0:
            in_offsets_list.append(0)
        else:
            in_offsets_list.append(int(gr[:rank].sum().item()))
    in_sizes_list = output_splits_list  # rows we receive from each peer

    # out_offsets: cumulative sum of output_splits
    out_offsets_list = [0] * world_size
    acc = 0
    for r in range(world_size):
        out_offsets_list[r] = acc
        acc += output_splits_list[r]

    in_offsets_t = torch.tensor(in_offsets_list, device=device, dtype=torch.int32)
    in_sizes_t = torch.tensor(in_sizes_list, device=device, dtype=torch.int32)
    out_offsets_t = torch.tensor(out_offsets_list, device=device, dtype=torch.int32)

    out_total_rows = max(total_out, 1)
    a2a_out = torch.empty((out_total_rows, hidden_dim), dtype=dtype, device=device)

    # Synchronize: ensure all peers have written their send buffers before reads
    send_hdl.barrier(channel=0)

    max_chunk = max(in_sizes_list) if in_sizes_list else 0
    if max_chunk > 0 and total_out > 0:
        ext.launch_a2a_read(
            peer_ptrs, a2a_out, in_offsets_t, in_sizes_t, out_offsets_t,
            world_size, max_chunk, hidden_dim
        )

    send_hdl.barrier(channel=1)

    global_permuted = a2a_out[:total_out] if total_out < out_total_rows else a2a_out

    # ---- sort_chunks_by_idxs ----
    num_local_experts = num_experts // world_size
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    split_sizes_list = num_global_tokens_per_local_expert.reshape(-1).tolist()

    if len(permute_order) > 0 and total_out > 0:
        K = len(split_sizes_list)
        src_starts = [0] * K
        acc = 0
        for i in range(K):
            src_starts[i] = acc
            acc += split_sizes_list[i]
        dst_chunk_sizes = [split_sizes_list[i] for i in permute_order]
        dst_starts = [0] * len(permute_order)
        acc = 0
        for i, sz in enumerate(dst_chunk_sizes):
            dst_starts[i] = acc
            acc += sz
        src_starts_reordered = [src_starts[i] for i in permute_order]
        out_total = acc
        out_tensor = torch.empty((out_total, hidden_dim), dtype=dtype, device=device)
        max_rows = max(dst_chunk_sizes) if dst_chunk_sizes else 0
        if max_rows > 0:
            ss = torch.tensor(src_starts_reordered, device=device, dtype=torch.int32)
            ds = torch.tensor(dst_starts, device=device, dtype=torch.int32)
            cs = torch.tensor(dst_chunk_sizes, device=device, dtype=torch.int32)
            ext.launch_reorder_chunks(global_permuted.contiguous(), out_tensor, ss, ds, cs, max_rows, hidden_dim)
        global_permuted = out_tensor

    sorted_indices_long = sorted_indices_i32.to(torch.int64)
    return global_permuted, routing_map, sorted_indices_long, org_hidden_states_shape