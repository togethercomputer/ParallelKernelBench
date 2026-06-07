"""
Problem 50: Fused MoE forward — balanced expert parallel (num_experts == world_size).

Optimized with custom CUDA: replaces dist.all_to_all_single and
dist.all_gather_into_tensor with symmetric-memory peer-pointer kernels.
The all-to-all is implemented as a device-side gather using UVA pointers
to peer symmetric buffers (one local expert per rank in this balanced regime).
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

// ---------- signal-pad barrier (relaxed and acq_rel) ----------
__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}
__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__global__ void barrier_kernel(
    uint64_t* signal_pad_ptrs,
    int rank,
    int world_size,
    uint64_t block_id
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

// All-gather of a small int64 vector via symmetric memory peers.
// Each rank has placed its data at offset (rank * elems_per_rank) in its symm buffer.
// We barrier, then read from each peer.
__global__ void allgather_int64_kernel(
    uint64_t* peer_ptrs,         // [world_size] symm buffer base ptrs
    uint64_t* signal_pad_ptrs,   // [world_size]
    int64_t* out,                // [world_size * elems_per_rank]
    int rank,
    int world_size,
    int elems_per_rank
) {
    // barrier first (use thread 0..world_size for signaling)
    unsigned int tid = threadIdx.x;
    if (blockIdx.x == 0 && tid < (unsigned)world_size) {
        uint64_t local_base = signal_pad_ptrs[rank];
        uint64_t remote_base = signal_pad_ptrs[tid];
        uint32_t* send_addr = reinterpret_cast<uint32_t*>(
            remote_base + 0 * (uint64_t)world_size + (uint64_t)rank);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
            local_base + 0 * (uint64_t)world_size + (uint64_t)tid);
        send_signal_acq_rel(send_addr);
        wait_signal_acq_rel(wait_addr);
    }
    __syncthreads();
    // grid-wide barrier — but we only have 1 block, so syncthreads is enough.

    int total = world_size * elems_per_rank;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < total; i += stride) {
        int r = i / elems_per_rank;
        int off = i % elems_per_rank;
        const int64_t* src = reinterpret_cast<const int64_t*>(peer_ptrs[r]);
        // each rank has stored its own data at offset (rank * elems_per_rank)
        out[i] = src[r * elems_per_rank + off];
    }

    __syncthreads();
    if (blockIdx.x == 0 && tid < (unsigned)world_size) {
        uint64_t local_base = signal_pad_ptrs[rank];
        uint64_t remote_base = signal_pad_ptrs[tid];
        uint32_t* send_addr = reinterpret_cast<uint32_t*>(
            remote_base + 1 * (uint64_t)world_size + (uint64_t)rank);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
            local_base + 1 * (uint64_t)world_size + (uint64_t)tid);
        send_signal_acq_rel(send_addr);
        wait_signal_acq_rel(wait_addr);
    }
}

void launch_allgather_int64(
    torch::Tensor peer_ptrs,
    torch::Tensor signal_pad_ptrs,
    torch::Tensor out,
    int64_t rank,
    int64_t world_size,
    int64_t elems_per_rank
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 64;
    int blocks = 1;
    allgather_int64_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint64_t*>(peer_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>()),
        out.data_ptr<int64_t>(),
        (int)rank, (int)world_size, (int)elems_per_rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// All-to-all (single) via symmetric memory.
// Each rank writes its full input into its symm buffer (already done by copy_).
// To gather, each peer needs to know the source offsets in our buffer.
// We implement: for each peer p, copy peer_p[ src_offsets_for_us[p] : src_offsets_for_us[p] + recv_count[p] ]
// into our out at out_offsets[p].
//
// peer_input_offsets_per_rank[p] = the offset in peer p's input buffer of the chunk destined to us.
// recv_counts[p] = how many rows from peer p.
// out_offsets[p] = where to put it in our output (cumsum of recv_counts).
__global__ void all_to_all_bf16_kernel(
    uint64_t* peer_input_ptrs,        // [world_size]
    __nv_bfloat16* out,               // local output [total_recv, hidden]
    const int64_t* recv_counts,       // [world_size]
    const int64_t* recv_offsets,      // [world_size]  (where to write in out)
    const int64_t* src_offsets,       // [world_size]  (where to read in peer p's input)
    int world_size,
    int hidden
) {
    // Each block handles one (peer, row) pair-ish. Use 2D grid: x=row, y=peer.
    int peer = blockIdx.y;
    if (peer >= world_size) return;

    int64_t cnt = recv_counts[peer];
    if (cnt == 0) return;

    int64_t row = blockIdx.x;
    if (row >= cnt) return;

    int64_t src_row = src_offsets[peer] + row;
    int64_t dst_row = recv_offsets[peer] + row;

    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_input_ptrs[peer])
                               + src_row * hidden;
    __nv_bfloat16* dst = out + dst_row * hidden;

    // copy hidden elements with vectorized loads (4x bf16 = 8 bytes)
    int tid = threadIdx.x;
    int nth = blockDim.x;
    // use float4 (16 bytes = 8 bf16)
    int hidden_v8 = hidden / 8;
    const float4* sv = reinterpret_cast<const float4*>(src);
    float4* dv = reinterpret_cast<float4*>(dst);
    for (int i = tid; i < hidden_v8; i += nth) {
        dv[i] = sv[i];
    }
    int rem_start = hidden_v8 * 8;
    for (int i = rem_start + tid; i < hidden; i += nth) {
        dst[i] = src[i];
    }
}

void launch_all_to_all_bf16(
    torch::Tensor peer_input_ptrs,
    torch::Tensor out,
    torch::Tensor recv_counts,
    torch::Tensor recv_offsets,
    torch::Tensor src_offsets,
    int64_t world_size,
    int64_t hidden,
    int64_t max_rows
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (max_rows == 0) return;
    dim3 grid((unsigned)max_rows, (unsigned)world_size);
    int threads = 128;
    all_to_all_bf16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<uint64_t*>(peer_input_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        recv_counts.data_ptr<int64_t>(),
        recv_offsets.data_ptr<int64_t>(),
        src_offsets.data_ptr<int64_t>(),
        (int)world_size, (int)hidden);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_all_to_all_f32(
    torch::Tensor peer_input_ptrs,
    torch::Tensor out,
    torch::Tensor recv_counts,
    torch::Tensor recv_offsets,
    torch::Tensor src_offsets,
    int64_t world_size,
    int64_t hidden,
    int64_t max_rows
) {
    // reuse bf16 kernel via separate float kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (max_rows == 0) return;
    // For f32, we can call the same structure but cast — easier: just do byte-level memcpy via float4 too
    // hidden floats -> hidden_v4 = hidden/4 float4 (16B = 4 floats)
    auto launch = [&]() {
        // Implement inline via lambda: we'll dispatch through a tiny kernel below
    };
    // Just call a dedicated kernel:
    // (define inline to keep file compact)
    extern __global__ void all_to_all_f32_kernel(
        uint64_t*, float*, const int64_t*, const int64_t*, const int64_t*, int, int);
    dim3 grid((unsigned)max_rows, (unsigned)world_size);
    int threads = 128;
    all_to_all_f32_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<uint64_t*>(peer_input_ptrs.data_ptr<int64_t>()),
        out.data_ptr<float>(),
        recv_counts.data_ptr<int64_t>(),
        recv_offsets.data_ptr<int64_t>(),
        src_offsets.data_ptr<int64_t>(),
        (int)world_size, (int)hidden);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void all_to_all_f32_kernel(
    uint64_t* peer_input_ptrs,
    float* out,
    const int64_t* recv_counts,
    const int64_t* recv_offsets,
    const int64_t* src_offsets,
    int world_size,
    int hidden
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;
    int64_t cnt = recv_counts[peer];
    if (cnt == 0) return;
    int64_t row = blockIdx.x;
    if (row >= cnt) return;

    int64_t src_row = src_offsets[peer] + row;
    int64_t dst_row = recv_offsets[peer] + row;

    const float* src = reinterpret_cast<const float*>(peer_input_ptrs[peer])
                       + src_row * hidden;
    float* dst = out + dst_row * hidden;

    int tid = threadIdx.x;
    int nth = blockDim.x;
    int hidden_v4 = hidden / 4;
    const float4* sv = reinterpret_cast<const float4*>(src);
    float4* dv = reinterpret_cast<float4*>(dst);
    for (int i = tid; i < hidden_v4; i += nth) {
        dv[i] = sv[i];
    }
    int rem_start = hidden_v4 * 4;
    for (int i = rem_start + tid; i < hidden; i += nth) {
        dst[i] = src[i];
    }
}

// Barrier-only kernel for synchronization between phases
void launch_barrier(
    torch::Tensor signal_pad_ptrs,
    int64_t rank,
    int64_t world_size,
    int64_t block_id
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    barrier_kernel<<<1, 64, 0, stream>>>(
        reinterpret_cast<uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>()),
        (int)rank, (int)world_size, (uint64_t)block_id);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_allgather_int64", &launch_allgather_int64, "AG int64 via symm");
    m.def("launch_all_to_all_bf16", &launch_all_to_all_bf16, "A2A bf16 via symm");
    m.def("launch_all_to_all_f32", &launch_all_to_all_f32, "A2A f32 via symm");
    m.def("launch_barrier", &launch_barrier, "Symm barrier");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_ep_balanced_ext", CUDA_SRC)
    return _ext


# ---------- symmetric memory caches ----------

_ag_cache = {}      # all-gather small int buffer
_a2a_in_cache = {}  # input symm buffer for a2a
_a2a_out_cache = {} # output buffer (regular)
_peer_ptrs_cache = {}
_signal_cache = {}


def _get_ag_buf(world_size: int, elems_per_rank: int, device, dtype, group):
    key = (world_size, elems_per_rank, device, dtype)
    if key in _ag_cache:
        return _ag_cache[key]
    total = world_size * elems_per_rank
    buf = symm_mem.empty(total, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    peer_ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    sig_ptrs = torch.tensor(list(hdl.signal_pad_ptrs), device=device, dtype=torch.int64)
    out = torch.empty(total, device=device, dtype=dtype)
    _ag_cache[key] = (buf, hdl, peer_ptrs, sig_ptrs, out)
    return _ag_cache[key]


def _get_a2a_in(num_rows_cap: int, hidden: int, device, dtype, group):
    key = (num_rows_cap, hidden, device, dtype)
    if key in _a2a_in_cache:
        return _a2a_in_cache[key]
    buf = symm_mem.empty((num_rows_cap, hidden), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    peer_ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    sig_ptrs = torch.tensor(list(hdl.signal_pad_ptrs), device=device, dtype=torch.int64)
    _a2a_in_cache[key] = (buf, hdl, peer_ptrs, sig_ptrs)
    return _a2a_in_cache[key]


# ---------- Custom AllToAll Function (autograd-aware, but uses custom CUDA in fwd; bwd uses dist for safety) ----------

class _AllToAllCustom(torch.autograd.Function):
    """
    Forward: custom symm-mem all_to_all.
    Backward: dist.all_to_all_single (rare path; backward not required by problem 50).
    """
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input.contiguous()

        input = input.contiguous()
        rank = dist.get_rank(group)
        hidden = input.size(1)
        dtype = input.dtype
        device = input.device

        in_rows = input.size(0)
        out_rows = sum(output_split_sizes) if output_split_sizes is not None else in_rows

        # Capacity: max across ranks. We use the larger of in_rows and out_rows, padded.
        # We need a symm buffer big enough on every rank. Use a power-of-2-ish growable cap.
        # Use a global cap across all calls (simple): reuse with at least max(in_rows) grown.
        cap_key = (hidden, dtype)
        cap = max(in_rows, 1)
        # round up
        cap_pow = 1
        while cap_pow < cap:
            cap_pow *= 2
        cap_pow = max(cap_pow, 64)

        # Track existing cap; grow if needed
        existing_cap = 0
        for k in _a2a_in_cache.keys():
            if k[1] == hidden and k[3] == dtype and k[2] == device:
                existing_cap = max(existing_cap, k[0])
        use_cap = max(existing_cap, cap_pow)

        in_buf, in_hdl, peer_ptrs, sig_ptrs = _get_a2a_in(use_cap, hidden, device, dtype, group)

        # Copy local input into symm buffer
        in_buf[:in_rows].copy_(input)

        # Build offsets / counts
        # input_split_sizes: rows we send to each peer
        # output_split_sizes: rows we receive from each peer (= peer's input_split_sizes[rank])
        input_splits_t = torch.tensor(input_split_sizes if input_split_sizes is not None
                                      else [in_rows // world_size] * world_size,
                                      device=device, dtype=torch.int64)
        output_splits_t = torch.tensor(output_split_sizes if output_split_sizes is not None
                                       else [out_rows // world_size] * world_size,
                                       device=device, dtype=torch.int64)

        # src_offsets[p] = where in peer p's input the chunk for us starts.
        # That is: prefix-sum over peer p's input_split_sizes up to index `rank`.
        # We need to know each peer's input_split_sizes. We have output_split_sizes locally,
        # and that equals: for each p, output_split_sizes[p] = peer_p.input_split_sizes[rank].
        # But we need peer_p.input_split_sizes[0..rank-1] to compute the offset.
        #
        # Easier: gather all input_split_sizes globally via small all-gather.
        # all_input_splits[p, q] = rank p's input_split_sizes[q]
        ws = world_size
        all_input_splits_flat = _allgather_int64(input_splits_t, ws, group)
        all_input_splits = all_input_splits_flat.view(ws, ws)  # [from_rank, to_rank]
        # src_offsets[p] = sum over q < rank of all_input_splits[p, q]
        # = cumsum along to_rank dim, take column `rank`'s prefix
        cum = torch.cumsum(all_input_splits, dim=1)  # [ws, ws]
        # offset in peer p's buffer of chunk going to `rank`
        if rank == 0:
            src_offsets = torch.zeros(ws, device=device, dtype=torch.int64)
        else:
            src_offsets = cum[:, rank - 1].contiguous()

        recv_offsets = torch.zeros(ws, device=device, dtype=torch.int64)
        if ws > 1:
            recv_offsets[1:] = torch.cumsum(output_splits_t, dim=0)[:-1]

        # Output buffer
        output = torch.empty((out_rows, hidden), device=device, dtype=dtype)

        ext = _get_ext()
        # Barrier so all peers have finished writing their in_buf
        ext.launch_barrier(sig_ptrs, rank, ws, 2)

        max_rows = int(output_splits_t.max().item()) if ws > 0 else 0
        if max_rows > 0:
            if dtype == torch.bfloat16:
                ext.launch_all_to_all_bf16(
                    peer_ptrs, output, output_splits_t, recv_offsets, src_offsets,
                    ws, hidden, max_rows)
            elif dtype == torch.float32:
                ext.launch_all_to_all_f32(
                    peer_ptrs, output, output_splits_t, recv_offsets, src_offsets,
                    ws, hidden, max_rows)
            else:
                # fallback
                ext.launch_barrier(sig_ptrs, rank, ws, 3)
                return _fallback_a2a(group, input, output_split_sizes, input_split_sizes)

        # Barrier so no peer reuses the in_buf before all readers done
        ext.launch_barrier(sig_ptrs, rank, ws, 3)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # rarely used in problem 50; fall back to dist
        if dist.get_world_size(group=ctx.group) == 1:
            return None, grad_output.contiguous(), None, None
        grad_output = grad_output.contiguous()
        if ctx.input_split_sizes is None:
            grad_input = torch.empty_like(grad_output)
        else:
            grad_input = torch.empty(
                size=(sum(ctx.input_split_sizes), grad_output.size(1)),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
        dist.all_to_all_single(
            grad_input, grad_output,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )
        return None, grad_input, None, None


def _fallback_a2a(group, input, output_split_sizes, input_split_sizes):
    if output_split_sizes is None:
        out = torch.empty_like(input)
    else:
        out = torch.empty(
            size=(sum(output_split_sizes), input.size(1)),
            dtype=input.dtype, device=input.device)
    dist.all_to_all_single(
        out, input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group)
    return out


def _allgather_int64(local: torch.Tensor, world_size: int, group) -> torch.Tensor:
    """All-gather a 1-D int64 tensor across world. Returns flat [ws*n]."""
    assert local.dtype == torch.int64
    n = local.numel()
    device = local.device
    buf, hdl, peer_ptrs, sig_ptrs, out = _get_ag_buf(world_size, n, device, torch.int64, group)
    rank = dist.get_rank(group)
    # write our chunk into our own buffer at offset rank*n
    buf[rank * n: (rank + 1) * n].copy_(local)
    ext = _get_ext()
    ext.launch_allgather_int64(peer_ptrs, sig_ptrs, out, rank, world_size, n)
    return out


def _all_to_all(
    group: dist.ProcessGroup,
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
) -> torch.Tensor:
    return _AllToAllCustom.apply(group, input, output_split_sizes, input_split_sizes)


# ---------- Preprocess (uses our custom allgather) ----------

def _preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
):
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))
    input_splits = (
        num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    )
    num_local_tokens_per_expert_flat = num_local_tokens_per_expert.contiguous().view(-1).to(torch.int64)
    n_local = num_local_tokens_per_expert_flat.numel()

    # Custom symm-mem all-gather instead of dist.all_gather_into_tensor
    num_global_tokens_per_expert_flat = _allgather_int64(num_local_tokens_per_expert_flat, ep_size, ep_group)

    num_global_tokens_per_expert = num_global_tokens_per_expert_flat.view(ep_size, n_local)
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[
        :, start_idx:end_idx
    ].contiguous()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0).to(
        torch.device("cpu"), non_blocking=True
    )
    num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert.view(
        -1, num_local_experts
    ).to(torch.device("cpu"), non_blocking=True)
    return (
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert_cpu,
        num_global_sum_tokens_per_local_expert,
    )


# ---------- helpers ----------

def _permute(tokens, routing_map):
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]
    routing_map = routing_map.bool()
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, sorted_indices


def _sort_chunks_by_idxs(input, split_sizes, sorted_idxs):
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()
    chunks = torch.split(input, split_sizes, dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)


def _generate_weights_idx(routing_weights, selected_experts, num_experts):
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros(
        (num_tokens, num_experts),
        dtype=routing_weights.dtype,
        device=routing_weights.device,
    )
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx


def _unpermute(tokens, routing_weights, hidden_states_shape, permutation_mapping, routing_map):
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]
    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)
    expanded_mapping = permutation_mapping.unsqueeze(1).expand(-1, hidden_dim)
    unpermuted_tokens.scatter_add_(0, expanded_mapping, tokens)
    return unpermuted_tokens


def token_pre_all2all(
    hidden_states, expert_mask, num_experts,
    input_splits, output_splits, num_global_tokens_per_local_expert,
    group=None,
):
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
        raise RuntimeError(f"EP split mismatch: {expected_tokens} != {actual_tokens}")

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
    expert_outputs, routing_weights, selected_experts, num_experts,
    input_splits, output_splits, num_global_tokens_per_local_expert,
    routing_map, local_input_permutation_mapping, org_hidden_states_shape,
    group=None,
):
    group = group or dist.group.WORLD
    num_local_experts = num_experts // dist.get_world_size(group)
    unpermute_order = (
        torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    )
    split_sizes = num_global_tokens_per_local_expert.T.ravel().tolist()
    expert_outputs = _sort_chunks_by_idxs(expert_outputs, split_sizes, unpermute_order)
    unpermute_outputs = _all_to_all(group, expert_outputs, input_splits, output_splits)
    weights_idx = _generate_weights_idx(routing_weights, selected_experts, num_experts)
    unpermute_outputs = _unpermute(
        unpermute_outputs, weights_idx, org_hidden_states_shape,
        local_input_permutation_mapping, routing_map,
    )
    return unpermute_outputs


def expert_forward(x, gate_proj, up_proj, down_proj):
    gate = torch.nn.functional.silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


def solution(
    hidden_states,
    gate_weight,
    gate_bias,
    gate_proj,
    up_proj,
    down_proj,
    num_experts,
    top_k,
    group=None,
):
    group = group or dist.group.WORLD
    # Eagerly compile extension (rank 0 first to avoid races)
    if dist.is_initialized():
        if dist.get_rank(group) == 0:
            _get_ext()
        dist.barrier(group=group)
    _get_ext()

    hidden_dim = hidden_states.size(-1)
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
        hidden_states, expert_mask, num_experts,
        input_splits, output_splits, num_global_tokens_per_local_expert, group,
    )

    expert_outputs = expert_forward(
        global_permuted_hidden_states, gate_proj, up_proj, down_proj
    )

    out = tokens_post_all2all(
        expert_outputs, routing_weights, selected_experts, num_experts,
        input_splits, output_splits, num_global_tokens_per_local_expert,
        routing_map, local_input_permutation_mapping, org_hidden_states_shape, group,
    )
    return out