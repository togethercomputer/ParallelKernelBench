"""
GraphBolt cooperative feature exchange via symmetric memory + custom CUDA.

Strategy:
- Each rank writes its gathered rows directly into peers' symmetric buffers
  using UVA device pointers (one kernel that gathers + scatters cross-GPU).
- A single fused kernel performs: index gather from local_features, then
  per-peer remote store using NVLink P2P writes.
- Signal-pad blockwise barrier provides arrival/completion sync without NCCL.
"""

from typing import List, Optional

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

__device__ __forceinline__ void send_signal_acqrel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acqrel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

// One-block all-rank barrier using signal pads.
__device__ void global_barrier(
    const uint64_t* signal_pad_ptrs,
    uint64_t slot,
    int rank,
    int world_size,
    bool acqrel
) {
    int tid = threadIdx.x;
    if (tid < world_size) {
        uint64_t remote_base = signal_pad_ptrs[tid];
        uint64_t local_base = signal_pad_ptrs[rank];
        uint32_t* send_addr = reinterpret_cast<uint32_t*>(
            remote_base + slot * (uint64_t)world_size + (uint64_t)rank);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
            local_base + slot * (uint64_t)world_size + (uint64_t)tid);
        if (acqrel) {
            send_signal_acqrel(send_addr);
            wait_signal_acqrel(wait_addr);
        } else {
            send_signal_relaxed(send_addr);
            wait_signal_relaxed(wait_addr);
        }
    }
    __syncthreads();
}

// Gather local features by index, then scatter to peer symmetric buffers.
// Each peer receives counts_received[peer] rows, written into its recv buffer
// at offset recv_offsets[my_rank_in_peer_view].
//
// To keep things simple: rank r sends to peer p the chunk corresponding to
// rotated index. The python side computes per-peer (dst_rank, src_offset_idx,
// num_rows, dst_row_offset_in_peer_buf).
//
// Layout of plan (int64 per peer, P peers):
//   plan[p].dst_rank
//   plan[p].src_idx_offset   // offset into seed_inverse_ids
//   plan[p].num_rows
//   plan[p].dst_row_offset   // row offset within recv buffer of peer
__global__ void gather_scatter_kernel(
    const __nv_bfloat16* __restrict__ local_features,  // [N, H]
    const int64_t* __restrict__ seed_inverse_ids,      // [total_send_rows]
    const int64_t* __restrict__ plan,                  // [P*4]
    const uint64_t* __restrict__ recv_buf_ptrs,        // [W] peer recv buffers
    int world_size,
    int H,
    int hidden_bytes_per_row,  // H * 2
    int P                       // number of peer entries
) {
    // Each block handles one peer entry's rows (subset).
    int peer_idx = blockIdx.y;
    if (peer_idx >= P) return;

    int64_t dst_rank        = plan[peer_idx * 4 + 0];
    int64_t src_idx_offset  = plan[peer_idx * 4 + 1];
    int64_t num_rows        = plan[peer_idx * 4 + 2];
    int64_t dst_row_offset  = plan[peer_idx * 4 + 3];

    if (num_rows == 0) return;

    __nv_bfloat16* dst_base = reinterpret_cast<__nv_bfloat16*>(recv_buf_ptrs[dst_rank]);

    // Grid stride over rows
    for (int64_t r = blockIdx.x; r < num_rows; r += gridDim.x) {
        int64_t local_row = seed_inverse_ids[src_idx_offset + r];
        const __nv_bfloat16* src_row = local_features + local_row * H;
        __nv_bfloat16* dst_row = dst_base + (dst_row_offset + r) * H;

        // Vectorized copy: try 8 bf16 = 16 bytes (uint4)
        int tid = threadIdx.x;
        int bsz = blockDim.x;

        if ((H % 8) == 0 && (((uintptr_t)src_row & 15) == 0) && (((uintptr_t)dst_row & 15) == 0)) {
            int n4 = H / 8;
            const uint4* s4 = reinterpret_cast<const uint4*>(src_row);
            uint4* d4 = reinterpret_cast<uint4*>(dst_row);
            for (int i = tid; i < n4; i += bsz) {
                d4[i] = s4[i];
            }
        } else {
            for (int i = tid; i < H; i += bsz) {
                dst_row[i] = src_row[i];
            }
        }
    }
}

__global__ void barrier_kernel(
    const uint64_t* signal_pad_ptrs,
    uint64_t slot,
    int rank,
    int world_size,
    int acqrel
) {
    global_barrier(signal_pad_ptrs, slot, rank, world_size, acqrel != 0);
}

void launch_gather_scatter(
    torch::Tensor local_features,        // bf16 [N,H]
    torch::Tensor seed_inverse_ids,      // int64
    torch::Tensor plan,                   // int64 [P*4]
    torch::Tensor recv_buf_ptrs,          // int64 [W]
    int64_t world_size,
    int64_t H,
    int64_t P,
    int64_t max_rows_per_peer
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 128;
    int row_blocks = (int)std::min<int64_t>(max_rows_per_peer, 256);
    if (row_blocks < 1) row_blocks = 1;
    dim3 grid(row_blocks, (unsigned)P, 1);

    gather_scatter_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(local_features.data_ptr<at::BFloat16>()),
        seed_inverse_ids.data_ptr<int64_t>(),
        plan.data_ptr<int64_t>(),
        reinterpret_cast<const uint64_t*>(recv_buf_ptrs.data_ptr<int64_t>()),
        (int)world_size,
        (int)H,
        (int)(H * 2),
        (int)P
    );
}

void launch_barrier(
    torch::Tensor signal_pad_ptrs,  // int64 [W]
    int64_t slot,
    int64_t rank,
    int64_t world_size,
    int64_t acqrel
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    barrier_kernel<<<1, (unsigned)world_size, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>()),
        (uint64_t)slot,
        (int)rank,
        (int)world_size,
        (int)acqrel
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_scatter", &launch_gather_scatter, "fused gather + p2p scatter");
    m.def("launch_barrier", &launch_barrier, "signal-pad barrier");
}
'''


_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_feat_exchange_ext", CUDA_SRC)
    return _ext


# Cache symmetric recv buffer per (capacity_rows, H, dtype, device).
_buf_cache = {}

def _get_recv_buf(capacity_rows: int, H: int, dtype: torch.dtype, device: torch.device):
    key = (capacity_rows, H, dtype, str(device))
    entry = _buf_cache.get(key)
    if entry is not None:
        return entry
    buf = symm_mem.empty((capacity_rows, H), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    entry = (buf, hdl, ptrs)
    _buf_cache[key] = entry
    return entry


_barrier_slot = [0]


@torch.no_grad()
def solution(
    local_features: torch.Tensor,
    seed_inverse_ids: torch.Tensor,
    counts_sent: List[int],
    counts_received: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD

    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        gathered = local_features[seed_inverse_ids]
        out = local_features.new_empty((sum(counts_sent),) + local_features.shape[1:])
        # single rank: counts_sent == counts_received, just copy
        if gathered.numel() > 0:
            out.copy_(gathered)
        return out

    # Only handle 2D bf16 CUDA tensors with the fast path; else fallback.
    if (local_features.dtype != torch.bfloat16
            or not local_features.is_cuda
            or local_features.dim() != 2):
        # Fallback: reference-style implementation
        return _reference_solution(local_features, seed_inverse_ids,
                                   counts_sent, counts_received, group)

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    H = local_features.shape[1]
    device = local_features.device

    # The reference, after _shift, calls dist.all_to_all with:
    #   outputs_unshifted = shift(split(out, counts_sent))
    #   inputs_unshifted  = shift(split(gathered, counts_received))
    # In dist.all_to_all the i-th input is sent to rank i and i-th output
    # comes from rank i.
    #
    # _shift(chunks): cutoff = W - rank; return chunks[cutoff:] + chunks[:cutoff]
    #   So unshifted[i] = chunks[(i + cutoff) mod W] = chunks[(i - rank) mod W]
    # That means: chunks index j -> unshifted index (j + rank) mod W.
    # Therefore: input chunk j (split of gathered with counts_received[j])
    # is sent to peer rank (j + rank) mod W.
    # And output chunk j (split of out with counts_sent[j]) is received from
    # peer rank (j + rank) mod W.

    # Build send plan: for each j, send counts_received[j] rows starting at
    # cumulative offset to peer dst = (j + rank) % W, placed at peer's recv buf
    # at offset = peer's prefix sum over its counts_sent for slot j_peer where
    # peer_rank = dst, source rank = our rank.
    # On peer 'dst', counts_sent[j_peer] is the rows received from source rank
    # (j_peer + dst) mod W = our rank => j_peer = (rank - dst) mod W.
    #
    # We need peer's counts_sent prefix sums to know offsets. We don't have
    # those directly here; but counts_received on this rank corresponds to
    # what peers will accept — actually each rank's counts_sent[j] equals the
    # source rank=(j+dst)%W's counts_received[j']. The harness guarantees
    # consistency, but to compute offsets on peer we need peer's counts_sent.
    #
    # Simpler: each rank lays out its OWN recv buffer using counts_sent (which
    # is what it expects to receive). Peers writing into our buffer must use
    # offsets based on our counts_sent. So we need every rank to know every
    # other rank's counts_sent prefix at the slot corresponding to itself.
    #
    # Solution: do an all_gather of counts_sent vector via a small CPU/CUDA
    # collective. World size is small (<=8). We use a tiny symm_mem int64
    # buffer.

    counts_sent_t = torch.tensor(counts_sent, device=device, dtype=torch.int64)
    counts_recv_t = torch.tensor(counts_received, device=device, dtype=torch.int64)

    # Gather all ranks' counts_sent into [W, W]
    all_counts_sent = _all_gather_int64(counts_sent_t, world_size, device)
    # all_counts_sent[r, j] = rank r's counts_sent[j] = rows rank r receives
    # from rank (j + r) % W.

    # Compute total recv rows = sum(counts_sent) for our rank
    total_recv = int(counts_sent_t.sum().item())
    total_send = int(counts_recv_t.sum().item())

    # Allocate / reuse symmetric recv buffer with sufficient capacity.
    # Use a capacity that grows; round up.
    capacity = max(total_recv, 1)
    # Round up to reduce reallocation churn
    pow2 = 1
    while pow2 < capacity:
        pow2 *= 2
    capacity = pow2

    recv_buf, hdl, recv_buf_ptrs = _get_recv_buf(capacity, H, local_features.dtype, device)

    # Build plan on CPU (W is small)
    # For each j in [0, W):
    #   dst = (j + rank) % W
    #   src_idx_offset = prefix sum of counts_received up to j
    #   num_rows = counts_received[j]
    #   On peer dst, we are source rank = our rank. Peer dst's slot j_peer
    #   such that (j_peer + dst) % W == rank => j_peer = (rank - dst) % W.
    #   dst_row_offset = sum(all_counts_sent[dst, 0..j_peer-1])
    all_counts_sent_cpu = all_counts_sent.cpu().tolist()
    counts_recv_cpu = counts_received

    plan_list = []
    src_prefix = 0
    for j in range(world_size):
        dst = (j + rank) % world_size
        nrows = counts_recv_cpu[j]
        j_peer = (rank - dst) % world_size
        dst_row_offset = sum(all_counts_sent_cpu[dst][:j_peer])
        plan_list.append([dst, src_prefix, nrows, dst_row_offset])
        src_prefix += nrows

    plan_t = torch.tensor(plan_list, device=device, dtype=torch.int64).flatten()

    # Gather source rows into a contiguous buffer? We can do it inline in the
    # kernel by reading directly from local_features via seed_inverse_ids.
    # seed_inverse_ids is already the index list for all sends.

    ext = _get_ext()

    # Pre-barrier to ensure recv buffers are ready (no readers)
    slot = _barrier_slot[0] % 16
    _barrier_slot[0] += 1
    ext.launch_barrier(hdl.signal_pad_ptrs_dev, slot, rank, world_size, 0)

    max_rows = max((c for c in counts_recv_cpu), default=1)
    if max_rows < 1:
        max_rows = 1

    ext.launch_gather_scatter(
        local_features,
        seed_inverse_ids.to(torch.int64) if seed_inverse_ids.dtype != torch.int64 else seed_inverse_ids,
        plan_t,
        recv_buf_ptrs,
        world_size,
        H,
        world_size,
        max_rows,
    )

    # Post-barrier: ensure all peers finished writing into our buffer
    slot2 = _barrier_slot[0] % 16
    _barrier_slot[0] += 1
    ext.launch_barrier(hdl.signal_pad_ptrs_dev, slot2, rank, world_size, 1)

    # Slice recv_buf to total_recv and return as a fresh tensor (clone to
    # avoid aliasing the symm_mem buffer used next call).
    if total_recv == 0:
        out = local_features.new_empty((0, H))
    else:
        out = recv_buf[:total_recv].clone()

    return out


def _all_gather_int64(t: torch.Tensor, world_size: int, device) -> torch.Tensor:
    """Small all-gather of a 1D int64 tensor of length world_size."""
    # Use torch.distributed since this is a tiny one-off; not on hot path
    # for large tensors.
    out_list = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(out_list, t)
    return torch.stack(out_list, dim=0)


def _reference_solution(local_features, seed_inverse_ids, counts_sent,
                        counts_received, group):
    def _shift(chunks):
        cutoff = len(chunks) - dist.get_rank(group)
        return chunks[cutoff:] + chunks[:cutoff]

    gathered = local_features[seed_inverse_ids]
    out = local_features.new_empty((sum(counts_sent),) + local_features.shape[1:])
    outputs = _shift(list(torch.split(out, counts_sent)))
    inputs = _shift(list(torch.split(gathered, counts_received)))
    dist.all_to_all(outputs, inputs, group=group)
    return out