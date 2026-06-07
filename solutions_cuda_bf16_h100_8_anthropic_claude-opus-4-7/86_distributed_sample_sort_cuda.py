"""
Distributed sample sort using symmetric memory for collective operations.

Strategy:
- Local sort via torch.sort (uses CUB internally, hard to beat).
- Replace all_gather/all_to_all collectives with symmetric memory + custom CUDA
  kernels that read peer buffers directly via UVA pointers over NVLink.
- Use a single symmetric "exchange" buffer sized to the maximum local shard
  to host both the sample-gather and the variable all-to-all payloads.
- Splitter computation kept on host (small: world_size entries) but driven by
  device-side gathers.
"""

from typing import List, Optional, Tuple

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

// Copy from a peer symmetric buffer into a local destination.
__global__ void peer_copy_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        dst[idx] = src[idx];
    }
}

__global__ void peer_copy_i64_kernel(
    const int64_t* __restrict__ src,
    int64_t* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        dst[idx] = src[idx];
    }
}

void launch_peer_copy_bf16(int64_t src_ptr, torch::Tensor dst, int64_t n) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)src_ptr);
    peer_copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
        src, (__nv_bfloat16*)dst.data_ptr<at::BFloat16>(), n);
}

void launch_peer_copy_i64(int64_t src_ptr, torch::Tensor dst, int64_t n) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t* src = reinterpret_cast<const int64_t*>((uintptr_t)src_ptr);
    peer_copy_i64_kernel<<<blocks, threads, 0, stream>>>(
        src, dst.data_ptr<int64_t>(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_peer_copy_bf16", &launch_peer_copy_bf16, "peer copy bf16");
    m.def("launch_peer_copy_i64", &launch_peer_copy_i64, "peer copy int64");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("sample_sort_p2p_ext", CUDA_SRC)
    return _ext


# ---------------------------------------------------------------------------
# Symmetric buffer caches
# ---------------------------------------------------------------------------

_size_buf_cache = None  # for sizes (int64), one slot per rank
_data_buf_cache = None  # for bf16 payloads, max-shard-sized


def _get_size_buf(world_size: int, device: torch.device):
    global _size_buf_cache
    if _size_buf_cache is None:
        buf = symm_mem.empty(world_size, device=device, dtype=torch.int64)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        _size_buf_cache = (buf, hdl)
    return _size_buf_cache


def _get_data_buf(min_size: int, device: torch.device):
    """Symmetric bf16 buffer, grows monotonically."""
    global _data_buf_cache
    if _data_buf_cache is None or _data_buf_cache[0].numel() < min_size:
        size = max(min_size, 1)
        # Round up to reduce realloc churn
        size = max(size, 1024)
        buf = symm_mem.empty(size, device=device, dtype=torch.bfloat16)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        _data_buf_cache = (buf, hdl, size)
    return _data_buf_cache


def _all_gather_sizes(local_n: int, device: torch.device, group) -> List[int]:
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    buf, hdl = _get_size_buf(world_size, device)
    buf.zero_()
    buf[rank] = local_n
    hdl.barrier(channel=0)

    # Read all entries via peer pointers (each rank wrote its slot).
    out = torch.empty(world_size, dtype=torch.int64, device=device)
    ext = _get_ext()
    for r in range(world_size):
        peer_ptr = int(hdl.buffer_ptrs[r]) + r * 8  # offset to slot r
        ext.launch_peer_copy_i64(peer_ptr, out[r:r + 1], 1)
    hdl.barrier(channel=1)
    return out.cpu().tolist()


# ---------------------------------------------------------------------------
# Core helpers (mostly unchanged from reference, kept for correctness)
# ---------------------------------------------------------------------------


def _active_rank_info(rank: int, sizes: List[int]) -> Tuple[List[int], int]:
    active = [idx for idx, size in enumerate(sizes) if size > 0]
    sort_rank = active.index(rank) if rank in active else -1
    return active, sort_rank


def _extract_samples(sorted_local, sort_rank, n_samples):
    if sort_rank < 0 or sorted_local.numel() == 0:
        values = sorted_local.new_full((n_samples,), float("inf"))
        ranks = torch.full((n_samples,), -1, dtype=torch.long, device=sorted_local.device)
        positions = torch.full_like(ranks, -1)
        return values, ranks, positions

    local_n = sorted_local.numel()
    sample_idx = torch.arange(n_samples, dtype=torch.long, device=sorted_local.device)
    valid_count = min(n_samples, local_n)
    values = sorted_local.new_full((n_samples,), float("inf"))
    ranks = torch.full((n_samples,), -1, dtype=torch.long, device=sorted_local.device)
    positions = torch.full_like(ranks, -1)
    if n_samples < local_n:
        valid_positions = ((sample_idx + 1) * local_n).div(n_samples, rounding_mode="floor") - 1
    else:
        valid_positions = sample_idx[:valid_count]
    values[:valid_count] = sorted_local[valid_positions[:valid_count]]
    ranks[:valid_count] = sort_rank
    positions[:valid_count] = valid_positions[:valid_count]
    return values, ranks, positions


def _gather_splitters_via_pg(sample_values, sample_ranks, sample_positions, active_count, group):
    """Use stock all_gather for the small sample tensors (size = world_size each)."""
    world_size = dist.get_world_size(group=group)
    value_parts = [torch.empty_like(sample_values) for _ in range(world_size)]
    rank_parts = [torch.empty_like(sample_ranks) for _ in range(world_size)]
    pos_parts = [torch.empty_like(sample_positions) for _ in range(world_size)]
    dist.all_gather(value_parts, sample_values, group=group)
    dist.all_gather(rank_parts, sample_ranks, group=group)
    dist.all_gather(pos_parts, sample_positions, group=group)

    values = torch.cat(value_parts).detach().cpu().tolist()
    ranks = torch.cat(rank_parts).detach().cpu().tolist()
    positions = torch.cat(pos_parts).detach().cpu().tolist()
    samples = [
        (float(v), int(r), int(p))
        for v, r, p in zip(values, ranks, positions)
        if int(r) >= 0
    ]
    samples.sort(key=lambda x: (x[0], x[1], x[2]))

    splitters = []
    usable = len(samples)
    for sr in range(active_count - 1):
        index = (sr + 1) * usable // active_count - 1
        splitters.append(samples[max(0, min(index, usable - 1))])
    return splitters


def _split_positions(sorted_local, splitters, sort_rank):
    if sort_rank < 0:
        return [0] * (len(splitters) + 2)
    boundaries = [0]
    for value, splitter_rank, splitter_position in splitters:
        probe = torch.tensor(value, dtype=sorted_local.dtype, device=sorted_local.device)
        if sort_rank > splitter_rank:
            end = int(torch.searchsorted(sorted_local, probe, right=False).item())
        elif sort_rank < splitter_rank:
            end = int(torch.searchsorted(sorted_local, probe, right=True).item())
        else:
            end = int(splitter_position) + 1
        boundaries.append(max(boundaries[-1], min(end, sorted_local.numel())))
    boundaries.append(sorted_local.numel())
    return boundaries


# ---------------------------------------------------------------------------
# P2P variable all-to-all over symmetric memory
# ---------------------------------------------------------------------------


def _p2p_variable_all_to_all(
    send_chunks: List[torch.Tensor],
    group,
) -> torch.Tensor:
    """
    Each rank publishes its concatenated payload + per-destination offsets/counts
    into a symmetric buffer. Peers then pull their slice directly via UVA.

    Returns a flat tensor (concatenation in source-rank order) of received data.
    """
    device = send_chunks[0].device
    dtype = send_chunks[0].dtype
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    # Counts I'm sending to each dest
    send_counts = [int(c.numel()) for c in send_chunks]
    total_send = sum(send_counts)

    # 1) Exchange the count matrix using stock all_gather (small: world_size ints).
    sc_tensor = torch.tensor(send_counts, dtype=torch.int64, device=device)
    counts_parts = [torch.empty_like(sc_tensor) for _ in range(world_size)]
    dist.all_gather(counts_parts, sc_tensor, group=group)
    counts_matrix = torch.stack(counts_parts, dim=0).cpu()  # [src, dst]

    # recv_counts[src] = how much rank `rank` receives from `src`
    recv_counts = counts_matrix[:, rank].tolist()

    # send offsets within my published buffer
    send_offsets = [0]
    for c in send_counts:
        send_offsets.append(send_offsets[-1] + c)

    # 2) Allocate (or reuse) a symmetric buffer big enough on every rank.
    # Size needed locally = total_send. But peers must agree on a common stride
    # so each rank uses its OWN total_send for layout; we read peer offsets via
    # the counts matrix.
    max_total = int(counts_matrix.sum(dim=1).max().item())
    if max_total == 0:
        return torch.empty(0, dtype=dtype, device=device)

    pub_buf, pub_hdl, pub_cap = _get_data_buf(max_total, device)

    # 3) Pack send data into local symmetric buffer.
    if total_send > 0:
        offset = 0
        for chunk in send_chunks:
            n = chunk.numel()
            if n > 0:
                pub_buf[offset:offset + n].copy_(chunk)
                offset += n

    pub_hdl.barrier(channel=2)

    # 4) Pull from each peer into a local recv buffer.
    total_recv = sum(recv_counts)
    recv = torch.empty(total_recv, dtype=dtype, device=device)
    ext = _get_ext()

    # source-rank offsets in their published buffers (where their chunk to me starts)
    # offset in src rank's pub buffer = sum_{d<rank} counts_matrix[src, d]
    src_offsets_to_me = counts_matrix[:, :rank].sum(dim=1).tolist()

    write_off = 0
    elem_size = 2  # bf16
    for src in range(world_size):
        n = recv_counts[src]
        if n <= 0:
            continue
        peer_base = int(pub_hdl.buffer_ptrs[src])
        peer_ptr = peer_base + src_offsets_to_me[src] * elem_size
        ext.launch_peer_copy_bf16(peer_ptr, recv[write_off:write_off + n], n)
        write_off += n

    pub_hdl.barrier(channel=3)
    return recv


# ---------------------------------------------------------------------------
# Redistribute exact final balanced layout
# ---------------------------------------------------------------------------


def _target_range(rank: int, world_size: int, total: int) -> Tuple[int, int]:
    base = total // world_size
    extra = total % world_size
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end


def _redistribute_exact(merged: torch.Tensor, group) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    sizes = _all_gather_sizes(merged.numel(), merged.device, group)
    total = sum(sizes)

    bucket_start = sum(sizes[:rank])
    bucket_end = bucket_start + merged.numel()
    send_chunks: List[torch.Tensor] = []
    for dest in range(world_size):
        ts, te = _target_range(dest, world_size, total)
        s = max(bucket_start, ts)
        e = min(bucket_end, te)
        if s < e:
            send_chunks.append(merged[s - bucket_start:e - bucket_start].contiguous())
        else:
            send_chunks.append(merged.new_empty(0))
    return _p2p_variable_all_to_all(send_chunks, group)


# ---------------------------------------------------------------------------
# Public solution
# ---------------------------------------------------------------------------


@torch.no_grad()
def solution(local_shard: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    # Ensure ext compiled (rank 0 first, then barrier).
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()

    sorted_local = local_shard.sort().values.contiguous()

    initial_sizes = _all_gather_sizes(local_shard.numel(), local_shard.device, group)
    active_ranks, sort_rank = _active_rank_info(rank, initial_sizes)
    active_count = len(active_ranks)
    if active_count == 0:
        return local_shard.new_empty(0)

    sample_values, sample_ranks, sample_positions = _extract_samples(
        sorted_local, sort_rank, active_count
    )
    splitters = _gather_splitters_via_pg(
        sample_values, sample_ranks, sample_positions, active_count, group
    )
    boundaries = _split_positions(sorted_local, splitters, sort_rank)

    send_chunks = [sorted_local.new_empty(0) for _ in range(world_size)]
    for bucket, dest_rank in enumerate(active_ranks):
        send_chunks[dest_rank] = sorted_local[boundaries[bucket]:boundaries[bucket + 1]].contiguous()

    received = _p2p_variable_all_to_all(send_chunks, group)
    if received.numel() == 0:
        merged = local_shard.new_empty(0)
    else:
        merged = received.sort().values

    return _redistribute_exact(merged, group)