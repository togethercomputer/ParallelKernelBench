"""
Distributed GNN neighbor sampling with custom CUDA kernels and symmetric memory
for device-side all-to-all exchanges.

Strategy:
- Replace dist.all_to_all_single with symmetric-memory based exchanges using
  peer UVA pointers (NVLink P2P on H100).
- Fuse per-rank partitioning, sampling, and reply assembly into custom CUDA
  kernels to eliminate Python-side loops over nodes.
- Use device-side counters and signal pad barriers for synchronization.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// ---------------------------------------------------------------
// Partition src nodes by rank, count and produce per-rank send buffer
// ---------------------------------------------------------------
__global__ void count_partition_kernel(
    const int64_t* __restrict__ src,
    const int64_t* __restrict__ node_to_rank,
    int64_t* __restrict__ partition_ids,
    int64_t* __restrict__ send_counts,
    int64_t n,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int64_t v = src[idx];
    int64_t r = node_to_rank[v];
    partition_ids[idx] = r;
    atomicAdd((unsigned long long*)&send_counts[r], 1ULL);
}

__global__ void scatter_partition_kernel(
    const int64_t* __restrict__ src,
    const int64_t* __restrict__ partition_ids,
    const int64_t* __restrict__ send_offsets, // exclusive prefix sum
    int64_t* __restrict__ partition_orders,
    int64_t* __restrict__ send_buffer,
    int64_t* __restrict__ counter,           // per-rank running counters
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int64_t r = partition_ids[idx];
    int64_t order = atomicAdd((unsigned long long*)&counter[r], 1ULL);
    partition_orders[idx] = order;
    int64_t pos = send_offsets[r] + order;
    send_buffer[pos] = src[idx];
}

// ---------------------------------------------------------------
// CSC neighbor sampling kernel
// Each thread handles one node in recv_nodes
// ---------------------------------------------------------------
__global__ void csc_sample_count_kernel(
    const int64_t* __restrict__ nodes,
    const int64_t* __restrict__ colptr,
    int64_t* __restrict__ counts,  // size n
    int64_t* __restrict__ degs,    // size n
    int64_t n,
    int k
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int64_t v = nodes[idx];
    int64_t start = colptr[v];
    int64_t end = colptr[v + 1];
    int64_t deg = end - start;
    int64_t take = (k >= 0) ? min((int64_t)k, deg) : deg;
    counts[idx] = take;
    degs[idx] = deg;
}

__device__ __forceinline__ uint32_t lcg_step(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

__global__ void csc_sample_fill_kernel(
    const int64_t* __restrict__ nodes,
    const int64_t* __restrict__ colptr,
    const int64_t* __restrict__ row,
    const int64_t* __restrict__ counts,
    const int64_t* __restrict__ degs,
    const int64_t* __restrict__ offsets, // exclusive prefix sum of counts
    int64_t* __restrict__ out_nodes,
    int64_t* __restrict__ out_edges,
    int64_t n,
    int replace,
    uint64_t seed
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int64_t v = nodes[idx];
    int64_t start = colptr[v];
    int64_t take = counts[idx];
    int64_t deg = degs[idx];
    int64_t out_off = offsets[idx];

    uint32_t s = (uint32_t)(seed ^ ((uint64_t)v * 2654435761ULL) ^ (uint64_t)idx);

    if (take == 0) return;

    if (replace) {
        for (int64_t i = 0; i < take; ++i) {
            uint32_t r = lcg_step(s);
            int64_t pick = (int64_t)(r % (uint32_t)deg);
            out_nodes[out_off + i] = row[start + pick];
            out_edges[out_off + i] = start + pick;
        }
    } else {
        // Reservoir / Fisher-Yates partial: for small take, sample without replacement
        // Use a simple approach: if take == deg, pick all in order; else use rejection
        if (take == deg) {
            for (int64_t i = 0; i < deg; ++i) {
                out_nodes[out_off + i] = row[start + i];
                out_edges[out_off + i] = start + i;
            }
        } else {
            // Floyd's algorithm requires set membership; fall back to local array if small
            // For simplicity, use rejection with bitmap up to 64 elements
            // For larger, do Fisher-Yates in-place using stride pattern
            // Here: do partial Fisher-Yates by storing already-picked indices
            // We assume take is small (typical fanout 5-25)
            const int MAX_PICKS = 64;
            int64_t picks[MAX_PICKS];
            int64_t picked = 0;
            for (int64_t i = 0; i < take && picked < MAX_PICKS; ++i) {
                bool ok = false;
                int64_t cand = 0;
                int attempts = 0;
                while (!ok && attempts < 100) {
                    uint32_t r = lcg_step(s);
                    cand = (int64_t)(r % (uint32_t)deg);
                    ok = true;
                    for (int64_t j = 0; j < picked; ++j) {
                        if (picks[j] == cand) { ok = false; break; }
                    }
                    attempts++;
                }
                picks[picked++] = cand;
                out_nodes[out_off + i] = row[start + cand];
                out_edges[out_off + i] = start + cand;
            }
        }
    }
}

// ---------------------------------------------------------------
// Compute send_node_counts: sum sampled_counts per receiving rank's chunk
// ---------------------------------------------------------------
__global__ void sum_per_rank_kernel(
    const int64_t* __restrict__ sampled_counts,
    const int64_t* __restrict__ recv_offsets, // size world+1
    int64_t* __restrict__ send_node_counts,
    int world_size
) {
    int r = blockIdx.x;
    if (r >= world_size) return;
    int64_t start = recv_offsets[r];
    int64_t end = recv_offsets[r + 1];
    int64_t sum = 0;
    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        sum += sampled_counts[i];
    }
    __shared__ int64_t shm[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    if (lane == 0) shm[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        sum = (threadIdx.x < (blockDim.x + 31) / 32) ? shm[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
        if (threadIdx.x == 0) send_node_counts[r] = sum;
    }
}

// ---------------------------------------------------------------
// Reorder reply nodes/edges based on grouped_index, also expand dst
// ---------------------------------------------------------------
__global__ void reorder_replies_kernel(
    const int64_t* __restrict__ reply_nodes,
    const int64_t* __restrict__ reply_edges,
    const int64_t* __restrict__ reply_counts,    // size n_src
    const int64_t* __restrict__ reply_offsets,   // exclusive prefix size n_src
    const int64_t* __restrict__ grouped_index,   // permutation, size n_src
    const int64_t* __restrict__ src,             // dst nodes, size n_src
    const int64_t* __restrict__ ordered_offsets, // exclusive prefix of reply_counts[grouped_index], size n_src
    int64_t* __restrict__ out_nodes,
    int64_t* __restrict__ out_edges,
    int64_t* __restrict__ out_dst,
    int64_t n_src
) {
    int64_t i = blockIdx.x;
    if (i >= n_src) return;
    int64_t gi = grouped_index[i];
    int64_t cnt = reply_counts[gi];
    int64_t src_off = reply_offsets[gi];
    int64_t dst_off = ordered_offsets[i];
    int64_t dst_node = src[i];
    for (int64_t j = threadIdx.x; j < cnt; j += blockDim.x) {
        out_nodes[dst_off + j] = reply_nodes[src_off + j];
        out_edges[dst_off + j] = reply_edges[src_off + j];
        out_dst[dst_off + j] = dst_node;
    }
}

// ---------------------------------------------------------------
// Symmetric memory all-to-all-v: each rank writes its data into peer's buffer
// at known offsets. Use single-block per peer.
// ---------------------------------------------------------------
__global__ void p2p_alltoallv_int64_kernel(
    const int64_t* __restrict__ send_data,
    const int64_t* __restrict__ send_offsets,    // size world+1
    const int64_t* __restrict__ peer_buf_ptrs,   // size world: peer's recv buffer base
    const int64_t* __restrict__ peer_recv_offsets_ptrs, // size world: peer's recv_offsets array on each peer
    int rank,
    int world_size
) {
    int peer = blockIdx.x;
    if (peer >= world_size) return;
    int64_t my_send_start = send_offsets[peer];
    int64_t my_send_end = send_offsets[peer + 1];
    int64_t cnt = my_send_end - my_send_start;
    if (cnt == 0) return;
    // Where does peer expect my data? At peer's recv_offsets[rank]
    const int64_t* peer_recv_offsets = (const int64_t*)peer_recv_offsets_ptrs[peer];
    int64_t* peer_buf = (int64_t*)peer_buf_ptrs[peer];
    int64_t dst_start = peer_recv_offsets[rank];
    for (int64_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        peer_buf[dst_start + i] = send_data[my_send_start + i];
    }
}

void launch_count_partition(torch::Tensor src, torch::Tensor node_to_rank,
                             torch::Tensor partition_ids, torch::Tensor send_counts,
                             int64_t n, int world_size) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    count_partition_kernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<int64_t>(), node_to_rank.data_ptr<int64_t>(),
        partition_ids.data_ptr<int64_t>(), send_counts.data_ptr<int64_t>(),
        n, world_size);
}

void launch_scatter_partition(torch::Tensor src, torch::Tensor partition_ids,
                               torch::Tensor send_offsets, torch::Tensor partition_orders,
                               torch::Tensor send_buffer, torch::Tensor counter,
                               int64_t n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scatter_partition_kernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<int64_t>(), partition_ids.data_ptr<int64_t>(),
        send_offsets.data_ptr<int64_t>(), partition_orders.data_ptr<int64_t>(),
        send_buffer.data_ptr<int64_t>(), counter.data_ptr<int64_t>(), n);
}

void launch_csc_sample_count(torch::Tensor nodes, torch::Tensor colptr,
                              torch::Tensor counts, torch::Tensor degs,
                              int64_t n, int64_t k) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    csc_sample_count_kernel<<<blocks, threads, 0, stream>>>(
        nodes.data_ptr<int64_t>(), colptr.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(), degs.data_ptr<int64_t>(), n, (int)k);
}

void launch_csc_sample_fill(torch::Tensor nodes, torch::Tensor colptr, torch::Tensor row,
                             torch::Tensor counts, torch::Tensor degs, torch::Tensor offsets,
                             torch::Tensor out_nodes, torch::Tensor out_edges,
                             int64_t n, int64_t replace, int64_t seed) {
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    csc_sample_fill_kernel<<<blocks, threads, 0, stream>>>(
        nodes.data_ptr<int64_t>(), colptr.data_ptr<int64_t>(), row.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(), degs.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        out_nodes.data_ptr<int64_t>(), out_edges.data_ptr<int64_t>(),
        n, (int)replace, (uint64_t)seed);
}

void launch_sum_per_rank(torch::Tensor sampled_counts, torch::Tensor recv_offsets,
                          torch::Tensor send_node_counts, int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    sum_per_rank_kernel<<<world_size, 256, 0, stream>>>(
        sampled_counts.data_ptr<int64_t>(), recv_offsets.data_ptr<int64_t>(),
        send_node_counts.data_ptr<int64_t>(), world_size);
}

void launch_reorder_replies(torch::Tensor reply_nodes, torch::Tensor reply_edges,
                             torch::Tensor reply_counts, torch::Tensor reply_offsets,
                             torch::Tensor grouped_index, torch::Tensor src,
                             torch::Tensor ordered_offsets, torch::Tensor out_nodes,
                             torch::Tensor out_edges, torch::Tensor out_dst, int64_t n_src) {
    if (n_src == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reorder_replies_kernel<<<n_src, 64, 0, stream>>>(
        reply_nodes.data_ptr<int64_t>(), reply_edges.data_ptr<int64_t>(),
        reply_counts.data_ptr<int64_t>(), reply_offsets.data_ptr<int64_t>(),
        grouped_index.data_ptr<int64_t>(), src.data_ptr<int64_t>(),
        ordered_offsets.data_ptr<int64_t>(),
        out_nodes.data_ptr<int64_t>(), out_edges.data_ptr<int64_t>(),
        out_dst.data_ptr<int64_t>(), n_src);
}

void launch_p2p_alltoallv(torch::Tensor send_data, torch::Tensor send_offsets,
                           torch::Tensor peer_buf_ptrs, torch::Tensor peer_recv_offsets_ptrs,
                           int rank, int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    p2p_alltoallv_int64_kernel<<<world_size, 256, 0, stream>>>(
        send_data.data_ptr<int64_t>(), send_offsets.data_ptr<int64_t>(),
        peer_buf_ptrs.data_ptr<int64_t>(), peer_recv_offsets_ptrs.data_ptr<int64_t>(),
        rank, world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_partition", &launch_count_partition, "");
    m.def("scatter_partition", &launch_scatter_partition, "");
    m.def("csc_sample_count", &launch_csc_sample_count, "");
    m.def("csc_sample_fill", &launch_csc_sample_fill, "");
    m.def("sum_per_rank", &launch_sum_per_rank, "");
    m.def("reorder_replies", &launch_reorder_replies, "");
    m.def("p2p_alltoallv", &launch_p2p_alltoallv, "");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_sampling_ext", CUDA_SRC)
    return _ext


# Symmetric memory buffers reused across calls
_SYMM_CACHE = {}

def _get_symm_buf(name: str, size: int, dtype: torch.dtype, device: torch.device, group):
    key = (name, dtype)
    cur = _SYMM_CACHE.get(key)
    if cur is not None and cur[0].numel() >= size:
        return cur
    cap = max(size, 1)
    # Grow with some headroom
    cap = max(cap, 1024)
    if cur is not None:
        cap = max(cap, cur[0].numel() * 2)
    buf = symm_mem.empty(cap, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    res = (buf, hdl, ptrs)
    _SYMM_CACHE[key] = res
    return res


def _ensure_symm_capacity(name: str, size: int, dtype: torch.dtype, device: torch.device, group):
    """Get or grow a symmetric buffer; rendezvous is collective so all ranks must agree on size."""
    return _get_symm_buf(name, size, dtype, device, group)


def _alltoallv_symm(send_data: torch.Tensor, send_counts: torch.Tensor, group, name_prefix: str):
    """Device-side all-to-all-v using symmetric memory.
    send_data: int64 1D contiguous on device
    send_counts: int64 1D length world_size on device
    Returns recv_data, recv_counts (int64 device tensors)
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = send_data.device

    # Step 1: exchange counts (small, fixed-size). We need recv_counts on all ranks.
    # Use a symmetric buffer of size world*world; each rank writes its send_counts into peer's row.
    # Simpler: use NCCL all_to_all_single for the small counts (low overhead) -> but spec wants device-side.
    # Instead, write counts via symmetric memory.
    counts_size = world_size * world_size
    cbuf, chdl, cptrs = _ensure_symm_capacity("counts_" + name_prefix, counts_size, torch.int64, device, group)

    # Each rank writes send_counts[r] into peer r's row at position rank.
    # We do this via direct python-driven peer pointer writes? Easier: write entire send_counts into our own row of every peer.
    # Use a small kernel: we already have generic p2p_alltoallv but that's for variable; here it's fixed=1 element per peer.
    # Simpler: build a tiny send buffer of length world_size where send_buffer[r] = send_counts[r], 1 element each.
    # Then offsets are [0,1,2,...,world]. Reuse p2p kernel.

    # Build send_offsets for counts: each rank sends 1 to each peer
    # send_buffer is just send_counts itself.
    # Recv layout: each peer's row has world_size entries; rank r's data goes at position rank in peer's recv_offsets.
    # We need a "recv_offsets" array of size world+1 on each rank: [0,1,2,...,world].
    fixed_offsets = torch.arange(world_size + 1, dtype=torch.int64, device=device)

    # Allocate per-rank recv_offsets in a symmetric buffer so peers can read.
    rofs_buf, rofs_hdl, rofs_ptrs = _ensure_symm_capacity("rofs_fixed", world_size + 1, torch.int64, device, group)
    rofs_buf[:world_size + 1].copy_(fixed_offsets)
    rofs_hdl.barrier(channel=0)

    # Source send buffer for counts: just send_counts (length world_size), offsets [0,1,2,...]
    sbuf, shdl, sptrs = _ensure_symm_capacity("sbuf_" + name_prefix + "_counts", world_size, torch.int64, device, group)
    sbuf[:world_size].copy_(send_counts)
    shdl.barrier(channel=0)

    # cbuf is symmetric recv buffer
    chdl.barrier(channel=0)

    _get_ext().p2p_alltoallv(sbuf, fixed_offsets, cptrs, rofs_ptrs, rank, world_size)
    chdl.barrier(channel=0)

    # Now cbuf[0:world_size] contains my recv_counts (peer r wrote send_counts[rank] from peer r's perspective into cbuf[r])
    recv_counts = cbuf[:world_size].clone()

    # Step 2: exchange variable data
    total_recv = int(recv_counts.sum().item())
    total_send = int(send_counts.sum().item())

    # Compute send_offsets and recv_offsets (exclusive prefix)
    send_offsets = torch.zeros(world_size + 1, dtype=torch.int64, device=device)
    send_offsets[1:] = torch.cumsum(send_counts, dim=0)
    recv_offsets = torch.zeros(world_size + 1, dtype=torch.int64, device=device)
    recv_offsets[1:] = torch.cumsum(recv_counts, dim=0)

    # Allocate symmetric send buffer (we copy send_data into it)
    if total_send == 0 and total_recv == 0:
        return torch.empty(0, dtype=torch.int64, device=device), recv_counts

    sdbuf, sdhdl, sdptrs = _ensure_symm_capacity("sdbuf_" + name_prefix, max(total_send, 1), torch.int64, device, group)
    if total_send > 0:
        sdbuf[:total_send].copy_(send_data[:total_send])

    rdbuf, rdhdl, rdptrs = _ensure_symm_capacity("rdbuf_" + name_prefix, max(total_recv, 1), torch.int64, device, group)

    # Each rank's recv_offsets must be readable by peers
    rofbuf, rofhdl, rofptrs = _ensure_symm_capacity("rofbuf_" + name_prefix, world_size + 1, torch.int64, device, group)
    rofbuf[:world_size + 1].copy_(recv_offsets)

    sdhdl.barrier(channel=0)
    rdhdl.barrier(channel=0)
    rofhdl.barrier(channel=0)

    _get_ext().p2p_alltoallv(sdbuf, send_offsets, rdptrs, rofptrs, rank, world_size)
    rdhdl.barrier(channel=0)

    recv_data = rdbuf[:total_recv].clone() if total_recv > 0 else torch.empty(0, dtype=torch.int64, device=device)
    return recv_data, recv_counts


def _sample_one_hop_csc_cuda(nodes: torch.Tensor, k: int, colptr: torch.Tensor, row: torch.Tensor, replace: bool):
    """CUDA neighbor sample. Returns (out_nodes, out_edges, sampled_counts)."""
    n = nodes.numel()
    device = nodes.device
    if n == 0:
        z = torch.empty(0, dtype=torch.long, device=device)
        return z, z, z
    counts = torch.empty(n, dtype=torch.long, device=device)
    degs = torch.empty(n, dtype=torch.long, device=device)
    _get_ext().csc_sample_count(nodes, colptr, counts, degs, n, int(k))
    offsets = torch.zeros(n + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    total = int(offsets[-1].item())
    out_nodes = torch.empty(total, dtype=torch.long, device=device)
    out_edges = torch.empty(total, dtype=torch.long, device=device)
    if total > 0:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
        _get_ext().csc_sample_fill(nodes, colptr, row, counts, degs, offsets[:-1].contiguous(),
                                    out_nodes, out_edges, n, int(replace), int(seed))
    return out_nodes, out_edges, counts


@torch.no_grad()
def solution(
    seed_nodes: torch.Tensor,
    fanouts: List[int],
    local_adj_row_ptr: torch.Tensor,
    local_adj_col: torch.Tensor,
    node_to_rank: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    replace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = seed_nodes.device

    # Compile extension on rank 0 first
    if rank == 0:
        _get_ext()
    dist.barrier()
    _get_ext()

    seed = seed_nodes.to(dtype=torch.long, device=device).contiguous()
    node_to_rank = node_to_rank.to(dtype=torch.long, device=device).contiguous()
    local_adj_row_ptr = local_adj_row_ptr.to(dtype=torch.long, device=device).contiguous()
    local_adj_col = local_adj_col.to(dtype=torch.long, device=device).contiguous()

    src = seed.clone()
    node = src.clone()
    node_with_dupl = [seed.new_empty(0)]
    dst_with_dupl = [seed.new_empty(0)]
    edge_list = [seed.new_empty(0)]

    for fanout in fanouts:
        # Synchronize: all ranks must continue together (since collectives below are collective)
        # Use a small all-reduce of "are we still going" via dist; cheap.
        local_alive = torch.tensor([1 if src.numel() > 0 else 0], dtype=torch.long, device=device)
        # Actually we must have all ranks participate even if empty, since other ranks may need data from us.
        # So do not break early on individual ranks. Let an all-reduce decide global termination.
        any_alive = local_alive.clone()
        dist.all_reduce(any_alive, op=dist.ReduceOp.SUM, group=group)
        if int(any_alive.item()) == 0:
            break

        n_src = src.numel()

        if n_src > 0:
            partition_ids = torch.empty(n_src, dtype=torch.long, device=device)
            send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
            _get_ext().count_partition(src, node_to_rank, partition_ids, send_counts, n_src, world_size)

            send_offsets = torch.zeros(world_size + 1, dtype=torch.long, device=device)
            send_offsets[1:] = torch.cumsum(send_counts, dim=0)
            partition_orders = torch.empty(n_src, dtype=torch.long, device=device)
            counter = torch.zeros(world_size, dtype=torch.long, device=device)
            send_buffer = torch.empty(n_src, dtype=torch.long, device=device)
            _get_ext().scatter_partition(src, partition_ids, send_offsets, partition_orders,
                                          send_buffer, counter, n_src)
        else:
            partition_ids = torch.empty(0, dtype=torch.long, device=device)
            partition_orders = torch.empty(0, dtype=torch.long, device=device)
            send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
            send_buffer = torch.empty(0, dtype=torch.long, device=device)

        # Exchange nodes via symmetric memory
        recv_nodes, recv_counts = _alltoallv_symm(send_buffer, send_counts, group, "nodes")

        # Sample on this rank
        sampled_nodes, sampled_edges, sampled_counts = _sample_one_hop_csc_cuda(
            recv_nodes, int(fanout), local_adj_row_ptr, local_adj_col, replace
        )

        # Compute send_node_counts per receiving rank (sum sampled_counts within each chunk)
        recv_offsets = torch.zeros(world_size + 1, dtype=torch.long, device=device)
        recv_offsets[1:] = torch.cumsum(recv_counts, dim=0)
        send_node_counts = torch.zeros(world_size, dtype=torch.long, device=device)
        _get_ext().sum_per_rank(sampled_counts, recv_offsets, send_node_counts, world_size)

        # Exchange replies: nodes, edges, and per-source-node counts
        reply_nodes, _ = _alltoallv_symm(sampled_nodes, send_node_counts, group, "rnodes")
        reply_edges, _ = _alltoallv_symm(sampled_edges, send_node_counts, group, "redges")
        reply_counts, _ = _alltoallv_symm(sampled_counts, recv_counts, group, "rcounts")

        # Now reorder back to original src order
        if n_src > 0 and reply_counts.numel() > 0:
            # rank_offsets[r] = start index in send order of rank r
            rank_offsets = torch.zeros(world_size, dtype=torch.long, device=device)
            rank_offsets[1:] = torch.cumsum(send_counts, dim=0)[:-1]
            grouped_index = rank_offsets[partition_ids] + partition_orders

            reply_offsets = torch.zeros(reply_counts.numel() + 1, dtype=torch.long, device=device)
            reply_offsets[1:] = torch.cumsum(reply_counts, dim=0)

            # ordered_offsets: prefix of reply_counts[grouped_index]
            reordered_counts = reply_counts[grouped_index]
            ordered_offsets = torch.zeros(n_src + 1, dtype=torch.long, device=device)
            ordered_offsets[1:] = torch.cumsum(reordered_counts, dim=0)
            total_out = int(ordered_offsets[-1].item())

            out_node = torch.empty(total_out, dtype=torch.long, device=device)
            out_edge = torch.empty(total_out, dtype=torch.long, device=device)
            out_dst = torch.empty(total_out, dtype=torch.long, device=device)
            if total_out > 0:
                _get_ext().reorder_replies(reply_nodes, reply_edges, reply_counts,
                                            reply_offsets[:-1].contiguous(),
                                            grouped_index, src,
                                            ordered_offsets[:-1].contiguous(),
                                            out_node, out_edge, out_dst, n_src)
        else:
            out_node = seed.new_empty(0)
            out_edge = seed.new_empty(0)
            out_dst = seed.new_empty(0)

        if out_node.numel() == 0:
            src = seed.new_empty(0)
            continue

        # Dedup against accumulated node set
        # PyG remove_duplicates: stable first-occurrence in [node | out_node]
        node_combined = torch.cat([node, out_node])
        nc_np = node_combined.cpu().numpy()
        _, idx = np.unique(nc_np, return_index=True)
        idx_t = torch.from_numpy(idx).to(device).sort().values
        num_nodes_prev = node.numel()
        node = node_combined[idx_t]
        src = node_combined[idx_t[idx_t >= num_nodes_prev]]

        node_with_dupl.append(out_node)
        dst_with_dupl.append(out_dst)
        edge_list.append(out_edge)

    node_dupl = torch.cat(node_with_dupl)
    dst_dupl = torch.cat(dst_with_dupl)

    # Relabel
    if node_dupl.numel() == 0:
        row_out = node.new_empty(0)
        col_out = node.new_empty(0)
    else:
        max_id = int(node.max().item()) + 1
        assoc = torch.full((max_id,), -1, dtype=torch.long, device=device)
        assoc[node] = torch.arange(node.numel(), device=device)
        row_out = assoc[node_dupl]
        col_out = assoc[dst_dupl]

    return node, row_out, col_out, torch.cat(edge_list)