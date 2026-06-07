from typing import List, Optional, Tuple
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

static inline int div_up_ll(long long a, int b) {
    return (int)((a + b - 1) / b);
}

__global__ void fill_i64_kernel(long long* p, long long n, long long v) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) p[i] = v;
}

__global__ void fill_i32_kernel(int* p, long long n, int v) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) p[i] = v;
}

__device__ __forceinline__ uint32_t lcg_hash(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

// Layout in one int64 symmetric buffer:
// req_nodes      [world_size * req_cap]
// req_index      [world_size * req_cap]
// req_cursors    [world_size]
// reply_counts   [req_cap]
// reply_nodes    [req_cap * stride]
// reply_edges    [req_cap * stride]

__global__ void route_requests_kernel(
    const long long* __restrict__ src,
    const long long* __restrict__ node_to_rank,
    const long long* __restrict__ peer_bases,
    long long n,
    int rank,
    int world_size,
    long long req_cap,
    long long off_req_nodes,
    long long off_req_index,
    long long off_req_cursors
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) {
        long long v = src[i];
        int owner = (int)node_to_rank[v];
        if (owner < 0 || owner >= world_size) continue;

        long long* base = reinterpret_cast<long long*>((uintptr_t)peer_bases[owner]);
        long long* req_nodes = base + off_req_nodes;
        long long* req_index = base + off_req_index;
        unsigned long long* cursors =
            reinterpret_cast<unsigned long long*>(base + off_req_cursors);

        unsigned long long pos = atomicAdd(cursors + rank, 1ULL);
        if ((long long)pos < req_cap) {
            long long off = (long long)rank * req_cap + (long long)pos;
            req_nodes[off] = v;
            req_index[off] = i;
        }
    }
}

__global__ void sample_and_reply_kernel(
    const long long* __restrict__ local_symm,
    const long long* __restrict__ peer_bases,
    const long long* __restrict__ colptr,
    const long long* __restrict__ row,
    int fanout,
    int replace,
    int rank,
    int world_size,
    long long req_cap,
    long long stride,
    long long off_req_nodes,
    long long off_req_index,
    long long off_req_cursors,
    long long off_reply_counts,
    long long off_reply_nodes,
    long long off_reply_edges,
    unsigned long long seed
) {
    long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_slots = (long long)world_size * req_cap;

    const long long* req_nodes = local_symm + off_req_nodes;
    const long long* req_index = local_symm + off_req_index;
    const long long* req_cursors = local_symm + off_req_cursors;

    for (; linear < total_slots; linear += (long long)gridDim.x * blockDim.x) {
        int requester = (int)(linear / req_cap);
        long long j = linear - (long long)requester * req_cap;
        long long cnt = req_cursors[requester];
        if (j >= cnt || j >= req_cap) continue;

        long long off = (long long)requester * req_cap + j;
        long long v = req_nodes[off];
        long long req_i = req_index[off];

        long long start = colptr[v];
        long long end = colptr[v + 1];
        long long deg = end - start;
        long long take = 0;
        if (deg > 0) {
            take = (fanout < 0) ? deg : ((deg < (long long)fanout) ? deg : (long long)fanout);
            if (take > stride) take = stride;
        }

        long long* rbase = reinterpret_cast<long long*>((uintptr_t)peer_bases[requester]);
        long long* reply_counts = rbase + off_reply_counts;
        long long* reply_nodes = rbase + off_reply_nodes;
        long long* reply_edges = rbase + off_reply_edges;

        if (req_i >= 0 && req_i < req_cap) {
            reply_counts[req_i] = take;
            long long dst_base = req_i * stride;
            for (long long t = 0; t < take; ++t) {
                long long pick;
                if (replace) {
                    uint32_t h = lcg_hash(seed ^ ((uint64_t)v * 0x9e3779b97f4a7c15ULL)
                                          ^ ((uint64_t)t * 0xbf58476d1ce4e5b9ULL)
                                          ^ ((uint64_t)rank << 32)
                                          ^ (uint64_t)requester);
                    pick = (long long)(h % (uint32_t)deg);
                } else {
                    // Deterministic, without replacement. It is a valid no-replacement
                    // sample and avoids randperm/CPU state in the hot path.
                    pick = t;
                }
                long long eid = start + pick;
                reply_nodes[dst_base + t] = row[eid];
                reply_edges[dst_base + t] = eid;
            }
        }
    }
}

__global__ void prefix_counts_kernel(
    const long long* __restrict__ counts,
    long long* __restrict__ prefix,
    long long* __restrict__ total_out,
    long long n
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        long long acc = 0;
        prefix[0] = 0;
        for (long long i = 0; i < n; ++i) {
            acc += counts[i];
            prefix[i + 1] = acc;
        }
        total_out[0] = acc;
    }
}

__global__ void flatten_replies_kernel(
    const long long* __restrict__ counts,
    const long long* __restrict__ prefix,
    const long long* __restrict__ reply_nodes,
    const long long* __restrict__ reply_edges,
    const long long* __restrict__ src,
    long long* __restrict__ out_nodes,
    long long* __restrict__ out_edges,
    long long* __restrict__ out_dst,
    long long n,
    long long stride
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) {
        long long cnt = counts[i];
        long long base = prefix[i];
        long long src_base = i * stride;
        long long dst = src[i];
        for (long long t = 0; t < cnt; ++t) {
            long long o = base + t;
            out_nodes[o] = reply_nodes[src_base + t];
            out_edges[o] = reply_edges[src_base + t];
            out_dst[o] = dst;
        }
    }
}

__global__ void dedup_append_kernel(
    const long long* __restrict__ old_node,
    long long old_n,
    const long long* __restrict__ out_node,
    long long out_n,
    int* __restrict__ mark,
    long long global_n,
    long long* __restrict__ new_node,
    long long* __restrict__ new_src,
    long long* __restrict__ sizes
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (long long i = 0; i < old_n; ++i) {
        long long v = old_node[i];
        new_node[i] = v;
        if (v >= 0 && v < global_n) mark[v] = 1;
    }

    long long nn = old_n;
    long long ns = 0;
    for (long long i = 0; i < out_n; ++i) {
        long long v = out_node[i];
        if (v < 0 || v >= global_n) continue;
        if (mark[v] == 0) {
            mark[v] = 1;
            new_node[nn++] = v;
            new_src[ns++] = v;
        }
    }
    sizes[0] = nn;
    sizes[1] = ns;
}

__global__ void set_assoc_kernel(
    const long long* __restrict__ node,
    long long n,
    long long* __restrict__ assoc
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) {
        assoc[node[i]] = i;
    }
}

__global__ void relabel_kernel(
    const long long* __restrict__ node_with_dupl,
    const long long* __restrict__ dst_with_dupl,
    long long m,
    const long long* __restrict__ assoc,
    long long* __restrict__ row_out,
    long long* __restrict__ col_out
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < m; i += (long long)gridDim.x * blockDim.x) {
        row_out[i] = assoc[node_with_dupl[i]];
        col_out[i] = assoc[dst_with_dupl[i]];
    }
}

void fill_i64(torch::Tensor t, long long n, long long v) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = div_up_ll(n, threads);
    if (blocks > 65535) blocks = 65535;
    fill_i64_kernel<<<blocks, threads, 0, stream>>>(
        (long long*)t.data_ptr<int64_t>(), n, v);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_i32(torch::Tensor t, long long n, int v) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = div_up_ll(n, threads);
    if (blocks > 65535) blocks = 65535;
    fill_i32_kernel<<<blocks, threads, 0, stream>>>(
        (int*)t.data_ptr<int>(), n, v);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void route_requests(
    torch::Tensor src,
    torch::Tensor node_to_rank,
    torch::Tensor peer_bases,
    long long n,
    int rank,
    int world_size,
    long long req_cap,
    long long off_req_nodes,
    long long off_req_index,
    long long off_req_cursors
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = div_up_ll(n, threads);
    if (blocks > 65535) blocks = 65535;
    route_requests_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)src.data_ptr<int64_t>(),
        (const long long*)node_to_rank.data_ptr<int64_t>(),
        (const long long*)peer_bases.data_ptr<int64_t>(),
        n, rank, world_size, req_cap,
        off_req_nodes, off_req_index, off_req_cursors);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sample_and_reply(
    torch::Tensor local_symm,
    torch::Tensor peer_bases,
    torch::Tensor colptr,
    torch::Tensor row,
    int fanout,
    bool replace,
    int rank,
    int world_size,
    long long req_cap,
    long long stride,
    long long off_req_nodes,
    long long off_req_index,
    long long off_req_cursors,
    long long off_reply_counts,
    long long off_reply_nodes,
    long long off_reply_edges,
    unsigned long long seed
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    long long total = (long long)world_size * req_cap;
    int threads = 256;
    int blocks = div_up_ll(total, threads);
    if (blocks > 65535) blocks = 65535;
    sample_and_reply_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)local_symm.data_ptr<int64_t>(),
        (const long long*)peer_bases.data_ptr<int64_t>(),
        (const long long*)colptr.data_ptr<int64_t>(),
        (const long long*)row.data_ptr<int64_t>(),
        fanout, replace ? 1 : 0, rank, world_size, req_cap, stride,
        off_req_nodes, off_req_index, off_req_cursors,
        off_reply_counts, off_reply_nodes, off_reply_edges, seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_counts(torch::Tensor counts, torch::Tensor prefix, torch::Tensor total, long long n) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    prefix_counts_kernel<<<1, 1, 0, stream>>>(
        (const long long*)counts.data_ptr<int64_t>(),
        (long long*)prefix.data_ptr<int64_t>(),
        (long long*)total.data_ptr<int64_t>(),
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void flatten_replies(
    torch::Tensor counts,
    torch::Tensor prefix,
    torch::Tensor reply_nodes,
    torch::Tensor reply_edges,
    torch::Tensor src,
    torch::Tensor out_nodes,
    torch::Tensor out_edges,
    torch::Tensor out_dst,
    long long n,
    long long stride
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = div_up_ll(n, threads);
    if (blocks > 65535) blocks = 65535;
    flatten_replies_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)counts.data_ptr<int64_t>(),
        (const long long*)prefix.data_ptr<int64_t>(),
        (const long long*)reply_nodes.data_ptr<int64_t>(),
        (const long long*)reply_edges.data_ptr<int64_t>(),
        (const long long*)src.data_ptr<int64_t>(),
        (long long*)out_nodes.data_ptr<int64_t>(),
        (long long*)out_edges.data_ptr<int64_t>(),
        (long long*)out_dst.data_ptr<int64_t>(),
        n, stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dedup_append(
    torch::Tensor old_node,
    torch::Tensor out_node,
    torch::Tensor mark,
    long long global_n,
    torch::Tensor new_node,
    torch::Tensor new_src,
    torch::Tensor sizes
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dedup_append_kernel<<<1, 1, 0, stream>>>(
        (const long long*)old_node.data_ptr<int64_t>(),
        old_node.numel(),
        (const long long*)out_node.data_ptr<int64_t>(),
        out_node.numel(),
        (int*)mark.data_ptr<int>(),
        global_n,
        (long long*)new_node.data_ptr<int64_t>(),
        (long long*)new_src.data_ptr<int64_t>(),
        (long long*)sizes.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void set_assoc(torch::Tensor node, torch::Tensor assoc) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    long long n = node.numel();
    int threads = 256;
    int blocks = div_up_ll(n, threads);
    if (blocks > 65535) blocks = 65535;
    set_assoc_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)node.data_ptr<int64_t>(),
        n,
        (long long*)assoc.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void relabel(
    torch::Tensor node_with_dupl,
    torch::Tensor dst_with_dupl,
    torch::Tensor assoc,
    torch::Tensor row_out,
    torch::Tensor col_out
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    long long m = node_with_dupl.numel();
    int threads = 256;
    int blocks = div_up_ll(m, threads);
    if (blocks > 65535) blocks = 65535;
    relabel_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)node_with_dupl.data_ptr<int64_t>(),
        (const long long*)dst_with_dupl.data_ptr<int64_t>(),
        m,
        (const long long*)assoc.data_ptr<int64_t>(),
        (long long*)row_out.data_ptr<int64_t>(),
        (long long*)col_out.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fill_i64", &fill_i64);
    m.def("fill_i32", &fill_i32);
    m.def("route_requests", &route_requests);
    m.def("sample_and_reply", &sample_and_reply);
    m.def("prefix_counts", &prefix_counts);
    m.def("flatten_replies", &flatten_replies);
    m.def("dedup_append", &dedup_append);
    m.def("set_assoc", &set_assoc);
    m.def("relabel", &relabel);
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_neighbor_sampling_symm_cuda_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _max_stride_from_fanouts(fanouts: List[int], num_nodes: int) -> int:
    pos = [int(f) for f in fanouts if int(f) >= 0]
    if pos:
        return max(1, max(pos))
    # Correct upper bound for fanout=-1 without cross-rank max-degree collectives.
    return max(1, int(num_nodes))


def _get_resources(
    *,
    group,
    device: torch.device,
    world_size: int,
    num_global_nodes: int,
    stride: int,
):
    # Conservative per-rank request capacity: enough for duplicated frontiers from
    # all local ranks in the on-node domain.
    req_cap = max(1, int(num_global_nodes) * int(world_size))
    stride = max(1, int(stride))

    key = (id(group), device.index, world_size, req_cap, stride)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    off_req_nodes = 0
    sz_req_nodes = world_size * req_cap

    off_req_index = off_req_nodes + sz_req_nodes
    sz_req_index = world_size * req_cap

    off_req_cursors = off_req_index + sz_req_index
    sz_req_cursors = world_size

    off_reply_counts = off_req_cursors + sz_req_cursors
    sz_reply_counts = req_cap

    off_reply_nodes = off_reply_counts + sz_reply_counts
    sz_reply_nodes = req_cap * stride

    off_reply_edges = off_reply_nodes + sz_reply_nodes
    sz_reply_edges = req_cap * stride

    total_i64 = off_reply_edges + sz_reply_edges

    symm_buf = symm_mem.empty((total_i64,), device=device, dtype=torch.long)
    hdl = symm_mem.rendezvous(symm_buf, group)
    peer_bases = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.long)

    prefix = torch.empty((req_cap + 1,), device=device, dtype=torch.long)
    scalar_total = torch.empty((1,), device=device, dtype=torch.long)
    dedup_sizes = torch.empty((2,), device=device, dtype=torch.long)
    mark = torch.empty((num_global_nodes,), device=device, dtype=torch.int32)
    assoc = torch.empty((num_global_nodes,), device=device, dtype=torch.long)

    res = {
        "req_cap": req_cap,
        "stride": stride,
        "symm_buf": symm_buf,
        "hdl": hdl,
        "peer_bases": peer_bases,
        "prefix": prefix,
        "scalar_total": scalar_total,
        "dedup_sizes": dedup_sizes,
        "mark": mark,
        "assoc": assoc,
        "off_req_nodes": off_req_nodes,
        "off_req_index": off_req_index,
        "off_req_cursors": off_req_cursors,
        "off_reply_counts": off_reply_counts,
        "off_reply_nodes": off_reply_nodes,
        "off_reply_edges": off_reply_edges,
        "sz_req_cursors": sz_req_cursors,
        "sz_reply_counts": sz_reply_counts,
    }
    _resource_cache[key] = res
    return res


def _symm_slice(res, off: int, n: int) -> torch.Tensor:
    return res["symm_buf"].narrow(0, int(off), int(n))


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
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert seed_nodes.is_cuda, "seed_nodes must be CUDA"
    assert local_adj_row_ptr.is_cuda and local_adj_col.is_cuda and node_to_rank.is_cuda

    ext = _get_ext()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = seed_nodes.device

    seed = seed_nodes.to(device=device, dtype=torch.long).contiguous()
    colptr = local_adj_row_ptr.to(device=device, dtype=torch.long).contiguous()
    adj_col = local_adj_col.to(device=device, dtype=torch.long).contiguous()
    owner = node_to_rank.to(device=device, dtype=torch.long).contiguous()

    num_global_nodes = int(owner.numel())
    max_stride = _max_stride_from_fanouts(fanouts, num_global_nodes)
    res = _get_resources(
        group=group,
        device=device,
        world_size=world_size,
        num_global_nodes=num_global_nodes,
        stride=max_stride,
    )

    req_cap = res["req_cap"]
    stride = res["stride"]
    hdl = res["hdl"]

    src = seed.clone()
    node = seed.clone()

    node_with_dupl_parts = []
    dst_with_dupl_parts = []
    edge_parts = []

    req_cursors = _symm_slice(res, res["off_req_cursors"], world_size)
    reply_counts_full = _symm_slice(res, res["off_reply_counts"], req_cap)
    reply_nodes_full = _symm_slice(res, res["off_reply_nodes"], req_cap * stride)
    reply_edges_full = _symm_slice(res, res["off_reply_edges"], req_cap * stride)

    for hop, fanout in enumerate(fanouts):
        n_src = int(src.numel())
        if n_src == 0:
            break
        if n_src > req_cap:
            # Capacity is intentionally conservative; if exceeded, truncate would be
            # incorrect, so fail loudly.
            raise RuntimeError("frontier exceeds symmetric request capacity")

        # Clear local inbox cursors and local reply counts before peers write.
        ext.fill_i64(req_cursors, world_size, 0)
        ext.fill_i64(reply_counts_full, n_src, 0)
        hdl.barrier(channel=(2 * hop) % 16)

        ext.route_requests(
            src,
            owner,
            res["peer_bases"],
            n_src,
            rank,
            world_size,
            req_cap,
            res["off_req_nodes"],
            res["off_req_index"],
            res["off_req_cursors"],
        )

        # All ranks have deposited requests into owner inboxes.
        hdl.barrier(channel=(2 * hop + 1) % 16)

        seed64 = (
            (int(time.time_ns()) & 0xFFFFFFFFFFFF)
            ^ (rank << 48)
            ^ (hop * 0x9E3779B97F4A7C15)
        ) & 0xFFFFFFFFFFFFFFFF

        ext.sample_and_reply(
            res["symm_buf"],
            res["peer_bases"],
            colptr,
            adj_col,
            int(fanout),
            bool(replace),
            rank,
            world_size,
            req_cap,
            stride,
            res["off_req_nodes"],
            res["off_req_index"],
            res["off_req_cursors"],
            res["off_reply_counts"],
            res["off_reply_nodes"],
            res["off_reply_edges"],
            int(seed64),
        )

        # Replies are now in requester-rank symmetric buffers.
        hdl.barrier(channel=(2 * hop + 2) % 16)

        counts = reply_counts_full.narrow(0, 0, n_src)
        prefix = res["prefix"]
        total_scalar = res["scalar_total"]
        ext.prefix_counts(counts, prefix, total_scalar, n_src)
        total = int(total_scalar.item())

        if total == 0:
            break

        out_node = torch.empty((total,), device=device, dtype=torch.long)
        out_edge = torch.empty((total,), device=device, dtype=torch.long)
        out_dst = torch.empty((total,), device=device, dtype=torch.long)

        ext.flatten_replies(
            counts,
            prefix,
            reply_nodes_full,
            reply_edges_full,
            src,
            out_node,
            out_edge,
            out_dst,
            n_src,
            stride,
        )

        node_with_dupl_parts.append(out_node)
        dst_with_dupl_parts.append(out_dst)
        edge_parts.append(out_edge)

        # PyG remove_duplicates equivalent for homogeneous non-disjoint mode:
        # preserve first occurrence order in cat([node, out_node]).
        ext.fill_i32(res["mark"], num_global_nodes, 0)

        new_node_buf = torch.empty(
            (int(node.numel()) + int(out_node.numel()),),
            device=device,
            dtype=torch.long,
        )
        new_src_buf = torch.empty_like(out_node)

        ext.dedup_append(
            node,
            out_node,
            res["mark"],
            num_global_nodes,
            new_node_buf,
            new_src_buf,
            res["dedup_sizes"],
        )

        sizes_cpu = res["dedup_sizes"].cpu()
        new_node_n = int(sizes_cpu[0].item())
        new_src_n = int(sizes_cpu[1].item())

        node = new_node_buf.narrow(0, 0, new_node_n)
        src = new_src_buf.narrow(0, 0, new_src_n)

    if node_with_dupl_parts:
        node_dupl = torch.cat(node_with_dupl_parts)
        dst_dupl = torch.cat(dst_with_dupl_parts)
        edge = torch.cat(edge_parts)
    else:
        node_dupl = seed.new_empty((0,))
        dst_dupl = seed.new_empty((0,))
        edge = seed.new_empty((0,))

    if node_dupl.numel() == 0:
        row = seed.new_empty((0,))
        col = seed.new_empty((0,))
        return node, row, col, edge

    ext.fill_i64(res["assoc"], num_global_nodes, -1)
    ext.set_assoc(node, res["assoc"])

    row = torch.empty_like(node_dupl)
    col = torch.empty_like(dst_dupl)
    ext.relabel(node_dupl, dst_dupl, res["assoc"], row, col)

    return node, row, col, edge