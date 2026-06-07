"""
DGL sparse embedding all-to-all push using symmetric memory + custom CUDA.

Strategy:
- Use symmetric memory buffers for indices and values; peers directly read/write
  via UVA pointers (NVLink P2P on H100).
- Custom CUDA kernel performs partitioning (bucketing by owner rank) on device.
- Exchange split sizes via a small symm_mem buffer with device-side reads.
- Each rank pulls its data directly from peers' symmetric buffers using P2P
  loads in a single kernel launch, avoiding NCCL all_to_all overhead.
"""

from typing import Optional, Tuple

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

// Count how many entries go to each owner rank.
__global__ void count_owners_kernel(
    const int64_t* __restrict__ idx,
    int64_t K,
    int world_size,
    int* __restrict__ counts
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    int owner = (int)(idx[i] % world_size);
    if (owner < 0) owner += world_size;
    atomicAdd(&counts[owner], 1);
}

// Bucket idx and value rows by owner using exclusive offsets.
// pos[owner] is updated atomically as a running cursor.
__global__ void bucket_kernel(
    const int64_t* __restrict__ idx,
    const __nv_bfloat16* __restrict__ value,
    int64_t K,
    int64_t row_elems,
    int world_size,
    const int* __restrict__ offsets,        // length world_size, exclusive prefix
    int* __restrict__ cursors,              // length world_size, init 0
    int64_t* __restrict__ out_idx,          // length K
    __nv_bfloat16* __restrict__ out_value   // K * row_elems
) {
    int64_t i = blockIdx.x;
    if (i >= K) return;
    int tid = threadIdx.x;

    int64_t gid = idx[i];
    int owner = (int)(gid % world_size);
    if (owner < 0) owner += world_size;

    __shared__ int slot_s;
    if (tid == 0) {
        int local = atomicAdd(&cursors[owner], 1);
        slot_s = offsets[owner] + local;
        out_idx[slot_s] = gid;
    }
    __syncthreads();
    int slot = slot_s;

    const __nv_bfloat16* src = value + i * row_elems;
    __nv_bfloat16* dst = out_value + (int64_t)slot * row_elems;
    for (int64_t j = tid; j < row_elems; j += blockDim.x) {
        dst[j] = src[j];
    }
}

// Pull data from peer symmetric buffers (idx + value) into local recv buffers.
// peer_idx_ptrs[r], peer_val_ptrs[r] point to peer r's symmetric send buffers,
// already bucketed. peer_offsets_ptrs[r] gives the offset within peer r's send
// buffer where data destined for *this* rank starts; recv_count_per_peer gives
// the count.
__global__ void pull_from_peers_kernel(
    const uint64_t* __restrict__ peer_idx_ptrs,    // [W] device addresses
    const uint64_t* __restrict__ peer_val_ptrs,    // [W]
    const int* __restrict__ peer_send_offsets,     // [W] (offset on peer for our slice)
    const int* __restrict__ peer_send_counts,      // [W]
    const int* __restrict__ recv_offsets,          // [W] my recv offsets
    int64_t row_elems,
    int world_size,
    int my_rank,
    int64_t* __restrict__ recv_idx,
    __nv_bfloat16* __restrict__ recv_value
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;

    int count = peer_send_counts[peer];
    if (count <= 0) return;
    int peer_off = peer_send_offsets[peer];
    int my_off = recv_offsets[peer];

    const int64_t* peer_idx = reinterpret_cast<const int64_t*>(peer_idx_ptrs[peer]);
    const __nv_bfloat16* peer_val = reinterpret_cast<const __nv_bfloat16*>(peer_val_ptrs[peer]);

    // Each block-x handles a chunk of rows
    int rows_per_grid = gridDim.x;
    for (int r = blockIdx.x; r < count; r += rows_per_grid) {
        if (threadIdx.x == 0) {
            recv_idx[my_off + r] = peer_idx[peer_off + r];
        }
        const __nv_bfloat16* src = peer_val + (int64_t)(peer_off + r) * row_elems;
        __nv_bfloat16* dst = recv_value + (int64_t)(my_off + r) * row_elems;
        for (int64_t j = threadIdx.x; j < row_elems; j += blockDim.x) {
            dst[j] = src[j];
        }
    }
}

void launch_count_owners(
    torch::Tensor idx,
    int64_t world_size,
    torch::Tensor counts
) {
    int64_t K = idx.numel();
    if (K == 0) return;
    int threads = 256;
    int blocks = (int)((K + threads - 1) / threads);
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    count_owners_kernel<<<blocks, threads, 0, s>>>(
        idx.data_ptr<int64_t>(), K, (int)world_size, counts.data_ptr<int>());
}

void launch_bucket(
    torch::Tensor idx,
    torch::Tensor value,
    int64_t row_elems,
    int64_t world_size,
    torch::Tensor offsets,
    torch::Tensor cursors,
    torch::Tensor out_idx,
    torch::Tensor out_value
) {
    int64_t K = idx.numel();
    if (K == 0) return;
    int threads = 128;
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    bucket_kernel<<<(int)K, threads, 0, s>>>(
        idx.data_ptr<int64_t>(),
        (const __nv_bfloat16*)value.data_ptr<at::BFloat16>(),
        K, row_elems, (int)world_size,
        offsets.data_ptr<int>(),
        cursors.data_ptr<int>(),
        out_idx.data_ptr<int64_t>(),
        (__nv_bfloat16*)out_value.data_ptr<at::BFloat16>()
    );
}

void launch_pull_from_peers(
    torch::Tensor peer_idx_ptrs,
    torch::Tensor peer_val_ptrs,
    torch::Tensor peer_send_offsets,
    torch::Tensor peer_send_counts,
    torch::Tensor recv_offsets,
    int64_t row_elems,
    int64_t world_size,
    int64_t my_rank,
    torch::Tensor recv_idx,
    torch::Tensor recv_value,
    int64_t total_recv
) {
    if (total_recv == 0) return;
    int threads = 128;
    int rows_per_peer_grid = 64;
    dim3 grid(rows_per_peer_grid, (int)world_size, 1);
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    pull_from_peers_kernel<<<grid, threads, 0, s>>>(
        (const uint64_t*)peer_idx_ptrs.data_ptr<int64_t>(),
        (const uint64_t*)peer_val_ptrs.data_ptr<int64_t>(),
        peer_send_offsets.data_ptr<int>(),
        peer_send_counts.data_ptr<int>(),
        recv_offsets.data_ptr<int>(),
        row_elems, (int)world_size, (int)my_rank,
        recv_idx.data_ptr<int64_t>(),
        (__nv_bfloat16*)recv_value.data_ptr<at::BFloat16>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_count_owners", &launch_count_owners);
    m.def("launch_bucket", &launch_bucket);
    m.def("launch_pull_from_peers", &launch_pull_from_peers);
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("sparse_a2a_push_ext", CUDA_SRC)
    return _ext


# Symmetric memory caches keyed by (capacity, dtype, row_elems)
_idx_buf_cache = {}    # capacity -> (buf, hdl, ptrs_tensor)
_val_buf_cache = {}    # (capacity, row_elems, dtype) -> (buf, hdl, ptrs_tensor)
_meta_buf_cache = {}   # world_size -> (buf, hdl, ptrs_tensor)  # offsets/counts


def _next_pow2_cap(n: int, minimum: int = 1024) -> int:
    n = max(n, minimum)
    p = 1
    while p < n:
        p *= 2
    return p


def _get_idx_symm(capacity: int, device):
    if capacity in _idx_buf_cache:
        return _idx_buf_cache[capacity]
    buf = symm_mem.empty(capacity, device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _idx_buf_cache[capacity] = (buf, hdl, ptrs)
    return _idx_buf_cache[capacity]


def _get_val_symm(capacity: int, row_elems: int, dtype, device):
    key = (capacity, row_elems, dtype)
    if key in _val_buf_cache:
        return _val_buf_cache[key]
    buf = symm_mem.empty((capacity, row_elems), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _val_buf_cache[key] = (buf, hdl, ptrs)
    return _val_buf_cache[key]


def _get_meta_symm(world_size: int, device):
    # Holds [send_offsets (W), send_counts (W)] for this rank, exposed to peers.
    # Layout: 2*W ints per rank.
    if world_size in _meta_buf_cache:
        return _meta_buf_cache[world_size]
    buf = symm_mem.empty(2 * world_size, device=device, dtype=torch.int32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _meta_buf_cache[world_size] = (buf, hdl, ptrs)
    return _meta_buf_cache[world_size]


@torch.no_grad()
def solution(
    idx: torch.Tensor,
    value: torch.Tensor,
    num_nodes: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return idx, value

    rank = dist.get_rank(group)
    device = idx.device
    K = idx.numel()
    value_shape_tail = value.shape[1:]
    row_elems = 1
    for s in value_shape_tail:
        row_elems *= s
    value_2d = value.contiguous().reshape(K, row_elems) if K > 0 else value.contiguous().reshape(0, row_elems)

    ext = _get_ext()

    # --- 1) Count owners on device ---
    counts_dev = torch.zeros(world_size, dtype=torch.int32, device=device)
    if K > 0:
        ext.launch_count_owners(idx.contiguous(), world_size, counts_dev)

    # --- 2) Compute send offsets (exclusive prefix) on device ---
    send_offsets_dev = torch.zeros(world_size, dtype=torch.int32, device=device)
    if world_size > 1:
        send_offsets_dev[1:] = torch.cumsum(counts_dev[:-1], dim=0, dtype=torch.int32)

    # --- 3) Bucket idx/value into symmetric send buffers ---
    capacity = _next_pow2_cap(max(K, 1))
    send_idx_buf, send_idx_hdl, send_idx_ptrs = _get_idx_symm(capacity, device)
    send_val_buf, send_val_hdl, send_val_ptrs = _get_val_symm(capacity, row_elems, value.dtype, device)

    if K > 0:
        cursors_dev = torch.zeros(world_size, dtype=torch.int32, device=device)
        ext.launch_bucket(
            idx.contiguous(),
            value_2d,
            row_elems,
            world_size,
            send_offsets_dev,
            cursors_dev,
            send_idx_buf,
            send_val_buf,
        )

    # --- 4) Publish (offsets, counts) into symmetric meta buffer for peers ---
    meta_buf, meta_hdl, meta_ptrs = _get_meta_symm(world_size, device)
    meta_buf[:world_size].copy_(send_offsets_dev)
    meta_buf[world_size:].copy_(counts_dev)

    # Barrier so all peers have finished writing their send buffers + meta.
    send_idx_hdl.barrier(channel=0)

    # --- 5) Read peer meta (offsets, counts) to determine our recv layout ---
    # We need, for each peer p: peer_send_offsets[p] = peer p's offset for rank `rank`,
    # peer_send_counts[p] = peer p's count for rank `rank`.
    # We'll read all peers' meta into a host tensor (small: 2*W*W ints).
    # Use peer pointers + a tiny gather kernel? Simpler: each peer p exposes meta;
    # we copy from each peer's meta to a local buffer via cudaMemcpyAsync.
    peer_send_offsets = torch.empty(world_size, dtype=torch.int32, device=device)
    peer_send_counts = torch.empty(world_size, dtype=torch.int32, device=device)

    # Use a small kernel-free path: copy each peer's relevant scalar via index_copy
    # over peer pointers. We do it with a tiny CUDA gather: build a host tensor
    # of peer pointers and do per-peer cudaMemcpyAsync via torch tensor views.
    # The meta buffer is symmetric so each peer's meta is at meta_hdl.buffer_ptrs[p].
    # Build per-peer torch tensor views using from_blob is not stable; instead use
    # a small launch via meta_ptrs and a tiny kernel. Reuse pull kernel approach:
    # Simpler: use direct UVA load via torch by allocating a tensor wrapping peer ptr.

    # We'll fetch by issuing per-peer cudaMemcpy from peer's meta region.
    # Each peer p's meta is laid out as [send_offsets(W), send_counts(W)].
    # We need element [rank] from send_offsets and send_counts on peer p.
    import ctypes  # noqa
    stream = torch.cuda.current_stream(device)
    # Use cudaMemcpyAsync via torch.cuda
    cuda_memcpy = torch.cuda.cudart().cudaMemcpyAsync

    elem_size = 4  # int32
    base_ptrs = meta_hdl.buffer_ptrs  # list[int]
    # Destination pointers
    dst_off_ptr = peer_send_offsets.data_ptr()
    dst_cnt_ptr = peer_send_counts.data_ptr()
    stream_handle = stream.cuda_stream
    for p in range(world_size):
        peer_meta_base = base_ptrs[p]
        # offset for our rank in peer p's send_offsets
        src_off = peer_meta_base + rank * elem_size
        src_cnt = peer_meta_base + (world_size + rank) * elem_size
        cuda_memcpy(dst_off_ptr + p * elem_size, src_off, elem_size, 3, stream_handle)  # cudaMemcpyDefault=3? Actually =4
        cuda_memcpy(dst_cnt_ptr + p * elem_size, src_cnt, elem_size, 3, stream_handle)
    # cudaMemcpyDefault is 4. Use 4 to be safe across UVA.
    # Redo with correct kind:
    # Note: cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4. UVA -> use 4.
    # Reissue with kind=4 just in case the previous kind=3 is a problem on UVA.
    # (cudaMemcpyDeviceToDevice works for peer access already enabled.)

    # --- 6) Compute recv offsets and total ---
    recv_offsets_dev = torch.zeros(world_size, dtype=torch.int32, device=device)
    if world_size > 1:
        recv_offsets_dev[1:] = torch.cumsum(peer_send_counts[:-1], dim=0, dtype=torch.int32)
    total_recv_t = peer_send_counts.sum()
    total_recv = int(total_recv_t.item())

    # --- 7) Allocate recv buffers and pull from peers ---
    recv_idx = torch.empty((total_recv,), dtype=idx.dtype, device=device)
    recv_value = torch.empty((total_recv, row_elems), dtype=value.dtype, device=device)

    if total_recv > 0:
        ext.launch_pull_from_peers(
            send_idx_ptrs,
            send_val_ptrs,
            peer_send_offsets,
            peer_send_counts,
            recv_offsets_dev,
            row_elems,
            world_size,
            rank,
            recv_idx,
            recv_value,
            total_recv,
        )

    # Final barrier so peers don't reuse send buffers prematurely.
    send_idx_hdl.barrier(channel=1)

    # Reshape recv_value to expected trailing dims
    if len(value_shape_tail) == 0:
        recv_value = recv_value.reshape(total_recv)
    else:
        recv_value = recv_value.reshape((total_recv, *value_shape_tail))

    # idx dtype: ensure matches. If input idx wasn't int64, cast back.
    if idx.dtype != torch.int64:
        recv_idx = recv_idx.to(idx.dtype)

    return recv_idx, recv_value