"""
Ulysses all_to_all_tensor via symmetric memory + custom CUDA kernel.

Strategy:
- Each rank writes its source chunks into a symmetric memory buffer (one slot per peer).
- After a device-side barrier, each rank reads its slot from every peer's symmetric
  buffer via UVA peer pointers and writes directly into the output tensor at the
  correct gather_dim offset, performing the necessary transpose/concat in one kernel.
- This replaces dist.all_to_all + torch.cat with a single fused device-side exchange.
"""

from typing import Optional

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

// Copy local input chunks (split along scatter_dim) into the symmetric buffer
// laid out as [world_size, chunk_numel] where slot r holds the chunk destined
// for peer r.
__global__ void pack_chunks_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ symm_buf,
    int64_t outer,        // product of dims before scatter_dim
    int64_t scatter_size, // size of scatter_dim (full)
    int64_t inner,        // product of dims after scatter_dim
    int64_t chunk_scatter, // scatter_size / world_size
    int world_size
) {
    // total elements
    int64_t total = outer * scatter_size * inner;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    int64_t chunk_numel = outer * chunk_scatter * inner;

    for (int64_t idx = tid; idx < total; idx += stride) {
        // decode idx into (o, s, i)
        int64_t i = idx % inner;
        int64_t s = (idx / inner) % scatter_size;
        int64_t o = idx / (inner * scatter_size);

        int rank_dst = (int)(s / chunk_scatter);
        int64_t s_local = s - (int64_t)rank_dst * chunk_scatter;

        // dest layout per slot: [outer, chunk_scatter, inner]
        int64_t dst_off = (int64_t)rank_dst * chunk_numel
                          + o * (chunk_scatter * inner)
                          + s_local * inner
                          + i;
        symm_buf[dst_off] = x[idx];
    }
}

// Read slot 'rank' (which contains data peer r intended for me) from each peer
// and write it into the output tensor at the correct position along gather_dim.
// Output shape conceptually:
//   [outer_g, world_size * chunk_gather, inner_g]
// where the gather dimension is split into world_size segments, each segment
// corresponding to data received from peer r.
//
// Each peer's slot was packed with shape [outer, chunk_scatter, inner]
// from peer r's perspective. We need to interpret that layout in terms of
// the output's (outer_g, chunk_gather, inner_g) coordinate system.
//
// Note: outer*chunk_scatter*inner == outer_g*chunk_gather*inner_g
// (same number of elements). We pass the source layout dims and do a
// reshape-aware copy: each element in the slot is at flat index 'k'.
// We map flat index k -> (og, cg, ig) for the output write position.

__global__ void unpack_from_peers_kernel(
    const uint64_t* __restrict__ peer_ptrs, // world_size pointers (uintptr to bf16 buffers)
    __nv_bfloat16* __restrict__ out,
    int64_t outer_g,         // product of out dims before gather_dim
    int64_t gather_size,     // out gather_dim full size = world_size * chunk_gather
    int64_t inner_g,         // product of out dims after gather_dim
    int64_t chunk_gather,    // chunk along gather dim per peer
    int64_t chunk_numel,     // outer * chunk_scatter * inner == outer_g*chunk_gather*inner_g
    int world_size,
    int my_rank
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;

    const __nv_bfloat16* src_base =
        reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[peer])
        + (int64_t)my_rank * chunk_numel;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t k = tid; k < chunk_numel; k += stride) {
        // Decode k into output coordinates (og, cg, ig)
        // Flat layout of the slot, when reshaped onto output's
        // (outer_g, chunk_gather, inner_g), is the same flat order
        // because outer*chunk_scatter*inner reshapes contiguously.
        int64_t ig = k % inner_g;
        int64_t cg = (k / inner_g) % chunk_gather;
        int64_t og = k / (inner_g * chunk_gather);

        int64_t g = (int64_t)peer * chunk_gather + cg;
        int64_t out_off = og * (gather_size * inner_g) + g * inner_g + ig;
        out[out_off] = src_base[k];
    }
}

void launch_pack(
    torch::Tensor x,
    torch::Tensor symm_buf,
    int64_t outer,
    int64_t scatter_size,
    int64_t inner,
    int64_t chunk_scatter,
    int64_t world_size
) {
    int64_t total = outer * scatter_size * inner;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_chunks_kernel<<<(int)blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)symm_buf.data_ptr<at::BFloat16>(),
        outer, scatter_size, inner, chunk_scatter, (int)world_size);
}

void launch_unpack(
    torch::Tensor peer_ptrs_t,
    torch::Tensor out,
    int64_t outer_g,
    int64_t gather_size,
    int64_t inner_g,
    int64_t chunk_gather,
    int64_t chunk_numel,
    int64_t world_size,
    int64_t my_rank
) {
    const uint64_t* d_ptrs = (const uint64_t*)peer_ptrs_t.data_ptr<int64_t>();
    int threads = 256;
    int64_t blocks_x = (chunk_numel + threads - 1) / threads;
    if (blocks_x > 32768) blocks_x = 32768;
    dim3 grid((unsigned)blocks_x, (unsigned)world_size, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    unpack_from_peers_kernel<<<grid, threads, 0, stream>>>(
        d_ptrs,
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        outer_g, gather_size, inner_g, chunk_gather, chunk_numel,
        (int)world_size, (int)my_rank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pack", &launch_pack, "pack chunks into symmetric buffer");
    m.def("launch_unpack", &launch_unpack, "unpack from peer symmetric buffers into output");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_a2a_symm_ext", CUDA_SRC)
    return _ext


# Cache: keyed by (numel, dtype, device, group_id) -> (symm_buf, hdl, peer_ptrs_tensor)
_buf_cache = {}


def _get_symm_buf(numel: int, dtype: torch.dtype, device: torch.device, group):
    key = (numel, dtype, device, id(group))
    if key in _buf_cache:
        return _buf_cache[key]
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    peer_ptrs = torch.tensor(
        [int(p) for p in hdl.buffer_ptrs], device=device, dtype=torch.int64
    )
    _buf_cache[key] = (buf, hdl, peer_ptrs)
    return _buf_cache[key]


def solution(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    x = x.contiguous()
    assert x.dtype == torch.bfloat16, "this kernel is specialized for bf16"

    ndim = x.dim()
    if scatter_dim < 0:
        scatter_dim += ndim
    if gather_dim < 0:
        gather_dim += ndim

    in_shape = list(x.shape)
    scatter_size = in_shape[scatter_dim]
    assert scatter_size % world_size == 0
    chunk_scatter = scatter_size // world_size

    outer = 1
    for d in range(scatter_dim):
        outer *= in_shape[d]
    inner = 1
    for d in range(scatter_dim + 1, ndim):
        inner *= in_shape[d]

    chunk_numel = outer * chunk_scatter * inner
    total_numel = chunk_numel * world_size  # == x.numel()

    # Output shape: same as input but scatter_dim shrinks by world_size, gather_dim grows by world_size
    out_shape = list(in_shape)
    out_shape[scatter_dim] = chunk_scatter
    out_shape[gather_dim] = out_shape[gather_dim] * world_size

    # Compute output outer/inner around gather_dim from out_shape
    outer_g = 1
    for d in range(gather_dim):
        outer_g *= out_shape[d]
    inner_g = 1
    for d in range(gather_dim + 1, ndim):
        inner_g *= out_shape[d]
    gather_size = out_shape[gather_dim]
    chunk_gather = gather_size // world_size

    device = x.device
    out = torch.empty(out_shape, dtype=x.dtype, device=device)

    ext = _get_ext()
    buf, hdl, peer_ptrs = _get_symm_buf(total_numel, x.dtype, device, group)

    # Pack into symmetric buffer
    ext.launch_pack(x, buf, outer, scatter_size, inner, chunk_scatter, world_size)

    # Device-side barrier: ensure all peers have completed their pack before
    # we read from them.
    hdl.barrier(channel=0)

    # Pull from each peer's slot for this rank, writing directly into out
    my_rank = dist.get_rank(group)
    ext.launch_unpack(
        peer_ptrs, out,
        outer_g, gather_size, inner_g, chunk_gather, chunk_numel,
        world_size, my_rank,
    )

    # Ensure no peer races ahead and overwrites their buffer before we finish reading
    hdl.barrier(channel=1)

    return out