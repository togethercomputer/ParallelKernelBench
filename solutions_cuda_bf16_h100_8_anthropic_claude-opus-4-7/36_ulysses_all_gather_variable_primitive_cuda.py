"""
Ulysses variable-size all_gather using symmetric memory + custom CUDA copy.

Strategy:
- Phase 1: gather sizes via symm_mem int64 buffer + barrier (device-side).
- Phase 2: each rank stages its tensor into a symmetric buffer (max-size slot).
  A single CUDA kernel reads all peers' slots via UVA peer pointers and writes
  directly into the concatenated output at the proper offset along gather_dim.
- Avoids torch.cat and per-peer launches; one fused kernel performs the gather.
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
#include <cstdint>

// Generic byte-wise copy from peer slot into the right slice of out.
// Each peer's tensor occupies a contiguous block of `peer_bytes[r]` bytes,
// laid out as [outer, inner_r] where inner_r = inner_per_unit_r (varies by rank).
// The output has shape [outer, total_inner], where inner offset for rank r
// is inner_offsets[r] (in elements of the inner dim contributed by that rank,
// summed in bytes). We pass byte offsets directly.

extern "C" __global__ void gather_concat_kernel(
    const uint64_t* __restrict__ peer_ptrs,    // [world_size] device pointers (bytes)
    const int64_t* __restrict__ peer_inner_bytes, // [world_size] inner-row bytes per peer
    const int64_t* __restrict__ inner_byte_offsets, // [world_size] starting byte offset within out row
    int64_t outer,
    int64_t out_row_bytes,
    int world_size,
    uint8_t* __restrict__ out
) {
    // Each block handles one (rank, outer_idx) slab of bytes.
    // We tile outer*world_size onto blockIdx.y, and bytes onto blockIdx.x.
    int rank_id = blockIdx.z;
    if (rank_id >= world_size) return;

    int64_t inner_bytes = peer_inner_bytes[rank_id];
    if (inner_bytes <= 0) return;

    int64_t out_off = inner_byte_offsets[rank_id];
    const uint8_t* src_base = reinterpret_cast<const uint8_t*>(peer_ptrs[rank_id]);

    int64_t o = blockIdx.y;
    if (o >= outer) return;

    int64_t byte_idx_start = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const uint8_t* src_row = src_base + o * inner_bytes;
    uint8_t* dst_row = out + o * out_row_bytes + out_off;

    // Copy as 16-byte vectors when aligned
    if ((((uintptr_t)src_row | (uintptr_t)dst_row | (uintptr_t)inner_bytes) & 15ULL) == 0ULL) {
        int64_t n16 = inner_bytes >> 4;
        const float4* s = reinterpret_cast<const float4*>(src_row);
        float4* d = reinterpret_cast<float4*>(dst_row);
        for (int64_t i = byte_idx_start; i < n16; i += stride) {
            d[i] = s[i];
        }
    } else if ((((uintptr_t)src_row | (uintptr_t)dst_row | (uintptr_t)inner_bytes) & 7ULL) == 0ULL) {
        int64_t n8 = inner_bytes >> 3;
        const uint64_t* s = reinterpret_cast<const uint64_t*>(src_row);
        uint64_t* d = reinterpret_cast<uint64_t*>(dst_row);
        for (int64_t i = byte_idx_start; i < n8; i += stride) {
            d[i] = s[i];
        }
    } else {
        for (int64_t i = byte_idx_start; i < inner_bytes; i += stride) {
            dst_row[i] = src_row[i];
        }
    }
}

void launch_gather_concat(
    torch::Tensor peer_ptrs,        // int64 [W]
    torch::Tensor peer_inner_bytes, // int64 [W]
    torch::Tensor inner_byte_offsets, // int64 [W]
    int64_t outer,
    int64_t out_row_bytes,
    int64_t world_size,
    torch::Tensor out
) {
    TORCH_CHECK(peer_ptrs.is_cuda() && peer_ptrs.dtype() == torch::kInt64);
    TORCH_CHECK(out.is_cuda());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    // Choose blocks.x based on max inner bytes / 16 to saturate
    int64_t max_inner = 0;
    {
        auto pib_cpu = peer_inner_bytes.cpu();
        auto acc = pib_cpu.accessor<int64_t,1>();
        for (int i = 0; i < (int)world_size; ++i) max_inner = std::max(max_inner, acc[i]);
    }
    int64_t units = (max_inner + 15) / 16;
    int blocks_x = (int)std::min<int64_t>((units + threads - 1) / threads, 64);
    if (blocks_x < 1) blocks_x = 1;

    dim3 grid(blocks_x, (unsigned int)outer, (unsigned int)world_size);
    dim3 block(threads);

    gather_concat_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>()),
        peer_inner_bytes.data_ptr<int64_t>(),
        inner_byte_offsets.data_ptr<int64_t>(),
        outer,
        out_row_bytes,
        (int)world_size,
        reinterpret_cast<uint8_t*>(out.data_ptr())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_concat", &launch_gather_concat, "Gather concat from peer symm buffers");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_var_gather_ext", CUDA_SRC)
    return _ext


_size_buf_cache = {}  # (ndim, world_size, device) -> (buf, hdl)
_data_buf_cache = {}  # (nbytes_cap, world_size, device) -> (buf, hdl, ptrs_tensor)


def _get_size_buf(ndim, world_size, device):
    key = (ndim, world_size, device)
    if key not in _size_buf_cache:
        # symmetric buffer holding ndim int64 per rank slot, but symm_mem is per-rank;
        # each rank writes its own ndim, peers read from peer pointers.
        buf = symm_mem.empty(ndim, device=device, dtype=torch.int64)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
        _size_buf_cache[key] = (buf, hdl, ptrs)
    return _size_buf_cache[key]


def _get_data_buf(nbytes_cap, world_size, device):
    key = (nbytes_cap, world_size, device)
    if key not in _data_buf_cache:
        buf = symm_mem.empty(nbytes_cap, device=device, dtype=torch.uint8)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
        _data_buf_cache[key] = (buf, hdl, ptrs)
    return _data_buf_cache[key]


@torch.no_grad()
def solution(
    x: torch.Tensor,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    device = x.device
    dtype = x.dtype
    x = x.contiguous()
    ndim = x.dim()
    rank = dist.get_rank(group)

    _get_ext()

    # ---- Phase 1: exchange shapes via symm_mem ----
    size_buf, size_hdl, size_ptrs = _get_size_buf(ndim, world_size, device)
    # Write our shape
    my_shape = torch.tensor(list(x.size()), dtype=torch.int64, device=device)
    size_buf.copy_(my_shape)
    size_hdl.barrier(channel=0)

    # Read all peer shapes from peer pointers via a small gather using cudaMemcpy
    # Easier: each rank reads via direct pointer load. We can do this with a tiny CUDA op,
    # but simpler — copy from each peer pointer using torch from_blob is not safe. Use
    # cudaMemcpyAsync via torch.cuda APIs: build a [W, ndim] tensor and memcpy each row.
    all_shapes = torch.empty((world_size, ndim), dtype=torch.int64, device=device)
    stream = torch.cuda.current_stream(device).cuda_stream
    import ctypes
    cudart = torch.cuda.cudart()
    elem_bytes = ndim * 8
    for r in range(world_size):
        src_ptr = int(size_hdl.buffer_ptrs[r])
        dst_ptr = all_shapes[r].data_ptr()
        # cudaMemcpyAsync DeviceToDevice = 3
        cudart.cudaMemcpyAsync(dst_ptr, src_ptr, elem_bytes, 3, stream)

    # Need shapes on CPU to allocate output and compute offsets
    shapes_cpu = all_shapes.cpu()  # syncs
    shapes_list = [tuple(shapes_cpu[r].tolist()) for r in range(world_size)]

    # Validate: all dims except gather_dim must match
    out_shape = list(shapes_list[0])
    for r in range(1, world_size):
        for d in range(ndim):
            if d == gather_dim:
                continue
            assert shapes_list[r][d] == out_shape[d], "non-gather dims mismatch"
    out_shape[gather_dim] = sum(shapes_list[r][gather_dim] for r in range(world_size))
    out_shape = tuple(out_shape)

    # ---- Phase 2: stage data and gather ----
    elem_size = x.element_size()

    # Compute outer = prod(shape[:gather_dim]); each peer's inner bytes = prod(shape[gather_dim:]) * elem_size
    def _outer(shape):
        o = 1
        for d in range(gather_dim):
            o *= shape[d]
        return o
    def _inner(shape):
        i = 1
        for d in range(gather_dim, ndim):
            i *= shape[d]
        return i

    outer = _outer(out_shape)
    # All ranks must agree on outer (non-gather dims match), so outer is consistent.

    peer_inner_bytes = [_inner(shapes_list[r]) * elem_size for r in range(world_size)]
    peer_total_bytes = [outer * peer_inner_bytes[r] for r in range(world_size)]
    max_bytes = max(peer_total_bytes)

    # Symmetric data buffer: use a capacity that fits any peer's tensor.
    # Round up to reduce re-allocations.
    cap = 1
    while cap < max_bytes:
        cap *= 2
    cap = max(cap, 1024)

    data_buf, data_hdl, data_ptrs = _get_data_buf(cap, world_size, device)

    # Copy our x bytes into symmetric buffer
    my_bytes = peer_total_bytes[rank]
    if my_bytes > 0:
        # view x as bytes
        x_bytes = x.view(torch.uint8).reshape(-1)
        data_buf[:my_bytes].copy_(x_bytes[:my_bytes])

    data_hdl.barrier(channel=1)

    # Compute inner byte offsets in the output row
    inner_byte_offsets = [0] * world_size
    acc = 0
    for r in range(world_size):
        inner_byte_offsets[r] = acc
        acc += peer_inner_bytes[r]
    out_row_bytes = acc

    # Build device tensors for kernel args
    peer_inner_bytes_t = torch.tensor(peer_inner_bytes, dtype=torch.int64, device=device)
    inner_byte_offsets_t = torch.tensor(inner_byte_offsets, dtype=torch.int64, device=device)

    out = torch.empty(out_shape, dtype=dtype, device=device)

    _get_ext().launch_gather_concat(
        data_ptrs,
        peer_inner_bytes_t,
        inner_byte_offsets_t,
        int(outer),
        int(out_row_bytes),
        int(world_size),
        out.view(torch.uint8).reshape(-1),
    )

    data_hdl.barrier(channel=2)
    return out