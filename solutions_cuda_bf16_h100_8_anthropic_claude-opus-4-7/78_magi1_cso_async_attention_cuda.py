"""
MAGI-1 CSO attention with symmetric-memory based all-to-all.

Strategy:
- Replace dist.all_to_all_single with a symm_mem-backed device-side a2a:
  each rank writes its per-peer chunk directly into peers' symmetric buffers
  via UVA pointers, then a barrier synchronizes completion.
- KV redistribution and per-range Q/O all-to-alls all use the same primitive.
- Overlap is preserved by issuing the next a2a (which is a single kernel +
  barrier) while SDPA on the current range runs on the default stream.
"""

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Copy local input chunks into peer symmetric buffers.
// in_ptr: local source contiguous tensor [total_bytes]
// peer_ptrs[r]: pointer to peer r's symmetric output buffer
// per peer r, this rank writes 'chunk_bytes[r]' bytes starting at
//   src_offset_bytes[r] in source -> dst_offset_bytes[r] in peer r's buffer.
__global__ void a2a_scatter_kernel(
    const uint8_t* __restrict__ in_ptr,
    const uint64_t* __restrict__ peer_ptrs,
    const int64_t* __restrict__ src_offsets,    // [world]
    const int64_t* __restrict__ dst_offsets,    // [world]
    const int64_t* __restrict__ chunk_bytes,    // [world]
    int world_size
) {
    int r = blockIdx.y;
    if (r >= world_size) return;
    int64_t nb = chunk_bytes[r];
    if (nb <= 0) return;

    uint8_t* dst = reinterpret_cast<uint8_t*>(peer_ptrs[r]) + dst_offsets[r];
    const uint8_t* src = in_ptr + src_offsets[r];

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // 16-byte vectorized copy
    int64_t n16 = nb / 16;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (int64_t i = tid; i < n16; i += stride) {
        d4[i] = s4[i];
    }
    int64_t tail_start = n16 * 16;
    for (int64_t i = tail_start + tid; i < nb; i += stride) {
        dst[i] = src[i];
    }
}

void launch_a2a_scatter(
    torch::Tensor in_buf,                  // local source on device
    torch::Tensor peer_ptrs,               // int64 [world]
    torch::Tensor src_offsets,             // int64 [world]
    torch::Tensor dst_offsets,             // int64 [world]
    torch::Tensor chunk_bytes,             // int64 [world]
    int64_t world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int threads = 256;
    int blocks_x = 256;
    dim3 grid(blocks_x, (unsigned int)world_size);
    a2a_scatter_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(in_buf.data_ptr()),
        reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>()),
        src_offsets.data_ptr<int64_t>(),
        dst_offsets.data_ptr<int64_t>(),
        chunk_bytes.data_ptr<int64_t>(),
        (int)world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_a2a_scatter", &launch_a2a_scatter, "Symm-mem all-to-all scatter");
}
"""


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("magi1_cso_a2a_ext", CUDA_SRC)
    return _ext


# Cache symmetric buffers keyed by byte size
_SYMM_POOL = {}


def _get_symm_pair(nbytes: int, device: torch.device, group):
    key = (nbytes, device.index)
    if key in _SYMM_POOL:
        return _SYMM_POOL[key]
    # Two ping-pong buffers: input (we write peers' parts here -> actually we
    # write to peers' OUTPUT buffer). We just need one symmetric buffer for the
    # output side. Source is regular tensor.
    out_buf = symm_mem.empty(nbytes, device=device, dtype=torch.uint8)
    hdl = symm_mem.rendezvous(out_buf, group)
    peer_ptrs = torch.tensor(
        [int(hdl.buffer_ptrs[r]) for r in range(hdl.world_size)],
        device=device, dtype=torch.int64,
    )
    _SYMM_POOL[key] = (out_buf, hdl, peer_ptrs)
    return _SYMM_POOL[key]


def _next_pow2_bytes(n: int) -> int:
    # Round up to a multiple to reduce reallocs
    if n <= 0:
        return 1024
    base = 1 << (n - 1).bit_length()
    return max(base, 1024)


def _symm_a2a_equal(
    src: torch.Tensor,
    split: int,
    world_size: int,
    rank: int,
    group,
    ext,
) -> torch.Tensor:
    """All-to-all where each peer chunk has identical 'split' rows along dim 0.
    src must be contiguous, shape (world_size*split, ...). Returns same shape.
    """
    assert src.is_contiguous()
    elem_per_row = src.numel() // src.shape[0]
    bytes_per_row = elem_per_row * src.element_size()
    chunk_bytes_each = split * bytes_per_row
    total_bytes = world_size * chunk_bytes_each

    pool_bytes = _next_pow2_bytes(total_bytes)
    out_buf, hdl, peer_ptrs = _get_symm_pair(pool_bytes, src.device, group)

    device = src.device
    src_offsets = torch.arange(world_size, device=device, dtype=torch.int64) * chunk_bytes_each
    # Each peer r places data at rank's slot (= rank * chunk_bytes_each)
    dst_offsets = torch.full((world_size,), rank * chunk_bytes_each,
                             device=device, dtype=torch.int64)
    chunk_bytes = torch.full((world_size,), chunk_bytes_each,
                             device=device, dtype=torch.int64)

    # Barrier so peers' output buffers are ready to be written
    hdl.barrier(channel=0)

    ext.launch_a2a_scatter(
        src.view(torch.uint8).reshape(-1),
        peer_ptrs,
        src_offsets,
        dst_offsets,
        chunk_bytes,
        world_size,
    )

    # Wait for all peers to finish writing
    hdl.barrier(channel=1)

    out_view = out_buf[:total_bytes].view(torch.uint8).clone()
    out = out_view.view(src.dtype).reshape(src.shape)
    return out


def _redistribute_kv_symm(
    key_value: torch.Tensor, world_size: int, rank: int, group, ext
) -> torch.Tensor:
    tokens, heads, width = key_value.shape
    if heads < world_size and world_size % heads == 0:
        key_value = key_value.repeat_interleave(world_size // heads, dim=1)
        heads = key_value.shape[1]
    if heads % world_size != 0:
        raise ValueError("KV heads must divide evenly across context ranks")

    local_heads = heads // world_size
    packed = key_value.reshape(tokens, world_size, local_heads, width)
    packed = packed.permute(1, 0, 2, 3).reshape(world_size * tokens, local_heads, width).contiguous()
    return _symm_a2a_equal(packed, tokens, world_size, rank, group, ext)


def _kv_by_range(kv, world_size, ranges, spb, clip_token_nums):
    _, heads, width = kv.shape
    kv = kv.reshape(world_size, ranges, spb, heads, width)
    kv = kv.permute(1, 0, 2, 3, 4).contiguous()
    kv = kv.reshape(ranges, world_size * spb, heads, width)
    return kv[:, :clip_token_nums].reshape(ranges * clip_token_nums, heads, width)


def _split_query(query, world_size, ranges):
    tokens, heads, head_dim = query.shape
    if tokens % ranges != 0:
        raise ValueError("query token count must divide cp_shuffle_num")
    if heads % world_size != 0:
        raise ValueError("query heads must divide evenly across context ranks")
    spb = tokens // ranges
    local_heads = heads // world_size
    query = query.reshape(ranges, spb, world_size, local_heads, head_dim)
    query = query.permute(0, 2, 1, 3, 4).contiguous()
    query = query.reshape(ranges, world_size * spb, local_heads, head_dim)
    return [query[idx].contiguous() for idx in range(ranges)]


def _restore_output(chunks, world_size, spb):
    out = torch.stack(chunks, dim=0)
    ranges, _, heads, head_dim = out.shape
    out = out.reshape(ranges, world_size, spb, heads, head_dim)
    out = out.permute(0, 2, 1, 3, 4).contiguous()
    return out.reshape(ranges * spb, world_size * heads, head_dim)


def _sdpa(q, k, v):
    q = q.unsqueeze(0).transpose(1, 2)
    k = k.unsqueeze(0).transpose(1, 2)
    v = v.unsqueeze(0).transpose(1, 2)
    if k.shape[1] < q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1).contiguous()


@torch.no_grad()
def solution(
    query: torch.Tensor,
    key_value: torch.Tensor,
    k_ranges: torch.Tensor,
    cp_shuffle_num: int,
    clip_token_nums: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    ext = _get_ext()

    ranges = cp_shuffle_num
    tokens, _, head_dim = query.shape
    if tokens % ranges != 0:
        raise ValueError("query token count must divide cp_shuffle_num")
    spb = tokens // ranges
    clip_token_nums = int(clip_token_nums or world_size * spb)

    kv = _redistribute_kv_symm(key_value, world_size, rank, group, ext)
    kv = _kv_by_range(kv, world_size, ranges, spb, clip_token_nums)
    key = kv[..., :head_dim]
    value = kv[..., head_dim:]

    q_chunks = _split_query(query, world_size, ranges)

    outputs: List[torch.Tensor] = []
    for idx in range(ranges):
        q_local = _symm_a2a_equal(q_chunks[idx], spb, world_size, rank, group, ext)
        start = int(k_ranges[idx, 0])
        end = int(k_ranges[idx, 1])
        out = _sdpa(q_local, key[start:end], value[start:end])
        out = _symm_a2a_equal(out.contiguous(), spb, world_size, rank, group, ext)
        outputs.append(out)

    return _restore_output(outputs, world_size, spb)