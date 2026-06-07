"""
Ulysses sequence-parallel attention with custom CUDA all-to-all via symmetric memory.
Replaces dist.all_to_all_single with direct peer-to-peer copies through UVA pointers
on symm_mem buffers. Forward-only (no_grad) hot path.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Each rank has a symmetric input buffer of size [world_size, chunk_bytes].
// rank r writes its chunk for peer p at input[p][...]'s location ON peer p.
// After barrier, each rank's "output" buffer (its own input) contains
// concatenated chunks from all peers indexed by source rank.
//
// We implement a fused kernel: copy local chunk to all peers' slots.
// Each block handles one peer; threads stream the chunk via vectorized loads.

__global__ void a2a_push_kernel(
    const uint8_t* __restrict__ src,      // local source buffer, [world_size * chunk_bytes]
    const uint64_t* __restrict__ peer_dst_ptrs, // ptrs to each peer's destination buffer base
    int world_size,
    int rank,
    int64_t chunk_bytes
) {
    int peer = blockIdx.x;
    if (peer >= world_size) return;

    // Source: src + peer * chunk_bytes
    // Destination: peer_dst_ptrs[peer] + rank * chunk_bytes (slot indexed by source rank)
    const uint8_t* s = src + (int64_t)peer * chunk_bytes;
    uint8_t* d = reinterpret_cast<uint8_t*>(peer_dst_ptrs[peer]) + (int64_t)rank * chunk_bytes;

    // Vectorized 16-byte copies
    int64_t n_vec = chunk_bytes / 16;
    const int4* sv = reinterpret_cast<const int4*>(s);
    int4* dv = reinterpret_cast<int4*>(d);

    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.y; // we'll use gridDim.y for parallelism within a peer
    int block_y = blockIdx.y;
    int gtid = block_y * blockDim.x + tid;

    for (int64_t i = gtid; i < n_vec; i += stride) {
        dv[i] = sv[i];
    }

    // Tail bytes
    int64_t tail_start = n_vec * 16;
    for (int64_t i = tail_start + gtid; i < chunk_bytes; i += stride) {
        d[i] = s[i];
    }
}

void launch_a2a_push(
    torch::Tensor src,
    torch::Tensor peer_dst_ptrs,
    int64_t world_size,
    int64_t rank,
    int64_t chunk_bytes
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid((unsigned)world_size, 16, 1);
    dim3 block(256, 1, 1);
    a2a_push_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src.data_ptr()),
        reinterpret_cast<const uint64_t*>(peer_dst_ptrs.data_ptr<int64_t>()),
        (int)world_size, (int)rank, chunk_bytes
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_a2a_push", &launch_a2a_push, "all-to-all push via symm_mem");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_a2a_ext", CUDA_SRC)
    return _ext


_symm_cache = {}

def _get_symm(nbytes: int, device: torch.device):
    """Get a pair (src_buf, dst_buf, peer_dst_ptrs_tensor, hdl_dst) for a given size."""
    key = (nbytes, device.index)
    if key in _symm_cache:
        return _symm_cache[key]

    # Allocate symmetric buffers as bytes
    src_buf = symm_mem.empty(nbytes, device=device, dtype=torch.uint8)
    dst_buf = symm_mem.empty(nbytes, device=device, dtype=torch.uint8)
    hdl_src = symm_mem.rendezvous(src_buf, dist.group.WORLD)
    hdl_dst = symm_mem.rendezvous(dst_buf, dist.group.WORLD)

    peer_dst_ptrs = torch.tensor(
        [int(p) for p in hdl_dst.buffer_ptrs], device=device, dtype=torch.int64
    )

    entry = (src_buf, dst_buf, peer_dst_ptrs, hdl_src, hdl_dst)
    _symm_cache[key] = entry
    return entry


def _symm_all_to_all_bytes(input_flat_bytes: torch.Tensor, world_size: int, rank: int):
    """
    input_flat_bytes: contiguous uint8 tensor of shape [world_size * chunk_bytes].
    Returns output uint8 tensor of same shape, where output[r*chunk_bytes:(r+1)*chunk_bytes]
    came from peer r's input[rank*chunk_bytes:(rank+1)*chunk_bytes].
    """
    nbytes = input_flat_bytes.numel()
    chunk_bytes = nbytes // world_size
    src_buf, dst_buf, peer_dst_ptrs, hdl_src, hdl_dst = _get_symm(nbytes, input_flat_bytes.device)

    # Copy input into symmetric src buffer
    src_buf.copy_(input_flat_bytes)

    # Sync so all peers have populated their src buffers (we read no peer src; we push to peer dst)
    # Actually we push from local src to peer dst, so we need src ready locally and dst ready on peers.
    # Use a barrier on dst handle to ensure all peers are at the same point.
    hdl_dst.barrier(channel=0)

    _get_ext().launch_a2a_push(
        src_buf, peer_dst_ptrs, world_size, rank, chunk_bytes
    )

    # Wait for all peers to finish writing into our dst
    hdl_dst.barrier(channel=1)

    return dst_buf


def _all_to_all_dim(x: torch.Tensor, scatter_dim: int, gather_dim: int, world_size: int, rank: int) -> torch.Tensor:
    """
    Equivalent to dist.all_to_all on tensor split into world_size chunks along scatter_dim,
    then concatenated along gather_dim.
    """
    assert scatter_dim in (1, 2) and gather_dim in (1, 2)
    # Split scatter_dim into world_size chunks; rearrange so chunk index is leading.
    shape = list(x.shape)
    assert shape[scatter_dim] % world_size == 0
    chunk = shape[scatter_dim] // world_size

    # Bring scatter_dim chunks into leading position in a contiguous layout matching all_to_all semantics:
    # For all_to_all, input_list[r] is x.split(scatter_dim)[r]; output is cat along gather_dim.
    # We need a layout where leading dim is "rank" so we can do flat byte all-to-all.
    # Construct: x_split shape = [..., world_size, chunk, ...] then move world_size to dim 0.
    new_shape = shape[:scatter_dim] + [world_size, chunk] + shape[scatter_dim+1:]
    x_r = x.reshape(new_shape)
    # Move world_size axis (at scatter_dim) to dim 0
    perm = [scatter_dim] + [i for i in range(len(new_shape)) if i != scatter_dim]
    x_perm = x_r.permute(perm).contiguous()
    # Now x_perm shape: [world_size, ...]
    flat = x_perm.view(torch.uint8).reshape(-1)

    out_bytes = _symm_all_to_all_bytes(flat, world_size, rank)

    out_perm = out_bytes.view(x_perm.dtype).reshape(x_perm.shape)
    # out_perm[r] is the chunk that came from peer r (was input_list[rank] on peer r,
    # i.e., x.split(scatter_dim)[rank] on peer r). Concatenate along gather_dim.

    # Move dim 0 (which is "source rank") to gather_dim position to concat.
    # Current shape: [world_size, *other_dims_in_order_of_perm]
    # We need to inverse permute back to original layout but with world_size still as a chunk.
    # Strategy: think of out_perm as having the same logical meaning as x_r but where the
    # world_size dim now indexes source rank. Then we want to concatenate along gather_dim.
    inv_perm = [0] * len(new_shape)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    out_r = out_perm.permute(inv_perm).contiguous()
    # out_r shape == new_shape but world_size axis is at scatter_dim, indexing source rank.
    # Reshape merging world_size with gather_dim.
    # First, move world_size from scatter_dim to be adjacent to gather_dim.
    # out_r shape has world_size at scatter_dim. We want to merge it into gather_dim.
    # Move scatter_dim to just before gather_dim (or after, depending).
    if gather_dim > scatter_dim:
        # After moving world_size out of scatter_dim, gather_dim shifts down by 1.
        # Move axis scatter_dim to position gather_dim - 1 (so it's just before original gather_dim's data)
        # Actually we want to merge world_size into gather_dim: result shape's gather_dim becomes world_size * orig_gather_dim_size
        # So move world_size axis to position gather_dim, then merge with the original gather data which is now at gather_dim+1... wait.
        # Let's think simpler: out_r has shape new_shape = [..., world_size_at_scatter_dim, chunk_at_scatter_dim+1, ...]
        # No: new_shape splits scatter_dim into (world_size, chunk) at positions scatter_dim and scatter_dim+1.
        # gather_dim in original x is some other axis. In new_shape, if gather_dim < scatter_dim, it's at gather_dim.
        # If gather_dim > scatter_dim, it's at gather_dim + 1 (because we inserted world_size).
        gd_in_new = gather_dim + 1
    else:
        gd_in_new = gather_dim
    # We want to move axis at scatter_dim (the world_size axis) next to gd_in_new and merge.
    # Move it to position gd_in_new (so it sits just before gather data), then merge.
    axes = list(range(len(new_shape)))
    axes.remove(scatter_dim)
    # insert scatter_dim axis at position gd_in_new (adjusted because we removed scatter_dim)
    insert_pos = gd_in_new if gd_in_new < scatter_dim else gd_in_new - 1
    axes.insert(insert_pos, scatter_dim)
    out_moved = out_r.permute(axes).contiguous()
    # Now world_size axis is at insert_pos, and gather_dim's chunk is at insert_pos+1. Merge.
    final_shape = list(out_moved.shape)
    merged = final_shape[insert_pos] * final_shape[insert_pos + 1]
    final_shape = final_shape[:insert_pos] + [merged] + final_shape[insert_pos + 2:]
    return out_moved.reshape(final_shape)


def _local_attention(q, k, v, scale, causal=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal and q.size(1) > 1:
        S = scores.size(-1)
        causal_mask = torch.triu(
            torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    num_heads: int = 8,
    causal: bool = False,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        B, S_local, H = hidden_states.shape
        head_dim = H // num_heads
        qkv = F.linear(hidden_states, w_qkv)
        qkv = qkv.view(B, S_local, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)
        scale = head_dim ** -0.5
        attn_out = _local_attention(q, k, v, scale, causal=causal)
        out = attn_out.reshape(B, S_local, -1)
        return F.linear(out, w_o)

    # Warm up extension on rank 0 first to avoid race
    _get_ext()

    B, S_local, H = hidden_states.shape
    head_dim = (w_qkv.shape[0] // 3) // num_heads
    assert num_heads % world_size == 0

    qkv = F.linear(hidden_states, w_qkv)
    qkv = qkv.view(B, S_local, 3, num_heads, head_dim)
    q = qkv[:, :, 0].contiguous()  # [B, S_local, num_heads, head_dim]
    k = qkv[:, :, 1].contiguous()
    v = qkv[:, :, 2].contiguous()

    # Pre-A2A: gather seq, scatter heads. scatter_dim=2 (heads), gather_dim=1 (seq).
    # For each, scatter heads across world_size, gather seq.
    # Pad seq if needed
    S_total = S_local  # per-rank seq is S_local; after gather along seq it's S_local * world_size

    # Stack k and v along head dim to do a single all-to-all
    kv = torch.stack([k, v], dim=3).reshape(B, S_local, 2 * num_heads, head_dim).contiguous()

    q_g = _all_to_all_dim(q, scatter_dim=2, gather_dim=1, world_size=world_size, rank=rank)
    kv_g = _all_to_all_dim(kv, scatter_dim=2, gather_dim=1, world_size=world_size, rank=rank)

    S_full = q_g.size(1)
    kv_g = kv_g.reshape(B, S_full, num_heads // world_size, 2, head_dim)
    k_g = kv_g[:, :, :, 0, :].contiguous()
    v_g = kv_g[:, :, :, 1, :].contiguous()

    scale = head_dim ** -0.5
    attn_out = _local_attention(q_g, k_g, v_g, scale, causal=causal)
    # attn_out: [B, S_full, num_heads//world_size, head_dim]

    # Post-A2A: gather heads, scatter seq. scatter_dim=1 (seq), gather_dim=2 (heads)
    attn_out = attn_out.contiguous()
    attn_out = _all_to_all_dim(attn_out, scatter_dim=1, gather_dim=2, world_size=world_size, rank=rank)

    out = attn_out.reshape(B, attn_out.size(1), -1)
    return F.linear(out, w_o)