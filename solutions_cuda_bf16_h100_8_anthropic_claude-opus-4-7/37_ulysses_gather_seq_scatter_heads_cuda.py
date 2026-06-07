"""
Ulysses gather_seq_scatter_heads via symmetric memory all-to-all.

Strategy:
- Use symm_mem buffers for input/output staging.
- Each rank writes its scatter chunks into peer symmetric buffers via direct
  UVA stores (one CUDA kernel does the all-to-all by remote writes).
- Then a local kernel concatenates received chunks along gather_dim.
- Barriers via symm_mem signal pad inside kernels.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import ProcessGroup

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Copy a contiguous block of bytes between device pointers (peer or local).
__global__ void copy_bytes_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t nbytes
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // 16-byte vectorized copy when aligned
    int64_t n16 = nbytes / 16;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (int64_t i = idx; i < n16; i += stride) {
        d4[i] = s4[i];
    }
    int64_t tail_start = n16 * 16;
    for (int64_t i = tail_start + idx; i < nbytes; i += stride) {
        dst[i] = src[i];
    }
}

// Generic strided copy from a 3D logical view [outer, mid, inner] in bf16 elements.
// src layout: src[o, m, i] = src_base[o*src_outer_stride + m*src_mid_stride + i]
// dst layout: dst[o, m, i] = dst_base[o*dst_outer_stride + m*dst_mid_stride + i]
__global__ void strided_copy_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t outer, int64_t mid, int64_t inner,
    int64_t src_outer_stride, int64_t src_mid_stride,
    int64_t dst_outer_stride, int64_t dst_mid_stride
) {
    int64_t total = outer * mid * inner;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t t = idx; t < total; t += stride) {
        int64_t i = t % inner;
        int64_t m = (t / inner) % mid;
        int64_t o = t / (inner * mid);
        dst[o * dst_outer_stride + m * dst_mid_stride + i] =
            src[o * src_outer_stride + m * src_mid_stride + i];
    }
}

void launch_copy_bytes(uint64_t src_ptr, uint64_t dst_ptr, int64_t nbytes) {
    if (nbytes <= 0) return;
    int threads = 256;
    int64_t n16 = nbytes / 16;
    int64_t units = n16 > 0 ? n16 : nbytes;
    int blocks = (int)((units + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    copy_bytes_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src_ptr),
        reinterpret_cast<uint8_t*>(dst_ptr),
        nbytes);
}

void launch_strided_copy_bf16(
    uint64_t src_ptr, uint64_t dst_ptr,
    int64_t outer, int64_t mid, int64_t inner,
    int64_t src_outer_stride, int64_t src_mid_stride,
    int64_t dst_outer_stride, int64_t dst_mid_stride
) {
    int64_t total = outer * mid * inner;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    strided_copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_ptr),
        reinterpret_cast<__nv_bfloat16*>(dst_ptr),
        outer, mid, inner,
        src_outer_stride, src_mid_stride,
        dst_outer_stride, dst_mid_stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_copy_bytes", &launch_copy_bytes, "Peer/local byte copy");
    m.def("launch_strided_copy_bf16", &launch_strided_copy_bf16, "Strided bf16 copy");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_a2a_ext", CUDA_SRC)
    return _ext


_buf_cache = {}


def _get_symm_buf(nbytes: int, device: torch.device, group):
    # Round up to multiple of 16 for alignment
    nbytes = (nbytes + 15) // 16 * 16
    key = (nbytes, device.index, id(group))
    if key in _buf_cache:
        return _buf_cache[key]
    # allocate as bytes via int8 tensor of length nbytes
    buf = symm_mem.empty(nbytes, device=device, dtype=torch.int8)
    hdl = symm_mem.rendezvous(buf, group)
    _buf_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: int = 0,
) -> torch.Tensor:
    if group is None:
        return x

    sp_world = dist.get_world_size(group)
    if sp_world == 1:
        if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
            slc = [slice(None)] * x.dim()
            padding_size = x.size(seq_dim) - unpadded_dim_size
            if padding_size > 0:
                slc[seq_dim] = slice(0, -padding_size)
                x = x[tuple(slc)]
        return x

    rank = dist.get_rank(group)
    device = x.device

    assert x.dtype == torch.bfloat16, "This optimized path expects bfloat16"

    x = x.contiguous()
    ext = _get_ext()

    # Logical view: collapse dims into [outer, scatter_dim_size, inner]
    # where outer = prod(dims before head_dim), scatter_dim_size = x.size(head_dim),
    # inner = prod(dims after head_dim). We split head_dim into sp_world chunks.
    # For all-to-all, rank r sends chunk r (along head_dim) to rank r.
    # After all-to-all, recv buffer at rank R has, for each source rank s, the
    # chunk that s sent. We then need to concatenate along seq_dim.

    shape = list(x.shape)
    H = shape[head_dim]
    S = shape[seq_dim]
    assert H % sp_world == 0, "head_dim must be divisible by sp_world"
    assert S % sp_world == 0, "seq_dim must be divisible by sp_world"

    h_per = H // sp_world

    # Build "outer" and "inner" relative to head_dim
    outer_h = 1
    for i in range(head_dim):
        outer_h *= shape[i]
    inner_h = 1
    for i in range(head_dim + 1, len(shape)):
        inner_h *= shape[i]

    # Each chunk along head_dim has size: outer_h * h_per * inner_h elements (bf16)
    chunk_elems = outer_h * h_per * inner_h
    chunk_bytes = chunk_elems * 2  # bf16

    total_bytes = chunk_bytes * sp_world

    # Allocate symm send and recv buffers
    send_buf, send_hdl = _get_symm_buf(total_bytes, device, group)
    recv_buf, recv_hdl = _get_symm_buf(total_bytes, device, group)

    # Step 1: pack x into send_buf such that send_buf[r*chunk_bytes:(r+1)*chunk_bytes]
    # contains the chunk to send to rank r. The chunk corresponds to slicing head_dim
    # at [r*h_per:(r+1)*h_per]. We can do this with strided_copy:
    # source: x viewed as [outer_h, sp_world, h_per, inner_h]
    # dest:   send_buf viewed as [sp_world, outer_h, h_per, inner_h]
    # i.e., transpose first two dims. Rearrange so chunk r is contiguous in send_buf.

    # We do sp_world strided copies (one per chunk). For each rank r, copy
    # src: x[..., r*h_per:(r+1)*h_per, ...] (in head_dim) -> send_buf chunk r
    # In send_buf chunk r, layout is [outer_h, h_per, inner_h] contiguous.
    src_base_ptr = x.data_ptr()
    send_base_ptr = send_buf.data_ptr()

    src_outer_stride = H * inner_h  # stride for outer index (elements)
    src_mid_stride = inner_h  # stride for h dimension within head_dim

    for r in range(sp_world):
        src_ptr_r = src_base_ptr + (r * h_per * inner_h) * 2
        dst_ptr_r = send_base_ptr + r * chunk_bytes
        ext.launch_strided_copy_bf16(
            src_ptr_r, dst_ptr_r,
            outer_h, h_per, inner_h,
            src_outer_stride, src_mid_stride,
            h_per * inner_h, inner_h,
        )

    # Barrier: ensure all ranks finished packing send_buf and are ready for peer reads
    send_hdl.barrier(channel=0)

    # Step 2: each rank r writes its chunk r into peer's recv_buf at slot=rank.
    # That is: for each peer p, we copy send_buf[p*chunk_bytes : (p+1)*chunk_bytes]
    # to peer_p's recv_buf[rank*chunk_bytes : (rank+1)*chunk_bytes].
    for p in range(sp_world):
        peer_recv_ptr = int(recv_hdl.buffer_ptrs[p])
        dst_ptr = peer_recv_ptr + rank * chunk_bytes
        src_ptr = send_base_ptr + p * chunk_bytes
        ext.launch_copy_bytes(src_ptr, dst_ptr, chunk_bytes)

    # Barrier: ensure all peer writes to our recv_buf are done before reading
    recv_hdl.barrier(channel=1)

    # Step 3: assemble output tensor. After all-to-all on head_dim, we have
    # received sp_world chunks; each chunk c was originally at source rank c
    # and has the slice [c*h_per:(c+1)*h_per] of the *original* head dim BUT only
    # 1/sp_world of the seq dim (the part that source rank c held).
    # Wait — re-think: in the reference, scatter_dim=head_dim, gather_dim=seq_dim.
    # Each rank starts with full head dim H but a 1/sp_world slice of seq.
    # all_to_all splits head into sp_world chunks of size h_per, sends chunk r to rank r.
    # After that, rank R has h_per heads, but full seq (concatenated from all sources).
    # Source rank s sent its chunk R (heads R*h_per:(R+1)*h_per) to us; that chunk has
    # the seq slice that rank s held.
    # We need to concatenate along seq_dim in source-rank order.

    # The received recv_buf layout: [sp_world, outer_h, h_per, inner_h] contiguous
    # where the first dim is source rank s (we wrote slot=rank from peer p, but each
    # peer p wrote its chunk=rank, and we wrote our send chunk p to peer p's slot=rank;
    # so in our recv_buf, slot s contains the chunk we received FROM source rank s,
    # which is heads [rank*h_per:(rank+1)*h_per] from rank s's original tensor with
    # rank s's seq slice).
    #
    # outer_h in this packing corresponds to dims before head_dim of original x, which
    # includes seq_dim if seq_dim < head_dim.
    #
    # We need to construct output with shape:
    #   shape_out = shape; shape_out[head_dim] = h_per; shape_out[seq_dim] = S (full)
    # and seq_dim entries from source rank s go to seq positions [s*S_local:(s+1)*S_local]
    # where S_local = S (since each rank holds the full local seq before).
    # Wait — each rank holds 1/sp_world of S already in input. So input seq size at this
    # dim is S_in = S (the input's seq_dim size). After all-to-all gather on seq, output
    # seq size = S_in * sp_world.

    S_in = shape[seq_dim]
    S_out = S_in * sp_world

    out_shape = list(shape)
    out_shape[head_dim] = h_per
    out_shape[seq_dim] = S_out
    output = torch.empty(out_shape, dtype=x.dtype, device=device)

    # Now we need to scatter recv_buf chunks into output along seq_dim.
    # recv_buf chunk s has layout [outer_h, h_per, inner_h] which logically corresponds
    # to original x's [dims_before_head_dim, h_per, dims_after_head_dim] for source rank s.
    # outer_h decomposes as (dims_before_head_dim of original x, in order). seq_dim might be
    # one of those dims (if seq_dim < head_dim) or in inner_h (if seq_dim > head_dim).

    # Easier approach: view recv_buf as a tensor with shape:
    # [sp_world] + shape_with_h_per
    # where shape_with_h_per = shape but with head_dim replaced by h_per.
    shape_with_h_per = list(shape)
    shape_with_h_per[head_dim] = h_per
    recv_view = recv_buf.view(torch.bfloat16).view([sp_world] + shape_with_h_per)

    # Now concatenate along seq_dim. seq position in output: source-rank-major.
    # output[..., seq_dim slice s*S_in:(s+1)*S_in, ...] = recv_view[s]
    # Use torch.cat over the sp_world dim along seq_dim+1 (since sp_world is dim 0).
    # Actually: recv_view has shape [W, ..., S_in (at seq_dim+1), ..., h_per, ...].
    # We want to move dim 0 next to seq_dim and merge.

    # Permute so that the W dim is right before seq_dim, then reshape.
    perm = list(range(recv_view.dim()))
    # recv_view dims: 0=W, 1..=original dims (seq_dim is at 1+seq_dim, head_dim at 1+head_dim).
    src_seq_axis = 1 + seq_dim
    # Move axis 0 to position src_seq_axis (so W ends up at src_seq_axis, and seq follows).
    perm.remove(0)
    perm.insert(src_seq_axis, 0)
    recv_perm = recv_view.permute(perm).contiguous()
    # Now shape: [..., W, S_in, ...] with W at position seq_dim, S_in at seq_dim+1.
    # Merge them.
    new_shape = list(recv_perm.shape)
    merged = new_shape[seq_dim] * new_shape[seq_dim + 1]
    new_shape = new_shape[:seq_dim] + [merged] + new_shape[seq_dim + 2:]
    output = recv_perm.view(new_shape)

    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = output.size(seq_dim) - unpadded_dim_size
        if padding_size > 0:
            slc = [slice(None)] * output.dim()
            slc[seq_dim] = slice(0, -padding_size)
            output = output[tuple(slc)].contiguous()

    return output