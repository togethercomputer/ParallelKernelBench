import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import ProcessGroup
from typing import Optional

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

__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__global__ void global_barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size,
    uint64_t block_id
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

// All-to-all + concat-on-head_dim kernel, BF16.
//
// Logical layout:
//   Source x has shape [..., S, ..., H, ...] but we collapse to
//   [outer_pre_seq, S_local_total, mid, H_total, inner] where:
//     - seq_dim partitions S_local_total into W chunks of S_chunk = S_local_total/W
//     - head_dim has H_total heads, after a2a result has H_total*W heads
//
// For the post-attention gather_heads_scatter_seq:
//   scatter_dim = seq_dim, gather_dim = head_dim
//   Input on rank r: shape with S = S_local_total (full), H = H_local
//   Output on rank r: S_chunk on seq, H_local*W on head
//
// We split input along seq_dim into W chunks. Chunk c goes to rank c.
// On rank c, the data from sender r becomes the r-th block along head_dim.
//
// We write directly into peer symmetric output buffers:
//   For each (outer, s_local, mid, h, inner) in the c-th seq slice,
//   target rank = c, target offset on head_dim = my_rank * H_local + h.

__global__ void a2a_scatter_seq_gather_head_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    const uint64_t* __restrict__ dst_ptrs,  // [W] dst buffer pointers (one per peer)
    int64_t outer_pre_seq,
    int64_t S_chunk,        // S_local_total / W
    int64_t mid,
    int64_t H_local,
    int64_t inner,
    int W,
    int my_rank
) {
    // Total elements per chunk per rank
    const int64_t per_chunk = outer_pre_seq * S_chunk * mid * H_local * inner;

    // grid.y = chunk index c (peer), grid.x = element within chunk
    const int c = blockIdx.y;
    const int64_t total = per_chunk;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Source base: chunk c starts at seq offset c*S_chunk
    // Destination layout on peer c:
    //   shape [outer_pre_seq, S_chunk, mid, H_local*W, inner]
    //   our writes go to head slice [my_rank*H_local : (my_rank+1)*H_local]

    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(dst_ptrs[c]);

    // Source strides (input shape: [outer_pre_seq, S_local_total, mid, H_local, inner])
    // S_local_total = S_chunk * W
    const int64_t S_local_total = S_chunk * (int64_t)W;
    const int64_t src_stride_inner = 1;
    const int64_t src_stride_h = inner;
    const int64_t src_stride_mid = H_local * inner;
    const int64_t src_stride_s = mid * H_local * inner;
    const int64_t src_stride_outer = S_local_total * mid * H_local * inner;

    // Destination strides (output shape: [outer_pre_seq, S_chunk, mid, H_local*W, inner])
    const int64_t H_total = H_local * (int64_t)W;
    const int64_t dst_stride_inner = 1;
    const int64_t dst_stride_h = inner;
    const int64_t dst_stride_mid = H_total * inner;
    const int64_t dst_stride_s = mid * H_total * inner;
    const int64_t dst_stride_outer = S_chunk * mid * H_total * inner;

    const int64_t head_offset_dst = (int64_t)my_rank * H_local;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += stride) {
        // Decompose idx into (o, s, m, h, i)
        int64_t i = idx % inner;
        int64_t t = idx / inner;
        int64_t h = t % H_local;
        t = t / H_local;
        int64_t m = t % mid;
        t = t / mid;
        int64_t s = t % S_chunk;
        int64_t o = t / S_chunk;

        int64_t src_off = o * src_stride_outer
                        + ((int64_t)c * S_chunk + s) * src_stride_s
                        + m * src_stride_mid
                        + h * src_stride_h
                        + i;
        int64_t dst_off = o * dst_stride_outer
                        + s * dst_stride_s
                        + m * dst_stride_mid
                        + (head_offset_dst + h) * dst_stride_h
                        + i;
        dst[dst_off] = src[src_off];
    }
}

void launch_global_barrier(
    torch::Tensor signal_pad_ptrs,
    int64_t rank,
    int64_t world_size,
    int64_t block_id
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = world_size;
    if (threads < 32) threads = 32;
    global_barrier_kernel<<<1, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>()),
        (int)rank, (int)world_size, (uint64_t)block_id);
}

void launch_a2a_scatter_gather_bf16(
    torch::Tensor src,
    torch::Tensor dst_ptrs,  // int64 [W]
    int64_t outer_pre_seq,
    int64_t S_chunk,
    int64_t mid,
    int64_t H_local,
    int64_t inner,
    int64_t world_size,
    int64_t my_rank
) {
    int64_t per_chunk = outer_pre_seq * S_chunk * mid * H_local * inner;
    int threads = 256;
    int64_t blocks_x_64 = (per_chunk + threads - 1) / threads;
    if (blocks_x_64 > 4096) blocks_x_64 = 4096;
    int blocks_x = (int)blocks_x_64;
    if (blocks_x < 1) blocks_x = 1;
    dim3 grid(blocks_x, (unsigned int)world_size, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    a2a_scatter_seq_gather_head_bf16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint64_t*>(dst_ptrs.data_ptr<int64_t>()),
        outer_pre_seq, S_chunk, mid, H_local, inner,
        (int)world_size, (int)my_rank
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_global_barrier", &launch_global_barrier, "Symm-mem global barrier");
    m.def("launch_a2a_scatter_gather_bf16", &launch_a2a_scatter_gather_bf16, "Fused A2A scatter-seq gather-head BF16");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_gather_heads_scatter_seq_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _get_buffers(in_shape, out_shape, dtype, device, group):
    key = (tuple(in_shape), tuple(out_shape), dtype, device, group)
    if key in _resource_cache:
        return _resource_cache[key]

    in_buf = symm_mem.empty(in_shape, device=device, dtype=dtype)
    in_hdl = symm_mem.rendezvous(in_buf, group)

    out_buf = symm_mem.empty(out_shape, device=device, dtype=dtype)
    out_hdl = symm_mem.rendezvous(out_buf, group)

    dst_ptrs = torch.tensor(out_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (in_buf, in_hdl, out_buf, out_hdl, dst_ptrs)
    _resource_cache[key] = res
    return res


_barrier_counter = [0]


@torch.no_grad()
def solution(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    if group is None:
        return x

    sp_world = dist.get_world_size(group)
    if sp_world == 1:
        return x

    # Pad seq dim to multiple of sp_world
    dim_size = x.size(seq_dim)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        shape = list(x.shape)
        shape[seq_dim] = padding_size
        pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=seq_dim)

    x = x.contiguous()
    rank = dist.get_rank(group)

    # Normalize dims
    nd = x.dim()
    sd = seq_dim if seq_dim >= 0 else seq_dim + nd
    hd = head_dim if head_dim >= 0 else head_dim + nd

    # Collapse shape to [outer_pre_seq, S, mid, H, inner] where:
    #   outer_pre_seq = prod(dims before min(sd,hd))
    #   The two "feature" dims are seq and head; we need both. They may be in either order.
    # Strategy: handle generically by collapsing by min/max position.
    # For Ulysses post-attn: x is typically [b, s, h, d]: sd=1, hd=2. Common case sd < hd.
    # We'll require sd != hd and handle sd<hd or sd>hd.

    assert sd != hd
    if sd < hd:
        # outer = dims [0..sd), mid = dims (sd..hd), inner = dims (hd..end)
        outer_pre_seq = 1
        for i in range(0, sd):
            outer_pre_seq *= x.shape[i]
        S = x.shape[sd]
        mid = 1
        for i in range(sd + 1, hd):
            mid *= x.shape[i]
        H_local = x.shape[hd]
        inner = 1
        for i in range(hd + 1, nd):
            inner *= x.shape[i]
        x_view = x.reshape(outer_pre_seq, S, mid, H_local, inner)
    else:
        # hd < sd: need to put seq before head. Permute: bring hd before sd.
        # But to keep contiguous logic, we just transpose into a canonical view.
        # Construct: outer = [0..hd), then head, then mid=(hd..sd), then seq, then inner=(sd..end)
        # We need shape [outer, S, mid, H, inner] with seq before head. Swap head and seq.
        outer_pre_head = 1
        for i in range(0, hd):
            outer_pre_head *= x.shape[i]
        H_local = x.shape[hd]
        mid = 1
        for i in range(hd + 1, sd):
            mid *= x.shape[i]
        S = x.shape[sd]
        inner = 1
        for i in range(sd + 1, nd):
            inner *= x.shape[i]
        # original collapsed: [outer_pre_head, H_local, mid, S, inner]
        x_view = x.reshape(outer_pre_head, H_local, mid, S, inner).transpose(1, 3).contiguous()
        # now [outer_pre_head, S, mid, H_local, inner]
        outer_pre_seq = outer_pre_head
        x_view = x_view.reshape(outer_pre_seq, S, mid, H_local, inner)

    assert S % sp_world == 0
    S_chunk = S // sp_world
    H_total = H_local * sp_world

    # Output collapsed shape: [outer_pre_seq, S_chunk, mid, H_total, inner]
    in_shape = (outer_pre_seq, S, mid, H_local, inner)
    out_shape = (outer_pre_seq, S_chunk, mid, H_total, inner)

    in_buf, in_hdl, out_buf, out_hdl, dst_ptrs = _get_buffers(
        in_shape, out_shape, x.dtype, x.device, group
    )

    # Copy local input into symmetric input buffer (not strictly needed since we
    # only read locally, but keeps allocations stable). We can read directly from x_view.
    # We'll skip copying to in_buf and read x_view directly.

    ext = _get_ext()

    # Pre-barrier: ensure all peers ready (out_buf safe to write)
    _barrier_counter[0] = (_barrier_counter[0] + 1) % 64
    bid = _barrier_counter[0]
    ext.launch_global_barrier(out_hdl.signal_pad_ptrs_dev, out_hdl.rank, out_hdl.world_size, bid)

    # Launch fused A2A + gather kernel: write directly into peer out_bufs
    ext.launch_a2a_scatter_gather_bf16(
        x_view, dst_ptrs,
        outer_pre_seq, S_chunk, mid, H_local, inner,
        sp_world, rank
    )

    # Post-barrier: ensure all peers finished writing into our out_buf
    _barrier_counter[0] = (_barrier_counter[0] + 1) % 64
    bid = _barrier_counter[0]
    ext.launch_global_barrier(out_hdl.signal_pad_ptrs_dev, out_hdl.rank, out_hdl.world_size, bid)

    # Reshape result to user-facing shape
    if sd < hd:
        # original x shape with S replaced by S_chunk and H_local replaced by H_total
        final_shape = list(x.shape)
        final_shape[sd] = S_chunk
        final_shape[hd] = H_total
        result = out_buf.reshape(final_shape).clone()
    else:
        # hd < sd. We canonicalized by swapping. Now reverse: output collapsed is
        # [outer_pre_seq, S_chunk, mid, H_total, inner], but original wanted head before seq.
        # Build shape [outer_pre_head, H_total, mid, S_chunk, inner], then reshape to user shape.
        tmp = out_buf.reshape(outer_pre_seq, S_chunk, mid, H_total, inner).transpose(1, 3).contiguous()
        # tmp shape: [outer_pre_seq, H_total, mid, S_chunk, inner]
        final_shape = list(x.shape)
        final_shape[hd] = H_total
        final_shape[sd] = S_chunk
        result = tmp.reshape(final_shape).clone()

    return result