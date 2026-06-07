"""
RoPE + all-gather fused with custom CUDA: writes RoPE output directly into
symmetric memory at the correct rank slot, then peer-copies via UVA pointers.
Uses per-rank channels for pipelined copies overlapping with the local RoPE.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// RoPE kernel: writes [B, S_local, H, D] embedded output to dst pointer.
// cos/sin shape: [B, S_local, D]
__global__ void rope_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,    // [B, S, H, D]
    const __nv_bfloat16* __restrict__ cos,  // [B, S, D]
    const __nv_bfloat16* __restrict__ sin,  // [B, S, D]
    __nv_bfloat16* __restrict__ out,        // [B, S, H, D]
    int B, int S, int H, int D
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int bs = blockIdx.z;
    int b = bs / S;
    int s = bs % S;
    if (d >= D) return;

    int half = D / 2;
    int x_idx = ((b * S + s) * H + h) * D + d;
    int cs_idx = (b * S + s) * D + d;

    float xv = __bfloat162float(x[x_idx]);
    float cv = __bfloat162float(cos[cs_idx]);
    float sv = __bfloat162float(sin[cs_idx]);

    // rotate_half: if d < half, pair with x[d+half] negated; else pair with x[d-half]
    float xr;
    if (d < half) {
        int pair_idx = ((b * S + s) * H + h) * D + (d + half);
        xr = -__bfloat162float(x[pair_idx]);
    } else {
        int pair_idx = ((b * S + s) * H + h) * D + (d - half);
        xr = __bfloat162float(x[pair_idx]);
    }

    float result = xv * cv + xr * sv;
    out[x_idx] = __float2bfloat16(result);
}

// Bulk copy kernel: copy from a remote pointer into local destination
__global__ void copy_kernel_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // Use vectorized copy via uint4 when aligned
    int64_t n_vec = n / 8;
    const uint4* src_v = reinterpret_cast<const uint4*>(src);
    uint4* dst_v = reinterpret_cast<uint4*>(dst);
    for (int64_t i = idx; i < n_vec; i += stride) {
        dst_v[i] = src_v[i];
    }
    int64_t tail_start = n_vec * 8;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

void launch_rope_bf16(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin,
    int64_t out_ptr,
    int B, int S, int H, int D
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 block(128);
    dim3 grid((D + 127) / 128, H, B * S);
    rope_kernel_bf16<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)cos.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)sin.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)(uintptr_t)out_ptr,
        B, S, H, D);
}

void launch_copy_bf16(
    int64_t src_ptr, int64_t dst_ptr, int64_t n, int64_t stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)(uintptr_t)stream_ptr;
    int threads = 256;
    int blocks = 1024;
    copy_kernel_bf16<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)(uintptr_t)src_ptr,
        (__nv_bfloat16*)(uintptr_t)dst_ptr,
        n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_rope_bf16", &launch_rope_bf16, "RoPE BF16 -> dst pointer");
    m.def("launch_copy_bf16", &launch_copy_bf16, "P2P copy BF16");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("rope_allgather_ext_v1", CUDA_SRC)
    return _ext


_cache = {}

def _get_resources(B, S_local, H, D, dtype, device, world_size):
    key = (B, S_local, H, D, dtype, device, world_size)
    if key in _cache:
        return _cache[key]
    S_global = S_local * world_size
    # Symmetric buffers for q and k full output
    q_buf = symm_mem.empty((B, S_global, H, D), device=device, dtype=dtype)
    k_buf = symm_mem.empty((B, S_global, H, D), device=device, dtype=dtype)
    q_hdl = symm_mem.rendezvous(q_buf, dist.group.WORLD)
    k_hdl = symm_mem.rendezvous(k_buf, dist.group.WORLD)
    # Side streams for peer copies
    streams = [torch.cuda.Stream(device=device) for _ in range(world_size)]
    res = (q_buf, k_buf, q_hdl, k_hdl, streams)
    _cache[key] = res
    return res


@torch.no_grad()
def solution(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    cos_local: torch.Tensor,
    sin_local: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, S_local, H, D = q_local.shape
    device = q_local.device
    dtype = q_local.dtype

    q_local = q_local.contiguous()
    k_local = k_local.contiguous()
    cos_local = cos_local.contiguous()
    sin_local = sin_local.contiguous()

    if not dist.is_initialized():
        # local fallback
        ext = _get_ext()
        q_out = torch.empty_like(q_local)
        k_out = torch.empty_like(k_local)
        ext.launch_rope_bf16(q_local, cos_local, sin_local, int(q_out.data_ptr()), B, S_local, H, D)
        ext.launch_rope_bf16(k_local, cos_local, sin_local, int(k_out.data_ptr()), B, S_local, H, D)
        return q_out, k_out

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    ext = _get_ext()

    q_buf, k_buf, q_hdl, k_hdl, streams = _get_resources(
        B, S_local, H, D, dtype, device, world_size
    )

    # Compute RoPE directly into our slice of the symmetric buffer
    slice_elems = B * S_local * H * D
    q_slice_ptr = int(q_buf.data_ptr()) + rank * slice_elems * q_buf.element_size()
    k_slice_ptr = int(k_buf.data_ptr()) + rank * slice_elems * k_buf.element_size()

    ext.launch_rope_bf16(q_local, cos_local, sin_local, q_slice_ptr, B, S_local, H, D)
    ext.launch_rope_bf16(k_local, cos_local, sin_local, k_slice_ptr, B, S_local, H, D)

    # Barrier to ensure all ranks have written their slices
    q_hdl.barrier(channel=0)

    # Pull peer slices via P2P UVA, overlapping across streams
    cur_stream = torch.cuda.current_stream(device)
    main_event = torch.cuda.Event()
    main_event.record(cur_stream)

    elem_size = q_buf.element_size()
    for i in range(1, world_size):
        peer = (rank + i) % world_size
        s = streams[i % len(streams)]
        s.wait_event(main_event)
        with torch.cuda.stream(s):
            q_src = int(q_hdl.buffer_ptrs[peer]) + peer * slice_elems * elem_size
            q_dst = int(q_buf.data_ptr()) + peer * slice_elems * elem_size
            k_src = int(k_hdl.buffer_ptrs[peer]) + peer * slice_elems * elem_size
            k_dst = int(k_buf.data_ptr()) + peer * slice_elems * elem_size
            ext.launch_copy_bf16(q_src, q_dst, slice_elems, s.cuda_stream)
            ext.launch_copy_bf16(k_src, k_dst, slice_elems, s.cuda_stream)

    # Sync side streams back to current
    for s in streams:
        ev = torch.cuda.Event()
        ev.record(s)
        cur_stream.wait_event(ev)

    # Final barrier so peers don't reuse buffers prematurely
    q_hdl.barrier(channel=1)

    return q_buf.clone(), k_buf.clone()