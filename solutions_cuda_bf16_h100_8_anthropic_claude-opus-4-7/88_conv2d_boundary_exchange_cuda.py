"""
Distributed Conv2d with patch-parallel boundary exchange using symmetric memory.

Strategy:
- Each rank publishes its top/bottom halo rows into a symmetric memory buffer.
- A custom CUDA kernel pulls neighbor halos directly from peer GPUs via UVA
  pointers (symm_mem.buffer_ptrs) into a locally-padded input tensor.
- The local conv2d then runs with width-only padding.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Copy halos from peers into local padded buffer.
// Layout of symm buffer per rank: [top_halo (boundary rows) | bottom_halo (boundary rows)]
// Each "row" is B*C*W elements of bf16.
//
// padded_x: [B, C, H_local + 2*boundary, W]
// We write:
//   - top pad rows [0, boundary) <- previous rank's bottom halo (or zeros if rank==0)
//   - middle [boundary, boundary+H_local) <- local x (already filled by caller via copy)
//   - bottom pad rows [boundary+H_local, H_local+2*boundary) <- next rank's top halo (or zeros)

extern "C" __global__ void fill_halos_kernel(
    const uint64_t* __restrict__ peer_ptrs,  // [world_size]
    __nv_bfloat16* __restrict__ padded_x,    // [B, C, H_pad, W]
    int B, int C, int H_local, int W, int boundary,
    int rank, int world_size
) {
    // Each thread handles one element of the halo region (top and/or bottom pad).
    // Total halo elements per side: B * C * boundary * W
    int64_t halo_size = (int64_t)B * C * boundary * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int H_pad = H_local + 2 * boundary;
    int64_t plane = (int64_t)H_pad * W;

    // halo layout per peer: [top (boundary*B*C*W) | bottom (boundary*B*C*W)]
    int64_t per_side = halo_size;
    int64_t per_peer = 2 * per_side;

    // Top halo fill
    if (tid < halo_size) {
        // Decompose tid into (b, c, h, w) where h in [0, boundary)
        int64_t idx = tid;
        int w = idx % W; idx /= W;
        int h = idx % boundary; idx /= boundary;
        int c = idx % C; idx /= C;
        int b = (int)idx;

        __nv_bfloat16 val;
        if (rank == 0) {
            val = __float2bfloat16(0.0f);
        } else {
            // Read from previous rank's bottom halo
            const __nv_bfloat16* peer_buf =
                reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[rank - 1]);
            // Index in peer buffer: bottom side starts at per_side
            int64_t peer_idx = per_side
                + ((int64_t)b * C + c) * boundary * W
                + (int64_t)h * W + w;
            val = peer_buf[peer_idx];
        }
        // Write to padded_x at row h (top pad)
        int64_t out_idx = ((int64_t)b * C + c) * plane + (int64_t)h * W + w;
        padded_x[out_idx] = val;
    }

    // Bottom halo fill (use second wave of threads beyond halo_size)
    int64_t tid2 = tid - halo_size;
    if (tid2 >= 0 && tid2 < halo_size) {
        int64_t idx = tid2;
        int w = idx % W; idx /= W;
        int h = idx % boundary; idx /= boundary;
        int c = idx % C; idx /= C;
        int b = (int)idx;

        __nv_bfloat16 val;
        if (rank == world_size - 1) {
            val = __float2bfloat16(0.0f);
        } else {
            // Read from next rank's top halo
            const __nv_bfloat16* peer_buf =
                reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[rank + 1]);
            int64_t peer_idx = ((int64_t)b * C + c) * boundary * W
                + (int64_t)h * W + w;
            val = peer_buf[peer_idx];
        }
        int64_t out_row = boundary + H_local + h;
        int64_t out_idx = ((int64_t)b * C + c) * plane + out_row * W + w;
        padded_x[out_idx] = val;
    }
}

// Pack local x's top and bottom halos into the symmetric publish buffer.
extern "C" __global__ void pack_halos_kernel(
    const __nv_bfloat16* __restrict__ x,   // [B, C, H_local, W]
    __nv_bfloat16* __restrict__ symm_buf,  // [2 * B * C * boundary * W]
    int B, int C, int H_local, int W, int boundary
) {
    int64_t halo_size = (int64_t)B * C * boundary * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    int64_t plane = (int64_t)H_local * W;

    if (tid < halo_size) {
        int64_t idx = tid;
        int w = idx % W; idx /= W;
        int h = idx % boundary; idx /= boundary;
        int c = idx % C; idx /= C;
        int b = (int)idx;
        // top: rows [0, boundary)
        int64_t in_idx = ((int64_t)b * C + c) * plane + (int64_t)h * W + w;
        symm_buf[tid] = x[in_idx];
    }

    int64_t tid2 = tid - halo_size;
    if (tid2 >= 0 && tid2 < halo_size) {
        int64_t idx = tid2;
        int w = idx % W; idx /= W;
        int h = idx % boundary; idx /= boundary;
        int c = idx % C; idx /= C;
        int b = (int)idx;
        // bottom: rows [H_local - boundary, H_local)
        int row = H_local - boundary + h;
        int64_t in_idx = ((int64_t)b * C + c) * plane + (int64_t)row * W + w;
        symm_buf[halo_size + tid2] = x[in_idx];
    }
}

// Copy local x into the middle of padded_x (rows [boundary, boundary+H_local))
extern "C" __global__ void copy_middle_kernel(
    const __nv_bfloat16* __restrict__ x,        // [B,C,H_local,W]
    __nv_bfloat16* __restrict__ padded_x,       // [B,C,H_pad,W]
    int B, int C, int H_local, int W, int boundary
) {
    int64_t total = (int64_t)B * C * H_local * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int64_t idx = tid;
    int w = idx % W; idx /= W;
    int h = idx % H_local; idx /= H_local;
    int c = idx % C; idx /= C;
    int b = (int)idx;
    int H_pad = H_local + 2 * boundary;
    int64_t in_idx = ((int64_t)b * C + c) * (int64_t)H_local * W + (int64_t)h * W + w;
    int64_t out_idx = ((int64_t)b * C + c) * (int64_t)H_pad * W + (int64_t)(h + boundary) * W + w;
    padded_x[out_idx] = x[in_idx];
}

void launch_pack_halos(
    torch::Tensor x,
    torch::Tensor symm_buf,
    int B, int C, int H_local, int W, int boundary
) {
    int64_t halo_size = (int64_t)B * C * boundary * W;
    int64_t total = 2 * halo_size;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_halos_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)symm_buf.data_ptr<at::BFloat16>(),
        B, C, H_local, W, boundary);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fill_halos(
    torch::Tensor peer_ptrs,
    torch::Tensor padded_x,
    int B, int C, int H_local, int W, int boundary,
    int rank, int world_size
) {
    int64_t halo_size = (int64_t)B * C * boundary * W;
    int64_t total = 2 * halo_size;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fill_halos_kernel<<<blocks, threads, 0, stream>>>(
        (const uint64_t*)peer_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)padded_x.data_ptr<at::BFloat16>(),
        B, C, H_local, W, boundary,
        rank, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_copy_middle(
    torch::Tensor x,
    torch::Tensor padded_x,
    int B, int C, int H_local, int W, int boundary
) {
    int64_t total = (int64_t)B * C * H_local * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    copy_middle_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)padded_x.data_ptr<at::BFloat16>(),
        B, C, H_local, W, boundary);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_halos", &launch_pack_halos, "Pack local halos into symm buffer");
    m.def("fill_halos", &launch_fill_halos, "Fill halos in padded buffer from peers");
    m.def("copy_middle", &launch_copy_middle, "Copy local x to middle of padded");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("distrifuser_halo_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_resources(B, C, H_local, W, boundary, dtype, device, group):
    key = (B, C, H_local, W, boundary, dtype, device.index)
    if key in _cache:
        return _cache[key]

    halo_size = B * C * boundary * W
    symm_buf = symm_mem.empty(2 * halo_size, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(symm_buf, group)
    peer_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    H_pad = H_local + 2 * boundary
    padded_x = torch.empty((B, C, H_pad, W), device=device, dtype=dtype)

    res = (symm_buf, hdl, peer_ptrs, padded_x)
    _cache[key] = res
    return res


@torch.no_grad()
def solution(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int = 1,
    padding: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    boundary = int(padding)

    if boundary == 0 or world_size == 1:
        return F.conv2d(x, weight, bias, stride=stride, padding=padding)

    if x.dtype != torch.bfloat16:
        # Fallback to reference behavior for non-bf16
        local = torch.stack([x[:, :, :boundary, :], x[:, :, -boundary:, :]], dim=0)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local.contiguous(), group=group)
        pieces = []
        if rank == 0:
            pieces.append(x.new_zeros(*x.shape[:2], boundary, x.shape[-1]))
        else:
            pieces.append(gathered[rank - 1][1])
        pieces.append(x)
        if rank == world_size - 1:
            pieces.append(x.new_zeros(*x.shape[:2], boundary, x.shape[-1]))
        else:
            pieces.append(gathered[rank + 1][0])
        padded_x = torch.cat(pieces, dim=2)
        return F.conv2d(padded_x, weight, bias, stride=stride, padding=(0, padding))

    x = x.contiguous()
    B, C, H_local, W = x.shape

    ext = _get_ext()
    symm_buf, hdl, peer_ptrs, padded_x = _get_resources(
        B, C, H_local, W, boundary, x.dtype, x.device, group
    )

    # Pack halos into symmetric buffer (publish)
    ext.pack_halos(x, symm_buf, B, C, H_local, W, boundary)

    # Copy local x into the middle of padded buffer (overlap with peer pack)
    ext.copy_middle(x, padded_x, B, C, H_local, W, boundary)

    # Sync so all peers have published their halos
    hdl.barrier(channel=0)

    # Pull halos directly from peer GPUs via UVA pointers
    ext.fill_halos(peer_ptrs, padded_x, B, C, H_local, W, boundary,
                   rank, world_size)

    # Run the conv with only width padding
    out = F.conv2d(padded_x, weight, bias, stride=stride, padding=(0, padding))

    # Make sure peers don't overwrite their symm_buf before we've consumed it
    hdl.barrier(channel=1)

    return out