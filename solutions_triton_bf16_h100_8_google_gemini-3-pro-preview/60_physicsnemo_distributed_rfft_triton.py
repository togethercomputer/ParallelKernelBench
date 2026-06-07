"""
Optimized Distributed 2D Real FFT with fused zero-copy All-to-All Transpose.
"""

import math
from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

template<typename scalar_t>
__global__ void all_to_all_transpose_5d_kernel(
    const scalar_t* __restrict__ src,
    const uintptr_t* peer_ptrs,
    int world_size,
    int rank,
    int chunk_H,
    int W,
    int dim0_is_first,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,
    int64_t src_stride0, int64_t src_stride1, int64_t src_stride2, int64_t src_stride3, int64_t src_stride4,
    int64_t dst_stride0, int64_t dst_stride1, int64_t dst_stride2, int64_t dst_stride3, int64_t dst_stride4
) {
    int64_t total_elements_per_chunk = s0 * s1 * s2 * s3 * s4;
    int64_t total_elements = total_elements_per_chunk * world_size;
    
    // Grid-stride loop ensures we never exceed max grid size while safely copying all elements
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        int c = idx / total_elements_per_chunk;
        int64_t elem_idx = idx % total_elements_per_chunk;

        // Unravel 5D index within the current chunk
        int64_t i4 = elem_idx % s4;
        int64_t temp = elem_idx / s4;
        int64_t i3 = temp % s3;
        temp /= s3;
        int64_t i2 = temp % s2;
        temp /= s2;
        int64_t i1 = temp % s1;
        int64_t i0 = temp / s1;

        int64_t src_offset = i0 * src_stride0 + i1 * src_stride1 + i2 * src_stride2 + i3 * src_stride3 + i4 * src_stride4;
        int64_t dst_offset = i0 * dst_stride0 + i1 * dst_stride1 + i2 * dst_stride2 + i3 * dst_stride3 + i4 * dst_stride4;

        if (dim0_is_first) {
            src_offset += c * chunk_H * src_stride1;
            dst_offset += rank * W * dst_stride3;
        } else {
            src_offset += c * chunk_H * src_stride3;
            dst_offset += rank * W * dst_stride1;
        }

        scalar_t* dst = reinterpret_cast<scalar_t*>(peer_ptrs[c]);
        dst[dst_offset] = src[src_offset];
    }
}

void all_to_all_transpose_cuda(
    torch::Tensor src,
    int64_t peer_ptrs_ptr,
    int world_size,
    int rank,
    int chunk_H,
    int W,
    int dim0_is_first,
    std::vector<int64_t> sizes,
    std::vector<int64_t> src_strides,
    std::vector<int64_t> dst_strides,
    int element_size
) {
    int64_t total_elements_per_chunk = sizes[0] * sizes[1] * sizes[2] * sizes[3] * sizes[4];
    int64_t total_elements = total_elements_per_chunk * world_size;
    int threads = 256;
    // Cap blocks to maintain safe grid sizes for huge tensors, letting grid-stride handle the rest
    int blocks = std::min((int)((total_elements + threads - 1) / threads), 262144);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uintptr_t* peer_ptrs = reinterpret_cast<const uintptr_t*>(peer_ptrs_ptr);

    // Vectorized transfers over NVLink natively based on complex sizes (e.g. 8 bytes for ComplexFloat)
    if (element_size == 4) {
        all_to_all_transpose_5d_kernel<int32_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const int32_t*>(src.data_ptr()), peer_ptrs,
            world_size, rank, chunk_H, W, dim0_is_first,
            sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
            src_strides[0], src_strides[1], src_strides[2], src_strides[3], src_strides[4],
            dst_strides[0], dst_strides[1], dst_strides[2], dst_strides[3], dst_strides[4]
        );
    } else if (element_size == 8) {
        all_to_all_transpose_5d_kernel<int64_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const int64_t*>(src.data_ptr()), peer_ptrs,
            world_size, rank, chunk_H, W, dim0_is_first,
            sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
            src_strides[0], src_strides[1], src_strides[2], src_strides[3], src_strides[4],
            dst_strides[0], dst_strides[1], dst_strides[2], dst_strides[3], dst_strides[4]
        );
    } else if (element_size == 16) {
        all_to_all_transpose_5d_kernel<ulonglong2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const ulonglong2*>(src.data_ptr()), peer_ptrs,
            world_size, rank, chunk_H, W, dim0_is_first,
            sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
            src_strides[0], src_strides[1], src_strides[2], src_strides[3], src_strides[4],
            dst_strides[0], dst_strides[1], dst_strides[2], dst_strides[3], dst_strides[4]
        );
    } else {
        TORCH_CHECK(false, "Unsupported element size for all-to-all transpose.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("all_to_all_transpose_cuda", &all_to_all_transpose_cuda,
          "All-to-all transpose kernel using UVA symmetric memory");
}
'''

_ext = None
_symm_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("all_to_all_transpose_ext", CUDA_SRC)
    return _ext


def _get_symm_state(shape, dtype, device, group):
    """Caches symmetric memory buffers and peer pointer tensors."""
    global _symm_cache
    key = (tuple(shape), dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]

    numel = math.prod(shape)
    buf = symm_mem.empty(numel, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    peer_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    out = buf.view(shape)
    
    state = (out, hdl, peer_ptrs)
    _symm_cache[key] = state
    return state


def _get_5d_params(shape, dim0, dim1, world_size):
    """Collapses N-dim shape into 5 dimensions for the fast zero-copy fused scatter kernel."""
    H, W = shape[dim0], shape[dim1]
    chunk_H = H // world_size

    if dim0 < dim1:
        s0, s1 = math.prod(shape[:dim0]), chunk_H
        s2 = math.prod(shape[dim0+1:dim1])
        s3, s4 = W, math.prod(shape[dim1+1:])

        src_strides = [s4 * W * s2 * H, s4 * W * s2, s4 * W, s4, 1]
        dst_strides = [s4 * (W * world_size) * s2 * chunk_H, s4 * (W * world_size) * s2, s4 * (W * world_size), s4, 1]
        
        return (s0, s1, s2, s3, s4), src_strides, dst_strides, True, chunk_H, W
    else:
        s0, s1 = math.prod(shape[:dim1]), W
        s2 = math.prod(shape[dim1+1:dim0])
        s3, s4 = chunk_H, math.prod(shape[dim0+1:])

        src_strides = [s4 * H * s2 * W, s4 * H * s2, s4 * H, s4, 1]
        dst_strides = [s4 * chunk_H * s2 * (W * world_size), s4 * chunk_H * s2, s4 * chunk_H, s4, 1]
        
        return (s0, s1, s2, s3, s4), src_strides, dst_strides, False, chunk_H, W


def _truncate(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Return a contiguous slice tensor[..., :size, ...] along dim."""
    slices = [slice(None)] * tensor.ndim
    slices[dim % tensor.ndim] = slice(0, size)
    return tensor[tuple(slices)].contiguous()


@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Sequence[int],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    ndim = x.ndim
    dim0 = int(dim[0]) % ndim
    dim1 = int(dim[1]) % ndim

    if rank == 0:
        _get_ext()
    dist.barrier(group)
    ext = _get_ext()

    # 1. Transform the replicated spatial dimension.
    x1 = torch.fft.fft(x, n=int(s[0]), dim=dim0, norm=norm).contiguous()

    if world_size == 1:
        x2 = torch.fft.fft(x1, n=int(s[1]), dim=dim1, norm=norm)
        return _truncate(x2, dim1, x2.shape[dim1] // 2 + 1)

    # 2. Setup Transpose via Fused Device-Side Data Movement
    dst_shape = list(x1.shape)
    dst_shape[dim0] = x1.shape[dim0] // world_size
    dst_shape[dim1] = x1.shape[dim1] * world_size

    out_buf, hdl, peer_ptrs = _get_symm_state(tuple(dst_shape), x1.dtype, x1.device, group)
    sizes, src_strides, dst_strides, dim0_is_first, chunk_H, W = _get_5d_params(x1.shape, dim0, dim1, world_size)

    # Sync prior to scatter (so we do not clobber a previous forward's workspace)
    hdl.barrier(channel=0)

    # Direct UV memory scatter bypasses contiguous allocations and multiple splits/cats
    ext.all_to_all_transpose_cuda(
        x1,
        peer_ptrs.data_ptr(),
        world_size,
        rank,
        chunk_H,
        W,
        int(dim0_is_first),
        list(sizes),
        src_strides,
        dst_strides,
        x1.element_size()
    )

    # Synchronize ensuring all ranks are finished pushing to this chunk's layout
    hdl.barrier(channel=0)

    # 3. Transform the now-replicated second dimension.
    x2 = torch.fft.fft(out_buf, n=int(s[1]), dim=dim1, norm=norm)

    # 4. Keep the real-input half spectrum along the second transform dimension.
    return _truncate(x2, dim1, x2.shape[dim1] // 2 + 1)