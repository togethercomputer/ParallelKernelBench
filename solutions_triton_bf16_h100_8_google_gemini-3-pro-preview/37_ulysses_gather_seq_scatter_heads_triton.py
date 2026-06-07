"""
Strategy:
- **Device-Side Communication**: We replace `dist.all_to_all` and complex reshaping with a single custom fused JIT CUDA kernel. Peers expose their local input buffers via `torch.distributed._symmetric_memory`, allowing direct UVA peer-to-peer reads.
- **Compute-Communication Overlap**: While this operator doesn't contain independent compute to overlap with, the kernel completely bypasses the host by pulling directly from remote device pointers, overlapping memory transactions inherently across NVLink.
- **Coalesced Memory Access**: The Python wrapper dynamically coalesces contiguous tensor dimensions, and the CUDA kernel vectorizes memory copies up to 128-bit (`uint4`) where shapes align, maximizing bus utilization.
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
#include <vector>
#include <cstdint>

#define MAX_DIMS 8

struct PtrArray {
    const void* ptrs[16];
};

struct TensorDescriptor {
    int ndim;
    int shape[MAX_DIMS];
    int stride_x[MAX_DIMS];
    int stride_out[MAX_DIMS];
};

template <typename T>
__global__ void ulysses_gather_scatter_kernel(
    PtrArray x_ptrs,
    T* __restrict__ out,
    TensorDescriptor desc,
    int world_size,
    int rank,
    int chunk_s_stride_s_x_vec,
    int chunk_g_stride_g_out_vec,
    int num_elements_per_chunk
) {
    int p = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements_per_chunk) {
        int temp = tid;
        int offset_x = 0;
        int offset_out = 0;

        for (int i = desc.ndim - 1; i >= 0; --i) {
            int dim_size = desc.shape[i];
            int idx = temp % dim_size;
            temp /= dim_size;
            offset_x += idx * desc.stride_x[i];
            offset_out += idx * desc.stride_out[i];
        }

        offset_x += rank * chunk_s_stride_s_x_vec;
        offset_out += p * chunk_g_stride_g_out_vec;

        const T* x_peer = reinterpret_cast<const T*>(x_ptrs.ptrs[p]);
        out[offset_out] = x_peer[offset_x];
    }
}

void ulysses_gather_scatter_bf16(
    std::vector<int64_t> x_ptrs_int,
    torch::Tensor out,
    std::vector<int> shape,
    std::vector<int> stride_x,
    std::vector<int> stride_out,
    int world_size,
    int rank,
    int chunk_s,
    int chunk_g,
    int stride_s_x,
    int stride_g_out
) {
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");

    PtrArray ptrs;
    for (int i = 0; i < world_size; ++i) {
        ptrs.ptrs[i] = reinterpret_cast<const void*>(x_ptrs_int[i]);
    }

    int ndim = shape.size();
    TORCH_CHECK(ndim <= MAX_DIMS, "Too many dimensions");

    int num_elements = 1;
    for (int s : shape) num_elements *= s;

    int vec_size = 1;
    if (ndim > 0 && stride_x[ndim - 1] == 1 && stride_out[ndim - 1] == 1) {
        bool div8 = (shape[ndim - 1] % 8 == 0);
        if ((chunk_s * stride_s_x) % 8 != 0) div8 = false;
        if ((chunk_g * stride_g_out) % 8 != 0) div8 = false;
        for (int i = 0; i < ndim - 1; ++i) {
            if (stride_x[i] % 8 != 0 || stride_out[i] % 8 != 0) div8 = false;
        }
        if (div8 && (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0)) {
            bool all_aligned = true;
            for (int i = 0; i < world_size; ++i) {
                if (x_ptrs_int[i] % 16 != 0) all_aligned = false;
            }
            if (all_aligned) vec_size = 8;
        }

        if (vec_size == 1) {
            bool div4 = (shape[ndim - 1] % 4 == 0);
            if ((chunk_s * stride_s_x) % 4 != 0) div4 = false;
            if ((chunk_g * stride_g_out) % 4 != 0) div4 = false;
            for (int i = 0; i < ndim - 1; ++i) {
                if (stride_x[i] % 4 != 0 || stride_out[i] % 4 != 0) div4 = false;
            }
            if (div4 && (reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0)) {
                bool all_aligned = true;
                for (int i = 0; i < world_size; ++i) {
                    if (x_ptrs_int[i] % 8 != 0) all_aligned = false;
                }
                if (all_aligned) vec_size = 4;
            }
        }

        if (vec_size == 1) {
            bool div2 = (shape[ndim - 1] % 2 == 0);
            if ((chunk_s * stride_s_x) % 2 != 0) div2 = false;
            if ((chunk_g * stride_g_out) % 2 != 0) div2 = false;
            for (int i = 0; i < ndim - 1; ++i) {
                if (stride_x[i] % 2 != 0 || stride_out[i] % 2 != 0) div2 = false;
            }
            if (div2 && (reinterpret_cast<uintptr_t>(out.data_ptr()) % 4 == 0)) {
                bool all_aligned = true;
                for (int i = 0; i < world_size; ++i) {
                    if (x_ptrs_int[i] % 4 != 0) all_aligned = false;
                }
                if (all_aligned) vec_size = 2;
            }
        }
    }

    TensorDescriptor desc;
    desc.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        if (i == ndim - 1) {
            desc.shape[i] = shape[i] / vec_size;
            desc.stride_x[i] = 1;
            desc.stride_out[i] = 1;
        } else {
            desc.shape[i] = shape[i];
            desc.stride_x[i] = stride_x[i] / vec_size;
            desc.stride_out[i] = stride_out[i] / vec_size;
        }
    }

    int chunk_s_stride_s_x_vec = (chunk_s * stride_s_x) / vec_size;
    int chunk_g_stride_g_out_vec = (chunk_g * stride_g_out) / vec_size;
    int num_elements_pass = num_elements / vec_size;

    const int threads = 256;
    const int blocks = (num_elements_pass + threads - 1) / threads;
    dim3 grid(blocks, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (vec_size == 8) {
        ulysses_gather_scatter_kernel<uint4><<<grid, threads, 0, stream>>>(
            ptrs, reinterpret_cast<uint4*>(out.data_ptr()), desc,
            world_size, rank, chunk_s_stride_s_x_vec, chunk_g_stride_g_out_vec, num_elements_pass
        );
    } else if (vec_size == 4) {
        ulysses_gather_scatter_kernel<uint2><<<grid, threads, 0, stream>>>(
            ptrs, reinterpret_cast<uint2*>(out.data_ptr()), desc,
            world_size, rank, chunk_s_stride_s_x_vec, chunk_g_stride_g_out_vec, num_elements_pass
        );
    } else if (vec_size == 2) {
        ulysses_gather_scatter_kernel<uint32_t><<<grid, threads, 0, stream>>>(
            ptrs, reinterpret_cast<uint32_t*>(out.data_ptr()), desc,
            world_size, rank, chunk_s_stride_s_x_vec, chunk_g_stride_g_out_vec, num_elements_pass
        );
    } else {
        ulysses_gather_scatter_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            ptrs, reinterpret_cast<uint16_t*>(out.data_ptr()), desc,
            world_size, rank, chunk_s_stride_s_x_vec, chunk_g_stride_g_out_vec, num_elements_pass
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ulysses_gather_scatter_bf16", &ulysses_gather_scatter_bf16, "UVA all-to-all gather scatter bf16");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_gather_scatter_ext", CUDA_SRC)
    return _ext

def coalesce_dims(shape, stride1, stride2):
    new_shape = []
    new_s1 = []
    new_s2 = []
    for s, st1, st2 in zip(shape, stride1, stride2):
        if s > 1:
            new_shape.append(s)
            new_s1.append(st1)
            new_s2.append(st2)
    if not new_shape:
        return [1], [0], [0]

    res_shape = [new_shape[-1]]
    res_s1 = [new_s1[-1]]
    res_s2 = [new_s2[-1]]

    for i in range(len(new_shape) - 2, -1, -1):
        if new_s1[i] == res_shape[-1] * res_s1[-1] and new_s2[i] == res_shape[-1] * res_s2[-1]:
            res_shape[-1] *= new_shape[i]
        else:
            res_shape.append(new_shape[i])
            res_s1.append(new_s1[i])
            res_s2.append(new_s2[i])

    res_shape.reverse()
    res_s1.reverse()
    res_s2.reverse()
    return res_shape, res_s1, res_s2

_symm_cache = {}

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = (n, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache[key] = (buf, hdl)
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

    assert x.dtype == torch.bfloat16, "Optimized kernel expects bfloat16 precision"

    sp_world = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if rank == 0:
        _get_ext()
    dist.barrier(group)

    buf, hdl = _get_symm_state(x.numel(), x.dtype, x.device, group)
    
    # Pack input contiguously into symmetric memory view
    buf_view = buf.view(x.shape)
    buf_view.copy_(x)
    hdl.barrier(channel=0)

    out_shape = list(x.shape)
    out_shape[seq_dim] *= sp_world
    out_shape[head_dim] //= sp_world
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    S = list(x.shape)
    chunk_g = S[seq_dim]
    chunk_s = S[head_dim] // sp_world

    # The shape of the specific data block exchanged between peer pairs
    C_shape = list(x.shape)
    C_shape[head_dim] = chunk_s

    # Dynamic strides inside the continuous memory bounds
    x_stride = list(buf_view.stride())
    out_stride = list(out.stride())

    stride_s_x = x_stride[head_dim]
    stride_g_out = out_stride[seq_dim]

    c_shape_coalesced, s_x_coalesced, s_out_coalesced = coalesce_dims(C_shape, x_stride, out_stride)

    remote_ptrs = [int(ptr) for ptr in hdl.buffer_ptrs]

    _get_ext().ulysses_gather_scatter_bf16(
        remote_ptrs, out, c_shape_coalesced, s_x_coalesced, s_out_coalesced,
        sp_world, rank, chunk_s, chunk_g, stride_s_x, stride_g_out
    )

    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = out.size(seq_dim) - unpadded_dim_size
        slc = [slice(None)] * out.dim()
        slc[seq_dim] = slice(0, -padding_size)
        out = out[tuple(slc)]

    return out