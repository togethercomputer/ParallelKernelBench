"""
Strategy:
1. **Device-Side Communication (UVA)**: We fuse the padding/conjugation (`_conj_pad_2d`) and the all-to-all transpose (`_all_to_all_transpose`) into two custom CUDA C++ kernels. They read directly from peer memory using `torch.distributed._symmetric_memory` (UVA), entirely bypassing NCCL and host-driven chunking overhead.
2. **Compute-Communication Overlap**: Memory transfers over NVLink are tightly coupled with the physical data reformatting (conjugate flips and transpositions). The transpose kernel reads directly from the remote FFT output buffers in symmetric memory, perfectly overlapping memory loads with the global scatter/gather logic.
3. **Zero-Copy Re-use**: We allocate a single contiguous symmetric byte buffer sized to hold both the initial shard and the intermediate complex spectrum. This avoids multiple rendezvous delays and prevents buffer contention, permitting high-throughput pipelined execution.
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
#include <c10/util/complex.h>
#include <vector>

struct PtrArray8 {
    const void* ptrs[8];
};

__global__ void conj_pad_2d_kernel(
    PtrArray8 symm_x_ptrs,
    c10::complex<float>* __restrict__ my_x_pad,
    int B, int N0_local, int N1_half, int N1_full, int N0_full, int rank
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i_local = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (j >= N1_full || i_local >= N0_local || b >= B) return;

    int64_t out_idx = (int64_t)b * N0_local * N1_full + i_local * N1_full + j;
    const c10::complex<float>* src_ptrs[8];
    #pragma unroll
    for(int i=0; i<8; ++i) {
        src_ptrs[i] = (const c10::complex<float>*)symm_x_ptrs.ptrs[i];
    }

    if (j < N1_half) {
        my_x_pad[out_idx] = src_ptrs[rank][(int64_t)b * N0_local * N1_half + i_local * N1_half + j];
    } else {
        int flipped_j = N1_full - j;
        int i_global = rank * N0_local + i_local;
        int flipped_i_global = (i_global == 0) ? 0 : (N0_full - i_global);

        int src_r = flipped_i_global / N0_local;
        int src_i_local = flipped_i_global % N0_local;

        c10::complex<float> val = src_ptrs[src_r][(int64_t)b * N0_local * N1_half + src_i_local * N1_half + flipped_j];
        my_x_pad[out_idx] = c10::complex<float>(val.real(), -val.imag());
    }
}

__global__ void all_to_all_transpose_kernel(
    PtrArray8 symm_x1_ptrs,
    c10::complex<float>* __restrict__ my_x1_tran,
    int B, int N0_local, int N1_full, int N1_local, int N0_full, int rank, int world_size
) {
    int j_local = blockIdx.x * blockDim.x + threadIdx.x;
    int i_local = blockIdx.y * blockDim.y + threadIdx.y;
    int bp = blockIdx.z * blockDim.z + threadIdx.z;

    int p = bp % world_size;
    int b = bp / world_size;

    if (j_local >= N1_local || i_local >= N0_local || b >= B) return;

    int64_t out_idx = (int64_t)b * N0_full * N1_local + (p * N0_local + i_local) * N1_local + j_local;
    int64_t in_idx = (int64_t)b * N0_local * N1_full + i_local * N1_full + (rank * N1_local + j_local);

    const c10::complex<float>* src_p = (const c10::complex<float>*)symm_x1_ptrs.ptrs[p];
    my_x1_tran[out_idx] = src_p[in_idx];
}

void conj_pad_2d_cuda(
    std::vector<int64_t> symm_ptrs,
    torch::Tensor my_x_pad,
    int B, int N0_local, int N1_half, int N1_full, int N0_full, int rank
) {
    PtrArray8 ptrs_struct;
    for(size_t i=0; i<symm_ptrs.size() && i<8; ++i) {
        ptrs_struct.ptrs[i] = (const void*)symm_ptrs[i];
    }
    
    dim3 threads(32, 8, 1);
    dim3 blocks((N1_full + 31)/32, (N0_local + 7)/8, B);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    conj_pad_2d_kernel<<<blocks, threads, 0, stream>>>(
        ptrs_struct,
        (c10::complex<float>*)my_x_pad.data_ptr(),
        B, N0_local, N1_half, N1_full, N0_full, rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void all_to_all_transpose_cuda(
    std::vector<int64_t> symm_ptrs,
    torch::Tensor my_x1_tran,
    int B, int N0_local, int N1_full, int N1_local, int N0_full, int rank, int world_size
) {
    PtrArray8 ptrs_struct;
    for(size_t i=0; i<symm_ptrs.size() && i<8; ++i) {
        ptrs_struct.ptrs[i] = (const void*)symm_ptrs[i];
    }

    dim3 threads(32, 8, 1);
    dim3 blocks((N1_local + 31)/32, (N0_local + 7)/8, B * world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    all_to_all_transpose_kernel<<<blocks, threads, 0, stream>>>(
        ptrs_struct,
        (c10::complex<float>*)my_x1_tran.data_ptr(),
        B, N0_local, N1_full, N1_local, N0_full, rank, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conj_pad_2d_cuda", &conj_pad_2d_cuda, "Conj Pad 2D CUDA");
    m.def("all_to_all_transpose_cuda", &all_to_all_transpose_cuda, "All to All Transpose CUDA");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("physicsnemo_irfft_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(max_bytes: int, group: dist.ProcessGroup, device: torch.device):
    global _symm_cache
    if group not in _symm_cache or _symm_cache[group]['size'] < max_bytes:
        buf = symm_mem.empty(max_bytes, dtype=torch.uint8, device=device)
        hdl = symm_mem.rendezvous(buf, group)
        _symm_cache[group] = {'size': max_bytes, 'buf': buf, 'hdl': hdl}
    return _symm_cache[group]['buf'], _symm_cache[group]['hdl']

@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Optional[Sequence[int]],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    assert world_size <= 8, "Fast custom symmetric kernel path assumes world_size <= 8"

    x_in = x if x.dtype == torch.complex64 else x.to(torch.complex64)
    ndim = x_in.ndim
    
    dim0, dim1 = int(dim[0]), int(dim[1])
    dim0 = dim0 if dim0 >= 0 else dim0 + ndim
    dim1 = dim1 if dim1 >= 0 else dim1 + ndim

    if s is not None:
        first_dim_size = int(s[0])
        last_dim_size = int(s[1])
    else:
        first_dim_size = int(x_in.shape[dim0])
        last_dim_size = int(2 * (x_in.shape[dim1] - 1))

    # Move active dimensions to the end for the 3D-oriented custom kernel
    dims = list(range(ndim))
    dims.remove(dim0)
    dims.remove(dim1)
    perm = dims + [dim0, dim1]
    
    x_perm = x_in.permute(perm).contiguous()
    
    B = math.prod(x_perm.shape[:-2]) if ndim > 2 else 1
    N0_local = x_perm.shape[-2]
    N1_half = x_perm.shape[-1]
    
    x_3d = x_perm.view(B, N0_local, N1_half)

    N0_full = N0_local * world_size
    N1_full = last_dim_size
    N1_local = N1_full // world_size

    x_bytes = x_3d.numel() * 8
    x1_bytes = B * N0_local * N1_full * 8
    total_bytes = x_bytes + x1_bytes

    buf, hdl = _get_symm_state(total_bytes, group, x_in.device)

    # 1. Provide input memory directly via symmetric UVA mapping
    buf_x = buf[:x_bytes].view(torch.complex64)
    buf_x[:x_3d.numel()].copy_(x_3d.flatten())
    torch.cuda.current_stream().synchronize()
    hdl.barrier(channel=0)

    symm_x_ptrs = [int(p) for p in hdl.buffer_ptrs]

    x_pad = torch.empty(B, N0_local, N1_full, dtype=torch.complex64, device=x_in.device)
    _get_ext().conj_pad_2d_cuda(symm_x_ptrs, x_pad, B, N0_local, N1_half, N1_full, N0_full, rank)

    # 2. First FFT, written directly to the offset segment inside the combined symmetric buffer
    buf_x1 = buf[x_bytes : x_bytes + x1_bytes].view(torch.complex64)
    x1_symm_view = buf_x1.view(B, N0_local, N1_full)
    torch.fft.ifft(x_pad, n=N1_full, dim=-1, norm=norm, out=x1_symm_view)
    
    torch.cuda.current_stream().synchronize()
    hdl.barrier(channel=1)

    symm_x1_ptrs = [int(p) + x_bytes for p in hdl.buffer_ptrs]

    # 3. Transpose via direct multi-rank pulls 
    x1_tran = torch.empty(B, N0_full, N1_local, dtype=torch.complex64, device=x_in.device)
    _get_ext().all_to_all_transpose_cuda(symm_x1_ptrs, x1_tran, B, N0_local, N1_full, N1_local, N0_full, rank, world_size)

    # 4. Final transform and real chunk extraction
    x2 = torch.fft.ifft(x1_tran, n=first_dim_size, dim=-2, norm=norm)
    out_3d = torch.real(x2)

    # Un-permute back to the user's original dimensions
    out_shape_perm = list(x_perm.shape[:-2]) + [first_dim_size, N1_local]
    out_perm = out_3d.view(*out_shape_perm)

    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    out = out_perm.permute(inv_perm).contiguous()
    return out