"""
Strategy:
1. Replaced NCCL `all_to_all_single` and PyTorch `cat`/`reshape` ops with a direct NVLink PULL kernel using Symmetric Memory.
2. Flattened the complex N-dimensional data routing into a 5D logical tensor offset calculation directly inside the CUDA kernel, eliminating multi-step memory traffic (no intermediate splits, transposes, or concats).
3. The sender simply copies its local data to its symmetric buffer (1 contiguous write), and the receiver directly pulls its required scattered slices into the correct gathered layout (1 NVLink read, 1 contiguous write). This achieves fewer memory ops than the reference all-to-all.
4. Used `uint4` (128-bit) vectorized loads/stores on the inner-most dimension (typically `head_dim` size, highly divisible by 8) for maximum P2P bandwidth utilization.
5. Employs double-buffering for symmetric memory allocations to eliminate read-after-write hazards across consecutive calls without blocking the host stream.
"""

import math
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
#include <cstdint>

template <int VEC_SIZE>
__global__ void ulysses_pull_kernel(
    const uint64_t* __restrict__ peer_ptrs,
    void* __restrict__ out_ptr,
    int64_t A, int64_t B, int64_t C, int64_t D, int64_t E_vec,
    int P, int my_rank, bool scatter_first, int64_t N_out_vec
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_out_vec) return;

    int64_t e = idx % E_vec;
    int64_t tmp = idx / E_vec;

    int64_t rank_j;
    int64_t in_idx;
    
    if (scatter_first) {
        // B is scatter (head_dim), D is gather (seq_dim)
        // Output shape: [A, B/P, C, D*P, E_vec]
        int64_t d_out = tmp % (D * P);
        tmp = tmp / (D * P);
        int64_t c = tmp % C;
        tmp = tmp / C;
        int64_t b_out = tmp % (B / P);
        int64_t a = tmp / (B / P);

        rank_j = d_out / D;
        int64_t d_in = d_out % D;
        int64_t b_in = my_rank * (B / P) + b_out;

        in_idx = (((a * B + b_in) * C + c) * D + d_in) * E_vec + e;
    } else {
        // B is gather (seq_dim), D is scatter (head_dim)
        // Output shape: [A, B*P, C, D/P, E_vec]
        int64_t d_out = tmp % (D / P);
        tmp = tmp / (D / P);
        int64_t c = tmp % C;
        tmp = tmp / C;
        int64_t b_out = tmp % (B * P);
        int64_t a = tmp / (B * P);

        rank_j = b_out / B;
        int64_t b_in = b_out % B;
        int64_t d_in = my_rank * (D / P) + d_out;

        in_idx = (((a * B + b_in) * C + c) * D + d_in) * E_vec + e;
    }

    if constexpr (VEC_SIZE == 8) {
        const uint4* src = reinterpret_cast<const uint4*>(peer_ptrs[rank_j]);
        uint4* out = reinterpret_cast<uint4*>(out_ptr);
        out[idx] = src[in_idx];
    } else if constexpr (VEC_SIZE == 4) {
        const uint2* src = reinterpret_cast<const uint2*>(peer_ptrs[rank_j]);
        uint2* out = reinterpret_cast<uint2*>(out_ptr);
        out[idx] = src[in_idx];
    } else if constexpr (VEC_SIZE == 2) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(peer_ptrs[rank_j]);
        uint32_t* out = reinterpret_cast<uint32_t*>(out_ptr);
        out[idx] = src[in_idx];
    } else {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(peer_ptrs[rank_j]);
        uint16_t* out = reinterpret_cast<uint16_t*>(out_ptr);
        out[idx] = src[in_idx];
    }
}

void launch_ulysses_pull(
    torch::Tensor peer_ptrs,
    torch::Tensor out,
    int64_t A, int64_t B, int64_t C, int64_t D, int64_t E,
    int P, int my_rank, bool scatter_first
) {
    int64_t N_out = A * B * C * D * E;

    int vec_size = 1;
    if (E % 8 == 0) vec_size = 8;
    else if (E % 4 == 0) vec_size = 4;
    else if (E % 2 == 0) vec_size = 2;

    int64_t E_vec = E / vec_size;
    int64_t N_out_vec = N_out / vec_size;

    int threads = 256;
    int blocks = (N_out_vec + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint64_t* ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>());
    void* out_ptr = out.data_ptr();

    if (vec_size == 8) {
        ulysses_pull_kernel<8><<<blocks, threads, 0, stream>>>(
            ptrs, out_ptr, A, B, C, D, E_vec, P, my_rank, scatter_first, N_out_vec);
    } else if (vec_size == 4) {
        ulysses_pull_kernel<4><<<blocks, threads, 0, stream>>>(
            ptrs, out_ptr, A, B, C, D, E_vec, P, my_rank, scatter_first, N_out_vec);
    } else if (vec_size == 2) {
        ulysses_pull_kernel<2><<<blocks, threads, 0, stream>>>(
            ptrs, out_ptr, A, B, C, D, E_vec, P, my_rank, scatter_first, N_out_vec);
    } else {
        ulysses_pull_kernel<1><<<blocks, threads, 0, stream>>>(
            ptrs, out_ptr, A, B, C, D, E_vec, P, my_rank, scatter_first, N_out_vec);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_ulysses_pull", &launch_ulysses_pull, "Ulysses NVLink Pull Kernel");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_pull_ext", CUDA_SRC)
    return _ext

_step_counter = {}
_symm_cache = {}

def _power_of_2(n):
    if n <= 0: return 0
    return 1 << (n - 1).bit_length()

def _get_symm_buffer(numel: int, dtype: torch.dtype, device: torch.device, group: ProcessGroup):
    global _step_counter
    group_id = id(group)
    step = _step_counter.get(group_id, 0)
    _step_counter[group_id] = step + 1
    buf_idx = step % 2  # Double-buffering prevents Read-After-Write hazards without host blocking

    best_key = None
    for k in _symm_cache:
        k_numel, k_dtype, k_device, k_group, k_idx = k
        if k_dtype == dtype and k_device == device and k_group == group and k_idx == buf_idx:
            if k_numel >= numel:
                if best_key is None or k_numel < best_key[0]:
                    best_key = k
                    
    if best_key is not None:
        buf, hdl, ptrs = _symm_cache[best_key]
        return buf[:numel].view(-1), hdl, ptrs

    alloc_numel = max(_power_of_2(numel), 1024 * 1024) 
    buf = symm_mem.empty(alloc_numel, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache[(alloc_numel, dtype, device, group, buf_idx)] = (buf, hdl, ptrs)
    return buf[:numel].view(-1), hdl, ptrs


@torch.no_grad()
def solution(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: int = 0,
) -> torch.Tensor:
    if group is None or not dist.is_initialized():
        return x

    sp_world = dist.get_world_size(group)
    my_rank = dist.get_rank(group)

    if my_rank == 0:
        _get_ext()
    dist.barrier(group=group)
    ext = _get_ext()

    scatter_dim = head_dim
    gather_dim = seq_dim
    dims = list(x.shape)
    
    # Pre-calculate 5D flattening constants based on dim order
    if scatter_dim < gather_dim:
        scatter_first = True
        A = math.prod(dims[:scatter_dim])
        B = dims[scatter_dim]
        C = math.prod(dims[scatter_dim+1:gather_dim])
        D = dims[gather_dim]
        E = math.prod(dims[gather_dim+1:])
    else:
        scatter_first = False
        A = math.prod(dims[:gather_dim])
        B = dims[gather_dim]
        C = math.prod(dims[gather_dim+1:scatter_dim])
        D = dims[scatter_dim]
        E = math.prod(dims[scatter_dim+1:])

    # Prepare symmetric buffer
    x_contig = x.contiguous()
    numel = x_contig.numel()
    symm_buf, hdl, ptrs = _get_symm_buffer(numel, x_contig.dtype, x_contig.device, group)
    
    # Local contiguous write followed by symmetric memory stream barrier
    symm_buf.copy_(x_contig.view(-1))
    hdl.barrier(channel=0)

    # Allocate local output tensor explicitly
    out_shape = list(x.shape)
    out_shape[seq_dim] *= sp_world
    out_shape[head_dim] //= sp_world
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # Dispatch custom PULL NVLink kernel
    ext.launch_ulysses_pull(
        ptrs, out, A, B, C, D, E, sp_world, my_rank, scatter_first
    )

    # Clean unpadding natively (acts on memory views seamlessly)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = out.size(seq_dim) - unpadded_dim_size
        slc = [slice(None)] * out.dim()
        slc[seq_dim] = slice(0, -padding_size)
        out = out[tuple(slc)]

    return out