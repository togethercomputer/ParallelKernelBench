"""
Strategy:
1. **Device-Side Communication**: We replace NCCL's host-driven `all_to_all` and PyTorch's multiple `split`/`cat`/`reshape` operations with a single, fused custom CUDA pull kernel.
2. **Symmetric Memory (UVA)**: The local input sequence is staged into a cached `symm_mem` buffer. All ranks then use UVA peer pointers to read their required chunks directly from the peers' symmetric buffers.
3. **Compute-Communication Overlap & Fusion**: Sequence padding is fused directly into the staging copy step, avoiding extra allocations. The pull kernel resolves the scatter-gather multi-dimensional routing on-the-fly and writes the final contiguous output.
4. **Bandwidth Optimization**: The kernel detects the innermost contiguous dimension (`E`) and automatically vectorizes loads/stores (up to 128-bit) to saturate NVLink bandwidth.
"""

from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.distributed._symmetric_memory as symm_mem

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

template <int VEC>
__global__ void ulysses_pull_kernel(
    const __nv_bfloat16* const* __restrict__ peer_ptrs,
    __nv_bfloat16* __restrict__ out,
    int rank,
    int world_size,
    int64_t A,
    int64_t B,
    int64_t C,
    int64_t D,
    int64_t E,
    bool seq_first,
    int64_t numel_vec
) {
    int64_t vec_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= numel_vec) return;
    
    int64_t idx = vec_idx * VEC;
    
    int64_t a, b_out, c, d_out, e;
    int64_t temp = idx;
    
    e = temp % E; temp /= E;
    
    int64_t D_out = seq_first ? (D * world_size) : (D / world_size);
    d_out = temp % D_out; temp /= D_out;
    
    c = temp % C; temp /= C;
    
    int64_t B_out = seq_first ? (B / world_size) : (B * world_size);
    b_out = temp % B_out; temp /= B_out;
    
    a = temp;
    
    int p;
    int64_t b_in, d_in;
    
    if (seq_first) {
        // B is seq_dim, D is head_dim
        p = d_out / D;
        d_in = d_out % D;
        b_in = rank * B_out + b_out;
    } else {
        // B is head_dim, D is seq_dim
        p = b_out / B;
        b_in = b_out % B;
        d_in = rank * D_out + d_out;
    }
    
    int64_t in_idx = a * (B * C * D * E) + b_in * (C * D * E) + c * (D * E) + d_in * E + e;
    const __nv_bfloat16* src_ptr = peer_ptrs[p];
    
    if constexpr (VEC == 8) {
        *reinterpret_cast<uint4*>(&out[idx]) = *reinterpret_cast<const uint4*>(&src_ptr[in_idx]);
    } else if constexpr (VEC == 4) {
        *reinterpret_cast<uint2*>(&out[idx]) = *reinterpret_cast<const uint2*>(&src_ptr[in_idx]);
    } else if constexpr (VEC == 2) {
        *reinterpret_cast<uint32_t*>(&out[idx]) = *reinterpret_cast<const uint32_t*>(&src_ptr[in_idx]);
    } else {
        out[idx] = src_ptr[in_idx];
    }
}

void launch_ulysses_pull(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor out,
    int rank,
    int world_size,
    int64_t A,
    int64_t B,
    int64_t C,
    int64_t D,
    int64_t E,
    bool seq_first,
    int64_t numel
) {
    const __nv_bfloat16* const* peer_ptrs = (const __nv_bfloat16* const*)peer_ptrs_tensor.data_ptr<int64_t>();
    __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    
    if (E % 8 == 0) {
        int blocks = (numel / 8 + threads - 1) / threads;
        ulysses_pull_kernel<8><<<blocks, threads, 0, stream>>>(
            peer_ptrs, out_ptr, rank, world_size, A, B, C, D, E, seq_first, numel / 8);
    } else if (E % 4 == 0) {
        int blocks = (numel / 4 + threads - 1) / threads;
        ulysses_pull_kernel<4><<<blocks, threads, 0, stream>>>(
            peer_ptrs, out_ptr, rank, world_size, A, B, C, D, E, seq_first, numel / 4);
    } else if (E % 2 == 0) {
        int blocks = (numel / 2 + threads - 1) / threads;
        ulysses_pull_kernel<2><<<blocks, threads, 0, stream>>>(
            peer_ptrs, out_ptr, rank, world_size, A, B, C, D, E, seq_first, numel / 2);
    } else {
        int blocks = (numel + threads - 1) / threads;
        ulysses_pull_kernel<1><<<blocks, threads, 0, stream>>>(
            peer_ptrs, out_ptr, rank, world_size, A, B, C, D, E, seq_first, numel);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_ulysses_pull", &launch_ulysses_pull, "Ulysses gather-scatter pull kernel");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        from utils.cuda_helpers import compile_cuda_extension
        _ext = compile_cuda_extension("ulysses_gather_scatter_bf16", CUDA_SRC)
    return _ext


_symm_cache = {}
def _get_symm_state(shape_tuple, dtype, device, group):
    key = (shape_tuple, dtype, device, group)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(shape_tuple, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group=group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    res = (buf, hdl, ptrs_tensor)
    _symm_cache[key] = res
    return res


@torch.no_grad()
def solution(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    if group is None or dist.get_world_size(group) <= 1:
        return x

    seq_dim = seq_dim % x.ndim
    head_dim = head_dim % x.ndim
    sp_world = dist.get_world_size(group)
    dim_size = x.size(seq_dim)

    # Fallback to stock PyTorch for unsupported datatypes
    if x.dtype != torch.bfloat16:
        if dim_size % sp_world != 0:
            padding_size = sp_world - (dim_size % sp_world)
            pad_shape = list(x.shape)
            pad_shape[seq_dim] = padding_size
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=seq_dim)
        input_list = [t.contiguous() for t in torch.tensor_split(x, sp_world, seq_dim)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(sp_world)]
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=head_dim).contiguous()

    # JIT Compile
    if _ext is None:
        if dist.get_rank(group) == 0:
            _get_ext()
        dist.barrier(group)
        _get_ext()

    # 1. Determine padded shape
    padded_shape = list(x.shape)
    needs_padding = (dim_size % sp_world != 0)
    if needs_padding:
        padded_shape[seq_dim] += sp_world - (dim_size % sp_world)

    # 2. Grab symm_mem staging buffers from cache
    buf, hdl, ptrs_tensor = _get_symm_state(tuple(padded_shape), x.dtype, x.device, group)
    
    # 3. Synchronize stream before overwriting staging memory
    hdl.barrier(channel=0)
    
    # 4. Copy current input to symm_mem buffer (fuse padding logic here)
    if needs_padding:
        slices = [slice(None)] * x.ndim
        slices[seq_dim] = slice(0, dim_size)
        buf[tuple(slices)].copy_(x)
        
        slices_pad = [slice(None)] * x.ndim
        slices_pad[seq_dim] = slice(dim_size, None)
        buf[tuple(slices_pad)].zero_()
    else:
        buf.copy_(x)
        
    # 5. Synchronize stream to ensure all peers have finished writing
    hdl.barrier(channel=1)
    
    # 6. Allocate independent contiguous output
    out_shape = list(padded_shape)
    out_shape[seq_dim] //= sp_world
    out_shape[head_dim] *= sp_world
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # 7. Collapse shapes down to 5D for exact mapping
    A, B, C, D, E = 1, 1, 1, 1, 1
    dim1 = min(seq_dim, head_dim)
    dim2 = max(seq_dim, head_dim)
    
    for i in range(dim1): A *= padded_shape[i]
    B = padded_shape[dim1]
    for i in range(dim1 + 1, dim2): C *= padded_shape[i]
    D = padded_shape[dim2]
    for i in range(dim2 + 1, len(padded_shape)): E *= padded_shape[i]
    
    seq_first = (seq_dim < head_dim)
    
    # 8. Pull from peer UVA pointers directly to local output
    _get_ext().launch_ulysses_pull(
        ptrs_tensor,
        out,
        dist.get_rank(group),
        sp_world,
        A, B, C, D, E,
        seq_first,
        out.numel()
    )
    
    return out