"""
Strategy:
- **Device-Side Communication (Push via UVA):** Instead of NCCL collectives, we allocate the sequence parallel output buffer in symmetric memory. Using UVA pointers, each rank directly "pushes" its scatter chunks to the correctly calculated offset in remote peers' symmetric memory, bypassing host overhead and collective bottlenecks.
- **Compute-Communication Overlap:** The target peer chunks are pushed concurrently by launching the custom copy kernel onto `W` distinct CUDA streams. This parallelizes the NVLink writes, saturating the interconnect bandwidth.
- **Zero-Copy Reshape Fusion:** A custom 5D indexing CUDA kernel natively computes multi-dimensional strides to map the linear source chunk to the appropriate gathered position in the remote buffer. This eliminates the opaque sequence of `split`, `reshape`, `transpose`, and `cat` typically required in stock PyTorch All-to-All paths.
"""

import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void copy_5d_chunk_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t A, int64_t B, int64_t C,
    int64_t x_chunk_size, int64_t y_chunk_size,
    int64_t X_in, int64_t Y_in,
    int64_t X_out, int64_t Y_out,
    int64_t x_offset_in, int64_t y_offset_in,
    int64_t x_offset_out, int64_t y_offset_out,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int64_t temp = idx;
    int64_t c = temp % C;
    temp /= C;
    int64_t y_c = temp % y_chunk_size;
    temp /= y_chunk_size;
    int64_t b = temp % B;
    temp /= B;
    int64_t x_c = temp % x_chunk_size;
    int64_t a = temp / x_chunk_size;

    int64_t x_in_idx = x_offset_in + x_c;
    int64_t y_in_idx = y_offset_in + y_c;
    int64_t flat_in = ((((a * X_in) + x_in_idx) * B + b) * Y_in + y_in_idx) * C + c;

    int64_t x_out_idx = x_offset_out + x_c;
    int64_t y_out_idx = y_offset_out + y_c;
    int64_t flat_out = ((((a * X_out) + x_out_idx) * B + b) * Y_out + y_out_idx) * C + c;

    dst[flat_out] = src[flat_in];
}

void uva_push_5d_bf16(
    torch::Tensor src,
    int64_t dst_ptr,
    int64_t A, int64_t B, int64_t C,
    int64_t x_chunk_size, int64_t y_chunk_size,
    int64_t X_in, int64_t Y_in,
    int64_t X_out, int64_t Y_out,
    int64_t x_offset_in, int64_t y_offset_in,
    int64_t x_offset_out, int64_t y_offset_out,
    int64_t stream_int
) {
    int64_t numel = A * x_chunk_size * B * y_chunk_size * C;
    if (numel == 0) return;

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);

    __nv_bfloat16* dst_data = reinterpret_cast<__nv_bfloat16*>(dst_ptr);
    const __nv_bfloat16* src_data = reinterpret_cast<const __nv_bfloat16*>(src.data_ptr());

    copy_5d_chunk_kernel<<<blocks, threads, 0, stream>>>(
        src_data, dst_data,
        A, B, C,
        x_chunk_size, y_chunk_size,
        X_in, Y_in,
        X_out, Y_out,
        x_offset_in, y_offset_in,
        x_offset_out, y_offset_out,
        numel
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_push_5d_bf16", &uva_push_5d_bf16, "UVA 5D chunk copy for bf16 push");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_push_5d_bf16_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device, group):
    global _symm_cache
    key = (n, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
        
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl

_streams = None

def _get_streams(W: int):
    global _streams
    if _streams is None or len(_streams) < W:
        _streams = [torch.cuda.Stream() for _ in range(W)]
    return _streams[:W]

def _pad_tensor(x: torch.Tensor, dim: int, padding_size: int, padding_value: int = 0) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.full(shape, padding_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)

def compute_5d_args(shape_in, shape_out, scatter_dim, gather_dim, p, rank, W):
    min_dim = min(scatter_dim, gather_dim)
    max_dim = max(scatter_dim, gather_dim)
    
    A = math.prod(shape_in[:min_dim]) if min_dim > 0 else 1
    X_in = shape_in[min_dim]
    B = math.prod(shape_in[min_dim+1:max_dim]) if max_dim > min_dim + 1 else 1
    Y_in = shape_in[max_dim] if max_dim > min_dim else 1
    C = math.prod(shape_in[max_dim+1:]) if max_dim + 1 < len(shape_in) else 1
    
    X_out = shape_out[min_dim]
    Y_out = shape_out[max_dim] if max_dim > min_dim else 1
    
    if scatter_dim < gather_dim:
        x_chunk_size = X_in // W
        y_chunk_size = Y_in
        x_offset_in = p * x_chunk_size
        y_offset_in = 0
        x_offset_out = 0
        y_offset_out = rank * y_chunk_size
    elif scatter_dim > gather_dim:
        x_chunk_size = X_in
        y_chunk_size = Y_in // W
        x_offset_in = 0
        y_offset_in = p * y_chunk_size
        x_offset_out = rank * x_chunk_size
        y_offset_out = 0
    else:
        x_chunk_size = X_in // W
        y_chunk_size = 1
        x_offset_in = p * x_chunk_size
        y_offset_in = 0
        x_offset_out = rank * x_chunk_size
        y_offset_out = 0
        
    return (A, B, C, x_chunk_size, y_chunk_size, 
            X_in, Y_in, X_out, Y_out, 
            x_offset_in, y_offset_in, x_offset_out, y_offset_out)


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
        return x.contiguous()
        
    dim_size = x.size(seq_dim)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
        
    x = x.contiguous()
    assert x.dtype == torch.bfloat16, "Custom push kernel expects BF16 input."
    
    out_shape = list(x.shape)
    out_shape[seq_dim] //= sp_world
    if seq_dim != head_dim:
        out_shape[head_dim] *= sp_world
    else:
        out_shape[seq_dim] *= sp_world
    out_shape = tuple(out_shape)
    
    numel = math.prod(out_shape)
    
    rank = dist.get_rank(group)
    ext = _get_ext()
    streams = _get_streams(sp_world)
    
    buf, hdl = _get_symm_state(numel, x.dtype, x.device, group)
    out = buf.view(*out_shape)
    
    # Ensure all previous usages of symmetric memory are clear
    hdl.barrier(channel=0)
    
    # Scatter (Push) over distinct parallel streams mapping to peers
    for p in range(sp_world):
        with torch.cuda.stream(streams[p]):
            args = compute_5d_args(x.shape, out.shape, seq_dim, head_dim, p, rank, sp_world)
            if p == rank:
                dst_ptr = out.data_ptr()
            else:
                dst_ptr = int(hdl.buffer_ptrs[p])
            
            ext.uva_push_5d_bf16(
                x, dst_ptr, *args, streams[p].cuda_stream
            )
            
    # Wait for all chunks to be effectively written via parallel streams
    for p in range(sp_world):
        torch.cuda.current_stream().wait_stream(streams[p])
        
    # Synchronize guarantees all peers finished writing into our outbound buffer
    hdl.barrier(channel=0)
    
    return out