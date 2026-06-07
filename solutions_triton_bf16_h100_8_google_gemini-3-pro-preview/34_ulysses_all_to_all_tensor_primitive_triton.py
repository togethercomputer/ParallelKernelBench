"""
Strategy:
1. Device-Side Communication via UVA: We bypass the NCCL overhead by directly writing (pushing) over NVLink into remote peers' symmetric memory buffers (`ulysses_push`), eliminating host-side collective launches.
2. Fused Reshape/Concatenation: The multidimensional chunking (scatter) and concatenation (gather) are fused directly into the communication kernel. Threads compute the exact input and output vector offsets, bypassing PyTorch `tensor_split` and `cat`.
3. Compute-Communication Overlap: Multi-dimensional indexing is simplified to purely 32-bit math and hidden by NVLink latencies. Double buffering of symmetric memory allows pipelined operations across successive calls.
4. Maximum Bandwidth: The kernel dynamically identifies the innermost contiguous dimension and automatically vectorizes loads/stores (up to 128-bit uint4), maximizing interconnect throughput.
"""

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename T>
__global__ void ulysses_push_kernel(
    const T* __restrict__ input,
    const void* const* __restrict__ dst_ptrs,
    int W, int src_rank,
    uint32_t A, uint32_t S1, uint32_t B, uint32_t S2, uint32_t C,
    bool s_less_than_g
) {
    int dst_rank = blockIdx.y;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t chunk_numel = A * S1 * B * S2 * C;

    if (idx < chunk_numel) {
        // purely 32-bit arithmetic for multi-dimensional index
        uint32_t c = idx % C;
        uint32_t temp = idx / C;
        uint32_t i2 = temp % S2;
        temp = temp / S2;
        uint32_t b = temp % B;
        temp = temp / B;
        uint32_t i1 = temp % S1;
        uint32_t a = temp / S1;

        // 64-bit flat index calculation to prevent offset overflow
        uint64_t in_idx, out_idx;
        if (s_less_than_g) {
            in_idx = ((((uint64_t)a * (W * S1) + (i1 + dst_rank * S1)) * B + b) * S2 + i2) * C + c;
            out_idx = ((((uint64_t)a * S1 + i1) * B + b) * (W * S2) + (i2 + src_rank * S2)) * C + c;
        } else {
            in_idx = ((((uint64_t)a * S1 + i1) * B + b) * (W * S2) + (i2 + dst_rank * S2)) * C + c;
            out_idx = ((((uint64_t)a * (W * S1) + (i1 + src_rank * S1)) * B + b) * S2 + i2) * C + c;
        }

        T* dst_ptr = reinterpret_cast<T*>(const_cast<void*>(dst_ptrs[dst_rank]));
        dst_ptr[out_idx] = input[in_idx];
    }
}

void ulysses_push(
    torch::Tensor input,
    torch::Tensor dst_ptrs_tensor,
    int W, int src_rank,
    uint32_t A, uint32_t S1, uint32_t B, uint32_t S2, uint32_t C,
    bool s_less_than_g,
    int vec_size
) {
    uint32_t chunk_numel = A * S1 * B * S2 * C;
    int threads = 256;
    int blocks_x = (chunk_numel + threads - 1) / threads;
    dim3 blocks(blocks_x, W, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const void* const* dst_ptrs = reinterpret_cast<const void* const*>(dst_ptrs_tensor.data_ptr<int64_t>());

    if (vec_size == 8) {
        ulysses_push_kernel<uint4><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(input.data_ptr()),
            dst_ptrs, W, src_rank, A, S1, B, S2, C, s_less_than_g
        );
    } else if (vec_size == 4) {
        ulysses_push_kernel<uint2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint2*>(input.data_ptr()),
            dst_ptrs, W, src_rank, A, S1, B, S2, C, s_less_than_g
        );
    } else if (vec_size == 2) {
        ulysses_push_kernel<uint32_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(input.data_ptr()),
            dst_ptrs, W, src_rank, A, S1, B, S2, C, s_less_than_g
        );
    } else {
        ulysses_push_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(input.data_ptr()),
            dst_ptrs, W, src_rank, A, S1, B, S2, C, s_less_than_g
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ulysses_push", &ulysses_push, "Ulysses All-to-All Push Kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_alltoall_uva_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
_symm_idx = 0

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_idx
    key = (n, dtype, device, group)
    if key not in _symm_cache:
        bufs = []
        hdls = []
        ptrs = []
        # Allocate a double buffer to safely interleave computation and communication
        for _ in range(2):
            b = symm_mem.empty(n, device=device, dtype=dtype)
            h = symm_mem.rendezvous(b, group)
            bufs.append(b)
            hdls.append(h)
            # Cache the tensor containing device pointers locally
            ptrs.append(torch.tensor(h.buffer_ptrs, dtype=torch.int64, device=device))
        _symm_cache[key] = (bufs, hdls, ptrs)
    
    bufs, hdls, ptrs = _symm_cache[key]
    idx = _symm_idx % 2
    _symm_idx += 1
    return bufs[idx], hdls[idx], ptrs[idx]

@torch.no_grad()
def solution(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    assert x.element_size() == 2, "Custom kernel optimized for 16-bit precisions (e.g. bfloat16, float16)"

    rank = dist.get_rank(group)
    x = x.contiguous()
    
    if rank == 0:
        _get_ext()
    dist.barrier(group)

    shape = list(x.shape)
    shape_s = shape[scatter_dim] // world_size
    shape_g = shape[gather_dim]
    
    dim1 = min(scatter_dim, gather_dim)
    dim2 = max(scatter_dim, gather_dim)
    
    # 5-Segment decomposition of the tensor dimensions
    A = math.prod(shape[:dim1]) if dim1 > 0 else 1
    S1 = shape_s if dim1 == scatter_dim else shape_g
    B = math.prod(shape[dim1+1:dim2]) if dim2 > dim1 + 1 else 1
    S2 = shape_g if dim2 == gather_dim else shape_s
    C = math.prod(shape[dim2+1:]) if dim2 + 1 < len(shape) else 1
    
    out_shape = list(shape)
    out_shape[scatter_dim] = shape_s
    out_shape[gather_dim] = shape_g * world_size
    out_numel = math.prod(out_shape)
    
    buf, hdl, ptrs_tensor = _get_symm_state(out_numel, x.dtype, x.device, group)
    
    # Identify max vectorization factor safely by finding the innermost non-degenerate dimension
    dims = [A, S1, B, S2, C]
    last_dim = 4
    for i in range(4, -1, -1):
        if dims[i] > 1:
            last_dim = i
            break
            
    vec_size = 1
    if dims[last_dim] % 8 == 0:
        vec_size = 8
    elif dims[last_dim] % 4 == 0:
        vec_size = 4
    elif dims[last_dim] % 2 == 0:
        vec_size = 2

    # Scale down the vectorization axis sizes
    dims[last_dim] //= vec_size
    A_v, S1_v, B_v, S2_v, C_v = dims

    s_less_than_g = scatter_dim < gather_dim

    # Wait for peers to finish reading from the double buffer element we're about to write over
    hdl.barrier(channel=0)

    _get_ext().ulysses_push(
        x, ptrs_tensor, world_size, rank,
        A_v, S1_v, B_v, S2_v, C_v, s_less_than_g, vec_size
    )

    # Wait for peers to finish pushing our incoming chunks to our UVA memory
    hdl.barrier(channel=1)

    return buf.view(out_shape)