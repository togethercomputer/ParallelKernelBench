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
#include <cstdint>

// A custom push kernel that directly scatters local FFT results into
// the remote, concatenated target tensors via NVLink symmetric memory.
template<typename T>
__global__ void push_kernel(
    const T* __restrict__ x1,
    const uint64_t* __restrict__ remote_ptrs,
    uint32_t N0, uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N4,
    uint32_t chunk0, uint32_t D1, uint32_t D1_mul_W,
    uint32_t my_rank,
    bool dim0_lt_dim1,
    uint64_t total_elements
) {
    for (uint64_t flat_idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; 
         flat_idx < total_elements; 
         flat_idx += (uint64_t)gridDim.x * blockDim.x) {
         
        // N-Dimensional Indexing mapped over a flattened generic 5D abstraction
        uint64_t temp = flat_idx;
        uint32_t i4 = temp % N4; temp /= N4;
        uint32_t i3 = temp % N3; temp /= N3;
        uint32_t i2 = temp % N2; temp /= N2;
        uint32_t i1 = temp % N1; temp /= N1;
        uint32_t i0 = temp;

        uint32_t r;
        uint64_t out_flat;

        if (dim0_lt_dim1) {
            r = i1 / chunk0;
            uint32_t out_i1 = i1 % chunk0;
            uint32_t out_i3 = my_rank * D1 + i3;
            // Native concatenated flat offset calculation
            out_flat = ((( (uint64_t)i0 * chunk0 + out_i1 ) * N2 + i2 ) * D1_mul_W + out_i3 ) * N4 + i4;
        } else {
            r = i3 / chunk0;
            uint32_t out_i3 = i3 % chunk0;
            uint32_t out_i1 = my_rank * D1 + i1;
            // Native concatenated flat offset calculation
            out_flat = ((( (uint64_t)i0 * D1_mul_W + out_i1 ) * N2 + i2 ) * chunk0 + out_i3 ) * N4 + i4;
        }

        // Direct device-side scatter across NVLink into remote peer memory
        T* dest = (T*)remote_ptrs[r];
        dest[out_flat] = x1[flat_idx];
    }
}

void launch_push_kernel(
    torch::Tensor x1,
    torch::Tensor remote_ptrs,
    int64_t N0, int64_t N1, int64_t N2, int64_t N3, int64_t N4,
    int64_t chunk0, int64_t D1, int64_t D1_mul_W,
    int my_rank,
    bool dim0_lt_dim1
) {
    uint64_t total_elements = N0 * N1 * N2 * N3 * N4;
    if (total_elements == 0) return;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 1024 * 64) blocks = 1024 * 64;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* ptrs = (const uint64_t*)remote_ptrs.data_ptr<int64_t>();
    
    // Copy vectorization mappings depending on standard complex shapes
    if (x1.element_size() == 8) {
        push_kernel<int64_t><<<blocks, threads, 0, stream>>>(
            (const int64_t*)x1.data_ptr(), ptrs,
            N0, N1, N2, N3, N4, chunk0, D1, D1_mul_W,
            my_rank, dim0_lt_dim1, total_elements
        );
    } else if (x1.element_size() == 16) {
        push_kernel<int4><<<blocks, threads, 0, stream>>>(
            (const int4*)x1.data_ptr(), ptrs,
            N0, N1, N2, N3, N4, chunk0, D1, D1_mul_W,
            my_rank, dim0_lt_dim1, total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported element size for direct push");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_push_kernel", &launch_push_kernel, "Custom symmetric push kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("physicsnemo_fft_push_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(shape, dtype, device, group):
    key = (tuple(shape), dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return buf, hdl, ptrs_tensor

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
    my_rank = dist.get_rank(group)

    dim0, dim1 = int(dim[0]), int(dim[1])
    ndim = x.ndim
    
    # Handle negative dimension indexing
    dim0 = dim0 if dim0 >= 0 else dim0 + ndim
    dim1 = dim1 if dim1 >= 0 else dim1 + ndim

    # 1. Transform the replicated spatial dimension.
    # Output of fft for float32/bfloat16 returns complex64 (8 bytes per element)
    x1 = torch.fft.fft(x, n=int(s[0]), dim=dim0, norm=norm).contiguous()

    if world_size == 1:
        x1_tran = x1
    else:
        # 2. All-to-all transpose fused into Custom Symmetric Memory Push
        shape_in = list(x1.shape)
        D0 = shape_in[dim0]
        D1 = shape_in[dim1]
        chunk0 = D0 // world_size

        shape_tran = list(shape_in)
        shape_tran[dim0] = chunk0
        shape_tran[dim1] = D1 * world_size

        buf, hdl, ptrs_tensor = _get_symm_state(shape_tran, x1.dtype, x1.device, group)

        # Convert varying arbitrary N-dimensional space to strict 5D configuration abstraction
        d_min = min(dim0, dim1)
        d_max = max(dim0, dim1)

        N0 = math.prod(shape_in[:d_min])
        N1 = shape_in[d_min]
        N2 = math.prod(shape_in[d_min+1:d_max])
        N3 = shape_in[d_max]
        N4 = math.prod(shape_in[d_max+1:])

        dim0_lt_dim1 = (dim0 < dim1)

        # Build extension serially from Rank 0 locally to prevent compilation race conditions
        if my_rank == 0:
            _get_ext()
        dist.barrier(group=group)

        # Ensure local symmetric buffer is clear and ready to receive writes
        hdl.barrier(channel=0)

        _get_ext().launch_push_kernel(
            x1, ptrs_tensor,
            N0, N1, N2, N3, N4,
            chunk0, D1, D1 * world_size,
            my_rank, dim0_lt_dim1
        )

        # Stream sync followed by a dist barrier guarantees all peers have finished writing safely
        torch.cuda.current_stream().synchronize()
        dist.barrier(group=group)
        
        x1_tran = buf

    # 3. Transform the now-replicated second dimension natively
    x2 = torch.fft.fft(x1_tran, n=int(s[1]), dim=dim1, norm=norm)

    # 4. Truncate returning real-input half spectrum shape mapping
    return _truncate(x2, dim1, x2.shape[dim1] // 2 + 1)