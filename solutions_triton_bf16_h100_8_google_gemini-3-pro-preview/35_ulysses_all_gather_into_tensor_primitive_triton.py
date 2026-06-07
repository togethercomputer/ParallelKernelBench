from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

template <int MAX_RANKS>
struct PtrArray {
    const uint16_t* ptrs[MAX_RANKS];
};

template <int MAX_RANKS>
__global__ void all_gather_kernel_bf16_vec_2d(
    PtrArray<MAX_RANKS> remote_ptrs,
    int64_t elements_per_rank,
    uint16_t* __restrict__ out,
    int my_rank
) {
    int rank = blockIdx.y;
    // Skip processing for our own rank as it's already copied locally
    if (rank == my_rank) return;
    
    int64_t vecs_per_rank = elements_per_rank / 8;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    const uint4* src = reinterpret_cast<const uint4*>(remote_ptrs.ptrs[rank]);
    uint4* dst = reinterpret_cast<uint4*>(out + rank * elements_per_rank);
    
    for (int64_t i = idx; i < vecs_per_rank; i += stride) {
        dst[i] = src[i];
    }
}

template <int MAX_RANKS>
__global__ void all_gather_kernel_bf16_scalar_2d(
    PtrArray<MAX_RANKS> remote_ptrs,
    int64_t elements_per_rank,
    uint16_t* __restrict__ out,
    int my_rank
) {
    int rank = blockIdx.y;
    if (rank == my_rank) return;
    
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    const uint16_t* src = remote_ptrs.ptrs[rank];
    uint16_t* dst = out + rank * elements_per_rank;
    
    for (int64_t i = idx; i < elements_per_rank; i += stride) {
        dst[i] = src[i];
    }
}

void ulysses_all_gather_cuda(
    std::vector<int64_t> remote_ptr_ints,
    int64_t elements_per_rank,
    torch::Tensor out,
    int my_rank
) {
    int world_size = remote_ptr_ints.size();
    TORCH_CHECK(world_size <= 32, "Max 32 ranks supported");
    
    PtrArray<32> ptrs;
    for (int i = 0; i < world_size; ++i) {
        ptrs.ptrs[i] = reinterpret_cast<const uint16_t*>(remote_ptr_ints[i]);
    }
    
    const int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (elements_per_rank % 8 == 0) {
        int blocks_x = std::min<int>((elements_per_rank / 8 + threads - 1) / threads, 1024);
        if (blocks_x == 0) blocks_x = 1;
        dim3 grid(blocks_x, world_size);
        
        all_gather_kernel_bf16_vec_2d<32><<<grid, threads, 0, stream>>>(
            ptrs, elements_per_rank, reinterpret_cast<uint16_t*>(out.data_ptr()), my_rank
        );
    } else {
        int blocks_x = std::min<int>((elements_per_rank + threads - 1) / threads, 1024);
        if (blocks_x == 0) blocks_x = 1;
        dim3 grid(blocks_x, world_size);
        
        all_gather_kernel_bf16_scalar_2d<32><<<grid, threads, 0, stream>>>(
            ptrs, elements_per_rank, reinterpret_cast<uint16_t*>(out.data_ptr()), my_rank
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("all_gather", &ulysses_all_gather_cuda, "UVA all gather bf16");
}
'''

_ext = None
_compile_done = False

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_all_gather_uva_ext", CUDA_SRC)
    return _ext

def _ensure_compiled():
    global _compile_done
    if not _compile_done:
        if dist.get_rank() == 0:
            _get_ext()
        dist.barrier()
        _get_ext()
        _compile_done = True

_symm_cache = {}

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    key = (n, dtype, id(group))
    if key in _symm_cache:
        return _symm_cache[key]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl

def solution(
    x: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    x = x.contiguous()
    dim_size = list(x.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=x.dtype, device=x.device)

    n = x.numel()
    if n == 0:
        return output

    _ensure_compiled()
    buf, hdl = _get_symm_state(n, x.dtype, x.device, group)
    my_rank = dist.get_rank(group)

    # Overlap opportunity: Write into our symm_mem buffer for peers to read, 
    # while concurrently doing the local copy to the final output slot.
    buf.copy_(x.view(-1))
    chunk_size = x.size(0)
    output[my_rank * chunk_size : (my_rank + 1) * chunk_size].copy_(x)

    # Assure all peers have written their chunk to their respective symm memory.
    hdl.barrier(channel=0)

    # Execute custom all-gather to pull direct UVA from peers 
    # (skipping our own slot handled above)
    remote_ptrs = [int(hdl.buffer_ptrs[i]) for i in range(world_size)]
    _get_ext().all_gather(remote_ptrs, n, output, my_rank)

    return output