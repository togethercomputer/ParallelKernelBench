# Strategy:
# 1. Device-side communication: Instead of opaque NCCL rings, we allocate symmetric
#    memory for the output tensor and use Hopper's NVLink hardware multicast 
#    (`multimem.st`) to PUSH each rank's local shard directly into the correct slice 
#    of all W peers' output buffers simultaneously via a single custom kernel.
# 2. Optimal bandwidth: The 1-to-W hardware broadcast reduces memory reads on the 
#    sender by Wx compared to standard P2P push, perfectly saturating NVLink. 
# 3. Double-buffering & Overlap: We rotate through a pool of symmetric memory buffers. 
#    This structure safely hides stream synchronization latency, requires only one 
#    device-side barrier per invocation, and allows overlap with unrelated downstream ops.

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
#include <algorithm>

// ---------------------------------------------------------------------------
// Hardware Multicast Push (Hopper multimem.st)
// Broadcasts 16-byte chunks (e.g. 8x bfloat16) to all ranks simultaneously.
// ---------------------------------------------------------------------------
__global__ void multimem_push_16B(
    const uint4* __restrict__ local_x,
    uint64_t multicast_ptr,
    int64_t numel_16b,
    int64_t offset_16b
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < numel_16b; idx += gridDim.x * blockDim.x) {
        uint4 val = local_x[idx];
        uint64_t dst = multicast_ptr + (offset_16b + idx) * 16;
        asm volatile(
            "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
            :: "l"(dst), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
            : "memory"
        );
    }
}

// ---------------------------------------------------------------------------
// Fallback: Standard P2P Push (For older architectures or missing NVSwitch)
// ---------------------------------------------------------------------------
__global__ void p2p_push_16B_kernel(
    const uint4* __restrict__ local_x,
    const uint64_t* __restrict__ dst_ptrs,
    int world_size,
    int64_t numel_16b,
    int64_t offset_16b
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < numel_16b; idx += gridDim.x * blockDim.x) {
        uint4 val = local_x[idx];
        #pragma unroll 8
        for (int r = 0; r < world_size; ++r) {
            uint4* dst = (uint4*)(dst_ptrs[r] + (offset_16b + idx) * 16);
            *dst = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Unaligned Fallback (2-byte chunks for bfloat16 / float16 elements)
// ---------------------------------------------------------------------------
__global__ void p2p_push_2B_kernel(
    const uint16_t* __restrict__ local_x,
    const uint64_t* __restrict__ dst_ptrs,
    int world_size,
    int64_t numel_2b,
    int64_t offset_2b
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < numel_2b; idx += gridDim.x * blockDim.x) {
        uint16_t val = local_x[idx];
        #pragma unroll 8
        for (int r = 0; r < world_size; ++r) {
            uint16_t* dst = (uint16_t*)(dst_ptrs[r] + (offset_2b + idx) * 2);
            *dst = val;
        }
    }
}

void launch_multimem_push(
    torch::Tensor local_x,
    uint64_t multicast_ptr,
    torch::Tensor ptrs_tensor,
    int world_size,
    int rank,
    int64_t numel_bytes
) {
    if (numel_bytes == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    
    // Ensure memory pointer and byte sizes are fully 16-byte aligned.
    bool is_16b_aligned = (reinterpret_cast<uintptr_t>(local_x.data_ptr()) % 16 == 0) &&
                          (numel_bytes % 16 == 0);
    
    if (is_16b_aligned) {
        int64_t numel_16b = numel_bytes / 16;
        int64_t offset_16b = (rank * numel_bytes) / 16; 
        int64_t max_blocks = 65535;
        int64_t blocks = std::min(max_blocks, (numel_16b + threads - 1) / threads);
        multimem_push_16B<<<blocks, threads, 0, stream>>>(
            (const uint4*)local_x.data_ptr(),
            multicast_ptr,
            numel_16b,
            offset_16b
        );
    } else {
        int64_t numel_2b = numel_bytes / 2;
        int64_t offset_2b = (rank * numel_bytes) / 2;
        int64_t max_blocks = 65535;
        int64_t blocks = std::min(max_blocks, (numel_2b + threads - 1) / threads);
        p2p_push_2B_kernel<<<blocks, threads, 0, stream>>>(
            (const uint16_t*)local_x.data_ptr(),
            (const uint64_t*)ptrs_tensor.data_ptr(),
            world_size,
            numel_2b,
            offset_2b
        );
    }
}

void launch_p2p_push(
    torch::Tensor local_x,
    torch::Tensor ptrs_tensor,
    int world_size,
    int rank,
    int64_t numel_bytes
) {
    if (numel_bytes == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    
    bool is_16b_aligned = (reinterpret_cast<uintptr_t>(local_x.data_ptr()) % 16 == 0) &&
                          (numel_bytes % 16 == 0);
                          
    if (is_16b_aligned) {
        int64_t numel_16b = numel_bytes / 16;
        int64_t offset_16b = (rank * numel_bytes) / 16; 
        int64_t max_blocks = 65535;
        int64_t blocks = std::min(max_blocks, (numel_16b + threads - 1) / threads);
        p2p_push_16B_kernel<<<blocks, threads, 0, stream>>>(
            (const uint4*)local_x.data_ptr(),
            (const uint64_t*)ptrs_tensor.data_ptr(),
            world_size,
            numel_16b,
            offset_16b
        );
    } else {
        int64_t numel_2b = numel_bytes / 2;
        int64_t offset_2b = (rank * numel_bytes) / 2;
        int64_t max_blocks = 65535;
        int64_t blocks = std::min(max_blocks, (numel_2b + threads - 1) / threads);
        p2p_push_2B_kernel<<<blocks, threads, 0, stream>>>(
            (const uint16_t*)local_x.data_ptr(),
            (const uint64_t*)ptrs_tensor.data_ptr(),
            world_size,
            numel_2b,
            offset_2b
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_push", &launch_multimem_push, "Multicast push onto symmetric pointers");
    m.def("launch_p2p_push", &launch_p2p_push, "Peer-to-peer push fallback");
}
'''

_ext = None
_ext_compiled = False


def _get_ext_safe(rank: int, group: dist.ProcessGroup):
    global _ext_compiled, _ext
    if not _ext_compiled:
        # Protect compilation lock races across ranks by staggering the entry
        if rank == 0:
            _ext = compile_cuda_extension("ulysses_allgather_ext", CUDA_SRC)
        dist.barrier(group)
        if rank != 0:
            _ext = compile_cuda_extension("ulysses_allgather_ext", CUDA_SRC)
        _ext_compiled = True
    return _ext


_symm_cache = {}


def _get_symm_output(shape, dtype, device, group):
    """
    Allocates and caches a pool of 2 symmetric memory buffers for double buffering.
    This safely guarantees isolation between iterations in the pipeline schedule.
    """
    key = (tuple(shape), dtype, device)
    if key not in _symm_cache:
        pool = []
        for _ in range(2):
            buf = symm_mem.empty(shape, device=device, dtype=dtype)
            hdl = symm_mem.rendezvous(buf, group)
            ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
            pool.append((buf, hdl, ptrs))
        _symm_cache[key] = {
            "pool": pool,
            "idx": 0
        }
    
    cache = _symm_cache[key]
    idx = cache["idx"]
    buf, hdl, ptrs = cache["pool"][idx]
    # Advance to the next buffer for the following invocation
    cache["idx"] = (idx + 1) % 2
    return buf, hdl, ptrs


@torch.no_grad()
def solution(
    x: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    x = x.contiguous()
    rank = dist.get_rank(group)
    
    dim_size = list(x.size())
    # Expand 0-th dimension to hold the entire gathered size
    dim_size[0] = dim_size[0] * world_size
    
    ext = _get_ext_safe(rank, group)
    buf, hdl, ptrs_tensor = _get_symm_output(dim_size, x.dtype, x.device, group)
    
    numel_bytes = x.numel() * x.element_size()
    multicast_ptr = int(hdl.multicast_ptr) if hasattr(hdl, 'multicast_ptr') else 0
    
    # Push local shard directly to the appropriate slice in all ranks' symmetric outputs.
    if multicast_ptr != 0:
        ext.launch_multimem_push(x, multicast_ptr, ptrs_tensor, world_size, rank, numel_bytes)
    else:
        ext.launch_p2p_push(x, ptrs_tensor, world_size, rank, numel_bytes)
        
    # Queue a device-side stream barrier enforcing complete delivery visibility before clone
    hdl.barrier(channel=0)
    
    # Return an independent instance isolated from future rotations of symmetric buffer
    return buf.clone()