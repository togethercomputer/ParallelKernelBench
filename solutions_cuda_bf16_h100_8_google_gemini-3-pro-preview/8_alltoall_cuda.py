"""
Strategy:
- **Device-Side Communication:** Instead of NCCL, we use `torch.distributed._symmetric_memory` to allocate UVA-accessible device memory. A custom CUDA kernel performs parallel "PULL" operations directly over NVLink, allowing each rank to independently fetch its designated data chunk from peers' symmetric buffers without host synchronization.
- **Maximized Memory Bandwidth:** The kernel dynamically verifies memory alignment and degrades gracefully, using 128-bit (`uint4`) vectorized loads and stores whenever chunks are 16-byte aligned. This maximizes NVLink and global memory bus utilization.
- **Compute-Communication Overlap & Stream Semantics:** By utilizing `symm_mem.rendezvous.barrier()`, the synchronization is fully stream-ordered. The entire operation—local copy, peer synchronization, and PULL data movement—is enqueued asynchronously, allowing the host to immediately return and the CUDA scheduler to execute the collective efficiently while the CPU proceeds.
"""

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

__global__ void pull_all_to_all_kernel(
    const uint64_t* __restrict__ peer_ptrs,
    uint8_t* __restrict__ out,
    int64_t chunk_bytes,
    int world_size,
    int rank
) {
    // Each block in the Y dimension handles reading from a specific peer
    int peer = blockIdx.y;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    const uint8_t* peer_buf = reinterpret_cast<const uint8_t*>(peer_ptrs[peer]);
    int64_t peer_start = (int64_t)rank * chunk_bytes;
    int64_t out_start = (int64_t)peer * chunk_bytes;
    
    // Attempt the widest possible vectorized load/store based on alignment
    if ((reinterpret_cast<uintptr_t>(peer_buf) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(out) % 16 == 0) &&
        (peer_start % 16 == 0) && 
        (out_start % 16 == 0) &&
        (chunk_bytes % 16 == 0)) 
    {
        int64_t chunk_128b = chunk_bytes / 16;
        const uint4* peer_buf_128 = reinterpret_cast<const uint4*>(peer_buf + peer_start);
        uint4* out_buf_128 = reinterpret_cast<uint4*>(out + out_start);
        
        for (int64_t offset = tid; offset < chunk_128b; offset += (int64_t)gridDim.x * blockDim.x) {
            out_buf_128[offset] = peer_buf_128[offset];
        }
    } 
    else if ((reinterpret_cast<uintptr_t>(peer_buf) % 8 == 0) &&
             (reinterpret_cast<uintptr_t>(out) % 8 == 0) &&
             (peer_start % 8 == 0) && 
             (out_start % 8 == 0) &&
             (chunk_bytes % 8 == 0))
    {
        int64_t chunk_64b = chunk_bytes / 8;
        const uint64_t* peer_buf_64 = reinterpret_cast<const uint64_t*>(peer_buf + peer_start);
        uint64_t* out_buf_64 = reinterpret_cast<uint64_t*>(out + out_start);
        
        for (int64_t offset = tid; offset < chunk_64b; offset += (int64_t)gridDim.x * blockDim.x) {
            out_buf_64[offset] = peer_buf_64[offset];
        }
    }
    else if ((reinterpret_cast<uintptr_t>(peer_buf) % 4 == 0) &&
             (reinterpret_cast<uintptr_t>(out) % 4 == 0) &&
             (peer_start % 4 == 0) && 
             (out_start % 4 == 0) &&
             (chunk_bytes % 4 == 0))
    {
        int64_t chunk_32b = chunk_bytes / 4;
        const uint32_t* peer_buf_32 = reinterpret_cast<const uint32_t*>(peer_buf + peer_start);
        uint32_t* out_buf_32 = reinterpret_cast<uint32_t*>(out + out_start);
        
        for (int64_t offset = tid; offset < chunk_32b; offset += (int64_t)gridDim.x * blockDim.x) {
            out_buf_32[offset] = peer_buf_32[offset];
        }
    }
    else if ((reinterpret_cast<uintptr_t>(peer_buf) % 2 == 0) &&
             (reinterpret_cast<uintptr_t>(out) % 2 == 0) &&
             (peer_start % 2 == 0) && 
             (out_start % 2 == 0) &&
             (chunk_bytes % 2 == 0))
    {
        int64_t chunk_16b = chunk_bytes / 2;
        const uint16_t* peer_buf_16 = reinterpret_cast<const uint16_t*>(peer_buf + peer_start);
        uint16_t* out_buf_16 = reinterpret_cast<uint16_t*>(out + out_start);
        
        for (int64_t offset = tid; offset < chunk_16b; offset += (int64_t)gridDim.x * blockDim.x) {
            out_buf_16[offset] = peer_buf_16[offset];
        }
    }
    else {
        for (int64_t offset = tid; offset < chunk_bytes; offset += (int64_t)gridDim.x * blockDim.x) {
            out[out_start + offset] = peer_buf[peer_start + offset];
        }
    }
}

void launch_pull_all_to_all(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor out,
    int64_t chunk_bytes,
    int world_size,
    int rank
) {
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs_tensor.data_ptr<int64_t>());
    uint8_t* d_out = reinterpret_cast<uint8_t*>(out.data_ptr());
    
    int threads = 256;
    int max_blocks_x = 1024;
    
    int64_t max_elements = chunk_bytes; 
    if (chunk_bytes % 16 == 0) max_elements = chunk_bytes / 16;
    else if (chunk_bytes % 8 == 0) max_elements = chunk_bytes / 8;
    else if (chunk_bytes % 4 == 0) max_elements = chunk_bytes / 4;
    else if (chunk_bytes % 2 == 0) max_elements = chunk_bytes / 2;

    int blocks_x = std::min((int)((max_elements + threads - 1) / threads), max_blocks_x);
    if (blocks_x < 1) blocks_x = 1;

    dim3 blocks(blocks_x, world_size, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    pull_all_to_all_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, d_out, chunk_bytes, world_size, rank
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pull_all_to_all", &launch_pull_all_to_all, "Pull-based All-to-All P2P kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("all_to_all_pull_ext", CUDA_SRC)
    return _ext

_resource_cache = {}

def _get_resources(shape, dtype, device):
    """Caches symmetric memory buffers and rendezvous handles to prevent reallocation overhead."""
    key = (shape, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]

    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, ptrs_tensor)
    _resource_cache[key] = res
    return res

@torch.no_grad()
def solution(
    tensor: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_contiguous(), "Input tensor must be contiguous"
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert tensor.shape[0] == world_size, \
        f"First dimension ({tensor.shape[0]}) must equal world_size ({world_size})"
        
    buf, hdl, ptrs_tensor = _get_resources(tensor.shape, tensor.dtype, tensor.device)
    
    # Copy local data into symmetric memory so it's accessible to peers
    buf.copy_(tensor)
    
    # Enqueue stream-ordered barrier: wait for all peers to finish writing their buffers
    hdl.barrier(channel=0)
    
    # Calculate bytes per rank chunk
    chunk_bytes = (tensor.numel() // world_size) * tensor.element_size()
    out = torch.empty_like(tensor)
    
    # PULL execution: read slices asynchronously over NVLink/UVA directly to local output
    _get_ext().launch_pull_all_to_all(ptrs_tensor, out, chunk_bytes, world_size, rank)
    
    # Enqueue stream-ordered barrier: prevent overwriting the buffer in subsequent calls
    # before peers have safely concluded pulling
    hdl.barrier(channel=0)
    
    return out