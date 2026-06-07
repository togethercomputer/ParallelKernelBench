"""
Strategy:
- **Topology-Aware Direct Pull**: On Hopper's fully connected NVSwitch, a multi-step tree gather actually increases total NVLink traffic and latency steps. We instead use a flat 1-step P2P pull where the destination rank directly reads from all peers, minimizing latency and saturating the destination's RX bandwidth.
- **Symmetric Memory Staging**: All ranks stage their input chunk into a lightweight `torch.distributed._symmetric_memory` buffer, avoiding a W-sized allocation on every rank and providing stable device pointers for the destination rank.
- **Custom Vectorized CUDA Kernel**: A custom JIT-compiled kernel on the destination rank uses perfectly aligned memory accesses (up to 128-bit `uint4`) to pull from all peers in a single grid launch. This eliminates host-side `cudaMemcpyAsync` loop bottlenecks and maintains peak SM utilization.
- **Zero Local Copies on Destination**: The destination rank skips staging entirely, directly writing its own input tensor to the output buffer, cutting memory operations on the bottleneck rank to the theoretical minimum.
- **Compute-Communication Overlap**: The schedule perfectly scopes device barriers such that non-destination ranks can safely return and launch independent compute while the destination rank is still busy pulling their data.
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

// ---------------------------------------------------------------------------
// Direct Pull Gather Kernel
// ---------------------------------------------------------------------------
// The destination rank launches this kernel to directly pull data from all
// peers' symmetric memory buffers over NVLink into a single contiguous output.
__global__ void direct_pull_gather_kernel(
    const uint64_t* __restrict__ ptrs,
    uint8_t* __restrict__ out,
    int64_t chunk_bytes,
    int dst
) {
    // blockIdx.y represents the peer rank we are pulling from
    int peer = blockIdx.y;
    
    // The destination rank skips itself as it handles its local copy directly
    if (peer == dst) return;
    
    const uint8_t* src = (const uint8_t*)ptrs[peer];
    uint8_t* dst_ptr = out + peer * chunk_bytes;
    
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    // Dynamically use the widest memory instructions possible based on alignment
    if (chunk_bytes % 16 == 0) {
        int64_t chunk_16 = chunk_bytes / 16;
        const uint4* src_16 = reinterpret_cast<const uint4*>(src);
        uint4* dst_16 = reinterpret_cast<uint4*>(dst_ptr);
        for (int64_t i = idx; i < chunk_16; i += stride) {
            dst_16[i] = src_16[i];
        }
    } else if (chunk_bytes % 8 == 0) {
        int64_t chunk_8 = chunk_bytes / 8;
        const uint2* src_8 = reinterpret_cast<const uint2*>(src);
        uint2* dst_8 = reinterpret_cast<uint2*>(dst_ptr);
        for (int64_t i = idx; i < chunk_8; i += stride) {
            dst_8[i] = src_8[i];
        }
    } else if (chunk_bytes % 4 == 0) {
        int64_t chunk_4 = chunk_bytes / 4;
        const uint32_t* src_4 = reinterpret_cast<const uint32_t*>(src);
        uint32_t* dst_4 = reinterpret_cast<uint32_t*>(dst_ptr);
        for (int64_t i = idx; i < chunk_4; i += stride) {
            dst_4[i] = src_4[i];
        }
    } else if (chunk_bytes % 2 == 0) {
        int64_t chunk_2 = chunk_bytes / 2;
        const uint16_t* src_2 = reinterpret_cast<const uint16_t*>(src);
        uint16_t* dst_2 = reinterpret_cast<uint16_t*>(dst_ptr);
        for (int64_t i = idx; i < chunk_2; i += stride) {
            dst_2[i] = src_2[i];
        }
    } else {
        for (int64_t i = idx; i < chunk_bytes; i += stride) {
            dst_ptr[i] = src[i];
        }
    }
}

void launch_gather_pull(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t chunk_bytes,
    int dst
) {
    int world_size = ptrs_tensor.size(0);
    const uint64_t* ptrs = (const uint64_t*)ptrs_tensor.data_ptr<int64_t>();
    uint8_t* out_ptr = (uint8_t*)out.data_ptr();
    
    // Match the host-side element count to the device-side loop stride
    int64_t align_size = 1;
    if (chunk_bytes % 16 == 0) align_size = 16;
    else if (chunk_bytes % 8 == 0) align_size = 8;
    else if (chunk_bytes % 4 == 0) align_size = 4;
    else if (chunk_bytes % 2 == 0) align_size = 2;
    
    int64_t elements = chunk_bytes / align_size;
    
    int threads = 256;
    int blocks_x = (elements + threads - 1) / threads;
    if (blocks_x == 0) blocks_x = 1;
    
    // Cap blocks to prevent grid over-subscription; grid-stride loops handle the rest
    if (blocks_x > 1024) blocks_x = 1024;
    
    // 2D Grid: X maps to data chunks, Y maps to peer ranks
    dim3 blocks(blocks_x, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    direct_pull_gather_kernel<<<blocks, threads, 0, stream>>>(ptrs, out_ptr, chunk_bytes, dst);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_pull", &launch_gather_pull, "Direct pull kernel for device-side gather");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gather_direct_pull_ext", CUDA_SRC)
    return _ext


_resource_cache = {}

def _get_resources(shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    Allocates and caches symmetric memory buffers to completely bypass 
    process rendezvous overheads on repeated calls.
    """
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
    dst: int = 0,
) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    
    tensor = tensor.contiguous()
    buf, hdl, ptrs_tensor = _get_resources(tensor.shape, tensor.dtype, tensor.device)
    
    # 1. Non-destination ranks quickly stage their input tensor into symmetric memory.
    # The destination rank skips this entirely to save memory operations.
    if rank != dst:
        buf.copy_(tensor)
        
    # Synchronization ensures symmetric buffers are ready before the pull kernel fires.
    hdl.barrier(channel=0)
    
    # 2. Destination rank pulls from all symmetric memory staging buffers.
    if rank == dst:
        out = torch.empty((world_size, *tensor.shape), dtype=tensor.dtype, device=tensor.device)
        
        # Safely copy own local tensor without invoking NVLink overhead.
        out[dst].copy_(tensor)
        
        # Fire vectorized pull-kernel over NVLink mappings directly into the output tensor.
        chunk_bytes = tensor.numel() * tensor.element_size()
        _get_ext().launch_gather_pull(ptrs_tensor, out, chunk_bytes, dst)
        
    # Post-sync protects symmetric buffers from next-call overwrite.
    hdl.barrier(channel=1)
    
    if rank == dst:
        return out
    else:
        return tensor