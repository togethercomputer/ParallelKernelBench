"""
Strategy:
- **UVA & Symmetric Memory:** Replaces NCCL `broadcast` with direct device-to-device transfers over NVLink using PyTorch's `_symmetric_memory`.
- **Hybrid Broadcast Algorithm:**
  - **Direct Pull (Small Tensors):** For payloads ≤ 1MB, non-root ranks pull directly from the source rank's symmetric buffer to minimize barrier latency.
  - **Binomial Tree (Large Tensors):** For larger payloads, implements a recursive doubling (binomial tree) schedule. This overlaps communication by recursively turning receivers into senders, maximizing NVLink bisection bandwidth utilization and preventing bottlenecks on the source rank.
- **Custom CUDA Extension:** Data transfers use a JIT-compiled CUDA kernel optimized for dense payloads (like BF16). It treats data as raw bytes, uses 128-bit (`uint4`) vectorized loads, and grid-stride loops to saturate memory bandwidth while supporting any arbitrary numeric dtype.
"""

import math
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void uva_copy_bytes_kernel(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    int64_t n_vec
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n_vec; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void uva_copy_rem_bytes_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t offset,
    int64_t n_rem
) {
    int idx = threadIdx.x;
    if (idx < n_rem) {
        dst[offset + idx] = src[offset + idx];
    }
}

void uva_broadcast(
    int64_t src_ptr,
    int64_t dst_ptr,
    int64_t n_bytes
) {
    const int threads = 256;
    int64_t n_vec = n_bytes / 16;
    int64_t n_rem = n_bytes % 16;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (n_vec > 0) {
        const uint4* src_vec = reinterpret_cast<const uint4*>(static_cast<uintptr_t>(src_ptr));
        uint4* dst_vec = reinterpret_cast<uint4*>(static_cast<uintptr_t>(dst_ptr));
        int blocks = (int)std::min((int64_t)65536, (n_vec + threads - 1) / threads);
        uva_copy_bytes_kernel<<<blocks, threads, 0, stream>>>(src_vec, dst_vec, n_vec);
    }
    
    if (n_rem > 0) {
        const uint8_t* src_r = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(src_ptr));
        uint8_t* dst_r = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_ptr));
        uva_copy_rem_bytes_kernel<<<1, 32, 0, stream>>>(src_r, dst_r, n_vec * 16, n_rem);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_broadcast", &uva_broadcast, "UVA Broadcast copy in bytes");
}
'''

_ext = None

def ensure_ext():
    global _ext
    if _ext is None:
        if dist.get_rank() == 0:
            _ext = compile_cuda_extension("uva_broadcast_ext", CUDA_SRC)
        dist.barrier()
        if dist.get_rank() != 0:
            _ext = compile_cuda_extension("uva_broadcast_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(n_bytes: int, device: torch.device):
    global _symm_cache
    if "n_bytes" in _symm_cache and _symm_cache["n_bytes"] >= n_bytes:
        return _symm_cache["buf"], _symm_cache["hdl"]
    
    buf = symm_mem.empty(n_bytes, dtype=torch.uint8, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache["n_bytes"] = n_bytes
    _symm_cache["buf"] = buf
    _symm_cache["hdl"] = hdl
    return buf, hdl

@torch.no_grad()
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    ext = ensure_ext()
    n_bytes = tensor.numel() * tensor.element_size()
    
    # Fast path for empty tensors
    if n_bytes == 0:
        return tensor.clone() if rank == src else torch.empty_like(tensor)
        
    buf, hdl = _get_symm_state(n_bytes, tensor.device)
    
    # Root rank copies its tensor payload into its shared symmetric buffer
    if rank == src:
        ext.uva_broadcast(tensor.data_ptr(), buf.data_ptr(), n_bytes)
        
    hdl.barrier(channel=0)
    
    # Threshold for direct pull vs binomial tree (1 MB)
    if n_bytes <= 1024 * 1024:
        # Latency-optimized path: Direct multi-pull from src
        if rank == src:
            out = tensor.clone()
        else:
            out = torch.empty_like(tensor)
            remote_ptr = int(hdl.buffer_ptrs[src])
            ext.uva_broadcast(remote_ptr, out.data_ptr(), n_bytes)
    else:
        # Bandwidth-optimized path: Recursive doubling (binomial tree)
        rel_rank = (rank - src) % world_size
        num_steps = math.ceil(math.log2(world_size))
        
        for s in range(num_steps):
            d = 1 << s
            # Receivers in this step pull from their corresponding sender
            if d <= rel_rank < 2 * d:
                sender_rel = rel_rank - d
                sender_abs = (sender_rel + src) % world_size
                remote_ptr = int(hdl.buffer_ptrs[sender_abs])
                ext.uva_broadcast(remote_ptr, buf.data_ptr(), n_bytes)
            
            hdl.barrier(channel=0)
            
        if rank == src:
            out = tensor.clone()
        else:
            out = torch.empty_like(tensor)
            ext.uva_broadcast(buf.data_ptr(), out.data_ptr(), n_bytes)
            
    # Final barrier ensures all reads from symmetric buf are complete 
    # before returning, preventing subsequent calls from overwriting buf prematurely.
    hdl.barrier(channel=0)
    
    return out