"""
Strategy:
- **Device-Side Push:** Avoids repeated NCCL/host collective calls by allocating symmetric memory buffers (`symm_mem`) on all ranks, enabling direct peer-to-peer memory mappings via UVA. The source rank executes a single custom CUDA kernel that pushes data directly to all peers' symmetric buffers simultaneously.
- **Compute–Communication Overlap:** By launching a grid-stride kernel spanning multiple SMs (one grid Y-dimension per peer), the data push maximally exploits the bidirectional NVLink bandwidth. The copy dynamically falls back from 128-bit vectorization to granular access, ensuring high throughput without blocking host CPU, masked behind stream-ordered barriers (`hdl.barrier()`).
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

__global__ void scatter_push_kernel(
    const char* __restrict__ input,
    const uint64_t* __restrict__ peer_ptrs,
    int64_t chunk_bytes,
    int world_size,
    int src_rank
) {
    int rank_idx = blockIdx.y;
    if (rank_idx >= world_size || rank_idx == src_rank) return;
    
    char* out = reinterpret_cast<char*>(peer_ptrs[rank_idx]);
    const char* in = input + rank_idx * chunk_bytes;
    
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    // Calculate maximum alignment for vectorization
    int align = 1;
    if (((reinterpret_cast<uintptr_t>(in) % 16) == 0) && ((reinterpret_cast<uintptr_t>(out) % 16) == 0)) align = 16;
    else if (((reinterpret_cast<uintptr_t>(in) % 8) == 0) && ((reinterpret_cast<uintptr_t>(out) % 8) == 0)) align = 8;
    else if (((reinterpret_cast<uintptr_t>(in) % 4) == 0) && ((reinterpret_cast<uintptr_t>(out) % 4) == 0)) align = 4;
    else if (((reinterpret_cast<uintptr_t>(in) % 2) == 0) && ((reinterpret_cast<uintptr_t>(out) % 2) == 0)) align = 2;

    if (align == 16) {
        int64_t num_vec = chunk_bytes / 16;
        const uint4* in_vec = reinterpret_cast<const uint4*>(in);
        uint4* out_vec = reinterpret_cast<uint4*>(out);
        for (int64_t i = tid; i < num_vec; i += stride) {
            out_vec[i] = in_vec[i];
        }
        if (tid == 0) {
            for (int64_t i = num_vec * 16; i < chunk_bytes; i++) out[i] = in[i];
        }
    } else if (align == 8) {
        int64_t num_vec = chunk_bytes / 8;
        const uint2* in_vec = reinterpret_cast<const uint2*>(in);
        uint2* out_vec = reinterpret_cast<uint2*>(out);
        for (int64_t i = tid; i < num_vec; i += stride) {
            out_vec[i] = in_vec[i];
        }
        if (tid == 0) {
            for (int64_t i = num_vec * 8; i < chunk_bytes; i++) out[i] = in[i];
        }
    } else if (align == 4) {
        int64_t num_vec = chunk_bytes / 4;
        const uint32_t* in_vec = reinterpret_cast<const uint32_t*>(in);
        uint32_t* out_vec = reinterpret_cast<uint32_t*>(out);
        for (int64_t i = tid; i < num_vec; i += stride) {
            out_vec[i] = in_vec[i];
        }
        if (tid == 0) {
            for (int64_t i = num_vec * 4; i < chunk_bytes; i++) out[i] = in[i];
        }
    } else if (align == 2) {
        int64_t num_vec = chunk_bytes / 2;
        const uint16_t* in_vec = reinterpret_cast<const uint16_t*>(in);
        uint16_t* out_vec = reinterpret_cast<uint16_t*>(out);
        for (int64_t i = tid; i < num_vec; i += stride) {
            out_vec[i] = in_vec[i];
        }
        if (tid == 0) {
            for (int64_t i = num_vec * 2; i < chunk_bytes; i++) out[i] = in[i];
        }
    } else {
        for (int64_t i = tid; i < chunk_bytes; i += stride) {
            out[i] = in[i];
        }
    }
}

void launch_scatter_push(
    torch::Tensor input,
    torch::Tensor peer_ptrs_tensor,
    int64_t chunk_bytes,
    int world_size,
    int src_rank
) {
    const char* d_input = reinterpret_cast<const char*>(input.data_ptr());
    const uint64_t* d_peer_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs_tensor.data_ptr<int64_t>());
    
    int threads = 256;
    int64_t max_vec = (chunk_bytes + 15) / 16;
    int64_t blocks_per_rank = (max_vec + threads - 1) / threads;
    if (blocks_per_rank > 256) blocks_per_rank = 256;
    if (blocks_per_rank < 1) blocks_per_rank = 1;
    
    dim3 blocks(static_cast<unsigned int>(blocks_per_rank), static_cast<unsigned int>(world_size));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    scatter_push_kernel<<<blocks, threads, 0, stream>>>(
        d_input, d_peer_ptrs, chunk_bytes, world_size, src_rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_scatter_push", &launch_scatter_push, "Scatter push kernel via symmetric memory UVA");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        if dist.is_initialized():
            if dist.get_rank() == 0:
                _ext = compile_cuda_extension("scatter_cuda_ext", CUDA_SRC)
            dist.barrier()
        _ext = compile_cuda_extension("scatter_cuda_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(chunk_shape, dtype, device):
    key = (tuple(chunk_shape), dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(chunk_shape, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache[key] = (buf, hdl, ptrs)
    return buf, hdl, ptrs

_channel = 0
def _next_channel():
    global _channel
    c = _channel
    _channel = (_channel + 1) % 256
    return c


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor.clone()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Handle potentially empty tensors directly
    if tensor.numel() == 0:
        if rank == src:
            return tensor[src].clone()
        else:
            return tensor.clone()

    tensor = tensor.contiguous()

    if rank == src:
        assert tensor.shape[0] == world_size, f"Source tensor must have {world_size} chunks"
        chunk_shape = tensor.shape[1:]
    else:
        chunk_shape = tensor.shape

    _get_ext()  # Ensure extension is loaded
    
    buf, hdl, ptrs = _get_symm_state(chunk_shape, tensor.dtype, tensor.device)
    chunk_bytes = buf.numel() * buf.element_size()
    
    # Synchronization 1: Ensure peers have finished consuming the symmetric buffer
    # from any previous operations before source starts overwriting it.
    c1 = _next_channel()
    hdl.barrier(channel=c1)
    
    if rank == src:
        _get_ext().launch_scatter_push(tensor, ptrs, chunk_bytes, world_size, src)
        
    # Synchronization 2: Block peers from reading until source has completed pushing
    # data to their symmetric buffer partitions.
    c2 = _next_channel()
    hdl.barrier(channel=c2)
    
    if rank == src:
        out = tensor[src].clone()
    else:
        out = buf.clone()
        
    return out