"""
Strategy:
- **Device-side communication & fusion**: Bypasses `dist.all_gather` and the subsequent `torch.cat`. Uses symmetric memory and a custom CUDA kernel that directly pulls `A_local` shards from all peers over NVLink and natively writes them into their final contiguous positions in `A_global`.
- **Compute-communication overlap**: Partitions the M-dimension of the GEMM into chunks. While the main stream computes the dense Tensor Core matmul for chunk `c`, a concurrent CUDA stream pulls the symmetric memory shards for chunk `c+1`. This perfectly hides communication latency without destroying arithmetic intensity (which would happen if we partitioned along K).
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

template <typename T>
__global__ void gather_concat_kernel_row_peer(
    const uint64_t* __restrict__ peer_ptrs,
    T* __restrict__ out,
    int M_chunk,
    int K_local,
    int world_size,
    int64_t peer_offset_elements
) {
    int total_tasks = M_chunk * world_size;
    int task_idx = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (task_idx < total_tasks) {
        int row = task_idx / world_size;
        int peer = task_idx % world_size;
        
        const T* peer_buf = reinterpret_cast<const T*>(peer_ptrs[peer]) + peer_offset_elements;
        T* out_row = out + row * (K_local * world_size) + peer * K_local;
        const T* in_row = peer_buf + row * K_local;
        
        for (int col = threadIdx.x; col < K_local; col += blockDim.x) {
            out_row[col] = in_row[col];
        }
    }
}

__global__ void gather_concat_kernel_128_row_peer(
    const uint64_t* __restrict__ peer_ptrs,
    int4* __restrict__ out,
    int M_chunk,
    int K_local_128,
    int world_size,
    int64_t peer_offset_128
) {
    int total_tasks = M_chunk * world_size;
    int task_idx = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (task_idx < total_tasks) {
        int row = task_idx / world_size;
        int peer = task_idx % world_size;
        
        const int4* peer_buf = reinterpret_cast<const int4*>(peer_ptrs[peer]) + peer_offset_128;
        int4* out_row = out + row * (K_local_128 * world_size) + peer * K_local_128;
        const int4* in_row = peer_buf + row * K_local_128;
        
        for (int col = threadIdx.x; col < K_local_128; col += blockDim.x) {
            out_row[col] = in_row[col];
        }
    }
}

void launch_gather_concat(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor out,
    int M_chunk,
    int K_local,
    int world_size,
    int element_size,
    int64_t peer_offset_elements
) {
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    int64_t total_bytes = (int64_t)M_chunk * K_local * world_size * element_size;
    
    if (total_bytes % 16 == 0 && (K_local * element_size) % 16 == 0 && (peer_offset_elements * element_size) % 16 == 0) {
        int K_local_128 = (K_local * element_size) / 16;
        int64_t peer_offset_128 = (peer_offset_elements * element_size) / 16;
        
        dim3 block(32, 8); // 32 threads for inner copy, 8 parallel tasks
        int total_tasks = M_chunk * world_size;
        int grid = (total_tasks + block.y - 1) / block.y;
        
        gather_concat_kernel_128_row_peer<<<grid, block, 0, stream>>>(
            d_ptrs, reinterpret_cast<int4*>(out.data_ptr()), M_chunk, K_local_128, world_size, peer_offset_128);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        dim3 block(32, 8);
        int total_tasks = M_chunk * world_size;
        int grid = (total_tasks + block.y - 1) / block.y;
        
        if (element_size == 2) {
            gather_concat_kernel_row_peer<int16_t><<<grid, block, 0, stream>>>(
                d_ptrs, reinterpret_cast<int16_t*>(out.data_ptr()), M_chunk, K_local, world_size, peer_offset_elements);
        } else if (element_size == 4) {
            gather_concat_kernel_row_peer<int32_t><<<grid, block, 0, stream>>>(
                d_ptrs, reinterpret_cast<int32_t*>(out.data_ptr()), M_chunk, K_local, world_size, peer_offset_elements);
        } else {
            gather_concat_kernel_row_peer<int8_t><<<grid, block, 0, stream>>>(
                d_ptrs, reinterpret_cast<int8_t*>(out.data_ptr()), M_chunk * element_size, K_local * element_size, world_size, peer_offset_elements * element_size);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_concat", &launch_gather_concat, "P2P fetch and concatenate chunks");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("p2p_gather_concat_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    key = (n, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
            
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]

@torch.no_grad()
def solution(A_local: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    world_size = dist.get_world_size()
    
    M, K_local = A_local.shape
    K_B, N = B.shape
    K_global = world_size * K_local
    assert K_B == K_global, f"B must have K dimension = world_size * K_local"
    
    # Pre-compile
    _get_ext()
    
    # Expose A_local layout to peers
    buf, hdl, ptrs_tensor = _get_symm_state(M * K_local, A_local.dtype, A_local.device)
    
    # Copy data synchronously to symm_mem so peers can safely pull it
    buf.copy_(A_local.contiguous().flatten())
    hdl.barrier(channel=0)
    
    # Barrier completes on current stream. Create an event so the comm stream knows it's safe to start pulling.
    comp_stream = torch.cuda.current_stream()
    sync_event = torch.cuda.Event()
    sync_event.record(comp_stream)
    
    comm_stream = torch.cuda.Stream()
    comm_stream.wait_event(sync_event)
    
    # Partition M intelligently into chunks to overlap computation with communication
    NUM_CHUNKS = 1
    if M % 4 == 0 and (M // 4) >= 128:
        NUM_CHUNKS = 4
    elif M % 2 == 0 and (M // 2) >= 128:
        NUM_CHUNKS = 2
        
    M_chunk = M // NUM_CHUNKS
    
    # Pre-allocate fully materialized targets 
    A_global = torch.empty((M, K_global), dtype=A_local.dtype, device=A_local.device)
    C_out = torch.empty((M, N), dtype=A_local.dtype, device=A_local.device)
    
    comm_events = [torch.cuda.Event() for _ in range(NUM_CHUNKS)]
    
    # Launch purely pipelined communication asynchronously
    for c in range(NUM_CHUNKS):
        with torch.cuda.stream(comm_stream):
            offset = c * M_chunk * K_local
            out_slice = A_global[c * M_chunk : (c + 1) * M_chunk, :]
            _get_ext().launch_gather_concat(
                ptrs_tensor,
                out_slice,
                M_chunk,
                K_local,
                world_size,
                A_local.element_size(),
                offset
            )
            comm_events[c].record(comm_stream)
            
    # Launch pipelined compute as chunks become available
    for c in range(NUM_CHUNKS):
        # Wait for the chunk's communication to complete
        comm_events[c].wait(comp_stream)
        out_slice = A_global[c * M_chunk : (c + 1) * M_chunk, :]
        C_chunk = C_out[c * M_chunk : (c + 1) * M_chunk, :]
        torch.matmul(out_slice, B.contiguous(), out=C_chunk)
        
    # Prevent successive steps / functions from un-registering or overwriting active buffers
    dist.barrier()
    
    return C_out