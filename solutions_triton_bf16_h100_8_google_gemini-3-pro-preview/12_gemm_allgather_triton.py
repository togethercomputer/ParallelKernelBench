"""
Strategy:
- Replace the NCCL `all_gather` with a custom UVA-based P2P gather via PyTorch Symmetric Memory.
- Maximize compute-communication overlap by splitting the M dimension into pipelined chunks.
- While cuBLAS computes the GEMM `C_chunk = A_global_chunk @ B` on the main stream, a custom vectorized CUDA kernel concurrently fetches the next chunk of `A` from all peers directly into `A_global` over NVLink using a separate stream.
- This effectively hides the bandwidth-bound communication of A behind the math-heavy GEMM.
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
#include <vector>

struct RemotePtrs {
    const void* ptrs[16];
};

__global__ void allgather_a_kernel_16byte(
    RemotePtrs remote,
    void* __restrict__ out_global,
    int64_t M,
    int64_t K_local_bytes,
    int world_size
) {
    int64_t K_local_vec = K_local_bytes / 16;
    int64_t K_global_vec = (world_size * K_local_bytes) / 16;
    int64_t total_vecs = M * K_local_vec * world_size;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_vecs) {
        int64_t peer = idx / (M * K_local_vec);
        int64_t local_idx = idx % (M * K_local_vec);
        
        int64_t m = local_idx / K_local_vec;
        int64_t k_vec = local_idx % K_local_vec;
        
        int64_t out_vec_idx = m * K_global_vec + peer * K_local_vec + k_vec;
        
        const float4* src = reinterpret_cast<const float4*>(remote.ptrs[peer]);
        float4* dst = reinterpret_cast<float4*>(out_global);
        
        dst[out_vec_idx] = src[local_idx];
    }
}

__global__ void allgather_a_kernel_byte2(
    RemotePtrs remote,
    void* __restrict__ out_global,
    int64_t M,
    int64_t K_local_words,
    int world_size
) {
    int64_t K_global_words = world_size * K_local_words;
    int64_t total_words = M * K_local_words * world_size;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_words) {
        int64_t peer = idx / (M * K_local_words);
        int64_t local_idx = idx % (M * K_local_words);
        
        int64_t m = local_idx / K_local_words;
        int64_t k_vec = local_idx % K_local_words;
        
        int64_t out_idx = m * K_global_words + peer * K_local_words + k_vec;
        
        const uint16_t* src = reinterpret_cast<const uint16_t*>(remote.ptrs[peer]);
        uint16_t* dst = reinterpret_cast<uint16_t*>(out_global);
        
        dst[out_idx] = src[local_idx];
    }
}

void allgather_a_forward(
    std::vector<int64_t> remote_ptrs_int,
    torch::Tensor out_global,
    int64_t M,
    int64_t K_local,
    int world_size,
    int64_t stream_ptr
) {
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    TORCH_CHECK(world_size <= 16, "world_size > 16 not supported");
    RemotePtrs remote;
    for (int i = 0; i < world_size; i++) {
        remote.ptrs[i] = reinterpret_cast<const void*>(remote_ptrs_int[i]);
    }
    
    int64_t element_size = out_global.element_size();
    int64_t K_local_bytes = K_local * element_size;
    
    if (K_local_bytes % 16 == 0 && (reinterpret_cast<uintptr_t>(out_global.data_ptr()) % 16) == 0) {
        int64_t total_vecs = M * (K_local_bytes / 16) * world_size;
        int threads = 256;
        int blocks = (total_vecs + threads - 1) / threads;
        if (blocks > 0) {
            allgather_a_kernel_16byte<<<blocks, threads, 0, stream>>>(
                remote, out_global.data_ptr(), M, K_local_bytes, world_size
            );
        }
    } else {
        int64_t total_elements = M * (K_local_bytes / 2) * world_size;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        if (blocks > 0) {
            allgather_a_kernel_byte2<<<blocks, threads, 0, stream>>>(
                remote, out_global.data_ptr(), M, K_local_bytes / 2, world_size
            );
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allgather_a_forward", &allgather_a_forward, "UVA allgather A forward");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("allgather_gemm_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(size: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    key = (size, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]

    buf = symm_mem.empty(size, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    
    M, K_local = A_local.shape
    K_B, N = B.shape
    K_global = world_size * K_local
    assert K_B == K_global, f"B must have K dimension = world_size * K_local: {K_B} != {world_size} * {K_local}"
    
    C = torch.empty((M, N), dtype=A_local.dtype, device=A_local.device)
    if M == 0 or N == 0:
        return C
        
    buf, hdl = _get_symm_state(M * K_local, A_local.dtype, A_local.device)
    
    # Copy A_local to the symmetric memory buffer so peers can access it via UVA
    buf.view(M, K_local).copy_(A_local)
    
    # Wait for all ranks to populate their symmetric memory
    hdl.barrier(channel=0)
    
    A_global = torch.empty((M, K_global), dtype=A_local.dtype, device=A_local.device)
    
    num_chunks = 2 if M >= 256 else 1
    chunk_size = (M + num_chunks - 1) // num_chunks
    
    compute_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    
    # Ensure copy_stream does not start reading peers' buffers before the barrier is crossed
    copy_stream.wait_stream(compute_stream)
    
    remote_ptrs = [int(hdl.buffer_ptrs[i]) for i in range(world_size)]
    element_size = A_local.element_size()
    
    def get_chunk_bounds(i):
        return min(i * chunk_size, M), min((i + 1) * chunk_size, M)
        
    def dispatch_copy(i):
        m_start, m_end = get_chunk_bounds(i)
        m_chunk = m_end - m_start
        if m_chunk <= 0:
            return
            
        offset_bytes = m_start * K_local * element_size
        chunk_ptrs = [ptr + offset_bytes for ptr in remote_ptrs]
        
        with torch.cuda.stream(copy_stream):
            ext.allgather_a_forward(
                chunk_ptrs,
                A_global[m_start:m_end],
                m_chunk,
                K_local,
                world_size,
                copy_stream.cuda_stream
            )

    # Pre-queue the first copy chunk on copy_stream
    dispatch_copy(0)
    
    for i in range(num_chunks):
        m_start, m_end = get_chunk_bounds(i)
        if m_start >= M:
            break
            
        # Wait for the copy of chunk i to complete before computing on it
        compute_stream.wait_stream(copy_stream)
        
        # Pipeline: Queue the copy of chunk i+1 concurrently while computing chunk i
        if i + 1 < num_chunks:
            dispatch_copy(i + 1)
            
        # Compute GEMM for chunk i on compute_stream (cuBLAS)
        torch.matmul(A_global[m_start:m_end], B, out=C[m_start:m_end])
        
    # Global barrier ensures no rank returns and overwrites their A_local / symm_mem
    # in a subsequent iteration before other ranks have finished fetching it.
    dist.barrier()
    
    return C