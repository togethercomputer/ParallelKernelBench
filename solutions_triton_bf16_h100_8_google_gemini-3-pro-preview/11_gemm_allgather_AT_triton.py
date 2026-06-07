"""
Strategy:
- **Device-side Communication**: We replace host-driven NCCL `all_gather` with custom UVA peer-to-peer copies. Each rank exposes its `A_local` shard via `torch.distributed._symmetric_memory` to enable direct NVLink fetches.
- **Compute-Communication Overlap**: We slice the computation along the $M$ dimension into $W$ row-chunks. We use double buffering to pipeline the memory fetch and the GEMM math. 
- **Fused Execution**: While a custom C++ CUDA kernel asynchronously copies the required row-chunk of $A$ from all peers over NVLink, a custom Triton GEMM computes the previous chunk, completely hiding the communication latency behind the Tensor Core math.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# 1. Custom C++ CUDA Extension for UVA Gathering
# -----------------------------------------------------------------------------
CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void gather_chunks_kernel(
    const int64_t* __restrict__ symm_ptrs,
    scalar_t* __restrict__ dst,
    int64_t step_offset,
    int64_t elements_per_rank
) {
    int rank = blockIdx.y;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements_per_rank) {
        const scalar_t* src = reinterpret_cast<const scalar_t*>(symm_ptrs[rank]);
        dst[rank * elements_per_rank + idx] = src[step_offset + idx];
    }
}

void gather_chunks(
    torch::Tensor symm_ptrs,
    torch::Tensor dst,
    int64_t step_offset,
    int64_t elements_per_rank,
    int world_size,
    int64_t stream_ptr
) {
    int threads = 256;
    int blocks_x = (elements_per_rank + threads - 1) / threads;
    dim3 blocks(blocks_x, world_size);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dst.scalar_type(), "gather_chunks", ([&] {
        gather_chunks_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            symm_ptrs.data_ptr<int64_t>(),
            dst.data_ptr<scalar_t>(),
            step_offset,
            elements_per_rank
        );
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_chunks", &gather_chunks, "Gather A chunks via UVA pointers");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_gather_ext", CUDA_SRC)
    return _ext

# -----------------------------------------------------------------------------
# 2. Custom Triton Kernel for Fused Chunked GEMM
# -----------------------------------------------------------------------------
@triton.jit
def gemm_chunk_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K_local, W,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for r in range(W):
        a_base = A_ptr + r * M * K_local
        b_base = B_ptr + r * K_local * stride_bk
        
        for k in range(0, K_local, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            
            a_ptrs = a_base + (offs_m[:, None] * stride_am + offs_k[None, :])
            b_ptrs = b_base + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
            
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_local)
            b_mask = (offs_k[:, None] < K_local) & (offs_n[None, :] < N)
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            acc += tl.dot(a, b)
            
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)

# -----------------------------------------------------------------------------
# 3. Global Caches for minimal runtime overhead
# -----------------------------------------------------------------------------
_symm_cache = None
def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache = {"n": n, "dtype": dtype, "device": device, "buf": buf, "hdl": hdl}
    return buf, hdl

_buffer_cache = None
def _get_buffers(W, M_chunk, K_local, dtype, device):
    global _buffer_cache
    shape = (W, M_chunk, K_local)
    if _buffer_cache is not None:
        if _buffer_cache[0].shape == shape and _buffer_cache[0].dtype == dtype and _buffer_cache[0].device == device:
            return _buffer_cache
    _buffer_cache = [torch.empty(shape, device=device, dtype=dtype) for _ in range(2)]
    return _buffer_cache

_stream_cache = None
def _get_stream_and_events(W):
    global _stream_cache
    if _stream_cache is not None and len(_stream_cache[1]) == W:
        return _stream_cache
    stream = torch.cuda.Stream()
    events_copy = [torch.cuda.Event() for _ in range(W)]
    events_compute = [torch.cuda.Event() for _ in range(W)]
    _stream_cache = (stream, events_copy, events_compute)
    return _stream_cache


@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    rank = dist.get_rank()
    W = dist.get_world_size()
    
    M, K_local = A_local.shape
    K_B, N = B.shape
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    
    # 1. Prepare symmetric memory
    buf_A, hdl_A = _get_symm_state(A_local.numel(), A_local.dtype, A_local.device)
    buf_A.copy_(A_local.contiguous().view(-1))
    hdl_A.barrier(channel=0)
    
    symm_ptrs = torch.tensor(hdl_A.buffer_ptrs, dtype=torch.int64, device=A_local.device)
    C = torch.empty((M, N), device=A_local.device, dtype=A_local.dtype)
    B = B.contiguous()
    
    # 2. Prepare pipeline structures
    M_chunk = (M + W - 1) // W
    buffer = _get_buffers(W, M_chunk, K_local, A_local.dtype, A_local.device)
    copy_stream, copy_events, compute_events = _get_stream_and_events(W)
    compute_stream = torch.cuda.current_stream()
    
    # 3. Pipelined Chunked Execution
    for step in range(W):
        buf_idx = step % 2
        
        start_m = step * M_chunk
        end_m = min(M, start_m + M_chunk)
        current_m = end_m - start_m
        
        if current_m <= 0:
            continue

        # Prevent overwriting a buffer that is currently being read by the compute stream
        if step >= 2:
            copy_stream.wait_event(compute_events[step - 2])
            
        # [Stream 1] Asynchronous chunk copy via NVLink
        with torch.cuda.stream(copy_stream):
            step_offset = start_m * K_local
            elements_per_rank = current_m * K_local
            
            ext.gather_chunks(
                symm_ptrs,
                buffer[buf_idx],
                step_offset,
                elements_per_rank,
                W,
                copy_stream.cuda_stream
            )
            copy_events[step].record(copy_stream)

        # [Stream 2] Wait for current chunk to finish fetching, then execute compute
        compute_stream.wait_event(copy_events[step])
        
        grid = ((current_m + 127) // 128, (N + 127) // 128)
        C_ptr = C[start_m : end_m, :]
        
        gemm_chunk_kernel[grid](
            buffer[buf_idx], B, C_ptr,
            current_m, N, K_local, W,
            K_local, 1, 
            N, 1,       
            N, 1,       
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=3
        )
        compute_events[step].record(compute_stream)

    # 4. Cleanup & Synchronize across pipeline and loops
    compute_stream.wait_stream(copy_stream)
    dist.barrier()
    
    return C