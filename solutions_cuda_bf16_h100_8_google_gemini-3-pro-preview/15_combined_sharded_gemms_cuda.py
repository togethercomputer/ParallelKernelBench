"""
Strategy:
- **Algorithmic Reduction:** The reference code all-gathers `x` and computes the MLP on the full sequence, then throws away `(world_size - 1)/world_size` of the result via a masked reduce-scatter. We replace this entire sequence with a direct **All-to-All** routing of the required `M_local` sequence chunks to each rank, followed by the MLP computed only on the local shard. This mathematically identical approach reduces FLOPs and communication volume by `world_size`x.
- **Device-Side P2P Symmetric Memory:** We eliminate the PyTorch collective overhead by allocating an `x_full_loc` receiver buffer via `torch.distributed._symmetric_memory` and explicitly scattering the input tensors directly into peer memory using a custom vectorized CUDA kernel over NVLink.
- **Compute-Communication Pipelining:** We split the sequence dimension `M_local` into multiple chunks. Using dual CUDA streams and atomic device-side barriers (`hdl.barrier`), we overlap the cross-NVLink transfer of chunk `c+1` with the local GEMM and SiLU computations of chunk `c`, hiding the network transfer time behind the compute.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void all_to_all_chunk_kernel(
    const __nv_bfloat16* __restrict__ x_local,
    const long long* __restrict__ dest_ptrs,
    int M_local,
    int H_local,
    int H,
    int rank,
    int world_size,
    int chunk_start,
    int chunk_size
) {
    int64_t total_elements = (int64_t)chunk_size * H_local * world_size;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Fast vectorized path (128-bit loads/stores) when aligned
    if (H_local % 8 == 0) {
        int64_t total_vecs = total_elements / 8;
        for (int64_t idx = tid; idx < total_vecs; idx += gridDim.x * blockDim.x) {
            int64_t elem_idx = idx * 8;
            int dest_rank = elem_idx / (chunk_size * H_local);
            int rem = elem_idx % (chunk_size * H_local);
            int r = rem / H_local; 
            int c = rem % H_local;
            
            int src_r = dest_rank * M_local + chunk_start + r;
            int dst_r = chunk_start + r;
            
            const uint4* src = (const uint4*)x_local;
            uint4* dest = (uint4*)dest_ptrs[dest_rank];
            
            int src_vec_idx = (src_r * H_local + c) / 8;
            int dst_vec_idx = (dst_r * H + rank * H_local + c) / 8;
            
            dest[dst_vec_idx] = src[src_vec_idx];
        }
    } else {
        // Scalar fallback path
        for (int64_t idx = tid; idx < total_elements; idx += gridDim.x * blockDim.x) {
            int dest_rank = idx / (chunk_size * H_local);
            int rem = idx % (chunk_size * H_local);
            int r = rem / H_local;
            int c = rem % H_local;
            
            int src_r = dest_rank * M_local + chunk_start + r;
            int dst_r = chunk_start + r;
            
            const __nv_bfloat16* src = x_local;
            __nv_bfloat16* dest = (__nv_bfloat16*)dest_ptrs[dest_rank];
            
            dest[dst_r * H + rank * H_local + c] = src[src_r * H_local + c];
        }
    }
}

void launch_all_to_all_chunk(
    torch::Tensor x_local,
    torch::Tensor ptrs_tensor,
    int M_local,
    int H_local,
    int H,
    int rank,
    int world_size,
    int chunk_start,
    int chunk_size
) {
    int64_t total_elements = (int64_t)chunk_size * H_local * world_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (H_local % 8 == 0) {
        blocks = ((total_elements / 8) + threads - 1) / threads;
    }
    if (blocks == 0) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    all_to_all_chunk_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_local.data_ptr<at::BFloat16>()),
        reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>()),
        M_local,
        H_local,
        H,
        rank,
        world_size,
        chunk_start,
        chunk_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_all_to_all_chunk", &launch_all_to_all_chunk, "All-to-all chunk copy kernel");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("all_to_all_pipeline_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_buf(M_local, H, dtype, device):
    global _symm_cache
    key = (M_local, H, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty((M_local, H), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]

_stream_cache = None
def _get_stream1():
    global _stream_cache
    if _stream_cache is None:
        _stream_cache = torch.cuda.Stream()
    return _stream_cache

_events_cache = []
def _get_events(n):
    global _events_cache
    while len(_events_cache) < n:
        _events_cache.append(torch.cuda.Event())
    return _events_cache[:n]


@torch.no_grad()
def solution(
    x_local: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert x_local.is_cuda and W1.is_cuda and W2.is_cuda, "Inputs must be CUDA tensors"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if _ext is None:
        if rank == 0:
            _get_ext()
        dist.barrier()
        _get_ext()

    M, H_local = x_local.shape
    H, ffn_dim = W1.shape
    M_local = M // world_size

    x_local = x_local.contiguous()

    # Allocate symmetric receiver buffer for this rank's block of the sequence
    buf, hdl, ptrs_tensor = _get_symm_buf(M_local, H, x_local.dtype, x_local.device)
    
    stream1 = _get_stream1()
    # Make the comm stream wait for any pending x_local preparation on the default stream
    stream1.wait_stream(torch.cuda.current_stream())

    # Pipeline chunks for overlapping matmul computation with NVLink peer-to-peer copies
    num_chunks = 2 if (M_local >= 128 and M_local % 2 == 0) else 1
    chunk_size = M_local // num_chunks
    events = _get_events(num_chunks)

    # Launch communication sequence entirely on stream1
    with torch.cuda.stream(stream1):
        # Strict pre-write sync to ensure peers finished matmuls from previous steps
        hdl.barrier(channel=0)
        
        for c in range(num_chunks):
            chunk_start = c * chunk_size
            _get_ext().launch_all_to_all_chunk(
                x_local, ptrs_tensor, M_local, H_local, H,
                rank, world_size, chunk_start, chunk_size
            )
            # Ensure chunk c has fully arrived on all ranks before computation
            hdl.barrier(channel=0)
            events[c].record(stream1)

    y_local = torch.empty((M_local, H), dtype=x_local.dtype, device=x_local.device)

    # Launch computation on the default stream, synced with stream1 chunks
    for c in range(num_chunks):
        torch.cuda.current_stream().wait_event(events[c])
        chunk_start = c * chunk_size
        
        # Pull the fully assembled row block of x_full
        x_chunk = buf[chunk_start : chunk_start + chunk_size, :]
        
        # Execute MLP exclusively on this rank's required sequence shard
        z_chunk = torch.matmul(x_chunk, W1)
        a_chunk = F.silu(z_chunk)
        y_local[chunk_start : chunk_start + chunk_size, :] = torch.matmul(a_chunk, W2)

    # Prevent stream1 resources from being prematurely cleaned up
    torch.cuda.current_stream().wait_stream(stream1)

    # Matching the reference spec synchronization
    dist.barrier()
    return y_local