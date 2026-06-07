"""
Distributed GEMM with All-Gather (scatter).

Strategy:
- Maximize overlap by chunking the M dimension. Local compute (A_chunk @ B) is overlapped 
  with asynchronous peer-to-peer multicast of the previous chunk.
- Avoids NCCL by using a custom CUDA kernel that writes directly to peers' memory via 
  UVA and symmetric memory buffers.
- Exploits Hopper NVSwitch `multimem.st` (Hardware Broadcast) to write the output to all 
  peers simultaneously with a single instruction, drastically reducing NVLink traffic.
- Leverages device-side barriers (`hdl.barrier`) for lightning-fast GPU stream synchronization.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void multimem_st_128(const uint64_t* addr, uint4 val) {
    // Hardware broadcast: write 128 bits to all multicast peers simultaneously
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
        : "memory");
}

__global__ void push_multimem_128(
    const uint4* __restrict__ src,
    uint64_t multicast_ptr,
    int chunk_rows,
    int N_local_128,
    int N_128,
    int start_col_128
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = chunk_rows * N_local_128;
    if (idx >= total_elements) return;
    
    int row = idx / N_local_128;
    int col = idx % N_local_128;
    
    uint4 val = src[idx];
    int out_offset = row * N_128 + start_col_128 + col;
    
    // cast to uint64_t* which is 8 bytes, so we multiply offset by 2 to step by 16 bytes
    uint64_t* dst = reinterpret_cast<uint64_t*>(multicast_ptr) + out_offset * 2;
    multimem_st_128(dst, val);
}

__global__ void push_p2p_128(
    const uint4* __restrict__ src,
    const uint64_t* __restrict__ peer_ptrs,
    int world_size,
    int chunk_rows,
    int N_local_128,
    int N_128,
    int start_col_128
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = chunk_rows * N_local_128;
    if (idx >= total_elements) return;
    
    int row = idx / N_local_128;
    int col = idx % N_local_128;
    
    uint4 val = src[idx];
    int out_offset = row * N_128 + start_col_128 + col;
    
    #pragma unroll
    for (int p = 0; p < world_size; ++p) {
        uint4* dst = reinterpret_cast<uint4*>(peer_ptrs[p]);
        dst[out_offset] = val;
    }
}

template<typename T>
__global__ void push_p2p_scalar(
    const T* __restrict__ src,
    const uint64_t* __restrict__ peer_ptrs,
    int world_size,
    int chunk_rows,
    int N_local,
    int N,
    int start_col
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = chunk_rows * N_local;
    if (idx >= total_elements) return;
    
    int row = idx / N_local;
    int col = idx % N_local;
    
    T val = src[idx];
    int out_offset = row * N + start_col + col;
    
    #pragma unroll
    for (int p = 0; p < world_size; ++p) {
        T* dst = reinterpret_cast<T*>(peer_ptrs[p]);
        dst[out_offset] = val;
    }
}

void launch_push(
    torch::Tensor src,
    torch::Tensor peer_ptrs,
    int64_t multicast_ptr_int,
    int chunk_rows,
    int N_local,
    int N,
    int start_col
) {
    int element_size = src.element_size();
    bool use_128 = ((N_local * element_size) % 16 == 0) && 
                   ((N * element_size) % 16 == 0) && 
                   ((start_col * element_size) % 16 == 0);
                   
    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uint64_t multicast_ptr = static_cast<uint64_t>(multicast_ptr_int);
    
    if (use_128) {
        int N_local_128 = (N_local * element_size) / 16;
        int N_128 = (N * element_size) / 16;
        int start_col_128 = (start_col * element_size) / 16;
        int total = chunk_rows * N_local_128;
        if (total == 0) return;
        
        int blocks = (total + threads - 1) / threads;
        const uint4* src_ptr = reinterpret_cast<const uint4*>(src.data_ptr());
        
        if (multicast_ptr != 0) {
            push_multimem_128<<<blocks, threads, 0, stream>>>(
                src_ptr, multicast_ptr, chunk_rows, N_local_128, N_128, start_col_128
            );
        } else {
            const uint64_t* ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>());
            int world_size = peer_ptrs.size(0);
            push_p2p_128<<<blocks, threads, 0, stream>>>(
                src_ptr, ptrs, world_size, chunk_rows, N_local_128, N_128, start_col_128
            );
        }
    } else {
        int total = chunk_rows * N_local;
        if (total == 0) return;
        
        int blocks = (total + threads - 1) / threads;
        const uint64_t* ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>());
        int world_size = peer_ptrs.size(0);
        
        if (element_size == 4) {
            push_p2p_scalar<float><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const float*>(src.data_ptr()), ptrs, world_size, chunk_rows, N_local, N, start_col
            );
        } else if (element_size == 2) {
            push_p2p_scalar<uint16_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint16_t*>(src.data_ptr()), ptrs, world_size, chunk_rows, N_local, N, start_col
            );
        } else {
            push_p2p_scalar<uint8_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(src.data_ptr()), ptrs, world_size, chunk_rows, N_local*element_size, N*element_size, start_col*element_size
            );
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_push", &launch_push, "Asynchronously push chunk of C_local to all peers");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        if dist.get_rank() == 0:
            _ext = compile_cuda_extension("gemm_allscatter_push_ext", CUDA_SRC)
        dist.barrier()
        if dist.get_rank() != 0:
            _ext = compile_cuda_extension("gemm_allscatter_push_ext", CUDA_SRC)
    return _ext

_resource_cache = {}

def _get_resources(M: int, N_local: int, N: int, dtype: torch.dtype, device: torch.device):
    key = (M, N_local, N, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]

    # Global C symmetric buffer. Written natively via P2P/Multimem ST
    C_symm = symm_mem.empty((M, N), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(C_symm, dist.group.WORLD)
    
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    multicast_ptr = getattr(hdl, "multicast_ptr", None)
    multicast_ptr_int = int(multicast_ptr) if multicast_ptr is not None else 0
    
    # Preallocate compute target buffer to avoid recurrent allocations inside hot loop
    C_local_buffer = torch.empty((M, N_local), dtype=dtype, device=device)
    
    comm_stream = torch.cuda.Stream(device=device)
    events = [torch.cuda.Event() for _ in range(64)]
    
    res = (C_symm, hdl, ptrs_tensor, multicast_ptr_int, C_local_buffer, comm_stream, events)
    _resource_cache[key] = res
    return res

@torch.no_grad()
def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return torch.matmul(A, B)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, K = A.shape
    K_B, N_local = B.shape
    N = world_size * N_local

    _get_ext()
    
    (C_symm, hdl, ptrs_tensor, multicast_ptr_int, 
     C_local_buffer, comm_stream, events) = _get_resources(M, N_local, N, A.dtype, A.device)

    compute_stream = torch.cuda.current_stream()
    
    # Determine schedule (chunking) to allow comm stream to hide behind compute stream
    chunk_size = max(256, M // 4)
    if M <= 512:
        chunk_size = M
    num_chunks = (M + chunk_size - 1) // chunk_size
    if num_chunks > len(events):
        chunk_size = (M + len(events) - 1) // len(events)
        num_chunks = len(events)

    start_col = rank * N_local

    for i in range(num_chunks):
        start_m = i * chunk_size
        end_m = min((i + 1) * chunk_size, M)
        chunk_rows = end_m - start_m
        if chunk_rows <= 0:
            break
        
        # Slicing creates views into preallocated buffers
        A_chunk = A[start_m:end_m, :]
        C_local_chunk = C_local_buffer[start_m:end_m, :]
        
        # Step 1: Execute dense math on the main stream (uses Tensor Cores)
        torch.matmul(A_chunk, B, out=C_local_chunk)
        events[i].record(compute_stream)
        
        # Step 2: Push result to the rest of the world concurrently via separate stream
        comm_stream.wait_event(events[i])
        with torch.cuda.stream(comm_stream):
            _get_ext().launch_push(
                C_local_chunk,
                ptrs_tensor,
                multicast_ptr_int,
                chunk_rows,
                N_local,
                N,
                start_col
            )

    # Ensure compute stream tracks memory-copy stream
    compute_stream.wait_stream(comm_stream)

    # Fast hardware-assisted stream barrier 0: Ensure all peers finished pushing data out
    hdl.barrier(channel=0)

    # Once everyone is synchronized, form the local copy of the complete output
    out = C_symm.clone()

    # Fast hardware-assisted stream barrier 1: Ensure local reading logic finishes 
    # before returning, securing the `C_symm` buffer against being corrupted in the NEXT iteration
    hdl.barrier(channel=1)

    return out