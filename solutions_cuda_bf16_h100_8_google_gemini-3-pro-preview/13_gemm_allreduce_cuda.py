"""
Strategy:
1. **Device-side Communication**: Uses Hopper NVSwitch `multimem.ld_reduce` and `multimem.st` instructions to perform an in-network fused all-reduce broadcast directly on symmetric bfloat16 tensors. A fast P2P pointer-based fallback is implemented for other dtypes.
2. **Compute-Communication Overlap**: The GEMM is chunked along the M dimension. The `torch.matmul` computation (cuBLAS) is launched on the default stream, while the chunked all-reduce kernels operate asynchronously on a dedicated communication stream.
3. **Pipelining**: A reusable custom device-side blockwise barrier inside symmetric memory ensures safe, chunk-level synchronization without host intervention. This perfectly hides chunk $i$'s all-reduce latency behind chunk $i+1$'s GEMM computation.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Reusable blockwise barrier in Symmetric Memory
// ---------------------------------------------------------------------------
__device__ void blockwise_barrier_reusable(
    const uint64_t* __restrict__ sync_ptrs,
    uint64_t barrier_idx,
    uint64_t chunk_id,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    
    uint64_t local_base = sync_ptrs[rank];
    uint64_t remote_base = sync_ptrs[flat_tid];
    
    // offset in uint32_t elements
    // MAX_CHUNKS = 128, MAX_BLOCKS = 32
    uint64_t offset = (((barrier_idx * 128 + chunk_id) * gridDim.x) + block_id) * world_size;
    
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base) + offset + rank;
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base) + offset + flat_tid;
    
    uint32_t tmp;
    // Send signal (self-cleans from 0 to 1)
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp)
            : "l"(send_addr)
            : "memory");
    } while (tmp != 0u);
    
    // Wait signal and reset (self-cleans from 1 back to 0)
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp)
            : "l"(wait_addr)
            : "memory");
    } while (tmp != 1u);
}

// ---------------------------------------------------------------------------
// Multimem chunked all-reduce (BF16)
// ---------------------------------------------------------------------------
__global__ void multimem_allreduce_bf16_chunked_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ sync_ptrs,
    int64_t chunk_offset_128,
    int64_t chunk_numel_128,
    int world_size,
    int rank,
    int block_stride,
    int chunk_id
) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    
    // Wait for all ranks to complete GEMM for this chunk
    blockwise_barrier_reusable(sync_ptrs, 0, chunk_id, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank = (chunk_numel_128 + world_size - 1) / world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = block_id * block_stride;
         block_start < numel_per_rank;
         block_start += num_programs * block_stride)
    {
        const int64_t offsets = block_start + tid;
        if (offsets >= numel_per_rank) continue;
        
        const int64_t idx = rank * numel_per_rank + offsets;
        if (idx < chunk_numel_128) {
            uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + (chunk_offset_128 + idx) * 2;
            uint32_t x, y, z, w;
            asm volatile(
                "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(ptrs)
                : "memory");
            asm volatile(
                "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
                :
                : "l"(ptrs), "r"(x), "r"(y), "r"(z), "r"(w)
                : "memory");
        }
    }

    __syncthreads();
    // Barrier to ensure no early exits before multimem accesses complete
    blockwise_barrier_reusable(sync_ptrs, 1, chunk_id, block_id, rank, world_size);
}

// ---------------------------------------------------------------------------
// P2P chunked all-reduce (Fallback)
// ---------------------------------------------------------------------------
__global__ void p2p_allreduce_chunked_kernel(
    const uint64_t* __restrict__ sync_ptrs,
    const long long* __restrict__ ptrs,
    int64_t chunk_offset,
    int64_t chunk_numel,
    int world_size,
    int rank,
    int chunk_id,
    int dtype_enum
) {
    const uint64_t block_id = blockIdx.x;
    
    // Wait for all ranks to complete GEMM for this chunk
    blockwise_barrier_reusable(sync_ptrs, 0, chunk_id, block_id, rank, world_size);
    __syncthreads();

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < chunk_numel; idx += gridDim.x * blockDim.x) {
        if (dtype_enum == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < world_size; ++r) {
                const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
                sum += __bfloat162float(src[chunk_offset + idx]);
            }
            __nv_bfloat16* out = (__nv_bfloat16*)ptrs[rank];
            out[chunk_offset + idx] = __float2bfloat16(sum);
        } else if (dtype_enum == 1) {
            float sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < world_size; ++r) {
                const float* src = (const float*)ptrs[r];
                sum += src[chunk_offset + idx];
            }
            float* out = (float*)ptrs[rank];
            out[chunk_offset + idx] = sum;
        } else if (dtype_enum == 2) {
            float sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < world_size; ++r) {
                const __half* src = (const __half*)ptrs[r];
                sum += __half2float(src[chunk_offset + idx]);
            }
            __half* out = (__half*)ptrs[rank];
            out[chunk_offset + idx] = __float2half(sum);
        }
    }

    __syncthreads();
    // Barrier to ensure no rank modifies the symmetric buffer before others finish reading
    blockwise_barrier_reusable(sync_ptrs, 1, chunk_id, block_id, rank, world_size);
}

// ---------------------------------------------------------------------------
// Python Launchers
// ---------------------------------------------------------------------------
void launch_multimem_chunked(
    uint64_t multicast_ptr,
    torch::Tensor sync_ptrs_tensor,
    int64_t chunk_offset_128,
    int64_t chunk_numel_128,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride,
    int chunk_id,
    uint64_t stream_ptr
) {
    const uint64_t* d_sync = reinterpret_cast<const uint64_t*>(sync_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    multimem_allreduce_bf16_chunked_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_sync, chunk_offset_128, chunk_numel_128, world_size, rank, block_stride, chunk_id
    );
}

void launch_p2p_chunked(
    torch::Tensor sync_ptrs_tensor,
    torch::Tensor ptrs_tensor,
    int64_t chunk_offset,
    int64_t chunk_numel,
    int world_size,
    int rank,
    int chunk_id,
    int dtype_enum,
    uint64_t stream_ptr
) {
    const uint64_t* d_sync = reinterpret_cast<const uint64_t*>(sync_ptrs_tensor.data_ptr<int64_t>());
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    int threads = 512;
    int blocks = 32;
    if (chunk_numel < blocks * threads) {
        blocks = (chunk_numel + threads - 1) / threads;
        if (blocks == 0) blocks = 1;
    }
    
    p2p_allreduce_chunked_kernel<<<blocks, threads, 0, stream>>>(
        d_sync, d_ptrs, chunk_offset, chunk_numel, world_size, rank, chunk_id, dtype_enum
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_chunked", &launch_multimem_chunked);
    m.def("launch_p2p_chunked", &launch_p2p_chunked);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("chunked_gemm_allreduce", CUDA_SRC)
    return _ext

BYTES_PER_THREAD = 16
MAX_NUM_BLOCKS = 4
MAX_BLOCK_SIZE = 1024

def _multimem_launch_config(numel: int, world_size: int) -> tuple[int, int, int]:
    numel_per_thread = 8
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = max(32, world_size) # Guarantee enough threads for blockwise barriers
        while block_size < num_threads and block_size < MAX_BLOCK_SIZE:
            block_size *= 2
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min((num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE, MAX_NUM_BLOCKS)
    return num_blocks, block_size, block_size

_resource_cache = {}

def _get_resources(shape, dtype, device, world_size):
    key = (shape, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]
        
    C_symm = symm_mem.empty(shape, device=device, dtype=dtype)
    C_hdl = symm_mem.rendezvous(C_symm, dist.group.WORLD)
    C_ptrs = torch.tensor(C_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    # Preallocate self-cleaning blockwise barriers: 
    # Max barriers = 2, Max chunks = 128, Max blocks = 32
    sync_numel = 2 * 128 * 32 * world_size
    sync_buf = symm_mem.empty((sync_numel,), device=device, dtype=torch.int32)
    sync_buf.zero_() # Cleared once at creation; barriers handle local 1->0 cleanup
    sync_hdl = symm_mem.rendezvous(sync_buf, dist.group.WORLD)
    sync_ptrs = torch.tensor(sync_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    comm_stream = torch.cuda.Stream()
    events = [torch.cuda.Event() for _ in range(128)]
    
    res = {
        "C_symm": C_symm, "C_hdl": C_hdl, "C_ptrs": C_ptrs,
        "sync_buf": sync_buf, "sync_ptrs": sync_ptrs,
        "comm_stream": comm_stream, "events": events
    }
    _resource_cache[key] = res
    return res

def get_num_chunks(M: int) -> int:
    # Heuristics for overlapping
    if M <= 512:
        return 1
    elif M <= 2048:
        return 2
    else:
        return min(4, (M + 1023) // 1024)

@torch.no_grad()
def solution(A_local: torch.Tensor, B_local: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return torch.matmul(A_local, B_local)

    M, K = A_local.shape
    K_B, N = B_local.shape
    assert K == K_B
    
    if M == 0 or N == 0 or K == 0:
        C = torch.matmul(A_local, B_local)
        dist.all_reduce(C, op=dist.ReduceOp.SUM)
        return C

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Pre-compile on rank 0 sequentially to prevent NCCL/CUDA timeout issues
    if rank == 0:
        _get_ext()
    dist.barrier()
    
    num_chunks = get_num_chunks(M)
    chunk_size = (M + num_chunks - 1) // num_chunks
    num_chunks = (M + chunk_size - 1) // chunk_size # Evict empty chunks
    
    res = _get_resources((M, N), A_local.dtype, A_local.device, world_size)
    C_symm = res["C_symm"]
    C_hdl = res["C_hdl"]
    C_ptrs = res["C_ptrs"]
    sync_ptrs = res["sync_ptrs"]
    comm_stream = res["comm_stream"]
    events = res["events"]
    
    A_local = A_local.contiguous()
    B_local = B_local.contiguous()
    
    use_multimem = (A_local.dtype == torch.bfloat16) and (C_symm.numel() % 8 == 0) and hasattr(C_hdl, "multicast_ptr")
    if use_multimem:
        multicast_ptr = int(C_hdl.multicast_ptr)
        
    for i in range(num_chunks):
        start_m = i * chunk_size
        end_m = min(M, (i + 1) * chunk_size)
        if start_m >= M:
            break
            
        chunk_m = end_m - start_m
        chunk_numel = chunk_m * N
        chunk_offset = start_m * N
        
        # 1. Compute GEMM chunk natively overlapping previous iteration's async all-reduce
        torch.matmul(A_local[start_m:end_m, :], B_local, out=C_symm[start_m:end_m, :])
        events[i].record(torch.cuda.current_stream())
        
        # 2. Asynchronous All-Reduce
        comm_stream.wait_event(events[i])
        with torch.cuda.stream(comm_stream):
            if use_multimem and (chunk_numel % 8 == 0) and (chunk_offset % 8 == 0):
                chunk_numel_128 = chunk_numel // 8
                chunk_offset_128 = chunk_offset // 8
                num_blocks, block_size, block_stride = _multimem_launch_config(chunk_numel, world_size)
                
                _get_ext().launch_multimem_chunked(
                    multicast_ptr, sync_ptrs, chunk_offset_128, chunk_numel_128,
                    world_size, rank, num_blocks, block_size, block_stride, i, comm_stream.cuda_stream
                )
            else:
                dtype_enum = 0 if A_local.dtype == torch.bfloat16 else (1 if A_local.dtype == torch.float32 else 2)
                _get_ext().launch_p2p_chunked(
                    sync_ptrs, C_ptrs, chunk_offset, chunk_numel,
                    world_size, rank, i, dtype_enum, comm_stream.cuda_stream
                )
                
    torch.cuda.current_stream().wait_stream(comm_stream)
    
    # Strict wait ensuring safe buffer extraction before next `solution` call zeroes/touches local dependencies
    dist.barrier()
    return C_symm.clone()