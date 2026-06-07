"""
Strategy:
1. **Device-side Communication:** We replace NCCL all-reduce with a custom one-shot direct all-reduce CUDA kernel. We utilize `torch.distributed._symmetric_memory` to allocate the output chunks (`C_local`) and small sync flags. The device kernels perform UVA reads directly across peers over NVLink.
2. **Compute-Communication Overlap:** We slice the matrix along the M-dimension into chunks. `torch.matmul` runs on the main compute stream. Once a chunk's GEMM completes, a CUDA event triggers the all-reduce kernel on a separate communication stream. This perfectly pipelines GEMM compute with peer reduction without blocking the host.
3. **Optimized BF16 Reduction:** We implement a fully vectorized path using `uint4` memory transactions and tensor core math primitives (`__nv_bfloat162`) to extract maximum memory bandwidth and throughput from Hopper NVLink for the hot-path BF16 workloads.
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
#include <vector>
#include <stdexcept>

// Convert to/from float for numeric accumulation
template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }

struct PeerPtrs {
    const void* ptrs[8];
    int32_t* flags[8];
};

// Highly Optimized Vectorized Path for BF16
template <int WORLD_SIZE>
__global__ void chunked_allreduce_bf16_direct(
    PeerPtrs peers,
    void* C_out_v,
    int rank,
    int chunk_idx,
    size_t chunk_offset,
    size_t chunk_elements,
    int seq
) {
    const __nv_bfloat16* peer_C_local[WORLD_SIZE];
    #pragma unroll
    for(int i=0; i<WORLD_SIZE; i++) peer_C_local[i] = (const __nv_bfloat16*)peers.ptrs[i];
    int32_t* const* peer_flags = peers.flags;
    __nv_bfloat16* C_out = (__nv_bfloat16*)C_out_v;

    // Signal own chunk is ready
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        atomicMax(peer_flags[rank] + chunk_idx, seq);
    }
    
    // Spin-wait for all peers
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int p = 0; p < WORLD_SIZE; p++) {
            if (p == rank) continue;
            volatile int32_t* flag_ptr = peer_flags[p] + chunk_idx;
            while (*flag_ptr < seq) { }
        }
    }
    __syncthreads();

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if (chunk_offset % 8 == 0) {
        size_t vec_elements = chunk_elements / 8;
        size_t vec_offset = chunk_offset / 8;
        
        for (size_t i = tid; i < vec_elements; i += stride) {
            uint4 vals[WORLD_SIZE];
            
            #pragma unroll
            for (int p = 0; p < WORLD_SIZE; p++) {
                vals[p] = reinterpret_cast<const uint4*>(peer_C_local[p])[vec_offset + i];
            }
            
            auto sum_bf16x2 = [&](const uint32_t* p_vals) -> uint32_t {
                float2 s = make_float2(0,0);
                #pragma unroll
                for (int p = 0; p < WORLD_SIZE; p++) {
                    __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&p_vals[p]);
                    float2 f = __bfloat1622float2(b);
                    s.x += f.x;
                    s.y += f.y;
                }
                __nv_bfloat162 res = __float22bfloat162_rn(s);
                return *reinterpret_cast<uint32_t*>(&res);
            };

            uint32_t px[WORLD_SIZE], py[WORLD_SIZE], pz[WORLD_SIZE], pw[WORLD_SIZE];
            #pragma unroll
            for (int p = 0; p < WORLD_SIZE; p++) {
                px[p] = vals[p].x;
                py[p] = vals[p].y;
                pz[p] = vals[p].z;
                pw[p] = vals[p].w;
            }

            uint32_t out_x = sum_bf16x2(px);
            uint32_t out_y = sum_bf16x2(py);
            uint32_t out_z = sum_bf16x2(pz);
            uint32_t out_w = sum_bf16x2(pw);
            
            uint4 out_val = make_uint4(out_x, out_y, out_z, out_w);
            reinterpret_cast<uint4*>(C_out)[vec_offset + i] = out_val;
        }
        
        // Scalar Tail
        size_t tail_start = vec_elements * 8;
        for (size_t i = tail_start + tid; i < chunk_elements; i += stride) {
            float sum = 0.0f;
            #pragma unroll
            for (int p = 0; p < WORLD_SIZE; p++) {
                sum += __bfloat162float(peer_C_local[p][chunk_offset + i]);
            }
            C_out[chunk_offset + i] = __float2bfloat16(sum);
        }
    } else {
        for (size_t i = tid; i < chunk_elements; i += stride) {
            float sum = 0.0f;
            #pragma unroll
            for (int p = 0; p < WORLD_SIZE; p++) {
                sum += __bfloat162float(peer_C_local[p][chunk_offset + i]);
            }
            C_out[chunk_offset + i] = __float2bfloat16(sum);
        }
    }
    
    __threadfence_system();
}

// Fallback Scalar Path for Other Dtypes
template <typename T, int WORLD_SIZE>
__global__ void chunked_allreduce_generic(
    PeerPtrs peers,
    void* C_out_v,
    int rank,
    int chunk_idx,
    size_t chunk_offset,
    size_t chunk_elements,
    int seq
) {
    const T* peer_C_local[WORLD_SIZE];
    #pragma unroll
    for(int i=0; i<WORLD_SIZE; i++) peer_C_local[i] = (const T*)peers.ptrs[i];
    int32_t* const* peer_flags = peers.flags;
    T* C_out = (T*)C_out_v;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        atomicMax(peer_flags[rank] + chunk_idx, seq);
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int p = 0; p < WORLD_SIZE; p++) {
            if (p == rank) continue;
            volatile int32_t* flag_ptr = peer_flags[p] + chunk_idx;
            while (*flag_ptr < seq) {}
        }
    }
    __syncthreads();

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < chunk_elements; i += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int p = 0; p < WORLD_SIZE; p++) {
            sum += to_float<T>(peer_C_local[p][chunk_offset + i]);
        }
        C_out[chunk_offset + i] = from_float<T>(sum);
    }
    __threadfence_system();
}

void launch_allreduce(
    std::vector<int64_t> peer_c_ptrs,
    std::vector<int64_t> peer_flag_ptrs,
    int64_t c_out_ptr,
    int rank,
    int world_size,
    int chunk_idx,
    int64_t chunk_offset,
    int64_t chunk_elements,
    int seq,
    int dtype_enum
) {
    PeerPtrs peers;
    for (int i = 0; i < world_size; i++) {
        peers.ptrs[i] = reinterpret_cast<const void*>(peer_c_ptrs[i]);
        peers.flags[i] = reinterpret_cast<int32_t*>(peer_flag_ptrs[i]);
    }
    void* c_out = reinterpret_cast<void*>(c_out_ptr);

    int threads = 512;
    int blocks = std::min((int)((chunk_elements + threads - 1) / threads), 108 * 4);
    if (dtype_enum == 0) {
        blocks = std::min((int)((chunk_elements / 8 + threads - 1) / threads), 108 * 4);
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    #define DISPATCH_WS(WS) \
        if (dtype_enum == 0) { \
            chunked_allreduce_bf16_direct<WS><<<blocks, threads, 0, stream>>>(peers, c_out, rank, chunk_idx, chunk_offset, chunk_elements, seq); \
        } else if (dtype_enum == 1) { \
            chunked_allreduce_generic<__half, WS><<<blocks, threads, 0, stream>>>(peers, c_out, rank, chunk_idx, chunk_offset, chunk_elements, seq); \
        } else { \
            chunked_allreduce_generic<float, WS><<<blocks, threads, 0, stream>>>(peers, c_out, rank, chunk_idx, chunk_offset, chunk_elements, seq); \
        }

    switch (world_size) {
        case 1: DISPATCH_WS(1); break;
        case 2: DISPATCH_WS(2); break;
        case 3: DISPATCH_WS(3); break;
        case 4: DISPATCH_WS(4); break;
        case 5: DISPATCH_WS(5); break;
        case 6: DISPATCH_WS(6); break;
        case 7: DISPATCH_WS(7); break;
        case 8: DISPATCH_WS(8); break;
        default: throw std::runtime_error("Unsupported world size");
    }
    #undef DISPATCH_WS

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_allreduce", &launch_allreduce, "Chunked allreduce direct");
}
'''

_ext = None
_compiled = False
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_gemm_allreduce_direct", CUDA_SRC)
    return _ext

_cache = {}
_seq_num = 1

def _get_symm_state(M, N, dtype, device, num_chunks):
    global _cache
    key = (M, N, dtype, device, num_chunks)
    if key in _cache:
        return _cache[key]
    
    C_local_buf = symm_mem.empty((M, N), dtype=dtype, device=device)
    C_local_hdl = symm_mem.rendezvous(C_local_buf, dist.group.WORLD)
    
    flags_buf = symm_mem.empty((num_chunks,), dtype=torch.int32, device=device)
    flags_buf.zero_()
    flags_hdl = symm_mem.rendezvous(flags_buf, dist.group.WORLD)
    
    peer_c_ptrs = [int(C_local_hdl.buffer_ptrs[i]) for i in range(dist.get_world_size())]
    peer_flag_ptrs = [int(flags_hdl.buffer_ptrs[i]) for i in range(dist.get_world_size())]
    
    compute_events = [torch.cuda.Event() for _ in range(num_chunks)]
    comm_events = [torch.cuda.Event() for _ in range(num_chunks)]
    comm_stream = torch.cuda.Stream()
    
    state = {
        "C_local_buf": C_local_buf,
        "peer_c_ptrs": peer_c_ptrs,
        "peer_flag_ptrs": peer_flag_ptrs,
        "compute_events": compute_events,
        "comm_events": comm_events,
        "comm_stream": comm_stream
    }
    _cache[key] = state
    return state

@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B_local: torch.Tensor,
) -> torch.Tensor:
    global _seq_num, _compiled
    
    rank = dist.get_rank()
    if not _compiled:
        if rank == 0:
            _get_ext()
        dist.barrier()
        _compiled = True
        
    seq = _seq_num
    _seq_num += 1

    M, K = A_local.shape
    K_B, N = B_local.shape
    dtype = A_local.dtype
    device = A_local.device
    world_size = dist.get_world_size()
    
    if dtype == torch.bfloat16:
        dtype_enum = 0
    elif dtype == torch.float16:
        dtype_enum = 1
    elif dtype == torch.float32:
        dtype_enum = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # For safety/performance, standardize layouts
    A = A_local.contiguous()
    B = B_local.contiguous()
    
    NUM_CHUNKS = 4 if M >= 1024 else (2 if M >= 256 else 1)
    state = _get_symm_state(M, N, dtype, device, NUM_CHUNKS)
    
    C_local_buf = state["C_local_buf"]
    peer_c_ptrs = state["peer_c_ptrs"]
    peer_flag_ptrs = state["peer_flag_ptrs"]
    compute_events = state["compute_events"]
    comm_events = state["comm_events"]
    comm_stream = state["comm_stream"]
    
    C_out = torch.empty((M, N), dtype=dtype, device=device)
    compute_stream = torch.cuda.current_stream()
    ext = _get_ext()
    chunk_size = (M + NUM_CHUNKS - 1) // NUM_CHUNKS
    
    for c in range(NUM_CHUNKS):
        start = min(c * chunk_size, M)
        end = min((c + 1) * chunk_size, M)
        if start == end:
            continue
            
        # 1. Compute Local Block
        A_chunk = A[start:end]
        C_chunk = C_local_buf[start:end]
        torch.matmul(A_chunk, B, out=C_chunk)
        
        # 2. Record GEMM completion
        compute_events[c].record(compute_stream)
        
        # 3. Synchronize pipeline: Comm stream waits for compute stream chunk
        comm_stream.wait_event(compute_events[c])
        
        # 4. Device P2P AllReduce on Comm Stream
        chunk_offset = start * N
        chunk_elements = (end - start) * N
        
        with torch.cuda.stream(comm_stream):
            ext.launch_allreduce(
                peer_c_ptrs,
                peer_flag_ptrs,
                C_out.data_ptr(),
                rank,
                world_size,
                c,
                chunk_offset,
                chunk_elements,
                seq,
                dtype_enum
            )
            
        # 5. Record Reduction completion
        comm_events[c].record(comm_stream)
        
    # Wait for the reduction kernels on the Comm Stream to land before handing off Tensor
    for c in range(NUM_CHUNKS):
        compute_stream.wait_event(comm_events[c])
        
    return C_out