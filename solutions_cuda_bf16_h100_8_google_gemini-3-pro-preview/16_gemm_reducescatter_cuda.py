import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Vectorized helper for software reduction fallback
__device__ __forceinline__ void sum_bf16_8(float* acc, const uint4& val) {
    const __nv_bfloat162* v = reinterpret_cast<const __nv_bfloat162*>(&val);
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(v[i]);
        acc[2 * i] += f.x;
        acc[2 * i + 1] += f.y;
    }
}

__device__ __forceinline__ uint4 pack_bf16_8(const float* acc) {
    uint4 res;
    __nv_bfloat162* v = reinterpret_cast<__nv_bfloat162*>(&res);
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        v[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    }
    return res;
}

__global__ void reduce_scatter_fallback_kernel(
    const uint64_t* __restrict__ symm_C_ptrs,
    __nv_bfloat16* __restrict__ out_C,
    uint32_t* __restrict__ my_flags,
    uint32_t expected_value,
    int chunk_idx,
    int64_t chunk_size,
    int world_size
) {
    // 1. Wait for all peers to signal they have finished this chunk
    if (threadIdx.x == 0) {
        for (int p = 0; p < world_size; ++p) {
            uint32_t val = 0;
            do {
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(my_flags + p) : "memory");
            } while (val < expected_value);
        }
    }
    __syncthreads();

    // 2. Reduce the chunk
    int64_t offset = (int64_t)chunk_idx * chunk_size;
    int64_t num_vecs = chunk_size / 8;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < num_vecs; i += stride) {
        float sum[8] = {0.0f};
        for (int p = 0; p < world_size; ++p) {
            const uint64_t byte_offset = (offset + i * 8) * sizeof(__nv_bfloat16);
            const uint4* peer_C_vec = reinterpret_cast<const uint4*>(symm_C_ptrs[p] + byte_offset);
            uint4 val = peer_C_vec[0];
            sum_bf16_8(sum, val);
        }
        uint4 out_val = pack_bf16_8(sum);
        reinterpret_cast<uint4*>(out_C)[i] = out_val;
    }

    // Tail reduction for remaining elements if chunk_size is not perfectly divisible by 8
    if (tid == 0) {
        for (int64_t i = num_vecs * 8; i < chunk_size; ++i) {
            float sum = 0.0f;
            for (int p = 0; p < world_size; ++p) {
                const __nv_bfloat16* peer_C = reinterpret_cast<const __nv_bfloat16*>(symm_C_ptrs[p]);
                sum += __bfloat162float(peer_C[offset + i]);
            }
            out_C[i] = __float2bfloat16(sum);
        }
    }
}

__global__ void reduce_scatter_multimem_kernel(
    uint64_t multicast_base,
    __nv_bfloat16* __restrict__ out_C,
    uint32_t* __restrict__ my_flags,
    uint32_t expected_value,
    int chunk_idx,
    int64_t chunk_size,
    int world_size
) {
    // 1. Wait for all peers to signal they have finished this chunk
    if (threadIdx.x == 0) {
        for (int p = 0; p < world_size; ++p) {
            uint32_t val = 0;
            do {
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(val) : "l"(my_flags + p) : "memory");
            } while (val < expected_value);
        }
    }
    __syncthreads();

    // 2. Hardware NVSwitch multimem reduction
    int64_t offset = (int64_t)chunk_idx * chunk_size;
    int64_t num_vecs = chunk_size / 8;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < num_vecs; i += stride) {
        uint64_t byte_offset = (offset + i * 8) * sizeof(__nv_bfloat16);
        uint64_t ptr = multicast_base + byte_offset;
        
        uint32_t r0, r1, r2, r3;
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "l"(ptr)
            : "memory");
        
        uint32_t* out_dst = reinterpret_cast<uint32_t*>(out_C + i * 8);
        out_dst[0] = r0;
        out_dst[1] = r1;
        out_dst[2] = r2;
        out_dst[3] = r3;
    }
}

__global__ void send_signal_kernel(uint32_t* target_flags, int index, uint32_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Release consistency ensures prior symm_C matmul stores are visible to target device
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(target_flags + index), "r"(value) : "memory");
    }
}

void launch_send_signal(uint64_t target_flags_ptr, int index, uint32_t value) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    send_signal_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(target_flags_ptr), index, value);
}

void launch_reduce_scatter(
    uint64_t multicast_ptr,
    torch::Tensor symm_C_ptrs_tensor,
    torch::Tensor out_C,
    uint64_t my_flags_ptr,
    uint32_t expected_value,
    int chunk_idx,
    int64_t chunk_size,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 512;
    int blocks = std::min((int)((chunk_size / 8 + threads - 1) / threads), 4096);
    if (blocks == 0) blocks = 1;

    bool use_multimem = (multicast_ptr != 0) && (chunk_size % 8 == 0);

    if (use_multimem) {
        reduce_scatter_multimem_kernel<<<blocks, threads, 0, stream>>>(
            multicast_ptr,
            (__nv_bfloat16*)out_C.data_ptr<at::BFloat16>(),
            reinterpret_cast<uint32_t*>(my_flags_ptr),
            expected_value,
            chunk_idx,
            chunk_size,
            world_size
        );
    } else {
        const uint64_t* ptrs = (const uint64_t*)symm_C_ptrs_tensor.data_ptr<int64_t>();
        reduce_scatter_fallback_kernel<<<blocks, threads, 0, stream>>>(
            ptrs,
            (__nv_bfloat16*)out_C.data_ptr<at::BFloat16>(),
            reinterpret_cast<uint32_t*>(my_flags_ptr),
            expected_value,
            chunk_idx,
            chunk_size,
            world_size
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_send_signal", &launch_send_signal);
    m.def("launch_reduce_scatter", &launch_reduce_scatter);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gemm_reduce_scatter_ext", CUDA_SRC)
    return _ext

_resource_cache = {}
_buf_idx = 0
_invocation_count = 0

def _get_resources(shape, dtype, device, world_size):
    global _resource_cache
    key = (shape, dtype, device, world_size)
    if key in _resource_cache:
        return _resource_cache[key]
    
    buffers = []
    for _ in range(2):  # Double-buffering prevents overwriting during tight pipelined loops
        symm_C = symm_mem.empty(shape, device=device, dtype=dtype)
        symm_C_hdl = symm_mem.rendezvous(symm_C, dist.group.WORLD)
        symm_C_ptrs = torch.tensor(symm_C_hdl.buffer_ptrs, device=device, dtype=torch.int64)
        
        symm_flags = symm_mem.empty((world_size,), dtype=torch.int32, device=device)
        symm_flags.zero_()
        flags_hdl = symm_mem.rendezvous(symm_flags, dist.group.WORLD)
        flags_ptrs = flags_hdl.buffer_ptrs
        
        buffers.append((symm_C, symm_C_hdl, symm_C_ptrs, symm_flags, flags_hdl, flags_ptrs))
        
    torch.cuda.synchronize()
    dist.barrier()
    _resource_cache[key] = buffers
    return buffers

@torch.no_grad()
def solution(A_local: torch.Tensor, B_local: torch.Tensor) -> torch.Tensor:
    global _buf_idx, _invocation_count
    
    W = dist.get_world_size()
    rank = dist.get_rank()
    
    M, K_local = A_local.shape
    _, N = B_local.shape
    M_local = M // W
    
    _invocation_count += 1
    invoc_id = _invocation_count
    
    buffers = _get_resources((M, N), A_local.dtype, A_local.device, W)
    symm_C, symm_C_hdl, symm_C_ptrs, symm_flags, flags_hdl, flags_ptrs = buffers[_buf_idx]
    _buf_idx = (_buf_idx + 1) % 2
    
    ext = _get_ext()
    
    A_contig = A_local.contiguous()
    B_contig = B_local.contiguous()
    
    # Chunked overlap loop: Staggered chunks pipelined to minimize peer spin-wait.
    for i in range(W):
        c = (rank + i) % W
        start_row = c * M_local
        end_row = start_row + M_local
        
        A_slice = A_contig[start_row:end_row, :]
        C_slice = symm_C[start_row:end_row, :]
        
        # Output directly to asymmetric/symmetric shared device memory segment 
        torch.matmul(A_slice, B_contig, out=C_slice)
        
        # Async kernel queuing: Release device-resident chunk-level completion signal
        ext.launch_send_signal(flags_ptrs[c], rank, invoc_id)
        
    out_C = torch.empty((M_local, N), dtype=A_local.dtype, device=A_local.device)
    chunk_size = M_local * N
    
    multicast_ptr = int(symm_C_hdl.multicast_ptr) if hasattr(symm_C_hdl, 'multicast_ptr') and symm_C_hdl.multicast_ptr is not None else 0
    
    # Launch spin-wait device-side reduction; pulls via NVSwitch MMU multimem pointers where possible
    ext.launch_reduce_scatter(
        multicast_ptr,
        symm_C_ptrs,
        out_C,
        symm_flags.data_ptr(),
        invoc_id,
        rank,  # Focus purely on chunk subset
        chunk_size,
        W
    )
    
    return out_C