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

template <int MAX_PEERS=16>
struct PeerPointers {
    const void* ptrs[MAX_PEERS];
    int count;
};

// BF16 Optimized Vectorized Kernel
__global__ void reduce_scatter_bf16_vec_kernel(
    PeerPointers<16> peers,
    __nv_bfloat16* __restrict__ out,
    int64_t chunk_offset,
    int64_t chunk_size
) {
    int64_t vec_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (vec_idx < chunk_size) {
        int limit = (chunk_size - vec_idx < 8) ? (chunk_size - vec_idx) : 8;
        
        // Ensure strictly aligned 16-byte boundaries for uint4
        if (limit == 8 && (chunk_offset % 8 == 0)) {
            float sums[8] = {0.0f};
            
            #pragma unroll(8)
            for (int p = 0; p < peers.count; ++p) {
                const __nv_bfloat16* peer_ptr = reinterpret_cast<const __nv_bfloat16*>(peers.ptrs[p]) + chunk_offset;
                uint4 vals = *reinterpret_cast<const uint4*>(peer_ptr + vec_idx);
                
                float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vals.x));
                float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vals.y));
                float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vals.z));
                float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vals.w));
                
                sums[0] += f0.x; sums[1] += f0.y;
                sums[2] += f1.x; sums[3] += f1.y;
                sums[4] += f2.x; sums[5] += f2.y;
                sums[6] += f3.x; sums[7] += f3.y;
            }
            
            uint4 out_vals;
            *reinterpret_cast<__nv_bfloat162*>(&out_vals.x) = __floats2bfloat162_rn(sums[0], sums[1]);
            *reinterpret_cast<__nv_bfloat162*>(&out_vals.y) = __floats2bfloat162_rn(sums[2], sums[3]);
            *reinterpret_cast<__nv_bfloat162*>(&out_vals.z) = __floats2bfloat162_rn(sums[4], sums[5]);
            *reinterpret_cast<__nv_bfloat162*>(&out_vals.w) = __floats2bfloat162_rn(sums[6], sums[7]);
            
            *reinterpret_cast<uint4*>(out + vec_idx) = out_vals;
            
        } else {
            // Scalar fallback for bounds and non-aligned dimensions
            for (int i = 0; i < limit; ++i) {
                float sum = 0.0f;
                for (int p = 0; p < peers.count; ++p) {
                    sum += __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(peers.ptrs[p])[chunk_offset + vec_idx + i]);
                }
                out[vec_idx + i] = __float2bfloat16(sum);
            }
        }
    }
}

// FP16 Optimized Vectorized Kernel
__global__ void reduce_scatter_fp16_vec_kernel(
    PeerPointers<16> peers,
    __half* __restrict__ out,
    int64_t chunk_offset,
    int64_t chunk_size
) {
    int64_t vec_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (vec_idx < chunk_size) {
        int limit = (chunk_size - vec_idx < 8) ? (chunk_size - vec_idx) : 8;
        
        // Ensure strictly aligned 16-byte boundaries for uint4
        if (limit == 8 && (chunk_offset % 8 == 0)) {
            float sums[8] = {0.0f};
            
            #pragma unroll(8)
            for (int p = 0; p < peers.count; ++p) {
                const __half* peer_ptr = reinterpret_cast<const __half*>(peers.ptrs[p]) + chunk_offset;
                uint4 vals = *reinterpret_cast<const uint4*>(peer_ptr + vec_idx);
                
                float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&vals.x));
                float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&vals.y));
                float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&vals.z));
                float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&vals.w));
                
                sums[0] += f0.x; sums[1] += f0.y;
                sums[2] += f1.x; sums[3] += f1.y;
                sums[4] += f2.x; sums[5] += f2.y;
                sums[6] += f3.x; sums[7] += f3.y;
            }
            
            uint4 out_vals;
            *reinterpret_cast<__half2*>(&out_vals.x) = __floats2half2_rn(sums[0], sums[1]);
            *reinterpret_cast<__half2*>(&out_vals.y) = __floats2half2_rn(sums[2], sums[3]);
            *reinterpret_cast<__half2*>(&out_vals.z) = __floats2half2_rn(sums[4], sums[5]);
            *reinterpret_cast<__half2*>(&out_vals.w) = __floats2half2_rn(sums[6], sums[7]);
            
            *reinterpret_cast<uint4*>(out + vec_idx) = out_vals;
            
        } else {
            // Scalar fallback
            for (int i = 0; i < limit; ++i) {
                float sum = 0.0f;
                for (int p = 0; p < peers.count; ++p) {
                    sum += __half2float(reinterpret_cast<const __half*>(peers.ptrs[p])[chunk_offset + vec_idx + i]);
                }
                out[vec_idx + i] = __float2half(sum);
            }
        }
    }
}

// Generic Support Kernel for FP32, INT32, etc.
template <typename T>
__global__ void reduce_scatter_generic_kernel(
    PeerPointers<16> peers,
    T* __restrict__ out,
    int64_t chunk_offset,
    int64_t chunk_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < chunk_size) {
        T sum = 0;
        for (int p = 0; p < peers.count; ++p) {
            sum += reinterpret_cast<const T*>(peers.ptrs[p])[chunk_offset + idx];
        }
        out[idx] = sum;
    }
}

void reduce_scatter_cuda(
    std::vector<int64_t> peer_ptrs_ints,
    torch::Tensor out,
    int64_t chunk_offset,
    int64_t chunk_size
) {
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    int world_size = peer_ptrs_ints.size();
    TORCH_CHECK(world_size <= 16, "max 16 peers supported");

    PeerPointers<16> peers;
    peers.count = world_size;
    for (int i = 0; i < world_size; ++i) {
        peers.ptrs[i] = reinterpret_cast<const void*>(peer_ptrs_ints[i]);
    }

    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (out.dtype() == torch::kBFloat16) {
        int64_t num_vecs = (chunk_size + 7) / 8;
        int blocks = (num_vecs + threads - 1) / threads;
        reduce_scatter_bf16_vec_kernel<<<blocks, threads, 0, stream>>>(
            peers, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), chunk_offset, chunk_size
        );
    } else if (out.dtype() == torch::kFloat16) {
        int64_t num_vecs = (chunk_size + 7) / 8;
        int blocks = (num_vecs + threads - 1) / threads;
        reduce_scatter_fp16_vec_kernel<<<blocks, threads, 0, stream>>>(
            peers, reinterpret_cast<__half*>(out.data_ptr<at::Half>()), chunk_offset, chunk_size
        );
    } else {
        int blocks = (chunk_size + threads - 1) / threads;
        AT_DISPATCH_ALL_TYPES(out.scalar_type(), "reduce_scatter_generic", [&] {
            reduce_scatter_generic_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                peers, out.data_ptr<scalar_t>(), chunk_offset, chunk_size
            );
        });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_scatter_cuda", &reduce_scatter_cuda, "UVA reduce_scatter direct-fetch");
}
'''

_ext = None
_symm_cache = {}

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("reduce_scatter_direct_uva_ext", CUDA_SRC)
    return _ext

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    key = (n, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert tensor.shape[0] % world_size == 0, \
        f"First dimension ({tensor.shape[0]}) must be divisible by world_size ({world_size})"
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    
    ext = _get_ext()
    n = tensor.numel()
    
    buf, hdl = _get_symm_state(n, tensor.dtype, tensor.device)
    
    # Wait for all peer threads to safely finish relying on the symmetric buffer from preceding calls
    hdl.barrier(channel=0)
    
    # Expose current input directly into symmetric memory pool
    buf.copy_(tensor.flatten())
    
    # Wait for all copies on all GPUs to finalize so kernels access cleanly updated arrays 
    hdl.barrier(channel=1)
    
    chunk_size_dim0 = tensor.shape[0] // world_size
    out_shape = (chunk_size_dim0,) + tensor.shape[1:]
    chunk_elements = n // world_size
    chunk_offset = rank * chunk_elements
    
    out = torch.empty(chunk_elements, dtype=tensor.dtype, device=tensor.device)
    peer_ptrs = [int(hdl.buffer_ptrs[p]) for p in range(world_size)]
    
    ext.reduce_scatter_cuda(peer_ptrs, out, chunk_offset, chunk_elements)
    
    return out.view(out_shape)