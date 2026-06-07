import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdint>

#define MAX_RANKS 8

template<typename T>
struct PtrArray {
    const T* ptrs[MAX_RANKS];
    int num_ranks;
};

// Accumulator traits to force safe accumulation in higher precision for 16-bit types
template<typename T>
struct Accumulator {
    typedef T Type;
};

template<>
struct Accumulator<__half> {
    typedef float Type;
};

template<>
struct Accumulator<__nv_bfloat16> {
    typedef float Type;
};

template<typename T>
__device__ inline typename Accumulator<T>::Type to_acc(T x) { return x; }

template<>
__device__ inline float to_acc<__half>(__half x) { return __half2float(x); }

template<>
__device__ inline float to_acc<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename T>
__device__ inline T from_acc(typename Accumulator<T>::Type x) { return x; }

template<>
__device__ inline __half from_acc<__half>(float x) { return __float2half(x); }

template<>
__device__ inline __nv_bfloat16 from_acc<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Universal scalar reduction kernel for generic / unaligned cases
template<typename T>
__global__ void allreduce_scalar_kernel(
    PtrArray<T> arr,
    T* __restrict__ out,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        typename Accumulator<T>::Type sum = to_acc<T>(arr.ptrs[0][idx]);
        for (int i = 1; i < arr.num_ranks; ++i) {
            sum += to_acc<T>(arr.ptrs[i][idx]);
        }
        out[idx] = from_acc<T>(sum);
    }
}

// Highly optimized vectorized BF16 reduction kernel for H100 NVLink 
__global__ void allreduce_bf16_vec_kernel(
    PtrArray<__nv_bfloat16> arr,
    __nv_bfloat16* __restrict__ out,
    int64_t n
) {
    int64_t idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx < n) {
        int64_t rem = n - idx;
        if (rem >= 8) {
            float sums[8] = {0};
            for (int r = 0; r < arr.num_ranks; ++r) {
                // 128-bit vectorized load
                float4 vals = *reinterpret_cast<const float4*>(arr.ptrs[r] + idx);
                const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&vals);
                #pragma unroll
                for(int i = 0; i < 4; ++i) {
                    float2 f2 = __bfloat1622float2(v2[i]);
                    sums[i*2] += f2.x;
                    sums[i*2+1] += f2.y;
                }
            }
            float4 out_vals;
            __nv_bfloat162* out_v2 = reinterpret_cast<__nv_bfloat162*>(&out_vals);
            #pragma unroll
            for(int i = 0; i < 4; ++i) {
                out_v2[i] = __floats2bfloat162_rn(sums[i*2], sums[i*2+1]);
            }
            // 128-bit vectorized store
            *reinterpret_cast<float4*>(out + idx) = out_vals;
        } else {
            // Scalar fallback for remainder elements at the tail
            for(int i = 0; i < rem; ++i) {
                float sum = 0.0f;
                for (int r = 0; r < arr.num_ranks; ++r) {
                    sum += __bfloat162float(arr.ptrs[r][idx + i]);
                }
                out[idx + i] = __float2bfloat16(sum);
            }
        }
    }
}

void allreduce_cuda(
    std::vector<int64_t> remote_ptrs,
    torch::Tensor out,
    int64_t n,
    int dtype_idx
) {
    int num_ranks = remote_ptrs.size();
    TORCH_CHECK(num_ranks <= MAX_RANKS, "Too many ranks mapped for symmetric PtrArray");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    bool all_aligned = (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);

    // dtype_idx: 0=float32, 1=float16, 2=bfloat16, 3=int32, 4=int64
    if (dtype_idx == 2) {
        PtrArray<__nv_bfloat16> arr;
        arr.num_ranks = num_ranks;
        for (int i = 0; i < num_ranks; ++i) {
            arr.ptrs[i] = reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(remote_ptrs[i]));
            if (reinterpret_cast<uintptr_t>(arr.ptrs[i]) % 16 != 0) all_aligned = false;
        }
        if (all_aligned) {
            int64_t num_vec = (n + 7) / 8;
            int threads = 256;
            int blocks = (num_vec + threads - 1) / threads;
            allreduce_bf16_vec_kernel<<<blocks, threads, 0, stream>>>(
                arr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n);
        } else {
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            allreduce_scalar_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                arr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n);
        }
    } else if (dtype_idx == 0) {
        PtrArray<float> arr; arr.num_ranks = num_ranks;
        for (int i = 0; i < num_ranks; ++i) arr.ptrs[i] = reinterpret_cast<const float*>(static_cast<uintptr_t>(remote_ptrs[i]));
        int threads = 256; int blocks = (n + threads - 1) / threads;
        allreduce_scalar_kernel<float><<<blocks, threads, 0, stream>>>(arr, reinterpret_cast<float*>(out.data_ptr()), n);
    } else if (dtype_idx == 1) {
        PtrArray<__half> arr; arr.num_ranks = num_ranks;
        for (int i = 0; i < num_ranks; ++i) arr.ptrs[i] = reinterpret_cast<const __half*>(static_cast<uintptr_t>(remote_ptrs[i]));
        int threads = 256; int blocks = (n + threads - 1) / threads;
        allreduce_scalar_kernel<__half><<<blocks, threads, 0, stream>>>(arr, reinterpret_cast<__half*>(out.data_ptr()), n);
    } else if (dtype_idx == 3) {
        PtrArray<int32_t> arr; arr.num_ranks = num_ranks;
        for (int i = 0; i < num_ranks; ++i) arr.ptrs[i] = reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(remote_ptrs[i]));
        int threads = 256; int blocks = (n + threads - 1) / threads;
        allreduce_scalar_kernel<int32_t><<<blocks, threads, 0, stream>>>(arr, reinterpret_cast<int32_t*>(out.data_ptr()), n);
    } else if (dtype_idx == 4) {
        PtrArray<int64_t> arr; arr.num_ranks = num_ranks;
        for (int i = 0; i < num_ranks; ++i) arr.ptrs[i] = reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(remote_ptrs[i]));
        int threads = 256; int blocks = (n + threads - 1) / threads;
        allreduce_scalar_kernel<int64_t><<<blocks, threads, 0, stream>>>(arr, reinterpret_cast<int64_t*>(out.data_ptr()), n);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom UVA allreduce");
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allreduce_cuda", &allreduce_cuda, "Symmetric Memory UVA flat allreduce sum");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_allreduce_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    key = (dtype, device)
    
    if key in _symm_cache:
        c = _symm_cache[key]
        if c["n"] >= n:
            return c["buf"], c["hdl"]
            
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[key] = {"n": n, "buf": buf, "hdl": hdl}
    return buf, hdl

DTYPE_TO_IDX = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int32: 3,
    torch.int64: 4
}


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda and tensor.is_contiguous(), "Tensor must be a contiguous CUDA tensor"
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor.clone()
        
    n = tensor.numel()
    if n == 0:
        return tensor.clone()

    dtype_idx = DTYPE_TO_IDX.get(tensor.dtype, -1)
    
    # Fallback to standard NCCL for non-numeric or unsupported discrete dtypes
    if dtype_idx == -1:
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    # Compile extension once per node reliably
    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()

    buf, hdl = _get_symm_state(n, tensor.dtype, tensor.device)
    
    # Write phase: Broadcast local data to symmetric buffer mapping
    buf[:n].copy_(tensor.view(-1))
    
    # Synchronization: Wait until all peers have published their arrays
    hdl.barrier(channel=0)
    
    remote_ptrs = [int(hdl.buffer_ptrs[i]) for i in range(world_size)]
    out = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    
    # Device kernel executes peer fetches and reductions intrinsically overlapping communication
    _get_ext().allreduce_cuda(remote_ptrs, out, n, dtype_idx)
    
    # Final Synchronization: Ensure local symmetric buffer is not overwritten until peers read it 
    hdl.barrier(channel=1)

    return out.reshape_as(tensor)