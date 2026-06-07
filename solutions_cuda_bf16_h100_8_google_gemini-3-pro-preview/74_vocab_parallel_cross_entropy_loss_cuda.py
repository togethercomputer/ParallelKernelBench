import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional, Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <typename T>
struct Vec2;

template <>
struct Vec2<__nv_bfloat16> {
    using type = __nv_bfloat162;
    static __device__ __forceinline__ float2 to_float2(type v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        return __bfloat1622float2(v);
#else
        return {__bfloat162float(v.x), __bfloat162float(v.y)};
#endif
    }
    static __device__ __forceinline__ type from_float2(float2 f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        return __floats2bfloat162_rn(f.x, f.y);
#else
        type v;
        v.x = __float2bfloat16(f.x);
        v.y = __float2bfloat16(f.y);
        return v;
#endif
    }
    static __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float f) {
        return __float2bfloat16(f);
    }
};

template <>
struct Vec2<float> {
    using type = float2;
    static __device__ __forceinline__ float2 to_float2(type v) { return v; }
    static __device__ __forceinline__ type from_float2(float2 f) { return f; }
    static __device__ __forceinline__ float to_float(float v) { return v; }
    static __device__ __forceinline__ float from_float(float f) { return f; }
};

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : -1e20f;
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    __syncthreads();
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    __syncthreads();
    return val;
}

template <typename T>
__global__ void kernel_local_max(
    const T* __restrict__ logits,
    float* __restrict__ sym_M_local,
    int N, int P
) {
    using V2 = typename Vec2<T>::type;
    __shared__ float shared_reduce[32];
    
    for (int i = blockIdx.x; i < N; i += gridDim.x) {
        float local_max = -1e20f;
        if (P % 2 == 0) {
            int P2 = P / 2;
            const V2* logits2 = (const V2*)(logits + i * P);
            for (int j = threadIdx.x; j < P2; j += blockDim.x) {
                float2 fvals = Vec2<T>::to_float2(logits2[j]);
                local_max = max(local_max, fvals.x);
                local_max = max(local_max, fvals.y);
            }
        } else {
            for (int j = threadIdx.x; j < P; j += blockDim.x) {
                float val = Vec2<T>::to_float(logits[i * P + j]);
                local_max = max(local_max, val);
            }
        }
        local_max = block_reduce_max(local_max, shared_reduce);
        if (threadIdx.x == 0) {
            sym_M_local[i] = local_max;
        }
    }
}

template <typename T>
__global__ void kernel_shift_and_local_sums(
    T* __restrict__ logits,
    const long long* __restrict__ sym_ptrs,
    const int64_t* __restrict__ target,
    float* __restrict__ sym_S_local,
    float* __restrict__ sym_t_local,
    int N, int P, int vocab_start, int vocab_end, int world_size
) {
    using V2 = typename Vec2<T>::type;
    __shared__ float shared_reduce[32];
    __shared__ float shared_M_global;
    
    for (int i = blockIdx.x; i < N; i += gridDim.x) {
        if (threadIdx.x == 0) {
            float m = -1e20f;
            for (int r = 0; r < world_size; ++r) {
                const float* remote_M = (const float*)sym_ptrs[r];
                m = max(m, remote_M[i]);
            }
            shared_M_global = m;
        }
        __syncthreads();
        float M_global = shared_M_global;

        int target_idx = target[i] - vocab_start;
        float local_sum_exp = 0.0f;
        float local_t = 0.0f;

        if (P % 2 == 0) {
            int P2 = P / 2;
            V2* logits2 = (V2*)(logits + i * P);
            for (int j = threadIdx.x; j < P2; j += blockDim.x) {
                V2 vals = logits2[j];
                float2 fvals = Vec2<T>::to_float2(vals);
                fvals.x -= M_global;
                fvals.y -= M_global;
                logits2[j] = Vec2<T>::from_float2(fvals);
                
                local_sum_exp += expf(fvals.x) + expf(fvals.y);
                if (j * 2 == target_idx) local_t = fvals.x;
                else if (j * 2 + 1 == target_idx) local_t = fvals.y;
            }
        } else {
            for (int j = threadIdx.x; j < P; j += blockDim.x) {
                float val = Vec2<T>::to_float(logits[i * P + j]);
                val -= M_global;
                logits[i * P + j] = Vec2<T>::from_float(val);
                local_sum_exp += expf(val);
                if (j == target_idx) local_t = val;
            }
        }

        float total_sum_exp = block_reduce_sum(local_sum_exp, shared_reduce);
        float total_t = block_reduce_sum(local_t, shared_reduce);

        if (threadIdx.x == 0) {
            sym_S_local[i] = total_sum_exp;
            sym_t_local[i] = total_t;
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void kernel_global_sums_and_loss(
    const long long* __restrict__ sym_ptrs,
    T* __restrict__ loss,
    int N, int world_size
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float S_global = 0.0f;
        float t_global = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const float* remote_S = ((const float*)sym_ptrs[r]) + N;
            const float* remote_t = ((const float*)sym_ptrs[r]) + 2 * N;
            S_global += remote_S[i];
            t_global += remote_t[i];
        }
        float final_loss = logf(S_global) - t_global;
        loss[i] = Vec2<T>::from_float(final_loss);
    }
}

void launch_kernel_local_max(
    torch::Tensor logits,
    int64_t sym_M_local,
    int N, int P, int dtype_enum
) {
    int threads = 256;
    int blocks = std::min(N, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (dtype_enum == 0) {
        kernel_local_max<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            (__nv_bfloat16*)logits.data_ptr<at::BFloat16>(), (float*)sym_M_local, N, P);
    } else {
        kernel_local_max<float><<<blocks, threads, 0, stream>>>(
            logits.data_ptr<float>(), (float*)sym_M_local, N, P);
    }
}

void launch_kernel_shift_and_local_sums(
    torch::Tensor logits,
    torch::Tensor sym_ptrs,
    torch::Tensor target,
    int64_t sym_S_local,
    int64_t sym_t_local,
    int N, int P, int vocab_start, int vocab_end, int world_size, int dtype_enum
) {
    int threads = 256;
    int blocks = std::min(N, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const long long* d_sym_ptrs = (const long long*)sym_ptrs.data_ptr<int64_t>();
    const int64_t* d_target = target.data_ptr<int64_t>();
    
    if (dtype_enum == 0) {
        kernel_shift_and_local_sums<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            (__nv_bfloat16*)logits.data_ptr<at::BFloat16>(), d_sym_ptrs, d_target,
            (float*)sym_S_local, (float*)sym_t_local, N, P, vocab_start, vocab_end, world_size);
    } else {
        kernel_shift_and_local_sums<float><<<blocks, threads, 0, stream>>>(
            logits.data_ptr<float>(), d_sym_ptrs, d_target,
            (float*)sym_S_local, (float*)sym_t_local, N, P, vocab_start, vocab_end, world_size);
    }
}

void launch_kernel_global_sums_and_loss(
    torch::Tensor sym_ptrs,
    torch::Tensor loss,
    int N, int world_size, int dtype_enum
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const long long* d_sym_ptrs = (const long long*)sym_ptrs.data_ptr<int64_t>();
    
    if (dtype_enum == 0) {
        kernel_global_sums_and_loss<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            d_sym_ptrs, (__nv_bfloat16*)loss.data_ptr<at::BFloat16>(), N, world_size);
    } else {
        kernel_global_sums_and_loss<float><<<blocks, threads, 0, stream>>>(
            d_sym_ptrs, loss.data_ptr<float>(), N, world_size);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel_local_max", &launch_kernel_local_max);
    m.def("launch_kernel_shift_and_local_sums", &launch_kernel_shift_and_local_sums);
    m.def("launch_kernel_global_sums_and_loss", &launch_kernel_global_sums_and_loss);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_vocab_ce_ce", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(N: int, device: torch.device, group):
    global _symm_cache
    key = (device,)
    if key in _symm_cache:
        c = _symm_cache[key]
        if c["N"] >= N:
            return c["buf"], c["hdl"], c["ptrs"]
            
    buf = symm_mem.empty(3 * N, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _symm_cache[key] = {"N": N, "buf": buf, "hdl": hdl, "ptrs": ptrs}
    return buf, hdl, ptrs

@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    
    assert vocab_parallel_logits.is_contiguous(), "vocab_parallel_logits must be contiguous"
    
    dtype = vocab_parallel_logits.dtype
    if dtype == torch.bfloat16:
        dtype_enum = 0
    elif dtype == torch.float32:
        dtype_enum = 1
    else:
        raise ValueError("Only BF16 and F32 are supported")

    logits_2d = vocab_parallel_logits.view(-1, vocab_parallel_logits.shape[-1])
    target_1d = target.reshape(-1).contiguous()
    
    N, P = logits_2d.shape
    vocab_start = rank * P
    vocab_end = vocab_start + P

    # 3xN symmetric staging buffer
    buf, hdl, sym_ptrs = _get_symm_state(N, vocab_parallel_logits.device, group)
    
    sym_M_local = int(hdl.buffer_ptrs[rank])
    sym_S_local = sym_M_local + N * 4 
    sym_t_local = sym_M_local + 2 * N * 4

    # 1. Local block-reduced maximum
    _get_ext().launch_kernel_local_max(
        logits_2d, sym_M_local, N, P, dtype_enum
    )
    
    # Fast device-side stream barrier
    hdl.barrier(channel=0)
    
    # 2. Inplace Logits shift + Compute Partial Sum-Exps
    _get_ext().launch_kernel_shift_and_local_sums(
        logits_2d, sym_ptrs, target_1d,
        sym_S_local, sym_t_local,
        N, P, vocab_start, vocab_end, world_size, dtype_enum
    )
    
    # Eager peer synchronization for metrics 
    hdl.barrier(channel=1)
    
    # 3. Pull globals & formulate standard Loss
    loss_1d = torch.empty_like(target_1d, dtype=dtype)
    _get_ext().launch_kernel_global_sums_and_loss(
        sym_ptrs, loss_1d, N, world_size, dtype_enum
    )
    
    return loss_1d.view(target.shape)