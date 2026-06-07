"""
Strategy:
1. Fused Adam + Chunked Pull-based AllGather: We use a single custom CUDA kernel to overlap the parameter update with AllGather communication, maximizing GPU utilization.
2. Push-based Synchronization: Each block computes its local Adam shard and immediately pushes a completion step-counter flag to all peers via symmetric memory.
3. Fast Spin-Wait: Peers wait on their *local* symmetric flag buffer, minimizing NVLink polling traffic.
4. Symmetric Exchange Buffer: We allocate a shared symmetric exchange buffer of size `P` to hold the updated local shard, which peers then pull into their final output tensor. This keeps memory overhead to a bare minimum `O(P)` instead of caching `O(world_size * P)`.
5. Device-side Barrier: We use `symm_mem` channel barriers at the end of the step to safely allow buffer reuse across sequential optimizer calls without blocking the CPU.
"""

import math
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

template <typename T>
struct AdamMath;

template <>
struct AdamMath<float> {
    static __device__ __forceinline__ float to_float(float x) { return x; }
    static __device__ __forceinline__ float from_float(float x) { return x; }
};

template <>
struct AdamMath<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }
};

template <typename scalar_t>
__global__ void fused_adam_allgather_kernel(
    const scalar_t* __restrict__ g,
    scalar_t* __restrict__ m,
    scalar_t* __restrict__ v,
    scalar_t* __restrict__ w,
    scalar_t* __restrict__ local_gathered,
    scalar_t* __restrict__ my_exchange,
    const uint64_t* __restrict__ flag_ptrs,
    const uint64_t* __restrict__ exchange_ptrs,
    float beta1,
    float beta2,
    float lr,
    float eps,
    float bc1,
    float bc2,
    int64_t P,
    int W,
    int B,
    int my_rank,
    int step
) {
    int b_global = blockIdx.x;
    int r = b_global / B;
    int local_b = b_global % B;

    int64_t align_elements = 16 / sizeof(scalar_t);
    int64_t chunk_size = (P + B - 1) / B;
    chunk_size = ((chunk_size + align_elements - 1) / align_elements) * align_elements;
    
    int64_t start = local_b * chunk_size;
    int64_t end = start + chunk_size;
    if (end > P) end = P;

    if (r == my_rank) {
        // 1. Compute Adam on our rank's shard
        if (start < P) {
            for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
                float gi = AdamMath<scalar_t>::to_float(g[i]);
                float mi = AdamMath<scalar_t>::to_float(m[i]);
                float vi = AdamMath<scalar_t>::to_float(v[i]);
                float wi = AdamMath<scalar_t>::to_float(w[i]);

                mi = mi * beta1 + gi * (1.0f - beta1);
                vi = vi * beta2 + gi * gi * (1.0f - beta2);
                
                float m_hat = mi / bc1;
                float v_hat = vi / bc2;
                
                wi += m_hat / (sqrtf(v_hat) + eps) * (-lr);

                scalar_t out_val = AdamMath<scalar_t>::from_float(wi);
                m[i] = AdamMath<scalar_t>::from_float(mi);
                v[i] = AdamMath<scalar_t>::from_float(vi);
                w[i] = out_val;
                
                local_gathered[r * P + i] = out_val;
                my_exchange[i] = out_val;
            }
        }
        
        __syncthreads();
        // 2. Push signal to all peers that this chunk is ready
        if (threadIdx.x == 0) {
            __threadfence_system();
            for (int p = 0; p < W; ++p) {
                volatile int* peer_flag_ptr = reinterpret_cast<volatile int*>(flag_ptrs[p]);
                peer_flag_ptr[my_rank * B + local_b] = step;
            }
        }
    } else {
        // 1. Spin-wait on LOCAL flag memory for the peer to finish
        if (threadIdx.x == 0) {
            int* my_flag_ptr = reinterpret_cast<int*>(flag_ptrs[my_rank]);
            volatile int* wait_flag = (volatile int*)(&my_flag_ptr[r * B + local_b]);
            while (*wait_flag < step) {
#if __CUDA_ARCH__ >= 700
                asm volatile("nanosleep.u32 20;" ::: "memory");
#endif
            }
            __threadfence_system();
        }
        __syncthreads();
        
        // 2. Pull data from peer's exchange buffer via UVA
        if (start < P) {
            const scalar_t* peer_exchange = reinterpret_cast<const scalar_t*>(exchange_ptrs[r]);
            int64_t n = end - start;
            const scalar_t* src_ptr = peer_exchange + start;
            scalar_t* dst_ptr = local_gathered + r * P + start;
            
            int64_t i = threadIdx.x;
            if (((uintptr_t)src_ptr % 16 == 0) && ((uintptr_t)dst_ptr % 16 == 0)) {
                int64_t n_vec = n / align_elements;
                const ulong2* src_vec = reinterpret_cast<const ulong2*>(src_ptr);
                ulong2* dst_vec = reinterpret_cast<ulong2*>(dst_ptr);
                for (int64_t vi = threadIdx.x; vi < n_vec; vi += blockDim.x) {
                    dst_vec[vi] = src_vec[vi];
                }
                i = n_vec * align_elements + threadIdx.x;
            }
            for (; i < n; i += blockDim.x) {
                dst_ptr[i] = src_ptr[i];
            }
        }
    }
}

void launch_bf16(
    torch::Tensor g, torch::Tensor m, torch::Tensor v, torch::Tensor w,
    torch::Tensor local_gathered, torch::Tensor my_exchange,
    torch::Tensor flag_ptrs, torch::Tensor exchange_ptrs,
    float beta1, float beta2, float lr, float eps, float bc1, float bc2,
    int64_t P, int W, int B, int my_rank, int step,
    int blocks, int threads
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_adam_allgather_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(m.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(local_gathered.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(my_exchange.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint64_t*>(flag_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(exchange_ptrs.data_ptr<int64_t>()),
        beta1, beta2, lr, eps, bc1, bc2,
        P, W, B, my_rank, step
    );
}

void launch_fp32(
    torch::Tensor g, torch::Tensor m, torch::Tensor v, torch::Tensor w,
    torch::Tensor local_gathered, torch::Tensor my_exchange,
    torch::Tensor flag_ptrs, torch::Tensor exchange_ptrs,
    float beta1, float beta2, float lr, float eps, float bc1, float bc2,
    int64_t P, int W, int B, int my_rank, int step,
    int blocks, int threads
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_adam_allgather_kernel<float><<<blocks, threads, 0, stream>>>(
        g.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(),
        local_gathered.data_ptr<float>(), my_exchange.data_ptr<float>(),
        reinterpret_cast<const uint64_t*>(flag_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(exchange_ptrs.data_ptr<int64_t>()),
        beta1, beta2, lr, eps, bc1, bc2,
        P, W, B, my_rank, step
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_bf16", &launch_bf16, "Fused Adam + AllGather for bfloat16");
    m.def("launch_fp32", &launch_fp32, "Fused Adam + AllGather for float32");
}
'''

_ext = None
_ext_compiled = False

def _ensure_ext():
    global _ext, _ext_compiled
    if not _ext_compiled:
        if dist.get_rank() == 0:
            _ext = compile_cuda_extension("fused_adam_allgather_ext", CUDA_SRC)
        dist.barrier()
        if dist.get_rank() != 0:
            _ext = compile_cuda_extension("fused_adam_allgather_ext", CUDA_SRC)
        _ext_compiled = True
    return _ext

_exchange_state = None
_sync_step = 1

def _get_exchange_state(P: int, dtype: torch.dtype, device: torch.device, W: int):
    global _exchange_state
    
    if _exchange_state is None or _exchange_state['P'] < P or _exchange_state['dtype'] != dtype:
        new_P = max(P, _exchange_state['P'] if _exchange_state else 0)
        
        buf = symm_mem.empty(new_P, dtype=dtype, device=device)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
        
        flags = symm_mem.empty(W * 128, dtype=torch.int32, device=device)
        flags.zero_()
        flags_hdl = symm_mem.rendezvous(flags, dist.group.WORLD)
        flag_ptrs = torch.tensor(flags_hdl.buffer_ptrs, dtype=torch.int64, device=device)
        
        _exchange_state = {
            'P': new_P, 'dtype': dtype,
            'buf': buf, 'hdl': hdl, 'ptrs': ptrs,
            'flags': flags, 'flags_hdl': flags_hdl, 'flag_ptrs': flag_ptrs
        }
        
    return _exchange_state


@torch.no_grad()
def solution(
    grad_shard: Tensor,
    master_shard: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> Tensor:
    global _sync_step
    
    W = dist.get_world_size()
    rank = dist.get_rank()
    P = grad_shard.numel()
    device = grad_shard.device
    dtype = master_shard.dtype
    
    ext = _ensure_ext()
    state = _get_exchange_state(P, dtype, device, W)
    
    g = grad_shard.contiguous()
    m = exp_avg.contiguous()
    v = exp_avg_sq.contiguous()
    w = master_shard.contiguous()
    
    gathered = torch.empty(W * P, dtype=dtype, device=device)
    
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    
    B = 128
    threads = 256
    blocks = W * B
    
    if dtype == torch.bfloat16:
        ext.launch_bf16(
            g, m, v, w,
            gathered, state['buf'], state['flag_ptrs'], state['ptrs'],
            beta1, beta2, lr, eps, bc1, bc2,
            P, W, B, rank, _sync_step,
            blocks, threads
        )
    elif dtype == torch.float32:
        ext.launch_fp32(
            g, m, v, w,
            gathered, state['buf'], state['flag_ptrs'], state['ptrs'],
            beta1, beta2, lr, eps, bc1, bc2,
            P, W, B, rank, _sync_step,
            blocks, threads
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
        
    _sync_step += 1
    
    # Fast device-side barrier prevents proceeding CPU streams from 
    # enqueuing kernels that might overwrite the reused `exchange_buf` 
    # before all peer pulls have finished asynchronously.
    state['hdl'].barrier(channel=0)
    
    return gathered

__all__ = ["solution"]