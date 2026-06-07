from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

template <typename T>
__device__ __forceinline__ float ld(const T* p, int64_t i);

template <>
__device__ __forceinline__ float ld<float>(const float* p, int64_t i) {
    return p[i];
}

template <>
__device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}

template <typename T>
__device__ __forceinline__ void st(T* p, int64_t i, float v);

template <>
__device__ __forceinline__ void st<float>(float* p, int64_t i, float v) {
    p[i] = v;
}

template <>
__device__ __forceinline__ void st<__nv_bfloat16>(__nv_bfloat16* p, int64_t i, float v) {
    p[i] = __float2bfloat16(v);
}

template <typename T>
__global__ void pack_params_kernel(
    const T* __restrict__ W1,
    const T* __restrict__ b1,
    const T* __restrict__ W2,
    const T* __restrict__ b2,
    T* __restrict__ flat,
    int64_t nW1,
    int64_t nb1,
    int64_t nW2,
    int64_t nb2,
    int64_t total
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        if (i < nW1) {
            flat[i] = W1[i];
        } else if (i < nW1 + nb1) {
            flat[i] = b1[i - nW1];
        } else if (i < nW1 + nb1 + nW2) {
            flat[i] = W2[i - nW1 - nb1];
        } else {
            flat[i] = b2[i - nW1 - nb1 - nW2];
        }
    }
}

template <typename T>
__global__ void copy_from_uva_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

template <typename T>
__global__ void add_bias_relu_kernel(
    T* __restrict__ z,
    const T* __restrict__ b,
    int64_t rows,
    int64_t cols
) {
    int64_t n = rows * cols;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t c = i % cols;
        float v = ld<T>(z, i) + ld<T>(b, c);
        if (v < 0.0f) v = 0.0f;
        st<T>(z, i, v);
    }
}

template <typename T>
__global__ void add_bias_mse_grad_kernel(
    T* __restrict__ out_as_grad,
    const T* __restrict__ y,
    const T* __restrict__ b,
    float scale,
    int64_t rows,
    int64_t cols
) {
    int64_t n = rows * cols;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t c = i % cols;
        float pred = ld<T>(out_as_grad, i) + ld<T>(b, c);
        float diff = pred - ld<T>(y, i);
        st<T>(out_as_grad, i, diff * scale);
    }
}

template <typename T>
__global__ void relu_backward_kernel(
    const T* __restrict__ dh,
    const T* __restrict__ h,
    T* __restrict__ dz,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float hv = ld<T>(h, i);
        float gv = hv > 0.0f ? ld<T>(dh, i) : 0.0f;
        st<T>(dz, i, gv);
    }
}

template <typename T>
__global__ void bias_reduce_kernel(
    const T* __restrict__ grad,
    T* __restrict__ db,
    int64_t rows,
    int64_t cols
) {
    extern __shared__ float smem[];
    int64_t c = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int64_t r = tid; r < rows; r += blockDim.x) {
        sum += ld<T>(grad, r * cols + c);
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) st<T>(db, c, smem[0]);
}

template <typename T>
__global__ void pack_grads_kernel(
    const T* __restrict__ dW1,
    const T* __restrict__ db1,
    const T* __restrict__ dW2,
    const T* __restrict__ db2,
    T* __restrict__ flat,
    int64_t nW1,
    int64_t nb1,
    int64_t nW2,
    int64_t nb2,
    int64_t total
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        if (i < nW1) {
            flat[i] = dW1[i];
        } else if (i < nW1 + nb1) {
            flat[i] = db1[i - nW1];
        } else if (i < nW1 + nb1 + nW2) {
            flat[i] = dW2[i - nW1 - nb1];
        } else {
            flat[i] = db2[i - nW1 - nb1 - nW2];
        }
    }
}

template <typename P, typename S>
__global__ void adam_reduce_scatter_update_kernel(
    const long long* __restrict__ grad_ptrs,
    const P* __restrict__ flat_w,
    const S* __restrict__ m_in,
    const S* __restrict__ v_in,
    P* __restrict__ w_part_out,
    S* __restrict__ m_out,
    S* __restrict__ v_out,
    int64_t part,
    int64_t start,
    int world_size,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float inv_world,
    float lr,
    float bc1,
    float bc2,
    float eps
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < part; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t gi = start + i;

        float gsum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const P* gp = reinterpret_cast<const P*>((uintptr_t)grad_ptrs[r]);
                gsum += ld<P>(gp, gi);
            }
        }
        float g = gsum * inv_world;

        float m = beta1 * ld<S>(m_in, i) + one_minus_beta1 * g;
        float v = beta2 * ld<S>(v_in, i) + one_minus_beta2 * g * g;

        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float w = ld<P>(flat_w, gi) - lr * (m_hat / (sqrtf(v_hat) + eps));

        st<P>(w_part_out, i, w);
        st<S>(m_out, i, m);
        st<S>(v_out, i, v);
    }
}

template <typename T>
__global__ void allgather_partitions_kernel(
    const long long* __restrict__ part_ptrs,
    T* __restrict__ flat_out,
    int64_t part,
    int world_size,
    int64_t total
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        int r = (int)(i / part);
        int64_t off = i - (int64_t)r * part;
        if (r < world_size) {
            const T* src = reinterpret_cast<const T*>((uintptr_t)part_ptrs[r]);
            flat_out[i] = src[off];
        }
    }
}

static inline int blocks_for(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

void pack_params(
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2,
    torch::Tensor flat
) {
    int64_t nW1 = W1.numel();
    int64_t nb1 = b1.numel();
    int64_t nW2 = W2.numel();
    int64_t nb2 = b2.numel();
    int64_t total = flat.numel();

    const int threads = 256;
    int blocks = blocks_for(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (flat.scalar_type() == torch::kBFloat16) {
        pack_params_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(W1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(W2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(flat.data_ptr<at::BFloat16>()),
            nW1, nb1, nW2, nb2, total);
    } else {
        pack_params_kernel<float><<<blocks, threads, 0, stream>>>(
            W1.data_ptr<float>(), b1.data_ptr<float>(),
            W2.data_ptr<float>(), b2.data_ptr<float>(),
            flat.data_ptr<float>(),
            nW1, nb1, nW2, nb2, total);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_from_uva(int64_t src_ptr, torch::Tensor dst, int64_t n) {
    const int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dst.scalar_type() == torch::kBFloat16) {
        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)src_ptr);
        copy_from_uva_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            src, reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()), n);
    } else {
        const float* src = reinterpret_cast<const float*>((uintptr_t)src_ptr);
        copy_from_uva_kernel<float><<<blocks, threads, 0, stream>>>(
            src, dst.data_ptr<float>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_bias_relu(torch::Tensor z, torch::Tensor b, int64_t rows, int64_t cols) {
    int64_t n = rows * cols;
    const int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (z.scalar_type() == torch::kBFloat16) {
        add_bias_relu_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(z.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            rows, cols);
    } else {
        add_bias_relu_kernel<float><<<blocks, threads, 0, stream>>>(
            z.data_ptr<float>(), b.data_ptr<float>(), rows, cols);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_bias_mse_grad(torch::Tensor out_as_grad, torch::Tensor y, torch::Tensor b, double scale, int64_t rows, int64_t cols) {
    int64_t n = rows * cols;
    const int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (out_as_grad.scalar_type() == torch::kBFloat16) {
        add_bias_mse_grad_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out_as_grad.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
            (float)scale, rows, cols);
    } else {
        add_bias_mse_grad_kernel<float><<<blocks, threads, 0, stream>>>(
            out_as_grad.data_ptr<float>(), y.data_ptr<float>(), b.data_ptr<float>(),
            (float)scale, rows, cols);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void relu_backward(torch::Tensor dh, torch::Tensor h, torch::Tensor dz) {
    int64_t n = dh.numel();
    const int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dh.scalar_type() == torch::kBFloat16) {
        relu_backward_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dz.data_ptr<at::BFloat16>()),
            n);
    } else {
        relu_backward_kernel<float><<<blocks, threads, 0, stream>>>(
            dh.data_ptr<float>(), h.data_ptr<float>(), dz.data_ptr<float>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bias_reduce(torch::Tensor grad, torch::Tensor db, int64_t rows, int64_t cols) {
    const int threads = 256;
    size_t shmem = threads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (grad.scalar_type() == torch::kBFloat16) {
        bias_reduce_kernel<__nv_bfloat16><<<cols, threads, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(db.data_ptr<at::BFloat16>()),
            rows, cols);
    } else {
        bias_reduce_kernel<float><<<cols, threads, shmem, stream>>>(
            grad.data_ptr<float>(), db.data_ptr<float>(), rows, cols);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_grads(
    torch::Tensor dW1,
    torch::Tensor db1,
    torch::Tensor dW2,
    torch::Tensor db2,
    torch::Tensor flat
) {
    int64_t nW1 = dW1.numel();
    int64_t nb1 = db1.numel();
    int64_t nW2 = dW2.numel();
    int64_t nb2 = db2.numel();
    int64_t total = flat.numel();

    const int threads = 256;
    int blocks = blocks_for(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (flat.scalar_type() == torch::kBFloat16) {
        pack_grads_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(dW1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(db1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(dW2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(db2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(flat.data_ptr<at::BFloat16>()),
            nW1, nb1, nW2, nb2, total);
    } else {
        pack_grads_kernel<float><<<blocks, threads, 0, stream>>>(
            dW1.data_ptr<float>(), db1.data_ptr<float>(),
            dW2.data_ptr<float>(), db2.data_ptr<float>(),
            flat.data_ptr<float>(),
            nW1, nb1, nW2, nb2, total);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adam_reduce_scatter_update(
    torch::Tensor grad_ptrs,
    torch::Tensor flat_w,
    torch::Tensor m_in,
    torch::Tensor v_in,
    torch::Tensor w_part_out,
    torch::Tensor m_out,
    torch::Tensor v_out,
    int64_t part,
    int64_t start,
    int world_size,
    double beta1,
    double beta2,
    double lr,
    double bc1,
    double bc2,
    double eps
) {
    const int threads = 256;
    int blocks = blocks_for(part, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* ptrs = reinterpret_cast<const long long*>(grad_ptrs.data_ptr<int64_t>());
    float b1 = (float)beta1;
    float b2 = (float)beta2;
    float omb1 = (float)(1.0 - beta1);
    float omb2 = (float)(1.0 - beta2);
    float invw = 1.0f / (float)world_size;

    bool p_bf16 = flat_w.scalar_type() == torch::kBFloat16;
    bool s_bf16 = m_in.scalar_type() == torch::kBFloat16;

    if (p_bf16 && s_bf16) {
        adam_reduce_scatter_update_kernel<__nv_bfloat16, __nv_bfloat16><<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<const __nv_bfloat16*>(flat_w.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(m_in.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v_in.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(w_part_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(m_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr<at::BFloat16>()),
            part, start, world_size, b1, b2, omb1, omb2, invw,
            (float)lr, (float)bc1, (float)bc2, (float)eps);
    } else if (p_bf16 && !s_bf16) {
        adam_reduce_scatter_update_kernel<__nv_bfloat16, float><<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<const __nv_bfloat16*>(flat_w.data_ptr<at::BFloat16>()),
            m_in.data_ptr<float>(), v_in.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(w_part_out.data_ptr<at::BFloat16>()),
            m_out.data_ptr<float>(), v_out.data_ptr<float>(),
            part, start, world_size, b1, b2, omb1, omb2, invw,
            (float)lr, (float)bc1, (float)bc2, (float)eps);
    } else if (!p_bf16 && s_bf16) {
        adam_reduce_scatter_update_kernel<float, __nv_bfloat16><<<blocks, threads, 0, stream>>>(
            ptrs,
            flat_w.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(m_in.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(v_in.data_ptr<at::BFloat16>()),
            w_part_out.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(m_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr<at::BFloat16>()),
            part, start, world_size, b1, b2, omb1, omb2, invw,
            (float)lr, (float)bc1, (float)bc2, (float)eps);
    } else {
        adam_reduce_scatter_update_kernel<float, float><<<blocks, threads, 0, stream>>>(
            ptrs,
            flat_w.data_ptr<float>(),
            m_in.data_ptr<float>(), v_in.data_ptr<float>(),
            w_part_out.data_ptr<float>(),
            m_out.data_ptr<float>(), v_out.data_ptr<float>(),
            part, start, world_size, b1, b2, omb1, omb2, invw,
            (float)lr, (float)bc1, (float)bc2, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void allgather_partitions(torch::Tensor part_ptrs, torch::Tensor flat_out, int64_t part, int world_size) {
    int64_t total = flat_out.numel();
    const int threads = 256;
    int blocks = blocks_for(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const long long* ptrs = reinterpret_cast<const long long*>(part_ptrs.data_ptr<int64_t>());

    if (flat_out.scalar_type() == torch::kBFloat16) {
        allgather_partitions_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<__nv_bfloat16*>(flat_out.data_ptr<at::BFloat16>()),
            part, world_size, total);
    } else {
        allgather_partitions_kernel<float><<<blocks, threads, 0, stream>>>(
            ptrs, flat_out.data_ptr<float>(), part, world_size, total);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_params", &pack_params, "Pack W1,b1,W2,b2 into flat buffer");
    m.def("copy_from_uva", &copy_from_uva, "Copy from UVA pointer to local tensor");
    m.def("add_bias_relu", &add_bias_relu, "Fused bias add + ReLU");
    m.def("add_bias_mse_grad", &add_bias_mse_grad, "Fused bias add + MSE dloss/dout");
    m.def("relu_backward", &relu_backward, "Fused ReLU backward");
    m.def("bias_reduce", &bias_reduce, "Reduce batch dimension for bias grad");
    m.def("pack_grads", &pack_grads, "Pack gradients into flat buffer");
    m.def("adam_reduce_scatter_update", &adam_reduce_scatter_update,
          "UVA reduce-scatter SUM/avg fused with Adam partition update");
    m.def("allgather_partitions", &allgather_partitions,
          "UVA all-gather of updated optimizer partitions");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("zero2_bf16_h100_symmem_cuda_ext", CUDA_SRC)
    return _ext


_resource_cache: Dict[Tuple, Tuple[Tensor, object, Tensor, object, Tensor, object, Tensor, Tensor]] = {}


def _dtype_ok(dtype: torch.dtype) -> bool:
    return dtype in (torch.bfloat16, torch.float32)


def _get_resources(
    total: int,
    part: int,
    param_dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
):
    key = (
        total,
        part,
        param_dtype,
        state_dtype,
        device.index,
        dist.get_world_size(),
    )
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    flat_param = symm_mem.empty(total, device=device, dtype=param_dtype)
    param_hdl = symm_mem.rendezvous(flat_param, dist.group.WORLD)

    flat_grad = symm_mem.empty(total, device=device, dtype=param_dtype)
    grad_hdl = symm_mem.rendezvous(flat_grad, dist.group.WORLD)

    part_buf = symm_mem.empty(part, device=device, dtype=param_dtype)
    part_hdl = symm_mem.rendezvous(part_buf, dist.group.WORLD)

    grad_ptrs = torch.tensor(grad_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    part_ptrs = torch.tensor(part_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (flat_param, param_hdl, flat_grad, grad_hdl, part_buf, part_hdl, grad_ptrs, part_ptrs)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_part: Tensor,
    exp_avg_sq_part: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    ZeRO-2 step with device-side symmetric-memory collectives:
    rank-0 parameter broadcast via UVA copy, reduce-scatter fused with Adam,
    and all-gather via peer reads of updated partitions. Dense GEMMs still use
    cuBLAS/Tensor Cores; surrounding pointwise/reduction/packing work is fused CUDA.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert step >= 1
    assert _dtype_ok(W1.dtype), "optimized path supports BF16/FP32 parameters"
    assert W1.dtype == b1.dtype == W2.dtype == b2.dtype
    assert X_local.dtype == W1.dtype and y_local.dtype == W1.dtype
    assert exp_avg_part.dtype == exp_avg_sq_part.dtype
    assert _dtype_ok(exp_avg_part.dtype), "optimizer state must be BF16 or FP32"

    ext = _get_ext()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if not W1.is_contiguous():
        W1 = W1.contiguous()
    if not b1.is_contiguous():
        b1 = b1.contiguous()
    if not W2.is_contiguous():
        W2 = W2.contiguous()
    if not b2.is_contiguous():
        b2 = b2.contiguous()
    if not X_local.is_contiguous():
        X_local = X_local.contiguous()
    if not y_local.is_contiguous():
        y_local = y_local.contiguous()
    if not exp_avg_part.is_contiguous():
        exp_avg_part = exp_avg_part.contiguous()
    if not exp_avg_sq_part.is_contiguous():
        exp_avg_sq_part = exp_avg_sq_part.contiguous()

    nW1 = W1.numel()
    nb1 = b1.numel()
    nW2 = W2.numel()
    nb2 = b2.numel()
    total = nW1 + nb1 + nW2 + nb2
    part = exp_avg_part.numel()
    assert total == part * world_size

    (
        flat_param,
        param_hdl,
        flat_grad,
        grad_hdl,
        part_buf,
        part_hdl,
        grad_ptrs,
        part_ptrs,
    ) = _get_resources(total, part, W1.dtype, exp_avg_part.dtype, W1.device)

    # Broadcast flattened parameters from rank 0 using symmetric memory.
    if rank == 0:
        ext.pack_params(W1, b1, W2, b2, flat_param)
    param_hdl.barrier(channel=0)

    rank0_param_ptr = int(param_hdl.buffer_ptrs[0])
    ext.copy_from_uva(rank0_param_ptr, flat_param, total)

    # Views into the broadcast flat parameter buffer.
    off0 = 0
    off1 = off0 + nW1
    off2 = off1 + nb1
    off3 = off2 + nW2

    W1v = flat_param.narrow(0, off0, nW1).view_as(W1)
    b1v = flat_param.narrow(0, off1, nb1).view_as(b1)
    W2v = flat_param.narrow(0, off2, nW2).view_as(W2)
    b2v = flat_param.narrow(0, off3, nb2).view_as(b2)

    # Manual forward/backward. GEMMs dispatch to H100 tensor cores for BF16.
    batch = X_local.shape[0]
    hidden = W1.shape[0]
    out_dim = W2.shape[0]

    h = torch.matmul(X_local, W1v.transpose(0, 1))
    ext.add_bias_relu(h, b1v, batch, hidden)

    dout = torch.matmul(h, W2v.transpose(0, 1))
    mse_scale = 2.0 / float(batch * out_dim)
    ext.add_bias_mse_grad(dout, y_local, b2v, mse_scale, batch, out_dim)

    dW2 = torch.matmul(dout.transpose(0, 1), h)
    db2 = torch.empty_like(b2v)
    ext.bias_reduce(dout, db2, batch, out_dim)

    dh = torch.matmul(dout, W2v)
    dz1 = torch.empty_like(dh)
    ext.relu_backward(dh, h, dz1)

    dW1 = torch.matmul(dz1.transpose(0, 1), X_local)
    db1 = torch.empty_like(b1v)
    ext.bias_reduce(dz1, db1, batch, hidden)

    # Publish full local gradient in symmetric memory.
    ext.pack_grads(dW1, db1, dW2, db2, flat_grad)
    grad_hdl.barrier(channel=0)

    # Reduce-scatter average + Adam update for this rank's shard.
    m_part = torch.empty_like(exp_avg_part)
    v_part = torch.empty_like(exp_avg_sq_part)

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    start = rank * part

    ext.adam_reduce_scatter_update(
        grad_ptrs,
        flat_param,
        exp_avg_part,
        exp_avg_sq_part,
        part_buf,
        m_part,
        v_part,
        part,
        start,
        world_size,
        float(beta1),
        float(beta2),
        float(lr),
        float(bc1),
        float(bc2),
        float(eps),
    )

    # Device-side all-gather of updated partitions into flat_param.
    part_hdl.barrier(channel=0)
    ext.allgather_partitions(part_ptrs, flat_param, part, world_size)

    out_W1 = flat_param.narrow(0, off0, nW1).view_as(W1)
    out_b1 = flat_param.narrow(0, off1, nb1).view_as(b1)
    out_W2 = flat_param.narrow(0, off2, nW2).view_as(W2)
    out_b2 = flat_param.narrow(0, off3, nb2).view_as(b2)

    return out_W1, out_b1, out_W2, out_b2, m_part, v_part


__all__ = ["solution"]