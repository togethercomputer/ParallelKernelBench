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
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).dtype() == torch::kBFloat16, #x " must be bfloat16")

static inline void check_cublas(cublasStatus_t st, const char* msg) {
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, msg, " cublasStatus=", (int)st);
}

static inline __nv_bfloat16* bf16_ptr(torch::Tensor t) {
    return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

static inline const __nv_bfloat16* cbf16_ptr(torch::Tensor t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

__device__ __forceinline__ float bf162f(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f2bf16(const float x) {
    return __float2bfloat16(x);
}

__global__ void pack_params_kernel(
    const __nv_bfloat16* __restrict__ W1,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ W2,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ flat,
    int64_t nW1,
    int64_t nb1,
    int64_t nW2,
    int64_t nb2
) {
    int64_t total = nW1 + nb1 + nW2 + nb2;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        if (idx < nW1) {
            flat[idx] = W1[idx];
        } else if (idx < nW1 + nb1) {
            flat[idx] = b1[idx - nW1];
        } else if (idx < nW1 + nb1 + nW2) {
            flat[idx] = W2[idx - nW1 - nb1];
        } else {
            flat[idx] = b2[idx - nW1 - nb1 - nW2];
        }
    }
}

__global__ void unpack_params_kernel(
    const __nv_bfloat16* __restrict__ flat,
    __nv_bfloat16* __restrict__ W1,
    __nv_bfloat16* __restrict__ b1,
    __nv_bfloat16* __restrict__ W2,
    __nv_bfloat16* __restrict__ b2,
    int64_t nW1,
    int64_t nb1,
    int64_t nW2,
    int64_t nb2
) {
    int64_t total = nW1 + nb1 + nW2 + nb2;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        if (idx < nW1) {
            W1[idx] = flat[idx];
        } else if (idx < nW1 + nb1) {
            b1[idx - nW1] = flat[idx];
        } else if (idx < nW1 + nb1 + nW2) {
            W2[idx - nW1 - nb1] = flat[idx];
        } else {
            b2[idx - nW1 - nb1 - nW2] = flat[idx];
        }
    }
}

__global__ void pack_grads_kernel(
    const __nv_bfloat16* __restrict__ gW1,
    const __nv_bfloat16* __restrict__ gb1,
    const __nv_bfloat16* __restrict__ gW2,
    const __nv_bfloat16* __restrict__ gb2,
    __nv_bfloat16* __restrict__ flat,
    int64_t nW1,
    int64_t nb1,
    int64_t nW2,
    int64_t nb2
) {
    int64_t total = nW1 + nb1 + nW2 + nb2;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        if (idx < nW1) {
            flat[idx] = gW1[idx];
        } else if (idx < nW1 + nb1) {
            flat[idx] = gb1[idx - nW1];
        } else if (idx < nW1 + nb1 + nW2) {
            flat[idx] = gW2[idx - nW1 - nb1];
        } else {
            flat[idx] = gb2[idx - nW1 - nb1 - nW2];
        }
    }
}

__global__ void bias_relu_kernel(
    __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ b,
    int64_t rows,
    int64_t cols
) {
    int64_t n = rows * cols;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t c = idx % cols;
        float v = bf162f(h[idx]) + bf162f(b[c]);
        h[idx] = f2bf16(v > 0.0f ? v : 0.0f);
    }
}

__global__ void bias_mse_dout_kernel(
    __nv_bfloat16* __restrict__ out_as_dout,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ y,
    int64_t rows,
    int64_t cols,
    float scale
) {
    int64_t n = rows * cols;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t c = idx % cols;
        float o = bf162f(out_as_dout[idx]) + bf162f(b[c]);
        float yy = bf162f(y[idx]);
        out_as_dout[idx] = f2bf16((o - yy) * scale);
    }
}

__global__ void relu_backward_kernel(
    __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h_relu,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float mask = bf162f(h_relu[idx]) > 0.0f ? 1.0f : 0.0f;
        dh[idx] = f2bf16(bf162f(dh[idx]) * mask);
    }
}

__global__ void reduce_bias_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ bgrad,
    int64_t rows,
    int64_t cols
) {
    int col = blockIdx.x;
    float sum = 0.0f;

    for (int64_t r = threadIdx.x; r < rows; r += blockDim.x) {
        sum += bf162f(x[r * cols + col]);
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        bgrad[col] = f2bf16(smem[0]);
    }
}

__global__ void adam_shard_f32mom_kernel(
    const int64_t* __restrict__ peer_bases,
    const __nv_bfloat16* __restrict__ rank0_params,
    float* __restrict__ m_out,
    float* __restrict__ v_out,
    const float* __restrict__ m_in,
    const float* __restrict__ v_in,
    __nv_bfloat16* __restrict__ shard_out,
    int64_t grad_off,
    int64_t shard_off,
    int64_t start,
    int64_t part,
    int world_size,
    float inv_world,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float bc1,
    float bc2,
    float lr,
    float eps
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < part; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t global = start + i;
        float gsum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* gbase =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)peer_bases[r]);
                gsum += bf162f(gbase[grad_off + global]);
            }
        }
        float g = gsum * inv_world;

        float m = m_in[i] * beta1 + g * one_minus_beta1;
        float v = v_in[i] * beta2 + g * g * one_minus_beta2;
        m_out[i] = m;
        v_out[i] = v;

        float mh = m / bc1;
        float vh = v / bc2;
        float w = bf162f(rank0_params[global]);
        w += -lr * (mh / (sqrtf(vh) + eps));
        shard_out[shard_off + i] = f2bf16(w);
    }
}

__global__ void adam_shard_bf16mom_kernel(
    const int64_t* __restrict__ peer_bases,
    const __nv_bfloat16* __restrict__ rank0_params,
    __nv_bfloat16* __restrict__ m_out,
    __nv_bfloat16* __restrict__ v_out,
    const __nv_bfloat16* __restrict__ m_in,
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ shard_out,
    int64_t grad_off,
    int64_t shard_off,
    int64_t start,
    int64_t part,
    int world_size,
    float inv_world,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float bc1,
    float bc2,
    float lr,
    float eps
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < part; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t global = start + i;
        float gsum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* gbase =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)peer_bases[r]);
                gsum += bf162f(gbase[grad_off + global]);
            }
        }
        float g = gsum * inv_world;

        float m = bf162f(m_in[i]) * beta1 + g * one_minus_beta1;
        float v = bf162f(v_in[i]) * beta2 + g * g * one_minus_beta2;
        m_out[i] = f2bf16(m);
        v_out[i] = f2bf16(v);

        float mh = m / bc1;
        float vh = v / bc2;
        float w = bf162f(rank0_params[global]);
        w += -lr * (mh / (sqrtf(vh) + eps));
        shard_out[shard_off + i] = f2bf16(w);
    }
}

__global__ void gather_shards_kernel(
    const int64_t* __restrict__ peer_bases,
    __nv_bfloat16* __restrict__ full_out,
    int64_t shard_off,
    int64_t part,
    int world_size
) {
    int64_t total = part * (int64_t)world_size;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int src_rank = (int)(idx / part);
        int64_t j = idx - (int64_t)src_rank * part;
        const __nv_bfloat16* src =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)peer_bases[src_rank]);
        full_out[idx] = src[shard_off + j];
    }
}

static inline int blocks_for(int64_t n, int threads=256) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

// Row-major BF16 C[M,N] = A[M,K] @ W[N,K]^T.
void gemm_linear_bf16(torch::Tensor A, torch::Tensor W, torch::Tensor C) {
    CHECK_CUDA(A); CHECK_CUDA(W); CHECK_CUDA(C);
    CHECK_CONTIG(A); CHECK_CONTIG(W); CHECK_CONTIG(C);
    CHECK_BF16(A); CHECK_BF16(W); CHECK_BF16(C);

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = W.size(0);
    TORCH_CHECK(W.size(1) == K);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream()), "cublasSetStream");
    check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");

    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            (int)N,
            (int)M,
            (int)K,
            &alpha,
            cbf16_ptr(W),
            CUDA_R_16BF,
            (int)K,
            cbf16_ptr(A),
            CUDA_R_16BF,
            (int)K,
            &beta,
            bf16_ptr(C),
            CUDA_R_16BF,
            (int)N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "gemm_linear_bf16");
}

// Row-major dW[N,K] = dY[M,N]^T @ H[M,K].
void gemm_grad_weight_bf16(torch::Tensor dY, torch::Tensor H, torch::Tensor dW) {
    CHECK_CUDA(dY); CHECK_CUDA(H); CHECK_CUDA(dW);
    CHECK_CONTIG(dY); CHECK_CONTIG(H); CHECK_CONTIG(dW);
    CHECK_BF16(dY); CHECK_BF16(H); CHECK_BF16(dW);

    int64_t M = dY.size(0);
    int64_t N = dY.size(1);
    int64_t K = H.size(1);
    TORCH_CHECK(H.size(0) == M);
    TORCH_CHECK(dW.size(0) == N && dW.size(1) == K);

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream()), "cublasSetStream");
    check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");

    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            (int)K,
            (int)N,
            (int)M,
            &alpha,
            cbf16_ptr(H),
            CUDA_R_16BF,
            (int)K,
            cbf16_ptr(dY),
            CUDA_R_16BF,
            (int)N,
            &beta,
            bf16_ptr(dW),
            CUDA_R_16BF,
            (int)K,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "gemm_grad_weight_bf16");
}

// Row-major dX[M,K] = dY[M,N] @ W[N,K].
void gemm_dinput_bf16(torch::Tensor dY, torch::Tensor W, torch::Tensor dX) {
    CHECK_CUDA(dY); CHECK_CUDA(W); CHECK_CUDA(dX);
    CHECK_CONTIG(dY); CHECK_CONTIG(W); CHECK_CONTIG(dX);
    CHECK_BF16(dY); CHECK_BF16(W); CHECK_BF16(dX);

    int64_t M = dY.size(0);
    int64_t N = dY.size(1);
    int64_t K = W.size(1);
    TORCH_CHECK(W.size(0) == N);
    TORCH_CHECK(dX.size(0) == M && dX.size(1) == K);

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream()), "cublasSetStream");
    check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");

    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            (int)K,
            (int)M,
            (int)N,
            &alpha,
            cbf16_ptr(W),
            CUDA_R_16BF,
            (int)K,
            cbf16_ptr(dY),
            CUDA_R_16BF,
            (int)N,
            &beta,
            bf16_ptr(dX),
            CUDA_R_16BF,
            (int)K,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "gemm_dinput_bf16");
}

void pack_params(
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2,
    torch::Tensor workspace,
    int64_t param_off
) {
    CHECK_BF16(W1); CHECK_BF16(b1); CHECK_BF16(W2); CHECK_BF16(b2); CHECK_BF16(workspace);
    int64_t nW1 = W1.numel(), nb1 = b1.numel(), nW2 = W2.numel(), nb2 = b2.numel();
    int64_t total = nW1 + nb1 + nW2 + nb2;
    __nv_bfloat16* dst = bf16_ptr(workspace) + param_off;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_params_kernel<<<blocks_for(total), 256, 0, stream>>>(
        cbf16_ptr(W1), cbf16_ptr(b1), cbf16_ptr(W2), cbf16_ptr(b2), dst,
        nW1, nb1, nW2, nb2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unpack_params_from_ptr(
    int64_t flat_ptr,
    torch::Tensor W1,
    torch::Tensor b1,
    torch::Tensor W2,
    torch::Tensor b2
) {
    CHECK_BF16(W1); CHECK_BF16(b1); CHECK_BF16(W2); CHECK_BF16(b2);
    int64_t nW1 = W1.numel(), nb1 = b1.numel(), nW2 = W2.numel(), nb2 = b2.numel();
    int64_t total = nW1 + nb1 + nW2 + nb2;
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)flat_ptr);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    unpack_params_kernel<<<blocks_for(total), 256, 0, stream>>>(
        src, bf16_ptr(W1), bf16_ptr(b1), bf16_ptr(W2), bf16_ptr(b2),
        nW1, nb1, nW2, nb2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_grads(
    torch::Tensor gW1,
    torch::Tensor gb1,
    torch::Tensor gW2,
    torch::Tensor gb2,
    torch::Tensor workspace,
    int64_t grad_off
) {
    CHECK_BF16(gW1); CHECK_BF16(gb1); CHECK_BF16(gW2); CHECK_BF16(gb2); CHECK_BF16(workspace);
    int64_t nW1 = gW1.numel(), nb1 = gb1.numel(), nW2 = gW2.numel(), nb2 = gb2.numel();
    int64_t total = nW1 + nb1 + nW2 + nb2;
    __nv_bfloat16* dst = bf16_ptr(workspace) + grad_off;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_grads_kernel<<<blocks_for(total), 256, 0, stream>>>(
        cbf16_ptr(gW1), cbf16_ptr(gb1), cbf16_ptr(gW2), cbf16_ptr(gb2), dst,
        nW1, nb1, nW2, nb2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_bias_relu(torch::Tensor h, torch::Tensor b) {
    CHECK_BF16(h); CHECK_BF16(b);
    int64_t rows = h.size(0), cols = h.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    bias_relu_kernel<<<blocks_for(rows * cols), 256, 0, stream>>>(
        bf16_ptr(h), cbf16_ptr(b), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_bias_make_dout(torch::Tensor out_as_dout, torch::Tensor b, torch::Tensor y, float scale) {
    CHECK_BF16(out_as_dout); CHECK_BF16(b); CHECK_BF16(y);
    int64_t rows = out_as_dout.size(0), cols = out_as_dout.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    bias_mse_dout_kernel<<<blocks_for(rows * cols), 256, 0, stream>>>(
        bf16_ptr(out_as_dout), cbf16_ptr(b), cbf16_ptr(y), rows, cols, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void relu_backward(torch::Tensor dh, torch::Tensor h_relu) {
    CHECK_BF16(dh); CHECK_BF16(h_relu);
    int64_t n = dh.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    relu_backward_kernel<<<blocks_for(n), 256, 0, stream>>>(
        bf16_ptr(dh), cbf16_ptr(h_relu), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_bias(torch::Tensor x, torch::Tensor bgrad) {
    CHECK_BF16(x); CHECK_BF16(bgrad);
    int64_t rows = x.size(0), cols = x.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_bias_kernel<<<(int)cols, 256, 0, stream>>>(
        cbf16_ptr(x), bf16_ptr(bgrad), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void adam_shard(
    torch::Tensor peer_bases,
    int64_t rank0_param_ptr,
    torch::Tensor m_in,
    torch::Tensor v_in,
    torch::Tensor m_out,
    torch::Tensor v_out,
    torch::Tensor workspace,
    int64_t grad_off,
    int64_t shard_off,
    int64_t start,
    int64_t part,
    int world_size,
    float beta1,
    float beta2,
    float bc1,
    float bc2,
    float lr,
    float eps
) {
    CHECK_CUDA(peer_bases); CHECK_CONTIG(peer_bases);
    CHECK_BF16(workspace);
    TORCH_CHECK(peer_bases.dtype() == torch::kInt64);
    TORCH_CHECK(m_in.dtype() == m_out.dtype());
    TORCH_CHECK(v_in.dtype() == v_out.dtype());
    TORCH_CHECK(m_in.dtype() == v_in.dtype());

    const int64_t* bases = peer_bases.data_ptr<int64_t>();
    const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)rank0_param_ptr);
    float inv_world = 1.0f / (float)world_size;
    float omb1 = 1.0f - beta1;
    float omb2 = 1.0f - beta2;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int blocks = blocks_for(part, 256);

    if (m_in.dtype() == torch::kFloat32) {
        adam_shard_f32mom_kernel<<<blocks, 256, 0, stream>>>(
            bases, p0,
            m_out.data_ptr<float>(), v_out.data_ptr<float>(),
            m_in.data_ptr<float>(), v_in.data_ptr<float>(),
            bf16_ptr(workspace),
            grad_off, shard_off, start, part, world_size, inv_world,
            beta1, beta2, omb1, omb2, bc1, bc2, lr, eps);
    } else {
        TORCH_CHECK(m_in.dtype() == torch::kBFloat16, "moments must be float32 or bfloat16");
        adam_shard_bf16mom_kernel<<<blocks, 256, 0, stream>>>(
            bases, p0,
            bf16_ptr(m_out), bf16_ptr(v_out),
            cbf16_ptr(m_in), cbf16_ptr(v_in),
            bf16_ptr(workspace),
            grad_off, shard_off, start, part, world_size, inv_world,
            beta1, beta2, omb1, omb2, bc1, bc2, lr, eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_shards(torch::Tensor peer_bases, torch::Tensor full_out, int64_t shard_off, int64_t part, int world_size) {
    CHECK_CUDA(peer_bases); CHECK_CONTIG(peer_bases);
    CHECK_BF16(full_out);
    TORCH_CHECK(peer_bases.dtype() == torch::kInt64);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t total = part * (int64_t)world_size;
    gather_shards_kernel<<<blocks_for(total), 256, 0, stream>>>(
        peer_bases.data_ptr<int64_t>(), bf16_ptr(full_out), shard_off, part, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_params", &pack_params, "Pack BF16 params into symmetric flat buffer");
    m.def("unpack_params_from_ptr", &unpack_params_from_ptr, "Unpack BF16 params from UVA ptr");
    m.def("pack_grads", &pack_grads, "Pack BF16 grads into symmetric flat buffer");

    m.def("gemm_linear_bf16", &gemm_linear_bf16, "BF16 linear GEMM");
    m.def("gemm_grad_weight_bf16", &gemm_grad_weight_bf16, "BF16 grad weight GEMM");
    m.def("gemm_dinput_bf16", &gemm_dinput_bf16, "BF16 input grad GEMM");

    m.def("add_bias_relu", &add_bias_relu, "Bias + ReLU");
    m.def("add_bias_make_dout", &add_bias_make_dout, "Bias + MSE backward dOut");
    m.def("relu_backward", &relu_backward, "ReLU backward");
    m.def("reduce_bias", &reduce_bias, "Column reduction for bias grad");

    m.def("adam_shard", &adam_shard, "Peer-load reduced ZeRO-1 Adam shard update");
    m.def("gather_shards", &gather_shards, "Peer-load all-gather of updated shards");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("zero1_bf16_h100_symm_cuda_ext", CUDA_SRC)
    return _ext


_resource_cache: Dict[Tuple, dict] = {}


def _numel4(W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor) -> int:
    return W1.numel() + b1.numel() + W2.numel() + b2.numel()


def _get_resources(
    X: Tensor,
    y: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_part: Tensor,
    world_size: int,
) -> dict:
    total = _numel4(W1, b1, W2, b2)
    part = exp_avg_part.numel()
    key = (
        torch.cuda.current_device(),
        tuple(X.shape),
        tuple(y.shape),
        tuple(W1.shape),
        tuple(b1.shape),
        tuple(W2.shape),
        tuple(b2.shape),
        X.dtype,
        W1.dtype,
        exp_avg_part.dtype,
        total,
        part,
        world_size,
    )
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    device = X.device

    # Symmetric BF16 workspace layout, in elements:
    # [0:total]                    rank0 broadcast params
    # [total:2*total]              per-rank local full gradient
    # [2*total:2*total+part]       per-rank updated owned shard
    workspace = symm_mem.empty((2 * total + part,), device=device, dtype=torch.bfloat16)
    hdl = symm_mem.rendezvous(workspace, dist.group.WORLD)
    peer_bases = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = {
        "workspace": workspace,
        "hdl": hdl,
        "peer_bases": peer_bases,
        "param_off": 0,
        "grad_off": total,
        "shard_off": 2 * total,
        "total": total,
        "part": part,
        "W1_work": torch.empty_like(W1),
        "b1_work": torch.empty_like(b1),
        "W2_work": torch.empty_like(W2),
        "b2_work": torch.empty_like(b2),
        "H": torch.empty((X.shape[0], W1.shape[0]), device=device, dtype=torch.bfloat16),
        "DOUT": torch.empty((X.shape[0], W2.shape[0]), device=device, dtype=torch.bfloat16),
        "DH": torch.empty((X.shape[0], W1.shape[0]), device=device, dtype=torch.bfloat16),
        "gW1": torch.empty_like(W1),
        "gb1": torch.empty_like(b1),
        "gW2": torch.empty_like(W2),
        "gb2": torch.empty_like(b2),
        "full_flat": torch.empty((total,), device=device, dtype=torch.bfloat16),
        "W1_out": torch.empty_like(W1),
        "b1_out": torch.empty_like(b1),
        "W2_out": torch.empty_like(W2),
        "b2_out": torch.empty_like(b2),
        "m_out": torch.empty_like(exp_avg_part),
        "v_out": torch.empty_like(exp_avg_part),
    }
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
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert step >= 1
    assert X_local.is_cuda and y_local.is_cuda
    assert W1.is_cuda and b1.is_cuda and W2.is_cuda and b2.is_cuda
    assert X_local.dtype == torch.bfloat16
    assert y_local.dtype == torch.bfloat16
    assert W1.dtype == torch.bfloat16
    assert b1.dtype == torch.bfloat16
    assert W2.dtype == torch.bfloat16
    assert b2.dtype == torch.bfloat16
    assert exp_avg_part.dtype in (torch.float32, torch.bfloat16)
    assert exp_avg_sq_part.dtype == exp_avg_part.dtype

    X = X_local.contiguous()
    y = y_local.contiguous()
    W1 = W1.contiguous()
    b1 = b1.contiguous()
    W2 = W2.contiguous()
    b2 = b2.contiguous()
    m_in = exp_avg_part.contiguous()
    v_in = exp_avg_sq_part.contiguous()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    total = _numel4(W1, b1, W2, b2)
    part = m_in.numel()
    assert total == part * world_size

    ext = _get_ext()
    res = _get_resources(X, y, W1, b1, W2, b2, m_in, world_size)

    workspace = res["workspace"]
    hdl = res["hdl"]
    peer_bases = res["peer_bases"]
    param_off = res["param_off"]
    grad_off = res["grad_off"]
    shard_off = res["shard_off"]

    # Device-side broadcast source: rank 0 packs the canonical full replica once.
    if rank == 0:
        ext.pack_params(W1, b1, W2, b2, workspace, param_off)

    hdl.barrier(channel=0)

    rank0_param_ptr = int(hdl.buffer_ptrs[0]) + param_off * 2
    ext.unpack_params_from_ptr(
        rank0_param_ptr,
        res["W1_work"],
        res["b1_work"],
        res["W2_work"],
        res["b2_work"],
    )

    # Forward: H = relu(X @ W1.T + b1), DOUT temp = X2 @ W2.T + b2, then DOUT = dLoss/dOut.
    ext.gemm_linear_bf16(X, res["W1_work"], res["H"])
    ext.add_bias_relu(res["H"], res["b1_work"])

    ext.gemm_linear_bf16(res["H"], res["W2_work"], res["DOUT"])
    mse_scale = 2.0 / float(y.numel())
    ext.add_bias_make_dout(res["DOUT"], res["b2_work"], y, float(mse_scale))

    # Backward.
    ext.gemm_grad_weight_bf16(res["DOUT"], res["H"], res["gW2"])
    ext.reduce_bias(res["DOUT"], res["gb2"])

    ext.gemm_dinput_bf16(res["DOUT"], res["W2_work"], res["DH"])
    ext.relu_backward(res["DH"], res["H"])

    ext.gemm_grad_weight_bf16(res["DH"], X, res["gW1"])
    ext.reduce_bias(res["DH"], res["gb1"])

    # Publish this rank's complete local gradient in symmetric memory.
    ext.pack_grads(res["gW1"], res["gb1"], res["gW2"], res["gb2"], workspace, grad_off)

    hdl.barrier(channel=1)

    # Fused ZeRO shard reduce + Adam: only reduce [rank*part:(rank+1)*part].
    start = rank * part
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    ext.adam_shard(
        peer_bases,
        rank0_param_ptr,
        m_in,
        v_in,
        res["m_out"],
        res["v_out"],
        workspace,
        grad_off,
        shard_off,
        start,
        part,
        world_size,
        float(beta1),
        float(beta2),
        float(bc1),
        float(bc2),
        float(lr),
        float(eps),
    )

    hdl.barrier(channel=2)

    # Device-side all-gather of updated shards, then unpack flat replica.
    ext.gather_shards(peer_bases, res["full_flat"], shard_off, part, world_size)
    ext.unpack_params_from_ptr(
        int(res["full_flat"].data_ptr()),
        res["W1_out"],
        res["b1_out"],
        res["W2_out"],
        res["b2_out"],
    )

    return (
        res["W1_out"],
        res["b1_out"],
        res["W2_out"],
        res["b2_out"],
        res["m_out"],
        res["v_out"],
    )


__all__ = ["solution"]