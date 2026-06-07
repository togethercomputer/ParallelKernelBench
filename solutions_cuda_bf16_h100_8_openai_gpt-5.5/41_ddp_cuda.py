from __future__ import annotations

import math
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  asm("trap;"); }} while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
  printf("CUBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
  asm("trap;"); }} while (0)

static inline cublasHandle_t get_blas_handle() {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream()));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    return handle;
}

__device__ __forceinline__ float bf162f(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f2bf16(const float x) {
    return __float2bfloat16_rn(x);
}

template <typename T>
__global__ void pack4_kernel(
    T* __restrict__ flat,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const T* __restrict__ c,
    const T* __restrict__ d,
    int64_t na,
    int64_t nb,
    int64_t nc,
    int64_t nd
) {
    int64_t n = na + nb + nc + nd;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        if (i < na) {
            flat[i] = a[i];
        } else if (i < na + nb) {
            flat[i] = b[i - na];
        } else if (i < na + nb + nc) {
            flat[i] = c[i - na - nb];
        } else {
            flat[i] = d[i - na - nb - nc];
        }
    }
}

template <typename T>
__global__ void copy_from_uva_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void add_bias_relu_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int64_t rows,
    int64_t cols
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t col = i % cols;
        float v = bf162f(x[i]) + bf162f(bias[col]);
        v = v > 0.0f ? v : 0.0f;
        x[i] = f2bf16(v);
    }
}

__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int64_t rows,
    int64_t cols
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t col = i % cols;
        float v = bf162f(x[i]) + bf162f(bias[col]);
        x[i] = f2bf16(v);
    }
}

__global__ void mse_grad_bf16_kernel(
    const __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ dout,
    float scale,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float g = (bf162f(out[i]) - bf162f(y[i])) * scale;
        dout[i] = f2bf16(g);
    }
}

__global__ void relu_backward_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float m = bf162f(h[i]) > 0.0f ? 1.0f : 0.0f;
        dh[i] = f2bf16(bf162f(dh[i]) * m);
    }
}

__global__ void bias_grad_bf16_kernel(
    const __nv_bfloat16* __restrict__ grad_mat,
    __nv_bfloat16* __restrict__ grad_bias,
    int64_t rows,
    int64_t cols
) {
    __shared__ float smem[256];
    int col = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int64_t r = tid; r < rows; r += blockDim.x) {
        sum += bf162f(grad_mat[r * cols + col]);
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        grad_bias[col] = f2bf16(smem[0]);
    }
}

__global__ void allreduce_adam_bf16_kernel(
    __nv_bfloat16* __restrict__ p,
    void* __restrict__ m_void,
    void* __restrict__ v_void,
    const long long* __restrict__ grad_ptrs,
    int world_size,
    int64_t n,
    int moment_dtype,  // 0=bf16, 1=float32
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float inv_bc1,
    float inv_bc2,
    float eps
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float gsum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(
                    static_cast<uintptr_t>(grad_ptrs[r]));
                gsum += bf162f(gp[i]);
            }
        }
        float g = gsum / (float)world_size;

        float m_old, v_old;
        if (moment_dtype == 0) {
            __nv_bfloat16* m = reinterpret_cast<__nv_bfloat16*>(m_void);
            __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(v_void);
            m_old = bf162f(m[i]);
            v_old = bf162f(v[i]);
            float m_new = beta1 * m_old + one_minus_beta1 * g;
            float v_new = beta2 * v_old + one_minus_beta2 * g * g;
            float upd = (m_new * inv_bc1) / (sqrtf(v_new * inv_bc2) + eps);
            float p_new = bf162f(p[i]) - lr * upd;
            p[i] = f2bf16(p_new);
            m[i] = f2bf16(m_new);
            v[i] = f2bf16(v_new);
        } else {
            float* m = reinterpret_cast<float*>(m_void);
            float* v = reinterpret_cast<float*>(v_void);
            float m_new = beta1 * m[i] + one_minus_beta1 * g;
            float v_new = beta2 * v[i] + one_minus_beta2 * g * g;
            float upd = (m_new * inv_bc1) / (sqrtf(v_new * inv_bc2) + eps);
            float p_new = bf162f(p[i]) - lr * upd;
            p[i] = f2bf16(p_new);
            m[i] = m_new;
            v[i] = v_new;
        }
    }
}

// C[M,N] row-major = X[M,K] row-major * W[N,K]^T row-major.
void linear_forward_bf16(torch::Tensor X, torch::Tensor W, torch::Tensor C,
                         int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && C.is_cuda());
    TORCH_CHECK(X.dtype() == torch::kBFloat16 && W.dtype() == torch::kBFloat16 && C.dtype() == torch::kBFloat16);
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t h = get_blas_handle();
    CUBLAS_CHECK(cublasGemmEx(
        h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        W.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)K,
        X.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)K,
        &beta,
        C.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// C[M,N] row-major = A[M,K] row-major * B[K,N] row-major.
void row_nn_bf16(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                 int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16 && C.dtype() == torch::kBFloat16);
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t h = get_blas_handle();
    CUBLAS_CHECK(cublasGemmEx(
        h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        B.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)N,
        A.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)K,
        &beta,
        C.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// C[M,N] row-major = A[K,M]^T row-major * B[K,N] row-major.
void row_at_b_bf16(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                   int64_t K, int64_t M, int64_t N) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16 && C.dtype() == torch::kBFloat16);
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t h = get_blas_handle();
    CUBLAS_CHECK(cublasGemmEx(
        h,
        CUBLAS_OP_N, CUBLAS_OP_T,
        (int)N, (int)M, (int)K,
        &alpha,
        B.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)N,
        A.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)M,
        &beta,
        C.data_ptr<at::BFloat16>(), CUDA_R_16BF, (int)N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static inline int blocks_for(int64_t n) {
    int b = (int)((n + 255) / 256);
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return b;
}

void pack4_bf16(torch::Tensor flat, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor d) {
    int64_t na = a.numel(), nb = b.numel(), nc = c.numel(), nd = d.numel();
    int64_t n = na + nb + nc + nd;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    pack4_kernel<__nv_bfloat16><<<blocks_for(n), 256, 0, stream>>>(
        (__nv_bfloat16*)flat.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)a.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)b.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)c.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)d.data_ptr<at::BFloat16>(),
        na, nb, nc, nd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack4_f32(torch::Tensor flat, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor d) {
    int64_t na = a.numel(), nb = b.numel(), nc = c.numel(), nd = d.numel();
    int64_t n = na + nb + nc + nd;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    pack4_kernel<float><<<blocks_for(n), 256, 0, stream>>>(
        flat.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), d.data_ptr<float>(),
        na, nb, nc, nd);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_from_uva_bf16(torch::Tensor dst, int64_t src_ptr, int64_t n) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(src_ptr));
    copy_from_uva_kernel<__nv_bfloat16><<<blocks_for(n), 256, 0, stream>>>(
        (__nv_bfloat16*)dst.data_ptr<at::BFloat16>(), src, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_from_uva_f32(torch::Tensor dst, int64_t src_ptr, int64_t n) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const float* src = reinterpret_cast<const float*>(static_cast<uintptr_t>(src_ptr));
    copy_from_uva_kernel<float><<<blocks_for(n), 256, 0, stream>>>(
        dst.data_ptr<float>(), src, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_bias_relu_bf16(torch::Tensor x, torch::Tensor bias, int64_t rows, int64_t cols) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t n = rows * cols;
    add_bias_relu_bf16_kernel<<<blocks_for(n), 256, 0, stream>>>(
        (__nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)bias.data_ptr<at::BFloat16>(),
        rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_bias_bf16(torch::Tensor x, torch::Tensor bias, int64_t rows, int64_t cols) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t n = rows * cols;
    add_bias_bf16_kernel<<<blocks_for(n), 256, 0, stream>>>(
        (__nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)bias.data_ptr<at::BFloat16>(),
        rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mse_grad_bf16(torch::Tensor out, torch::Tensor y, torch::Tensor dout, float scale, int64_t n) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mse_grad_bf16_kernel<<<blocks_for(n), 256, 0, stream>>>(
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)y.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)dout.data_ptr<at::BFloat16>(),
        scale, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void relu_backward_inplace_bf16(torch::Tensor dh, torch::Tensor h, int64_t n) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    relu_backward_inplace_bf16_kernel<<<blocks_for(n), 256, 0, stream>>>(
        (__nv_bfloat16*)dh.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)h.data_ptr<at::BFloat16>(),
        n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bias_grad_bf16(torch::Tensor grad_mat, torch::Tensor grad_bias, int64_t rows, int64_t cols) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bias_grad_bf16_kernel<<<(int)cols, 256, 0, stream>>>(
        (__nv_bfloat16*)grad_mat.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)grad_bias.data_ptr<at::BFloat16>(),
        rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void allreduce_adam_bf16(
    torch::Tensor p,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor grad_ptrs,
    int64_t n,
    int moment_dtype,
    float lr,
    float beta1,
    float beta2,
    float bc1,
    float bc2,
    float eps
) {
    int world_size = (int)grad_ptrs.size(0);
    float omb1 = 1.0f - beta1;
    float omb2 = 1.0f - beta2;
    float inv_bc1 = 1.0f / bc1;
    float inv_bc2 = 1.0f / bc2;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    allreduce_adam_bf16_kernel<<<blocks_for(n), 256, 0, stream>>>(
        (__nv_bfloat16*)p.data_ptr<at::BFloat16>(),
        m.data_ptr(),
        v.data_ptr(),
        (const long long*)grad_ptrs.data_ptr<int64_t>(),
        world_size,
        n,
        moment_dtype,
        lr,
        beta1,
        beta2,
        omb1,
        omb2,
        inv_bc1,
        inv_bc2,
        eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack4_bf16", &pack4_bf16);
    m.def("pack4_f32", &pack4_f32);
    m.def("copy_from_uva_bf16", &copy_from_uva_bf16);
    m.def("copy_from_uva_f32", &copy_from_uva_f32);
    m.def("linear_forward_bf16", &linear_forward_bf16);
    m.def("row_nn_bf16", &row_nn_bf16);
    m.def("row_at_b_bf16", &row_at_b_bf16);
    m.def("add_bias_relu_bf16", &add_bias_relu_bf16);
    m.def("add_bias_bf16", &add_bias_bf16);
    m.def("mse_grad_bf16", &mse_grad_bf16);
    m.def("relu_backward_inplace_bf16", &relu_backward_inplace_bf16);
    m.def("bias_grad_bf16", &bias_grad_bf16);
    m.def("allreduce_adam_bf16", &allreduce_adam_bf16);
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ddp_bf16_symm_adam_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _numel4(W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor) -> tuple[int, int, int, int, int]:
    n1 = W1.numel()
    n2 = b1.numel()
    n3 = W2.numel()
    n4 = b2.numel()
    return n1, n2, n3, n4, n1 + n2 + n3 + n4


def _moment_dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    raise AssertionError("Adam moment tensors must be bfloat16 or float32")


def _pack4(ext, flat: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor):
    if flat.dtype == torch.bfloat16:
        ext.pack4_bf16(flat, a, b, c, d)
    elif flat.dtype == torch.float32:
        ext.pack4_f32(flat, a, b, c, d)
    else:
        raise AssertionError("unsupported dtype")


def _copy_from_rank0(ext, dst: Tensor, hdl, n: int):
    src = int(hdl.buffer_ptrs[0])
    if dst.dtype == torch.bfloat16:
        ext.copy_from_uva_bf16(dst, src, n)
    elif dst.dtype == torch.float32:
        ext.copy_from_uva_f32(dst, src, n)
    else:
        raise AssertionError("unsupported dtype")


def _get_resources(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_W1: Tensor,
    exp_avg_sq_W1: Tensor,
):
    n1, n2, n3, n4, total = _numel4(W1, b1, W2, b2)
    key = (
        X_local.shape,
        y_local.shape,
        W1.shape,
        b1.shape,
        W2.shape,
        b2.shape,
        W1.dtype,
        exp_avg_W1.dtype,
        exp_avg_sq_W1.dtype,
        X_local.device,
        dist.get_world_size(),
    )
    if key in _resource_cache:
        return _resource_cache[key]

    device = X_local.device
    param_dtype = W1.dtype
    m_dtype = exp_avg_W1.dtype
    v_dtype = exp_avg_sq_W1.dtype

    init_p = symm_mem.empty((total,), device=device, dtype=param_dtype)
    init_m = symm_mem.empty((total,), device=device, dtype=m_dtype)
    init_v = symm_mem.empty((total,), device=device, dtype=v_dtype)
    grad_symm = symm_mem.empty((total,), device=device, dtype=param_dtype)

    hdl_p = symm_mem.rendezvous(init_p, dist.group.WORLD)
    hdl_m = symm_mem.rendezvous(init_m, dist.group.WORLD)
    hdl_v = symm_mem.rendezvous(init_v, dist.group.WORLD)
    hdl_g = symm_mem.rendezvous(grad_symm, dist.group.WORLD)

    p = torch.empty((total,), device=device, dtype=param_dtype)
    m = torch.empty((total,), device=device, dtype=m_dtype)
    v = torch.empty((total,), device=device, dtype=v_dtype)

    local_n = X_local.shape[0]
    hidden = W1.shape[0]
    out_dim = W2.shape[0]

    h_act = torch.empty((local_n, hidden), device=device, dtype=param_dtype)
    out = torch.empty((local_n, out_dim), device=device, dtype=param_dtype)
    dout = torch.empty((local_n, out_dim), device=device, dtype=param_dtype)
    dh = torch.empty((local_n, hidden), device=device, dtype=param_dtype)

    grad_ptrs = torch.tensor(hdl_g.buffer_ptrs, device=device, dtype=torch.int64)

    res = {
        "n1": n1,
        "n2": n2,
        "n3": n3,
        "n4": n4,
        "total": total,
        "init_p": init_p,
        "init_m": init_m,
        "init_v": init_v,
        "grad": grad_symm,
        "hdl_p": hdl_p,
        "hdl_m": hdl_m,
        "hdl_v": hdl_v,
        "hdl_g": hdl_g,
        "p": p,
        "m": m,
        "v": v,
        "h": h_act,
        "out": out,
        "dout": dout,
        "dh": dh,
        "grad_ptrs": grad_ptrs,
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
    exp_avg_W1: Tensor,
    exp_avg_b1: Tensor,
    exp_avg_W2: Tensor,
    exp_avg_b2: Tensor,
    exp_avg_sq_W1: Tensor,
    exp_avg_sq_b1: Tensor,
    exp_avg_sq_W2: Tensor,
    exp_avg_sq_b2: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, ...]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert step >= 1
    assert X_local.is_cuda and y_local.is_cuda
    assert W1.dtype == torch.bfloat16 and b1.dtype == torch.bfloat16
    assert W2.dtype == torch.bfloat16 and b2.dtype == torch.bfloat16
    assert X_local.dtype == torch.bfloat16 and y_local.dtype == torch.bfloat16
    assert W1.is_contiguous() and b1.is_contiguous() and W2.is_contiguous() and b2.is_contiguous()
    assert X_local.is_contiguous() and y_local.is_contiguous()

    ext = _get_ext()
    rank = dist.get_rank()

    res = _get_resources(
        X_local, y_local, W1, b1, W2, b2, exp_avg_W1, exp_avg_sq_W1
    )

    n1 = res["n1"]
    n2 = res["n2"]
    n3 = res["n3"]
    n4 = res["n4"]
    total = res["total"]

    if rank == 0:
        _pack4(ext, res["init_p"], W1, b1, W2, b2)
        _pack4(ext, res["init_m"], exp_avg_W1, exp_avg_b1, exp_avg_W2, exp_avg_b2)
        _pack4(ext, res["init_v"], exp_avg_sq_W1, exp_avg_sq_b1, exp_avg_sq_W2, exp_avg_sq_b2)

    res["hdl_p"].barrier(channel=0)
    res["hdl_m"].barrier(channel=1)
    res["hdl_v"].barrier(channel=2)

    _copy_from_rank0(ext, res["p"], res["hdl_p"], total)
    _copy_from_rank0(ext, res["m"], res["hdl_m"], total)
    _copy_from_rank0(ext, res["v"], res["hdl_v"], total)

    p = res["p"]
    m = res["m"]
    v = res["v"]
    grad = res["grad"]

    W1_l = p.narrow(0, 0, n1).view_as(W1)
    b1_l = p.narrow(0, n1, n2).view_as(b1)
    W2_l = p.narrow(0, n1 + n2, n3).view_as(W2)
    b2_l = p.narrow(0, n1 + n2 + n3, n4).view_as(b2)

    gW1 = grad.narrow(0, 0, n1).view_as(W1)
    gb1 = grad.narrow(0, n1, n2).view_as(b1)
    gW2 = grad.narrow(0, n1 + n2, n3).view_as(W2)
    gb2 = grad.narrow(0, n1 + n2 + n3, n4).view_as(b2)

    local_n = X_local.shape[0]
    d_in = X_local.shape[1]
    hidden = W1.shape[0]
    out_dim = W2.shape[0]

    h_act = res["h"]
    out = res["out"]
    dout = res["dout"]
    dh = res["dh"]

    # Forward: h = relu(X @ W1.T + b1), out = h @ W2.T + b2.
    ext.linear_forward_bf16(X_local, W1_l, h_act, local_n, hidden, d_in)
    ext.add_bias_relu_bf16(h_act, b1_l, local_n, hidden)

    ext.linear_forward_bf16(h_act, W2_l, out, local_n, out_dim, hidden)
    ext.add_bias_bf16(out, b2_l, local_n, out_dim)

    # Backward for mean squared error.
    scale = 2.0 / float(local_n * out_dim)
    ext.mse_grad_bf16(out, y_local, dout, scale, dout.numel())

    # gW2 = dout.T @ h, gb2 = sum(dout).
    ext.row_at_b_bf16(dout, h_act, gW2, local_n, out_dim, hidden)
    ext.bias_grad_bf16(dout, gb2, local_n, out_dim)

    # dh = dout @ W2; dz1 = dh * relu'(h).
    ext.row_nn_bf16(dout, W2_l, dh, local_n, hidden, out_dim)
    ext.relu_backward_inplace_bf16(dh, h_act, dh.numel())

    # gW1 = dz1.T @ X, gb1 = sum(dz1).
    ext.row_at_b_bf16(dh, X_local, gW1, local_n, hidden, d_in)
    ext.bias_grad_bf16(dh, gb1, local_n, hidden)

    # Symmetric-memory gradient visibility, then fused UVA all-reduce average + Adam.
    res["hdl_g"].barrier(channel=3)

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    moment_dtype = _moment_dtype_enum(m.dtype)
    assert moment_dtype == _moment_dtype_enum(v.dtype)

    ext.allreduce_adam_bf16(
        p,
        m,
        v,
        res["grad_ptrs"],
        total,
        moment_dtype,
        float(lr),
        float(beta1),
        float(beta2),
        float(bc1),
        float(bc2),
        float(eps),
    )

    W1_o = p.narrow(0, 0, n1).view_as(W1)
    b1_o = p.narrow(0, n1, n2).view_as(b1)
    W2_o = p.narrow(0, n1 + n2, n3).view_as(W2)
    b2_o = p.narrow(0, n1 + n2 + n3, n4).view_as(b2)

    mW1 = m.narrow(0, 0, n1).view_as(exp_avg_W1)
    mb1 = m.narrow(0, n1, n2).view_as(exp_avg_b1)
    mW2 = m.narrow(0, n1 + n2, n3).view_as(exp_avg_W2)
    mb2 = m.narrow(0, n1 + n2 + n3, n4).view_as(exp_avg_b2)

    vW1 = v.narrow(0, 0, n1).view_as(exp_avg_sq_W1)
    vb1 = v.narrow(0, n1, n2).view_as(exp_avg_sq_b1)
    vW2 = v.narrow(0, n1 + n2, n3).view_as(exp_avg_sq_W2)
    vb2 = v.narrow(0, n1 + n2 + n3, n4).view_as(exp_avg_sq_b2)

    return (W1_o, b1_o, W2_o, b2_o, mW1, mb1, mW2, mb2, vW1, vb1, vW2, vb2)


__all__ = ["solution"]