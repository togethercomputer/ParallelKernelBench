from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cmath>

#define CUDA_CHECK(cmd) do {                                      \
    cudaError_t e = (cmd);                                        \
    TORCH_CHECK(e == cudaSuccess, "CUDA error: ",                 \
                cudaGetErrorString(e));                           \
} while (0)

#define CUBLAS_CHECK(cmd) do {                                    \
    cublasStatus_t s = (cmd);                                     \
    TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", s); \
} while (0)

template <typename T>
__global__ void copy3_kernel(
    T* __restrict__ d1, const T* __restrict__ s1, int64_t n1,
    T* __restrict__ d2, const T* __restrict__ s2, int64_t n2,
    T* __restrict__ d3, const T* __restrict__ s3, int64_t n3
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t nmax = max(n1, max(n2, n3));
    for (; idx < nmax; idx += stride) {
        if (idx < n1) d1[idx] = s1[idx];
        if (idx < n2) d2[idx] = s2[idx];
        if (idx < n3) d3[idx] = s3[idx];
    }
}

template <typename T>
__global__ void gather_dim0_pair_kernel(
    const long long* __restrict__ ptrs1,
    const long long* __restrict__ ptrs2,
    T* __restrict__ dst1,
    T* __restrict__ dst2,
    int n_tp,
    int n_fsdp,
    int tp_rank,
    int64_t rows_shard,
    int64_t cols
) {
    const int64_t shard_elems = rows_shard * cols;
    const int64_t total = (int64_t)n_fsdp * shard_elems;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int fsdp_src = (int)(idx / shard_elems);
        int64_t local_idx = idx - (int64_t)fsdp_src * shard_elems;
        int src_rank = fsdp_src * n_tp + tp_rank;

        const T* src1 = reinterpret_cast<const T*>((uintptr_t)ptrs1[src_rank]);
        const T* src2 = reinterpret_cast<const T*>((uintptr_t)ptrs2[src_rank]);

        dst1[idx] = src1[local_idx];
        dst2[idx] = src2[local_idx];
    }
}

template <typename T>
__global__ void gather_dim1_kernel(
    const long long* __restrict__ ptrs,
    T* __restrict__ dst,
    int n_tp,
    int n_fsdp,
    int tp_rank,
    int64_t rows,
    int64_t cols_shard
) {
    const int64_t full_cols = (int64_t)n_fsdp * cols_shard;
    const int64_t total = rows * full_cols;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t r = idx / full_cols;
        int64_t c = idx - r * full_cols;

        int fsdp_src = (int)(c / cols_shard);
        int64_t lc = c - (int64_t)fsdp_src * cols_shard;
        int src_rank = fsdp_src * n_tp + tp_rank;

        const T* src = reinterpret_cast<const T*>((uintptr_t)ptrs[src_rank]);
        dst[idx] = src[r * cols_shard + lc];
    }
}

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ x1,
    const __nv_bfloat16* __restrict__ x2,
    __nv_bfloat16* __restrict__ z,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float a = __bfloat162float(x1[idx]);
        float b = __bfloat162float(x2[idx]);
        float v = (a / (1.0f + expf(-a))) * b;
        z[idx] = __float2bfloat16(v);
    }
}

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ z,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float a = x1[idx];
        float b = x2[idx];
        z[idx] = (a / (1.0f + expf(-a))) * b;
    }
}

__global__ void allreduce_tp_bf16_kernel(
    const long long* __restrict__ y_ptrs,
    __nv_bfloat16* __restrict__ out,
    int n_tp,
    int fsdp_rank,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int base_rank = fsdp_rank * n_tp;

    for (; idx < n; idx += stride) {
        float acc = 0.0f;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < n_tp) {
                const __nv_bfloat16* src =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)y_ptrs[base_rank + t]);
                acc += __bfloat162float(src[idx]);
            }
        }
        out[idx] = __float2bfloat16(acc);
    }
}

__global__ void allreduce_tp_f32_kernel(
    const long long* __restrict__ y_ptrs,
    float* __restrict__ out,
    int n_tp,
    int fsdp_rank,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int base_rank = fsdp_rank * n_tp;

    for (; idx < n; idx += stride) {
        float acc = 0.0f;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < n_tp) {
                const float* src =
                    reinterpret_cast<const float*>((uintptr_t)y_ptrs[base_rank + t]);
                acc += src[idx];
            }
        }
        out[idx] = acc;
    }
}

static inline int launch_blocks(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

void copy3(
    torch::Tensor d1, torch::Tensor s1,
    torch::Tensor d2, torch::Tensor s2,
    torch::Tensor d3, torch::Tensor s3,
    int64_t n1,
    int64_t n2,
    int64_t n3,
    int dtype_enum
) {
    TORCH_CHECK(d1.is_cuda() && s1.is_cuda(), "copy3 tensors must be CUDA");
    const int threads = 256;
    const int blocks = launch_blocks(max(n1, max(n2, n3)), threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        copy3_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(d1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(s1.data_ptr<at::BFloat16>()),
            n1,
            reinterpret_cast<__nv_bfloat16*>(d2.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(s2.data_ptr<at::BFloat16>()),
            n2,
            reinterpret_cast<__nv_bfloat16*>(d3.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(s3.data_ptr<at::BFloat16>()),
            n3
        );
    } else {
        copy3_kernel<float><<<blocks, threads, 0, stream>>>(
            d1.data_ptr<float>(), s1.data_ptr<float>(), n1,
            d2.data_ptr<float>(), s2.data_ptr<float>(), n2,
            d3.data_ptr<float>(), s3.data_ptr<float>(), n3
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_dim0_pair(
    torch::Tensor ptrs1,
    torch::Tensor ptrs2,
    torch::Tensor dst1,
    torch::Tensor dst2,
    int n_tp,
    int n_fsdp,
    int tp_rank,
    int64_t rows_shard,
    int64_t cols,
    int dtype_enum
) {
    const int64_t total = (int64_t)n_fsdp * rows_shard * cols;
    const int threads = 256;
    const int blocks = launch_blocks(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* p1 = reinterpret_cast<const long long*>(ptrs1.data_ptr<int64_t>());
    const long long* p2 = reinterpret_cast<const long long*>(ptrs2.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        gather_dim0_pair_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            p1, p2,
            reinterpret_cast<__nv_bfloat16*>(dst1.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dst2.data_ptr<at::BFloat16>()),
            n_tp, n_fsdp, tp_rank, rows_shard, cols
        );
    } else {
        gather_dim0_pair_kernel<float><<<blocks, threads, 0, stream>>>(
            p1, p2,
            dst1.data_ptr<float>(),
            dst2.data_ptr<float>(),
            n_tp, n_fsdp, tp_rank, rows_shard, cols
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_dim1(
    torch::Tensor ptrs,
    torch::Tensor dst,
    int n_tp,
    int n_fsdp,
    int tp_rank,
    int64_t rows,
    int64_t cols_shard,
    int dtype_enum
) {
    const int64_t total = rows * cols_shard * (int64_t)n_fsdp;
    const int threads = 256;
    const int blocks = launch_blocks(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* p = reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        gather_dim1_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            p,
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
            n_tp, n_fsdp, tp_rank, rows, cols_shard
        );
    } else {
        gather_dim1_kernel<float><<<blocks, threads, 0, stream>>>(
            p,
            dst.data_ptr<float>(),
            n_tp, n_fsdp, tp_rank, rows, cols_shard
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void silu_mul(torch::Tensor x1, torch::Tensor x2, torch::Tensor z, int64_t n, int dtype_enum) {
    const int threads = 256;
    const int blocks = launch_blocks(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        silu_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(z.data_ptr<at::BFloat16>()),
            n
        );
    } else {
        silu_mul_f32_kernel<<<blocks, threads, 0, stream>>>(
            x1.data_ptr<float>(), x2.data_ptr<float>(), z.data_ptr<float>(), n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm_rowmajor(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                   int64_t M64, int64_t N64, int64_t K64, int dtype_enum) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "GEMM tensors must be CUDA");
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX, "GEMM dims too large");

    int M = (int)M64;
    int N = (int)N64;
    int K = (int)K64;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Row-major C[M,N] = A[M,K] @ B[K,N].
    // Interpret as column-major C^T[N,M] = B^T[N,K] @ A^T[K,M].
    if (dtype_enum == 0) {
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            reinterpret_cast<const void*>(B.data_ptr<at::BFloat16>()),
            CUDA_R_16BF,
            N,
            reinterpret_cast<const void*>(A.data_ptr<at::BFloat16>()),
            CUDA_R_16BF,
            K,
            &beta,
            reinterpret_cast<void*>(C.data_ptr<at::BFloat16>()),
            CUDA_R_16BF,
            N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    } else {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            B.data_ptr<float>(),
            N,
            A.data_ptr<float>(),
            K,
            &beta,
            C.data_ptr<float>(),
            N
        ));
    }
}

void allreduce_tp(
    torch::Tensor y_ptrs,
    torch::Tensor out,
    int n_tp,
    int fsdp_rank,
    int64_t n,
    int dtype_enum
) {
    const int threads = 256;
    const int blocks = launch_blocks(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* p = reinterpret_cast<const long long*>(y_ptrs.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        allreduce_tp_bf16_kernel<<<blocks, threads, 0, stream>>>(
            p,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n_tp,
            fsdp_rank,
            n
        );
    } else {
        allreduce_tp_f32_kernel<<<blocks, threads, 0, stream>>>(
            p,
            out.data_ptr<float>(),
            n_tp,
            fsdp_rank,
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy3", &copy3, "copy three local shards into symmetric buffers");
    m.def("gather_dim0_pair", &gather_dim0_pair, "FSDP gather dim0 for W1/W2 via UVA");
    m.def("gather_dim1", &gather_dim1, "FSDP gather dim1 for W3 via UVA");
    m.def("silu_mul", &silu_mul, "fused SiLU(x1) * x2");
    m.def("gemm_rowmajor", &gemm_rowmajor, "row-major BF16/FP32 GEMM");
    m.def("allreduce_tp", &allreduce_tp, "TP all-reduce SUM via UVA peer loads");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_tp_bf16_h100_symm_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    raise AssertionError("optimized path supports torch.bfloat16 and torch.float32")


def _ptr_tensor(hdl, device: torch.device) -> Tensor:
    return torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)


def _get_resources(
    x_shape,
    w1_shape,
    w2_shape,
    w3_shape,
    dtype: torch.dtype,
    device: torch.device,
    n_tp: int,
    n_fsdp: int,
):
    key = (tuple(x_shape), tuple(w1_shape), tuple(w2_shape), tuple(w3_shape),
           dtype, device, n_tp, n_fsdp)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    Bf, D = x_shape
    D_shard, Htp = w1_shape
    Htp3, D_shard3 = w3_shape

    w1_sym = symm_mem.empty(w1_shape, device=device, dtype=dtype)
    hdl_w1 = symm_mem.rendezvous(w1_sym, dist.group.WORLD)

    w2_sym = symm_mem.empty(w2_shape, device=device, dtype=dtype)
    hdl_w2 = symm_mem.rendezvous(w2_sym, dist.group.WORLD)

    w3_sym = symm_mem.empty(w3_shape, device=device, dtype=dtype)
    hdl_w3 = symm_mem.rendezvous(w3_sym, dist.group.WORLD)

    y_sym = symm_mem.empty((Bf, D), device=device, dtype=dtype)
    hdl_y = symm_mem.rendezvous(y_sym, dist.group.WORLD)

    W1_full = torch.empty((D_shard * n_fsdp, Htp), device=device, dtype=dtype)
    W2_full = torch.empty((D_shard * n_fsdp, Htp), device=device, dtype=dtype)
    W3_full = torch.empty((Htp3, D_shard3 * n_fsdp), device=device, dtype=dtype)

    x1 = torch.empty((Bf, Htp), device=device, dtype=dtype)
    x2 = torch.empty((Bf, Htp), device=device, dtype=dtype)
    z = torch.empty((Bf, Htp), device=device, dtype=dtype)
    out = torch.empty((Bf, D), device=device, dtype=dtype)

    ptr_w1 = _ptr_tensor(hdl_w1, device)
    ptr_w2 = _ptr_tensor(hdl_w2, device)
    ptr_w3 = _ptr_tensor(hdl_w3, device)
    ptr_y = _ptr_tensor(hdl_y, device)

    comm_stream = torch.cuda.Stream(device=device)
    comm_event = torch.cuda.Event(blocking=False, interprocess=False)

    cached = {
        "w1_sym": w1_sym,
        "w2_sym": w2_sym,
        "w3_sym": w3_sym,
        "y_sym": y_sym,
        "hdl_w1": hdl_w1,
        "hdl_y": hdl_y,
        "W1_full": W1_full,
        "W2_full": W2_full,
        "W3_full": W3_full,
        "x1": x1,
        "x2": x2,
        "z": z,
        "out": out,
        "ptr_w1": ptr_w1,
        "ptr_w2": ptr_w2,
        "ptr_w3": ptr_w3,
        "ptr_y": ptr_y,
        "comm_stream": comm_stream,
        "comm_event": comm_event,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    x_local: Tensor,
    W1_shard: Tensor,
    W2_shard: Tensor,
    W3_shard: Tensor,
    n_tp: int,
    n_fsdp: int,
) -> Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size == n_tp * n_fsdp

    assert x_local.is_cuda and W1_shard.is_cuda and W2_shard.is_cuda and W3_shard.is_cuda
    assert W1_shard.dtype == W2_shard.dtype == W3_shard.dtype == x_local.dtype
    dtype_enum = _dtype_enum(x_local.dtype)

    x = x_local.contiguous()
    w1_in = W1_shard.contiguous()
    w2_in = W2_shard.contiguous()
    w3_in = W3_shard.contiguous()

    Bf, D = x.shape
    D_shard, Htp = w1_in.shape
    D_shard2, Htp2 = w2_in.shape
    Htp3, D_shard3 = w3_in.shape

    assert D_shard == D_shard2
    assert Htp == Htp2 == Htp3
    assert D_shard * n_fsdp == D
    assert D_shard3 * n_fsdp == D

    tp_rank = rank % n_tp
    fsdp_rank = rank // n_tp

    ext = _get_ext()
    res = _get_resources(
        x.shape,
        w1_in.shape,
        w2_in.shape,
        w3_in.shape,
        x.dtype,
        x.device,
        n_tp,
        n_fsdp,
    )

    # Publish this rank's FSDP shards into symmetric memory.
    ext.copy3(
        res["w1_sym"], w1_in,
        res["w2_sym"], w2_in,
        res["w3_sym"], w3_in,
        w1_in.numel(),
        w2_in.numel(),
        w3_in.numel(),
        dtype_enum,
    )

    # Device-visible global sync for peer reads of symmetric shard buffers.
    res["hdl_w1"].barrier(channel=0)

    main_stream = torch.cuda.current_stream(device=x.device)
    comm_stream = res["comm_stream"]
    comm_event = res["comm_event"]

    # Overlap W3 column gather with W1/W2 row gathers + first two GEMMs.
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_stream(main_stream)
        ext.gather_dim1(
            res["ptr_w3"],
            res["W3_full"],
            n_tp,
            n_fsdp,
            tp_rank,
            Htp,
            D_shard3,
            dtype_enum,
        )
        comm_event.record(comm_stream)

    ext.gather_dim0_pair(
        res["ptr_w1"],
        res["ptr_w2"],
        res["W1_full"],
        res["W2_full"],
        n_tp,
        n_fsdp,
        tp_rank,
        D_shard,
        Htp,
        dtype_enum,
    )

    ext.gemm_rowmajor(x, res["W1_full"], res["x1"], Bf, Htp, D, dtype_enum)
    ext.gemm_rowmajor(x, res["W2_full"], res["x2"], Bf, Htp, D, dtype_enum)

    ext.silu_mul(res["x1"], res["x2"], res["z"], Bf * Htp, dtype_enum)

    main_stream.wait_event(comm_event)

    # TP-local partial output is written directly to symmetric memory.
    ext.gemm_rowmajor(res["z"], res["W3_full"], res["y_sym"], Bf, D, Htp, dtype_enum)

    # Sync TP partials, then reduce peers in the same FSDP row.
    res["hdl_y"].barrier(channel=1)

    ext.allreduce_tp(
        res["ptr_y"],
        res["out"],
        n_tp,
        fsdp_rank,
        Bf * D,
        dtype_enum,
    )

    return res["out"].reshape_as(x_local)


__all__ = ["solution"]