# solutions_cuda_bf16_h100_8_openai_gpt-5.5/15_combined_sharded_gemms_cuda.py

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>

#define CUBLAS_CHECK(cmd) do {                                      \
    cublasStatus_t _status = (cmd);                                 \
    TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                   \
                "cuBLAS error: ", static_cast<int>(_status));       \
} while (0)

__global__ void copy_bytes_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int64_t nbytes
) {
    int64_t nvec = nbytes >> 4;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src);
    uint4* __restrict__ d4 = reinterpret_cast<uint4*>(dst);

    for (int64_t i = tid; i < nvec; i += stride) {
        d4[i] = s4[i];
    }

    int rem = (int)(nbytes & 15);
    if (rem && tid < rem) {
        dst[(nvec << 4) + tid] = src[(nvec << 4) + tid];
    }
}

__global__ void silu_round_bf16_kernel(
    const float* __restrict__ z,
    __nv_bfloat16* __restrict__ a,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        // Reference path materializes BF16 z before F.silu(z).
        float x = __bfloat162float(__float2bfloat16(z[i]));
        float y = x / (1.0f + expf(-x));
        a[i] = __float2bfloat16(y);
    }
}

__global__ void silu_f32_kernel(
    const float* __restrict__ z,
    float* __restrict__ a,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        float x = z[i];
        a[i] = x / (1.0f + expf(-x));
    }
}

static inline void launch_silu_round_bf16(torch::Tensor z, torch::Tensor a) {
    int64_t n = z.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    silu_round_bf16_kernel<<<blocks, threads, 0, stream>>>(
        z.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static inline void launch_silu_f32(torch::Tensor z, torch::Tensor a) {
    int64_t n = z.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    silu_f32_kernel<<<blocks, threads, 0, stream>>>(
        z.data_ptr<float>(),
        a.data_ptr<float>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Row-major C[M,N] = A[M,K] @ B[K,N], BF16 inputs, FP32 output.
// Implemented as column-major C^T[N,M] = B^T[N,K] @ A^T[K,M].
static inline void gemm_rowmajor_bf16_to_f32(
    cublasHandle_t handle,
    const at::BFloat16* A,
    const at::BFloat16* B,
    float* C,
    int64_t M,
    int64_t N,
    int64_t K,
    float beta
) {
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)N,
        (int)M,
        (int)K,
        &alpha,
        reinterpret_cast<const void*>(B),
        CUDA_R_16BF,
        (int)N,
        reinterpret_cast<const void*>(A),
        CUDA_R_16BF,
        (int)K,
        &beta,
        reinterpret_cast<void*>(C),
        CUDA_R_32F,
        (int)N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// Row-major C[M,N] = A[M,K] @ B[K,N], BF16 inputs, BF16 output.
static inline void gemm_rowmajor_bf16_to_bf16(
    cublasHandle_t handle,
    const at::BFloat16* A,
    const at::BFloat16* B,
    at::BFloat16* C,
    int64_t M,
    int64_t N,
    int64_t K
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)N,
        (int)M,
        (int)K,
        &alpha,
        reinterpret_cast<const void*>(B),
        CUDA_R_16BF,
        (int)N,
        reinterpret_cast<const void*>(A),
        CUDA_R_16BF,
        (int)K,
        &beta,
        reinterpret_cast<void*>(C),
        CUDA_R_16BF,
        (int)N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

static inline void gemm_rowmajor_f32(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t N,
    int64_t K,
    float beta
) {
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)N,
        (int)M,
        (int)K,
        &alpha,
        reinterpret_cast<const void*>(B),
        CUDA_R_32F,
        (int)N,
        reinterpret_cast<const void*>(A),
        CUDA_R_32F,
        (int)K,
        &beta,
        reinterpret_cast<void*>(C),
        CUDA_R_32F,
        (int)N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

void publish_copy(torch::Tensor src, torch::Tensor dst, int64_t nbytes) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA tensors");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");
    TORCH_CHECK(nbytes <= src.nbytes() && nbytes <= dst.nbytes(), "invalid byte count");

    int threads = 256;
    int64_t nvec = (nbytes + 15) >> 4;
    int blocks = (int)((nvec + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    copy_bytes_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(src.data_ptr()),
        reinterpret_cast<char*>(dst.data_ptr()),
        nbytes
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_mlp_bf16(
    std::vector<int64_t> x_ptrs,
    torch::Tensor W1,
    torch::Tensor W2,
    torch::Tensor z_f32,
    torch::Tensor a_bf16,
    torch::Tensor y_bf16,
    int64_t M_total,
    int64_t H_local,
    int64_t F_dim,
    int64_t rank
) {
    TORCH_CHECK(W1.is_cuda() && W2.is_cuda() && z_f32.is_cuda() && a_bf16.is_cuda() && y_bf16.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(W1.is_contiguous() && W2.is_contiguous() && z_f32.is_contiguous() &&
                a_bf16.is_contiguous() && y_bf16.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(W1.dtype() == torch::kBFloat16 && W2.dtype() == torch::kBFloat16 &&
                a_bf16.dtype() == torch::kBFloat16 && y_bf16.dtype() == torch::kBFloat16,
                "BF16 path requires BF16 W1/W2/a/y");
    TORCH_CHECK(z_f32.dtype() == torch::kFloat32, "z must be float32");

    int64_t world_size = (int64_t)x_ptrs.size();
    int64_t M_local = y_bf16.size(0);
    int64_t H = H_local * world_size;

    TORCH_CHECK(W1.size(0) == H && W1.size(1) == F_dim, "bad W1 shape");
    TORCH_CHECK(W2.size(0) == F_dim && W2.size(1) == H, "bad W2 shape");
    TORCH_CHECK(z_f32.size(0) == M_local && z_f32.size(1) == F_dim, "bad z shape");
    TORCH_CHECK(a_bf16.size(0) == M_local && a_bf16.size(1) == F_dim, "bad a shape");
    TORCH_CHECK(M_total % world_size == 0, "M must be divisible by world_size");

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    int64_t row0 = rank * M_local;

    float* z = z_f32.data_ptr<float>();
    const at::BFloat16* W1p = W1.data_ptr<at::BFloat16>();

    // z = sum_r x_r[local_rows] @ W1_r
    // x_r is read directly from peer symmetric memory through UVA/NVLink.
    for (int64_t r = 0; r < world_size; ++r) {
        const at::BFloat16* x_remote =
            reinterpret_cast<const at::BFloat16*>(static_cast<uintptr_t>(x_ptrs[(size_t)r]))
            + row0 * H_local;
        const at::BFloat16* W1_shard = W1p + r * H_local * F_dim;
        float beta = (r == 0) ? 0.0f : 1.0f;

        gemm_rowmajor_bf16_to_f32(
            handle,
            x_remote,
            W1_shard,
            z,
            M_local,
            F_dim,
            H_local,
            beta
        );
    }

    launch_silu_round_bf16(z_f32, a_bf16);

    gemm_rowmajor_bf16_to_bf16(
        handle,
        a_bf16.data_ptr<at::BFloat16>(),
        W2.data_ptr<at::BFloat16>(),
        y_bf16.data_ptr<at::BFloat16>(),
        M_local,
        H,
        F_dim
    );
}

void compute_mlp_f32(
    std::vector<int64_t> x_ptrs,
    torch::Tensor W1,
    torch::Tensor W2,
    torch::Tensor z_f32,
    torch::Tensor a_f32,
    torch::Tensor y_f32,
    int64_t M_total,
    int64_t H_local,
    int64_t F_dim,
    int64_t rank
) {
    TORCH_CHECK(W1.is_cuda() && W2.is_cuda() && z_f32.is_cuda() && a_f32.is_cuda() && y_f32.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(W1.is_contiguous() && W2.is_contiguous() && z_f32.is_contiguous() &&
                a_f32.is_contiguous() && y_f32.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(W1.dtype() == torch::kFloat32 && W2.dtype() == torch::kFloat32 &&
                z_f32.dtype() == torch::kFloat32 && a_f32.dtype() == torch::kFloat32 &&
                y_f32.dtype() == torch::kFloat32,
                "F32 path requires float32 tensors");

    int64_t world_size = (int64_t)x_ptrs.size();
    int64_t M_local = y_f32.size(0);
    int64_t H = H_local * world_size;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    int64_t row0 = rank * M_local;
    float* z = z_f32.data_ptr<float>();
    const float* W1p = W1.data_ptr<float>();

    for (int64_t r = 0; r < world_size; ++r) {
        const float* x_remote =
            reinterpret_cast<const float*>(static_cast<uintptr_t>(x_ptrs[(size_t)r]))
            + row0 * H_local;
        const float* W1_shard = W1p + r * H_local * F_dim;
        float beta = (r == 0) ? 0.0f : 1.0f;

        gemm_rowmajor_f32(
            handle,
            x_remote,
            W1_shard,
            z,
            M_local,
            F_dim,
            H_local,
            beta
        );
    }

    launch_silu_f32(z_f32, a_f32);

    gemm_rowmajor_f32(
        handle,
        a_f32.data_ptr<float>(),
        W2.data_ptr<float>(),
        y_f32.data_ptr<float>(),
        M_local,
        H,
        F_dim,
        0.0f
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("publish_copy", &publish_copy, "Async device publish copy into symmetric memory");
    m.def("compute_mlp_bf16", &compute_mlp_bf16,
          "Sequence-parallel MLP using peer UVA BF16 shards and tensor-core GEMMs");
    m.def("compute_mlp_f32", &compute_mlp_f32,
          "Sequence-parallel MLP using peer UVA F32 shards");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("combined_sharded_mlp_symm_uva_bf16_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _cache_key(x_local: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, world_size: int):
    M, H_local = x_local.shape
    H, F_dim = W1.shape
    return (
        int(M),
        int(H_local),
        int(H),
        int(F_dim),
        x_local.dtype,
        x_local.device.index,
        int(world_size),
    )


def _get_resources(x_local: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, rank: int, world_size: int):
    key = _cache_key(x_local, W1, W2, world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    M, H_local = x_local.shape
    H, F_dim = W1.shape
    M_local = M // world_size
    device = x_local.device
    dtype = x_local.dtype

    x_sym = symm_mem.empty((M, H_local), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(x_sym, dist.group.WORLD)

    z_f32 = torch.empty((M_local, F_dim), device=device, dtype=torch.float32)

    if dtype == torch.bfloat16:
        a = torch.empty((M_local, F_dim), device=device, dtype=torch.bfloat16)
        y = torch.empty((M_local, H), device=device, dtype=torch.bfloat16)
    elif dtype == torch.float32:
        a = torch.empty((M_local, F_dim), device=device, dtype=torch.float32)
        y = torch.empty((M_local, H), device=device, dtype=torch.float32)
    else:
        raise TypeError(f"unsupported dtype {dtype}; optimized path supports bfloat16 and float32")

    ptrs = [int(p) for p in hdl.buffer_ptrs]

    res = {
        "x_sym": x_sym,
        "hdl": hdl,
        "z": z_f32,
        "a": a,
        "y": y,
        "ptrs": ptrs,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    x_local: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert x_local.is_cuda and W1.is_cuda and W2.is_cuda, "Inputs must be CUDA tensors"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    M, H_local = x_local.shape
    H, ffn_dim = W1.shape
    ffn2, H_out = W2.shape

    assert ffn_dim == ffn2, f"W1 and W2 inner dims must match: {ffn_dim} vs {ffn2}"
    assert H_out == H, f"W2 out dim must match gathered hidden H: {H_out} vs {H}"
    assert H == H_local * world_size, (
        f"Hidden must split across ranks: H={H}, H_local={H_local}, world_size={world_size}"
    )
    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
    assert x_local.dtype == W1.dtype == W2.dtype, "x_local/W1/W2 dtype mismatch"

    if x_local.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {x_local.dtype}; expected BF16 or F32")

    ext = _get_ext()

    x_in = x_local if x_local.is_contiguous() else x_local.contiguous()
    W1c = W1 if W1.is_contiguous() else W1.contiguous()
    W2c = W2 if W2.is_contiguous() else W2.contiguous()

    res = _get_resources(x_in, W1c, W2c, rank, world_size)
    x_sym = res["x_sym"]
    hdl = res["hdl"]
    z = res["z"]
    a = res["a"]
    y = res["y"]
    ptrs = res["ptrs"]

    # Publish this rank's hidden shard into symmetric memory. Peers consume it
    # directly through UVA inside the GEMM loop; no NCCL all-gather is used.
    ext.publish_copy(x_in, x_sym, x_in.numel() * x_in.element_size())
    hdl.barrier(channel=0)

    if x_in.dtype == torch.bfloat16:
        ext.compute_mlp_bf16(
            ptrs,
            W1c,
            W2c,
            z,
            a,
            y,
            int(M),
            int(H_local),
            int(ffn_dim),
            int(rank),
        )
    else:
        ext.compute_mlp_f32(
            ptrs,
            W1c,
            W2c,
            z,
            a,
            y,
            int(M),
            int(H_local),
            int(ffn_dim),
            int(rank),
        )

    # Protect symmetric x buffer reuse by fast symmetric-memory synchronization.
    hdl.barrier(channel=1)
    return y