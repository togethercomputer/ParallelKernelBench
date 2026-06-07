# Strategy:
# - Avoid forming full [M, N] partials and avoid NCCL reduce-scatter entirely.
# - Publish A_local/B_local in symmetric memory, then each rank directly computes only its output row shard.
# - GEMM reads peer shards through UVA pointers; communication is pulled by device-side loads inside the compute kernel.
# - BF16 fast path uses WMMA tensor cores for aligned 16x16x16 tiles; scalar CUDA fallbacks preserve correctness for tails/dtypes.

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
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
#endif

// -----------------------------------------------------------------------------
// BF16 tensor-core direct reduce-scatter GEMM:
//
// out[m, n] = sum_p A_p[rank*M_local + m, :] @ B_p[:, n]
//
// This computes the reduce-scatter result directly, using UVA peer pointers.
// To better match the reference numerics, each peer GEMM contribution is rounded
// to BF16 before being accumulated into the cross-rank sum.
// -----------------------------------------------------------------------------

__global__ void direct_rs_bf16_wmma_kernel(
    const long long* __restrict__ A_ptrs,
    const long long* __restrict__ B_ptrs,
    __nv_bfloat16* __restrict__ out,
    int M,
    int K,
    int N,
    int M_local,
    int rank,
    int world_size
) {
    const int tile_n = blockIdx.x * 16;
    const int tile_m = blockIdx.y * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int global_m = rank * M_local + tile_m;

    #pragma unroll 1
    for (int p = 0; p < world_size; ++p) {
        const __nv_bfloat16* __restrict__ A =
            reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(A_ptrs[p]));
        const __nv_bfloat16* __restrict__ B =
            reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(B_ptrs[p]));

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> peer_acc;
        wmma::fill_fragment(peer_acc, 0.0f);

        #pragma unroll 1
        for (int kk = 0; kk < K; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;

            const __nv_bfloat16* a_tile = A + global_m * K + kk;
            const __nv_bfloat16* b_tile = B + kk * N + tile_n;

            wmma::load_matrix_sync(a_frag, a_tile, K);
            wmma::load_matrix_sync(b_frag, b_tile, N);
            wmma::mma_sync(peer_acc, a_frag, b_frag, peer_acc);
        }

        // Emulate per-rank BF16 partial materialization before the reduction.
        #pragma unroll
        for (int i = 0; i < peer_acc.num_elements; ++i) {
            acc.x[i] += __bfloat162float(__float2bfloat16(peer_acc.x[i]));
        }
    }

    __shared__ float smem[16 * 16];
    wmma::store_matrix_sync(smem, acc, 16, wmma::mem_row_major);
    __syncthreads();

    const int tid = threadIdx.x;
    for (int i = tid; i < 16 * 16; i += blockDim.x) {
        const int r = i / 16;
        const int c = i - r * 16;
        out[(tile_m + r) * N + tile_n + c] = __float2bfloat16(smem[i]);
    }
}


// -----------------------------------------------------------------------------
// Scalar correctness fallbacks for non-16-aligned BF16 and non-BF16 dtypes.
// -----------------------------------------------------------------------------

__global__ void direct_rs_bf16_scalar_kernel(
    const long long* __restrict__ A_ptrs,
    const long long* __restrict__ B_ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t K,
    int64_t N,
    int64_t M_local,
    int rank,
    int world_size,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t m = idx / N;
        const int64_t n = idx - m * N;
        const int64_t gm = (int64_t)rank * M_local + m;

        float acc = 0.0f;

        #pragma unroll 1
        for (int p = 0; p < world_size; ++p) {
            const __nv_bfloat16* __restrict__ A =
                reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(A_ptrs[p]));
            const __nv_bfloat16* __restrict__ B =
                reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(B_ptrs[p]));

            float peer = 0.0f;
            #pragma unroll 1
            for (int64_t k = 0; k < K; ++k) {
                peer += __bfloat162float(A[gm * K + k]) * __bfloat162float(B[k * N + n]);
            }
            acc += __bfloat162float(__float2bfloat16(peer));
        }

        out[idx] = __float2bfloat16(acc);
    }
}


__global__ void direct_rs_f32_scalar_kernel(
    const long long* __restrict__ A_ptrs,
    const long long* __restrict__ B_ptrs,
    float* __restrict__ out,
    int64_t K,
    int64_t N,
    int64_t M_local,
    int rank,
    int world_size,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t m = idx / N;
        const int64_t n = idx - m * N;
        const int64_t gm = (int64_t)rank * M_local + m;

        float acc = 0.0f;

        #pragma unroll 1
        for (int p = 0; p < world_size; ++p) {
            const float* __restrict__ A =
                reinterpret_cast<const float*>(static_cast<uintptr_t>(A_ptrs[p]));
            const float* __restrict__ B =
                reinterpret_cast<const float*>(static_cast<uintptr_t>(B_ptrs[p]));

            float peer = 0.0f;
            #pragma unroll 1
            for (int64_t k = 0; k < K; ++k) {
                peer += A[gm * K + k] * B[k * N + n];
            }
            acc += peer;
        }

        out[idx] = acc;
    }
}


__global__ void direct_rs_f16_scalar_kernel(
    const long long* __restrict__ A_ptrs,
    const long long* __restrict__ B_ptrs,
    __half* __restrict__ out,
    int64_t K,
    int64_t N,
    int64_t M_local,
    int rank,
    int world_size,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t m = idx / N;
        const int64_t n = idx - m * N;
        const int64_t gm = (int64_t)rank * M_local + m;

        float acc = 0.0f;

        #pragma unroll 1
        for (int p = 0; p < world_size; ++p) {
            const __half* __restrict__ A =
                reinterpret_cast<const __half*>(static_cast<uintptr_t>(A_ptrs[p]));
            const __half* __restrict__ B =
                reinterpret_cast<const __half*>(static_cast<uintptr_t>(B_ptrs[p]));

            float peer = 0.0f;
            #pragma unroll 1
            for (int64_t k = 0; k < K; ++k) {
                peer += __half2float(A[gm * K + k]) * __half2float(B[k * N + n]);
            }
            acc += __half2float(__float2half(peer));
        }

        out[idx] = __float2half(acc);
    }
}


void direct_rs_gemm(
    torch::Tensor A_ptrs,
    torch::Tensor B_ptrs,
    torch::Tensor out,
    int64_t M64,
    int64_t K64,
    int64_t N64,
    int rank,
    int world_size,
    int dtype_enum
) {
    TORCH_CHECK(A_ptrs.is_cuda() && B_ptrs.is_cuda(), "pointer tensors must be CUDA");
    TORCH_CHECK(A_ptrs.dtype() == torch::kInt64 && B_ptrs.dtype() == torch::kInt64,
                "pointer tensors must be int64");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(M64 % world_size == 0, "M must be divisible by world_size");

    const int64_t M_local64 = M64 / world_size;
    const int64_t total = M_local64 * N64;
    if (total == 0) {
        return;
    }

    const long long* A_p = reinterpret_cast<const long long*>(A_ptrs.data_ptr<int64_t>());
    const long long* B_p = reinterpret_cast<const long long*>(B_ptrs.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        // BF16 fast tensor-core path for fully aligned tiles.
        if ((M_local64 % 16 == 0) && (N64 % 16 == 0) && (K64 % 16 == 0) &&
            M64 <= INT_MAX && K64 <= INT_MAX && N64 <= INT_MAX && M_local64 <= INT_MAX) {
            dim3 grid((unsigned int)(N64 / 16), (unsigned int)(M_local64 / 16), 1);
            direct_rs_bf16_wmma_kernel<<<grid, 32, 0, stream>>>(
                A_p,
                B_p,
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                (int)M64,
                (int)K64,
                (int)N64,
                (int)M_local64,
                rank,
                world_size
            );
        } else {
            const int threads = 256;
            int blocks = (int)((total + threads - 1) / threads);
            if (blocks > 65535) blocks = 65535;
            direct_rs_bf16_scalar_kernel<<<blocks, threads, 0, stream>>>(
                A_p,
                B_p,
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                K64,
                N64,
                M_local64,
                rank,
                world_size,
                total
            );
        }
    } else if (dtype_enum == 1) {
        const int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        direct_rs_f32_scalar_kernel<<<blocks, threads, 0, stream>>>(
            A_p,
            B_p,
            out.data_ptr<float>(),
            K64,
            N64,
            M_local64,
            rank,
            world_size,
            total
        );
    } else if (dtype_enum == 2) {
        const int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        direct_rs_f16_scalar_kernel<<<blocks, threads, 0, stream>>>(
            A_p,
            B_p,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            K64,
            N64,
            M_local64,
            rank,
            world_size,
            total
        );
    } else {
        TORCH_CHECK(false, "unsupported dtype for direct_rs_gemm");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("direct_rs_gemm", &direct_rs_gemm,
          "Direct distributed GEMM reduce-scatter via symmetric-memory UVA pointers");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("direct_gemm_reducescatter_symm_bf16_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"unsupported dtype: {dtype}")


def _get_resources(A_shape, B_shape, dtype, device, world_size):
    key = (tuple(A_shape), tuple(B_shape), dtype, device, world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    M, K = A_shape
    Kb, N = B_shape
    assert K == Kb
    assert M % world_size == 0
    M_local = M // world_size

    A_buf = symm_mem.empty((M, K), device=device, dtype=dtype)
    B_buf = symm_mem.empty((K, N), device=device, dtype=dtype)

    A_hdl = symm_mem.rendezvous(A_buf, dist.group.WORLD)
    B_hdl = symm_mem.rendezvous(B_buf, dist.group.WORLD)

    out = torch.empty((M_local, N), device=device, dtype=dtype)

    A_ptrs_dev = torch.tensor(A_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    B_ptrs_dev = torch.tensor(B_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (A_buf, B_buf, A_hdl, B_hdl, out, A_ptrs_dev, B_ptrs_dev)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B_local: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B_local.is_cuda, "Inputs must be CUDA tensors"
    assert A_local.dtype == B_local.dtype, "A_local and B_local must have same dtype"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not A_local.is_contiguous():
        A_local = A_local.contiguous()
    if not B_local.is_contiguous():
        B_local = B_local.contiguous()

    M, K_local = A_local.shape
    K_B, N = B_local.shape

    assert K_local == K_B, (
        f"A_local and B_local must have matching K_local dimension: {K_local} != {K_B}"
    )
    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"

    dtype_enum = _dtype_enum(A_local.dtype)

    A_buf, B_buf, A_hdl, B_hdl, out, A_ptrs_dev, B_ptrs_dev = _get_resources(
        A_local.shape,
        B_local.shape,
        A_local.dtype,
        A_local.device,
        world_size,
    )

    # Publish this rank's K-shards into symmetric memory.  The following
    # symmetric barriers make peer UVA reads safe without using NCCL collectives.
    A_buf.copy_(A_local)
    B_buf.copy_(B_local)

    A_hdl.barrier(channel=0)
    B_hdl.barrier(channel=0)

    _get_ext().direct_rs_gemm(
        A_ptrs_dev,
        B_ptrs_dev,
        out,
        int(M),
        int(K_local),
        int(N),
        int(rank),
        int(world_size),
        int(dtype_enum),
    )

    # Protect buffer reuse across consecutive invocations: no rank overwrites its
    # symmetric inputs for the next call until all peer GEMMs have consumed them.
    A_hdl.barrier(channel=1)
    B_hdl.barrier(channel=1)

    return out