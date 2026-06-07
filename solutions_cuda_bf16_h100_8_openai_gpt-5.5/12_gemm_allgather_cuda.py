# solutions_cuda_bf16_h100_8_openai_gpt-5.5/12_gemm_allgather_cuda.py

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

using namespace nvcuda;

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

static constexpr int TILE_M = 16;
static constexpr int TILE_N = 16;
static constexpr int TILE_K = 16;
static constexpr int WARPS_PER_BLOCK = 8;

// -----------------------------------------------------------------------------
// D2D copy into symmetric memory on current stream.
// -----------------------------------------------------------------------------

void copy_to_symm(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA tensors");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");
    TORCH_CHECK(src.nbytes() == dst.nbytes(), "src/dst byte sizes must match");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        src.nbytes(),
        cudaMemcpyDeviceToDevice,
        stream));
}

// -----------------------------------------------------------------------------
// BF16 tensor-core GEMM:
//   C[M,N] = sum_r A_r[M,Klocal] @ B[r*Klocal:(r+1)*Klocal, N]
// A_r pointers are UVA peer pointers from symmetric memory.
// B is local replicated row-major.
// C is row-major BF16.
// -----------------------------------------------------------------------------

__global__ void allshard_gemm_bf16_wmma_kernel(
    const long long* __restrict__ a_ptrs,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int64_t M,
    int64_t Klocal,
    int64_t N,
    int world_size
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    const int64_t tiles_n = (N + TILE_N - 1) / TILE_N;
    const int64_t tiles_m = (M + TILE_M - 1) / TILE_M;
    const int64_t tile_linear = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (warp_id >= WARPS_PER_BLOCK || tile_linear >= tiles_m * tiles_n) {
        return;
    }

    const int64_t tile_m = tile_linear / tiles_n;
    const int64_t tile_n = tile_linear - tile_m * tiles_n;

    __shared__ __nv_bfloat16 shA[WARPS_PER_BLOCK][TILE_M * TILE_K];
    __shared__ __nv_bfloat16 shB[WARPS_PER_BLOCK][TILE_K * TILE_N];
    __shared__ float shC[WARPS_PER_BLOCK][TILE_M * TILE_N];

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int r = 0; r < world_size; ++r) {
        const __nv_bfloat16* __restrict__ A =
            reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(a_ptrs[r]));

        for (int64_t kk = 0; kk < Klocal; kk += TILE_K) {
            for (int idx = lane; idx < TILE_M * TILE_K; idx += WARP_SIZE) {
                const int i = idx / TILE_K;
                const int j = idx - i * TILE_K;

                const int64_t row = tile_m * TILE_M + i;
                const int64_t colk = kk + j;

                __nv_bfloat16 v = __float2bfloat16(0.0f);
                if (row < M && colk < Klocal) {
                    v = A[row * Klocal + colk];
                }
                shA[warp_id][idx] = v;
            }

            for (int idx = lane; idx < TILE_K * TILE_N; idx += WARP_SIZE) {
                const int i = idx / TILE_N;
                const int j = idx - i * TILE_N;

                const int64_t brow = kk + i;
                const int64_t bcol = tile_n * TILE_N + j;

                __nv_bfloat16 v = __float2bfloat16(0.0f);
                if (brow < Klocal && bcol < N) {
                    v = B[((int64_t)r * Klocal + brow) * N + bcol];
                }
                shB[warp_id][idx] = v;
            }

            __syncwarp();

            wmma::load_matrix_sync(a_frag, shA[warp_id], TILE_K);
            wmma::load_matrix_sync(b_frag, shB[warp_id], TILE_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

            __syncwarp();
        }
    }

    wmma::store_matrix_sync(shC[warp_id], acc_frag, TILE_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < TILE_M * TILE_N; idx += WARP_SIZE) {
        const int i = idx / TILE_N;
        const int j = idx - i * TILE_N;

        const int64_t row = tile_m * TILE_M + i;
        const int64_t col = tile_n * TILE_N + j;

        if (row < M && col < N) {
            C[row * N + col] = __float2bfloat16(shC[warp_id][idx]);
        }
    }
}

// -----------------------------------------------------------------------------
// FP16 tensor-core path, same fused remote-read algorithm.
// -----------------------------------------------------------------------------

__global__ void allshard_gemm_f16_wmma_kernel(
    const long long* __restrict__ a_ptrs,
    const half* __restrict__ B,
    half* __restrict__ C,
    int64_t M,
    int64_t Klocal,
    int64_t N,
    int world_size
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    const int64_t tiles_n = (N + TILE_N - 1) / TILE_N;
    const int64_t tiles_m = (M + TILE_M - 1) / TILE_M;
    const int64_t tile_linear = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (warp_id >= WARPS_PER_BLOCK || tile_linear >= tiles_m * tiles_n) {
        return;
    }

    const int64_t tile_m = tile_linear / tiles_n;
    const int64_t tile_n = tile_linear - tile_m * tiles_n;

    __shared__ half shA[WARPS_PER_BLOCK][TILE_M * TILE_K];
    __shared__ half shB[WARPS_PER_BLOCK][TILE_K * TILE_N];
    __shared__ float shC[WARPS_PER_BLOCK][TILE_M * TILE_N];

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int r = 0; r < world_size; ++r) {
        const half* __restrict__ A =
            reinterpret_cast<const half*>(static_cast<uintptr_t>(a_ptrs[r]));

        for (int64_t kk = 0; kk < Klocal; kk += TILE_K) {
            for (int idx = lane; idx < TILE_M * TILE_K; idx += WARP_SIZE) {
                const int i = idx / TILE_K;
                const int j = idx - i * TILE_K;

                const int64_t row = tile_m * TILE_M + i;
                const int64_t colk = kk + j;

                half v = __float2half(0.0f);
                if (row < M && colk < Klocal) {
                    v = A[row * Klocal + colk];
                }
                shA[warp_id][idx] = v;
            }

            for (int idx = lane; idx < TILE_K * TILE_N; idx += WARP_SIZE) {
                const int i = idx / TILE_N;
                const int j = idx - i * TILE_N;

                const int64_t brow = kk + i;
                const int64_t bcol = tile_n * TILE_N + j;

                half v = __float2half(0.0f);
                if (brow < Klocal && bcol < N) {
                    v = B[((int64_t)r * Klocal + brow) * N + bcol];
                }
                shB[warp_id][idx] = v;
            }

            __syncwarp();

            wmma::load_matrix_sync(a_frag, shA[warp_id], TILE_K);
            wmma::load_matrix_sync(b_frag, shB[warp_id], TILE_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

            __syncwarp();
        }
    }

    wmma::store_matrix_sync(shC[warp_id], acc_frag, TILE_N, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < TILE_M * TILE_N; idx += WARP_SIZE) {
        const int i = idx / TILE_N;
        const int j = idx - i * TILE_N;

        const int64_t row = tile_m * TILE_M + i;
        const int64_t col = tile_n * TILE_N + j;

        if (row < M && col < N) {
            C[row * N + col] = __float2half(shC[warp_id][idx]);
        }
    }
}

// -----------------------------------------------------------------------------
// FP32 correctness fallback: direct remote-read GEMM on CUDA cores.
// -----------------------------------------------------------------------------

__global__ void allshard_gemm_f32_kernel(
    const long long* __restrict__ a_ptrs,
    const float* __restrict__ B,
    float* __restrict__ C,
    int64_t M,
    int64_t Klocal,
    int64_t N,
    int world_size
) {
    const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t row = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;

    for (int r = 0; r < world_size; ++r) {
        const float* __restrict__ A =
            reinterpret_cast<const float*>(static_cast<uintptr_t>(a_ptrs[r]));

        for (int64_t k = 0; k < Klocal; ++k) {
            acc += A[row * Klocal + k] * B[((int64_t)r * Klocal + k) * N + col];
        }
    }

    C[row * N + col] = acc;
}

void launch_allshard_gemm(
    torch::Tensor ptrs_tensor,
    torch::Tensor B,
    torch::Tensor C,
    int64_t M,
    int64_t Klocal,
    int64_t N,
    int world_size
) {
    TORCH_CHECK(ptrs_tensor.is_cuda(), "ptrs_tensor must be CUDA");
    TORCH_CHECK(ptrs_tensor.scalar_type() == torch::kInt64, "ptrs_tensor must be int64");
    TORCH_CHECK(ptrs_tensor.is_contiguous(), "ptrs_tensor must be contiguous");

    TORCH_CHECK(B.is_cuda() && C.is_cuda(), "B/C must be CUDA");
    TORCH_CHECK(B.is_contiguous() && C.is_contiguous(), "B/C must be contiguous");
    TORCH_CHECK(B.scalar_type() == C.scalar_type(), "B/C dtype mismatch");

    if (M == 0 || N == 0) {
        return;
    }

    const long long* ptrs = reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (B.scalar_type() == torch::kBFloat16) {
        const int64_t tiles_m = (M + TILE_M - 1) / TILE_M;
        const int64_t tiles_n = (N + TILE_N - 1) / TILE_N;
        const int64_t total_tiles = tiles_m * tiles_n;
        const int blocks = (int)((total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        allshard_gemm_bf16_wmma_kernel<<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(
            ptrs,
            reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
            M,
            Klocal,
            N,
            world_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    if (B.scalar_type() == torch::kFloat16) {
        const int64_t tiles_m = (M + TILE_M - 1) / TILE_M;
        const int64_t tiles_n = (N + TILE_N - 1) / TILE_N;
        const int64_t total_tiles = tiles_m * tiles_n;
        const int blocks = (int)((total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

        allshard_gemm_f16_wmma_kernel<<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, stream>>>(
            ptrs,
            reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M,
            Klocal,
            N,
            world_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    if (B.scalar_type() == torch::kFloat32) {
        dim3 block(16, 16);
        dim3 grid((unsigned int)((N + block.x - 1) / block.x),
                  (unsigned int)((M + block.y - 1) / block.y));

        allshard_gemm_f32_kernel<<<grid, block, 0, stream>>>(
            ptrs,
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M,
            Klocal,
            N,
            world_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    TORCH_CHECK(false, "custom allshard GEMM supports bfloat16, float16, and float32");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_to_symm", &copy_to_symm, "Async D2D copy into symmetric memory");
    m.def("launch_allshard_gemm", &launch_allshard_gemm,
          "Fused all-gather-as-UVA-loads + GEMM");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("allshard_gemm_symm_wmma_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _get_resources(A_shape, B_shape, dtype, device):
    key = (tuple(A_shape), tuple(B_shape), dtype, device, dist.get_world_size())
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    M, Klocal = A_shape
    _, N = B_shape

    a_symm = symm_mem.empty((M, Klocal), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(a_symm, dist.group.WORLD)

    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    out = torch.empty((M, N), device=device, dtype=dtype)

    cached = {
        "a_symm": a_symm,
        "hdl": hdl,
        "ptrs_tensor": ptrs_tensor,
        "out": out,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A_local.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"
    assert A_local.dtype == B.dtype, "A_local and B dtype must match"

    world_size = dist.get_world_size()

    M, Klocal = A_local.shape
    Kb, N = B.shape
    assert Kb == world_size * Klocal, (
        f"B must have K dimension = world_size * K_local: {Kb} != {world_size} * {Klocal}"
    )

    ext = _get_ext()
    res = _get_resources(A_local.shape, B.shape, A_local.dtype, A_local.device)

    a_symm = res["a_symm"]
    hdl = res["hdl"]
    ptrs_tensor = res["ptrs_tensor"]
    out = res["out"]

    # Publish this rank's A shard into symmetric memory, then use a symmetric-memory
    # barrier so peer UVA reads in the GEMM see the completed write.
    ext.copy_to_symm(A_local, a_symm)
    hdl.barrier(channel=0)

    # Fused distributed GEMM: no materialized A_global and no NCCL all_gather.
    ext.launch_allshard_gemm(
        ptrs_tensor,
        B,
        out,
        int(M),
        int(Klocal),
        int(N),
        int(world_size),
    )

    return out