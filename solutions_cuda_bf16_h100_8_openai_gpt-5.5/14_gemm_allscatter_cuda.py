# Distributed BF16 GEMM + all-scatter for H100:
# - Compute local A @ B shard with a custom BF16 WMMA CUDA kernel.
# - Fuse all-scatter into the GEMM epilogue: each rank writes its computed column shard
#   directly into every rank's symmetric output buffer using UVA/NVLink P2P stores.
# - No NCCL all_gather/cat on the hot path; synchronization uses symmetric-memory rendezvous/barriers.

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

#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

// -----------------------------------------------------------------------------
// BF16 tensor-core GEMM fused with all-scatter.
//
// Each warp computes one 16x16 tile of the local output shard:
//   C_local[:, :] = A[M,K] @ B[K,N_local]
//
// In the epilogue, the tile is written to every rank's symmetric output buffer:
//   out_peer[m, rank*N_local + n] = C_local[m,n]
//
// Different ranks write disjoint column ranges, so no atomics are needed.
// -----------------------------------------------------------------------------

__global__ void bf16_wmma_gemm_scatter_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const long long* __restrict__ out_ptrs,
    int64_t M,
    int64_t K,
    int64_t N_local,
    int64_t N_total,
    int rank,
    int world_size,
    int64_t tiles_n,
    int64_t total_tiles
) {
#if __CUDA_ARCH__ >= 800
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const int64_t tile_id = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (tile_id >= total_tiles) {
        return;
    }

    const int64_t tile_m = tile_id / tiles_n;
    const int64_t tile_n = tile_id - tile_m * tiles_n;

    const int64_t row0 = tile_m * 16;
    const int64_t col0 = tile_n * 16;

    __shared__ __nv_bfloat16 As[WARPS_PER_BLOCK * 16 * 16];
    __shared__ __nv_bfloat16 Bs[WARPS_PER_BLOCK * 16 * 16];
    __shared__ float Cs[WARPS_PER_BLOCK * 16 * 16];

    __nv_bfloat16* As_w = As + warp_id * 256;
    __nv_bfloat16* Bs_w = Bs + warp_id * 256;
    float* Cs_w = Cs + warp_id * 256;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int64_t kk = 0; kk < K; kk += 16) {
        for (int i = lane; i < 256; i += WARP_SIZE) {
            const int r = i / 16;
            const int c = i - r * 16;

            const int64_t a_r = row0 + r;
            const int64_t a_c = kk + c;
            const int64_t b_r = kk + r;
            const int64_t b_c = col0 + c;

            As_w[i] = (a_r < M && a_c < K) ? A[a_r * K + a_c] : __float2bfloat16(0.0f);
            Bs_w[i] = (b_r < K && b_c < N_local) ? B[b_r * N_local + b_c] : __float2bfloat16(0.0f);
        }

        __syncwarp();

        wmma::load_matrix_sync(a_frag, As_w, 16);
        wmma::load_matrix_sync(b_frag, Bs_w, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncwarp();
    }

    wmma::store_matrix_sync(Cs_w, c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    for (int i = lane; i < 256; i += WARP_SIZE) {
        const int r = i / 16;
        const int c = i - r * 16;

        const int64_t m = row0 + r;
        const int64_t n_local = col0 + c;

        if (m < M && n_local < N_local) {
            const int64_t n_global = (int64_t)rank * N_local + n_local;
            const __nv_bfloat16 v = __float2bfloat16(Cs_w[i]);

            #pragma unroll
            for (int peer = 0; peer < 8; ++peer) {
                if (peer < world_size) {
                    __nv_bfloat16* dst =
                        reinterpret_cast<__nv_bfloat16*>((uintptr_t)out_ptrs[peer]);
                    dst[m * N_total + n_global] = v;
                }
            }
        }
    }

    __threadfence_system();
#endif
}

// -----------------------------------------------------------------------------
// Generic CUDA-core fallbacks for fp32/fp16. These are intentionally simple;
// benchmark target is BF16, which uses the WMMA path above.
// -----------------------------------------------------------------------------

__global__ void f32_gemm_scatter_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const long long* __restrict__ out_ptrs,
    int64_t M,
    int64_t K,
    int64_t N_local,
    int64_t N_total,
    int rank,
    int world_size,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        const int64_t m = idx / N_local;
        const int64_t n_local = idx - m * N_local;

        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            acc += A[m * K + k] * B[k * N_local + n_local];
        }

        const int64_t n_global = (int64_t)rank * N_local + n_local;

        #pragma unroll
        for (int peer = 0; peer < 8; ++peer) {
            if (peer < world_size) {
                float* dst = reinterpret_cast<float*>((uintptr_t)out_ptrs[peer]);
                dst[m * N_total + n_global] = acc;
            }
        }
    }

    __threadfence_system();
}

__global__ void f16_gemm_scatter_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const long long* __restrict__ out_ptrs,
    int64_t M,
    int64_t K,
    int64_t N_local,
    int64_t N_total,
    int rank,
    int world_size,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        const int64_t m = idx / N_local;
        const int64_t n_local = idx - m * N_local;

        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            acc += __half2float(A[m * K + k]) * __half2float(B[k * N_local + n_local]);
        }

        const int64_t n_global = (int64_t)rank * N_local + n_local;
        const __half v = __float2half(acc);

        #pragma unroll
        for (int peer = 0; peer < 8; ++peer) {
            if (peer < world_size) {
                __half* dst = reinterpret_cast<__half*>((uintptr_t)out_ptrs[peer]);
                dst[m * N_total + n_global] = v;
            }
        }
    }

    __threadfence_system();
}

void launch_gemm_scatter(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor out_ptrs,
    int64_t M,
    int64_t K,
    int64_t N_local,
    int64_t N_total,
    int rank,
    int world_size,
    int dtype_enum
) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(out_ptrs.is_cuda(), "out_ptrs must be CUDA");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(out_ptrs.dtype() == torch::kInt64, "out_ptrs must be int64");
    TORCH_CHECK(world_size <= 8, "This H100 SXM path assumes world_size <= 8");

    const long long* ptrs = reinterpret_cast<const long long*>(out_ptrs.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        TORCH_CHECK(A.dtype() == torch::kBFloat16, "BF16 dtype mismatch");
        TORCH_CHECK(B.dtype() == torch::kBFloat16, "BF16 dtype mismatch");

        const int64_t tiles_m = (M + 15) / 16;
        const int64_t tiles_n = (N_local + 15) / 16;
        const int64_t total_tiles = tiles_m * tiles_n;

        const dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
        const dim3 grid((unsigned int)((total_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));

        bf16_wmma_gemm_scatter_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
            ptrs,
            M,
            K,
            N_local,
            N_total,
            rank,
            world_size,
            tiles_n,
            total_tiles
        );
    } else if (dtype_enum == 1) {
        TORCH_CHECK(A.dtype() == torch::kFloat32, "FP32 dtype mismatch");
        TORCH_CHECK(B.dtype() == torch::kFloat32, "FP32 dtype mismatch");

        const int threads = 256;
        int blocks = (int)((M * N_local + threads - 1) / threads);
        if (blocks > 65535) {
            blocks = 65535;
        }

        f32_gemm_scatter_kernel<<<blocks, threads, 0, stream>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            ptrs,
            M,
            K,
            N_local,
            N_total,
            rank,
            world_size,
            M * N_local
        );
    } else if (dtype_enum == 2) {
        TORCH_CHECK(A.dtype() == torch::kFloat16, "FP16 dtype mismatch");
        TORCH_CHECK(B.dtype() == torch::kFloat16, "FP16 dtype mismatch");

        const int threads = 256;
        int blocks = (int)((M * N_local + threads - 1) / threads);
        if (blocks > 65535) {
            blocks = 65535;
        }

        f16_gemm_scatter_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
            ptrs,
            M,
            K,
            N_local,
            N_total,
            rank,
            world_size,
            M * N_local
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for custom GEMM scatter");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gemm_scatter", &launch_gemm_scatter,
          "BF16 WMMA GEMM fused with symmetric-memory all-scatter");
}
'''


_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("bf16_gemm_allscatter_symm_uva_ext", CUDA_SRC)
    return _ext


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"Unsupported dtype for custom GEMM all-scatter: {dtype}")


def _device_key(device: torch.device):
    if device.index is None:
        return torch.cuda.current_device()
    return device.index


def _get_resources(M: int, N_total: int, dtype: torch.dtype, device: torch.device):
    """
    Two symmetric output buffers are kept per shape to avoid immediately reusing
    the buffer returned by the previous call. Every rank creates/rendezvous in
    the same order.
    """
    key = (M, N_total, dtype, _device_key(device))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    outs = []
    hdls = []
    ptr_tensors = []

    for _ in range(2):
        out = symm_mem.empty((M, N_total), device=device, dtype=dtype)
        hdl = symm_mem.rendezvous(out, dist.group.WORLD)
        ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

        outs.append(out)
        hdls.append(hdl)
        ptr_tensors.append(ptrs)

    cached = {
        "outs": outs,
        "hdls": hdls,
        "ptrs": ptr_tensors,
        "next": 0,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dim() == 2 and B.dim() == 2, "A and B must be matrices"
    assert A.dtype == B.dtype, "A and B must have same dtype"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size <= 8, "This implementation targets <=8 H100 SXM GPUs"

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    M, K = A.shape
    K_B, N_local = B.shape
    assert K == K_B, f"A and B must have matching K dimension: {K} != {K_B}"

    N_total = N_local * world_size
    dtype_id = _dtype_enum(A.dtype)

    _get_ext()

    state = _get_resources(M, N_total, A.dtype, A.device)
    buf_idx = state["next"]
    state["next"] = 1 - buf_idx

    out = state["outs"][buf_idx]
    hdl = state["hdls"][buf_idx]
    ptrs = state["ptrs"][buf_idx]

    _ext.launch_gemm_scatter(
        A,
        B,
        ptrs,
        int(M),
        int(K),
        int(N_local),
        int(N_total),
        int(rank),
        int(world_size),
        int(dtype_id),
    )

    # Make local P2P stores complete before this rank participates in the
    # symmetric-memory barrier; after the barrier every column shard has landed
    # in every rank's output buffer.
    torch.cuda.current_stream().synchronize()
    hdl.barrier(channel=0)

    return out