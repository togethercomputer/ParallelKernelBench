# Distributed GEMM + all-reduce via custom CUDA, symmetric memory, UVA peer loads.
# Target: BF16 on H100 SXM. No NCCL collectives on the hot path.

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

#include <stdint.h>

using namespace nvcuda;

static constexpr int TILE_M = 16;
static constexpr int TILE_N = 16;
static constexpr int TILE_K = 16;
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

__global__ void init_i32_kernel(int* p, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        p[i] = 0;
    }
}

template <typename T>
__device__ __forceinline__ float to_float_dev(T x);

template <>
__device__ __forceinline__ float to_float_dev<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float to_float_dev<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float_dev<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float_dev(float x);

template <>
__device__ __forceinline__ float from_float_dev<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ __half from_float_dev<__half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float_dev<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

// Device-side rank barrier scoped to one logical tile/phase.
// flags layout per rank: [num_tiles * 2, world_size], int32
// Every rank writes its phase value into every peer's slot and waits until every
// peer wrote into this rank's local slots.
__device__ __forceinline__ void warp_rank_barrier(
    const long long* __restrict__ flag_ptrs,
    int barrier_id,
    int world_size,
    int rank,
    int value,
    int lane
) {
    if (lane < world_size) {
        unsigned int* remote_base =
            reinterpret_cast<unsigned int*>(static_cast<uintptr_t>(flag_ptrs[lane]));
        unsigned int* local_base =
            reinterpret_cast<unsigned int*>(static_cast<uintptr_t>(flag_ptrs[rank]));

        unsigned int* send_addr =
            remote_base + (int64_t)barrier_id * world_size + rank;
        unsigned int* wait_addr =
            local_base + (int64_t)barrier_id * world_size + lane;

        __threadfence_system();
        atomicExch_system(send_addr, (unsigned int)value);

        unsigned int seen = 0;
        do {
            seen = atomicAdd_system(wait_addr, 0u);
            if (seen != (unsigned int)value) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
        } while (seen != (unsigned int)value);
    }
    __syncwarp();
}

// -----------------------------------------------------------------------------
// BF16 tensor-core persistent tiled GEMM + per-tile UVA all-reduce.
// Fast path requires M,N,K all multiples of 16.
// One warp owns one 16x16 C tile.
// -----------------------------------------------------------------------------

__global__ void fused_gemm_allreduce_bf16_wmma_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C_local_symm,
    const long long* __restrict__ c_ptrs,
    __nv_bfloat16* __restrict__ Out,
    const long long* __restrict__ flag_ptrs,
    int M,
    int K,
    int N,
    int tiles_m,
    int tiles_n,
    int num_tiles,
    int world_size,
    int rank,
    int epoch_value
) {
    __shared__ float smem[WARPS_PER_BLOCK][TILE_M * TILE_N];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total_warps = gridDim.x * WARPS_PER_BLOCK;

    for (int tile_id = global_warp; tile_id < num_tiles; tile_id += total_warps) {
        const int tm = tile_id / tiles_n;
        const int tn = tile_id - tm * tiles_n;

        const int row0 = tm * TILE_M;
        const int col0 = tn * TILE_N;

        wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K,
                       __nv_bfloat16, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            const __nv_bfloat16* a_tile = A + (int64_t)row0 * K + k0;
            const __nv_bfloat16* b_tile = B + (int64_t)k0 * N + col0;
            wmma::load_matrix_sync(a_frag, a_tile, K);
            wmma::load_matrix_sync(b_frag, b_tile, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* warp_smem = &smem[warp_id][0];
        wmma::store_matrix_sync(warp_smem, c_frag, TILE_N, wmma::mem_row_major);
        __syncwarp();

        for (int e = lane; e < TILE_M * TILE_N; e += 32) {
            const int i = e / TILE_N;
            const int j = e - i * TILE_N;
            C_local_symm[(int64_t)(row0 + i) * N + (col0 + j)] =
                __float2bfloat16_rn(warp_smem[e]);
        }

        __threadfence_system();
        __syncwarp();

        // Phase 0: all ranks have written their local tile.
        warp_rank_barrier(
            flag_ptrs,
            tile_id * 2,
            world_size,
            rank,
            epoch_value * 2,
            lane
        );

        // Reduce this tile directly from peer symmetric buffers via UVA.
        for (int e = lane; e < TILE_M * TILE_N; e += 32) {
            const int i = e / TILE_N;
            const int j = e - i * TILE_N;
            const int64_t off = (int64_t)(row0 + i) * N + (col0 + j);

            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r < world_size) {
                    const __nv_bfloat16* peer_c =
                        reinterpret_cast<const __nv_bfloat16*>(
                            static_cast<uintptr_t>(c_ptrs[r]));
                    sum += __bfloat162float(peer_c[off]);
                }
            }
            Out[off] = __float2bfloat16_rn(sum);
        }

        __threadfence_system();
        __syncwarp();

        // Phase 1: all ranks have finished reading this tile, so it is safe for
        // a faster rank to reuse/overwrite it in a subsequent call.
        warp_rank_barrier(
            flag_ptrs,
            tile_id * 2 + 1,
            world_size,
            rank,
            epoch_value * 2 + 1,
            lane
        );
    }
}

// -----------------------------------------------------------------------------
// Generic scalar fallback. Correct for BF16/FP16/FP32 and arbitrary dimensions.
// Still uses the same per-tile device-side symmetric-memory all-reduce.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void fused_gemm_allreduce_scalar_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C_local_symm,
    const long long* __restrict__ c_ptrs,
    T* __restrict__ Out,
    const long long* __restrict__ flag_ptrs,
    int M,
    int K,
    int N,
    int tiles_m,
    int tiles_n,
    int num_tiles,
    int world_size,
    int rank,
    int epoch_value
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total_warps = gridDim.x * WARPS_PER_BLOCK;

    for (int tile_id = global_warp; tile_id < num_tiles; tile_id += total_warps) {
        const int tm = tile_id / tiles_n;
        const int tn = tile_id - tm * tiles_n;

        const int row0 = tm * TILE_M;
        const int col0 = tn * TILE_N;
        const int valid_m = min(TILE_M, M - row0);
        const int valid_n = min(TILE_N, N - col0);
        const int elems = valid_m * valid_n;

        for (int e = lane; e < elems; e += 32) {
            const int i = e / valid_n;
            const int j = e - i * valid_n;
            const int row = row0 + i;
            const int col = col0 + j;

            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float av = to_float_dev<T>(A[(int64_t)row * K + k]);
                float bv = to_float_dev<T>(B[(int64_t)k * N + col]);
                acc += av * bv;
            }
            C_local_symm[(int64_t)row * N + col] = from_float_dev<T>(acc);
        }

        __threadfence_system();
        __syncwarp();

        warp_rank_barrier(
            flag_ptrs,
            tile_id * 2,
            world_size,
            rank,
            epoch_value * 2,
            lane
        );

        for (int e = lane; e < elems; e += 32) {
            const int i = e / valid_n;
            const int j = e - i * valid_n;
            const int row = row0 + i;
            const int col = col0 + j;
            const int64_t off = (int64_t)row * N + col;

            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r < world_size) {
                    const T* peer_c =
                        reinterpret_cast<const T*>(
                            static_cast<uintptr_t>(c_ptrs[r]));
                    sum += to_float_dev<T>(peer_c[off]);
                }
            }
            Out[off] = from_float_dev<T>(sum);
        }

        __threadfence_system();
        __syncwarp();

        warp_rank_barrier(
            flag_ptrs,
            tile_id * 2 + 1,
            world_size,
            rank,
            epoch_value * 2 + 1,
            lane
        );
    }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

void init_i32(torch::Tensor t) {
    TORCH_CHECK(t.is_cuda(), "init_i32: tensor must be CUDA");
    TORCH_CHECK(t.dtype() == torch::kInt32, "init_i32: tensor must be int32");
    TORCH_CHECK(t.is_contiguous(), "init_i32: tensor must be contiguous");

    int64_t n = t.numel();
    if (n == 0) return;

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    init_i32_kernel<<<blocks, threads, 0, stream>>>(t.data_ptr<int>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fused_gemm_allreduce(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C_symm,
    torch::Tensor c_ptrs,
    torch::Tensor Out,
    torch::Tensor flag_ptrs,
    int64_t M64,
    int64_t K64,
    int64_t N64,
    int world_size,
    int rank,
    int epoch_value
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C_symm.is_cuda() && Out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() &&
                C_symm.is_contiguous() && Out.is_contiguous(),
                "A, B, C_symm, Out must be contiguous");
    TORCH_CHECK(c_ptrs.is_cuda() && flag_ptrs.is_cuda(),
                "pointer tensors must be CUDA");
    TORCH_CHECK(c_ptrs.dtype() == torch::kInt64 &&
                flag_ptrs.dtype() == torch::kInt64,
                "pointer tensors must be int64");
    TORCH_CHECK(world_size >= 1 && world_size <= 8,
                "this H100 on-node implementation expects world_size in [1, 8]");

    int M = (int)M64;
    int K = (int)K64;
    int N = (int)N64;

    const int tiles_m = (M + TILE_M - 1) / TILE_M;
    const int tiles_n = (N + TILE_N - 1) / TILE_N;
    const int num_tiles = tiles_m * tiles_n;

    if (num_tiles == 0) return;

    cudaDeviceProp prop;
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // Keep blocks resident-ish and deterministic across ranks. Each block has
    // four persistent warps that walk tiles in the same order.
    int blocks_needed = (num_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int blocks = blocks_needed < prop.multiProcessorCount
        ? blocks_needed
        : prop.multiProcessorCount;
    if (blocks < 1) blocks = 1;

    dim3 grid(blocks);
    dim3 block(THREADS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* cptr =
        reinterpret_cast<const long long*>(c_ptrs.data_ptr<int64_t>());
    const long long* fptr =
        reinterpret_cast<const long long*>(flag_ptrs.data_ptr<int64_t>());

    if (A.dtype() == torch::kBFloat16) {
        TORCH_CHECK(B.dtype() == torch::kBFloat16 &&
                    C_symm.dtype() == torch::kBFloat16 &&
                    Out.dtype() == torch::kBFloat16,
                    "BF16 path requires all tensors BF16");

        const bool aligned = ((M % 16) == 0) && ((N % 16) == 0) && ((K % 16) == 0);

        const __nv_bfloat16* Ap =
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
        const __nv_bfloat16* Bp =
            reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>());
        __nv_bfloat16* Cp =
            reinterpret_cast<__nv_bfloat16*>(C_symm.data_ptr<at::BFloat16>());
        __nv_bfloat16* Op =
            reinterpret_cast<__nv_bfloat16*>(Out.data_ptr<at::BFloat16>());

        if (aligned) {
            fused_gemm_allreduce_bf16_wmma_kernel<<<grid, block, 0, stream>>>(
                Ap, Bp, Cp, cptr, Op, fptr,
                M, K, N, tiles_m, tiles_n, num_tiles,
                world_size, rank, epoch_value
            );
        } else {
            fused_gemm_allreduce_scalar_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                Ap, Bp, Cp, cptr, Op, fptr,
                M, K, N, tiles_m, tiles_n, num_tiles,
                world_size, rank, epoch_value
            );
        }
    } else if (A.dtype() == torch::kFloat16) {
        TORCH_CHECK(B.dtype() == torch::kFloat16 &&
                    C_symm.dtype() == torch::kFloat16 &&
                    Out.dtype() == torch::kFloat16,
                    "FP16 path requires all tensors FP16");

        const __half* Ap =
            reinterpret_cast<const __half*>(A.data_ptr<at::Half>());
        const __half* Bp =
            reinterpret_cast<const __half*>(B.data_ptr<at::Half>());
        __half* Cp =
            reinterpret_cast<__half*>(C_symm.data_ptr<at::Half>());
        __half* Op =
            reinterpret_cast<__half*>(Out.data_ptr<at::Half>());

        fused_gemm_allreduce_scalar_kernel<__half><<<grid, block, 0, stream>>>(
            Ap, Bp, Cp, cptr, Op, fptr,
            M, K, N, tiles_m, tiles_n, num_tiles,
            world_size, rank, epoch_value
        );
    } else if (A.dtype() == torch::kFloat32) {
        TORCH_CHECK(B.dtype() == torch::kFloat32 &&
                    C_symm.dtype() == torch::kFloat32 &&
                    Out.dtype() == torch::kFloat32,
                    "FP32 path requires all tensors FP32");

        fused_gemm_allreduce_scalar_kernel<float><<<grid, block, 0, stream>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C_symm.data_ptr<float>(),
            cptr,
            Out.data_ptr<float>(),
            fptr,
            M, K, N, tiles_m, tiles_n, num_tiles,
            world_size, rank, epoch_value
        );
    } else {
        TORCH_CHECK(false, "supported dtypes: bfloat16, float16, float32");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_i32", &init_i32, "Initialize int32 CUDA tensor to zero");
    m.def("launch_fused_gemm_allreduce", &launch_fused_gemm_allreduce,
          "Fused GEMM + symmetric-memory UVA all-reduce");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gemm_allreduce_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_resources(M: int, K: int, N: int, dtype: torch.dtype, device: torch.device):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    tiles_m = _ceil_div(M, 16)
    tiles_n = _ceil_div(N, 16)
    num_tiles = tiles_m * tiles_n

    key = (M, K, N, dtype, device, world_size)
    res = _resource_cache.get(key, None)
    if res is not None:
        res["epoch"] += 1
        if res["epoch"] > 1_000_000_000:
            # Avoid int wrap in long-running processes.
            _get_ext().init_i32(res["flags"])
            res["flag_hdl"].barrier(channel=0)
            res["epoch"] = 1
        return res

    # Symmetric local partial C buffer. Peers load these via UVA.
    c_symm = symm_mem.empty((M, N), device=device, dtype=dtype)
    c_hdl = symm_mem.rendezvous(c_symm, dist.group.WORLD)

    # Two device-side barrier phases per tile, one slot per peer rank.
    flags = symm_mem.empty((max(num_tiles, 1) * 2 * world_size,),
                           device=device,
                           dtype=torch.int32)
    flag_hdl = symm_mem.rendezvous(flags, dist.group.WORLD)

    out = torch.empty((M, N), device=device, dtype=dtype)

    c_ptrs = torch.tensor(c_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    flag_ptrs = torch.tensor(flag_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    _get_ext().init_i32(flags)
    # Setup-only symmetric-memory barrier so no rank observes uninitialized flags.
    flag_hdl.barrier(channel=0)

    res = {
        "M": M,
        "K": K,
        "N": N,
        "dtype": dtype,
        "device": device,
        "world_size": world_size,
        "rank": rank,
        "num_tiles": num_tiles,
        "c_symm": c_symm,
        "c_hdl": c_hdl,
        "flags": flags,
        "flag_hdl": flag_hdl,
        "out": out,
        "c_ptrs": c_ptrs,
        "flag_ptrs": flag_ptrs,
        "epoch": 1,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B_local: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B_local.is_cuda, "Inputs must be CUDA tensors"

    if not A_local.is_contiguous():
        A_local = A_local.contiguous()
    if not B_local.is_contiguous():
        B_local = B_local.contiguous()

    M, K = A_local.shape
    K_B, N = B_local.shape
    assert K == K_B, f"A_local and B_local must have matching K dimension: {K} != {K_B}"
    assert A_local.dtype == B_local.dtype, "A_local and B_local must have same dtype"

    res = _get_resources(M, K, N, A_local.dtype, A_local.device)

    _get_ext().launch_fused_gemm_allreduce(
        A_local,
        B_local,
        res["c_symm"],
        res["c_ptrs"],
        res["out"],
        res["flag_ptrs"],
        M,
        K,
        N,
        dist.get_world_size(),
        dist.get_rank(),
        res["epoch"],
    )

    return res["out"]