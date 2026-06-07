import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

// =============================================================================
// Device-side signal helpers for symmetric-memory block barriers
// =============================================================================

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old) : "l"(addr) : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old) : "l"(addr) : "memory");
    } while (old != 1u);
}

__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old) : "l"(addr) : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old) : "l"(addr) : "memory");
    } while (old != 1u);
}

__device__ __forceinline__ void blockwise_barrier_relaxed(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned tid = threadIdx.x;
    if (tid >= (unsigned)world_size) return;

    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t peer_base = signal_pad_ptrs[tid];

    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        peer_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);

    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

__device__ __forceinline__ void blockwise_barrier_acq_rel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned tid = threadIdx.x;
    if (tid >= (unsigned)world_size) return;

    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t peer_base = signal_pad_ptrs[tid];

    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        peer_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);

    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

// =============================================================================
// BF16 WMMA GEMM: partial C = A_local[M, K_local] @ B_shard[K_local, N]
// Row-major inputs/outputs.
// One warp computes one 16x16 tile.
// =============================================================================

__global__ void gemm_bf16_wmma_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M,
    int Kloc,
    int N,
    int rank
) {
#if __CUDA_ARCH__ >= 800
    const int tile_n = blockIdx.x;
    const int tile_m = blockIdx.y;
    const int row0 = tile_m * 16;
    const int col0 = tile_n * 16;
    const int tid = threadIdx.x & 31;

    extern __shared__ unsigned char smem_raw[];
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* Bs = As + 256;
    float* Cs = reinterpret_cast<float*>(Bs + 256);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    const __nv_bfloat16* B_shard = B + (int64_t)rank * (int64_t)Kloc * (int64_t)N;

    for (int k0 = 0; k0 < Kloc; k0 += 16) {
        for (int i = tid; i < 256; i += 32) {
            int r = i >> 4;
            int c = i & 15;

            int ar = row0 + r;
            int ac = k0 + c;
            int br = k0 + r;
            int bc = col0 + c;

            As[i] = (ar < M && ac < Kloc)
                ? A[(int64_t)ar * Kloc + ac]
                : __float2bfloat16(0.0f);

            Bs[i] = (br < Kloc && bc < N)
                ? B_shard[(int64_t)br * N + bc]
                : __float2bfloat16(0.0f);
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, As, 16);
        wmma::load_matrix_sync(b_frag, Bs, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(Cs, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < 256; i += 32) {
        int r = i >> 4;
        int c = i & 15;
        int rr = row0 + r;
        int cc = col0 + c;
        if (rr < M && cc < N) {
            C[(int64_t)rr * N + cc] = __float2bfloat16(Cs[i]);
        }
    }
#endif
}

// =============================================================================
// Scalar fallback GEMMs for fp32/fp16 correctness outside BF16 benchmark path
// =============================================================================

__global__ void gemm_f32_scalar_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int Kloc,
    int N,
    int rank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)M * N;
    const float* B_shard = B + (int64_t)rank * Kloc * N;

    for (int64_t linear = idx; linear < total; linear += (int64_t)gridDim.x * blockDim.x) {
        int m = linear / N;
        int n = linear - (int64_t)m * N;
        float acc = 0.0f;
        for (int k = 0; k < Kloc; ++k) {
            acc += A[(int64_t)m * Kloc + k] * B_shard[(int64_t)k * N + n];
        }
        C[linear] = acc;
    }
}

__global__ void gemm_f16_scalar_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M,
    int Kloc,
    int N,
    int rank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)M * N;
    const half* B_shard = B + (int64_t)rank * Kloc * N;

    for (int64_t linear = idx; linear < total; linear += (int64_t)gridDim.x * blockDim.x) {
        int m = linear / N;
        int n = linear - (int64_t)m * N;
        float acc = 0.0f;
        for (int k = 0; k < Kloc; ++k) {
            acc += __half2float(A[(int64_t)m * Kloc + k]) *
                   __half2float(B_shard[(int64_t)k * N + n]);
        }
        C[linear] = __float2half(acc);
    }
}

// =============================================================================
// NVSwitch multimem BF16 all-reduce over symmetric partial C.
// Reduces 8 BF16 elements per logical element.
// =============================================================================

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& x,
    uint32_t& y,
    uint32_t& z,
    uint32_t& w
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
        : "l"(addr)
        : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(
    uint64_t* addr,
    uint32_t x,
    uint32_t y,
    uint32_t z,
    uint32_t w
) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w)
        : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t n_vec128,
    int world_size,
    int rank,
    int block_stride
) {
#if __CUDA_ARCH__ >= 900
    const uint64_t bid = blockIdx.x;

    blockwise_barrier_acq_rel(signal_pad_ptrs, bid, rank, world_size);
    __syncthreads();

    const int64_t per_rank =
        (n_vec128 + (int64_t)world_size - 1) / (int64_t)world_size;

    const int tid = threadIdx.x;
    const int nblocks = gridDim.x;

    for (int64_t local_i = (int64_t)bid * block_stride + tid;
         local_i < per_rank;
         local_i += (int64_t)nblocks * block_stride) {
        int64_t vec_idx = (int64_t)rank * per_rank + local_i;
        if (vec_idx < n_vec128) {
            uint64_t* p = reinterpret_cast<uint64_t*>(multicast_base) + vec_idx * 2;
            uint32_t x, y, z, w;
            multimem_ld_reduce_bf16x4(p, x, y, z, w);
            multimem_st_bf16x4(p, x, y, z, w);
        }
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, bid, rank, world_size);
#endif
}

// =============================================================================
// UVA peer-pointer all-reduce fallback
// =============================================================================

__global__ void allreduce_bf16_peer_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* p =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[r]);
                acc += __bfloat162float(p[idx]);
            }
        }
        out[idx] = __float2bfloat16(acc);
    }
}

__global__ void allreduce_f32_peer_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const float* p = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
                acc += p[idx];
            }
        }
        out[idx] = acc;
    }
}

__global__ void allreduce_f16_peer_kernel(
    const long long* __restrict__ ptrs,
    half* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const half* p = reinterpret_cast<const half*>((uintptr_t)ptrs[r]);
                acc += __half2float(p[idx]);
            }
        }
        out[idx] = __float2half(acc);
    }
}

// dtype_enum: 0=bf16, 1=float32, 2=float16
void launch_local_gemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int rank,
    int dtype_enum
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "contiguous tensors required");

    int M = (int)A.size(0);
    int Kloc = (int)A.size(1);
    int N = (int)B.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        dim3 block(32);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        size_t smem = 256 * sizeof(__nv_bfloat16) * 2 + 256 * sizeof(float);
        gemm_bf16_wmma_kernel<<<grid, block, smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
            M, Kloc, N, rank);
    } else {
        int64_t total = (int64_t)M * N;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;

        if (dtype_enum == 1) {
            gemm_f32_scalar_kernel<<<blocks, threads, 0, stream>>>(
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                C.data_ptr<float>(),
                M, Kloc, N, rank);
        } else {
            gemm_f16_scalar_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
                reinterpret_cast<half*>(C.data_ptr<at::Half>()),
                M, Kloc, N, rank);
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t n_vec128,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride
) {
    const uint64_t* sig =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr,
        sig,
        n_vec128,
        world_size,
        rank,
        block_stride);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_peer_allreduce(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n,
    int dtype_enum
) {
    int world_size = (int)ptrs_tensor.size(0);
    const long long* ptrs =
        reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>());

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        allreduce_bf16_peer_kernel<<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            world_size,
            n);
    } else if (dtype_enum == 1) {
        allreduce_f32_peer_kernel<<<blocks, threads, 0, stream>>>(
            ptrs,
            out.data_ptr<float>(),
            world_size,
            n);
    } else {
        allreduce_f16_peer_kernel<<<blocks, threads, 0, stream>>>(
            ptrs,
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            world_size,
            n);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_local_gemm", &launch_local_gemm, "Local sharded GEMM");
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16,
          "NVSwitch multimem BF16 all-reduce");
    m.def("launch_peer_allreduce", &launch_peer_allreduce,
          "UVA peer-pointer all-reduce fallback");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gemm_allgather_at_symm_wmma_bf16_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.float32:
        return 1
    if dtype is torch.float16:
        return 2
    raise TypeError(f"unsupported dtype for custom CUDA path: {dtype}")


def _get_resources(shape, dtype, device):
    key = (tuple(shape), dtype, device, dist.get_world_size())
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    partial = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(partial, dist.group.WORLD)

    out = torch.empty(shape, device=device, dtype=dtype)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (partial, hdl, out, ptrs)
    _resource_cache[key] = cached
    return cached


WARP_SIZE = 32
MAX_NUM_BLOCKS = 4
MAX_BLOCK_SIZE = 1024
BYTES_PER_MULTIMEM_THREAD = 16


def _multimem_launch_config(numel_bf16: int, world_size: int):
    elems_per_thread = BYTES_PER_MULTIMEM_THREAD // 2
    num_threads = (numel_bf16 // elems_per_thread + world_size - 1) // world_size

    if num_threads <= 1:
        return 1, 1, 1

    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads:
            block_size <<= 1
        return 1, block_size, block_size

    block_size = MAX_BLOCK_SIZE
    num_blocks = min(
        (num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE,
        MAX_NUM_BLOCKS,
    )
    return num_blocks, block_size, block_size


@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "inputs must be CUDA tensors"

    if not A_local.is_contiguous():
        A_local = A_local.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    M, K_local = A_local.shape
    K_B, N = B.shape
    assert K_B == world_size * K_local, (
        f"B must have K dimension = world_size * K_local: {K_B} != {world_size * K_local}"
    )
    assert A_local.dtype == B.dtype, "A_local and B must have same dtype"

    dtype = A_local.dtype
    dtype_id = _dtype_enum(dtype)

    ext = _get_ext()

    partial, hdl, out, ptrs = _get_resources((M, N), dtype, A_local.device)

    # Local rank computes its contribution:
    #   partial = A_local @ B[rank*K_local:(rank+1)*K_local, :]
    ext.launch_local_gemm(A_local, B, partial, rank, dtype_id)

    n = M * N

    if world_size == 1:
        return partial

    # Fast BF16 path: reduce symmetric partial-C buffers in-switch with multimem.
    # Requires 16-byte vector alignment: 8 BF16 elements per vector.
    if dtype is torch.bfloat16 and (n % 8 == 0):
        n_vec128 = n // 8
        num_blocks, block_size, block_stride = _multimem_launch_config(n, world_size)
        ext.launch_multimem_allreduce_bf16(
            int(hdl.multicast_ptr),
            hdl.signal_pad_ptrs_dev,
            n_vec128,
            world_size,
            rank,
            num_blocks,
            block_size,
            block_stride,
        )
        return partial

    # Fallback for odd sizes / fp16 / fp32:
    # Ensure local GEMM has completed before exposing the symmetric buffer, then
    # do explicit UVA peer loads in a CUDA kernel. This is still NCCL-free.
    torch.cuda.current_stream().synchronize()
    hdl.barrier(channel=0)
    ext.launch_peer_allreduce(ptrs, out, n, dtype_id)
    return out