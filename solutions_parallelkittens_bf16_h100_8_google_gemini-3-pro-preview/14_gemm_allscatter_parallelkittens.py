"""
Strategy:
- **Device-Side Communication & Overlap**: Instead of computing a full local GEMM shard and executing a host-driven `all_gather` collective, we fuse the GEMM and communication. The ThunderKittens kernel computes the local shard $C_{\text{local}} = A \times B_{\text{local}}$ in 128x128 tiles. As soon as a tile finishes computing via WGMMA, it uses asynchronous TMA stores to broadcast the result to its final offset in the symmetric `C` matrix on all peers over NVLink.
- **Pipelining**: Within the kernel, double-buffered TMA loads overlap with MMA compute. At the grid level, blocking on TMA stores natively overlaps with independent blocks executing on other SMs, maximizing compute and network saturation without extra buffers.
"""

import os
import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source 
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace gemm_allgather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_THREADS = NUM_WARPGROUPS * 128;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 64;

    using st_A = st_bf<BM, BK>;
    using st_B = st_bf<BK, BN>;
    using st_C = st_bf<BM, BN>;

    using layout_A = pgl<gl<bf16, -1, -1, -1, -1, st_A>, NUM_DEVICES, false>;
    using layout_B = pgl<gl<bf16, -1, -1, -1, -1, st_B>, NUM_DEVICES, false>;
    using layout_C = pgl<gl<bf16, -1, -1, -1, -1, st_C>, NUM_DEVICES, false>;

    layout_A A;
    layout_B B;
    layout_C C;
    
    int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(B.cols() / BN, A.rows() / BM);
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(2 * sizeof(st_A) + 2 * sizeof(st_B) + sizeof(st_C) + 2048);
    }
};

__device__ inline void kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    
    globals::st_A (&smem_A)[2] = allocator.allocate<globals::st_A, 2>();
    globals::st_B (&smem_B)[2] = allocator.allocate<globals::st_B, 2>();
    globals::st_C &smem_C = allocator.allocate<globals::st_C>();

    int i_n = blockIdx.x;
    int i_m = blockIdx.y;
    int K_blocks = G.A.cols() / globals::BK;

    __shared__ semaphore arrived_A[2];
    __shared__ semaphore arrived_B[2];

    if (threadIdx.x == 0) {
        init_semaphore(arrived_A[0], 0, 1);
        init_semaphore(arrived_A[1], 0, 1);
        init_semaphore(arrived_B[0], 0, 1);
        init_semaphore(arrived_B[1], 0, 1);
    }
    __syncthreads();

    int tic = 0, toc = 1;

    if (threadIdx.x == 0) {
        tma::expect_bytes(arrived_A[tic], sizeof(globals::st_A));
        tma::expect_bytes(arrived_B[tic], sizeof(globals::st_B));
        tma::load_async(smem_A[tic], G.A[G.dev_idx], {0, 0, i_m, 0}, arrived_A[tic]);
        tma::load_async(smem_B[tic], G.B[G.dev_idx], {0, 0, 0, i_n}, arrived_B[tic]);
    }

    rt_bf<globals::BM, globals::BN> accum;
    zero(accum);

    for (int k = 0; k < K_blocks; k++, tic = toc, toc ^= 1) {
        if (k < K_blocks - 1) {
            if (threadIdx.x == 0) {
                tma::expect_bytes(arrived_A[toc], sizeof(globals::st_A));
                tma::expect_bytes(arrived_B[toc], sizeof(globals::st_B));
                tma::load_async(smem_A[toc], G.A[G.dev_idx], {0, 0, i_m, k + 1}, arrived_A[toc]);
                tma::load_async(smem_B[toc], G.B[G.dev_idx], {0, 0, k + 1, i_n}, arrived_B[toc]);
            }
        }
        
        int phase = k / 2;
        wait(arrived_A[tic], phase);
        wait(arrived_B[tic], phase);
        
        warpgroup::mma_AB(accum, smem_A[tic], smem_B[tic]);
    }

    warpgroup::mma_async_wait();
    
    // Write out the MAC result to shared memory
    warpgroup::store(smem_C, accum);
    __syncthreads();

    // Broadcast the result to all peers directly into their symmetric memory blocks
    if (threadIdx.x == 0) {
        int N_local_blocks = G.B.cols() / globals::BN;
        int dst_col_block = G.dev_idx * N_local_blocks + i_n;
        
        #pragma unroll
        for(int d = 0; d < globals::NUM_DEVICES; d++) {
            tma::store_async(G.C[d], smem_C, {0, 0, i_m, dst_col_block});
        }
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group 0;");
    }
    __syncthreads();
}

} // namespace gemm_allgather

namespace barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace barrier

void entrypoint(
    kittens::py::TKParallelTensor &A,
    kittens::py::TKParallelTensor &B,
    kittens::py::TKParallelTensor &C,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(A, B, C, barrier);

    gemm_allgather::globals G {
        .A = kittens::py::parallel_tensor_to_pgl<typename gemm_allgather::globals::layout_A>(A),
        .B = kittens::py::parallel_tensor_to_pgl<typename gemm_allgather::globals::layout_B>(B),
        .C = kittens::py::parallel_tensor_to_pgl<typename gemm_allgather::globals::layout_C>(C),
        .dev_idx = A.local_rank_
    };

    barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<barrier::config, barrier::globals, barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<gemm_allgather::config, gemm_allgather::globals, gemm_allgather::kernel>(G);
    kittens::py::launch_kernel<barrier::config, barrier::globals, barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_gemm_allgather", &entrypoint);
}
'''

TK_CUDA_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-DNDEBUG",
]

_ext = None
_ext_jit_ready = False


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_gemmallgather_ext",
            CUDA_SRC,
            extra_cuda_cflags=TK_CUDA_FLAGS,
            extra_include_paths=[
                os.path.join(TK_ROOT, "include"),
                os.path.join(TK_ROOT, "prototype"),
            ],
            extra_ldflags=["-lcuda"],
        )
    return _ext


def _ensure_ext_jit():
    """Compile/load extension once; avoid per-call ``dist.barrier()`` in timed hot path."""
    global _ext_jit_ready
    if _ext_jit_ready:
        return _get_ext()
    rank = dist.get_rank()
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


@torch.no_grad()
def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    world_size = dist.get_world_size()
    assert world_size == 8, "This ThunderKittens kernel is built for NUM_DEVICES=8"
    
    M, K = A.shape
    K_B, N_local = B.shape
    assert K == K_B, f"A and B must have matching K dimension: {K} != {K_B}"
    
    ext = _ensure_ext_jit()
    original_dtype = A.dtype

    # Quantize blocks cleanly into ThunderKittens tile sizes
    pad_M = (M + 127) // 128 * 128
    pad_K = (K + 63) // 64 * 64
    pad_N_local = (N_local + 127) // 128 * 128
    
    # Simple hack to circumvent shape collision if `pad_M == pad_N_local` inside 
    # parallelkittens caching map. Adding an unused dummy tile column shifts
    # uniquely without perturbing valid matrix bounds in memory.
    if pad_M == pad_N_local:
        pad_N_local += 128
        
    pad_N_total = world_size * pad_N_local
    
    A_tk = get_or_create_parallel_tensor(ext, (1, 1, pad_M, pad_K), torch.bfloat16, multicast=False)
    B_tk = get_or_create_parallel_tensor(ext, (1, 1, pad_K, pad_N_local), torch.bfloat16, multicast=False)
    C_tk = get_or_create_parallel_tensor(ext, (1, 1, pad_M, pad_N_total), torch.bfloat16, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    
    # Clear padded borders to bypass contamination and fill tensors
    A_flat_len = pad_M * pad_K
    A_tk.data_.reshape(-1)[:A_flat_len].view(pad_M, pad_K).zero_()
    A_tk.data_.reshape(-1)[:A_flat_len].view(pad_M, pad_K)[:M, :K].copy_(A)

    B_flat_len = pad_K * pad_N_local
    B_tk.data_.reshape(-1)[:B_flat_len].view(pad_K, pad_N_local).zero_()
    B_tk.data_.reshape(-1)[:B_flat_len].view(pad_K, pad_N_local)[:K, :N_local].copy_(B)
    
    ext.tk_gemm_allgather(A_tk, B_tk, C_tk, barrier_tk)
    
    C_flat_len = pad_M * pad_N_total
    C_out_full = C_tk.data_.reshape(-1)[:C_flat_len].view(pad_M, pad_N_total)
    
    if pad_N_local == N_local:
        C_final = C_out_full[:M, :world_size * N_local].clone()
    else:
        chunks = []
        for i in range(world_size):
            chunks.append(C_out_full[:M, i * pad_N_local : i * pad_N_local + N_local])
        C_final = torch.cat(chunks, dim=1)
        
    return C_final.to(original_dtype)