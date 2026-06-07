"""
ThunderKittens pipelined GEMM + reduce-scatter.

Strategy:
- Overlap Compute and Comm: We partition the M dimension into world_size chunks. PyTorch GEMMs are issued sequentially on the default stream, while a custom ThunderKittens P2P kernel executes on a comm stream, overlapping Tensor Core math with NVLink memory transfers.
- Device-side Data Movement: We allocate a symmetric TKParallelTensor buffer. Each rank explicitly pulls its designated output chunk from all peers' memory using direct one-sided vectorized loads, summing inline and bypassing intermediate buffers.
- Stream-Native Barriers: Fine-grained device-side barriers (barrier_all) are launched per-chunk on the communication stream, synchronizing memory access seamlessly without stalling the host CPU.
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
# Embedded .cu source: P2P Pull Reduce-Scatter
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace rs_chunk {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    // We treat the parallel tensors as 1D logical layouts and calculate offsets manually
    using layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, false>;
    
    layout c_local;
    layout c_partial;
    barrier_t<NUM_DEVICES> barrier;
    
    int dev_idx;
    int chunk_idx;
    int num_elems;
    int padded_elems;
    
    __host__ inline dim3 grid() const {
        return dim3((num_elems + config::NUM_THREADS * 8 - 1) / (config::NUM_THREADS * 8));
    }
};

__device__ inline void kernel(const globals& G) {
    // Wait for all peers to finish the GEMM computation for this chunk
    barrier_all(G.barrier, {0}, G.dev_idx);
    
    // Only the rank responsible for this chunk does the pull-and-reduce
    if (G.dev_idx == G.chunk_idx) {
        const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int base_idx = vec_idx * 8;
        
        if (base_idx < G.num_elems) {
            float sum[8] = {0.0f};
            int valid = (G.num_elems - base_idx > 8) ? 8 : (G.num_elems - base_idx);
            
            #pragma unroll
            for (int p = 0; p < globals::NUM_DEVICES; p++) {
                bf16* peer_ptr = (bf16*)&G.c_partial[p](0,0,0,0);
                
                if (valid == 8) {
                    // Fast path: fully vectorized 16-byte load
                    uint4 vec = *(uint4*)(&peer_ptr[G.chunk_idx * G.padded_elems + base_idx]);
                    bf16* vals = (bf16*)&vec;
                    for(int i = 0; i < 8; ++i) {
                        sum[i] += __bfloat162float(vals[i]);
                    }
                } else {
                    // Edge case path
                    for(int i = 0; i < valid; ++i) {
                        sum[i] += __bfloat162float(peer_ptr[G.chunk_idx * G.padded_elems + base_idx + i]);
                    }
                }
            }
            
            bf16* out_ptr = (bf16*)&G.c_local[G.dev_idx](0,0,0,0);
            if (valid == 8) {
                uint4 out_vec;
                bf16* out_vals = (bf16*)&out_vec;
                for(int i = 0; i < 8; ++i) {
                    out_vals[i] = __float2bfloat16(sum[i]);
                }
                *(uint4*)(&out_ptr[base_idx]) = out_vec;
            } else {
                for(int i = 0; i < valid; ++i) {
                    out_ptr[base_idx + i] = __float2bfloat16(sum[i]);
                }
            }
        }
    }
}

} // namespace rs_chunk

namespace sync_only {
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 128;
};
struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    int dev_idx;
    __host__ inline dim3 grid() const { return dim3(1); }
};
__device__ inline void kernel(const globals& G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}
} // namespace sync_only

void rs_entrypoint(
    kittens::py::TKParallelTensor &c_local,
    kittens::py::TKParallelTensor &c_partial,
    kittens::py::TKParallelTensor &barrier,
    int chunk_idx,
    int num_elems,
    int padded_elems
) {
    kittens::py::parallel_tensor_check(c_local, c_partial, barrier);
    
    rs_chunk::globals G {
        .c_local = kittens::py::parallel_tensor_to_pgl<typename rs_chunk::globals::layout>(c_local),
        .c_partial = kittens::py::parallel_tensor_to_pgl<typename rs_chunk::globals::layout>(c_partial),
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<rs_chunk::globals::NUM_DEVICES>>(barrier),
        .dev_idx = c_local.local_rank_,
        .chunk_idx = chunk_idx,
        .num_elems = num_elems,
        .padded_elems = padded_elems
    };
    
    kittens::py::launch_kernel<rs_chunk::config, rs_chunk::globals, rs_chunk::kernel>(G);
}

void barrier_entrypoint(
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(barrier);
    
    sync_only::globals G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<sync_only::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    
    kittens::py::launch_kernel<sync_only::config, sync_only::globals, sync_only::kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_reduce_scatter_chunk", &rs_entrypoint);
    m.def("tk_barrier_only", &barrier_entrypoint);
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
NUM_DEVICES = 8


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_gemm_reducescatter_ext",
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
def solution(A_local: torch.Tensor, B_local: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B_local.is_cuda, "Inputs must be CUDA tensors"
    
    world_size = dist.get_world_size()
    assert world_size == NUM_DEVICES, f"Expected {NUM_DEVICES} ranks for this compiled kernel"
    
    M, K_local = A_local.shape
    K_B, N = B_local.shape
    assert K_local == K_B, "A_local and B_local inner dims mismatch"
    assert M % world_size == 0, "M must be divisible by world_size"
    
    M_local = M // world_size
    num_elems = M_local * N
    
    # Pad alignment to 2048 elements for safety with flat VMM allocations
    padded_elems = ((num_elems + 2047) // 2048) * 2048
    
    ext = _ensure_ext_jit()

    # Cached TKParallelTensor buffers
    C_partial_tk = get_or_create_parallel_tensor(ext, (world_size, padded_elems), torch.bfloat16, multicast=False)
    C_local_tk = get_or_create_parallel_tensor(ext, (padded_elems,), torch.bfloat16, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    
    gemm_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()
    gemm_events = [torch.cuda.Event() for _ in range(world_size)]
    
    c_partial_views = []
    for c in range(world_size):
        v = C_partial_tk.data_[c, :num_elems].view(M_local, N)
        c_partial_views.append(v)
        
    # Pipelined Compute and Communication
    for c in range(world_size):
        # 1. Compute chunk c into symmetric buffer directly
        A_chunk = A_local[c * M_local : (c + 1) * M_local, :]
        torch.matmul(A_chunk, B_local, out=c_partial_views[c])
        gemm_events[c].record(gemm_stream)
        
        # 2. Launch TK pull-kernel to overlap with the next GEMM
        comm_stream.wait_event(gemm_events[c])
        with torch.cuda.stream(comm_stream):
            ext.tk_reduce_scatter_chunk(
                C_local_tk, C_partial_tk, barrier_tk, c, num_elems, padded_elems
            )
            
    # Launch final device-side sync on the comm stream. This ensures no peer returns and 
    # executes its next loop call (which could overwrite C_partial) before all reads conclude.
    with torch.cuda.stream(comm_stream):
        ext.tk_barrier_only(barrier_tk)
        
    gemm_stream.wait_stream(comm_stream)

    # Return local result from the symmetrical buffer
    C_local = torch.empty((M_local, N), dtype=A_local.dtype, device=A_local.device)
    C_local.copy_(C_local_tk.data_[:num_elems].view(M_local, N))
    
    return C_local