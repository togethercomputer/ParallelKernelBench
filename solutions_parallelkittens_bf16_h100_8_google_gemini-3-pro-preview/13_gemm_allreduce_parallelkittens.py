"""
ThunderKittens Distributed GEMM with All-Reduce

Strategy:
We optimize the GEMM + All-Reduce map-reduce pattern by pipelining local matrix 
multiplications with device-side communication. Instead of computing the full 
local GEMM and then launching a monolithic collective, we slice the M dimension 
into chunks. The CPU submits the compute (cuBLAS) for chunk `i` and then asynchronously 
dispatches a custom ThunderKittens multimem reduction for that chunk on a separate 
communication stream. This hides the NVSwitch/NVLink network latency behind the dense 
tensor-core math of the subsequent chunks, maximizing overlapping and throughput on Hopper.
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
# Embedded .cu source (Pipelined all_reduce entrypoint + barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_reduce {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 2;
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout tensor;
    const int dev_idx;
    int offset;
    int chunk_numel;

    __host__ inline dim3 grid() const {
        return dim3(chunk_numel / NUM_ELEMS_PER_BLOCK / NUM_DEVICES);
    }
};

__device__ inline void kernel(const globals &G) {
    const size_t N_per_dev = G.chunk_numel / globals::NUM_DEVICES;
    const size_t idx = G.offset + N_per_dev * G.dev_idx +
                                  globals::NUM_ELEMS_PER_BLOCK * blockIdx.x +
                                  globals::NUM_ELEMS_PER_INST * threadIdx.x;

    bf16_2 tmp;
    multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<bf16_2*>(&G.tensor.mc_ptr[idx]));
    multimem<bf16_2>::st(reinterpret_cast<bf16_2*>(&G.tensor.mc_ptr[idx]), tmp);
}

} // namespace all_reduce

namespace all_reduce_barrier {

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

} // namespace all_reduce_barrier

void entrypoint(
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier,
    int offset,
    int chunk_numel
) {
    kittens::py::parallel_tensor_check(tensor, barrier);

    TORCH_CHECK(chunk_numel % (all_reduce::globals::NUM_DEVICES * all_reduce::globals::NUM_ELEMS_PER_BLOCK) == 0,
        "chunk_numel must be divisible by NUM_DEVICES * NUM_ELEMS_PER_BLOCK");

    all_reduce::globals all_reduce_G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename all_reduce::globals::parallel_layout>(tensor),
        .dev_idx = tensor.local_rank_,
        .offset = offset,
        .chunk_numel = chunk_numel
    };

    all_reduce_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_reduce_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<all_reduce::config, all_reduce::globals, all_reduce::kernel>(all_reduce_G);
    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_reduce_chunk", &entrypoint);
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
NUM_THREADS = 256  # NUM_WARPGROUPS(2) * WARPGROUP_WARPS(4) * WARP_THREADS(32)
NUM_ELEMS_PER_INST = 2
NUM_ELEMS_PER_BLOCK = NUM_THREADS * NUM_ELEMS_PER_INST
ALIGNMENT = NUM_DEVICES * NUM_ELEMS_PER_BLOCK  # 4096

_stream_comm = None
_events_compute = []


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_gemm_allreduce_ext",
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


def _get_comm_resources(chunks):
    """Reuse CUDA streams and events to minimize per-iteration allocation overhead."""
    global _stream_comm, _events_compute
    if _stream_comm is None:
        _stream_comm = torch.cuda.Stream()
    while len(_events_compute) < chunks:
        _events_compute.append(torch.cuda.Event())
    return _stream_comm, _events_compute[:chunks]


@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B_local: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B_local.is_cuda, "Inputs must be CUDA tensors"
    
    world = dist.get_world_size()
    ext = _ensure_ext_jit()

    original_dtype = A_local.dtype
    M, K = A_local.shape
    K_B, N = B_local.shape
    assert K == K_B, f"A_local and B_local must have matching K dimension: {K} != {K_B}"

    A_bf16 = A_local.to(torch.bfloat16).contiguous()
    B_bf16 = B_local.to(torch.bfloat16).contiguous()

    # Pipelining: overlap chunk N+1 compute with chunk N all-reduce.
    # Fallback to single chunk for tiny payloads to avoid slicing overhead.
    chunks = min(2, M)
    if M * N < 512 * 512:
        chunks = 1

    M_per_chunk = (M + chunks - 1) // chunks
    chunk_numel = M_per_chunk * N
    aligned_chunk_numel = ((chunk_numel + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    total_elements = aligned_chunk_numel * chunks

    tensor_tk = get_or_create_parallel_tensor(ext, (total_elements,), torch.bfloat16, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    stream_compute = torch.cuda.current_stream()
    stream_comm, events_compute_done = _get_comm_resources(chunks)

    # 1. Pipeline schedule: Compute GEMM chunk -> record event -> asynchronously trigger multimem reduce.
    for i in range(chunks):
        start_M = i * M_per_chunk
        end_M = min(start_M + M_per_chunk, M)
        actual_M = end_M - start_M
        actual_numel = actual_M * N
        offset = i * aligned_chunk_numel

        if actual_M > 0:
            # We construct a view directly into the symmetric VMM parallel tensor memory 
            out_view = tensor_tk.data_[offset : offset + actual_numel].view(actual_M, N)
            torch.matmul(A_bf16[start_M:end_M], B_bf16, out=out_view)
            
            # Zero out any padded tail to avoid incorporating garbage in the sum reduction
            if aligned_chunk_numel > actual_numel:
                tensor_tk.data_[offset + actual_numel : offset + aligned_chunk_numel].zero_()

        events_compute_done[i].record(stream_compute)

        # Offload synchronization & reduction collective to comm stream
        with torch.cuda.stream(stream_comm):
            stream_comm.wait_event(events_compute_done[i])
            if actual_M > 0:
                ext.tk_all_reduce_chunk(tensor_tk, barrier_tk, offset, aligned_chunk_numel)

    stream_compute.wait_stream(stream_comm)

    # 2. Gather results from the VMM buffer
    C = torch.empty((M, N), dtype=original_dtype, device=A_local.device)
    for i in range(chunks):
        start_M = i * M_per_chunk
        end_M = min(start_M + M_per_chunk, M)
        actual_M = end_M - start_M
        actual_numel = actual_M * N
        offset = i * aligned_chunk_numel
        
        if actual_M > 0:
            C[start_M:end_M] = tensor_tk.data_[offset : offset + actual_numel].view(actual_M, N).to(original_dtype)

    return C