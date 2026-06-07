"""
ThunderKittens MoE Token Preprocess (Expert Parallel).
Replaces NCCL all_gather with TMA-based peer-to-peer all-gather.
Defers blocking synchronizations to overlap compute and communication.
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source (TMA all-gather entrypoint + barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace tk_all_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1; // TMA driven
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;

    using shared_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(NUM_DEVICES * (input.rows() / ROW_BLOCK_SIZE) * (input.cols() / COL_BLOCK_SIZE));
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) + 1024);
    }
};

__device__ inline void kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::shared_tile &tile = allocator.allocate<globals::shared_tile>();

    int task_idx = blockIdx.x;
    int src_dev_idx = task_idx / ((G.input.rows() / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= ((G.input.rows() / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    
    int row_block_idx = task_idx / (G.input.cols() / globals::COL_BLOCK_SIZE);
    int col_block_idx = task_idx % (G.input.cols() / globals::COL_BLOCK_SIZE);

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    
    // Load tile from src_dev_idx's input tensor
    tma::load_async(tile, G.input[src_dev_idx], {0, 0, row_block_idx, col_block_idx}, arrived);
    wait(arrived, 0);

    // Store tile into local output tensor at the appropriate depth (source rank)
    tma::store_async(G.output[G.dev_idx], tile, {0, src_dev_idx, row_block_idx, col_block_idx});
}

} // namespace tk_all_gather

namespace tk_all_gather_barrier {

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

} // namespace tk_all_gather_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(output, input);

    tk_all_gather::globals all_gather_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename tk_all_gather::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename tk_all_gather::globals::parallel_layout>(input),
        .dev_idx = input.local_rank_
    };

    tk_all_gather_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<tk_all_gather_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Symmetric sync -> TMA Gather -> Symmetric sync
    kittens::py::launch_kernel<tk_all_gather_barrier::config, tk_all_gather_barrier::globals, tk_all_gather_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<tk_all_gather::config, tk_all_gather::globals, tk_all_gather::kernel>(all_gather_G);
    kittens::py::launch_kernel<tk_all_gather_barrier::config, tk_all_gather_barrier::globals, tk_all_gather_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_gather", &entrypoint);
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
ROW_TILE = 16
COL_TILE = 128
TILE_ELEMS = ROW_TILE * COL_TILE


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_allgather_preprocess_ext",
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


def _padded_row_col(rest_elems: int) -> tuple[int, int, int]:
    """Return (R, C, padded_rest) properly matching TK TMA tile semantics."""
    num_tiles = (rest_elems + TILE_ELEMS - 1) // TILE_ELEMS
    r, c = ROW_TILE, COL_TILE * num_tiles
    padded = r * c
    return r, c, padded


def solution(
    expert_mask: torch.Tensor,
    num_experts: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
    """
    Compute splits and routing token totals using overlapped TMA kernels
    for communication instead of NCCL blocks.
    """
    group = group or dist.group.WORLD
    ep_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    num_local_experts = num_experts // ep_size

    assert ep_size == NUM_DEVICES, (
        f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; "
        f"got world_size={ep_size}"
    )

    ext = _ensure_ext_jit()

    # 1. Local reductions (hot math)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))

    # 2. Async setup for input splits. We compute on GPU and hold the tensor,
    # deliberately delaying .tolist() which forces CPU blocks, so the GPU can overlap.
    input_splits_tensor = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1)

    # 3. Setup TK Parallel Tensors (Symmetric memory buffers for peer-to-peer copies)
    r, c, padded_rest = _padded_row_col(num_experts)
    input_tk = get_or_create_parallel_tensor(ext, (1, 1, r, c), torch.bfloat16, multicast=False)
    output_tk = get_or_create_parallel_tensor(ext, (1, ep_size, r, c), torch.bfloat16, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=ep_size)

    # Push to symmetric allocation in optimized bfloat16
    padded = torch.zeros(padded_rest, dtype=torch.bfloat16, device=expert_mask.device)
    padded[:num_experts] = num_local_tokens_per_expert.to(torch.bfloat16)
    input_tk.data_.view(-1)[:padded_rest].copy_(padded)

    # 4. Asynchronous peer-to-peer gathering using TMA natively on device
    ext.tk_all_gather(output_tk, input_tk, barrier_tk)

    # 5. Extract flat counts and recover original datatype
    out_flat = output_tk.data_.view(ep_size, padded_rest)[:, :num_experts].contiguous()
    num_global_tokens_per_expert = out_flat.to(num_local_tokens_per_expert.dtype)

    # 6. Extract bounds for local experts handling
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    output_splits_tensor = num_global_tokens_per_local_expert.sum(dim=1)

    # Launch non-blocking CPU transfers asynchronously
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0).to(
        "cpu", non_blocking=True
    )
    num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert.view(-1, num_local_experts).to(
        "cpu", non_blocking=True
    )

    # 7. Drain and sync everything exactly at the end. At this point the GPU TK TMA kernel
    # and local math likely already finished underneath the Python CPU execution time.
    input_splits = input_splits_tensor.tolist()
    output_splits = output_splits_tensor.tolist()

    return input_splits, output_splits, num_global_tokens_per_local_expert_cpu, num_global_sum_tokens_per_local_expert