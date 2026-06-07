"""
Optimized Ulysses all_to_all_tensor primitive using ThunderKittens TMA.

Strategy:
1. Device-Side Data Movement: Replaces host-driven `torch.tensor_split` + `dist.all_to_all` + `torch.cat`
   with a direct, fused TMA scatter-gather kernel mapping directly to the underlying NVSwitch/NVLink fabric.
2. Overlap Communication and Memory Permutations: By permuting and casting into a flat buffer view, we 
   pack the scattered chunks safely. The TK kernel then asynchronously routes them (`tma::store_async`) 
   to their destinations simultaneously across all 8 GPUs.
3. Eliminate Redundant Collectives: The customized parallel tensor allocation and unified compilation 
   avoids unnecessary Python loop overheads, completing the all-to-all purely through device-side 
   cooperative transfers and barriers.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source (all_to_all entrypoint + barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_to_all {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;
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
        return dim3((input.cols() / globals::COL_BLOCK_SIZE) *
                    (input.rows() / globals::ROW_BLOCK_SIZE) *
                    input.depth() * input.batch());
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) + 1024);
    }
};

template <int SCATTER_AXIS, int GATHER_AXIS>
__device__ inline void kernel(const globals &G) {
    static_assert(0 <= SCATTER_AXIS && SCATTER_AXIS < 4 && 0 <= GATHER_AXIS && GATHER_AXIS < 4,
        "Scatter and gather axes must be 0, 1, 2, or 3");
    static_assert(SCATTER_AXIS != GATHER_AXIS, "Scatter and gather axes must be different");

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::shared_tile &tile = allocator.allocate<globals::shared_tile>();

    int task_idx = blockIdx.x;
    int batch_idx = task_idx / (G.input.depth() * (G.input.rows() / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.input.depth() * (G.input.rows() / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    int depth_idx = task_idx / (G.input.rows() / globals::ROW_BLOCK_SIZE * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.input.rows() / globals::ROW_BLOCK_SIZE * (G.input.cols() / globals::COL_BLOCK_SIZE));
    int row_block_idx = task_idx / (G.input.cols() / globals::COL_BLOCK_SIZE);
    task_idx %= (G.input.cols() / globals::COL_BLOCK_SIZE);
    int col_block_idx = task_idx;

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    tma::load_async(tile, G.input[G.dev_idx], {batch_idx, depth_idx, row_block_idx, col_block_idx}, arrived);

    int dst_dev_idx;

    if constexpr (SCATTER_AXIS == 0) {
        dst_dev_idx = batch_idx / G.output.batch();
        batch_idx %= G.output.batch();
    } else if constexpr (SCATTER_AXIS == 1) {
        dst_dev_idx = depth_idx / G.output.depth();
        depth_idx %= G.output.depth();
    } else if constexpr (SCATTER_AXIS == 2) {
        dst_dev_idx = row_block_idx / (G.output.rows() / globals::ROW_BLOCK_SIZE);
        row_block_idx %= (G.output.rows() / globals::ROW_BLOCK_SIZE);
    } else {
        dst_dev_idx = col_block_idx / (G.output.cols() / globals::COL_BLOCK_SIZE);
        col_block_idx %= (G.output.cols() / globals::COL_BLOCK_SIZE);
    }

    if constexpr (GATHER_AXIS == 0) {
        batch_idx += G.input.batch() * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 1) {
        depth_idx += G.input.depth() * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 2) {
        row_block_idx += (G.input.rows() / globals::ROW_BLOCK_SIZE) * G.dev_idx;
    } else {
        col_block_idx += (G.input.cols() / globals::COL_BLOCK_SIZE) * G.dev_idx;
    }

    wait(arrived, 0);
    tma::store_async(G.output[dst_dev_idx], tile,
        {batch_idx, depth_idx, row_block_idx, col_block_idx});
}

} // namespace all_to_all

namespace all_to_all_barrier {

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

} // namespace all_to_all_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int scatter_axis,
    int gather_axis
) {
    TORCH_CHECK(0 <= scatter_axis && scatter_axis < 4 && 0 <= gather_axis && gather_axis < 4,
        "Scatter and gather axes must be 0, 1, 2, or 3");
    TORCH_CHECK(scatter_axis != gather_axis, "Scatter and gather axes must be different");

    kittens::py::parallel_tensor_check(output, input);

    all_to_all::globals all_to_all_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_to_all::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename all_to_all::globals::parallel_layout>(input),
        .dev_idx = input.local_rank_
    };

    all_to_all_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_to_all_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);

    if (scatter_axis == 0 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<0, 1>>(all_to_all_G);
    else if (scatter_axis == 0 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<0, 2>>(all_to_all_G);
    else if (scatter_axis == 0 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<0, 3>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<1, 0>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<1, 2>>(all_to_all_G);
    else if (scatter_axis == 1 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<1, 3>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<2, 0>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<2, 1>>(all_to_all_G);
    else if (scatter_axis == 2 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<2, 3>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<3, 0>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<3, 1>>(all_to_all_G);
    else if (scatter_axis == 3 && gather_axis == 2)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<3, 2>>(all_to_all_G);
    else
        TORCH_CHECK(false, "Invalid scatter and gather axes");

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_to_all", &entrypoint);
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


def _padded_row_col(rest_elems: int) -> tuple[int, int, int]:
    """Return (R, C, padded_rest) with R=16, C multiple of 128, R*C >= rest_elems."""
    num_tiles = (rest_elems + TILE_ELEMS - 1) // TILE_ELEMS
    r, c = ROW_TILE, COL_TILE * num_tiles
    padded = r * c
    return r, c, padded


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_ulysses_alltoall_ext",
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
def solution(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    w = dist.get_world_size(group)
    if w == 1:
        return x.contiguous()
        
    assert w == NUM_DEVICES, (
        f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; "
        f"got world_size={w}"
    )

    x = x.contiguous()
    original_shape = x.shape
    original_dtype = x.dtype
    
    scatter_dim = scatter_dim % x.dim()
    gather_dim = gather_dim % x.dim()

    if x.numel() == 0:
        out_shape = list(original_shape)
        out_shape[scatter_dim] //= w
        out_shape[gather_dim] *= w
        return torch.empty(out_shape, dtype=original_dtype, device=x.device)

    # 1. Prepare scattered dimensions and chunks
    shape = list(original_shape)
    assert shape[scatter_dim] % w == 0, f"scatter_dim size {shape[scatter_dim]} not divisible by world_size {w}"
    shape[scatter_dim] //= w
    
    new_shape = shape.copy()
    new_shape.insert(scatter_dim, w)
    
    x_view = x.view(new_shape)
    
    perm = list(range(len(new_shape)))
    perm.remove(scatter_dim)
    perm.insert(0, scatter_dim)
    
    # Chunk preparation directly maps identically to `dist.all_to_all_single`
    x_contig = x_view.permute(perm).contiguous()
    chunk_shape = x_contig.shape[1:]
    rest = x_contig[0].numel()
    
    r, c, padded_rest = _padded_row_col(rest)
    
    ext = _ensure_ext_jit()
    
    input_tk = get_or_create_parallel_tensor(
        ext, (w, 1, r, c), torch.bfloat16, multicast=False
    )
    output_tk = get_or_create_parallel_tensor(
        ext, (1, w, r, c), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=w)
    
    # Explicit single memory transfer formatting to padded chunks mapped via IPC / NVSwitch
    input_tk.data_.reshape(w, padded_rest)[:, :rest].copy_(x_contig.reshape(w, rest))
    
    # Launch ThunderKittens TMA All-To-All mapping `batch_idx -> rank`, `depth_idx -> source`
    ext.tk_all_to_all(output_tk, input_tk, barrier_tk, 0, 1)
    
    # Gather chunks directly from target buffer slicing away padding safely
    out_flat = output_tk.data_.reshape(w, padded_rest)[:, :rest]
    
    # Form fully contiguous output allocating to original memory precision concurrently
    out_contig = torch.empty((w, *chunk_shape), dtype=original_dtype, device=x.device)
    out_contig.reshape(w, rest).copy_(out_flat)
    
    # 2. Post-process received chunks via gathering axes semantics
    out_view = out_contig.view(w, *shape)
    
    perm2 = list(range(1, len(shape) + 1))
    perm2.insert(gather_dim, 0)
    
    out_perm = out_view.permute(perm2).contiguous()
    
    final_shape = shape.copy()
    final_shape[gather_dim] *= w
    
    return out_perm.view(final_shape)