"""
ThunderKittens scatter via pull-based TMA over peer-to-peer memory.

Replaces NCCL scatter with a symmetric PGL kernel. The source rank loads its
full multi-chunk tensor into an IPC-mapped parallel tensor. All 8 GPUs then
execute a pull via TMA load directly from the source rank's memory into their
own local output buffers. Hardware copy engines handle the cross-device data
movement.

Requires: ThunderKittens headers at $THUNDERKITTENS_ROOT/include.
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
# Embedded .cu source (scatter entrypoint + barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace tk_scatter {

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
    int src_rank;
    int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((output.cols() / globals::COL_BLOCK_SIZE) *
                    (output.rows() / globals::ROW_BLOCK_SIZE) *
                    output.depth() * output.batch());
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
    int batch_idx = task_idx / (G.output.depth() * (G.output.rows() / globals::ROW_BLOCK_SIZE) * (G.output.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.output.depth() * (G.output.rows() / globals::ROW_BLOCK_SIZE) * (G.output.cols() / globals::COL_BLOCK_SIZE));
    int depth_idx = task_idx / (G.output.rows() / globals::ROW_BLOCK_SIZE * (G.output.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.output.rows() / globals::ROW_BLOCK_SIZE * (G.output.cols() / globals::COL_BLOCK_SIZE));
    int row_block_idx = task_idx / (G.output.cols() / globals::COL_BLOCK_SIZE);
    task_idx %= (G.output.cols() / globals::COL_BLOCK_SIZE);
    int col_block_idx = task_idx;

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    
    // Each rank pulls its designated chunk from the src_rank
    // The chunk index in the src_rank tensor corresponds to G.dev_idx
    tma::load_async(tile, G.input[G.src_rank], {G.dev_idx, depth_idx, row_block_idx, col_block_idx}, arrived);

    wait(arrived, 0);
    // Write out locally to its own output tensor
    tma::store_async(G.output[G.dev_idx], tile, {batch_idx, depth_idx, row_block_idx, col_block_idx});
}

} // namespace tk_scatter

namespace tk_scatter_barrier {

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

} // namespace tk_scatter_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int src_rank
) {
    kittens::py::parallel_tensor_check(output, input);

    tk_scatter::globals scatter_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename tk_scatter::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename tk_scatter::globals::parallel_layout>(input),
        .src_rank = src_rank,
        .dev_idx = input.local_rank_
    };

    tk_scatter_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<tk_scatter_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Synchronize to ensure src_rank's buffer is populated before pulls start
    kittens::py::launch_kernel<tk_scatter_barrier::config, tk_scatter_barrier::globals, tk_scatter_barrier::kernel>(barrier_G);
    
    kittens::py::launch_kernel<tk_scatter::config, tk_scatter::globals, tk_scatter::kernel>(scatter_G);
    
    // Synchronize to ensure all ranks finish pulling before subsequent calls can overwrite src_rank's buffer
    kittens::py::launch_kernel<tk_scatter_barrier::config, tk_scatter_barrier::globals, tk_scatter_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_scatter", &entrypoint);
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
            "tk_scatter_ext",
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
    """Return (R, C, padded_rest) with R=16, C multiple of 128, R*C >= rest_elems."""
    num_tiles = (rest_elems + TILE_ELEMS - 1) // TILE_ELEMS
    r, c = ROW_TILE, COL_TILE * num_tiles
    padded = r * c
    return r, c, padded


@torch.no_grad()
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    assert tensor.is_cuda and tensor.is_contiguous()

    world = dist.get_world_size()
    assert world == NUM_DEVICES, (
        f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; "
        f"got world_size={world}"
    )

    rank = dist.get_rank()
    ext = _ensure_ext_jit()

    original_dtype = tensor.dtype

    if rank == src:
        assert tensor.shape[0] == world, (
            f"First dimension ({tensor.shape[0]}) must equal world_size ({world})"
        )
        chunk_shape = tensor.shape[1:]
        chunk_elems = tensor[0].numel()
    else:
        chunk_shape = tensor.shape
        chunk_elems = tensor.numel()

    r, c, padded_rest = _padded_row_col(chunk_elems)

    # Input on src has shape [world_size, 1, R, C] to accommodate all chunks
    input_tk = get_or_create_parallel_tensor(
        ext, (world, 1, r, c), torch.bfloat16, multicast=False
    )
    # Output on every rank has shape [1, 1, R, C]
    output_tk = get_or_create_parallel_tensor(
        ext, (1, 1, r, c), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    if rank == src:
        # Format the batched tensor for the TK buffer
        flat = tensor.to(torch.bfloat16).reshape(world, -1).contiguous()
        padded = torch.zeros(world, padded_rest, dtype=torch.bfloat16, device=tensor.device)
        padded[:, :chunk_elems] = flat
        inp_4 = padded.view(world, 1, r, c)
        n = inp_4.numel()
        # Copy to the IPC-mapped parallel tensor layout
        input_tk.data_.reshape(-1)[:n].copy_(inp_4.reshape(-1))

    # All ranks launch the kernel (pull from src_rank's memory)
    ext.tk_scatter(output_tk, input_tk, barrier_tk, src)

    # Extract the resulting chunk logically populated on every rank
    out_flat = output_tk.data_.reshape(-1)[:padded_rest]
    return out_flat[:chunk_elems].contiguous().reshape(chunk_shape).to(original_dtype)