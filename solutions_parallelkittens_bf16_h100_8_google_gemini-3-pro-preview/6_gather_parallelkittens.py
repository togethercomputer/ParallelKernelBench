"""
ThunderKittens Gather via direct TMA push to destination.

Optimized for 8x H100 (Hopper) connected via NVLink.
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

namespace gather {

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
    int dst_dev_idx;
    int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / globals::COL_BLOCK_SIZE) *
                    (input.rows() / globals::ROW_BLOCK_SIZE));
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
    int row_block_idx = task_idx / (G.input.cols() / globals::COL_BLOCK_SIZE);
    int col_block_idx = task_idx % (G.input.cols() / globals::COL_BLOCK_SIZE);

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    
    // Load local input block
    tma::load_async(tile, G.input[G.dev_idx], {0, 0, row_block_idx, col_block_idx}, arrived);
    wait(arrived, 0);

    // Push block directly to the destination rank
    tma::store_async(G.output[G.dst_dev_idx], tile,
        {G.dev_idx, 0, row_block_idx, col_block_idx});
}

} // namespace gather

namespace gather_barrier {

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

} // namespace gather_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int dst
) {
    TORCH_CHECK(0 <= dst && dst < gather::globals::NUM_DEVICES, "dst rank must be valid");
    kittens::py::parallel_tensor_check(output, input);

    gather::globals gather_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename gather::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename gather::globals::parallel_layout>(input),
        .dst_dev_idx = dst,
        .dev_idx = input.local_rank_
    };

    gather_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<gather_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Synchronize to ensure all ranks are ready to TMA push
    kittens::py::launch_kernel<gather_barrier::config, gather_barrier::globals, gather_barrier::kernel>(barrier_G);
    
    // Execute gather (direct TMA store to dst)
    kittens::py::launch_kernel<gather::config, gather::globals, gather::kernel>(gather_G);
    
    // Synchronize to ensure all TMA stores have completed system-wide before host access
    kittens::py::launch_kernel<gather_barrier::config, gather_barrier::globals, gather_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_gather", &entrypoint);
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
            "tk_gather_ext",
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
def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    assert tensor.is_cuda and tensor.is_contiguous()

    world = dist.get_world_size()
    assert world == NUM_DEVICES, (
        f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; "
        f"got world_size={world}"
    )

    ext = _ensure_ext_jit()

    original_shape = tensor.shape
    original_dtype = tensor.dtype

    flat = tensor.to(torch.bfloat16).reshape(-1).contiguous()
    rest = flat.numel()
    
    # Calculate padding to map data correctly into TMA hardware tiles (16 x 128)
    r, c, padded_rest = _padded_row_col(rest)

    padded = torch.zeros(padded_rest, dtype=torch.bfloat16, device=tensor.device)
    padded[:rest] = flat
    inp_4 = padded.view(1, 1, r, c)

    # Establish PGL VMM layouts (shape symmetric across all NVLink ranks)
    input_tk = get_or_create_parallel_tensor(
        ext, (1, 1, r, c), torch.bfloat16, multicast=False
    )
    output_tk = get_or_create_parallel_tensor(
        ext, (world, 1, r, c), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    # Fill current rank's chunk
    n = inp_4.numel()
    input_tk.data_.reshape(-1)[:n].copy_(inp_4.reshape(-1))

    # Kernel execution: load local chunk, push via TMA to output_tk on rank `dst`
    ext.tk_gather(output_tk, input_tk, barrier_tk, dst)

    rank = dist.get_rank()
    if rank == dst:
        # Destination rank: Extract stacked slices avoiding alignment padding
        out_n = output_tk.data_.reshape(-1)[:world * n]
        out_flat = out_n.view(world, padded_rest)[:, :rest].contiguous()
        return out_flat.reshape(world, *original_shape).to(original_dtype)
    else:
        # Non-destination ranks: return input tensor unchanged
        return tensor