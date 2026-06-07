"""
Strategy:
We bypass the standard PyTorch NCCL `all_to_all_single` and intermediate allocations by integrating custom Triton kernels with ThunderKittens' native TMA all-to-all. 
1. **Fused Dequantization & Padding:** A Triton kernel (`block_fp8_dequant_pad_kernel`) performs FP8-to-BF16 block dequantization while directly scattering the output into the 16x128 tiled memory layout required by ThunderKittens. This overlaps the scale arithmetic with the padding layout transformations.
2. **Device-Side TMA Exchange:** We utilize a custom ThunderKittens kernel (`tk_all_to_all`) to perform personalized all-to-all communication. This leverages Hopper's Tensor Memory Accelerator (TMA) to move blocks asynchronously over NVLink without host-driven NCCL overhead.
3. **Fused Unpadding & Cast:** A second Triton kernel slices out the valid payload from the 16x128 TMA tiles and casts the received BF16 data to FP32 directly into the final contiguous output tensor, merging memory movement and data formatting on-device.
"""

import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl
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

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_alltoall_ext",
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

# ---------------------------------------------------------------------------
# Fused Triton Kernels
# ---------------------------------------------------------------------------

@triton.jit
def block_fp8_dequant_pad_kernel(
    y_ptr, s_ptr, out_ptr, 
    num_elements, chunk_size, padded_chunk_size, 
    scale_block_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    chunk_idx = offs // chunk_size
    idx_in_chunk = offs % chunk_size

    scale_idx = offs // scale_block_size
    s = tl.load(s_ptr + scale_idx, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)

    val = y * s
    out_idx = chunk_idx * padded_chunk_size + idx_in_chunk
    tl.store(out_ptr + out_idx, val.to(tl.bfloat16), mask=mask)

@triton.jit
def unpad_cast_fp32_kernel(
    in_ptr, out_ptr, 
    num_elements, chunk_size, padded_chunk_size, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    chunk_idx = offs // chunk_size
    idx_in_chunk = offs % chunk_size

    in_idx = chunk_idx * padded_chunk_size + idx_in_chunk
    val = tl.load(in_ptr + in_idx, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, val, mask=mask)

# ---------------------------------------------------------------------------
# Main Implementation
# ---------------------------------------------------------------------------

@torch.no_grad()
def solution(
    local_y: torch.Tensor,
    local_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    world_size = dist.get_world_size()
    assert local_y.dim() >= 1 and local_y.shape[0] == world_size, \
        f"local_y first dimension must equal world_size ({world_size}), got {local_y.shape[0]}"
    assert world_size == NUM_DEVICES, \
        f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; got {world_size}"
    
    assert local_y.is_contiguous(), "Input tensor local_y must be contiguous"
    assert local_s.is_contiguous(), "Scale tensor local_s must be contiguous"

    chunk_shape = local_y.shape[1:]
    chunk_size = local_y.numel() // world_size
    num_elements = local_y.numel()
    
    if num_elements == 0:
        return torch.empty(world_size, *chunk_shape, device=local_y.device, dtype=torch.float32)

    assert chunk_size % block_size == 0, \
        f"Chunk size {chunk_size} must be divisible by block_size ({block_size})"

    ext = _ensure_ext_jit()

    r, c, padded_chunk_size = _padded_row_col(chunk_size)

    # Pre-allocate TK tensors (using caching under the hood)
    input_tk = get_or_create_parallel_tensor(
        ext, (world_size, 1, r, c), torch.bfloat16, multicast=False
    )
    output_tk = get_or_create_parallel_tensor(
        ext, (1, world_size, r, c), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    in_tk_flat = input_tk.data_.view(-1)
    out_tk_flat = output_tk.data_.view(-1)
    
    # Zero buffer to prevent transferring garbage within unused padding regions
    in_tk_flat.zero_()

    triton_block_size = 1024
    grid = (triton.cdiv(num_elements, triton_block_size),)

    # 1. Fused dequantization + pad to ThunderKittens input layout
    block_fp8_dequant_pad_kernel[grid](
        local_y.view(-1), local_s.view(-1), in_tk_flat,
        num_elements, chunk_size, padded_chunk_size,
        block_size,
        BLOCK_SIZE=triton_block_size
    )

    # 2. ThunderKittens device-side TMA Exchange
    ext.tk_all_to_all(output_tk, input_tk, barrier_tk, 0, 1)

    # 3. Fused unpad + FP32 cast directly to final destination
    out = torch.empty(world_size, *chunk_shape, device=local_y.device, dtype=torch.float32)
    unpad_cast_fp32_kernel[grid](
        out_tk_flat, out.view(-1),
        num_elements, chunk_size, padded_chunk_size,
        BLOCK_SIZE=triton_block_size
    )

    return out