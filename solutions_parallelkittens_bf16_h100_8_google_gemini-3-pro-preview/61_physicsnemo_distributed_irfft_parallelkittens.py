"""
Strategy:
1. Communication Volume Reduction: Modified `_conj_pad_2d` to extract, all-gather, flip, and scatter *only* the padded region, slicing the required communication bandwidth in half compared to the reference's full-tensor NCCL collective.
2. Device-side Comm Patterns: Swapped `torch.distributed` collectives (`all_to_all` and `all_gather`) for custom ThunderKittens Hopper TMA kernels. These kernels pull/push tiles across W=8 peers directly over NVLink using explicit PGL addressing and symmetric memory, eliminating CPU host overhead.
3. Lossless BF16 Transfer: Since the specified kernels are compiled for `bf16` arrays but the FFT uses `complex64`, we bitcast the data structures via contiguous `torch.view` down to `bfloat16` prior to P2P transport. This attains max NVLink throughput while recovering exact IEEE identical floats on arrival.
"""

import os
from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded ThunderKittens CUDA Source: All-to-all and All-gather via TMA
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

// ============================================================================
// ALL-TO-ALL
// ============================================================================
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


// ============================================================================
// ALL-GATHER
// ============================================================================
namespace all_gather {
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

    parallel_layout output; // [W, 1, R, C]
    parallel_layout input;  // [1, 1, R, C]
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

__device__ inline void kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::shared_tile &tile = allocator.allocate<globals::shared_tile>();

    int task_idx = blockIdx.x;
    int depth_idx = 0; // Fixed because input depth is 1
    int row_block_idx = task_idx / (G.input.cols() / globals::COL_BLOCK_SIZE);
    int col_block_idx = task_idx % (G.input.cols() / globals::COL_BLOCK_SIZE);

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);

    // Pull from every device and store to the batch index matching the source
    for (int src = 0; src < globals::NUM_DEVICES; src++) {
        tma::expect_bytes(arrived, sizeof(tile));
        tma::load_async(tile, G.input[src], {0, depth_idx, row_block_idx, col_block_idx}, arrived);
        wait(arrived, src);

        tma::store_async(G.output[G.dev_idx], tile, {src, depth_idx, row_block_idx, col_block_idx});
        tma::store_commit_group();
        tma::store_async_wait();
    }
}
} // namespace all_gather


// ============================================================================
// BARRIER
// ============================================================================
namespace ext_barrier {
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
} // namespace ext_barrier


void entrypoint_all_to_all(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int scatter_axis,
    int gather_axis
) {
    kittens::py::parallel_tensor_check(output, input);

    all_to_all::globals G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_to_all::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename all_to_all::globals::parallel_layout>(input),
        .dev_idx = input.local_rank_
    };
    ext_barrier::globals b_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<ext_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<ext_barrier::config, ext_barrier::globals, ext_barrier::kernel>(b_G);

    if (scatter_axis == 0 && gather_axis == 1)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<0, 1>>(G);
    else if (scatter_axis == 1 && gather_axis == 0)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<1, 0>>(G);
    else
        TORCH_CHECK(false, "Unsupported axes");

    kittens::py::launch_kernel<ext_barrier::config, ext_barrier::globals, ext_barrier::kernel>(b_G);
}

void entrypoint_all_gather(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(output, input);

    all_gather::globals G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_gather::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename all_gather::globals::parallel_layout>(input),
        .dev_idx = input.local_rank_
    };
    ext_barrier::globals b_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<ext_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<ext_barrier::config, ext_barrier::globals, ext_barrier::kernel>(b_G);
    kittens::py::launch_kernel<all_gather::config, all_gather::globals, all_gather::kernel>(G);
    kittens::py::launch_kernel<ext_barrier::config, ext_barrier::globals, ext_barrier::kernel>(b_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_to_all", &entrypoint_all_to_all);
    m.def("tk_all_gather", &entrypoint_all_gather);
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
            "tk_fft_comms_ext",
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
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


def _padded_row_col(rest_elems: int) -> tuple[int, int, int]:
    num_tiles = (rest_elems + TILE_ELEMS - 1) // TILE_ELEMS
    r, c = ROW_TILE, COL_TILE * num_tiles
    padded = r * c
    return r, c, padded


def tk_all_to_all_call(tensor: torch.Tensor, ext, barrier_tk, scatter_axis=0, gather_axis=1) -> torch.Tensor:
    """Wrapper mapping arbitrary tensors perfectly to TK WxWxRxC bfloat16 TMA layouts via bitcasting."""
    w = dist.get_world_size()
    
    # Exact bitcast to bfloat16 view; avoids losing fp32/complex64 precision required for FFT correctness
    if tensor.is_complex():
        t_bits = torch.view_as_real(tensor).flatten().contiguous().view(torch.bfloat16)
    else:
        t_bits = tensor.flatten().contiguous().view(torch.bfloat16)
        
    rest = t_bits.numel() // w
    r, c, padded_rest = _padded_row_col(rest)
    
    padded = torch.zeros(w, padded_rest, dtype=torch.bfloat16, device=tensor.device)
    padded[:, :rest] = t_bits.view(w, rest)
    inp_4 = padded.view(w, 1, r, c)
    
    input_tk = get_or_create_parallel_tensor(ext, (w, 1, r, c), torch.bfloat16, multicast=False)
    output_tk = get_or_create_parallel_tensor(ext, (1, w, r, c), torch.bfloat16, multicast=False)
    
    input_tk.data_.reshape(-1)[:inp_4.numel()].copy_(inp_4.reshape(-1))
    
    ext.tk_all_to_all(output_tk, input_tk, barrier_tk, scatter_axis, gather_axis)
    
    out_flat = output_tk.data_.reshape(1, w, r, c)[0].reshape(w, padded_rest)[:, :rest].contiguous()
    
    if tensor.is_complex():
        out_f32 = out_flat.view(torch.float32).view(w, *tensor.shape[1:], 2)
        return torch.view_as_complex(out_f32)
    else:
        return out_flat.view(tensor.dtype).view(w, *tensor.shape[1:])


def tk_all_gather_call(tensor: torch.Tensor, ext, barrier_tk) -> torch.Tensor:
    """TMA all-gather gathering 1/W blocks directly over NVLink with bitcasted BF16 layout."""
    w = dist.get_world_size()
    
    if tensor.is_complex():
        t_bits = torch.view_as_real(tensor).flatten().contiguous().view(torch.bfloat16)
    else:
        t_bits = tensor.flatten().contiguous().view(torch.bfloat16)
        
    rest = t_bits.numel()
    r, c, padded_rest = _padded_row_col(rest)
    
    padded = torch.zeros(1, padded_rest, dtype=torch.bfloat16, device=tensor.device)
    padded[0, :rest] = t_bits
    inp_4 = padded.view(1, 1, r, c)
    
    input_tk = get_or_create_parallel_tensor(ext, (1, 1, r, c), torch.bfloat16, multicast=False)
    output_tk = get_or_create_parallel_tensor(ext, (w, 1, r, c), torch.bfloat16, multicast=False)
    
    input_tk.data_.reshape(-1)[:inp_4.numel()].copy_(inp_4.reshape(-1))
    
    ext.tk_all_gather(output_tk, input_tk, barrier_tk)
    
    out_flat = output_tk.data_.reshape(w, padded_rest)[:, :rest].contiguous()
    
    if tensor.is_complex():
        out_f32 = out_flat.view(torch.float32).view(w, *tensor.shape, 2)
        return torch.view_as_complex(out_f32)
    else:
        return out_flat.view(tensor.dtype).view(w, *tensor.shape)


def _pad_zero(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Zero-pad tensor along dim to size."""
    dim = dim % tensor.ndim
    pad = [0] * (2 * (tensor.ndim - dim))
    pad[1] = size - tensor.shape[dim]
    return F.pad(tensor, pad, mode="constant", value=0.0)


def _conj_pad_2d_tk(
    tensor: torch.Tensor,
    pad_dim: int,
    other_dim: int,
    size: int,
    ext,
    barrier_tk,
) -> torch.Tensor:
    """Pad the RFFT half spectrum natively exchanging *only* the padded region via TMA."""
    pad_dim = pad_dim % tensor.ndim
    other_dim = other_dim % tensor.ndim
    orig_size = tensor.shape[pad_dim]

    # 1. Pad zeroes natively
    tensor_pad = _pad_zero(tensor, pad_dim, size)
    
    # 2. Local filling via complex conjugate map for the resident chunk
    lhs_slice = [slice(0, s) for s in tensor.shape]
    lhs_slice[pad_dim] = slice(orig_size, size)
    rhs_slice = [slice(0, s) for s in tensor.shape]
    rhs_slice[pad_dim] = slice(1, size - orig_size + 1)
    tensor_pad[tuple(lhs_slice)] = torch.flip(torch.conj(tensor_pad[tuple(rhs_slice)]), dims=[pad_dim])

    # 3. Only gather the small padded portion (~half communication footprint vs reference)
    local_pad_region = tensor_pad[tuple(lhs_slice)].contiguous()
    gathered_pad = tk_all_gather_call(local_pad_region, ext, barrier_tk)
    full_pad_region = torch.cat(list(gathered_pad), dim=other_dim)
    
    # 4. Flip the full pad chunk dimension symmetrically across ranks
    full_pad_dim_size = full_pad_region.shape[other_dim]
    flip_slice = [slice(0, s) for s in full_pad_region.shape]
    flip_slice[other_dim] = slice(1, full_pad_dim_size)
    full_pad_region[tuple(flip_slice)] = torch.flip(full_pad_region[tuple(flip_slice)], dims=[other_dim])
    
    # 5. Extract my chunk exclusively
    rank = dist.get_rank()
    my_chunk_size = full_pad_dim_size // dist.get_world_size()
    my_pad_slice = [slice(0, s) for s in full_pad_region.shape]
    my_pad_slice[other_dim] = slice(rank * my_chunk_size, (rank + 1) * my_chunk_size)
    
    tensor_pad[tuple(lhs_slice)] = full_pad_region[tuple(my_pad_slice)]
    return tensor_pad


@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Optional[Sequence[int]],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Optimized PhysicsNeMo-style distributed 2D inverse real FFT.
    """
    ext = _ensure_ext_jit()
    world_size = dist.get_world_size(group) if group is not None else dist.get_world_size()
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    dim0, dim1 = int(dim[0]), int(dim[1])
    if s is not None:
        first_dim_size = int(s[0])
        last_dim_size = int(s[1])
    else:
        first_dim_size = int(x.shape[dim0])
        last_dim_size = int(2 * (x.shape[dim1] - 1))

    # 1. Half-bandwidth conjugate rebuild
    x_pad = _conj_pad_2d_tk(x, pad_dim=dim1, other_dim=dim0, size=last_dim_size, ext=ext, barrier_tk=barrier_tk)

    # 2. Transform the replicated second dimension
    x1 = torch.fft.ifft(x_pad, n=last_dim_size, dim=dim1, norm=norm)

    # 3. Fast device-side TMA NVLink Transpose
    chunk_size = x1.shape[dim1] // world_size
    send_chunks = torch.stack(list(torch.split(x1, chunk_size, dim=dim1)), dim=0).contiguous()
    recv_chunks = tk_all_to_all_call(send_chunks, ext, barrier_tk)
    x1_tran = torch.cat(list(recv_chunks), dim=dim0)

    # 4. Final transform mapping correctly reconstructed spatial sharding 
    x2 = torch.fft.ifft(x1_tran, n=first_dim_size, dim=dim0, norm=norm)
    return torch.real(x2).contiguous()