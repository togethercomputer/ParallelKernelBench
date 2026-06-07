"""
Strategy:
- Algorithmic Reduction: We eliminate the redundant global Matmuls and `reduce_scatter` by transforming the initial All-Gather into an All-to-All. Rank `r` only requests the `M_local` rows it needs for its sequence-parallel compute block, drastically reducing communication volume by 8x and compute by 8x.
- Custom TK All-to-All TMA: Data movement uses a custom ThunderKittens TMA kernel directly scattering input blocks into the destination's correctly-strided layout `[M_local, H]`, leveraging device-side UVA P2P without intermediate buffers or opaque PyTorch collectives.
- Compute-Communication Overlap: The `M_local` rows are split into chunks. A parallel CUDA stream runs the TK collective for the next chunk concurrently with PyTorch Tensor Core Matmuls and SiLU compute on the current chunk, completely hiding the reduced communication latency.
"""

import os
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
# Embedded .cu source for All-to-All Gather via TMA
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace tk_all_to_all_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int TILE_M = 16;
    static constexpr int TILE_H = 128;

    using shared_tile = st_bf<TILE_M, TILE_H>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    int m_offset_blocks;
    int m_size_blocks;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(m_size_blocks * (input.cols() / globals::TILE_H) * NUM_DEVICES);
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
    int dst_dev_idx = task_idx / (G.m_size_blocks * (G.input.cols() / globals::TILE_H));
    task_idx %= (G.m_size_blocks * (G.input.cols() / globals::TILE_H));
    int row_block_idx = G.m_offset_blocks + task_idx / (G.input.cols() / globals::TILE_H);
    int col_block_idx = task_idx % (G.input.cols() / globals::TILE_H);

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    
    // input shape: [W, 1, M_pad, H_pad]
    tma::load_async(tile, G.input[G.dev_idx], {dst_dev_idx, 0, row_block_idx, col_block_idx}, arrived);
    
    wait(arrived, 0);

    // output shape: [1, 1, M_pad, W * H_pad]
    int out_col_block_idx = G.dev_idx * (G.input.cols() / globals::TILE_H) + col_block_idx;
    
    tma::store_async(G.output[dst_dev_idx], tile, {0, 0, row_block_idx, out_col_block_idx});
}

} // namespace tk_all_to_all_gather

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
    int m_offset_blocks,
    int m_size_blocks
) {
    TORCH_CHECK(m_size_blocks > 0, "m_size_blocks must be positive");
    kittens::py::parallel_tensor_check(output, input);

    tk_all_to_all_gather::globals all_to_all_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename tk_all_to_all_gather::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename tk_all_to_all_gather::globals::parallel_layout>(input),
        .m_offset_blocks = m_offset_blocks,
        .m_size_blocks = m_size_blocks,
        .dev_idx = input.local_rank_
    };

    all_to_all_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_to_all_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<tk_all_to_all_gather::config, tk_all_to_all_gather::globals, tk_all_to_all_gather::kernel>(all_to_all_G);
    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_to_all_gather", &entrypoint);
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
            "tk_alltoallgather_ext",
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
def solution(
    x_local: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    world_size = dist.get_world_size()
    assert world_size == NUM_DEVICES, f"This ThunderKittens kernel expects NUM_DEVICES={NUM_DEVICES}"
    
    ext = _ensure_ext_jit()

    M, H_local = x_local.shape
    H, F_dim = W1.shape
    
    M_local = M // world_size
    
    original_dtype = x_local.dtype
    x_local = x_local.to(torch.bfloat16).contiguous()
    W1 = W1.to(torch.bfloat16)
    W2 = W2.to(torch.bfloat16)

    y_local_full = torch.empty((M_local, H), dtype=torch.bfloat16, device=x_local.device)
    
    if M_local == 0:
        dist.barrier()
        return y_local_full.to(original_dtype)

    M_pad = ((M_local + 15) // 16) * 16
    H_pad = ((H_local + 127) // 128) * 128
    H_out_pad = world_size * H_pad

    # Shared pre-allocated UVA descriptors
    input_tk = get_or_create_parallel_tensor(
        ext, (world_size, 1, M_pad, H_pad), torch.bfloat16, multicast=False
    )
    output_tk = get_or_create_parallel_tensor(
        ext, (1, 1, M_pad, H_out_pad), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    # Prepare interleaved slices for TMA: rank `i`'s input requires sending sequence chunk `i` to rank `i`
    x_local_view = x_local.view(world_size, M_local, H_local)
    padded_in = torch.zeros(world_size, 1, M_pad, H_pad, dtype=torch.bfloat16, device=x_local.device)
    padded_in[:, 0, :M_local, :H_local] = x_local_view
    
    in_numel = world_size * M_pad * H_pad
    input_tk.data_.reshape(-1)[:in_numel].copy_(padded_in.reshape(-1))

    # Chunk M_local logic for optimal communication overlap
    m_blocks = M_pad // 16
    if m_blocks >= 2:
        blocks_1 = m_blocks // 2
        blocks_2 = m_blocks - blocks_1
        chunks = [(0, blocks_1), (blocks_1, blocks_2)]
    else:
        chunks = [(0, m_blocks)]

    s_main = torch.cuda.current_stream()
    s_comm = torch.cuda.Stream()

    # Kickoff the first All-to-All block transfer
    offset_0, size_0 = chunks[0]
    with torch.cuda.stream(s_comm):
        ext.tk_all_to_all_gather(output_tk, input_tk, barrier_tk, offset_0, size_0)

    for i in range(len(chunks)):
        offset, size = chunks[i]
        
        # Wait for this chunk's TMA P2P delivery to wrap up
        s_main.wait_stream(s_comm)
        
        # Pre-issue the next chunk's comm immediately
        if i + 1 < len(chunks):
            next_offset, next_size = chunks[i+1]
            with torch.cuda.stream(s_comm):
                ext.tk_all_to_all_gather(output_tk, input_tk, barrier_tk, next_offset, next_size)
                
        valid_start = min(offset * 16, M_local)
        valid_end = min((offset + size) * 16, M_local)
        if valid_start >= valid_end:
            continue
            
        # Extract the gathered valid elements from the TK memory footprint
        out_numel = M_pad * H_out_pad
        out_view = output_tk.data_.reshape(-1)[:out_numel].view(M_pad, world_size, H_pad)
        x_chunk_pad = out_view[valid_start:valid_end, :, :]
        
        # Slicing the tensor removes the internal H padding natively
        x_chunk = x_chunk_pad[:, :, :H_local].contiguous().view(-1, H)
        
        # Reduced compute step: instead of `[M, H] @ W1`, only `[M_local, H] @ W1`
        z_chunk = torch.matmul(x_chunk, W1)
        a_chunk = F.silu(z_chunk)
        block_chunk = torch.matmul(a_chunk, W2)
        
        y_local_full[valid_start:valid_end, :] = block_chunk

    dist.barrier()
    return y_local_full.to(original_dtype)