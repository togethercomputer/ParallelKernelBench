import os
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded ThunderKittens Kernel
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
    static constexpr int COL_BLOCK_SIZE = 64;

    using shared_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    const int dev_idx;
    int row_start;
    int num_rows;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / globals::COL_BLOCK_SIZE) *
                    (num_rows / globals::ROW_BLOCK_SIZE) *
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
    int batch_idx = task_idx / (G.input.depth() * (G.num_rows / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.input.depth() * (G.num_rows / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    int depth_idx = task_idx / (G.num_rows / globals::ROW_BLOCK_SIZE * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.num_rows / globals::ROW_BLOCK_SIZE * (G.input.cols() / globals::COL_BLOCK_SIZE));
    int row_block_idx = task_idx / (G.input.cols() / globals::COL_BLOCK_SIZE);
    task_idx %= (G.input.cols() / globals::COL_BLOCK_SIZE);
    int col_block_idx = task_idx;

    // Shift row block for chunking
    row_block_idx += G.row_start / globals::ROW_BLOCK_SIZE;

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    tma::load_async(tile, G.input[G.dev_idx], {batch_idx, depth_idx, row_block_idx, col_block_idx}, arrived);

    int dst_dev_idx;

    if constexpr (SCATTER_AXIS == 2) {
        dst_dev_idx = row_block_idx / (G.output.rows() / globals::ROW_BLOCK_SIZE);
        row_block_idx %= (G.output.rows() / globals::ROW_BLOCK_SIZE);
    } else if constexpr (SCATTER_AXIS == 3) {
        dst_dev_idx = col_block_idx / (G.output.cols() / globals::COL_BLOCK_SIZE);
        col_block_idx %= (G.output.cols() / globals::COL_BLOCK_SIZE);
    } else {
        dst_dev_idx = 0; // Unused
    }

    if constexpr (GATHER_AXIS == 2) {
        row_block_idx += (G.input.rows() / globals::ROW_BLOCK_SIZE) * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 3) {
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
    int gather_axis,
    int row_start,
    int num_rows
) {
    TORCH_CHECK(0 <= scatter_axis && scatter_axis < 4 && 0 <= gather_axis && gather_axis < 4,
        "Scatter and gather axes must be 0, 1, 2, or 3");
    TORCH_CHECK(scatter_axis != gather_axis, "Scatter and gather axes must be different");

    kittens::py::parallel_tensor_check(output, input);

    all_to_all::globals all_to_all_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_to_all::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename all_to_all::globals::parallel_layout>(input),
        .dev_idx = input.local_rank_,
        .row_start = row_start,
        .num_rows = num_rows
    };

    all_to_all_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_to_all_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);

    if (scatter_axis == 2 && gather_axis == 3)
        kittens::py::launch_kernel<all_to_all::config, all_to_all::globals, all_to_all::kernel<2, 3>>(all_to_all_G);
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

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_ulysses_a2a_ext",
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


def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = False,
) -> torch.Tensor:
    # Retained as fallback for single GPU runs
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal and q.size(1) > 1:
        S = scores.size(-1)
        causal_mask = torch.triu(
            torch.ones(S, S, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    num_heads: int = 8,
    causal: bool = False,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    
    if world_size == 1:
        B, S_local, H = hidden_states.shape
        head_dim = H // num_heads
        qkv = F.linear(hidden_states, w_qkv)
        qkv = qkv.view(B, S_local, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)
        scale = head_dim**-0.5
        attn_out = _local_attention(q, k, v, scale, causal=causal)
        out = attn_out.reshape(B, S_local, -1)
        return F.linear(out, w_o)

    B, S_local, H = hidden_states.shape
    head_dim = (w_qkv.shape[0] // 3) // num_heads
    chunk_size = (num_heads // world_size) * head_dim
    
    assert num_heads % world_size == 0, "num_heads must be divisible by world_size"
    assert S_local % 16 == 0, "S_local must be a multiple of 16 for TK alignments"
    assert chunk_size % 64 == 0, "chunk_size must be a multiple of 64 for TK alignments"

    ext = _ensure_ext_jit()

    # Pre-allocate TK buffers out of VMM
    tk_qkv_in = get_or_create_parallel_tensor(ext, (B, 1, S_local * 3, world_size * chunk_size), torch.bfloat16, multicast=False)
    tk_qkv_out = get_or_create_parallel_tensor(ext, (B, 1, S_local * world_size * 3, chunk_size), torch.bfloat16, multicast=False)
    
    tk_out_in = get_or_create_parallel_tensor(ext, (B, 1, S_local * world_size, chunk_size), torch.bfloat16, multicast=False)
    tk_out_out = get_or_create_parallel_tensor(ext, (B, 1, S_local, world_size * chunk_size), torch.bfloat16, multicast=False)
    
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    NUM_CHUNKS = 2
    if S_local % NUM_CHUNKS != 0:
        NUM_CHUNKS = 1
    S_chunk = S_local // NUM_CHUNKS

    s1 = torch.cuda.current_stream()
    s2 = torch.cuda.Stream()
    events_compute = [torch.cuda.Event() for _ in range(NUM_CHUNKS)]
    events_comm = [torch.cuda.Event() for _ in range(NUM_CHUNKS)]

    # Compute QKV efficiently overlapping with first All-to-All via sequence chunks
    w_qkv_t = w_qkv.t()
    tk_qkv_in_view = tk_qkv_in.data_[:B * S_local * 3 * H].view(B, S_local, -1)

    for c in range(NUM_CHUNKS):
        start = c * S_chunk
        end = (c + 1) * S_chunk
        
        # F.linear naturally outputs [B, S_local, 3 * num_heads * head_dim] mapping implicitly to TK coordinates 
        torch.matmul(hidden_states[:, start:end, :], w_qkv_t, out=tk_qkv_in_view[:, start:end, :])
        events_compute[c].record(s1)
        
        s2.wait_event(events_compute[c])
        with torch.cuda.stream(s2):
            ext.tk_all_to_all(tk_qkv_out, tk_qkv_in, barrier_tk, 3, 2, start * 3, S_chunk * 3)
            events_comm[c].record(s2)

    for c in range(NUM_CHUNKS):
        s1.wait_event(events_comm[c])

    # Extract fully gathered Q, K, V elements from parallel buffer. 
    qkv = tk_qkv_out.data_[:B * S_local * world_size * 3 * chunk_size].view(B, S_local * world_size, 3, num_heads // world_size, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    
    # Inline Local Attention 
    scale = head_dim**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal and q.size(1) > 1:
        S = scores.size(-1)
        causal_mask = torch.triu(
            torch.ones(S, S, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    
    # Write attention out to next TK tensor VMM, mapping back to flattened coordinates
    tk_out_in_view = tk_out_in.data_[:B * S_local * world_size * chunk_size].view(B, S_local * world_size, num_heads // world_size, head_dim)
    torch.matmul(attn, v, out=tk_out_in_view)
    
    # Run second All-to-All gathering heads and scattering sequence segments.
    ext.tk_all_to_all(tk_out_out, tk_out_in, barrier_tk, 2, 3, 0, S_local * world_size)
    
    # Perform output projection
    out = tk_out_out.data_[:B * S_local * world_size * chunk_size].view(B, S_local, -1)
    return F.linear(out, w_o)