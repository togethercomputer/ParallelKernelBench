import os
from typing import List, Optional

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source (all_to_all TMA + scatter_add reduction)
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

void tk_all_to_all_entry(
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

__global__ void scatter_add_kernel(
    const __nv_bfloat16* __restrict__ out,
    const int64_t* __restrict__ seed_inverse_ids,
    __nv_bfloat16* __restrict__ grad_input,
    int N_recv,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_recv * H) {
        int row = idx / H;
        int col = idx % H;
        int64_t dst_row = seed_inverse_ids[row];
        // Native Hopper hardware bfloat16 atomic add
        atomicAdd(&grad_input[dst_row * H + col], out[idx]);
    }
}

void tk_scatter_add_entry(
    torch::Tensor out,
    torch::Tensor seed_inverse_ids,
    torch::Tensor grad_input
) {
    int N_recv = out.size(0);
    int H = out.size(1);
    int total_elems = N_recv * H;
    if (total_elems == 0) return;
    
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;
    
    scatter_add_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        seed_inverse_ids.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
        N_recv,
        H
    );
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_to_all", &tk_all_to_all_entry);
    m.def("tk_scatter_add", &tk_scatter_add_entry);
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
            "tk_gnn_all2all_bw_ext",
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


def _shift(chunks: List[torch.Tensor], group: dist.ProcessGroup) -> List[torch.Tensor]:
    cutoff = len(chunks) - dist.get_rank(group)
    return chunks[cutoff:] + chunks[:cutoff]

def _unshift(chunks: List[torch.Tensor], group: dist.ProcessGroup) -> List[torch.Tensor]:
    cutoff = dist.get_rank(group)
    return chunks[cutoff:] + chunks[:cutoff]


@torch.no_grad()
def solution(
    grad_output: torch.Tensor,
    seed_inverse_ids: torch.Tensor,
    seed_size: int,
    counts_sent: List[int],
    counts_received: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    group = group or dist.group.WORLD
    W = dist.get_world_size(group)
    assert W == NUM_DEVICES, f"Expected world size of {NUM_DEVICES} for this extension"
    
    original_dtype = grad_output.dtype
    H = grad_output.shape[1] if grad_output.dim() > 1 else 1
    ext = _ensure_ext_jit()

    # Determine global max to allow padding for ThunderKittens TMA symmetric-size requirements
    local_max_elems = 0
    for c in counts_sent + counts_received:
        local_max_elems = max(local_max_elems, c * H)
        
    local_max_tensor = torch.tensor([local_max_elems], dtype=torch.long, device=grad_output.device)
    dist.all_reduce(local_max_tensor, op=dist.ReduceOp.MAX, group=group)
    global_max_elems = local_max_tensor.item()
    
    # Map padded tensor block to TK TMA compatible sizing parameters
    TILE_ELEMS = 16 * 128
    num_tiles = (global_max_elems + TILE_ELEMS - 1) // TILE_ELEMS
    if num_tiles == 0:
        num_tiles = 1
    r_dim = 16
    c_dim = 128 * num_tiles
    padded_chunk_size = r_dim * c_dim
    
    padded_send = torch.zeros((W, padded_chunk_size), dtype=torch.bfloat16, device=grad_output.device)
    
    inputs = list(torch.split(grad_output, counts_sent))
    shifted_inputs = _shift(inputs, group)
    
    # Copy shifted input fragments into padded send blocks
    for i, inp in enumerate(shifted_inputs):
        n = inp.numel()
        if n > 0:
            padded_send[i, :n] = inp.to(torch.bfloat16).view(-1)
            
    input_tk = get_or_create_parallel_tensor(
        ext, (W, 1, r_dim, c_dim), torch.bfloat16, multicast=False
    )
    output_tk = get_or_create_parallel_tensor(
        ext, (1, W, r_dim, c_dim), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=W)
    
    n_total = W * padded_chunk_size
    input_tk.data_.view(-1)[:n_total].copy_(padded_send.view(-1))
    
    # 1. Device-side asymmetric TMA all-to-all
    ext.tk_all_to_all(output_tk, input_tk, barrier_tk, 0, 1)
    
    padded_recv = output_tk.data_.view(-1)[:n_total].view(W, padded_chunk_size)
    shifted_counts_received = _shift(counts_received, group)
    
    shifted_outputs = []
    for i, count in enumerate(shifted_counts_received):
        n = count * H
        if n > 0:
            shifted_outputs.append(padded_recv[i, :n].view(count, H))
        else:
            shifted_outputs.append(torch.empty((0, H), dtype=torch.bfloat16, device=grad_output.device))
            
    # Unshift back to natural rotation logic of GraphBolt outputs
    outputs = _unshift(shifted_outputs, group)
    if sum(counts_received) > 0:
        flat_out = torch.cat(outputs)
    else:
        flat_out = torch.empty((0, H), dtype=torch.bfloat16, device=grad_output.device)
        
    grad_input = torch.zeros((seed_size, H), dtype=torch.bfloat16, device=grad_output.device)
    
    # 2. Custom fast scatter-add exploiting native bfloat16 atomic reductions (avoids sparse tensor overhead)
    ext.tk_scatter_add(flat_out, seed_inverse_ids, grad_input)
    
    return grad_input.to(original_dtype)