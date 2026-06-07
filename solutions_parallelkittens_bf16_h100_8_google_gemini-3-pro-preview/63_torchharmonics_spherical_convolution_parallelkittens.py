"""
Distributed DISCO spherical convolution forward.

Optimized with ParallelKittens / ThunderKittens CUDA integrations and
overlapping reduction schedules.
"""

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
# Embedded .cu source: ThunderKittens Subgroup All-to-All via TMA
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <vector>

using namespace kittens;

namespace all_to_all_subgroup {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = NUM_DEVICES_PLACEHOLDER;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;

    using shared_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    int group_ranks[NUM_DEVICES];
    int my_subgroup_idx;
    int subgroup_size;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / COL_BLOCK_SIZE) *
                    (input.rows() / ROW_BLOCK_SIZE) *
                    input.depth() * subgroup_size);
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
    int col_blocks = G.input.cols() / globals::COL_BLOCK_SIZE;
    int row_blocks = G.input.rows() / globals::ROW_BLOCK_SIZE;
    int depth = G.input.depth();
    
    int scatter_idx = task_idx / (depth * row_blocks * col_blocks);
    task_idx %= (depth * row_blocks * col_blocks);
    
    int depth_idx = task_idx / (row_blocks * col_blocks);
    task_idx %= (row_blocks * col_blocks);
    
    int row_block_idx = task_idx / col_blocks;
    int col_block_idx = task_idx % col_blocks;

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    
    // Load from my physical memory, representing the scatter_idx chunk intended for peer
    tma::load_async(tile, G.input[G.group_ranks[G.my_subgroup_idx]], {scatter_idx, depth_idx, row_block_idx, col_block_idx}, arrived);

    int dst_dev_idx = G.group_ranks[scatter_idx];
    int gather_idx = G.my_subgroup_idx;

    wait(arrived, 0);
    // Write TMA scatter to peer's layout index allocated for me
    tma::store_async(G.output[dst_dev_idx], tile, {gather_idx, depth_idx, row_block_idx, col_block_idx});
}

} // namespace all_to_all_subgroup

namespace all_to_all_barrier {
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};
struct globals {
    static constexpr int NUM_DEVICES = NUM_DEVICES_PLACEHOLDER;
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
    std::vector<int> group_ranks,
    int my_subgroup_idx
) {
    kittens::py::parallel_tensor_check(output, input);

    all_to_all_subgroup::globals G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_to_all_subgroup::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename all_to_all_subgroup::globals::parallel_layout>(input),
        .my_subgroup_idx = my_subgroup_idx,
        .subgroup_size = static_cast<int>(group_ranks.size())
    };
    for(size_t i = 0; i < group_ranks.size(); ++i) {
        G.group_ranks[i] = group_ranks[i];
    }

    all_to_all_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_to_all_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<all_to_all_subgroup::config, all_to_all_subgroup::globals, all_to_all_subgroup::kernel>(G);
    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_subgroup_all_to_all", &entrypoint);
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
        world_size = dist.get_world_size() if dist.is_initialized() else 8
        src = CUDA_SRC.replace("NUM_DEVICES_PLACEHOLDER", str(world_size))
        
        _ext = compile_cuda_extension(
            "tk_disco_s2_ext",
            src,
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


def _tk_all_to_all_subgroup(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    ext,
) -> torch.Tensor:
    """
    Subgroup scatter (dim=0 after permute) and gather (dim=-1) orchestrating TK TMA kernels.
    Handles padding transparently.
    """
    group_ranks = dist.get_process_group_ranks(group)
    N = len(group_ranks)
    my_rank = dist.get_rank()
    my_subgroup_idx = group_ranks.index(my_rank)
    world_size = dist.get_world_size()
    
    B, C, R, C_inner = tensor.shape
    C_chunk = C // N
    
    # Extract to uniform shape [N, Depth, Rows, Cols] mapping
    x = tensor.view(B, N, C_chunk, R, C_inner).permute(1, 0, 2, 3, 4).reshape(N, B * C_chunk, R, C_inner).contiguous()
    
    pad_R = (16 - (R % 16)) % 16
    pad_C = (128 - (C_inner % 128)) % 128
    if pad_R > 0 or pad_C > 0:
        x_pad = torch.nn.functional.pad(x, (0, pad_C, 0, pad_R))
    else:
        x_pad = x
        
    R_pad = R + pad_R
    C_pad = C_inner + pad_C
    
    input_tk = get_or_create_parallel_tensor(ext, (N, B * C_chunk, R_pad, C_pad), torch.bfloat16, multicast=False)
    output_tk = get_or_create_parallel_tensor(ext, (N, B * C_chunk, R_pad, C_pad), torch.bfloat16, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    
    input_tk.data_.view(-1)[:x_pad.numel()].copy_(x_pad.flatten())
    
    ext.tk_subgroup_all_to_all(output_tk, input_tk, barrier_tk, group_ranks, my_subgroup_idx)
    
    y_pad = output_tk.data_.view(N, B * C_chunk, R_pad, C_pad)
    if pad_R > 0 or pad_C > 0:
        y = y_pad[:, :, :R, :C_inner]
    else:
        y = y_pad
        
    # Layout un-permute back from [N, B, C_chunk, R, C_inner] to [B, C_chunk, R, N * C_inner]
    y = y.view(N, B, C_chunk, R, C_inner).permute(1, 2, 3, 0, 4).reshape(B, C_chunk, R, N * C_inner).contiguous()
    return y


@torch.no_grad()
def solution(
    x: torch.Tensor,
    psi: torch.Tensor,
    weight: torch.Tensor,
    groups: int,
    nlon_out: int,
    nlon_in: int,
    azimuth_group: Optional[dist.ProcessGroup] = None,
    polar_group: Optional[dist.ProcessGroup] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    azimuth_group = azimuth_group or dist.group.WORLD
    polar_group = polar_group or dist.group.WORLD
    azimuth_size = dist.get_world_size(group=azimuth_group)
    polar_size = dist.get_world_size(group=polar_group)
    
    ext = _ensure_ext_jit()

    # 1. Forward Transpose (Azimuth). Maps cleanly to the TK TMA scattered logic.
    if azimuth_size > 1:
        x = _tk_all_to_all_subgroup(x.to(torch.bfloat16), azimuth_group, ext)

    # 2 & 3. Ovelapped DISCO Math with Polar Reduce-Scatter Collectives
    batch_size, n_chans, nlat_in, nlon_in_cur = x.shape
    kernel_size, nlat_out, _ = psi.shape
    pscale = nlon_in_cur // nlon_out
    
    B_C = batch_size * n_chans
    x_flat = x.view(B_C, nlat_in, nlon_in_cur).permute(1, 2, 0).to(torch.bfloat16)
    
    nlat_out_local = nlat_out // polar_size if polar_size > 1 else nlat_out
    
    # Pre-allocate full loop space to host asynchronous reduction blocks
    y_local = torch.empty(
        nlon_out, kernel_size, nlat_out_local, B_C,
        device=x.device, dtype=torch.float32
    )
    
    reqs = []
    psi_bf = psi.to(torch.bfloat16)
    
    for pout in range(nlon_out):
        # Implicitly expand memory without copying allocations
        x_exp = x_flat.reshape(1, nlat_in * nlon_in_cur, B_C).expand(kernel_size, -1, -1)
        curr_y = torch.bmm(psi_bf, x_exp)
        
        if pout < nlon_out - 1:
            x_flat = torch.roll(x_flat, -pscale, dims=1)
            
        # Hide the communication: chunk reduce_scatter immediately on computation boundary
        if polar_size > 1:
            curr_y_float = curr_y.float().contiguous()
            curr_y_chunks = list(torch.split(curr_y_float, nlat_out_local, dim=1))
            req = dist.reduce_scatter(
                y_local[pout], curr_y_chunks, group=polar_group, async_op=True
            )
            reqs.append(req)
        else:
            y_local[pout].copy_(curr_y)
            
    if polar_size > 1:
        for req in reqs:
            req.wait()
            
    x = y_local.permute(3, 1, 2, 0).reshape(batch_size, n_chans, kernel_size, nlat_out_local, nlon_out).to(torch.bfloat16)

    # 5. Backward Transpose (Azimuth). Recycles the exact same TK bidirectional layout permutation kernel.
    if azimuth_size > 1:
        K = x.shape[2]
        x_perm = x.view(batch_size, n_chans, K * nlat_out_local, nlon_out).permute(0, 3, 2, 1).contiguous()
        y_perm = _tk_all_to_all_subgroup(x_perm, azimuth_group, ext)
        x = y_perm.permute(0, 3, 2, 1).view(batch_size, azimuth_size * n_chans, K, nlat_out_local, nlon_out // azimuth_size).contiguous()

    # 6. Grouped channel mixing explicit bmm (instead of unoptimizable nested tensor einsums).
    B, C, K, H, W = x.shape
    C_out = weight.shape[0]
    groupsize = C // groups
    C_out_group = C_out // groups
    
    w_bmm = weight.view(groups, C_out_group, groupsize * K).to(torch.bfloat16)
    x_bmm = x.view(B, groups, groupsize * K, H * W).permute(1, 2, 0, 3).reshape(groups, groupsize * K, B * H * W).to(torch.bfloat16)
    
    out = torch.bmm(w_bmm, x_bmm)
    out = out.view(groups, C_out_group, B, H, W).permute(2, 0, 1, 3, 4).reshape(B, C_out, H, W)
    
    # 7. Bias
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1).to(torch.bfloat16)
        
    return out