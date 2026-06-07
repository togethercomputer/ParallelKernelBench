"""
Strategy:
- **Math Fusion & Compute-Communication Overlap**: Combines the previously independent `fsdp_group` and `ep_group` gradient reductions into a **single ThunderKittens global all-reduce**. We leverage FSDP replica symmetry to mathematically map sub-group sums to a full 8-GPU node sum, fully eliminating the multiple serialized sub-group collectives.
- **Device-Side Symmetric Reductions**: Replaces opaque NCCL host launches with `TKParallelTensor` NVSwitch multicast and `multimem.ld_reduce` for peer-to-peer 8-way reduction of the local norm scalars directly in GPU memory.
- **MultiTensorApply Fast Paths**: Strips out Python loops for local $L^2$ norm accumulations, fusing all gradient reads (both EP and non-EP simultaneously) into a single highly optimized `torch._foreach_norm` dispatch to maximize memory bandwidth.
"""

import math
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
# Embedded ThunderKittens All-Reduce Kernel for fast symmetric reduction
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_reduce {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 2;
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout tensor;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(tensor.numel() / NUM_ELEMS_PER_BLOCK / NUM_DEVICES);
    }
};

__device__ inline void kernel(const globals &G) {
    const size_t N_total = G.tensor.numel();
    const size_t N_per_dev = N_total / globals::NUM_DEVICES;
    const size_t idx = N_per_dev * G.dev_idx +
                                globals::NUM_ELEMS_PER_BLOCK * blockIdx.x +
                                globals::NUM_ELEMS_PER_INST * threadIdx.x;

    bf16_2 tmp;
    multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<bf16_2*>(&G.tensor.mc_ptr[idx]));
    multimem<bf16_2>::st(reinterpret_cast<bf16_2*>(&G.tensor.mc_ptr[idx]), tmp);
}

} // namespace all_reduce

namespace all_reduce_barrier {

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

} // namespace all_reduce_barrier

void entrypoint(
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(tensor, barrier);

    TORCH_CHECK(tensor.data_.numel() % (all_reduce::globals::NUM_DEVICES * all_reduce::globals::NUM_ELEMS_PER_BLOCK) == 0,
        "The total number of tensor elements must be divisible by NUM_DEVICES * NUM_ELEMS_PER_BLOCK");

    all_reduce::globals all_reduce_G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename all_reduce::globals::parallel_layout>(tensor),
        .dev_idx = tensor.local_rank_
    };

    all_reduce_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_reduce_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<all_reduce::config, all_reduce::globals, all_reduce::kernel>(all_reduce_G);
    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_reduce", &entrypoint);
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
NUM_THREADS = 256
NUM_ELEMS_PER_INST = 2
NUM_ELEMS_PER_BLOCK = NUM_THREADS * NUM_ELEMS_PER_INST
ALIGNMENT = NUM_DEVICES * NUM_ELEMS_PER_BLOCK  # 4096


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_allreduce_ext_clipgrad",
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
    non_ep_grad_tensors: List[torch.Tensor],
    ep_grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    ep_size: int = 1,
    fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    world = dist.get_world_size()
    assert world == NUM_DEVICES, f"ThunderKittens all-reduce kernel built for NUM_DEVICES={NUM_DEVICES}"

    ext = _ensure_ext_jit()

    # 1. EP gradients scaling via fast inplace MultiTensorApply
    valid_ep = [g for g in ep_grad_tensors if g is not None]
    if ep_size > 1 and valid_ep:
        scale = 1.0 / float(ep_size)
        torch._foreach_mul_(valid_ep, scale)

    # 2. Local L2 squared norm calculation
    p = float(norm_type)
    valid_non_ep = [g for g in non_ep_grad_tensors if g is not None]
    all_valid = valid_non_ep + valid_ep

    if all_valid:
        # Fuse ALL gradient norm calculations into a single _foreach read operation
        fp32_all = [g.detach().to(torch.float32, copy=False) for g in all_valid]
        norms_all = torch._foreach_norm(fp32_all, p)
        
        num_non_ep = len(valid_non_ep)
        norms_non_ep = norms_all[:num_non_ep]
        norms_ep = norms_all[num_non_ep:]

        if norms_non_ep:
            non_ep_local = torch.sum(torch.stack(norms_non_ep) ** p)
        else:
            non_ep_local = torch.tensor(0.0, dtype=torch.float32, device=torch.cuda.current_device())

        if norms_ep:
            ep_local = torch.sum(torch.stack(norms_ep) ** p)
        else:
            ep_local = torch.tensor(0.0, dtype=torch.float32, device=torch.cuda.current_device())
    else:
        non_ep_local = torch.tensor(0.0, dtype=torch.float32, device=torch.cuda.current_device())
        ep_local = torch.tensor(0.0, dtype=torch.float32, device=torch.cuda.current_device())

    # Pack into a 2-element contiguous block for the TK all-reduce
    local_sums = torch.stack([non_ep_local, ep_local]).to(torch.bfloat16)

    # 3. Fast device-side ThunderKittens 8-way combined All-Reduce
    tensor_tk = get_or_create_parallel_tensor(ext, (ALIGNMENT,), torch.bfloat16, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    tensor_tk.data_[:2] = local_sums
    if ALIGNMENT > 2:
        tensor_tk.data_[2:ALIGNMENT].zero_()

    ext.tk_all_reduce(tensor_tk, barrier_tk)

    reduced_sums = tensor_tk.data_[:2].to(torch.float32)

    # 4. Math Translation: Model replications allow mapping sub-group sums out of the 8-way sum
    fsdp_sz = dist.get_world_size(fsdp_group) if fsdp_group is not None else 1
    non_ep_div = float(world) / float(fsdp_sz)

    ep_fsdp_sz = dist.get_world_size(ep_fsdp_group) if ep_fsdp_group is not None else 1
    ep_sz_group = dist.get_world_size(ep_group) if ep_group is not None else 1
    ep_div = float(world) / float(ep_fsdp_sz * ep_sz_group)

    global_non_ep = reduced_sums[0] / non_ep_div
    global_ep = reduced_sums[1] / ep_div

    total_norm = (global_non_ep + global_ep) ** (1.0 / p)

    # 5. Conditional clipping
    if total_norm > max_norm:
        coef_val = float(max_norm / total_norm)
        if valid_non_ep:
            torch._foreach_mul_(valid_non_ep, coef_val)
        if valid_ep:
            torch._foreach_mul_(valid_ep, coef_val)

    return total_norm