"""
Strategy:
- **Flatten & Fuse:** Instead of three separate `dist.all_reduce` calls, we pack the three LoRA gradient tensors into a single contiguous buffer. This mitigates repeated kernel and collective launch overheads.
- **Device-Side Communication:** We deploy ThunderKittens' `pgl` layout with NVSwitch multicast via symmetric memory (`TKParallelTensor`).
- **In-Network Reduction:** Our custom CUDA kernel exploits Hopper's `multimem::ld_reduce` for a direct, low-latency sum across all 8 devices on the node—drastically shrinking the footprint compared to a stock NCCL hot path.
"""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source for Fused Multimem All-Reduce
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
NUM_THREADS = 256  # NUM_WARPGROUPS(2) * WARPGROUP_WARPS(4) * WARP_THREADS(32)
NUM_ELEMS_PER_INST = 2
NUM_ELEMS_PER_BLOCK = NUM_THREADS * NUM_ELEMS_PER_INST
ALIGNMENT = NUM_DEVICES * NUM_ELEMS_PER_BLOCK  # 4096


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_lora_allreduce_ext",
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
    if dist.is_initialized() and dist.get_rank() == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


@torch.no_grad()
def solution(
    grad_fc1_1_lora_A: torch.Tensor,
    grad_fc1_2_lora_A: torch.Tensor,
    grad_fc2_lora_B: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not dist.is_initialized():
        return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B

    world = dist.get_world_size(group)
    
    # Fallback to pure NCCL if not on a full 8-GPU node arrangement
    if world != NUM_DEVICES:
        group = group or dist.group.WORLD
        dist.all_reduce(grad_fc1_1_lora_A, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(grad_fc1_2_lora_A, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(grad_fc2_lora_B, op=dist.ReduceOp.SUM, group=group)
        return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B

    ext = _ensure_ext_jit()

    # Create 1D views to fuse into a single contiguous TK array
    f1 = grad_fc1_1_lora_A.view(-1)
    f2 = grad_fc1_2_lora_A.view(-1)
    f3 = grad_fc2_lora_B.view(-1)

    n1, n2, n3 = f1.numel(), f2.numel(), f3.numel()
    n = n1 + n2 + n3

    # Pad out to mults of (NUM_DEVICES * block_alignment)
    padded = ((n + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    tensor_tk = get_or_create_parallel_tensor(ext, (padded,), torch.bfloat16, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    # Coalesced scatter into symmetric memory
    tensor_tk.data_[:n1].copy_(f1)
    tensor_tk.data_[n1 : n1 + n2].copy_(f2)
    tensor_tk.data_[n1 + n2 : n].copy_(f3)
    if padded > n:
        tensor_tk.data_[n:].zero_()

    # Single-shot all-reduce kernel (barrier -> multimem reduce/store -> barrier)
    ext.tk_all_reduce(tensor_tk, barrier_tk)

    # In-place gather
    grad_fc1_1_lora_A.copy_(tensor_tk.data_[:n1].view(grad_fc1_1_lora_A.shape))
    grad_fc1_2_lora_A.copy_(tensor_tk.data_[n1 : n1 + n2].view(grad_fc1_2_lora_A.shape))
    grad_fc2_lora_B.copy_(tensor_tk.data_[n1 + n2 : n].view(grad_fc2_lora_B.shape))

    return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B