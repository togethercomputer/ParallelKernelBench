"""
ThunderKittens Grad Acc Loss Integration

Replaces the CPU-syncing `.item()` check with a device-side mask and utilizes
a high-speed ThunderKittens PGL multicast all-reduce for the scalar aggregation.
"""

import os
import torch
import torch.distributed as dist
from typing import Tuple, Optional

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source: ThunderKittens All-Reduce (SUM)
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
            "tk_allreduce_ext",
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


def _tk_all_reduce_sum(tensor: torch.Tensor, ext) -> torch.Tensor:
    """Wrapper to map PyTorch arbitrary tensors through the fixed TK buffer alignment."""
    world = dist.get_world_size()
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    flat = tensor.to(torch.bfloat16).reshape(-1).contiguous()
    n = flat.numel()

    # Pad to kernel alignment (NUM_DEVICES * NUM_ELEMS_PER_BLOCK)
    padded = ((n + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    # Cached TKParallelTensor (VMM + multicast)
    tensor_tk = get_or_create_parallel_tensor(ext, (padded,), torch.bfloat16, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    # Copy input into the VMM-allocated parallel tensor
    tensor_tk.data_[:n] = flat
    if padded > n:
        tensor_tk.data_[n:].zero_()

    # Run the TK all-reduce (barrier → reduce → barrier)
    ext.tk_all_reduce(tensor_tk, barrier_tk)

    result = tensor_tk.data_[:n].clone()
    return result.to(original_dtype).reshape(original_shape)


@torch.no_grad()
def solution(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    grad_normalized_loss: torch.Tensor,
    grad_loss_sum: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates explicitly configured forward/backward metrics utilizing 
    device-side masks and TK multicaset kernels for minimum synchronization.
    """
    ext = _ensure_ext_jit()

    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------
    
    # Eliminate host-device synchronization by replacing .item() with device logic
    loss_clean = torch.nan_to_num(loss)
    loss_val = torch.where(local_valid_tokens == 0, loss_clean, loss)

    loss_sum = loss_val * local_valid_tokens

    # TK All-Reduce Sum Collective replacing NCCL dist.all_reduce
    loss_sum = _tk_all_reduce_sum(loss_sum, ext)

    normalized_loss = loss_sum / global_valid_tokens

    # ---------------------------------------------------------
    # Backward Pass
    # ---------------------------------------------------------
    grad_from_normalized = grad_normalized_loss * local_valid_tokens / global_valid_tokens

    if grad_loss_sum is not None:
        grad_from_sum = grad_loss_sum * local_valid_tokens
    else:
        grad_from_sum = torch.zeros_like(grad_normalized_loss)

    grad_loss = grad_from_normalized + grad_from_sum

    return normalized_loss, loss_sum, grad_loss