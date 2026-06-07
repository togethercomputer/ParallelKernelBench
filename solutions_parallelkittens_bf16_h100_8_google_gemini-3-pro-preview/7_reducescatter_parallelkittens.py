"""
ThunderKittens Reduce-Scatter via NVSwitch multimem.

Strategy:
- We replace the NCCL `reduce_scatter_tensor` with a custom ThunderKittens kernel 
  that leverages Hopper NVSwitch `multimem::ld_reduce`.
- Each rank pushes its locally computed chunks to a contiguous symmetric buffer (`TKParallelTensor`).
- A single fused kernel executes a barrier, reads its assigned reduced chunk directly 
  from the multicast NVSwitch pointer (performing implicit hardware-accelerated summation 
  across all ranks), writes the result locally, and hits a final barrier.
- This entirely bypasses host-driven NCCL loops, operating at maximum NVSwitch bandwidth 
  with no extra memory traffic.
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

namespace reduce_scatter {

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

    using input_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;

    input_layout input;
    bf16* output_ptr;
    const int dev_idx;
    const int chunk_size;

    __host__ inline dim3 grid() const {
        return dim3(chunk_size / NUM_ELEMS_PER_BLOCK);
    }
};

__device__ inline void kernel(const globals &G) {
    const size_t out_idx = globals::NUM_ELEMS_PER_BLOCK * blockIdx.x +
                           globals::NUM_ELEMS_PER_INST * threadIdx.x;
    
    // Each device's chunk in the padded layout is at exactly G.chunk_size * G.dev_idx
    const size_t in_idx = G.chunk_size * G.dev_idx + out_idx;

    bf16_2 tmp;
    // Hardware multicast NVSwitch reduction across all ranks at this offset
    multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<bf16_2*>(&G.input.mc_ptr[in_idx]));
    
    // Write reduced data cleanly to local output tensor
    *reinterpret_cast<bf16_2*>(&G.output_ptr[out_idx]) = tmp;
}

} // namespace reduce_scatter

namespace reduce_scatter_barrier {

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

} // namespace reduce_scatter_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(output, input, barrier);

    int chunk_size = output.data_.numel();

    TORCH_CHECK(chunk_size % reduce_scatter::globals::NUM_ELEMS_PER_BLOCK == 0,
        "The number of output tensor elements must be divisible by NUM_ELEMS_PER_BLOCK");
    TORCH_CHECK(input.data_.numel() == chunk_size * reduce_scatter::globals::NUM_DEVICES,
        "Input tensor must be NUM_DEVICES times larger than output tensor");

    reduce_scatter::globals reduce_scatter_G {
        .input = kittens::py::parallel_tensor_to_pgl<typename reduce_scatter::globals::input_layout>(input),
        .output_ptr = reinterpret_cast<kittens::bf16*>(output.data_.data_ptr<at::BFloat16>()),
        .dev_idx = input.local_rank_,
        .chunk_size = chunk_size
    };

    reduce_scatter_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<reduce_scatter_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Sync input readiness -> Reduce & Store locally -> Sync completion
    kittens::py::launch_kernel<reduce_scatter_barrier::config, reduce_scatter_barrier::globals, reduce_scatter_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<reduce_scatter::config, reduce_scatter::globals, reduce_scatter::kernel>(reduce_scatter_G);
    kittens::py::launch_kernel<reduce_scatter_barrier::config, reduce_scatter_barrier::globals, reduce_scatter_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_reduce_scatter", &entrypoint);
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
ALIGNMENT = NUM_ELEMS_PER_BLOCK  # 512


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_reducescatter_ext",
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


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.is_cuda and tensor.is_contiguous()

    world = dist.get_world_size()
    assert world == NUM_DEVICES, f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; got world_size={world}"
    assert tensor.shape[0] % world == 0, f"First dimension ({tensor.shape[0]}) must be divisible by world_size ({world})"

    ext = _ensure_ext_jit()

    original_shape = tensor.shape
    chunk_dim0 = original_shape[0] // world
    out_shape = (chunk_dim0,) + original_shape[1:]
    original_dtype = tensor.dtype

    n = tensor.numel()
    n_chunk = n // world

    # Pad chunk size to kernel alignment bounds (NUM_ELEMS_PER_BLOCK)
    padded_n_chunk = ((n_chunk + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    # Create / Fetch symmetric buffers for multicast group
    input_tk = get_or_create_parallel_tensor(ext, (world, padded_n_chunk), torch.bfloat16, multicast=True)
    output_tk = get_or_create_parallel_tensor(ext, (padded_n_chunk,), torch.bfloat16, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    # Format local data per-chunk and push to symmetric parallel tensor
    flat_chunks = tensor.to(torch.bfloat16).reshape(world, n_chunk)
    if padded_n_chunk > n_chunk:
        padded_input = torch.zeros((world, padded_n_chunk), dtype=torch.bfloat16, device=tensor.device)
        padded_input[:, :n_chunk] = flat_chunks
        input_tk.data_.copy_(padded_input)
    else:
        input_tk.data_.copy_(flat_chunks)

    # Launch kernel sequence to NVSwitch reduction
    ext.tk_reduce_scatter(output_tk, input_tk, barrier_tk)

    # Truncate any alignment padding and restore output shape characteristics
    result = output_tk.data_[:n_chunk].clone()
    return result.to(original_dtype).reshape(out_shape)