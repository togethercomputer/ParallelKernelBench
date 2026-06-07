"""
Strategy:
- **Device-Side Multicast**: Instead of PyTorch's two-phase all-gather (which bounces metadata through host lists and executes repeated peer copies followed by `torch.cat`), we map a single symmetric TK parallel tensor on device via NVSwitch multicast.
- **Compute-Communication Fused Mapping**: Each rank computes the direct destination offset for its chunk and pushes local data to the global multicast pointer (`mc_ptr`). This fuses the network transmission and the `torch.cat` concatenation into a single broadcast step.
- **Vectorized Stores**: Data movement is heavily vectorized using `uint4` (128-bit) stores when the inner dimensions are aligned, maximizing fabric bandwidth and bypassing L1 cache to saturate the NVLink multicast window.
- **Minimal Host Overhead**: We only sync once on the host to negotiate exact shapes, then use device-side barriers (`barrier_all`) to ensure completion before and after the fabric writes.
"""

import os
import math
import torch
import torch.distributed as dist
from typing import Optional

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source for Ulysses variable all-gather via Multicast
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <algorithm>

using namespace kittens;

namespace ulysses_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout output;
    const bf16* local_input;

    int outer_size;
    int rank_dim;
    int inner_size;
    int total_dim;
    int rank_offset;

    __host__ inline dim3 grid() const {
        size_t numel = (size_t)outer_size * rank_dim * (inner_size % 8 == 0 ? inner_size / 8 : inner_size);
        size_t g = (numel + config::NUM_THREADS - 1) / config::NUM_THREADS;
        return dim3(g > 0 ? std::min((size_t)65536, g) : 1);
    }
};

__device__ inline void kernel_vec8(const globals &G) {
    size_t vec_inner = G.inner_size / 8;
    size_t chunk_size = (size_t)G.rank_dim * vec_inner;
    size_t total_elems = (size_t)G.outer_size * chunk_size;

    using vec_t = uint4; // 16 bytes = 8 bf16
    const vec_t* in_vec = reinterpret_cast<const vec_t*>(G.local_input);
    vec_t* out_vec = reinterpret_cast<vec_t*>(G.output.mc_ptr);

    // Grid-stride loop mapped directly over 128-bit boundary offsets
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_elems; tid += gridDim.x * blockDim.x) {
        size_t o = tid / chunk_size;
        size_t idx_in_chunk = tid % chunk_size;

        size_t in_idx = o * chunk_size + idx_in_chunk;
        size_t out_idx = o * ((size_t)G.total_dim * vec_inner) + ((size_t)G.rank_offset * vec_inner) + idx_in_chunk;

        out_vec[out_idx] = in_vec[in_idx];
    }
}

__device__ inline void kernel_scalar(const globals &G) {
    size_t chunk_size = (size_t)G.rank_dim * G.inner_size;
    size_t total_elems = (size_t)G.outer_size * chunk_size;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_elems; tid += gridDim.x * blockDim.x) {
        size_t o = tid / chunk_size;
        size_t idx_in_chunk = tid % chunk_size;

        size_t in_idx = o * chunk_size + idx_in_chunk;
        size_t out_idx = o * ((size_t)G.total_dim * G.inner_size) + ((size_t)G.rank_offset * G.inner_size) + idx_in_chunk;

        G.output.mc_ptr[out_idx] = G.local_input[in_idx];
    }
}

} // namespace ulysses_gather

namespace gather_barrier {
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
} // namespace gather_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    long long local_input_ptr,
    kittens::py::TKParallelTensor &barrier,
    int outer_size,
    int rank_dim,
    int inner_size,
    int total_dim,
    int rank_offset
) {
    kittens::py::parallel_tensor_check(output, barrier);

    ulysses_gather::globals G {
        .output = kittens::py::parallel_tensor_to_pgl<typename ulysses_gather::globals::parallel_layout>(output),
        .local_input = reinterpret_cast<const bf16*>(local_input_ptr),
        .outer_size = outer_size,
        .rank_dim = rank_dim,
        .inner_size = inner_size,
        .total_dim = total_dim,
        .rank_offset = rank_offset
    };

    gather_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<gather_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Synchronize to ensure safety of symmetric buffers
    kittens::py::launch_kernel<gather_barrier::config, gather_barrier::globals, gather_barrier::kernel>(barrier_G);

    if (rank_dim > 0) {
        if (inner_size % 8 == 0) {
            kittens::py::launch_kernel<ulysses_gather::config, ulysses_gather::globals, ulysses_gather::kernel_vec8>(G);
        } else {
            kittens::py::launch_kernel<ulysses_gather::config, ulysses_gather::globals, ulysses_gather::kernel_scalar>(G);
        }
    }

    // Synchronize to ensure fabric writes have completed before the PyTorch stream continues
    kittens::py::launch_kernel<gather_barrier::config, gather_barrier::globals, gather_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_ulysses_gather", &entrypoint);
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
            "tk_ulysses_gather_ext",
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


def solution(
    x: torch.Tensor,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        return x.contiguous()

    assert world_size == NUM_DEVICES, (
        f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; "
        f"got world_size={world_size}"
    )

    device = x.device
    original_dtype = x.dtype
    
    # Negative dim support and contiguous memory layout
    if gather_dim < 0:
        gather_dim += x.ndim
    x = x.contiguous()

    # 1. Gather sizes to negotiate the symmetric tensor bounds
    #    (Required host sync to compute proper shapes and offsets)
    x_size = torch.tensor([x.shape[gather_dim]], dtype=torch.int64, device=device)
    size_list = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(size_list, x_size, group=group)
    sizes = [s.item() for s in size_list]

    # Calculate flat dimensions equivalent to the block copy
    outer_size = math.prod(x.shape[:gather_dim]) if gather_dim > 0 else 1
    inner_size = math.prod(x.shape[gather_dim+1:]) if gather_dim < x.ndim - 1 else 1
    
    total_dim = sum(sizes)
    rank_dim = sizes[rank]
    rank_offset = sum(sizes[:rank])

    # Pre-calculate intended output shape
    out_shape = list(x.shape)
    out_shape[gather_dim] = total_dim

    ext = _ensure_ext_jit()

    # Cast strictly to kernel execution type
    x_bf16 = x.to(torch.bfloat16)
    
    # 2. ThunderKittens allocation / mapping 
    total_elems = outer_size * total_dim * inner_size
    padded_elems = ((total_elems + 1023) // 1024) * 1024
    if padded_elems == 0:
        padded_elems = 1024

    output_tk = get_or_create_parallel_tensor(
        ext,
        (padded_elems,),
        torch.bfloat16,
        multicast=True
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    # 3. Kernel launch mapping direct local chunk offset into the multicast peer layout
    ext.tk_ulysses_gather(
        output_tk,
        x_bf16.data_ptr(),
        barrier_tk,
        outer_size,
        rank_dim,
        inner_size,
        total_dim,
        rank_offset
    )

    # Slice output precisely based on exact elements written dynamically
    result = output_tk.data_.view(-1)[:total_elems].view(*out_shape).clone()
    
    return result.to(original_dtype)