"""
Strategy:
To eliminate the overhead of stock `torch.distributed.all_gather_into_tensor`, we use a custom ThunderKittens multicast kernel. 
- **Device-Side Multicast**: Instead of moving data through NCCL's ring/tree abstractions, the kernel directly maps a shared parallel tensor across all Hopper GPUs in the node. Each rank reads its local input chunk and uses `st.global.cs` to write directly to the multicast address space.
- **Hardware-Level Broadcast**: The NVLink fabric broadcasts the store to all peer GPUs simultaneously, completing the all-gather in a single memory phase. 
- **Zero Host Overhead**: We cache the VMM parallel tensor and barrier allocations, avoiding repetitive IPC exchanges on the hot path, and we vectorize memory operations (using `int4` for 16-byte chunks) to saturate the memory bus.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source (all_gather entrypoint + barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include <torch/extension.h>
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <torch/csrc/utils/pybind.h>

using namespace kittens;

namespace all_gather {

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
    bf16* output_mc_ptr;
    const bf16* input_ptr;
    size_t chunk_elems;
    int dev_idx;

    __host__ inline dim3 grid() const {
        size_t int4_count = chunk_elems / 8; // 8 bf16s per int4
        int blocks = (int4_count + config::NUM_THREADS - 1) / config::NUM_THREADS;
        // Cap blocks to saturate GPU but allow grid-stride loop
        if (blocks > 1024) blocks = 1024;
        return dim3(blocks > 0 ? blocks : 1);
    }
};

__device__ inline void kernel(const globals &G) {
    const size_t int4_count = G.chunk_elems / 8;
    
    // Grid-stride loop for arbitrary chunk sizes
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < int4_count; 
         idx += gridDim.x * blockDim.x) {
        
        // Read 16 bytes (8 bf16 elements) from local input
        int4 val = reinterpret_cast<const int4*>(G.input_ptr)[idx];
        
        // Write to multicast pointer via streaming store (bypasses L1, hits L2/fabric)
        int4* dst = reinterpret_cast<int4*>(G.output_mc_ptr + G.dev_idx * G.chunk_elems) + idx;
        asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};"
            : : "l"(dst), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
    }
}

} // namespace all_gather

namespace all_gather_barrier {

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

} // namespace all_gather_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    torch::Tensor input,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(output, barrier);
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "Input must be BF16");

    size_t chunk_elems = input.numel();
    TORCH_CHECK(chunk_elems % 8 == 0, "Chunk size must be multiple of 8 for int4 vectorization");
    
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, all_gather::globals::NUM_DEVICES, true>;
    auto output_pgl = kittens::py::parallel_tensor_to_pgl<parallel_layout>(output);
    
    all_gather::globals ag_G {
        .output_mc_ptr = output_pgl.mc_ptr,
        .input_ptr = reinterpret_cast<const bf16*>(input.data_ptr<at::BFloat16>()),
        .chunk_elems = chunk_elems,
        .dev_idx = output.local_rank_
    };

    all_gather_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_gather_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // 1. Sync before writing to avoid stomping on previous iterations
    kittens::py::launch_kernel<all_gather_barrier::config, all_gather_barrier::globals, all_gather_barrier::kernel>(barrier_G);
    
    // 2. Multicast broadcast 
    if (chunk_elems > 0) {
        kittens::py::launch_kernel<all_gather::config, all_gather::globals, all_gather::kernel>(ag_G);
    }
    
    // 3. Sync after writing to ensure all L2s are updated before Python consumes
    kittens::py::launch_kernel<all_gather_barrier::config, all_gather_barrier::globals, all_gather_barrier::kernel>(barrier_G);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_gather", &entrypoint);
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
            "tk_allgather_ext",
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
    """Compile/load extension once; avoid per-call dist.barrier() in timed hot path."""
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
    x: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    
    if world_size == 1:
        return x.contiguous()
        
    assert world_size == NUM_DEVICES, (
        f"This ThunderKittens kernel is built for NUM_DEVICES={NUM_DEVICES}; "
        f"got world_size={world_size}"
    )

    if x.numel() == 0:
        dim_size = list(x.size())
        dim_size[0] = dim_size[0] * world_size
        return torch.empty(dim_size, dtype=x.dtype, device=x.device)

    x = x.contiguous()
    original_dtype = x.dtype
    if original_dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    ext = _ensure_ext_jit()

    dim_size = list(x.size())
    dim_size[0] = dim_size[0] * world_size

    # Flatten the per-rank chunk to process uniformly
    flat = x.view(-1)
    n = flat.numel()

    # Kernel relies on 16-byte (8 bf16s) vectorization
    pad_n = ((n + 7) // 8) * 8
    
    if pad_n > n:
        padded_x = torch.zeros(pad_n, dtype=x.dtype, device=x.device)
        padded_x[:n] = flat
    else:
        padded_x = flat

    # Request/Re-use VMM mapping spanning all devices in group
    output_tk = get_or_create_parallel_tensor(
        ext, (world_size * pad_n,), torch.bfloat16, multicast=True
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    # Perform device-side multicast exchange
    ext.tk_all_gather(output_tk, padded_x, barrier_tk)

    if pad_n > n:
        # Re-pack padded shards before reshaping to strict primitive spec
        out = torch.empty((world_size, pad_n), dtype=x.dtype, device=x.device)
        out.copy_(output_tk.data_[: world_size * pad_n].view(world_size, pad_n))
        out_compact = out[:, :n].contiguous().view(dim_size)
        
        if original_dtype != torch.bfloat16:
            return out_compact.to(original_dtype)
        return out_compact
    else:
        # Directly view contiguous array block
        out = output_tk.data_[: world_size * n].view(dim_size)
        
        if original_dtype != torch.bfloat16:
            return out.to(original_dtype)
        # We must clone because output_tk.data_ is a static pool mapping
        # and may be overwritten on the next call to `solution`.
        return out.clone()