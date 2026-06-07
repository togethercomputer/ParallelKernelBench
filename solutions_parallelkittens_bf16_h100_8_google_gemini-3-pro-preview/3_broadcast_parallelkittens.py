"""
Strategy:
- Use ParallelKittens `TKParallelTensor` with NVSwitch multicast to broadcast data to all ranks simultaneously in a single NVLink hop.
- The root rank uses `uint4` (16-byte) write-through stores to the multicast pointer, achieving peak device-side memory bandwidth and directly writing to all destination devices.
- Avoids host round-trips and multiple `torch.distributed` operations by synchronizing via ThunderKittens device-side `barrier_all` before and after the multicast stores, overlapping the physical broadcast across all SMs.
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
# Embedded .cu source (broadcast entrypoint + barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace broadcast {

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
    static constexpr int NUM_ELEMS_PER_INST = 8; // 16 bytes = 8 bf16s
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout tensor;
    bf16* local_ptr;
    const int dev_idx;
    const int src_rank;

    __host__ inline dim3 grid() const {
        return dim3(tensor.numel() / NUM_ELEMS_PER_BLOCK);
    }
};

// Use 16-byte write-through global stores to the multicast address space
__device__ inline void st_uint4(void* ptr, const uint4& val) {
    asm volatile("st.global.wt.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                 : "memory");
}

__device__ inline void kernel(const globals &G) {
    // Only the source rank streams its local data into the NVSwitch multicast pointer
    if (G.dev_idx == G.src_rank) {
        const size_t idx = globals::NUM_ELEMS_PER_BLOCK * blockIdx.x +
                           globals::NUM_ELEMS_PER_INST * threadIdx.x;
        
        // Data is pre-padded so `idx` will always fall within bounds
        uint4 tmp = *reinterpret_cast<uint4*>(&G.local_ptr[idx]);
        st_uint4(&G.tensor.mc_ptr[idx], tmp);
    }
}

} // namespace broadcast

namespace broadcast_barrier {

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

} // namespace broadcast_barrier

void entrypoint(
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier,
    int src_rank
) {
    kittens::py::parallel_tensor_check(tensor, barrier);

    TORCH_CHECK(tensor.data_.numel() % broadcast::globals::NUM_ELEMS_PER_BLOCK == 0,
        "The total number of tensor elements must be divisible by NUM_ELEMS_PER_BLOCK");

    broadcast::globals broadcast_G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename broadcast::globals::parallel_layout>(tensor),
        .local_ptr = reinterpret_cast<bf16*>(tensor.data_.data_ptr<at::BFloat16>()),
        .dev_idx = tensor.local_rank_,
        .src_rank = src_rank
    };

    broadcast_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<broadcast_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<broadcast_barrier::config, broadcast_barrier::globals, broadcast_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<broadcast::config, broadcast::globals, broadcast::kernel>(broadcast_G);
    kittens::py::launch_kernel<broadcast_barrier::config, broadcast_barrier::globals, broadcast_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_broadcast", &entrypoint);
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
NUM_ELEMS_PER_INST = 8  # float4 = 16 bytes = 8 bf16s
NUM_ELEMS_PER_BLOCK = NUM_THREADS * NUM_ELEMS_PER_INST
ALIGNMENT = NUM_ELEMS_PER_BLOCK  # 2048


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_broadcast_ext",
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
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    assert tensor.is_cuda and tensor.is_contiguous()

    world = dist.get_world_size()
    assert world == NUM_DEVICES, f"Expected {NUM_DEVICES} ranks, got {world}."

    ext = _ensure_ext_jit()

    original_shape = tensor.shape
    original_dtype = tensor.dtype

    flat = tensor.to(torch.bfloat16).reshape(-1).contiguous()
    n = flat.numel()

    if n == 0:
        return torch.empty(original_shape, dtype=original_dtype, device=tensor.device)

    # Pad out to allow uniform fast 16-byte accesses across all blocks without bounds checking
    padded = ((n + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    # Request the cached, symmetric VMM-allocated tensor
    tensor_tk = get_or_create_parallel_tensor(ext, (padded,), torch.bfloat16, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    # Copy input into the VMM-allocated parallel tensor, but only on the broadcasting root
    if dist.get_rank() == src:
        tensor_tk.data_[:n] = flat
        if padded > n:
            tensor_tk.data_[n:].zero_()

    # Issue device-side broadcast:
    # 1. barrier_all (syncs local write) -> 2. broadcast to multicasts -> 3. barrier_all (syncs read)
    ext.tk_broadcast(tensor_tk, barrier_tk, src)

    # Harvest the received data
    result = tensor_tk.data_[:n].clone()
    return result.to(original_dtype).reshape(original_shape)