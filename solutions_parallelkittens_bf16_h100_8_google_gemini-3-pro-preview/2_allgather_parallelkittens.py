"""
Strategy:
- **Device-side NVLink Multicast**: By compiling with ThunderKittens' `pgl` layout and fetching the `mc_ptr`, we utilize Hopper's native NVLink multicast to perform all-gather as an O(1) store operation per element.
- **Zero Host Overhead / Custom Scheduling**: We bypass stock NCCL and its host-side dispatch latency. Each GPU just loads its local tensor chunk and fires a 16-byte vectorized cache-bypassing store directly into the multicast fabric.
- **Hardware-accelerated Output Delivery**: The hardware switch broadcasts these writes to the symmetric physical offset in the VMM-allocated result tensor on all peers simultaneously, fully saturating the node's bisection bandwidth without an explicit software tree/ring.
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
# Embedded .cu source (all_gather entrypoint + barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

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
    static constexpr int NUM_ELEMS_PER_INST = 8; // 8 bf16s = 16 bytes = uint4
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    // Output mapped across all ranks with multicast enabled
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout output;
    const bf16* input;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        size_t chunk_size = output.numel() / NUM_DEVICES;
        int blocks = (chunk_size + NUM_ELEMS_PER_BLOCK - 1) / NUM_ELEMS_PER_BLOCK;
        if (blocks == 0) blocks = 1;
        return dim3(blocks);
    }
};

__device__ inline void kernel(const globals &G) {
    const size_t chunk_size = G.output.numel() / globals::NUM_DEVICES;
    const size_t idx = globals::NUM_ELEMS_PER_BLOCK * blockIdx.x + globals::NUM_ELEMS_PER_INST * threadIdx.x;

    if (idx < chunk_size) {
        // Vectorized 16-byte load from local input
        uint4 tmp = *reinterpret_cast<const uint4*>(&G.input[idx]);
        
        // Target physical offset in the assembled tensor
        const size_t out_idx = chunk_size * G.dev_idx + idx;
        
        // Cache-bypassing (.cg) multicast store via inline PTX
        asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" 
            :: "l"(reinterpret_cast<uint4*>(&G.output.mc_ptr[out_idx])), 
               "r"(tmp.x), "r"(tmp.y), "r"(tmp.z), "r"(tmp.w) : "memory");
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
    const torch::Tensor &input,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(output, barrier);
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    all_gather::globals all_gather_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_gather::globals::parallel_layout>(output),
        .input = reinterpret_cast<const bf16*>(input.data_ptr()),
        .dev_idx = output.local_rank_
    };

    all_gather_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_gather_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Synchronization barrier before mapping phase (if overlapping usages exist)
    kittens::py::launch_kernel<all_gather_barrier::config, all_gather_barrier::globals, all_gather_barrier::kernel>(barrier_G);
    
    // Core payload: parallel NVLink multicasts
    kittens::py::launch_kernel<all_gather::config, all_gather::globals, all_gather::kernel>(all_gather_G);
    
    // Trailing barrier ensures completion of network-bound cache bypass stores globally
    kittens::py::launch_kernel<all_gather_barrier::config, all_gather_barrier::globals, all_gather_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

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
NUM_THREADS = 256
NUM_ELEMS_PER_INST = 8 
NUM_ELEMS_PER_BLOCK = NUM_THREADS * NUM_ELEMS_PER_INST  # 2048
ALIGNMENT = NUM_ELEMS_PER_BLOCK


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
    assert world == NUM_DEVICES, (
        f"This ThunderKittens kernel is scaled for NUM_DEVICES={NUM_DEVICES}; "
        f"got world_size={world}"
    )

    ext = _ensure_ext_jit()

    original_shape = tensor.shape
    original_dtype = tensor.dtype

    flat = tensor.to(torch.bfloat16).reshape(-1).contiguous()
    n = flat.numel()

    # Align chunk size to 16-byte bounds safely handled by 2048-elem blocks
    padded_n = ((n + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    if padded_n == 0:
        padded_n = ALIGNMENT

    # VMM allocation cached across calls; maps the symmetric target layout for multicast writes
    output_tk = get_or_create_parallel_tensor(
        ext, (world, padded_n), torch.bfloat16, multicast=True
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    if padded_n > n:
        padded_inp = torch.zeros(padded_n, dtype=torch.bfloat16, device=tensor.device)
        padded_inp[:n] = flat
    else:
        padded_inp = flat

    # Dispatch device kernel. Returns when symmetric multicast is globally visible.
    ext.tk_all_gather(output_tk, padded_inp, barrier_tk)

    # Slice out precisely the logical layout natively on the current device
    out_flat = output_tk.data_.view(world, padded_n)[:, :n].clone()
    
    return out_flat.to(original_dtype).reshape((world,) + original_shape)