"""
Strategy:
- **Device-side Multicast All-Gather**: Replaces stock `dist.all_gather` and `torch.cat` with a custom ThunderKittens kernel that uses Hopper's `multimem::st`. Each rank broadcasts its `A_local` chunk directly to the correct strided columns of the globally shared `A_global` tensor over NVLink in a single operation.
- **Zero-Copy Assembly**: Eliminates host-driven slice concatenations by natively writing into the strided offsets of the continuous `A_global` buffer.
- **Maximized Compute Efficiency**: Operating at maximum NVLink bandwidth via hardware multicast, we seamlessly deliver a contiguous `A_global` to a single, monolithic `torch.matmul(A_global, B)`, perfectly retaining cuBLAS wave quantization efficiency without the memory bandwidth overhead of chunked accumulations.
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
#include <torch/csrc/utils/pybind.h>

using namespace kittens;

namespace all_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;
    parallel_layout A_global;
    const bf16* local_ptr; 
    
    int M;
    int K_local;
    int K_global;
    int dev_idx;

    __host__ inline dim3 grid() const {
        long long total_vecs = ((long long)M * K_local) / 2;
        int threads = config::NUM_THREADS;
        int blocks = (total_vecs + threads - 1) / threads;
        if (blocks > 108 * 4) blocks = 108 * 4; // Saturate GPU
        if (blocks < 1) blocks = 1;
        return dim3(blocks);
    }
};

__device__ inline void kernel(const globals &G) {
    long long total_vecs = ((long long)G.M * G.K_local) / 2;
    if (total_vecs == 0) return;

    for (long long vec_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         vec_idx < total_vecs; 
         vec_idx += blockDim.x * gridDim.x) {
        
        long long elem_idx = vec_idx * 2;
        long long row = elem_idx / G.K_local;
        long long col = elem_idx % G.K_local;

        // Map correctly to the strided columns of the global tensor
        long long dst_elem_idx = row * G.K_global + (G.dev_idx * G.K_local) + col;

        bf16_2 val = *(reinterpret_cast<const bf16_2*>(&G.local_ptr[elem_idx]));
        multimem<bf16_2>::st(reinterpret_cast<bf16_2*>(&G.A_global.mc_ptr[dst_elem_idx]), val);
    }
}

} // namespace all_gather

namespace barrier_ns {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace barrier_ns

void tk_all_gather(
    kittens::py::TKParallelTensor &A_global_tk,
    torch::Tensor A_local,
    kittens::py::TKParallelTensor &barrier
) {
    int M = A_local.size(0);
    int K_local = A_local.size(1);
    
    TORCH_CHECK(A_local.is_contiguous(), "A_local must be contiguous");
    TORCH_CHECK(K_local % 2 == 0, "K_local must be even for bf16_2 operations");

    all_gather::globals ag_G {
        .A_global = kittens::py::parallel_tensor_to_pgl<typename all_gather::globals::parallel_layout>(A_global_tk),
        .local_ptr = reinterpret_cast<const bf16*>(A_local.data_ptr()),
        .M = M,
        .K_local = K_local,
        .K_global = K_local * all_gather::globals::NUM_DEVICES,
        .dev_idx = A_global_tk.local_rank_
    };

    barrier_ns::globals bar_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<barrier_ns::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Synchronize peers -> Hardware Multicast All-Gather -> Synchronize peers
    kittens::py::launch_kernel<barrier_ns::config, barrier_ns::globals, barrier_ns::kernel>(bar_G);
    kittens::py::launch_kernel<all_gather::config, all_gather::globals, all_gather::kernel>(ag_G);
    kittens::py::launch_kernel<barrier_ns::config, barrier_ns::globals, barrier_ns::kernel>(bar_G);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_gather", &tk_all_gather);
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
        _ext = compile_cuda_extension(
            "tk_allgather_gemm_ext",
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
def solution(A_local: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    world_size = dist.get_world_size()
    M, K_local = A_local.shape
    K_global = world_size * K_local
    
    # cuBLAS / Tensor Cores strongly prefer even alignments
    assert K_local % 2 == 0, "K_local must be even for vectorized operations."
    assert B.shape[0] == K_global, f"B must have K dimension = world_size * K_local"
    
    ext = _ensure_ext_jit()
    
    original_dtype = A_local.dtype
    A_local_bf16 = A_local.to(torch.bfloat16).contiguous()
    B_bf16 = B.to(torch.bfloat16).contiguous()
    
    # Obtain or create cached TKParallelTensor mapping NVSwitch multimem
    A_global_tk = get_or_create_parallel_tensor(
        ext, (M, K_global), torch.bfloat16, multicast=True
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    
    # Lightning fast hardware-driven Multicast All-Gather
    ext.tk_all_gather(A_global_tk, A_local_bf16, barrier_tk)
    
    # Retrieve identically-sized view of the globally mapped tensor
    A_global = A_global_tk.data_.view(M, K_global)
    
    # High-efficiency monolithic GEMM using cuBLAS Tensor Cores
    C = torch.matmul(A_global, B_bf16)
    
    # Match reference boundary logic
    dist.barrier()
    
    return C.to(original_dtype)