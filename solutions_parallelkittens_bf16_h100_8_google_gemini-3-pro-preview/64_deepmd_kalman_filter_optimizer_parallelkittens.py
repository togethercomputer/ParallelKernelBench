"""
Optimized DeepMD blockwise local Kalman-filter optimizer update using ThunderKittens.

Uses ThunderKittens PGL and NVSwitch multimem multicast arrays to perform ultra-low
latency device-side All-Reduce for the Kalman gain scalar and All-Gather for the 
updated parameter weights. Replaces dense allocations with in-place rank-1 updates.
"""

import os
from typing import List, Tuple

import torch
import torch.distributed as dist

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source (TK Reduce + TK Gather + Barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

// Common barrier kernel for synchronization
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
}

// Scalar All-Reduce (SUM) using ld_reduce
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
}

// Padded array All-Gather using multimem.st (broadcast)
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
        static constexpr int NUM_ELEMS_PER_INST = 2;
        static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

        using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;
        parallel_layout tensor;
        const int dev_idx;
        const int chunk_size;

        __host__ inline dim3 grid() const {
            return dim3((chunk_size + NUM_ELEMS_PER_BLOCK - 1) / NUM_ELEMS_PER_BLOCK);
        }
    };

    __device__ inline void kernel(const globals &G) {
        const size_t idx = globals::NUM_ELEMS_PER_BLOCK * blockIdx.x + globals::NUM_ELEMS_PER_INST * threadIdx.x;
        if (idx < G.chunk_size) {
            size_t offset = G.dev_idx * G.chunk_size + idx;
            // Load local rank's segment
            bf16_2 tmp = *reinterpret_cast<bf16_2*>(&G.tensor.data_[offset]);
            // Multicast to all ranks
            multimem<bf16_2>::st(reinterpret_cast<bf16_2*>(&G.tensor.mc_ptr[offset]), tmp);
        }
    }
}

void entrypoint_reduce(
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(tensor, barrier);
    TORCH_CHECK(tensor.data_.numel() % (all_reduce::globals::NUM_DEVICES * all_reduce::globals::NUM_ELEMS_PER_BLOCK) == 0,
        "The total number of tensor elements must be divisible by NUM_DEVICES * NUM_ELEMS_PER_BLOCK");

    all_reduce::globals reduce_G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename all_reduce::globals::parallel_layout>(tensor),
        .dev_idx = tensor.local_rank_
    };

    all_reduce_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_reduce_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<all_reduce::config, all_reduce::globals, all_reduce::kernel>(reduce_G);
    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
}

void entrypoint_gather(
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier,
    int chunk_size
) {
    kittens::py::parallel_tensor_check(tensor, barrier);
    TORCH_CHECK(chunk_size % all_gather::globals::NUM_ELEMS_PER_INST == 0, "chunk_size must be even");

    all_gather::globals gather_G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename all_gather::globals::parallel_layout>(tensor),
        .dev_idx = tensor.local_rank_,
        .chunk_size = chunk_size
    };

    all_reduce_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_reduce_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<all_gather::config, all_gather::globals, all_gather::kernel>(gather_G);
    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_reduce", &entrypoint_reduce);
    m.def("tk_all_gather", &entrypoint_gather);
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

# Topology cache to elide all_gather_object during weights reconstruction.
_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_deepmd_ext",
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
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            _get_ext()
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


@torch.no_grad()
def solution(
    H: List[torch.Tensor],
    error: torch.Tensor,
    weights: List[torch.Tensor],
    P: List[torch.Tensor],
    kalman_lambda: float,
    kalman_nue: float = 0.9987,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:

    device = weights[0].device
    dtype = weights[0].dtype
    weights_num = len(weights)

    lam = torch.tensor(kalman_lambda, dtype=dtype, device=device)
    err = error.to(device=device, dtype=dtype)

    # 1. Fast blockwise precomputations (avoids memory allocation where possible).
    K_list = []
    hk_sum = torch.zeros((), dtype=dtype, device=device)
    
    for i in range(weights_num):
        k_i = torch.mm(P[i], H[i])
        K_list.append(k_i)
        # Vector dot is natively fast, avoids full matrix multiplication paths.
        hk_sum += torch.vdot(H[i].squeeze(1), k_i.squeeze(1))

    tmp_local = lam * weights_num + hk_sum

    if dist.is_initialized():
        world_size = dist.get_world_size()
        assert world_size == 8, "This ThunderKittens kernel is built for NUM_DEVICES=8"
        
        ext = _ensure_ext_jit()
        
        # All-Reduce: sum denominator scalar via TK NVSwitch PGL
        ALIGNMENT = 8 * 512
        reduce_tk = get_or_create_parallel_tensor(ext, (ALIGNMENT,), torch.bfloat16, multicast=True)
        barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
        
        reduce_tk.data_[0] = tmp_local.view(-1)[0].to(torch.bfloat16)
        if ALIGNMENT > 1:
            reduce_tk.data_[1:].zero_()
            
        ext.tk_all_reduce(reduce_tk, barrier_tk)
        tmp_global = reduce_tk.data_[0].to(dtype)
    else:
        tmp_global = tmp_local

    A = 1.0 / tmp_global
    A_item = A.item()
    A_err_item = (A * err).item()
    inv_lam_item = (1.0 / lam).item()

    # 2. Local updates: utilize in-place torch.addr_ instead of allocating K @ K.T intermediates.
    for i in range(weights_num):
        K = K_list[i]
        K_vec = K.squeeze(1)
        
        weights[i].add_(K, alpha=A_err_item)
        P[i].addr_(K_vec, K_vec, beta=1.0, alpha=-A_item).mul_(inv_lam_item)

    # 3. Distributed Weights Gathering via Multicast broadcast
    if dist.is_initialized():
        local_shape = [w.shape[0] for w in weights]
        shape_tuple = tuple(local_shape)
        
        # Populate shape cache once per process to bypass repeated all_gather_object.
        if shape_tuple not in _cache:
            shape_list_t = [
                torch.zeros(len(local_shape), dtype=torch.int64, device=device)
                for _ in range(world_size)
            ]
            local_shape_tensor = torch.tensor(local_shape, dtype=torch.int64, device=device)
            dist.all_gather(shape_list_t, local_shape_tensor)
            
            shape_list = [t.tolist() for t in shape_list_t]
            sizes = [sum(s) for s in shape_list]
            max_size = max(sizes)
            
            CHUNK_ALIGN = 512
            padded_chunk = ((max_size + CHUNK_ALIGN - 1) // CHUNK_ALIGN) * CHUNK_ALIGN
            
            gather_tk = get_or_create_parallel_tensor(
                ext, (world_size, padded_chunk), torch.bfloat16, multicast=True
            )
            
            _cache[shape_tuple] = (shape_list, sizes, padded_chunk, gather_tk)
        
        shape_list, sizes, padded_chunk, gather_tk = _cache[shape_tuple]
        
        rank = dist.get_rank()
        local_size = sizes[rank]
        
        flat_weights = torch.cat([w.reshape(-1) for w in weights], dim=0).to(torch.bfloat16)
        
        # Scatter local segment linearly; zero out padding block.
        gather_tk.data_[rank, :local_size] = flat_weights
        if local_size < padded_chunk:
            gather_tk.data_[rank, local_size:].zero_()
            
        ext.tk_all_gather(gather_tk, barrier_tk, padded_chunk)
        
        # Rematerialize split weights list using cached topology shapes.
        result = []
        for r in range(world_size):
            r_size = sizes[r]
            r_data = gather_tk.data_[r, :r_size].to(dtype)
            
            r_shapes = shape_list[r]
            splits = torch.split(r_data, r_shapes)
            for s in splits:
                result.append(s.reshape(-1, 1))
                
        weights = result

    # 4. Decay Kalman factor using scalar promotion logic seamlessly matching original type logic.
    kalman_lambda_next = kalman_nue * lam + 1.0 - kalman_nue

    return weights, P, kalman_lambda_next