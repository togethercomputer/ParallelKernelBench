"""
Standalone L2 clip_grad_norm: FSDP2 path WITHOUT EP.
Optimized with ThunderKittens PGL multicast and custom fused CUDA reductions.
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
# Embedded .cu source for fused computation and TK all-reduce
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

using namespace kittens;

// ============================================================================
// 1. Local Norm Squared Reduction (Fused across tensors)
// ============================================================================

__global__ void local_norm_sq_kernel(const __nv_bfloat16* g, int numel, float* acc) {
    // Accumulate locally into double to ensure complete numerical stability
    double thread_sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float v = __bfloat162float(g[i]);
        thread_sum += (double)v * (double)v;
    }
    
    float sum = static_cast<float>(thread_sum);

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Block reduction
    extern __shared__ float shared[];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();
    
    if (wid == 0) {
        float val = (lane < (blockDim.x / 32)) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            atomicAdd(acc, val);
        }
    }
}

void compute_local_norm_sq(std::vector<at::Tensor> tensors, at::Tensor acc) {
    float* acc_ptr = acc.data_ptr<float>();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Zero out the entire buffer (including trailing TK alignment paddings)
    cudaMemsetAsync(acc_ptr, 0, acc.numel() * sizeof(float), stream);

    for (const auto& t : tensors) {
        if (!t.defined() || t.numel() == 0) continue;
        int numel = t.numel();
        int threads = 256;
        int blocks = std::min((numel + threads - 1) / threads, 1024);
        int shared_mem = (threads / 32) * sizeof(float);
        
        local_norm_sq_kernel<<<blocks, threads, shared_mem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()), numel, acc_ptr
        );
    }
}

// ============================================================================
// 2. ThunderKittens Hopper Float All-Reduce (Multicast)
// ============================================================================

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
    static constexpr int NUM_ELEMS_PER_INST = 1; 
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    using parallel_layout = pgl<gl<float, -1, -1, -1, -1>, NUM_DEVICES, true>;

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

    float tmp;
    multimem<float>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<float*>(&G.tensor.mc_ptr[idx]));
    multimem<float>::st(reinterpret_cast<float*>(&G.tensor.mc_ptr[idx]), tmp);
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

void tk_all_reduce(
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(tensor, barrier);

    TORCH_CHECK(tensor.data_.numel() % (all_reduce::globals::NUM_DEVICES * all_reduce::globals::NUM_ELEMS_PER_BLOCK) == 0,
        "Total tensor elements must be divisible by NUM_DEVICES * NUM_ELEMS_PER_BLOCK");

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

// ============================================================================
// 3. Batched Scale Pass
// ============================================================================

__global__ void scale_tensors_kernel(__nv_bfloat16* g, int numel, float coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float v = __bfloat162float(g[i]);
        g[i] = __float2bfloat16(v * coef);
    }
}

void scale_tensors(std::vector<at::Tensor> tensors, float coef) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    for (const auto& t : tensors) {
        if (!t.defined() || t.numel() == 0) continue;
        int numel = t.numel();
        int threads = 256;
        int blocks = std::min((numel + threads - 1) / threads, 1024);
        scale_tensors_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()), numel, coef
        );
    }
}

// ============================================================================
// PyBind Initialization
// ============================================================================

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("compute_local_norm_sq", &compute_local_norm_sq);
    m.def("tk_all_reduce", &tk_all_reduce);
    m.def("scale_tensors", &scale_tensors);
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

# Hardcoded node size corresponding to standard H100 cluster limits
NUM_DEVICES = 8


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_clip_grad_norm_ext",
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
    """Compile/load extension once; avoid per-call barriers in timed hot path."""
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
    grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    fsdp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    assert float(norm_type) == 2.0, "Only L2 norm is supported by this optimized path"
    
    world = dist.get_world_size(group=fsdp_group) if fsdp_group is not None else dist.get_world_size()
    assert world == NUM_DEVICES, f"This ThunderKittens kernel is compiled for NUM_DEVICES={NUM_DEVICES}; got world_size={world}"
    
    ext = _ensure_ext_jit()
    
    # TK requires the local buffer space to neatly split into thread blocks (ALIGNMENT elements)
    ALIGNMENT = world * 256
    tensor_tk = get_or_create_parallel_tensor(ext, (ALIGNMENT,), torch.float32, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)
    
    valid_tensors = [t for t in grad_tensors if t is not None]
    
    # 1. Pipeline local fused accumulation directly into symmetric tensor slot [0]
    ext.compute_local_norm_sq(valid_tensors, tensor_tk.data_)
    
    # 2. ThunderKittens NVSwitch barrier + load/reduce multicast + barrier
    ext.tk_all_reduce(tensor_tk, barrier_tk)
    
    # 3. Read accumulated scalar norms and globally trigger scalings
    total_norm_sq = tensor_tk.data_[0].clone()
    total_norm = total_norm_sq.sqrt()
    
    if total_norm > max_norm:
        coef = max_norm / total_norm.item()
        ext.scale_tensors(valid_tensors, coef)
        
    return total_norm