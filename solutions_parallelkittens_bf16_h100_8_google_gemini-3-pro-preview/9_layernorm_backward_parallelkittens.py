"""
ThunderKittens Fused LayerNorm Backward Aggregation.

Fuses the local token-wise sum over (dY) and (dY * X_hat) with a device-side
all-reduce using ThunderKittens PGL and NVSwitch multimem/multicast operations.
Optimized for H100 (sm_90a) BF16 workflows.

Requires: ThunderKittens headers at $THUNDERKITTENS_ROOT/include.
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
# Embedded .cu source (Compute + Barriers + All-Reduce)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>

using namespace kittens;

namespace fused_ln_bwd {

struct compute_globals {
    const __nv_bfloat16* X_hat;
    const __nv_bfloat16* dY;
    __nv_bfloat16* local_out;
    int B;
    int H;
    int padded_H;
    int B_chunk;
};

__global__ void compute_kernel(const compute_globals G) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int b_start = blockIdx.y * G.B_chunk;
    int b_end = min(b_start + G.B_chunk, G.B);
    
    if (h >= G.H) return;
    
    float d_gamma_sum = 0.0f;
    float d_beta_sum = 0.0f;
    
    // Contiguous load: threads in a warp read consecutive h
    for (int b = b_start; b < b_end; ++b) {
        float x  = __bfloat162float(G.X_hat[b * G.H + h]);
        float dy = __bfloat162float(G.dY[b * G.H + h]);
        d_gamma_sum += x * dy;
        d_beta_sum  += dy;
    }
    
    if (G.B_chunk < G.B) {
        atomicAdd(&G.local_out[h * 2], __float2bfloat16(d_gamma_sum));
        atomicAdd(&G.local_out[h * 2 + 1], __float2bfloat16(d_beta_sum));
    } else {
        G.local_out[h * 2]     = __float2bfloat16(d_gamma_sum);
        G.local_out[h * 2 + 1] = __float2bfloat16(d_beta_sum);
    }
}

} // namespace fused_ln_bwd

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
    torch::Tensor X_hat,
    torch::Tensor dY,
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(tensor, barrier);
    
    int B = X_hat.size(0);
    int H = X_hat.size(1);
    
    TORCH_CHECK(tensor.data_.numel() % (all_reduce::globals::NUM_DEVICES * all_reduce::globals::NUM_ELEMS_PER_BLOCK) == 0,
        "The total number of tensor elements must be divisible by NUM_DEVICES * NUM_ELEMS_PER_BLOCK");

    // 1. Launch compute kernel to reduce tokens -> feature gradients directly into PGL memory
    int threads_per_block = 256;
    int blocks_x = (H + threads_per_block - 1) / threads_per_block;
    
    int grid_y = 16;
    int B_chunk = (B + grid_y - 1) / grid_y;
    if (B_chunk == 0) B_chunk = 1;
    grid_y = (B + B_chunk - 1) / B_chunk;

    dim3 grid(blocks_x, grid_y);
    dim3 block(threads_per_block);

    fused_ln_bwd::compute_globals compute_G {
        .X_hat = reinterpret_cast<const __nv_bfloat16*>(X_hat.data_ptr<at::BFloat16>()),
        .dY = reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr<at::BFloat16>()),
        .local_out = reinterpret_cast<__nv_bfloat16*>(tensor.data_.data_ptr<at::BFloat16>()),
        .B = B,
        .H = H,
        .padded_H = static_cast<int>(tensor.data_.size(0)),
        .B_chunk = B_chunk
    };
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fused_ln_bwd::compute_kernel<<<grid, block, 0, stream>>>(compute_G);

    // 2. Local-sync barrier
    all_reduce_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_reduce_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);

    // 3. HW-accelerated cross-GPU all-reduce
    all_reduce::globals all_reduce_G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename all_reduce::globals::parallel_layout>(tensor),
        .dev_idx = tensor.local_rank_
    };
    kittens::py::launch_kernel<all_reduce::config, all_reduce::globals, all_reduce::kernel>(all_reduce_G);

    // 4. Global-sync barrier
    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_fused_ln_bwd", &entrypoint);
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
NUM_ELEMS_PER_INST = 2
NUM_ELEMS_PER_BLOCK = NUM_THREADS * NUM_ELEMS_PER_INST
ALIGNMENT_ELEMS = NUM_DEVICES * NUM_ELEMS_PER_BLOCK  # 4096 elements
ALIGNMENT_H = ALIGNMENT_ELEMS // 2                   # 2048


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_fused_ln_bwd_ext",
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
def solution(
    X_hat: torch.Tensor,
    dY: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert X_hat.is_cuda and dY.is_cuda, "Inputs must be CUDA tensors"
    assert X_hat.is_contiguous() and dY.is_contiguous(), "Inputs must be contiguous"
    assert X_hat.shape == dY.shape, "X_hat and dY must have the same shape"

    world = dist.get_world_size()
    assert world == NUM_DEVICES, f"Expected NUM_DEVICES={NUM_DEVICES}, got {world}"

    B, H = X_hat.shape
    ext = _ensure_ext_jit()

    original_dtype = X_hat.dtype

    # Hardware target fastpath precision
    X_bf16 = X_hat.to(torch.bfloat16)
    dY_bf16 = dY.to(torch.bfloat16)

    # Output dimensions padded to satisfy the NVSwitch alignment (NUM_DEVICES * 512 elements)
    padded_H = ((H + ALIGNMENT_H - 1) // ALIGNMENT_H) * ALIGNMENT_H

    # Prepare symmetrical device buffers ([padded_H, 2] multiplexes d_gamma and d_beta)
    tensor_tk = get_or_create_parallel_tensor(ext, (padded_H, 2), torch.bfloat16, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    # Erase padding and prev outputs to prep `atomicAdd` aggregation
    tensor_tk.data_.zero_()

    # Invoke fused device pipeline (Compute -> Barrier -> Multimem LD/ST -> Barrier)
    ext.tk_fused_ln_bwd(X_bf16, dY_bf16, tensor_tk, barrier_tk)

    # Pluck original shape out of the padded symmetrical buffer
    out = tensor_tk.data_[:H, :]
    d_gamma = out[:, 0].clone()
    d_beta = out[:, 1].clone()

    return d_gamma.to(original_dtype), d_beta.to(original_dtype)