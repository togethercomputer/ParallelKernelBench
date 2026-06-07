import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source (Fused SigLIP with Peer TMA + Barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace siglip {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 4;
    static constexpr int NUM_WARPGROUPS = 2; // 8 warps = 256 threads
    static constexpr int NUM_THREADS = NUM_WARPGROUPS * WARPGROUP_WARPS * WARP_THREADS; 
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    int B;
    int flat_D;
    float scale;
    float bias;
    
    using shared_tile = st_bf<64, 64>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;
    
    parallel_layout image;
    parallel_layout text;
    float* loss_out;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((B + 63) / 64, (B + 63) / 64, NUM_DEVICES);
    }
    
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(2 * sizeof(shared_tile) + sizeof(st_fl<64, 64>) + 1024);
    }
};

__device__ inline float logsigmoid_neg(float x) {
    // computes -logsigmoid(x) = log(1 + exp(-x))
    float abs_x = fabsf(x);
    float val = log1pf(expf(-abs_x));
    return (x < 0.0f) ? (-x + val) : val;
}

__device__ inline void kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    
    globals::shared_tile &a_smem = allocator.allocate<globals::shared_tile>();
    globals::shared_tile &b_smem = allocator.allocate<globals::shared_tile>();
    st_fl<64, 64> &c_smem = allocator.allocate<st_fl<64, 64>>();
    
    int row_idx = blockIdx.x;
    int col_idx = blockIdx.y;
    int target_dev = blockIdx.z;
    
    int num_k_blocks = G.flat_D / 64;
    
    rt_fl<64, 64> acc;
    zero(acc);
    
    __shared__ semaphore arrived;

    for (int k = 0; k < num_k_blocks; ++k) {
        __syncthreads(); // Ensure smem is consumed from previous loop before overwriting
        if (threadIdx.x == 0) {
            init_semaphore(arrived, 0, 1);
            tma::expect_bytes(arrived, 2 * sizeof(globals::shared_tile));
            tma::load_async(a_smem, G.image[G.dev_idx], {0, 0, row_idx, k}, arrived);
            tma::load_async(b_smem, G.text[target_dev], {0, 0, col_idx, k}, arrived);
        }
        __syncthreads(); // Ensure initialized semaphore is visible to all
        wait(arrived, 0);
        
        rt_bf<64, 64> a_reg;
        rt_bf<64, 64> b_reg;
        load(a_reg, a_smem);
        load(b_reg, b_smem);
        
        mma_ABt(acc, a_reg, b_reg, acc);
    }
    
    store(c_smem, acc);
    __syncthreads();
    
    float block_loss = 0.0f;
    float* c_ptr = (float*)&c_smem;
    
    // Element-wise log-sigmoid loss
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        int r = i / 64;
        int c = i % 64;
        
        int global_r = row_idx * 64 + r;
        int global_c = col_idx * 64 + c;
        
        if (global_r < G.B && global_c < G.B) {
            float val = c_ptr[i];
            float logits = G.scale * val + G.bias;
            // Diagonals of local device match are positive pairs
            float label = (target_dev == G.dev_idx && global_r == global_c) ? 1.0f : -1.0f;
            
            float x = label * logits;
            block_loss += logsigmoid_neg(x);
        }
    }
    
    // Warp and Block reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        block_loss += __shfl_down_sync(0xffffffff, block_loss, offset);
    }
    
    __shared__ float warp_sums[8];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = block_loss;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            int block_id = (blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
            G.loss_out[block_id] = val;
        }
    }
}

} // namespace siglip


namespace siglip_barrier {

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

} // namespace siglip_barrier


void entrypoint(
    kittens::py::TKParallelTensor &image,
    kittens::py::TKParallelTensor &text,
    kittens::py::TKParallelTensor &loss_out,
    kittens::py::TKParallelTensor &barrier,
    int B,
    int flat_D,
    float scale,
    float bias
) {
    kittens::py::parallel_tensor_check(image, text, loss_out, barrier);

    siglip::globals siglip_G {
        .B = B,
        .flat_D = flat_D,
        .scale = scale,
        .bias = bias,
        .image = kittens::py::parallel_tensor_to_pgl<typename siglip::globals::parallel_layout>(image),
        .text = kittens::py::parallel_tensor_to_pgl<typename siglip::globals::parallel_layout>(text),
        .loss_out = reinterpret_cast<float*>(loss_out.data_.data_ptr()),
        .dev_idx = image.local_rank_
    };

    siglip_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<siglip_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Global barrier to guarantee all data has safely reached PGL peers before reads.
    kittens::py::launch_kernel<siglip_barrier::config, siglip_barrier::globals, siglip_barrier::kernel>(barrier_G);
    
    // Cross-rank local TMA loads -> Fused Compute -> Reduction logic
    kittens::py::launch_kernel<siglip::config, siglip::globals, siglip::kernel>(siglip_G);
    
    // Barrier to ensure TMA loads are done before tensors are freed/rewritten in dynamic models.
    kittens::py::launch_kernel<siglip_barrier::config, siglip_barrier::globals, siglip_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_siglip", &entrypoint);
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
            "tk_siglip_ext",
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
def solution(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
    logit_bias: float = 0.0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Per-rank SigLIP loss with text features fully distributed via ThunderKittens TMA reads.
    Replaces host-driven O(N) bidir loop with direct NVLink peer memory accesses.
    """
    assert image_features.is_cuda and image_features.is_contiguous()
    assert text_features.is_cuda and text_features.is_contiguous()

    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    assert world_size == NUM_DEVICES, f"This kernel is fixed for NUM_DEVICES={NUM_DEVICES}"

    ext = _ensure_ext_jit()

    B, D = image_features.shape
    
    # Pad to strictly TK supported multiples for memory tiling
    flat_B = ((B + 63) // 64) * 64
    flat_D = ((D + 63) // 64) * 64

    inp_image = torch.zeros((1, 1, flat_B, flat_D), dtype=torch.bfloat16, device=image_features.device)
    inp_image[0, 0, :B, :D] = image_features.to(torch.bfloat16)

    inp_text = torch.zeros((1, 1, flat_B, flat_D), dtype=torch.bfloat16, device=text_features.device)
    inp_text[0, 0, :B, :D] = text_features.to(torch.bfloat16)

    n = inp_image.numel()

    # Preallocate TK tensors matching the topology PGL requires
    image_tk = get_or_create_parallel_tensor(ext, (1, 1, flat_B, flat_D), torch.bfloat16, multicast=False)
    text_tk = get_or_create_parallel_tensor(ext, (1, 1, flat_B, flat_D), torch.bfloat16, multicast=False)
    
    image_tk.data_.reshape(-1)[:n].copy_(inp_image.reshape(-1))
    text_tk.data_.reshape(-1)[:n].copy_(inp_text.reshape(-1))

    # Grid output to prevent atomics precision loss across millions of summed logits
    num_row_blocks = flat_B // 64
    num_col_blocks = flat_B // 64
    grid_size = num_row_blocks * num_col_blocks * NUM_DEVICES

    loss_tk = get_or_create_parallel_tensor(ext, (grid_size,), torch.float32, multicast=False)
    loss_tk.data_.zero_()

    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    scale_val = float(logit_scale.item() if isinstance(logit_scale, torch.Tensor) else logit_scale)
    bias_val = float(logit_bias.item() if isinstance(logit_bias, torch.Tensor) else logit_bias)

    # Launch cross-rank block fused kernel
    ext.tk_siglip(image_tk, text_tk, loss_tk, barrier_tk, B, flat_D, scale_val, bias_val)

    # Local reduce all blocks and divide by logical local batch
    total_loss = loss_tk.data_[:grid_size].sum() / B
    
    return total_loss