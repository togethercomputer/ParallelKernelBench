"""
Distributed link-prediction ranking over positive and negative scores.
Optimized with ThunderKittens device-side compute and NVSwitch Multicast.

Strategy:
- Mathematical equivalence: Local independent ranking followed by a 1D AllGather is equivalent 
  to a 2D AllGather followed by global ranking. This slashes communication from O(P * K) to O(P).
- Device-side fusion: A ThunderKittens custom kernel uses a warp-per-row reduction to efficiently 
  compute the rankings directly from the local BFloat16 tensors.
- Overlap & Multicast: The kernel natively writes the final ranking using inline NVSwitch 
  multicast (`st.global.nc.mcast`), overlapping the local compute completely with the collective 
  distribution of the results.
"""

from typing import Optional
import os

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>

using namespace kittens;

namespace compute_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 256;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    using parallel_layout = pgl<gl<float, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout tensor;
    const __nv_bfloat16* pos_scores;
    const __nv_bfloat16* neg_scores;
    int P;
    int max_P;
    int K;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        int rows_per_block = config::NUM_THREADS / 32;
        return dim3((max_P + rows_per_block - 1) / rows_per_block);
    }
};

__device__ inline void kernel(const globals &G) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row_idx = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (row_idx < G.max_P) {
        float rank_val = 0.0f;
        // Compute ranking locally using mathematical equivalence
        // PyTorch's descending stable sort implies pos_score stays ahead of equal elements.
        // Therefore, position is just 1 + (number of strictly greater negatives).
        if (row_idx < G.P) {
            float pos = __bfloat162float(G.pos_scores[row_idx]);
            int count = 0;
            // Coalesced warp access over the negative samples
            for(int k = lane; k < G.K; k += 32) {
                float neg = __bfloat162float(G.neg_scores[row_idx * G.K + k]);
                if (neg > pos) {
                    count++;
                }
            }
            
            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                count += __shfl_down_sync(0xffffffff, count, offset);
            }
            
            if (lane == 0) {
                rank_val = (float)(count + 1);
            }
        }
        
        // Directly multicast the result to all GPUs, fusing communication with compute
        if (lane == 0) {
            int my_offset = G.dev_idx * G.max_P + row_idx;
            asm volatile("st.global.nc.mcast.f32 [%0], %1;\n" 
                :: "l"(&G.tensor.mc_ptr[my_offset]), "f"(rank_val) 
                : "memory");
        }
    }
}

} // namespace compute_gather

namespace compute_gather_barrier {

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

} // namespace compute_gather_barrier

void entrypoint(
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier,
    uintptr_t pos_scores_ptr,
    uintptr_t neg_scores_ptr,
    int P,
    int max_P,
    int K
) {
    kittens::py::parallel_tensor_check(tensor, barrier);

    compute_gather::globals G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename compute_gather::globals::parallel_layout>(tensor),
        .pos_scores = reinterpret_cast<const __nv_bfloat16*>(pos_scores_ptr),
        .neg_scores = reinterpret_cast<const __nv_bfloat16*>(neg_scores_ptr),
        .P = P,
        .max_P = max_P,
        .K = K,
        .dev_idx = tensor.local_rank_
    };

    compute_gather_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<compute_gather_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<compute_gather_barrier::config, compute_gather_barrier::globals, compute_gather_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<compute_gather::config, compute_gather::globals, compute_gather::kernel>(G);
    kittens::py::launch_kernel<compute_gather_barrier::config, compute_gather_barrier::globals, compute_gather_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_compute_gather", &entrypoint);
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
            "tk_gnn_ranking_ext",
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
    local_pos_scores: torch.Tensor,
    local_neg_scores: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    if world_size == 1:
        scores = torch.cat([local_pos_scores.view(-1, 1), local_neg_scores], dim=1)
        _, indices = torch.sort(torch.sigmoid(scores), dim=1, descending=True)
        return torch.nonzero(indices == 0)[:, 1].view(-1).detach() + 1
        
    assert world_size == 8, f"ThunderKittens kernel hardcoded for 8 devices, got {world_size}"
    
    assert local_pos_scores.dtype == torch.bfloat16, "Expected bfloat16 pos_scores"
    assert local_neg_scores.dtype == torch.bfloat16, "Expected bfloat16 neg_scores"
    
    local_pos_scores = local_pos_scores.contiguous()
    local_neg_scores = local_neg_scores.contiguous()
    
    ext = _ensure_ext_jit()

    P = local_pos_scores.shape[0]
    K = local_neg_scores.shape[1]

    # Quick sync to determine total padded shape for the communication collective
    sizes = torch.zeros(world_size, dtype=torch.long, device=local_pos_scores.device)
    sizes[rank] = P
    dist.all_reduce(sizes, op=dist.ReduceOp.SUM, group=group)
    
    max_P = sizes.max().item()
    # Pad aggressively to avoid constant VMM reallocation when graphs fluctuate slightly
    max_P_padded = ((max_P + 4095) // 4096) * 4096

    tensor_tk = get_or_create_parallel_tensor(
        ext, (world_size * max_P_padded,), torch.float32, multicast=True
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    pos_ptr = local_pos_scores.data_ptr()
    neg_ptr = local_neg_scores.data_ptr()

    # Launch fusion kernel: locally computes independent rankings & NVSwitch multicasts immediately
    ext.tk_compute_gather(tensor_tk, barrier_tk, pos_ptr, neg_ptr, P, max_P_padded, K)

    # Reconstruct exact GraphStorm format: concatenate valid sub-slices from each rank's multicast payload
    data_view = tensor_tk.data_.view(world_size, max_P_padded)
    sizes_list = sizes.tolist()
    
    res = [data_view[i, :sizes_list[i]] for i in range(world_size)]
    return torch.cat(res, dim=0).to(torch.long)