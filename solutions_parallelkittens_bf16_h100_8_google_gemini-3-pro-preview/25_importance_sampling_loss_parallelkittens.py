import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Any
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
#include <cuda_bf16.h>
#include <math.h>
#include <algorithm>

using namespace kittens;

// Custom atomic min/max for float mappings
__device__ void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// Fused kernel for importance sampling pointwise ops and local stats reduction
__global__ void local_compute_kernel(
    const __nv_bfloat16* __restrict__ per_token_ce,
    const __nv_bfloat16* __restrict__ old_logprobs,
    const __nv_bfloat16* __restrict__ advantages,
    const int64_t* __restrict__ labels,
    int ignore_index,
    int N,
    __nv_bfloat16* __restrict__ per_token_pg,
    __nv_bfloat16* __restrict__ per_token_logprobs,
    __nv_bfloat16* __restrict__ w,
    float* __restrict__ local_stats
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum_n_valid = 0;
    float sum_pg = 0;
    float sum_surrogate = 0;
    float sum_ratio = 0;
    float min_ratio = INFINITY;
    float max_ratio = -INFINITY;
    float sum_k3 = 0;
    float sum_entropy = 0;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        int64_t label = labels[i];
        float ce = __bfloat162float(per_token_ce[i]);
        float old_lp = __bfloat162float(old_logprobs[i]);
        float adv = __bfloat162float(advantages[i]);
        
        float new_lp = -ce;
        per_token_logprobs[i] = __float2bfloat16(new_lp);
        
        if (label != ignore_index) {
            sum_n_valid += 1.0f;
            
            float delta = new_lp - old_lp;
            delta = fmaxf(-20.0f, fminf(20.0f, delta));
            float ratio = expf(delta);
            
            float pg = -(ratio * adv);
            per_token_pg[i] = __float2bfloat16(pg);
            sum_pg += pg;
            
            float w_val = ratio * adv; 
            w[i] = __float2bfloat16(w_val);
            sum_surrogate += w_val * ce;
            
            sum_ratio += ratio;
            min_ratio = fminf(min_ratio, ratio);
            max_ratio = fmaxf(max_ratio, ratio);
            
            float k3 = ratio - delta - 1.0f;
            sum_k3 += k3;
            
            sum_entropy += ce;
        } else {
            per_token_pg[i] = __float2bfloat16(0.0f);
            w[i] = __float2bfloat16(0.0f);
        }
    }
    
    // Warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_n_valid += __shfl_down_sync(mask, sum_n_valid, offset);
        sum_pg += __shfl_down_sync(mask, sum_pg, offset);
        sum_surrogate += __shfl_down_sync(mask, sum_surrogate, offset);
        sum_ratio += __shfl_down_sync(mask, sum_ratio, offset);
        min_ratio = fminf(min_ratio, __shfl_down_sync(mask, min_ratio, offset));
        max_ratio = fmaxf(max_ratio, __shfl_down_sync(mask, max_ratio, offset));
        sum_k3 += __shfl_down_sync(mask, sum_k3, offset);
        sum_entropy += __shfl_down_sync(mask, sum_entropy, offset);
    }
    
    // Leader threads commit to global shared buffer safely
    if (idx % 32 == 0) {
        atomicAdd(&local_stats[0], sum_n_valid);
        atomicAdd(&local_stats[1], sum_pg);
        atomicAdd(&local_stats[2], sum_surrogate);
        atomicAdd(&local_stats[3], sum_ratio);
        atomicMinFloat(&local_stats[4], min_ratio);
        atomicMaxFloat(&local_stats[5], max_ratio);
        atomicAdd(&local_stats[6], sum_k3);
        atomicAdd(&local_stats[7], sum_entropy);
    }
}

namespace grpo_barrier {
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
} // namespace grpo_barrier

// Compact P2P Read kernel targeting symmetric memory via peer pointers
namespace grpo_reduce {
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};
struct globals {
    static constexpr int NUM_DEVICES = 8;
    float* peer_ptrs[NUM_DEVICES];
    float* global_out;
    int dev_idx;
};
__device__ inline void kernel(const globals &G) {
    float sums[8] = {0};
    sums[4] = INFINITY;
    sums[5] = -INFINITY;
    
    #pragma unroll
    for (int i = 0; i < G.NUM_DEVICES; i++) {
        sums[0] += G.peer_ptrs[i][0]; // n_valid
        sums[1] += G.peer_ptrs[i][1]; // pg_sum
        // [2] local_surrogate_sum not cross-device reduced
        sums[3] += G.peer_ptrs[i][3]; // sum_ratio
        sums[4] = fminf(sums[4], G.peer_ptrs[i][4]); // min_ratio
        sums[5] = fmaxf(sums[5], G.peer_ptrs[i][5]); // max_ratio
        sums[6] += G.peer_ptrs[i][6]; // k3
        sums[7] += G.peer_ptrs[i][7]; // entropy
    }
    
    // Store back strictly to local results
    G.global_out[0] = sums[0];
    G.global_out[1] = sums[1];
    G.global_out[2] = G.peer_ptrs[G.dev_idx][2]; 
    G.global_out[3] = sums[3];
    G.global_out[4] = sums[4];
    G.global_out[5] = sums[5];
    G.global_out[6] = sums[6];
    G.global_out[7] = sums[7];
}
} // namespace grpo_reduce

void entrypoint(
    const torch::Tensor& per_token_ce,
    const torch::Tensor& old_logprobs,
    const torch::Tensor& advantages,
    const torch::Tensor& labels,
    int ignore_index,
    torch::Tensor& per_token_pg,
    torch::Tensor& per_token_logprobs,
    torch::Tensor& w,
    kittens::py::TKParallelTensor& local_stats_tk,
    torch::Tensor& global_stats,
    kittens::py::TKParallelTensor& barrier
) {
    int N = per_token_ce.numel();
    int num_threads = 256;
    int num_blocks = std::min((N + num_threads - 1) / num_threads, 1024);
    
    float* local_stats_ptr = reinterpret_cast<float*>(local_stats_tk.data_.data_ptr());
    
    // 1. Compute local values into local arrays natively
    local_compute_kernel<<<num_blocks, num_threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(per_token_ce.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(old_logprobs.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(advantages.data_ptr<at::BFloat16>()),
        labels.data_ptr<int64_t>(),
        ignore_index,
        N,
        reinterpret_cast<__nv_bfloat16*>(per_token_pg.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(per_token_logprobs.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
        local_stats_ptr
    );
    
    // 2. Safely barrier for visibility
    grpo_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<grpo_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    kittens::py::launch_kernel<grpo_barrier::config, grpo_barrier::globals, grpo_barrier::kernel>(barrier_G);
    
    // 3. One pass 8-float reduction fetching via fast symmetric peer pointers
    grpo_reduce::globals reduce_G;
    reduce_G.global_out = global_stats.data_ptr<float>();
    reduce_G.dev_idx = local_stats_tk.local_rank_;
    for(int i = 0; i < grpo_reduce::globals::NUM_DEVICES; i++) {
        reduce_G.peer_ptrs[i] = reinterpret_cast<float*>(local_stats_tk.ptrs_[i]);
    }
    kittens::py::launch_kernel<grpo_reduce::config, grpo_reduce::globals, grpo_reduce::kernel>(reduce_G);
    
    // 4. Safely barrier before exit to shield the TK array overwrites on the next loop iteration
    kittens::py::launch_kernel<grpo_barrier::config, grpo_barrier::globals, grpo_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_grpo_step", &entrypoint);
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
_init_vals = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_grpo_ext",
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
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    ext = _ensure_ext_jit()
    world_size = dist.get_world_size()

    # 1. Compute massive dense arrays explicitly in highly-optimized PyTorch components
    logits = F.linear(hidden_states, weight)
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.to(torch.int64).contiguous().view(-1)
    
    # Needs to require grad to pipe correctly back into logits
    per_token_ce = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='none')
    
    # 2. Allocate variables
    old_logprobs_flat = old_logprobs.contiguous().view(-1)
    advantages_flat = advantages.contiguous().view(-1)
    
    per_token_pg = torch.empty_like(per_token_ce)
    per_token_logprobs = torch.empty_like(per_token_ce)
    w = torch.empty_like(per_token_ce)
    
    # 3. Setup ThunderKittens reduction structure and zero buffers natively async
    tk_stats = get_or_create_parallel_tensor(ext, (8,), torch.float32, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    global_stats = torch.empty(8, dtype=torch.float32, device=hidden_states.device)
    
    global _init_vals
    if _init_vals is None or _init_vals.device != hidden_states.device:
        _init_vals = torch.tensor([0.0, 0.0, 0.0, 0.0, float('inf'), float('-inf'), 0.0, 0.0], 
                                  dtype=torch.float32, device=hidden_states.device)
    tk_stats.data_.copy_(_init_vals)
    
    # 4. Fused device-side operation
    ext.tk_grpo_step(
        per_token_ce,
        old_logprobs_flat,
        advantages_flat,
        labels_flat,
        ignore_index,
        per_token_pg,
        per_token_logprobs,
        w,
        tk_stats,
        global_stats,
        barrier_tk
    )
    
    # 5. Connect autograd graph
    n_valid_global = global_stats[0].clamp(min=1.0)
    global_pg_sum = global_stats[1]
    
    # Recover autograd graph cleanly; w is implicitly detached
    surrogate = (w * per_token_ce).sum() / n_valid_global
    true_pg = global_pg_sum / n_valid_global
    
    # Forward pass equals `true_pg`, backward is `surrogate` gradient
    loss = true_pg + surrogate - surrogate.detach()
    
    # Metrics
    ratio_mean = global_stats[3] / n_valid_global
    min_ratio = global_stats[4]
    max_ratio = global_stats[5]
    k3_mean = global_stats[6] / n_valid_global
    entropy_mean = global_stats[7] / n_valid_global
    metrics = torch.stack([ratio_mean, min_ratio, max_ratio, k3_mean, entropy_mean])
    
    return loss, None, per_token_logprobs.view_as(labels), per_token_pg.view_as(labels), metrics