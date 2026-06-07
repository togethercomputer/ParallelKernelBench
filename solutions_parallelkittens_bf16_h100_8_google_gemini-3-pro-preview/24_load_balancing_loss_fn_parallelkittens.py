"""
Strategy:
1. Optimize the local device-side computation by replacing memory-heavy standard ops 
   (like `one_hot` followed by `mean` and `unsqueeze`, which allocate huge intermediates 
   of shape [N, top_k, num_experts]) with fast `bincount` and matrix-vector multiplications 
   (`routing_weights.T @ am_repeated`). This eliminates massive memory spikes and bandwidth 
   bottlenecks on the hot path before communication even begins.
2. Replace opaque `torch.distributed.all_reduce` with a custom ThunderKittens PGL kernel 
   using Hopper's multimem and NVSwitch multicast. The scalar loss is mapped directly 
   into symmetric memory and all-reduced via `multimem.ld_reduce<ADD>`, keeping 
   synchronization entirely device-side and minimizing host overhead for the reduction.
"""

import os
import torch
import torch.distributed as dist
from typing import Union, Tuple, Optional
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source (ThunderKittens All-Reduce via Multicast)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

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
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(tensor, barrier);

    TORCH_CHECK(tensor.data_.numel() % (all_reduce::globals::NUM_DEVICES * all_reduce::globals::NUM_ELEMS_PER_BLOCK) == 0,
        "The total number of tensor elements must be divisible by NUM_DEVICES * NUM_ELEMS_PER_BLOCK");

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

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_reduce", &entrypoint);
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
ALIGNMENT = NUM_DEVICES * NUM_ELEMS_PER_BLOCK


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_allreduce_ext",
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
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


@torch.no_grad()
def solution(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    num_experts: int,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    if isinstance(gate_logits, (tuple, list)):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )
    else:
        compute_device = gate_logits.device
        concatenated_gate_logits = gate_logits

    # Fast local soft-max and top-k router selection
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    N = concatenated_gate_logits.shape[0]
    
    # Fast local binning & token probabilities, bypassing huge `one_hot` mask allocations
    if attention_mask is None:
        tokens_per_expert = torch.zeros((top_k, num_experts), dtype=torch.float32, device=compute_device)
        for k in range(top_k):
            counts = torch.bincount(selected_experts[:, k], minlength=num_experts)
            tokens_per_expert[k] = counts.float()
        tokens_per_expert /= N
        
        router_prob_per_expert = routing_weights.mean(dim=0)
    else:
        am_1d = attention_mask.reshape(-1).to(dtype=torch.float32, device=compute_device)
        num_hidden_layers = N // am_1d.shape[0]
        am_repeated = am_1d.repeat(num_hidden_layers)
        
        sum_am = am_repeated.sum()
        
        tokens_per_expert = torch.zeros((top_k, num_experts), dtype=torch.float32, device=compute_device)
        for k in range(top_k):
            counts = torch.bincount(selected_experts[:, k], weights=am_repeated, minlength=num_experts)
            tokens_per_expert[k] = counts.float()
        tokens_per_expert /= sum_am
        
        # Matrix-vector mult eliminates expanding/allocating massive attention masks
        router_prob_per_expert = (routing_weights.T @ am_repeated) / sum_am

    # Local scalar loss (equivalent to summing over all expert elements after multiplication)
    tokens_sum = tokens_per_expert.sum(dim=0)
    overall_loss = torch.dot(tokens_sum, router_prob_per_expert) * num_experts

    # Device-side all-reduce over ThunderKittens PGL NVSwitch Multicast
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        assert world == NUM_DEVICES, f"ThunderKittens kernel built for {NUM_DEVICES} devices, but got {world}"
        
        ext = _ensure_ext_jit()
        
        flat = overall_loss.to(torch.bfloat16).view(-1).contiguous()
        n = flat.numel()
        padded = ((n + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        
        tensor_tk = get_or_create_parallel_tensor(ext, (padded,), torch.bfloat16, multicast=True)
        barrier_tk = get_or_create_barrier(ext, num_devices=world)
        
        tensor_tk.data_[:n] = flat
        if padded > n:
            tensor_tk.data_[n:padded].zero_()
            
        ext.tk_all_reduce(tensor_tk, barrier_tk)
        
        result = tensor_tk.data_[:n].clone()
        overall_loss = result.to(overall_loss.dtype).view(overall_loss.shape) / world

    return overall_loss