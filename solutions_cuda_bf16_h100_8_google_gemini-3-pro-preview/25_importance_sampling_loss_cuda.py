"""
Strategy:
1. Kernel Fusion: Combines elementwise importance sampling ops (exp, clamp, mask) and local reductions (sum, min, max, entropy, kl) into a single custom CUDA kernel. This minimizes memory bandwidth and replaces 7 separate NCCL all-reduces.
2. Device-Side Communication: The local kernel atomically accumulates reductions directly into a `torch.distributed._symmetric_memory` buffer. A lightweight UVA kernel (`reduce_global_stats_kernel`) gathers and reduces the 7 scalar metrics from all peers via direct peer-to-peer access, avoiding host-driven `dist.all_reduce` overhead.
3. Compute-Communication Overlap: The cross-rank UVA reduction and symmetric memory barrier are offloaded to a dedicated communication stream. Concurrently, the default stream computes the PyTorch `local_surrogate_sum` (necessary for autograd). The streams synchronize right before assembling the final loss, effectively hiding the barrier and peer-read latency behind independent local computation.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple, Any
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ void atomicMinFloat(float* addr, float val) {
    if (isnan(val)) return;
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) break;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    if (isnan(val)) return;
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void init_stats_kernel(float* stats) {
    if (threadIdx.x == 0) {
        stats[0] = 0.0f;       // n_valid
        stats[1] = 0.0f;       // pg_sum
        stats[2] = 0.0f;       // ratio_sum
        stats[3] = INFINITY;   // ratio_min
        stats[4] = -INFINITY;  // ratio_max
        stats[5] = 0.0f;       // k3_sum
        stats[6] = 0.0f;       // entropy_sum
        stats[7] = 0.0f;       // padding
    }
}

template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ per_token_ce,
    const int64_t* __restrict__ labels,
    const scalar_t* __restrict__ old_logprobs,
    const scalar_t* __restrict__ advantages,
    scalar_t* __restrict__ w_out,
    scalar_t* __restrict__ per_token_pg_out,
    scalar_t* __restrict__ per_token_logprobs_out,
    float* __restrict__ local_stats,
    int ignore_index,
    int n
) {
    __shared__ float s_n_valid[32];
    __shared__ float s_pg_sum[32];
    __shared__ float s_ratio_sum[32];
    __shared__ float s_ratio_min[32];
    __shared__ float s_ratio_max[32];
    __shared__ float s_k3_sum[32];
    __shared__ float s_entropy_sum[32];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    float l_n_valid = 0;
    float l_pg_sum = 0;
    float l_ratio_sum = 0;
    float l_ratio_min = INFINITY;
    float l_ratio_max = -INFINITY;
    float l_k3_sum = 0;
    float l_entropy_sum = 0;

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        int64_t label = labels[idx];
        float ce = static_cast<float>(per_token_ce[idx]);
        float old_lp = static_cast<float>(old_logprobs[idx]);
        float adv = static_cast<float>(advantages[idx]);

        float new_lp = -ce;
        per_token_logprobs_out[idx] = static_cast<scalar_t>(new_lp);

        if (label != ignore_index) {
            l_n_valid = 1.0f;
            float delta = new_lp - old_lp;
            delta = fmaxf(-20.0f, fminf(20.0f, delta));
            float ratio = expf(delta);
            float pg = -ratio * adv;

            w_out[idx] = static_cast<scalar_t>(ratio * adv);
            per_token_pg_out[idx] = static_cast<scalar_t>(pg);

            l_pg_sum = pg;
            l_ratio_sum = ratio;
            l_ratio_min = ratio;
            l_ratio_max = ratio;
            l_k3_sum = ratio - delta - 1.0f;
            l_entropy_sum = ce;
        } else {
            w_out[idx] = static_cast<scalar_t>(0.0f);
            per_token_pg_out[idx] = static_cast<scalar_t>(0.0f);
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        l_n_valid += __shfl_down_sync(0xffffffff, l_n_valid, offset);
        l_pg_sum += __shfl_down_sync(0xffffffff, l_pg_sum, offset);
        l_ratio_sum += __shfl_down_sync(0xffffffff, l_ratio_sum, offset);
        l_ratio_min = fminf(l_ratio_min, __shfl_down_sync(0xffffffff, l_ratio_min, offset));
        l_ratio_max = fmaxf(l_ratio_max, __shfl_down_sync(0xffffffff, l_ratio_max, offset));
        l_k3_sum += __shfl_down_sync(0xffffffff, l_k3_sum, offset);
        l_entropy_sum += __shfl_down_sync(0xffffffff, l_entropy_sum, offset);
    }

    if (lane == 0) {
        s_n_valid[warp] = l_n_valid;
        s_pg_sum[warp] = l_pg_sum;
        s_ratio_sum[warp] = l_ratio_sum;
        s_ratio_min[warp] = l_ratio_min;
        s_ratio_max[warp] = l_ratio_max;
        s_k3_sum[warp] = l_k3_sum;
        s_entropy_sum[warp] = l_entropy_sum;
    }
    __syncthreads();

    // Block reduction
    if (warp == 0) {
        int num_warps = blockDim.x / 32;
        l_n_valid = (lane < num_warps) ? s_n_valid[lane] : 0;
        l_pg_sum = (lane < num_warps) ? s_pg_sum[lane] : 0;
        l_ratio_sum = (lane < num_warps) ? s_ratio_sum[lane] : 0;
        l_ratio_min = (lane < num_warps) ? s_ratio_min[lane] : INFINITY;
        l_ratio_max = (lane < num_warps) ? s_ratio_max[lane] : -INFINITY;
        l_k3_sum = (lane < num_warps) ? s_k3_sum[lane] : 0;
        l_entropy_sum = (lane < num_warps) ? s_entropy_sum[lane] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            l_n_valid += __shfl_down_sync(0xffffffff, l_n_valid, offset);
            l_pg_sum += __shfl_down_sync(0xffffffff, l_pg_sum, offset);
            l_ratio_sum += __shfl_down_sync(0xffffffff, l_ratio_sum, offset);
            l_ratio_min = fminf(l_ratio_min, __shfl_down_sync(0xffffffff, l_ratio_min, offset));
            l_ratio_max = fmaxf(l_ratio_max, __shfl_down_sync(0xffffffff, l_ratio_max, offset));
            l_k3_sum += __shfl_down_sync(0xffffffff, l_k3_sum, offset);
            l_entropy_sum += __shfl_down_sync(0xffffffff, l_entropy_sum, offset);
        }

        if (lane == 0) {
            atomicAdd(&local_stats[0], l_n_valid);
            atomicAdd(&local_stats[1], l_pg_sum);
            atomicAdd(&local_stats[2], l_ratio_sum);
            if (l_ratio_min < INFINITY) atomicMinFloat(&local_stats[3], l_ratio_min);
            if (l_ratio_max > -INFINITY) atomicMaxFloat(&local_stats[4], l_ratio_max);
            atomicAdd(&local_stats[5], l_k3_sum);
            atomicAdd(&local_stats[6], l_entropy_sum);
        }
    }
}

__global__ void reduce_global_stats_kernel(
    const long long* __restrict__ peer_ptrs,
    float* __restrict__ global_out,
    int world_size
) {
    int tid = threadIdx.x;
    if (tid >= 7) return;

    float val;
    if (tid == 0 || tid == 1 || tid == 2 || tid == 5 || tid == 6) val = 0.0f;
    else if (tid == 3) val = INFINITY;
    else if (tid == 4) val = -INFINITY;

    for (int r = 0; r < world_size; r++) {
        const float* peer_stats = (const float*)peer_ptrs[r];
        float p_val = peer_stats[tid];
        if (tid == 0 || tid == 1 || tid == 2 || tid == 5 || tid == 6) {
            val += p_val;
        } else if (tid == 3) {
            val = fminf(val, p_val);
        } else if (tid == 4) {
            val = fmaxf(val, p_val);
        }
    }
    
    global_out[tid] = val;
}

void launch_compute_stats(
    torch::Tensor per_token_ce,
    torch::Tensor labels,
    torch::Tensor old_logprobs,
    torch::Tensor advantages,
    torch::Tensor w_out,
    torch::Tensor per_token_pg_out,
    torch::Tensor per_token_logprobs_out,
    torch::Tensor local_stats,
    int ignore_index
) {
    int n = per_token_ce.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    init_stats_kernel<<<1, 1, 0, stream>>>(local_stats.data_ptr<float>());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, per_token_ce.scalar_type(), "compute_stats_kernel", ([&] {
        compute_stats_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            per_token_ce.data_ptr<scalar_t>(),
            labels.data_ptr<int64_t>(),
            old_logprobs.data_ptr<scalar_t>(),
            advantages.data_ptr<scalar_t>(),
            w_out.data_ptr<scalar_t>(),
            per_token_pg_out.data_ptr<scalar_t>(),
            per_token_logprobs_out.data_ptr<scalar_t>(),
            local_stats.data_ptr<float>(),
            ignore_index,
            n
        );
    }));
}

void launch_reduce_global_stats(
    torch::Tensor peer_ptrs,
    torch::Tensor global_out
) {
    int world_size = peer_ptrs.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_global_stats_kernel<<<1, 32, 0, stream>>>(
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        global_out.data_ptr<float>(),
        world_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_compute_stats", &launch_compute_stats, "Compute local elementwise metrics and local reductions");
    m.def("launch_reduce_global_stats", &launch_reduce_global_stats, "Reduce global metrics from peers via UVA");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("grpo_importance_sampling_ext", CUDA_SRC)
    return _ext

_resource_cache = None
def _get_resources(device):
    global _resource_cache
    if _resource_cache is not None:
        return _resource_cache

    buf = symm_mem.empty((8,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    global_out = torch.empty((8,), device=device, dtype=torch.float32)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    comm_stream = torch.cuda.Stream(device=device)

    _resource_cache = (buf, hdl, global_out, ptrs_tensor, comm_stream)
    return _resource_cache

def solution(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # 1. Compute logits and per-token cross entropy via heavily-optimized PyTorch ops
    logits = F.linear(hidden_states, weight)
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.contiguous().view(-1)
    
    # per_token_ce retains the grad_fn necessary for surrogate loss backpropagation
    per_token_ce = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='none')
    per_token_ce_contig = per_token_ce.contiguous() if not per_token_ce.is_contiguous() else per_token_ce
    
    old_logprobs_flat = old_logprobs.contiguous().view(-1)
    advantages_flat = advantages.contiguous().view(-1)
    
    w = torch.empty_like(per_token_ce_contig)
    per_token_pg = torch.empty_like(per_token_ce_contig)
    per_token_logprobs = torch.empty_like(per_token_ce_contig)
    
    buf, hdl, global_out, ptrs_tensor, comm_stream = _get_resources(hidden_states.device)
    ext = _get_ext()
    
    # 2. Fuse all elementwise operations and local reductions into a single kernel run
    ext.launch_compute_stats(
        per_token_ce_contig, labels_flat, old_logprobs_flat, advantages_flat,
        w, per_token_pg, per_token_logprobs, buf, ignore_index
    )
    
    ready_event = torch.cuda.Event()
    ready_event.record()
    
    # 3. OVERLAP: Compute the autograd-tracked surrogate sum completely concurrently with cross-rank communication
    # The kernel inherently detached `w`. `per_token_ce` will trace back through the graph.
    local_surrogate_sum = (w * per_token_ce).sum()
    
    # 4. Device-side communication for global stats handling via UVA
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(ready_event)
        # Block-level sync over symmetric memory ensures all peers have populated local_stats
        hdl.barrier(channel=0)
        # Gather and reduce the 7 scalar metrics directly via UVA peer pointers
        ext.launch_reduce_global_stats(ptrs_tensor, global_out)
        done_event = torch.cuda.Event()
        done_event.record()
        
    torch.cuda.current_stream().wait_event(done_event)
    
    # 5. Extract finalized global stats and formulate final composite metrics & loss
    n_valid_global = global_out[0].clamp(min=1.0)
    true_pg = global_out[1] / n_valid_global
    ratio_mean = global_out[2] / n_valid_global
    ratio_min = global_out[3]
    ratio_max = global_out[4]
    k3_mean = global_out[5] / n_valid_global
    entropy_mean = global_out[6] / n_valid_global
    
    surrogate = local_surrogate_sum / n_valid_global
    
    # The loss triggers exactly the same gradients due to surrogate component
    loss = true_pg.detach() + surrogate - surrogate.detach()
    metrics = torch.stack([ratio_mean, ratio_min, ratio_max, k3_mean, entropy_mean])
    
    per_token_logprobs = per_token_logprobs.view_as(labels)
    per_token_loss = per_token_pg.view_as(labels)
    
    return loss, None, per_token_logprobs, per_token_loss, metrics