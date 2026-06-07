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
#include <cuda_bf16.h>
#include <vector>
#include <cfloat>

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceMax(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__inline__ __device__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void forward_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const __nv_bfloat16* __restrict__ old_logprobs,
    const __nv_bfloat16* __restrict__ advantages,
    const int ignore_index,
    const int N,
    const int V,
    __nv_bfloat16* __restrict__ per_token_logprobs,
    __nv_bfloat16* __restrict__ per_token_loss,
    float* __restrict__ w_out,
    float* __restrict__ local_stats
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int64_t label = labels[row];
    bool valid = (label != ignore_index);

    __shared__ float s_reduce[32];

    float local_max = -FLT_MAX;
    if (valid) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            float val = __bfloat162float(logits[row * V + i]);
            if (val > local_max) local_max = val;
        }
    }
    float row_max = blockReduceMax(local_max, s_reduce);

    float local_sum = 0.0f;
    float label_logit = 0.0f;
    if (valid) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            float val = __bfloat162float(logits[row * V + i]);
            local_sum += expf(val - row_max);
            if (i == label) label_logit = val;
        }
    }
    float row_sum = blockReduceSum(local_sum, s_reduce);
    float row_label_logit = blockReduceSum(label_logit, s_reduce);

    if (threadIdx.x == 0) {
        if (valid) {
            float log_sum_exp = row_max + logf(row_sum);
            float logprob = row_label_logit - log_sum_exp;
            
            float old_lp = __bfloat162float(old_logprobs[row]);
            float delta = logprob - old_lp;
            if (delta < -20.0f) delta = -20.0f;
            if (delta > 20.0f) delta = 20.0f;
            
            float ratio = expf(delta);
            float adv = __bfloat162float(advantages[row]);
            float pg = -ratio * adv;
            
            per_token_logprobs[row] = __float2bfloat16(logprob);
            per_token_loss[row] = __float2bfloat16(pg);
            w_out[row] = ratio * adv;
            
            float k3 = ratio - delta - 1.0f;
            float entropy = -logprob;
            
            atomicAdd(&local_stats[0], 1.0f);
            atomicAdd(&local_stats[1], pg);
            atomicAdd(&local_stats[2], ratio);
            atomicMin((int*)&local_stats[3], __float_as_int(ratio));
            atomicMax((int*)&local_stats[4], __float_as_int(ratio));
            atomicAdd(&local_stats[5], k3);
            atomicAdd(&local_stats[6], entropy);
        } else {
            per_token_logprobs[row] = __float2bfloat16(0.0f);
            per_token_loss[row] = __float2bfloat16(0.0f);
            w_out[row] = 0.0f;
        }
    }
}

struct PeerPtrs {
    const float* ptrs[8];
};

__global__ void reduce_stats_kernel(
    PeerPtrs peers,
    int world_size,
    float* global_stats
) {
    if (threadIdx.x < 7 && blockIdx.x == 0) {
        int idx = threadIdx.x;
        float val;
        if (idx == 3) val = FLT_MAX;
        else val = 0.0f;
        
        for (int p = 0; p < world_size; p++) {
            float p_val = peers.ptrs[p][idx];
            if (idx == 3) {
                if (p_val < val) val = p_val;
            } else if (idx == 4) {
                if (p_val > val) val = p_val;
            } else {
                val += p_val;
            }
        }
        global_stats[idx] = val;
    }
}

__global__ void backward_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const float* __restrict__ w_in,
    const float* __restrict__ global_stats,
    const float* __restrict__ grad_loss_ptr,
    const int ignore_index,
    const int N,
    const int V,
    __nv_bfloat16* __restrict__ d_logits
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int64_t label = labels[row];
    if (label == ignore_index) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            d_logits[row * V + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    float n_valid_global = global_stats[0];
    if (n_valid_global < 1.0f) n_valid_global = 1.0f;

    float grad_loss = *grad_loss_ptr;
    float scale = w_in[row] * grad_loss / n_valid_global;

    if (scale == 0.0f) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            d_logits[row * V + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    __shared__ float s_reduce[32];

    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float val = __bfloat162float(logits[row * V + i]);
        if (val > local_max) local_max = val;
    }
    float row_max = blockReduceMax(local_max, s_reduce);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float val = __bfloat162float(logits[row * V + i]);
        local_sum += expf(val - row_max);
    }
    float row_sum = blockReduceSum(local_sum, s_reduce);

    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float val = __bfloat162float(logits[row * V + i]);
        float prob = expf(val - row_max) / row_sum;
        if (i == label) {
            prob -= 1.0f;
        }
        float grad = prob * scale;
        d_logits[row * V + i] = __float2bfloat16(grad);
    }
}

void forward_pass(
    torch::Tensor logits,
    torch::Tensor labels,
    torch::Tensor old_logprobs,
    torch::Tensor advantages,
    int ignore_index,
    torch::Tensor per_token_logprobs,
    torch::Tensor per_token_loss,
    torch::Tensor w_out,
    torch::Tensor local_stats
) {
    int N = logits.size(0);
    int V = logits.size(1);
    int threads = 256;
    int blocks = N;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    forward_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        labels.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(old_logprobs.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(advantages.data_ptr<at::BFloat16>()),
        ignore_index,
        N, V,
        reinterpret_cast<__nv_bfloat16*>(per_token_logprobs.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(per_token_loss.data_ptr<at::BFloat16>()),
        w_out.data_ptr<float>(),
        local_stats.data_ptr<float>()
    );
}

void reduce_stats(
    std::vector<int64_t> peer_ptrs,
    int world_size,
    torch::Tensor global_stats
) {
    TORCH_CHECK(world_size <= 8, "This optimization is optimized for 8 NVLink connected GPUs");
    PeerPtrs p;
    for (int i = 0; i < world_size; i++) {
        p.ptrs[i] = reinterpret_cast<const float*>(static_cast<uintptr_t>(peer_ptrs[i]));
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_stats_kernel<<<1, 32, 0, stream>>>(p, world_size, global_stats.data_ptr<float>());
}

void backward_pass(
    torch::Tensor logits,
    torch::Tensor labels,
    torch::Tensor w_in,
    torch::Tensor global_stats,
    torch::Tensor grad_loss,
    int ignore_index,
    torch::Tensor d_logits
) {
    int N = logits.size(0);
    int V = logits.size(1);
    int threads = 256;
    int blocks = N;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    backward_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        labels.data_ptr<int64_t>(),
        w_in.data_ptr<float>(),
        global_stats.data_ptr<float>(),
        grad_loss.data_ptr<float>(),
        ignore_index,
        N, V,
        reinterpret_cast<__nv_bfloat16*>(d_logits.data_ptr<at::BFloat16>())
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_pass", &forward_pass, "Fused GRPO Forward");
    m.def("reduce_stats", &reduce_stats, "UVA reduce stats");
    m.def("backward_pass", &backward_pass, "Fused GRPO Backward");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_grpo_loss_ext", CUDA_SRC)
    return _ext


_symm_cache = None
def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"], c["global_stats"]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    global_stats = torch.empty(n, device=device, dtype=dtype)
    _symm_cache = {"n": n, "dtype": dtype, "device": device, 
                   "buf": buf, "hdl": hdl, "global_stats": global_stats}
    return buf, hdl, global_stats


class FusedGRPOLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits_flat, labels_flat, old_logprobs_flat, advantages_flat, ignore_index,
                symm_buf, hdl, peer_ptrs, global_stats, world_size):
        
        N, V = logits_flat.shape
        per_token_logprobs = torch.empty(N, dtype=torch.bfloat16, device=logits_flat.device)
        per_token_loss = torch.empty(N, dtype=torch.bfloat16, device=logits_flat.device)
        w_out = torch.empty(N, dtype=torch.float32, device=logits_flat.device)
        
        # Init specific symmetric reduction flags
        symm_buf.zero_()
        symm_buf[3] = 3.402823466e+38  # FLT_MAX

        # Kernel 1: Fused row-wise logic onto fast L1/L2 layout
        _get_ext().forward_pass(
            logits_flat, labels_flat, old_logprobs_flat, advantages_flat,
            ignore_index, per_token_logprobs, per_token_loss, w_out, symm_buf
        )
        
        # Async barrier overlapping cross-device buffers
        hdl.barrier(channel=0)
        
        # Kernel 2: Inter-GPU stats reduction
        _get_ext().reduce_stats(peer_ptrs, world_size, global_stats)
        
        # Clone global stats to support sequential/accumulated loss passes effectively
        global_stats_saved = global_stats.clone()
        ctx.save_for_backward(logits_flat, labels_flat, w_out, global_stats_saved)
        ctx.ignore_index = ignore_index
        
        n_valid = global_stats[0].clamp(min=1.0)
        true_pg = global_stats[1] / n_valid
        
        metrics = torch.stack([
            global_stats[2] / n_valid,
            global_stats[3],
            global_stats[4],
            global_stats[5] / n_valid,
            global_stats[6] / n_valid
        ])
        
        return true_pg, per_token_logprobs, per_token_loss, metrics

    @staticmethod
    def backward(ctx, grad_loss, grad_logprobs, grad_loss_pt, grad_metrics):
        logits_flat, labels_flat, w_out, global_stats = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        
        d_logits = torch.empty_like(logits_flat)
        
        # Kernel 3: Surrogate backprop directly to d_logits without explicit intermediate tracking nodes
        _get_ext().backward_pass(
            logits_flat, labels_flat, w_out, global_stats,
            grad_loss, ignore_index, d_logits
        )
        
        return d_logits, None, None, None, None, None, None, None, None, None


def solution(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, torch.Tensor]:

    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()
    _get_ext()

    # cuBLAS accelerated pass purely handling massive dense math
    logits = F.linear(hidden_states, weight) 

    # Layout conform formatting required cleanly by kernels
    logits_flat = logits.view(-1, logits.size(-1)).contiguous()
    labels_flat = labels.view(-1).contiguous()
    old_logprobs_flat = old_logprobs.to(torch.bfloat16).view(-1).contiguous()
    advantages_flat = advantages.to(torch.bfloat16).view(-1).contiguous()
    
    world_size = dist.get_world_size()
    buf, hdl, global_stats = _get_symm_state(7, torch.float32, logits.device)
    peer_ptrs = [int(hdl.buffer_ptrs[p]) for p in range(world_size)]
    
    loss, per_token_logprobs_flat, per_token_loss_flat, metrics = FusedGRPOLoss.apply(
        logits_flat, labels_flat, old_logprobs_flat, advantages_flat, ignore_index,
        buf, hdl, peer_ptrs, global_stats, world_size
    )
    
    per_token_logprobs = per_token_logprobs_flat.view_as(labels)
    per_token_loss = per_token_loss_flat.view_as(labels)
    
    return loss, None, per_token_logprobs, per_token_loss, metrics