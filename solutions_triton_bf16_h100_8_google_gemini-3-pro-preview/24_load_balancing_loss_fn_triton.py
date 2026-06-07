import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Union, Tuple, Optional
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <limits.h>
#include <algorithm>

// Kernel 1: Local Stats Aggregation (Warp-level cooperative processing)
__global__ void local_stats_kernel(
    const __nv_bfloat16* __restrict__ gate_logits,
    const bool* __restrict__ attention_mask,
    float* __restrict__ global_stats,
    int N, int E, int K, int BS
) {
    extern __shared__ float smem[];
    float* s_P = smem;           // size E
    float* s_C = smem + E;       // size E
    float* s_W = smem + 2 * E;   // size 1
    
    for (int i = threadIdx.x; i < 2 * E + 1; i += blockDim.x) {
        smem[i] = 0.0f;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int global_warp_id = blockIdx.x * (blockDim.x / 32) + warp_id;
    int num_warps = gridDim.x * (blockDim.x / 32);

    for (int row = global_warp_id; row < N; row += num_warps) {
        float w = 1.0f;
        if (attention_mask != nullptr) {
            w = attention_mask[row % BS] ? 1.0f : 0.0f;
        }
        if (w == 0.0f) continue;

        // 1. Warp Find Max
        float max_val = -INFINITY;
        for (int e = lane_id; e < E; e += 32) {
            float val = __bfloat162float(gate_logits[row * E + e]);
            if (val > max_val) max_val = val;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        max_val = __shfl_sync(0xffffffff, max_val, 0);

        // 2. Compute sum of exp
        float sum_exp = 0.0f;
        for (int e = lane_id; e < E; e += 32) {
            float val = __bfloat162float(gate_logits[row * E + e]);
            sum_exp += expf(val - max_val);
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        }
        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

        // 3. Find Top-K indices
        int selected_indices[8]; 
        for (int k = 0; k < K; ++k) selected_indices[k] = -1;

        for (int k = 0; k < K; ++k) {
            float local_max = -INFINITY;
            int local_max_idx = INT_MAX;
            for (int e = lane_id; e < E; e += 32) {
                float val = __bfloat162float(gate_logits[row * E + e]);
                bool selected = false;
                for (int j = 0; j < k; ++j) {
                    if (selected_indices[j] == e) selected = true;
                }
                if (!selected && val > local_max) {
                    local_max = val;
                    local_max_idx = e;
                }
            }

            float max_v = local_max;
            int max_i = local_max_idx;
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_v = __shfl_down_sync(0xffffffff, max_v, offset);
                int other_i = __shfl_down_sync(0xffffffff, max_i, offset);
                if (other_v > max_v || (other_v == max_v && other_i < max_i)) {
                    max_v = other_v;
                    max_i = other_i;
                }
            }
            max_i = __shfl_sync(0xffffffff, max_i, 0);
            selected_indices[k] = max_i;
        }

        // 4. Accumulate routing probabilities
        for (int e = lane_id; e < E; e += 32) {
            float val = expf(__bfloat162float(gate_logits[row * E + e]) - max_val) / sum_exp;
            atomicAdd(&s_P[e], w * val);
        }

        // 5. Accumulate count and total weights
        if (lane_id == 0) {
            for (int k = 0; k < K; ++k) {
                if (selected_indices[k] != INT_MAX) {
                    atomicAdd(&s_C[selected_indices[k]], w);
                }
            }
            atomicAdd(&s_W[0], w);
        }
    }
    __syncthreads();

    // Flush to global memory
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        atomicAdd(&global_stats[e], s_P[e]);
        atomicAdd(&global_stats[E + e], s_C[e]);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&global_stats[2 * E], s_W[0]);
    }
}

// Kernel 2: Compute purely local loss scalar
__global__ void compute_local_loss_kernel(
    const float* __restrict__ global_stats,
    float* __restrict__ symm_buf,
    int E
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float w_sum = global_stats[2 * E];
        float loss = 0.0f;
        if (w_sum > 0.0f) {
            float sum_cp = 0.0f;
            for (int e = 0; e < E; ++e) {
                sum_cp += global_stats[e] * global_stats[E + e];
            }
            loss = sum_cp / (w_sum * w_sum) * E;
        }
        symm_buf[0] = loss;
    }
}

struct PeerPtrs { const float* ptrs[8]; }; // Max 8 Hopper SXM GPUs per node domain

// Kernel 3: Device-side NVLink multi-gpu reduction of the load balancing scalar
__global__ void cross_rank_reduce_kernel(PeerPtrs peers, float* __restrict__ out, int world_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total_loss = 0.0f;
        for (int p = 0; p < world_size; ++p) {
            total_loss += peers.ptrs[p][0];
        }
        out[0] = total_loss / world_size;
    }
}

// Kernel 4: Analytical Gradient Computation
__global__ void backward_kernel(
    __nv_bfloat16* __restrict__ grad_x,
    const __nv_bfloat16* __restrict__ gate_logits,
    const bool* __restrict__ attention_mask,
    const float* __restrict__ global_stats,
    float grad_output,
    int N, int E, int BS, int world_size
) {
    extern __shared__ float s_G[]; 
    
    float w_sum = global_stats[2 * E];
    if (w_sum <= 0.0f) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * E; i += gridDim.x * blockDim.x) {
            grad_x[i] = __float2bfloat16(0.0f);
        }
        return;
    }
    
    // Gradient scale per expert factor
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float c_e = global_stats[E + e]; 
        s_G[e] = (c_e * E) / (w_sum * w_sum * world_size) * grad_output;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int global_warp_id = blockIdx.x * (blockDim.x / 32) + warp_id;
    int num_warps = gridDim.x * (blockDim.x / 32);

    for (int row = global_warp_id; row < N; row += num_warps) {
        float w = 1.0f;
        if (attention_mask != nullptr) {
            w = attention_mask[row % BS] ? 1.0f : 0.0f;
        }
        if (w == 0.0f) {
            for (int e = lane_id; e < E; e += 32) grad_x[row * E + e] = __float2bfloat16(0.0f);
            continue;
        }

        // Recompute Softmax Probabilities
        float max_val = -INFINITY;
        for (int e = lane_id; e < E; e += 32) {
            float val = __bfloat162float(gate_logits[row * E + e]);
            if (val > max_val) max_val = val;
        }
        for (int offset = 16; offset > 0; offset /= 2) max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        max_val = __shfl_sync(0xffffffff, max_val, 0);

        float sum_exp = 0.0f;
        for (int e = lane_id; e < E; e += 32) {
            sum_exp += expf(__bfloat162float(gate_logits[row * E + e]) - max_val);
        }
        for (int offset = 16; offset > 0; offset /= 2) sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
        sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

        float s_i = 0.0f;
        for (int e = lane_id; e < E; e += 32) {
            float r_ie = expf(__bfloat162float(gate_logits[row * E + e]) - max_val) / sum_exp;
            s_i += r_ie * s_G[e];
        }
        for (int offset = 16; offset > 0; offset /= 2) s_i += __shfl_down_sync(0xffffffff, s_i, offset);
        s_i = __shfl_sync(0xffffffff, s_i, 0);

        for (int e = lane_id; e < E; e += 32) {
            float r_ie = expf(__bfloat162float(gate_logits[row * E + e]) - max_val) / sum_exp;
            float g = r_ie * w * (s_G[e] - s_i);
            grad_x[row * E + e] = __float2bfloat16(g);
        }
    }
}

void compute_loss(
    torch::Tensor gate_logits, std::optional<torch::Tensor> attention_mask,
    torch::Tensor global_stats, torch::Tensor symm_buf, int N, int E, int K, int BS
) {
    TORCH_CHECK(K <= 8, "top_k > 8 is not supported in the compiled fast-path");
    TORCH_CHECK(E <= 4096, "num_experts > 4096 is extremely unusual and not supported");
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = std::max(1, std::min(1024, (N + (threads / 32) - 1) / (threads / 32)));
    int smem_size = (2 * E + 1) * sizeof(float);
    
    const bool* mask_ptr = attention_mask.has_value() ? attention_mask.value().data_ptr<bool>() : nullptr;
    const __nv_bfloat16* logits_ptr = reinterpret_cast<const __nv_bfloat16*>(gate_logits.data_ptr<at::BFloat16>());
    
    local_stats_kernel<<<blocks, threads, smem_size, stream>>>(
        logits_ptr, mask_ptr, global_stats.data_ptr<float>(), N, E, K, BS
    );

    compute_local_loss_kernel<<<1, 1, 0, stream>>>(
        global_stats.data_ptr<float>(), symm_buf.data_ptr<float>(), E
    );
}

void reduce_loss(torch::Tensor out, std::vector<int64_t> peer_ptrs_int, int world_size) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    PeerPtrs peers;
    for (int i = 0; i < world_size && i < 8; ++i) {
        peers.ptrs[i] = reinterpret_cast<const float*>(peer_ptrs_int[i]);
    }
    cross_rank_reduce_kernel<<<1, 1, 0, stream>>>(peers, out.data_ptr<float>(), world_size);
}

void compute_backward(
    torch::Tensor grad_x, torch::Tensor gate_logits, std::optional<torch::Tensor> attention_mask,
    torch::Tensor global_stats, float grad_output, int N, int E, int BS, int world_size
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = std::max(1, std::min(1024, (N + (threads / 32) - 1) / (threads / 32)));
    int smem_size = E * sizeof(float);
    
    const bool* mask_ptr = attention_mask.has_value() ? attention_mask.value().data_ptr<bool>() : nullptr;
    const __nv_bfloat16* logits_ptr = reinterpret_cast<const __nv_bfloat16*>(gate_logits.data_ptr<at::BFloat16>());
    __nv_bfloat16* grad_ptr = reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr<at::BFloat16>());
    
    backward_kernel<<<blocks, threads, smem_size, stream>>>(
        grad_ptr, logits_ptr, mask_ptr, global_stats.data_ptr<float>(), grad_output, N, E, BS, world_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_loss", &compute_loss);
    m.def("reduce_loss", &reduce_loss);
    m.def("compute_backward", &compute_backward);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_moe_load_balancing", CUDA_SRC)
    return _ext


_symm_cache = None
def _get_symm_state(device: torch.device):
    global _symm_cache
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    
    if _symm_cache is None:
        if world_size > 1:
            buf = symm_mem.empty(1, device=device, dtype=torch.float32)
            hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
            peer_ptrs = [int(p) for p in hdl.buffer_ptrs]
        else:
            buf = torch.empty(1, device=device, dtype=torch.float32)
            hdl = None
            peer_ptrs = [buf.data_ptr()]
        _symm_cache = (buf, hdl, peer_ptrs, world_size)
    else:
        buf, hdl, peer_ptrs, world_size = _symm_cache
        
    out = torch.empty((), device=device, dtype=torch.float32)
    return {"buf": buf, "hdl": hdl, "out": out, "peer_ptrs": peer_ptrs, "world_size": world_size}


class MoELoadBalancingLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, concatenated_gate_logits, attention_mask, num_experts, top_k):
        N, E = concatenated_gate_logits.shape
        BS = attention_mask.numel() if attention_mask is not None else 1
        
        ext = _get_ext()
        symm_state = _get_symm_state(concatenated_gate_logits.device)
        
        # New tensor mapped properly into the computation trace per forward
        global_stats = torch.zeros(2 * E + 1, device=concatenated_gate_logits.device, dtype=torch.float32)
        
        ext.compute_loss(
            concatenated_gate_logits,
            attention_mask,
            global_stats,
            symm_state["buf"],
            N, E, top_k, BS
        )
        
        if symm_state["hdl"] is not None:
            symm_state["hdl"].barrier(channel=0)
            
        ext.reduce_loss(
            symm_state["out"],
            symm_state["peer_ptrs"],
            symm_state["world_size"]
        )
        
        ctx.save_for_backward(concatenated_gate_logits, attention_mask, global_stats)
        ctx.E, ctx.BS, ctx.world_size = E, BS, symm_state["world_size"]
        
        return symm_state["out"]
        
    @staticmethod
    def backward(ctx, grad_output):
        concatenated_gate_logits, attention_mask, global_stats = ctx.saved_tensors
        N = concatenated_gate_logits.shape[0]
        
        grad_x = torch.empty_like(concatenated_gate_logits)
        
        _get_ext().compute_backward(
            grad_x,
            concatenated_gate_logits,
            attention_mask,
            global_stats,
            grad_output.item(),
            N, ctx.E, ctx.BS, ctx.world_size
        )
        
        return grad_x, None, None, None


def solution(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    num_experts: int,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(gate_logits, (tuple, list)):
        concatenated = torch.cat(gate_logits, dim=0)
    else:
        concatenated = gate_logits
        
    if concatenated.dtype != torch.bfloat16 or not concatenated.is_contiguous():
        concatenated = concatenated.to(dtype=torch.bfloat16, memory_format=torch.contiguous_format)
        
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool or not attention_mask.is_contiguous():
            attention_mask = attention_mask.to(dtype=torch.bool, memory_format=torch.contiguous_format)

    return MoELoadBalancingLoss.apply(concatenated, attention_mask, num_experts, top_k)