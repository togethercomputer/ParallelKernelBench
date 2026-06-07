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

// -------------------------------------------------------------------------
// 1. Forward Kernel: Single pass to compute local M_e, C_e, and W_total
// -------------------------------------------------------------------------
__global__ void moe_load_balance_warp_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ mask,
    float* __restrict__ global_m_e,
    float* __restrict__ global_c_e,
    float* __restrict__ global_w_total,
    int total_tokens,
    int mask_size,
    int num_experts,
    int top_k
) {
    extern __shared__ float smem[];
    float* s_m_e = smem;
    float* s_c_e = smem + num_experts;
    float* s_w_total = smem + 2 * num_experts;

    for (int i = threadIdx.x; i < 2 * num_experts + 1; i += blockDim.x) {
        smem[i] = 0.0f;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int token_idx = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (token_idx < total_tokens) {
        float w = 1.0f;
        if (mask != nullptr && mask_size > 0) {
            w = mask[token_idx % mask_size];
        }

        if (w > 0.0f) {
            float max_val = -1e20f;
            const __nv_bfloat16* token_logits = logits + token_idx * num_experts;
            
            // Pass 1: Max
            for (int i = lane_id; i < num_experts; i += 32) {
                float val = __bfloat162float(token_logits[i]);
                if (val > max_val) max_val = val;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
            }
            max_val = __shfl_sync(0xffffffff, max_val, 0);

            // Pass 2: Sum Exp
            float sum_exp = 0.0f;
            for (int i = lane_id; i < num_experts; i += 32) {
                float val = __bfloat162float(token_logits[i]);
                sum_exp += expf(val - max_val);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
            }
            sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

            // Pass 3: Softmax Probabilities and accumulation
            float thread_probs[128]; // Safe up to 4096 experts per warp
            int thread_indices[128];
            int num_items = 0;

            for (int i = lane_id; i < num_experts; i += 32) {
                float val = __bfloat162float(token_logits[i]);
                float prob = expf(val - max_val) / sum_exp;
                if (num_items < 128) {
                    thread_probs[num_items] = prob;
                    thread_indices[num_items] = i;
                    num_items++;
                }
                atomicAdd(&s_c_e[i], w * prob);
            }

            // Select Top-K with consistent tie-breaking behavior
            for (int k = 0; k < top_k; k++) {
                float local_max_prob = -1.0f;
                int local_max_idx = -1;
                for (int i = 0; i < num_items; i++) {
                    if (thread_probs[i] > local_max_prob) {
                        local_max_prob = thread_probs[i];
                        local_max_idx = thread_indices[i];
                    }
                }

                float warp_max_prob = local_max_prob;
                int warp_max_idx = local_max_idx;

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    float other_prob = __shfl_down_sync(0xffffffff, warp_max_prob, offset);
                    int other_idx = __shfl_down_sync(0xffffffff, warp_max_idx, offset);
                    if (other_prob > warp_max_prob || (other_prob == warp_max_prob && other_idx < warp_max_idx)) {
                        warp_max_prob = other_prob;
                        warp_max_idx = other_idx;
                    }
                }
                
                warp_max_prob = __shfl_sync(0xffffffff, warp_max_prob, 0);
                warp_max_idx = __shfl_sync(0xffffffff, warp_max_idx, 0);

                if (lane_id == 0) {
                    atomicAdd(&s_m_e[warp_max_idx], w);
                }

                // Mask out selected probability for the next K iteration
                for (int i = 0; i < num_items; i++) {
                    if (thread_indices[i] == warp_max_idx) {
                        thread_probs[i] = -2.0f; 
                    }
                }
            }

            if (lane_id == 0) atomicAdd(s_w_total, w);
        }
    }

    __syncthreads();

    // Flush shared memory to global arrays
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        if (s_m_e[i] > 0.0f) atomicAdd(&global_m_e[i], s_m_e[i]);
        if (s_c_e[i] > 0.0f) atomicAdd(&global_c_e[i], s_c_e[i]);
    }
    if (threadIdx.x == 0) {
        if (s_w_total[0] > 0.0f) atomicAdd(global_w_total, s_w_total[0]);
    }
}

// -------------------------------------------------------------------------
// 2. Compute Local Loss
// -------------------------------------------------------------------------
__global__ void compute_local_loss_kernel(
    const float* __restrict__ m_e,
    const float* __restrict__ c_e,
    const float* __restrict__ w_total,
    float* __restrict__ local_loss,
    int num_experts
) {
    float w = w_total[0];
    if (w <= 0.0f) w = 1.0f; // Avoid NaN division if completely masked out

    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        sum += (m_e[i] / w) * (c_e[i] / w);
    }

    static __shared__ float shared_sum[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) shared_sum[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < (blockDim.x / 32)) ? shared_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            local_loss[0] = sum * num_experts;
        }
    }
}

// -------------------------------------------------------------------------
// 3. UVA Symmetric Memory Scalar All-Reduce
// -------------------------------------------------------------------------
__global__ void symm_allreduce_scalar_kernel(
    const long long* __restrict__ peer_ptrs,
    float* __restrict__ out_val,
    int world_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < world_size; i++) {
            const float* ptr = reinterpret_cast<const float*>(peer_ptrs[i]);
            sum += *ptr;
        }
        out_val[0] = sum / world_size;
    }
}

// -------------------------------------------------------------------------
// 4. Backward Pass Kernel: Recompute probabilities and emit exact analytical grad
// -------------------------------------------------------------------------
__global__ void moe_load_balance_backward_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ mask,
    const float* __restrict__ m_e,
    float w_total,
    __nv_bfloat16* __restrict__ grad_logits,
    float grad_output_scaled,
    int total_tokens,
    int mask_size,
    int num_experts
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int token_idx = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (token_idx >= total_tokens) return;

    float w = 1.0f;
    if (mask != nullptr && mask_size > 0) w = mask[token_idx % mask_size];

    __nv_bfloat16* out_ptr = grad_logits + token_idx * num_experts;

    if (w <= 0.0f) {
        for (int i = lane_id; i < num_experts; i += 32) out_ptr[i] = __float2bfloat16(0.0f);
        return;
    }

    float W_sq = w_total * w_total;
    if (W_sq <= 0.0f) W_sq = 1.0f;
    float scale = (float(num_experts) / W_sq) * grad_output_scaled * w;

    const __nv_bfloat16* token_logits = logits + token_idx * num_experts;
    
    float max_val = -1e20f;
    for (int i = lane_id; i < num_experts; i += 32) {
        float val = __bfloat162float(token_logits[i]);
        if (val > max_val) max_val = val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    float sum_exp = 0.0f;
    for (int i = lane_id; i < num_experts; i += 32) {
        float val = __bfloat162float(token_logits[i]);
        sum_exp += expf(val - max_val);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    // Compute Analytical Expected Grad component
    float expected_G = 0.0f;
    for (int i = lane_id; i < num_experts; i += 32) {
        float val = __bfloat162float(token_logits[i]);
        float p = expf(val - max_val) / sum_exp;
        expected_G += p * m_e[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) expected_G += __shfl_down_sync(0xffffffff, expected_G, offset);
    expected_G = __shfl_sync(0xffffffff, expected_G, 0);

    // Emit final grad components safely
    for (int i = lane_id; i < num_experts; i += 32) {
        float val = __bfloat162float(token_logits[i]);
        float p = expf(val - max_val) / sum_exp;
        float dx = scale * p * (m_e[i] - expected_G);
        out_ptr[i] = __float2bfloat16(dx);
    }
}

// -------------------------------------------------------------------------
// Bindings
// -------------------------------------------------------------------------
void launch_moe_load_balance(
    torch::Tensor logits, std::optional<torch::Tensor> mask,
    torch::Tensor m_e, torch::Tensor c_e, torch::Tensor w_total, int top_k
) {
    int total_tokens = logits.size(0);
    int num_experts = logits.size(1);
    int mask_size = mask.has_value() ? mask->size(0) : 0;
    const float* mask_ptr = mask.has_value() ? mask->data_ptr<float>() : nullptr;
    
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (total_tokens + warps_per_block - 1) / warps_per_block;
    int smem_size = (2 * num_experts + 1) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    moe_load_balance_warp_kernel<<<blocks, threads, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()), mask_ptr,
        m_e.data_ptr<float>(), c_e.data_ptr<float>(), w_total.data_ptr<float>(),
        total_tokens, mask_size, num_experts, top_k
    );
}

void launch_compute_local_loss(torch::Tensor m_e, torch::Tensor c_e, torch::Tensor w_total, torch::Tensor local_loss) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    compute_local_loss_kernel<<<1, 256, 0, stream>>>(
        m_e.data_ptr<float>(), c_e.data_ptr<float>(), w_total.data_ptr<float>(),
        local_loss.data_ptr<float>(), m_e.size(0)
    );
}

void launch_symm_allreduce(torch::Tensor peer_ptrs, torch::Tensor out_val) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    symm_allreduce_scalar_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<const long long*>(peer_ptrs.data_ptr<int64_t>()),
        out_val.data_ptr<float>(), peer_ptrs.size(0)
    );
}

void launch_moe_load_balance_backward(
    torch::Tensor logits, std::optional<torch::Tensor> mask, torch::Tensor m_e,
    float w_total, torch::Tensor grad_logits, float grad_output_scaled, int num_experts
) {
    int total_tokens = logits.size(0);
    int mask_size = mask.has_value() ? mask->size(0) : 0;
    const float* mask_ptr = mask.has_value() ? mask->data_ptr<float>() : nullptr;

    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (total_tokens + warps_per_block - 1) / warps_per_block;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    moe_load_balance_backward_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()), mask_ptr,
        m_e.data_ptr<float>(), w_total,
        reinterpret_cast<__nv_bfloat16*>(grad_logits.data_ptr<at::BFloat16>()),
        grad_output_scaled, total_tokens, mask_size, num_experts
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_moe_load_balance", &launch_moe_load_balance);
    m.def("launch_compute_local_loss", &launch_compute_local_loss);
    m.def("launch_symm_allreduce", &launch_symm_allreduce);
    m.def("launch_moe_load_balance_backward", &launch_moe_load_balance_backward);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_load_balance_fast_ext", CUDA_SRC, extra_compile_args={'nvcc': ['-O3']})
    return _ext


_symm_cache = {}
def _get_symm_state(device: torch.device):
    if device in _symm_cache:
        return _symm_cache[device]
    
    buf = symm_mem.empty((1,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = torch.empty((1,), device=device, dtype=torch.float32)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    _symm_cache[device] = (buf, hdl, out, ptrs_tensor)
    return _symm_cache[device]


class CustomMoELoadBalanceLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_logits, attention_mask, num_experts, top_k):
        _get_ext()  # warm up JIT
        
        if isinstance(gate_logits, (tuple, list)):
            ctx.is_tuple = True
            ctx.shapes = [g.shape for g in gate_logits]
            ctx.devices = [g.device for g in gate_logits]
            compute_device = gate_logits[0].device
            logits = torch.cat([g.to(compute_device) for g in gate_logits], dim=0).contiguous()
        else:
            ctx.is_tuple = False
            logits = gate_logits.contiguous()

        mask_tensor = None
        if attention_mask is not None:
            mask_tensor = attention_mask.reshape(-1).float().contiguous()

        ctx.save_for_backward(logits, mask_tensor)
        ctx.num_experts = num_experts

        m_e = torch.zeros(num_experts, device=logits.device, dtype=torch.float32)
        c_e = torch.zeros(num_experts, device=logits.device, dtype=torch.float32)
        w_total = torch.zeros(1, device=logits.device, dtype=torch.float32)

        _get_ext().launch_moe_load_balance(logits, mask_tensor, m_e, c_e, w_total, top_k)

        ctx.m_e = m_e
        ctx.w_total = w_total

        is_dist = dist.is_available() and dist.is_initialized()
        ctx.world_size = dist.get_world_size() if is_dist else 1

        if is_dist:
            buf, hdl, out, ptrs_tensor = _get_symm_state(logits.device)
            _get_ext().launch_compute_local_loss(m_e, c_e, w_total, buf)
            
            # Using channel-ordered barrier directly on the symmetric handle
            hdl.barrier(channel=0) 
            _get_ext().launch_symm_allreduce(ptrs_tensor, out)
            global_loss = out.clone().reshape(())
        else:
            out = torch.empty(1, device=logits.device, dtype=torch.float32)
            _get_ext().launch_compute_local_loss(m_e, c_e, w_total, out)
            global_loss = out.reshape(())
            
        return global_loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, mask_tensor = ctx.saved_tensors
        m_e = ctx.m_e
        w_total_val = ctx.w_total.item()
        num_experts = ctx.num_experts

        # Because we used a scalar sum all-reduce internally on forward (Loss = sum(Li)/N), 
        # its backward gradient passes locally out purely scaled by local distribution.
        grad_output_scaled = grad_output.item() / ctx.world_size
        grad_logits = torch.empty_like(logits)

        _get_ext().launch_moe_load_balance_backward(
            logits, mask_tensor, m_e, w_total_val, grad_logits,
            grad_output_scaled, num_experts
        )

        if ctx.is_tuple:
            grads = []
            offset = 0
            for shape, device in zip(ctx.shapes, ctx.devices):
                size = shape[0]
                grads.append(grad_logits[offset:offset+size].to(device))
                offset += size
            return tuple(grads), None, None, None
        else:
            return grad_logits, None, None, None


def solution(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    num_experts: int,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return CustomMoELoadBalanceLoss.apply(gate_logits, attention_mask, num_experts, top_k)