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
#include <cstdint>
#include <cfloat>

// Fused softmax + topk + accumulation kernel.
// For each token row [E], compute softmax, top-k experts, and atomically add
// 1.0 to tokens_per_expert[e] for each selected expert (or mask-weighted),
// and softmax probability to router_prob_per_expert[e] (or mask-weighted).
//
// Out tensors are float32, shape [E].
// If attention_mask_flat is null, every token contributes weight 1.
// Otherwise, mask[i] in {0,1}, and the row contributes mask[i].
//
// We need normalization counts:
//   tokens_per_expert[e] /= sum_over_tokens(mask)  (same for all e when no mask: =N)
//   router_prob_per_expert[e] /= sum_over_tokens(mask)
//
// Without mask: denom_tokens = N, denom_router = N. (Simple post-scale.)
// With mask: denom_tokens = top_k * sum(mask) for tokens_per_expert? Let's check:
// Reference:
//   tokens_per_expert = sum(expert_mask * expert_attention_mask, dim=0) / sum(expert_attention_mask, dim=0)
//   expert_attention_mask shape after reshape: [N, top_k, num_experts], values are mask[token]
//   sum over dim=0 of expert_attention_mask -> [top_k, num_experts], each = sum(mask)
// So tokens_per_expert[k,e] = (sum over tokens of mask[i]*1[expert@k==e]) / sum(mask)
// Then overall_loss uses tokens_per_expert (shape [top_k, num_experts]) * router_prob[1,num_experts]
// summed: sum over k,e of tokens_per_expert[k,e] * router_prob[e]
// = (1/sum(mask)) * sum_i mask[i] * sum_k 1[expert@k][e_selected_at_k]... 
// Actually simpler: aggregate per expert (sum across k) then it becomes equivalent.
//
// We compute:
//   tpe[e] = sum over (i, k) of mask[i] * 1[topk(i,k)==e]  -> divided by sum(mask) (NOT top_k, since each k row sums to sum(mask))
//   But tokens_per_expert is shape [top_k, num_experts] in reference.
//   sum_{k,e} tpe[k,e] * rpe[e]
// We can fuse: T[e] = sum over (i,k) mask[i]*1[topk(i,k)==e] / sum(mask)
//   Then sum_e T[e] * rpe[e] equals sum_{k,e} tpe[k,e] * rpe[e]. ✓
//
// And rpe[e] = sum_i mask[i]*softmax(i,e) / sum(mask).
//
// So we can collapse top_k dim. Final loss = num_experts * sum_e T[e] * rpe[e].

template <int MAX_E, int MAX_K>
__global__ void fused_moe_loss_kernel(
    const __nv_bfloat16* __restrict__ logits,  // [N, E] bf16
    const float* __restrict__ mask,             // [N] or null
    float* __restrict__ tpe,                    // [E] zeroed
    float* __restrict__ rpe,                    // [E] zeroed
    int N,
    int E,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr = logits + (int64_t)row * E;

    float w = (mask == nullptr) ? 1.0f : mask[row];

    // Load logits and compute softmax in shared memory.
    extern __shared__ float smem[];
    float* probs = smem;  // [E]

    // Load + max
    float local_max = -FLT_MAX;
    for (int e = tid; e < E; e += blockDim.x) {
        float v = __bfloat162float(row_ptr[e]);
        probs[e] = v;
        if (v > local_max) local_max = v;
    }
    __syncthreads();

    // Block reduce max
    __shared__ float s_max;
    typedef float fT;
    // simple reduction
    static __shared__ float redbuf[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    float m = local_max;
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, m, off);
        if (other > m) m = other;
    }
    if (lane == 0) redbuf[warp] = m;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) / 32;
        float mm = (tid < nwarps) ? redbuf[lane] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, mm, off);
            if (other > mm) mm = other;
        }
        if (tid == 0) s_max = mm;
    }
    __syncthreads();

    // exp and sum
    float local_sum = 0.0f;
    for (int e = tid; e < E; e += blockDim.x) {
        float v = expf(probs[e] - s_max);
        probs[e] = v;
        local_sum += v;
    }
    __syncthreads();
    float ss = local_sum;
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, off);
    }
    if (lane == 0) redbuf[warp] = ss;
    __syncthreads();
    __shared__ float s_sum;
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) / 32;
        float sv = (tid < nwarps) ? redbuf[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sv += __shfl_xor_sync(0xffffffff, sv, off);
        }
        if (tid == 0) s_sum = sv;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_sum;

    // Normalize and accumulate router_prob
    for (int e = tid; e < E; e += blockDim.x) {
        float p = probs[e] * inv_sum;
        probs[e] = p;
        if (w != 0.0f) {
            atomicAdd(&rpe[e], p * w);
        }
    }
    __syncthreads();

    // Top-k selection (single thread, K is small typically <=8)
    if (tid == 0 && w != 0.0f) {
        int picked[MAX_K];
        for (int k = 0; k < K; ++k) {
            float best = -FLT_MAX;
            int bi = -1;
            for (int e = 0; e < E; ++e) {
                float v = probs[e];
                bool taken = false;
                for (int kk = 0; kk < k; ++kk) {
                    if (picked[kk] == e) { taken = true; break; }
                }
                if (!taken && v > best) {
                    best = v;
                    bi = e;
                }
            }
            picked[k] = bi;
            if (bi >= 0) {
                atomicAdd(&tpe[bi], w);
            }
        }
    }
}

// Final reduction: loss = num_experts * sum_e (tpe[e]/denom_t) * (rpe[e]/denom_r)
__global__ void finalize_loss_kernel(
    const float* __restrict__ tpe,
    const float* __restrict__ rpe,
    float denom_t,
    float denom_r,
    int E,
    int num_experts,
    float* __restrict__ out  // scalar
) {
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int e = tid; e < E; e += blockDim.x) {
        acc += (tpe[e] / denom_t) * (rpe[e] / denom_r);
    }
    static __shared__ float buf[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, off);
    }
    if (lane == 0) buf[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) / 32;
        float v = (tid < nwarps) ? buf[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, off);
        }
        if (tid == 0) {
            out[0] = v * (float)num_experts;
        }
    }
}

void launch_fused_moe_loss(
    torch::Tensor logits_bf16,   // [N, E]
    torch::Tensor mask_or_empty, // [N] float32 or empty
    torch::Tensor tpe,           // [E] float32
    torch::Tensor rpe,           // [E] float32
    int64_t K
) {
    TORCH_CHECK(logits_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(tpe.dtype() == torch::kFloat32);
    TORCH_CHECK(rpe.dtype() == torch::kFloat32);
    int N = logits_bf16.size(0);
    int E = logits_bf16.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(tpe.data_ptr<float>(), 0, sizeof(float) * E, stream);
    cudaMemsetAsync(rpe.data_ptr<float>(), 0, sizeof(float) * E, stream);

    int threads = 128;
    if (E >= 256) threads = 256;
    int blocks = N;
    size_t shmem = sizeof(float) * E;

    const float* mask_ptr = mask_or_empty.numel() > 0 ? mask_or_empty.data_ptr<float>() : nullptr;

    // Dispatch on K with templated max
    fused_moe_loss_kernel<1024, 16><<<blocks, threads, shmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits_bf16.data_ptr<at::BFloat16>()),
        mask_ptr,
        tpe.data_ptr<float>(),
        rpe.data_ptr<float>(),
        N, E, (int)K
    );
}

void launch_finalize_loss(
    torch::Tensor tpe,
    torch::Tensor rpe,
    double denom_t,
    double denom_r,
    int64_t num_experts,
    torch::Tensor out
) {
    int E = tpe.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 128;
    if (E >= 256) threads = 256;
    finalize_loss_kernel<<<1, threads, 0, stream>>>(
        tpe.data_ptr<float>(),
        rpe.data_ptr<float>(),
        (float)denom_t,
        (float)denom_r,
        E,
        (int)num_experts,
        out.data_ptr<float>()
    );
}

// Custom all-reduce SUM for small float buffer over peer pointers.
__global__ void allreduce_small_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float s = 0.0f;
    for (int r = 0; r < world_size; ++r) {
        const float* p = (const float*)ptrs[r];
        s += p[idx];
    }
    out[idx] = s;
}

void launch_allreduce_small_f32(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n
) {
    int world_size = ptrs_tensor.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 32;
    int blocks = (n + threads - 1) / threads;
    allreduce_small_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const long long*)ptrs_tensor.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        world_size,
        (int)n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_moe_loss", &launch_fused_moe_loss);
    m.def("launch_finalize_loss", &launch_finalize_loss);
    m.def("launch_allreduce_small_f32", &launch_allreduce_small_f32);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_loss_fused_ext", CUDA_SRC)
    return _ext


_symm_cache = None
def _get_symm(device):
    global _symm_cache
    if _symm_cache is not None:
        return _symm_cache
    if not (dist.is_available() and dist.is_initialized()):
        return None
    buf = symm_mem.empty(1, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = torch.empty(1, device=device, dtype=torch.float32)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _symm_cache = (buf, hdl, out, ptrs_tensor)
    return _symm_cache


@torch.no_grad()
def solution(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    num_experts: int,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Concatenate
    if isinstance(gate_logits, (tuple, list)):
        compute_device = gate_logits[0].device
        concatenated = torch.cat(
            [g.to(compute_device) for g in gate_logits], dim=0
        )
    else:
        compute_device = gate_logits.device
        concatenated = gate_logits

    if concatenated.dtype != torch.bfloat16:
        concatenated = concatenated.to(torch.bfloat16)
    concatenated = concatenated.contiguous()

    N, E = concatenated.shape
    ext = _get_ext()

    tpe = torch.empty(E, device=compute_device, dtype=torch.float32)
    rpe = torch.empty(E, device=compute_device, dtype=torch.float32)

    # Build mask flattened to [N] if provided
    if attention_mask is None:
        mask_flat = torch.empty(0, device=compute_device, dtype=torch.float32)
        denom_t = float(N)
        denom_r = float(N)
    else:
        bsz, seqlen = attention_mask.shape
        num_layers = N // (bsz * seqlen)
        m = attention_mask.to(compute_device).to(torch.float32)
        # Replicate over layers
        mask_flat = m.reshape(1, bsz, seqlen).expand(num_layers, bsz, seqlen).reshape(-1).contiguous()
        s = float(mask_flat.sum().item())
        denom_t = s
        denom_r = s

    ext.launch_fused_moe_loss(concatenated, mask_flat, tpe, rpe, top_k)

    if dist.is_available() and dist.is_initialized():
        symm = _get_symm(compute_device)
        buf, hdl, out, ptrs_tensor = symm
        # finalize directly into symmetric buffer
        ext.launch_finalize_loss(tpe, rpe, denom_t, denom_r, num_experts, buf)
        hdl.barrier(channel=0)
        ext.launch_allreduce_small_f32(ptrs_tensor, out, 1)
        ws = dist.get_world_size()
        return (out / ws).reshape(()).clone()
    else:
        out = torch.empty(1, device=compute_device, dtype=torch.float32)
        ext.launch_finalize_loss(tpe, rpe, denom_t, denom_r, num_experts, out)
        return out.reshape(())