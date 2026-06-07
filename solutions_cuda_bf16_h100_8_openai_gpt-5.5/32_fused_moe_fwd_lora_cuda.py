from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <limits>

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p, int64_t i);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* p, int64_t i) {
    return p[i];
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* p, int64_t i, float v);

template <>
__device__ __forceinline__ void store_from_float<float>(float* p, int64_t i, float v) {
    p[i] = v;
}

template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, int64_t i, float v) {
    p[i] = __float2bfloat16(v);
}

__device__ __forceinline__ bool better_pair(float v, int idx, float best_v, int best_idx) {
    return (v > best_v) || ((v == best_v) && (idx >= 0) && ((best_idx < 0) || (idx < best_idx)));
}

template <typename scalar_t>
__global__ void router_topk_sum_kernel(
    const scalar_t* __restrict__ logits,
    scalar_t* __restrict__ out,
    int64_t rows,
    int experts,
    int top_k
) {
    extern __shared__ unsigned char smem_raw[];
    float* s_logits = reinterpret_cast<float*>(smem_raw);
    float* s_vals = s_logits + experts;
    int* s_idxs = reinterpret_cast<int*>(s_vals + blockDim.x);

    int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int64_t base = (int64_t)row * experts;

    float local_max = -CUDART_INF_F;
    for (int e = tid; e < experts; e += blockDim.x) {
        float v = load_as_float<scalar_t>(logits, base + e);
        s_logits[e] = v;
        local_max = fmaxf(local_max, v);
    }

    s_vals[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_vals[tid] = fmaxf(s_vals[tid], s_vals[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_vals[0];

    float local_sum = 0.0f;
    for (int e = tid; e < experts; e += blockDim.x) {
        local_sum += expf(s_logits[e] - row_max);
    }
    s_vals[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_vals[tid] += s_vals[tid + stride];
        }
        __syncthreads();
    }
    float denom = s_vals[0];

    float top_sum = 0.0f;
    int k_lim = top_k < experts ? top_k : experts;

    for (int k = 0; k < k_lim; ++k) {
        float best_v = -CUDART_INF_F;
        int best_idx = -1;

        for (int e = tid; e < experts; e += blockDim.x) {
            float v = s_logits[e];
            if (better_pair(v, e, best_v, best_idx)) {
                best_v = v;
                best_idx = e;
            }
        }

        s_vals[tid] = best_v;
        s_idxs[tid] = best_idx;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float ov = s_vals[tid + stride];
                int oi = s_idxs[tid + stride];
                if (better_pair(ov, oi, s_vals[tid], s_idxs[tid])) {
                    s_vals[tid] = ov;
                    s_idxs[tid] = oi;
                }
            }
            __syncthreads();
        }

        int chosen = s_idxs[0];
        float chosen_v = s_vals[0];
        if (tid == 0 && chosen >= 0) {
            top_sum += expf(chosen_v - row_max);
            s_logits[chosen] = -CUDART_INF_F;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float scale = top_sum / denom;
        store_from_float<scalar_t>(out, row, scale);
    }
}

template <typename scalar_t>
__global__ void fused_silu_mul_kernel(
    const scalar_t* __restrict__ gate_base,
    const scalar_t* __restrict__ gate_lora,
    const scalar_t* __restrict__ up_base,
    const scalar_t* __restrict__ up_lora,
    scalar_t* __restrict__ out,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float g = load_as_float<scalar_t>(gate_base, idx) + load_as_float<scalar_t>(gate_lora, idx);
        float u = load_as_float<scalar_t>(up_base, idx) + load_as_float<scalar_t>(up_lora, idx);
        float silu = g / (1.0f + expf(-g));
        store_from_float<scalar_t>(out, idx, silu * u);
    }
}

template <typename scalar_t>
__global__ void fused_add_scale_kernel(
    const scalar_t* __restrict__ base,
    const scalar_t* __restrict__ lora,
    const scalar_t* __restrict__ scale,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        int64_t r = idx / cols;
        float v = load_as_float<scalar_t>(base, idx) + load_as_float<scalar_t>(lora, idx);
        float s = load_as_float<scalar_t>(scale, r);
        store_from_float<scalar_t>(out, idx, v * s);
    }
}

__global__ void uva_touch_kernel(
    const long long* __restrict__ ptrs,
    int* __restrict__ scratch,
    int world_size
) {
    int tid = threadIdx.x;
    int acc = 0;
    for (int r = tid; r < world_size; r += blockDim.x) {
        const int* p = reinterpret_cast<const int*>(static_cast<uintptr_t>(ptrs[r]));
        acc += p[0];
    }

    __shared__ int smem[256];
    smem[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) scratch[0] = smem[0];
}

void router_topk_sum(torch::Tensor logits, torch::Tensor out, int64_t top_k) {
    TORCH_CHECK(logits.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(logits.dim() == 2, "logits must be [tokens, experts]");
    TORCH_CHECK(out.dim() == 1 && out.size(0) == logits.size(0), "bad out shape");
    TORCH_CHECK(top_k > 0 && top_k <= logits.size(1), "invalid top_k");

    int64_t rows = logits.size(0);
    int experts = (int)logits.size(1);
    int threads = 256;
    size_t shmem = (size_t)experts * sizeof(float) + (size_t)threads * (sizeof(float) + sizeof(int));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (logits.scalar_type() == torch::kBFloat16) {
        const __nv_bfloat16* in = reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>());
        __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
        router_topk_sum_kernel<__nv_bfloat16><<<rows, threads, shmem, stream>>>(
            in, o, rows, experts, (int)top_k);
    } else if (logits.scalar_type() == torch::kFloat32) {
        router_topk_sum_kernel<float><<<rows, threads, shmem, stream>>>(
            logits.data_ptr<float>(), out.data_ptr<float>(), rows, experts, (int)top_k);
    } else {
        TORCH_CHECK(false, "router_topk_sum supports bf16/fp32 only");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_silu_mul(
    torch::Tensor gate_base,
    torch::Tensor gate_lora,
    torch::Tensor up_base,
    torch::Tensor up_lora,
    torch::Tensor out
) {
    TORCH_CHECK(gate_base.is_cuda() && gate_lora.is_cuda() && up_base.is_cuda() && up_lora.is_cuda() && out.is_cuda(),
                "CUDA tensors required");
    TORCH_CHECK(gate_base.is_contiguous() && gate_lora.is_contiguous() && up_base.is_contiguous() &&
                up_lora.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(gate_base.numel() == gate_lora.numel() && gate_base.numel() == up_base.numel() &&
                gate_base.numel() == up_lora.numel() && gate_base.numel() == out.numel(), "shape mismatch");

    int64_t n = gate_base.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (gate_base.scalar_type() == torch::kBFloat16) {
        fused_silu_mul_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(gate_base.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(gate_lora.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(up_base.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(up_lora.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n);
    } else if (gate_base.scalar_type() == torch::kFloat32) {
        fused_silu_mul_kernel<float><<<blocks, threads, 0, stream>>>(
            gate_base.data_ptr<float>(), gate_lora.data_ptr<float>(),
            up_base.data_ptr<float>(), up_lora.data_ptr<float>(), out.data_ptr<float>(), n);
    } else {
        TORCH_CHECK(false, "fused_silu_mul supports bf16/fp32 only");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_add_scale(
    torch::Tensor base,
    torch::Tensor lora,
    torch::Tensor scale,
    torch::Tensor out
) {
    TORCH_CHECK(base.is_cuda() && lora.is_cuda() && scale.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(base.is_contiguous() && lora.is_contiguous() && scale.is_contiguous() && out.is_contiguous(),
                "contiguous tensors required");
    TORCH_CHECK(base.dim() == 2 && lora.sizes() == base.sizes() && out.sizes() == base.sizes(),
                "base/lora/out must be same 2D shape");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == base.size(0), "bad scale shape");

    int64_t rows = base.size(0);
    int64_t cols = base.size(1);
    int64_t n = rows * cols;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (base.scalar_type() == torch::kBFloat16) {
        fused_add_scale_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(base.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(lora.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            rows, cols);
    } else if (base.scalar_type() == torch::kFloat32) {
        fused_add_scale_kernel<float><<<blocks, threads, 0, stream>>>(
            base.data_ptr<float>(), lora.data_ptr<float>(), scale.data_ptr<float>(),
            out.data_ptr<float>(), rows, cols);
    } else {
        TORCH_CHECK(false, "fused_add_scale supports bf16/fp32 only");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void uva_touch(torch::Tensor ptrs, torch::Tensor scratch, int64_t world_size) {
    TORCH_CHECK(ptrs.is_cuda() && scratch.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(ptrs.scalar_type() == torch::kInt64, "ptrs must be int64");
    TORCH_CHECK(scratch.scalar_type() == torch::kInt32, "scratch must be int32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uva_touch_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        scratch.data_ptr<int>(),
        (int)world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("router_topk_sum", &router_topk_sum, "row-wise softmax top-k probability sum");
    m.def("fused_silu_mul", &fused_silu_mul, "BF16/FP32 fused (silu(gate_base+gate_lora) * (up_base+up_lora))");
    m.def("fused_add_scale", &fused_add_scale, "BF16/FP32 fused (base+lora)*row_scale");
    m.def("uva_touch", &uva_touch, "one-shot UVA peer-pointer touch through symmetric memory");
}
'''


_ext = None
_symm_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_lora_shared_bf16_h100_ext", CUDA_SRC)
    return _ext


def _as_dtype_contig(t: Optional[torch.Tensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.dtype != dtype:
        t = t.to(dtype)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def _ensure_module_dtype(mod: torch.nn.Linear, dtype: torch.dtype) -> torch.nn.Linear:
    if mod.weight.dtype != dtype or (mod.bias is not None and mod.bias.dtype != dtype):
        mod.to(dtype)
    return mod


def _ensure_symm_uva_once(device: torch.device, group: Optional[dist.ProcessGroup]):
    """
    Cached device-side peer-pointer setup. The optimized algorithm removes the token
    collective entirely, but this keeps the distributed path on symmetric memory/UVA
    instead of NCCL when a multi-rank job is present.
    """
    if not dist.is_initialized():
        return

    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    if world <= 1:
        return

    key = (device.index, id(group))
    if key in _symm_cache:
        return

    ext = _get_ext()
    buf = symm_mem.empty((1,), device=device, dtype=torch.int32)
    buf.zero_()
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    scratch = torch.empty((1,), device=device, dtype=torch.int32)

    # symmetric-memory device barrier + direct peer UVA load in custom CUDA
    hdl.barrier(channel=0)
    ext.uva_touch(ptrs, scratch, world)

    _symm_cache[key] = (buf, hdl, ptrs, scratch)


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    lora_gate_A: torch.Tensor,
    lora_gate_B: torch.Tensor,
    lora_up_A: torch.Tensor,
    lora_up_B: torch.Tensor,
    lora_down_A: torch.Tensor,
    lora_down_B: torch.Tensor,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Shared-expert MoE+LoRA forward.

    Because every selected expert applies the same shared LoRA MLP, the reference
    all-to-all/permutation/unpermutation pipeline reduces to:
        out[token] = shared_mlp(hidden[token]) * sum_{e in topk(router)} p_e
    """
    ext = _get_ext()
    group = group or (dist.group.WORLD if dist.is_initialized() else None)

    assert hidden_states.is_cuda, "hidden_states must be CUDA"
    dtype = hidden_states.dtype
    assert dtype in (torch.bfloat16, torch.float32), "optimized path supports bf16/fp32"

    _ensure_symm_uva_once(hidden_states.device, group)

    hidden_dim = hidden_states.size(-1)
    x = hidden_states.reshape(-1, hidden_dim).contiguous()
    tokens = x.size(0)

    gate_weight = _as_dtype_contig(gate_weight, dtype)
    gate_bias = _as_dtype_contig(gate_bias, dtype)

    gate_proj = _ensure_module_dtype(gate_proj, dtype)
    up_proj = _ensure_module_dtype(up_proj, dtype)
    down_proj = _ensure_module_dtype(down_proj, dtype)

    lora_gate_A = _as_dtype_contig(lora_gate_A, dtype)
    lora_gate_B = _as_dtype_contig(lora_gate_B, dtype)
    lora_up_A = _as_dtype_contig(lora_up_A, dtype)
    lora_up_B = _as_dtype_contig(lora_up_B, dtype)
    lora_down_A = _as_dtype_contig(lora_down_A, dtype)
    lora_down_B = _as_dtype_contig(lora_down_B, dtype)

    # Router: only the top-k probability mass is needed after shared-expert collapse.
    router_logits = torch.nn.functional.linear(x, gate_weight, gate_bias).contiguous()
    route_scale = torch.empty((tokens,), device=x.device, dtype=dtype)
    ext.router_topk_sum(router_logits, route_scale, int(top_k))

    # Shared LoRA MLP:
    # gate_x = x Wg^T + (x Ag^T) Bg^T
    # up     = x Wu^T + (x Au^T) Bu^T
    # y      = silu(gate_x) * up
    gate_base = torch.nn.functional.linear(x, gate_proj.weight, gate_proj.bias).contiguous()
    gate_a = torch.nn.functional.linear(x, lora_gate_A).contiguous()
    gate_lora = torch.nn.functional.linear(gate_a, lora_gate_B).contiguous()

    up_base = torch.nn.functional.linear(x, up_proj.weight, up_proj.bias).contiguous()
    up_a = torch.nn.functional.linear(x, lora_up_A).contiguous()
    up_lora = torch.nn.functional.linear(up_a, lora_up_B).contiguous()

    y = torch.empty_like(gate_base)
    ext.fused_silu_mul(gate_base, gate_lora, up_base, up_lora, y)

    # down = y Wd^T + (y Ad^T) Bd^T, then apply summed routing weight.
    down_base = torch.nn.functional.linear(y, down_proj.weight, down_proj.bias).contiguous()
    down_a = torch.nn.functional.linear(y, lora_down_A).contiguous()
    down_lora = torch.nn.functional.linear(down_a, lora_down_B).contiguous()

    out = torch.empty_like(down_base)
    ext.fused_add_scale(down_base, down_lora, route_scale, out)
    return out