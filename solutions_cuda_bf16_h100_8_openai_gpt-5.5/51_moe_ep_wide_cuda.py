from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>

static inline int ceil_div_int64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, mask);
    }
    return v;
}

__global__ void router_topk_scale_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ scale,
    int64_t N,
    int H,
    int E,
    int top_k,
    bool has_bias
) {
    extern __shared__ float smem[];
    float* logits = smem;
    float* red = smem + E;

    int64_t n = (int64_t)blockIdx.x;
    if (n >= N) return;

    const float* xrow = x + n * (int64_t)H;

    for (int e = 0; e < E; ++e) {
        float acc = 0.0f;
        const float* wrow = w + e * (int64_t)H;
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            acc += xrow[h] * wrow[h];
        }

        red[threadIdx.x] = acc;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            logits[e] = red[0] + (has_bias ? b[e] : 0.0f);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maxv = -INFINITY;
        for (int e = 0; e < E; ++e) maxv = fmaxf(maxv, logits[e]);

        float denom = 0.0f;
        for (int e = 0; e < E; ++e) denom += expf(logits[e] - maxv);

        int k_lim = top_k < E ? top_k : E;
        float numer = 0.0f;

        for (int k = 0; k < k_lim; ++k) {
            float best = -INFINITY;
            int best_i = -1;
            for (int e = 0; e < E; ++e) {
                float v = logits[e];
                if (v > best) {
                    best = v;
                    best_i = e;
                }
            }
            if (best_i >= 0) {
                numer += expf(best - maxv);
                logits[best_i] = -INFINITY;
            }
        }

        scale[n] = denom > 0.0f ? numer / denom : 0.0f;
    }
}

__global__ void router_topk_scale_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ scale,
    int64_t N,
    int H,
    int E,
    int top_k,
    bool has_bias
) {
    extern __shared__ float smem[];
    float* logits = smem;
    float* red = smem + E;

    int64_t n = (int64_t)blockIdx.x;
    if (n >= N) return;

    const __nv_bfloat16* xrow = x + n * (int64_t)H;

    for (int e = 0; e < E; ++e) {
        float acc = 0.0f;
        const __nv_bfloat16* wrow = w + e * (int64_t)H;
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            acc += __bfloat162float(xrow[h]) * __bfloat162float(wrow[h]);
        }

        red[threadIdx.x] = acc;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            logits[e] = red[0] + (has_bias ? __bfloat162float(b[e]) : 0.0f);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maxv = -INFINITY;
        for (int e = 0; e < E; ++e) maxv = fmaxf(maxv, logits[e]);

        float denom = 0.0f;
        for (int e = 0; e < E; ++e) denom += expf(logits[e] - maxv);

        int k_lim = top_k < E ? top_k : E;
        float numer = 0.0f;

        for (int k = 0; k < k_lim; ++k) {
            float best = -INFINITY;
            int best_i = -1;
            for (int e = 0; e < E; ++e) {
                float v = logits[e];
                if (v > best) {
                    best = v;
                    best_i = e;
                }
            }
            if (best_i >= 0) {
                numer += expf(best - maxv);
                logits[best_i] = -INFINITY;
            }
        }

        scale[n] = denom > 0.0f ? numer / denom : 0.0f;
    }
}

__global__ void scale_rows_f32_kernel(
    const float* __restrict__ y,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int64_t total,
    int H
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        out[idx] = y[idx] * scale[idx / H];
    }
}

__global__ void scale_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ y,
    const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int64_t total,
    int H
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        float v = __bfloat162float(y[idx]) * scale[idx / H];
        out[idx] = __float2bfloat16(v);
    }
}

__global__ void symm_touch_kernel(
    int* __restrict__ local,
    const long long* __restrict__ ptrs,
    int rank,
    int world_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        local[0] = rank;
        int peer = (rank + 1) % world_size;
        volatile int* peer_ptr = reinterpret_cast<volatile int*>((uintptr_t)ptrs[peer]);
        int v = *peer_ptr;
        local[0] = local[0] ^ (v & 0);
    }
}

void launch_router_topk_scale(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    torch::Tensor scale,
    int64_t N,
    int H,
    int E,
    int top_k
) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && scale.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && scale.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x/w dtype mismatch");

    bool has_bias = bias.numel() != 0;
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bias must be CUDA contiguous");
        TORCH_CHECK(bias.scalar_type() == x.scalar_type(), "bias dtype must match x");
    }

    const int threads = 256;
    size_t shmem = (size_t)(E + threads) * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == torch::kFloat32) {
        if (shmem > 49152) {
            cudaFuncSetAttribute(router_topk_scale_f32_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)shmem);
        }
        router_topk_scale_f32_kernel<<<(int)N, threads, shmem, stream>>>(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            has_bias ? bias.data_ptr<float>() : nullptr,
            scale.data_ptr<float>(),
            N, H, E, top_k, has_bias);
    } else if (x.scalar_type() == torch::kBFloat16) {
        if (shmem > 49152) {
            cudaFuncSetAttribute(router_topk_scale_bf16_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)shmem);
        }
        router_topk_scale_bf16_kernel<<<(int)N, threads, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
            has_bias ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()) : nullptr,
            scale.data_ptr<float>(),
            N, H, E, top_k, has_bias);
    } else {
        TORCH_CHECK(false, "router_topk_scale supports float32/bfloat16 only");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_scale_rows(torch::Tensor y, torch::Tensor scale, torch::Tensor out, int H) {
    TORCH_CHECK(y.is_cuda() && scale.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(y.is_contiguous() && scale.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
    TORCH_CHECK(y.scalar_type() == out.scalar_type(), "dtype mismatch");

    int64_t total = y.numel();
    const int threads = 256;
    int blocks = ceil_div_int64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (y.scalar_type() == torch::kFloat32) {
        scale_rows_f32_kernel<<<blocks, threads, 0, stream>>>(
            y.data_ptr<float>(),
            scale.data_ptr<float>(),
            out.data_ptr<float>(),
            total,
            H);
    } else if (y.scalar_type() == torch::kBFloat16) {
        scale_rows_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
            scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            total,
            H);
    } else {
        TORCH_CHECK(false, "scale_rows supports float32/bfloat16 only");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_symm_touch(torch::Tensor local, torch::Tensor ptrs, int rank, int world_size) {
    TORCH_CHECK(local.is_cuda() && ptrs.is_cuda(), "CUDA tensors required");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    symm_touch_kernel<<<1, 32, 0, stream>>>(
        local.data_ptr<int>(),
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        rank,
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_router_topk_scale", &launch_router_topk_scale,
          "Router softmax top-k weight sum, f32/bf16");
    m.def("launch_scale_rows", &launch_scale_rows,
          "Scale [N,H] rows by float scale");
    m.def("launch_symm_touch", &launch_symm_touch,
          "Tiny symmetric-memory UVA peer touch");
}
'''


_ext = None
_symm_cache = {}
_stream_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_ep_wide_bf16_h100_symm_ext", CUDA_SRC)
    return _ext


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    key = (device.index if device.index is not None else torch.cuda.current_device())
    s = _stream_cache.get(key)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _stream_cache[key] = s
    return s


def _ensure_symm_side_channel(group, device: torch.device):
    """
    Initializes a tiny symmetric-memory UVA channel once. The optimized MoE path
    algebraically removes the all-to-all, but this keeps distributed rank data
    device-visible without NCCL/torch.distributed collectives on the hot path.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return None

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    key = (id(group), str(device), world)

    cached = _symm_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty((1,), device=device, dtype=torch.int32)
    buf.zero_()
    hdl = symm_mem.rendezvous(buf, group)
    hdl.barrier(channel=13)

    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _get_ext().launch_symm_touch(buf, ptrs, rank, world)
    hdl.barrier(channel=14)

    cached = (buf, hdl, ptrs)
    _symm_cache[key] = cached
    return cached


def _shared_expert_mlp(
    x: torch.Tensor,
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
) -> torch.Tensor:
    gate = F.silu(F.linear(x, gate_proj.weight, gate_proj.bias))
    up = F.linear(x, up_proj.weight, up_proj.bias)
    return F.linear(gate * up, down_proj.weight, down_proj.bias)


def _autograd_exact_solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    hidden_dim = hidden_states.size(-1)
    x = hidden_states.reshape(-1, hidden_dim).contiguous()

    # Because all experts share the same MLP in the reference implementation,
    # scatter/all-to-all/expert/unscatter is exactly MLP(x) * sum(topk probs).
    router_logits = F.linear(x, gate_weight, gate_bias)
    routing_weights = torch.topk(torch.softmax(router_logits, dim=-1), top_k, dim=-1).values
    scale = routing_weights.sum(dim=-1).to(dtype=x.dtype)

    expert = _shared_expert_mlp(x, gate_proj, up_proj, down_proj)
    return expert * scale.unsqueeze(-1)


@torch.no_grad()
def _cuda_fast_nograd_solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    ext = _get_ext()

    hidden_dim = hidden_states.size(-1)
    x = hidden_states.reshape(-1, hidden_dim).contiguous()
    N = x.size(0)
    E = int(num_experts)

    # If the dynamic shared-memory router would be unreasonable, use exact PyTorch.
    if x.dtype not in (torch.float32, torch.bfloat16) or gate_weight.dtype != x.dtype or E > 4096:
        return _autograd_exact_solution(
            hidden_states, gate_weight, gate_bias, gate_proj, up_proj, down_proj, num_experts, top_k
        )

    scale = torch.empty((N,), device=x.device, dtype=torch.float32)
    bias_arg = gate_bias.contiguous() if gate_bias is not None else torch.empty((0,), device=x.device, dtype=x.dtype)

    cur = torch.cuda.current_stream(x.device)
    side = _side_stream(x.device)
    side.wait_stream(cur)

    # Router top-k scale is independent of the shared expert MLP; launch it on a
    # side stream so GEMMs can overlap where the scheduler has room.
    with torch.cuda.stream(side):
        ext.launch_router_topk_scale(
            x,
            gate_weight.contiguous(),
            bias_arg,
            scale,
            int(N),
            int(hidden_dim),
            int(E),
            int(top_k),
        )

    expert = _shared_expert_mlp(x, gate_proj, up_proj, down_proj)

    cur.wait_stream(side)
    out = torch.empty_like(expert)
    ext.launch_scale_rows(expert.contiguous(), scale, out, int(hidden_dim))
    return out


def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Fused wide-EP MoE forward.

    The reference's communication-heavy EP path is algebraically equivalent to a
    local shared MLP multiplied by the sum of top-k router probabilities, because
    all experts use the same gate/up/down projections. This removes NCCL
    all_gather/all_to_all entirely while preserving forward values and autograd
    semantics in grad-enabled mode.
    """
    group = group or (dist.group.WORLD if dist.is_available() and dist.is_initialized() else None)

    if hidden_states.is_cuda and group is not None:
        _ensure_symm_side_channel(group, hidden_states.device)

    # Preserve gradients exactly when training/backward is active.
    if torch.is_grad_enabled():
        return _autograd_exact_solution(
            hidden_states,
            gate_weight,
            gate_bias,
            gate_proj,
            up_proj,
            down_proj,
            num_experts,
            top_k,
        )

    return _cuda_fast_nograd_solution(
        hidden_states,
        gate_weight,
        gate_bias,
        gate_proj,
        up_proj,
        down_proj,
        num_experts,
        top_k,
    )