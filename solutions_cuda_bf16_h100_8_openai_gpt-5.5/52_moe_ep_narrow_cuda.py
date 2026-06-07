from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// -----------------------------------------------------------------------------
// Build deterministic expert-major token positions.
// Layout:
//   send_sym      [E, N, H]
//   expert_input  [EP, N, H] for the local expert, source-major fixed slots
//   expert_outsym [EP, N, H]
//   return_fixed  [E, N, H] on the original source rank
// -----------------------------------------------------------------------------

__global__ void build_pos_counts_kernel(
    const int64_t* __restrict__ selected,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ pos,
    int N,
    int K,
    int E
) {
    int e = blockIdx.x;
    if (e >= E || threadIdx.x != 0) return;

    int c = 0;
    for (int t = 0; t < N; ++t) {
        for (int k = 0; k < K; ++k) {
            int se = (int)selected[t * K + k];
            if (se == e) {
                pos[t * K + k] = c;
                ++c;
            }
        }
    }
    counts[e] = c;
}

__global__ void pack_send_f32_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ pos,
    float* __restrict__ send,
    int N,
    int H,
    int K,
    int E
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * K * H;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int h = idx % H;
        int tmp = idx / H;
        int k = tmp % K;
        int t = tmp / K;
        int e = (int)selected[t * K + k];
        int p = pos[t * K + k];
        send[((int64_t)e * N + p) * H + h] = hidden[(int64_t)t * H + h];
    }
}

__global__ void pack_send_bf16_kernel(
    const uint16_t* __restrict__ hidden,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ pos,
    uint16_t* __restrict__ send,
    int N,
    int H,
    int K,
    int E
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * K * H;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int h = idx % H;
        int tmp = idx / H;
        int k = tmp % K;
        int t = tmp / K;
        int e = (int)selected[t * K + k];
        int p = pos[t * K + k];
        send[((int64_t)e * N + p) * H + h] = hidden[(int64_t)t * H + h];
    }
}

__global__ void gather_pre_f32_kernel(
    const int64_t* __restrict__ count_ptrs,
    const int64_t* __restrict__ send_ptrs,
    float* __restrict__ expert_in,
    int rank,
    int EP,
    int N,
    int H
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)EP * N * H;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int h = idx % H;
        int row = idx / H;
        int j = row % N;
        int src = row / N;

        const int32_t* cptr = reinterpret_cast<const int32_t*>((uintptr_t)count_ptrs[src]);
        int c = cptr[rank];

        if (j < c) {
            const float* sptr = reinterpret_cast<const float*>((uintptr_t)send_ptrs[src]);
            expert_in[idx] = sptr[((int64_t)rank * N + j) * H + h];
        } else {
            expert_in[idx] = 0.0f;
        }
    }
}

__global__ void gather_pre_bf16_kernel(
    const int64_t* __restrict__ count_ptrs,
    const int64_t* __restrict__ send_ptrs,
    uint16_t* __restrict__ expert_in,
    int rank,
    int EP,
    int N,
    int H
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)EP * N * H;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int h = idx % H;
        int row = idx / H;
        int j = row % N;
        int src = row / N;

        const int32_t* cptr = reinterpret_cast<const int32_t*>((uintptr_t)count_ptrs[src]);
        int c = cptr[rank];

        if (j < c) {
            const uint16_t* sptr = reinterpret_cast<const uint16_t*>((uintptr_t)send_ptrs[src]);
            expert_in[idx] = sptr[((int64_t)rank * N + j) * H + h];
        } else {
            expert_in[idx] = 0;
        }
    }
}

__global__ void gather_post_f32_kernel(
    const int64_t* __restrict__ count_ptrs,
    const int64_t* __restrict__ out_ptrs,
    float* __restrict__ return_fixed,
    int rank,
    int E,
    int N,
    int H
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)E * N * H;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int h = idx % H;
        int row = idx / H;
        int j = row % N;
        int e = row / N;

        const int32_t* local_counts =
            reinterpret_cast<const int32_t*>((uintptr_t)count_ptrs[rank]);
        int c = local_counts[e];

        if (j < c) {
            const float* optr = reinterpret_cast<const float*>((uintptr_t)out_ptrs[e]);
            return_fixed[idx] = optr[((int64_t)rank * N + j) * H + h];
        } else {
            return_fixed[idx] = 0.0f;
        }
    }
}

__global__ void gather_post_bf16_kernel(
    const int64_t* __restrict__ count_ptrs,
    const int64_t* __restrict__ out_ptrs,
    uint16_t* __restrict__ return_fixed,
    int rank,
    int E,
    int N,
    int H
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)E * N * H;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int h = idx % H;
        int row = idx / H;
        int j = row % N;
        int e = row / N;

        const int32_t* local_counts =
            reinterpret_cast<const int32_t*>((uintptr_t)count_ptrs[rank]);
        int c = local_counts[e];

        if (j < c) {
            const uint16_t* optr = reinterpret_cast<const uint16_t*>((uintptr_t)out_ptrs[e]);
            return_fixed[idx] = optr[((int64_t)rank * N + j) * H + h];
        } else {
            return_fixed[idx] = 0;
        }
    }
}

__global__ void final_unpermute_f32_kernel(
    const float* __restrict__ return_fixed,
    const float* __restrict__ weights,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ pos,
    float* __restrict__ out,
    int N,
    int H,
    int K
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * H;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int h = idx % H;
        int t = idx / H;
        float acc = 0.0f;

        for (int k = 0; k < K; ++k) {
            int e = (int)selected[t * K + k];
            int p = pos[t * K + k];
            float w = weights[t * K + k];
            float v = return_fixed[((int64_t)e * N + p) * H + h];
            acc += v * w;
        }
        out[idx] = acc;
    }
}

__global__ void final_unpermute_bf16_kernel(
    const __nv_bfloat16* __restrict__ return_fixed,
    const float* __restrict__ weights,
    const int64_t* __restrict__ selected,
    const int32_t* __restrict__ pos,
    __nv_bfloat16* __restrict__ out,
    int N,
    int H,
    int K
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * H;
    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int h = idx % H;
        int t = idx / H;
        float acc = 0.0f;

        for (int k = 0; k < K; ++k) {
            int e = (int)selected[t * K + k];
            int p = pos[t * K + k];
            float w = weights[t * K + k];
            float v = __bfloat162float(return_fixed[((int64_t)e * N + p) * H + h]);
            acc += v * w;
        }
        out[idx] = __float2bfloat16(acc);
    }
}

// -----------------------------------------------------------------------------
// Fused SiLU(gate) * up for expert MLP.
// -----------------------------------------------------------------------------

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float g = gate[i];
        float s = g / (1.0f + expf(-g));
        out[i] = s * up[i];
    }
}

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float s = g / (1.0f + expf(-g));
        out[i] = __float2bfloat16(s * u);
    }
}

static inline int blocks_for(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

void launch_build_pack(
    torch::Tensor hidden,
    torch::Tensor selected,
    torch::Tensor send,
    torch::Tensor counts,
    torch::Tensor pos,
    int N,
    int H,
    int K,
    int E,
    int dtype_enum
) {
    CHECK_CUDA(hidden); CHECK_CUDA(selected); CHECK_CUDA(send);
    CHECK_CUDA(counts); CHECK_CUDA(pos);
    CHECK_CONTIG(hidden); CHECK_CONTIG(selected); CHECK_CONTIG(send);
    CHECK_CONTIG(counts); CHECK_CONTIG(pos);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    build_pos_counts_kernel<<<E, 1, 0, stream>>>(
        selected.data_ptr<int64_t>(),
        counts.data_ptr<int32_t>(),
        pos.data_ptr<int32_t>(),
        N, K, E
    );

    int64_t total = (int64_t)N * K * H;
    int threads = 256;
    int blocks = blocks_for(total, threads);

    if (dtype_enum == 0) {
        pack_send_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(hidden.data_ptr<at::BFloat16>()),
            selected.data_ptr<int64_t>(),
            pos.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(send.data_ptr<at::BFloat16>()),
            N, H, K, E
        );
    } else {
        pack_send_f32_kernel<<<blocks, threads, 0, stream>>>(
            hidden.data_ptr<float>(),
            selected.data_ptr<int64_t>(),
            pos.data_ptr<int32_t>(),
            send.data_ptr<float>(),
            N, H, K, E
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_pre(
    torch::Tensor count_ptrs,
    torch::Tensor send_ptrs,
    torch::Tensor expert_in,
    int rank,
    int EP,
    int N,
    int H,
    int dtype_enum
) {
    CHECK_CUDA(count_ptrs); CHECK_CUDA(send_ptrs); CHECK_CUDA(expert_in);
    CHECK_CONTIG(count_ptrs); CHECK_CONTIG(send_ptrs); CHECK_CONTIG(expert_in);

    int64_t total = (int64_t)EP * N * H;
    int threads = 256;
    int blocks = blocks_for(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        gather_pre_bf16_kernel<<<blocks, threads, 0, stream>>>(
            count_ptrs.data_ptr<int64_t>(),
            send_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<uint16_t*>(expert_in.data_ptr<at::BFloat16>()),
            rank, EP, N, H
        );
    } else {
        gather_pre_f32_kernel<<<blocks, threads, 0, stream>>>(
            count_ptrs.data_ptr<int64_t>(),
            send_ptrs.data_ptr<int64_t>(),
            expert_in.data_ptr<float>(),
            rank, EP, N, H
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_post(
    torch::Tensor count_ptrs,
    torch::Tensor out_ptrs,
    torch::Tensor return_fixed,
    int rank,
    int E,
    int N,
    int H,
    int dtype_enum
) {
    CHECK_CUDA(count_ptrs); CHECK_CUDA(out_ptrs); CHECK_CUDA(return_fixed);
    CHECK_CONTIG(count_ptrs); CHECK_CONTIG(out_ptrs); CHECK_CONTIG(return_fixed);

    int64_t total = (int64_t)E * N * H;
    int threads = 256;
    int blocks = blocks_for(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        gather_post_bf16_kernel<<<blocks, threads, 0, stream>>>(
            count_ptrs.data_ptr<int64_t>(),
            out_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<uint16_t*>(return_fixed.data_ptr<at::BFloat16>()),
            rank, E, N, H
        );
    } else {
        gather_post_f32_kernel<<<blocks, threads, 0, stream>>>(
            count_ptrs.data_ptr<int64_t>(),
            out_ptrs.data_ptr<int64_t>(),
            return_fixed.data_ptr<float>(),
            rank, E, N, H
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_final_unpermute(
    torch::Tensor return_fixed,
    torch::Tensor weights_f32,
    torch::Tensor selected,
    torch::Tensor pos,
    torch::Tensor out,
    int N,
    int H,
    int K,
    int dtype_enum
) {
    CHECK_CUDA(return_fixed); CHECK_CUDA(weights_f32); CHECK_CUDA(selected);
    CHECK_CUDA(pos); CHECK_CUDA(out);
    CHECK_CONTIG(return_fixed); CHECK_CONTIG(weights_f32); CHECK_CONTIG(selected);
    CHECK_CONTIG(pos); CHECK_CONTIG(out);

    int64_t total = (int64_t)N * H;
    int threads = 256;
    int blocks = blocks_for(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        final_unpermute_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(return_fixed.data_ptr<at::BFloat16>()),
            weights_f32.data_ptr<float>(),
            selected.data_ptr<int64_t>(),
            pos.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            N, H, K
        );
    } else {
        final_unpermute_f32_kernel<<<blocks, threads, 0, stream>>>(
            return_fixed.data_ptr<float>(),
            weights_f32.data_ptr<float>(),
            selected.data_ptr<int64_t>(),
            pos.data_ptr<int32_t>(),
            out.data_ptr<float>(),
            N, H, K
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_silu_mul(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor out,
    int64_t n,
    int dtype_enum
) {
    CHECK_CUDA(gate); CHECK_CUDA(up); CHECK_CUDA(out);
    CHECK_CONTIG(gate); CHECK_CONTIG(up); CHECK_CONTIG(out);

    int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        silu_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n
        );
    } else {
        silu_mul_f32_kernel<<<blocks, threads, 0, stream>>>(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            out.data_ptr<float>(),
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_build_pack", &launch_build_pack, "MoE build counts/positions and pack fixed-slot send buffer");
    m.def("launch_gather_pre", &launch_gather_pre, "MoE pre-expert UVA gather");
    m.def("launch_gather_post", &launch_gather_post, "MoE post-expert UVA gather");
    m.def("launch_final_unpermute", &launch_final_unpermute, "MoE weighted final unpermute");
    m.def("launch_silu_mul", &launch_silu_mul, "Fused SiLU(gate) * up");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_ep_narrow_symm_uva_bf16_ext", CUDA_SRC)
    return _ext


_EP_SUBGROUP_CACHE: dict[tuple[int, int], None | list] = {}
_RESOURCE_CACHE: dict[tuple, tuple] = {}


def _resolve_ep_group_for_narrow_moe(num_experts: int) -> dist.ProcessGroup:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    ws = dist.get_world_size()
    rank = dist.get_rank()
    key = (ws, num_experts)

    if key not in _EP_SUBGROUP_CACHE:
        if num_experts >= ws:
            _EP_SUBGROUP_CACHE[key] = None
        elif ws % num_experts != 0:
            raise ValueError(
                f"narrow EP requires world_size ({ws}) % num_experts ({num_experts}) == 0"
            )
        else:
            groups = []
            for r in range(ws // num_experts):
                ranks = list(range(r * num_experts, (r + 1) * num_experts))
                groups.append(dist.new_group(ranks))
            _EP_SUBGROUP_CACHE[key] = groups

    entry = _EP_SUBGROUP_CACHE[key]
    if entry is None:
        return dist.group.WORLD
    return entry[rank // num_experts]


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    raise TypeError(f"supported hot-path dtypes are bfloat16 and float32, got {dtype}")


def _get_resources(
    *,
    group: dist.ProcessGroup,
    num_experts: int,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    dtype: torch.dtype,
    device: torch.device,
):
    ep_size = dist.get_world_size(group)
    ep_rank = dist.get_rank(group)
    replica_id = dist.get_rank() // max(1, num_experts)

    key = (
        replica_id,
        ep_size,
        ep_rank,
        num_experts,
        num_tokens,
        hidden_dim,
        top_k,
        dtype,
        device.index,
    )
    if key in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[key]

    # Symmetric peer-visible state.
    counts_sym = symm_mem.empty((num_experts,), device=device, dtype=torch.int32)
    counts_hdl = symm_mem.rendezvous(counts_sym, group)

    send_sym = symm_mem.empty(
        (num_experts * num_tokens, hidden_dim), device=device, dtype=dtype
    )
    send_hdl = symm_mem.rendezvous(send_sym, group)

    expert_out_sym = symm_mem.empty(
        (ep_size * num_tokens, hidden_dim), device=device, dtype=dtype
    )
    out_hdl = symm_mem.rendezvous(expert_out_sym, group)

    count_ptrs = torch.tensor(counts_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    send_ptrs = torch.tensor(send_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    out_ptrs = torch.tensor(out_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    # Local scratch; fixed-slot shapes avoid CPU split lists and ragged allocation.
    pos = torch.empty((num_tokens, top_k), device=device, dtype=torch.int32)
    expert_in = torch.empty((ep_size * num_tokens, hidden_dim), device=device, dtype=dtype)
    return_fixed = torch.empty((num_experts * num_tokens, hidden_dim), device=device, dtype=dtype)
    final_out = torch.empty((num_tokens, hidden_dim), device=device, dtype=dtype)

    res = (
        counts_sym,
        counts_hdl,
        send_sym,
        send_hdl,
        expert_out_sym,
        out_hdl,
        count_ptrs,
        send_ptrs,
        out_ptrs,
        pos,
        expert_in,
        return_fixed,
        final_out,
    )
    _RESOURCE_CACHE[key] = res
    return res


def _expert_forward_fast(
    x: torch.Tensor,
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    dtype_enum: int,
) -> torch.Tensor:
    gate = gate_proj(x).contiguous()
    up = up_proj(x).contiguous()
    fused = torch.empty_like(gate)
    _get_ext().launch_silu_mul(gate, up, fused, fused.numel(), dtype_enum)
    return down_proj(fused).contiguous()


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
    Fused narrow-EP MoE forward using symmetric-memory UVA token exchange.
    Assumes the narrow regime used by the task: world_size > num_experts and
    world_size % num_experts == 0, hence one local expert per rank inside the
    EP subgroup.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    ws = dist.get_world_size()
    if group is None or (ws > num_experts and dist.get_world_size(group) != num_experts):
        group = _resolve_ep_group_for_narrow_moe(num_experts)

    ep_size = dist.get_world_size(group)
    ep_rank = dist.get_rank(group)

    if ep_size != num_experts:
        raise RuntimeError(
            "This optimized narrow-EP path requires ep_group size == num_experts "
            f"(got ep_size={ep_size}, num_experts={num_experts})."
        )

    if hidden_states.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("optimized path supports bfloat16/float32 hidden states")

    ext = _get_ext()
    dtype_enum = _dtype_enum(hidden_states.dtype)

    hidden_dim = hidden_states.size(-1)
    hidden = hidden_states.reshape(-1, hidden_dim).contiguous()
    num_tokens = hidden.size(0)

    # Router: keep numerically identical PyTorch top-k/softmax semantics.
    router_logits = F.linear(hidden, gate_weight, gate_bias)
    routing_weights, selected_experts = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1
    )
    selected_experts = selected_experts.contiguous()
    routing_weights_f32 = routing_weights.float().contiguous()

    (
        counts_sym,
        counts_hdl,
        send_sym,
        send_hdl,
        expert_out_sym,
        out_hdl,
        count_ptrs,
        send_ptrs,
        out_ptrs,
        pos,
        expert_in,
        return_fixed,
        final_out,
    ) = _get_resources(
        group=group,
        num_experts=num_experts,
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        top_k=top_k,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    # Local deterministic expert-major packing into symmetric send buffer.
    ext.launch_build_pack(
        hidden,
        selected_experts,
        send_sym,
        counts_sym,
        pos,
        num_tokens,
        hidden_dim,
        top_k,
        num_experts,
        dtype_enum,
    )

    # Make counts + payload visible to peers, then gather local-expert input via UVA.
    counts_hdl.barrier(channel=0)
    send_hdl.barrier(channel=0)

    ext.launch_gather_pre(
        count_ptrs,
        send_ptrs,
        expert_in,
        ep_rank,
        ep_size,
        num_tokens,
        hidden_dim,
        dtype_enum,
    )

    # Shared local expert MLP; GEMMs use the backend tensor-core implementation,
    # with custom fused activation between projections.
    expert_outputs = _expert_forward_fast(
        expert_in,
        gate_proj,
        up_proj,
        down_proj,
        dtype_enum,
    )

    # Publish local expert results in symmetric memory for original source ranks.
    expert_out_sym.copy_(expert_outputs)
    out_hdl.barrier(channel=1)

    # Gather each expert's results back to this source rank and weighted unpermute.
    ext.launch_gather_post(
        count_ptrs,
        out_ptrs,
        return_fixed,
        ep_rank,
        num_experts,
        num_tokens,
        hidden_dim,
        dtype_enum,
    )

    ext.launch_final_unpermute(
        return_fixed,
        routing_weights_f32,
        selected_experts,
        pos,
        final_out,
        num_tokens,
        hidden_dim,
        top_k,
        dtype_enum,
    )

    return final_out