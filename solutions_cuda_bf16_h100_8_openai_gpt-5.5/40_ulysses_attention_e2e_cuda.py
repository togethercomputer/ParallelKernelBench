import math
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor
from torch.distributed import ProcessGroup

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#define MAX_HP 128
#define REDUCE_THREADS 256

__device__ __forceinline__ float bf16_to_f32(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(const float x) {
    return __float2bfloat16_rn(x);
}

__global__ void pack_qkv_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,   // [B,S,3,H,D]
    __nv_bfloat16* __restrict__ sym,         // q at 0, kv at q_elems
    int64_t q_elems,
    int B,
    int S,
    int H,
    int D
) {
    const int64_t total = q_elems * 3; // q + 2*kv logical copy work
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        if (idx < q_elems) {
            // q: [B,S,H,D] from component 0
            int64_t t = idx;
            int d = (int)(t % D); t /= D;
            int h = (int)(t % H); t /= H;
            int s = (int)(t % S); t /= S;
            int b = (int)t;

            int64_t src = (((((int64_t)b * S + s) * 3 + 0) * H + h) * D + d);
            sym[idx] = qkv[src];
        } else {
            // kv: [B,S,2H,D], interleaved k/v per head, from components 1/2
            int64_t kv_idx = idx - q_elems;
            int64_t t = kv_idx;
            int d = (int)(t % D); t /= D;
            int h2 = (int)(t % (2 * H)); t /= (2 * H);
            int s = (int)(t % S); t /= S;
            int b = (int)t;

            int h = h2 >> 1;
            int comp = 1 + (h2 & 1); // 1=k, 2=v
            int64_t src = (((((int64_t)b * S + s) * 3 + comp) * H + h) * D + d);
            sym[q_elems + kv_idx] = qkv[src];
        }
    }
}

__global__ void pre_a2a_bf16_kernel(
    const long long* __restrict__ base_ptrs,
    __nv_bfloat16* __restrict__ q_pre,       // [B,S*P,Hp,D]
    __nv_bfloat16* __restrict__ kv_pre,      // [B,S*P,2Hp,D]
    int64_t q_off_bytes,
    int64_t kv_off_bytes,
    int B,
    int S,
    int H,
    int D,
    int P,
    int rank
) {
    const int Hp = H / P;
    const int Sg = S * P;

    const int64_t q_pre_elems = (int64_t)B * Sg * Hp * D;
    const int64_t kv_pre_elems = (int64_t)B * Sg * (2 * Hp) * D;
    const int64_t total = q_pre_elems + kv_pre_elems;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        if (idx < q_pre_elems) {
            int64_t t = idx;
            int d = (int)(t % D); t /= D;
            int hloc = (int)(t % Hp); t /= Hp;
            int sg = (int)(t % Sg); t /= Sg;
            int b = (int)t;

            int src_rank = sg / S;
            int s = sg - src_rank * S;
            int h = rank * Hp + hloc;

            const __nv_bfloat16* remote_q =
                reinterpret_cast<const __nv_bfloat16*>(
                    (uintptr_t)base_ptrs[src_rank] + (uintptr_t)q_off_bytes);

            int64_t src = ((((int64_t)b * S + s) * H + h) * D + d);
            q_pre[idx] = remote_q[src];
        } else {
            int64_t o = idx - q_pre_elems;
            int64_t t = o;
            int d = (int)(t % D); t /= D;
            int h2loc = (int)(t % (2 * Hp)); t /= (2 * Hp);
            int sg = (int)(t % Sg); t /= Sg;
            int b = (int)t;

            int src_rank = sg / S;
            int s = sg - src_rank * S;
            int h2 = rank * (2 * Hp) + h2loc;

            const __nv_bfloat16* remote_kv =
                reinterpret_cast<const __nv_bfloat16*>(
                    (uintptr_t)base_ptrs[src_rank] + (uintptr_t)kv_off_bytes);

            int64_t src = ((((int64_t)b * S + s) * (2 * H) + h2) * D + d);
            kv_pre[o] = remote_kv[src];
        }
    }
}

__global__ void local_head_attention_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,     // [B,Sg,Hp,D]
    const __nv_bfloat16* __restrict__ kv,    // [B,Sg,2Hp,D]
    __nv_bfloat16* __restrict__ attn_sym,    // [B,Sg,Hp,D]
    int rows,                                // B*Sg
    int Hp,
    int D,
    float scale,
    int causal
) {
    __shared__ float red[REDUCE_THREADS];
    __shared__ float probs[MAX_HP];

    const int qi = blockIdx.x % Hp;
    const int row = blockIdx.x / Hp;
    const int tid = threadIdx.x;

    const __nv_bfloat16* qrow = q + ((int64_t)row * Hp + qi) * D;
    const __nv_bfloat16* kvrow = kv + (int64_t)row * (2 * Hp) * D;
    __nv_bfloat16* outrow = attn_sym + ((int64_t)row * Hp + qi) * D;

    for (int kj = 0; kj < Hp; ++kj) {
        float acc = 0.0f;
        if (!(causal && Hp > 1 && kj > qi)) {
            const __nv_bfloat16* krow = kvrow + (2 * kj) * D;
            for (int d = tid; d < D; d += blockDim.x) {
                acc += bf16_to_f32(qrow[d]) * bf16_to_f32(krow[d]);
            }
        }

        red[tid] = acc;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            probs[kj] = (causal && Hp > 1 && kj > qi) ? -INFINITY : red[0] * scale;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = -INFINITY;
        for (int j = 0; j < Hp; ++j) m = fmaxf(m, probs[j]);

        float denom = 0.0f;
        for (int j = 0; j < Hp; ++j) {
            float e = expf(probs[j] - m);
            probs[j] = e;
            denom += e;
        }

        float inv = 1.0f / denom;
        for (int j = 0; j < Hp; ++j) probs[j] *= inv;
    }
    __syncthreads();

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int kj = 0; kj < Hp; ++kj) {
            const __nv_bfloat16* vrow = kvrow + (2 * kj + 1) * D;
            acc += probs[kj] * bf16_to_f32(vrow[d]);
        }
        outrow[d] = f32_to_bf16(acc);
    }
}

__global__ void post_a2a_bf16_kernel(
    const long long* __restrict__ base_ptrs,
    __nv_bfloat16* __restrict__ out,         // [B,S,H,D]
    int64_t attn_off_bytes,
    int B,
    int S,
    int H,
    int D,
    int P,
    int rank
) {
    const int Hp = H / P;
    const int Sg = S * P;
    const int64_t total = (int64_t)B * S * H * D;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t t = idx;
        int d = (int)(t % D); t /= D;
        int h = (int)(t % H); t /= H;
        int s = (int)(t % S); t /= S;
        int b = (int)t;

        int owner = h / Hp;
        int hloc = h - owner * Hp;
        int sg = rank * S + s;

        const __nv_bfloat16* remote_attn =
            reinterpret_cast<const __nv_bfloat16*>(
                (uintptr_t)base_ptrs[owner] + (uintptr_t)attn_off_bytes);

        int64_t src = ((((int64_t)b * Sg + sg) * Hp + hloc) * D + d);
        out[idx] = remote_attn[src];
    }
}

static inline int pick_blocks(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

void pack_qkv_bf16(torch::Tensor qkv, torch::Tensor sym, int B, int S, int H, int D) {
    TORCH_CHECK(qkv.is_cuda() && sym.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(qkv.dtype() == torch::kBFloat16 && sym.dtype() == torch::kBFloat16, "BF16 required");
    TORCH_CHECK(qkv.is_contiguous() && sym.is_contiguous(), "contiguous tensors required");

    int64_t q_elems = (int64_t)B * S * H * D;
    int threads = 256;
    int blocks = pick_blocks(q_elems * 3, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_qkv_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(sym.data_ptr<at::BFloat16>()),
        q_elems, B, S, H, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pre_a2a_bf16(
    torch::Tensor ptrs,
    torch::Tensor q_pre,
    torch::Tensor kv_pre,
    int64_t q_off_bytes,
    int64_t kv_off_bytes,
    int B,
    int S,
    int H,
    int D,
    int P,
    int rank
) {
    TORCH_CHECK(ptrs.is_cuda() && q_pre.is_cuda() && kv_pre.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(q_pre.dtype() == torch::kBFloat16 && kv_pre.dtype() == torch::kBFloat16, "BF16 required");
    TORCH_CHECK(q_pre.is_contiguous() && kv_pre.is_contiguous(), "contiguous tensors required");

    const int Hp = H / P;
    const int Sg = S * P;
    int64_t qn = (int64_t)B * Sg * Hp * D;
    int64_t kvn = (int64_t)B * Sg * (2 * Hp) * D;
    int threads = 256;
    int blocks = pick_blocks(qn + kvn, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pre_a2a_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(q_pre.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(kv_pre.data_ptr<at::BFloat16>()),
        q_off_bytes, kv_off_bytes, B, S, H, D, P, rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void local_head_attention_bf16(
    torch::Tensor q_pre,
    torch::Tensor kv_pre,
    torch::Tensor sym,
    int64_t attn_off_elems,
    int B,
    int Sg,
    int Hp,
    int D,
    double scale,
    bool causal
) {
    TORCH_CHECK(q_pre.is_cuda() && kv_pre.is_cuda() && sym.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(q_pre.dtype() == torch::kBFloat16 && kv_pre.dtype() == torch::kBFloat16 && sym.dtype() == torch::kBFloat16, "BF16 required");
    TORCH_CHECK(q_pre.is_contiguous() && kv_pre.is_contiguous() && sym.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(Hp <= MAX_HP, "too many local heads for this kernel");

    int rows = B * Sg;
    dim3 grid(rows * Hp);
    dim3 block(REDUCE_THREADS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(sym.data_ptr<at::BFloat16>());
    local_head_attention_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_pre.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(kv_pre.data_ptr<at::BFloat16>()),
        base + attn_off_elems,
        rows, Hp, D, (float)scale, causal ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void post_a2a_bf16(
    torch::Tensor ptrs,
    torch::Tensor out,
    int64_t attn_off_bytes,
    int B,
    int S,
    int H,
    int D,
    int P,
    int rank
) {
    TORCH_CHECK(ptrs.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "BF16 required");
    TORCH_CHECK(out.is_contiguous(), "contiguous tensor required");

    int64_t n = (int64_t)B * S * H * D;
    int threads = 256;
    int blocks = pick_blocks(n, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    post_a2a_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        attn_off_bytes, B, S, H, D, P, rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_qkv_bf16", &pack_qkv_bf16, "Pack Q/KV into symmetric BF16 workspace");
    m.def("pre_a2a_bf16", &pre_a2a_bf16, "UVA pre all-to-all for Ulysses BF16");
    m.def("local_head_attention_bf16", &local_head_attention_bf16, "Local per-token head attention BF16");
    m.def("post_a2a_bf16", &post_a2a_bf16, "UVA post all-to-all for Ulysses BF16");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_attention_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _get_resources(
    B: int,
    S: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    group: ProcessGroup,
):
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    key = (B, S, num_heads, head_dim, dtype, device.index, world, rank, id(group))

    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    local_q_elems = B * S * num_heads * head_dim
    total_sym_elems = local_q_elems * 4  # q + 2*kv + attn

    sym = symm_mem.empty((total_sym_elems,), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(sym, group)

    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    hp = num_heads // world
    q_pre = torch.empty((B, S * world, hp, head_dim), device=device, dtype=dtype)
    kv_pre = torch.empty((B, S * world, 2 * hp, head_dim), device=device, dtype=dtype)
    post = torch.empty((B, S, num_heads, head_dim), device=device, dtype=dtype)

    q_off_elems = 0
    kv_off_elems = local_q_elems
    attn_off_elems = local_q_elems * 3
    elem_size = torch.empty((), device=device, dtype=dtype).element_size()

    res = {
        "sym": sym,
        "hdl": hdl,
        "ptrs": ptrs,
        "q_pre": q_pre,
        "kv_pre": kv_pre,
        "post": post,
        "q_off_bytes": q_off_elems * elem_size,
        "kv_off_bytes": kv_off_elems * elem_size,
        "attn_off_elems": attn_off_elems,
        "attn_off_bytes": attn_off_elems * elem_size,
    }
    _resource_cache[key] = res
    return res


def _torch_single_rank_fallback(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool,
) -> torch.Tensor:
    B, S, _ = hidden_states.shape
    head_dim = (w_qkv.shape[0] // 3) // num_heads
    qkv = F.linear(hidden_states, w_qkv).view(B, S, 3, num_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    if causal and q.size(2) > 1:
        h = scores.size(-1)
        mask = torch.triu(torch.ones(h, h, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).reshape(B, S, -1)
    return F.linear(out, w_o)


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    num_heads: int = 8,
    causal: bool = False,
) -> torch.Tensor:
    """
    Per-rank Ulysses attention block:
      qkv projection -> custom symmetric-memory pre-a2a -> custom BF16 local attention
      -> custom symmetric-memory post-a2a -> output projection.
    """
    if not dist.is_initialized():
        return _torch_single_rank_fallback(hidden_states, w_qkv, w_o, num_heads, causal)

    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world == 1 and hidden_states.dtype != torch.bfloat16:
        return _torch_single_rank_fallback(hidden_states, w_qkv, w_o, num_heads, causal)

    assert hidden_states.is_cuda and w_qkv.is_cuda and w_o.is_cuda
    assert hidden_states.dtype == torch.bfloat16, "optimized path expects BF16 hidden_states"
    assert w_qkv.dtype == torch.bfloat16 and w_o.dtype == torch.bfloat16, "optimized path expects BF16 weights"

    B, S_local, _ = hidden_states.shape
    head_dim = (w_qkv.shape[0] // 3) // num_heads

    assert (w_qkv.shape[0] // 3) == num_heads * head_dim
    assert num_heads % world == 0, "num_heads must be divisible by world_size"

    ext = _get_ext()

    # Tensor-core GEMM retained for dense projection; following layout is consumed by CUDA packer.
    qkv = F.linear(hidden_states, w_qkv).contiguous().view(B, S_local, 3, num_heads, head_dim)

    res = _get_resources(
        B,
        S_local,
        num_heads,
        head_dim,
        hidden_states.dtype,
        hidden_states.device,
        group,
    )

    sym = res["sym"]
    hdl = res["hdl"]
    ptrs = res["ptrs"]
    q_pre = res["q_pre"]
    kv_pre = res["kv_pre"]
    post = res["post"]

    local_q_elems = B * S_local * num_heads * head_dim
    hp = num_heads // world

    # Local pack into symmetric workspace: q segment and interleaved k/v segment.
    ext.pack_qkv_bf16(qkv, sym, B, S_local, num_heads, head_dim)

    # Make packed q/kv visible, then peer-load all ranks' sequence chunks for this rank's head shard.
    hdl.barrier(channel=0)

    ext.pre_a2a_bf16(
        ptrs,
        q_pre,
        kv_pre,
        int(res["q_off_bytes"]),
        int(res["kv_off_bytes"]),
        B,
        S_local,
        num_heads,
        head_dim,
        world,
        rank,
    )

    # Local attention over the gathered sequence / local-head shard, written directly to symmetric attn segment.
    ext.local_head_attention_bf16(
        q_pre,
        kv_pre,
        sym,
        int(res["attn_off_elems"]),
        B,
        S_local * world,
        hp,
        head_dim,
        float(head_dim ** -0.5),
        bool(causal),
    )

    # Make attention segment visible, then peer-load owner head shards back to local sequence.
    hdl.barrier(channel=1)

    ext.post_a2a_bf16(
        ptrs,
        post,
        int(res["attn_off_bytes"]),
        B,
        S_local,
        num_heads,
        head_dim,
        world,
        rank,
    )

    # Tensor-core output projection.
    out_in = post.view(B, S_local, num_heads * head_dim)
    return F.linear(out_in, w_o)