from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {                         \
  cudaError_t err = cudaGetLastError();                              \
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));          \
} while (0)
#endif

// -----------------------------------------------------------------------------
// BF16 row-major GEMM: C[M,N] = A[M,K] @ W[N,K]^T
// Small, dependency-free CUDA GEMM. Accumulates FP32, stores BF16.
// -----------------------------------------------------------------------------

template<int BM, int BN, int BK>
__global__ void bf16_linear_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ __nv_bfloat16 As[BM][BK];
    __shared__ __nv_bfloat16 Ws[BK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float acc = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {
        int ak = kt + threadIdx.x;
        if (row < M && ak < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + ak];
        } else {
            As[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        int wk = kt + threadIdx.y;
        if (col < N && wk < K) {
            Ws[threadIdx.y][threadIdx.x] = W[col * K + wk];
        } else {
            Ws[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < BK; ++e) {
            acc += __bfloat162float(As[threadIdx.y][e]) *
                   __bfloat162float(Ws[e][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

void linear_bf16(torch::Tensor A, torch::Tensor W, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda() && W.is_cuda() && C.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(W.dtype() == torch::kBFloat16, "W must be BF16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be BF16");
    TORCH_CHECK(A.is_contiguous() && W.is_contiguous() && C.is_contiguous(), "tensors must be contiguous");

    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)W.size(0);

    TORCH_CHECK(W.size(1) == K, "W shape mismatch");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape mismatch");

    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 32;

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    bf16_linear_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
        M, N, K
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -----------------------------------------------------------------------------
// Split QKV projection output [B*S, 3*H*D] into Q [B,S,H,D]
// and symmetric combined KV buffer [2,B,S,H,D].
// -----------------------------------------------------------------------------

__global__ void split_qkv_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ kv,
    int64_t total,
    int H,
    int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t hd = (int64_t)H * D;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t m = idx / hd;
        int64_t r = idx - m * hd;
        q[idx] = qkv[m * (3 * hd) + r];
        kv[idx] = qkv[m * (3 * hd) + hd + r];
        kv[total + idx] = qkv[m * (3 * hd) + 2 * hd + r];
    }
}

void split_qkv_bf16(
    torch::Tensor qkv,
    torch::Tensor q,
    torch::Tensor kv,
    int H,
    int D
) {
    TORCH_CHECK(qkv.is_cuda() && q.is_cuda() && kv.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(qkv.dtype() == torch::kBFloat16, "qkv must be BF16");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be BF16");
    TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv must be BF16");
    TORCH_CHECK(qkv.is_contiguous() && q.is_contiguous() && kv.is_contiguous(), "tensors must be contiguous");

    int64_t total = q.numel();
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    split_qkv_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(kv.data_ptr<at::BFloat16>()),
        total, H, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -----------------------------------------------------------------------------
// Online-softmax CP attention over UVA symmetric KV pointers.
// q/context: [B,S,H,D]
// each kv shard: [2,B,S,H,D] with K first, V second.
// causal CP semantics match the Megatron ring reference:
// rank r attends to CP shards <= r; local shard uses triangular mask.
// -----------------------------------------------------------------------------

__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    v = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

__global__ void cp_attention_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const long long* __restrict__ kv_ptrs,
    __nv_bfloat16* __restrict__ out,
    int B,
    int S,
    int H,
    int D,
    int cp_world,
    int cp_rank,
    float scale,
    int causal
) {
    extern __shared__ float smem[];
    float* qbuf = smem;
    float* acc = smem + D;

    __shared__ float s_m;
    __shared__ float s_l;
    __shared__ float s_alpha;
    __shared__ float s_beta;

    int row_id = blockIdx.x;
    int h = row_id % H;
    int tmp = row_id / H;
    int s_q = tmp % S;
    int b = tmp / S;

    int64_t local_base = (((int64_t)b * S + s_q) * H + h) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        qbuf[d] = __bfloat162float(q[local_base + d]);
        acc[d] = 0.0f;
    }

    if (threadIdx.x == 0) {
        s_m = -INFINITY;
        s_l = 0.0f;
        s_alpha = 0.0f;
        s_beta = 0.0f;
    }
    __syncthreads();

    int64_t shard_elems = (int64_t)B * S * H * D;

    for (int src = 0; src < cp_world; ++src) {
        if (causal && src > cp_rank) {
            continue;
        }

        const __nv_bfloat16* kv_base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)kv_ptrs[src]);
        const __nv_bfloat16* k_base = kv_base;
        const __nv_bfloat16* v_base = kv_base + shard_elems;

        for (int s_k = 0; s_k < S; ++s_k) {
            if (causal && src == cp_rank && s_k > s_q) {
                continue;
            }

            int64_t key_base = (((int64_t)b * S + s_k) * H + h) * D;

            float dot = 0.0f;
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                dot += qbuf[d] * __bfloat162float(k_base[key_base + d]);
            }

            dot = block_reduce_sum(dot);

            if (threadIdx.x == 0) {
                float score = dot * scale;
                float old_m = s_m;
                float old_l = s_l;
                float new_m = fmaxf(old_m, score);
                float alpha = (old_l == 0.0f) ? 0.0f : __expf(old_m - new_m);
                float beta = __expf(score - new_m);
                float new_l = old_l * alpha + beta;

                s_m = new_m;
                s_l = new_l;
                s_alpha = alpha;
                s_beta = beta;
            }
            __syncthreads();

            float alpha = s_alpha;
            float beta = s_beta;

            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                float vv = __bfloat162float(v_base[key_base + d]);
                acc[d] = acc[d] * alpha + beta * vv;
            }
            __syncthreads();
        }
    }

    float inv_l = 0.0f;
    if (s_l > 0.0f) inv_l = 1.0f / s_l;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[local_base + d] = __float2bfloat16(acc[d] * inv_l);
    }
}

void cp_attention_bf16(
    torch::Tensor q,
    torch::Tensor kv_ptrs,
    torch::Tensor out,
    int B,
    int S,
    int H,
    int D,
    int cp_world,
    int cp_rank,
    double scale,
    bool causal
) {
    TORCH_CHECK(q.is_cuda() && kv_ptrs.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be BF16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be BF16");
    TORCH_CHECK(kv_ptrs.dtype() == torch::kInt64, "kv_ptrs must be int64");
    TORCH_CHECK(q.is_contiguous() && out.is_contiguous() && kv_ptrs.is_contiguous(), "tensors must be contiguous");

    int rows = B * S * H;
    int threads = 128;
    if (D > 128) threads = 256;
    if (D > 256) threads = 512;

    size_t shmem = (size_t)2 * D * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    cp_attention_bf16_kernel<<<rows, threads, shmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const long long*>(kv_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        B, S, H, D, cp_world, cp_rank, (float)scale, causal ? 1 : 0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -----------------------------------------------------------------------------
// TP all-reduce SUM via UVA symmetric peer pointers.
// -----------------------------------------------------------------------------

__global__ void allreduce_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world) {
                const __nv_bfloat16* p =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[r]);
                sum += __bfloat162float(p[i]);
            }
        }
        out[i] = __float2bfloat16(sum);
    }
}

void allreduce_bf16(torch::Tensor ptrs, torch::Tensor out, int64_t n) {
    TORCH_CHECK(ptrs.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(ptrs.dtype() == torch::kInt64, "ptrs must be int64");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be BF16");
    TORCH_CHECK(ptrs.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");

    int world = (int)ptrs.size(0);
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        world,
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_bf16", &linear_bf16, "BF16 linear C=A@W.T");
    m.def("split_qkv_bf16", &split_qkv_bf16, "Split packed QKV into Q and symmetric KV");
    m.def("cp_attention_bf16", &cp_attention_bf16, "CP attention over symmetric UVA KV");
    m.def("allreduce_bf16", &allreduce_bf16, "TP all-reduce SUM over symmetric UVA pointers");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        # Per-rank extension name avoids concurrent build-directory races.
        r = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        _ext = compile_cuda_extension(f"ring_attention_tp_cuda_bf16_h100_r{r}", CUDA_SRC)
    return _ext


_cp_cache = {}
_tp_cache = {}
_tmp_cache = {}


def _group_key(group: dist.ProcessGroup) -> int:
    return id(group)


def _get_cp_resources(
    B: int,
    S: int,
    H: int,
    D: int,
    device: torch.device,
    group: dist.ProcessGroup,
):
    key = (_group_key(group), B, S, H, D, device)
    cached = _cp_cache.get(key)
    if cached is not None:
        return cached

    kv = symm_mem.empty((2, B, S, H, D), device=device, dtype=torch.bfloat16)
    hdl = symm_mem.rendezvous(kv, group)

    q = torch.empty((B, S, H, D), device=device, dtype=torch.bfloat16)
    context = torch.empty((B, S, H, D), device=device, dtype=torch.bfloat16)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (kv, hdl, q, context, ptrs)
    _cp_cache[key] = cached
    return cached


def _get_tp_resources(
    shape,
    device: torch.device,
    group: dist.ProcessGroup,
):
    key = (_group_key(group), tuple(shape), device)
    cached = _tp_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty(shape, device=device, dtype=torch.bfloat16)
    hdl = symm_mem.rendezvous(buf, group)
    out = torch.empty(shape, device=device, dtype=torch.bfloat16)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (buf, hdl, out, ptrs)
    _tp_cache[key] = cached
    return cached


def _get_tmp(name: str, shape, device: torch.device):
    key = (name, tuple(shape), device)
    t = _tmp_cache.get(key)
    if t is None:
        t = torch.empty(shape, device=device, dtype=torch.bfloat16)
        _tmp_cache[key] = t
    return t


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Megatron-style CP+TP attention forward using custom CUDA kernels and
    symmetric-memory UVA communication. Optimized path expects BF16 CUDA tensors.
    """
    assert hidden_states.is_cuda and w_qkv.is_cuda and w_o.is_cuda
    assert hidden_states.dtype == torch.bfloat16
    assert w_qkv.dtype == torch.bfloat16
    assert w_o.dtype == torch.bfloat16
    assert dist.is_initialized(), "torch.distributed must be initialized"

    ext = _get_ext()

    tp_group = tp_group or dist.group.WORLD
    cp_group = cp_group or dist.group.WORLD

    tp_size = dist.get_world_size(tp_group)
    heads_local = num_heads // tp_size
    head_dim = w_qkv.shape[0] // 3 // heads_local
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    hidden_states = hidden_states.contiguous()
    w_qkv = w_qkv.contiguous()
    w_o = w_o.contiguous()

    B = int(hidden_states.shape[0])
    S = int(hidden_states.shape[1])
    hidden_size = int(hidden_states.shape[2])
    M = B * S

    # -------------------------------------------------------------------------
    # 1. Column-parallel QKV projection: [B*S, hidden] x [3*H*D, hidden]^T.
    # -------------------------------------------------------------------------
    hs2d = hidden_states.reshape(M, hidden_size)
    qkv_cols = int(w_qkv.shape[0])
    qkv = _get_tmp("qkv", (M, qkv_cols), hidden_states.device)
    ext.linear_bf16(hs2d, w_qkv, qkv)

    # -------------------------------------------------------------------------
    # 2. Publish K/V in CP symmetric memory, then compute attention by direct
    #    UVA reads from all visible CP shards.
    # -------------------------------------------------------------------------
    kv_symm, cp_hdl, q, context, cp_ptrs = _get_cp_resources(
        B, S, heads_local, head_dim, hidden_states.device, cp_group
    )

    ext.split_qkv_bf16(qkv, q, kv_symm, heads_local, head_dim)

    # Makes K/V writes visible to CP peers before the attention kernel reads UVA.
    cp_hdl.barrier(channel=0)

    ext.cp_attention_bf16(
        q,
        cp_ptrs,
        context,
        B,
        S,
        heads_local,
        head_dim,
        cp_hdl.world_size,
        cp_hdl.rank,
        float(softmax_scale),
        bool(causal),
    )

    # -------------------------------------------------------------------------
    # 3. Row-parallel output projection followed by custom TP all-reduce.
    # -------------------------------------------------------------------------
    context2d = context.reshape(M, heads_local * head_dim)
    partial = _get_tmp("partial_out", (M, int(w_o.shape[0])), hidden_states.device)
    ext.linear_bf16(context2d, w_o, partial)

    partial_3d = partial.reshape(B, S, int(w_o.shape[0]))

    if tp_size == 1:
        return partial_3d

    tp_buf, tp_hdl, final_out, tp_ptrs = _get_tp_resources(
        partial_3d.shape, hidden_states.device, tp_group
    )
    tp_buf.copy_(partial_3d)

    # Makes row-parallel partial output visible to TP peers.
    tp_hdl.barrier(channel=1)

    ext.allreduce_bf16(tp_ptrs, final_out, final_out.numel())
    return final_out