from typing import Optional, Tuple
import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).dtype() == torch::kBFloat16, #x " must be bfloat16")

// -----------------------------------------------------------------------------
// Small utilities
// -----------------------------------------------------------------------------

__device__ __forceinline__ float bf16_to_f32(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(const float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ uint32_t sys_load_u32(const uint32_t* addr) {
    uint32_t v;
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(addr)
                 : "memory");
    return v;
}

__device__ __forceinline__ void sys_store_u32(uint32_t* addr, uint32_t v) {
    asm volatile("st.global.release.sys.u32 [%0], %1;"
                 :
                 : "l"(addr), "r"(v)
                 : "memory");
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    return smem[0];
}

// -----------------------------------------------------------------------------
// BF16 copy + PP signal kernels
// -----------------------------------------------------------------------------

__global__ void copy_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void pp_wait_signal_kernel(uint64_t local_signal_base, int slot) {
    if (threadIdx.x == 0) {
        uint32_t* p = reinterpret_cast<uint32_t*>(local_signal_base) + slot;
        while (true) {
            uint32_t v = sys_load_u32(p);
            if (v == 1u) {
                sys_store_u32(p, 0u);
                break;
            }
            __nanosleep(128);
        }
    }
}

__global__ void pp_signal_kernel(uint64_t remote_signal_base, int slot) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        uint32_t* p = reinterpret_cast<uint32_t*>(remote_signal_base) + slot;
        sys_store_u32(p, 1u);
    }
}

// -----------------------------------------------------------------------------
// Naive BF16 GEMM: C[M,N] = A[M,K] @ W[N,K]^T
// Accumulates in FP32, stores BF16.
// -----------------------------------------------------------------------------

__global__ void linear_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ C,
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        acc += bf16_to_f32(A[m * K + k]) * bf16_to_f32(W[n * K + k]);
    }
    C[m * N + n] = f32_to_bf16(acc);
}

// -----------------------------------------------------------------------------
// Pack K/V from qkv[B,S,3,H,Dh] into symmetric kv[2,B,S,H,Dh]
// -----------------------------------------------------------------------------

__global__ void pack_kv_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ kv,
    int B,
    int S,
    int H,
    int Dh
) {
    int64_t n = (int64_t)B * S * H * Dh;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int d = idx % Dh;
        int t = idx / Dh;
        int h = t % H;
        t /= H;
        int s = t % S;
        int b = t / S;

        int64_t qkv_k_idx = (((((int64_t)b * S + s) * 3 + 1) * H + h) * Dh + d);
        int64_t qkv_v_idx = (((((int64_t)b * S + s) * 3 + 2) * H + h) * Dh + d);

        int64_t kv_k_idx = (((((int64_t)0 * B + b) * S + s) * H + h) * Dh + d);
        int64_t kv_v_idx = (((((int64_t)1 * B + b) * S + s) * H + h) * Dh + d);

        kv[kv_k_idx] = qkv[qkv_k_idx];
        kv[kv_v_idx] = qkv[qkv_v_idx];
    }
}

// -----------------------------------------------------------------------------
// CP attention by direct symmetric-memory UVA remote loads.
//
// qkv local: [B,S,3,H,Dh]
// each kv shard: [2,B,S,H,Dh]
// out: [B,S,H,Dh]
//
// Causal semantics match the reference ring:
//   rank r attends local causal block and all full key blocks from ranks < r.
// -----------------------------------------------------------------------------

__global__ void cp_attention_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const int64_t* __restrict__ kv_ptrs,
    __nv_bfloat16* __restrict__ out,
    int B,
    int S,
    int H,
    int Dh,
    float scale,
    int causal,
    int cp_rank,
    int cp_world
) {
    int row = blockIdx.x;
    int qs = row % S;
    int tmp = row / S;
    int h = tmp % H;
    int b = tmp / H;
    int tid = threadIdx.x;

    const int64_t q_base = (((((int64_t)b * S + qs) * 3 + 0) * H + h) * Dh);

    float max_score = -FLT_MAX;

    // Pass 1: max.
    for (int kr = 0; kr < cp_world; ++kr) {
        if (causal && kr > cp_rank) continue;

        const __nv_bfloat16* kv = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)kv_ptrs[kr]);

        for (int ks = 0; ks < S; ++ks) {
            if (causal && kr == cp_rank && ks > qs) continue;

            float partial = 0.0f;
            int64_t k_base = (((((int64_t)0 * B + b) * S + ks) * H + h) * Dh);
            for (int d = tid; d < Dh; d += blockDim.x) {
                partial += bf16_to_f32(qkv[q_base + d]) * bf16_to_f32(kv[k_base + d]);
            }
            float dot = block_sum(partial) * scale;
            max_score = fmaxf(max_score, dot);
        }
    }

    // Pass 2: denominator.
    float denom = 0.0f;
    for (int kr = 0; kr < cp_world; ++kr) {
        if (causal && kr > cp_rank) continue;

        const __nv_bfloat16* kv = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)kv_ptrs[kr]);

        for (int ks = 0; ks < S; ++ks) {
            if (causal && kr == cp_rank && ks > qs) continue;

            float partial = 0.0f;
            int64_t k_base = (((((int64_t)0 * B + b) * S + ks) * H + h) * Dh);
            for (int d = tid; d < Dh; d += blockDim.x) {
                partial += bf16_to_f32(qkv[q_base + d]) * bf16_to_f32(kv[k_base + d]);
            }
            float dot = block_sum(partial) * scale;
            denom += expf(dot - max_score);
        }
    }
    denom = fmaxf(denom, 1.0e-20f);

    // Pass 3: output. All threads participate in reductions for every output-Dh pass.
    int passes = (Dh + blockDim.x - 1) / blockDim.x;
    for (int pass = 0; pass < passes; ++pass) {
        int od = pass * blockDim.x + tid;
        float acc = 0.0f;

        for (int kr = 0; kr < cp_world; ++kr) {
            if (causal && kr > cp_rank) continue;

            const __nv_bfloat16* kv = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)kv_ptrs[kr]);

            for (int ks = 0; ks < S; ++ks) {
                if (causal && kr == cp_rank && ks > qs) continue;

                float partial = 0.0f;
                int64_t k_base = (((((int64_t)0 * B + b) * S + ks) * H + h) * Dh);
                for (int d = tid; d < Dh; d += blockDim.x) {
                    partial += bf16_to_f32(qkv[q_base + d]) * bf16_to_f32(kv[k_base + d]);
                }
                float dot = block_sum(partial) * scale;
                float p = expf(dot - max_score) / denom;

                if (od < Dh) {
                    int64_t v_idx = (((((int64_t)1 * B + b) * S + ks) * H + h) * Dh + od);
                    acc += p * bf16_to_f32(kv[v_idx]);
                }
            }
        }

        if (od < Dh) {
            int64_t out_idx = ((((int64_t)b * S + qs) * H + h) * Dh + od);
            out[out_idx] = f32_to_bf16(acc);
        }
    }
}

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------

void launch_copy_bf16(torch::Tensor src, torch::Tensor dst, int64_t n) {
    CHECK_CUDA(src); CHECK_CUDA(dst);
    CHECK_CONTIGUOUS(src); CHECK_CONTIGUOUS(dst);
    CHECK_BF16(src); CHECK_BF16(dst);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    blocks = blocks > 65535 ? 65535 : blocks;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pp_wait(uint64_t local_signal_base, int slot) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pp_wait_signal_kernel<<<1, 32, 0, stream>>>(local_signal_base, slot);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pp_signal(uint64_t remote_signal_base, int slot) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pp_signal_kernel<<<1, 32, 0, stream>>>(remote_signal_base, slot);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_linear_bf16(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor C,
    int64_t M,
    int64_t K,
    int64_t N
) {
    CHECK_CUDA(A); CHECK_CUDA(W); CHECK_CUDA(C);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(W); CHECK_CONTIGUOUS(C);
    CHECK_BF16(A); CHECK_BF16(W); CHECK_BF16(C);

    dim3 threads(16, 16);
    dim3 blocks((unsigned int)((N + 15) / 16), (unsigned int)((M + 15) / 16));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    linear_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr<at::BFloat16>()),
        M, K, N
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pack_kv(
    torch::Tensor qkv,
    torch::Tensor kv,
    int B,
    int S,
    int H,
    int Dh
) {
    CHECK_CUDA(qkv); CHECK_CUDA(kv);
    CHECK_CONTIGUOUS(qkv); CHECK_CONTIGUOUS(kv);
    CHECK_BF16(qkv); CHECK_BF16(kv);

    int64_t n = (int64_t)B * S * H * Dh;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    blocks = blocks > 65535 ? 65535 : blocks;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    pack_kv_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(kv.data_ptr<at::BFloat16>()),
        B, S, H, Dh
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_cp_attention_bf16(
    torch::Tensor qkv,
    torch::Tensor kv_ptrs,
    torch::Tensor out,
    int B,
    int S,
    int H,
    int Dh,
    double scale,
    bool causal,
    int cp_rank,
    int cp_world
) {
    CHECK_CUDA(qkv); CHECK_CUDA(kv_ptrs); CHECK_CUDA(out);
    CHECK_CONTIGUOUS(qkv); CHECK_CONTIGUOUS(kv_ptrs); CHECK_CONTIGUOUS(out);
    CHECK_BF16(qkv); CHECK_BF16(out);
    TORCH_CHECK(kv_ptrs.dtype() == torch::kInt64, "kv_ptrs must be int64");

    int rows = B * S * H;
    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    cp_attention_bf16_kernel<<<rows, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>()),
        kv_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        B, S, H, Dh,
        (float)scale,
        causal ? 1 : 0,
        cp_rank,
        cp_world
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_copy_bf16", &launch_copy_bf16, "BF16 copy");
    m.def("launch_pp_wait", &launch_pp_wait, "PP wait signal");
    m.def("launch_pp_signal", &launch_pp_signal, "PP signal");
    m.def("launch_linear_bf16", &launch_linear_bf16, "BF16 linear");
    m.def("launch_pack_kv", &launch_pack_kv, "Pack KV to symmetric buffer");
    m.def("launch_cp_attention_bf16", &launch_cp_attention_bf16, "CP attention via symmetric-memory UVA");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attention_pp_bf16_symm_cuda_ext", CUDA_SRC)
    return _ext


_cp_cache = {}
_pp_cache = {}


def _group_key(group: Optional[dist.ProcessGroup]) -> int:
    return 0 if group is None else id(group)


def _get_cp_resources(
    cp_group: dist.ProcessGroup,
    shape: Tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
):
    key = (_group_key(cp_group), shape, dtype, device)
    if key in _cp_cache:
        return _cp_cache[key]

    B, S, H, Dh = shape
    kv = symm_mem.empty((2, B, S, H, Dh), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(kv, cp_group)
    kv_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    qkv = torch.empty((B, S, 3, H, Dh), device=device, dtype=dtype)
    ctx = torch.empty((B, S, H, Dh), device=device, dtype=dtype)

    res = {
        "kv": kv,
        "hdl": hdl,
        "kv_ptrs": kv_ptrs,
        "qkv": qkv,
        "ctx": ctx,
    }
    _cp_cache[key] = res
    return res


def _get_pp_resources(
    pp_group: dist.ProcessGroup,
    tensor_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
):
    pp_size = dist.get_world_size(pp_group)
    key = (_group_key(pp_group), tensor_shape, dtype, device)
    if key in _pp_cache:
        return _pp_cache[key]

    data = symm_mem.empty(tensor_shape, device=device, dtype=dtype)
    # One signal slot per local PP rank; sender writes receiver.signal[sender_rank] = 1.
    signal = symm_mem.empty((pp_size,), device=device, dtype=torch.int32)
    signal.zero_()

    data_hdl = symm_mem.rendezvous(data, pp_group)
    sig_hdl = symm_mem.rendezvous(signal, pp_group)

    recv_tmp = torch.empty(tensor_shape, device=device, dtype=dtype)

    res = {
        "data": data,
        "data_hdl": data_hdl,
        "signal": signal,
        "sig_hdl": sig_hdl,
        "recv_tmp": recv_tmp,
    }
    _pp_cache[key] = res
    return res


def _pp_recv_forward_cuda(
    pp_group: dist.ProcessGroup,
    tensor_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ext = _get_ext()
    res = _get_pp_resources(pp_group, tensor_shape, dtype, device)

    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)
    prev_rank = (pp_rank - 1) % pp_size

    local_signal_ptr = int(res["sig_hdl"].buffer_ptrs[pp_rank])
    remote_data_ptr = int(res["data_hdl"].buffer_ptrs[prev_rank])

    # Wait for predecessor's device-side signal, then copy predecessor's symmetric buffer.
    ext.launch_pp_wait(local_signal_ptr, prev_rank)

    # Build a temporary tensor shell pointing is not possible from Python, so the copy kernel
    # is implemented through the symmetric local resource by first using the sender's UVA ptr
    # in a small tensor-free path would be ideal. Here, use data_hdl pointer via a custom copy
    # shell fallback by copying through this rank's recv_tmp with a direct UVA copy kernel below.
    # The existing launch_copy_bf16 expects tensors, so use a tiny custom remote-copy path by
    # aliasing through a cached symmetric receive tensor is avoided; instead, sender writes data
    # into its own symmetric buffer and this receive copies with torch.empty output plus CUDA copy
    # from the exposed pointer through the extension's generic pointer-copy substitute:
    _remote_copy_bf16(remote_data_ptr, res["recv_tmp"], res["recv_tmp"].numel())

    return res["recv_tmp"]


def _pp_send_forward_cuda(
    pp_group: dist.ProcessGroup,
    tensor: torch.Tensor,
) -> None:
    ext = _get_ext()
    res = _get_pp_resources(pp_group, tuple(tensor.shape), tensor.dtype, tensor.device)

    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)
    next_rank = (pp_rank + 1) % pp_size

    ext.launch_copy_bf16(tensor.contiguous(), res["data"], tensor.numel())

    remote_signal_ptr = int(res["sig_hdl"].buffer_ptrs[next_rank])
    ext.launch_pp_signal(remote_signal_ptr, pp_rank)


# A small remote pointer copy extension is compiled separately to keep launch_copy_bf16 tensor-only
# checks simple.
REMOTE_COPY_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__global__ void remote_copy_bf16_kernel(
    uint64_t src_ptr,
    __nv_bfloat16* __restrict__ dst,
    int64_t n
) {
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)src_ptr);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

void remote_copy_bf16(uint64_t src_ptr, torch::Tensor dst, int64_t n) {
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(dst.dtype() == torch::kBFloat16, "dst must be bf16");

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    blocks = blocks > 65535 ? 65535 : blocks;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    remote_copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
        src_ptr,
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("remote_copy_bf16", &remote_copy_bf16, "Remote UVA bf16 copy");
}
'''

_remote_ext = None


def _get_remote_ext():
    global _remote_ext
    if _remote_ext is None:
        _remote_ext = compile_cuda_extension("ring_attention_pp_remote_copy_bf16_ext", REMOTE_COPY_SRC)
    return _remote_ext


def _remote_copy_bf16(src_ptr: int, dst: torch.Tensor, n: int) -> None:
    _get_remote_ext().remote_copy_bf16(int(src_ptr), dst, int(n))


def _attention_block_cuda(
    hidden: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    scale: float,
    causal: bool,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    ext = _get_ext()

    assert hidden.is_cuda and w_qkv.is_cuda and w_o.is_cuda
    assert hidden.dtype == torch.bfloat16
    assert w_qkv.dtype == torch.bfloat16
    assert w_o.dtype == torch.bfloat16

    hidden = hidden.contiguous()
    w_qkv = w_qkv.contiguous()
    w_o = w_o.contiguous()

    B, S, Din = hidden.shape
    head_dim = w_qkv.shape[0] // 3 // num_heads
    qkv_out_dim = 3 * num_heads * head_dim
    out_dim = w_o.shape[0]

    res = _get_cp_resources(
        cp_group,
        (int(B), int(S), int(num_heads), int(head_dim)),
        hidden.dtype,
        hidden.device,
    )

    qkv = res["qkv"]
    kv = res["kv"]
    ctx = res["ctx"]

    # QKV projection.
    ext.launch_linear_bf16(
        hidden,
        w_qkv,
        qkv,
        int(B * S),
        int(Din),
        int(qkv_out_dim),
    )

    # Publish K/V into CP symmetric memory and synchronize visibility across CP ranks.
    ext.launch_pack_kv(qkv, kv, int(B), int(S), int(num_heads), int(head_dim))
    res["hdl"].barrier(channel=0)

    cp_rank = dist.get_rank(cp_group)
    cp_world = dist.get_world_size(cp_group)

    # Direct remote-read attention over all visible CP KV shards.
    ext.launch_cp_attention_bf16(
        qkv,
        res["kv_ptrs"],
        ctx,
        int(B),
        int(S),
        int(num_heads),
        int(head_dim),
        float(scale),
        bool(causal),
        int(cp_rank),
        int(cp_world),
    )

    # Output projection.
    out = torch.empty((B, S, out_dim), device=hidden.device, dtype=hidden.dtype)
    ext.launch_linear_bf16(
        ctx,
        w_o,
        out,
        int(B * S),
        int(num_heads * head_dim),
        int(out_dim),
    )
    return out


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Megatron-style CP+PP ring attention forward, implemented with CUDA kernels and
    symmetric-memory UVA communication. Optimized path expects BF16 CUDA tensors.
    """
    assert hidden_states.is_cuda
    assert hidden_states.dtype == torch.bfloat16
    assert w_qkv.dtype == torch.bfloat16
    assert w_o.dtype == torch.bfloat16
    assert dist.is_initialized(), "distributed must be initialized for symmetric-memory path"

    _get_ext()
    _get_remote_ext()

    cp_group = cp_group or dist.group.WORLD

    head_dim = w_qkv.shape[0] // 3 // num_heads
    scale = float(softmax_scale if softmax_scale is not None else head_dim ** -0.5)

    is_first = True
    is_last = True
    if pp_group is not None and dist.get_world_size(pp_group) > 1:
        pp_rank = dist.get_rank(pp_group)
        pp_size = dist.get_world_size(pp_group)
        is_first = pp_rank == 0
        is_last = pp_rank == pp_size - 1

    if is_first:
        stage_input = hidden_states.contiguous()
    else:
        stage_input = _pp_recv_forward_cuda(
            pp_group,
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
        )

    stage_output = _attention_block_cuda(
        stage_input,
        w_qkv,
        w_o,
        int(num_heads),
        float(scale),
        bool(causal),
        cp_group,
    )

    if (not is_last) and pp_group is not None:
        _pp_send_forward_cuda(pp_group, stage_output)

    return stage_output