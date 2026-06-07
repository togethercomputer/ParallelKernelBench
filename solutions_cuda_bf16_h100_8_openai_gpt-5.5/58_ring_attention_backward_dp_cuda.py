from typing import Optional, Tuple

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

template <typename T>
__device__ __forceinline__ float ld_val(const T* p, int64_t i) {
    return static_cast<float>(p[i]);
}

template <>
__device__ __forceinline__ float ld_val<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}

template <typename T>
__device__ __forceinline__ void st_val(T* p, int64_t i, float x) {
    p[i] = static_cast<T>(x);
}

template <>
__device__ __forceinline__ void st_val<__nv_bfloat16>(__nv_bfloat16* p, int64_t i, float x) {
    p[i] = __float2bfloat16(x);
}

__device__ __forceinline__ float load_lse(uint64_t base, int64_t i, int lse_dtype) {
    if (lse_dtype == 0) {
        const float* p = reinterpret_cast<const float*>(base);
        return p[i];
    } else {
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(base);
        return __bfloat162float(p[i]);
    }
}

template <typename T>
__global__ void rowdot_kernel(
    const T* __restrict__ pack,
    float* __restrict__ rowdot,
    int B,
    int S,
    int H,
    int D,
    int64_t n
) {
    __shared__ float sh[256];

    int row = blockIdx.x; // [B, S, H]
    int tid = threadIdx.x;

    int h = row % H;
    int tmp = row / H;
    int s = tmp % S;
    int b = tmp / S;

    int64_t base = (((int64_t)b * S + s) * H + h) * D;
    const T* dout = pack + 3 * n;
    const T* out  = pack + 4 * n;

    float acc = 0.0f;
    if (tid < D) {
        acc = ld_val<T>(dout, base + tid) * ld_val<T>(out, base + tid);
    }
    sh[tid] = acc;
    __syncthreads();

    #pragma unroll
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }

    if (tid == 0) {
        int64_t ridx = ((int64_t)b * H + h) * S + s; // [B,H,S]
        rowdot[ridx] = sh[0];
    }
}

template <typename T>
__global__ void dq_kernel(
    const uint64_t* __restrict__ pack_ptrs,
    const uint64_t* __restrict__ lse_ptrs,
    const uint64_t* __restrict__ row_ptrs,
    T* __restrict__ grad,
    int B,
    int S,
    int H,
    int D,
    int64_t n,
    float scale,
    bool causal,
    int cp_rank,
    int cp_world,
    int lse_dtype
) {
    __shared__ float sh_score[256];
    __shared__ float sh_dp[256];

    int row = blockIdx.x; // local [B,S,H] query row
    int tid = threadIdx.x;

    int h = row % H;
    int tmp = row / H;
    int sq = tmp % S;
    int b = tmp / S;

    uint64_t local_base_u = pack_ptrs[cp_rank];
    const T* local_pack = reinterpret_cast<const T*>(local_base_u);
    const T* qptr    = local_pack + 0 * n;
    const T* doutptr = local_pack + 3 * n;

    uint64_t local_lse_base = lse_ptrs[cp_rank];
    uint64_t local_row_base = row_ptrs[cp_rank];

    int64_t qbase = (((int64_t)b * S + sq) * H + h) * D;
    int64_t lidx = ((int64_t)b * H + h) * S + sq;
    float lse = load_lse(local_lse_base, lidx, lse_dtype);
    float rowdot = reinterpret_cast<const float*>(local_row_base)[lidx];

    float acc_dq = 0.0f;
    float q_d = 0.0f;
    if (tid < D) q_d = ld_val<T>(qptr, qbase + tid);

    for (int kr = 0; kr < cp_world; ++kr) {
        if (causal && kr > cp_rank) continue;

        const T* rpack = reinterpret_cast<const T*>(pack_ptrs[kr]);
        const T* kptr = rpack + 1 * n;
        const T* vptr = rpack + 2 * n;

        for (int sk = 0; sk < S; ++sk) {
            if (causal && kr == cp_rank && sk > sq) continue;

            int64_t kbase = (((int64_t)b * S + sk) * H + h) * D;

            float ps = 0.0f;
            float pd = 0.0f;
            if (tid < D) {
                float kd = ld_val<T>(kptr, kbase + tid);
                float vd = ld_val<T>(vptr, kbase + tid);
                float dod = ld_val<T>(doutptr, qbase + tid);
                ps = q_d * kd;
                pd = dod * vd;
            }
            sh_score[tid] = ps;
            sh_dp[tid] = pd;
            __syncthreads();

            #pragma unroll
            for (int off = 128; off > 0; off >>= 1) {
                if (tid < off) {
                    sh_score[tid] += sh_score[tid + off];
                    sh_dp[tid] += sh_dp[tid + off];
                }
                __syncthreads();
            }

            float prob = __expf(sh_score[0] * scale - lse);
            float ds = prob * (sh_dp[0] - rowdot);

            if (tid < D) {
                float kd = ld_val<T>(kptr, kbase + tid);
                acc_dq += ds * kd;
            }
            __syncthreads();
        }
    }

    if (tid < D) {
        st_val<T>(grad, qbase, acc_dq * scale);
    }
}

template <typename T>
__global__ void dkdv_kernel(
    const uint64_t* __restrict__ pack_ptrs,
    const uint64_t* __restrict__ lse_ptrs,
    const uint64_t* __restrict__ row_ptrs,
    T* __restrict__ grad,
    int B,
    int S,
    int H,
    int D,
    int64_t n,
    float scale,
    bool causal,
    int cp_rank,
    int cp_world,
    int lse_dtype
) {
    __shared__ float sh_score[256];
    __shared__ float sh_dp[256];

    int row = blockIdx.x; // local [B,S,H] key/value row
    int tid = threadIdx.x;

    int h = row % H;
    int tmp = row / H;
    int sk = tmp % S;
    int b = tmp / S;

    const T* local_pack = reinterpret_cast<const T*>(pack_ptrs[cp_rank]);
    const T* kptr_local = local_pack + 1 * n;
    const T* vptr_local = local_pack + 2 * n;

    int64_t kbase_local = (((int64_t)b * S + sk) * H + h) * D;

    float k_d = 0.0f;
    float v_d = 0.0f;
    if (tid < D) {
        k_d = ld_val<T>(kptr_local, kbase_local + tid);
        v_d = ld_val<T>(vptr_local, kbase_local + tid);
    }

    float acc_dk = 0.0f;
    float acc_dv = 0.0f;

    for (int qr = 0; qr < cp_world; ++qr) {
        if (causal && qr < cp_rank) continue;

        const T* qpack = reinterpret_cast<const T*>(pack_ptrs[qr]);
        const T* qptr = qpack + 0 * n;
        const T* doutptr = qpack + 3 * n;
        uint64_t q_lse_base = lse_ptrs[qr];
        const float* q_rowdot = reinterpret_cast<const float*>(row_ptrs[qr]);

        for (int sq = 0; sq < S; ++sq) {
            if (causal && qr == cp_rank && sq < sk) continue;

            int64_t qbase = (((int64_t)b * S + sq) * H + h) * D;
            int64_t lidx = ((int64_t)b * H + h) * S + sq;

            float ps = 0.0f;
            float pd = 0.0f;
            float q_d = 0.0f;
            float dout_d = 0.0f;

            if (tid < D) {
                q_d = ld_val<T>(qptr, qbase + tid);
                dout_d = ld_val<T>(doutptr, qbase + tid);
                ps = q_d * k_d;
                pd = dout_d * v_d;
            }

            sh_score[tid] = ps;
            sh_dp[tid] = pd;
            __syncthreads();

            #pragma unroll
            for (int off = 128; off > 0; off >>= 1) {
                if (tid < off) {
                    sh_score[tid] += sh_score[tid + off];
                    sh_dp[tid] += sh_dp[tid + off];
                }
                __syncthreads();
            }

            float lse = load_lse(q_lse_base, lidx, lse_dtype);
            float rowdot = q_rowdot[lidx];
            float prob = __expf(sh_score[0] * scale - lse);
            float ds = prob * (sh_dp[0] - rowdot);

            if (tid < D) {
                acc_dk += ds * q_d;
                acc_dv += prob * dout_d;
            }
            __syncthreads();
        }
    }

    if (tid < D) {
        st_val<T>(grad, n + kbase_local, acc_dk * scale);
        st_val<T>(grad, 2 * n + kbase_local, acc_dv);
    }
}

template <typename T>
__global__ void dp_avg_kernel(
    const uint64_t* __restrict__ grad_ptrs,
    T* __restrict__ out,
    int64_t total_n,
    int dp_world
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total_n; idx += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < dp_world) {
                const T* gp = reinterpret_cast<const T*>(grad_ptrs[r]);
                sum += ld_val<T>(gp, idx);
            }
        }
        st_val<T>(out, idx, sum / (float)dp_world);
    }
}

void launch_rowdot(torch::Tensor pack, torch::Tensor rowdot, int B, int S, int H, int D, int dtype_enum) {
    TORCH_CHECK(pack.is_cuda() && rowdot.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(D <= 256, "D > 256 is not supported by this BF16 H100 kernel");
    int64_t n = (int64_t)B * S * H * D;
    int rows = B * S * H;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        rowdot_kernel<__nv_bfloat16><<<rows, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(pack.data_ptr<at::BFloat16>()),
            rowdot.data_ptr<float>(), B, S, H, D, n);
    } else {
        rowdot_kernel<float><<<rows, 256, 0, stream>>>(
            pack.data_ptr<float>(), rowdot.data_ptr<float>(), B, S, H, D, n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_cp_backward(
    torch::Tensor pack_ptrs,
    torch::Tensor lse_ptrs,
    torch::Tensor row_ptrs,
    torch::Tensor grad,
    int B,
    int S,
    int H,
    int D,
    float scale,
    bool causal,
    int cp_rank,
    int cp_world,
    int dtype_enum,
    int lse_dtype_enum
) {
    TORCH_CHECK(pack_ptrs.is_cuda() && lse_ptrs.is_cuda() && row_ptrs.is_cuda() && grad.is_cuda(),
                "CUDA tensors required");
    TORCH_CHECK(D <= 256, "D > 256 is not supported by this BF16 H100 kernel");

    int64_t n = (int64_t)B * S * H * D;
    int rows = B * S * H;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint64_t* pp = reinterpret_cast<const uint64_t*>(pack_ptrs.data_ptr<int64_t>());
    const uint64_t* lp = reinterpret_cast<const uint64_t*>(lse_ptrs.data_ptr<int64_t>());
    const uint64_t* rp = reinterpret_cast<const uint64_t*>(row_ptrs.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>(grad.data_ptr<at::BFloat16>());
        dq_kernel<__nv_bfloat16><<<rows, 256, 0, stream>>>(
            pp, lp, rp, g, B, S, H, D, n, scale, causal, cp_rank, cp_world, lse_dtype_enum);
        dkdv_kernel<__nv_bfloat16><<<rows, 256, 0, stream>>>(
            pp, lp, rp, g, B, S, H, D, n, scale, causal, cp_rank, cp_world, lse_dtype_enum);
    } else {
        float* g = grad.data_ptr<float>();
        dq_kernel<float><<<rows, 256, 0, stream>>>(
            pp, lp, rp, g, B, S, H, D, n, scale, causal, cp_rank, cp_world, lse_dtype_enum);
        dkdv_kernel<float><<<rows, 256, 0, stream>>>(
            pp, lp, rp, g, B, S, H, D, n, scale, causal, cp_rank, cp_world, lse_dtype_enum);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_dp_avg(torch::Tensor grad_ptrs, torch::Tensor out, int64_t total_n, int dp_world, int dtype_enum) {
    TORCH_CHECK(grad_ptrs.is_cuda() && out.is_cuda(), "CUDA tensors required");

    int threads = 256;
    int blocks = (int)((total_n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint64_t* gp = reinterpret_cast<const uint64_t*>(grad_ptrs.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        dp_avg_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            gp, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), total_n, dp_world);
    } else {
        dp_avg_kernel<float><<<blocks, threads, 0, stream>>>(
            gp, out.data_ptr<float>(), total_n, dp_world);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_rowdot", &launch_rowdot, "row dot(dout, out)");
    m.def("launch_cp_backward", &launch_cp_backward, "CP ring-attention backward via UVA symmetric memory");
    m.def("launch_dp_avg", &launch_dp_avg, "DP average via UVA symmetric memory");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attn_bwd_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


_cp_cache = {}
_grad_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    raise TypeError("optimized solution supports torch.bfloat16 and torch.float32")


def _lse_dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.bfloat16:
        return 1
    raise TypeError("softmax_lse must be float32 or bfloat16")


def _group_key(group: dist.ProcessGroup) -> int:
    return id(group)


def _get_cp_resources(
    B: int,
    S: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    lse_dtype: torch.dtype,
    device: torch.device,
    cp_group: dist.ProcessGroup,
):
    key = (B, S, H, D, dtype, lse_dtype, device.index, _group_key(cp_group))
    cached = _cp_cache.get(key)
    if cached is not None:
        return cached

    n = B * S * H * D
    pack = symm_mem.empty((5, n), device=device, dtype=dtype)
    pack_hdl = symm_mem.rendezvous(pack, cp_group)

    lse = symm_mem.empty((B, H, S), device=device, dtype=lse_dtype)
    lse_hdl = symm_mem.rendezvous(lse, cp_group)

    rowdot = symm_mem.empty((B, H, S), device=device, dtype=torch.float32)
    row_hdl = symm_mem.rendezvous(rowdot, cp_group)

    pack_ptrs = torch.tensor(pack_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    lse_ptrs = torch.tensor(lse_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    row_ptrs = torch.tensor(row_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (pack, pack_hdl, lse, lse_hdl, rowdot, row_hdl, pack_ptrs, lse_ptrs, row_ptrs)
    _cp_cache[key] = cached
    return cached


def _get_grad_resources(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    dp_group: Optional[dist.ProcessGroup],
    use_dp: bool,
):
    key = (n, dtype, device.index, _group_key(dp_group) if dp_group is not None else -1, use_dp)
    cached = _grad_cache.get(key)
    if cached is not None:
        return cached

    if use_dp:
        grad = symm_mem.empty((3, n), device=device, dtype=dtype)
        grad_hdl = symm_mem.rendezvous(grad, dp_group)
        grad_ptrs = torch.tensor(grad_hdl.buffer_ptrs, device=device, dtype=torch.int64)
        avg = torch.empty((3, n), device=device, dtype=dtype)
    else:
        grad = torch.empty((3, n), device=device, dtype=dtype)
        grad_hdl = None
        grad_ptrs = None
        avg = None

    cached = (grad, grad_hdl, grad_ptrs, avg)
    _grad_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert q.is_cuda and k.is_cuda and v.is_cuda and dout.is_cuda and out.is_cuda
    assert q.dim() == 4, "q/k/v/dout/out must be [B,S,H,D]"
    assert softmax_lse.dim() == 3, "softmax_lse must be [B,H,S]"

    cp_group = cp_group or dist.group.WORLD
    cp_rank = dist.get_rank(cp_group)
    cp_world = dist.get_world_size(cp_group)

    B, S, H, D = q.shape
    n = B * S * H * D
    dtype = q.dtype
    lse_dtype = softmax_lse.dtype
    dtype_e = _dtype_enum(dtype)
    lse_dtype_e = _lse_dtype_enum(lse_dtype)

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    dout = dout.contiguous()
    out = out.contiguous()
    softmax_lse = softmax_lse.contiguous()

    assert k.shape == q.shape and v.shape == q.shape and dout.shape == q.shape and out.shape == q.shape
    assert softmax_lse.shape == (B, H, S)
    assert k.dtype == dtype and v.dtype == dtype and dout.dtype == dtype and out.dtype == dtype

    ext = _get_ext()
    device = q.device

    (
        pack,
        pack_hdl,
        lse_buf,
        lse_hdl,
        rowdot,
        row_hdl,
        pack_ptrs,
        lse_ptrs,
        row_ptrs,
    ) = _get_cp_resources(B, S, H, D, dtype, lse_dtype, device, cp_group)

    # Publish local operands into CP symmetric memory.
    pack[0].copy_(q.reshape(-1))
    pack[1].copy_(k.reshape(-1))
    pack[2].copy_(v.reshape(-1))
    pack[3].copy_(dout.reshape(-1))
    pack[4].copy_(out.reshape(-1))
    lse_buf.copy_(softmax_lse)

    pack_hdl.barrier(channel=0)
    lse_hdl.barrier(channel=0)

    # Precompute per-query row_dot = sum_d dout*out in symmetric memory so dK/dV can
    # read it from peer Q shards without recomputing it per key row.
    ext.launch_rowdot(pack, rowdot, B, S, H, D, dtype_e)
    row_hdl.barrier(channel=0)

    dp_world = dist.get_world_size(dp_group) if dp_group is not None else 1
    use_dp = dp_group is not None and dp_world > 1

    grad, grad_hdl, grad_ptrs, avg = _get_grad_resources(n, dtype, device, dp_group, use_dp)

    # CP backward directly over peer UVA pointers. This replaces the dual P2P ring:
    # local dQ reads all K/V shards; local dK/dV read all Q/dout/out/LSE shards.
    ext.launch_cp_backward(
        pack_ptrs,
        lse_ptrs,
        row_ptrs,
        grad,
        B,
        S,
        H,
        D,
        float(softmax_scale),
        bool(causal),
        int(cp_rank),
        int(cp_world),
        int(dtype_e),
        int(lse_dtype_e),
    )

    if use_dp:
        # Publish local CP gradients and average over DP replicas via direct peer loads.
        grad_hdl.barrier(channel=1)
        ext.launch_dp_avg(grad_ptrs, avg, 3 * n, int(dp_world), int(dtype_e))
        ret = avg
    else:
        ret = grad

    dq = ret[0].reshape(B, S, H, D)
    dk = ret[1].reshape(B, S, H, D)
    dv = ret[2].reshape(B, S, H, D)
    return dq, dk, dv