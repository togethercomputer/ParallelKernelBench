import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

static inline int ceil_div_i64_to_i32(int64_t a, int b) {
    int64_t v = (a + b - 1) / b;
    if (v > 65535) v = 65535;
    if (v < 1) v = 1;
    return (int)v;
}

// -----------------------------------------------------------------------------
// BF16 path: fused RoPE + all-gather via UVA remote stores into symmetric outputs
// -----------------------------------------------------------------------------

__global__ void rope_allgather_store_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cosv,
    const __nv_bfloat16* __restrict__ sinv,
    const long long* __restrict__ out_ptrs,
    int64_t n_local,
    int64_t n_global,
    int B,
    int S,
    int H,
    int D,
    int rank,
    int world_size
) {
    const int halfD = D >> 1;
    const int64_t Sg = (int64_t)S * (int64_t)world_size;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_local;
         idx += (int64_t)gridDim.x * blockDim.x) {

        int64_t t = idx;
        const int d = (int)(t % D);
        t /= D;
        const int h = (int)(t % H);
        t /= H;
        const int s = (int)(t % S);
        const int b = (int)(t / S);

        const int64_t pair_idx = (d < halfD) ? (idx + halfD) : (idx - halfD);
        const int64_t cs_idx = ((int64_t)b * S + s) * D + d;

        const float c = __bfloat162float(cosv[cs_idx]);
        const float ss = __bfloat162float(sinv[cs_idx]);

        const float qx = __bfloat162float(q[idx]);
        const float kx = __bfloat162float(k[idx]);

        float qr = __bfloat162float(q[pair_idx]);
        float kr = __bfloat162float(k[pair_idx]);
        if (d < halfD) {
            qr = -qr;
            kr = -kr;
        }

        const __nv_bfloat16 qout = __float2bfloat16(qx * c + qr * ss);
        const __nv_bfloat16 kout = __float2bfloat16(kx * c + kr * ss);

        const int64_t dst =
            (((int64_t)b * Sg + (int64_t)rank * S + s) * H + h) * D + d;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= world_size) break;
            __nv_bfloat16* base =
                reinterpret_cast<__nv_bfloat16*>((uintptr_t)out_ptrs[r]);
            base[dst] = qout;
            base[n_global + dst] = kout;
        }
    }
}

__global__ void rope_local_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cosv,
    const __nv_bfloat16* __restrict__ sinv,
    __nv_bfloat16* __restrict__ qout,
    __nv_bfloat16* __restrict__ kout,
    int64_t n_local,
    int B,
    int S,
    int H,
    int D
) {
    const int halfD = D >> 1;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_local;
         idx += (int64_t)gridDim.x * blockDim.x) {

        int64_t t = idx;
        const int d = (int)(t % D);
        t /= D;
        t /= H;
        const int s = (int)(t % S);
        const int b = (int)(t / S);

        const int64_t pair_idx = (d < halfD) ? (idx + halfD) : (idx - halfD);
        const int64_t cs_idx = ((int64_t)b * S + s) * D + d;

        const float c = __bfloat162float(cosv[cs_idx]);
        const float ss = __bfloat162float(sinv[cs_idx]);

        const float qx = __bfloat162float(q[idx]);
        const float kx = __bfloat162float(k[idx]);

        float qr = __bfloat162float(q[pair_idx]);
        float kr = __bfloat162float(k[pair_idx]);
        if (d < halfD) {
            qr = -qr;
            kr = -kr;
        }

        qout[idx] = __float2bfloat16(qx * c + qr * ss);
        kout[idx] = __float2bfloat16(kx * c + kr * ss);
    }
}

// -----------------------------------------------------------------------------
// FP32 fallback, still custom CUDA + symmetric-memory gather path
// -----------------------------------------------------------------------------

__global__ void rope_allgather_store_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cosv,
    const float* __restrict__ sinv,
    const long long* __restrict__ out_ptrs,
    int64_t n_local,
    int64_t n_global,
    int B,
    int S,
    int H,
    int D,
    int rank,
    int world_size
) {
    const int halfD = D >> 1;
    const int64_t Sg = (int64_t)S * (int64_t)world_size;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_local;
         idx += (int64_t)gridDim.x * blockDim.x) {

        int64_t t = idx;
        const int d = (int)(t % D);
        t /= D;
        const int h = (int)(t % H);
        t /= H;
        const int s = (int)(t % S);
        const int b = (int)(t / S);

        const int64_t pair_idx = (d < halfD) ? (idx + halfD) : (idx - halfD);
        const int64_t cs_idx = ((int64_t)b * S + s) * D + d;

        float qr = q[pair_idx];
        float kr = k[pair_idx];
        if (d < halfD) {
            qr = -qr;
            kr = -kr;
        }

        const float qout = q[idx] * cosv[cs_idx] + qr * sinv[cs_idx];
        const float kout = k[idx] * cosv[cs_idx] + kr * sinv[cs_idx];

        const int64_t dst =
            (((int64_t)b * Sg + (int64_t)rank * S + s) * H + h) * D + d;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= world_size) break;
            float* base = reinterpret_cast<float*>((uintptr_t)out_ptrs[r]);
            base[dst] = qout;
            base[n_global + dst] = kout;
        }
    }
}

__global__ void rope_local_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cosv,
    const float* __restrict__ sinv,
    float* __restrict__ qout,
    float* __restrict__ kout,
    int64_t n_local,
    int B,
    int S,
    int H,
    int D
) {
    const int halfD = D >> 1;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_local;
         idx += (int64_t)gridDim.x * blockDim.x) {

        int64_t t = idx;
        const int d = (int)(t % D);
        t /= D;
        t /= H;
        const int s = (int)(t % S);
        const int b = (int)(t / S);

        const int64_t pair_idx = (d < halfD) ? (idx + halfD) : (idx - halfD);
        const int64_t cs_idx = ((int64_t)b * S + s) * D + d;

        float qr = q[pair_idx];
        float kr = k[pair_idx];
        if (d < halfD) {
            qr = -qr;
            kr = -kr;
        }

        qout[idx] = q[idx] * cosv[cs_idx] + qr * sinv[cs_idx];
        kout[idx] = k[idx] * cosv[cs_idx] + kr * sinv[cs_idx];
    }
}

// -----------------------------------------------------------------------------
// FP16 fallback
// -----------------------------------------------------------------------------

__global__ void rope_allgather_store_f16_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ cosv,
    const half* __restrict__ sinv,
    const long long* __restrict__ out_ptrs,
    int64_t n_local,
    int64_t n_global,
    int B,
    int S,
    int H,
    int D,
    int rank,
    int world_size
) {
    const int halfD = D >> 1;
    const int64_t Sg = (int64_t)S * (int64_t)world_size;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_local;
         idx += (int64_t)gridDim.x * blockDim.x) {

        int64_t t = idx;
        const int d = (int)(t % D);
        t /= D;
        const int h = (int)(t % H);
        t /= H;
        const int s = (int)(t % S);
        const int b = (int)(t / S);

        const int64_t pair_idx = (d < halfD) ? (idx + halfD) : (idx - halfD);
        const int64_t cs_idx = ((int64_t)b * S + s) * D + d;

        const float c = __half2float(cosv[cs_idx]);
        const float ss = __half2float(sinv[cs_idx]);

        float qr = __half2float(q[pair_idx]);
        float kr = __half2float(k[pair_idx]);
        if (d < halfD) {
            qr = -qr;
            kr = -kr;
        }

        const half qout = __float2half_rn(__half2float(q[idx]) * c + qr * ss);
        const half kout = __float2half_rn(__half2float(k[idx]) * c + kr * ss);

        const int64_t dst =
            (((int64_t)b * Sg + (int64_t)rank * S + s) * H + h) * D + d;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= world_size) break;
            half* base = reinterpret_cast<half*>((uintptr_t)out_ptrs[r]);
            base[dst] = qout;
            base[n_global + dst] = kout;
        }
    }
}

__global__ void rope_local_f16_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ cosv,
    const half* __restrict__ sinv,
    half* __restrict__ qout,
    half* __restrict__ kout,
    int64_t n_local,
    int B,
    int S,
    int H,
    int D
) {
    const int halfD = D >> 1;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_local;
         idx += (int64_t)gridDim.x * blockDim.x) {

        int64_t t = idx;
        const int d = (int)(t % D);
        t /= D;
        t /= H;
        const int s = (int)(t % S);
        const int b = (int)(t / S);

        const int64_t pair_idx = (d < halfD) ? (idx + halfD) : (idx - halfD);
        const int64_t cs_idx = ((int64_t)b * S + s) * D + d;

        const float c = __half2float(cosv[cs_idx]);
        const float ss = __half2float(sinv[cs_idx]);

        float qr = __half2float(q[pair_idx]);
        float kr = __half2float(k[pair_idx]);
        if (d < halfD) {
            qr = -qr;
            kr = -kr;
        }

        qout[idx] = __float2half_rn(__half2float(q[idx]) * c + qr * ss);
        kout[idx] = __float2half_rn(__half2float(k[idx]) * c + kr * ss);
    }
}

void launch_rope_allgather_store(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cosv,
    torch::Tensor sinv,
    torch::Tensor out_ptrs,
    int B,
    int S,
    int H,
    int D,
    int rank,
    int world_size,
    int dtype_enum
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && cosv.is_cuda() && sinv.is_cuda(), "all inputs must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && cosv.is_contiguous() && sinv.is_contiguous(), "all inputs must be contiguous");
    TORCH_CHECK(out_ptrs.is_cuda() && out_ptrs.dtype() == torch::kInt64, "out_ptrs must be CUDA int64");
    TORCH_CHECK(D % 2 == 0, "RoPE head dimension D must be even");
    TORCH_CHECK(world_size <= 8, "this H100 node kernel expects world_size <= 8");

    const int64_t n_local = q.numel();
    const int64_t n_global = n_local * (int64_t)world_size;
    const int threads = 256;
    const int blocks = ceil_div_i64_to_i32(n_local, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* ptrs = (const long long*)out_ptrs.data_ptr<int64_t>();

    if (dtype_enum == 0) {
        rope_allgather_store_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(cosv.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(sinv.data_ptr<at::BFloat16>()),
            ptrs, n_local, n_global, B, S, H, D, rank, world_size);
    } else if (dtype_enum == 1) {
        rope_allgather_store_f32_kernel<<<blocks, threads, 0, stream>>>(
            q.data_ptr<float>(), k.data_ptr<float>(),
            cosv.data_ptr<float>(), sinv.data_ptr<float>(),
            ptrs, n_local, n_global, B, S, H, D, rank, world_size);
    } else {
        rope_allgather_store_f16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(cosv.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(sinv.data_ptr<at::Half>()),
            ptrs, n_local, n_global, B, S, H, D, rank, world_size);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_rope_local(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor cosv,
    torch::Tensor sinv,
    torch::Tensor qout,
    torch::Tensor kout,
    int B,
    int S,
    int H,
    int D,
    int dtype_enum
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && cosv.is_cuda() && sinv.is_cuda(), "all inputs must be CUDA");
    TORCH_CHECK(qout.is_cuda() && kout.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && cosv.is_contiguous() && sinv.is_contiguous(), "all inputs must be contiguous");
    TORCH_CHECK(qout.is_contiguous() && kout.is_contiguous(), "outputs must be contiguous");
    TORCH_CHECK(D % 2 == 0, "RoPE head dimension D must be even");

    const int64_t n_local = q.numel();
    const int threads = 256;
    const int blocks = ceil_div_i64_to_i32(n_local, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        rope_local_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(cosv.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(sinv.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qout.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(kout.data_ptr<at::BFloat16>()),
            n_local, B, S, H, D);
    } else if (dtype_enum == 1) {
        rope_local_f32_kernel<<<blocks, threads, 0, stream>>>(
            q.data_ptr<float>(), k.data_ptr<float>(),
            cosv.data_ptr<float>(), sinv.data_ptr<float>(),
            qout.data_ptr<float>(), kout.data_ptr<float>(),
            n_local, B, S, H, D);
    } else {
        rope_local_f16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(cosv.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(sinv.data_ptr<at::Half>()),
            reinterpret_cast<half*>(qout.data_ptr<at::Half>()),
            reinterpret_cast<half*>(kout.data_ptr<at::Half>()),
            n_local, B, S, H, D);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_rope_allgather_store", &launch_rope_allgather_store,
          "Fused RoPE + all-gather using symmetric-memory UVA remote stores");
    m.def("launch_rope_local", &launch_rope_local,
          "Local fused RoPE CUDA kernel");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("rope_allgather_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"unsupported dtype for fused RoPE all-gather: {dtype}")


def _contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _get_resources(
    B: int,
    S_local: int,
    H: int,
    D: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    key = (B, S_local, H, D, world_size, dtype, device)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    S_global = S_local * world_size

    # Symmetric output buffer layout:
    #   out_buf[0] -> q_global [B, S_global, H, D]
    #   out_buf[1] -> k_global [B, S_global, H, D]
    out_buf = symm_mem.empty((2, B, S_global, H, D), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(out_buf, dist.group.WORLD)

    ptrs_dev = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    q_view = out_buf[0]
    k_view = out_buf[1]

    cached = (out_buf, hdl, ptrs_dev, q_view, k_view)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    cos_local: torch.Tensor,
    sin_local: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused BF16-optimized RoPE + sequence all-gather.

    Distributed path:
      - allocate rank-local symmetric output [2, B, S_global, H, D]
      - each rank computes RoPE for its local [B, S_local, H, D]
      - each rank directly UVA-stores its result into every rank's symmetric output
      - symmetric-memory barrier replaces NCCL all_gather synchronization
    """
    assert q_local.is_cuda and k_local.is_cuda
    assert cos_local.is_cuda and sin_local.is_cuda
    assert q_local.dim() == 4
    assert k_local.shape == q_local.shape

    B = int(q_local.shape[0])
    S_local = int(q_local.shape[1])
    H = int(q_local.shape[2])
    D = int(q_local.shape[3])

    assert cos_local.shape == (B, S_local, D)
    assert sin_local.shape == (B, S_local, D)
    assert D % 2 == 0
    assert q_local.dtype == k_local.dtype == cos_local.dtype == sin_local.dtype

    dtype_enum = _dtype_enum(q_local.dtype)

    q = _contig(q_local)
    k = _contig(k_local)
    c = _contig(cos_local)
    s = _contig(sin_local)

    ext = _get_ext()

    if not dist.is_initialized():
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        ext.launch_rope_local(q, k, c, s, q_out, k_out, B, S_local, H, D, dtype_enum)
        return q_out, k_out

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        ext.launch_rope_local(q, k, c, s, q_out, k_out, B, S_local, H, D, dtype_enum)
        return q_out, k_out

    out_buf, hdl, ptrs_dev, q_global, k_global = _get_resources(
        B, S_local, H, D, world_size, q.dtype, q.device
    )

    ext.launch_rope_allgather_store(
        q,
        k,
        c,
        s,
        ptrs_dev,
        B,
        S_local,
        H,
        D,
        rank,
        world_size,
        dtype_enum,
    )

    # Ensures all ranks have completed their remote UVA stores into this rank's
    # symmetric output before q_global/k_global are consumed.
    hdl.barrier(channel=0)

    return q_global, k_global