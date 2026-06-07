from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

static inline int launch_blocks(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

// -----------------------------------------------------------------------------
// Pack local overlap slices into symmetric buffer [2, B, H, P]
//   symm[0] = last P values of local chunk A
//   symm[1] = last P values of local chunk B
// x is [B, H, 2*S]
// -----------------------------------------------------------------------------

__global__ void pack_overlaps_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ symm,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t P,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t p = idx % P;
        int64_t q = idx / P;
        int64_t h = q % H;
        q /= H;
        int64_t b = q % B;
        int64_t c = q / B;  // 0 or 1

        int64_t x_idx = ((b * H + h) * (2 * S)) + c * S + (S - P + p);
        symm[idx] = x[x_idx];
    }
}

__global__ void pack_overlaps_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ symm,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t P,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t p = idx % P;
        int64_t q = idx / P;
        int64_t h = q % H;
        q /= H;
        int64_t b = q % B;
        int64_t c = q / B;  // 0 or 1

        int64_t x_idx = ((b * H + h) * (2 * S)) + c * S + (S - P + p);
        symm[idx] = x[x_idx];
    }
}

// -----------------------------------------------------------------------------
// Tail convolution: t >= P needs only local x, so it can overlap communication.
// y[b,h,c,t] = sum_j weight[h,j] * padded[c,b,h,t+j]
// For t >= P, padded index always maps to local chunk at t+j-P.
// -----------------------------------------------------------------------------

__global__ void conv_tail_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t K,
    int64_t start_t,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t tail_len = S - start_t;
    int64_t P = K - 1;

    for (; idx < total; idx += stride) {
        int64_t tr = idx % tail_len;
        int64_t t = start_t + tr;
        int64_t q = idx / tail_len;
        int64_t h = q % H;
        q /= H;
        int64_t b = q % B;
        int64_t c = q / B;

        int64_t base = ((b * H + h) * (2 * S)) + c * S;
        float acc = 0.0f;

        for (int64_t j = 0; j < K; ++j) {
            float xv = __bfloat162float(x[base + t + j - P]);
            float wv = __bfloat162float(w[h * K + j]);
            acc += xv * wv;
        }

        out[base + t] = __float2bfloat16(acc);
    }
}

__global__ void conv_tail_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t K,
    int64_t start_t,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t tail_len = S - start_t;
    int64_t P = K - 1;

    for (; idx < total; idx += stride) {
        int64_t tr = idx % tail_len;
        int64_t t = start_t + tr;
        int64_t q = idx / tail_len;
        int64_t h = q % H;
        q /= H;
        int64_t b = q % B;
        int64_t c = q / B;

        int64_t base = ((b * H + h) * (2 * S)) + c * S;
        float acc = 0.0f;

        for (int64_t j = 0; j < K; ++j) {
            acc += x[base + t + j - P] * w[h * K + j];
        }

        out[base + t] = acc;
    }
}

// -----------------------------------------------------------------------------
// Prefix convolution: t < P needs context.
// chunk A context: previous rank's chunk A overlap, or zeros on first rank.
// chunk B context: next rank's chunk B overlap, or local chunk A overlap on last.
// Remote contexts are read by UVA pointers into symmetric memory.
// -----------------------------------------------------------------------------

__global__ void conv_prefix_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const uint64_t prev_ptr_u64,
    const uint64_t next_ptr_u64,
    __nv_bfloat16* __restrict__ out,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t K,
    int rank,
    int world_size,
    int64_t prefix_len,
    int64_t total
) {
    const __nv_bfloat16* prev_symm = reinterpret_cast<const __nv_bfloat16*>(prev_ptr_u64);
    const __nv_bfloat16* next_symm = reinterpret_cast<const __nv_bfloat16*>(next_ptr_u64);
    int64_t P = K - 1;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t t = idx % prefix_len;
        int64_t q = idx / prefix_len;
        int64_t h = q % H;
        q /= H;
        int64_t b = q % B;
        int64_t c = q / B;

        int64_t local_base = ((b * H + h) * (2 * S)) + c * S;
        float acc = 0.0f;

        int64_t ctx_count = P - t;
        if (ctx_count < 0) ctx_count = 0;
        if (ctx_count > K) ctx_count = K;

        // Context part.
        for (int64_t j = 0; j < ctx_count; ++j) {
            int64_t p = t + j;
            float xv = 0.0f;

            if (c == 0) {
                if (rank > 0) {
                    int64_t off = (((int64_t)0 * B + b) * H + h) * P + p;
                    xv = __bfloat162float(prev_symm[off]);
                }
            } else {
                if (rank < world_size - 1) {
                    int64_t off = (((int64_t)1 * B + b) * H + h) * P + p;
                    xv = __bfloat162float(next_symm[off]);
                } else {
                    // Reference fallback: recv_next_b = chunk_a.clone()
                    int64_t a_base = ((b * H + h) * (2 * S));
                    xv = __bfloat162float(x[a_base + (S - P + p)]);
                }
            }

            float wv = __bfloat162float(w[h * K + j]);
            acc += xv * wv;
        }

        // Local part.
        for (int64_t j = ctx_count; j < K; ++j) {
            int64_t local_t = t + j - P;
            float xv = __bfloat162float(x[local_base + local_t]);
            float wv = __bfloat162float(w[h * K + j]);
            acc += xv * wv;
        }

        out[local_base + t] = __float2bfloat16(acc);
    }
}

__global__ void conv_prefix_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const uint64_t prev_ptr_u64,
    const uint64_t next_ptr_u64,
    float* __restrict__ out,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t K,
    int rank,
    int world_size,
    int64_t prefix_len,
    int64_t total
) {
    const float* prev_symm = reinterpret_cast<const float*>(prev_ptr_u64);
    const float* next_symm = reinterpret_cast<const float*>(next_ptr_u64);
    int64_t P = K - 1;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t t = idx % prefix_len;
        int64_t q = idx / prefix_len;
        int64_t h = q % H;
        q /= H;
        int64_t b = q % B;
        int64_t c = q / B;

        int64_t local_base = ((b * H + h) * (2 * S)) + c * S;
        float acc = 0.0f;

        int64_t ctx_count = P - t;
        if (ctx_count < 0) ctx_count = 0;
        if (ctx_count > K) ctx_count = K;

        for (int64_t j = 0; j < ctx_count; ++j) {
            int64_t p = t + j;
            float xv = 0.0f;

            if (c == 0) {
                if (rank > 0) {
                    int64_t off = (((int64_t)0 * B + b) * H + h) * P + p;
                    xv = prev_symm[off];
                }
            } else {
                if (rank < world_size - 1) {
                    int64_t off = (((int64_t)1 * B + b) * H + h) * P + p;
                    xv = next_symm[off];
                } else {
                    int64_t a_base = ((b * H + h) * (2 * S));
                    xv = x[a_base + (S - P + p)];
                }
            }

            acc += xv * w[h * K + j];
        }

        for (int64_t j = ctx_count; j < K; ++j) {
            int64_t local_t = t + j - P;
            acc += x[local_base + local_t] * w[h * K + j];
        }

        out[local_base + t] = acc;
    }
}

void pack_overlaps(torch::Tensor x, torch::Tensor symm, int64_t B, int64_t H, int64_t S, int64_t P) {
    TORCH_CHECK(x.is_cuda() && symm.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.is_contiguous() && symm.is_contiguous(), "contiguous tensors required");
    if (P <= 0) return;

    int64_t total = 2 * B * H * P;
    const int threads = 256;
    int blocks = launch_blocks(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.dtype() == torch::kBFloat16) {
        pack_overlaps_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(symm.data_ptr<at::BFloat16>()),
            B, H, S, P, total);
    } else if (x.dtype() == torch::kFloat32) {
        pack_overlaps_f32_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), symm.data_ptr<float>(), B, H, S, P, total);
    } else {
        TORCH_CHECK(false, "supported dtypes: bfloat16, float32");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void conv_tail(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t K,
    int64_t start_t
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    if (start_t >= S) return;

    int64_t tail_len = S - start_t;
    int64_t total = 2 * B * H * tail_len;
    const int threads = 256;
    int blocks = launch_blocks(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.dtype() == torch::kBFloat16) {
        conv_tail_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            B, H, S, K, start_t, total);
    } else if (x.dtype() == torch::kFloat32) {
        conv_tail_f32_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(),
            B, H, S, K, start_t, total);
    } else {
        TORCH_CHECK(false, "supported dtypes: bfloat16, float32");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void conv_prefix(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t prev_ptr,
    int64_t next_ptr,
    torch::Tensor out,
    int64_t B,
    int64_t H,
    int64_t S,
    int64_t K,
    int rank,
    int world_size,
    int64_t prefix_len
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    if (prefix_len <= 0) return;

    int64_t total = 2 * B * H * prefix_len;
    const int threads = 256;
    int blocks = launch_blocks(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    uint64_t prev_u = static_cast<uint64_t>(prev_ptr);
    uint64_t next_u = static_cast<uint64_t>(next_ptr);

    if (x.dtype() == torch::kBFloat16) {
        conv_prefix_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            prev_u,
            next_u,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            B, H, S, K, rank, world_size, prefix_len, total);
    } else if (x.dtype() == torch::kFloat32) {
        conv_prefix_f32_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), prev_u, next_u, out.data_ptr<float>(),
            B, H, S, K, rank, world_size, prefix_len, total);
    } else {
        TORCH_CHECK(false, "supported dtypes: bfloat16, float32");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_overlaps", &pack_overlaps, "Pack Hyena zigzag overlap slices into symmetric memory");
    m.def("conv_tail", &conv_tail, "Local-only tail causal depthwise conv1d");
    m.def("conv_prefix", &conv_prefix, "Boundary prefix causal depthwise conv1d with UVA symmetric memory");
}
'''


_ext = None
_resource_cache = {}
_stream_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("hyena_cp_boundary_conv_bf16_uva_ext", CUDA_SRC)
    return _ext


def _device_key(device: torch.device) -> int:
    return torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()


def _get_comm_stream(device: torch.device) -> torch.cuda.Stream:
    dev_idx = _device_key(device)
    s = _stream_cache.get(dev_idx)
    if s is None:
        with torch.cuda.device(dev_idx):
            s = torch.cuda.Stream(device=dev_idx)
        _stream_cache[dev_idx] = s
    return s


def _get_symm_resource(
    B: int,
    H: int,
    P: int,
    dtype: torch.dtype,
    device: torch.device,
    group,
):
    key = (B, H, P, dtype, _device_key(device), id(group))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty((2, B, H, P), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)

    cached = (buf, hdl)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    x: torch.Tensor,
    weight: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Hyena context-parallel causal depthwise conv1d over local zigzag chunks.

    Replaces distributed P2P/NCCL and torch conv1d with:
      - symmetric-memory overlap exchange,
      - UVA peer reads for neighbor context,
      - custom CUDA BF16/FP32 direct depthwise causal convolution.
    """
    assert x.is_cuda, "x must be CUDA"
    assert weight.is_cuda, "weight must be CUDA"
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert x.dim() == 3, "x must be [B, H, 2*S]"
    assert weight.dim() == 3 and weight.shape[1] == 1, "weight must be [H, 1, K]"
    assert x.dtype == weight.dtype, "x and weight must have the same dtype"
    assert x.dtype in (torch.bfloat16, torch.float32), "supported dtypes: bfloat16, float32"

    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    B = int(x.shape[0])
    H = int(x.shape[1])
    local_seq = int(x.shape[2])
    S = local_seq // 2
    K = int(weight.shape[-1])
    P = K - 1

    assert local_seq == 2 * S, "local sequence length must be even"
    assert int(weight.shape[0]) == H, "weight hidden dimension must match x"
    assert P <= S, "kernel overlap K-1 must not exceed per-zigzag chunk length"

    x_c = x.contiguous()
    w_c = weight.contiguous().view(H, K)
    out = torch.empty_like(x_c)

    ext = _get_ext()

    # No context exchange needed for K == 1.
    if P == 0:
        ext.conv_tail(x_c, w_c, out, B, H, S, K, 0)
        return out.reshape_as(x)

    symm_buf, hdl = _get_symm_resource(B, H, P, x_c.dtype, x_c.device, group)

    prev_ptr = int(hdl.buffer_ptrs[rank - 1]) if rank > 0 else 0
    next_ptr = int(hdl.buffer_ptrs[rank + 1]) if rank < world_size - 1 else 0

    current_stream = torch.cuda.current_stream(x_c.device)
    comm_stream = _get_comm_stream(x_c.device)
    done_event = torch.cuda.Event(blocking=False, interprocess=False)

    # Comm stream: publish overlaps, then symmetric-memory barrier.
    comm_stream.wait_stream(current_stream)
    with torch.cuda.stream(comm_stream):
        ext.pack_overlaps(x_c, symm_buf, B, H, S, P)
        hdl.barrier(channel=0)
        done_event.record(comm_stream)

    # Compute stream: local-only tail runs while pack/barrier is in flight.
    tail_start = P
    if tail_start < S:
        ext.conv_tail(x_c, w_c, out, B, H, S, K, tail_start)

    # Boundary prefix needs peer context.
    current_stream.wait_event(done_event)
    prefix_len = min(P, S)
    ext.conv_prefix(
        x_c,
        w_c,
        prev_ptr,
        next_ptr,
        out,
        B,
        H,
        S,
        K,
        rank,
        world_size,
        prefix_len,
    )

    return out.reshape_as(x)