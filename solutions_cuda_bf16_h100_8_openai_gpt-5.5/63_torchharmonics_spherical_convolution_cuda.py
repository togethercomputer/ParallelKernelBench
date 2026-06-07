from typing import List, Optional, Tuple, Dict, Any

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

#define CUDA_CHECK_ERRORS() C10_CUDA_KERNEL_LAUNCH_CHECK()

// -----------------------------------------------------------------------------
// Basic packing kernels into padded symmetric buffers
// -----------------------------------------------------------------------------

__global__ void pack4d_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t B, int64_t C, int64_t H, int64_t W, int64_t maxW
) {
    int64_t n = B * C * H * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;
        t /= H;
        int64_t c = t % C;
        int64_t b = t / C;
        dst[((b * C + c) * H + h) * maxW + w] = src[idx];
    }
}

__global__ void pack4d_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t B, int64_t C, int64_t H, int64_t W, int64_t maxW
) {
    int64_t n = B * C * H * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;
        t /= H;
        int64_t c = t % C;
        int64_t b = t / C;
        dst[((b * C + c) * H + h) * maxW + w] = src[idx];
    }
}

__global__ void pack5d_cpad_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t B, int64_t C, int64_t K, int64_t H, int64_t W, int64_t maxC
) {
    int64_t n = B * C * K * H * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;
        t /= H;
        int64_t k = t % K;
        t /= K;
        int64_t c = t % C;
        int64_t b = t / C;
        dst[((((b * maxC + c) * K + k) * H + h) * W + w)] = src[idx];
    }
}

__global__ void pack5d_cpad_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t B, int64_t C, int64_t K, int64_t H, int64_t W, int64_t maxC
) {
    int64_t n = B * C * K * H * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;
        t /= H;
        int64_t k = t % K;
        t /= K;
        int64_t c = t % C;
        int64_t b = t / C;
        dst[((((b * maxC + c) * K + k) * H + h) * W + w)] = src[idx];
    }
}

// -----------------------------------------------------------------------------
// Azimuth transpose #1: gather channel chunk from all longitude shards
// Output: [B, C_this_rank, H, global_lon]
// -----------------------------------------------------------------------------

__global__ void az1_gather_bf16_kernel(
    const long long* __restrict__ ptrs,
    const int32_t* __restrict__ lon_offsets,
    const int32_t* __restrict__ lon_sizes,
    __nv_bfloat16* __restrict__ out,
    int az_size,
    int64_t B, int64_t Cglobal, int64_t H,
    int64_t maxW, int64_t Cchunk, int64_t chan_offset, int64_t Nlon
) {
    int64_t n = B * Cchunk * H * Nlon;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % Nlon;
        int64_t t = idx / Nlon;
        int64_t h = t % H;
        t /= H;
        int64_t c = t % Cchunk;
        int64_t b = t / Cchunk;
        int src_rank = 0;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r >= az_size) break;
            int lo = lon_offsets[r];
            int sz = lon_sizes[r];
            if (w >= lo && w < lo + sz) {
                src_rank = r;
                break;
            }
        }
        int64_t wl = w - lon_offsets[src_rank];
        int64_t cg = chan_offset + c;
        const __nv_bfloat16* base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[src_rank]);
        out[idx] = base[((b * Cglobal + cg) * H + h) * maxW + wl];
    }
}

__global__ void az1_gather_f32_kernel(
    const long long* __restrict__ ptrs,
    const int32_t* __restrict__ lon_offsets,
    const int32_t* __restrict__ lon_sizes,
    float* __restrict__ out,
    int az_size,
    int64_t B, int64_t Cglobal, int64_t H,
    int64_t maxW, int64_t Cchunk, int64_t chan_offset, int64_t Nlon
) {
    int64_t n = B * Cchunk * H * Nlon;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % Nlon;
        int64_t t = idx / Nlon;
        int64_t h = t % H;
        t /= H;
        int64_t c = t % Cchunk;
        int64_t b = t / Cchunk;
        int src_rank = 0;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r >= az_size) break;
            int lo = lon_offsets[r];
            int sz = lon_sizes[r];
            if (w >= lo && w < lo + sz) {
                src_rank = r;
                break;
            }
        }
        int64_t wl = w - lon_offsets[src_rank];
        int64_t cg = chan_offset + c;
        const float* base = reinterpret_cast<const float*>((uintptr_t)ptrs[src_rank]);
        out[idx] = base[((b * Cglobal + cg) * H + h) * maxW + wl];
    }
}

// -----------------------------------------------------------------------------
// Sparse DISCO S2 contraction.
// psi CSR rows are row = k * Hout + hout, columns = hin * Nlon + lon.
// Output layout: [B, C, K, Hout, Wout]
// -----------------------------------------------------------------------------

__global__ void disco_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_idx,
    const float* __restrict__ vals,
    __nv_bfloat16* __restrict__ out,
    int64_t B, int64_t C, int64_t Hin, int64_t Nlon,
    int64_t K, int64_t Hout, int64_t Wout
) {
    int64_t n = B * C * K * Hout * Wout;
    int64_t pscale = Nlon / Wout;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t wout = idx % Wout;
        int64_t t = idx / Wout;
        int64_t hout = t % Hout;
        t /= Hout;
        int64_t k = t % K;
        t /= K;
        int64_t c = t % C;
        int64_t b = t / C;

        int64_t row = k * Hout + hout;
        int32_t start = row_offsets[row];
        int32_t end = row_offsets[row + 1];

        float acc = 0.0f;
        for (int32_t p = start; p < end; ++p) {
            int64_t col = col_idx[p];
            int64_t hin = col / Nlon;
            int64_t lon0 = col - hin * Nlon;
            int64_t lon = lon0 + wout * pscale;
            if (lon >= Nlon) lon -= Nlon;
            float xv = __bfloat162float(x[((b * C + c) * Hin + hin) * Nlon + lon]);
            acc += vals[p] * xv;
        }
        out[idx] = __float2bfloat16(acc);
    }
}

__global__ void disco_f32_kernel(
    const float* __restrict__ x,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_idx,
    const float* __restrict__ vals,
    float* __restrict__ out,
    int64_t B, int64_t C, int64_t Hin, int64_t Nlon,
    int64_t K, int64_t Hout, int64_t Wout
) {
    int64_t n = B * C * K * Hout * Wout;
    int64_t pscale = Nlon / Wout;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t wout = idx % Wout;
        int64_t t = idx / Wout;
        int64_t hout = t % Hout;
        t /= Hout;
        int64_t k = t % K;
        t /= K;
        int64_t c = t % C;
        int64_t b = t / C;

        int64_t row = k * Hout + hout;
        int32_t start = row_offsets[row];
        int32_t end = row_offsets[row + 1];

        float acc = 0.0f;
        for (int32_t p = start; p < end; ++p) {
            int64_t col = col_idx[p];
            int64_t hin = col / Nlon;
            int64_t lon0 = col - hin * Nlon;
            int64_t lon = lon0 + wout * pscale;
            if (lon >= Nlon) lon -= Nlon;
            acc += vals[p] * x[((b * C + c) * Hin + hin) * Nlon + lon];
        }
        out[idx] = acc;
    }
}

// -----------------------------------------------------------------------------
// Polar reduce-scatter: sum across polar peer buffers, emit only local H shard.
// Input peer layout: [B, C, K, Hout, W]
// Output layout:     [B, C, K, Hloc, W]
// -----------------------------------------------------------------------------

__global__ void polar_reduce_scatter_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int polar_size,
    int64_t B, int64_t C, int64_t K, int64_t Hout, int64_t W,
    int64_t Hoff, int64_t Hloc
) {
    int64_t n = B * C * K * Hloc * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t hl = t % Hloc;
        t /= Hloc;
        int64_t k = t % K;
        t /= K;
        int64_t c = t % C;
        int64_t b = t / C;
        int64_t hg = Hoff + hl;

        float acc = 0.0f;
        for (int r = 0; r < polar_size; ++r) {
            const __nv_bfloat16* base =
                reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[r]);
            acc += __bfloat162float(base[((((b * C + c) * K + k) * Hout + hg) * W + w)]);
        }
        out[idx] = __float2bfloat16(acc);
    }
}

__global__ void polar_reduce_scatter_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int polar_size,
    int64_t B, int64_t C, int64_t K, int64_t Hout, int64_t W,
    int64_t Hoff, int64_t Hloc
) {
    int64_t n = B * C * K * Hloc * W;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t hl = t % Hloc;
        t /= Hloc;
        int64_t k = t % K;
        t /= K;
        int64_t c = t % C;
        int64_t b = t / C;
        int64_t hg = Hoff + hl;

        float acc = 0.0f;
        for (int r = 0; r < polar_size; ++r) {
            const float* base = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
            acc += base[((((b * C + c) * K + k) * Hout + hg) * W + w)];
        }
        out[idx] = acc;
    }
}

// -----------------------------------------------------------------------------
// Azimuth transpose #2: gather channel chunks from all channel-sharded ranks,
// keeping this rank's longitude output chunk.
// Peer layout: [B, maxCchunk, K, Hloc, Wglobal]
// Out layout:  [B, Cglobal,   K, Hloc, Wlocal]
// -----------------------------------------------------------------------------

__global__ void az2_gather_bf16_kernel(
    const long long* __restrict__ ptrs,
    const int32_t* __restrict__ chan_offsets,
    const int32_t* __restrict__ chan_sizes,
    __nv_bfloat16* __restrict__ out,
    int az_size,
    int64_t B, int64_t Cglobal, int64_t K, int64_t Hloc,
    int64_t Wglobal, int64_t Woff, int64_t Wlocal, int64_t maxC
) {
    int64_t n = B * Cglobal * K * Hloc * Wlocal;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t wl = idx % Wlocal;
        int64_t t = idx / Wlocal;
        int64_t h = t % Hloc;
        t /= Hloc;
        int64_t k = t % K;
        t /= K;
        int64_t c = t % Cglobal;
        int64_t b = t / Cglobal;

        int src_rank = 0;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r >= az_size) break;
            int co = chan_offsets[r];
            int cs = chan_sizes[r];
            if (c >= co && c < co + cs) {
                src_rank = r;
                break;
            }
        }
        int64_t cl = c - chan_offsets[src_rank];
        int64_t wg = Woff + wl;
        const __nv_bfloat16* base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[src_rank]);
        out[idx] = base[((((b * maxC + cl) * K + k) * Hloc + h) * Wglobal + wg)];
    }
}

__global__ void az2_gather_f32_kernel(
    const long long* __restrict__ ptrs,
    const int32_t* __restrict__ chan_offsets,
    const int32_t* __restrict__ chan_sizes,
    float* __restrict__ out,
    int az_size,
    int64_t B, int64_t Cglobal, int64_t K, int64_t Hloc,
    int64_t Wglobal, int64_t Woff, int64_t Wlocal, int64_t maxC
) {
    int64_t n = B * Cglobal * K * Hloc * Wlocal;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t wl = idx % Wlocal;
        int64_t t = idx / Wlocal;
        int64_t h = t % Hloc;
        t /= Hloc;
        int64_t k = t % K;
        t /= K;
        int64_t c = t % Cglobal;
        int64_t b = t / Cglobal;

        int src_rank = 0;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r >= az_size) break;
            int co = chan_offsets[r];
            int cs = chan_sizes[r];
            if (c >= co && c < co + cs) {
                src_rank = r;
                break;
            }
        }
        int64_t cl = c - chan_offsets[src_rank];
        int64_t wg = Woff + wl;
        const float* base = reinterpret_cast<const float*>((uintptr_t)ptrs[src_rank]);
        out[idx] = base[((((b * maxC + cl) * K + k) * Hloc + h) * Wglobal + wg)];
    }
}

// -----------------------------------------------------------------------------
// Grouped channel mixing.
// x:      [B, C, K, H, W]
// weight: [Cout, C/groups, K]
// out:    [B, Cout, H, W]
// -----------------------------------------------------------------------------

__global__ void mix_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int64_t B, int64_t C, int64_t K, int64_t H, int64_t W,
    int64_t Cout, int groups, int has_bias
) {
    int64_t n = B * Cout * H * W;
    int64_t group_in = C / groups;
    int64_t out_per_group = Cout / groups;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;
        t /= H;
        int64_t co = t % Cout;
        int64_t b = t / Cout;

        int64_t g = co / out_per_group;
        int64_t og = co - g * out_per_group;
        float acc = 0.0f;

        for (int64_t ci = 0; ci < group_in; ++ci) {
            int64_t cg = g * group_in + ci;
            for (int64_t k = 0; k < K; ++k) {
                float xv = __bfloat162float(x[((((b * C + cg) * K + k) * H + h) * W + w)]);
                float wv = __bfloat162float(weight[((g * out_per_group + og) * group_in + ci) * K + k]);
                acc += xv * wv;
            }
        }
        if (has_bias) acc += __bfloat162float(bias[co]);
        out[idx] = __float2bfloat16(acc);
    }
}

__global__ void mix_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t B, int64_t C, int64_t K, int64_t H, int64_t W,
    int64_t Cout, int groups, int has_bias
) {
    int64_t n = B * Cout * H * W;
    int64_t group_in = C / groups;
    int64_t out_per_group = Cout / groups;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t idx = tid; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;
        t /= H;
        int64_t co = t % Cout;
        int64_t b = t / Cout;

        int64_t g = co / out_per_group;
        int64_t og = co - g * out_per_group;
        float acc = 0.0f;

        for (int64_t ci = 0; ci < group_in; ++ci) {
            int64_t cg = g * group_in + ci;
            for (int64_t k = 0; k < K; ++k) {
                acc += x[((((b * C + cg) * K + k) * H + h) * W + w)] *
                       weight[((g * out_per_group + og) * group_in + ci) * K + k];
            }
        }
        if (has_bias) acc += bias[co];
        out[idx] = acc;
    }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

static inline int launch_blocks(int64_t n, int threads=256) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

void pack4d(torch::Tensor src, torch::Tensor dst, int64_t B, int64_t C, int64_t H, int64_t W, int64_t maxW, int dtype_enum) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = launch_blocks(B*C*H*W, threads);
    if (dtype_enum == 0) {
        pack4d_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
            B, C, H, W, maxW);
    } else {
        pack4d_f32_kernel<<<blocks, threads, 0, stream>>>(
            src.data_ptr<float>(), dst.data_ptr<float>(), B, C, H, W, maxW);
    }
    CUDA_CHECK_ERRORS();
}

void pack5d_cpad(torch::Tensor src, torch::Tensor dst, int64_t B, int64_t C, int64_t K, int64_t H, int64_t W, int64_t maxC, int dtype_enum) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = launch_blocks(B*C*K*H*W, threads);
    if (dtype_enum == 0) {
        pack5d_cpad_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
            B, C, K, H, W, maxC);
    } else {
        pack5d_cpad_f32_kernel<<<blocks, threads, 0, stream>>>(
            src.data_ptr<float>(), dst.data_ptr<float>(), B, C, K, H, W, maxC);
    }
    CUDA_CHECK_ERRORS();
}

void az1_gather(
    torch::Tensor ptrs, torch::Tensor lon_offsets, torch::Tensor lon_sizes,
    torch::Tensor out,
    int az_size,
    int64_t B, int64_t Cglobal, int64_t H,
    int64_t maxW, int64_t Cchunk, int64_t chan_offset, int64_t Nlon,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = launch_blocks(B*Cchunk*H*Nlon, threads);
    const long long* p = reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>());
    const int32_t* lo = lon_offsets.data_ptr<int32_t>();
    const int32_t* ls = lon_sizes.data_ptr<int32_t>();
    if (dtype_enum == 0) {
        az1_gather_bf16_kernel<<<blocks, threads, 0, stream>>>(
            p, lo, ls,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            az_size, B, Cglobal, H, maxW, Cchunk, chan_offset, Nlon);
    } else {
        az1_gather_f32_kernel<<<blocks, threads, 0, stream>>>(
            p, lo, ls, out.data_ptr<float>(),
            az_size, B, Cglobal, H, maxW, Cchunk, chan_offset, Nlon);
    }
    CUDA_CHECK_ERRORS();
}

void disco(
    torch::Tensor x, torch::Tensor row_offsets, torch::Tensor col_idx, torch::Tensor vals,
    torch::Tensor out,
    int64_t B, int64_t C, int64_t Hin, int64_t Nlon,
    int64_t K, int64_t Hout, int64_t Wout,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = launch_blocks(B*C*K*Hout*Wout, threads);
    if (dtype_enum == 0) {
        disco_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            row_offsets.data_ptr<int32_t>(),
            col_idx.data_ptr<int32_t>(),
            vals.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            B, C, Hin, Nlon, K, Hout, Wout);
    } else {
        disco_f32_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            row_offsets.data_ptr<int32_t>(),
            col_idx.data_ptr<int32_t>(),
            vals.data_ptr<float>(),
            out.data_ptr<float>(),
            B, C, Hin, Nlon, K, Hout, Wout);
    }
    CUDA_CHECK_ERRORS();
}

void polar_reduce_scatter(
    torch::Tensor ptrs, torch::Tensor out,
    int polar_size,
    int64_t B, int64_t C, int64_t K, int64_t Hout, int64_t W,
    int64_t Hoff, int64_t Hloc,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = launch_blocks(B*C*K*Hloc*W, threads);
    const long long* p = reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>());
    if (dtype_enum == 0) {
        polar_reduce_scatter_bf16_kernel<<<blocks, threads, 0, stream>>>(
            p, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            polar_size, B, C, K, Hout, W, Hoff, Hloc);
    } else {
        polar_reduce_scatter_f32_kernel<<<blocks, threads, 0, stream>>>(
            p, out.data_ptr<float>(),
            polar_size, B, C, K, Hout, W, Hoff, Hloc);
    }
    CUDA_CHECK_ERRORS();
}

void az2_gather(
    torch::Tensor ptrs, torch::Tensor chan_offsets, torch::Tensor chan_sizes,
    torch::Tensor out,
    int az_size,
    int64_t B, int64_t Cglobal, int64_t K, int64_t Hloc,
    int64_t Wglobal, int64_t Woff, int64_t Wlocal, int64_t maxC,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = launch_blocks(B*Cglobal*K*Hloc*Wlocal, threads);
    const long long* p = reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>());
    const int32_t* co = chan_offsets.data_ptr<int32_t>();
    const int32_t* cs = chan_sizes.data_ptr<int32_t>();
    if (dtype_enum == 0) {
        az2_gather_bf16_kernel<<<blocks, threads, 0, stream>>>(
            p, co, cs,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            az_size, B, Cglobal, K, Hloc, Wglobal, Woff, Wlocal, maxC);
    } else {
        az2_gather_f32_kernel<<<blocks, threads, 0, stream>>>(
            p, co, cs, out.data_ptr<float>(),
            az_size, B, Cglobal, K, Hloc, Wglobal, Woff, Wlocal, maxC);
    }
    CUDA_CHECK_ERRORS();
}

void mix(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out,
    int64_t B, int64_t C, int64_t K, int64_t H, int64_t W,
    int64_t Cout, int groups, int has_bias, int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = launch_blocks(B*Cout*H*W, threads);
    if (dtype_enum == 0) {
        mix_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            B, C, K, H, W, Cout, groups, has_bias);
    } else {
        mix_f32_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
            out.data_ptr<float>(),
            B, C, K, H, W, Cout, groups, has_bias);
    }
    CUDA_CHECK_ERRORS();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack4d", &pack4d, "pack 4d into padded symmetric buffer");
    m.def("pack5d_cpad", &pack5d_cpad, "pack 5d into channel-padded symmetric buffer");
    m.def("az1_gather", &az1_gather, "azimuth transpose gather #1 via UVA");
    m.def("disco", &disco, "sparse DISCO S2 contraction");
    m.def("polar_reduce_scatter", &polar_reduce_scatter, "polar reduce-scatter via UVA");
    m.def("az2_gather", &az2_gather, "azimuth transpose gather #2 via UVA");
    m.def("mix", &mix, "grouped channel mixing");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("disco_s2_symm_cuda_bf16_h100_ext", CUDA_SRC)
    return _ext


def _compute_split_shapes(size: int, num_chunks: int) -> List[int]:
    if num_chunks == 1:
        return [size]
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)
    return [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]


def _offsets_from_sizes(sizes: List[int]) -> List[int]:
    out = []
    s = 0
    for v in sizes:
        out.append(s)
        s += v
    return out


_symm_cache: Dict[Any, Tuple[torch.Tensor, Any, torch.Tensor]] = {}
_int_cache: Dict[Any, torch.Tensor] = {}
_psi_cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    raise TypeError("Only bfloat16 and float32 are supported by this CUDA implementation")


def _int32_tensor(vals: List[int], device: torch.device, key: Tuple[Any, ...]) -> torch.Tensor:
    k = ("i32", device, tuple(vals), key)
    t = _int_cache.get(k)
    if t is None:
        t = torch.tensor(vals, device=device, dtype=torch.int32)
        _int_cache[k] = t
    return t


def _get_symm(
    role: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, Any, torch.Tensor]:
    key = (role, tuple(shape), dtype, device, id(group))
    r = _symm_cache.get(key)
    if r is not None:
        return r
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    r = (buf, hdl, ptrs)
    _symm_cache[key] = r
    return r


def _prepare_psi(psi: torch.Tensor, device: torch.device):
    key = (id(psi), getattr(psi, "_version", 0), tuple(psi.shape), device)
    cached = _psi_cache.get(key)
    if cached is not None:
        return cached

    K = int(psi.shape[0])
    Hout = int(psi.shape[1])
    rows_total = K * Hout

    if psi.layout == torch.sparse_coo:
        coo = psi.coalesce().to(device)
        idx = coo.indices()
        vals = coo.values().to(device=device, dtype=torch.float32).contiguous()
        rows = (idx[0].to(torch.int64) * Hout + idx[1].to(torch.int64)).contiguous()
        cols = idx[2].to(device=device, dtype=torch.int32).contiguous()
    else:
        dense = psi.to(device)
        nz = dense.nonzero(as_tuple=False)
        rows = (nz[:, 0].to(torch.int64) * Hout + nz[:, 1].to(torch.int64)).contiguous()
        cols = nz[:, 2].to(device=device, dtype=torch.int32).contiguous()
        vals = dense[nz[:, 0], nz[:, 1], nz[:, 2]].to(torch.float32).contiguous()

    order = torch.argsort(rows)
    rows = rows[order]
    cols = cols[order].contiguous()
    vals = vals[order].contiguous()

    counts = torch.bincount(rows, minlength=rows_total).to(torch.int32)
    row_offsets = torch.empty(rows_total + 1, device=device, dtype=torch.int32)
    row_offsets[0] = 0
    row_offsets[1:] = torch.cumsum(counts, dim=0)

    out = (row_offsets.contiguous(), cols, vals)
    _psi_cache[key] = out
    return out


@torch.no_grad()
def solution(
    x: torch.Tensor,
    psi: torch.Tensor,
    weight: torch.Tensor,
    groups: int,
    nlon_out: int,
    nlon_in: int,
    azimuth_group: Optional[dist.ProcessGroup] = None,
    polar_group: Optional[dist.ProcessGroup] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert x.is_cuda, "x must be CUDA"
    assert dist.is_initialized(), "torch.distributed must be initialized"

    ext = _get_ext()
    azimuth_group = azimuth_group or dist.group.WORLD
    polar_group = polar_group or dist.group.WORLD

    az_size = dist.get_world_size(group=azimuth_group)
    az_rank = dist.get_rank(group=azimuth_group)
    pol_size = dist.get_world_size(group=polar_group)
    pol_rank = dist.get_rank(group=polar_group)

    dtype = x.dtype
    de = _dtype_enum(dtype)
    device = x.device

    if not x.is_contiguous():
        x = x.contiguous()

    B = int(x.shape[0])
    Cglobal = int(x.shape[1])
    Hin_local = int(x.shape[2])
    Wlocal_in = int(x.shape[3])

    lon_in_sizes = _compute_split_shapes(nlon_in, az_size)
    lon_in_offsets = _offsets_from_sizes(lon_in_sizes)
    chan_sizes = _compute_split_shapes(Cglobal, az_size)
    chan_offsets = _offsets_from_sizes(chan_sizes)

    # -------------------------------------------------------------------------
    # 1. Azimuth transpose: longitude becomes local, channels become sharded.
    # -------------------------------------------------------------------------
    if az_size > 1:
        max_win = max(lon_in_sizes)
        symm_in, hdl_az1, ptrs_az1 = _get_symm(
            "az1_in",
            (B, Cglobal, Hin_local, max_win),
            dtype,
            device,
            azimuth_group,
        )
        ext.pack4d(x, symm_in, B, Cglobal, Hin_local, Wlocal_in, max_win, de)
        hdl_az1.barrier(channel=0)

        Cchunk = chan_sizes[az_rank]
        x_local = torch.empty((B, Cchunk, Hin_local, nlon_in), device=device, dtype=dtype)
        lon_off_t = _int32_tensor(lon_in_offsets, device, ("lon_in_off", nlon_in, az_size))
        lon_sz_t = _int32_tensor(lon_in_sizes, device, ("lon_in_sz", nlon_in, az_size))
        ext.az1_gather(
            ptrs_az1,
            lon_off_t,
            lon_sz_t,
            x_local,
            az_size,
            B,
            Cglobal,
            Hin_local,
            max_win,
            Cchunk,
            chan_offsets[az_rank],
            nlon_in,
            de,
        )
    else:
        Cchunk = Cglobal
        x_local = x

    # -------------------------------------------------------------------------
    # 2. Sparse DISCO S2 contraction directly into a polar symmetric buffer
    #    when polar communication is needed.
    # -------------------------------------------------------------------------
    K = int(psi.shape[0])
    Hout = int(psi.shape[1])
    row_offsets, col_idx, vals = _prepare_psi(psi, device)

    if pol_size > 1:
        disco_buf, hdl_pol, ptrs_pol = _get_symm(
            "polar_disco",
            (B, Cchunk, K, Hout, nlon_out),
            dtype,
            device,
            polar_group,
        )
    else:
        disco_buf = torch.empty((B, Cchunk, K, Hout, nlon_out), device=device, dtype=dtype)
        hdl_pol = None
        ptrs_pol = None

    ext.disco(
        x_local,
        row_offsets,
        col_idx,
        vals,
        disco_buf,
        B,
        Cchunk,
        Hin_local,
        nlon_in,
        K,
        Hout,
        nlon_out,
        de,
    )

    # -------------------------------------------------------------------------
    # 3 + 4. Polar all-reduce fused with latitude scatter.
    # -------------------------------------------------------------------------
    if pol_size > 1:
        hdl_pol.barrier(channel=1)
        h_sizes = _compute_split_shapes(Hout, pol_size)
        h_offsets = _offsets_from_sizes(h_sizes)
        Hloc = h_sizes[pol_rank]
        Hoff = h_offsets[pol_rank]
        x_reduced = torch.empty((B, Cchunk, K, Hloc, nlon_out), device=device, dtype=dtype)
        ext.polar_reduce_scatter(
            ptrs_pol,
            x_reduced,
            pol_size,
            B,
            Cchunk,
            K,
            Hout,
            nlon_out,
            Hoff,
            Hloc,
            de,
        )
    else:
        Hloc = Hout
        x_reduced = disco_buf

    # -------------------------------------------------------------------------
    # 5. Azimuth transpose back: channels local, longitude sharded.
    # -------------------------------------------------------------------------
    lon_out_sizes = _compute_split_shapes(nlon_out, az_size)
    lon_out_offsets = _offsets_from_sizes(lon_out_sizes)
    Wlocal_out = lon_out_sizes[az_rank]

    if az_size > 1:
        max_cchunk = max(chan_sizes)
        symm_red, hdl_az2, ptrs_az2 = _get_symm(
            "az2_red",
            (B, max_cchunk, K, Hloc, nlon_out),
            dtype,
            device,
            azimuth_group,
        )
        ext.pack5d_cpad(
            x_reduced,
            symm_red,
            B,
            Cchunk,
            K,
            Hloc,
            nlon_out,
            max_cchunk,
            de,
        )
        hdl_az2.barrier(channel=2)

        x_full = torch.empty(
            (B, Cglobal, K, Hloc, Wlocal_out),
            device=device,
            dtype=dtype,
        )
        chan_off_t = _int32_tensor(chan_offsets, device, ("chan_off", Cglobal, az_size))
        chan_sz_t = _int32_tensor(chan_sizes, device, ("chan_sz", Cglobal, az_size))
        ext.az2_gather(
            ptrs_az2,
            chan_off_t,
            chan_sz_t,
            x_full,
            az_size,
            B,
            Cglobal,
            K,
            Hloc,
            nlon_out,
            lon_out_offsets[az_rank],
            Wlocal_out,
            max_cchunk,
            de,
        )
    else:
        x_full = x_reduced
        Wlocal_out = nlon_out

    # -------------------------------------------------------------------------
    # 6 + 7. Grouped channel mixing and optional bias.
    # -------------------------------------------------------------------------
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if weight.dtype != dtype:
        weight = weight.to(dtype)

    Cout = int(weight.shape[0])
    out = torch.empty((B, Cout, Hloc, Wlocal_out), device=device, dtype=dtype)

    if bias is None:
        bias_arg = torch.empty((0,), device=device, dtype=dtype)
        has_bias = 0
    else:
        bias_arg = bias.to(device=device, dtype=dtype).contiguous()
        has_bias = 1

    ext.mix(
        x_full,
        weight,
        bias_arg,
        out,
        B,
        Cglobal,
        K,
        Hloc,
        Wlocal_out,
        Cout,
        groups,
        has_bias,
        de,
    )

    return out