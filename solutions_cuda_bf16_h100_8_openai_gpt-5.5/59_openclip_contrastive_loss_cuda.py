"""
SigLIP contrastive loss using symmetric-memory UVA peer reads and custom CUDA.

Each rank publishes its local text block into symmetric memory, then a single
BF16 WMMA CUDA kernel directly reads every rank's text block over NVLink/UVA and
fuses image@text.T with SigLIP softplus loss accumulation. This removes NCCL/P2P
ring exchanges and keeps communication device-side while tensor-core tiles are
computed.
"""

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
#include <mma.h>

#include <cstdint>

using namespace nvcuda;

__device__ __forceinline__ float softplus_f32(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float round_bf16_to_f32(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

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

__global__ void copy_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

// One warp computes one 16x16 output tile for one text rank.
// A = local image [B,D], Bmat = peer text^T logically [D,B].
// Peer text is stored row-major [B,D], loaded into shared as col-major B tile.
__global__ void siglip_bf16_wmma_loss_kernel(
    const __nv_bfloat16* __restrict__ image,
    const int64_t* __restrict__ text_ptrs,
    float* __restrict__ accum,
    int B,
    int D,
    int world_size,
    int rank,
    float logit_scale,
    float logit_bias
) {
#if __CUDA_ARCH__ >= 800
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int text_rank = blockIdx.z;
    const int tid = threadIdx.x;

    __shared__ __nv_bfloat16 As[16 * 16];
    __shared__ __nv_bfloat16 Bs[16 * 16];
    __shared__ float Cs[16 * 16];

    const __nv_bfloat16* __restrict__ text =
        reinterpret_cast<const __nv_bfloat16*>(text_ptrs[text_rank]);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < D; k0 += 16) {
        for (int idx = tid; idx < 256; idx += 32) {
            const int i = idx >> 4;
            const int k = idx & 15;
            const int row = tile_m * 16 + i;
            const int col = k0 + k;

            float v = 0.0f;
            if (row < B && col < D) {
                v = __bfloat162float(image[(int64_t)row * D + col]);
            }
            As[idx] = __float2bfloat16(v);
        }

        // Col-major shared tile for matrix_b: offset = k + j*16.
        for (int idx = tid; idx < 256; idx += 32) {
            const int k = idx & 15;
            const int j = idx >> 4;
            const int text_row = tile_n * 16 + j;
            const int col = k0 + k;

            float v = 0.0f;
            if (text_row < B && col < D) {
                v = __bfloat162float(text[(int64_t)text_row * D + col]);
            }
            Bs[idx] = __float2bfloat16(v);
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, As, 16);
        wmma::load_matrix_sync(b_frag, Bs, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(Cs, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    float local = 0.0f;

    for (int idx = tid; idx < 256; idx += 32) {
        const int i = idx >> 4;
        const int j = idx & 15;
        const int image_row = tile_m * 16 + i;
        const int text_row = tile_n * 16 + j;

        if (image_row < B && text_row < B) {
            // Match the reference BF16 hot path more closely:
            // matmul result is BF16, scale and bias elementwise ops round BF16.
            float dot = round_bf16_to_f32(Cs[idx]);
            float logit = round_bf16_to_f32(logit_scale * dot);
            logit = round_bf16_to_f32(logit + logit_bias);

            const bool positive = (text_rank == rank) && (image_row == text_row);
            local += positive ? softplus_f32(-logit) : softplus_f32(logit);
        }
    }

    // Warp reduction, one warp per block.
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_down_sync(mask, local, off);
    }

    if (tid == 0) {
        atomicAdd(accum, local);
    }
#endif
}

// F32 correctness fallback. One block computes one logit with a D reduction.
__global__ void siglip_f32_loss_kernel(
    const float* __restrict__ image,
    const int64_t* __restrict__ text_ptrs,
    float* __restrict__ accum,
    int B,
    int D,
    int world_size,
    int rank,
    float logit_scale,
    float logit_bias
) {
    const int64_t total = (int64_t)world_size * B * B;
    const int64_t linear = blockIdx.x;
    if (linear >= total) return;

    const int text_rank = (int)(linear / ((int64_t)B * B));
    const int rem = (int)(linear - (int64_t)text_rank * B * B);
    const int i = rem / B;
    const int j = rem - i * B;

    const float* __restrict__ text =
        reinterpret_cast<const float*>(text_ptrs[text_rank]);

    float sum = 0.0f;
    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        sum += image[(int64_t)i * D + k] * text[(int64_t)j * D + k];
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float logit = logit_scale * smem[0] + logit_bias;
        const bool positive = (text_rank == rank) && (i == j);
        float term = positive ? softplus_f32(-logit) : softplus_f32(logit);
        atomicAdd(accum, term);
    }
}

__global__ void finalize_bf16_kernel(
    const float* __restrict__ accum,
    at::BFloat16* __restrict__ out,
    float inv_batch
) {
    if (threadIdx.x == 0) {
        float v = accum[0] * inv_batch;
        *reinterpret_cast<__nv_bfloat16*>(out) = __float2bfloat16(v);
    }
}

__global__ void finalize_f32_kernel(
    const float* __restrict__ accum,
    float* __restrict__ out,
    float inv_batch
) {
    if (threadIdx.x == 0) {
        out[0] = accum[0] * inv_batch;
    }
}

void copy_to_symm(torch::Tensor src, torch::Tensor dst, int64_t n, int dtype_enum) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "copy_to_symm tensors must be CUDA");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "copy_to_symm tensors must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    if (dtype_enum == 0) {
        copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
            n
        );
    } else {
        copy_f32_kernel<<<blocks, threads, 0, stream>>>(
            src.data_ptr<float>(),
            dst.data_ptr<float>(),
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void siglip_bf16_loss(
    torch::Tensor image,
    torch::Tensor text_ptrs,
    torch::Tensor accum,
    torch::Tensor out,
    int B,
    int D,
    int world_size,
    int rank,
    float logit_scale,
    float logit_bias
) {
    TORCH_CHECK(image.is_cuda() && text_ptrs.is_cuda() && accum.is_cuda() && out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
    TORCH_CHECK(image.dtype() == torch::kBFloat16, "image must be BF16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be BF16");
    TORCH_CHECK(accum.dtype() == torch::kFloat32, "accum must be float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(accum.data_ptr<float>(), 0, sizeof(float), stream);

    dim3 grid((B + 15) / 16, (B + 15) / 16, world_size);
    dim3 block(32);

    siglip_bf16_wmma_loss_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(image.data_ptr<at::BFloat16>()),
        text_ptrs.data_ptr<int64_t>(),
        accum.data_ptr<float>(),
        B,
        D,
        world_size,
        rank,
        logit_scale,
        logit_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    finalize_bf16_kernel<<<1, 1, 0, stream>>>(
        accum.data_ptr<float>(),
        out.data_ptr<at::BFloat16>(),
        1.0f / (float)B
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void siglip_f32_loss(
    torch::Tensor image,
    torch::Tensor text_ptrs,
    torch::Tensor accum,
    torch::Tensor out,
    int B,
    int D,
    int world_size,
    int rank,
    float logit_scale,
    float logit_bias
) {
    TORCH_CHECK(image.is_cuda() && text_ptrs.is_cuda() && accum.is_cuda() && out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
    TORCH_CHECK(image.dtype() == torch::kFloat32, "image must be float32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(accum.dtype() == torch::kFloat32, "accum must be float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(accum.data_ptr<float>(), 0, sizeof(float), stream);

    int64_t total = (int64_t)world_size * B * B;
    TORCH_CHECK(total <= 2147483647LL, "problem too large for f32 fallback grid");

    siglip_f32_loss_kernel<<<(int)total, 256, 0, stream>>>(
        image.data_ptr<float>(),
        text_ptrs.data_ptr<int64_t>(),
        accum.data_ptr<float>(),
        B,
        D,
        world_size,
        rank,
        logit_scale,
        logit_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    finalize_f32_kernel<<<1, 1, 0, stream>>>(
        accum.data_ptr<float>(),
        out.data_ptr<float>(),
        1.0f / (float)B
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_to_symm", &copy_to_symm, "copy local text into symmetric buffer");
    m.def("siglip_bf16_loss", &siglip_bf16_loss, "SigLIP BF16 loss over symmetric peer text buffers");
    m.def("siglip_f32_loss", &siglip_f32_loss, "SigLIP F32 fallback over symmetric peer text buffers");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("siglip_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _group_key(group: dist.ProcessGroup):
    return id(group)


def _get_resources(shape, dtype, device, group):
    key = (tuple(shape), dtype, device.index, _group_key(group))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    text_buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(text_buf, group)

    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    accum = torch.empty((), device=device, dtype=torch.float32)

    cached = {
        "text_buf": text_buf,
        "hdl": hdl,
        "ptrs": ptrs,
        "accum": accum,
        "world_size": dist.get_world_size(group),
        "rank": dist.get_rank(group),
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
    logit_bias: float = 0.0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert image_features.is_cuda and text_features.is_cuda
    assert image_features.dim() == 2 and text_features.dim() == 2
    assert image_features.shape == text_features.shape
    assert image_features.dtype == text_features.dtype

    if not image_features.is_contiguous():
        image_features = image_features.contiguous()
    if not text_features.is_contiguous():
        text_features = text_features.contiguous()

    dtype = image_features.dtype
    assert dtype in (torch.bfloat16, torch.float32), "optimized path supports BF16 and float32 fallback"

    B = int(image_features.size(0))
    D = int(image_features.size(1))
    assert B > 0 and D > 0

    ext = _get_ext()
    res = _get_resources(tuple(text_features.shape), dtype, text_features.device, group)

    text_buf = res["text_buf"]
    hdl = res["hdl"]
    ptrs = res["ptrs"]
    accum = res["accum"]
    world_size = int(res["world_size"])
    rank = int(res["rank"])

    dtype_enum = 0 if dtype is torch.bfloat16 else 1

    # Publish this rank's text block to symmetric memory, then synchronize all
    # ranks before device-side UVA reads. No NCCL/P2P ring collectives are used.
    ext.copy_to_symm(text_features, text_buf, text_features.numel(), dtype_enum)
    hdl.barrier(channel=0)

    out = torch.empty((), device=image_features.device, dtype=dtype)

    if dtype is torch.bfloat16:
        ext.siglip_bf16_loss(
            image_features,
            ptrs,
            accum,
            out,
            B,
            D,
            world_size,
            rank,
            float(logit_scale),
            float(logit_bias),
        )
    else:
        ext.siglip_f32_loss(
            image_features,
            ptrs,
            accum,
            out,
            B,
            D,
            world_size,
            rank,
            float(logit_scale),
            float(logit_bias),
        )

    return out