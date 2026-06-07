# Strategy:
# - Compute FP32 L2 sum-of-squares for BF16/FP tensors with custom CUDA reductions.
# - Write the scalar directly into symmetric memory and reduce it with a CUDA UVA peer-load kernel.
# - Avoid NCCL/torch.distributed collectives on the hot path; only symm_mem rendezvous/barriers are used.
# - Compute clip coefficient on device and apply in-place scaling with custom CUDA kernels, with no host tensor sync.

import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        shared[wid] = v;
    }
    __syncthreads();

    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
    }
    return v;
}

__global__ void sumsq_bf16_scalar_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ acc,
    int64_t n
) {
    float local = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float v = __bfloat162float(x[i]);
        local += v * v;
    }

    local = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(acc, local);
    }
}

__global__ void sumsq_bf16_vec2_kernel(
    const __nv_bfloat162* __restrict__ x2,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ acc,
    int64_t n_pairs,
    int has_tail
) {
    float local = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n_pairs; i += stride) {
        float2 v = __bfloat1622float2(x2[i]);
        local += v.x * v.x + v.y * v.y;
    }

    if (has_tail && blockIdx.x == 0 && threadIdx.x == 0) {
        float v = __bfloat162float(x[n_pairs * 2]);
        local += v * v;
    }

    local = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(acc, local);
    }
}

__global__ void sumsq_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ acc,
    int64_t n
) {
    float local = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float v = x[i];
        local += v * v;
    }

    local = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(acc, local);
    }
}

__global__ void sumsq_f16_kernel(
    const __half* __restrict__ x,
    float* __restrict__ acc,
    int64_t n
) {
    float local = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float v = __half2float(x[i]);
        local += v * v;
    }

    local = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(acc, local);
    }
}

__global__ void sumsq_f64_kernel(
    const double* __restrict__ x,
    float* __restrict__ acc,
    int64_t n
) {
    float local = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float v = static_cast<float>(x[i]);  // reference casts grads to fp32 before norm
        local += v * v;
    }

    local = block_reduce_sum(local);
    if (threadIdx.x == 0) {
        atomicAdd(acc, local);
    }
}

__global__ void finish_reduce_kernel(
    const int64_t* __restrict__ ptrs,
    float* __restrict__ total_norm,
    float* __restrict__ coef_out,
    float max_norm,
    int world_size
) {
    float s = 0.0f;
    int tid = threadIdx.x;

    if (tid < world_size) {
        const float* p = reinterpret_cast<const float*>((uintptr_t)ptrs[tid]);
        s = p[0];
    }

    s = block_reduce_sum(s);

    if (tid == 0) {
        float n = sqrtf(s);
        total_norm[0] = n;
        coef_out[0] = (n > max_norm) ? (max_norm / n) : 1.0f;
    }
}

__global__ void finish_local_kernel(
    const float* __restrict__ local_sum,
    float* __restrict__ total_norm,
    float* __restrict__ coef_out,
    float max_norm
) {
    if (threadIdx.x == 0) {
        float n = sqrtf(local_sum[0]);
        total_norm[0] = n;
        coef_out[0] = (n > max_norm) ? (max_norm / n) : 1.0f;
    }
}

__global__ void scale_bf16_scalar_kernel(
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ coef,
    int64_t n
) {
    float c = coef[0];
    if (c == 1.0f) {
        return;
    }

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float v = __bfloat162float(x[i]) * c;
        x[i] = __float2bfloat16(v);
    }
}

__global__ void scale_bf16_vec2_kernel(
    __nv_bfloat162* __restrict__ x2,
    __nv_bfloat16* __restrict__ x,
    const float* __restrict__ coef,
    int64_t n_pairs,
    int has_tail
) {
    float c = coef[0];
    if (c == 1.0f) {
        return;
    }

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n_pairs; i += stride) {
        float2 v = __bfloat1622float2(x2[i]);
        v.x *= c;
        v.y *= c;
        x2[i] = __float22bfloat162_rn(v);
    }

    if (has_tail && blockIdx.x == 0 && threadIdx.x == 0) {
        float v = __bfloat162float(x[n_pairs * 2]) * c;
        x[n_pairs * 2] = __float2bfloat16(v);
    }
}

__global__ void scale_f32_kernel(
    float* __restrict__ x,
    const float* __restrict__ coef,
    int64_t n
) {
    float c = coef[0];
    if (c == 1.0f) {
        return;
    }

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        x[i] *= c;
    }
}

__global__ void scale_f16_kernel(
    __half* __restrict__ x,
    const float* __restrict__ coef,
    int64_t n
) {
    float c = coef[0];
    if (c == 1.0f) {
        return;
    }

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float v = __half2float(x[i]) * c;
        x[i] = __float2half(v);
    }
}

__global__ void scale_f64_kernel(
    double* __restrict__ x,
    const float* __restrict__ coef,
    int64_t n
) {
    double c = static_cast<double>(coef[0]);
    if (c == 1.0) {
        return;
    }

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        x[i] *= c;
    }
}

static inline int blocks_for(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 4096) b = 4096;
    return static_cast<int>(b);
}

void local_sumsq(std::vector<torch::Tensor> tensors, torch::Tensor scalar_out) {
    CHECK_CUDA(scalar_out);
    CHECK_CONTIG(scalar_out);
    TORCH_CHECK(scalar_out.scalar_type() == at::kFloat, "scalar_out must be float32");
    TORCH_CHECK(scalar_out.numel() >= 1, "scalar_out must contain one element");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(scalar_out.data_ptr<float>(), 0, sizeof(float), stream);

    const int threads = 256;

    for (auto& t : tensors) {
        CHECK_CUDA(t);
        CHECK_CONTIG(t);

        int64_t n = t.numel();
        if (n == 0) {
            continue;
        }

        int blocks = blocks_for(n, threads);
        auto dt = t.scalar_type();

        if (dt == at::kBFloat16) {
            auto* p = reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
            uintptr_t addr = reinterpret_cast<uintptr_t>(p);
            if ((addr % alignof(__nv_bfloat162)) == 0 && n >= 2) {
                int64_t n_pairs = n >> 1;
                int has_tail = static_cast<int>(n & 1);
                int pair_blocks = blocks_for(n_pairs, threads);
                auto* p2 = reinterpret_cast<__nv_bfloat162*>(p);
                sumsq_bf16_vec2_kernel<<<pair_blocks, threads, 0, stream>>>(
                    p2, p, scalar_out.data_ptr<float>(), n_pairs, has_tail);
            } else {
                sumsq_bf16_scalar_kernel<<<blocks, threads, 0, stream>>>(
                    p, scalar_out.data_ptr<float>(), n);
            }
        } else if (dt == at::kFloat) {
            sumsq_f32_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<float>(), scalar_out.data_ptr<float>(), n);
        } else if (dt == at::kHalf) {
            auto* p = reinterpret_cast<__half*>(t.data_ptr<at::Half>());
            sumsq_f16_kernel<<<blocks, threads, 0, stream>>>(
                p, scalar_out.data_ptr<float>(), n);
        } else if (dt == at::kDouble) {
            sumsq_f64_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<double>(), scalar_out.data_ptr<float>(), n);
        } else {
            TORCH_CHECK(false, "unsupported grad dtype for CUDA clip_grad_norm");
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finish_reduce(
    torch::Tensor ptrs,
    torch::Tensor total_norm,
    torch::Tensor coef_out,
    double max_norm,
    int world_size
) {
    CHECK_CUDA(ptrs);
    CHECK_CUDA(total_norm);
    CHECK_CUDA(coef_out);
    CHECK_CONTIG(ptrs);
    CHECK_CONTIG(total_norm);
    CHECK_CONTIG(coef_out);

    TORCH_CHECK(ptrs.scalar_type() == at::kLong, "ptrs must be int64");
    TORCH_CHECK(total_norm.scalar_type() == at::kFloat, "total_norm must be float32");
    TORCH_CHECK(coef_out.scalar_type() == at::kFloat, "coef_out must be float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    finish_reduce_kernel<<<1, 32, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        total_norm.data_ptr<float>(),
        coef_out.data_ptr<float>(),
        static_cast<float>(max_norm),
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finish_local(
    torch::Tensor local_sum,
    torch::Tensor total_norm,
    torch::Tensor coef_out,
    double max_norm
) {
    CHECK_CUDA(local_sum);
    CHECK_CUDA(total_norm);
    CHECK_CUDA(coef_out);
    CHECK_CONTIG(local_sum);
    CHECK_CONTIG(total_norm);
    CHECK_CONTIG(coef_out);

    TORCH_CHECK(local_sum.scalar_type() == at::kFloat, "local_sum must be float32");
    TORCH_CHECK(total_norm.scalar_type() == at::kFloat, "total_norm must be float32");
    TORCH_CHECK(coef_out.scalar_type() == at::kFloat, "coef_out must be float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    finish_local_kernel<<<1, 1, 0, stream>>>(
        local_sum.data_ptr<float>(),
        total_norm.data_ptr<float>(),
        coef_out.data_ptr<float>(),
        static_cast<float>(max_norm)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scale_tensors(std::vector<torch::Tensor> tensors, torch::Tensor coef) {
    CHECK_CUDA(coef);
    CHECK_CONTIG(coef);
    TORCH_CHECK(coef.scalar_type() == at::kFloat, "coef must be float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int threads = 256;

    for (auto& t : tensors) {
        CHECK_CUDA(t);
        CHECK_CONTIG(t);

        int64_t n = t.numel();
        if (n == 0) {
            continue;
        }

        int blocks = blocks_for(n, threads);
        auto dt = t.scalar_type();

        if (dt == at::kBFloat16) {
            auto* p = reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
            uintptr_t addr = reinterpret_cast<uintptr_t>(p);
            if ((addr % alignof(__nv_bfloat162)) == 0 && n >= 2) {
                int64_t n_pairs = n >> 1;
                int has_tail = static_cast<int>(n & 1);
                int pair_blocks = blocks_for(n_pairs, threads);
                auto* p2 = reinterpret_cast<__nv_bfloat162*>(p);
                scale_bf16_vec2_kernel<<<pair_blocks, threads, 0, stream>>>(
                    p2, p, coef.data_ptr<float>(), n_pairs, has_tail);
            } else {
                scale_bf16_scalar_kernel<<<blocks, threads, 0, stream>>>(
                    p, coef.data_ptr<float>(), n);
            }
        } else if (dt == at::kFloat) {
            scale_f32_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<float>(), coef.data_ptr<float>(), n);
        } else if (dt == at::kHalf) {
            auto* p = reinterpret_cast<__half*>(t.data_ptr<at::Half>());
            scale_f16_kernel<<<blocks, threads, 0, stream>>>(
                p, coef.data_ptr<float>(), n);
        } else if (dt == at::kDouble) {
            scale_f64_kernel<<<blocks, threads, 0, stream>>>(
                t.data_ptr<double>(), coef.data_ptr<float>(), n);
        } else {
            TORCH_CHECK(false, "unsupported grad dtype for CUDA clip_grad_norm");
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("local_sumsq", &local_sumsq, "FP32 local sum of squares for grad tensors");
    m.def("finish_reduce", &finish_reduce, "UVA peer-load scalar all-reduce + clip coefficient");
    m.def("finish_local", &finish_local, "Local norm + clip coefficient");
    m.def("scale_tensors", &scale_tensors, "In-place grad scaling");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("clip_grad_norm_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


_group_cache = {}
_local_cache = {}


def _device_key(device: torch.device):
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    return idx


def _get_local_resources(device: torch.device):
    key = _device_key(device)
    res = _local_cache.get(key)
    if res is not None:
        return res

    scalar = torch.empty((1,), device=device, dtype=torch.float32)
    coef = torch.empty((1,), device=device, dtype=torch.float32)
    res = (scalar, coef)
    _local_cache[key] = res
    return res


def _get_group_resources(group, device: torch.device):
    key = (id(group), _device_key(device))
    res = _group_cache.get(key)
    if res is not None:
        return res

    scalar = symm_mem.empty((1,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(scalar, group)
    ptrs = torch.tensor([int(p) for p in hdl.buffer_ptrs], device=device, dtype=torch.int64)
    coef = torch.empty((1,), device=device, dtype=torch.float32)

    res = (scalar, hdl, ptrs, coef)
    _group_cache[key] = res
    return res


@torch.no_grad()
def solution(
    grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    fsdp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert float(norm_type) == 2.0, "optimized path implements L2 clip_grad_norm only"

    tensors = [t for t in grad_tensors if t is not None]

    if tensors:
        device = tensors[0].device
        for t in tensors:
            assert t.is_cuda, "grad tensors must be CUDA tensors"
            assert t.is_contiguous(), "grad tensors must be contiguous for the CUDA fast path"
            assert t.device == device, "all grad tensors must be on the same CUDA device"
    else:
        device = torch.device("cuda", torch.cuda.current_device())

    ext = _get_ext()
    total_norm = torch.empty((), device=device, dtype=torch.float32)

    use_group = fsdp_group is not None
    if use_group:
        assert dist.is_initialized(), "torch.distributed must be initialized when fsdp_group is provided"
        scalar, hdl, ptrs, coef = _get_group_resources(fsdp_group, device)

        # Local FP32 sum of squares goes directly into symmetric memory.
        ext.local_sumsq(tensors, scalar)

        # Publish scalar to peers.
        hdl.barrier(channel=0)

        # Device-side all-reduce of the scalar using UVA peer loads; no NCCL.
        ext.finish_reduce(ptrs, total_norm, coef, float(max_norm), int(hdl.world_size))

        # Ensure every rank has completed peer scalar reads before any future overwrite.
        hdl.barrier(channel=1)

        # In-place clipping is fully device-side and uses the device coefficient.
        ext.scale_tensors(tensors, coef)
    else:
        scalar, coef = _get_local_resources(device)
        ext.local_sumsq(tensors, scalar)
        ext.finish_local(scalar, total_norm, coef, float(max_norm))
        ext.scale_tensors(tensors, coef)

    return total_norm