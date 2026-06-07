from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension

# Strategy:
# - Fuse Adam math with the all-gather publish: each rank computes its updated shard once
#   and directly stores it into every rank's symmetric output buffer via UVA/NVLink.
# - Replace NCCL all_gather_into_tensor with peer stores plus a device-side symmetric-memory
#   signal-pad barrier, keeping communication on GPU and avoiding an extra model-sized temp.

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

static inline int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

__device__ __forceinline__ float load_bf16(const void* p, int64_t i) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);
}

__device__ __forceinline__ float load_f16(const void* p, int64_t i) {
    return __half2float(reinterpret_cast<const __half*>(p)[i]);
}

__device__ __forceinline__ float load_f32(const void* p, int64_t i) {
    return reinterpret_cast<const float*>(p)[i];
}

__device__ __forceinline__ void store_bf16(uint64_t base, int64_t i, float x) {
    reinterpret_cast<__nv_bfloat16*>(base)[i] = __float2bfloat16(x);
}

__device__ __forceinline__ void store_f16(uint64_t base, int64_t i, float x) {
    reinterpret_cast<__half*>(base)[i] = __float2half(x);
}

__device__ __forceinline__ void store_f32(uint64_t base, int64_t i, float x) {
    reinterpret_cast<float*>(base)[i] = x;
}

__device__ __forceinline__ float adam_update_f32(
    float g,
    float w,
    float m_old,
    float v_old,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float inv_bc1,
    float inv_bc2,
    float eps
) {
    float m = fmaf(beta1, m_old, one_minus_beta1 * g);
    float v = fmaf(beta2, v_old, one_minus_beta2 * g * g);
    float m_hat = m * inv_bc1;
    float v_hat = v * inv_bc2;
    return w - lr * (m_hat / (sqrtf(v_hat) + eps));
}

// all bf16 -> bf16 output
__global__ void adam_publish_bf16_kernel(
    const void* __restrict__ grad,
    const void* __restrict__ master,
    const void* __restrict__ exp_avg,
    const void* __restrict__ exp_avg_sq,
    const uint64_t* __restrict__ out_ptrs,
    int64_t p,
    int rank,
    int world_size,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float inv_bc1,
    float inv_bc2,
    float eps
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < p; i += stride) {
        float g = load_bf16(grad, i);
        float w = load_bf16(master, i);
        float m_old = load_bf16(exp_avg, i);
        float v_old = load_bf16(exp_avg_sq, i);
        float upd = adam_update_f32(
            g, w, m_old, v_old,
            lr, beta1, beta2,
            one_minus_beta1, one_minus_beta2,
            inv_bc1, inv_bc2, eps
        );

        int64_t out_i = (int64_t)rank * p + i;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                store_bf16(out_ptrs[r], out_i, upd);
            }
        }
    }
}

// all f32 -> f32 output
__global__ void adam_publish_f32_kernel(
    const void* __restrict__ grad,
    const void* __restrict__ master,
    const void* __restrict__ exp_avg,
    const void* __restrict__ exp_avg_sq,
    const uint64_t* __restrict__ out_ptrs,
    int64_t p,
    int rank,
    int world_size,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float inv_bc1,
    float inv_bc2,
    float eps
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < p; i += stride) {
        float g = load_f32(grad, i);
        float w = load_f32(master, i);
        float m_old = load_f32(exp_avg, i);
        float v_old = load_f32(exp_avg_sq, i);
        float upd = adam_update_f32(
            g, w, m_old, v_old,
            lr, beta1, beta2,
            one_minus_beta1, one_minus_beta2,
            inv_bc1, inv_bc2, eps
        );

        int64_t out_i = (int64_t)rank * p + i;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                store_f32(out_ptrs[r], out_i, upd);
            }
        }
    }
}

// bf16 grad, f32 master/state -> f32 output
__global__ void adam_publish_bf16grad_f32_kernel(
    const void* __restrict__ grad,
    const void* __restrict__ master,
    const void* __restrict__ exp_avg,
    const void* __restrict__ exp_avg_sq,
    const uint64_t* __restrict__ out_ptrs,
    int64_t p,
    int rank,
    int world_size,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float inv_bc1,
    float inv_bc2,
    float eps
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < p; i += stride) {
        float g = load_bf16(grad, i);
        float w = load_f32(master, i);
        float m_old = load_f32(exp_avg, i);
        float v_old = load_f32(exp_avg_sq, i);
        float upd = adam_update_f32(
            g, w, m_old, v_old,
            lr, beta1, beta2,
            one_minus_beta1, one_minus_beta2,
            inv_bc1, inv_bc2, eps
        );

        int64_t out_i = (int64_t)rank * p + i;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                store_f32(out_ptrs[r], out_i, upd);
            }
        }
    }
}

// fp16 grad, f32 master/state -> f32 output
__global__ void adam_publish_f16grad_f32_kernel(
    const void* __restrict__ grad,
    const void* __restrict__ master,
    const void* __restrict__ exp_avg,
    const void* __restrict__ exp_avg_sq,
    const uint64_t* __restrict__ out_ptrs,
    int64_t p,
    int rank,
    int world_size,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float inv_bc1,
    float inv_bc2,
    float eps
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < p; i += stride) {
        float g = load_f16(grad, i);
        float w = load_f32(master, i);
        float m_old = load_f32(exp_avg, i);
        float v_old = load_f32(exp_avg_sq, i);
        float upd = adam_update_f32(
            g, w, m_old, v_old,
            lr, beta1, beta2,
            one_minus_beta1, one_minus_beta2,
            inv_bc1, inv_bc2, eps
        );

        int64_t out_i = (int64_t)rank * p + i;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                store_f32(out_ptrs[r], out_i, upd);
            }
        }
    }
}

// all fp16 -> fp16 output
__global__ void adam_publish_f16_kernel(
    const void* __restrict__ grad,
    const void* __restrict__ master,
    const void* __restrict__ exp_avg,
    const void* __restrict__ exp_avg_sq,
    const uint64_t* __restrict__ out_ptrs,
    int64_t p,
    int rank,
    int world_size,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float inv_bc1,
    float inv_bc2,
    float eps
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < p; i += stride) {
        float g = load_f16(grad, i);
        float w = load_f16(master, i);
        float m_old = load_f16(exp_avg, i);
        float v_old = load_f16(exp_avg_sq, i);
        float upd = adam_update_f32(
            g, w, m_old, v_old,
            lr, beta1, beta2,
            one_minus_beta1, one_minus_beta2,
            inv_bc1, inv_bc2, eps
        );

        int64_t out_i = (int64_t)rank * p + i;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                store_f16(out_ptrs[r], out_i, upd);
            }
        }
    }
}

__device__ __forceinline__ void send_signal_release(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_acquire(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1u);
}

__global__ void symm_signal_barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size,
    int slot
) {
    int t = threadIdx.x;
    if (t >= world_size) {
        return;
    }

    __threadfence_system();

    uint32_t* local_base = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank]);
    uint32_t* peer_base = reinterpret_cast<uint32_t*>(signal_pad_ptrs[t]);

    uint32_t* send_addr = peer_base + (int64_t)slot * world_size + rank;
    uint32_t* wait_addr = local_base + (int64_t)slot * world_size + t;

    send_signal_release(send_addr);
    wait_signal_acquire(wait_addr);
}

void launch_adam_publish(
    torch::Tensor grad,
    torch::Tensor master,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor out_ptrs_tensor,
    int64_t p,
    int rank,
    int world_size,
    double lr,
    double beta1,
    double beta2,
    double inv_bc1,
    double inv_bc2,
    double eps,
    int mode
) {
    TORCH_CHECK(grad.is_cuda(), "grad must be CUDA");
    TORCH_CHECK(master.is_cuda(), "master must be CUDA");
    TORCH_CHECK(exp_avg.is_cuda(), "exp_avg must be CUDA");
    TORCH_CHECK(exp_avg_sq.is_cuda(), "exp_avg_sq must be CUDA");
    TORCH_CHECK(out_ptrs_tensor.is_cuda(), "out_ptrs_tensor must be CUDA");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(master.is_contiguous(), "master must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(), "exp_avg must be contiguous");
    TORCH_CHECK(exp_avg_sq.is_contiguous(), "exp_avg_sq must be contiguous");
    TORCH_CHECK(world_size <= 8, "optimized path assumes <= 8 ranks");

    const uint64_t* out_ptrs =
        reinterpret_cast<const uint64_t*>(out_ptrs_tensor.data_ptr<int64_t>());

    int threads = 256;
    int blocks = div_up_i64(p, threads);
    if (blocks > 65535) {
        blocks = 65535;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    float flr = static_cast<float>(lr);
    float fb1 = static_cast<float>(beta1);
    float fb2 = static_cast<float>(beta2);
    float fomb1 = static_cast<float>(1.0 - beta1);
    float fomb2 = static_cast<float>(1.0 - beta2);
    float fibc1 = static_cast<float>(inv_bc1);
    float fibc2 = static_cast<float>(inv_bc2);
    float feps = static_cast<float>(eps);

    const void* g = grad.data_ptr();
    const void* w = master.data_ptr();
    const void* m = exp_avg.data_ptr();
    const void* v = exp_avg_sq.data_ptr();

    if (mode == 0) {
        adam_publish_bf16_kernel<<<blocks, threads, 0, stream>>>(
            g, w, m, v, out_ptrs, p, rank, world_size,
            flr, fb1, fb2, fomb1, fomb2, fibc1, fibc2, feps);
    } else if (mode == 1) {
        adam_publish_f32_kernel<<<blocks, threads, 0, stream>>>(
            g, w, m, v, out_ptrs, p, rank, world_size,
            flr, fb1, fb2, fomb1, fomb2, fibc1, fibc2, feps);
    } else if (mode == 2) {
        adam_publish_bf16grad_f32_kernel<<<blocks, threads, 0, stream>>>(
            g, w, m, v, out_ptrs, p, rank, world_size,
            flr, fb1, fb2, fomb1, fomb2, fibc1, fibc2, feps);
    } else if (mode == 3) {
        adam_publish_f16grad_f32_kernel<<<blocks, threads, 0, stream>>>(
            g, w, m, v, out_ptrs, p, rank, world_size,
            flr, fb1, fb2, fomb1, fomb2, fibc1, fibc2, feps);
    } else if (mode == 4) {
        adam_publish_f16_kernel<<<blocks, threads, 0, stream>>>(
            g, w, m, v, out_ptrs, p, rank, world_size,
            flr, fb1, fb2, fomb1, fomb2, fibc1, fibc2, feps);
    } else {
        TORCH_CHECK(false, "unsupported dtype mode");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_symm_barrier(
    torch::Tensor signal_pad_ptrs_tensor,
    int rank,
    int world_size,
    int slot
) {
    TORCH_CHECK(signal_pad_ptrs_tensor.is_cuda(), "signal_pad_ptrs_tensor must be CUDA");
    const uint64_t* signal_ptrs =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    symm_signal_barrier_kernel<<<1, 32, 0, stream>>>(signal_ptrs, rank, world_size, slot);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_adam_publish", &launch_adam_publish,
          "Fused Adam update and symmetric-memory all-gather publish");
    m.def("launch_symm_barrier", &launch_symm_barrier,
          "Device-side symmetric-memory signal-pad barrier");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_adam_unshard_symm_uva_ext", CUDA_SRC)
    return _ext


# key: (p, dtype, device_index, world_size) -> (symmetric gather buffer, handle, ptr tensor)
_resource_cache: Dict[Tuple[int, torch.dtype, int, int], Tuple[Tensor, object, Tensor]] = {}


def _get_resources(p: int, dtype: torch.dtype, device: torch.device, world_size: int):
    dev_index = device.index
    if dev_index is None:
        dev_index = torch.cuda.current_device()
    key = (p, dtype, int(dev_index), int(world_size))

    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    gather_buf = symm_mem.empty((world_size * p,), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(gather_buf, dist.group.WORLD)

    ptrs = torch.tensor([int(x) for x in hdl.buffer_ptrs], device=device, dtype=torch.int64)

    res = (gather_buf, hdl, ptrs)
    _resource_cache[key] = res
    return res


def _dtype_mode(
    grad_shard: Tensor,
    master_shard: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
) -> int:
    gd = grad_shard.dtype
    wd = master_shard.dtype
    md = exp_avg.dtype
    vd = exp_avg_sq.dtype

    if gd == wd == md == vd == torch.bfloat16:
        return 0
    if gd == wd == md == vd == torch.float32:
        return 1
    if gd == torch.bfloat16 and wd == md == vd == torch.float32:
        return 2
    if gd == torch.float16 and wd == md == vd == torch.float32:
        return 3
    if gd == wd == md == vd == torch.float16:
        return 4

    raise AssertionError(
        "unsupported dtype combination for fused CUDA path: "
        f"grad={gd}, master={wd}, exp_avg={md}, exp_avg_sq={vd}"
    )


_barrier_slot = 0


@torch.no_grad()
def solution(
    grad_shard: Tensor,
    master_shard: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert step >= 1
    assert grad_shard.shape == master_shard.shape == exp_avg.shape == exp_avg_sq.shape
    assert grad_shard.is_cuda and master_shard.is_cuda and exp_avg.is_cuda and exp_avg_sq.is_cuda
    assert grad_shard.is_contiguous() and master_shard.is_contiguous()
    assert exp_avg.is_contiguous() and exp_avg_sq.is_contiguous()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    p = grad_shard.numel()
    assert p > 0
    assert world_size <= 8

    mode = _dtype_mode(grad_shard, master_shard, exp_avg, exp_avg_sq)

    ext = _get_ext()
    out, hdl, out_ptrs = _get_resources(p, master_shard.dtype, master_shard.device, world_size)

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    inv_bc1 = 1.0 / bc1
    inv_bc2 = 1.0 / bc2

    ext.launch_adam_publish(
        grad_shard,
        master_shard,
        exp_avg,
        exp_avg_sq,
        out_ptrs,
        p,
        rank,
        world_size,
        float(lr),
        float(beta1),
        float(beta2),
        float(inv_bc1),
        float(inv_bc2),
        float(eps),
        int(mode),
    )

    # GPU-side completion barrier: after this queued kernel completes, all ranks have
    # published their shard into every rank's symmetric output buffer.
    global _barrier_slot
    slot = _barrier_slot & 7
    _barrier_slot += 1
    ext.launch_symm_barrier(hdl.signal_pad_ptrs_dev, rank, world_size, slot)

    return out


__all__ = ["solution"]