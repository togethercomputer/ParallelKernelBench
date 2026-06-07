from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

static inline int blocks_for(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

__device__ __forceinline__ float bf16_to_f32(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(const float x) {
    return __float2bfloat16_rn(x);
}

__global__ void gather_params_bf16_kernel(
    const int64_t* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ full,
    int64_t p,
    int world_size,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int r = (int)(idx / p);
        int64_t j = idx - (int64_t)r * p;
        const __nv_bfloat16* src =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[r]);
        full[idx] = src[j];
    }
}

__global__ void add_bias_relu_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int64_t rows,
    int64_t cols
) {
    int64_t n = rows * cols;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        int64_t c = idx % cols;
        float v = bf16_to_f32(x[idx]) + bf16_to_f32(bias[c]);
        v = v > 0.0f ? v : 0.0f;
        x[idx] = f32_to_bf16(v);
    }
}

__global__ void add_bias_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int64_t rows,
    int64_t cols
) {
    int64_t n = rows * cols;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        int64_t c = idx % cols;
        float v = bf16_to_f32(x[idx]) + bf16_to_f32(bias[c]);
        x[idx] = f32_to_bf16(v);
    }
}

__global__ void make_mse_dout_bf16_kernel(
    const __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ dout,
    int64_t n,
    float scale
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float v = (bf16_to_f32(out[idx]) - bf16_to_f32(y[idx])) * scale;
        dout[idx] = f32_to_bf16(v);
    }
}

__global__ void relu_backward_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float mask = bf16_to_f32(h[idx]) > 0.0f ? 1.0f : 0.0f;
        dh[idx] = f32_to_bf16(bf16_to_f32(dh[idx]) * mask);
    }
}

__global__ void reduce_bias_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ bgrad,
    int64_t rows,
    int64_t cols
) {
    int c = blockIdx.x;
    float sum = 0.0f;

    for (int64_t r = threadIdx.x; r < rows; r += blockDim.x) {
        sum += bf16_to_f32(x[r * cols + c]);
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        bgrad[c] = f32_to_bf16(smem[0]);
    }
}

__global__ void pack_grads_bf16_kernel(
    __nv_bfloat16* __restrict__ flat,
    const __nv_bfloat16* __restrict__ g0,
    const __nv_bfloat16* __restrict__ g1,
    const __nv_bfloat16* __restrict__ g2,
    const __nv_bfloat16* __restrict__ g3,
    int64_t n0,
    int64_t n1,
    int64_t n2,
    int64_t n3,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    int64_t o1 = n0;
    int64_t o2 = n0 + n1;
    int64_t o3 = n0 + n1 + n2;

    for (; idx < total; idx += stride) {
        if (idx < n0) {
            flat[idx] = g0[idx];
        } else if (idx < o2) {
            flat[idx] = g1[idx - o1];
        } else if (idx < o3) {
            flat[idx] = g2[idx - o2];
        } else {
            flat[idx] = g3[idx - o3];
        }
    }
}

__global__ void reduce_scatter_adamw_bf16_kernel(
    const int64_t* __restrict__ grad_ptrs,
    const __nv_bfloat16* __restrict__ theta_in,
    const __nv_bfloat16* __restrict__ m_in,
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ theta_out,
    __nv_bfloat16* __restrict__ m_out,
    __nv_bfloat16* __restrict__ v_out,
    int64_t p,
    int64_t shard_offset,
    int world_size,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < p; idx += stride) {
        float gsum = 0.0f;

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r < world_size) {
                const __nv_bfloat16* peer =
                    reinterpret_cast<const __nv_bfloat16*>((uintptr_t)grad_ptrs[r]);
                gsum += bf16_to_f32(peer[shard_offset + idx]);
            }
        }

        float g = gsum / (float)world_size;
        g = bf16_to_f32(f32_to_bf16(g));

        float m_old = bf16_to_f32(m_in[idx]);
        float v_old = bf16_to_f32(v_in[idx]);

        float m_new_f = beta1 * m_old + (1.0f - beta1) * g;
        float v_new_f = beta2 * v_old + (1.0f - beta2) * g * g;

        __nv_bfloat16 m_new_b = f32_to_bf16(m_new_f);
        __nv_bfloat16 v_new_b = f32_to_bf16(v_new_f);

        float m_corr = bf16_to_f32(m_new_b) / bc1;
        float v_corr = bf16_to_f32(v_new_b) / bc2;
        float upd = m_corr / (sqrtf(v_corr) + eps);

        float theta_old = bf16_to_f32(theta_in[idx]);

        // Match AdamW reference ordering: Adam step rounded, then decoupled WD rounded.
        float t1 = theta_old - lr * upd;
        t1 = bf16_to_f32(f32_to_bf16(t1));
        float t2 = t1 - lr * weight_decay * theta_old;

        theta_out[idx] = f32_to_bf16(t2);
        m_out[idx] = m_new_b;
        v_out[idx] = v_new_b;
    }
}

void gather_params_bf16(torch::Tensor ptrs, torch::Tensor full, int64_t p) {
    TORCH_CHECK(full.is_cuda() && ptrs.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(full.scalar_type() == torch::kBFloat16, "BF16 full tensor required");

    int world_size = (int)ptrs.numel();
    int64_t total = full.numel();
    int threads = 256;
    int blocks = blocks_for(total, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_params_bf16_kernel<<<blocks, threads, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(full.data_ptr<at::BFloat16>()),
        p,
        world_size,
        total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_bias_relu_bf16(torch::Tensor x, torch::Tensor bias, int64_t rows, int64_t cols) {
    int64_t n = rows * cols;
    int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    add_bias_relu_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
        rows,
        cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_bias_bf16(torch::Tensor x, torch::Tensor bias, int64_t rows, int64_t cols) {
    int64_t n = rows * cols;
    int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    add_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
        rows,
        cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void make_mse_dout_bf16(torch::Tensor out, torch::Tensor y, torch::Tensor dout, float scale) {
    int64_t n = out.numel();
    int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    make_mse_dout_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dout.data_ptr<at::BFloat16>()),
        n,
        scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void relu_backward_inplace_bf16(torch::Tensor dh, torch::Tensor h) {
    int64_t n = dh.numel();
    int threads = 256;
    int blocks = blocks_for(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    relu_backward_inplace_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(dh.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_bias_bf16(torch::Tensor x, torch::Tensor bgrad, int64_t rows, int64_t cols) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_bias_bf16_kernel<<<(int)cols, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(bgrad.data_ptr<at::BFloat16>()),
        rows,
        cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_grads_bf16(
    torch::Tensor flat,
    torch::Tensor g0,
    torch::Tensor g1,
    torch::Tensor g2,
    torch::Tensor g3
) {
    int64_t n0 = g0.numel();
    int64_t n1 = g1.numel();
    int64_t n2 = g2.numel();
    int64_t n3 = g3.numel();
    int64_t total = n0 + n1 + n2 + n3;

    int threads = 256;
    int blocks = blocks_for(total, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    pack_grads_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(g0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(g1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(g2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(g3.data_ptr<at::BFloat16>()),
        n0,
        n1,
        n2,
        n3,
        total
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_scatter_adamw_bf16(
    torch::Tensor grad_ptrs,
    torch::Tensor theta_in,
    torch::Tensor m_in,
    torch::Tensor v_in,
    torch::Tensor theta_out,
    torch::Tensor m_out,
    torch::Tensor v_out,
    int64_t p,
    int64_t shard_offset,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1,
    float bc2
) {
    int world_size = (int)grad_ptrs.numel();
    int threads = 256;
    int blocks = blocks_for(p, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    reduce_scatter_adamw_bf16_kernel<<<blocks, threads, 0, stream>>>(
        grad_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<const __nv_bfloat16*>(theta_in.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(m_in.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_in.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(theta_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(m_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr<at::BFloat16>()),
        p,
        shard_offset,
        world_size,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bc1,
        bc2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_params_bf16", &gather_params_bf16, "symmetric-memory BF16 all-gather");
    m.def("add_bias_relu_bf16", &add_bias_relu_bf16, "BF16 bias + ReLU");
    m.def("add_bias_bf16", &add_bias_bf16, "BF16 bias add");
    m.def("make_mse_dout_bf16", &make_mse_dout_bf16, "BF16 MSE output gradient");
    m.def("relu_backward_inplace_bf16", &relu_backward_inplace_bf16, "BF16 ReLU backward");
    m.def("reduce_bias_bf16", &reduce_bias_bf16, "BF16 column reduction");
    m.def("pack_grads_bf16", &pack_grads_bf16, "pack four BF16 gradients");
    m.def("reduce_scatter_adamw_bf16", &reduce_scatter_adamw_bf16,
          "UVA reduce-scatter fused with AdamW");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_step_e2e_bf16_symm_h100_ext", CUDA_SRC)
    return _ext


_comm_cache = {}
_work_cache = {}


def _shape_key(param_shapes: Sequence[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in s) for s in param_shapes)


def _get_comm(p: int, total: int, dtype: torch.dtype, device: torch.device):
    key = (p, total, dtype, device)
    cached = _comm_cache.get(key)
    if cached is not None:
        return cached

    param_buf = symm_mem.empty(p, device=device, dtype=dtype)
    param_hdl = symm_mem.rendezvous(param_buf, dist.group.WORLD)
    param_ptrs = torch.tensor(param_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    grad_buf = symm_mem.empty(total, device=device, dtype=dtype)
    grad_hdl = symm_mem.rendezvous(grad_buf, dist.group.WORLD)
    grad_ptrs = torch.tensor(grad_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    full_flat = torch.empty(total, device=device, dtype=dtype)

    cached = {
        "param_buf": param_buf,
        "param_hdl": param_hdl,
        "param_ptrs": param_ptrs,
        "grad_buf": grad_buf,
        "grad_hdl": grad_hdl,
        "grad_ptrs": grad_ptrs,
        "full_flat": full_flat,
    }
    _comm_cache[key] = cached
    return cached


def _get_work(
    B: int,
    I: int,
    H: int,
    O: int,
    p: int,
    dtype: torch.dtype,
    device: torch.device,
):
    key = (B, I, H, O, p, dtype, device)
    cached = _work_cache.get(key)
    if cached is not None:
        return cached

    cached = {
        "h": torch.empty((B, H), device=device, dtype=dtype),
        "out": torch.empty((B, O), device=device, dtype=dtype),
        "dout": torch.empty((B, O), device=device, dtype=dtype),
        "dh": torch.empty((B, H), device=device, dtype=dtype),
        "dw1": torch.empty((H, I), device=device, dtype=dtype),
        "db1": torch.empty((H,), device=device, dtype=dtype),
        "dw2": torch.empty((O, H), device=device, dtype=dtype),
        "db2": torch.empty((O,), device=device, dtype=dtype),
        "theta": torch.empty((p,), device=device, dtype=dtype),
        "m": torch.empty((p,), device=device, dtype=dtype),
        "v": torch.empty((p,), device=device, dtype=dtype),
    }
    _work_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    X_local: Tensor,
    y_local: Tensor,
    flat_param_shard: Tensor,
    param_shapes: Sequence[tuple[int, ...]],
    exp_avg_shard: Tensor,
    exp_avg_sq_shard: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert step >= 1
    assert flat_param_shard.is_cuda
    assert flat_param_shard.dtype == torch.bfloat16, "optimized path expects BF16 parameters"
    assert X_local.dtype == torch.bfloat16 and y_local.dtype == torch.bfloat16
    assert exp_avg_shard.dtype == torch.bfloat16 and exp_avg_sq_shard.dtype == torch.bfloat16

    ext = _get_ext()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    p = flat_param_shard.numel()
    assert exp_avg_shard.numel() == p == exp_avg_sq_shard.numel()

    ps = _shape_key(param_shapes)
    assert len(ps) == 4, "expected MLP params: W1, b1, W2, b2"

    H, I = ps[0]
    assert ps[1] == (H,)
    O, H2 = ps[2]
    assert H2 == H
    assert ps[3] == (O,)

    total = sum(math.prod(s) for s in ps)
    assert total == p * world_size

    B = X_local.shape[0]
    assert X_local.shape[1] == I
    assert y_local.shape == (B, O)

    device = flat_param_shard.device
    dtype = flat_param_shard.dtype

    X = X_local if X_local.is_contiguous() else X_local.contiguous()
    y = y_local if y_local.is_contiguous() else y_local.contiguous()
    theta_in = flat_param_shard if flat_param_shard.is_contiguous() else flat_param_shard.contiguous()
    m_in = exp_avg_shard if exp_avg_shard.is_contiguous() else exp_avg_shard.contiguous()
    v_in = exp_avg_sq_shard if exp_avg_sq_shard.is_contiguous() else exp_avg_sq_shard.contiguous()

    comm = _get_comm(p, total, dtype, device)
    work = _get_work(B, I, H, O, p, dtype, device)

    param_buf = comm["param_buf"]
    param_hdl = comm["param_hdl"]
    param_ptrs = comm["param_ptrs"]
    full_flat = comm["full_flat"]

    # Publish local parameter shard, then gather peer shards through UVA pointers.
    param_buf.copy_(theta_in)
    param_hdl.barrier(channel=0)
    ext.gather_params_bf16(param_ptrs, full_flat, p)

    n_w1 = H * I
    n_b1 = H
    n_w2 = O * H
    n_b2 = O

    off_w1 = 0
    off_b1 = off_w1 + n_w1
    off_w2 = off_b1 + n_b1
    off_b2 = off_w2 + n_w2

    w1 = full_flat.narrow(0, off_w1, n_w1).view(H, I)
    b1 = full_flat.narrow(0, off_b1, n_b1)
    w2 = full_flat.narrow(0, off_w2, n_w2).view(O, H)
    b2 = full_flat.narrow(0, off_b2, n_b2)

    h = work["h"]
    out = work["out"]
    dout = work["dout"]
    dh = work["dh"]
    dw1 = work["dw1"]
    db1 = work["db1"]
    dw2 = work["dw2"]
    db2 = work["db2"]

    # Manual forward/backward. GEMMs dispatch to BF16 tensor cores; surrounding ops are fused CUDA kernels.
    torch.mm(X, w1.t(), out=h)
    ext.add_bias_relu_bf16(h, b1, B, H)

    torch.mm(h, w2.t(), out=out)
    ext.add_bias_bf16(out, b2, B, O)

    ext.make_mse_dout_bf16(out, y, dout, float(2.0 / (B * O)))

    torch.mm(dout.t(), h, out=dw2)
    ext.reduce_bias_bf16(dout, db2, B, O)

    torch.mm(dout, w2, out=dh)
    ext.relu_backward_inplace_bf16(dh, h)

    torch.mm(dh.t(), X, out=dw1)
    ext.reduce_bias_bf16(dh, db1, B, H)

    grad_buf = comm["grad_buf"]
    grad_hdl = comm["grad_hdl"]
    grad_ptrs = comm["grad_ptrs"]

    # Publish this rank's full local gradient in flat parameter order.
    ext.pack_grads_bf16(grad_buf, dw1, db1, dw2, db2)
    grad_hdl.barrier(channel=1)

    theta_out = work["theta"]
    m_out = work["m"]
    v_out = work["v"]

    # Avoid accidental in-place semantics if caller feeds back a cached output tensor.
    if theta_out.data_ptr() == theta_in.data_ptr():
        theta_out = torch.empty_like(theta_in)
    if m_out.data_ptr() == m_in.data_ptr():
        m_out = torch.empty_like(m_in)
    if v_out.data_ptr() == v_in.data_ptr():
        v_out = torch.empty_like(v_in)

    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)

    # Device-side reduce-scatter over only this rank's shard, fused with AdamW.
    shard_offset = rank * p
    ext.reduce_scatter_adamw_bf16(
        grad_ptrs,
        theta_in,
        m_in,
        v_in,
        theta_out,
        m_out,
        v_out,
        p,
        shard_offset,
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
        float(bc1),
        float(bc2),
    )

    return theta_out, m_out, v_out


__all__ = ["solution"]