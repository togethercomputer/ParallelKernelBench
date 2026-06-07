from __future__ import annotations

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

#define DTYPE_BF16 0
#define DTYPE_F32  1

static inline void check_cuda(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
}

__device__ __forceinline__ float load_f32(const float* p, int64_t i) {
    return p[i];
}

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}

__device__ __forceinline__ void store_f32(float* p, int64_t i, float v) {
    p[i] = v;
}

__device__ __forceinline__ void store_bf16(__nv_bfloat16* p, int64_t i, float v) {
    p[i] = __float2bfloat16(v);
}

// One CUDA thread handles one token. This is intentionally specialized for the
// balanced EP regime: num_experts == world_size <= 16, top_k <= 8, H commonly 64.
__global__ void router_pack_f32_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ gate_w,
    const float* __restrict__ gate_b,
    bool has_bias,
    float* __restrict__ xbuf,
    int* __restrict__ idxbuf,
    float* __restrict__ wtbuf,
    int* __restrict__ counts,
    int64_t T,
    int64_t H,
    int E,
    int K,
    int cap
) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float logits[16];
    float tmp[16];

    float maxv = -3.402823466e38f;
    for (int e = 0; e < E; ++e) {
        float acc = has_bias ? gate_b[e] : 0.0f;
        const float* gw = gate_w + (int64_t)e * H;
        const float* x = hidden + t * H;
        for (int64_t h = 0; h < H; ++h) {
            acc += x[h] * gw[h];
        }
        logits[e] = acc;
        tmp[e] = acc;
        maxv = fmaxf(maxv, acc);
    }

    float denom = 0.0f;
    for (int e = 0; e < E; ++e) denom += expf(logits[e] - maxv);

    for (int j = 0; j < K; ++j) {
        int best = 0;
        float bv = -3.402823466e38f;
        for (int e = 0; e < E; ++e) {
            if (tmp[e] > bv) {
                bv = tmp[e];
                best = e;
            }
        }
        tmp[best] = -3.402823466e38f;

        int pos = atomicAdd(counts + best, 1);
        if (pos < cap) {
            idxbuf[(int64_t)best * cap + pos] = (int)t;
            wtbuf[(int64_t)best * cap + pos] = expf(logits[best] - maxv) / denom;

            float* dst = xbuf + ((int64_t)best * cap + pos) * H;
            const float* src = hidden + t * H;
            for (int64_t h = 0; h < H; ++h) dst[h] = src[h];
        }
    }
}

__global__ void router_pack_bf16_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate_w,
    const __nv_bfloat16* __restrict__ gate_b,
    bool has_bias,
    __nv_bfloat16* __restrict__ xbuf,
    int* __restrict__ idxbuf,
    float* __restrict__ wtbuf,
    int* __restrict__ counts,
    int64_t T,
    int64_t H,
    int E,
    int K,
    int cap
) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float logits[16];
    float tmp[16];

    float maxv = -3.402823466e38f;
    for (int e = 0; e < E; ++e) {
        float acc = has_bias ? __bfloat162float(gate_b[e]) : 0.0f;
        const __nv_bfloat16* gw = gate_w + (int64_t)e * H;
        const __nv_bfloat16* x = hidden + t * H;
        for (int64_t h = 0; h < H; ++h) {
            acc += __bfloat162float(x[h]) * __bfloat162float(gw[h]);
        }
        logits[e] = acc;
        tmp[e] = acc;
        maxv = fmaxf(maxv, acc);
    }

    float denom = 0.0f;
    for (int e = 0; e < E; ++e) denom += expf(logits[e] - maxv);

    for (int j = 0; j < K; ++j) {
        int best = 0;
        float bv = -3.402823466e38f;
        for (int e = 0; e < E; ++e) {
            if (tmp[e] > bv) {
                bv = tmp[e];
                best = e;
            }
        }
        tmp[best] = -3.402823466e38f;

        int pos = atomicAdd(counts + best, 1);
        if (pos < cap) {
            idxbuf[(int64_t)best * cap + pos] = (int)t;
            wtbuf[(int64_t)best * cap + pos] = expf(logits[best] - maxv) / denom;

            __nv_bfloat16* dst = xbuf + ((int64_t)best * cap + pos) * H;
            const __nv_bfloat16* src = hidden + t * H;
            for (int64_t h = 0; h < H; ++h) dst[h] = src[h];
        }
    }
}

void router_pack(
    torch::Tensor hidden,
    torch::Tensor gate_w,
    torch::Tensor gate_b,
    bool has_bias,
    torch::Tensor xbuf,
    torch::Tensor idxbuf,
    torch::Tensor wtbuf,
    torch::Tensor counts,
    int64_t T,
    int64_t H,
    int E,
    int K,
    int cap,
    int dtype_enum
) {
    check_cuda(hidden, "hidden");
    check_cuda(gate_w, "gate_w");
    check_cuda(xbuf, "xbuf");
    check_cuda(idxbuf, "idxbuf");
    check_cuda(wtbuf, "wtbuf");
    check_cuda(counts, "counts");
    TORCH_CHECK(E <= 16, "router_pack supports world_size/num_experts <= 16");
    TORCH_CHECK(K <= 8, "router_pack supports top_k <= 8");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(counts.data_ptr<int>(), 0, E * sizeof(int), stream);

    int threads = 128;
    int blocks = (int)((T + threads - 1) / threads);

    if (dtype_enum == DTYPE_BF16) {
        router_pack_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(gate_w.data_ptr<at::BFloat16>()),
            has_bias ? reinterpret_cast<const __nv_bfloat16*>(gate_b.data_ptr<at::BFloat16>()) : nullptr,
            has_bias,
            reinterpret_cast<__nv_bfloat16*>(xbuf.data_ptr<at::BFloat16>()),
            idxbuf.data_ptr<int>(),
            wtbuf.data_ptr<float>(),
            counts.data_ptr<int>(),
            T, H, E, K, cap
        );
    } else {
        router_pack_f32_kernel<<<blocks, threads, 0, stream>>>(
            hidden.data_ptr<float>(),
            gate_w.data_ptr<float>(),
            has_bias ? gate_b.data_ptr<float>() : nullptr,
            has_bias,
            xbuf.data_ptr<float>(),
            idxbuf.data_ptr<int>(),
            wtbuf.data_ptr<float>(),
            counts.data_ptr<int>(),
            T, H, E, K, cap
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void gather_recv_f32_kernel(
    const long long* __restrict__ x_ptrs,
    const long long* __restrict__ cnt_ptrs,
    float* __restrict__ recv,
    int world,
    int rank,
    int cap,
    int64_t H
) {
    int64_t n = (int64_t)world * cap * H;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t h = i % H;
        int64_t q = i / H;
        int pos = (int)(q % cap);
        int src = (int)(q / cap);

        const int* cnt = reinterpret_cast<const int*>((uintptr_t)x_ptrs[0]); // dummy to placate compiler
        cnt = reinterpret_cast<const int*>((uintptr_t)cnt_ptrs[src]);
        int c = cnt[rank];

        float v = 0.0f;
        if (pos < c) {
            const float* xp = reinterpret_cast<const float*>((uintptr_t)x_ptrs[src]);
            v = xp[((int64_t)rank * cap + pos) * H + h];
        }
        recv[i] = v;
    }
}

__global__ void gather_recv_bf16_kernel(
    const long long* __restrict__ x_ptrs,
    const long long* __restrict__ cnt_ptrs,
    __nv_bfloat16* __restrict__ recv,
    int world,
    int rank,
    int cap,
    int64_t H
) {
    int64_t n = (int64_t)world * cap * H;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t h = i % H;
        int64_t q = i / H;
        int pos = (int)(q % cap);
        int src = (int)(q / cap);

        const int* cnt = reinterpret_cast<const int*>((uintptr_t)cnt_ptrs[src]);
        int c = cnt[rank];

        __nv_bfloat16 v = __float2bfloat16(0.0f);
        if (pos < c) {
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)x_ptrs[src]);
            v = xp[((int64_t)rank * cap + pos) * H + h];
        }
        recv[i] = v;
    }
}

void gather_recv(
    torch::Tensor x_ptrs,
    torch::Tensor cnt_ptrs,
    torch::Tensor recv,
    int world,
    int rank,
    int cap,
    int64_t H,
    int dtype_enum
) {
    check_cuda(x_ptrs, "x_ptrs");
    check_cuda(cnt_ptrs, "cnt_ptrs");
    check_cuda(recv, "recv");

    int64_t n = (int64_t)world * cap * H;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == DTYPE_BF16) {
        gather_recv_bf16_kernel<<<blocks, threads, 0, stream>>>(
            (const long long*)x_ptrs.data_ptr<int64_t>(),
            (const long long*)cnt_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(recv.data_ptr<at::BFloat16>()),
            world, rank, cap, H
        );
    } else {
        gather_recv_f32_kernel<<<blocks, threads, 0, stream>>>(
            (const long long*)x_ptrs.data_ptr<int64_t>(),
            (const long long*)cnt_ptrs.data_ptr<int64_t>(),
            recv.data_ptr<float>(),
            world, rank, cap, H
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void write_return_f32_kernel(
    const float* __restrict__ expert,
    float* __restrict__ ybuf,
    int world,
    int cap,
    int64_t H
) {
    int64_t n = (int64_t)world * cap * H;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        ybuf[i] = expert[i];
    }
}

__global__ void write_return_bf16_kernel(
    const __nv_bfloat16* __restrict__ expert,
    __nv_bfloat16* __restrict__ ybuf,
    int world,
    int cap,
    int64_t H
) {
    int64_t n = (int64_t)world * cap * H;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        ybuf[i] = expert[i];
    }
}

void write_return(
    torch::Tensor expert,
    torch::Tensor ybuf,
    int world,
    int cap,
    int64_t H,
    int dtype_enum
) {
    check_cuda(expert, "expert");
    check_cuda(ybuf, "ybuf");

    int64_t n = (int64_t)world * cap * H;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == DTYPE_BF16) {
        write_return_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(ybuf.data_ptr<at::BFloat16>()),
            world, cap, H
        );
    } else {
        write_return_f32_kernel<<<blocks, threads, 0, stream>>>(
            expert.data_ptr<float>(),
            ybuf.data_ptr<float>(),
            world, cap, H
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void scatter_acc_f32_kernel(
    const long long* __restrict__ y_ptrs,
    const int* __restrict__ counts,
    const int* __restrict__ idx,
    const float* __restrict__ wt,
    float* __restrict__ acc,
    int world,
    int rank,
    int cap,
    int64_t T,
    int64_t H
) {
    int64_t n = (int64_t)world * cap * H;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t h = i % H;
        int64_t q = i / H;
        int pos = (int)(q % cap);
        int dst = (int)(q / cap);

        int c = counts[dst];
        if (pos < c) {
            int tok = idx[(int64_t)dst * cap + pos];
            float w = wt[(int64_t)dst * cap + pos];
            const float* yp = reinterpret_cast<const float*>((uintptr_t)y_ptrs[dst]);
            float v = yp[((int64_t)rank * cap + pos) * H + h];
            atomicAdd(acc + (int64_t)tok * H + h, w * v);
        }
    }
}

__global__ void scatter_acc_bf16_kernel(
    const long long* __restrict__ y_ptrs,
    const int* __restrict__ counts,
    const int* __restrict__ idx,
    const float* __restrict__ wt,
    float* __restrict__ acc,
    int world,
    int rank,
    int cap,
    int64_t T,
    int64_t H
) {
    int64_t n = (int64_t)world * cap * H;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t h = i % H;
        int64_t q = i / H;
        int pos = (int)(q % cap);
        int dst = (int)(q / cap);

        int c = counts[dst];
        if (pos < c) {
            int tok = idx[(int64_t)dst * cap + pos];
            float w = wt[(int64_t)dst * cap + pos];
            const __nv_bfloat16* yp = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)y_ptrs[dst]);
            float v = __bfloat162float(yp[((int64_t)rank * cap + pos) * H + h]);
            atomicAdd(acc + (int64_t)tok * H + h, w * v);
        }
    }
}

__global__ void cast_out_f32_kernel(
    const float* __restrict__ acc,
    float* __restrict__ out,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) out[i] = acc[i];
}

__global__ void cast_out_bf16_kernel(
    const float* __restrict__ acc,
    __nv_bfloat16* __restrict__ out,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) out[i] = __float2bfloat16(acc[i]);
}

void final_combine(
    torch::Tensor y_ptrs,
    torch::Tensor counts,
    torch::Tensor idx,
    torch::Tensor wt,
    torch::Tensor acc,
    torch::Tensor out,
    int world,
    int rank,
    int cap,
    int64_t T,
    int64_t H,
    int dtype_enum
) {
    check_cuda(y_ptrs, "y_ptrs");
    check_cuda(counts, "counts");
    check_cuda(idx, "idx");
    check_cuda(wt, "wt");
    check_cuda(acc, "acc");
    check_cuda(out, "out");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(acc.data_ptr<float>(), 0, T * H * sizeof(float), stream);

    int64_t nscatter = (int64_t)world * cap * H;
    int threads = 256;
    int blocks = (int)((nscatter + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    if (dtype_enum == DTYPE_BF16) {
        scatter_acc_bf16_kernel<<<blocks, threads, 0, stream>>>(
            (const long long*)y_ptrs.data_ptr<int64_t>(),
            counts.data_ptr<int>(),
            idx.data_ptr<int>(),
            wt.data_ptr<float>(),
            acc.data_ptr<float>(),
            world, rank, cap, T, H
        );
    } else {
        scatter_acc_f32_kernel<<<blocks, threads, 0, stream>>>(
            (const long long*)y_ptrs.data_ptr<int64_t>(),
            counts.data_ptr<int>(),
            idx.data_ptr<int>(),
            wt.data_ptr<float>(),
            acc.data_ptr<float>(),
            world, rank, cap, T, H
        );
    }

    int64_t n = T * H;
    int cblocks = (int)((n + threads - 1) / threads);
    if (cblocks > 65535) cblocks = 65535;

    if (dtype_enum == DTYPE_BF16) {
        cast_out_bf16_kernel<<<cblocks, threads, 0, stream>>>(
            acc.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n
        );
    } else {
        cast_out_f32_kernel<<<cblocks, threads, 0, stream>>>(
            acc.data_ptr<float>(),
            out.data_ptr<float>(),
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void build_grad_expert_f32_kernel(
    const long long* __restrict__ grad_ptrs,
    const long long* __restrict__ cnt_ptrs,
    const long long* __restrict__ idx_ptrs,
    const long long* __restrict__ wt_ptrs,
    float* __restrict__ grad_expert,
    int world,
    int rank,
    int cap,
    int64_t H
) {
    int64_t n = (int64_t)world * cap * H;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t h = i % H;
        int64_t q = i / H;
        int pos = (int)(q % cap);
        int src = (int)(q / cap);

        const int* cnt = reinterpret_cast<const int*>((uintptr_t)cnt_ptrs[src]);
        int c = cnt[rank];

        float gv = 0.0f;
        if (pos < c) {
            const int* idx = reinterpret_cast<const int*>((uintptr_t)idx_ptrs[src]);
            const float* wt = reinterpret_cast<const float*>((uintptr_t)wt_ptrs[src]);
            const float* gout = reinterpret_cast<const float*>((uintptr_t)grad_ptrs[src]);
            int tok = idx[(int64_t)rank * cap + pos];
            float w = wt[(int64_t)rank * cap + pos];
            gv = w * gout[(int64_t)tok * H + h];
        }
        grad_expert[i] = gv;
    }
}

__global__ void build_grad_expert_bf16_kernel(
    const long long* __restrict__ grad_ptrs,
    const long long* __restrict__ cnt_ptrs,
    const long long* __restrict__ idx_ptrs,
    const long long* __restrict__ wt_ptrs,
    __nv_bfloat16* __restrict__ grad_expert,
    int world,
    int rank,
    int cap,
    int64_t H
) {
    int64_t n = (int64_t)world * cap * H;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int64_t h = i % H;
        int64_t q = i / H;
        int pos = (int)(q % cap);
        int src = (int)(q / cap);

        const int* cnt = reinterpret_cast<const int*>((uintptr_t)cnt_ptrs[src]);
        int c = cnt[rank];

        float gv = 0.0f;
        if (pos < c) {
            const int* idx = reinterpret_cast<const int*>((uintptr_t)idx_ptrs[src]);
            const float* wt = reinterpret_cast<const float*>((uintptr_t)wt_ptrs[src]);
            const __nv_bfloat16* gout = reinterpret_cast<const __nv_bfloat16*>((uintptr_t)grad_ptrs[src]);
            int tok = idx[(int64_t)rank * cap + pos];
            float w = wt[(int64_t)rank * cap + pos];
            gv = w * __bfloat162float(gout[(int64_t)tok * H + h]);
        }
        grad_expert[i] = __float2bfloat16(gv);
    }
}

void build_grad_expert(
    torch::Tensor grad_ptrs,
    torch::Tensor cnt_ptrs,
    torch::Tensor idx_ptrs,
    torch::Tensor wt_ptrs,
    torch::Tensor grad_expert,
    int world,
    int rank,
    int cap,
    int64_t H,
    int dtype_enum
) {
    check_cuda(grad_ptrs, "grad_ptrs");
    check_cuda(cnt_ptrs, "cnt_ptrs");
    check_cuda(idx_ptrs, "idx_ptrs");
    check_cuda(wt_ptrs, "wt_ptrs");
    check_cuda(grad_expert, "grad_expert");

    int64_t n = (int64_t)world * cap * H;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == DTYPE_BF16) {
        build_grad_expert_bf16_kernel<<<blocks, threads, 0, stream>>>(
            (const long long*)grad_ptrs.data_ptr<int64_t>(),
            (const long long*)cnt_ptrs.data_ptr<int64_t>(),
            (const long long*)idx_ptrs.data_ptr<int64_t>(),
            (const long long*)wt_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_expert.data_ptr<at::BFloat16>()),
            world, rank, cap, H
        );
    } else {
        build_grad_expert_f32_kernel<<<blocks, threads, 0, stream>>>(
            (const long long*)grad_ptrs.data_ptr<int64_t>(),
            (const long long*)cnt_ptrs.data_ptr<int64_t>(),
            (const long long*)idx_ptrs.data_ptr<int64_t>(),
            (const long long*)wt_ptrs.data_ptr<int64_t>(),
            grad_expert.data_ptr<float>(),
            world, rank, cap, H
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("router_pack", &router_pack, "Router top-k + fixed-capacity symmetric pack");
    m.def("gather_recv", &gather_recv, "Gather peer routed tokens into local expert batch");
    m.def("write_return", &write_return, "Write expert output into symmetric return slots");
    m.def("final_combine", &final_combine, "Peer read returned expert outputs and weighted scatter-add");
    m.def("build_grad_expert", &build_grad_expert, "Backward peer gather for expert output gradient");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_ep_balanced_symm_cuda_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    raise TypeError("optimized MoE path supports torch.bfloat16 and torch.float32")


def _ptr_tensor(ptrs, device):
    return torch.tensor([int(p) for p in ptrs], device=device, dtype=torch.int64)


def _get_resources(T: int, H: int, dtype: torch.dtype, device: torch.device, world: int, top_k: int, group):
    cap = T * top_k
    key = (T, H, dtype, device, world, top_k, id(group))
    res = _resource_cache.get(key)
    if res is not None:
        return res

    xbuf = symm_mem.empty((world, cap, H), device=device, dtype=dtype)
    ybuf = symm_mem.empty((world, cap, H), device=device, dtype=dtype)
    idxbuf = symm_mem.empty((world, cap), device=device, dtype=torch.int32)
    wtbuf = symm_mem.empty((world, cap), device=device, dtype=torch.float32)
    counts = symm_mem.empty((world,), device=device, dtype=torch.int32)
    gradbuf = symm_mem.empty((T, H), device=device, dtype=dtype)

    hx = symm_mem.rendezvous(xbuf, group)
    hy = symm_mem.rendezvous(ybuf, group)
    hidx = symm_mem.rendezvous(idxbuf, group)
    hwt = symm_mem.rendezvous(wtbuf, group)
    hcnt = symm_mem.rendezvous(counts, group)
    hg = symm_mem.rendezvous(gradbuf, group)

    recv = torch.empty((world * cap, H), device=device, dtype=dtype)
    out = torch.empty((T, H), device=device, dtype=dtype)
    acc = torch.empty((T, H), device=device, dtype=torch.float32)

    res = {
        "T": T,
        "H": H,
        "cap": cap,
        "world": world,
        "rank": dist.get_rank(group),
        "dtype": dtype,
        "dtype_enum": _dtype_enum(dtype),
        "xbuf": xbuf,
        "ybuf": ybuf,
        "idxbuf": idxbuf,
        "wtbuf": wtbuf,
        "counts": counts,
        "gradbuf": gradbuf,
        "hx": hx,
        "hy": hy,
        "hidx": hidx,
        "hwt": hwt,
        "hcnt": hcnt,
        "hg": hg,
        "recv": recv,
        "out": out,
        "acc": acc,
        "x_ptrs": _ptr_tensor(hx.buffer_ptrs, device),
        "y_ptrs": _ptr_tensor(hy.buffer_ptrs, device),
        "idx_ptrs": _ptr_tensor(hidx.buffer_ptrs, device),
        "wt_ptrs": _ptr_tensor(hwt.buffer_ptrs, device),
        "cnt_ptrs": _ptr_tensor(hcnt.buffer_ptrs, device),
        "grad_ptrs": _ptr_tensor(hg.buffer_ptrs, device),
    }
    _resource_cache[key] = res
    return res


class _PostCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs: torch.Tensor, res: dict) -> torch.Tensor:
        expert_outputs = expert_outputs.contiguous()
        ext = _get_ext()

        ext.write_return(
            expert_outputs,
            res["ybuf"],
            res["world"],
            res["cap"],
            res["H"],
            res["dtype_enum"],
        )

        # Device-side rendezvous: all owners have filled their return slots.
        res["hy"].barrier(channel=1)

        ext.final_combine(
            res["y_ptrs"],
            res["counts"],
            res["idxbuf"],
            res["wtbuf"],
            res["acc"],
            res["out"],
            res["world"],
            res["rank"],
            res["cap"],
            res["T"],
            res["H"],
            res["dtype_enum"],
        )

        ctx.res = res
        return res["out"]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        res = ctx.res
        ext = _get_ext()

        res["gradbuf"].copy_(grad_output.contiguous())
        res["hg"].barrier(channel=2)

        grad_expert = torch.empty(
            (res["world"] * res["cap"], res["H"]),
            device=grad_output.device,
            dtype=res["dtype"],
        )

        ext.build_grad_expert(
            res["grad_ptrs"],
            res["cnt_ptrs"],
            res["idx_ptrs"],
            res["wt_ptrs"],
            grad_expert,
            res["world"],
            res["rank"],
            res["cap"],
            res["H"],
            res["dtype_enum"],
        )

        res["hg"].barrier(channel=3)
        return grad_expert, None


def _expert_forward_cuda_backed(
    x: torch.Tensor,
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
) -> torch.Tensor:
    # Local dense MLP remains rank-local; with BF16 modules on H100 this uses tensor cores.
    gate = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_proj.weight, gate_proj.bias))
    up = torch.nn.functional.linear(x, up_proj.weight, up_proj.bias)
    return torch.nn.functional.linear(gate * up, down_proj.weight, down_proj.bias)


def solution(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    gate_proj: torch.nn.Linear,
    up_proj: torch.nn.Linear,
    down_proj: torch.nn.Linear,
    num_experts: int,
    top_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Balanced expert-parallel MoE forward using symmetric-memory UVA buffers
    instead of NCCL all_gather/all_to_all. Intended for num_experts == world_size.
    """
    assert hidden_states.is_cuda
    assert dist.is_initialized()
    group = group or dist.group.WORLD

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    assert num_experts == world, "balanced EP path requires num_experts == world_size"
    assert top_k <= 8
    assert world <= 16

    ext = _get_ext()

    hidden_dim = hidden_states.size(-1)
    hidden = hidden_states.reshape(-1, hidden_dim).contiguous()
    T = hidden.size(0)
    H = hidden.size(1)
    dtype = hidden.dtype
    dtype_enum = _dtype_enum(dtype)

    if gate_weight.dtype != dtype:
        gate_weight = gate_weight.to(dtype)
    gate_weight = gate_weight.contiguous()

    has_bias = gate_bias is not None
    if has_bias:
        if gate_bias.dtype != dtype:
            gate_bias = gate_bias.to(dtype)
        gate_bias_arg = gate_bias.contiguous()
    else:
        gate_bias_arg = torch.empty((0,), device=hidden.device, dtype=dtype)

    res = _get_resources(T, H, dtype, hidden.device, world, top_k, group)
    res["rank"] = rank

    # Router + local fixed-capacity pack into symmetric send buffer.
    ext.router_pack(
        hidden,
        gate_weight,
        gate_bias_arg,
        has_bias,
        res["xbuf"],
        res["idxbuf"],
        res["wtbuf"],
        res["counts"],
        T,
        H,
        num_experts,
        top_k,
        res["cap"],
        dtype_enum,
    )

    # Make routed token slots visible to expert owners.
    res["hx"].barrier(channel=0)
    res["hidx"].barrier(channel=0)
    res["hwt"].barrier(channel=0)
    res["hcnt"].barrier(channel=0)

    # Each rank gathers only the slots whose destination expert is this rank.
    ext.gather_recv(
        res["x_ptrs"],
        res["cnt_ptrs"],
        res["recv"],
        world,
        rank,
        res["cap"],
        H,
        dtype_enum,
    )

    # Rank-local expert computation.
    expert_outputs = _expert_forward_cuda_backed(
        res["recv"],
        gate_proj,
        up_proj,
        down_proj,
    ).contiguous()

    # Symmetric-memory return path + weighted scatter-add to original token order.
    out = _PostCombine.apply(expert_outputs, res)
    return out