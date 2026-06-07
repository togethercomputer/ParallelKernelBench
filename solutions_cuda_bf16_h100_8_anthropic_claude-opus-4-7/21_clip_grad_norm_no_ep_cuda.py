"""
FSDP2 clip_grad_norm using symmetric memory + multimem all-reduce on H100/NVSwitch.
- Custom CUDA kernel computes local sum of squares (BF16/FP32) directly into a
  symmetric memory scalar buffer.
- Multimem all-reduce (single-element FP32 in-switch SUM) replaces NCCL all_reduce.
- In-place clipping scaling is fused into a single kernel that scales all grads.
"""

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
#include <cstdint>

// ---- Signal-pad barrier ---------------------------------------------------
__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}
__device__ void blockwise_barrier(
    const uint64_t* signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

// ---- Sum of squares (BF16 / FP32) -----------------------------------------
template<int BLOCK>
__global__ void sumsq_bf16_kernel(
    const __nv_bfloat16* __restrict__ data,
    int64_t n,
    float* __restrict__ partial
) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * BLOCK + tid;
    int64_t stride = (int64_t)gridDim.x * BLOCK;
    float acc = 0.0f;
    for (int64_t i = idx; i < n; i += stride) {
        float v = __bfloat162float(data[i]);
        acc += v * v;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(partial, sdata[0]);
}

template<int BLOCK>
__global__ void sumsq_f32_kernel(
    const float* __restrict__ data,
    int64_t n,
    float* __restrict__ partial
) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * BLOCK + tid;
    int64_t stride = (int64_t)gridDim.x * BLOCK;
    float acc = 0.0f;
    for (int64_t i = idx; i < n; i += stride) {
        float v = data[i];
        acc += v * v;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(partial, sdata[0]);
}

void launch_sumsq(torch::Tensor t, torch::Tensor partial) {
    int64_t n = t.numel();
    if (n == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int BLOCK = 512;
    int blocks = (int)((n + BLOCK - 1) / BLOCK);
    if (blocks > 512) blocks = 512;
    if (t.scalar_type() == at::kBFloat16) {
        sumsq_bf16_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
            (const __nv_bfloat16*)t.data_ptr<at::BFloat16>(),
            n,
            partial.data_ptr<float>());
    } else if (t.scalar_type() == at::kFloat) {
        sumsq_f32_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
            t.data_ptr<float>(),
            n,
            partial.data_ptr<float>());
    } else {
        auto t32 = t.to(torch::kFloat32);
        sumsq_f32_kernel<BLOCK><<<blocks, BLOCK, 0, stream>>>(
            t32.data_ptr<float>(),
            n,
            partial.data_ptr<float>());
    }
}

// ---- Multimem all-reduce on a single FP32 scalar --------------------------
__global__ void multimem_allreduce_scalar_f32_kernel(
    uint64_t multicast_ptr,
    const uint64_t* signal_pad_ptrs,
    int rank,
    int world_size
) {
    blockwise_barrier(signal_pad_ptrs, 0, rank, world_size);
    __syncthreads();
    if (threadIdx.x == 0) {
        float val;
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
            : "=f"(val) : "l"(multicast_ptr) : "memory");
        asm volatile(
            "multimem.st.relaxed.sys.global.f32 [%0], %1;"
            :: "l"(multicast_ptr), "f"(val) : "memory");
    }
    __syncthreads();
    blockwise_barrier(signal_pad_ptrs, 1, rank, world_size);
}

void launch_multimem_allreduce_scalar(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int rank,
    int world_size
) {
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_scalar_f32_kernel<<<1, 32, 0, stream>>>(
        multicast_ptr, d_signal, rank, world_size);
}

// ---- Peer-pointer fallback for scalar all-reduce --------------------------
__global__ void p2p_allreduce_scalar_f32_kernel(
    const long long* ptrs,
    float* out,
    int world_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float s = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            s += *((const float*)ptrs[r]);
        }
        *out = s;
    }
}

void launch_p2p_allreduce_scalar(torch::Tensor ptrs_tensor, torch::Tensor out) {
    int world_size = ptrs_tensor.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    p2p_allreduce_scalar_f32_kernel<<<1, 32, 0, stream>>>(
        (const long long*)ptrs_tensor.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        world_size);
}

// ---- In-place scale -------------------------------------------------------
__global__ void scale_bf16_kernel(__nv_bfloat16* data, int64_t n, float coef) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float v = __bfloat162float(data[i]) * coef;
        data[i] = __float2bfloat16(v);
    }
}
__global__ void scale_f32_kernel(float* data, int64_t n, float coef) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        data[i] *= coef;
    }
}

void launch_scale(torch::Tensor t, double coef_d) {
    int64_t n = t.numel();
    if (n == 0) return;
    float coef = (float)coef_d;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    if (t.scalar_type() == at::kBFloat16) {
        scale_bf16_kernel<<<blocks, threads, 0, stream>>>(
            (__nv_bfloat16*)t.data_ptr<at::BFloat16>(), n, coef);
    } else if (t.scalar_type() == at::kFloat) {
        scale_f32_kernel<<<blocks, threads, 0, stream>>>(
            t.data_ptr<float>(), n, coef);
    } else {
        t.mul_(coef_d);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_sumsq", &launch_sumsq, "Sum of squares -> partial[0]");
    m.def("launch_multimem_allreduce_scalar", &launch_multimem_allreduce_scalar,
          "Multimem all-reduce a single fp32 scalar via multicast pointer");
    m.def("launch_p2p_allreduce_scalar", &launch_p2p_allreduce_scalar,
          "P2P all-reduce a single fp32 scalar via peer pointers");
    m.def("launch_scale", &launch_scale, "In-place scale by coef");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("clip_grad_norm_noep_ext", CUDA_SRC)
    return _ext


_symm_state = None
def _get_symm_state(device):
    global _symm_state
    if _symm_state is not None:
        return _symm_state
    buf = symm_mem.empty(1, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    out = torch.empty(1, device=device, dtype=torch.float32)
    _symm_state = (buf, hdl, ptrs_tensor, out)
    return _symm_state


@torch.no_grad()
def solution(
    grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    fsdp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    p = float(norm_type)

    # Find device
    dev = None
    for t in grad_tensors:
        if t is not None:
            dev = t.device
            break
    if dev is None:
        dev = torch.device("cuda", torch.cuda.current_device())

    ext = _get_ext()

    if dist.is_initialized() and fsdp_group is not None:
        # Use symmetric memory scalar buffer
        buf, hdl, ptrs_tensor, _out = _get_symm_state(dev)
        buf.zero_()

        # L2: accumulate sum of squares directly into symmetric buffer
        if abs(p - 2.0) < 1e-9:
            for t in grad_tensors:
                if t is None:
                    continue
                tc = t.detach()
                if not tc.is_contiguous():
                    tc = tc.contiguous()
                ext.launch_sumsq(tc, buf)
        else:
            # Generic p (rare here): fall back to torch.norm path
            acc = torch.zeros(1, device=dev, dtype=torch.float32)
            for t in grad_tensors:
                if t is None:
                    continue
                gn = torch.norm(t.detach().to(torch.float32), p=p)
                acc = acc + (gn ** p)
            buf.copy_(acc)

        # Try multimem all-reduce (NVSwitch) on the scalar; fallback to P2P sum
        try:
            multicast_ptr = int(hdl.multicast_ptr) if hdl.multicast_ptr else 0
        except Exception:
            multicast_ptr = 0

        if multicast_ptr != 0:
            ext.launch_multimem_allreduce_scalar(
                multicast_ptr,
                hdl.signal_pad_ptrs_dev,
                hdl.rank,
                hdl.world_size,
            )
            total_p = buf
        else:
            hdl.barrier(channel=0)
            ext.launch_p2p_allreduce_scalar(ptrs_tensor, _out)
            total_p = _out
            hdl.barrier(channel=1)
    else:
        # Single rank
        total_p = torch.zeros(1, device=dev, dtype=torch.float32)
        if abs(p - 2.0) < 1e-9:
            for t in grad_tensors:
                if t is None:
                    continue
                tc = t.detach()
                if not tc.is_contiguous():
                    tc = tc.contiguous()
                ext.launch_sumsq(tc, total_p)
        else:
            for t in grad_tensors:
                if t is None:
                    continue
                gn = torch.norm(t.detach().to(torch.float32), p=p)
                total_p = total_p + (gn ** p)

    total_norm = total_p.squeeze() ** (1.0 / p)

    # In-place clip
    max_norm_t = float(max_norm)
    tn_val = total_norm.item()
    if tn_val > max_norm_t and tn_val > 0.0:
        coef = max_norm_t / tn_val
        for t in grad_tensors:
            if t is not None:
                ext.launch_scale(t.detach(), coef)

    return total_norm