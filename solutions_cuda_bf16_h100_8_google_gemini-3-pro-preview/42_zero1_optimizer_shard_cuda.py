"""
Strategy:
1. **Symmetric Memory & Persistent Buffers**: We allocate `flat_p_buf` (parameters) and `flat_g_buf` (gradients) in PyTorch symmetric memory, caching them across steps to eliminate PyTorch's native memory reallocation and process group overheads.
2. **Fused Reduce-Scatter + Adam + All-Gather**: Instead of a full `dist.all_reduce` followed by slicing and a final `all_gather`, we completely fuse the communication and optimizer step. Each rank loads only its partition of gradients.
3. **NVSwitch Multimem Hardware Acceleration**: If operating in BF16, the kernel uses `multimem.ld_reduce` to instantly sum gradient partitions across the NVLink switch directly into registers, executes the Adam step, and immediately broadcasts the updated weights to all peers simultaneously via `multimem.st`.
4. **UVA Fallback for Remainder/Dtypes**: Handles sizes non-divisible by 8 or FP32 inputs via direct device-to-device peer memory accesses.
5. **Device-Side Sync**: Synchronization between forward/backward passes and the fused optimizer is handled using fast device-side barriers (`hdl.barrier()`).
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <typename moment_t>
__device__ __forceinline__ float2 do_adam_bf16x2_tmpl(
    uint32_t g_sum_u32, uint32_t w_u32,
    moment_t* m_ptr, moment_t* v_ptr,
    float scale_g, float lr, float beta1, float beta2, float eps,
    float bc1, float bc2
) {
    float2 g = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&g_sum_u32));
    float2 w = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&w_u32));
    
    g.x *= scale_g;
    g.y *= scale_g;
    
    float m0 = (float)m_ptr[0];
    float m1 = (float)m_ptr[1];
    float v0 = (float)v_ptr[0];
    float v1 = (float)v_ptr[1];
    
    m0 = beta1 * m0 + (1.0f - beta1) * g.x;
    m1 = beta1 * m1 + (1.0f - beta1) * g.y;
    
    v0 = beta2 * v0 + (1.0f - beta2) * g.x * g.x;
    v1 = beta2 * v1 + (1.0f - beta2) * g.y * g.y;
    
    m_ptr[0] = (moment_t)m0;
    m_ptr[1] = (moment_t)m1;
    v_ptr[0] = (moment_t)v0;
    v_ptr[1] = (moment_t)v1;
    
    float m_hat0 = m0 / bc1;
    float m_hat1 = m1 / bc1;
    
    float v_hat0 = v0 / bc2;
    float v_hat1 = v1 / bc2;
    
    w.x -= lr * m_hat0 / (sqrtf(v_hat0) + eps);
    w.y -= lr * m_hat1 / (sqrtf(v_hat1) + eps);
    
    return w;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float2 w) {
#if __CUDA_ARCH__ >= 800
    __nv_bfloat162 res = __floats2bfloat162_rn(w.x, w.y);
    return *reinterpret_cast<uint32_t*>(&res);
#else
    return 0;
#endif
}

template <typename moment_t>
__global__ void fused_zero1_multimem_bf16_kernel(
    uint64_t g_multicast_base,
    uint64_t p_multicast_base,
    const __nv_bfloat16* __restrict__ local_w,
    moment_t* __restrict__ m_part,
    moment_t* __restrict__ v_part,
    int64_t part_start,
    int64_t part_size,
    int world_size,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2
) {
    int64_t idx_8 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    float scale_g = 1.0f / (float)world_size;

    if (idx_8 + 7 < part_size) {
        int64_t global_idx = part_start + idx_8;
        
        uint64_t* g_ptr = reinterpret_cast<uint64_t*>(g_multicast_base + global_idx * 2);
        uint64_t* p_ptr = reinterpret_cast<uint64_t*>(p_multicast_base + global_idx * 2);
        
        uint32_t gx, gy, gz, gw;
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=r"(gx), "=r"(gy), "=r"(gz), "=r"(gw)
            : "l"(g_ptr)
            : "memory");
            
        const uint32_t* local_w_u32 = reinterpret_cast<const uint32_t*>(&local_w[global_idx]);
        uint32_t wx = local_w_u32[0];
        uint32_t wy = local_w_u32[1];
        uint32_t wz = local_w_u32[2];
        uint32_t ww = local_w_u32[3];
        
        float2 w01 = do_adam_bf16x2_tmpl(gx, wx, &m_part[idx_8],   &v_part[idx_8],   scale_g, lr, beta1, beta2, eps, bc1, bc2);
        float2 w23 = do_adam_bf16x2_tmpl(gy, wy, &m_part[idx_8+2], &v_part[idx_8+2], scale_g, lr, beta1, beta2, eps, bc1, bc2);
        float2 w45 = do_adam_bf16x2_tmpl(gz, wz, &m_part[idx_8+4], &v_part[idx_8+4], scale_g, lr, beta1, beta2, eps, bc1, bc2);
        float2 w67 = do_adam_bf16x2_tmpl(gw, ww, &m_part[idx_8+6], &v_part[idx_8+6], scale_g, lr, beta1, beta2, eps, bc1, bc2);
        
        uint32_t out_x = pack_bf16x2(w01);
        uint32_t out_y = pack_bf16x2(w23);
        uint32_t out_z = pack_bf16x2(w45);
        uint32_t out_w = pack_bf16x2(w67);
        
        asm volatile(
            "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(p_ptr), "r"(out_x), "r"(out_y), "r"(out_z), "r"(out_w)
            : "memory");
    }
}

template <typename weight_t, typename grad_t, typename moment_t>
__global__ void fused_zero1_uva_kernel(
    const long long* __restrict__ g_ptrs,
    const long long* __restrict__ p_ptrs,
    weight_t* __restrict__ local_w,
    moment_t* __restrict__ m_part,
    moment_t* __restrict__ v_part,
    int64_t part_start,
    int64_t part_size,
    int64_t m_part_offset,
    int world_size,
    float lr, float beta1, float beta2, float eps,
    float bc1, float bc2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < part_size) {
        int64_t global_idx = part_start + idx;
        int64_t local_idx = m_part_offset + idx;
        
        float g_sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const grad_t* peer_g = (const grad_t*)g_ptrs[r];
            g_sum += (float)peer_g[global_idx];
        }
        g_sum /= world_size;

        float m = (float)m_part[local_idx];
        float v = (float)v_part[local_idx];
        
        m = beta1 * m + (1.0f - beta1) * g_sum;
        v = beta2 * v + (1.0f - beta2) * g_sum * g_sum;
        
        m_part[local_idx] = (moment_t)m;
        v_part[local_idx] = (moment_t)v;
        
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        
        float w = (float)local_w[global_idx];
        w -= lr * m_hat / (sqrtf(v_hat) + eps);
        
        weight_t new_w = (weight_t)w;
        
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            weight_t* peer_p = (weight_t*)p_ptrs[r];
            peer_p[global_idx] = new_w;
        }
    }
}

__global__ void broadcast_uva_kernel_bf16(
    __nv_bfloat16* local_p, const __nv_bfloat16* src_p, int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < numel; idx += (int64_t)blockDim.x * gridDim.x) {
        local_p[idx] = src_p[idx];
    }
}
__global__ void broadcast_uva_kernel_f32(
    float* local_p, const float* src_p, int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < numel; idx += (int64_t)blockDim.x * gridDim.x) {
        local_p[idx] = src_p[idx];
    }
}

void launch_fused_multimem_bf16(
    uint64_t g_multicast, uint64_t p_multicast,
    torch::Tensor local_w, torch::Tensor m_part, torch::Tensor v_part,
    int64_t part_start, int64_t part_size, int world_size,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2
) {
    int threads = 256;
    int blocks = (part_size / 8 + threads - 1) / threads;
    if (blocks == 0) return;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (m_part.scalar_type() == at::ScalarType::Float) {
        fused_zero1_multimem_bf16_kernel<float><<<blocks, threads, 0, stream>>>(
            g_multicast, p_multicast,
            reinterpret_cast<const __nv_bfloat16*>(local_w.data_ptr()),
            m_part.data_ptr<float>(),
            v_part.data_ptr<float>(),
            part_start, part_size, world_size,
            lr, beta1, beta2, eps, bc1, bc2
        );
    } else {
        fused_zero1_multimem_bf16_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            g_multicast, p_multicast,
            reinterpret_cast<const __nv_bfloat16*>(local_w.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(m_part.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(v_part.data_ptr()),
            part_start, part_size, world_size,
            lr, beta1, beta2, eps, bc1, bc2
        );
    }
}

template <typename weight_t, typename moment_t>
void dispatch_uva_kernel(
    torch::Tensor g_ptrs, torch::Tensor p_ptrs,
    torch::Tensor local_w, torch::Tensor m_part, torch::Tensor v_part,
    int64_t part_start, int64_t part_size, int64_t m_part_offset, int world_size,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (part_size + threads - 1) / threads;
    if (blocks == 0) return;
    
    const long long* d_g_ptrs = (const long long*)g_ptrs.data_ptr<int64_t>();
    const long long* d_p_ptrs = (const long long*)p_ptrs.data_ptr<int64_t>();
    
    fused_zero1_uva_kernel<weight_t, weight_t, moment_t><<<blocks, threads, 0, stream>>>(
        d_g_ptrs, d_p_ptrs,
        reinterpret_cast<weight_t*>(local_w.data_ptr()),
        reinterpret_cast<moment_t*>(m_part.data_ptr()),
        reinterpret_cast<moment_t*>(v_part.data_ptr()),
        part_start, part_size, m_part_offset, world_size,
        lr, beta1, beta2, eps, bc1, bc2
    );
}

void launch_fused_uva(
    torch::Tensor g_ptrs, torch::Tensor p_ptrs,
    torch::Tensor local_w, torch::Tensor m_part, torch::Tensor v_part,
    int64_t part_start, int64_t part_size, int64_t m_part_offset, int world_size,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (local_w.scalar_type() == at::ScalarType::BFloat16) {
        if (m_part.scalar_type() == at::ScalarType::Float) {
            dispatch_uva_kernel<__nv_bfloat16, float>(g_ptrs, p_ptrs, local_w, m_part, v_part, part_start, part_size, m_part_offset, world_size, lr, beta1, beta2, eps, bc1, bc2, stream);
        } else {
            dispatch_uva_kernel<__nv_bfloat16, __nv_bfloat16>(g_ptrs, p_ptrs, local_w, m_part, v_part, part_start, part_size, m_part_offset, world_size, lr, beta1, beta2, eps, bc1, bc2, stream);
        }
    } else {
        if (m_part.scalar_type() == at::ScalarType::Float) {
            dispatch_uva_kernel<float, float>(g_ptrs, p_ptrs, local_w, m_part, v_part, part_start, part_size, m_part_offset, world_size, lr, beta1, beta2, eps, bc1, bc2, stream);
        }
    }
}

void launch_uva_broadcast(torch::Tensor local_p, int64_t src_ptr, int64_t numel) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (local_p.scalar_type() == at::ScalarType::BFloat16) {
        broadcast_uva_kernel_bf16<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(local_p.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(src_ptr),
            numel
        );
    } else {
        broadcast_uva_kernel_f32<<<blocks, threads, 0, stream>>>(
            local_p.data_ptr<float>(),
            reinterpret_cast<const float*>(src_ptr),
            numel
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_multimem_bf16", &launch_fused_multimem_bf16, "Fused Multimem BF16 Kernel");
    m.def("launch_fused_uva", &launch_fused_uva, "Fused UVA Kernel");
    m.def("launch_uva_broadcast", &launch_uva_broadcast, "UVA Broadcast Kernel");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_zero1_opt_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    key = (n, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    p_hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    p_ptrs = torch.tensor(p_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    g_buf = symm_mem.empty(n, device=device, dtype=dtype)
    g_hdl = symm_mem.rendezvous(g_buf, dist.group.WORLD)
    g_ptrs = torch.tensor(g_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    res = (buf, p_hdl, g_buf, g_hdl, p_ptrs, g_ptrs)
    _symm_cache[key] = res
    return res

def solution(
    X_local: Tensor,
    y_local: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    exp_avg_part: Tensor,
    exp_avg_sq_part: Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    templates = [W1, b1, W2, b2]
    flat_p = _flatten_dense_tensors(templates)
    numel = flat_p.numel()
    part = exp_avg_part.numel()
    start = rank * part
    assert numel == part * world_size
    
    buf, p_hdl, g_buf, g_hdl, p_ptrs, g_ptrs = _get_symm_state(numel, flat_p.dtype, flat_p.device)
    
    # 1. Sync & ensure weights match Rank 0 (replaces initial dist.broadcast)
    p_hdl.barrier(channel=0)
    buf.copy_(flat_p)
    p_hdl.barrier(channel=1)
    if rank != 0:
        _get_ext().launch_uva_broadcast(buf, p_ptrs[0].item(), numel)
    p_hdl.barrier(channel=2)
    
    # 2. Forward / Backward Pass
    param_views = _unflatten_dense_tensors(buf, templates)
    params = [t.detach().requires_grad_(True) for t in param_views]
    
    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()
    
    # 3. Flatten Grads into Symm Mem
    flat_g = _flatten_dense_tensors([p.grad for p in params])
    g_hdl.barrier(channel=0)
    g_buf.copy_(flat_g)
    g_hdl.barrier(channel=1)
    
    # 4. Fused Reduce-Scatter + Adam + All-Gather
    m_part = exp_avg_part.clone()
    v_part = exp_avg_sq_part.clone()
    
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    
    use_multimem = (buf.dtype == torch.bfloat16) and getattr(g_hdl, 'multicast_ptr', 0) != 0 and getattr(p_hdl, 'multicast_ptr', 0) != 0

    if use_multimem:
        numel_8 = part // 8
        if numel_8 > 0:
            _get_ext().launch_fused_multimem_bf16(
                int(g_hdl.multicast_ptr), int(p_hdl.multicast_ptr),
                buf, m_part, v_part,
                start, numel_8 * 8, world_size,
                lr, beta1, beta2, eps, bc1, bc2
            )
        remainder = part % 8
        if remainder > 0:
            _get_ext().launch_fused_uva(
                g_ptrs, p_ptrs,
                buf, m_part, v_part,
                start + numel_8 * 8, remainder, numel_8 * 8, world_size,
                lr, beta1, beta2, eps, bc1, bc2
            )
    else:
        _get_ext().launch_fused_uva(
            g_ptrs, p_ptrs,
            buf, m_part, v_part,
            start, part, 0, world_size,
            lr, beta1, beta2, eps, bc1, bc2
        )
        
    p_hdl.barrier(channel=3)
    
    out_params = _unflatten_dense_tensors(buf, templates)
    return (*out_params, m_part, v_part)

__all__ = ["solution"]