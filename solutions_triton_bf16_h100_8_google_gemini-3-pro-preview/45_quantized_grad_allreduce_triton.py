from __future__ import annotations

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
#include <vector>
#include <cstdint>

#define MAX_WORLD_SIZE 16

struct DevicePtrs {
    const int8_t* qs[MAX_WORLD_SIZE];
    const float* scales[MAX_WORLD_SIZE];
    const uint32_t* flags[MAX_WORLD_SIZE];
    const uint32_t* done_flags[MAX_WORLD_SIZE];
};

__global__ void fused_quantize_reduce_kernel(
    const __nv_bfloat16* __restrict__ grad,
    int8_t* __restrict__ symm_q,
    float* __restrict__ symm_scale,
    uint32_t* __restrict__ symm_flags,
    uint32_t* __restrict__ symm_done_flags,
    DevicePtrs ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t n,
    int block_size,
    int nb,
    int world_size,
    int rank,
    uint32_t step
) {
    // Persistent kernel loop mapping thread blocks to logical chunks
    for (int b = blockIdx.x; b < nb; b += gridDim.x) {
        int64_t start_idx = (int64_t)b * block_size;
        
        // --- Phase 0: Wait for peers to consume previous iteration's data ---
        if (step > 1 && threadIdx.x < world_size) {
            int p = threadIdx.x;
            if (p != rank) {
                const uint32_t* p_done = ptrs.done_flags[p] + b;
                uint32_t ready = 0;
                while (ready != step - 1) { 
                    asm volatile("ld.global.sys.b32 %0, [%1];" : "=r"(ready) : "l"(p_done) : "memory");
                }
            }
        }
        __syncthreads();
        
        // --- Phase 1: Local Quantize ---
        float max_val = 0.0f;
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            int64_t idx = start_idx + i;
            float val = 0.0f;
            if (idx < n) {
                val = __bfloat162float(grad[idx]);
            }
            max_val = fmaxf(max_val, fabsf(val));
        }
        
        // Warp and block reduction for block max
        unsigned int mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(mask, max_val, offset));
        }
        __shared__ float s_max[32];
        int lane = threadIdx.x % 32;
        int warp = threadIdx.x / 32;
        if (lane == 0) s_max[warp] = max_val;
        __syncthreads();
        
        if (warp == 0) {
            float val = (lane < (blockDim.x + 31) / 32) ? s_max[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(mask, val, offset));
            }
            if (lane == 0) {
                s_max[0] = fmaxf(val, 1e-8f) / 127.0f;
            }
        }
        __syncthreads();
        
        float scale = s_max[0];
        if (threadIdx.x == 0) {
            symm_scale[b] = scale;
        }
        
        // Apply scaling, round-to-even, and clamp to INT8 bounds
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            int64_t idx = start_idx + i;
            float val = 0.0f;
            if (idx < n) {
                val = __bfloat162float(grad[idx]);
            }
            float q_f = rintf(val / scale);
            int32_t q_i = (int32_t)q_f;
            if (q_i > 127) q_i = 127;
            if (q_i < -127) q_i = -127;
            
            symm_q[start_idx + i] = (int8_t)q_i;
        }
        
        // Ensure local quantization is visible globally over NVLink
        __threadfence_system();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            asm volatile("st.global.sys.b32 [%0], %1;" : : "l"(symm_flags + b), "r"(step) : "memory");
        }
        
        // --- Phase 2: Global Reduce via Spin-Wait ---
        if (threadIdx.x < world_size) {
            int p = threadIdx.x;
            if (p != rank) {
                const uint32_t* p_flag = ptrs.flags[p] + b;
                uint32_t ready = 0;
                while (ready != step) {
                    asm volatile("ld.global.sys.b32 %0, [%1];" : "=r"(ready) : "l"(p_flag) : "memory");
                }
            }
        }
        __syncthreads();
        
        __shared__ float s_scales[32]; 
        if (threadIdx.x < world_size) {
            s_scales[threadIdx.x] = ptrs.scales[threadIdx.x][b];
        }
        __syncthreads();
        
        float inv_ws = 1.0f / (float)world_size;
        
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            int64_t idx = start_idx + i;
            if (idx >= n) continue;
            
            float sum = 0.0f;
            for (int p = 0; p < world_size; p++) {
                int8_t q = ptrs.qs[p][idx];
                sum += (float)q * s_scales[p];
            }
            out[idx] = __float2bfloat16(sum * inv_ws);
        }
        
        // --- Phase 3: Signal Consumption Complete ---
        __threadfence_system();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            asm volatile("st.global.sys.b32 [%0], %1;" : : "l"(symm_done_flags + b), "r"(step) : "memory");
        }
    }
}

void fused_quantize_reduce_bf16(
    torch::Tensor grad,
    torch::Tensor symm_buf,
    int64_t offset_scale,
    int64_t offset_flags,
    int64_t offset_done,
    std::vector<int64_t> peer_buf_ptrs,
    torch::Tensor out,
    int64_t n,
    int block_size,
    int world_size,
    int rank,
    uint32_t step
) {
    TORCH_CHECK(grad.is_cuda() && out.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(world_size <= MAX_WORLD_SIZE, "world_size too large");

    int8_t* symm_q = reinterpret_cast<int8_t*>(symm_buf.data_ptr<uint8_t>());
    float* symm_scale = reinterpret_cast<float*>(symm_buf.data_ptr<uint8_t>() + offset_scale);
    uint32_t* symm_flags = reinterpret_cast<uint32_t*>(symm_buf.data_ptr<uint8_t>() + offset_flags);
    uint32_t* symm_done_flags = reinterpret_cast<uint32_t*>(symm_buf.data_ptr<uint8_t>() + offset_done);

    DevicePtrs ptrs;
    for (int i = 0; i < world_size; ++i) {
        uint8_t* base = reinterpret_cast<uint8_t*>(peer_buf_ptrs[i]);
        ptrs.qs[i] = reinterpret_cast<const int8_t*>(base);
        ptrs.scales[i] = reinterpret_cast<const float*>(base + offset_scale);
        ptrs.flags[i] = reinterpret_cast<const uint32_t*>(base + offset_flags);
        ptrs.done_flags[i] = reinterpret_cast<const uint32_t*>(base + offset_done);
    }

    int nb = (n + block_size - 1) / block_size;
    
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, grad.device().index());
    
    // Launch no more blocks than the GPU can strictly co-reside to guarantee deadlock-free execution
    int grids = num_sms;
    if (grids > nb) grids = nb;
    
    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    fused_quantize_reduce_kernel<<<grids, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()),
        symm_q,
        symm_scale,
        symm_flags,
        symm_done_flags,
        ptrs,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        n,
        block_size,
        nb,
        world_size,
        rank,
        step
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_quantize_reduce_bf16", &fused_quantize_reduce_bf16, "Fused UVA P2P int8 quantize and reduce (bf16)");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_quantize_reduce_ext", CUDA_SRC)
    return _ext


_symm_cache = None
_step_counter = 1

def _get_symm_state(n: int, block_size: int, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["block_size"] == block_size:
            return c

    padded_n = ((n + block_size - 1) // block_size) * block_size
    nb = padded_n // block_size

    size_q = padded_n
    size_scale = nb * 4
    size_flags = nb * 4
    size_done = nb * 4

    offset_scale = (size_q + 127) // 128 * 128
    offset_flags = (offset_scale + size_scale + 127) // 128 * 128
    offset_done = (offset_flags + size_flags + 127) // 128 * 128
    total_bytes = offset_done + size_done

    buf = symm_mem.empty(total_bytes, device=device, dtype=torch.uint8)

    # Initialize sync flags to cleanly start at zero
    buf[offset_flags : offset_flags + size_flags].zero_()
    buf[offset_done : offset_done + size_done].zero_()

    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    hdl.barrier(channel=0)

    peer_buf_ptrs = [int(hdl.buffer_ptrs[p]) for p in range(dist.get_world_size())]

    _symm_cache = {
        "n": n,
        "block_size": block_size,
        "buf": buf,
        "hdl": hdl,
        "peer_buf_ptrs": peer_buf_ptrs,
        "offset_scale": offset_scale,
        "offset_flags": offset_flags,
        "offset_done": offset_done,
    }
    return _symm_cache


@torch.no_grad()
def solution(
    flat_grad: Tensor,
    block_size: int,
) -> Tensor:
    global _step_counter

    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert block_size >= 1
    assert flat_grad.dtype == torch.bfloat16, "Grad must be in bf16 precision"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    n = flat_grad.numel()
    orig_shape = flat_grad.shape
    if n == 0:
        return flat_grad.clone()

    if rank == 0:
        _get_ext()
    dist.barrier()

    state = _get_symm_state(n, block_size, flat_grad.device)
    out = torch.empty_like(flat_grad)

    _get_ext().fused_quantize_reduce_bf16(
        flat_grad,
        state["buf"],
        state["offset_scale"],
        state["offset_flags"],
        state["offset_done"],
        state["peer_buf_ptrs"],
        out,
        n,
        block_size,
        world_size,
        rank,
        _step_counter
    )

    _step_counter += 1

    return out.reshape(orig_shape)

__all__ = ["solution"]