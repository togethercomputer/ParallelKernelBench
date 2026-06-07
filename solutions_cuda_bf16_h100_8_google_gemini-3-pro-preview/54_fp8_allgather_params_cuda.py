"""
Strategy:
1. **Pipelined device-side communication**: Instead of a PyTorch-level BF16 all-gather after a local FP8 round-trip, we directly fuse the quantization and communicate the *compressed FP8 buffers* across peers using `torch.distributed._symmetric_memory`.
2. **Fused Gather + Dequantize**: A custom multi-block CUDA kernel utilizes direct peer-to-peer memory access over NVLink (pull-based). Each block pulls FP8 parameter shards and `scale` variables from its designated peer, dequantizes them locally on the fly, and streams the restored BF16 values straight into the full output tensor.
3. **Optimized NVLink throughput**: Uses vectorized 16-byte memory instructions (`uint4`) for all global memory reads and writes over NVLink, saturating the bus bandwidth and significantly speeding up the memory-bound all-gather.
4. **No host syncs**: We use `scale` variables directly via device pointers between our custom kernels without copying back to CPU, keeping the execution entirely asynchronous.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor
from utils.cuda_helpers import compile_cuda_extension

_FP8_E4M3_MAX = 448.0

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ float e4m3_to_float(uint8_t x) {
    float res;
    uint32_t ix = x;
    asm volatile("cvt.f32.e4m3 %0, %1;" : "=f"(res) : "r"(ix));
    return res;
}

__global__ void quantize_fused_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ out_fp8,
    const float* __restrict__ scale_ptr,
    int64_t p
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    
    bool aligned = (((uintptr_t)input % 16) == 0) && (((uintptr_t)out_fp8 % 16) == 0);
    
    if (aligned) {
        int64_t p_16 = p / 16;
        int64_t offset_16 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride_16 = (int64_t)gridDim.x * blockDim.x;
        
        for (int64_t i = offset_16; i < p_16; i += stride_16) {
            uint4 in_bf16_0 = reinterpret_cast<const uint4*>(input)[i * 2];
            uint4 in_bf16_1 = reinterpret_cast<const uint4*>(input)[i * 2 + 1];
            
            __nv_bfloat16 bf16_vals[16];
            ((uint4*)bf16_vals)[0] = in_bf16_0;
            ((uint4*)bf16_vals)[1] = in_bf16_1;
            
            uint8_t bytes[16];
            #pragma unroll
            for(int j=0; j<16; ++j) {
                float val_f32 = __bfloat162float(bf16_vals[j]);
                float scaled = val_f32 * inv_scale;
                uint32_t fp8_val;
                asm volatile("cvt.rn.satfinite.e4m3.f32 %0, %1;" : "=r"(fp8_val) : "f"(scaled));
                bytes[j] = (uint8_t)fp8_val;
            }
            
            reinterpret_cast<uint4*>(out_fp8)[i] = *(uint4*)bytes;
        }
        
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (int64_t i = p_16 * 16; i < p; ++i) {
                float val_f32 = __bfloat162float(input[i]);
                float scaled = val_f32 * inv_scale;
                uint32_t fp8_val;
                asm volatile("cvt.rn.satfinite.e4m3.f32 %0, %1;" : "=r"(fp8_val) : "f"(scaled));
                out_fp8[i] = (uint8_t)fp8_val;
            }
        }
    } else {
        int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)gridDim.x * blockDim.x;
        for (int64_t i = tid; i < p; i += stride) {
            float val_f32 = __bfloat162float(input[i]);
            float scaled = val_f32 * inv_scale;
            uint32_t fp8_val;
            asm volatile("cvt.rn.satfinite.e4m3.f32 %0, %1;" : "=r"(fp8_val) : "f"(scaled));
            out_fp8[i] = (uint8_t)fp8_val;
        }
    }
}

__global__ void dequantize_and_gather_kernel(
    const uint64_t* __restrict__ peer_fp8_ptrs,
    const uint64_t* __restrict__ peer_scale_ptrs,
    __nv_bfloat16* __restrict__ out_full,
    int64_t p,
    int world_size
) {
    int r = blockIdx.y;
    if (r >= world_size) return;
    
    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        const float* scale_ptr = reinterpret_cast<const float*>(peer_scale_ptrs[r]);
        shared_scale = *scale_ptr;
    }
    __syncthreads();
    
    const uint8_t* src_fp8 = reinterpret_cast<const uint8_t*>(peer_fp8_ptrs[r]);
    float scale = shared_scale;
    __nv_bfloat16* out_rank_ptr = out_full + (int64_t)r * p;
    
    bool aligned = (((uintptr_t)src_fp8 % 16) == 0) && (((uintptr_t)out_rank_ptr % 16) == 0);
    
    if (aligned) {
        int64_t p_16 = p / 16;
        int64_t offset_16 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride_16 = (int64_t)gridDim.x * blockDim.x;
        
        for (int64_t i = offset_16; i < p_16; i += stride_16) {
            uint4 fp8_vec = reinterpret_cast<const uint4*>(src_fp8)[i];
            
            uint8_t bytes[16];
            *(uint4*)bytes = fp8_vec;
            
            __nv_bfloat16 out_bf16[16];
            #pragma unroll
            for(int j=0; j<16; ++j) {
                float val_f32 = e4m3_to_float(bytes[j]);
                out_bf16[j] = __float2bfloat16(val_f32 * scale);
            }
            
            reinterpret_cast<uint4*>(out_rank_ptr)[i * 2] = ((uint4*)out_bf16)[0];
            reinterpret_cast<uint4*>(out_rank_ptr)[i * 2 + 1] = ((uint4*)out_bf16)[1];
        }
        
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            for (int64_t i = p_16 * 16; i < p; ++i) {
                uint8_t val_fp8 = src_fp8[i];
                float val_f32 = e4m3_to_float(val_fp8);
                out_rank_ptr[i] = __float2bfloat16(val_f32 * scale);
            }
        }
    } else {
        int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)gridDim.x * blockDim.x;
        for (int64_t i = tid; i < p; i += stride) {
            uint8_t val_fp8 = src_fp8[i];
            float val_f32 = e4m3_to_float(val_fp8);
            out_rank_ptr[i] = __float2bfloat16(val_f32 * scale);
        }
    }
}

void launch_quantize(
    torch::Tensor input,
    torch::Tensor out_fp8,
    torch::Tensor scale_tensor,
    int64_t p
) {
    int threads = 256;
    int blocks = std::min((int)((p/16 + threads - 1) / threads), 2048);
    if (blocks == 0) blocks = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    quantize_fused_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        out_fp8.data_ptr<uint8_t>(),
        scale_tensor.data_ptr<float>(),
        p
    );
}

void launch_gather(
    torch::Tensor peer_fp8_ptrs,
    torch::Tensor peer_scale_ptrs,
    torch::Tensor out_full,
    int64_t p,
    int world_size
) {
    int threads = 256;
    int blocks_x = std::min((int)((p/16 + threads - 1) / threads), 1024);
    if (blocks_x == 0) blocks_x = 1;
    dim3 blocks(blocks_x, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    dequantize_and_gather_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(peer_fp8_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(peer_scale_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(out_full.data_ptr<at::BFloat16>()),
        p,
        world_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_quantize", &launch_quantize, "Quantize BF16 to FP8 E4M3");
    m.def("launch_gather", &launch_gather, "Gather FP8 and dequantize to BF16");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fp8_allgather_ext", CUDA_SRC)
    return _ext

_resource_cache = {}

def _get_resources(p: int, device: torch.device):
    key = (p, device)
    if key in _resource_cache:
        return _resource_cache[key]
    
    fp8_buf = symm_mem.empty(p, device=device, dtype=torch.uint8)
    fp8_hdl = symm_mem.rendezvous(fp8_buf, dist.group.WORLD)
    fp8_ptrs = torch.tensor(fp8_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    scale_buf = symm_mem.empty(1, device=device, dtype=torch.float32)
    scale_hdl = symm_mem.rendezvous(scale_buf, dist.group.WORLD)
    scale_ptrs = torch.tensor(scale_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    res = (fp8_buf, fp8_hdl, fp8_ptrs, scale_buf, scale_hdl, scale_ptrs)
    _resource_cache[key] = res
    return res

@torch.no_grad()
def solution(flat_param_shard: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"

    world_size = dist.get_world_size()
    p = flat_param_shard.numel()
    
    # Fallback to PyTorch reference if it is not BF16
    if flat_param_shard.dtype != torch.bfloat16:
        cur_abs_max = flat_param_shard.abs().max().to(torch.float32)
        out_hist = torch.roll(amax_history, shifts=-1, dims=0)
        out_hist[-1] = cur_abs_max.to(dtype=out_hist.dtype)
        scale = out_hist.max().clamp(min=1e-12).to(torch.float32) / _FP8_E4M3_MAX
        
        xf = flat_param_shard.float()
        qs = xf / scale
        q = qs.to(torch.float8_e4m3fn)
        recon = (q.float() * scale).to(dtype=flat_param_shard.dtype)
        
        full = torch.empty(world_size * p, dtype=flat_param_shard.dtype, device=flat_param_shard.device)
        dist.all_gather_into_tensor(full, recon.contiguous())
        return full, out_hist

    # Accelerated path using Custom NVLink Gathering
    ext = _get_ext()
    flat_param_shard = flat_param_shard.contiguous()
    fp8_buf, fp8_hdl, fp8_ptrs, scale_buf, scale_hdl, scale_ptrs = _get_resources(p, flat_param_shard.device)

    # 1. Update AMAX purely on-device using native PyTorch
    cur_abs_max = flat_param_shard.abs().max().float()
    updated_hist = torch.roll(amax_history, shifts=-1, dims=0)
    updated_hist[-1] = cur_abs_max.to(updated_hist.dtype)

    # 2. Compute dynamic scale and deposit it onto symmetric pointer
    scale = updated_hist.max().clamp(min=1e-12).float() / _FP8_E4M3_MAX
    scale_buf.copy_(scale.view(-1))

    # 3. Fast device-local quantization into our outgoing buffer
    ext.launch_quantize(flat_param_shard, fp8_buf, scale_buf, p)

    # 4. Synchronize so all symmetric scales and fp8 buffers are fully written across the group
    fp8_hdl.barrier(channel=0)

    # 5. Pull from peers and execute inline unpacking
    full = torch.empty(world_size * p, dtype=torch.bfloat16, device=flat_param_shard.device)
    ext.launch_gather(fp8_ptrs, scale_ptrs, full, p, world_size)

    # 6. Safety barrier ensuring no rank will overwrite its buffer in immediate consecutive loops
    fp8_hdl.barrier(channel=0)

    return full, updated_hist

__all__ = ["solution"]