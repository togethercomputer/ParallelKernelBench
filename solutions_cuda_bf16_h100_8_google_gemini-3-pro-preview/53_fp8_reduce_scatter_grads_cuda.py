"""
Strategy:
- **Device-Side Fusion:** Replaced the host-driven PyTorch simulation (FP32 conversion -> FP8 quantize -> BF16 dequantize -> NCCL reduce-scatter) with pure device-side kernels. The rolling history and dynamic scaling operate asynchronously on the GPU stream, entirely sidestepping host synchronization.
- **True FP8 Wire Protocol via Symmetric Memory:** Skipped standard NCCL completely. A custom fused `quantize_kernel` directly scales and converts BF16 gradients into FP8 E4M3, then pushes them to a `symm_mem` buffer alongside the local scaling factor. This inherently cuts the peer-to-peer NVLink communication payload in half (from BF16 to FP8).
- **Zero-Copy Dequantize & Reduce-Scatter:** A custom `reduce_scatter_kernel` directly taps into the FP8 symmetric buffers of all peers. It performs vectorized (4-element `uint32_t`) zero-copy reads, dequantizes on the fly using the gathered per-rank scales, accurately accumulates the global sum in float precision, divides by `world_size`, and stores the final averaged BF16 shard directly to the local output tensor.
- **Compute-Communication Overlap & Maximised Bandwidth:** The kernels utilise vectorized 64-bit/32-bit loads to push Hopper's memory bandwidth to the limit. We broadcast the scale factors via fast block shared memory. Inter-rank synchronisation strictly relies on lightweight device-stream barriers via `symm_mem`.
"""

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
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>

__global__ void quantize_kernel_vec4(
    const at::BFloat16* __restrict__ input,
    const float* __restrict__ scale,
    c10::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ symm_scale,
    int64_t n
) {
    float s = *scale;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *symm_scale = s;
    }
    float inv_s = 1.0f / s;
    int64_t n_vec4 = n / 4;
    
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n_vec4; idx += (int64_t)gridDim.x * blockDim.x) {
        uint64_t in4 = reinterpret_cast<const uint64_t*>(input)[idx];
        at::BFloat16 v0 = reinterpret_cast<const at::BFloat16*>(&in4)[0];
        at::BFloat16 v1 = reinterpret_cast<const at::BFloat16*>(&in4)[1];
        at::BFloat16 v2 = reinterpret_cast<const at::BFloat16*>(&in4)[2];
        at::BFloat16 v3 = reinterpret_cast<const at::BFloat16*>(&in4)[3];
        
        c10::Float8_e4m3fn q0 = static_cast<c10::Float8_e4m3fn>(static_cast<float>(v0) * inv_s);
        c10::Float8_e4m3fn q1 = static_cast<c10::Float8_e4m3fn>(static_cast<float>(v1) * inv_s);
        c10::Float8_e4m3fn q2 = static_cast<c10::Float8_e4m3fn>(static_cast<float>(v2) * inv_s);
        c10::Float8_e4m3fn q3 = static_cast<c10::Float8_e4m3fn>(static_cast<float>(v3) * inv_s);
        
        uint32_t out4;
        reinterpret_cast<c10::Float8_e4m3fn*>(&out4)[0] = q0;
        reinterpret_cast<c10::Float8_e4m3fn*>(&out4)[1] = q1;
        reinterpret_cast<c10::Float8_e4m3fn*>(&out4)[2] = q2;
        reinterpret_cast<c10::Float8_e4m3fn*>(&out4)[3] = q3;
        
        reinterpret_cast<uint32_t*>(output)[idx] = out4;
    }
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int64_t i = n_vec4 * 4; i < n; ++i) {
            output[i] = static_cast<c10::Float8_e4m3fn>(static_cast<float>(input[i]) * inv_s);
        }
    }
}

__global__ void reduce_scatter_kernel_vec4(
    const uint64_t* __restrict__ peer_fp8_ptrs,
    const uint64_t* __restrict__ peer_scale_ptrs,
    at::BFloat16* __restrict__ out_shard,
    int world_size,
    int rank,
    int64_t shard_elems
) {
    extern __shared__ float shared_scales[];
    if (threadIdx.x < world_size) {
        const float* scale_ptr = reinterpret_cast<const float*>(peer_scale_ptrs[threadIdx.x]);
        shared_scales[threadIdx.x] = *scale_ptr;
    }
    __syncthreads();

    int64_t vec_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t shard_vec4 = shard_elems / 4;
    
    for (; vec_idx < shard_vec4; vec_idx += (int64_t)gridDim.x * blockDim.x) {
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int64_t global_vec_idx = rank * shard_vec4 + vec_idx;
        
        #pragma unroll
        for (int p = 0; p < world_size; ++p) {
            float scale = shared_scales[p];
            const uint32_t* fp8_ptr = reinterpret_cast<const uint32_t*>(peer_fp8_ptrs[p]);
            uint32_t q4 = fp8_ptr[global_vec_idx];
            
            c10::Float8_e4m3fn q0 = reinterpret_cast<const c10::Float8_e4m3fn*>(&q4)[0];
            c10::Float8_e4m3fn q1 = reinterpret_cast<const c10::Float8_e4m3fn*>(&q4)[1];
            c10::Float8_e4m3fn q2 = reinterpret_cast<const c10::Float8_e4m3fn*>(&q4)[2];
            c10::Float8_e4m3fn q3 = reinterpret_cast<const c10::Float8_e4m3fn*>(&q4)[3];
            
            sum[0] += static_cast<float>(q0) * scale;
            sum[1] += static_cast<float>(q1) * scale;
            sum[2] += static_cast<float>(q2) * scale;
            sum[3] += static_cast<float>(q3) * scale;
        }
        
        float inv_ws = 1.0f / world_size;
        at::BFloat16 out0 = static_cast<at::BFloat16>(sum[0] * inv_ws);
        at::BFloat16 out1 = static_cast<at::BFloat16>(sum[1] * inv_ws);
        at::BFloat16 out2 = static_cast<at::BFloat16>(sum[2] * inv_ws);
        at::BFloat16 out3 = static_cast<at::BFloat16>(sum[3] * inv_ws);
        
        uint64_t out4;
        reinterpret_cast<at::BFloat16*>(&out4)[0] = out0;
        reinterpret_cast<at::BFloat16*>(&out4)[1] = out1;
        reinterpret_cast<at::BFloat16*>(&out4)[2] = out2;
        reinterpret_cast<at::BFloat16*>(&out4)[3] = out3;
        
        reinterpret_cast<uint64_t*>(out_shard)[vec_idx] = out4;
    }
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int64_t i = shard_vec4 * 4; i < shard_elems; ++i) {
            float sum = 0.0f;
            int64_t global_idx = rank * shard_elems + i;
            for (int p = 0; p < world_size; ++p) {
                float scale = shared_scales[p];
                const c10::Float8_e4m3fn* fp8_ptr = reinterpret_cast<const c10::Float8_e4m3fn*>(peer_fp8_ptrs[p]);
                c10::Float8_e4m3fn q = fp8_ptr[global_idx];
                sum += static_cast<float>(q) * scale;
            }
            out_shard[i] = static_cast<at::BFloat16>(sum / world_size);
        }
    }
}

void launch_quantize(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor output,
    torch::Tensor symm_scale
) {
    int64_t n = input.numel();
    int threads = 512;
    int blocks = std::max(1, std::min((int)((n/4 + threads - 1) / threads), 65535));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    quantize_kernel_vec4<<<blocks, threads, 0, stream>>>(
        input.data_ptr<at::BFloat16>(),
        scale.data_ptr<float>(),
        reinterpret_cast<c10::Float8_e4m3fn*>(output.data_ptr()),
        symm_scale.data_ptr<float>(),
        n
    );
}

void launch_reduce_scatter(
    torch::Tensor peer_fp8_ptrs_tensor,
    torch::Tensor peer_scale_ptrs_tensor,
    torch::Tensor out_shard,
    int world_size,
    int rank,
    int64_t shard_elems
) {
    int threads = 512;
    int blocks = std::max(1, std::min((int)((shard_elems/4 + threads - 1) / threads), 65535));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t shared_mem_size = world_size * sizeof(float);
    
    reduce_scatter_kernel_vec4<<<blocks, threads, shared_mem_size, stream>>>(
        reinterpret_cast<const uint64_t*>(peer_fp8_ptrs_tensor.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(peer_scale_ptrs_tensor.data_ptr<int64_t>()),
        out_shard.data_ptr<at::BFloat16>(),
        world_size,
        rank,
        shard_elems
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_quantize", &launch_quantize);
    m.def("launch_reduce_scatter", &launch_reduce_scatter);
}
'''

_ext = None
_symm_cache = {}

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fp8_rs_cuda_ext", CUDA_SRC)
    return _ext

def _get_symm_state(n: int, device: torch.device):
    global _symm_cache
    if n in _symm_cache:
        return _symm_cache[n]
    
    # FP8 E4M3 symmetric buffer
    fp8_buf = symm_mem.empty(n, dtype=torch.float8_e4m3fn, device=device)
    hdl_fp8 = symm_mem.rendezvous(fp8_buf, dist.group.WORLD)
    fp8_ptrs = torch.tensor(hdl_fp8.buffer_ptrs, dtype=torch.int64, device=device)
    
    # Scale per-rank symmetric buffer
    scale_buf = symm_mem.empty(1, dtype=torch.float32, device=device)
    hdl_scale = symm_mem.rendezvous(scale_buf, dist.group.WORLD)
    scale_ptrs = torch.tensor(hdl_scale.buffer_ptrs, dtype=torch.int64, device=device)
    
    state = (fp8_buf, hdl_fp8, fp8_ptrs, scale_buf, hdl_scale, scale_ptrs)
    _symm_cache[n] = state
    return state


@torch.no_grad()
def solution(flat_grads: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    flat_grads = flat_grads.contiguous()
    n = flat_grads.numel()
    shard_elems = n // world_size
    
    assert n % world_size == 0, f"flat_grads numel {n} must be divisible by world_size {world_size}"
    
    ext = _get_ext()
    fp8_buf, hdl_fp8, fp8_ptrs, scale_buf, hdl_scale, scale_ptrs = _get_symm_state(n, flat_grads.device)

    # Completely asynchronous calculations on the device; no host/GPU sync invoked.
    cur_abs_max = flat_grads.abs().max().to(torch.float32)
    updated_hist = torch.roll(amax_history, shifts=-1, dims=0)
    updated_hist[-1] = cur_abs_max
    
    scale = updated_hist.max().clamp(min=1e-12) / 448.0
    
    # Ensure any previous reduction's reads on the symmetric buffers have cleanly finished.
    hdl_fp8.barrier(channel=0)
    
    # 1. Fuse scale-out and convert BF16 -> FP8 + store scale to device symm memory
    ext.launch_quantize(flat_grads, scale, fp8_buf, scale_buf)
    
    # Wait for all peers to write their FP8 arrays and scalar multipliers to symmetric memory
    hdl_fp8.barrier(channel=0)
    
    out_shard = torch.empty(shard_elems, dtype=flat_grads.dtype, device=flat_grads.device)
    
    # 2. Fully fused peer-reads of FP8, dequantize using respective scales, average, and save to BF16 shard
    ext.launch_reduce_scatter(fp8_ptrs, scale_ptrs, out_shard, world_size, rank, shard_elems)
    
    return out_shard, updated_hist

__all__ = ["solution"]