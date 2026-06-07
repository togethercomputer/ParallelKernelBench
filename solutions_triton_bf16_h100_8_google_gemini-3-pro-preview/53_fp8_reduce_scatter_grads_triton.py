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
#include <cuda_fp8.h>
#include <cuda_bf16.h>

__global__ void quantize_fp8_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    int n,
    bool use_vec
) {
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;

    if (use_vec) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int vec_idx = idx * 16;
        if (vec_idx >= n) return;

        const uint4* in_v = reinterpret_cast<const uint4*>(input + vec_idx);
        uint4 in_val0 = in_v[0];
        uint4 in_val1 = in_v[1];
        const __nv_bfloat16* bf_vals0 = reinterpret_cast<const __nv_bfloat16*>(&in_val0);
        const __nv_bfloat16* bf_vals1 = reinterpret_cast<const __nv_bfloat16*>(&in_val1);

        __nv_fp8_e4m3 out_q[16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float f = __bfloat162float(bf_vals0[i]);
            out_q[i] = __nv_fp8_e4m3(f * inv_scale);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float f = __bfloat162float(bf_vals1[i]);
            out_q[8 + i] = __nv_fp8_e4m3(f * inv_scale);
        }

        *reinterpret_cast<uint4*>(output + vec_idx) = *reinterpret_cast<uint4*>(out_q);
    } else {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float f = __bfloat162float(input[idx]);
        __nv_fp8_e4m3 q(f * inv_scale);
        output[idx] = *(reinterpret_cast<uint8_t*>(&q));
    }
}

__global__ void reduce_scatter_fp8_kernel(
    const uint8_t* const* peer_ptrs,
    const float* const* peer_scale_ptrs,
    __nv_bfloat16* __restrict__ out,
    int shard_elems,
    int shard_idx,
    int world_size,
    bool use_vec
) {
    __shared__ float scales[16]; // safely handles world_sizes up to 16
    if (threadIdx.x < world_size) {
        scales[threadIdx.x] = *peer_scale_ptrs[threadIdx.x];
    }
    __syncthreads();

    if (use_vec) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int vec_idx = idx * 16;
        if (vec_idx >= shard_elems) return;

        int global_idx = shard_idx * shard_elems + vec_idx;
        float sums[16] = {0};

        for (int p = 0; p < world_size; ++p) {
            float scale = scales[p];
            uint4 v = *reinterpret_cast<const uint4*>(peer_ptrs[p] + global_idx);
            const __nv_fp8_e4m3* q = reinterpret_cast<const __nv_fp8_e4m3*>(&v);
            
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                // Mimic precise reference behavior: convert fp8 dequant to BF16 prior to summing 
                // simulating the exact precision truncation of sending over actual NCCL
                __nv_bfloat16 recon = __float2bfloat16((float)(q[i]) * scale);
                sums[i] += __bfloat162float(recon);
            }
        }
        
        float inv_ws = 1.0f / (float)world_size;
        __nv_bfloat162 bfloat_vals[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            // Apply BF16 cast after sum and prior to division simulating PyTorch div_() rules
            __nv_bfloat16 sum0_bf16 = __float2bfloat16(sums[2*i]);
            __nv_bfloat16 sum1_bf16 = __float2bfloat16(sums[2*i+1]);
            
            float val0 = __bfloat162float(sum0_bf16) * inv_ws;
            float val1 = __bfloat162float(sum1_bf16) * inv_ws;
            
            bfloat_vals[i].x = __float2bfloat16(val0);
            bfloat_vals[i].y = __float2bfloat16(val1);
        }
        uint4* out_v = reinterpret_cast<uint4*>(out + vec_idx);
        out_v[0] = reinterpret_cast<uint4*>(bfloat_vals)[0];
        out_v[1] = reinterpret_cast<uint4*>(bfloat_vals)[1];
    } else {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= shard_elems) return;

        int global_idx = shard_idx * shard_elems + idx;
        float sum = 0.0f;
        for (int p = 0; p < world_size; ++p) {
            float scale = scales[p];
            __nv_fp8_e4m3 q = reinterpret_cast<const __nv_fp8_e4m3*>(peer_ptrs[p])[global_idx];
            __nv_bfloat16 recon = __float2bfloat16((float)(q) * scale);
            sum += __bfloat162float(recon);
        }
        __nv_bfloat16 sum_bf16 = __float2bfloat16(sum);
        float final_val = __bfloat162float(sum_bf16) / (float)world_size;
        out[idx] = __float2bfloat16(final_val);
    }
}

void launch_quantize(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    int n,
    bool use_vec
) {
    int threads = 256;
    int blocks = use_vec ? (n / 16 + threads - 1) / threads : (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    quantize_fp8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint8_t*>(output.data_ptr<uint8_t>()),
        scale.data_ptr<float>(),
        n,
        use_vec
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce_scatter(
    torch::Tensor peer_ptrs_tensor, 
    torch::Tensor peer_scale_ptrs_tensor, 
    torch::Tensor out,
    int shard_elems,
    int shard_idx,
    int world_size,
    bool use_vec
) {
    int threads = 256;
    int blocks = use_vec ? (shard_elems / 16 + threads - 1) / threads : (shard_elems + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    reduce_scatter_fp8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t* const*>(peer_ptrs_tensor.data_ptr<uint64_t>()),
        reinterpret_cast<const float* const*>(peer_scale_ptrs_tensor.data_ptr<uint64_t>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        shard_elems,
        shard_idx,
        world_size,
        use_vec
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_fp8", &launch_quantize, "Quantize to FP8 directly within symmetrical buffers");
    m.def("reduce_scatter_fp8", &launch_reduce_scatter, "UVA UVA vectorized fusion RS over FP8");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "fp8_rs_ext",
            CUDA_SRC,
            extra_compile_args={'nvcc': ['-O3', '-std=c++17']}
        )
    return _ext

_symm_cache = None
def _get_symm_state(n: int, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n:
            return c["fp8_buf"], c["hdl_fp8"], c["scale_buf"], c["hdl_scale"], c["ptrs_tensor"], c["scale_ptrs_tensor"]

    fp8_buf = symm_mem.empty(n, device=device, dtype=torch.uint8)
    hdl_fp8 = symm_mem.rendezvous(fp8_buf, dist.group.WORLD)
    
    scale_buf = symm_mem.empty(1, device=device, dtype=torch.float32)
    hdl_scale = symm_mem.rendezvous(scale_buf, dist.group.WORLD)
    
    ptrs_tensor = torch.tensor(hdl_fp8.buffer_ptrs, dtype=torch.int64, device=device)
    scale_ptrs_tensor = torch.tensor(hdl_scale.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache = {
        "n": n,
        "fp8_buf": fp8_buf,
        "hdl_fp8": hdl_fp8,
        "scale_buf": scale_buf,
        "hdl_scale": hdl_scale,
        "ptrs_tensor": ptrs_tensor,
        "scale_ptrs_tensor": scale_ptrs_tensor
    }
    return fp8_buf, hdl_fp8, scale_buf, hdl_scale, ptrs_tensor, scale_ptrs_tensor


@torch.no_grad()
def solution(flat_grads: Tensor, amax_history: Tensor) -> tuple[Tensor, Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n = flat_grads.numel()
    shard_elems = n // world_size

    if rank == 0:
        _get_ext()
    dist.barrier()

    fp8_buf, hdl_fp8, scale_buf, hdl_scale, ptrs_tensor, scale_ptrs_tensor = _get_symm_state(n, flat_grads.device)

    # 1. Update lightweight historical parameters conventionally
    cur_abs_max = flat_grads.abs().max().to(torch.float32)
    updated_hist = torch.roll(amax_history, shifts=-1, dims=0)
    updated_hist[-1] = cur_abs_max.to(dtype=updated_hist.dtype)
    scale = updated_hist.max().clamp(min=1e-12).to(torch.float32) / _FP8_E4M3_MAX

    # 2. Expose scaling context via symmetric memory
    scale_buf.copy_(scale)

    # 3. Quantize locally right into accessible symmetric memory
    use_vec = (n % 16 == 0) and (shard_elems % 16 == 0)
    _get_ext().quantize_fp8(flat_grads, fp8_buf, scale_buf, n, use_vec)

    # 4. Strict Barrier preventing peers from UVA-fetching before the rank is ready
    hdl_fp8.barrier(channel=0)
    
    # 5. Overlapped load, sum, dequantize loop natively operating on UVA 
    out_shard = torch.empty(shard_elems, dtype=flat_grads.dtype, device=flat_grads.device)
    _get_ext().reduce_scatter_fp8(
        ptrs_tensor,
        scale_ptrs_tensor,
        out_shard,
        shard_elems,
        rank,
        world_size,
        use_vec
    )
    
    # 6. Barrier blocking sequential calls from overwriting current step iterations
    hdl_fp8.barrier(channel=1)
    
    return out_shard, updated_hist

__all__ = ["solution"]