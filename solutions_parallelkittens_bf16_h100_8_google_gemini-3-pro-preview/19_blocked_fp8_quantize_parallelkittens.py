"""
Strategy:
1. Fuse the local block FP8 quantization with the communication step.
2. Utilize ThunderKittens' `multimem::st` (NVSwitch multicast) to perform an O(1) broadcast of the locally 
   quantized blocks and scales directly into the global all-gathered tensors of all peers.
3. By treating the all-gathered target buffer as a symmetric parallel tensor and computing appropriate 
   rank-based offsets, each GPU natively "all-gathers" simply by storing its own slice to the multicast 
   address, completely hiding communication behind the quantization math.
4. Use `__shfl_down_sync` warp reductions to quickly compute block scales and `__nv_fp8_e4m3` intrinsics 
   for single-pass natively saturated conversion.
"""

import os
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from typing import Tuple

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source for fused TK quantization and multicast broadcast
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_fp8.h>
#include <torch/extension.h>

using namespace kittens;

namespace quantize {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    const __nv_bfloat16* input;
    
    using layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;
    layout y_global;
    layout s_global;
    
    int64_t N;
    int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((N + config::NUM_WARPS - 1) / config::NUM_WARPS);
    }
};

__device__ inline void kernel(const globals &G) {
    int64_t block_idx = (int64_t)blockIdx.x * config::NUM_WARPS + threadIdx.x / WARP_THREADS;
    if (block_idx >= G.N) return;

    int warp_tid = threadIdx.x % WARP_THREADS;
    int64_t local_offset = block_idx * 128 + warp_tid * 4;
    
    const uint16_t* p_u16 = reinterpret_cast<const uint16_t*>(&G.input[local_offset]);
    float x[4];
    
    #pragma unroll
    for(int i=0; i<4; ++i) {
        uint32_t f_u32 = ((uint32_t)p_u16[i]) << 16;
        x[i] = *reinterpret_cast<float*>(&f_u32);
    }
    
    float local_max = 0.0f;
    #pragma unroll
    for(int i=0; i<4; ++i) {
        local_max = fmaxf(local_max, fabsf(x[i]));
    }
    
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
    }
    
    float block_max = __shfl_sync(mask, local_max, 0);
    float s = block_max / 448.0f;
    float s_safe = (s == 0.0f) ? 1.0f : s;
    
    if (warp_tid == 0) {
        int64_t global_s_idx = (int64_t)G.dev_idx * G.N + block_idx;
        bf16_2 s_val = *reinterpret_cast<bf16_2*>(&s_safe);
        // Multiply by 2 because mc_ptr points to bf16 (2 bytes) and we write 4 bytes
        kittens::multimem<bf16_2>::st(reinterpret_cast<bf16_2*>(&G.s_global.mc_ptr[global_s_idx * 2]), s_val);
    }
    
    uint32_t y_u32 = 0;
    uint8_t* p_y = reinterpret_cast<uint8_t*>(&y_u32);
    #pragma unroll
    for(int i=0; i<4; ++i) {
        float x_scaled = x[i] / s_safe;
        __nv_fp8_e4m3 y_fp8(x_scaled);
        p_y[i] = *reinterpret_cast<uint8_t*>(&y_fp8);
    }
    
    int64_t global_y_u32_idx = ((int64_t)G.dev_idx * G.N * 128) / 4 + block_idx * 32 + warp_tid;
    bf16_2 y_val = *reinterpret_cast<bf16_2*>(&y_u32);
    // Multiply by 2 because mc_ptr points to bf16 (2 bytes) and we write 4 bytes
    kittens::multimem<bf16_2>::st(reinterpret_cast<bf16_2*>(&G.y_global.mc_ptr[global_y_u32_idx * 2]), y_val);
}

} // namespace quantize

namespace quantize_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace quantize_barrier

void entrypoint(
    torch::Tensor local_input,
    kittens::py::TKParallelTensor &y_tk,
    kittens::py::TKParallelTensor &s_tk,
    kittens::py::TKParallelTensor &barrier
) {
    TORCH_CHECK(local_input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(local_input.numel() % 128 == 0, "Elements must be multiple of 128");

    kittens::py::parallel_tensor_check(y_tk, s_tk, barrier);

    int64_t N = local_input.numel() / 128;

    quantize::globals G {
        .input = reinterpret_cast<const __nv_bfloat16*>(local_input.data_ptr<at::BFloat16>()),
        .y_global = kittens::py::parallel_tensor_to_pgl<typename quantize::globals::layout>(y_tk),
        .s_global = kittens::py::parallel_tensor_to_pgl<typename quantize::globals::layout>(s_tk),
        .N = N,
        .dev_idx = y_tk.local_rank_
    };

    quantize_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<quantize_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<quantize_barrier::config, quantize_barrier::globals, quantize_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<quantize::config, quantize::globals, quantize::kernel>(G);
    kittens::py::launch_kernel<quantize_barrier::config, quantize_barrier::globals, quantize_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_quantize", &entrypoint);
}
'''

TK_CUDA_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-DNDEBUG",
]

_ext = None
_ext_jit_ready = False

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_quantize_ext",
            CUDA_SRC,
            extra_cuda_cflags=TK_CUDA_FLAGS,
            extra_include_paths=[
                os.path.join(TK_ROOT, "include"),
                os.path.join(TK_ROOT, "prototype"),
            ],
            extra_ldflags=["-lcuda"],
        )
    return _ext

def _ensure_ext_jit():
    global _ext_jit_ready
    if _ext_jit_ready:
        return _get_ext()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext

# Fallback Triton kernel for single-GPU / missing distributed scenarios
@triton.jit
def block_fp8_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    s_safe = tl.where(s == 0.0, 1.0, s)
    y = (x / s_safe).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)

@torch.no_grad()
def solution(local_tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert local_tensor.is_contiguous(), "Input tensor must be contiguous"
    assert local_tensor.size(-1) % block_size == 0, "Last dimension must be divisible by block_size"
    assert block_size == 128, "This optimized kernel requires block_size=128"
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    if world_size == 1:
        # Fallback for local testing or single GPU runs
        y_local = torch.empty_like(local_tensor, dtype=torch.float8_e4m3fn)
        s_local = local_tensor.new_empty(
            *local_tensor.size()[:-1], local_tensor.size(-1) // block_size, dtype=torch.float32
        )
        grid = (triton.cdiv(local_tensor.numel(), block_size),)
        block_fp8_quant_kernel[grid](local_tensor, y_local, s_local, BLOCK_SIZE=block_size)
        return y_local, s_local

    assert world_size == 8, "ThunderKittens kernel built for 8 GPUs"
    
    ext = _ensure_ext_jit()
    
    original_shape = local_tensor.shape
    local_tensor_bf16 = local_tensor.to(torch.bfloat16)
    
    L = local_tensor_bf16.numel()
    N = L // 128
    
    # y_tk needs W * L bytes of output (float8)
    # We allocate it as bfloat16, meaning we allocate (W * L // 2) bf16 elements.
    y_tk = get_or_create_parallel_tensor(ext, (world_size * L // 2,), torch.bfloat16, multicast=True)
    
    # s_tk needs W * N floats of output (4 bytes each).
    # We allocate it as bfloat16, meaning we allocate (W * N * 2) bf16 elements.
    s_tk = get_or_create_parallel_tensor(ext, (world_size * N * 2,), torch.bfloat16, multicast=True)
    
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    
    # Run the fused quantization + multicast broadcast
    ext.tk_quantize(local_tensor_bf16, y_tk, s_tk, barrier_tk)
    
    # Shape of reference output equivalent to `torch.cat(gather_list, dim=0)`
    target_shape = list(original_shape)
    if len(target_shape) > 0:
        target_shape[0] *= world_size
    else:
        target_shape = [world_size]
        
    s_target_shape = list(original_shape)
    s_target_shape[-1] = s_target_shape[-1] // 128
    if len(s_target_shape) > 0:
        s_target_shape[0] *= world_size
    else:
        s_target_shape = [world_size]
        
    # Cast symmetrical buffers back into desired view shapes
    y_global = y_tk.data_.view(-1)[:world_size * L // 2].view(torch.uint8).view(torch.float8_e4m3fn)
    y_global = y_global.reshape(target_shape)
    
    s_global = s_tk.data_.view(-1)[:world_size * N * 2].view(torch.float32)
    s_global = s_global.reshape(s_target_shape)
    
    return y_global, s_global