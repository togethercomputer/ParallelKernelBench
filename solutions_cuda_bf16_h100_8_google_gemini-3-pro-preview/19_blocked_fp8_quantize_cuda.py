"""
Strategy:
1. **Fused Compute & Push**: We replace the separate Triton quantization and NCCL all-gather with a single custom CUDA kernel. Each GPU locally quantizes its BF16 data to FP8, computes the FP32 scales, and directly pushes the results to all peers' memory via NVSwitch and UVA pointers.
2. **Zero-Copy All-Gather**: Symmetric memory buffers are allocated for the *entire* global output tensors. Each rank pushes its computed chunks directly to its respective non-overlapping slice (`rank * local_numel`) in every peer's buffer. This completely eliminates write conflicts and the need for a secondary communication pass.
3. **Compute–Communication Overlap & Vectorization**: The kernel processes data in 512-element tiles via shared memory. Packed 128-bit (`uint4`) stores are issued over NVLink to maximize cross-device bandwidth. The hardware overlaps these stores with the arithmetic (abs max reduction, scaling) of subsequent loops.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>

__global__ void quantize_and_push_kernel(
    const __nv_bfloat16* __restrict__ x,
    const long long* __restrict__ y_ptrs,
    const long long* __restrict__ s_ptrs,
    int64_t numel,
    int64_t block_size,
    int64_t local_y_offset,
    int64_t local_s_offset,
    int world_size
) {
    int64_t chunk_idx = blockIdx.x;
    int64_t num_chunks = numel / block_size;
    int tile_size = 512;
    
    // Shared memory for quantizing a tile and block reductions
    __shared__ uint8_t shared_fp8[512];
    __shared__ float shared_max[32];
    
    for (int64_t c = chunk_idx; c < num_chunks; c += gridDim.x) {
        int64_t base_idx = c * block_size;
        
        // 1. Find max abs for the block
        float local_max = 0.0f;
        for (int64_t i = threadIdx.x; i < block_size; i += blockDim.x) {
            float val = __bfloat162float(x[base_idx + i]);
            local_max = fmaxf(local_max, fabsf(val));
        }
        
        // Warp reduce max
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        if (lane == 0) shared_max[wid] = local_max;
        __syncthreads();
        
        if (wid == 0) {
            float val = (lane < (blockDim.x + 31) / 32) ? shared_max[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (lane == 0) shared_max[0] = val;
        }
        __syncthreads();
        
        float max_val = shared_max[0];
        float scale = max_val / 448.0f;
        if (scale == 0.0f) scale = 1.0f; // Prevent division by zero
        float inv_scale = 1.0f / scale;
        
        // 2. Quantize and Push via UVA in tiles
        for (int64_t tile_start = 0; tile_start < block_size; tile_start += tile_size) {
            int64_t current_tile_size = min((int64_t)tile_size, block_size - tile_start);
            
            // Quantize tile into shared memory
            for (int64_t i = threadIdx.x; i < current_tile_size; i += blockDim.x) {
                float val = __bfloat162float(x[base_idx + tile_start + i]);
                float q_val = val * inv_scale;
                __nv_fp8_e4m3 fp8_val(q_val);
                shared_fp8[i] = *(reinterpret_cast<uint8_t*>(&fp8_val));
            }
            __syncthreads();
            
            // Check if 16-byte aligned for uint4 vectorized stores
            bool can_vectorize = ((local_y_offset + base_idx + tile_start) % 16 == 0);
            
            if (can_vectorize) {
                int num_uint4 = current_tile_size / 16;
                for (int i = threadIdx.x; i < num_uint4; i += blockDim.x) {
                    uint4 packed = reinterpret_cast<uint4*>(shared_fp8)[i];
                    int64_t global_offset = local_y_offset + base_idx + tile_start + i * 16;
                    
                    #pragma unroll
                    for (int r = 0; r < world_size; ++r) {
                        uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y_ptrs[r]);
                        reinterpret_cast<uint4*>(y_ptr + global_offset)[0] = packed;
                    }
                }
                
                // Handle remainder if tile size is not a multiple of 16
                int remainder_start = num_uint4 * 16;
                if (current_tile_size > remainder_start) {
                    for (int i = remainder_start + threadIdx.x; i < current_tile_size; i += blockDim.x) {
                        uint8_t val = shared_fp8[i];
                        int64_t global_offset = local_y_offset + base_idx + tile_start + i;
                        
                        #pragma unroll
                        for (int r = 0; r < world_size; ++r) {
                            uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y_ptrs[r]);
                            y_ptr[global_offset] = val;
                        }
                    }
                }
            } else {
                // Scalar fallback if unaligned
                for (int64_t i = threadIdx.x; i < current_tile_size; i += blockDim.x) {
                    uint8_t val = shared_fp8[i];
                    int64_t global_offset = local_y_offset + base_idx + tile_start + i;
                    
                    #pragma unroll
                    for (int r = 0; r < world_size; ++r) {
                        uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y_ptrs[r]);
                        y_ptr[global_offset] = val;
                    }
                }
            }
            __syncthreads();
        }
        
        // 3. Write scale to symmetric memory
        if (threadIdx.x == 0) {
            int64_t global_s_idx = local_s_offset + c;
            #pragma unroll
            for (int r = 0; r < world_size; ++r) {
                float* s_ptr = reinterpret_cast<float*>(s_ptrs[r]);
                s_ptr[global_s_idx] = scale;
            }
        }
    }
}

void launch_quantize_and_push(
    torch::Tensor x,
    torch::Tensor y_ptrs_tensor,
    torch::Tensor s_ptrs_tensor,
    int64_t block_size,
    int64_t local_y_offset,
    int64_t local_s_offset,
    int world_size
) {
    int64_t numel = x.numel();
    int threads = 256;
    // Launch enough blocks for full SM occupancy
    int blocks = std::min((int)(numel / block_size), 1024);
    if (blocks == 0) blocks = 1;
    
    const __nv_bfloat16* d_x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    const long long* d_y_ptrs = reinterpret_cast<const long long*>(y_ptrs_tensor.data_ptr<int64_t>());
    const long long* d_s_ptrs = reinterpret_cast<const long long*>(s_ptrs_tensor.data_ptr<int64_t>());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    quantize_and_push_kernel<<<blocks, threads, 0, stream>>>(
        d_x, d_y_ptrs, d_s_ptrs, numel, block_size, local_y_offset, local_s_offset, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_quantize_and_push", &launch_quantize_and_push, "Fused block quantize to FP8 and NVLink push to peers");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_fp8_quant_push", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(shape_y, shape_s, device):
    key = (shape_y, shape_s, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    # Allocate full global tensors symmetrically
    buf_y = symm_mem.empty(shape_y, dtype=torch.uint8, device=device)
    hdl_y = symm_mem.rendezvous(buf_y, dist.group.WORLD)
    
    buf_s = symm_mem.empty(shape_s, dtype=torch.float32, device=device)
    hdl_s = symm_mem.rendezvous(buf_s, dist.group.WORLD)
    
    # Track UVA pointers into tensors to feed the kernel array lookup
    y_ptrs_tensor = torch.tensor(hdl_y.buffer_ptrs, dtype=torch.int64, device=device)
    s_ptrs_tensor = torch.tensor(hdl_s.buffer_ptrs, dtype=torch.int64, device=device)
    
    res = (buf_y, y_ptrs_tensor, buf_s, s_ptrs_tensor)
    _symm_cache[key] = res
    return res

@torch.no_grad()
def solution(local_tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Multi-GPU Block FP8 Quantization and All-Gather.
    Executes a custom CUDA kernel that locally calculates block scales, quantizes to FP8,
    and leverages UVA symmetric memory to push data over NVLink to all peers in one pass.
    """
    assert local_tensor.is_contiguous(), "Input tensor must be contiguous"
    assert local_tensor.size(-1) % block_size == 0, "Last dimension must be divisible by block_size"
    
    if local_tensor.dtype != torch.bfloat16:
        local_tensor = local_tensor.to(torch.bfloat16)
        
    device = local_tensor.device
    
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
        
    # Determine the post-concatenation global shapes 
    local_shape = list(local_tensor.shape)
    global_shape_y = [world_size * local_shape[0]] + local_shape[1:] if len(local_shape) > 0 else [world_size]
    
    local_shape_s = list(local_shape)
    local_shape_s[-1] = local_shape_s[-1] // block_size
    global_shape_s = [world_size * local_shape_s[0]] + local_shape_s[1:] if len(local_shape_s) > 0 else [world_size]
    
    if world_size > 1:
        buf_y, y_ptrs_tensor, buf_s, s_ptrs_tensor = _get_symm_state(
            tuple(global_shape_y), tuple(global_shape_s), device
        )
    else:
        buf_y = torch.empty(tuple(global_shape_y), dtype=torch.uint8, device=device)
        buf_s = torch.empty(tuple(global_shape_s), dtype=torch.float32, device=device)
        y_ptrs_tensor = torch.tensor([buf_y.data_ptr()], dtype=torch.int64, device=device)
        s_ptrs_tensor = torch.tensor([buf_s.data_ptr()], dtype=torch.int64, device=device)
        
    local_numel = local_tensor.numel()
    local_s_numel = local_numel // block_size
    
    # Since concatenation happens along dim=0, global flattened offsets scale perfectly with rank
    local_y_offset = rank * local_numel
    local_s_offset = rank * local_s_numel
    
    # Launch purely fused compute and NVLink scatter pass
    _get_ext().launch_quantize_and_push(
        local_tensor,
        y_ptrs_tensor,
        s_ptrs_tensor,
        block_size,
        local_y_offset,
        local_s_offset,
        world_size
    )
    
    if world_size > 1:
        # Guarantee writes emitted from the stream are physically visible to all peers before read
        torch.cuda.current_stream().synchronize()
        dist.barrier()
        
    y_global = buf_y.view(torch.float8_e4m3fn)
    return y_global, buf_s