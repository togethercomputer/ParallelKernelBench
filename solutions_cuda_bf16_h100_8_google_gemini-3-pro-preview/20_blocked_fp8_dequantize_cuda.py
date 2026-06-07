import math
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension
import triton
import triton.language as tl

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>

// -------------------------------------------------------------------------
// 1-byte element kernel (FP8, int8, uint8)
// -------------------------------------------------------------------------
template <typename T>
__global__ void dequant_alltoall_kernel_1byte(
    const uint8_t* __restrict__ y_ptr, 
    const float* __restrict__ s_ptr,
    const int64_t* __restrict__ peer_ptrs,
    int64_t chunk_numel,
    int blocks_per_chunk,
    int block_size,
    int rank,
    int world_size
) {
    int64_t block_idx = blockIdx.x; 
    int dest_rank = block_idx / blocks_per_chunk;
    int chunk_block_idx = block_idx % blocks_per_chunk;

    float* dest_ptr = reinterpret_cast<float*>(peer_ptrs[dest_rank]);
    
    // We write to the rank-th chunk of the destination buffer
    int64_t dest_offset = (int64_t)rank * chunk_numel + (int64_t)chunk_block_idx * block_size;
    int64_t src_offset = block_idx * block_size;
    
    float scale = s_ptr[block_idx];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (block_size % 16 == 0) {
        int num_vec = block_size / 16;
        for (int i = tid; i < num_vec; i += stride) {
            uint4 packed_y = *(reinterpret_cast<const uint4*>(y_ptr + src_offset + i * 16));
            
            uint8_t bytes[16];
            *(reinterpret_cast<uint4*>(bytes)) = packed_y;
            
            float out[16];
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                T val;
                memcpy(&val, &bytes[j], 1);
                out[j] = static_cast<float>(val) * scale;
            }
            
            float4* dest_float4 = reinterpret_cast<float4*>(dest_ptr + dest_offset + i * 16);
            dest_float4[0] = make_float4(out[0], out[1], out[2], out[3]);
            dest_float4[1] = make_float4(out[4], out[5], out[6], out[7]);
            dest_float4[2] = make_float4(out[8], out[9], out[10], out[11]);
            dest_float4[3] = make_float4(out[12], out[13], out[14], out[15]);
        }
    } else {
        for (int i = tid; i < block_size; i += stride) {
            T val;
            memcpy(&val, y_ptr + src_offset + i, 1);
            dest_ptr[dest_offset + i] = static_cast<float>(val) * scale;
        }
    }
}

// -------------------------------------------------------------------------
// 2-byte element kernel (BFloat16, Float16 fallbacks)
// -------------------------------------------------------------------------
template <typename T>
__global__ void dequant_alltoall_kernel_2byte(
    const uint16_t* __restrict__ y_ptr, 
    const float* __restrict__ s_ptr,
    const int64_t* __restrict__ peer_ptrs,
    int64_t chunk_numel,
    int blocks_per_chunk,
    int block_size,
    int rank,
    int world_size
) {
    int64_t block_idx = blockIdx.x; 
    int dest_rank = block_idx / blocks_per_chunk;
    int chunk_block_idx = block_idx % blocks_per_chunk;

    float* dest_ptr = reinterpret_cast<float*>(peer_ptrs[dest_rank]);
    int64_t dest_offset = (int64_t)rank * chunk_numel + (int64_t)chunk_block_idx * block_size;
    int64_t src_offset = block_idx * block_size;
    
    float scale = s_ptr[block_idx];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (block_size % 8 == 0) {
        int num_vec = block_size / 8;
        for (int i = tid; i < num_vec; i += stride) {
            uint4 packed_y = *(reinterpret_cast<const uint4*>(y_ptr + src_offset + i * 8));
            
            uint16_t bytes[8];
            *(reinterpret_cast<uint4*>(bytes)) = packed_y;
            
            float out[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                T val;
                memcpy(&val, &bytes[j], 2);
                out[j] = static_cast<float>(val) * scale;
            }
            
            float4* dest_float4 = reinterpret_cast<float4*>(dest_ptr + dest_offset + i * 8);
            dest_float4[0] = make_float4(out[0], out[1], out[2], out[3]);
            dest_float4[1] = make_float4(out[4], out[5], out[6], out[7]);
        }
    } else {
        for (int i = tid; i < block_size; i += stride) {
            T val;
            memcpy(&val, y_ptr + src_offset + i, 2);
            dest_ptr[dest_offset + i] = static_cast<float>(val) * scale;
        }
    }
}

// -------------------------------------------------------------------------
// 4-byte element kernel (Float32 fallback)
// -------------------------------------------------------------------------
template <typename T>
__global__ void dequant_alltoall_kernel_4byte(
    const float* __restrict__ y_ptr, 
    const float* __restrict__ s_ptr,
    const int64_t* __restrict__ peer_ptrs,
    int64_t chunk_numel,
    int blocks_per_chunk,
    int block_size,
    int rank,
    int world_size
) {
    int64_t block_idx = blockIdx.x; 
    int dest_rank = block_idx / blocks_per_chunk;
    int chunk_block_idx = block_idx % blocks_per_chunk;

    float* dest_ptr = reinterpret_cast<float*>(peer_ptrs[dest_rank]);
    int64_t dest_offset = (int64_t)rank * chunk_numel + (int64_t)chunk_block_idx * block_size;
    int64_t src_offset = block_idx * block_size;
    
    float scale = s_ptr[block_idx];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (block_size % 4 == 0) {
        int num_vec = block_size / 4;
        for (int i = tid; i < num_vec; i += stride) {
            float4 packed_y = *(reinterpret_cast<const float4*>(y_ptr + src_offset + i * 4));
            
            float4 out;
            out.x = packed_y.x * scale;
            out.y = packed_y.y * scale;
            out.z = packed_y.z * scale;
            out.w = packed_y.w * scale;
            
            float4* dest_float4 = reinterpret_cast<float4*>(dest_ptr + dest_offset + i * 4);
            dest_float4[0] = out;
        }
    } else {
        for (int i = tid; i < block_size; i += stride) {
            dest_ptr[dest_offset + i] = y_ptr[src_offset + i] * scale;
        }
    }
}

// -------------------------------------------------------------------------
// Launcher Dispatch
// -------------------------------------------------------------------------
void launch_dequant_alltoall(
    torch::Tensor y,
    torch::Tensor s,
    torch::Tensor peer_ptrs_tensor,
    int64_t chunk_numel,
    int blocks_per_chunk,
    int block_size,
    int rank,
    int world_size
) {
    auto s_ptr = s.data_ptr<float>();
    auto peer_ptrs = peer_ptrs_tensor.data_ptr<int64_t>();
    
    int num_blocks = world_size * blocks_per_chunk;
    
    int threads = 128;
    if (block_size % 16 == 0) {
        int num_vec = block_size / 16;
        if (num_vec <= 32) threads = 32;
        else if (num_vec <= 64) threads = 64;
        else if (num_vec <= 128) threads = 128;
        else threads = 256;
    } else {
        if (block_size <= 32) threads = 32;
        else if (block_size <= 64) threads = 64;
        else if (block_size <= 128) threads = 128;
        else threads = 256;
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto scalar_type = y.scalar_type();
    
    // Dispatch based on scalar type, natively supporting FP8
    if (scalar_type == torch::kFloat8_e4m3fn) {
        auto y_ptr = reinterpret_cast<const uint8_t*>(y.data_ptr());
        dequant_alltoall_kernel_1byte<c10::Float8_e4m3fn><<<num_blocks, threads, 0, stream>>>(
            y_ptr, s_ptr, peer_ptrs, chunk_numel, blocks_per_chunk, block_size, rank, world_size);
    } else if (scalar_type == torch::kFloat8_e5m2) {
        auto y_ptr = reinterpret_cast<const uint8_t*>(y.data_ptr());
        dequant_alltoall_kernel_1byte<c10::Float8_e5m2><<<num_blocks, threads, 0, stream>>>(
            y_ptr, s_ptr, peer_ptrs, chunk_numel, blocks_per_chunk, block_size, rank, world_size);
    } else if (scalar_type == torch::kInt8) {
        auto y_ptr = reinterpret_cast<const uint8_t*>(y.data_ptr());
        dequant_alltoall_kernel_1byte<int8_t><<<num_blocks, threads, 0, stream>>>(
            y_ptr, s_ptr, peer_ptrs, chunk_numel, blocks_per_chunk, block_size, rank, world_size);
    } else if (scalar_type == torch::kUInt8) {
        auto y_ptr = reinterpret_cast<const uint8_t*>(y.data_ptr());
        dequant_alltoall_kernel_1byte<uint8_t><<<num_blocks, threads, 0, stream>>>(
            y_ptr, s_ptr, peer_ptrs, chunk_numel, blocks_per_chunk, block_size, rank, world_size);
    } else if (scalar_type == torch::kBFloat16) {
        auto y_ptr = reinterpret_cast<const uint16_t*>(y.data_ptr());
        dequant_alltoall_kernel_2byte<at::BFloat16><<<num_blocks, threads, 0, stream>>>(
            y_ptr, s_ptr, peer_ptrs, chunk_numel, blocks_per_chunk, block_size, rank, world_size);
    } else if (scalar_type == torch::kFloat16) {
        auto y_ptr = reinterpret_cast<const uint16_t*>(y.data_ptr());
        dequant_alltoall_kernel_2byte<at::Half><<<num_blocks, threads, 0, stream>>>(
            y_ptr, s_ptr, peer_ptrs, chunk_numel, blocks_per_chunk, block_size, rank, world_size);
    } else if (scalar_type == torch::kFloat32) {
        auto y_ptr = reinterpret_cast<const float*>(y.data_ptr());
        dequant_alltoall_kernel_4byte<float><<<num_blocks, threads, 0, stream>>>(
            y_ptr, s_ptr, peer_ptrs, chunk_numel, blocks_per_chunk, block_size, rank, world_size);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for FP8 dequant_alltoall.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_dequant_alltoall", &launch_dequant_alltoall, "Fused FP8 dequantize and alltoall via symmetric memory");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_dequant_alltoall_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(shape, dtype, device):
    global _symm_cache
    key = (tuple(shape), dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]

    n = math.prod(shape)
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]

@torch.no_grad()
def solution(
    local_y: torch.Tensor,
    local_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    if local_y.numel() == 0:
        return torch.empty_like(local_y, dtype=torch.float32)

    assert dist.is_initialized(), "torch.distributed must be initialized"
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Determine chunking invariants mapped directly into the CUDA dispatch logic
    chunk_numel = local_y.numel() // world_size
    blocks_per_chunk = chunk_numel // block_size

    # Isolate compilation serialization
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    # Set up symmetric memory arrays returning expected Float32 dtype
    output_shape = local_y.shape
    out_dtype = torch.float32
    buf, hdl, ptrs_tensor = _get_symm_state(output_shape, out_dtype, local_y.device)

    # Secure the symmetric buffer is safe to write remotely
    hdl.barrier(channel=0)

    # Launch optimized UVA-mediated direct writes 
    ext.launch_dequant_alltoall(
        local_y, local_s, ptrs_tensor,
        chunk_numel, blocks_per_chunk, block_size,
        rank, world_size
    )

    # Assure completion of all peer-issued NVLink transfers mapping into rank local buf memory
    hdl.barrier(channel=0)

    # Return a new tensor ensuring isolation from the subsequent cache lifecycle updates
    out = buf.view(output_shape).clone()
    return out