"""
Strategy:
1. NVSwitch Multimem Reductions: We use Hopper's `multimem.ld_reduce` to perform the reduce-scatter. Instead of passing data back and forth over the network, each rank writes to its symmetric memory buffer and then individually reads its specific output chunk via a hardware multicast pointer. The NVSwitch performs the reduction transparently on read.
2. Zero Network Over-Fetch: The kernel maps threads only to the exact sub-slice of the global reduction destined for the calling rank. This ensures we naturally perform a reduce-scatter without manually coordinating a scatter-phase or slicing an all-reduced buffer.
3. Stream-Aware Overlap: PyTorch stream semantics are maintained using `hdl.barrier()`. Multi-channel barriers coordinate read/write-visibility over NVLink independently per stream, permitting the device computation and reduction to seamlessly interleave with minimal CPU overhead.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void reducescatter_multimem_bf16_kernel(
    uint64_t multicast_base,
    __nv_bfloat16* __restrict__ out,
    int64_t chunk_numel,
    int rank
) {
    int64_t total_vecs = chunk_numel / 8;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop processing 128 bits (8 x bf16) at a time per thread
    for (; idx < total_vecs; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t global_idx = (int64_t)rank * chunk_numel + idx * 8;
        uint64_t addr = multicast_base + global_idx * 2;
        
        uint32_t r0, r1, r2, r3;
        // Perform an in-switch load-and-reduce from all peers matching this multicast address
        asm volatile(
            "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "l"(addr)
            : "memory");
        
        uint4* out_ptr = reinterpret_cast<uint4*>(out);
        out_ptr[idx] = make_uint4(r0, r1, r2, r3);
    }
}

__global__ void reducescatter_fallback_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t chunk_numel,
    int rank,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < chunk_numel; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t global_idx = rank * chunk_numel + idx;
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[global_idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

__global__ void reducescatter_fallback_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int64_t chunk_numel,
    int rank,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < chunk_numel; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t global_idx = rank * chunk_numel + idx;
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const float* src = (const float*)ptrs[r];
            sum += src[global_idx];
        }
        out[idx] = sum;
    }
}

void launch_multimem_reducescatter_bf16(
    uint64_t multicast_ptr,
    torch::Tensor out,
    int64_t chunk_numel,
    int rank
) {
    int threads = 256;
    int64_t total_vecs = chunk_numel / 8;
    int blocks = (total_vecs + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reducescatter_multimem_bf16_kernel<<<blocks, threads, 0, stream>>>(
        multicast_ptr,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        chunk_numel,
        rank
    );
}

void launch_fallback_reducescatter(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t chunk_numel,
    int rank,
    int dtype_enum
) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    
    int threads = 256;
    int blocks = (chunk_numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (dtype_enum == 0) {
        reducescatter_fallback_bf16_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), chunk_numel, rank, world_size
        );
    } else {
        reducescatter_fallback_f32_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs, out.data_ptr<float>(), chunk_numel, rank, world_size
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_reducescatter_bf16", &launch_multimem_reducescatter_bf16, "Multimem reduce-scatter BF16");
    m.def("launch_fallback_reducescatter", &launch_fallback_reducescatter, "Fallback reduce-scatter via UVA");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("reducescatter_symm_ext", CUDA_SRC)
    return _ext

_resource_cache = {}

def _get_resources(shape, dtype, device):
    key = (shape, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]
    
    # Pre-allocate buffer via Symmetric Memory to keep working sets in UVA address spaces
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    res = (buf, hdl, ptrs_tensor)
    _resource_cache[key] = res
    return res

@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert tensor.shape[0] % world_size == 0, \
        f"First dimension ({tensor.shape[0]}) must be divisible by world_size ({world_size})"
        
    input_tensor = tensor.contiguous()
    N = input_tensor.numel()
    chunk_size = input_tensor.shape[0] // world_size
    
    # Fast paths for zero element arrays
    if N == 0:
        out_shape = (chunk_size,) + input_tensor.shape[1:]
        return torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    chunk_numel = N // world_size
    buf, hdl, ptrs_tensor = _get_resources(input_tensor.shape, input_tensor.dtype, input_tensor.device)
    
    # Coordinate symmetric streams: ensure no peers are overwriting the buffer locally or via UVA
    hdl.barrier(channel=0)
    buf.copy_(input_tensor)
    
    # Ensure all ranks have mapped their latest slices into symmetric memory
    hdl.barrier(channel=1)
    
    out_shape = (chunk_size,) + input_tensor.shape[1:]
    out = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    multicast_ptr = getattr(hdl, "multicast_ptr", 0)
    use_multimem = (
        input_tensor.dtype == torch.bfloat16 
        and chunk_numel % 8 == 0 
        and multicast_ptr != 0
    )
    
    # Perform chunked local reduction with Hopper Multimem or robust fallback loop natively on-device
    if use_multimem:
        _get_ext().launch_multimem_reducescatter_bf16(multicast_ptr, out, chunk_numel, rank)
    else:
        dtype_enum = 0 if input_tensor.dtype == torch.bfloat16 else 1
        _get_ext().launch_fallback_reducescatter(ptrs_tensor, out, chunk_numel, rank, dtype_enum)
        
    return out