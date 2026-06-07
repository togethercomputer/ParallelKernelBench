import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

struct Ptrs {
    const void* p[16];
};

template <typename T>
__global__ void pull_gather_kernel(
    Ptrs ptrs, 
    uint8_t* __restrict__ out, 
    int64_t src_offset_elements, 
    int64_t dst_offset_elements, 
    int64_t chunk_elements, 
    int64_t total_elements_per_rank
) {
    int r = blockIdx.y;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_elements) {
        const T* src = reinterpret_cast<const T*>((const uint8_t*)ptrs.p[r] + src_offset_elements * sizeof(T));
        T* dst = reinterpret_cast<T*>(out + r * total_elements_per_rank * sizeof(T) + dst_offset_elements * sizeof(T));
        dst[idx] = src[idx];
    }
}

void pull_gather(
    std::vector<int64_t> ptrs_int,
    torch::Tensor out,
    int64_t src_offset_bytes,
    int64_t dst_offset_bytes,
    int64_t chunk_bytes,
    int64_t total_bytes_per_rank,
    int world_size
) {
    TORCH_CHECK(world_size <= 16, "Max 16 ranks supported in this custom kernel");
    
    Ptrs ptrs;
    for (int i = 0; i < world_size; ++i) {
        ptrs.p[i] = (const void*)ptrs_int[i];
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uint8_t* out_ptr = (uint8_t*)out.data_ptr();
    
    int threads = 256;
    
    // Dynamically choose optimal vectorization while guaranteeing memory alignment
    if (chunk_bytes % 16 == 0 && src_offset_bytes % 16 == 0 && dst_offset_bytes % 16 == 0 && total_bytes_per_rank % 16 == 0) {
        int64_t chunk_elements = chunk_bytes / 16;
        int blocks = (chunk_elements + threads - 1) / threads;
        dim3 grid(blocks, world_size);
        pull_gather_kernel<float4><<<grid, threads, 0, stream>>>(ptrs, out_ptr, src_offset_bytes/16, dst_offset_bytes/16, chunk_elements, total_bytes_per_rank/16);
    } else if (chunk_bytes % 8 == 0 && src_offset_bytes % 8 == 0 && dst_offset_bytes % 8 == 0 && total_bytes_per_rank % 8 == 0) {
        int64_t chunk_elements = chunk_bytes / 8;
        int blocks = (chunk_elements + threads - 1) / threads;
        dim3 grid(blocks, world_size);
        pull_gather_kernel<float2><<<grid, threads, 0, stream>>>(ptrs, out_ptr, src_offset_bytes/8, dst_offset_bytes/8, chunk_elements, total_bytes_per_rank/8);
    } else if (chunk_bytes % 4 == 0 && src_offset_bytes % 4 == 0 && dst_offset_bytes % 4 == 0 && total_bytes_per_rank % 4 == 0) {
        int64_t chunk_elements = chunk_bytes / 4;
        int blocks = (chunk_elements + threads - 1) / threads;
        dim3 grid(blocks, world_size);
        pull_gather_kernel<float><<<grid, threads, 0, stream>>>(ptrs, out_ptr, src_offset_bytes/4, dst_offset_bytes/4, chunk_elements, total_bytes_per_rank/4);
    } else if (chunk_bytes % 2 == 0 && src_offset_bytes % 2 == 0 && dst_offset_bytes % 2 == 0 && total_bytes_per_rank % 2 == 0) {
        int64_t chunk_elements = chunk_bytes / 2;
        int blocks = (chunk_elements + threads - 1) / threads;
        dim3 grid(blocks, world_size);
        pull_gather_kernel<uint16_t><<<grid, threads, 0, stream>>>(ptrs, out_ptr, src_offset_bytes/2, dst_offset_bytes/2, chunk_elements, total_bytes_per_rank/2);
    } else {
        int64_t chunk_elements = chunk_bytes;
        int blocks = (chunk_elements + threads - 1) / threads;
        dim3 grid(blocks, world_size);
        pull_gather_kernel<uint8_t><<<grid, threads, 0, stream>>>(ptrs, out_ptr, src_offset_bytes, dst_offset_bytes, chunk_elements, total_bytes_per_rank);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pull_gather", &pull_gather, "Pull gather from multiple symm mem peer pointers");
}
'''

_ext = None
_symm_cache = None
_copy_stream = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("pull_gather_uva_ext", CUDA_SRC)
    return _ext

def _get_symm_state(numel: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["numel"] == numel and c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"]

    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache = {"numel": numel, "dtype": dtype, "device": device, "buf": buf, "hdl": hdl}
    return buf, hdl

def _get_copy_stream():
    global _copy_stream
    if _copy_stream is None:
        _copy_stream = torch.cuda.Stream()
    return _copy_stream

@torch.no_grad()
def solution(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda and tensor.is_contiguous(), "input tensor must be contiguous and on CUDA"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Pre-compile the custom CUDA extension sequentially to prevent JIT race conditions
    if rank == 0:
        _get_ext()
    dist.barrier()

    n_elements = tensor.numel()
    buf, hdl = _get_symm_state(n_elements, tensor.dtype, tensor.device)

    # Establish chunk boundaries for overlapping (cap at 4 chunks max to limit barrier channel rotation overhead)
    element_size = tensor.element_size()
    total_bytes = n_elements * element_size
    max_chunk_bytes = 4 * 1024 * 1024
    num_chunks = min(4, max(1, (total_bytes + max_chunk_bytes - 1) // max_chunk_bytes))

    align_elements = max(1, 16 // element_size)
    chunks = []
    
    for i in range(num_chunks):
        start = (n_elements * i) // num_chunks
        start = (start // align_elements) * align_elements
        
        end = (n_elements * (i + 1)) // num_chunks
        if i == num_chunks - 1:
            end = n_elements
        else:
            end = (end // align_elements) * align_elements
            
        if start < end:
            chunks.append((start, end))
            
    num_chunks = len(chunks)

    # Prepare destination buffer mapping
    if rank == dst:
        out = torch.empty((world_size, *tensor.shape), dtype=tensor.dtype, device=tensor.device)
    else:
        out = tensor

    copy_stream = _get_copy_stream()
    tensor_flat = tensor.view(-1)
    
    for i, (start, end) in enumerate(chunks):
        chunk_elements = end - start
        
        # Non-blocking copy onto parallel stream
        with torch.cuda.stream(copy_stream):
            buf[start:end].copy_(tensor_flat[start:end])
            
        # Ensure compute stream tracks dependencies dynamically (does not block CPU execution)
        torch.cuda.current_stream().wait_stream(copy_stream)
        
        # Synchronize ranks across chunk lifecycle
        hdl.barrier(channel=i)
        
        # Pull phase: Destination rank launches NVLink P2P custom pull kernel on main stream 
        # (This efficiently overlaps with the next iteration's local `copy_stream` copy)
        if rank == dst:
            src_offset_bytes = start * element_size
            dst_offset_bytes = start * element_size
            chunk_bytes = chunk_elements * element_size
            total_bytes_per_rank = n_elements * element_size
            
            _get_ext().pull_gather(
                hdl.buffer_ptrs,
                out,
                src_offset_bytes,
                dst_offset_bytes,
                chunk_bytes,
                total_bytes_per_rank,
                world_size
            )

    # Final semantic barrier guarantees the receiver completes pulling before any rank safely cycles
    hdl.barrier(channel=num_chunks)
    
    return out