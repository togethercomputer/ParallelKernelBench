"""
Strategy:
1. **Device-Side Communication (UVA)**: Replaces NCCL's `all_to_all_single` by pre-allocating a symmetric memory input buffer. Ranks copy local inputs to this buffer and execute a custom compiled CUDA kernel that directly pulls data from peers' memory spaces over NVLink.
2. **Compute-Communication Overlap**: The entire sequence (local copy -> sync -> UVA pull -> sync) is fully enqueued on the GPU stream asynchronously, returning control to the host immediately. The device kernel dynamically vectorizes memory accesses (up to 16 bytes per thread) based on runtime pointer alignment, fully saturating the high-bandwidth NVLink and seamlessly overlapping latency.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename T>
__device__ __forceinline__ void copy_chunk(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t size, int tid, int step) {
    size_t num_elems = size / sizeof(T);
    const T* src_t = reinterpret_cast<const T*>(src);
    T* dst_t = reinterpret_cast<T*>(dst);
    
    // Grid-stride loop for vectorized copy
    for (size_t i = tid; i < num_elems; i += step) {
        dst_t[i] = src_t[i];
    }
    
    // Handle remainder bytes
    if (tid < (size % sizeof(T))) {
        size_t offset = num_elems * sizeof(T) + tid;
        dst[offset] = src[offset];
    }
}

__global__ void all_to_all_pull_kernel(
    const uintptr_t* __restrict__ remote_ptrs,
    uint8_t* __restrict__ out_ptr,
    size_t chunk_size_bytes,
    int rank,
    int world_size
) {
    int s = blockIdx.x; // source rank
    int b = blockIdx.y; // block index for this chunk
    int num_blocks = gridDim.y;
    int tid = b * blockDim.x + threadIdx.x;
    int step = num_blocks * blockDim.x;

    // Source pointer logic: Pull from rank `s` symmetric buffer, specific offset for my `rank`
    const uint8_t* src_base = reinterpret_cast<const uint8_t*>(remote_ptrs[s]);
    const uint8_t* src_chunk = src_base + rank * chunk_size_bytes;
    
    // Destination pointer logic: Write to my local output buffer, specific offset for source rank `s`
    uint8_t* dst_chunk = out_ptr + s * chunk_size_bytes;

    // Check alignment dynamically to maximize load/store width
    bool align16 = (((uintptr_t)src_chunk % 16) == 0) && (((uintptr_t)dst_chunk % 16) == 0);
    bool align8  = (((uintptr_t)src_chunk % 8) == 0) && (((uintptr_t)dst_chunk % 8) == 0);
    bool align4  = (((uintptr_t)src_chunk % 4) == 0) && (((uintptr_t)dst_chunk % 4) == 0);
    bool align2  = (((uintptr_t)src_chunk % 2) == 0) && (((uintptr_t)dst_chunk % 2) == 0);

    if (align16) {
        copy_chunk<ulonglong2>(src_chunk, dst_chunk, chunk_size_bytes, tid, step);
    } else if (align8) {
        copy_chunk<uint64_t>(src_chunk, dst_chunk, chunk_size_bytes, tid, step);
    } else if (align4) {
        copy_chunk<uint32_t>(src_chunk, dst_chunk, chunk_size_bytes, tid, step);
    } else if (align2) {
        copy_chunk<uint16_t>(src_chunk, dst_chunk, chunk_size_bytes, tid, step);
    } else {
        copy_chunk<uint8_t>(src_chunk, dst_chunk, chunk_size_bytes, tid, step);
    }
}

void all_to_all_uva_pull(
    torch::Tensor remote_ptrs_tensor,
    torch::Tensor out_tensor,
    size_t chunk_size_bytes,
    int rank,
    int world_size
) {
    TORCH_CHECK(remote_ptrs_tensor.is_cuda(), "remote_ptrs must be CUDA");
    TORCH_CHECK(out_tensor.is_cuda(), "out_tensor must be CUDA");
    TORCH_CHECK(out_tensor.is_contiguous(), "out_tensor must be contiguous");

    const uintptr_t* remote_ptrs = reinterpret_cast<const uintptr_t*>(remote_ptrs_tensor.data_ptr<int64_t>());
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out_tensor.data_ptr());

    const int threads = 256;
    size_t elems = chunk_size_bytes / 16;
    if (elems == 0) elems = chunk_size_bytes;
    
    // Scale thread blocks dynamically with chunk size (max 32 blocks per chunk to prevent overscheduling)
    int blocks_per_chunk = 32;
    if (elems < 32 * threads) {
        blocks_per_chunk = std::max(1, (int)((elems + threads - 1) / threads));
    }

    dim3 grid(world_size, blocks_per_chunk);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    all_to_all_pull_kernel<<<grid, threads, 0, stream>>>(
        remote_ptrs, out_ptr, chunk_size_bytes, rank, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("all_to_all_uva_pull", &all_to_all_uva_pull, "UVA pull execution for all_to_all collective");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        # Standard pattern to prevent simultaneous host compilations causing filesystem races
        if dist.get_rank() == 0:
            _ext = compile_cuda_extension("all_to_all_uva_ext", CUDA_SRC)
        dist.barrier()
        if dist.get_rank() != 0:
            _ext = compile_cuda_extension("all_to_all_uva_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    """Caches and returns symmetric memory allocations and device pointers."""
    global _symm_cache
    key = (n, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    
    # Cache remote pointers directly into a CUDA tensor for the Triton/CUDA kernel
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_contiguous(), "Input tensor must be a contiguous CUDA tensor"
    assert tensor.is_cuda, "Input tensor must reside on a CUDA device"
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert tensor.shape[0] == world_size, \
        f"First dimension ({tensor.shape[0]}) must equal world_size ({world_size})"
    
    n = tensor.numel()
    if n == 0:
        return torch.empty_like(tensor)
        
    chunk_size = n // world_size
    chunk_size_bytes = chunk_size * tensor.element_size()
    
    # Ensure extension is ready
    _get_ext()
    
    # Retrieve symmetric memory structures based on current shape profile
    buf, hdl, ptrs_tensor = _get_symm_state(n, tensor.dtype, tensor.device)
    
    # 1. Pipeline local data copy into symmetric exchange buffer
    buf.copy_(tensor.reshape(-1))
    
    # 2. Asynchronous device barrier ensuring all chunks from peers are fully visible
    hdl.barrier(channel=0)
    
    out = torch.empty_like(tensor)
    
    # 3. Fast device-side UVA pull directly from peer memory spaces
    _get_ext().all_to_all_uva_pull(
        ptrs_tensor,
        out,
        chunk_size_bytes,
        rank,
        world_size
    )
    
    # 4. Trailing device barrier preventing succeeding invocations from mutating the cache too early
    hdl.barrier(channel=0)
    
    return out