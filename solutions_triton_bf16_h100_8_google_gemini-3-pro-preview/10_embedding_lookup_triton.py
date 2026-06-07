import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Optimized vectorized kernel for D_16 (number of uint16_t elements) divisible by 8 (16 bytes)
__global__ void uva_embedding_lookup_kernel_vec8(
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ shard_ptrs,
    uint16_t* __restrict__ output,
    int64_t N,
    int64_t D_16,
    int64_t shard_size,
    int world_size
) {
    int64_t vec_D = D_16 / 8;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N * vec_D) {
        int64_t idx = tid / vec_D;
        int64_t d_vec = tid % vec_D;
        
        int64_t global_index = indices[idx];
        if (global_index < 0) global_index = 0;
        
        int target_rank = global_index / shard_size;
        if (target_rank >= world_size) target_rank = world_size - 1;
        
        int64_t local_offset = global_index % shard_size;
        
        const uint4* target_shard = reinterpret_cast<const uint4*>(shard_ptrs[target_rank]);
        uint4* out_vec = reinterpret_cast<uint4*>(output);
        
        out_vec[idx * vec_D + d_vec] = target_shard[local_offset * vec_D + d_vec];
    }
}

// Scalar fallback kernel for irregular dimension sizes
__global__ void uva_embedding_lookup_kernel_scalar(
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ shard_ptrs,
    uint16_t* __restrict__ output,
    int64_t N,
    int64_t D_16,
    int64_t shard_size,
    int world_size
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N * D_16) {
        int64_t idx = tid / D_16;
        int64_t d = tid % D_16;
        
        int64_t global_index = indices[idx];
        if (global_index < 0) global_index = 0;
        
        int target_rank = global_index / shard_size;
        if (target_rank >= world_size) target_rank = world_size - 1;
        
        int64_t local_offset = global_index % shard_size;
        
        const uint16_t* target_shard = reinterpret_cast<const uint16_t*>(shard_ptrs[target_rank]);
        
        output[idx * D_16 + d] = target_shard[local_offset * D_16 + d];
    }
}

void uva_embedding_lookup(
    torch::Tensor indices,
    torch::Tensor shard_ptrs,
    torch::Tensor output,
    int64_t shard_size,
    int world_size
) {
    int64_t N = indices.numel();
    int64_t D = output.size(1);
    int64_t element_size = output.element_size();
    
    TORCH_CHECK((D * element_size) % 2 == 0, "Embedding byte size must be a multiple of 2");
    
    // Abstract the copy block width as multiples of 16-bits (uint16_t)
    // Allows transparent support for float32/float16/bfloat16.
    int64_t D_16 = (D * element_size) / 2;
    
    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (D_16 % 8 == 0) {
        int64_t total_vecs = N * (D_16 / 8);
        int blocks = (total_vecs + threads - 1) / threads;
        if (blocks > 0) {
            uva_embedding_lookup_kernel_vec8<<<blocks, threads, 0, stream>>>(
                indices.data_ptr<int64_t>(),
                shard_ptrs.data_ptr<int64_t>(),
                static_cast<uint16_t*>(output.data_ptr()),
                N, D_16, shard_size, world_size
            );
        }
    } else {
        int64_t total_elems = N * D_16;
        int blocks = (total_elems + threads - 1) / threads;
        if (blocks > 0) {
            uva_embedding_lookup_kernel_scalar<<<blocks, threads, 0, stream>>>(
                indices.data_ptr<int64_t>(),
                shard_ptrs.data_ptr<int64_t>(),
                static_cast<uint16_t*>(output.data_ptr()),
                N, D_16, shard_size, world_size
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_embedding_lookup", &uva_embedding_lookup, "UVA Direct Peer Embedding Lookup");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_embedding_ext", CUDA_SRC)
    return _ext

_symm_cache = None
def _get_symm_state(shard_size: int, D: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["shard_size"] == shard_size and c["D"] == D and c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"], c["ptrs"]

    # Use a 1D internal array mapping for symmetric memory compatibility
    buf_1d = symm_mem.empty(shard_size * D, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf_1d, dist.group.WORLD)
    buf = buf_1d.view(shard_size, D)
    
    # Store UVA pointers in a device tensor directly accessible by the CUDA kernel
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache = {
        "shard_size": shard_size,
        "D": D,
        "dtype": dtype,
        "device": device,
        "buf": buf,
        "hdl": hdl,
        "ptrs": ptrs
    }
    return buf, hdl, ptrs

@torch.no_grad()
def solution(
    indices: torch.Tensor,
    local_shard: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert indices.is_cuda and local_shard.is_cuda, "Inputs must be CUDA tensors"
    assert indices.dtype == torch.long, "indices must be torch.long"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_size = local_shard.shape[0]
    embed_dim = local_shard.shape[1]
    device = local_shard.device
    
    indices = indices.contiguous()
    if indices.device != device:
        indices = indices.to(device)
    
    # Compile kernel collectively
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    
    # Materialize cached symmetric environment and expose local weights
    buf, hdl, ptrs = _get_symm_state(shard_size, embed_dim, local_shard.dtype, device)
    buf.copy_(local_shard)
    
    # Barrier 0: Ensure all peers have flushed memory into their symmetric buffers
    hdl.barrier(channel=0)
    
    output_vectors = torch.empty((indices.numel(), embed_dim), dtype=local_shard.dtype, device=device)
    
    # Execute highly optimized custom CUDA loop
    if indices.numel() > 0:
        ext.uva_embedding_lookup(indices, ptrs, output_vectors, shard_size, world_size)
    
    # Barrier 1: Prevent successive call overlaps (safeguards `buf` overriding before peer readers conclude)
    hdl.barrier(channel=1)
    
    return output_vectors