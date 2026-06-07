import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

struct PtrArray {
    const void* ptrs[8]; // Assumes max world size of 8 for a single node
};

// 16-byte vectorized pull to saturate NVLink
__global__ void pull_gather_vec_kernel(
    PtrArray peer_ptrs,
    uint8_t* __restrict__ global_out,
    int64_t chunk_bytes,
    int64_t local_bytes,
    int world_size
) {
    int64_t chunk_vecs = chunk_bytes / sizeof(uint4);
    int64_t local_vecs = local_bytes / sizeof(uint4);
    int64_t total_vecs = chunk_vecs * world_size;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < total_vecs) {
        int rank = tid / chunk_vecs;
        int64_t vec_offset = tid % chunk_vecs;
        
        const uint4* peer_vec_ptr = reinterpret_cast<const uint4*>(peer_ptrs.ptrs[rank]);
        uint4* out_vec_ptr = reinterpret_cast<uint4*>(global_out);
        
        out_vec_ptr[rank * local_vecs + vec_offset] = peer_vec_ptr[vec_offset];
    }
}

// Remainder handling
__global__ void pull_gather_rem_kernel(
    PtrArray peer_ptrs,
    uint8_t* __restrict__ global_out,
    int64_t chunk_bytes,
    int64_t local_bytes,
    int world_size
) {
    int64_t vec_bytes = (chunk_bytes / sizeof(uint4)) * sizeof(uint4);
    int64_t rem_bytes = chunk_bytes - vec_bytes;
    int64_t total_rem = rem_bytes * world_size;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < total_rem) {
        int rank = tid / rem_bytes;
        int64_t rem_offset = tid % rem_bytes;
        
        const uint8_t* peer_byte_ptr = static_cast<const uint8_t*>(peer_ptrs.ptrs[rank]);
        global_out[rank * local_bytes + vec_bytes + rem_offset] = peer_byte_ptr[vec_bytes + rem_offset];
    }
}

void pull_gather(
    std::vector<int64_t> peer_ptrs_int,
    torch::Tensor global_out,
    int64_t chunk_bytes,
    int64_t local_bytes,
    int64_t global_offset_bytes,
    int world_size
) {
    PtrArray peer_ptrs;
    for(int i = 0; i < world_size; ++i) {
        peer_ptrs.ptrs[i] = reinterpret_cast<const void*>(peer_ptrs_int[i]);
    }

    const int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uint8_t* out_ptr = reinterpret_cast<uint8_t*>(global_out.data_ptr()) + global_offset_bytes;

    int64_t chunk_vecs = chunk_bytes / sizeof(uint4);
    if (chunk_vecs > 0) {
        int64_t total_vecs = chunk_vecs * world_size;
        int blocks = (total_vecs + threads - 1) / threads;
        pull_gather_vec_kernel<<<blocks, threads, 0, stream>>>(
            peer_ptrs, out_ptr, chunk_bytes, local_bytes, world_size
        );
    }

    int64_t rem_bytes = chunk_bytes % sizeof(uint4);
    if (rem_bytes > 0) {
        int64_t total_rem = rem_bytes * world_size;
        int blocks = (total_rem + threads - 1) / threads;
        pull_gather_rem_kernel<<<blocks, threads, 0, stream>>>(
            peer_ptrs, out_ptr, chunk_bytes, local_bytes, world_size
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pull_gather", &pull_gather, "UVA vectorized pull gather");
}
'''

_ext = None
_ext_loaded = False

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_quant_pull_ext", CUDA_SRC)
    return _ext

def ensure_ext():
    global _ext_loaded
    if not _ext_loaded:
        if dist.is_initialized():
            if dist.get_rank() == 0:
                _get_ext()
            dist.barrier()
        _get_ext()
        _ext_loaded = True

_symm_cache = {}

def get_symm_state(shape, dtype, device):
    key = (shape, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(shape, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group=dist.group.WORLD)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl

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
    ensure_ext()

    n_elements = local_tensor.numel()
    n_blocks = n_elements // block_size

    if not dist.is_initialized() or dist.get_world_size() == 1:
        y_local = torch.empty_like(local_tensor, dtype=torch.float8_e4m3fn)
        s_local = local_tensor.new_empty(*local_tensor.size()[:-1], local_tensor.size(-1) // block_size, dtype=torch.float32)
        grid = (n_blocks,)
        block_fp8_quant_kernel[grid](local_tensor, y_local, s_local, BLOCK_SIZE=block_size)
        return y_local, s_local

    world_size = dist.get_world_size()
    ext = _get_ext()
    
    y_shape = local_tensor.size()
    s_shape = (*local_tensor.size()[:-1], local_tensor.size(-1) // block_size)
    
    # Fast symmetric memory setup caching
    y_symm, y_hdl = get_symm_state(y_shape, torch.float8_e4m3fn, local_tensor.device)
    s_symm, s_hdl = get_symm_state(s_shape, torch.float32, local_tensor.device)
    
    y_global = torch.empty((world_size, *y_shape), dtype=torch.float8_e4m3fn, device=local_tensor.device)
    s_global = torch.empty((world_size, *s_shape), dtype=torch.float32, device=local_tensor.device)

    # Convert to 1D views for easy continuous pointer offsets
    local_1d = local_tensor.view(-1)
    y_symm_1d = y_symm.view(-1)
    s_symm_1d = s_symm.view(-1)

    y_local_bytes = n_elements * 1
    s_local_bytes = n_blocks * 4

    do_chunk = n_blocks >= 64
    
    if do_chunk:
        chunk0_blocks = n_blocks // 2
        chunk1_blocks = n_blocks - chunk0_blocks
        chunk0_elements = chunk0_blocks * block_size
        chunk1_elements = chunk1_blocks * block_size
        
        stream_comp = torch.cuda.Stream()
        stream_comm = torch.cuda.Stream()
        
        # Phase 1: Compute Chunk 0
        with torch.cuda.stream(stream_comp):
            block_fp8_quant_kernel[(chunk0_blocks,)](
                local_1d, y_symm_1d, s_symm_1d, BLOCK_SIZE=block_size
            )
        stream_comp.synchronize()
        y_hdl.barrier(channel=0) 
        
        y_ptrs_0 = y_hdl.buffer_ptrs
        s_ptrs_0 = s_hdl.buffer_ptrs
        
        # Phase 2: Pull Chunk 0 AND Compute Chunk 1 simultaneously (Compute-Communication overlap)
        with torch.cuda.stream(stream_comm):
            ext.pull_gather(y_ptrs_0, y_global, chunk0_elements * 1, y_local_bytes, 0, world_size)
            ext.pull_gather(s_ptrs_0, s_global, chunk0_blocks * 4, s_local_bytes, 0, world_size)
            
        with torch.cuda.stream(stream_comp):
            block_fp8_quant_kernel[(chunk1_blocks,)](
                local_1d[chunk0_elements:], y_symm_1d[chunk0_elements:], s_symm_1d[chunk0_blocks:], BLOCK_SIZE=block_size
            )
        stream_comp.synchronize()
        y_hdl.barrier(channel=1)
        
        y_ptrs_1 = [p + chunk0_elements * 1 for p in y_hdl.buffer_ptrs]
        s_ptrs_1 = [p + chunk0_blocks * 4 for p in s_hdl.buffer_ptrs]
        
        # Phase 3: Pull Chunk 1
        with torch.cuda.stream(stream_comm):
            ext.pull_gather(y_ptrs_1, y_global, chunk1_elements * 1, y_local_bytes, chunk0_elements * 1, world_size)
            ext.pull_gather(s_ptrs_1, s_global, chunk1_blocks * 4, s_local_bytes, chunk0_blocks * 4, world_size)
            
        stream_comm.synchronize()
    else:
        grid = (n_blocks,)
        block_fp8_quant_kernel[grid](local_1d, y_symm_1d, s_symm_1d, BLOCK_SIZE=block_size)
        y_hdl.barrier(channel=0)
        
        ext.pull_gather(y_hdl.buffer_ptrs, y_global, y_local_bytes, y_local_bytes, 0, world_size)
        ext.pull_gather(s_hdl.buffer_ptrs, s_global, s_local_bytes, s_local_bytes, 0, world_size)

    # Barrier 2 ensures no ranks restart and overwrite buffers while others finish reading
    y_hdl.barrier(channel=2)

    # Replicate structural properties of typical dim=0 torch.cat operations 
    y_out = y_global.view(-1, *y_shape[1:]) if y_shape else y_global.view(-1)
    s_out = s_global.view(-1, *s_shape[1:]) if s_shape else s_global.view(-1)
    
    return y_out, s_out