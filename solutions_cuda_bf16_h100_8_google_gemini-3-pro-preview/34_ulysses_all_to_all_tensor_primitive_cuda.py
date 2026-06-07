"""
Optimized all_to_all_tensor for sequence parallelism (Ulysses).

Strategy:
- Device-side Communication: Uses `torch.distributed._symmetric_memory` to allocate
  persistent symmetric input buffers on each rank.
- Compute-Communication Fusion: Instead of allocating intermediate lists of chunk tensors
  and launching multi-step NCCL collectives, we launch a single custom CUDA P2P kernel.
- Pull-based P2P over NVLink: The kernel allows each rank to read its required chunks
  directly from the symmetric input buffers of all peers. 
- Fast Indexing: Multidimensional tensor coordinates are collapsed on the host into a 
  minimal set of outer loops, leaving the largest possible contiguous innermost dimension
  mapped directly to thread blocks for perfectly coalesced memory accesses.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

#define MAX_WORLD_SIZE 32

template<typename scalar_t>
struct PeerPtrs {
    const scalar_t* ptrs[MAX_WORLD_SIZE];
};

struct ShapeStrides {
    int64_t shape[8];
    int64_t stride_in[8];
    int64_t stride_out[8];
};

template<typename scalar_t>
__global__ void all_to_all_pull_kernel(
    PeerPtrs<scalar_t> peers,
    scalar_t* __restrict__ out_ptr,
    int rank,
    int world_size,
    int64_t numel_chunk,
    int64_t inner_size,
    ShapeStrides ss,
    int64_t orig_stride_in_scatter,
    int64_t orig_stride_out_gather,
    int64_t c_sc,
    int64_t S_ga
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y; 
    
    if (tid < numel_chunk) {
        int64_t outer_idx = tid / inner_size;
        int64_t inner_idx = tid % inner_size;
        
        int64_t temp = outer_idx;
        int64_t offset_in = rank * c_sc * orig_stride_in_scatter + inner_idx;
        int64_t offset_out = p * S_ga * orig_stride_out_gather + inner_idx;
        
        #pragma unroll
        for (int d = 7; d >= 0; --d) {
            int64_t size = ss.shape[d];
            if (size > 1) {
                int64_t coord = temp % size;
                temp = temp / size;
                offset_in += coord * ss.stride_in[d];
                offset_out += coord * ss.stride_out[d];
            }
        }
        
        out_ptr[offset_out] = peers.ptrs[p][offset_in];
    }
}

void launch_all_to_all_pull(
    std::vector<int64_t> peer_ptrs_vec,
    torch::Tensor out_tensor,
    int rank,
    int world_size,
    int64_t numel_chunk,
    int64_t inner_size,
    std::vector<int64_t> outer_shape,
    std::vector<int64_t> outer_stride_in,
    std::vector<int64_t> outer_stride_out,
    int64_t orig_stride_in_scatter,
    int64_t orig_stride_out_gather,
    int64_t c_sc,
    int64_t S_ga
) {
    TORCH_CHECK(world_size <= MAX_WORLD_SIZE, "world_size exceeds MAX_WORLD_SIZE");
    
    ShapeStrides ss;
    for (int i = 0; i < 8; ++i) {
        ss.shape[i] = outer_shape[i];
        ss.stride_in[i] = outer_stride_in[i];
        ss.stride_out[i] = outer_stride_out[i];
    }
    
    int threads = 256;
    int blocks_x = (numel_chunk + threads - 1) / threads;
    dim3 blocks(blocks_x, world_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (out_tensor.dtype() == torch::kBFloat16) {
        PeerPtrs<__nv_bfloat16> peers;
        for (int i = 0; i < world_size; ++i) {
            peers.ptrs[i] = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs_vec[i]);
        }
        __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out_tensor.data_ptr<at::BFloat16>());
        all_to_all_pull_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            peers, out_ptr, rank, world_size, numel_chunk, inner_size, ss,
            orig_stride_in_scatter, orig_stride_out_gather, c_sc, S_ga
        );
    } else if (out_tensor.dtype() == torch::kFloat32) {
        PeerPtrs<float> peers;
        for (int i = 0; i < world_size; ++i) {
            peers.ptrs[i] = reinterpret_cast<const float*>(peer_ptrs_vec[i]);
        }
        float* out_ptr = reinterpret_cast<float*>(out_tensor.data_ptr<float>());
        all_to_all_pull_kernel<float><<<blocks, threads, 0, stream>>>(
            peers, out_ptr, rank, world_size, numel_chunk, inner_size, ss,
            orig_stride_in_scatter, orig_stride_out_gather, c_sc, S_ga
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only bfloat16 and float32 are supported.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_all_to_all_pull", &launch_all_to_all_pull, "Ulysses all-to-all pull kernel");
}
'''

_ext = None
_compiled = False


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_all_to_all_pull", CUDA_SRC)
    return _ext


def _ensure_compiled(group: dist.ProcessGroup):
    global _compiled
    if not _compiled:
        rank = dist.get_rank(group)
        if rank == 0:
            _get_ext()
        dist.barrier(group)
        if rank != 0:
            _get_ext()
        _compiled = True


_symm_cache = {}


def _get_symm_state(shape_in: tuple, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = (shape_in, dtype, device, id(group))
    if key in _symm_cache:
        return _symm_cache[key]
        
    buf = symm_mem.empty(shape_in, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    peer_ptrs = [int(p) for p in hdl.buffer_ptrs]
    
    _symm_cache[key] = (buf, hdl, peer_ptrs)
    return buf, hdl, peer_ptrs


_arg_cache = {}


def _get_kernel_args(shape_in: tuple, scatter_dim: int, gather_dim: int, world_size: int):
    key = (shape_in, scatter_dim, gather_dim, world_size)
    if key in _arg_cache:
        return _arg_cache[key]
        
    shape_out = list(shape_in)
    shape_out[scatter_dim] = shape_in[scatter_dim] // world_size
    shape_out[gather_dim] = shape_in[gather_dim] * world_size
    
    def get_strides(shape):
        strides = [1] * len(shape)
        for i in range(len(shape)-2, -1, -1):
            strides[i] = strides[i+1] * shape[i+1]
        return strides
        
    stride_in = get_strides(shape_in)
    stride_out = get_strides(shape_out)
    
    chunk_shape = list(shape_in)
    chunk_shape[scatter_dim] = shape_in[scatter_dim] // world_size
    
    new_chunk = [chunk_shape[-1]]
    new_stride_in = [stride_in[-1]]
    new_stride_out = [stride_out[-1]]
    
    for d in range(len(chunk_shape)-2, -1, -1):
        # Collapse contiguous dimensions avoiding div/mod overhead in kernel
        if (stride_in[d] == new_stride_in[0] * new_chunk[0] and
            stride_out[d] == new_stride_out[0] * new_chunk[0]):
            new_chunk[0] *= chunk_shape[d]
        else:
            new_chunk.insert(0, chunk_shape[d])
            new_stride_in.insert(0, stride_in[d])
            new_stride_out.insert(0, stride_out[d])
            
    inner_size = new_chunk[-1]
    
    outer_shape = new_chunk[:-1]
    outer_stride_in = new_stride_in[:-1]
    outer_stride_out = new_stride_out[:-1]
    
    if len(outer_shape) > 8:
        raise ValueError("Too many tensor dimensions after collapsing.")
    
    while len(outer_shape) < 8:
        outer_shape.insert(0, 1)
        outer_stride_in.insert(0, 0)
        outer_stride_out.insert(0, 0)
        
    numel_chunk = 1
    for s in chunk_shape:
        numel_chunk *= s
        
    orig_stride_in_scatter = stride_in[scatter_dim]
    orig_stride_out_gather = stride_out[gather_dim]
    c_sc = chunk_shape[scatter_dim]
    S_ga = chunk_shape[gather_dim]
    
    res = (
        int(numel_chunk), int(inner_size), 
        [int(x) for x in outer_shape], 
        [int(x) for x in outer_stride_in], 
        [int(x) for x in outer_stride_out],
        int(orig_stride_in_scatter), int(orig_stride_out_gather), int(c_sc), int(S_ga)
    )
    _arg_cache[key] = res
    return res


@torch.no_grad()
def solution(
    x: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x.contiguous()

    x = x.contiguous()
    
    shape_out = list(x.shape)
    shape_out[scatter_dim] = x.shape[scatter_dim] // world_size
    shape_out[gather_dim] = x.shape[gather_dim] * world_size
    
    if x.numel() == 0:
        return torch.empty(shape_out, dtype=x.dtype, device=x.device)

    _ensure_compiled(group)
    rank = dist.get_rank(group)
    
    buf, hdl, peer_ptrs = _get_symm_state(
        tuple(x.shape), x.dtype, x.device, group
    )
    
    args = _get_kernel_args(
        tuple(x.shape), scatter_dim, gather_dim, world_size
    )
    
    out_tensor = torch.empty(shape_out, dtype=x.dtype, device=x.device)
    
    # Wait for peers to finish reading from the symmetric buffer of the previous iteration
    hdl.barrier(channel=0)
    
    # Push local chunk to the symmetric buffer for peers to read
    buf.copy_(x)
    
    # Wait for peers to finish writing to their symmetric buffers
    hdl.barrier(channel=1)
    
    # Launch direct fused P2P pulling operations
    _get_ext().launch_all_to_all_pull(
        peer_ptrs,
        out_tensor,
        rank,
        world_size,
        *args
    )
    
    return out_tensor