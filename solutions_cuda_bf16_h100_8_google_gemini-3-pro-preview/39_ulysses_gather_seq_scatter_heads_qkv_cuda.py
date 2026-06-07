"""
Strategy:
This solution bypasses NCCL to provide an optimal NVLink P2P sequence parallel all-to-all schedule. We exploit symmetric memory to directly map the scattered-head and gathered-sequence chunking onto a single fused CUDA kernel. By computing the multi-dimensional stride mapping purely mathematically within the kernel, each device pulls exactly its required `(seq, head)` sub-chunks directly from peers' HBM without any intermediate slicing or concatenations. This maximizes bidirectional NVLink bandwidth, eliminates PyTorch op overhead, and fuses what would normally be several reshapes and chunking operations into one single-launch vectorized pull. We also implement the reverse mapping for a zero-overhead backward pass.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional, Any
from torch.distributed import ProcessGroup
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

template <typename T, int VEC_SIZE>
__global__ void all2all_pull_kernel(
    const uint64_t* __restrict__ symm_ptrs,
    T* __restrict__ out,
    int64_t W,
    int64_t me,
    int64_t SM,
    int64_t h_vecs,
    int64_t num_segments,
    int64_t total_vecs,
    bool is_backward
) {
    int64_t v_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v_idx >= total_vecs) return;

    int64_t chunk_idx = v_idx / h_vecs;
    int64_t v_offset = v_idx % h_vecs;

    int64_t j = chunk_idx / num_segments;
    int64_t segment_idx = chunk_idx % num_segments;

    int64_t a = segment_idx / SM;
    int64_t rem = segment_idx % SM;

    int64_t dest_chunk_idx;
    int64_t src_chunk_idx;
    
    if (!is_backward) {
        dest_chunk_idx = (a * W + j) * SM + rem;
        src_chunk_idx = segment_idx * W + me;
    } else {
        dest_chunk_idx = segment_idx * W + j;
        src_chunk_idx = (a * W + me) * SM + rem;
    }

    int64_t dest_idx = dest_chunk_idx * h_vecs + v_offset;
    int64_t src_idx = src_chunk_idx * h_vecs + v_offset;

    using VecType = typename std::aligned_storage<sizeof(T) * VEC_SIZE, sizeof(T) * VEC_SIZE>::type;
    
    const T* src_ptr = reinterpret_cast<const T*>(symm_ptrs[j]);
    const VecType* src_vec = reinterpret_cast<const VecType*>(src_ptr);
    VecType* out_vec = reinterpret_cast<VecType*>(out);
    
    out_vec[dest_idx] = src_vec[src_idx];
}

void launch_all2all_pull(
    torch::Tensor symm_ptrs_tensor,
    torch::Tensor out,
    int64_t W,
    int64_t me,
    int64_t SM,
    int64_t h,
    int64_t num_segments,
    bool is_backward
) {
    size_t el_size = out.element_size();
    int64_t vec_elements = 1;
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(out.data_ptr());

    if (h % (16 / el_size) == 0 && ptr_val % 16 == 0) {
        vec_elements = 16 / el_size;
    } else if (h % (8 / el_size) == 0 && ptr_val % 8 == 0) {
        vec_elements = 8 / el_size;
    } else if (h % (4 / el_size) == 0 && ptr_val % 4 == 0) {
        vec_elements = 4 / el_size;
    }

    int64_t h_vecs = h / vec_elements;
    int64_t total_vecs = num_segments * W * h_vecs;

    int64_t threads = 256;
    int blocks = (int)((total_vecs + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* symm_ptrs = reinterpret_cast<const uint64_t*>(symm_ptrs_tensor.data_ptr<int64_t>());
    
    #define DISPATCH_KERNEL(T, V) \
        all2all_pull_kernel<T, V><<<blocks, threads, 0, stream>>>( \
            symm_ptrs, out.data_ptr<T>(), W, me, SM, h_vecs, num_segments, total_vecs, is_backward)

    if (out.dtype() == torch::kBFloat16) {
        if (vec_elements == 8) { DISPATCH_KERNEL(at::BFloat16, 8); }
        else if (vec_elements == 4) { DISPATCH_KERNEL(at::BFloat16, 4); }
        else if (vec_elements == 2) { DISPATCH_KERNEL(at::BFloat16, 2); }
        else { DISPATCH_KERNEL(at::BFloat16, 1); }
    } else if (out.dtype() == torch::kFloat16) {
        if (vec_elements == 8) { DISPATCH_KERNEL(at::Half, 8); }
        else if (vec_elements == 4) { DISPATCH_KERNEL(at::Half, 4); }
        else if (vec_elements == 2) { DISPATCH_KERNEL(at::Half, 2); }
        else { DISPATCH_KERNEL(at::Half, 1); }
    } else if (out.dtype() == torch::kFloat32) {
        if (vec_elements == 4) { DISPATCH_KERNEL(float, 4); }
        else if (vec_elements == 2) { DISPATCH_KERNEL(float, 2); }
        else { DISPATCH_KERNEL(float, 1); }
    }
    
    #undef DISPATCH_KERNEL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_all2all_pull", &launch_all2all_pull, "Symmetric Memory NVLink All-to-All Pull");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_fused_qkv_all2all_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(shape, dtype, device, group):
    key = (tuple(shape), dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(shape, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]

class FusedQKVAllToAllSymm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv_tensor, seq_dim, group, unpadded_dim_size, restore_shape):
        W = dist.get_world_size(group)
        ctx.seq_dim = seq_dim
        ctx.group = group
        ctx.unpadded_dim_size = unpadded_dim_size
        ctx.restore_shape = restore_shape
        ctx.orig_shape = qkv_tensor.shape
        ctx.W = W
        
        if W == 1:
            return qkv_tensor

        orig_shape = qkv_tensor.shape
        qkv_proj_dim = orig_shape[-1]
        
        bef_all2all_shape = list(orig_shape)
        bef_all2all_shape = bef_all2all_shape[:-1] + [3, qkv_proj_dim // 3]
        
        qkv_tensor = qkv_tensor.contiguous()
        
        buf, hdl, ptrs_tensor = _get_symm_state(bef_all2all_shape, qkv_tensor.dtype, qkv_tensor.device, group)
        
        buf.view(-1).copy_(qkv_tensor.view(-1))
        hdl.barrier(channel=0)
        
        A = 1
        for i in range(seq_dim):
            A *= bef_all2all_shape[i]
        S = bef_all2all_shape[seq_dim]
        M = 1
        for i in range(seq_dim + 1, len(bef_all2all_shape) - 1):
            M *= bef_all2all_shape[i]
        H = bef_all2all_shape[-1]
        
        h = H // W
        num_segments = A * S * M
        
        gathered_shape = list(bef_all2all_shape)
        gathered_shape[seq_dim] = W * S
        gathered_shape[-1] = h
        
        out = torch.empty(gathered_shape, dtype=qkv_tensor.dtype, device=qkv_tensor.device)
        me = dist.get_rank(group)
        
        _get_ext().launch_all2all_pull(ptrs_tensor, out, W, me, S * M, h, num_segments, False)
        
        hdl.barrier(channel=0)
        
        ctx.S_M = S * M
        ctx.h = h
        ctx.num_segments = num_segments
        ctx.bef_all2all_shape = bef_all2all_shape
        ctx.gathered_shape = gathered_shape
        
        final_out = out
        
        if restore_shape:
            out_shape = list(orig_shape)
            out_shape[seq_dim] *= W
            out_shape[-1] = qkv_proj_dim // W
            final_out = final_out.view(out_shape)
        
        if unpadded_dim_size and unpadded_dim_size % W != 0:
            padding_size = final_out.size(seq_dim) - unpadded_dim_size
            slc = [slice(None)] * final_out.dim()
            slc[seq_dim] = slice(0, -padding_size)
            final_out = final_out[tuple(slc)]
            
        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        W = ctx.W
        if W == 1:
            return grad_output, None, None, None, None
            
        grad_output = grad_output.contiguous()
        
        if ctx.unpadded_dim_size and ctx.unpadded_dim_size % W != 0:
            padding_size = ctx.gathered_shape[ctx.seq_dim] - ctx.unpadded_dim_size
            shape = list(grad_output.shape)
            shape[ctx.seq_dim] = padding_size
            pad = torch.zeros(shape, dtype=grad_output.dtype, device=grad_output.device)
            grad_output = torch.cat([grad_output, pad], dim=ctx.seq_dim)
            
        grad_output = grad_output.view(ctx.gathered_shape)
        
        buf, hdl, ptrs_tensor = _get_symm_state(ctx.gathered_shape, grad_output.dtype, grad_output.device, ctx.group)
        buf.view(-1).copy_(grad_output.view(-1))
        hdl.barrier(channel=0)
        
        grad_input = torch.empty(ctx.bef_all2all_shape, dtype=grad_output.dtype, device=grad_output.device)
        me = dist.get_rank(ctx.group)
        
        _get_ext().launch_all2all_pull(ptrs_tensor, grad_input, W, me, ctx.S_M, ctx.h, ctx.num_segments, True)
        
        hdl.barrier(channel=0)
        
        grad_input = grad_input.view(ctx.orig_shape)
        return grad_input, None, None, None, None


def solution(
    qkv_tensor: torch.Tensor,
    seq_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
) -> torch.Tensor:
    """
    Per-rank inputs:
      qkv_tensor: fused QKV [..., qkv_proj_dim]; last dim divisible by 3 and world_size.
      seq_dim: sequence dimension (gather dim for all_to_all).
      group: SP process group (default world).
      unpadded_dim_size: if set and not divisible by world_size, unpad output.
      restore_shape: if True, output shape matches input ndim with seq_dim and last dim resized.

    Returns (per rank):
      output: tensor after fused QKV all_to_all (and optional reshape/unpad).
    """
    group = group or (dist.group.WORLD if dist.is_initialized() else None)
    if not group:
        return qkv_tensor

    return FusedQKVAllToAllSymm.apply(
        qkv_tensor,
        seq_dim,
        group,
        unpadded_dim_size or 0,
        restore_shape
    )