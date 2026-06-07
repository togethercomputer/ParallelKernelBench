"""
Strategy:
1. **Device-Side Communication (UVA)**: Replaced multiple `torch.distributed.all_to_all` calls, chunking, and list concatenations with a single custom CUDA Pull kernel. Symmetric memory is allocated once, and peers read directly from each other's buffers over NVLink using UVA pointers.
2. **Fused Reshape and All-To-All**: The reference implementation heavily relies on intermediate `view`, `tensor_split`, and `cat` operations. The custom kernel maps threads directly to the final `out` tensor layout and computes the precise source indices in remote peers, effectively fusing the `gather`, `scatter`, and all reshapes into one memory-bound operation.
3. **Vectorized Memory Access**: To maximize memory bandwidth over NVLink, the CUDA kernel leverages `float4` (16-byte) vectorized loads/stores whenever the chunk size aligns to 8 `bfloat16` elements (which is practically guaranteed for typical head sizes in BF16).
4. **Compute-Communication Overlap**: Since this is a pure communication operator without adjacent compute, overlap is achieved internally by relying on the GPU's memory subsystem to pipeline the remote NVLink reads and local L2/HBM writes across warps.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Any, Optional, Tuple
from torch.distributed import ProcessGroup
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

struct PtrArrayVec {
    float4* ptrs[16];
};

struct PtrArrayBf16 {
    uint16_t* ptrs[16];
};

__global__ void ulysses_pull_kernel_vec(
    PtrArrayVec X_ptrs,
    float4* __restrict__ Y,
    int prefix, int S, int mid, int W, int K_vec, int c,
    int64_t total_vec_elements
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vec_elements) return;

    int k = idx % K_vec;
    int64_t temp = idx / K_vec;
    int t = temp % 3;
    temp /= 3;
    int m = temp % mid;
    temp /= mid;
    int s = temp % S;
    temp /= S;
    int r = temp % W;
    int p = temp / W;

    int64_t p64 = p, S64 = S, s64 = s, mid64 = mid, m64 = m, t64 = t, W64 = W, c64 = c, K_vec64 = K_vec, k64 = k;
    int64_t src_idx = (((((p64 * S64) + s64) * mid64 + m64) * 3 + t64) * W64 + c64) * K_vec64 + k64;

    Y[idx] = X_ptrs.ptrs[r][src_idx];
}

__global__ void ulysses_pull_kernel_bf16(
    PtrArrayBf16 X_ptrs,
    uint16_t* __restrict__ Y,
    int prefix, int S, int mid, int W, int K, int c,
    int64_t total_elements
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int k = idx % K;
    int64_t temp = idx / K;
    int t = temp % 3;
    temp /= 3;
    int m = temp % mid;
    temp /= mid;
    int s = temp % S;
    temp /= S;
    int r = temp % W;
    int p = temp / W;

    int64_t p64 = p, S64 = S, s64 = s, mid64 = mid, m64 = m, t64 = t, W64 = W, c64 = c, K64 = K, k64 = k;
    int64_t src_idx = (((((p64 * S64) + s64) * mid64 + m64) * 3 + t64) * W64 + c64) * K64 + k64;

    Y[idx] = X_ptrs.ptrs[r][src_idx];
}

void ulysses_pull_bf16(
    std::vector<int64_t> remote_X_ptrs,
    torch::Tensor local_Y,
    int prefix, int S, int mid, int W, int K, int c
) {
    TORCH_CHECK(local_Y.is_cuda(), "local_Y must be CUDA");
    TORCH_CHECK(local_Y.dtype() == torch::kBFloat16, "local_Y must be bfloat16");
    TORCH_CHECK(local_Y.is_contiguous(), "local_Y must be contiguous");

    int64_t total_elements = local_Y.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int threads = 256;

    if (K % 8 == 0 && total_elements % 8 == 0) {
        int K_vec = K / 8;
        int64_t total_vec = total_elements / 8;
        const int blocks = (total_vec + threads - 1) / threads;
        PtrArrayVec ptrs_struct;
        for (int i = 0; i < W; ++i) {
            ptrs_struct.ptrs[i] = reinterpret_cast<float4*>(remote_X_ptrs[i]);
        }
        ulysses_pull_kernel_vec<<<blocks, threads, 0, stream>>>(
            ptrs_struct,
            reinterpret_cast<float4*>(local_Y.data_ptr()),
            prefix, S, mid, W, K_vec, c, total_vec
        );
    } else {
        const int blocks = (total_elements + threads - 1) / threads;
        PtrArrayBf16 ptrs_struct;
        for (int i = 0; i < W; ++i) {
            ptrs_struct.ptrs[i] = reinterpret_cast<uint16_t*>(remote_X_ptrs[i]);
        }
        ulysses_pull_kernel_bf16<<<blocks, threads, 0, stream>>>(
            ptrs_struct,
            reinterpret_cast<uint16_t*>(local_Y.data_ptr()),
            prefix, S, mid, W, K, c, total_elements
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ulysses_pull_bf16", &ulysses_pull_bf16, "Ulysses fused QKV pull via UVA");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_pull_uva_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(numel: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    key = (numel, dtype, device, group)
    if key in _symm_cache:
        return _symm_cache[key]

    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl


# ----- Helper functions for fallback Autograd -----

def _pad_tensor(x: torch.Tensor, dim: int, padding_size: int, padding_value: int = 0) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.full(shape, padding_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)

def _all_to_all_single(x: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup):
    sp_world_size = dist.get_world_size(group)
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (
            x.reshape(
                [gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size]
                + list(x.shape[2:])
            )
            .transpose(0, 1)
            .reshape(
                [gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size]
                + list(x.shape[2:])
            )
            .contiguous()
        )
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x.contiguous(), group=group)
    if scatter_dim == 0:
        output = torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
    return output

def _all_to_all(local_input: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup):
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()

def _all_to_all_tensor(x: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup):
    if scatter_dim <= 1 and gather_dim <= 1:
        return _all_to_all_single(x, scatter_dim, gather_dim, group)
    return _all_to_all(x, scatter_dim, gather_dim, group)


# ----- Custom Autograd Function -----

class _FusedUlyssesPull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv_tensor, seq_dim, group, unpadded_dim_size, restore_shape):
        ctx.seq_dim = seq_dim
        ctx.group = group
        ctx.unpadded_dim_size = unpadded_dim_size
        ctx.restore_shape = restore_shape
        ctx.orig_shape = qkv_tensor.shape

        orig_shape = list(qkv_tensor.shape)
        numel = qkv_tensor.numel()
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        if not group or world_size == 1:
            return qkv_tensor

        # Calculate logical dimensions
        prefix = 1
        for i in range(seq_dim):
            prefix *= orig_shape[i]
        S = orig_shape[seq_dim]
        mid = 1
        for i in range(seq_dim + 1, len(orig_shape) - 1):
            mid *= orig_shape[i]
        
        qkv_proj_dim = orig_shape[-1]
        K = (qkv_proj_dim // 3) // world_size

        qkv_tensor = qkv_tensor.contiguous()

        buf, hdl = _get_symm_state(numel, qkv_tensor.dtype, qkv_tensor.device, group)

        # 1. Sync before writing to shared symm mem
        hdl.barrier(channel=0)
        
        # 2. Local contiguous copy to our symm mem slice
        buf.view(-1).copy_(qkv_tensor.view(-1))
        
        # 3. Sync to ensure peers can now read our slice
        hdl.barrier(channel=1)

        # Allocate fresh output local tensor (prevents benchmark leaks/mutations)
        out_flat = torch.empty(numel, dtype=qkv_tensor.dtype, device=qkv_tensor.device)
        remote_ptrs = [int(hdl.buffer_ptrs[i]) for i in range(world_size)]

        # 4. Pull directly from peers into final layout contiguously
        _get_ext().ulysses_pull_bf16(
            remote_ptrs,
            out_flat,
            prefix, S, mid, world_size, K, rank
        )

        # Re-shape layout appropriately
        if restore_shape:
            out_shape = list(orig_shape)
            out_shape[seq_dim] *= world_size
            out_shape[-1] = qkv_proj_dim // world_size
            out = out_flat.view(out_shape)
        else:
            out_shape = list(orig_shape)
            out_shape[seq_dim] *= world_size
            out_shape[-1] = 3
            out_shape.append(K)
            out = out_flat.view(out_shape)

        if unpadded_dim_size and unpadded_dim_size % world_size != 0:
            padding_size = out_shape[seq_dim] - unpadded_dim_size
            slc = [slice(None)] * len(out_shape)
            slc[seq_dim] = slice(0, -padding_size)
            out = out[tuple(slc)]

        return out

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        sp_world = dist.get_world_size(group)
        
        if ctx.unpadded_dim_size and ctx.unpadded_dim_size % sp_world != 0:
            padding_size = ctx.orig_shape[ctx.seq_dim] * sp_world - ctx.unpadded_dim_size
            grad_output = _pad_tensor(grad_output, ctx.seq_dim, padding_size)
            
        if ctx.restore_shape:
            bef_shape = list(ctx.orig_shape)
            qkv_proj_dim = bef_shape[-1]
            bef_shape = bef_shape[:-1] + [3, qkv_proj_dim // 3]
            bef_shape[ctx.seq_dim] *= sp_world
            bef_shape[-1] = bef_shape[-1] // sp_world
            grad_output = grad_output.view(bef_shape)
            
        scatter_dim = len(ctx.orig_shape)
        gather_dim = ctx.seq_dim
        
        # Standard reverse AllToAll to fulfill correct autograd chain
        grad_input = _all_to_all_tensor(grad_output, gather_dim, scatter_dim, group)
        grad_input = grad_input.view(ctx.orig_shape)
        
        return grad_input, None, None, None, None


# ----- Main Interface -----

def solution(
    qkv_tensor: torch.Tensor,
    seq_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    
    if dist.get_rank(group) == 0:
        _get_ext()
    dist.barrier(group)

    return _FusedUlyssesPull.apply(
        qkv_tensor, seq_dim, group, unpadded_dim_size or 0, restore_shape
    )
"""