import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

import triton
import triton.language as tl

# Custom CUDA extension for fully device-side reductions via symmetric memory.
CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// buf memory layout per rank (4 floats):
// [0]: non_ep_total local sum_sq
// [1]: ep_total local sum_sq
// [2]: ep_total intermediate/final sum
// [3]: non_ep_total final sum

__global__ void reduce_groups_kernel(
    float* local_buf,
    const int64_t* remote_ptrs,
    const int32_t* fsdp_ranks, int fsdp_size,
    const int32_t* ep_fsdp_ranks, int ep_fsdp_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float non_ep_sum = 0.0f;
        if (fsdp_size > 0) {
            for (int i = 0; i < fsdp_size; ++i) {
                int r = fsdp_ranks[i];
                const float* peer_buf = reinterpret_cast<const float*>(remote_ptrs[r]);
                non_ep_sum += peer_buf[0];
            }
        } else {
            non_ep_sum = local_buf[0];
        }

        float ep_sum = 0.0f;
        if (ep_fsdp_size > 0) {
            for (int i = 0; i < ep_fsdp_size; ++i) {
                int r = ep_fsdp_ranks[i];
                const float* peer_buf = reinterpret_cast<const float*>(remote_ptrs[r]);
                ep_sum += peer_buf[1];
            }
        } else {
            ep_sum = local_buf[1];
        }

        local_buf[3] = non_ep_sum;
        local_buf[2] = ep_sum;
    }
}

__global__ void reduce_ep_group_kernel(
    float* local_buf,
    const int64_t* remote_ptrs,
    const int32_t* ep_ranks, int ep_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float ep_sum = 0.0f;
        if (ep_size > 0) {
            for (int i = 0; i < ep_size; ++i) {
                int r = ep_ranks[i];
                const float* peer_buf = reinterpret_cast<const float*>(remote_ptrs[r]);
                ep_sum += peer_buf[2];
            }
        } else {
            ep_sum = local_buf[2];
        }
        local_buf[2] = ep_sum;
    }
}

void reduce_groups_step1(
    torch::Tensor local_buf,
    torch::Tensor remote_ptrs,
    torch::Tensor fsdp_ranks,
    torch::Tensor ep_fsdp_ranks
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_groups_kernel<<<1, 1, 0, stream>>>(
        local_buf.data_ptr<float>(),
        remote_ptrs.data_ptr<int64_t>(),
        fsdp_ranks.numel() > 0 ? fsdp_ranks.data_ptr<int32_t>() : nullptr, fsdp_ranks.numel(),
        ep_fsdp_ranks.numel() > 0 ? ep_fsdp_ranks.data_ptr<int32_t>() : nullptr, ep_fsdp_ranks.numel()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_groups_step2(
    torch::Tensor local_buf,
    torch::Tensor remote_ptrs,
    torch::Tensor ep_ranks
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_ep_group_kernel<<<1, 1, 0, stream>>>(
        local_buf.data_ptr<float>(),
        remote_ptrs.data_ptr<int64_t>(),
        ep_ranks.numel() > 0 ? ep_ranks.data_ptr<int32_t>() : nullptr, ep_ranks.numel()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_groups_step1", &reduce_groups_step1);
    m.def("reduce_groups_step2", &reduce_groups_step2);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("clip_grad_norm_ep_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    if device not in _symm_cache:
        buf = symm_mem.empty(n, device=device, dtype=dtype)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        _symm_cache[device] = (buf, hdl)
    return _symm_cache[device]

_group_cache = {}
def get_group_ranks(group: Optional[dist.ProcessGroup], device: torch.device) -> torch.Tensor:
    if group is None:
        return torch.empty(0, dtype=torch.int32, device=device)
    gid = id(group)
    if gid not in _group_cache:
        if hasattr(dist, "get_process_group_ranks"):
            ranks = dist.get_process_group_ranks(group)
        else:
            ranks = [dist.get_global_rank(group, i) for i in range(dist.get_world_size(group))]
        _group_cache[gid] = torch.tensor(ranks, dtype=torch.int32, device=device)
    return _group_cache[gid]

def get_remote_ptrs(hdl, device: torch.device) -> torch.Tensor:
    if not hasattr(hdl, "_remote_ptrs_tensor"):
        hdl._remote_ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    return hdl._remote_ptrs_tensor

@triton.jit
def _norm_sq_scale_kernel(
    ptr,
    size,
    scale,
    out_ptr,
    out_idx,
    BLOCK_SIZE: tl.constexpr
):
    """Fuses optional gradient scaling and local sum-of-squares calculation."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Conditionally branch out scale to avoid modifying tensor if unneeded 
    if scale != 1.0:
        x_f32 = x_f32 * scale
        tl.store(ptr + offsets, x_f32.to(x.dtype), mask=mask)

    sq = x_f32 * x_f32
    sum_sq = tl.sum(sq, axis=0)

    # Accumulate globally
    tl.atomic_add(out_ptr + out_idx, sum_sq)

@triton.jit
def _clip_scale_kernel(
    ptr,
    size,
    coef_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """Conditionally applies the clip scale to all gradients, ignoring memory traffic if coef == 1.0."""
    coef = tl.load(coef_ptr)
    if coef < 1.0:
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        
        x = tl.load(ptr + offsets, mask=mask)
        x_scaled = x.to(tl.float32) * coef
        tl.store(ptr + offsets, x_scaled.to(x.dtype), mask=mask)


@torch.no_grad()
def solution(
    non_ep_grad_tensors: List[torch.Tensor],
    ep_grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    ep_size: int = 1,
    fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    if norm_type != 2.0:
        raise NotImplementedError("This optimized path only supports L2 norm (norm_type=2.0).")

    device = next(
        (t.device for t in non_ep_grad_tensors + ep_grad_tensors if t is not None), 
        torch.device("cuda")
    )

    ext = _get_ext()
    buf, hdl = _get_symm_state(4, torch.float32, device)

    # Zero accumulation buffer for this collective cycle
    buf.zero_()

    # Step 1: Locally fuse scaling and accumulate sum of squares for both EP & Non-EP streams
    ep_scale = 1.0 / float(ep_size) if ep_size > 1 and ep_grad_tensors else 1.0

    for t in non_ep_grad_tensors:
        if t is not None and (size := t.numel()) > 0:
            grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
            _norm_sq_scale_kernel[grid](t, size, 1.0, buf, 0, BLOCK_SIZE=1024)

    for t in ep_grad_tensors:
        if t is not None and (size := t.numel()) > 0:
            grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
            _norm_sq_scale_kernel[grid](t, size, ep_scale, buf, 1, BLOCK_SIZE=1024)

    # Barrier 0: Stream-ordered flush ensuring Triton blocks are committed to symmetric memory
    hdl.barrier(channel=0)

    # Step 2: Reduce over FSDP group and initial EP-FSDP group
    remote_ptrs = get_remote_ptrs(hdl, device)
    fsdp_ranks = get_group_ranks(fsdp_group, device)
    ep_fsdp_ranks = get_group_ranks(ep_fsdp_group, device)

    ext.reduce_groups_step1(buf, remote_ptrs, fsdp_ranks, ep_fsdp_ranks)

    # Barrier 1: Wait for intermediate step 1 group accumulation to finalize
    hdl.barrier(channel=1)

    # Step 3: Reduce over final EP group for orthogonal EP grads
    ep_ranks = get_group_ranks(ep_group, device)
    ext.reduce_groups_step2(buf, remote_ptrs, ep_ranks)

    # Step 4: Device-side clip coefficient evaluation (avoids CPU/GPU branching)
    total_norm_sq = buf[3] + buf[2]
    total_norm = torch.sqrt(total_norm_sq)
    max_norm_t = torch.tensor(max_norm, device=device, dtype=torch.float32)
    
    # Calculate scale dynamically. If total_norm <= max_norm, coef effectively becomes 1.0
    coef = torch.clamp(max_norm_t / total_norm, max=1.0)

    # Step 5: Conditionally scale all groups inplace if coefficient forces reduction
    for grads in (non_ep_grad_tensors, ep_grad_tensors):
        for t in grads:
            if t is not None and (size := t.numel()) > 0:
                grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
                _clip_scale_kernel[grid](t, size, coef, BLOCK_SIZE=1024)

    return total_norm