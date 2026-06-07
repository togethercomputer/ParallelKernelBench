import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional
import triton
import triton.language as tl
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

struct PtrArray {
    uint64_t ptrs[32]; // Accommodate standard single-node (8) or multi-node up to 32
};

__global__ void uva_gather_kernel_bf16_vec(
    PtrArray remote_ptrs,
    void* __restrict__ out,
    int64_t elements_per_rank,
    int world_size
) {
    // Vectorized 128-bit load/store: 8 bfloat16 elements per int4
    int64_t vec_per_rank = elements_per_rank / 8; 
    int64_t total_vecs = vec_per_rank * world_size;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_vecs) {
        int rank = idx / vec_per_rank;
        int64_t offset = idx % vec_per_rank;
        
        const int4* src = reinterpret_cast<const int4*>(remote_ptrs.ptrs[rank]);
        int4* dst = reinterpret_cast<int4*>(out);
        
        dst[idx] = src[offset];
    }
}

__global__ void uva_gather_kernel_bf16_scalar(
    PtrArray remote_ptrs,
    void* __restrict__ out,
    int64_t elements_per_rank,
    int world_size
) {
    int64_t total_elements = elements_per_rank * world_size;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int rank = idx / elements_per_rank;
        int64_t offset = idx % elements_per_rank;
        
        const uint16_t* src = reinterpret_cast<const uint16_t*>(remote_ptrs.ptrs[rank]);
        uint16_t* dst = reinterpret_cast<uint16_t*>(out);
        
        dst[idx] = src[offset];
    }
}

void uva_gather_bf16(
    std::vector<int64_t> remote_ptrs_list,
    torch::Tensor out,
    int64_t elements_per_rank
) {
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    
    int world_size = remote_ptrs_list.size();
    TORCH_CHECK(world_size <= 32, "world_size > 32 not supported by PtrArray");
    
    PtrArray remote_ptrs;
    for (int i = 0; i < world_size; ++i) {
        remote_ptrs.ptrs[i] = remote_ptrs_list[i];
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (elements_per_rank % 8 == 0) {
        int64_t vec_per_rank = elements_per_rank / 8;
        int64_t total_vecs = vec_per_rank * world_size;
        int threads = 256;
        int blocks = (total_vecs + threads - 1) / threads;
        uva_gather_kernel_bf16_vec<<<blocks, threads, 0, stream>>>(
            remote_ptrs, out.data_ptr(), elements_per_rank, world_size
        );
    } else {
        int64_t total_elements = elements_per_rank * world_size;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        uva_gather_kernel_bf16_scalar<<<blocks, threads, 0, stream>>>(
            remote_ptrs, out.data_ptr(), elements_per_rank, world_size
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_gather_bf16", &uva_gather_bf16, "UVA direct gather kernel for 16-bit tensors");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_gather_bf16_ext", CUDA_SRC)
    return _ext

_symm_cache = None

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype and c["device"] == device and c["group"] is group:
            return c["buf"], c["hdl"]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache = {"n": n, "dtype": dtype, "device": device, "group": group, "buf": buf, "hdl": hdl}
    return buf, hdl


@triton.jit
def fused_siglip_loss_kernel(
    image_ptr, text_ptr, loss_out_ptr,
    scale, bias,
    B, WB, D,
    stride_ib, stride_id,
    stride_twb, stride_td,
    is_local_block: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_WB: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    
    # Store block local losses in an array, 1 slot per row
    loss_acc = tl.zeros([BLOCK_B], dtype=tl.float32)
    
    for wb_start in range(0, WB, BLOCK_WB):
        offs_wb = wb_start + tl.arange(0, BLOCK_WB)
        mask_wb = offs_wb < WB
        
        acc_logits = tl.zeros([BLOCK_B, BLOCK_WB], dtype=tl.float32)
        
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            
            i_ptrs = image_ptr + offs_b[:, None] * stride_ib + offs_d[None, :] * stride_id
            t_ptrs = text_ptr + offs_d[:, None] * stride_td + offs_wb[None, :] * stride_twb
            
            i_mask = mask_b[:, None] & (offs_d[None, :] < D)
            t_mask = (offs_d[:, None] < D) & mask_wb[None, :]
            
            i_vals = tl.load(i_ptrs, mask=i_mask, other=0.0)
            t_vals = tl.load(t_ptrs, mask=t_mask, other=0.0)
            
            acc_logits += tl.dot(i_vals, t_vals)
            
        logits = acc_logits * scale + bias
        
        if is_local_block:
            is_pos = offs_b[:, None] == offs_wb[None, :]
            labels = tl.where(is_pos, 1.0, -1.0)
        else:
            labels = -1.0
            
        # Numerically stable fused siglip computation
        z = -labels * logits
        abs_z = tl.abs(z)
        loss_val = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-abs_z))
        
        valid_mask = mask_b[:, None] & mask_wb[None, :]
        loss_val = tl.where(valid_mask, loss_val, 0.0)
        
        loss_acc += tl.sum(loss_val, axis=1)
        
    # Atomically add individual row accumulations to the global scalar
    add_ptrs = loss_out_ptr + tl.arange(0, BLOCK_B) * 0 
    tl.atomic_add(add_ptrs, loss_acc, mask=mask_b)


@torch.no_grad()
def solution(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
    logit_bias: float = 0.0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    
    if rank == 0:
        _get_ext()
    dist.barrier(group)

    image_features = image_features.contiguous()
    text_features = text_features.contiguous()
    B, D = text_features.shape
    n = text_features.numel()
    
    buf, hdl = _get_symm_state(n, text_features.dtype, text_features.device, group)
    
    hdl.barrier(channel=0)
    buf.copy_(text_features.view(-1))
    hdl.barrier(channel=1)
    
    # Pre-allocate output logic array initializing to 0
    loss_out = torch.zeros(1, dtype=torch.float32, device=image_features.device)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    
    stream2 = None
    gathered_remote = None
    
    # Overlap Schedule: Dispatch communication on stream2 while running compute on current stream
    if world_size > 1:
        remote_ptrs = [int(hdl.buffer_ptrs[p]) for p in range(world_size) if p != rank]
                
        stream2 = torch.cuda.Stream(device=image_features.device)
        with torch.cuda.stream(stream2):
            gathered_remote = torch.empty(
                ((world_size - 1) * B, D), 
                dtype=text_features.dtype, 
                device=image_features.device
            )
            _get_ext().uva_gather_bf16(remote_ptrs, gathered_remote, n)
            
    # Independent Compute: Local block
    fused_siglip_loss_kernel[grid](
        image_features, text_features, loss_out,
        logit_scale, logit_bias,
        B, B, D,
        image_features.stride(0), image_features.stride(1),
        text_features.stride(0), text_features.stride(1),
        is_local_block=True,
        BLOCK_B=32, BLOCK_WB=64, BLOCK_D=64, num_warps=4, num_stages=3
    )
    
    # Dependent Compute: Compute against gathered memory after stream sync
    if world_size > 1:
        torch.cuda.current_stream().wait_stream(stream2)
        fused_siglip_loss_kernel[grid](
            image_features, gathered_remote, loss_out,
            logit_scale, logit_bias,
            B, (world_size - 1) * B, D,
            image_features.stride(0), image_features.stride(1),
            gathered_remote.stride(0), gathered_remote.stride(1),
            is_local_block=False,
            BLOCK_B=32, BLOCK_WB=64, BLOCK_D=64, num_warps=4, num_stages=3
        )
        
    return loss_out[0] / B