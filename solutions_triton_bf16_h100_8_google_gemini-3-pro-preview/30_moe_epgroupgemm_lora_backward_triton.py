import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional, Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

__global__ void symmetric_allreduce_bf16_vectorized_kernel(
    const __nv_bfloat16** peer_ptrs,
    __nv_bfloat16* out,
    int64_t n,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int64_t n8 = n / 8;
    
    // Vectorized path for multiples of 8 elements
    for (int64_t i = idx; i < n8; i += stride) {
        float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        
        for (int r = 0; r < world_size; r++) {
            // Read 8 __nv_bfloat16 values simultaneously using float4
            float4 val = reinterpret_cast<const float4*>(peer_ptrs[r])[i];
            const __nv_bfloat162* vals = reinterpret_cast<const __nv_bfloat162*>(&val);
            
            float2 v0 = __bfloat1622float2(vals[0]);
            float2 v1 = __bfloat1622float2(vals[1]);
            float2 v2 = __bfloat1622float2(vals[2]);
            float2 v3 = __bfloat1622float2(vals[3]);
            
            sums[0] += v0.x; sums[1] += v0.y;
            sums[2] += v1.x; sums[3] += v1.y;
            sums[4] += v2.x; sums[5] += v2.y;
            sums[6] += v3.x; sums[7] += v3.y;
        }
        
        // Pack back into __nv_bfloat162
        __nv_bfloat162 res0 = __floats2bfloat162_rn(sums[0], sums[1]);
        __nv_bfloat162 res1 = __floats2bfloat162_rn(sums[2], sums[3]);
        __nv_bfloat162 res2 = __floats2bfloat162_rn(sums[4], sums[5]);
        __nv_bfloat162 res3 = __floats2bfloat162_rn(sums[6], sums[7]);
        
        float4 out_val;
        __nv_bfloat162* out_vals = reinterpret_cast<__nv_bfloat162*>(&out_val);
        out_vals[0] = res0;
        out_vals[1] = res1;
        out_vals[2] = res2;
        out_vals[3] = res3;
        
        reinterpret_cast<float4*>(out)[i] = out_val;
    }

    // Handle remainder elements
    int64_t rem_start = n8 * 8;
    for (int64_t i = rem_start + idx; i < n; i += stride) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; r++) {
            sum += __bfloat162float(peer_ptrs[r][i]);
        }
        out[i] = __float2bfloat16(sum);
    }
}

void symmetric_allreduce_bf16(
    int64_t peer_ptrs_ptr,
    torch::Tensor out,
    int64_t n,
    int world_size
) {
    const int threads = 256;
    int blocks = std::min((int)((n/8 + threads - 1) / threads), 1024);
    if (blocks == 0) blocks = 1;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const __nv_bfloat16** peer_ptrs = reinterpret_cast<const __nv_bfloat16**>(peer_ptrs_ptr);

    symmetric_allreduce_bf16_vectorized_kernel<<<blocks, threads, 0, stream>>>(
        peer_ptrs,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        n,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symmetric_allreduce_bf16", &symmetric_allreduce_bf16, "UVA symmetric allreduce bf16");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symmetric_allreduce_bf16_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    world_size = dist.get_world_size(group)
    
    key = (group, dtype, device)
    if key in _symm_cache:
        c = _symm_cache[key]
        if c["n"] >= n:
            return c["buf"][:n], c["hdl"], c["out_buf"][:n], c["peer_ptrs"]

    # Pre-allocate to prevent repeated re-allocations if parameter count grows 
    alloc_n = max(n, 1024 * 1024) 
    
    buf = symm_mem.empty(alloc_n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    out_buf = torch.empty(alloc_n, device=device, dtype=dtype)
    peer_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache[key] = {
        "n": alloc_n,
        "buf": buf,
        "hdl": hdl,
        "out_buf": out_buf,
        "peer_ptrs": peer_ptrs
    }
    
    return buf[:n], hdl, out_buf[:n], peer_ptrs


@torch.no_grad()
def solution(
    grad_fc1_1_lora_A: torch.Tensor,
    grad_fc1_2_lora_A: torch.Tensor,
    grad_fc2_lora_B: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    In-place summation of shared LoRA gradients replacing grouped all_reduce collectives.
    Overlaps NVLink data reads via UVA with compute-bound FP32 reductions. 
    """
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    
    if world_size == 1:
        return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B
        
    dtype = grad_fc1_1_lora_A.dtype
    if dtype != torch.bfloat16:
        # Fallback for unexpected non-BF16 calls
        dist.all_reduce(grad_fc1_1_lora_A, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(grad_fc1_2_lora_A, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(grad_fc2_lora_B, op=dist.ReduceOp.SUM, group=group)
        return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B

    n1 = grad_fc1_1_lora_A.numel()
    n2 = grad_fc1_2_lora_A.numel()
    n3 = grad_fc2_lora_B.numel()
    total_n = n1 + n2 + n3
    
    # Initialize extension symmetrically to avoid locking issues
    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()

    buf, hdl, out_buf, peer_ptrs = _get_symm_state(total_n, dtype, grad_fc1_1_lora_A.device, group)
    
    # Pack memory from varying shapes
    buf[:n1].copy_(grad_fc1_1_lora_A.flatten())
    buf[n1:n1+n2].copy_(grad_fc1_2_lora_A.flatten())
    buf[n1+n2:].copy_(grad_fc2_lora_B.flatten())
    
    # Wait until all ranks have populated their slice in symm_mem
    hdl.barrier(channel=0)
    
    _get_ext().symmetric_allreduce_bf16(peer_ptrs.data_ptr(), out_buf, total_n, world_size)
    
    # Ensure kernel completion across ranks before next loop might overwrite symm_mem buffer
    hdl.barrier(channel=0)
    
    # Dispatch summed values back natively into the referenced inputs
    grad_fc1_1_lora_A.copy_(out_buf[:n1].view_as(grad_fc1_1_lora_A))
    grad_fc1_2_lora_A.copy_(out_buf[n1:n1+n2].view_as(grad_fc1_2_lora_A))
    grad_fc2_lora_B.copy_(out_buf[n1+n2:].view_as(grad_fc2_lora_B))
    
    return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B