import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension

# We embed the C++ and CUDA source code for our custom fused sequence-parallel kernels.
# 1. rope_local_kernel: Applies RoPE explicitly and writes to symmetric memory directly.
# 2. pull_gather_kernel: Linearly pulls from all remote peer symmetric memory buffers over NVLink into a contiguous global tensor.
CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

struct PtrArray {
    const __nv_bfloat16* ptrs[8];
};

__global__ void rope_local_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int64_t B, int64_t S_local, int64_t H, int64_t D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    int64_t elements_per_thread = 8; // Processing 8 elements for d1 and 8 for d2
    int64_t half_D = D / 2;
    int64_t half_D_vecs = half_D / elements_per_thread;
    
    int64_t total_vecs = B * S_local * H * half_D_vecs;
    if (idx < total_vecs) {
        int64_t d_vec = idx % half_D_vecs;
        int64_t tmp = idx / half_D_vecs;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t s = tmp % S_local;
        int64_t b = tmp / S_local;
        
        int64_t d1 = d_vec * elements_per_thread;
        int64_t d2 = d1 + half_D;
        
        int64_t offset1 = b * (S_local * H * D) + s * (H * D) + h * D + d1;
        int64_t offset2 = b * (S_local * H * D) + s * (H * D) + h * D + d2;
        
        int64_t cos_offset1 = b * (S_local * D) + s * D + d1;
        int64_t cos_offset2 = b * (S_local * D) + s * D + d2;
        
        // 128-bit vectorized loads over 8 bfloat16 elements
        float4 q1_f4 = *reinterpret_cast<const float4*>(q + offset1);
        float4 q2_f4 = *reinterpret_cast<const float4*>(q + offset2);
        float4 k1_f4 = *reinterpret_cast<const float4*>(k + offset1);
        float4 k2_f4 = *reinterpret_cast<const float4*>(k + offset2);
        
        float4 c1_f4 = *reinterpret_cast<const float4*>(cos + cos_offset1);
        float4 c2_f4 = *reinterpret_cast<const float4*>(cos + cos_offset2);
        float4 s1_f4 = *reinterpret_cast<const float4*>(sin + cos_offset1);
        float4 s2_f4 = *reinterpret_cast<const float4*>(sin + cos_offset2);
        
        const __nv_bfloat162* q1_ptr = reinterpret_cast<const __nv_bfloat162*>(&q1_f4);
        const __nv_bfloat162* q2_ptr = reinterpret_cast<const __nv_bfloat162*>(&q2_f4);
        const __nv_bfloat162* c1_ptr = reinterpret_cast<const __nv_bfloat162*>(&c1_f4);
        const __nv_bfloat162* s1_ptr = reinterpret_cast<const __nv_bfloat162*>(&s1_f4);
        const __nv_bfloat162* c2_ptr = reinterpret_cast<const __nv_bfloat162*>(&c2_f4);
        const __nv_bfloat162* s2_ptr = reinterpret_cast<const __nv_bfloat162*>(&s2_f4);
        const __nv_bfloat162* k1_ptr = reinterpret_cast<const __nv_bfloat162*>(&k1_f4);
        const __nv_bfloat162* k2_ptr = reinterpret_cast<const __nv_bfloat162*>(&k2_f4);
        
        float4 out_q1_f4, out_q2_f4, out_k1_f4, out_k2_f4;
        __nv_bfloat162* out_q1 = reinterpret_cast<__nv_bfloat162*>(&out_q1_f4);
        __nv_bfloat162* out_q2 = reinterpret_cast<__nv_bfloat162*>(&out_q2_f4);
        __nv_bfloat162* out_k1 = reinterpret_cast<__nv_bfloat162*>(&out_k1_f4);
        __nv_bfloat162* out_k2 = reinterpret_cast<__nv_bfloat162*>(&out_k2_f4);
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f_q1 = __bfloat1622float2(q1_ptr[i]);
            float2 f_q2 = __bfloat1622float2(q2_ptr[i]);
            float2 f_c1 = __bfloat1622float2(c1_ptr[i]);
            float2 f_s1 = __bfloat1622float2(s1_ptr[i]);
            float2 f_c2 = __bfloat1622float2(c2_ptr[i]);
            float2 f_s2 = __bfloat1622float2(s2_ptr[i]);
            
            float2 f_k1 = __bfloat1622float2(k1_ptr[i]);
            float2 f_k2 = __bfloat1622float2(k2_ptr[i]);
            
            float2 f_out_q1, f_out_q2, f_out_k1, f_out_k2;
            
            // Query RoPE formula
            f_out_q1.x = f_q1.x * f_c1.x - f_q2.x * f_s1.x;
            f_out_q1.y = f_q1.y * f_c1.y - f_q2.y * f_s1.y;
            f_out_q2.x = f_q2.x * f_c2.x + f_q1.x * f_s2.x;
            f_out_q2.y = f_q2.y * f_c2.y + f_q1.y * f_s2.y;
            
            // Key RoPE formula
            f_out_k1.x = f_k1.x * f_c1.x - f_k2.x * f_s1.x;
            f_out_k1.y = f_k1.y * f_c1.y - f_k2.y * f_s1.y;
            f_out_k2.x = f_k2.x * f_c2.x + f_k1.x * f_s2.x;
            f_out_k2.y = f_k2.y * f_c2.y + f_k1.y * f_s2.y;
            
            out_q1[i] = __floats2bfloat162_rn(f_out_q1.x, f_out_q1.y);
            out_q2[i] = __floats2bfloat162_rn(f_out_q2.x, f_out_q2.y);
            out_k1[i] = __floats2bfloat162_rn(f_out_k1.x, f_out_k1.y);
            out_k2[i] = __floats2bfloat162_rn(f_out_k2.x, f_out_k2.y);
        }
        
        // 128-bit vectorized stores
        *reinterpret_cast<float4*>(q_out + offset1) = out_q1_f4;
        *reinterpret_cast<float4*>(q_out + offset2) = out_q2_f4;
        *reinterpret_cast<float4*>(k_out + offset1) = out_k1_f4;
        *reinterpret_cast<float4*>(k_out + offset2) = out_k2_f4;
    }
}

__global__ void pull_gather_kernel(
    PtrArray q_ptrs,
    PtrArray k_ptrs,
    __nv_bfloat16* __restrict__ q_global,
    __nv_bfloat16* __restrict__ k_global,
    int64_t B, int64_t S_local, int64_t H, int64_t D, int64_t world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    int64_t chunk_size = S_local * H * D;
    int64_t total_vecs = (B * world_size * chunk_size) / 8;
    
    if (idx < total_vecs) {
        int64_t out_offset = idx * 8; // Offset in terms of bfloat16 elements
        
        int64_t d_h_s = out_offset % chunk_size;
        int64_t tmp = out_offset / chunk_size;
        int64_t r = tmp % world_size;
        int64_t b = tmp / world_size;
        
        int64_t in_offset = b * chunk_size + d_h_s;
        
        // Direct peer-to-peer read across NVLink via symmetric memory mapped UVA pointer
        float4 q_val = *reinterpret_cast<const float4*>(q_ptrs.ptrs[r] + in_offset);
        float4 k_val = *reinterpret_cast<const float4*>(k_ptrs.ptrs[r] + in_offset);
        
        // Linear local contiguous store
        *reinterpret_cast<float4*>(q_global + out_offset) = q_val;
        *reinterpret_cast<float4*>(k_global + out_offset) = k_val;
    }
}

void compute_rope_local(
    torch::Tensor q, torch::Tensor k,
    torch::Tensor cos, torch::Tensor sin,
    torch::Tensor q_symm, torch::Tensor k_symm
) {
    int64_t B = q.size(0);
    int64_t S_local = q.size(1);
    int64_t H = q.size(2);
    int64_t D = q.size(3);
    
    TORCH_CHECK(D % 16 == 0, "D must be a multiple of 16 for aggressive float4 128-bit vectorization.");
    
    int64_t half_D_vecs = (D / 2) / 8;
    int64_t total_vecs = B * S_local * H * half_D_vecs;
    int threads = 256;
    int blocks = (total_vecs + threads - 1) / threads;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    rope_local_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(q_symm.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_symm.data_ptr<at::BFloat16>()),
        B, S_local, H, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pull_gather(
    std::vector<int64_t> q_symm_ptrs_int,
    std::vector<int64_t> k_symm_ptrs_int,
    torch::Tensor q_global, torch::Tensor k_global,
    int64_t B, int64_t S_local, int64_t H, int64_t D, int64_t world_size
) {
    PtrArray q_ptrs, k_ptrs;
    for (int64_t i = 0; i < world_size; ++i) {
        q_ptrs.ptrs[i] = reinterpret_cast<const __nv_bfloat16*>(q_symm_ptrs_int[i]);
        k_ptrs.ptrs[i] = reinterpret_cast<const __nv_bfloat16*>(k_symm_ptrs_int[i]);
    }
    
    int64_t chunk_size = S_local * H * D;
    int64_t total_vecs = (B * world_size * chunk_size) / 8;
    int threads = 256;
    int blocks = (total_vecs + threads - 1) / threads;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    pull_gather_kernel<<<blocks, threads, 0, stream>>>(
        q_ptrs, k_ptrs,
        reinterpret_cast<__nv_bfloat16*>(q_global.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(k_global.data_ptr<at::BFloat16>()),
        B, S_local, H, D, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_rope_local", &compute_rope_local, "Fused Kernel: Compute RoPE into local symmetric memory");
    m.def("pull_gather", &pull_gather, "Fused Kernel: Direct NVLink memory pull into concatenated global format");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("rope_allgather_ext", CUDA_SRC)
    return _ext

_symm_state = None
def _get_symm_state(B: int, S_local: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
    global _symm_state
    numel = B * S_local * H * D
    if _symm_state is not None:
        c = _symm_state
        if c["numel"] == numel and c["dtype"] == dtype and c["device"] == device:
            return c["q_symm"], c["k_symm"], c["hdl_q"], c["hdl_k"]
            
    q_symm = symm_mem.empty(numel, dtype=dtype, device=device)
    k_symm = symm_mem.empty(numel, dtype=dtype, device=device)
    
    hdl_q = symm_mem.rendezvous(q_symm, dist.group.WORLD)
    hdl_k = symm_mem.rendezvous(k_symm, dist.group.WORLD)
    
    _symm_state = {
        "numel": numel, "dtype": dtype, "device": device,
        "q_symm": q_symm, "k_symm": k_symm,
        "hdl_q": hdl_q, "hdl_k": hdl_k
    }
    return q_symm, k_symm, hdl_q, hdl_k


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)

@torch.no_grad()
def solution(
    q_local: torch.Tensor, 
    k_local: torch.Tensor, 
    cos_local: torch.Tensor, 
    sin_local: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # Standard PyTorch fallback for unsupported properties or uninitialized environments 
    if not dist.is_initialized() or dist.get_world_size() == 1 or q_local.dtype != torch.bfloat16 or q_local.shape[-1] % 16 != 0:
        cos = cos_local.unsqueeze(2)
        sin = sin_local.unsqueeze(2)
        q_embed_local = (q_local * cos) + (rotate_half(q_local) * sin)
        k_embed_local = (k_local * cos) + (rotate_half(k_local) * sin)
        
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return q_embed_local, k_embed_local
            
        world_size = dist.get_world_size()
        q_gather_list = [torch.empty_like(q_embed_local) for _ in range(world_size)]
        k_gather_list = [torch.empty_like(k_embed_local) for _ in range(world_size)]
        
        dist.all_gather(q_gather_list, q_embed_local.contiguous())
        dist.all_gather(k_gather_list, k_embed_local.contiguous())
        
        q_embed_global = torch.cat(q_gather_list, dim=1)
        k_embed_global = torch.cat(k_gather_list, dim=1)
        return q_embed_global, k_embed_global

    world_size = dist.get_world_size()
    B, S_local, H, D = q_local.shape
    ext = _get_ext()
    
    q_symm, k_symm, hdl_q, hdl_k = _get_symm_state(B, S_local, H, D, q_local.dtype, q_local.device)
    
    # Guarantee we do not overwrite the persistent symm_mem buffer while a peer is still pulling from the previous iteration
    hdl_q.barrier(channel=0)
    
    ext.compute_rope_local(
        q_local.contiguous(), 
        k_local.contiguous(), 
        cos_local.contiguous(), 
        sin_local.contiguous(), 
        q_symm, 
        k_symm
    )
    
    # A single barrier ensures both 'q' and 'k' RoPE computes are visible to peers before any P2P loads happen
    hdl_q.barrier(channel=0)
    
    q_global = torch.empty((B, S_local * world_size, H, D), dtype=q_local.dtype, device=q_local.device)
    k_global = torch.empty((B, S_local * world_size, H, D), dtype=k_local.dtype, device=k_local.device)
    
    ext.pull_gather(
        hdl_q.buffer_ptrs, hdl_k.buffer_ptrs,
        q_global, k_global,
        B, S_local, H, D, world_size
    )
    
    return q_global, k_global