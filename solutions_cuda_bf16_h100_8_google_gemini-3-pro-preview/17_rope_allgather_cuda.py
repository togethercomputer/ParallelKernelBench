import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <int V>
__device__ __forceinline__ void compute_rope(
    void* q_v, void* k_v, void* q_p, void* k_p, void* c_v, void* s_v,
    void* qo_v, void* ko_v, int d_idx, int half_D
) {
    __nv_bfloat16* q = (__nv_bfloat16*)q_v;
    __nv_bfloat16* k = (__nv_bfloat16*)k_v;
    __nv_bfloat16* qp = (__nv_bfloat16*)q_p;
    __nv_bfloat16* kp = (__nv_bfloat16*)k_p;
    __nv_bfloat16* c = (__nv_bfloat16*)c_v;
    __nv_bfloat16* s = (__nv_bfloat16*)s_v;
    __nv_bfloat16* qo = (__nv_bfloat16*)qo_v;
    __nv_bfloat16* ko = (__nv_bfloat16*)ko_v;
    
    #pragma unroll
    for (int i = 0; i < V; ++i) {
        float q_f = __bfloat162float(q[i]);
        float k_f = __bfloat162float(k[i]);
        float qp_f = __bfloat162float(qp[i]);
        float kp_f = __bfloat162float(kp[i]);
        float c_f = __bfloat162float(c[i]);
        float s_f = __bfloat162float(s[i]);
        
        float q_rot = (d_idx < half_D) ? (-qp_f) : (qp_f);
        float k_rot = (d_idx < half_D) ? (-kp_f) : (kp_f);
        
        qo[i] = __float2bfloat16(q_f * c_f + q_rot * s_f);
        ko[i] = __float2bfloat16(k_f * c_f + k_rot * s_f);
    }
}

template <int VEC_SIZE>
__global__ void rope_multicast_kernel(
    const uint8_t* __restrict__ q_local,
    const uint8_t* __restrict__ k_local,
    const uint8_t* __restrict__ cos_local,
    const uint8_t* __restrict__ sin_local,
    uint64_t mcast_q,
    const uint64_t* __restrict__ peer_ptrs_q,
    uint64_t mcast_k,
    const uint64_t* __restrict__ peer_ptrs_k,
    int B, int S_local, int H, int D,
    int world_size, int rank
) {
    int num_elements = B * S_local * H * D;
    int num_vecs = num_elements / VEC_SIZE;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    int half_D = D / 2;
    
    for (int i = tid; i < num_vecs; i += stride) {
        int d_idx = (i * VEC_SIZE) % D;
        int h_idx = ((i * VEC_SIZE) / D) % H;
        int s_idx = ((i * VEC_SIZE) / (D * H)) % S_local;
        int b_idx = ((i * VEC_SIZE) / (D * H * S_local));
        
        int partner_d_idx = (d_idx < half_D) ? (d_idx + half_D) : (d_idx - half_D);
        
        size_t offset_main = (size_t)i * VEC_SIZE * 2;
        size_t offset_partner = ((size_t)b_idx * S_local * H * D + (size_t)s_idx * H * D + (size_t)h_idx * D + partner_d_idx) * 2;
        size_t offset_cos_sin = ((size_t)b_idx * S_local * D + (size_t)s_idx * D + d_idx) * 2;
        
        int s_global = rank * S_local + s_idx;
        size_t offset_global = ((size_t)b_idx * (world_size * S_local) * H * D + (size_t)s_global * H * D + (size_t)h_idx * D + d_idx) * 2;
        
        if constexpr (VEC_SIZE == 8) {
            uint4 q_vec = *(uint4*)(q_local + offset_main);
            uint4 k_vec = *(uint4*)(k_local + offset_main);
            uint4 q_partner = *(uint4*)(q_local + offset_partner);
            uint4 k_partner = *(uint4*)(k_local + offset_partner);
            uint4 cos_vec = *(uint4*)(cos_local + offset_cos_sin);
            uint4 sin_vec = *(uint4*)(sin_local + offset_cos_sin);
            
            uint4 q_out, k_out;
            compute_rope<8>(&q_vec, &k_vec, &q_partner, &k_partner, &cos_vec, &sin_vec, &q_out, &k_out, d_idx, half_D);
            
            if (mcast_q != 0) {
                uint64_t addr_q = mcast_q + offset_global;
                uint64_t addr_k = mcast_k + offset_global;
                asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr_q), "r"(q_out.x), "r"(q_out.y), "r"(q_out.z), "r"(q_out.w) : "memory");
                asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(addr_k), "r"(k_out.x), "r"(k_out.y), "r"(k_out.z), "r"(k_out.w) : "memory");
            } else {
                for (int r = 0; r < world_size; ++r) {
                    *(uint4*)(peer_ptrs_q[r] + offset_global) = q_out;
                    *(uint4*)(peer_ptrs_k[r] + offset_global) = k_out;
                }
            }
        }
        else if constexpr (VEC_SIZE == 4) {
            uint2 q_vec = *(uint2*)(q_local + offset_main);
            uint2 k_vec = *(uint2*)(k_local + offset_main);
            uint2 q_partner = *(uint2*)(q_local + offset_partner);
            uint2 k_partner = *(uint2*)(k_local + offset_partner);
            uint2 cos_vec = *(uint2*)(cos_local + offset_cos_sin);
            uint2 sin_vec = *(uint2*)(sin_local + offset_cos_sin);
            
            uint2 q_out, k_out;
            compute_rope<4>(&q_vec, &k_vec, &q_partner, &k_partner, &cos_vec, &sin_vec, &q_out, &k_out, d_idx, half_D);
            
            if (mcast_q != 0) {
                uint64_t addr_q = mcast_q + offset_global;
                uint64_t addr_k = mcast_k + offset_global;
                asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1, %2};" :: "l"(addr_q), "r"(q_out.x), "r"(q_out.y) : "memory");
                asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1, %2};" :: "l"(addr_k), "r"(k_out.x), "r"(k_out.y) : "memory");
            } else {
                for (int r = 0; r < world_size; ++r) {
                    *(uint2*)(peer_ptrs_q[r] + offset_global) = q_out;
                    *(uint2*)(peer_ptrs_k[r] + offset_global) = k_out;
                }
            }
        }
        else if constexpr (VEC_SIZE == 2) {
            uint32_t q_vec = *(uint32_t*)(q_local + offset_main);
            uint32_t k_vec = *(uint32_t*)(k_local + offset_main);
            uint32_t q_partner = *(uint32_t*)(q_local + offset_partner);
            uint32_t k_partner = *(uint32_t*)(k_local + offset_partner);
            uint32_t cos_vec = *(uint32_t*)(cos_local + offset_cos_sin);
            uint32_t sin_vec = *(uint32_t*)(sin_local + offset_cos_sin);
            
            uint32_t q_out, k_out;
            compute_rope<2>(&q_vec, &k_vec, &q_partner, &k_partner, &cos_vec, &sin_vec, &q_out, &k_out, d_idx, half_D);
            
            if (mcast_q != 0) {
                uint64_t addr_q = mcast_q + offset_global;
                uint64_t addr_k = mcast_k + offset_global;
                asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" :: "l"(addr_q), "r"(q_out) : "memory");
                asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" :: "l"(addr_k), "r"(k_out) : "memory");
            } else {
                for (int r = 0; r < world_size; ++r) {
                    *(uint32_t*)(peer_ptrs_q[r] + offset_global) = q_out;
                    *(uint32_t*)(peer_ptrs_k[r] + offset_global) = k_out;
                }
            }
        }
        else if constexpr (VEC_SIZE == 1) {
            uint16_t q_vec = *(uint16_t*)(q_local + offset_main);
            uint16_t k_vec = *(uint16_t*)(k_local + offset_main);
            uint16_t q_partner = *(uint16_t*)(q_local + offset_partner);
            uint16_t k_partner = *(uint16_t*)(k_local + offset_partner);
            uint16_t cos_vec = *(uint16_t*)(cos_local + offset_cos_sin);
            uint16_t sin_vec = *(uint16_t*)(sin_local + offset_cos_sin);
            
            uint16_t q_out, k_out;
            compute_rope<1>(&q_vec, &k_vec, &q_partner, &k_partner, &cos_vec, &sin_vec, &q_out, &k_out, d_idx, half_D);
            
            for (int r = 0; r < world_size; ++r) {
                *(uint16_t*)(peer_ptrs_q[r] + offset_global) = q_out;
                *(uint16_t*)(peer_ptrs_k[r] + offset_global) = k_out;
            }
        }
    }
}

void launch_rope_multicast(
    torch::Tensor q_local,
    torch::Tensor k_local,
    torch::Tensor cos_local,
    torch::Tensor sin_local,
    uint64_t mcast_q,
    torch::Tensor peer_ptrs_q_tensor,
    uint64_t mcast_k,
    torch::Tensor peer_ptrs_k_tensor,
    int world_size,
    int rank
) {
    int B = q_local.size(0);
    int S_local = q_local.size(1);
    int H = q_local.size(2);
    int D = q_local.size(3);
    
    const uint64_t* peer_ptrs_q = (const uint64_t*)peer_ptrs_q_tensor.data_ptr<int64_t>();
    const uint64_t* peer_ptrs_k = (const uint64_t*)peer_ptrs_k_tensor.data_ptr<int64_t>();
    
    int threads = 256;
    int num_elements = B * S_local * H * D;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (D % 16 == 0) {
        int blocks = (num_elements / 8 + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        rope_multicast_kernel<8><<<blocks, threads, 0, stream>>>(
            (const uint8_t*)q_local.data_ptr(), (const uint8_t*)k_local.data_ptr(),
            (const uint8_t*)cos_local.data_ptr(), (const uint8_t*)sin_local.data_ptr(),
            mcast_q, peer_ptrs_q, mcast_k, peer_ptrs_k,
            B, S_local, H, D, world_size, rank
        );
    } else if (D % 8 == 0) {
        int blocks = (num_elements / 4 + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        rope_multicast_kernel<4><<<blocks, threads, 0, stream>>>(
            (const uint8_t*)q_local.data_ptr(), (const uint8_t*)k_local.data_ptr(),
            (const uint8_t*)cos_local.data_ptr(), (const uint8_t*)sin_local.data_ptr(),
            mcast_q, peer_ptrs_q, mcast_k, peer_ptrs_k,
            B, S_local, H, D, world_size, rank
        );
    } else if (D % 4 == 0) {
        int blocks = (num_elements / 2 + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        rope_multicast_kernel<2><<<blocks, threads, 0, stream>>>(
            (const uint8_t*)q_local.data_ptr(), (const uint8_t*)k_local.data_ptr(),
            (const uint8_t*)cos_local.data_ptr(), (const uint8_t*)sin_local.data_ptr(),
            mcast_q, peer_ptrs_q, mcast_k, peer_ptrs_k,
            B, S_local, H, D, world_size, rank
        );
    } else {
        int blocks = (num_elements + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        rope_multicast_kernel<1><<<blocks, threads, 0, stream>>>(
            (const uint8_t*)q_local.data_ptr(), (const uint8_t*)k_local.data_ptr(),
            (const uint8_t*)cos_local.data_ptr(), (const uint8_t*)sin_local.data_ptr(),
            mcast_q, peer_ptrs_q, mcast_k, peer_ptrs_k,
            B, S_local, H, D, world_size, rank
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_rope_multicast", &launch_rope_multicast, "Fused RoPE and Multicast All-Gather");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("rope_multicast_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(shape, dtype, device):
    key = (shape, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
        
    buf_q = symm_mem.empty(shape, dtype=dtype, device=device)
    hdl_q = symm_mem.rendezvous(buf_q, dist.group.WORLD)
    
    buf_k = symm_mem.empty(shape, dtype=dtype, device=device)
    hdl_k = symm_mem.rendezvous(buf_k, dist.group.WORLD)
    
    ptrs_q = torch.tensor(hdl_q.buffer_ptrs, device=device, dtype=torch.int64)
    ptrs_k = torch.tensor(hdl_k.buffer_ptrs, device=device, dtype=torch.int64)
    
    mcast_q = int(hdl_q.multicast_ptr) if hasattr(hdl_q, 'multicast_ptr') and hdl_q.multicast_ptr else 0
    mcast_k = int(hdl_k.multicast_ptr) if hasattr(hdl_k, 'multicast_ptr') and hdl_k.multicast_ptr else 0
    
    res = (buf_q, hdl_q, ptrs_q, mcast_q, buf_k, hdl_k, ptrs_k, mcast_k)
    _symm_cache[key] = res
    return res

def rotate_half(x: torch.Tensor) -> torch.Tensor:
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
    
    if not dist.is_initialized():
        cos = cos_local.unsqueeze(2)
        sin = sin_local.unsqueeze(2)
        q_embed_local = (q_local * cos) + (rotate_half(q_local) * sin)
        k_embed_local = (k_local * cos) + (rotate_half(k_local) * sin)
        return q_embed_local, k_embed_local
        
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    
    B, S_local, H, D = q_local.shape
    global_shape = (B, S_local * world_size, H, D)
    
    buf_q, hdl_q, ptrs_q, mcast_q, buf_k, hdl_k, ptrs_k, mcast_k = _get_symm_state(global_shape, q_local.dtype, q_local.device)
    
    # Isolate cross-GPU traffic dependencies via lightweight symmetric memory device barriers
    hdl_q.barrier(channel=0)
    hdl_k.barrier(channel=0)
    
    _get_ext().launch_rope_multicast(
        q_local.contiguous(), k_local.contiguous(),
        cos_local.contiguous(), sin_local.contiguous(),
        mcast_q, ptrs_q, mcast_k, ptrs_k,
        world_size, rank
    )
    
    hdl_q.barrier(channel=0)
    hdl_k.barrier(channel=0)
    
    return buf_q.clone(), buf_k.clone()