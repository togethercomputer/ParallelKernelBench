import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Initial P2P Broadcast
// ---------------------------------------------------------------------------

__global__ void p2p_copy_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += gridDim.x * blockDim.x) {
        dst[idx] = src[idx];
    }
}

void p2p_copy(int64_t src_ptr, torch::Tensor dst, int64_t n) {
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(src_ptr);
    __nv_bfloat16* d = reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>());
    int threads = 512;
    int blocks = std::min<int>(65535, (n + threads - 1) / threads);
    p2p_copy_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(src, d, n);
}

// ---------------------------------------------------------------------------
// Math and Multimem Intrinsics
// ---------------------------------------------------------------------------

__device__ __forceinline__ float2 unpack_bf16x2(uint32_t v) {
    __nv_bfloat162 tmp = *reinterpret_cast<__nv_bfloat162*>(&v);
    return __bfloat1622float2(tmp);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float2 v) {
    __nv_bfloat162 tmp = __float22bfloat162_rn(v);
    return *reinterpret_cast<uint32_t*>(&tmp);
}

template <typename StateT>
__device__ __forceinline__ float load_state(StateT* ptr, int64_t idx);

template <>
__device__ __forceinline__ float load_state<float>(float* ptr, int64_t idx) {
    return ptr[idx];
}

template <>
__device__ __forceinline__ float load_state<__nv_bfloat16>(__nv_bfloat16* ptr, int64_t idx) {
    return __bfloat162float(ptr[idx]);
}

template <typename StateT>
__device__ __forceinline__ void store_state(StateT* ptr, int64_t idx, float val);

template <>
__device__ __forceinline__ void store_state<float>(float* ptr, int64_t idx, float val) {
    ptr[idx] = val;
}

template <>
__device__ __forceinline__ void store_state<__nv_bfloat16>(__nv_bfloat16* ptr, int64_t idx, float val) {
    ptr[idx] = __float2bfloat16(val);
}

template <typename StateT>
__device__ __forceinline__ void process_8_elements_generic(
    uint32_t g_val, uint32_t w_val,
    float* m_vals, float* v_vals,
    uint32_t& w_out,
    float inv_world_size, float lr, float beta1, float beta2, float eps, float bc1, float bc2
) {
    float2 g = unpack_bf16x2(g_val);
    g.x *= inv_world_size;
    g.y *= inv_world_size;
    
    float2 w = unpack_bf16x2(w_val);
    
    // Element 1
    float m_x = m_vals[0] * beta1 + g.x * (1.0f - beta1);
    float v_x = v_vals[0] * beta2 + g.x * g.x * (1.0f - beta2);
    m_vals[0] = m_x;
    v_vals[0] = v_x;
    w.x += (m_x / bc1) / (sqrtf(v_x / bc2) + eps) * (-lr);
    
    // Element 2
    float m_y = m_vals[1] * beta1 + g.y * (1.0f - beta1);
    float v_y = v_vals[1] * beta2 + g.y * g.y * (1.0f - beta2);
    m_vals[1] = m_y;
    v_vals[1] = v_y;
    w.y += (m_y / bc1) / (sqrtf(v_y / bc2) + eps) * (-lr);
    
    w_out = pack_bf16x2(w);
}

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr)
        : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr,
    uint32_t x, uint32_t y, uint32_t z, uint32_t w
) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w)
        : "memory");
}

// ---------------------------------------------------------------------------
// Fused kernels
// ---------------------------------------------------------------------------

template <typename StateT>
__global__ void fused_multimem_kernel(
    uint64_t multicast_grad_ptr,
    uint64_t multicast_weight_ptr,
    const __nv_bfloat16* __restrict__ local_w_part,
    StateT* __restrict__ m_part,
    StateT* __restrict__ v_part,
    int64_t part_128,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2, float inv_world_size,
    int rank
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    
    for (int64_t i = idx; i < part_128; i += stride) {
        int64_t global_idx = rank * part_128 + i;
        uint64_t* g_ptr = reinterpret_cast<uint64_t*>(multicast_grad_ptr) + global_idx * 2;
        uint64_t* w_ptr = reinterpret_cast<uint64_t*>(multicast_weight_ptr) + global_idx * 2;
        
        uint32_t g0, g1, g2, g3;
        multimem_ld_reduce_bf16x4(g_ptr, g0, g1, g2, g3);
        
        const uint32_t* local_w = reinterpret_cast<const uint32_t*>(local_w_part) + i * 4;
        uint32_t w0 = local_w[0], w1 = local_w[1], w2 = local_w[2], w3 = local_w[3];
        
        int64_t m_offset = i * 8;
        float m_vals[8], v_vals[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            m_vals[j] = load_state<StateT>(m_part, m_offset + j);
            v_vals[j] = load_state<StateT>(v_part, m_offset + j);
        }
        
        uint32_t w0_out, w1_out, w2_out, w3_out;
        process_8_elements_generic<StateT>(g0, w0, &m_vals[0], &v_vals[0], w0_out, inv_world_size, lr, beta1, beta2, eps, bc1, bc2);
        process_8_elements_generic<StateT>(g1, w1, &m_vals[2], &v_vals[2], w1_out, inv_world_size, lr, beta1, beta2, eps, bc1, bc2);
        process_8_elements_generic<StateT>(g2, w2, &m_vals[4], &v_vals[4], w2_out, inv_world_size, lr, beta1, beta2, eps, bc1, bc2);
        process_8_elements_generic<StateT>(g3, w3, &m_vals[6], &v_vals[6], w3_out, inv_world_size, lr, beta1, beta2, eps, bc1, bc2);
        
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            store_state<StateT>(m_part, m_offset + j, m_vals[j]);
            store_state<StateT>(v_part, m_offset + j, v_vals[j]);
        }
        
        multimem_st_bf16x4(w_ptr, w0_out, w1_out, w2_out, w3_out);
    }
}

template <typename StateT>
__global__ void p2p_fused_kernel(
    const uint64_t* __restrict__ peer_grad_ptrs,
    const uint64_t* __restrict__ peer_weight_ptrs,
    __nv_bfloat16* __restrict__ local_w_part,
    StateT* __restrict__ m_part,
    StateT* __restrict__ v_part,
    int64_t part,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2, float inv_world_size,
    int rank, int world_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    
    for (int64_t i = idx; i < part; i += stride) {
        int64_t global_idx = rank * part + i;
        
        float sum_g = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            __nv_bfloat16* peer_g = reinterpret_cast<__nv_bfloat16*>(peer_grad_ptrs[r]);
            sum_g += __bfloat162float(peer_g[global_idx]);
        }
        float g = sum_g * inv_world_size;
        
        float m = load_state<StateT>(m_part, i);
        float v = load_state<StateT>(v_part, i);
        
        m = m * beta1 + g * (1.0f - beta1);
        v = v * beta2 + g * g * (1.0f - beta2);
        
        store_state<StateT>(m_part, i, m);
        store_state<StateT>(v_part, i, v);
        
        float m_hat = m / bc1;
        float v_hat = v / bc2;
        
        float w = __bfloat162float(local_w_part[i]);
        w += m_hat / (sqrtf(v_hat) + eps) * (-lr);
        __nv_bfloat16 new_w = __float2bfloat16(w);
        
        for (int r = 0; r < world_size; ++r) {
            __nv_bfloat16* peer_w = reinterpret_cast<__nv_bfloat16*>(peer_weight_ptrs[r]);
            peer_w[global_idx] = new_w;
        }
    }
}

void fused_step(
    int64_t multicast_grad_ptr,
    int64_t multicast_weight_ptr,
    torch::Tensor grad_ptrs,
    torch::Tensor weight_ptrs,
    torch::Tensor weight_buf,
    torch::Tensor m_part,
    torch::Tensor v_part,
    int64_t part,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2, float inv_world_size,
    int rank, int world_size
) {
    bool use_multimem = (multicast_grad_ptr != 0) && (multicast_weight_ptr != 0) && (part % 8 == 0);
    
    int threads = 512;
    int blocks = std::min<int>(65535, (int)((part + threads - 1) / threads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const uint64_t* d_g_ptrs = reinterpret_cast<const uint64_t*>(grad_ptrs.data_ptr<int64_t>());
    const uint64_t* d_w_ptrs = reinterpret_cast<const uint64_t*>(weight_ptrs.data_ptr<int64_t>());
    __nv_bfloat16* local_w_part = reinterpret_cast<__nv_bfloat16*>(weight_buf.data_ptr<at::BFloat16>()) + rank * part;
    
    if (m_part.dtype() == torch::kFloat32) {
        float* m = m_part.data_ptr<float>();
        float* v = v_part.data_ptr<float>();
        if (use_multimem) {
            int64_t part_128 = part / 8;
            int blocks128 = std::min<int>(65535, (int)((part_128 + threads - 1) / threads));
            fused_multimem_kernel<float><<<blocks128, threads, 0, stream>>>(
                multicast_grad_ptr, multicast_weight_ptr, local_w_part, m, v, part_128,
                lr, beta1, beta2, eps, bc1, bc2, inv_world_size, rank
            );
        } else {
            p2p_fused_kernel<float><<<blocks, threads, 0, stream>>>(
                d_g_ptrs, d_w_ptrs, local_w_part, m, v, part,
                lr, beta1, beta2, eps, bc1, bc2, inv_world_size, rank, world_size
            );
        }
    } else if (m_part.dtype() == torch::kBFloat16) {
        __nv_bfloat16* m = reinterpret_cast<__nv_bfloat16*>(m_part.data_ptr<at::BFloat16>());
        __nv_bfloat16* v = reinterpret_cast<__nv_bfloat16*>(v_part.data_ptr<at::BFloat16>());
        if (use_multimem) {
            int64_t part_128 = part / 8;
            int blocks128 = std::min<int>(65535, (int)((part_128 + threads - 1) / threads));
            fused_multimem_kernel<__nv_bfloat16><<<blocks128, threads, 0, stream>>>(
                multicast_grad_ptr, multicast_weight_ptr, local_w_part, m, v, part_128,
                lr, beta1, beta2, eps, bc1, bc2, inv_world_size, rank
            );
        } else {
            p2p_fused_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                d_g_ptrs, d_w_ptrs, local_w_part, m, v, part,
                lr, beta1, beta2, eps, bc1, bc2, inv_world_size, rank, world_size
            );
        }
    } else {
        TORCH_CHECK(false, "Unsupported dtype for m_part/v_part");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("p2p_copy", &p2p_copy, "P2P symmetric memory copy");
    m.def("fused_step", &fused_step, "Fused Multimem Reduce-Scatter, Adam, All-Gather");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("zero2_fused_opt_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_resources(n: int, dtype: torch.dtype, device: torch.device):
    key = (n, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
        
    grad_buf = symm_mem.empty(n, dtype=dtype, device=device)
    hdl_g = symm_mem.rendezvous(grad_buf, dist.group.WORLD)
    grad_ptrs = torch.tensor(hdl_g.buffer_ptrs, dtype=torch.int64, device=device)
    
    weight_buf = symm_mem.empty(n, dtype=dtype, device=device)
    hdl_w = symm_mem.rendezvous(weight_buf, dist.group.WORLD)
    weight_ptrs = torch.tensor(hdl_w.buffer_ptrs, dtype=torch.int64, device=device)
    
    res = (grad_buf, hdl_g, grad_ptrs, weight_buf, hdl_w, weight_ptrs)
    _symm_cache[key] = res
    return res

def solution(
    X_local: Tensor, y_local: Tensor,
    W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor,
    exp_avg_part: Tensor, exp_avg_sq_part: Tensor,
    lr: float, beta1: float, beta2: float, eps: float, step: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    templates = [W1, b1, W2, b2]
    flat_p = _flatten_dense_tensors(templates)
    assert flat_p.dtype == torch.bfloat16, "Kernel is highly optimized for BF16 weights and gradients"
    
    n = flat_p.numel()
    part = exp_avg_part.numel()
    grad_buf, hdl_g, grad_ptrs, weight_buf, hdl_w, weight_ptrs = _get_symm_resources(n, flat_p.dtype, flat_p.device)
    
    # Fast initial broadcast to sync weights via P2P
    if rank == 0:
        weight_buf.copy_(flat_p)
    hdl_w.barrier(channel=0)
    if rank != 0:
        _get_ext().p2p_copy(weight_ptrs[0].item(), weight_buf, n)
    hdl_w.barrier(channel=1)
    
    # Establish PyTorch parameters backed directly by symmetric memory
    param_views = _unflatten_dense_tensors(weight_buf, templates)
    params = [t.detach().requires_grad_(True) for t in param_views]
    
    # Forward / Backward pass
    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()
    
    # Flatten gradient mappings seamlessly to our symmetric buffer
    flat_g = _flatten_dense_tensors([p.grad for p in params]).contiguous()
    grad_buf.copy_(flat_g)
    hdl_g.barrier(channel=0)
    
    # Prepare local partition for optimizer step
    m_part = exp_avg_part.clone()
    v_part = exp_avg_sq_part.clone()
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    inv_world_size = 1.0 / world_size
    
    multicast_grad_ptr = int(hdl_g.multicast_ptr) if hdl_g.has_multicast else 0
    multicast_weight_ptr = int(hdl_w.multicast_ptr) if hdl_w.has_multicast else 0
    
    # Dispatched fused kernel: Hardware Reduce-Scatter -> Adam -> Hardware Broadcast Update
    _get_ext().fused_step(
        multicast_grad_ptr,
        multicast_weight_ptr,
        grad_ptrs,
        weight_ptrs,
        weight_buf,
        m_part,
        v_part,
        part,
        lr, beta1, beta2, eps, bc1, bc2, inv_world_size,
        rank, world_size
    )
    
    # Barrier ensures execution finishes cleanly before tensors are copied off the persistent buffer
    hdl_w.barrier(channel=2)
    
    out_flat_p = weight_buf.clone()
    out_params = _unflatten_dense_tensors(out_flat_p, templates)
    
    return (*out_params, m_part, v_part)

__all__ = ["solution"]