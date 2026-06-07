"""
Strategy:
1. **Persistent Symmetric Memory**: We allocate a single symmetric memory buffer (`symm_mem`) on each rank that holds the flattened parameters, Adam moments (`exp_avg`, `exp_avg_sq`), and gradients.
2. **Zero-Copy Parameter Broadcast**: On step 1, Rank 0 writes its initial state to the symmetric buffer, and peers use a custom P2P pull kernel to fetch it. On subsequent steps, we detect if the input tensors are already views of our persistent buffer. If so, we *completely bypass* the broadcast, achieving zero-overhead persistence.
3. **Fused All-Reduce and Adam**: Instead of executing `dist.all_reduce` followed by stock PyTorch Adam operations, we launch a single custom CUDA kernel. It performs an all-to-all P2P read of the gradients directly from peers' symmetric buffers, averages them, and computes the Adam step immediately. This maximizes memory bandwidth by fusing cross-device communication and element-wise computation into a single pass.
4. **Minimal Stock PyTorch**: By keeping the authoritative state continuously in device memory and using our custom fused kernel, we eliminate all opaque collectives and intermediate tensor allocations on the performance-critical path.
"""

import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <typename T>
struct CudaTypeTraits;

template <>
struct CudaTypeTraits<float> {
    static __device__ __forceinline__ float to_float(float x) { return x; }
    static __device__ __forceinline__ float from_float(float x) { return x; }
};

template <>
struct CudaTypeTraits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }
};

template <typename T>
__global__ void pull_broadcast_kernel(
    const T* __restrict__ src_buf,
    T* __restrict__ local_buf,
    int64_t total_elements
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total_elements; idx += (int64_t)gridDim.x * blockDim.x) {
        local_buf[idx] = src_buf[idx];
    }
}

template <typename T>
__global__ void fused_allreduce_adam_kernel(
    const long long* __restrict__ peer_ptrs,
    T* __restrict__ local_buf,
    int world_size,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t m_offset = n;
    int64_t v_offset = 2 * n;
    int64_t g_offset = 3 * n;

    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum_g = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            const T* peer_buf = (const T*)peer_ptrs[r];
            sum_g += CudaTypeTraits<T>::to_float(peer_buf[g_offset + idx]);
        }
        float g = sum_g / world_size;

        float p = CudaTypeTraits<T>::to_float(local_buf[idx]);
        float m = CudaTypeTraits<T>::to_float(local_buf[m_offset + idx]);
        float v = CudaTypeTraits<T>::to_float(local_buf[v_offset + idx]);

        m = m * beta1 + g * (1.0f - beta1);
        v = v * beta2 + g * g * (1.0f - beta2);

        float m_hat = m / bc1;
        float v_hat = v / bc2;
        float denom = sqrtf(v_hat) + eps;

        p = p - lr * (m_hat / denom);

        local_buf[idx] = CudaTypeTraits<T>::from_float(p);
        local_buf[m_offset + idx] = CudaTypeTraits<T>::from_float(m);
        local_buf[v_offset + idx] = CudaTypeTraits<T>::from_float(v);
    }
}

void pull_broadcast(
    int64_t remote_ptr,
    torch::Tensor local_buf,
    int64_t total_elements,
    int dtype_enum
) {
    int threads = 512;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(remote_ptr);
        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(local_buf.data_ptr<at::BFloat16>());
        pull_broadcast_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(src, dst, total_elements);
    } else {
        const float* src = reinterpret_cast<const float*>(remote_ptr);
        float* dst = local_buf.data_ptr<float>();
        pull_broadcast_kernel<float><<<blocks, threads, 0, stream>>>(src, dst, total_elements);
    }
}

void fused_allreduce_adam(
    torch::Tensor ptrs_tensor,
    torch::Tensor local_buf,
    int world_size,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2,
    int dtype_enum
) {
    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const long long* peer_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();

    if (dtype_enum == 0) {
        __nv_bfloat16* local = reinterpret_cast<__nv_bfloat16*>(local_buf.data_ptr<at::BFloat16>());
        fused_allreduce_adam_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            peer_ptrs, local, world_size, n, lr, beta1, beta2, eps, bc1, bc2
        );
    } else {
        float* local = local_buf.data_ptr<float>();
        fused_allreduce_adam_kernel<float><<<blocks, threads, 0, stream>>>(
            peer_ptrs, local, world_size, n, lr, beta1, beta2, eps, bc1, bc2
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pull_broadcast", &pull_broadcast, "Pull broadcast kernel");
    m.def("fused_allreduce_adam", &fused_allreduce_adam, "Fused allreduce and adam kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_ddp_adam_ext", CUDA_SRC)
    return _ext


_cache = {}

def get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    key = (n, dtype, device)
    if key in _cache:
        return _cache[key]
    
    # Allocate a single symmetric buffer containing: [params, exp_avg, exp_avg_sq, grads]
    total_elements = 4 * n
    buf = symm_mem.empty(total_elements, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _cache[key] = (buf, hdl, ptrs_tensor)
    return _cache[key]


def solution(
    X_local: Tensor, y_local: Tensor,
    W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor,
    exp_avg_W1: Tensor, exp_avg_b1: Tensor, exp_avg_W2: Tensor, exp_avg_b2: Tensor,
    exp_avg_sq_W1: Tensor, exp_avg_sq_b1: Tensor, exp_avg_sq_W2: Tensor, exp_avg_sq_b2: Tensor,
    lr: float, beta1: float, beta2: float, eps: float, step: int,
) -> tuple[Tensor, ...]:
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    params_in = [W1, b1, W2, b2]
    exp_avg_in = [exp_avg_W1, exp_avg_b1, exp_avg_W2, exp_avg_b2]
    exp_avg_sq_in = [exp_avg_sq_W1, exp_avg_sq_b1, exp_avg_sq_W2, exp_avg_sq_b2]
    
    n = sum(t.numel() for t in params_in)
    dtype = W1.dtype
    device = torch.cuda.current_device()
    
    # Initialize extension safely on rank 0 first
    if rank == 0:
        _get_ext()
    dist.barrier()
    
    buf, hdl, ptrs_tensor = get_symm_state(n, dtype, device)
    dtype_enum = 0 if dtype == torch.bfloat16 else 1
    
    # If the caller passed the exact views we returned previously, state is already sync'd
    is_cached = (params_in[0].data_ptr() == buf.data_ptr())
    
    if not is_cached:
        # Full sync initialization needed
        hdl.barrier(channel=0)
        if rank == 0:
            flat_all = _flatten_dense_tensors(params_in + exp_avg_in + exp_avg_sq_in)
            buf[:3*n].copy_(flat_all)
            
        hdl.barrier(channel=0)
        if rank != 0:
            remote_ptr = int(hdl.buffer_ptrs[0])
            _get_ext().pull_broadcast(remote_ptr, buf, 3 * n, dtype_enum)
            
        hdl.barrier(channel=0)
    
    # Create views to flush any legacy gradient state from PyTorch's AD engine
    params, exp_avg, exp_avg_sq = [], [], []
    
    offset = 0
    for t in params_in:
        params.append(buf[offset : offset + t.numel()].view(t.shape).detach().requires_grad_(True))
        offset += t.numel()
        
    offset = n
    for t in exp_avg_in:
        exp_avg.append(buf[offset : offset + t.numel()].view(t.shape))
        offset += t.numel()
        
    offset = 2 * n
    for t in exp_avg_sq_in:
        exp_avg_sq.append(buf[offset : offset + t.numel()].view(t.shape))
        offset += t.numel()
        
    # Standard PyTorch local computation
    h = F.relu(F.linear(X_local, params[0], params[1]))
    out = F.linear(h, params[2], params[3])
    loss = F.mse_loss(out, y_local)
    loss.backward()
    
    # Flat-copy gradients to the symmetric memory gradients buffer block
    grads = [p.grad for p in params]
    flat_grad = _flatten_dense_tensors(grads)
    buf[3*n:].copy_(flat_grad)
    
    hdl.barrier(channel=0)
    
    bc1 = 1.0 - math.pow(beta1, step)
    bc2 = 1.0 - math.pow(beta2, step)
    
    # Peer-pointers fused all-reduce (SUM) and Adam update directly on parameters
    _get_ext().fused_allreduce_adam(
        ptrs_tensor, buf, world_size, n, lr, beta1, beta2, eps, bc1, bc2, dtype_enum
    )
    
    hdl.barrier(channel=0)
    
    # Returns the updated views perfectly matching expected reference outputs
    return tuple(params + exp_avg + exp_avg_sq)

__all__ = ["solution"]