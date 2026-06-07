"""
Strategy:
For scalar (or very small) metrics like loss, the critical path is latency—specifically PyTorch elementwise kernel launches and NCCL host syncs. 
By fusing the forward pass (NaN checks, multiplication, inter-GPU reduction, division) and the backward pass into a single custom UVA CUDA kernel, we eliminate all PyTorch overhead and NCCL host roundtrips.
We use `torch.distributed._symmetric_memory` to allocate symmetric buffers and custom signal pads. A single kernel reads inputs, exchanges data directly via NVLink using `atom.global.release.sys`/`acquire.sys` flip-flop barriers, and writes the final outputs. This maximizes compute-communication overlap by keeping the entire operation on-device without returning to the host.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple, Optional
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Flip-flop barriers using system-wide acquire/release atomics for memory consistency across NVLink
__device__ __forceinline__ void send_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp)
            : "l"(addr)
            : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp)
            : "l"(addr)
            : "memory");
    } while (tmp != 1u);
}

__device__ void blockwise_barrier(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size
) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) {
        return;
    }
    uint32_t* remote_pad = reinterpret_cast<uint32_t*>(signal_pad_ptrs[flat_tid]);
    uint32_t* send_addr = &remote_pad[rank];

    uint32_t* local_pad = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank]);
    uint32_t* wait_addr = &local_pad[flat_tid];

    send_signal(send_addr);
    wait_signal(wait_addr);
}

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

template <>
struct CudaTypeTraits<__half> {
    static __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
    static __device__ __forceinline__ __half from_float(float x) { return __float2half(x); }
};

template <typename T>
__global__ void fused_loss_fw_bw_kernel(
    const T* __restrict__ loss,
    const T* __restrict__ local_valid_tokens,
    const T* __restrict__ global_valid_tokens,
    const T* __restrict__ grad_normalized_loss,
    const T* __restrict__ grad_loss_sum,
    T* __restrict__ normalized_loss_out,
    T* __restrict__ loss_sum_out,
    T* __restrict__ grad_loss_out,
    const uint64_t* __restrict__ symm_buffer_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size,
    int numel
) {
    float lvt = CudaTypeTraits<T>::to_float(local_valid_tokens[0]);
    float gvt = CudaTypeTraits<T>::to_float(global_valid_tokens[0]);

    for (int idx = threadIdx.x; idx < numel; idx += blockDim.x) {
        float l = CudaTypeTraits<T>::to_float(loss[idx]);
        if (lvt == 0.0f) {
            if (isnan(l) || isinf(l)) l = 0.0f;
        }
        float l_sum = l * lvt;
        float* my_symm_buf = reinterpret_cast<float*>(symm_buffer_ptrs[rank]);
        my_symm_buf[idx] = l_sum;
    }

    __syncthreads();
    blockwise_barrier(signal_pad_ptrs, rank, world_size);
    __syncthreads();

    for (int idx = threadIdx.x; idx < numel; idx += blockDim.x) {
        float total_l_sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            float* peer_symm_buf = reinterpret_cast<float*>(symm_buffer_ptrs[r]);
            total_l_sum += peer_symm_buf[idx];
        }

        float norm_loss = total_l_sum / gvt;
        float gnl = CudaTypeTraits<T>::to_float(grad_normalized_loss[idx]);
        float gls = 0.0f;
        if (grad_loss_sum != nullptr) {
            gls = CudaTypeTraits<T>::to_float(grad_loss_sum[idx]);
        }

        float grad_from_norm = gnl * lvt / gvt;
        float grad_from_sum = gls * lvt;
        float grad_l = grad_from_norm + grad_from_sum;

        normalized_loss_out[idx] = CudaTypeTraits<T>::from_float(norm_loss);
        loss_sum_out[idx] = CudaTypeTraits<T>::from_float(total_l_sum);
        grad_loss_out[idx] = CudaTypeTraits<T>::from_float(grad_l);
    }

    __syncthreads();
    blockwise_barrier(signal_pad_ptrs, rank, world_size);
}

void launch_fused_loss(
    torch::Tensor loss,
    torch::Tensor local_valid_tokens,
    torch::Tensor global_valid_tokens,
    torch::Tensor grad_normalized_loss,
    c10::optional<torch::Tensor> grad_loss_sum,
    torch::Tensor normalized_loss_out,
    torch::Tensor loss_sum_out,
    torch::Tensor grad_loss_out,
    torch::Tensor symm_buffer_ptrs,
    torch::Tensor signal_pad_ptrs,
    int rank,
    int world_size
) {
    int numel = loss.numel();
    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (loss.dtype() == torch::kBFloat16) {
        fused_loss_fw_bw_kernel<__nv_bfloat16><<<1, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(loss.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(local_valid_tokens.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(global_valid_tokens.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_normalized_loss.data_ptr<at::BFloat16>()),
            grad_loss_sum.has_value() ? reinterpret_cast<const __nv_bfloat16*>(grad_loss_sum.value().data_ptr<at::BFloat16>()) : nullptr,
            reinterpret_cast<__nv_bfloat16*>(normalized_loss_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(loss_sum_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(grad_loss_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const uint64_t*>(symm_buffer_ptrs.data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>()),
            rank,
            world_size,
            numel
        );
    } else if (loss.dtype() == torch::kFloat32) {
        fused_loss_fw_bw_kernel<float><<<1, threads, 0, stream>>>(
            loss.data_ptr<float>(),
            local_valid_tokens.data_ptr<float>(),
            global_valid_tokens.data_ptr<float>(),
            grad_normalized_loss.data_ptr<float>(),
            grad_loss_sum.has_value() ? grad_loss_sum.value().data_ptr<float>() : nullptr,
            normalized_loss_out.data_ptr<float>(),
            loss_sum_out.data_ptr<float>(),
            grad_loss_out.data_ptr<float>(),
            reinterpret_cast<const uint64_t*>(symm_buffer_ptrs.data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>()),
            rank,
            world_size,
            numel
        );
    } else if (loss.dtype() == torch::kFloat16) {
        fused_loss_fw_bw_kernel<__half><<<1, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(loss.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(local_valid_tokens.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(global_valid_tokens.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad_normalized_loss.data_ptr<at::Half>()),
            grad_loss_sum.has_value() ? reinterpret_cast<const __half*>(grad_loss_sum.value().data_ptr<at::Half>()) : nullptr,
            reinterpret_cast<__half*>(normalized_loss_out.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(loss_sum_out.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(grad_loss_out.data_ptr<at::Half>()),
            reinterpret_cast<const uint64_t*>(symm_buffer_ptrs.data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>()),
            rank,
            world_size,
            numel
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only float32, float16, and bfloat16 are supported");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_loss", &launch_fused_loss, "Fused loss fw bw with symm_mem allreduce");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        if dist.is_initialized():
            if dist.get_rank() == 0:
                _ext = compile_cuda_extension("fused_loss_acc", CUDA_SRC)
            dist.barrier()
            if dist.get_rank() != 0:
                _ext = compile_cuda_extension("fused_loss_acc", CUDA_SRC)
        else:
            _ext = compile_cuda_extension("fused_loss_acc", CUDA_SRC)
    return _ext

_resource_cache = {}

def _get_resources(numel: int, device: torch.device, world_size: int):
    key = (numel, device, world_size)
    if key in _resource_cache:
        return _resource_cache[key]
        
    buf = symm_mem.empty(numel, dtype=torch.float32, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    buf_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    pad = symm_mem.empty((world_size,), dtype=torch.int32, device=device)
    pad_hdl = symm_mem.rendezvous(pad, dist.group.WORLD)
    pad.zero_()
    pad_ptrs = torch.tensor(pad_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    # Guarantee that pad zeroes are globally visible before they are actively requested by kernels
    dist.barrier()
    
    res = (buf, buf_ptrs, pad_ptrs)
    _resource_cache[key] = res
    return res

@torch.no_grad()
def solution(
    loss: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    global_valid_tokens: torch.Tensor,
    grad_normalized_loss: torch.Tensor,
    grad_loss_sum: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    if not dist.is_initialized() or dist.get_world_size() == 1:
        if local_valid_tokens.item() == 0:
            loss = torch.nan_to_num(loss)
        loss_sum = loss * local_valid_tokens
        normalized_loss = loss_sum / global_valid_tokens
        
        grad_from_norm = grad_normalized_loss * local_valid_tokens / global_valid_tokens
        if grad_loss_sum is not None:
            grad_from_sum = grad_loss_sum * local_valid_tokens
        else:
            grad_from_sum = torch.zeros_like(grad_normalized_loss)
            
        return normalized_loss, loss_sum, grad_from_norm + grad_from_sum
        
    loss = loss.contiguous()
    local_valid_tokens = local_valid_tokens.contiguous()
    global_valid_tokens = global_valid_tokens.contiguous()
    grad_normalized_loss = grad_normalized_loss.contiguous()
    if grad_loss_sum is not None:
        grad_loss_sum = grad_loss_sum.contiguous()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    numel = loss.numel()
    
    ext = _get_ext()
    buf, buf_ptrs, pad_ptrs = _get_resources(numel, loss.device, world_size)
    
    normalized_loss_out = torch.empty_like(loss)
    loss_sum_out = torch.empty_like(loss)
    grad_loss_out = torch.empty_like(loss)
    
    ext.launch_fused_loss(
        loss,
        local_valid_tokens,
        global_valid_tokens,
        grad_normalized_loss,
        grad_loss_sum,
        normalized_loss_out,
        loss_sum_out,
        grad_loss_out,
        buf_ptrs,
        pad_ptrs,
        rank,
        world_size
    )
    
    return normalized_loss_out, loss_sum_out, grad_loss_out