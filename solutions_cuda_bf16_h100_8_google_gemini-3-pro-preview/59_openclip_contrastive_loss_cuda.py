from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>

// Vectorized fast copy using 128-bit loads for bandwidth optimization over NVLink
__global__ void copy_128bit_kernel(
    uint4* __restrict__ dst,
    const uint4* __restrict__ src,
    int64_t n_128bit
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n_128bit; idx += (int64_t)gridDim.x * blockDim.x) {
        dst[idx] = src[idx];
    }
}

// Fallback generic elementwise copy
template <typename scalar_t>
__global__ void copy_generic_kernel(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        dst[idx] = src[idx];
    }
}

void async_copy(
    torch::Tensor dst,
    int64_t src_ptr,
    int64_t n_elements,
    int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int element_size = dst.element_size();
    
    // Check 16-byte alignment to enable vectorized 128-bit transfer
    if ((n_elements * element_size) % 16 == 0) {
        int64_t n_128 = (n_elements * element_size) / 16;
        int threads = 256;
        int blocks = std::min<int64_t>(65535, (n_128 + threads - 1) / threads);
        copy_128bit_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<uint4*>(dst.data_ptr()),
            reinterpret_cast<const uint4*>(src_ptr),
            n_128
        );
    } else {
        int threads = 256;
        int blocks = std::min<int64_t>(65535, (n_elements + threads - 1) / threads);
        AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dst.scalar_type(), "async_copy_generic", [&] {
            copy_generic_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                dst.data_ptr<scalar_t>(),
                reinterpret_cast<const scalar_t*>(src_ptr),
                n_elements
            );
        });
    }
}

// Fused kernel calculating the stable softplus and reducing to a scalar loss.
template <typename scalar_t>
__global__ void siglip_loss_forward_kernel(
    const scalar_t* __restrict__ logits,
    float scale,
    float bias,
    float* __restrict__ loss_out,
    int batch_size,
    bool is_local
) {
    int64_t total_elements = (int64_t)batch_size * batch_size;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    for (int64_t i = idx; i < total_elements; i += (int64_t)gridDim.x * blockDim.x) {
        int r = i / batch_size;
        int c = i % batch_size;
        
        float x = static_cast<float>(logits[i]);
        x = x * scale + bias;
        
        float label = -1.0f;
        if (is_local && r == c) {
            label = 1.0f;  // Match on local batch diagonal
        }
        
        float z = label * x;
        // stable softplus to represent -logsigmoid
        float neg_z = -z;
        float max_val = neg_z > 0.0f ? neg_z : 0.0f;
        float term = expf(-fabsf(neg_z));
        float loss_val = max_val + log1pf(term);
        
        local_sum += loss_val;
    }
    
    // Warp-level reduction
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    
    // Block-level reduction through shared memory
    __shared__ float shared_sum[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x / 32)) ? shared_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane_id == 0) {
            atomicAdd(loss_out, val / static_cast<float>(batch_size));
        }
    }
}

void siglip_loss_forward(
    torch::Tensor logits,
    float scale,
    float bias,
    torch::Tensor loss_out,
    int batch_size,
    bool is_local
) {
    int threads = 256;
    int64_t total = (int64_t)batch_size * batch_size;
    int blocks = std::min<int64_t>(1024, (total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, logits.scalar_type(), "siglip_loss_forward", [&] {
        siglip_loss_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            logits.data_ptr<scalar_t>(),
            scale,
            bias,
            loss_out.data_ptr<float>(),
            batch_size,
            is_local
        );
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("async_copy", &async_copy, "Async vector copy from device memory over UVA");
    m.def("siglip_loss_forward", &siglip_loss_forward, "Fused SigLIP loss kernel");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("siglip_overlap_ext", CUDA_SRC)
    return _ext

def _init_ext(group: Optional[dist.ProcessGroup] = None):
    global _ext
    if _ext is None:
        if dist.is_initialized():
            if dist.get_rank(group) == 0:
                _get_ext()
            dist.barrier(group)
        _get_ext()

_symm_cache = {}
def _get_symm_state(shape, dtype, device, group):
    key = (shape, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(shape, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
    logit_bias: float = 0.0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    group = group or (dist.group.WORLD if dist.is_initialized() else None)
    _init_ext(group)
    
    world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    rank = dist.get_rank(group) if dist.is_initialized() else 0

    batch_size = image_features.size(0)
    n_elements = text_features.numel()
    logit_scale_f = float(logit_scale)
    logit_bias_f = float(logit_bias)
    
    loss_scalar = torch.zeros(1, dtype=torch.float32, device=image_features.device)
    
    # Fast path for single GPU deployments without ring/overlap
    if world_size == 1:
        logits0 = torch.matmul(image_features, text_features.T)
        _get_ext().siglip_loss_forward(logits0.contiguous(), logit_scale_f, logit_bias_f, loss_scalar, batch_size, True)
        return loss_scalar.to(dtype=image_features.dtype).squeeze()

    buf, hdl = _get_symm_state(text_features.shape, text_features.dtype, text_features.device, group)
    
    # Broadcast current rank's text chunk to symm memory segment
    buf.copy_(text_features)
    hdl.barrier(channel=0)

    # Establish dual buffers and streams for NVLink overlap prefetching pipeline
    bufA = torch.empty_like(text_features)
    bufB = torch.empty_like(text_features)
    stream_copy = torch.cuda.Stream()
    
    event_copy_done = [torch.cuda.Event(enable_timing=False) for _ in range(world_size)]
    event_compute_done = [torch.cuda.Event(enable_timing=False) for _ in range(world_size)]

    # Step 0: Pre-fetch for step 1 while computing local 
    p1 = (rank + 1) % world_size
    _get_ext().async_copy(bufA, int(hdl.buffer_ptrs[p1]), n_elements, stream_copy.cuda_stream)
    event_copy_done[1].record(stream_copy)

    # Step 0: Calculate initial local matching block
    logits0 = torch.matmul(image_features, buf.T)
    _get_ext().siglip_loss_forward(logits0.contiguous(), logit_scale_f, logit_bias_f, loss_scalar, batch_size, True)
    event_compute_done[0].record()

    # Step 1 ... World_size-1: Progress down ring sequence iteratively
    for s in range(1, world_size):
        curr_buf = bufA if (s % 2) != 0 else bufB
        next_buf = bufB if (s % 2) != 0 else bufA
        
        # Pre-fetch the next peer sequentially into the free secondary buffer
        if s < world_size - 1:
            next_p = (rank + s + 1) % world_size
            # Keep copy stream waiting until primary stream safely consumes this secondary buffer (from compute step s-1)
            stream_copy.wait_event(event_compute_done[s-1])
            _get_ext().async_copy(next_buf, int(hdl.buffer_ptrs[next_p]), n_elements, stream_copy.cuda_stream)
            event_copy_done[s+1].record(stream_copy)
            
        # Guarantee prefetching to curr_buf has thoroughly settled before matrix multiplication
        torch.cuda.current_stream().wait_event(event_copy_done[s])
        
        logits = torch.matmul(image_features, curr_buf.T)
        _get_ext().siglip_loss_forward(logits.contiguous(), logit_scale_f, logit_bias_f, loss_scalar, batch_size, False)
        event_compute_done[s].record()

    # Hardware block until all parallel reading passes and ensures symm cache coherence
    hdl.barrier(channel=0)

    return loss_scalar.to(dtype=image_features.dtype).squeeze()