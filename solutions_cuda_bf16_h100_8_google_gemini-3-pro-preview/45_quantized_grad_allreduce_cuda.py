import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <pybind11/stl.h>

struct PeerPtrs {
    const float* ptrs[8];
};

__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory"
        );
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory"
        );
    } while (tmp != 1u);
}

template<typename T>
struct CudaTypeTraits;

template<>
struct CudaTypeTraits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }
};

template<>
struct CudaTypeTraits<float> {
    static __device__ __forceinline__ float to_float(float x) { return x; }
    static __device__ __forceinline__ float from_float(float x) { return x; }
};

template<typename T>
__global__ void fused_quant_dequant_reduce_kernel(
    const T* __restrict__ input,
    float* __restrict__ symm_buf,
    PeerPtrs peer_ptrs,
    T* __restrict__ out,
    int64_t n,
    int64_t block_size,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int world_size,
    int rank,
    float inv_world_size
) {
    int64_t nb = (n + block_size - 1) / block_size;
    int bid = blockIdx.x;
    
    // Persistent threadblock loop over chunks
    for (int64_t chunk_idx = bid; chunk_idx < nb; chunk_idx += gridDim.x) {
        int64_t start_idx = chunk_idx * block_size;
        int64_t end_idx = start_idx + block_size;
        if (end_idx > n) end_idx = n;
        
        // --- 1. Compute Max for Chunk ---
        float local_max = 0.0f;
        for (int64_t i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
            float val = CudaTypeTraits<T>::to_float(input[i]);
            val = fabsf(val);
            if (val > local_max) local_max = val;
        }

        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(mask, local_max, offset));
        }

        __shared__ float warp_max[32];
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;

        if (lane_id == 0) warp_max[warp_id] = local_max;
        __syncthreads();

        float block_max = 0.0f;
        if (warp_id == 0) {
            block_max = (lane_id < (blockDim.x / 32)) ? warp_max[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                block_max = fmaxf(block_max, __shfl_down_sync(mask, block_max, offset));
            }
            if (lane_id == 0) {
                if (block_max < 1e-8f) block_max = 1e-8f;
                warp_max[0] = block_max / 127.0f; 
            }
        }
        __syncthreads();

        // --- 2. Quantize & Dequantize ---
        float scale = warp_max[0];
        float inv_scale = 1.0f / scale;

        for (int64_t i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
            float val = CudaTypeTraits<T>::to_float(input[i]);
            float q = roundf(val * inv_scale);
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            symm_buf[i] = q * scale;
        }
        __syncthreads();

        // --- 3. Chunk-level Device Barrier ---
        // channel_id avoids overlapping channel 0 which is used for global PyTorch barriers.
        uint64_t channel_id = 1 + (chunk_idx % 65535);
        if (threadIdx.x < world_size) {
            unsigned int flat_tid = threadIdx.x;
            uint64_t local_base = signal_pad_ptrs[rank];
            uint64_t remote_base = signal_pad_ptrs[flat_tid];
            
            uint32_t* send_addr = reinterpret_cast<uint32_t*>(
                remote_base + channel_id * (uint64_t)world_size * 4 + (uint64_t)rank * 4);
            uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
                local_base + channel_id * (uint64_t)world_size * 4 + (uint64_t)flat_tid * 4);
                
            send_signal_acq_rel(send_addr);
            wait_signal_acq_rel(wait_addr);
        }
        __syncthreads();

        // --- 4. Peer-to-Peer FP32 Reduce ---
        for (int64_t i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
            float sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r < world_size) {
                    sum += peer_ptrs.ptrs[r][i];
                }
            }
            sum *= inv_world_size;
            out[i] = CudaTypeTraits<T>::from_float(sum);
        }
        __syncthreads();
    }
}

void launch_fused_quant_reduce(
    torch::Tensor input,
    torch::Tensor symm_buf,
    std::vector<int64_t> ptrs,
    torch::Tensor out,
    int64_t block_size,
    torch::Tensor signal_pad_ptrs_tensor,
    int world_size,
    int rank
) {
    TORCH_CHECK(world_size <= 8, "world_size > 8 is not supported");
    
    int64_t n = input.numel();
    int64_t nb = (n + block_size - 1) / block_size;
    
    int threads = 256;
    int blocks = nb < 132 ? nb : 132;

    PeerPtrs peer_ptrs;
    for (int i = 0; i < world_size; i++) {
        peer_ptrs.ptrs[i] = reinterpret_cast<const float*>(ptrs[i]);
    }

    const uint64_t* d_signal = reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    float inv_world_size = 1.0f / world_size;

    if (input.dtype() == torch::kBFloat16) {
        fused_quant_dequant_reduce_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            symm_buf.data_ptr<float>(),
            peer_ptrs,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            n,
            block_size,
            d_signal,
            world_size,
            rank,
            inv_world_size
        );
    } else if (input.dtype() == torch::kFloat32) {
        fused_quant_dequant_reduce_kernel<float><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            symm_buf.data_ptr<float>(),
            peer_ptrs,
            out.data_ptr<float>(),
            n,
            block_size,
            d_signal,
            world_size,
            rank,
            inv_world_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only BF16 and FP32 are supported.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_quant_reduce", &launch_fused_quant_reduce, "Fused quant-dequant and chunked P2P reduce");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_quant_reduce_ext", CUDA_SRC)
    return _ext

_resource_cache = {}

def _get_resources(n: int, dtype: torch.dtype, device: torch.device):
    key = (n, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]

    # symm_buf must be explicitly float32 to perfectly align with reference FP32 intermediate accumulation.
    buf = symm_mem.empty(n, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    out = torch.empty(n, device=device, dtype=dtype)
    
    res = (buf, hdl, out)
    _resource_cache[key] = res
    return res

@torch.no_grad()
def solution(
    flat_grad: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert block_size >= 1

    world_size = dist.get_world_size()
    assert world_size <= 8, "world_size > 8 is not supported by this NVLink optimized kernel"
    
    orig_shape = flat_grad.shape
    x = flat_grad.contiguous().view(-1)
    n = x.numel()
    dtype = x.dtype

    if n == 0:
        return flat_grad.clone()

    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    buf, hdl, out = _get_resources(n, dtype, x.device)

    # Issue a global barrier prior to starting the persistent kernel to prevent
    # ranks racing ahead and overwriting the symmetric buffer before slow peers read it.
    hdl.barrier(channel=0)

    ext.launch_fused_quant_reduce(
        x,
        buf,
        hdl.buffer_ptrs,
        out,
        block_size,
        hdl.signal_pad_ptrs_dev,
        world_size,
        dist.get_rank()
    )

    return out.view(orig_shape)

__all__ = ["solution"]