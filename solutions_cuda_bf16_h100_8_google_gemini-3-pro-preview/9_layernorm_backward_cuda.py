import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ----------------------------------------------------------------------------
// Type conversions mapping to float and float2
// ----------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float(float v) { return v; }

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
template <> __device__ __forceinline__ float from_float(float v) { return v; }

template <typename T, typename T2> __device__ __forceinline__ float2 cvt_float2(T2 v);
template <> __device__ __forceinline__ float2 cvt_float2<__nv_bfloat16, __nv_bfloat162>(__nv_bfloat162 v) { return __bfloat1622float2(v); }
template <> __device__ __forceinline__ float2 cvt_float2<__half, __half2>(__half2 v) { return __half22float2(v); }

// ----------------------------------------------------------------------------
// Kernel 1: Local Fused Reduction (Scalar Fallback)
// ----------------------------------------------------------------------------
template <typename T>
__global__ void local_reduce_scalar_kernel(
    const T* __restrict__ X_hat,
    const T* __restrict__ dY,
    float* __restrict__ d_gamma_local,
    float* __restrict__ d_beta_local,
    int B, int H
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;

    int b_chunk = (B + gridDim.y - 1) / gridDim.y;
    int b_begin = blockIdx.y * b_chunk;
    int b_end = b_begin + b_chunk;
    if (b_end > B) b_end = B;

    float sg = 0.0f;
    float sb = 0.0f;

    // Load elements natively, perform precision multiply-add in FP32
    for (int b = b_begin; b < b_end; ++b) {
        float x = to_float(X_hat[b * H + h]);
        float dy = to_float(dY[b * H + h]);
        sg += dy * x;
        sb += dy;
    }

    // Safely scatter partial accumulation into the symmetric zeroed buffer
    atomicAdd(&d_gamma_local[h], sg);
    atomicAdd(&d_beta_local[h], sb);
}

// ----------------------------------------------------------------------------
// Kernel 1: Local Fused Reduction (Vectorized 8-element loads for BF16/FP16)
// ----------------------------------------------------------------------------
template <typename T, typename T2>
__global__ void local_reduce_vec8_kernel(
    const T* __restrict__ X_hat,
    const T* __restrict__ dY,
    float* __restrict__ d_gamma_local,
    float* __restrict__ d_beta_local,
    int B, int H
) {
    int h_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int h_start = h_vec * 8;
    if (h_start >= H) return;

    int b_chunk = (B + gridDim.y - 1) / gridDim.y;
    int b_begin = blockIdx.y * b_chunk;
    int b_end = b_begin + b_chunk;
    if (b_end > B) b_end = B;

    float sg[8] = {0};
    float sb[8] = {0};

    // Vectorized read mapping: 1x float4 grabs 16 bytes = 8x 16-bit elements
    const float4* x_ptr = reinterpret_cast<const float4*>(X_hat);
    const float4* dy_ptr = reinterpret_cast<const float4*>(dY);
    int vec_H = H / 8;

    for (int b = b_begin; b < b_end; ++b) {
        int idx = b * vec_H + h_vec;
        float4 x_v = x_ptr[idx];
        float4 dy_v = dy_ptr[idx];

        const T2* x_h2 = (const T2*)&x_v;
        const T2* dy_h2 = (const T2*)&dy_v;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 x_f2 = cvt_float2<T, T2>(x_h2[i]);
            float2 dy_f2 = cvt_float2<T, T2>(dy_h2[i]);
            sg[i*2 + 0] += dy_f2.x * x_f2.x;
            sg[i*2 + 1] += dy_f2.y * x_f2.y;
            sb[i*2 + 0] += dy_f2.x;
            sb[i*2 + 1] += dy_f2.y;
        }
    }

    // Scatter atomic adds blockwise chunk sums into unified global symmetric memory
    for (int i = 0; i < 8; ++i) {
        atomicAdd(&d_gamma_local[h_start + i], sg[i]);
        atomicAdd(&d_beta_local[h_start + i], sb[i]);
    }
}

// ----------------------------------------------------------------------------
// Kernel 2: NVLink Peer Pointers Cross-Rank Reduce
// ----------------------------------------------------------------------------
template <typename T>
__global__ void cross_rank_reduce_kernel(
    const long long* __restrict__ ptrs,
    T* __restrict__ out_gamma,
    T* __restrict__ out_beta,
    int world_size,
    int H
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;

    float sum_g = 0.0f;
    float sum_b = 0.0f;

    #pragma unroll
    for (int r = 0; r < world_size; ++r) {
        const float* base = (const float*)ptrs[r];
        sum_g += base[h];            // First half of buffer is gamma
        sum_b += base[H + h];        // Second half of buffer is beta
    }

    out_gamma[h] = from_float<T>(sum_g);
    out_beta[h]  = from_float<T>(sum_b);
}

// ----------------------------------------------------------------------------
// C++ PyBind Dispatchers
// ----------------------------------------------------------------------------
void launch_fused_layernorm_backward(
    torch::Tensor X_hat,
    torch::Tensor dY,
    torch::Tensor buf,
    int dtype_enum
) {
    int B = X_hat.size(0);
    int H = X_hat.size(1);

    // Spread along the batch dim if batch is large enough to saturate Hopper execution
    int blocks_y = std::min(32, (B + 127) / 128);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // 0 out symmetric partial accumulation buffer mapping before filling
    cudaMemsetAsync(buf.data_ptr<float>(), 0, 2 * H * sizeof(float), stream);

    float* d_gamma_local = buf.data_ptr<float>();
    float* d_beta_local = buf.data_ptr<float>() + H;

    if (dtype_enum == 0) { // bf16
        if (H % 8 == 0) {
            dim3 threads(128);
            dim3 blocks(H / 8 / 128 + (H / 8 % 128 != 0), blocks_y);
            local_reduce_vec8_kernel<__nv_bfloat16, __nv_bfloat162><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(X_hat.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr<at::BFloat16>()),
                d_gamma_local, d_beta_local, B, H
            );
        } else {
            dim3 threads(256);
            dim3 blocks((H + 255) / 256, blocks_y);
            local_reduce_scalar_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(X_hat.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr<at::BFloat16>()),
                d_gamma_local, d_beta_local, B, H
            );
        }
    } else if (dtype_enum == 1) { // fp16
        if (H % 8 == 0) {
            dim3 threads(128);
            dim3 blocks(H / 8 / 128 + (H / 8 % 128 != 0), blocks_y);
            local_reduce_vec8_kernel<__half, __half2><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __half*>(X_hat.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(dY.data_ptr<at::Half>()),
                d_gamma_local, d_beta_local, B, H
            );
        } else {
            dim3 threads(256);
            dim3 blocks((H + 255) / 256, blocks_y);
            local_reduce_scalar_kernel<__half><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __half*>(X_hat.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(dY.data_ptr<at::Half>()),
                d_gamma_local, d_beta_local, B, H
            );
        }
    } else { // fp32
        dim3 threads(256);
        dim3 blocks((H + 255) / 256, blocks_y);
        local_reduce_scalar_kernel<float><<<blocks, threads, 0, stream>>>(
            X_hat.data_ptr<float>(), dY.data_ptr<float>(),
            d_gamma_local, d_beta_local, B, H
        );
    }
}

void launch_cross_rank_reduce(
    torch::Tensor ptrs,
    torch::Tensor out_gamma,
    torch::Tensor out_beta,
    int world_size,
    int H,
    int dtype_enum
) {
    int threads = 256;
    int blocks = (H + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* d_ptrs = (const long long*)ptrs.data_ptr<int64_t>();

    if (dtype_enum == 0) {
        cross_rank_reduce_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            d_ptrs,
            reinterpret_cast<__nv_bfloat16*>(out_gamma.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_beta.data_ptr<at::BFloat16>()),
            world_size, H
        );
    } else if (dtype_enum == 1) {
        cross_rank_reduce_kernel<__half><<<blocks, threads, 0, stream>>>(
            d_ptrs,
            reinterpret_cast<__half*>(out_gamma.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(out_beta.data_ptr<at::Half>()),
            world_size, H
        );
    } else {
        cross_rank_reduce_kernel<float><<<blocks, threads, 0, stream>>>(
            d_ptrs, out_gamma.data_ptr<float>(), out_beta.data_ptr<float>(), world_size, H
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_layernorm_backward", &launch_fused_layernorm_backward, "Fused local layernorm backward over B");
    m.def("launch_cross_rank_reduce", &launch_cross_rank_reduce, "Cross-rank symmetric memory reduce");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_layernorm_bw_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_resources(H: int, device: torch.device):
    key = (H, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    # Pre-allocate a single FP32 symmetric buffer of size (2, H) to hold both gamma and beta local sums cleanly
    buf = symm_mem.empty((2, H), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    
    # Create the tensor of UVA pointers holding symmetrical peering routes globally
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return buf, hdl, ptrs_tensor

@torch.no_grad()
def solution(
    X_hat: torch.Tensor,
    dY: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not dist.is_initialized():
        d_beta = dY.sum(dim=0)
        d_gamma = (dY * X_hat).sum(dim=0)
        return d_gamma, d_beta

    B, H = X_hat.shape
    dtype = X_hat.dtype
    device = X_hat.device

    if dtype == torch.bfloat16:
        dtype_enum = 0
    elif dtype == torch.float16:
        dtype_enum = 1
    elif dtype == torch.float32:
        dtype_enum = 2
    else:
        # Graceful fallback for non-supported exotic dtypes
        d_beta = dY.sum(dim=0)
        d_gamma = (dY * X_hat).sum(dim=0)
        dist.all_reduce(d_beta, op=dist.ReduceOp.SUM)
        dist.all_reduce(d_gamma, op=dist.ReduceOp.SUM)
        return d_gamma, d_beta

    buf, hdl, ptrs_tensor = _get_resources(H, device)

    if dist.get_rank() == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    # Phase 1: Local Fused Elementwise + Reduction into [2, H] Float32 Symmetric Memory Buffer 
    ext.launch_fused_layernorm_backward(X_hat.contiguous(), dY.contiguous(), buf, dtype_enum)

    # Device side sync acting natively on current stream ensures chunk stores have fully landed on all peers
    hdl.barrier(channel=0)

    d_gamma = torch.empty(H, device=device, dtype=dtype)
    d_beta = torch.empty(H, device=device, dtype=dtype)
    
    # Phase 2: Direct NVLink access cross-rank sum bypassing heavy NCCL dispatch
    ext.launch_cross_rank_reduce(ptrs_tensor, d_gamma, d_beta, hdl.world_size, H, dtype_enum)

    return d_gamma, d_beta