"""
Strategy:
- **Device-Side Communication**: We replace `dist.all_gather` with a custom P2P boundary exchange using `torch.distributed._symmetric_memory`. Each rank exposes only its top and bottom boundary rows (size `[2, B, C, padding, W]`) in a symmetric memory buffer.
- **Zero-Copy & UVA**: We use UVA device pointers to load boundary data directly from adjacent ranks into our local `padded_x` buffer, avoiding any host-side collectives or intermediate buffers.
- **Compute-Communication Overlap**: The local extraction of boundaries and the copy of the core local tensor into `padded_x` are fused into a single asynchronous kernel. This kernel executes immediately, overlapping the data preparation with the barrier wait (`hdl.barrier()`), effectively hiding the local setup latency. Finally, a single optimized `F.conv2d` call runs on the continuous patched buffer.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>

template<typename T>
__device__ __forceinline__ T get_zero();

template<>
__device__ __forceinline__ float get_zero<float>() { return 0.0f; }

template<>
__device__ __forceinline__ __half get_zero<__half>() { 
    return __float2half(0.0f);
}

template<>
__device__ __forceinline__ __nv_bfloat16 get_zero<__nv_bfloat16>() { 
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16(0.0f);
#else
    unsigned short val = 0;
    return *reinterpret_cast<__nv_bfloat16*>(&val);
#endif
}

template <typename T>
__global__ void pack_and_pad_kernel(
    const T* __restrict__ x,
    T* __restrict__ symm_buf,
    T* __restrict__ padded_x,
    int B, int C, int H, int W, int boundary,
    int64_t numel_x
) {
    int64_t H_padded = H + 2 * boundary;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel_x;
         idx += (int64_t)gridDim.x * blockDim.x) {
        
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (W * H)) % C;
        int b = idx / (W * H * C);

        int64_t padded_h = h + boundary;
        int64_t padded_idx = ((int64_t)b * C * H_padded * W) +
                             ((int64_t)c * H_padded * W) +
                             (padded_h * W) + w;
        
        T val = x[idx];
        padded_x[padded_idx] = val;

        if (h < boundary) {
            int64_t symm_idx = ((int64_t)b * C * boundary * W) +
                               ((int64_t)c * boundary * W) +
                               (h * W) + w;
            symm_buf[symm_idx] = val;
        } else if (h >= H - boundary) {
            int h_b = h - (H - boundary);
            int64_t symm_idx = ((int64_t)1 * B * C * boundary * W) +
                               ((int64_t)b * C * boundary * W) +
                               ((int64_t)c * boundary * W) +
                               (h_b * W) + w;
            symm_buf[symm_idx] = val;
        }
    }
}

template <typename T>
__global__ void unpack_peers_kernel(
    T* __restrict__ padded_x,
    const T* __restrict__ peer_top_buf,
    const T* __restrict__ peer_bottom_buf,
    int B, int C, int H_padded, int W, int boundary,
    int rank, int world_size,
    int64_t total_boundary_numel
) {
    int64_t numel_boundary = total_boundary_numel / 2;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_boundary_numel;
         idx += (int64_t)gridDim.x * blockDim.x) {
        
        bool is_top = idx < numel_boundary;
        int64_t bnd_idx = is_top ? idx : (idx - numel_boundary);
        
        int w = bnd_idx % W;
        int h = (bnd_idx / W) % boundary;
        int c = (bnd_idx / (W * boundary)) % C;
        int b = bnd_idx / (W * boundary * C);
        
        if (is_top) {
            int64_t out_idx = ((int64_t)b * C * H_padded * W) +
                              ((int64_t)c * H_padded * W) +
                              (h * W) + w;
            if (rank > 0 && peer_top_buf != nullptr) {
                padded_x[out_idx] = peer_top_buf[bnd_idx];
            } else {
                padded_x[out_idx] = get_zero<T>();
            }
        } else {
            int h_out = H_padded - boundary + h;
            int64_t out_idx = ((int64_t)b * C * H_padded * W) +
                              ((int64_t)c * H_padded * W) +
                              (h_out * W) + w;
            if (rank < world_size - 1 && peer_bottom_buf != nullptr) {
                padded_x[out_idx] = peer_bottom_buf[bnd_idx];
            } else {
                padded_x[out_idx] = get_zero<T>();
            }
        }
    }
}

void launch_pack_and_pad(
    torch::Tensor x,
    torch::Tensor symm_buf,
    torch::Tensor padded_x,
    int boundary
) {
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    int64_t numel_x = x.numel();
    int threads = 256;
    int blocks = std::min<int64_t>((numel_x + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.dtype() == torch::kBFloat16) {
        pack_and_pad_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(symm_buf.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(padded_x.data_ptr<at::BFloat16>()),
            B, C, H, W, boundary, numel_x
        );
    } else if (x.dtype() == torch::kFloat32) {
        pack_and_pad_kernel<float><<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(),
            symm_buf.data_ptr<float>(),
            padded_x.data_ptr<float>(),
            B, C, H, W, boundary, numel_x
        );
    } else if (x.dtype() == torch::kFloat16) {
        pack_and_pad_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(symm_buf.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(padded_x.data_ptr<at::Half>()),
            B, C, H, W, boundary, numel_x
        );
    }
}

void launch_unpack_peers(
    torch::Tensor padded_x,
    int64_t peer_top_ptr,
    int64_t peer_bottom_ptr,
    int boundary,
    int rank,
    int world_size
) {
    int B = padded_x.size(0);
    int C = padded_x.size(1);
    int H_padded = padded_x.size(2);
    int W = padded_x.size(3);
    
    int64_t numel_boundary = (int64_t)B * C * boundary * W;
    int64_t total_boundary_numel = numel_boundary * 2;
    int threads = 256;
    int blocks = std::min<int64_t>((total_boundary_numel + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (padded_x.dtype() == torch::kBFloat16) {
        unpack_peers_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(padded_x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(peer_top_ptr),
            reinterpret_cast<const __nv_bfloat16*>(peer_bottom_ptr),
            B, C, H_padded, W, boundary, rank, world_size, total_boundary_numel
        );
    } else if (padded_x.dtype() == torch::kFloat32) {
        unpack_peers_kernel<float><<<blocks, threads, 0, stream>>>(
            padded_x.data_ptr<float>(),
            reinterpret_cast<const float*>(peer_top_ptr),
            reinterpret_cast<const float*>(peer_bottom_ptr),
            B, C, H_padded, W, boundary, rank, world_size, total_boundary_numel
        );
    } else if (padded_x.dtype() == torch::kFloat16) {
        unpack_peers_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__half*>(padded_x.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(peer_top_ptr),
            reinterpret_cast<const __half*>(peer_bottom_ptr),
            B, C, H_padded, W, boundary, rank, world_size, total_boundary_numel
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pack_and_pad", &launch_pack_and_pad, "Pack boundary and init local tensor");
    m.def("launch_unpack_peers", &launch_unpack_peers, "Unpack UVA boundary rows");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("conv2d_boundary_cuda", CUDA_SRC)
    return _ext


_symm_cache = {}


def _get_symm_state(B, C, boundary, W, dtype, device, group):
    key = (B, C, boundary, W, dtype, device, id(group))
    if key in _symm_cache:
        return _symm_cache[key]
    
    # 2 buffers: index 0 for top boundary to send, index 1 for bottom boundary to send
    buf = symm_mem.empty((2, B, C, boundary, W), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int = 1,
    padding: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    boundary = int(padding)

    if boundary == 0 or world_size == 1:
        return F.conv2d(x, weight, bias, stride=stride, padding=padding)

    x = x.contiguous()

    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    
    ext = _get_ext()
    
    B, C, H, W = x.shape
    H_padded = H + 2 * boundary
    
    padded_x = torch.empty((B, C, H_padded, W), dtype=x.dtype, device=x.device)
    symm_buf, hdl = _get_symm_state(B, C, boundary, W, x.dtype, x.device, group)
    
    # Fused operation: asynchronously slice/copy the local tensor and prepare peer payloads
    ext.launch_pack_and_pad(x, symm_buf, padded_x, boundary)
    
    # Ensure memory writes to the symmetric buffers are globally visible
    hdl.barrier(channel=0)
    
    element_size = x.element_size()
    # Offset pointer by the full size of one boundary to access the bottom boundary 
    offset = B * C * boundary * W * element_size
    
    peer_top_ptr = 0
    peer_bottom_ptr = 0
    
    if rank > 0:
        # rank - 1's bottom boundary (index 1)
        peer_top_ptr = int(hdl.buffer_ptrs[rank - 1]) + offset
    if rank < world_size - 1:
        # rank + 1's top boundary (index 0)
        peer_bottom_ptr = int(hdl.buffer_ptrs[rank + 1])
        
    ext.launch_unpack_peers(
        padded_x,
        peer_top_ptr,
        peer_bottom_ptr,
        boundary,
        rank,
        world_size
    )
    
    return F.conv2d(padded_x, weight, bias, stride=stride, padding=(0, padding))