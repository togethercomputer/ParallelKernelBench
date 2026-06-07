"""
Strategy:
1.  **Eliminate intermediate reshaping and padding**: The stock PyTorch path performs expensive contiguous copies to carve out boundary chunks, pack them into an overlapped buffer, run `F.conv1d`, and then permute/reshape the output back into zigzag format. We replace this entire sequence with a single fused causal depthwise 1D convolution kernel.
2.  **Device-Side Communication (Symmetric Memory & UVA)**: We use `torch.distributed._symmetric_memory` to allocate a dedicated device buffer for the halo bounds (`chunk_a` and `chunk_b`). We use a lightweight kernel to pack the boundary slices. After a blockwise barrier, each rank fetches its required overlapping regions directly from its peers' symmetric memory via direct UVA pointers.
3.  **Compute-Communication Overlap**: The causal depthwise convolution kernel is parallelized into blocks over sequence chunks (tiles of size `T=1024`). Only the threads working on the leading boundary of a sequence chunk actually dereference the peer UVA pointer. The latency of these remote loads is seamlessly hidden by the GPU warp scheduler automatically executing warps from the vast majority of other sequence tiles that perform strictly local HBM loads. 
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Pack the boundary halo overlapping parts (chunk_a and chunk_b) 
// into a contiguous symmetric memory buffer.
// ---------------------------------------------------------------------------
__global__ void pack_symm_buf_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ symm_buf,
    int B, int H, int S, int K
) {
    int64_t total_elements = (int64_t)2 * B * H * (K - 1);
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int k_idx = idx % (K - 1);
        int64_t temp = idx / (K - 1);
        int bh = temp % (B * H);
        int chunk = temp / (B * H);

        int x_idx;
        if (chunk == 0) {
            x_idx = S - (K - 1) + k_idx;
        } else {
            x_idx = 2 * S - (K - 1) + k_idx;
        }
        symm_buf[idx] = x[bh * (2 * S) + x_idx];
    }
}

// ---------------------------------------------------------------------------
// Fused causal depthwise convolution over the zigzag layout.
// Automatically pulls halo padding from peer UVA pointers (ptr_prev_a, ptr_next_b)
// and writes directly into the complex output zigzag permuted shape.
// ---------------------------------------------------------------------------
__global__ void causal_depthwise_conv1d_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ ptr_prev_a,
    const __nv_bfloat16* __restrict__ ptr_next_b,
    __nv_bfloat16* __restrict__ out,
    int B, int H, int S, int K, int T, int grid_x
) {
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* smem_in = smem;                         // Size: T + K - 1
    __nv_bfloat16* smem_w = smem + (T + K - 1);            // Size: K

    // Use a 1D grid to circumvent the 65535 limit on gridDim.y
    int64_t block_idx = blockIdx.x;
    int chunk = block_idx % 2;
    int64_t temp = block_idx / 2;
    int bh = temp % (B * H);
    int tile_idx = temp / (B * H);

    int h = bh % H;
    int tid = threadIdx.x;

    int out_start = tile_idx * T;
    if (out_start >= S) return;
    int out_end = out_start + T;
    if (out_end > S) out_end = S;
    int out_len = out_end - out_start;

    // Load kernel weights into Shared Memory
    for (int i = tid; i < K; i += blockDim.x) {
        smem_w[i] = weight[h * K + i];
    }

    // Load input sequence + padding block into Shared Memory
    int load_len = out_len + K - 1;
    for (int i = tid; i < load_len; i += blockDim.x) {
        int logical_idx = out_start + i;
        __nv_bfloat16 val;
        
        if (logical_idx < K - 1) {
            if (chunk == 0) {
                if (ptr_prev_a != nullptr) {
                    // Fetch directly from rank - 1 peer via UVA
                    val = ptr_prev_a[bh * (K - 1) + logical_idx];
                } else {
                    val = __float2bfloat16(0.0f);
                }
            } else {
                if (ptr_next_b != nullptr) {
                    // Fetch directly from rank + 1 peer via UVA
                    val = ptr_next_b[bh * (K - 1) + logical_idx];
                } else {
                    val = __float2bfloat16(0.0f);
                }
            }
        } else {
            // Fetch from local input tensor
            int x_idx = logical_idx - (K - 1);
            if (chunk == 0) {
                val = x[bh * (2 * S) + x_idx];
            } else {
                val = x[bh * (2 * S) + S + x_idx];
            }
        }
        smem_in[i] = val;
    }

    __syncthreads();

    // Compute causal depthwise 1D convolution and scatter seamlessly into the final layout
    for (int i = tid; i < out_len; i += blockDim.x) {
        float sum = 0.0f;
        
        #pragma unroll(4)
        for (int k = 0; k < K; ++k) {
            sum += __bfloat162float(smem_in[i + k]) * __bfloat162float(smem_w[k]);
        }
        
        int out_x_idx = out_start + i;
        if (chunk == 0) {
            out[bh * (2 * S) + out_x_idx] = __float2bfloat16(sum);
        } else {
            out[bh * (2 * S) + S + out_x_idx] = __float2bfloat16(sum);
        }
    }
}

void launch_pack_symm_buf(
    torch::Tensor x,
    torch::Tensor symm_buf,
    int B, int H, int S, int K
) {
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "Must be BF16");
    int64_t total_elements = (int64_t)2 * B * H * (K - 1);
    if (total_elements == 0) return;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_symm_buf_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)symm_buf.data_ptr<at::BFloat16>(),
        B, H, S, K
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_causal_depthwise_conv1d(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t ptr_prev_a,
    int64_t ptr_next_b,
    torch::Tensor out,
    int B, int H, int S, int K, int T
) {
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "Must be BF16");
    int grid_x = (S + T - 1) / T;
    int64_t total_blocks = (int64_t)grid_x * B * H * 2;
    int threads = 256;

    size_t smem_size = (T + 2 * K - 1) * sizeof(__nv_bfloat16);
    
    if (smem_size > 49152) {
        cudaFuncSetAttribute(
            causal_depthwise_conv1d_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    causal_depthwise_conv1d_kernel<<<total_blocks, threads, smem_size, stream>>>(
        (const __nv_bfloat16*)x.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)weight.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)ptr_prev_a,
        (const __nv_bfloat16*)ptr_next_b,
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        B, H, S, K, T, grid_x
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pack_symm_buf", &launch_pack_symm_buf, "Pack bounds to symm memory");
    m.def("launch_causal_depthwise_conv1d", &launch_causal_depthwise_conv1d, "Fused halo exchange and grouped depthwise conv1d");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("hyena_conv1d_fused_symm", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(size: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = (size, dtype, group)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(size, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _symm_cache[key] = (buf, hdl)
    return buf, hdl

@torch.no_grad()
def solution(
    x: torch.Tensor,
    weight: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Per-rank Hyena causal depthwise conv1d over zigzag CP chunks, leveraging UVA direct memory
    access and fusing boundary communication perfectly behind independent local convolutions.
    """
    group = group or dist.group.WORLD
    group_ranks = dist.get_process_group_ranks(group)
    group_rank = dist.get_rank(group)
    group_world_size = len(group_ranks)

    batch, hidden, local_seq = x.shape
    S = local_seq // 2
    K = weight.shape[-1]
    pad_size = K - 1

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)

    ext = _get_ext()

    ptr_prev_a = 0
    ptr_next_b = 0

    if pad_size > 0:
        # Buffer size logically hosts (chunk_a + chunk_b) tightly packed
        symm_size = 2 * batch * hidden * pad_size
        buf, hdl = _get_symm_state(symm_size, x.dtype, x.device, group)

        # Stage our local communication bounds into the globally addressable buffer
        ext.launch_pack_symm_buf(x, buf, batch, hidden, S, K)
        hdl.barrier(channel=0)

        # Map Rank - 1 peer pointer for chunk_a
        if group_rank > 0:
            prev_g_rank = group_ranks[group_rank - 1]
            ptr_prev_a = int(hdl.buffer_ptrs[prev_g_rank])

        # Map Rank + 1 peer pointer for chunk_b
        if group_rank < group_world_size - 1:
            next_g_rank = group_ranks[group_rank + 1]
            # Offset into peer buffer to specifically target chunk_b
            chunk_size_bytes = batch * hidden * pad_size * x.element_size()
            ptr_next_b = int(hdl.buffer_ptrs[next_g_rank]) + chunk_size_bytes
        else:
            # Re-read locally created chunk_a mimicking PyTorch reference clone().contiguous()
            ptr_next_b = int(hdl.buffer_ptrs[group_ranks[group_rank]])

    # Tile block setup for balancing shared memory load.
    T = 1024 
    
    # Fire the fused depthwise convolution kernel which seamlessly streams border inputs
    # through the resolved symmetric memory direct pointers into shared memory, and
    # scatters final output values accurately avoiding PyTorch permute/reshape.
    ext.launch_causal_depthwise_conv1d(
        x, weight, ptr_prev_a, ptr_next_b, out, batch, hidden, S, K, T
    )

    return out