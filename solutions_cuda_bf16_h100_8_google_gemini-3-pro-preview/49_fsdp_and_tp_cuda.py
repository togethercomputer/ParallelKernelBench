from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// 1. FSDP Gather Kernels (Directly copying full weights from peer's UVA memory)
// ---------------------------------------------------------------------------
__global__ void gather_w1_w2_kernel(
    const uint64_t* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ w1_full,
    __nv_bfloat16* __restrict__ w2_full,
    int n_fsdp,
    int n_tp,
    int tp_rank,
    int64_t K
) {
    int64_t total_elements = n_fsdp * K;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; tid < total_elements; tid += stride) {
        int j = tid / K;
        int64_t idx = tid % K;
        int peer_rank = j * n_tp + tp_rank; // Ranks sharing the same tp_rank
        const __nv_bfloat16* peer_buf = (const __nv_bfloat16*)ptrs[peer_rank];
        
        w1_full[tid] = peer_buf[idx];
        w2_full[tid] = peer_buf[K + idx];
    }
}

__global__ void gather_w3_kernel(
    const uint64_t* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ w3_full,
    int n_fsdp,
    int n_tp,
    int tp_rank,
    int rows,
    int cols
) {
    int64_t K = (int64_t)rows * cols;
    int64_t total_elements = n_fsdp * K;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; tid < total_elements; tid += stride) {
        int j = tid / K;
        int64_t idx = tid % K;
        int peer_rank = j * n_tp + tp_rank;
        
        int r = idx / cols;
        int c = idx % cols;
        
        const __nv_bfloat16* peer_buf = (const __nv_bfloat16*)ptrs[peer_rank];
        
        // Strided copy since W3 is gathered along dim=1
        int64_t out_idx = (int64_t)r * (cols * n_fsdp) + j * cols + c;
        w3_full[out_idx] = peer_buf[2 * K + idx];
    }
}

// ---------------------------------------------------------------------------
// 2. Fused SwiGLU (z = silu(x1) * x2)
// ---------------------------------------------------------------------------
__global__ void swiglu_bf16x2_kernel(
    const __nv_bfloat162* __restrict__ x1,
    const __nv_bfloat162* __restrict__ x2,
    __nv_bfloat162* __restrict__ z,
    int64_t numel_2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel_2; i += stride) {
        float2 v1 = __bfloat1622float2(x1[i]);
        float2 v2 = __bfloat1622float2(x2[i]);
        
        float sig_x = 1.0f / (1.0f + expf(-v1.x));
        float sig_y = 1.0f / (1.0f + expf(-v1.y));
        
        float2 res;
        res.x = v1.x * sig_x * v2.x;
        res.y = v1.y * sig_y * v2.y;
        
        z[i] = __float22bfloat162_rn(res);
    }
}

__global__ void swiglu_odd_kernel(
    const __nv_bfloat16* __restrict__ x1,
    const __nv_bfloat16* __restrict__ x2,
    __nv_bfloat16* __restrict__ z,
    int64_t idx
) {
    float val1 = __bfloat162float(x1[idx]);
    float val2 = __bfloat162float(x2[idx]);
    float sig = 1.0f / (1.0f + expf(-val1));
    z[idx] = __float2bfloat16(val1 * sig * val2);
}

// ---------------------------------------------------------------------------
// 3. Tensor Parallel All-Reduce
// ---------------------------------------------------------------------------
__global__ void tp_allreduce_bf16x2_kernel(
    const uint64_t* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ y_out,
    int n_tp,
    int n_fsdp,
    int fsdp_rank,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t numel_2 = numel / 2;
    
    for (int64_t i = idx; i < numel_2; i += stride) {
        float2 sum = {0.0f, 0.0f};
        for (int p = 0; p < n_tp; ++p) {
            int peer_rank = fsdp_rank * n_tp + p;
            const __nv_bfloat162* peer_buf = (const __nv_bfloat162*)ptrs[peer_rank];
            float2 val = __bfloat1622float2(peer_buf[i]);
            sum.x += val.x;
            sum.y += val.y;
        }
        ((__nv_bfloat162*)y_out)[i] = __float22bfloat162_rn(sum);
    }
    
    if (idx == 0 && (numel % 2) != 0) {
        int64_t last_idx = numel - 1;
        float sum = 0.0f;
        for (int p = 0; p < n_tp; ++p) {
            int peer_rank = fsdp_rank * n_tp + p;
            const __nv_bfloat16* peer_buf = (const __nv_bfloat16*)ptrs[peer_rank];
            sum += __bfloat162float(peer_buf[last_idx]);
        }
        y_out[last_idx] = __float2bfloat16(sum);
    }
}

// ---------------------------------------------------------------------------
// Host Bindings
// ---------------------------------------------------------------------------
void launch_gather_w1_w2(
    torch::Tensor ptrs, torch::Tensor w1_full, torch::Tensor w2_full,
    int n_fsdp, int n_tp, int tp_rank, int64_t K
) {
    int threads = 256;
    int blocks = (n_fsdp * K + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_w1_w2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(w1_full.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(w2_full.data_ptr<at::BFloat16>()),
        n_fsdp, n_tp, tp_rank, K
    );
}

void launch_gather_w3(
    torch::Tensor ptrs, torch::Tensor w3_full,
    int n_fsdp, int n_tp, int tp_rank, int rows, int cols
) {
    int64_t K = (int64_t)rows * cols;
    int threads = 256;
    int blocks = (n_fsdp * K + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_w3_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(w3_full.data_ptr<at::BFloat16>()),
        n_fsdp, n_tp, tp_rank, rows, cols
    );
}

void launch_swiglu(torch::Tensor x1, torch::Tensor x2, torch::Tensor z, int64_t numel) {
    int threads = 256;
    int64_t numel_2 = numel / 2;
    if (numel_2 > 0) {
        int blocks = (numel_2 + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        swiglu_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat162*>(x1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat162*>(x2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat162*>(z.data_ptr<at::BFloat16>()),
            numel_2
        );
    }
    if (numel % 2 != 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        swiglu_odd_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x1.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(z.data_ptr<at::BFloat16>()),
            numel - 1
        );
    }
}

void launch_tp_allreduce(
    torch::Tensor ptrs, torch::Tensor y_out,
    int n_tp, int n_fsdp, int fsdp_rank, int64_t numel
) {
    int threads = 256;
    int blocks = (numel / 2 + threads - 1) / threads;
    if (blocks == 0 && numel > 0) blocks = 1; 
    if (blocks > 65535) blocks = 65535;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (numel > 0) {
        tp_allreduce_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(ptrs.data_ptr<int64_t>()),
            reinterpret_cast<__nv_bfloat16*>(y_out.data_ptr<at::BFloat16>()),
            n_tp, n_fsdp, fsdp_rank, numel
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_w1_w2", &launch_gather_w1_w2, "Gather W1 and W2 via P2P");
    m.def("launch_gather_w3", &launch_gather_w3, "Gather W3 via P2P");
    m.def("launch_swiglu", &launch_swiglu, "Fused SwiGLU bf16");
    m.def("launch_tp_allreduce", &launch_tp_allreduce, "TP AllReduce via P2P");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_tp_fused_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(key, shape, dtype, device):
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(shape, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    _symm_cache[key] = (buf, hdl, ptrs)
    return buf, hdl, ptrs

_local_cache = {}
def _get_local_buffer(key, shape, dtype, device):
    if key in _local_cache:
        return _local_cache[key]
    buf = torch.empty(shape, dtype=dtype, device=device)
    _local_cache[key] = buf
    return buf

_stream2 = None
def _get_stream2():
    global _stream2
    if _stream2 is None:
        _stream2 = torch.cuda.Stream()
    return _stream2


@torch.no_grad()
def solution(
    x_local: Tensor,
    W1_shard: Tensor,
    W2_shard: Tensor,
    W3_shard: Tensor,
    n_tp: int,
    n_fsdp: int,
) -> Tensor:
    rank = dist.get_rank()
    fsdp_rank = rank // n_tp
    tp_rank = rank % n_tp
    
    d_fsdp, d_ff_tp = W1_shard.shape
    K = d_fsdp * d_ff_tp
    device = x_local.device
    dtype = x_local.dtype
    ext = _get_ext()
    
    # 1. Acquire symmetric buffer for FSDP weights and copy our local shards over
    shards_key = ("shards", 3 * K, dtype, device)
    shards_symm, hdl_shards, ptrs_shards = _get_symm_state(shards_key, [3 * K], dtype, device)
    
    shards_symm[0:K].copy_(W1_shard.view(-1))
    shards_symm[K:2*K].copy_(W2_shard.view(-1))
    shards_symm[2*K:3*K].copy_(W3_shard.view(-1))
    
    # Barrier ensures all peers have flushed their local weights to Symmetric Memory 
    hdl_shards.barrier(channel=0)
    
    # 2. Reconstruct locally missing chunks by grabbing them off peer devices
    W1_full = _get_local_buffer(("w1", d_fsdp * n_fsdp, d_ff_tp), (d_fsdp * n_fsdp, d_ff_tp), dtype, device)
    W2_full = _get_local_buffer(("w2", d_fsdp * n_fsdp, d_ff_tp), (d_fsdp * n_fsdp, d_ff_tp), dtype, device)
    W3_full = _get_local_buffer(("w3", d_ff_tp, d_fsdp * n_fsdp), (d_ff_tp, d_fsdp * n_fsdp), dtype, device)
    
    # Pull W1 and W2 onto Default Stream to unlock first matmuls
    ext.launch_gather_w1_w2(ptrs_shards, W1_full, W2_full, n_fsdp, n_tp, tp_rank, K)
    
    # Overlap: Schedule W3's Gather independently on background stream
    stream2 = _get_stream2()
    stream2.wait_stream(torch.cuda.current_stream()) # Stream 2 awaits the Barrier flush prior to pulling
    with torch.cuda.stream(stream2):
        ext.launch_gather_w3(ptrs_shards, W3_full, n_fsdp, n_tp, tp_rank, d_ff_tp, d_fsdp)
        
    # Overlap: Compute hidden states (x1, x2) and execute customized SwiGLU alongside W3's comm
    x1 = torch.mm(x_local, W1_full)
    x2 = torch.mm(x_local, W2_full)
    
    z = _get_local_buffer(("z", x1.shape[0], x1.shape[1]), x1.shape, dtype, device)
    ext.launch_swiglu(x1, x2, z, x1.numel())
    
    # Re-sync prior to resolving W3
    torch.cuda.current_stream().wait_stream(stream2)
    y_partial = torch.mm(z, W3_full)
    
    # 3. Complete chunk reduction using an identical P2P paradigm across the TP ranks
    y_numel = y_partial.numel()
    y_key = ("y", y_partial.shape, dtype, device)
    y_symm, hdl_y, ptrs_y = _get_symm_state(y_key, y_partial.shape, dtype, device)
    y_out = torch.empty_like(y_partial)
    
    y_symm.copy_(y_partial.view(-1))
    hdl_y.barrier(channel=0)
    
    ext.launch_tp_allreduce(ptrs_y, y_out, n_tp, n_fsdp, fsdp_rank, y_numel)
    
    return y_out

__all__ = ["solution"]