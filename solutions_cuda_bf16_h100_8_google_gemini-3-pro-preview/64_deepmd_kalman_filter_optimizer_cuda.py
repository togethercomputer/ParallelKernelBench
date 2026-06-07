"""
Strategy:
1. **Device-side communication**: Replaced `dist.all_reduce` for the scalar `tmp` and `dist.all_gather` for parameter updates with direct UVA pointer accesses from `torch.distributed._symmetric_memory`.
2. **Fused Compute & Overlap**: Grouped computations into batched CUDA kernels. Kernel 1 fuses GEMV (`P_i @ H_i`) with dot product, directly using atomic additions on a symmetric memory scalar. Kernel 2 reads all peers' scalars directly to compute the denominator, applies parameter and covariance updates, and natively writes updated weights to a symmetric buffer.
3. **Optimized Shape Gather**: `all_gather_object` is extremely slow and only needed once (shapes stay constant across iterations); this layout is cached so hot-path calls only do memory accesses.
4. **Gather offload**: A third CUDA kernel uses UVA reads to efficiently aggregate peer data into a contiguous output buffer, skipping PyTorch allocations and NCCL collective latency.
"""

from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void kernel1(
    const int64_t* __restrict__ P_ptrs,
    const int64_t* __restrict__ H_ptrs,
    __nv_bfloat16* __restrict__ K_flat,
    const int* __restrict__ N_array,
    const int* __restrict__ Offsets_array,
    float lam,
    float* __restrict__ symm_tmp
) {
    int block_idx = blockIdx.x;
    int n = N_array[block_idx];
    int offset = Offsets_array[block_idx];
    
    const __nv_bfloat16* P = (const __nv_bfloat16*)P_ptrs[block_idx];
    const __nv_bfloat16* H = (const __nv_bfloat16*)H_ptrs[block_idx];
    __nv_bfloat16* K = K_flat + offset;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    float local_dot = 0.0f;

    for (int row = tid; row < n; row += threads) {
        float sum = 0.0f;
        for (int col = 0; col < n; col++) {
            sum += __bfloat162float(P[row * n + col]) * __bfloat162float(H[col]);
        }
        K[row] = __float2bfloat16(sum);
        local_dot += sum * __bfloat162float(H[row]);
    }

    static __shared__ float shared_dot[32];
    int lane = tid % 32;
    int wid = tid / 32;

    #pragma unroll
    for (int offset_shfl = 16; offset_shfl > 0; offset_shfl /= 2) {
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset_shfl);
    }

    if (lane == 0) {
        shared_dot[wid] = local_dot;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (threads + 31) / 32) ? shared_dot[tid] : 0.0f;
        #pragma unroll
        for (int offset_shfl = 16; offset_shfl > 0; offset_shfl /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset_shfl);
        }
        if (tid == 0) {
            atomicAdd(symm_tmp, val + lam);
        }
    }
}

__global__ void kernel2(
    const int64_t* __restrict__ symm_tmp_ptrs,
    int world_size,
    const int64_t* __restrict__ P_ptrs,
    const int64_t* __restrict__ W_ptrs,
    const __nv_bfloat16* __restrict__ K_flat,
    __nv_bfloat16* __restrict__ symm_weights,
    const int* __restrict__ N_array,
    const int* __restrict__ Offsets_array,
    const __nv_bfloat16* __restrict__ err_ptr,
    float lam
) {
    __shared__ float A;
    __shared__ float err;

    int block_idx = blockIdx.x;
    int n = N_array[block_idx];
    int offset = Offsets_array[block_idx];
    
    if (threadIdx.x == 0) {
        float global_tmp = 0.0f;
        for (int r = 0; r < world_size; r++) {
            float* peer_tmp = (float*)symm_tmp_ptrs[r];
            global_tmp += *peer_tmp;
        }
        A = 1.0f / global_tmp;
        err = __bfloat162float(*err_ptr);
    }
    __syncthreads();

    float local_A = A;
    float local_err = err;
    
    __nv_bfloat16* P = (__nv_bfloat16*)P_ptrs[block_idx];
    __nv_bfloat16* W = (__nv_bfloat16*)W_ptrs[block_idx];
    const __nv_bfloat16* K = K_flat + offset;
    __nv_bfloat16* W_out = symm_weights + offset;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    for (int row = tid; row < n; row += threads) {
        float w_val = __bfloat162float(W[row]);
        float k_val = __bfloat162float(K[row]);
        float new_w = w_val + local_A * local_err * k_val;
        W[row] = __float2bfloat16(new_w);
        W_out[row] = __float2bfloat16(new_w);
    }

    int total_elements = n * n;
    for (int idx = tid; idx < total_elements; idx += threads) {
        int row = idx / n;
        int col = idx % n;
        float p_val = __bfloat162float(P[idx]);
        float kr = __bfloat162float(K[row]);
        float kc = __bfloat162float(K[col]);
        
        float new_p = (p_val - local_A * kr * kc) / lam;
        P[idx] = __float2bfloat16(new_p);
    }
}

__global__ void gather_kernel(
    const int64_t* __restrict__ symm_weights_ptrs,
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ rank_offsets,
    const int* __restrict__ rank_sizes,
    int world_size
) {
    int block_idx = blockIdx.x; 
    if (block_idx >= world_size) return;

    int offset = rank_offsets[block_idx];
    int size = rank_sizes[block_idx];
    const __nv_bfloat16* src = (const __nv_bfloat16*)symm_weights_ptrs[block_idx];
    __nv_bfloat16* dst = out + offset;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    for (int i = tid; i < size; i += threads) {
        dst[i] = src[i];
    }
}

void launch_kernel1(
    torch::Tensor P_ptrs, torch::Tensor H_ptrs, torch::Tensor K_flat,
    torch::Tensor N_array, torch::Tensor Offsets_array, float lam,
    torch::Tensor symm_tmp, int weights_num
) {
    int threads = 256;
    int blocks = weights_num;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    kernel1<<<blocks, threads, 0, stream>>>(
        P_ptrs.data_ptr<int64_t>(),
        H_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)K_flat.data_ptr<at::BFloat16>(),
        N_array.data_ptr<int>(),
        Offsets_array.data_ptr<int>(),
        lam,
        symm_tmp.data_ptr<float>()
    );
}

void launch_kernel2(
    torch::Tensor symm_tmp_ptrs, int world_size,
    torch::Tensor P_ptrs, torch::Tensor W_ptrs, torch::Tensor K_flat,
    torch::Tensor symm_weights, torch::Tensor N_array, torch::Tensor Offsets_array,
    torch::Tensor err_tensor, float lam, int weights_num
) {
    int threads = 256;
    int blocks = weights_num;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    kernel2<<<blocks, threads, 0, stream>>>(
        symm_tmp_ptrs.data_ptr<int64_t>(),
        world_size,
        P_ptrs.data_ptr<int64_t>(),
        W_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)K_flat.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)symm_weights.data_ptr<at::BFloat16>(),
        N_array.data_ptr<int>(),
        Offsets_array.data_ptr<int>(),
        (__nv_bfloat16*)err_tensor.data_ptr<at::BFloat16>(),
        lam
    );
}

void launch_gather_kernel(
    torch::Tensor symm_weights_ptrs, torch::Tensor out,
    torch::Tensor rank_offsets, torch::Tensor rank_sizes,
    int world_size
) {
    int threads = 1024;
    int blocks = world_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_kernel<<<blocks, threads, 0, stream>>>(
        symm_weights_ptrs.data_ptr<int64_t>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        rank_offsets.data_ptr<int>(),
        rank_sizes.data_ptr<int>(),
        world_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel1", &launch_kernel1, "Kalman blockwise kernel 1");
    m.def("launch_kernel2", &launch_kernel2, "Kalman blockwise kernel 2");
    m.def("launch_gather_kernel", &launch_gather_kernel, "Kalman blockwise gather");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("deepmd_kalman_opt_ext", CUDA_SRC)
    return _ext

_cache = {}

def _get_resources(H, weights, P):
    if not dist.is_initialized():
        world_size = 1
    else:
        world_size = dist.get_world_size()

    device = weights[0].device
    local_shapes = tuple(w.shape[0] for w in weights)
    cache_key = (world_size, local_shapes)

    if cache_key in _cache:
        return _cache[cache_key]

    weights_num = len(weights)
    if weights_num > 0:
        N_array = torch.tensor(local_shapes, dtype=torch.int32, device=device)
        Offsets_array = torch.cat([
            torch.tensor([0], device=device, dtype=torch.int32), 
            torch.cumsum(N_array, dim=0)[:-1].to(torch.int32)
        ])
    else:
        N_array = torch.empty(0, dtype=torch.int32, device=device)
        Offsets_array = torch.empty(0, dtype=torch.int32, device=device)
        
    total_local_weights = sum(local_shapes)

    if world_size > 1:
        shape_list = [None for _ in range(world_size)]
        dist.all_gather_object(shape_list, list(local_shapes))
    else:
        shape_list = [list(local_shapes)]
        
    all_split_sizes = []
    rank_sizes = []
    for shapes in shape_list:
        all_split_sizes.extend(shapes)
        rank_sizes.append(sum(shapes))
        
    total_world_weights = sum(rank_sizes)
    
    symm_tmp_buf = symm_mem.empty((1,), dtype=torch.float32, device=device)
    if world_size > 1:
        hdl_tmp = symm_mem.rendezvous(symm_tmp_buf, dist.group.WORLD)
        symm_tmp_ptrs = torch.tensor(hdl_tmp.buffer_ptrs, dtype=torch.int64, device=device)
    else:
        hdl_tmp = None
        symm_tmp_ptrs = torch.tensor([symm_tmp_buf.data_ptr()], dtype=torch.int64, device=device)

    if total_local_weights > 0:
        symm_weights_buf = symm_mem.empty((total_local_weights,), dtype=torch.bfloat16, device=device)
    else:
        symm_weights_buf = torch.empty((0,), dtype=torch.bfloat16, device=device)
        
    if world_size > 1 and total_local_weights > 0:
        hdl_weights = symm_mem.rendezvous(symm_weights_buf, dist.group.WORLD)
        symm_weights_ptrs = torch.tensor(hdl_weights.buffer_ptrs, dtype=torch.int64, device=device)
    else:
        hdl_weights = None
        symm_weights_ptrs = torch.tensor([symm_weights_buf.data_ptr() if total_local_weights > 0 else 0], dtype=torch.int64, device=device)

    rank_offsets = [0]
    for s in rank_sizes[:-1]:
        rank_offsets.append(rank_offsets[-1] + s)
        
    rank_sizes_tensor = torch.tensor(rank_sizes, dtype=torch.int32, device=device)
    rank_offsets_tensor = torch.tensor(rank_offsets, dtype=torch.int32, device=device)
    
    K_flat = torch.empty(total_local_weights, dtype=torch.bfloat16, device=device)
    out_gathered = torch.empty(total_world_weights, dtype=torch.bfloat16, device=device)

    res = {
        "N_array": N_array,
        "Offsets_array": Offsets_array,
        "all_split_sizes": all_split_sizes,
        "symm_tmp_buf": symm_tmp_buf,
        "hdl_tmp": hdl_tmp,
        "symm_tmp_ptrs": symm_tmp_ptrs,
        "symm_weights_buf": symm_weights_buf,
        "hdl_weights": hdl_weights,
        "symm_weights_ptrs": symm_weights_ptrs,
        "rank_sizes_tensor": rank_sizes_tensor,
        "rank_offsets_tensor": rank_offsets_tensor,
        "K_flat": K_flat,
        "out_gathered": out_gathered,
        "world_size": world_size,
    }
    _cache[cache_key] = res
    return res

@torch.no_grad()
def solution(
    H: List[torch.Tensor],
    error: torch.Tensor,
    weights: List[torch.Tensor],
    P: List[torch.Tensor],
    kalman_lambda: float,
    kalman_nue: float = 0.9987,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:

    weights_num = len(weights)
    lam_val = float(kalman_lambda)
    lam_next_val = kalman_nue * lam_val + 1.0 - kalman_nue

    if weights_num == 0:
        if dist.is_initialized() and dist.get_world_size() > 1:
            device = error.device
            shape_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(shape_list, [])
            return [], P, torch.tensor(lam_next_val, dtype=torch.bfloat16, device=device)
        return [], P, torch.tensor(lam_next_val, dtype=torch.bfloat16, device=error.device if error is not None else torch.device("cuda"))

    device = weights[0].device
    res = _get_resources(H, weights, P)
    
    P_ptrs_dev = torch.tensor([p.data_ptr() for p in P], dtype=torch.int64, device=device)
    H_ptrs_dev = torch.tensor([h.data_ptr() for h in H], dtype=torch.int64, device=device)
    W_ptrs_dev = torch.tensor([w.data_ptr() for w in weights], dtype=torch.int64, device=device)

    res["symm_tmp_buf"].zero_()
    err_dev = error.to(device=device, dtype=torch.bfloat16)

    ext = _get_ext()
    
    ext.launch_kernel1(
        P_ptrs_dev, H_ptrs_dev, res["K_flat"],
        res["N_array"], res["Offsets_array"], lam_val,
        res["symm_tmp_buf"], weights_num
    )
    
    if res["hdl_tmp"] is not None:
        res["hdl_tmp"].barrier(channel=0)
        
    ext.launch_kernel2(
        res["symm_tmp_ptrs"], res["world_size"],
        P_ptrs_dev, W_ptrs_dev, res["K_flat"],
        res["symm_weights_buf"], res["N_array"], res["Offsets_array"],
        err_dev, lam_val, weights_num
    )
    
    if res["hdl_weights"] is not None:
        res["hdl_weights"].barrier(channel=0)
        ext.launch_gather_kernel(
            res["symm_weights_ptrs"], res["out_gathered"],
            res["rank_offsets_tensor"], res["rank_sizes_tensor"],
            res["world_size"]
        )
        out = res["out_gathered"]
    else:
        out = res["symm_weights_buf"]
        
    if len(res["all_split_sizes"]) > 0:
        gathered_tensors = torch.split(out, res["all_split_sizes"])
        weights_out = [t.view(-1, 1) for t in gathered_tensors]
    else:
        weights_out = []
    
    lam_next_tensor = torch.tensor(lam_next_val, dtype=weights[0].dtype, device=device)
    
    return weights_out, P, lam_next_tensor