import math
from typing import List, Optional

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
#include <algorithm>

__global__ void fused_scale_norm_kernel(
    const void** __restrict__ ptrs,
    const int64_t* __restrict__ sizes,
    const int* __restrict__ dtypes,
    int num_tensors,
    float scale,
    float* __restrict__ out_sum
) {
    extern __shared__ float sdata[];
    float sum = 0.0f;
    int tid = threadIdx.x;
    
    for (int t = 0; t < num_tensors; t++) {
        int64_t size = sizes[t];
        int dtype = dtypes[t];
        
        if (dtype == 0) {
            __nv_bfloat16* data = (__nv_bfloat16*)ptrs[t];
            for (int64_t i = blockIdx.x * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
                float val = __bfloat162float(data[i]);
                if (scale != 1.0f) {
                    val *= scale;
                    data[i] = __float2bfloat16(val);
                }
                sum += val * val;
            }
        } else if (dtype == 1) {
            float* data = (float*)ptrs[t];
            for (int64_t i = blockIdx.x * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
                float val = data[i];
                if (scale != 1.0f) {
                    val *= scale;
                    data[i] = val;
                }
                sum += val * val;
            }
        } else if (dtype == 2) {
            __half* data = (__half*)ptrs[t];
            for (int64_t i = blockIdx.x * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
                float val = __half2float(data[i]);
                if (scale != 1.0f) {
                    val *= scale;
                    data[i] = __float2half(val);
                }
                sum += val * val;
            }
        }
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(out_sum, sdata[0]);
    }
}

__global__ void clip_scale_kernel(
    const void** __restrict__ ptrs,
    const int64_t* __restrict__ sizes,
    const int* __restrict__ dtypes,
    int num_tensors,
    float scale
) {
    int tid = threadIdx.x;
    for (int t = 0; t < num_tensors; t++) {
        int64_t size = sizes[t];
        int dtype = dtypes[t];
        
        if (dtype == 0) {
            __nv_bfloat16* data = (__nv_bfloat16*)ptrs[t];
            for (int64_t i = blockIdx.x * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
                float val = __bfloat162float(data[i]);
                data[i] = __float2bfloat16(val * scale);
            }
        } else if (dtype == 1) {
            float* data = (float*)ptrs[t];
            for (int64_t i = blockIdx.x * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
                data[i] *= scale;
            }
        } else if (dtype == 2) {
            __half* data = (__half*)ptrs[t];
            for (int64_t i = blockIdx.x * blockDim.x + tid; i < size; i += gridDim.x * blockDim.x) {
                float val = __half2float(data[i]);
                data[i] = __float2half(val * scale);
            }
        }
    }
}

__global__ void uva_reduce_step1_kernel(
    const uint64_t* __restrict__ symm_ptrs,
    const int* __restrict__ fsdp_ranks, int num_fsdp,
    const int* __restrict__ ep_fsdp_ranks, int num_ep_fsdp,
    float* __restrict__ out_non_ep,
    int my_rank
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float non_ep_total = 0.0f;
        for (int i = 0; i < num_fsdp; i++) {
            int r = fsdp_ranks[i];
            float* peer_buf = (float*)symm_ptrs[r];
            non_ep_total += peer_buf[0];
        }
        
        float ep_fsdp_total = 0.0f;
        for (int i = 0; i < num_ep_fsdp; i++) {
            int r = ep_fsdp_ranks[i];
            float* peer_buf = (float*)symm_ptrs[r];
            ep_fsdp_total += peer_buf[1];
        }
        
        float* my_buf = (float*)symm_ptrs[my_rank];
        my_buf[2] = ep_fsdp_total;
        *out_non_ep = non_ep_total;
    }
}

__global__ void uva_reduce_step2_kernel(
    const uint64_t* __restrict__ symm_ptrs,
    const int* __restrict__ ep_ranks, int num_ep,
    float* __restrict__ out_ep
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float ep_total = 0.0f;
        for (int i = 0; i < num_ep; i++) {
            int r = ep_ranks[i];
            float* peer_buf = (float*)symm_ptrs[r];
            ep_total += peer_buf[2];
        }
        *out_ep = ep_total;
    }
}

void compute_norm_and_scale(
    torch::Tensor ptrs_tensor,
    torch::Tensor sizes_tensor,
    torch::Tensor dtypes_tensor,
    int num_tensors,
    float scale,
    torch::Tensor out_sum
) {
    if (num_tensors == 0) return;
    int threads = 256;
    int blocks = std::max(1, std::min(1024, (int)num_tensors * 4));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    fused_scale_norm_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        (const void**)ptrs_tensor.data_ptr<int64_t>(),
        sizes_tensor.data_ptr<int64_t>(),
        dtypes_tensor.data_ptr<int32_t>(),
        num_tensors,
        scale,
        out_sum.data_ptr<float>()
    );
}

void apply_clip_scale(
    torch::Tensor ptrs_tensor,
    torch::Tensor sizes_tensor,
    torch::Tensor dtypes_tensor,
    int num_tensors,
    float scale
) {
    if (num_tensors == 0) return;
    int threads = 256;
    int blocks = std::max(1, std::min(1024, (int)num_tensors * 4));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    clip_scale_kernel<<<blocks, threads, 0, stream>>>(
        (const void**)ptrs_tensor.data_ptr<int64_t>(),
        sizes_tensor.data_ptr<int64_t>(),
        dtypes_tensor.data_ptr<int32_t>(),
        num_tensors,
        scale
    );
}

void launch_uva_reduce_step1(
    torch::Tensor symm_ptrs,
    torch::Tensor fsdp_ranks,
    torch::Tensor ep_fsdp_ranks,
    torch::Tensor out_non_ep,
    int my_rank
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uva_reduce_step1_kernel<<<1, 1, 0, stream>>>(
        (const uint64_t*)symm_ptrs.data_ptr<int64_t>(),
        fsdp_ranks.data_ptr<int32_t>(), fsdp_ranks.size(0),
        ep_fsdp_ranks.data_ptr<int32_t>(), ep_fsdp_ranks.size(0),
        out_non_ep.data_ptr<float>(),
        my_rank
    );
}

void launch_uva_reduce_step2(
    torch::Tensor symm_ptrs,
    torch::Tensor ep_ranks,
    torch::Tensor out_ep
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uva_reduce_step2_kernel<<<1, 1, 0, stream>>>(
        (const uint64_t*)symm_ptrs.data_ptr<int64_t>(),
        ep_ranks.data_ptr<int32_t>(), ep_ranks.size(0),
        out_ep.data_ptr<float>()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_norm_and_scale", &compute_norm_and_scale);
    m.def("apply_clip_scale", &apply_clip_scale);
    m.def("launch_uva_reduce_step1", &launch_uva_reduce_step1);
    m.def("launch_uva_reduce_step2", &launch_uva_reduce_step2);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("clip_grad_norm_uva_ext", CUDA_SRC)
    return _ext

_symm_state = None
def _get_symm_state(device):
    global _symm_state
    if _symm_state is not None:
        return _symm_state
    
    # [non_ep_sum, ep_sum, ep_sum_round1, padding]
    buf = symm_mem.empty(4, dtype=torch.float32, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    symm_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_state = (buf, hdl, symm_ptrs)
    return _symm_state

_tensor_info_cache = {}
def _get_tensor_info_cached(tensors, key, device):
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None, 0
        
    cache_key = (key, tuple((t.data_ptr(), t.numel(), t.dtype) for t in tensors))
    if cache_key in _tensor_info_cache:
        return _tensor_info_cache[cache_key]
        
    if len(_tensor_info_cache) > 100:
        _tensor_info_cache.clear()
        
    ptrs = [t.data_ptr() for t in tensors]
    sizes = [t.numel() for t in tensors]
    dtypes = []
    for t in tensors:
        if t.dtype == torch.bfloat16: dtypes.append(0)
        elif t.dtype == torch.float32: dtypes.append(1)
        elif t.dtype == torch.float16: dtypes.append(2)
        else: raise ValueError(f"Unsupported dtype: {t.dtype}")
    
    ptrs_t = torch.tensor(ptrs, dtype=torch.int64, device=device)
    sizes_t = torch.tensor(sizes, dtype=torch.int64, device=device)
    dtypes_t = torch.tensor(dtypes, dtype=torch.int32, device=device)
    res = ((ptrs_t, sizes_t, dtypes_t), len(tensors))
    _tensor_info_cache[cache_key] = res
    return res

_ranks_cache = {}
def _get_ranks_tensor_cached(group, device):
    if group is None:
        ranks = [dist.get_rank()] if dist.is_initialized() else [0]
    else:
        ranks = dist.get_process_group_ranks(group)
        
    key = tuple(ranks)
    if key not in _ranks_cache:
        _ranks_cache[key] = torch.tensor(ranks, dtype=torch.int32, device=device)
    return _ranks_cache[key]

@torch.no_grad()
def solution(
    non_ep_grad_tensors: List[torch.Tensor],
    ep_grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    ep_size: int = 1,
    fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    ext = _get_ext()
    dev = next((t.device for t in non_ep_grad_tensors + ep_grad_tensors if t is not None), torch.device("cuda"))
    
    non_ep_info, non_ep_count = _get_tensor_info_cached(non_ep_grad_tensors, "non_ep", dev)
    ep_info, ep_count = _get_tensor_info_cached(ep_grad_tensors, "ep", dev)
    
    if not dist.is_initialized():
        # Single GPU Fallback
        buf = torch.zeros(2, dtype=torch.float32, device=dev)
        if non_ep_count > 0:
            ext.compute_norm_and_scale(non_ep_info[0], non_ep_info[1], non_ep_info[2], non_ep_count, 1.0, buf[0:1])
        if ep_count > 0:
            ep_scale = 1.0 / float(ep_size) if ep_size > 1 else 1.0
            ext.compute_norm_and_scale(ep_info[0], ep_info[1], ep_info[2], ep_count, ep_scale, buf[1:2])
            
        total_norm_tensor = torch.sqrt(buf[0] + buf[1])
        total_norm_val = total_norm_tensor.item()
        if total_norm_val > max_norm:
            coef = max_norm / total_norm_val
            if non_ep_count > 0:
                ext.apply_clip_scale(non_ep_info[0], non_ep_info[1], non_ep_info[2], non_ep_count, coef)
            if ep_count > 0:
                ext.apply_clip_scale(ep_info[0], ep_info[1], ep_info[2], ep_count, coef)
        return total_norm_tensor

    buf, hdl, symm_ptrs = _get_symm_state(dev)
    my_rank = dist.get_rank()
    
    fsdp_ranks = _get_ranks_tensor_cached(fsdp_group, dev)
    ep_fsdp_ranks = _get_ranks_tensor_cached(ep_fsdp_group, dev)
    ep_ranks = _get_ranks_tensor_cached(ep_group, dev)
    
    buf[:2].zero_()

    if non_ep_count > 0:
        ext.compute_norm_and_scale(non_ep_info[0], non_ep_info[1], non_ep_info[2], non_ep_count, 1.0, buf[0:1])
        
    if ep_count > 0:
        ep_scale = 1.0 / float(ep_size) if ep_size > 1 else 1.0
        ext.compute_norm_and_scale(ep_info[0], ep_info[1], ep_info[2], ep_count, ep_scale, buf[1:2])

    out_norms = torch.empty(2, dtype=torch.float32, device=dev)
    
    # Barrier 0: Blocks device stream until local sums are computed
    hdl.barrier(channel=0)
    
    ext.launch_uva_reduce_step1(symm_ptrs, fsdp_ranks, ep_fsdp_ranks, out_norms[0:1], my_rank)
    
    # Barrier 1: Blocks device stream until step1 writes (ep_fsdp sub-totals) are visible
    hdl.barrier(channel=1)
    
    ext.launch_uva_reduce_step2(symm_ptrs, ep_ranks, out_norms[1:2])
    
    # Barrier 2: Ensures all streams have consumed the buffer before the next iteration can `zero_()` it
    hdl.barrier(channel=2)

    total_norm_tensor = torch.sqrt(out_norms[0] + out_norms[1])
    total_norm_val = total_norm_tensor.item()
    
    if total_norm_val > max_norm:
        coef = max_norm / total_norm_val
        if non_ep_count > 0:
            ext.apply_clip_scale(non_ep_info[0], non_ep_info[1], non_ep_info[2], non_ep_count, coef)
        if ep_count > 0:
            ext.apply_clip_scale(ep_info[0], ep_info[1], ep_info[2], ep_count, coef)

    return total_norm_tensor