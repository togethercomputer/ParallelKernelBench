import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import List, Optional, Tuple
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

struct PtrArray {
    const void* ptrs[32];
};

template <typename T>
__global__ void reduce_expert_mask_kernel(
    const T* __restrict__ mask,
    T* __restrict__ local_counts,
    int num_experts,
    int inner_size
) {
    int expert_idx = blockIdx.x;
    if (expert_idx >= num_experts) return;

    float sum = 0.0f;
    // Each thread processes elements in a strided fashion
    for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
        sum += static_cast<float>(mask[expert_idx * inner_size + i]);
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block reduce
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 0) {
        shared[warp] = sum;
    }
    __syncthreads();

    sum = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            local_counts[expert_idx] = static_cast<T>(sum);
        }
    }
}

template <typename T>
__global__ void gather_and_split_kernel(
    PtrArray remote_ptrs, 
    int rank,
    int ep_size,
    int num_local_experts,
    T* out_global_tokens,
    T* out_global_sum,
    int32_t* out_input_splits,
    int32_t* out_output_splits
) {
    const T* const* ptrs = reinterpret_cast<const T* const*>(remote_ptrs.ptrs);

    // Compute the dense slice matrix & sum over local experts
    for (int j = threadIdx.x; j < num_local_experts; j += blockDim.x) {
        float sum_global = 0.0f;
        for (int i = 0; i < ep_size; i++) {
            T val = ptrs[i][rank * num_local_experts + j];
            out_global_tokens[i * num_local_experts + j] = val;
            sum_global += static_cast<float>(val);
        }
        out_global_sum[j] = static_cast<T>(sum_global);
    }
    
    // We only need the first ep_size threads to compute split distributions
    if (threadIdx.x < ep_size) {
        int i = threadIdx.x;
        
        // input splits
        float sum_in = 0.0f;
        for (int jj = 0; jj < num_local_experts; ++jj) {
            sum_in += static_cast<float>(ptrs[rank][i * num_local_experts + jj]);
        }
        out_input_splits[i] = static_cast<int32_t>(sum_in);

        // output splits
        float sum_out = 0.0f;
        for (int jj = 0; jj < num_local_experts; ++jj) {
            sum_out += static_cast<float>(ptrs[i][rank * num_local_experts + jj]);
        }
        out_output_splits[i] = static_cast<int32_t>(sum_out);
    }
}

void run_reduce(
    torch::Tensor mask,
    torch::Tensor local_counts,
    int num_experts,
    int inner_size
) {
    const int threads = 256;
    const int blocks = num_experts;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (mask.scalar_type() == torch::kBFloat16) {
        reduce_expert_mask_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(mask.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(local_counts.data_ptr<at::BFloat16>()),
            num_experts,
            inner_size
        );
    } else if (mask.scalar_type() == torch::kFloat32) {
        reduce_expert_mask_kernel<float><<<blocks, threads, 0, stream>>>(
            mask.data_ptr<float>(),
            local_counts.data_ptr<float>(),
            num_experts,
            inner_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only BF16 and FP32 are supported.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_gather_and_split(
    std::vector<int64_t> remote_ptr_ints,
    int rank,
    int ep_size,
    int num_local_experts,
    torch::Tensor out_global_tokens,
    torch::Tensor out_global_sum,
    torch::Tensor out_input_splits,
    torch::Tensor out_output_splits
) {
    TORCH_CHECK(ep_size <= 32, "ep_size > 32 not supported in struct caching");
    PtrArray ptr_array;
    for (int i = 0; i < ep_size; i++) {
        ptr_array.ptrs[i] = reinterpret_cast<const void*>(remote_ptr_ints[i]);
    }

    const int threads = 256;
    const int blocks = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (out_global_tokens.scalar_type() == torch::kBFloat16) {
        gather_and_split_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            ptr_array,
            rank,
            ep_size,
            num_local_experts,
            reinterpret_cast<__nv_bfloat16*>(out_global_tokens.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_global_sum.data_ptr<at::BFloat16>()),
            out_input_splits.data_ptr<int32_t>(),
            out_output_splits.data_ptr<int32_t>()
        );
    } else if (out_global_tokens.scalar_type() == torch::kFloat32) {
        gather_and_split_kernel<float><<<blocks, threads, 0, stream>>>(
            ptr_array,
            rank,
            ep_size,
            num_local_experts,
            out_global_tokens.data_ptr<float>(),
            out_global_sum.data_ptr<float>(),
            out_input_splits.data_ptr<int32_t>(),
            out_output_splits.data_ptr<int32_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only BF16 and FP32 are supported.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_reduce", &run_reduce, "Reduce expert mask to local counts");
    m.def("run_gather_and_split", &run_gather_and_split, "Gather and compute splits via UVA");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_token_preprocess_uva", CUDA_SRC)
    return _ext

_symm_cache = None
def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype and c["group"] == group:
            return c["buf"], c["hdl"], c["ptrs"]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = [int(p) for p in hdl.buffer_ptrs]
    _symm_cache = {"n": n, "dtype": dtype, "buf": buf, "hdl": hdl, "group": group, "ptrs": ptrs}
    return buf, hdl, ptrs

_tensor_cache = None
def _get_tensors(ep_size: int, num_local_experts: int, dtype: torch.dtype, device: torch.device):
    global _tensor_cache
    if _tensor_cache is not None:
        c = _tensor_cache
        if c["ep_size"] == ep_size and c["num_local_experts"] == num_local_experts and c["dtype"] == dtype:
            return c["out_global_tokens"], c["out_global_sum"], c["out_input_splits"], c["out_output_splits"]
            
    out_global_tokens = torch.empty((ep_size, num_local_experts), dtype=dtype, device=device)
    out_global_sum = torch.empty((num_local_experts,), dtype=dtype, device=device)
    out_input_splits = torch.empty((ep_size,), dtype=torch.int32, device=device)
    out_output_splits = torch.empty((ep_size,), dtype=torch.int32, device=device)
    
    _tensor_cache = {
        "ep_size": ep_size,
        "num_local_experts": num_local_experts,
        "dtype": dtype,
        "out_global_tokens": out_global_tokens,
        "out_global_sum": out_global_sum,
        "out_input_splits": out_input_splits,
        "out_output_splits": out_output_splits
    }
    return out_global_tokens, out_global_sum, out_input_splits, out_output_splits

@torch.no_grad()
def solution(
    expert_mask: torch.Tensor,
    num_experts: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    ep_size = group.size()
    rank = dist.get_rank(group)
    
    # Avoid hang compiling custom C++ by protecting it locally
    if rank == 0:
        _get_ext()
    dist.barrier(group)
    
    ext = _get_ext()
    
    num_local_experts = num_experts // ep_size
    expert_mask = expert_mask.contiguous()
    inner_size = expert_mask.numel() // num_experts
    
    # Pre-warm symmetrical allocation caches mapped via UVA
    buf, hdl, remote_ptrs = _get_symm_state(num_experts, expert_mask.dtype, expert_mask.device, group)
    
    # Phase 1: Local expert_mask fusion and symmetric memory load
    ext.run_reduce(expert_mask, buf, num_experts, inner_size)
    
    # Enforce symmetric memory write ordering before reading remote addresses 
    hdl.barrier(channel=0)
    
    # Phase 2: Compute full slice and distribute tokens using direct peer memory access
    out_global_tokens, out_global_sum, out_input_splits, out_output_splits = _get_tensors(
        ep_size, num_local_experts, expert_mask.dtype, expert_mask.device
    )
    
    ext.run_gather_and_split(
        remote_ptrs,
        rank,
        ep_size,
        num_local_experts,
        out_global_tokens,
        out_global_sum,
        out_input_splits,
        out_output_splits
    )
    
    # Trivial CPU-side `.tolist` guarantees the Python `List[int]` signature implicitly synchronizes 
    input_splits = out_input_splits.tolist()
    output_splits = out_output_splits.tolist()
    
    # Pinned DMA transfers to CPU overlap subsequent execution logic downstream 
    cpu_global_tokens = out_global_tokens.to(torch.device("cpu"), non_blocking=True)
    cpu_global_sum = out_global_sum.to(torch.device("cpu"), non_blocking=True)
    
    return input_splits, output_splits, cpu_global_tokens, cpu_global_sum