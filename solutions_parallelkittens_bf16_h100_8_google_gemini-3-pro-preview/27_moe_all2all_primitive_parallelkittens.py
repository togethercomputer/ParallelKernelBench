"""
Strategy: We exploit device-side P2P PULL via TKParallelTensor to bypass NCCL and avoid any CPU overhead.
1. We allocate a persistent TKParallelTensor for the inputs and an 8-element split sizes array, pre-exchanging NVLink handles.
2. We map a device-side sizes-gather and a vectorized PULL kernel to the default stream, executing purely on the device.
3. Each block is assigned to an output row, computing its remote offset locally using the gathered sizes and pulling 16-byte (float4) vectorized chunks directly into the dynamically allocated PyTorch output tensor.
4. ThunderKittens barriers wrap the kernel to ensure safe NVLink accesses without CPU-side synchronization.
"""

import os
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>
#include <torch/extension.h>

using namespace kittens;

struct globals {
    static constexpr int NUM_DEVICES = 8;
    bf16* input_ptrs[NUM_DEVICES];
    bf16* local_output_ptr;
    int* splits_ptrs[NUM_DEVICES];
    int* local_splits;
    int hidden_dim;
    int dev_idx;
    int total_output_rows;
    barrier_t<NUM_DEVICES> barrier;
};

__global__ void gather_splits_kernel(globals G) {
    // 1 block, 64 threads fetches the 8x8 size matrix into a local array
    if (threadIdx.x < 64) {
        int i = threadIdx.x / 8;
        int j = threadIdx.x % 8;
        G.local_splits[threadIdx.x] = G.splits_ptrs[i][j];
    }
}

__global__ void all_to_all_kernel(globals G) {
    int row = blockIdx.x;
    if (row >= G.total_output_rows) return;
    
    // Find which remote rank this row comes from
    int src_rank = G.NUM_DEVICES - 1;
    int row_offset_in_chunk = row;
    for (int i = 0; i < G.NUM_DEVICES; ++i) {
        int size = G.local_splits[i * G.NUM_DEVICES + G.dev_idx];
        if (row_offset_in_chunk < size) {
            src_rank = i;
            break;
        }
        row_offset_in_chunk -= size;
    }
    
    // Compute the base row offset on the remote rank
    int src_base_row = 0;
    for (int j = 0; j < G.dev_idx; ++j) {
        src_base_row += G.local_splits[src_rank * G.NUM_DEVICES + j];
    }
    
    int src_row = src_base_row + row_offset_in_chunk;
    
    bf16* src_row_ptr = G.input_ptrs[src_rank] + src_row * G.hidden_dim;
    bf16* dst_row_ptr = G.local_output_ptr + row * G.hidden_dim;
    
    // Vectorized copy using float4 (16 bytes = 8 bf16s) if perfectly aligned
    if (G.hidden_dim % 8 == 0) {
        float4* s = reinterpret_cast<float4*>(src_row_ptr);
        float4* d = reinterpret_cast<float4*>(dst_row_ptr);
        int cols = G.hidden_dim / 8;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            d[i] = s[i];
        }
    } else if (G.hidden_dim % 4 == 0) {
        float2* s = reinterpret_cast<float2*>(src_row_ptr);
        float2* d = reinterpret_cast<float2*>(dst_row_ptr);
        int cols = G.hidden_dim / 4;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            d[i] = s[i];
        }
    } else if (G.hidden_dim % 2 == 0) {
        float* s = reinterpret_cast<float*>(src_row_ptr);
        float* d = reinterpret_cast<float*>(dst_row_ptr);
        int cols = G.hidden_dim / 2;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            d[i] = s[i];
        }
    } else {
        // Fallback for odd hidden dims
        for (int i = threadIdx.x; i < G.hidden_dim; i += blockDim.x) {
            dst_row_ptr[i] = src_row_ptr[i];
        }
    }
}

__global__ void barrier_kernel(globals G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

void entrypoint(
    kittens::py::TKParallelTensor &input_tk,
    torch::Tensor &output_tensor,
    kittens::py::TKParallelTensor &splits_tk,
    kittens::py::TKParallelTensor &barrier_tk,
    torch::Tensor &local_splits,
    int hidden_dim,
    int total_output_rows
) {
    globals G;
    G.hidden_dim = hidden_dim;
    G.dev_idx = input_tk.local_rank_;
    G.total_output_rows = total_output_rows;
    G.local_output_ptr = reinterpret_cast<bf16*>(output_tensor.data_ptr<at::BFloat16>());
    G.local_splits = local_splits.data_ptr<int>();
    
    for (int i = 0; i < globals::NUM_DEVICES; ++i) {
        G.input_ptrs[i] = reinterpret_cast<bf16*>(input_tk.ptrs_[i]);
        G.splits_ptrs[i] = reinterpret_cast<int*>(splits_tk.ptrs_[i]);
    }
    
    G.barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<globals::NUM_DEVICES>>(barrier_tk);

    // Ensure all peers have completed their host->device split sizes and D2D tensor copies
    barrier_kernel<<<1, 1>>>(G);
    
    // Pull the 8x8 splits matrix into local GPU memory for fast access by the blocks
    gather_splits_kernel<<<1, 64>>>(G);

    // Launch one block per output row
    int num_blocks = total_output_rows;
    if (num_blocks > 0) {
        int threads_per_block = 256;
        all_to_all_kernel<<<num_blocks, threads_per_block>>>(G);
    }
    
    // Sync to ensure everyone is done pulling from our input_tk before proceeding
    barrier_kernel<<<1, 1>>>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_moe_all_to_all", &entrypoint);
}
'''

TK_CUDA_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-DNDEBUG",
]

_ext = None
_ext_jit_ready = False

# Fallback allocation guard: allocate up to 128M elements (~256MB) persistently for input payloads
MAX_ELEMS = 128 * 1024 * 1024 


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_moe_alltoall_ext",
            CUDA_SRC,
            extra_cuda_cflags=TK_CUDA_FLAGS,
            extra_include_paths=[
                os.path.join(TK_ROOT, "include"),
                os.path.join(TK_ROOT, "prototype"),
            ],
            extra_ldflags=["-lcuda"],
        )
    return _ext


def _ensure_ext_jit():
    global _ext_jit_ready
    if _ext_jit_ready:
        return _get_ext()
    rank = dist.get_rank()
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


def solution(
    local_tensor: torch.Tensor,
    input_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    output_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return local_tensor.contiguous()

    local_tensor = local_tensor.contiguous()
    hidden_dim = local_tensor.size(1)

    if output_split_sizes is None:
        out_size = local_tensor.size(0)
    else:
        out_size = sum(output_split_sizes) if isinstance(output_split_sizes, list) else int(output_split_sizes.sum().item())
        
    output = torch.empty(
        (out_size, hidden_dim),
        dtype=local_tensor.dtype,
        device=local_tensor.device,
    )

    # Check alignment + bounds; fallback dynamically if size or environment deviates
    if (world_size != 8 or 
        local_tensor.dtype != torch.bfloat16 or 
        local_tensor.numel() > MAX_ELEMS):
        dist.all_to_all_single(
            output,
            local_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    ext = _ensure_ext_jit()

    if input_split_sizes is None:
        in_splits = torch.tensor([local_tensor.size(0) // world_size] * world_size, dtype=torch.int32, device=local_tensor.device)
    elif isinstance(input_split_sizes, list):
        in_splits = torch.tensor(input_split_sizes, dtype=torch.int32, device=local_tensor.device)
    else:
        in_splits = input_split_sizes.to(torch.int32).to(local_tensor.device)

    # Request our persistent symmetric memory blocks
    input_tk = get_or_create_parallel_tensor(ext, (MAX_ELEMS,), torch.bfloat16, multicast=False)
    splits_tk = get_or_create_parallel_tensor(ext, (8,), torch.int32, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    local_splits = torch.empty((8, 8), dtype=torch.int32, device=local_tensor.device)

    # Queue input tensor copies async onto the default stream
    numel_in = local_tensor.numel()
    if numel_in > 0:
        input_tk.data_[:numel_in].copy_(local_tensor.view(-1))
    
    splits_tk.data_[:world_size].copy_(in_splits)

    # Launch fused P2P PULL kernel
    ext.tk_moe_all_to_all(
        input_tk,
        output,
        splits_tk,
        barrier_tk,
        local_splits,
        hidden_dim,
        out_size
    )

    return output