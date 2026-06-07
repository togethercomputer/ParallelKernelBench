import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded C++ / CUDA source
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>

using namespace kittens;

struct Pointers {
    __nv_bfloat16* ptrs[8];
};

template <typename VecT>
__global__ void p2p_alltoall_kernel(
    Pointers out_ptrs,
    const __nv_bfloat16* in_ptr,
    int D0, int D1, int D2, int D3, int D4,
    int W, int src_rank, bool S_less_than_G,
    size_t total_vecs
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vecs) return;

    // Convert 1D idx to 5D indices
    int i4 = idx % D4;
    size_t tmp = idx / D4;
    int i3 = tmp % D3;
    tmp /= D3;
    int i2 = tmp % D2;
    tmp /= D2;
    int i1 = tmp % D1;
    int i0 = tmp / D1;

    int dst_rank;
    size_t out_idx;

    if (S_less_than_G) {
        // D1 is Scatter, D3 is Gather
        int chunk_s = D1 / W;
        dst_rank = i1 / chunk_s;
        int i1_out = i1 % chunk_s;
        int i3_out = src_rank * D3 + i3;

        // out_shape = [D0, chunk_s, D2, D3 * W, D4]
        out_idx = ((((size_t)i0 * chunk_s + i1_out) * D2 + i2) * (D3 * W) + i3_out) * D4 + i4;
    } else {
        // D1 is Gather, D3 is Scatter
        int chunk_s = D3 / W;
        dst_rank = i3 / chunk_s;
        int i3_out = i3 % chunk_s;
        int i1_out = src_rank * D1 + i1;

        // out_shape = [D0, D1 * W, D2, chunk_s, D4]
        out_idx = ((((size_t)i0 * (D1 * W) + i1_out) * D2 + i2) * chunk_s + i3_out) * D4 + i4;
    }

    // Direct P2P NVLink write to destination symmetric buffer
    reinterpret_cast<VecT*>(out_ptrs.ptrs[dst_rank])[out_idx] = reinterpret_cast<const VecT*>(in_ptr)[idx];
}

namespace all_to_all_barrier {
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};
struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};
__device__ inline void kernel(const globals &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}
} // namespace all_to_all_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    torch::Tensor input,
    kittens::py::TKParallelTensor &barrier,
    int D0, int D1, int D2, int D3, int D4,
    bool S_less_than_G
) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input must be bfloat16");

    int W = all_to_all_barrier::globals::NUM_DEVICES;
    int src_rank = barrier.local_rank_;

    Pointers out_ptrs;
    for (int i = 0; i < W; ++i) {
        out_ptrs.ptrs[i] = reinterpret_cast<__nv_bfloat16*>(output.data_ptrs_[i]);
    }
    const __nv_bfloat16* in_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>());

    all_to_all_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_to_all_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Sync before P2P writes
    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);

    size_t total_elems = (size_t)D0 * D1 * D2 * D3 * D4;
    int threads = 256;
    
    if (total_elems > 0) {
        // Vectorize along the innermost contiguous dimension
        if (D4 % 8 == 0) {
            size_t total_vecs = total_elems / 8;
            int blocks = (total_vecs + threads - 1) / threads;
            p2p_alltoall_kernel<float4><<<blocks, threads>>>(
                out_ptrs, in_ptr, D0, D1, D2, D3, D4 / 8, W, src_rank, S_less_than_G, total_vecs);
        } else if (D4 % 4 == 0) {
            size_t total_vecs = total_elems / 4;
            int blocks = (total_vecs + threads - 1) / threads;
            p2p_alltoall_kernel<float2><<<blocks, threads>>>(
                out_ptrs, in_ptr, D0, D1, D2, D3, D4 / 4, W, src_rank, S_less_than_G, total_vecs);
        } else if (D4 % 2 == 0) {
            size_t total_vecs = total_elems / 2;
            int blocks = (total_vecs + threads - 1) / threads;
            p2p_alltoall_kernel<int32_t><<<blocks, threads>>>(
                out_ptrs, in_ptr, D0, D1, D2, D3, D4 / 2, W, src_rank, S_less_than_G, total_vecs);
        } else {
            size_t total_vecs = total_elems;
            int blocks = (total_vecs + threads - 1) / threads;
            p2p_alltoall_kernel<__nv_bfloat16><<<blocks, threads>>>(
                out_ptrs, in_ptr, D0, D1, D2, D3, D4, W, src_rank, S_less_than_G, total_vecs);
        }
    }

    // Sync after P2P writes to ensure visibility on destination ranks
    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_to_all", &entrypoint);
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

NUM_DEVICES = 8
ALIGNMENT = 1024 * 1024  # Cache TKParallelTensor via aligned shapes

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_ulysses_alltoall_ext",
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


def _pad_tensor(x: torch.Tensor, dim: int, padding_size: int, padding_value: int = 0) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.full(shape, padding_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


@torch.no_grad()
def solution(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    if group is None:
        return x

    assert x.is_cuda, "Input must be on CUDA device"
    
    sp_world = dist.get_world_size(group)
    assert sp_world == NUM_DEVICES, f"This ThunderKittens kernel assumes NUM_DEVICES={NUM_DEVICES}"
    
    dim_size = x.size(seq_dim)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)

    ext = _ensure_ext_jit()

    original_dtype = x.dtype
    x_bf16 = x.to(torch.bfloat16).contiguous()
    shape = list(x_bf16.shape)
    n = x_bf16.numel()

    # Pre-calculate 5D logical bounds based on scatter/gather layout splits
    s_dim = seq_dim
    g_dim = head_dim
    S_less_than_G = (s_dim < g_dim)
    
    D0, D1, D2, D3, D4 = 1, 1, 1, 1, 1
    min_dim = min(s_dim, g_dim)
    max_dim = max(s_dim, g_dim)
    
    for s in shape[:min_dim]: D0 *= s
    D1 = shape[min_dim]
    for s in shape[min_dim+1:max_dim]: D2 *= s
    D3 = shape[max_dim]
    for s in shape[max_dim+1:]: D4 *= s

    out_shape = list(shape)
    out_shape[s_dim] = shape[s_dim] // sp_world
    out_shape[g_dim] = shape[g_dim] * sp_world

    # Cache aligned parallel tensor mapping to symmetric memory for IPC handling
    padded_n = ((n + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    output_tk = get_or_create_parallel_tensor(
        ext, (padded_n,), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=sp_world)

    # Perform P2P NVLink scatter+gather transpose via custom device barrier synchronization
    ext.tk_all_to_all(
        output_tk,
        x_bf16,
        barrier_tk,
        D0, D1, D2, D3, D4,
        S_less_than_G
    )

    out = output_tk.data_[:n].clone()
    return out.view(out_shape).to(original_dtype)