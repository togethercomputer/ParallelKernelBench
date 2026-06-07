"""
Strategy:
1. Overlap split exchange with local compute: We compute the histogram of destination ranks (`send_splits`), and asynchronously `all_gather` these splits across all ranks while independently sorting the local data (`owner`, `idx`, `value`) to group it by destination rank.
2. Global deterministic offsets: Since the all-gather gives every rank the complete communication matrix, each rank can deterministically compute the exact read/write offsets for every peer without any further coordinate exchange.
3. Direct Device-to-Device (P2P) Push: We allocate a symmetric buffer (`TKParallelTensor`) large enough for the maximum received size, padded to a power of 2 for VMM reuse. A custom ThunderKittens CUDA kernel directly pushes each peer's chunk into its exact final offset in the peer's destination buffer using P2P NVLink stores.
4. Barrier & Slice: A single device-side barrier ensures all P2P writes have landed. Each rank simply slices the exact number of elements it received, avoiding all variable-length padding artifacts and host synchronization overheads associated with NCCL's variable-length `all_to_all_single`.
"""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source for Direct P2P Push
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>

using namespace kittens;

// Barrier namespace
namespace tk_barrier {
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
}

// Device-to-Device (P2P) Push Kernel
// Each block handles copying the packed chunk for a specific peer directly into their destination memory.
template <typename T>
__global__ void p2p_push_kernel(
    T* dst_0, T* dst_1, T* dst_2, T* dst_3,
    T* dst_4, T* dst_5, T* dst_6, T* dst_7,
    const T* src_data,
    const int* src_offsets,
    const int* dst_offsets,
    const int* counts,
    int D
) {
    T* dst_ptrs[8] = {dst_0, dst_1, dst_2, dst_3, dst_4, dst_5, dst_6, dst_7};
    
    int dst_rank = blockIdx.y;
    int count = counts[dst_rank];
    if (count == 0) return;

    int total_elems = count * D;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    T* dst = dst_ptrs[dst_rank];
    int src_off = src_offsets[dst_rank] * D;
    int dst_off = dst_offsets[dst_rank] * D;

    // Fast path: use vectorized uint4 copies if aligned
    if (reinterpret_cast<uintptr_t>(dst) % 16 == 0 && 
        reinterpret_cast<uintptr_t>(src_data) % 16 == 0 &&
        src_off % (16 / sizeof(T)) == 0 &&
        dst_off % (16 / sizeof(T)) == 0 &&
        total_elems % (16 / sizeof(T)) == 0) {
        
        int4* dst_vec = reinterpret_cast<int4*>(dst + dst_off);
        const int4* src_vec = reinterpret_cast<const int4*>(src_data + src_off);
        int vec_elems = total_elems / (16 / sizeof(T));
        
        for (int i = tid; i < vec_elems; i += stride) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        // Fallback flat copy
        for (int i = tid; i < total_elems; i += stride) {
            dst[dst_off + i] = src_data[src_off + i];
        }
    }
}

void entrypoint(
    kittens::py::TKParallelTensor &tk_dst_idx,
    kittens::py::TKParallelTensor &tk_dst_val,
    torch::Tensor src_idx,
    torch::Tensor src_val,
    torch::Tensor src_offsets,
    torch::Tensor dst_offsets,
    torch::Tensor counts,
    kittens::py::TKParallelTensor &barrier,
    int D_idx,
    int D_val
) {
    int dev_idx = tk_dst_idx.local_rank_;
    
    tk_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<tk_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = dev_idx
    };

    // Extract raw pointers for index (int64_t)
    int64_t* dst_idx_ptrs[8];
    for(int i=0; i<8; ++i) {
        dst_idx_ptrs[i] = reinterpret_cast<int64_t*>(tk_dst_idx.data_ptrs_[i]);
    }

    // Extract raw pointers for values (__nv_bfloat16)
    __nv_bfloat16* dst_val_ptrs[8];
    for(int i=0; i<8; ++i) {
        dst_val_ptrs[i] = reinterpret_cast<__nv_bfloat16*>(tk_dst_val.data_ptrs_[i]);
    }

    // 1. Barrier before write to ensure destination VMM buffers are ready and untouched by previous rounds
    kittens::py::launch_kernel<tk_barrier::config, tk_barrier::globals, tk_barrier::kernel>(barrier_G);

    // 2. Launch NVLink direct P2P writes
    dim3 grid(32, 8); // 32 blocks per peer, 8 peers (y-dimension)
    dim3 block(256);
    
    if (src_idx.numel() > 0) {
        p2p_push_kernel<int64_t><<<grid, block>>>(
            dst_idx_ptrs[0], dst_idx_ptrs[1], dst_idx_ptrs[2], dst_idx_ptrs[3],
            dst_idx_ptrs[4], dst_idx_ptrs[5], dst_idx_ptrs[6], dst_idx_ptrs[7],
            src_idx.data_ptr<int64_t>(),
            src_offsets.data_ptr<int>(),
            dst_offsets.data_ptr<int>(),
            counts.data_ptr<int>(),
            D_idx
        );
    }

    if (src_val.numel() > 0) {
        p2p_push_kernel<__nv_bfloat16><<<grid, block>>>(
            dst_val_ptrs[0], dst_val_ptrs[1], dst_val_ptrs[2], dst_val_ptrs[3],
            dst_val_ptrs[4], dst_val_ptrs[5], dst_val_ptrs[6], dst_val_ptrs[7],
            reinterpret_cast<const __nv_bfloat16*>(src_val.data_ptr<at::BFloat16>()),
            src_offsets.data_ptr<int>(),
            dst_offsets.data_ptr<int>(),
            counts.data_ptr<int>(),
            D_val
        );
    }

    // 3. Barrier after write to ensure all data has landed before the host attempts to slice
    kittens::py::launch_kernel<tk_barrier::config, tk_barrier::globals, tk_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_p2p_push", &entrypoint);
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


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_p2p_push_ext",
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
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


@torch.no_grad()
def solution(
    idx: torch.Tensor,
    value: torch.Tensor,
    num_nodes: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return idx, value

    assert world_size == 8, "This ThunderKittens integration expects exactly 8 ranks per node."
    me = dist.get_rank(group)

    idx_orig_dtype = idx.dtype
    if idx.dtype != torch.int64:
        idx = idx.to(torch.int64)

    # 1. Bucket local updates (launch stream: default)
    owner = (idx % world_size).long()
    send_splits = torch.bincount(owner, minlength=world_size)

    # 2. Async All-Gather of splits. Exchanges only 8 elements per rank.
    all_send_splits = torch.empty(world_size, world_size, dtype=torch.long, device=idx.device)
    gather_work = dist.all_gather_into_tensor(all_send_splits, send_splits, group=group, async_op=True)

    # 3. Overlap compute while Gather is in-flight: Sort & Pack based on destination rank.
    perm = torch.argsort(owner, stable=True)
    send_idx_packed = idx[perm].contiguous()
    send_value_packed = value[perm]

    # 4. Wait for split sizes. Since all ranks now have the FULL exchange matrix, offsets are deterministic.
    gather_work.wait()

    # Derived read/write coordinates entirely via local math:
    src_offsets = torch.zeros(world_size, dtype=torch.int32, device=idx.device)
    if world_size > 1:
        src_offsets[1:] = torch.cumsum(all_send_splits[me, :-1], dim=0).to(torch.int32)

    dst_offsets = all_send_splits[:me, :].sum(dim=0).to(torch.int32).contiguous()
    counts = all_send_splits[me, :].to(torch.int32).contiguous()

    my_recv_count = int(all_send_splits[:, me].sum().item())
    global_max_recv = int(all_send_splits.sum(dim=0).max().item())

    if global_max_recv == 0:
        return (
            torch.empty((0,), dtype=idx_orig_dtype, device=idx.device),
            torch.empty((0, *value.shape[1:]), dtype=value.dtype, device=value.device)
        )

    # 5. Get TK symmetric memory. Pad to next power of 2 for stable caching of Virtual Memory allocations.
    ext = _ensure_ext_jit()
    padded_max_recv = max(256, 1 << (global_max_recv - 1).bit_length())

    tk_dst_idx = get_or_create_parallel_tensor(ext, (padded_max_recv,), torch.int64, multicast=False)

    D = value.shape[1:].numel() if value.dim() > 1 else 1
    val_dtype = torch.bfloat16
    send_value_packed_bf16 = send_value_packed.to(val_dtype).reshape(-1, D).contiguous()
    tk_dst_val = get_or_create_parallel_tensor(ext, (padded_max_recv, D), val_dtype, multicast=False)

    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    # 6. Execute direct device-to-device push (bypassing PyTorch host looping and variable-size collectives)
    ext.tk_p2p_push(
        tk_dst_idx,
        tk_dst_val,
        send_idx_packed,
        send_value_packed_bf16,
        src_offsets,
        dst_offsets,
        counts,
        barrier_tk,
        1,
        D
    )

    # 7. Zero-overhead slice to exact boundary based on deterministically computed sizes
    recv_idx = tk_dst_idx.data_[:my_recv_count].clone().to(idx_orig_dtype)
    recv_value = tk_dst_val.data_[:my_recv_count].reshape(my_recv_count, *value.shape[1:]).clone().to(value.dtype)

    return recv_idx, recv_value