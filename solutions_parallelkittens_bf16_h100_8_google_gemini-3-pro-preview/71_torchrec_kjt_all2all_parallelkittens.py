"""
ThunderKittens KJT All-To-All Pull-Permute.

Replaces the NCCL all-to-all and PyTorch segment-gather permutation loops with
a fused PGL NVLink kernel. The kernel directly pulls variable-sized jagged segments 
from peer workspaces into their final permuted output locations.
"""

import os
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source for Pull-Permute Kernels
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <torch/extension.h>

using namespace kittens;

namespace barrier_ns {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int NUM_THREADS = 1;
    };
    struct globals {
        static constexpr int NUM_DEVICES = 8;
        barrier_t<NUM_DEVICES> barrier;
        const int dev_idx;
    };
    __device__ void kernel(const globals& G) {
        barrier_all(G.barrier, {0}, G.dev_idx);
    }
}

template<typename T>
struct pull_permute_globals {
    static constexpr int NUM_DEVICES = 8;
    // PGL array of rank memory blocks
    using parallel_layout = pgl<gl<T, -1, -1, -1, -1>, NUM_DEVICES, false>;
    
    parallel_layout send_workspace;
    T* out;
    
    const int* dst_offsets;
    const int* peer_ids;
    const int* src_offsets;
    
    int num_segments;
    int total_elems;
    int alloc_size;
    int rank;

    __host__ inline dim3 grid() const {
        return dim3((total_elems + 255) / 256);
    }
};

namespace pull_permute {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int NUM_THREADS = 256;
    };
}

template<typename T>
__device__ inline void pull_permute_kernel(const pull_permute_globals<T> &G) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= G.total_elems) return;
    
    // Binary search to find which segment this thread's element belongs to
    int low = 0, high = G.num_segments - 1;
    while (low < high) {
        int mid = (low + high + 1) / 2;
        if (G.dst_offsets[mid] <= tid) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    int seg = low;
    
    int peer_id = G.peer_ids[seg];
    int offset_in_seg = tid - G.dst_offsets[seg];
    int src_offset = G.src_offsets[seg] + offset_in_seg;
    
    // Direct NVLink read from the peer's workspace
    const T* src_base = reinterpret_cast<const T*>(G.send_workspace[peer_id].data);
    
    // The workspace on the peer is structured as 8 blocks of size `alloc_size`
    G.out[tid] = src_base[G.rank * G.alloc_size + src_offset];
}

void tk_barrier_entry(kittens::py::TKParallelTensor &barrier) {
    barrier_ns::globals G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<8>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    kittens::py::launch_kernel<barrier_ns::config, barrier_ns::globals, barrier_ns::kernel>(G);
}

void entrypoint_bf16(
    kittens::py::TKParallelTensor &workspace,
    torch::Tensor &out,
    torch::Tensor &dst_offsets,
    torch::Tensor &peer_ids,
    torch::Tensor &src_offsets,
    int num_segments,
    int total_elems,
    int alloc_size,
    int rank
) {
    pull_permute_globals<bf16> G {
        .send_workspace = kittens::py::parallel_tensor_to_pgl<typename pull_permute_globals<bf16>::parallel_layout>(workspace),
        .out = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
        .dst_offsets = dst_offsets.data_ptr<int>(),
        .peer_ids = peer_ids.data_ptr<int>(),
        .src_offsets = src_offsets.data_ptr<int>(),
        .num_segments = num_segments,
        .total_elems = total_elems,
        .alloc_size = alloc_size,
        .rank = rank
    };
    kittens::py::launch_kernel<pull_permute::config, pull_permute_globals<bf16>, pull_permute_kernel<bf16>>(G);
}

void entrypoint_int32(
    kittens::py::TKParallelTensor &workspace,
    torch::Tensor &out,
    torch::Tensor &dst_offsets,
    torch::Tensor &peer_ids,
    torch::Tensor &src_offsets,
    int num_segments,
    int total_elems,
    int alloc_size,
    int rank
) {
    pull_permute_globals<int> G {
        .send_workspace = kittens::py::parallel_tensor_to_pgl<typename pull_permute_globals<int>::parallel_layout>(workspace),
        .out = out.data_ptr<int>(),
        .dst_offsets = dst_offsets.data_ptr<int>(),
        .peer_ids = peer_ids.data_ptr<int>(),
        .src_offsets = src_offsets.data_ptr<int>(),
        .num_segments = num_segments,
        .total_elems = total_elems,
        .alloc_size = alloc_size,
        .rank = rank
    };
    kittens::py::launch_kernel<pull_permute::config, pull_permute_globals<int>, pull_permute_kernel<int>>(G);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_barrier", &tk_barrier_entry);
    m.def("tk_pull_permute_bf16", &entrypoint_bf16);
    m.def("tk_pull_permute_int32", &entrypoint_int32);
}
'''

TK_CUDA_FLAGS = [
    "-std=c++20", "--use_fast_math", "--expt-extended-lambda", "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER", "-gencode", "arch=compute_90a,code=sm_90a",
    "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-Wno-psabi", "-Xcompiler=-fno-strict-aliasing", "-DNDEBUG",
]

_ext = None
_ext_jit_ready = False
_tk_workspaces = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_kjt_all2all",
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


def _ensure_workspace(name: str, dtype: torch.dtype, req_size: int, ext):
    global _tk_workspaces
    if name not in _tk_workspaces or _tk_workspaces[name].data_.shape[-1] < req_size:
        alloc_size = max(req_size, 4 * 1024 * 1024)
        _tk_workspaces[name] = get_or_create_parallel_tensor(
            ext, (8, alloc_size), dtype, multicast=False
        )
    return _tk_workspaces[name]


def _get_recat(
    local_split: int,
    num_splits: int,
    stagger: int = 1,
    device: Optional[torch.device] = None,
    batch_size_per_rank: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    if local_split == 0:
        return None
    feature_order = [
        x + num_splits // stagger * y
        for x in range(num_splits // stagger)
        for y in range(stagger)
    ]
    if batch_size_per_rank is None:
        recat = [
            feature_idx + rank_idx * local_split
            for feature_idx in range(local_split)
            for rank_idx in feature_order
        ]
    else:
        rank_offsets = [0]
        for batch_size in batch_size_per_rank[:-1]:
            rank_offsets.append(rank_offsets[-1] + local_split * batch_size)
        recat = [
            rank_offsets[rank_idx] + feature_idx * batch_size_per_rank[rank_idx] + b
            for feature_idx in range(local_split)
            for rank_idx in feature_order
            for b in range(batch_size_per_rank[rank_idx])
        ]
    return torch.tensor(recat, device=device, dtype=torch.long)


def _sum_by_splits(values: List[int], splits: List[int]) -> List[int]:
    out: List[int] = []
    offset = 0
    for split in splits:
        out.append(sum(values[offset : offset + split]))
        offset += split
    return out


def _lengths_per_key(lengths: torch.Tensor, stride_per_key: List[int]) -> List[int]:
    out: List[int] = []
    offset = 0
    for stride in stride_per_key:
        out.append(int(lengths[offset : offset + stride].sum().item()))
        offset += stride
    return out


@torch.no_grad()
def solution(
    lengths: torch.Tensor,
    values: torch.Tensor,
    key_splits: List[int],
    batch_size: int,
    pg: Optional[dist.ProcessGroup] = None,
    weights: Optional[torch.Tensor] = None,
    stride_per_key: Optional[List[int]] = None,
    stagger: int = 1,
) -> Dict[str, torch.Tensor]:
    pg = pg or dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    device = lengths.device
    ext = _ensure_ext_jit()

    num_features = sum(key_splits)
    variable_stride = stride_per_key is not None
    if stride_per_key is None:
        stride_per_key = [batch_size] * num_features

    length_per_key = _lengths_per_key(lengths, stride_per_key)
    length_splits = _sum_by_splits(stride_per_key, key_splits)
    value_splits = _sum_by_splits(length_per_key, key_splits)

    input_splits = [length_splits, value_splits]
    if variable_stride:
        input_splits.append(key_splits)
    
    split_tensors = [
        torch.tensor(splits, dtype=torch.long, device=device) for splits in input_splits
    ]
    if not variable_stride:
        split_tensors.append(
            torch.full((world_size,), batch_size, dtype=torch.long, device=device)
        )
    if variable_stride:
        stride_per_key_tensor = torch.tensor(stride_per_key, dtype=torch.long, device=device)

    meta_input = torch.stack(split_tensors, dim=1).flatten()
    meta_output = torch.empty_like(meta_input)
    dist.all_to_all_single(meta_output, meta_input, group=pg)
    
    meta_rows = [
        [int(item) for item in row]
        for row in meta_output.view(world_size, -1).T.tolist()
    ]
    
    if variable_stride:
        output_splits = meta_rows
        stride_per_rank = None
        # output_splits[2] is received stride_per_key arrays from each rank
        recv_strides_t = torch.tensor(output_splits[2], device=device, dtype=torch.long)
    else:
        output_splits = meta_rows[:-1]
        stride_per_rank = meta_rows[-1]

    # Pre-allocate distributed TK parallel tensors for P2P routing
    max_l_req = max(length_splits)
    max_v_req = max(value_splits)
    
    max_reqs = torch.tensor([max_l_req, max_v_req], device=device, dtype=torch.long)
    dist.all_reduce(max_reqs, op=dist.ReduceOp.MAX)
    global_req_l, global_req_v = max_reqs.tolist()

    ws_l = _ensure_workspace("lengths", torch.int32, global_req_l, ext)
    ws_v = _ensure_workspace("values", torch.bfloat16, global_req_v, ext)
    ws_w = _ensure_workspace("weights", torch.bfloat16, global_req_v, ext) if weights is not None else None
    ws_barrier = get_or_create_barrier(ext, num_devices=world_size)

    # 1. Scatter payloads locally into the outbound P2P registered windows
    l_start, v_start = 0, 0
    lengths_i32 = lengths.to(torch.int32)
    for j in range(world_size):
        l_len, v_len = length_splits[j], value_splits[j]
        ws_l.data_[j, :l_len].copy_(lengths_i32[l_start : l_start + l_len])
        ws_v.data_[j, :v_len].copy_(values[v_start : v_start + v_len])
        if weights is not None:
            ws_w.data_[j, :v_len].copy_(weights[v_start : v_start + v_len])
        l_start += l_len
        v_start += v_len

    local_split = key_splits[rank]
    
    # 2. Determine generalized Length Segments layout
    if variable_stride:
        recat_lengths = _get_recat(local_split, world_size, stagger, device=device)
        if recat_lengths is None:
            recat_lengths = torch.arange(world_size * local_split, device=device)
        lengths_segment_sizes = recv_strides_t.flatten()
        S_lengths = world_size * local_split
        l_peer_block_sizes = [local_split] * world_size
    else:
        single_batch_per_rank = all(stride == stride_per_rank[0] for stride in stride_per_rank)
        if single_batch_per_rank:
            recat_lengths = _get_recat(local_split, world_size, stagger, device=device)
            if recat_lengths is None:
                recat_lengths = torch.arange(world_size * local_split, device=device)
            lengths_segment_sizes = torch.full((world_size * local_split,), stride_per_rank[0], device=device, dtype=torch.long)
            S_lengths = world_size * local_split
            l_peer_block_sizes = [local_split] * world_size
        else:
            recat_lengths = _get_recat(local_split, world_size, stagger, device=device, batch_size_per_rank=stride_per_rank)
            if recat_lengths is None:
                total_len = sum(local_split * b for b in stride_per_rank)
                recat_lengths = torch.arange(total_len, device=device)
            S_lengths = len(recat_lengths)
            lengths_segment_sizes = torch.ones(S_lengths, device=device, dtype=torch.long)
            l_peer_block_sizes = [local_split * b for b in stride_per_rank]

    # Pre-calculate segment coordinates logically
    l_peer_ids_unpermuted = torch.tensor(
        [i for i, size in enumerate(l_peer_block_sizes) for _ in range(size)],
        device=device, dtype=torch.int32
    )
    
    l_global_offsets = torch.zeros(S_lengths + 1, dtype=torch.long, device=device)
    l_global_offsets[1:] = torch.cumsum(lengths_segment_sizes, dim=0)
    l_src_offsets_unpermuted = l_global_offsets[:-1].clone()
    
    base_idx = 0
    for i in range(world_size):
        block_len = l_peer_block_sizes[i]
        l_src_offsets_unpermuted[base_idx : base_idx + block_len] -= l_global_offsets[base_idx]
        base_idx += block_len

    # Apply permutation to map segments to the output ordering
    l_peer_ids = l_peer_ids_unpermuted[recat_lengths]
    l_src_offsets = l_src_offsets_unpermuted[recat_lengths].to(torch.int32)
    l_permuted_sizes = lengths_segment_sizes[recat_lengths]
    l_dst_offsets = torch.zeros(S_lengths + 1, dtype=torch.int32, device=device)
    l_dst_offsets[1:] = torch.cumsum(l_permuted_sizes, dim=0).to(torch.int32)
    total_l = l_dst_offsets[-1].item()

    lengths_out = torch.empty(total_l, device=device, dtype=torch.int32)

    # 3. Stream data from peers using ParallelKittens
    ext.tk_barrier(ws_barrier)
    if total_l > 0:
        ext.tk_pull_permute_int32(
            ws_l, lengths_out, l_dst_offsets, l_peer_ids, l_src_offsets,
            S_lengths, total_l, ws_l.data_.shape[-1], rank
        )

    # 4. Resolve Value Segment boundaries (driven purely by the fully-permuted lengths)
    lengths_out_long = lengths_out.to(torch.long)
    l_cumsum = torch.zeros(total_l + 1, dtype=torch.long, device=device)
    l_cumsum[1:] = torch.cumsum(lengths_out_long, dim=0)
    
    v_permuted_sizes = l_cumsum[l_dst_offsets[1:].long()] - l_cumsum[l_dst_offsets[:-1].long()]
    inverse_recat = torch.argsort(recat_lengths)
    v_sizes_unpermuted = v_permuted_sizes[inverse_recat]

    v_global_offsets = torch.zeros(S_lengths + 1, dtype=torch.long, device=device)
    v_global_offsets[1:] = torch.cumsum(v_sizes_unpermuted, dim=0)
    v_src_offsets_unpermuted = v_global_offsets[:-1].clone()
    
    base_idx = 0
    for i in range(world_size):
        block_len = l_peer_block_sizes[i]
        v_src_offsets_unpermuted[base_idx : base_idx + block_len] -= v_global_offsets[base_idx]
        base_idx += block_len

    v_src_offsets = v_src_offsets_unpermuted[recat_lengths].to(torch.int32)
    v_dst_offsets = torch.zeros(S_lengths + 1, dtype=torch.int32, device=device)
    v_dst_offsets[1:] = torch.cumsum(v_permuted_sizes, dim=0).to(torch.int32)
    total_v = v_dst_offsets[-1].item()

    values_out = torch.empty(total_v, device=device, dtype=torch.bfloat16)

    if total_v > 0:
        ext.tk_pull_permute_bf16(
            ws_v, values_out, v_dst_offsets, l_peer_ids, v_src_offsets,
            S_lengths, total_v, ws_v.data_.shape[-1], rank
        )
    weights_out = None
    if weights is not None:
        weights_out = torch.empty(total_v, device=device, dtype=torch.bfloat16)
        if total_v > 0:
            ext.tk_pull_permute_bf16(
                ws_w, weights_out, v_dst_offsets, l_peer_ids, v_src_offsets,
                S_lengths, total_v, ws_w.data_.shape[-1], rank
            )

    result: Dict[str, torch.Tensor] = {
        "lengths": lengths_out,
        "values": values_out,
    }

    if variable_stride:
        stride_per_key_per_rank = recv_strides_t.T
        if stagger > 1:
            order = torch.arange(world_size, device=device).view(stagger, -1).T.reshape(-1)
            stride_per_key_per_rank = stride_per_key_per_rank[:, order]
        result["stride_per_key_per_rank"] = stride_per_key_per_rank
    else:
        result["stride"] = torch.tensor(sum(stride_per_rank), device=device)
        result["stride_per_rank"] = torch.tensor(stride_per_rank, device=device)
        
    if weights is not None:
        result["weights"] = weights_out

    return result