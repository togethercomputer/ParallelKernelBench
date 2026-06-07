"""
Distributed link-prediction ranking using symmetric memory all-gather.

Strategy:
- Use symm_mem buffers for variable-size all-gather: each rank writes its data
  into its own slot of a symmetric buffer, then peers read directly via UVA.
- Fuse the ranking computation (sort positions of positive among negatives)
  into a single CUDA kernel that avoids materializing a full sort.
- For ranking, we only need to count how many negatives have score > pos_score
  (with sigmoid being monotonic, we compare raw scores directly).
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
#include <cstdint>

// Gather variable-size data from peer symmetric buffers via UVA.
// Each peer's buffer is `cap` elements; only the first `sizes[r]` are valid.
__global__ void gather_concat_kernel(
    const long long* __restrict__ peer_ptrs,   // [world_size]
    const long long* __restrict__ sizes,        // [world_size]
    const long long* __restrict__ offsets,      // [world_size] prefix sum
    __nv_bfloat16* __restrict__ out,            // [total * stride]
    int world_size,
    int stride,
    long long total
) {
    int r = blockIdx.y;
    if (r >= world_size) return;
    long long n_r = sizes[r];
    long long off_r = offsets[r];
    const __nv_bfloat16* src = (const __nv_bfloat16*)peer_ptrs[r];
    long long total_elems = n_r * (long long)stride;

    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = tid; i < total_elems; i += step) {
        out[off_r * (long long)stride + i] = src[i];
    }
}

// For each positive score, count how many negatives have a strictly greater
// score, plus rank-1-based: rank = 1 + count(neg > pos) + 1 (for ties handled
// like the sort: stable sort with descending puts equals after, so rank is
// 1 + count(neg > pos)). torch.sort with descending=True: equal values keep
// stable order; pos is at index 0, so it appears before equal negs.
// Thus rank = 1 + count(neg > pos).
__global__ void rank_kernel(
    const __nv_bfloat16* __restrict__ pos_scores,  // [P]
    const __nv_bfloat16* __restrict__ neg_scores,  // [P, K]
    long long* __restrict__ rankings,              // [P]
    long long P,
    long long K
) {
    long long p = blockIdx.x;
    if (p >= P) return;

    extern __shared__ long long sdata[];

    float pos_val = __bfloat162float(pos_scores[p]);
    const __nv_bfloat16* row = neg_scores + p * K;

    long long count = 0;
    for (long long k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(row[k]);
        if (v > pos_val) count++;
    }

    sdata[threadIdx.x] = count;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        rankings[p] = sdata[0] + 1;
    }
}

void launch_gather_concat(
    torch::Tensor peer_ptrs,
    torch::Tensor sizes,
    torch::Tensor offsets,
    torch::Tensor out,
    int world_size,
    int stride,
    long long total
) {
    if (total == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    long long max_per = 0;
    auto sizes_cpu = sizes.cpu();
    auto* sp = sizes_cpu.data_ptr<long long>();
    for (int i = 0; i < world_size; ++i) if (sp[i] > max_per) max_per = sp[i];
    long long max_elems = max_per * (long long)stride;
    int blocks_x = (int)std::min((long long)1024, (max_elems + threads - 1) / threads);
    if (blocks_x < 1) blocks_x = 1;
    dim3 grid(blocks_x, world_size);
    gather_concat_kernel<<<grid, threads, 0, stream>>>(
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        (const long long*)sizes.data_ptr<int64_t>(),
        (const long long*)offsets.data_ptr<int64_t>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        world_size, stride, total);
}

void launch_rank(
    torch::Tensor pos_scores,
    torch::Tensor neg_scores,
    torch::Tensor rankings,
    long long P,
    long long K
) {
    if (P == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    if (K < 256) {
        threads = 64;
        if (K >= 128) threads = 128;
    }
    if (K >= 512) threads = 512;
    dim3 grid((unsigned int)P);
    size_t shmem = threads * sizeof(long long);
    rank_kernel<<<grid, threads, shmem, stream>>>(
        (const __nv_bfloat16*)pos_scores.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)neg_scores.data_ptr<at::BFloat16>(),
        rankings.data_ptr<int64_t>(),
        P, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_concat", &launch_gather_concat, "Gather concat from peers");
    m.def("launch_rank", &launch_rank, "Compute ranks");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_neg_scoring_ext", CUDA_SRC)
    return _ext


# Symmetric memory cache: keyed by (capacity, stride, dtype)
_symm_cache = {}

def _get_symm_buf(capacity: int, stride: int, dtype: torch.dtype, device: torch.device, group):
    key = (capacity, stride, dtype, device.index)
    if key in _symm_cache:
        return _symm_cache[key]
    if stride == 1:
        shape = (capacity,)
    else:
        shape = (capacity, stride)
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]


def _all_gather_var(data: torch.Tensor, rank: int, world_size: int, group) -> torch.Tensor:
    """All-gather variable-length data along dim 0 using symmetric memory."""
    if world_size == 1:
        return data.contiguous()

    data = data.contiguous()
    local_n = data.shape[0]
    stride = 1
    if data.ndim > 1:
        for s in data.shape[1:]:
            stride *= s

    # Exchange sizes via small all_reduce (cheap, infrequent)
    sizes = torch.zeros(world_size, dtype=torch.long, device=data.device)
    sizes[rank] = local_n
    dist.all_reduce(sizes, op=dist.ReduceOp.SUM, group=group)

    sizes_cpu = sizes.cpu()
    max_n = int(sizes_cpu.max().item())
    total = int(sizes_cpu.sum().item())

    if total == 0:
        out_shape = (0,) + tuple(data.shape[1:])
        return torch.empty(out_shape, dtype=data.dtype, device=data.device)

    # Allocate symmetric buffer with capacity = max over ranks
    # Round capacity up to avoid frequent reallocations
    cap = max(max_n, 1)
    # Pad cap to next power-of-2 minimum 64 for cache friendliness
    pad_cap = 64
    while pad_cap < cap:
        pad_cap *= 2

    buf, hdl, ptrs_tensor = _get_symm_buf(pad_cap, stride, data.dtype, data.device, group)

    # Write local data into our slot
    if local_n > 0:
        if data.ndim == 1:
            buf[:local_n].copy_(data)
        else:
            buf.view(pad_cap, stride)[:local_n].copy_(data.view(local_n, stride))

    # Barrier so peers can read
    hdl.barrier(channel=0)

    # Build offsets
    offsets = torch.zeros(world_size, dtype=torch.long, device=data.device)
    cumsum = 0
    offsets_cpu = [0] * world_size
    for r in range(world_size):
        offsets_cpu[r] = cumsum
        cumsum += int(sizes_cpu[r].item())
    offsets.copy_(torch.tensor(offsets_cpu, dtype=torch.long))

    out_shape = (total,) + tuple(data.shape[1:])
    out = torch.empty(out_shape, dtype=data.dtype, device=data.device)

    sizes_dev = sizes_cpu.to(data.device)

    _get_ext().launch_gather_concat(
        ptrs_tensor, sizes_dev, offsets, out.view(total, stride) if data.ndim > 1 else out.view(total, 1),
        world_size, stride, total
    )

    hdl.barrier(channel=1)
    return out


@torch.no_grad()
def solution(
    local_pos_scores: torch.Tensor,
    local_neg_scores: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # Ensure extension compiled (rank 0 first to avoid race on shared cache)
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()

    pos_scores = _all_gather_var(local_pos_scores, rank, world_size, group)
    neg_scores = _all_gather_var(local_neg_scores, rank, world_size, group)

    P = pos_scores.shape[0]
    if P == 0:
        return torch.empty(0, dtype=torch.long, device=pos_scores.device)

    K = neg_scores.shape[1] if neg_scores.ndim > 1 else 0

    # Match reference dtype exactly
    if pos_scores.dtype != torch.bfloat16:
        # Fallback: use reference path for non-bf16
        scores = torch.cat([pos_scores.view(-1, 1), neg_scores], dim=1)
        _, indices = torch.sort(torch.sigmoid(scores), dim=1, descending=True)
        return torch.nonzero(indices == 0)[:, 1].view(-1).detach() + 1

    rankings = torch.empty(P, dtype=torch.long, device=pos_scores.device)
    pos_c = pos_scores.contiguous()
    neg_c = neg_scores.contiguous()
    _get_ext().launch_rank(pos_c, neg_c, rankings, P, K)
    return rankings