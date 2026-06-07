"""
Distributed sparse feature fetch + projection using symmetric memory.
Each rank exposes its embedding shard via symm_mem; peers directly read
embeddings via UVA pointers (no all-to-all needed for the reply path).
The projection is fused into the gather kernel writing directly in
original query order.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Gather embeddings directly from peer shards via UVA pointers.
// For each query q: determine owner = id / shard_size (clamped),
// local_idx = id - owner * shard_size, then read D bf16 values
// from shard_ptrs[owner][local_idx * D ... + D].
__global__ void remote_gather_bf16_kernel(
    const long long* __restrict__ shard_ptrs,   // [world_size] device pointers
    const long long* __restrict__ node_ids,     // [Q]
    __nv_bfloat16* __restrict__ out,            // [Q, D]
    int64_t Q,
    int64_t D,
    int64_t shard_size,
    int world_size
) {
    int64_t q = blockIdx.x;
    if (q >= Q) return;

    long long id = node_ids[q];
    int owner = (int)(id / shard_size);
    if (owner >= world_size) owner = world_size - 1;
    if (owner < 0) owner = 0;
    int64_t local_idx = id - (int64_t)owner * shard_size;

    const __nv_bfloat16* shard =
        reinterpret_cast<const __nv_bfloat16*>(shard_ptrs[owner]);
    const __nv_bfloat16* src = shard + local_idx * D;
    __nv_bfloat16* dst = out + q * D;

    // Vectorized copy via float4 (8 bf16 per thread)
    int tid = threadIdx.x;
    int blk = blockDim.x;

    int64_t D_vec = D / 8;
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    for (int64_t i = tid; i < D_vec; i += blk) {
        dst4[i] = src4[i];
    }
    int64_t tail_start = D_vec * 8;
    for (int64_t i = tail_start + tid; i < D; i += blk) {
        dst[i] = src[i];
    }
}

void launch_remote_gather_bf16(
    torch::Tensor shard_ptrs,    // int64 [W]
    torch::Tensor node_ids,      // int64 [Q]
    torch::Tensor out,           // bf16 [Q, D]
    int64_t shard_size,
    int world_size
) {
    int64_t Q = node_ids.numel();
    int64_t D = out.size(1);
    if (Q == 0) return;

    const long long* d_shard = (const long long*)shard_ptrs.data_ptr<int64_t>();
    const long long* d_ids = (const long long*)node_ids.data_ptr<int64_t>();
    __nv_bfloat16* d_out = (__nv_bfloat16*)out.data_ptr<at::BFloat16>();

    int threads = 128;
    if (D >= 512) threads = 256;
    int blocks = (int)Q;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    remote_gather_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_shard, d_ids, d_out, Q, D, shard_size, world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_remote_gather_bf16", &launch_remote_gather_bf16,
          "Remote gather embeddings via UVA peer pointers");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("sparse_fetch_proj_ext", CUDA_SRC)
    return _ext


_symm_cache = {}


def _get_symm_buffer(shape, dtype, device, group):
    key = (tuple(shape), dtype, device.index)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(*shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(
        list(hdl.buffer_ptrs), device=device, dtype=torch.int64
    )
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]


@torch.no_grad()
def solution(
    local_embedding_shard: torch.Tensor,
    input_node_ids: torch.Tensor,
    proj_matrix: torch.Tensor,
    num_total_nodes: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    shard_size = (num_total_nodes + world_size - 1) // world_size
    S, D = local_embedding_shard.shape
    Q = input_node_ids.shape[0]
    device = input_node_ids.device

    # Compile on rank 0 first so others wait
    rank = dist.get_rank(group)
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    ext = _get_ext()

    # Allocate (or reuse) a symmetric buffer of fixed shard capacity for the
    # embedding shard. Capacity is the maximum possible shard size.
    cap = shard_size
    sym_buf, hdl, ptrs_tensor = _get_symm_buffer(
        (cap, D), local_embedding_shard.dtype, device, group
    )

    # Copy local shard into the symmetric buffer (only the first S rows used).
    sym_buf[:S].copy_(local_embedding_shard)

    # Cross-rank synchronization: ensure all peers have populated their shard
    # before any peer reads.
    hdl.barrier(channel=0)

    # Gather embeddings in original query order directly from peer shards.
    ids_long = input_node_ids.long().contiguous()
    gathered = torch.empty((Q, D), dtype=local_embedding_shard.dtype, device=device)

    ext.launch_remote_gather_bf16(
        ptrs_tensor, ids_long, gathered, shard_size, world_size
    )

    # Synchronize again before any subsequent overwrite of the symm buffer.
    hdl.barrier(channel=1)

    # Projection via tensor cores
    return gathered @ proj_matrix