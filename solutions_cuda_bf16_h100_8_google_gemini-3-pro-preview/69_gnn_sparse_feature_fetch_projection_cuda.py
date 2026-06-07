"""
Strategy:
1. Eliminate `all_to_all` and `argsort` communication overhead by caching embedding shards in `symm_mem` and leveraging NVLink UVA (peer direct memory access).
2. Fetch queried embeddings directly from remote memory (using a custom CUDA read kernel) natively into the correct query order—bypassing sort/unsort completely.
3. Maximize compute-communication overlap via double-buffering. Queries are chunked: while the current chunk is being projected (GEMM) via Tensor Cores on the default stream, the next chunk's embeddings are actively fetched over NVLink on a background stream.
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

__global__ void uva_fetch_kernel(
    const int64_t* __restrict__ queries,
    const uint64_t* __restrict__ shard_ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t num_queries,
    int64_t shard_size,
    int embed_dim,
    int world_size
) {
    int64_t q_idx = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (q_idx >= num_queries) return;

    int64_t node_id = queries[q_idx];
    int owner = node_id / shard_size;
    if (owner >= world_size) owner = world_size - 1;
    int64_t local_id = node_id - owner * shard_size;

    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(shard_ptrs[owner]) + local_id * embed_dim;
    __nv_bfloat16* dst = out + q_idx * embed_dim;

    int d = threadIdx.x;
    int stride = blockDim.x;

    if (embed_dim % 8 == 0) {
        int vecs = embed_dim / 8;
        const float4* src_v = reinterpret_cast<const float4*>(src);
        float4* dst_v = reinterpret_cast<float4*>(dst);
        for (int i = d; i < vecs; i += stride) {
            dst_v[i] = src_v[i];
        }
    } else if (embed_dim % 4 == 0) {
        int vecs = embed_dim / 4;
        const float2* src_v = reinterpret_cast<const float2*>(src);
        float2* dst_v = reinterpret_cast<float2*>(dst);
        for (int i = d; i < vecs; i += stride) {
            dst_v[i] = src_v[i];
        }
    } else if (embed_dim % 2 == 0) {
        int vecs = embed_dim / 2;
        const float* src_v = reinterpret_cast<const float*>(src);
        float* dst_v = reinterpret_cast<float*>(dst);
        for (int i = d; i < vecs; i += stride) {
            dst_v[i] = src_v[i];
        }
    } else {
        for (int i = d; i < embed_dim; i += stride) {
            dst[i] = src[i];
        }
    }
}

void launch_uva_fetch(
    torch::Tensor queries,
    torch::Tensor shard_ptrs,
    torch::Tensor out,
    int64_t shard_size,
    int world_size
) {
    int64_t num_queries = queries.size(0);
    if (num_queries == 0) return;
    
    int embed_dim = out.size(1);
    
    int vec_size = 1;
    if (embed_dim % 8 == 0) vec_size = 8;
    else if (embed_dim % 4 == 0) vec_size = 4;
    else if (embed_dim % 2 == 0) vec_size = 2;
    
    int vecs = embed_dim / vec_size;
    int tx = vecs;
    if (tx > 32) tx = 32;
    int ty = 256 / tx;
    if (ty == 0) ty = 1;
    
    dim3 threads(tx, ty);
    dim3 blocks((num_queries + ty - 1) / ty);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    uva_fetch_kernel<<<blocks, threads, 0, stream>>>(
        queries.data_ptr<int64_t>(),
        reinterpret_cast<const uint64_t*>(shard_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        num_queries,
        shard_size,
        embed_dim,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_uva_fetch", &launch_uva_fetch, "UVA fetch of remote embeddings");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_sparse_fetch_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(shape, dtype, device, group):
    key = (shape, dtype, device, group)
    if key in _symm_cache:
        return _symm_cache[key]

    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, ptrs_tensor)
    _symm_cache[key] = res
    return res

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
    rank = dist.get_rank(group)
    shard_size = (num_total_nodes + world_size - 1) // world_size
    embed_dim = local_embedding_shard.shape[1]
    device = local_embedding_shard.device

    if rank == 0:
        _get_ext()
    dist.barrier(group=group)

    if local_embedding_shard.dtype != torch.bfloat16:
        local_embedding_shard = local_embedding_shard.to(torch.bfloat16)
    if proj_matrix.dtype != torch.bfloat16:
        proj_matrix = proj_matrix.to(torch.bfloat16)
    if input_node_ids.dtype != torch.int64:
        input_node_ids = input_node_ids.to(torch.int64)
        
    input_node_ids = input_node_ids.contiguous().view(-1)
    proj_matrix = proj_matrix.contiguous()

    # Wait for trailing tasks from any previous iterations prior to overwriting symm buf
    dist.barrier(group=group)

    buf, hdl, ptrs_tensor = _get_symm_state(
        local_embedding_shard.shape, 
        local_embedding_shard.dtype, 
        device, 
        group
    )
    buf.copy_(local_embedding_shard)

    # Assure all embedding shards are registered/copied completely before UVA fetching begins
    dist.barrier(group=group)

    Q = input_node_ids.size(0)
    C = 32768  # Batch fetch queries into robust cache-friendly GEMM chunks
    num_chunks = (Q + C - 1) // C

    # Fast-path for small query workloads
    if num_chunks <= 1:
        gathered = torch.empty((Q, embed_dim), dtype=torch.bfloat16, device=device)
        if Q > 0:
            _get_ext().launch_uva_fetch(
                input_node_ids, ptrs_tensor, gathered, shard_size, world_size
            )
        return torch.matmul(gathered, proj_matrix)

    # Double-buffering path for overlapped comms / compute
    out = torch.empty((Q, proj_matrix.size(1)), dtype=torch.bfloat16, device=device)
    bufA = torch.empty((C, embed_dim), dtype=torch.bfloat16, device=device)
    bufB = torch.empty((C, embed_dim), dtype=torch.bfloat16, device=device)

    fetch_stream = torch.cuda.Stream(device=device)
    comp_stream = torch.cuda.current_stream(device=device)

    fetch_events = [torch.cuda.Event() for _ in range(num_chunks)]
    comp_events = [torch.cuda.Event(), torch.cuda.Event()]

    # Signal both buffers as conceptually "ready" prior to start
    comp_events[0].record(comp_stream)
    comp_events[1].record(comp_stream)

    # Kick off the initial fetch payload
    with torch.cuda.stream(fetch_stream):
        chunk_queries = input_node_ids[0:C]
        if chunk_queries.numel() > 0:
            _get_ext().launch_uva_fetch(
                chunk_queries, ptrs_tensor, bufA, shard_size, world_size
            )
        fetch_events[0].record(fetch_stream)

    for i in range(num_chunks):
        start_idx = i * C
        end_idx = min(start_idx + C, Q)
        current_buf = bufA if (i % 2 == 0) else bufB
        current_comp_event_idx = i % 2

        # 1. Pipeline barrier: await the active memory chunk fetch
        comp_stream.wait_event(fetch_events[i])

        # 2. TensorCore local matrix-multiplication pass
        chunk_queries_len = end_idx - start_idx
        if chunk_queries_len > 0:
            torch.mm(current_buf[:chunk_queries_len], proj_matrix, out=out[start_idx:end_idx])

        # Mark active buffer free for future fetches
        comp_events[current_comp_event_idx].record(comp_stream)

        # 3. Schedule upcoming UVA stream fetch overlapping existing host loop & compute
        if i + 1 < num_chunks:
            next_start = (i + 1) * C
            next_end = min(next_start + C, Q)
            next_buf = bufB if (i % 2 == 0) else bufA
            next_comp_event_idx = (i + 1) % 2

            with torch.cuda.stream(fetch_stream):
                fetch_stream.wait_event(comp_events[next_comp_event_idx])
                next_chunk_queries = input_node_ids[next_start:next_end]
                if next_chunk_queries.numel() > 0:
                    _get_ext().launch_uva_fetch(
                        next_chunk_queries, ptrs_tensor, next_buf, shard_size, world_size
                    )
                fetch_events[i+1].record(fetch_stream)

    # Guard ensuring all asynchronous reads safely complete prior to returning execution
    comp_stream.wait_stream(fetch_stream)
    return out