"""
Strategy:
- Replaced the PyTorch `all_to_all_single` collect/scatter and CPU-side argsort with a custom 3-stage push/pull architecture over NVLink via `torch.distributed._symmetric_memory`.
- Avoided routing overhead: `route_queries_kernel` directly writes requested IDs to peers' `symm_mem` buffers (UVA), fully bypassing `argsort`.
- Used atomic counters per-peer to pack queries directly into destination buffers, maximizing bandwidth without cross-rank atomics.
- Fused the un-sorting process: `gather_kernel` pulls responses from peers and writes them exactly to their original query offsets, eliminating another sort.
- Maximized device execution: The sparse fetch is executed as three lightweight kernels with device-side barriers, directly yielding a dense contiguous tensor ready for the projection GEMM.
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

__global__ void route_queries_kernel(
    const int64_t* __restrict__ input_node_ids,
    const int64_t* __restrict__ remote_base_ptrs,
    int32_t* __restrict__ local_send_counts,
    int32_t* __restrict__ query_dest_idx,
    int64_t req_offset,
    int64_t shard_size,
    int world_size,
    int rank,
    int MAX_Q,
    int num_queries
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries) {
        int64_t node_id = input_node_ids[idx];
        int owner = min((int)(node_id / shard_size), world_size - 1);
        int64_t local_id = node_id - owner * shard_size;
        
        int offset = atomicAdd(&local_send_counts[owner], 1);
        
        int64_t owner_base = remote_base_ptrs[owner];
        int64_t* owner_req_buf = (int64_t*)(owner_base + req_offset) + rank * MAX_Q;
        
        owner_req_buf[offset] = local_id;
        query_dest_idx[idx] = owner * MAX_Q + offset;
    }
}

__global__ void write_counts_kernel(
    const int32_t* __restrict__ local_send_counts,
    const int64_t* __restrict__ remote_base_ptrs,
    int64_t count_offset,
    int rank,
    int world_size
) {
    int owner = threadIdx.x;
    if (owner < world_size) {
        int count = local_send_counts[owner];
        int64_t owner_base = remote_base_ptrs[owner];
        int32_t* owner_count_buf = (int32_t*)(owner_base + count_offset);
        owner_count_buf[rank] = count;
    }
}

__global__ void lookup_kernel(
    const uint8_t* __restrict__ local_base_ptr,
    const __nv_bfloat16* __restrict__ local_embedding_shard,
    int64_t count_offset,
    int64_t req_offset,
    int64_t resp_offset,
    int MAX_Q,
    int D,
    int world_size
) {
    const int32_t* count_buf = (const int32_t*)(local_base_ptr + count_offset);
    
    for (int rank_i = blockIdx.y; rank_i < world_size; rank_i += gridDim.y) {
        int count = count_buf[rank_i];
        for (int q_idx = blockIdx.x * blockDim.y + threadIdx.y; q_idx < count; q_idx += gridDim.x * blockDim.y) {
            const int64_t* req_buf = (const int64_t*)(local_base_ptr + req_offset) + rank_i * MAX_Q;
            __nv_bfloat16* resp_buf = (__nv_bfloat16*)(local_base_ptr + resp_offset) + rank_i * MAX_Q * D;
            
            int64_t local_id = req_buf[q_idx];
            const __nv_bfloat16* emb_src = local_embedding_shard + local_id * D;
            __nv_bfloat16* emb_dst = resp_buf + q_idx * D;
            
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                emb_dst[d] = emb_src[d];
            }
        }
    }
}

__global__ void gather_kernel(
    const int32_t* __restrict__ query_dest_idx,
    const int64_t* __restrict__ remote_base_ptrs,
    __nv_bfloat16* __restrict__ gathered_emb,
    int64_t resp_offset,
    int MAX_Q,
    int D,
    int num_queries,
    int rank
) {
    int q_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (q_idx < num_queries) {
        int dest_info = query_dest_idx[q_idx];
        int owner = dest_info / MAX_Q;
        int offset = dest_info % MAX_Q;
        
        int64_t owner_base = remote_base_ptrs[owner];
        const __nv_bfloat16* resp_buf = (const __nv_bfloat16*)(owner_base + resp_offset) + rank * MAX_Q * D;
        const __nv_bfloat16* emb_src = resp_buf + offset * D;
        __nv_bfloat16* emb_dst = gathered_emb + q_idx * D;
        
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            emb_dst[d] = emb_src[d];
        }
    }
}

void run_kernels(
    torch::Tensor input_node_ids,
    torch::Tensor remote_base_ptrs,
    torch::Tensor local_send_counts,
    torch::Tensor query_dest_idx,
    int64_t req_offset,
    int64_t count_offset,
    int64_t resp_offset,
    int64_t shard_size,
    int world_size,
    int rank,
    int MAX_Q,
    int num_queries,
    int64_t local_base_ptr_val,
    torch::Tensor local_embedding_shard,
    torch::Tensor gathered_emb,
    int D,
    int step
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (step == 0) {
        cudaMemsetAsync(local_send_counts.data_ptr(), 0, world_size * sizeof(int32_t), stream);
        
        int threads = 256;
        int blocks = (num_queries + threads - 1) / threads;
        if (blocks > 0) {
            route_queries_kernel<<<blocks, threads, 0, stream>>>(
                input_node_ids.data_ptr<int64_t>(),
                remote_base_ptrs.data_ptr<int64_t>(),
                local_send_counts.data_ptr<int32_t>(),
                query_dest_idx.data_ptr<int32_t>(),
                req_offset, shard_size, world_size, rank, MAX_Q, num_queries
            );
        }
        write_counts_kernel<<<1, world_size, 0, stream>>>(
            local_send_counts.data_ptr<int32_t>(),
            remote_base_ptrs.data_ptr<int64_t>(),
            count_offset, rank, world_size
        );
    } 
    else if (step == 1) {
        const uint8_t* local_base_ptr = reinterpret_cast<const uint8_t*>(local_base_ptr_val);
        dim3 block(32, 8);
        dim3 grid(256, world_size);
        lookup_kernel<<<grid, block, 0, stream>>>(
            local_base_ptr,
            reinterpret_cast<const __nv_bfloat16*>(local_embedding_shard.data_ptr()),
            count_offset, req_offset, resp_offset, MAX_Q, D, world_size
        );
    }
    else if (step == 2) {
        dim3 block(32, 8);
        dim3 grid((num_queries + block.y - 1) / block.y);
        if (grid.x > 0) {
            gather_kernel<<<grid, block, 0, stream>>>(
                query_dest_idx.data_ptr<int32_t>(),
                remote_base_ptrs.data_ptr<int64_t>(),
                reinterpret_cast<__nv_bfloat16*>(gathered_emb.data_ptr()),
                resp_offset, MAX_Q, D, num_queries, rank
            );
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernels", &run_kernels, "GNN Sparse Fetch Kernels");
}
'''

class SymmState:
    def __init__(self):
        self.capacity = 0
        self.D = 0
        self.world_size = 0
        self.buf = None
        self.hdl = None
        self.remote_base_ptrs = None
        self.local_send_counts = None
        self.query_dest_idx_capacity = 0
        self.query_dest_idx = None
        self.gathered_emb = None

_symm_state = SymmState()
_ext = None


@torch.no_grad()
def solution(
    local_embedding_shard: torch.Tensor,
    input_node_ids: torch.Tensor,
    proj_matrix: torch.Tensor,
    num_total_nodes: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    global _ext, _symm_state
    
    if _ext is None:
        _ext = compile_cuda_extension("gnn_sparse_fetch_ext", CUDA_SRC)
        
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    shard_size = (num_total_nodes + world_size - 1) // world_size
    D = local_embedding_shard.shape[1]
    num_queries = input_node_ids.shape[0]
    device = input_node_ids.device
    
    # 1. Determine if we need to resize the persistent symm_mem buffer.
    #    A lightweight all-reduce ensures all peers agree on MAX_Q dynamically.
    local_q = torch.tensor([num_queries], dtype=torch.int32, device=device)
    dist.all_reduce(local_q, op=dist.ReduceOp.MAX, group=group)
    global_max_q = local_q.item()
    
    if (global_max_q > _symm_state.capacity or 
        D != _symm_state.D or 
        world_size != _symm_state.world_size):
        
        new_capacity = max(1048576, int(global_max_q * 1.5))
        req_offset = 256
        resp_offset = req_offset + 8 * world_size * new_capacity
        resp_offset = (resp_offset + 255) // 256 * 256
        total_bytes = resp_offset + 2 * world_size * new_capacity * D
        
        _symm_state.buf = symm_mem.empty(total_bytes, dtype=torch.uint8, device=device)
        _symm_state.hdl = symm_mem.rendezvous(_symm_state.buf, group=group)
        _symm_state.remote_base_ptrs = torch.tensor(
            _symm_state.hdl.buffer_ptrs, dtype=torch.int64, device=device
        )
        _symm_state.capacity = new_capacity
        _symm_state.D = D
        _symm_state.world_size = world_size
        _symm_state.local_send_counts = torch.zeros(world_size, dtype=torch.int32, device=device)

    # Reallocate local response caches if necessary
    if num_queries > _symm_state.query_dest_idx_capacity:
        new_q_cap = max(1048576, int(num_queries * 1.5))
        _symm_state.query_dest_idx = torch.empty(new_q_cap, dtype=torch.int32, device=device)
        _symm_state.gathered_emb = torch.empty((new_q_cap, D), dtype=torch.bfloat16, device=device)
        _symm_state.query_dest_idx_capacity = new_q_cap

    MAX_Q = _symm_state.capacity
    req_offset = 256
    count_offset = 0
    resp_offset = req_offset + 8 * world_size * MAX_Q
    resp_offset = (resp_offset + 255) // 256 * 256
    
    local_base_ptr_val = _symm_state.remote_base_ptrs[rank].item()
    
    # Kernel 1: Scatter query offsets natively to peers via UVA
    _ext.run_kernels(
        input_node_ids, _symm_state.remote_base_ptrs, _symm_state.local_send_counts,
        _symm_state.query_dest_idx, req_offset, count_offset, resp_offset,
        shard_size, world_size, rank, MAX_Q, num_queries, local_base_ptr_val,
        local_embedding_shard, _symm_state.gathered_emb, D, 0
    )
    
    # Wait for queries to arrive locally
    _symm_state.hdl.barrier(channel=0)
    
    # Kernel 2: Lookup features and write securely locally into symmetric memory 
    _ext.run_kernels(
        input_node_ids, _symm_state.remote_base_ptrs, _symm_state.local_send_counts,
        _symm_state.query_dest_idx, req_offset, count_offset, resp_offset,
        shard_size, world_size, rank, MAX_Q, num_queries, local_base_ptr_val,
        local_embedding_shard, _symm_state.gathered_emb, D, 1
    )
    
    # Wait for lookups to finish computation on peers
    _symm_state.hdl.barrier(channel=1)
    
    # Kernel 3: Direct read of queried embeddings from peer memory, naturally sorting to original offsets
    _ext.run_kernels(
        input_node_ids, _symm_state.remote_base_ptrs, _symm_state.local_send_counts,
        _symm_state.query_dest_idx, req_offset, count_offset, resp_offset,
        shard_size, world_size, rank, MAX_Q, num_queries, local_base_ptr_val,
        local_embedding_shard, _symm_state.gathered_emb, D, 2
    )
    
    # Dense Projection Matmul efficiently processes un-sorted gathered results 
    gathered = _symm_state.gathered_emb[:num_queries]
    out = torch.matmul(gathered, proj_matrix)
    
    return out