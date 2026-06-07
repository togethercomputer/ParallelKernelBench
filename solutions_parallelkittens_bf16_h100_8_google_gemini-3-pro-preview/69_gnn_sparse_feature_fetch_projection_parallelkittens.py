"""
Strategy:
1. **Device-Side P2P Communication**: We eliminate three PyTorch `all_to_all` passes and two `argsort` operations by exposing the local embedding shards via ThunderKittens symmetric `TKParallelTensor` (`pgl`). A custom Hopper CUDA kernel executes direct peer-to-peer (P2P) loads over NVLink, writing directly to the requested output index. This natively preserves the query order without any host-side coordination.
2. **Compute-Communication Overlap**: The sparse gather is highly memory-bandwidth bound over NVLink, while the embedding projection is a dense math-bound Tensor Core GEMM. We split the queries into chunks and use CUDA stream double-buffering: while Chunk $i$ executes its dense projection, Chunk $i+1$ performs the NVLink P2P gather concurrently, entirely hiding the communication latency.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source: P2P Gather + Barries using ThunderKittens
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>

using namespace kittens;

// ============================================================================
// Barrier Module
// ============================================================================
namespace barrier {

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

} // namespace barrier

void tk_barrier(kittens::py::TKParallelTensor &barrier) {
    barrier::globals G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    kittens::py::launch_kernel<barrier::config, barrier::globals, barrier::kernel>(G);
}


// ============================================================================
// NVLink P2P Gather Kernel
// ============================================================================
namespace p2p_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = 4;
    static constexpr int NUM_THREADS = NUM_WARPS * 32; // 128 threads
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    // Flat PGL layout: Multicast is false, we just use this to cleanly resolve symmetric memory pointers
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, false>;
    
    parallel_layout embeddings;
    const int64_t* input_node_ids;
    bf16* output_gathered;
    
    int num_queries;
    int shard_size;
    int embed_dim;

    __host__ inline dim3 grid() const {
        return dim3((num_queries + config::NUM_WARPS - 1) / config::NUM_WARPS);
    }
};

__device__ inline void kernel(const globals &G) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    int q_idx = blockIdx.x * config::NUM_WARPS + warp_id;
    if (q_idx >= G.num_queries) return;

    int64_t node_id = G.input_node_ids[q_idx];
    int owner = node_id / G.shard_size;
    int local_id = node_id % G.shard_size;
    
    // Clamp to valid range (should be guaranteed by inputs)
    if (owner >= globals::NUM_DEVICES) owner = globals::NUM_DEVICES - 1;

    // Use the pointer resolved by the ParallelKittens PGL broker
    bf16* src_ptr = (bf16*)G.embeddings[owner].data;
    bf16* dst_ptr = G.output_gathered + q_idx * G.embed_dim;

    // Warp-level vectorized read from peer GPU memory
    for (int d = lane_id; d < G.embed_dim; d += 32) {
        dst_ptr[d] = src_ptr[local_id * G.embed_dim + d];
    }
}

} // namespace p2p_gather

__global__ void __launch_bounds__(p2p_gather::config::NUM_THREADS, 1) 
gather_kernel_wrapper(p2p_gather::globals G) {
    p2p_gather::kernel(G);
}

void tk_p2p_gather(
    kittens::py::TKParallelTensor &embeddings,
    torch::Tensor input_node_ids,
    torch::Tensor output_gathered,
    int num_queries,
    int shard_size,
    int embed_dim
) {
    p2p_gather::globals G {
        .embeddings = kittens::py::parallel_tensor_to_pgl<typename p2p_gather::globals::parallel_layout>(embeddings),
        .input_node_ids = input_node_ids.data_ptr<int64_t>(),
        .output_gathered = reinterpret_cast<bf16*>(output_gathered.data_ptr<at::BFloat16>()),
        .num_queries = num_queries,
        .shard_size = shard_size,
        .embed_dim = embed_dim
    };

    // Grab the current PyTorch stream to execute asynchronously and enable overlapping
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid = G.grid();
    dim3 block(p2p_gather::config::NUM_THREADS);
    
    gather_kernel_wrapper<<<grid, block, 0, stream>>>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_p2p_gather", &tk_p2p_gather);
    m.def("tk_barrier", &tk_barrier);
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
            "tk_gnn_p2p_ext",
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
    assert world_size == 8, f"This kernel is built for 8 GPUs, got world_size={world_size}"
    
    ext = _ensure_ext_jit()
    
    shard_size = (num_total_nodes + world_size - 1) // world_size
    embed_dim = local_embedding_shard.shape[1]
    
    # 1. Provide symmetric access to embedding shards using TKParallelTensor cache
    embeddings_tk = get_or_create_parallel_tensor(
        ext, (shard_size, embed_dim), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    
    # Flat copy over (resolving size if the rank owns slightly fewer than `shard_size` items)
    n = local_embedding_shard.numel()
    embeddings_tk.data_.reshape(-1)[:n].copy_(local_embedding_shard.reshape(-1))
    
    # Force synchronization so all peers can safely pull from this array
    ext.tk_barrier(barrier_tk)
    
    num_queries = input_node_ids.shape[0]
    out_dim = proj_matrix.shape[1]
    
    # 2. Compute-Communication Overlap via Double-Buffered Streams
    num_chunks = 2
    chunk_size = (num_queries + num_chunks - 1) // num_chunks
    streams = [torch.cuda.Stream() for _ in range(num_chunks)]
    out_chunks = []
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, num_queries)
        if start >= end:
            break
            
        with torch.cuda.stream(streams[i]):
            queries_chunk = input_node_ids[start:end]
            gathered_chunk = torch.empty(
                (end - start, embed_dim), 
                dtype=torch.bfloat16, 
                device=input_node_ids.device
            )
            
            # Asynchronous custom P2P read kernel
            ext.tk_p2p_gather(
                embeddings_tk,
                queries_chunk,
                gathered_chunk,
                end - start,
                shard_size,
                embed_dim
            )
            
            # Immediately perform dense projection while the next gather runs
            out_chunk = torch.matmul(gathered_chunk, proj_matrix)
            out_chunks.append(out_chunk)
            
    # Resolve pipelines
    for s in streams:
        torch.cuda.current_stream().wait_stream(s)
        
    out = torch.cat(out_chunks, dim=0) if out_chunks else torch.empty(
        (0, out_dim), dtype=proj_matrix.dtype, device=proj_matrix.device
    )
    
    # Protect embeddings_tk buffer from being recycled or overwritten before all remote peers finish reads
    ext.tk_barrier(barrier_tk)
    
    return out