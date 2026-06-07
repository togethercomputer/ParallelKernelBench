"""
Strategy:
- **Device-Side Communication**: Instead of exchanging indices via NCCL and performing local lookups, we allocate the embedding shards symmetrically via `TKParallelTensor` so all GPUs have direct P2P memory access. A custom kernel directly reads the required remote embeddings over NVLink into the local output buffer.
- **Compute-Communication Overlap**: Peer NVLink loads are intrinsically executed asynchronously by the hardware. By vectorizing the memory operations and using warp-stride loops, the remote memory access latency is seamlessly hidden by the SM's warp scheduling, bypassing the latency penalties of `dist.all_to_all_single`.
- **Zero-Overhead Harness**: We use `get_or_create_parallel_tensor` to maintain the symmetric buffer. TK-native barriers guarantee safe copy-in and cross-GPU read semantics without repeatedly allocating or stalling via heavy PyTorch collectives.
"""

import os
import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source (Embedding Lookup entrypoint + TK barrier)
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>

struct globals {
    static constexpr int NUM_DEVICES = 8;
    nv_bfloat16* shards[NUM_DEVICES];
    const int64_t* indices;
    nv_bfloat16* output;
    int N;
    int shard_size;
    int embed_dim;
};

__global__ void lookup_kernel(globals G) {
    // Each row of block handles one index lookup; x-dim provides vectorization
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= G.N) return;

    int64_t global_id = G.indices[idx];
    int target_rank = global_id / G.shard_size;
    int local_offset = global_id % G.shard_size;

    // Safety clamps
    if (target_rank < 0) target_rank = 0;
    if (target_rank >= globals::NUM_DEVICES) target_rank = globals::NUM_DEVICES - 1;
    if (local_offset < 0) local_offset = 0;
    if (local_offset >= G.shard_size) local_offset = G.shard_size - 1;

    const nv_bfloat16* src = G.shards[target_rank] + (size_t)local_offset * (size_t)G.embed_dim;
    nv_bfloat16* dst = G.output + (size_t)idx * (size_t)G.embed_dim;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Check 16-byte alignment and compatible vectorization size
    bool aligned = ((reinterpret_cast<uintptr_t>(src) % 16) == 0) &&
                   ((reinterpret_cast<uintptr_t>(dst) % 16) == 0) &&
                   ((G.embed_dim % 8) == 0);

    if (aligned) {
        int d_vec = G.embed_dim / 8; // 8 bf16s per uint4
        const uint4* src_v = reinterpret_cast<const uint4*>(src);
        uint4* dst_v = reinterpret_cast<uint4*>(dst);
        for (int i = tid; i < d_vec; i += stride) {
            dst_v[i] = src_v[i];
        }
    } else {
        // Fallback for unaligned or irregularly sized dimensions
        for (int i = tid; i < G.embed_dim; i += stride) {
            dst[i] = src[i];
        }
    }
}

namespace lookup_barrier {
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};
struct globals {
    static constexpr int NUM_DEVICES = 8;
    kittens::barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};
__device__ inline void kernel(const globals &G) {
    kittens::barrier_all(G.barrier, {0}, G.dev_idx);
}
} // namespace lookup_barrier

void entrypoint(
    kittens::py::TKParallelTensor &shard_tk,
    kittens::py::TKParallelTensor &barrier,
    torch::Tensor indices,
    torch::Tensor output,
    int shard_size,
    int embed_dim
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be on CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be on CUDA");
    
    globals G;
    for (int i = 0; i < globals::NUM_DEVICES; i++) {
        G.shards[i] = reinterpret_cast<nv_bfloat16*>(shard_tk.ptrs_[i]);
    }
    G.indices = indices.data_ptr<int64_t>();
    G.output = reinterpret_cast<nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    G.N = indices.numel();
    G.shard_size = shard_size;
    G.embed_dim = embed_dim;

    lookup_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<kittens::barrier_t<lookup_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Barrier 1: ensure all peers have finished copying their local_shard into the symmetric shard_tk
    kittens::py::launch_kernel<lookup_barrier::config, lookup_barrier::globals, lookup_barrier::kernel>(barrier_G);

    // Launch lookup (32 threads per index, 8 indices per block)
    dim3 block(32, 8);
    dim3 grid((G.N + block.y - 1) / block.y);
    if (G.N > 0) {
        lookup_kernel<<<grid, block>>>(G);
    }

    // Barrier 2: ensure no peer overwrites its shard_tk (in next iteration) before others finish reading
    kittens::py::launch_kernel<lookup_barrier::config, lookup_barrier::globals, lookup_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_embedding_lookup", &entrypoint);
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
            "tk_embedding_lookup_ext",
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
    """Compile/load extension once; avoid per-call ``dist.barrier()`` in timed hot path."""
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
    indices: torch.Tensor,
    local_shard: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    world_size = dist.get_world_size()
    assert world_size == 8, f"This ThunderKittens kernel is built for NUM_DEVICES=8; got {world_size}"
    
    ext = _ensure_ext_jit()
    
    indices = indices.contiguous().to(torch.cuda.current_device())
    
    shard_size, embed_dim = local_shard.shape
    original_dtype = local_shard.dtype
    
    local_shard_bf16 = local_shard.to(torch.bfloat16).contiguous()
    
    # 1. Acquire symmetric memory for the table shards
    shard_tk = get_or_create_parallel_tensor(
        ext, (shard_size, embed_dim), torch.bfloat16, multicast=False
    )
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    
    # 2. Copy our local shard to the symmetric memory. 
    # Must use copy_ to ensure writing into the underlying VMM buffer mapped for peers.
    shard_tk.data_.copy_(local_shard_bf16)
    
    out_bf16 = torch.empty((indices.numel(), embed_dim), dtype=torch.bfloat16, device=indices.device)
    
    # 3. Launch unified kernel (barrier -> p2p lookup over NVLink -> barrier)
    ext.tk_embedding_lookup(
        shard_tk,
        barrier_tk,
        indices,
        out_bf16,
        shard_size,
        embed_dim
    )
    
    return out_bf16.to(original_dtype)