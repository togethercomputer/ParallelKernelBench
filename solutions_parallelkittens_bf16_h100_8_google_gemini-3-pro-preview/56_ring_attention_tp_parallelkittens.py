import os
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded ThunderKittens C++ / CUDA Source
# ---------------------------------------------------------------------------

CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>
#include <math.h>

using namespace kittens;

// ============================================================================
// TMA PEER COPY: Direct P2P DMA pull from symmetric memory
// ============================================================================
namespace peer_copy {
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 32; // Minimal threads; TMA offloads copy
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;

    using shared_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    int peer_idx;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3((input.cols() / globals::COL_BLOCK_SIZE) *
                    (input.rows() / globals::ROW_BLOCK_SIZE) *
                    input.depth() * input.batch());
    }

    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) + 1024);
    }
};

__device__ inline void kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::shared_tile &tile = allocator.allocate<globals::shared_tile>();

    int task_idx = blockIdx.x;
    int batch_idx = task_idx / (G.input.depth() * (G.input.rows() / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.input.depth() * (G.input.rows() / globals::ROW_BLOCK_SIZE) * (G.input.cols() / globals::COL_BLOCK_SIZE));
    int depth_idx = task_idx / (G.input.rows() / globals::ROW_BLOCK_SIZE * (G.input.cols() / globals::COL_BLOCK_SIZE));
    task_idx %= (G.input.rows() / globals::ROW_BLOCK_SIZE * (G.input.cols() / globals::COL_BLOCK_SIZE));
    int row_block_idx = task_idx / (G.input.cols() / globals::COL_BLOCK_SIZE);
    task_idx %= (G.input.cols() / globals::COL_BLOCK_SIZE);
    int col_block_idx = task_idx;

    __shared__ semaphore arrived;
    init_semaphore(arrived, 0, 1);
    tma::expect_bytes(arrived, sizeof(tile));
    
    // Asynchronous DMA pull from peer
    tma::load_async(tile, G.input[G.peer_idx], {batch_idx, depth_idx, row_block_idx, col_block_idx}, arrived);
    wait(arrived, 0);

    // Asynchronous DMA push to local buffer
    tma::store_async(G.output[G.dev_idx], tile, {batch_idx, depth_idx, row_block_idx, col_block_idx});
    tma::store_commit_group();
    tma::store_async_wait();
}
} // namespace peer_copy

void launch_peer_copy(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    int peer_idx
) {
    peer_copy::globals G {
        .output = kittens::py::parallel_tensor_to_pgl<typename peer_copy::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename peer_copy::globals::parallel_layout>(input),
        .peer_idx = peer_idx,
        .dev_idx = input.local_rank_
    };
    kittens::py::launch_kernel<peer_copy::config, peer_copy::globals, peer_copy::kernel>(G);
}

// ============================================================================
// BARRIER: Cluster synchronization
// ============================================================================
namespace sync_barrier {
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
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

void launch_barrier(kittens::py::TKParallelTensor &barrier) {
    sync_barrier::globals G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<sync_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    kittens::py::launch_kernel<sync_barrier::config, sync_barrier::globals, sync_barrier::kernel>(G);
}

// ============================================================================
// FUSED MERGE LSE: Numerically stable accumulation over chunks
// ============================================================================
namespace merge_lse {
struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 256;
};

struct globals {
    float* out;
    float* lse;
    const bf16* block_out;
    const float* block_lse;
    int numel_out;
    int D;
    int H;
    int S;
};

__device__ inline void kernel(const globals &G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < G.numel_out) {
        int lse_idx = idx / G.D;
        
        float current_lse = G.lse[lse_idx];
        float new_lse = G.block_lse[lse_idx];
        
        float current_out = G.out[idx];
        float new_out = __bfloat162float(G.block_out[idx]);
        
        bool new_is_inf = isinf(new_lse) && new_lse < 0;
        bool cur_is_inf = isinf(current_lse) && current_lse < 0;
        
        float sig;
        if (new_is_inf && cur_is_inf) {
            sig = 0.0f;
        } else if (cur_is_inf) {
            sig = 1.0f;
        } else if (new_is_inf) {
            sig = 0.0f;
        } else {
            float diff_lse = new_lse - current_lse;
            sig = 1.0f / (1.0f + expf(-diff_lse));
        }
        
        float updated_out = current_out - sig * (current_out - new_out);
        G.out[idx] = updated_out;
        
        // Single thread per head-dim group updates the shared LSE value
        if ((idx % G.D) == 0) {
            if (new_is_inf && cur_is_inf) {
                // remains -inf
            } else if (cur_is_inf) {
                G.lse[lse_idx] = new_lse;
            } else if (new_is_inf) {
                // remains current
            } else {
                float diff = current_lse - new_lse;
                float ls = -log1pf(expf(-diff));
                G.lse[lse_idx] = current_lse - ls;
            }
        }
    }
}
} // namespace merge_lse

void launch_merge(
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor block_out,
    torch::Tensor block_lse,
    int D, int H, int S
) {
    merge_lse::globals G {
        .out = out.data_ptr<float>(),
        .lse = lse.data_ptr<float>(),
        .block_out = reinterpret_cast<const bf16*>(block_out.data_ptr<at::BFloat16>()),
        .block_lse = block_lse.data_ptr<float>(),
        .numel_out = static_cast<int>(out.numel()),
        .D = D,
        .H = H,
        .S = S
    };
    dim3 grid((G.numel_out + merge_lse::config::NUM_THREADS - 1) / merge_lse::config::NUM_THREADS);
    kittens::py::launch_kernel<merge_lse::config, merge_lse::globals, merge_lse::kernel>(G, grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_peer_copy", &launch_peer_copy);
    m.def("tk_barrier", &launch_barrier);
    m.def("tk_merge", &launch_merge);
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
            "tk_ring_attn_ext",
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
    if dist.is_initialized() and dist.get_rank() == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


# ---------------------------------------------------------------------------
# Python Attention implementation interacting with TK extensions
# ---------------------------------------------------------------------------

def _local_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: float, causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes dense local attention, returning block outputs structured for the TK merge."""
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(q.size(1), k.size(1), device=q.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
    block_lse = torch.logsumexp(scores, dim=-1)
    block_out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).contiguous()
    
    # Transpose and guarantee contiguous memory structure for predictable TK merge addressing
    block_lse = block_lse.transpose(-2, -1).contiguous()
    return block_out.to(torch.bfloat16), block_lse


def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    tp_group: Optional[dist.ProcessGroup] = None,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    cp_group = cp_group or dist.group.WORLD

    tp_size = dist.get_world_size(tp_group)
    cp_size = dist.get_world_size(cp_group)
    heads_local = num_heads // tp_size
    head_dim = w_qkv.shape[0] // 3 // heads_local
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    B, S = hidden_states.shape[:2]
    qkv = F.linear(hidden_states, w_qkv).view(B, S, 3, heads_local, head_dim)
    q, k, v = qkv.unbind(dim=2)

    # Fast-path for single CP rank
    if cp_size == 1:
        block_out, _ = _local_attn(q.contiguous(), k.contiguous(), v.contiguous(), float(softmax_scale), causal)
        out = block_out.to(q.dtype)
        out = F.linear(out.reshape(B, S, -1), w_o)
        if tp_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
        return out

    ext = _ensure_ext_jit()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # Determine TMA optimal padding for Parallel Tensors
    flat_k = k.view(-1)
    flat_v = v.view(-1)
    n = flat_k.numel()
    
    TILE_ELEMS = 16 * 128
    num_tiles = (n + TILE_ELEMS - 1) // TILE_ELEMS
    r, c = 16, 128 * num_tiles

    # Static wrappers holding current rank's source data (peers pull from these)
    k_tk = get_or_create_parallel_tensor(ext, (1, 1, r, c), torch.bfloat16, multicast=False)
    v_tk = get_or_create_parallel_tensor(ext, (1, 1, r, c), torch.bfloat16, multicast=False)
    
    # Double-buffers mapped natively on device to pipeline peer fetching
    f1_k_tk = get_or_create_parallel_tensor(ext, (1, 1, r, c), torch.bfloat16, multicast=False)
    f1_v_tk = get_or_create_parallel_tensor(ext, (1, 1, r, c), torch.bfloat16, multicast=False)
    f2_k_tk = get_or_create_parallel_tensor(ext, (1, 1, r, c), torch.bfloat16, multicast=False)
    f2_v_tk = get_or_create_parallel_tensor(ext, (1, 1, r, c), torch.bfloat16, multicast=False)

    barrier_tk = get_or_create_barrier(ext, num_devices=8)

    # Initialize symmetric buffers and globally barrier
    k_tk.data_.view(-1)[:n].copy_(flat_k)
    v_tk.data_.view(-1)[:n].copy_(flat_v)
    ext.tk_barrier(barrier_tk)

    # Persistent output tracking allocations
    out = torch.zeros((B, S, heads_local, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.full((B, S, heads_local), float("-inf"), dtype=torch.float32, device=q.device)

    copy_stream = torch.cuda.Stream()
    local_cp = dist.get_rank(cp_group)
    local_tp = dist.get_rank(tp_group)

    # Issue initial background fetch for step 1
    if cp_size > 1:
        peer_cp = (local_cp - 1 + cp_size) % cp_size
        peer_global = peer_cp * tp_size + local_tp
        with torch.cuda.stream(copy_stream):
            ext.tk_peer_copy(f1_k_tk, k_tk, peer_global)
            ext.tk_peer_copy(f1_v_tk, v_tk, peer_global)

    # Process local step 0
    if (not causal) or 0 <= local_cp:
        block_out, block_lse = _local_attn(q, k, v, float(softmax_scale), causal=causal)
        ext.tk_merge(out, lse, block_out, block_lse, head_dim, heads_local, S)

    # Pipelined schedule over peers
    for step in range(1, cp_size):
        torch.cuda.current_stream().wait_stream(copy_stream)
        
        # Unwrap current step's buffers correctly
        cur_k = f1_k_tk.data_.view(-1)[:n].view(B, S, heads_local, head_dim)
        cur_v = f1_v_tk.data_.view(-1)[:n].view(B, S, heads_local, head_dim)
        
        # Launch background fetch for step n+1 into alternating buffer
        if step + 1 < cp_size:
            peer_cp = (local_cp - step - 1 + cp_size) % cp_size
            peer_global = peer_cp * tp_size + local_tp
            with torch.cuda.stream(copy_stream):
                ext.tk_peer_copy(f2_k_tk, k_tk, peer_global)
                ext.tk_peer_copy(f2_v_tk, v_tk, peer_global)

        # Execute attention on fetched KV and immediately device-fuse to accumulators
        if (not causal) or step <= local_cp:
            block_out, block_lse = _local_attn(q, cur_k, cur_v, float(softmax_scale), causal=False)
            ext.tk_merge(out, lse, block_out, block_lse, head_dim, heads_local, S)
            
        # Swap buffers for the next cycle
        f1_k_tk, f2_k_tk = f2_k_tk, f1_k_tk
        f1_v_tk, f2_v_tk = f2_v_tk, f1_v_tk

    # Finalize Out -> Row Parallel Projection -> TP sum
    out = out.to(q.dtype)
    out = F.linear(out.reshape(B, S, -1), w_o)
    if tp_size > 1:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)

    return out