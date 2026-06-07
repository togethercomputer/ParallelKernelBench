import os
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
# Embedded .cu source for ParallelKittens TMA Fetch and Barrier
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_THREADS = 1; 
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK = 64;
    static constexpr int COL_BLOCK = 64;
    
    using shared_tile = st_bf<ROW_BLOCK, COL_BLOCK>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, shared_tile>, NUM_DEVICES, false>;
    
    parallel_layout k_pgl;
    parallel_layout v_pgl;
    
    gl<bf16, -1, -1, -1, -1, shared_tile> k_local;
    gl<bf16, -1, -1, -1, -1, shared_tile> v_local;
    
    int peer_rank;
    
    __host__ inline dim3 grid() const {
        return dim3(
            (k_pgl.cols() / COL_BLOCK) * 
            (k_pgl.rows() / ROW_BLOCK) * 
            k_pgl.depth() * 
            k_pgl.batch()
        );
    }
    
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(shared_tile) * 2 + 1024);
    }
};

__device__ inline void kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    
    globals::shared_tile &k_tile = allocator.allocate<globals::shared_tile>();
    globals::shared_tile &v_tile = allocator.allocate<globals::shared_tile>();
    
    int task_idx = blockIdx.x;
    int batch_idx = task_idx / (G.k_pgl.depth() * (G.k_pgl.rows() / globals::ROW_BLOCK) * (G.k_pgl.cols() / globals::COL_BLOCK));
    task_idx %= (G.k_pgl.depth() * (G.k_pgl.rows() / globals::ROW_BLOCK) * (G.k_pgl.cols() / globals::COL_BLOCK));
    
    int depth_idx = task_idx / (G.k_pgl.rows() / globals::ROW_BLOCK * (G.k_pgl.cols() / globals::COL_BLOCK));
    task_idx %= (G.k_pgl.rows() / globals::ROW_BLOCK * (G.k_pgl.cols() / globals::COL_BLOCK));
    
    int row_block_idx = task_idx / (G.k_pgl.cols() / globals::COL_BLOCK);
    task_idx %= (G.k_pgl.cols() / globals::COL_BLOCK);
    
    int col_block_idx = task_idx;
    
    __shared__ semaphore arrived_k, arrived_v;
    init_semaphore(arrived_k, 0, 1);
    init_semaphore(arrived_v, 0, 1);
    
    tma::expect_bytes(arrived_k, sizeof(k_tile));
    tma::expect_bytes(arrived_v, sizeof(v_tile));
    
    tma::load_async(k_tile, G.k_pgl[G.peer_rank], {batch_idx, depth_idx, row_block_idx, col_block_idx}, arrived_k);
    tma::load_async(v_tile, G.v_pgl[G.peer_rank], {batch_idx, depth_idx, row_block_idx, col_block_idx}, arrived_v);
    
    wait(arrived_k, 0);
    wait(arrived_v, 0);
    
    tma::store_async(G.k_local, k_tile, {batch_idx, depth_idx, row_block_idx, col_block_idx});
    tma::store_async(G.v_local, v_tile, {batch_idx, depth_idx, row_block_idx, col_block_idx});
}

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

void tk_fetch_kv(
    kittens::py::TKParallelTensor &k_pgl,
    kittens::py::TKParallelTensor &v_pgl,
    torch::Tensor k_local,
    torch::Tensor v_local,
    int peer_rank
) {
    globals G {
        .k_pgl = kittens::py::parallel_tensor_to_pgl<typename globals::parallel_layout>(k_pgl),
        .v_pgl = kittens::py::parallel_tensor_to_pgl<typename globals::parallel_layout>(v_pgl),
        .k_local = kittens::py::tensor_to_gl<bf16, -1, -1, -1, -1, globals::shared_tile>(k_local),
        .v_local = kittens::py::tensor_to_gl<bf16, -1, -1, -1, -1, globals::shared_tile>(v_local),
        .peer_rank = peer_rank
    };
    kittens::py::launch_kernel<config, globals, kernel>(G);
}

void tk_barrier_fn(kittens::py::TKParallelTensor &barrier) {
    tk_barrier::globals G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<tk_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    kittens::py::launch_kernel<tk_barrier::config, tk_barrier::globals, tk_barrier::kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_fetch_kv", &tk_fetch_kv);
    m.def("tk_barrier", &tk_barrier_fn);
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
            "tk_ring_fetch_ext",
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


# ---------------------------------------------------------------------------
# Torch Compiled Local Compute Kernels
# ---------------------------------------------------------------------------
@torch.compile(fullgraph=True, mode="reduce-overhead")
def _local_attn_bhsd(
    qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor,
    scale: float, causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """qh, kh, vh: [B,H,S,D] -> out: [B,S,H,D], lse: [B,H,S]"""
    qh = qh.float()
    kh = kh.float()
    vh = vh.float()
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(qh.size(2), kh.size(2), device=qh.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    block_lse = torch.logsumexp(scores, dim=-1)
    block_out = torch.matmul(torch.softmax(scores, dim=-1), vh)
    return block_out.transpose(1, 2).contiguous(), block_lse

@torch.compile(fullgraph=True, mode="reduce-overhead")
def _merge_out_lse_compiled(
    out: torch.Tensor, lse: torch.Tensor,
    block_out: torch.Tensor, block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(-1)
    out = out - torch.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - torch.nn.functional.logsigmoid(lse - block_lse)
    return out, lse


# ---------------------------------------------------------------------------
# Hot-Path Entrypoint
# ---------------------------------------------------------------------------
@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    assert world_size == 8, "ThunderKittens kernel assumes NUM_DEVICES=8"

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
        
    # Re-stride to align with TK GL [B, H, S, D] layout internally, avoiding memory copies.
    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    
    B, H, S, D = kh.shape
    
    if world_size == 1:
        block_out, block_lse = _local_attn_bhsd(qh, kh, vh, softmax_scale, causal)
        out, lse = block_out.to(torch.float32), block_lse.transpose(-2, -1).unsqueeze(-1)
        return out.to(q.dtype)

    ext = _ensure_ext_jit()

    # Align sequence and depth dimensions for TK 64x64 TMA chunks
    pad_S = (64 - (S % 64)) % 64
    pad_D = (64 - (D % 64)) % 64

    if pad_S > 0 or pad_D > 0:
        kh_pad = F.pad(kh, (0, pad_D, 0, pad_S))
        vh_pad = F.pad(vh, (0, pad_D, 0, pad_S))
    else:
        kh_pad = kh
        vh_pad = vh

    # Expose local K and V physically in symmetric memory
    k_tk = get_or_create_parallel_tensor(ext, kh_pad.shape, torch.bfloat16, multicast=False)
    v_tk = get_or_create_parallel_tensor(ext, vh_pad.shape, torch.bfloat16, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    k_tk.data_[:kh_pad.numel()].copy_(kh_pad.view(-1))
    v_tk.data_[:vh_pad.numel()].copy_(vh_pad.view(-1))

    ext.tk_barrier(barrier_tk)

    compute_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()

    buf_k = [torch.empty_like(kh_pad), torch.empty_like(kh_pad)]
    buf_v = [torch.empty_like(vh_pad), torch.empty_like(vh_pad)]

    out, lse = None, None

    # Overlap bootstrap: Prefetch step 1 asynchronously
    if world_size > 1:
        with torch.cuda.stream(comm_stream):
            peer_1 = (rank - 1) % world_size
            ext.tk_fetch_kv(k_tk, v_tk, buf_k[1%2], buf_v[1%2], peer_1)

    for step in range(world_size):
        skip_compute = causal and (step > rank)

        if not skip_compute:
            # Await the overlapped block read from peer rank
            compute_stream.wait_stream(comm_stream)

            if step == 0:
                cur_k_bhsd = kh
                cur_v_bhsd = vh
            else:
                cur_k_pad = buf_k[step % 2]
                cur_v_pad = buf_v[step % 2]
                cur_k_bhsd = cur_k_pad[:, :, :S, :D]
                cur_v_bhsd = cur_v_pad[:, :, :S, :D]

            block_out, block_lse = _local_attn_bhsd(
                qh, cur_k_bhsd, cur_v_bhsd, softmax_scale, causal=(causal and step == 0)
            )
            
            if out is None:
                out = block_out.to(torch.float32)
                lse = block_lse.transpose(-2, -1).unsqueeze(-1)
            else:
                out, lse = _merge_out_lse_compiled(out, lse, block_out, block_lse)

        # Trigger async peer fetch for next step's blocks, hiding latency behind the next projection
        if step + 1 < world_size:
            next_peer = (rank - (step + 1)) % world_size
            with torch.cuda.stream(comm_stream):
                # Ensure the current projection matrix has fully resolved the background buffer
                comm_stream.wait_stream(compute_stream)
                ext.tk_fetch_kv(k_tk, v_tk, buf_k[(step + 1) % 2], buf_v[(step + 1) % 2], next_peer)

    torch.cuda.current_stream().synchronize()

    # Prevent sequential layer iterations from immediately overwriting PGL tensors another peer is reading
    ext.tk_barrier(barrier_tk)

    return out.to(q.dtype)