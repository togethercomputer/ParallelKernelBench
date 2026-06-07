import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import get_or_create_parallel_tensor

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source: TMA KV Fetch + Fused LSE Merge
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace kittens;

namespace tma_fetch {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int MIN_BLOCKS_PER_SM = 8;
        static constexpr int NUM_THREADS = 1;
    };

    template <int D>
    struct globals {
        static constexpr int NUM_DEVICES = 8;
        static constexpr int ROW_BLOCK_SIZE = 16;
        static constexpr int COL_BLOCK_SIZE = D;

        using tile_t = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
        using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, tile_t>, NUM_DEVICES, false>;
        using local_layout = gl<bf16, -1, -1, -1, -1, tile_t>;

        parallel_layout K_pgl;
        parallel_layout V_pgl;
        local_layout next_K;
        local_layout next_V;
        int peer_idx;

        __host__ inline dim3 grid() const {
            return dim3(next_K.batch() * next_K.depth() * (next_K.rows() / ROW_BLOCK_SIZE));
        }

        __host__ inline int dynamic_shared_memory() const {
            return static_cast<int>(2 * sizeof(tile_t) + 1024);
        }
    };

    template <int D>
    __device__ inline void kernel(const globals<D> &G) {
        extern __shared__ int __shm[];
        tma_swizzle_allocator allocator((int*)&__shm[0]);
        typename globals<D>::tile_t &k_tile = allocator.allocate<typename globals<D>::tile_t>();
        typename globals<D>::tile_t &v_tile = allocator.allocate<typename globals<D>::tile_t>();

        int task_idx = blockIdx.x;
        int c_idx = 0; // col dimension is exactly D, handled by 1 block
        int r_idx = task_idx % (G.next_K.rows() / globals<D>::ROW_BLOCK_SIZE); task_idx /= (G.next_K.rows() / globals<D>::ROW_BLOCK_SIZE);
        int d_idx = task_idx % G.next_K.depth(); task_idx /= G.next_K.depth();
        int b_idx = task_idx;

        __shared__ semaphore arrived;
        init_semaphore(arrived, 0, 1);
        
        tma::expect_bytes(arrived, sizeof(k_tile) + sizeof(v_tile));
        tma::load_async(k_tile, G.K_pgl[G.peer_idx], {b_idx, d_idx, r_idx, c_idx}, arrived);
        tma::load_async(v_tile, G.V_pgl[G.peer_idx], {b_idx, d_idx, r_idx, c_idx}, arrived);
        
        wait(arrived, 0);
        
        tma::store_async(G.next_K, k_tile, {b_idx, d_idx, r_idx, c_idx});
        tma::store_async(G.next_V, v_tile, {b_idx, d_idx, r_idx, c_idx});
    }
}

__global__ void merge_lse_kernel(
    float* __restrict__ out,
    const float* __restrict__ lse,
    const __nv_bfloat16* __restrict__ block_out,
    const float* __restrict__ block_lse,
    int num_elements,
    int head_dim,
    int seq_len,
    int num_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int d_idx = idx % head_dim;
        int tmp = idx / head_dim;
        int h_idx = tmp % num_heads;
        tmp = tmp / num_heads;
        int s_idx = tmp % seq_len;
        int b_idx = tmp / seq_len;
        
        int lse_idx = (b_idx * num_heads + h_idx) * seq_len + s_idx;
        
        float bo_val = __bfloat162float(block_out[idx]);
        float bl_val = block_lse[lse_idx];
        
        float o_val = out[idx];
        float l_val = lse[lse_idx];
        
        float diff = bl_val - l_val;
        float sig = 1.0f / (1.0f + expf(-diff));
        float new_o = o_val - sig * (o_val - bo_val);
        
        out[idx] = new_o;
    }
}

template <int D>
void launch_fetch_impl(
    kittens::py::TKParallelTensor &K_pgl,
    kittens::py::TKParallelTensor &V_pgl,
    torch::Tensor next_K,
    torch::Tensor next_V,
    int peer_idx
) {
    auto next_K_tk = kittens::py::tensor_to_gl<typename tma_fetch::globals<D>::local_layout>(next_K);
    auto next_V_tk = kittens::py::tensor_to_gl<typename tma_fetch::globals<D>::local_layout>(next_V);

    tma_fetch::globals<D> G {
        .K_pgl = kittens::py::parallel_tensor_to_pgl<typename tma_fetch::globals<D>::parallel_layout>(K_pgl),
        .V_pgl = kittens::py::parallel_tensor_to_pgl<typename tma_fetch::globals<D>::parallel_layout>(V_pgl),
        .next_K = next_K_tk,
        .next_V = next_V_tk,
        .peer_idx = peer_idx
    };
    kittens::py::launch_kernel<tma_fetch::config, tma_fetch::globals<D>, tma_fetch::kernel<D>>(G);
}

void launch_fetch(
    kittens::py::TKParallelTensor &K_pgl,
    kittens::py::TKParallelTensor &V_pgl,
    torch::Tensor next_K,
    torch::Tensor next_V,
    int peer_idx,
    int head_dim
) {
    if (head_dim == 64) launch_fetch_impl<64>(K_pgl, V_pgl, next_K, next_V, peer_idx);
    else if (head_dim == 128) launch_fetch_impl<128>(K_pgl, V_pgl, next_K, next_V, peer_idx);
    else TORCH_CHECK(false, "head_dim must be 64 or 128 for optimized TMA fetch");
}

void launch_merge(
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor block_out,
    torch::Tensor block_lse
) {
    int num_elements = out.numel();
    int head_dim = out.size(3);
    int num_heads = out.size(2);
    int seq_len = out.size(1);
    
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    merge_lse_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(),
        lse.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(block_out.data_ptr<at::BFloat16>()),
        block_lse.data_ptr<float>(),
        num_elements,
        head_dim,
        seq_len,
        num_heads
    );
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("launch_fetch", &launch_fetch);
    m.def("launch_merge", &launch_merge);
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

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_ring_attn_ext", CUDA_SRC, extra_cuda_cflags=TK_CUDA_FLAGS,
            extra_include_paths=[os.path.join(TK_ROOT, "include"), os.path.join(TK_ROOT, "prototype")],
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
# Torch + TK Python Interop
# ---------------------------------------------------------------------------

def _local_attn_math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, causal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(q.size(1), k.size(1), device=q.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    block_lse = torch.logsumexp(scores, dim=-1)
    block_out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).contiguous()
    return block_out, block_lse


def _pp_recv_forward(pp_group: dist.ProcessGroup, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    prev_rank = dist.get_global_rank(pp_group, (dist.get_rank(pp_group) - 1) % dist.get_world_size(pp_group))
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.irecv(buf, prev_rank, group=pp_group).wait()
    return buf


def _pp_send_forward(pp_group: dist.ProcessGroup, tensor: torch.Tensor) -> None:
    next_rank = dist.get_global_rank(pp_group, (dist.get_rank(pp_group) + 1) % dist.get_world_size(pp_group))
    dist.isend(tensor.contiguous(), next_rank, group=pp_group).wait()


def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    ext = _ensure_ext_jit()
    cp_group = cp_group or dist.group.WORLD
    
    is_first, is_last = True, True
    if pp_group is not None and dist.get_world_size(pp_group) > 1:
        pp_rank, pp_size = dist.get_rank(pp_group), dist.get_world_size(pp_group)
        is_first, is_last = (pp_rank == 0), (pp_rank == pp_size - 1)

    # 1. Pipeline-parallel step boundary
    stage_input = hidden_states if is_first else _pp_recv_forward(
        pp_group, tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device
    )

    # 2. Extract Q, K, V
    B, S, D_hidden = stage_input.shape
    head_dim = w_qkv.shape[0] // 3 // num_heads
    scale = float(softmax_scale if softmax_scale is not None else head_dim ** -0.5)

    qkv = F.linear(stage_input, w_qkv).view(B, S, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # 3. Context-Parallel Ring Attention Setup
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)

    if cp_size == 1:
        block_out, _ = _local_attn_math(q, k, v, scale, causal)
        stage_output = F.linear(block_out.to(q.dtype).reshape(B, S, -1), w_o)
    else:
        # Pre-pad to 16 for TK blocks
        padded_s = ((S + 15) // 16) * 16
        
        # TK Symmetric Memory mapping for TMA peer fetch
        K_tk = get_or_create_parallel_tensor(ext, (B, num_heads, padded_s, head_dim), torch.bfloat16, multicast=False)
        V_tk = get_or_create_parallel_tensor(ext, (B, num_heads, padded_s, head_dim), torch.bfloat16, multicast=False)

        k_tr = k.transpose(1, 2).contiguous()
        v_tr = v.transpose(1, 2).contiguous()
        
        K_tk.data_[:, :, :S, :].copy_(k_tr)
        V_tk.data_[:, :, :S, :].copy_(v_tr)
        
        # Ensure all peers have flushed memory
        dist.barrier(cp_group)

        fetch_stream = torch.cuda.Stream()
        out, lse = None, None
        k_current, v_current = k_tr, v_tr

        for step in range(cp_size):
            next_k, next_v = None, None
            
            # Initiate Async TMA Fetch of the NEXT chunk
            if step + 1 != cp_size:
                peer_cp_rank = (cp_rank - (step + 1)) % cp_size
                peer_global_rank = dist.get_global_rank(cp_group, peer_cp_rank)
                
                next_k_full = torch.empty((B, num_heads, padded_s, head_dim), dtype=torch.bfloat16, device=q.device)
                next_v_full = torch.empty((B, num_heads, padded_s, head_dim), dtype=torch.bfloat16, device=q.device)
                
                with torch.cuda.stream(fetch_stream):
                    ext.launch_fetch(K_tk, V_tk, next_k_full, next_v_full, peer_global_rank % 8, head_dim)
                
                next_k = next_k_full[:, :, :S, :]
                next_v = next_v_full[:, :, :S, :]

            # Overlapped Math compute & Merge
            if (not causal) or step <= cp_rank:
                k_loc = k_current.transpose(1, 2)
                v_loc = v_current.transpose(1, 2)
                block_out, block_lse = _local_attn_math(q, k_loc, v_loc, scale, causal=(causal and step == 0))

                if out is None:
                    out = block_out.to(torch.float32).clone()
                    lse = block_lse.to(torch.float32).clone()
                else:
                    ext.launch_merge(out, lse, block_out, block_lse)
                    lse.copy_(lse - F.logsigmoid(lse - block_lse))

            # Synchronize background stream before next iteration
            if step + 1 != cp_size:
                torch.cuda.current_stream().wait_stream(fetch_stream)
                k_current, v_current = next_k, next_v

        stage_output = F.linear(out.to(q.dtype).reshape(B, S, -1), w_o)

    # 4. Pipeline-parallel step boundary
    if not is_last and pp_group is not None:
        _pp_send_forward(pp_group, stage_output)

    return stage_output