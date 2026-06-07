import os
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

# ---------------------------------------------------------------------------
# Embedded .cu source for ThunderKittens kernels
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"
#include <cuda_bf16.h>

using namespace kittens;

namespace shared {
    struct barrier_config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int NUM_BLOCKS = 1;
        static constexpr int NUM_THREADS = 1;
        static constexpr int DYNAMIC_SHARED_MEMORY = 0;
    };
    struct barrier_globals {
        static constexpr int NUM_DEVICES = 8;
        barrier_t<NUM_DEVICES> barrier;
        const int dev_idx;
    };
    __device__ inline void barrier_kernel(const barrier_globals &G) {
        barrier_all(G.barrier, {0}, G.dev_idx);
    }
}

namespace shift_all {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int NUM_THREADS = 256;
    };
    struct globals {
        static constexpr int NUM_DEVICES = 8;
        using layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, false>;
        layout k_in, v_in, dk_in, dv_in;
        layout k_out, v_out, dk_out, dv_out;
        int dev_idx;
        int cp_size;
        int numel;
        __host__ inline dim3 grid() const { return dim3((numel + config::NUM_THREADS * 2 - 1) / (config::NUM_THREADS * 2)); }
    };
    __device__ inline void kernel(const globals &G) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx * 2 < G.numel) {
            int cp_rank = G.dev_idx % G.cp_size;
            int dp_rank = G.dev_idx / G.cp_size;
            int prev_dev = dp_rank * G.cp_size + (cp_rank - 1 + G.cp_size) % G.cp_size;
            
            *(int*)(&G.k_out[G.dev_idx].data[idx * 2]) = *(int*)(&G.k_in[prev_dev].data[idx * 2]);
            *(int*)(&G.v_out[G.dev_idx].data[idx * 2]) = *(int*)(&G.v_in[prev_dev].data[idx * 2]);
            *(int*)(&G.dk_out[G.dev_idx].data[idx * 2]) = *(int*)(&G.dk_in[prev_dev].data[idx * 2]);
            *(int*)(&G.dv_out[G.dev_idx].data[idx * 2]) = *(int*)(&G.dv_in[prev_dev].data[idx * 2]);
        }
    }
}

namespace shift_dkv {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int NUM_THREADS = 256;
    };
    struct globals {
        static constexpr int NUM_DEVICES = 8;
        using layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, false>;
        layout dk_in, dv_in;
        layout dk_out, dv_out;
        int dev_idx;
        int cp_size;
        int numel;
        __host__ inline dim3 grid() const { return dim3((numel + config::NUM_THREADS * 2 - 1) / (config::NUM_THREADS * 2)); }
    };
    __device__ inline void kernel(const globals &G) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx * 2 < G.numel) {
            int cp_rank = G.dev_idx % G.cp_size;
            int dp_rank = G.dev_idx / G.cp_size;
            int prev_dev = dp_rank * G.cp_size + (cp_rank - 1 + G.cp_size) % G.cp_size;
            
            *(int*)(&G.dk_out[G.dev_idx].data[idx * 2]) = *(int*)(&G.dk_in[prev_dev].data[idx * 2]);
            *(int*)(&G.dv_out[G.dev_idx].data[idx * 2]) = *(int*)(&G.dv_in[prev_dev].data[idx * 2]);
        }
    }
}

namespace dp_all_reduce {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int NUM_THREADS = 256;
    };
    struct globals {
        static constexpr int NUM_DEVICES = 8;
        using layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, false>;
        layout dq, dk, dv;
        layout dq_out, dk_out, dv_out;
        int dev_idx;
        int cp_size;
        int dp_size;
        int numel;
        __host__ inline dim3 grid() const { return dim3((numel + config::NUM_THREADS * 2 - 1) / (config::NUM_THREADS * 2)); }
    };
    __device__ inline void kernel(const globals &G) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx * 2 < G.numel) {
            int cp_rank = G.dev_idx % G.cp_size;
            
            float2 sum_dq = {0.f, 0.f};
            float2 sum_dk = {0.f, 0.f};
            float2 sum_dv = {0.f, 0.f};
            
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            for (int d = 0; d < G.dp_size; ++d) {
                int peer = d * G.cp_size + cp_rank;
                
                bf16_2 val_dq = *(bf16_2*)(&G.dq[peer].data[idx * 2]);
                float2 f_dq = __bfloat1622float2(val_dq);
                sum_dq.x += f_dq.x; sum_dq.y += f_dq.y;
                
                bf16_2 val_dk = *(bf16_2*)(&G.dk[peer].data[idx * 2]);
                float2 f_dk = __bfloat1622float2(val_dk);
                sum_dk.x += f_dk.x; sum_dk.y += f_dk.y;
                
                bf16_2 val_dv = *(bf16_2*)(&G.dv[peer].data[idx * 2]);
                float2 f_dv = __bfloat1622float2(val_dv);
                sum_dv.x += f_dv.x; sum_dv.y += f_dv.y;
            }
            
            sum_dq.x /= G.dp_size; sum_dq.y /= G.dp_size;
            sum_dk.x /= G.dp_size; sum_dk.y /= G.dp_size;
            sum_dv.x /= G.dp_size; sum_dv.y /= G.dp_size;
            
            *(bf16_2*)(&G.dq_out[G.dev_idx].data[idx * 2]) = __float22bfloat162_rn(sum_dq);
            *(bf16_2*)(&G.dk_out[G.dev_idx].data[idx * 2]) = __float22bfloat162_rn(sum_dk);
            *(bf16_2*)(&G.dv_out[G.dev_idx].data[idx * 2]) = __float22bfloat162_rn(sum_dv);
            #endif
        }
    }
}

// Host entrypoints
void tk_barrier(kittens::py::TKParallelTensor &barrier) {
    shared::barrier_globals bg {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<shared::barrier_globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    kittens::py::launch_kernel<shared::barrier_config, shared::barrier_globals, shared::barrier_kernel>(bg);
}

void tk_shift_all(
    kittens::py::TKParallelTensor &k_out, kittens::py::TKParallelTensor &v_out, kittens::py::TKParallelTensor &dk_out, kittens::py::TKParallelTensor &dv_out,
    kittens::py::TKParallelTensor &k_in, kittens::py::TKParallelTensor &v_in, kittens::py::TKParallelTensor &dk_in, kittens::py::TKParallelTensor &dv_in,
    int cp_size, int actual_numel
) {
    shift_all::globals g {
        .k_in = kittens::py::parallel_tensor_to_pgl<shift_all::globals::layout>(k_in),
        .v_in = kittens::py::parallel_tensor_to_pgl<shift_all::globals::layout>(v_in),
        .dk_in = kittens::py::parallel_tensor_to_pgl<shift_all::globals::layout>(dk_in),
        .dv_in = kittens::py::parallel_tensor_to_pgl<shift_all::globals::layout>(dv_in),
        .k_out = kittens::py::parallel_tensor_to_pgl<shift_all::globals::layout>(k_out),
        .v_out = kittens::py::parallel_tensor_to_pgl<shift_all::globals::layout>(v_out),
        .dk_out = kittens::py::parallel_tensor_to_pgl<shift_all::globals::layout>(dk_out),
        .dv_out = kittens::py::parallel_tensor_to_pgl<shift_all::globals::layout>(dv_out),
        .dev_idx = k_in.local_rank_,
        .cp_size = cp_size,
        .numel = actual_numel
    };
    kittens::py::launch_kernel<shift_all::config, shift_all::globals, shift_all::kernel>(g);
}

void tk_shift_dkv(
    kittens::py::TKParallelTensor &dk_out, kittens::py::TKParallelTensor &dv_out,
    kittens::py::TKParallelTensor &dk_in, kittens::py::TKParallelTensor &dv_in,
    int cp_size, int actual_numel
) {
    shift_dkv::globals g {
        .dk_in = kittens::py::parallel_tensor_to_pgl<shift_dkv::globals::layout>(dk_in),
        .dv_in = kittens::py::parallel_tensor_to_pgl<shift_dkv::globals::layout>(dv_in),
        .dk_out = kittens::py::parallel_tensor_to_pgl<shift_dkv::globals::layout>(dk_out),
        .dv_out = kittens::py::parallel_tensor_to_pgl<shift_dkv::globals::layout>(dv_out),
        .dev_idx = dk_in.local_rank_,
        .cp_size = cp_size,
        .numel = actual_numel
    };
    kittens::py::launch_kernel<shift_dkv::config, shift_dkv::globals, shift_dkv::kernel>(g);
}

void tk_dp_all_reduce(
    kittens::py::TKParallelTensor &dq, kittens::py::TKParallelTensor &dk, kittens::py::TKParallelTensor &dv,
    kittens::py::TKParallelTensor &dq_out, kittens::py::TKParallelTensor &dk_out, kittens::py::TKParallelTensor &dv_out,
    int cp_size, int dp_size, int actual_numel
) {
    dp_all_reduce::globals g {
        .dq = kittens::py::parallel_tensor_to_pgl<dp_all_reduce::globals::layout>(dq),
        .dk = kittens::py::parallel_tensor_to_pgl<dp_all_reduce::globals::layout>(dk),
        .dv = kittens::py::parallel_tensor_to_pgl<dp_all_reduce::globals::layout>(dv),
        .dq_out = kittens::py::parallel_tensor_to_pgl<dp_all_reduce::globals::layout>(dq_out),
        .dk_out = kittens::py::parallel_tensor_to_pgl<dp_all_reduce::globals::layout>(dk_out),
        .dv_out = kittens::py::parallel_tensor_to_pgl<dp_all_reduce::globals::layout>(dv_out),
        .dev_idx = dq.local_rank_,
        .cp_size = cp_size,
        .dp_size = dp_size,
        .numel = actual_numel
    };
    kittens::py::launch_kernel<dp_all_reduce::config, dp_all_reduce::globals, dp_all_reduce::kernel>(g);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_barrier", &tk_barrier);
    m.def("tk_shift_all", &tk_shift_all);
    m.def("tk_shift_dkv", &tk_shift_dkv);
    m.def("tk_dp_all_reduce", &tk_dp_all_reduce);
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
            "tk_ring_attn_bwd", CUDA_SRC,
            extra_cuda_cflags=TK_CUDA_FLAGS,
            extra_include_paths=[
                os.path.join(os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens"), "include"),
                os.path.join(os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens"), "prototype"),
            ],
            extra_ldflags=["-lcuda"],
        )
    return _ext

def _ensure_ext_jit():
    global _ext_jit_ready
    if _ext_jit_ready:
        return _get_ext()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext

def _local_attn_backward(
    dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    out: torch.Tensor, softmax_lse: torch.Tensor,
    scale: float, causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    doh = dout.transpose(1, 2).float()
    outh = out.transpose(1, 2).float()

    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        sq, sk = q.size(1), k.size(1)
        mask = torch.triu(torch.ones(sq, sk, device=q.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = torch.exp(scores - softmax_lse)
    dP = torch.matmul(doh, vh.transpose(-2, -1))
    row_dot = (doh * outh).sum(dim=-1, keepdim=True)
    dS = probs * (dP - row_dot)

    dQ = torch.matmul(dS, kh) * scale
    dK = torch.matmul(dS.transpose(-2, -1), qh) * scale
    dV = torch.matmul(probs.transpose(-2, -1), doh)

    return (
        dQ.transpose(1, 2).contiguous(),
        dK.transpose(1, 2).contiguous(),
        dV.transpose(1, 2).contiguous(),
    )

def alloc_pair(ext, padded_size, base_offset):
    # Differentiate shape allocations to circumvent caching identical underlying buffers
    t0 = get_or_create_parallel_tensor(ext, (padded_size + base_offset,), torch.bfloat16, False)
    t1 = get_or_create_parallel_tensor(ext, (padded_size + base_offset + 1,), torch.bfloat16, False)
    return [t0, t1]

@torch.no_grad()
def solution(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _ensure_ext_jit()

    world = dist.get_world_size()
    assert world == 8, "This ThunderKittens integration expects an 8-GPU domain (NUM_DEVICES=8)."
    
    cp_group = cp_group or dist.group.WORLD
    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)
    
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    actual_numel = q.numel()
    padded = ((actual_numel + 511) // 512) * 512
    shape = q.shape

    # Pre-allocate double buffers to safely fuse computation + pipeline shifts.
    tk_k  = alloc_pair(ext, padded, 10)
    tk_v  = alloc_pair(ext, padded, 20)
    tk_dk = alloc_pair(ext, padded, 30)
    tk_dv = alloc_pair(ext, padded, 40)

    tk_dq = get_or_create_parallel_tensor(ext, (padded + 50,), torch.bfloat16, False)
    
    barrier_tk = get_or_create_barrier(ext, num_devices=8)
    
    # Initialize 0th index with current rank's K and V
    tk_k[0].data_[:actual_numel].view(shape).copy_(k)
    tk_v[0].data_[:actual_numel].view(shape).copy_(v)

    lse_4d = softmax_lse.unsqueeze(-1)
    
    for step in range(cp_size):
        cur_idx = step % 2
        nxt_idx = (step + 1) % 2
        
        v_k_cur = tk_k[cur_idx].data_[:actual_numel].view(shape)
        v_v_cur = tk_v[cur_idx].data_[:actual_numel].view(shape)
        v_dk_cur = tk_dk[cur_idx].data_[:actual_numel].view(shape)
        v_dv_cur = tk_dv[cur_idx].data_[:actual_numel].view(shape)
        v_dq = tk_dq.data_[:actual_numel].view(shape)
        
        if step <= cp_rank or not causal:
            block_dq, block_dk, block_dv = _local_attn_backward(
                dout, q, v_k_cur, v_v_cur, out, lse_4d, float(softmax_scale), causal=(causal and step == 0)
            )
            if step == 0:
                v_dq.copy_(block_dq)
                v_dk_cur.copy_(block_dk)
                v_dv_cur.copy_(block_dv)
            else:
                v_dq.add_(block_dq)
                # Adds directly into in-place TK buffer (accumulating received gradients + local computed)
                v_dk_cur.copy_(block_dk + v_dk_cur)
                v_dv_cur.copy_(block_dv + v_dv_cur)
                
        ext.tk_barrier(barrier_tk)
        
        # P2P rotate buffers to adjacent peer logic completely device-side
        if step + 1 != cp_size:
            ext.tk_shift_all(
                tk_k[nxt_idx], tk_v[nxt_idx], tk_dk[nxt_idx], tk_dv[nxt_idx],
                tk_k[cur_idx], tk_v[cur_idx], tk_dk[cur_idx], tk_dv[cur_idx],
                cp_size, actual_numel
            )
        else:
            ext.tk_shift_dkv(
                tk_dk[nxt_idx], tk_dv[nxt_idx],
                tk_dk[cur_idx], tk_dv[cur_idx],
                cp_size, actual_numel
            )
            
        ext.tk_barrier(barrier_tk)

    final_idx = cp_size % 2
    dp_size = dist.get_world_size(dp_group) if dp_group is not None else 1

    if dp_size > 1:
        tk_dq_out = get_or_create_parallel_tensor(ext, (padded + 60,), torch.bfloat16, False)
        tk_dk_out = get_or_create_parallel_tensor(ext, (padded + 70,), torch.bfloat16, False)
        tk_dv_out = get_or_create_parallel_tensor(ext, (padded + 80,), torch.bfloat16, False)

        ext.tk_barrier(barrier_tk)
        ext.tk_dp_all_reduce(
            tk_dq, tk_dk[final_idx], tk_dv[final_idx],
            tk_dq_out, tk_dk_out, tk_dv_out,
            cp_size, dp_size, actual_numel
        )
        ext.tk_barrier(barrier_tk)

        final_dq = tk_dq_out.data_[:actual_numel].view(shape).clone()
        final_dk = tk_dk_out.data_[:actual_numel].view(shape).clone()
        final_dv = tk_dv_out.data_[:actual_numel].view(shape).clone()
    else:
        final_dq = tk_dq.data_[:actual_numel].view(shape).clone()
        final_dk = tk_dk[final_idx].data_[:actual_numel].view(shape).clone()
        final_dv = tk_dv[final_idx].data_[:actual_numel].view(shape).clone()

    return final_dq, final_dk, final_dv