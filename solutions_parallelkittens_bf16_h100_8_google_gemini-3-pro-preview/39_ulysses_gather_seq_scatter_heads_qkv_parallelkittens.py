import os
import math
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")


# ---------------------------------------------------------------------------
# Embedded .cu source: Fused Reshape & All-To-All over NVLink PGL
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace fused_all_to_all {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    // We treat the buffer as a flat array via gl to handle arbitrary striding
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, false>;

    parallel_layout output;
    parallel_layout input;
    
    int B;
    int S;
    int R;
    int C;
    int dev_idx;
    int total_vec;

    __host__ inline dim3 grid() const {
        return dim3((total_vec + config::NUM_THREADS - 1) / config::NUM_THREADS);
    }
};

template <int VEC_SIZE> struct Vec;
template <> struct Vec<8> { using type = float4; };
template <> struct Vec<4> { using type = float2; };
template <> struct Vec<2> { using type = float; };
template <> struct Vec<1> { using type = uint16_t; };

template<int VEC_SIZE>
__device__ inline void kernel(const globals &G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= G.total_vec || G.C == 0) return;

    int C_vec = G.C / VEC_SIZE;
    
    int c_v = idx % C_vec;
    int tmp = idx / C_vec;
    int r = tmp % G.R;
    tmp /= G.R;
    int s = tmp % G.S;
    int b = tmp / G.S;
    
    int c = c_v * VEC_SIZE;
    
    // Chunk boundary logic
    int w_dst = c / (G.C / globals::NUM_DEVICES);
    int c_out = c % (G.C / globals::NUM_DEVICES);
    
    // Sequence gather concatenation logic
    int s_out = G.dev_idx * G.S + s;
    
    // Flatten 4D indices to 1D offsets
    int src_offset = ((b * G.S + s) * G.R + r) * G.C + c;
    int dst_offset = ((b * (G.S * globals::NUM_DEVICES) + s_out) * G.R + r) * (G.C / globals::NUM_DEVICES) + c_out;
    
    using V = typename Vec<VEC_SIZE>::type;
    V val = *reinterpret_cast<const V*>(&G.input[G.dev_idx].data[src_offset]);
    *reinterpret_cast<V*>(&G.output[w_dst].data[dst_offset]) = val;
}

} // namespace fused_all_to_all

namespace all_to_all_barrier {

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

} // namespace all_to_all_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    kittens::py::TKParallelTensor &input,
    kittens::py::TKParallelTensor &barrier,
    int B, int S, int R, int C
) {
    kittens::py::parallel_tensor_check(output, input, barrier);

    int C_per_W = C / fused_all_to_all::globals::NUM_DEVICES;
    
    // Opt for highest aligned vector size to saturate NVLink
    int vec_size = 1;
    if (C_per_W > 0 && C_per_W % 8 == 0) vec_size = 8;
    else if (C_per_W > 0 && C_per_W % 4 == 0) vec_size = 4;
    else if (C_per_W > 0 && C_per_W % 2 == 0) vec_size = 2;

    fused_all_to_all::globals all_to_all_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename fused_all_to_all::globals::parallel_layout>(output),
        .input = kittens::py::parallel_tensor_to_pgl<typename fused_all_to_all::globals::parallel_layout>(input),
        .B = B,
        .S = S,
        .R = R,
        .C = C,
        .dev_idx = input.local_rank_,
        .total_vec = (B * S * R * C) / vec_size
    };

    all_to_all_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_to_all_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);

    if (vec_size == 8) {
        kittens::py::launch_kernel<fused_all_to_all::config, fused_all_to_all::globals, fused_all_to_all::kernel<8>>(all_to_all_G);
    } else if (vec_size == 4) {
        kittens::py::launch_kernel<fused_all_to_all::config, fused_all_to_all::globals, fused_all_to_all::kernel<4>>(all_to_all_G);
    } else if (vec_size == 2) {
        kittens::py::launch_kernel<fused_all_to_all::config, fused_all_to_all::globals, fused_all_to_all::kernel<2>>(all_to_all_G);
    } else {
        kittens::py::launch_kernel<fused_all_to_all::config, fused_all_to_all::globals, fused_all_to_all::kernel<1>>(all_to_all_G);
    }

    kittens::py::launch_kernel<all_to_all_barrier::config, all_to_all_barrier::globals, all_to_all_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_fused_all_to_all", &entrypoint);
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
            "tk_fused_all_to_all_ext",
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
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


# ---------------------------------------------------------------------------
# Fallback Reference Implementations (Exact compatibility for non-H100/world!=8)
# ---------------------------------------------------------------------------
def _pad_tensor(x: Tensor, dim: int, padding_size: int, padding_value: int = 0) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.full(shape, padding_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)

def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[tuple(slc)]

def _all_to_all_single(x: Tensor, scatter_dim: int, gather_dim: int, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
    group = group or dist.group.WORLD
    sp_world_size = dist.get_world_size(group)
    if scatter_dim != 0:
        gather_dim_bef = x.shape[gather_dim]
        scatter_dim_bef = x.shape[scatter_dim]
        x = (x.reshape([gather_dim_bef, sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
             .transpose(0, 1)
             .reshape([gather_dim_bef * sp_world_size, scatter_dim_bef // sp_world_size] + list(x.shape[2:]))
             .contiguous())
    output = torch.empty_like(x)
    comm = dist.all_to_all_single(output, x.contiguous(), group=group, async_op=async_op)
    if scatter_dim == 0:
        output = torch.cat(output.split(x.size(0) // sp_world_size), dim=gather_dim)
    return output

def _all_to_all(local_input: Tensor, scatter_dim: int, gather_dim: int, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
    group = group or dist.group.WORLD
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    return torch.cat(output_list, dim=gather_dim).contiguous()

def _all_to_all_tensor(x: Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup, async_op: bool = False):
    if scatter_dim <= 1 and gather_dim <= 1:
        return _all_to_all_single(x, scatter_dim, gather_dim, group, async_op)
    return _all_to_all(x, scatter_dim, gather_dim, group, async_op)

class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, local_input: Tensor, scatter_dim: int, gather_dim: int, async_op: bool) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return _all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

def gather_seq_scatter_heads_qkv(
    qkv_tensor: Tensor, seq_dim: int, unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True, async_op: bool = False, group: Optional[ProcessGroup] = None
) -> Tensor:
    group = group or dist.group.WORLD
    if not group: return qkv_tensor
    sp_world = dist.get_world_size(group)
    orig_shape = qkv_tensor.shape
    scatter_dim = qkv_tensor.dim()
    bef_all2all_shape = list(orig_shape)
    qkv_proj_dim = bef_all2all_shape[-1]
    bef_all2all_shape = bef_all2all_shape[:-1] + [3, qkv_proj_dim // 3]
    qkv_tensor = qkv_tensor.view(bef_all2all_shape)
    qkv_tensor = _SeqAllToAll.apply(group, qkv_tensor, scatter_dim, seq_dim, async_op)
    if restore_shape:
        out_shape = list(orig_shape)
        out_shape[seq_dim] *= sp_world
        out_shape[-1] = qkv_proj_dim // sp_world
        qkv_tensor = qkv_tensor.view(out_shape)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = qkv_tensor.size(seq_dim) - unpadded_dim_size
        qkv_tensor = _unpad_tensor(qkv_tensor, seq_dim, padding_size)
    return qkv_tensor


# ---------------------------------------------------------------------------
# ParallelKittens Implementation
# ---------------------------------------------------------------------------
@torch.no_grad()
def solution(
    qkv_tensor: torch.Tensor,
    seq_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    if not group:
        return qkv_tensor

    sp_world = dist.get_world_size(group)
    if sp_world == 1:
        return qkv_tensor

    orig_shape = qkv_tensor.shape
    qkv_proj_dim = orig_shape[-1]
    bef_shape = list(orig_shape[:-1]) + [3, qkv_proj_dim // 3]

    # Calculate collapsed logical dimensions
    B = math.prod(bef_shape[:seq_dim]) if seq_dim > 0 else 1
    S = bef_shape[seq_dim]
    R = math.prod(bef_shape[seq_dim + 1 : -1])
    C = bef_shape[-1]
    numel = B * S * R * C

    # The custom ThunderKittens collective is hard-coded for world_size=8
    # Fallback to pure PyTorch NCCL path if constraints are missed (e.g. padding not divisible)
    if sp_world != 8 or C % sp_world != 0 or numel == 0:
        return gather_seq_scatter_heads_qkv(
            qkv_tensor,
            seq_dim=seq_dim,
            unpadded_dim_size=unpadded_dim_size,
            restore_shape=restore_shape,
            async_op=False,
            group=group,
        )

    ext = _ensure_ext_jit()

    original_dtype = qkv_tensor.dtype
    if original_dtype != torch.bfloat16:
        qkv_tensor = qkv_tensor.to(torch.bfloat16)

    input_tk = get_or_create_parallel_tensor(ext, (numel,), torch.bfloat16, multicast=False)
    output_tk = get_or_create_parallel_tensor(ext, (numel,), torch.bfloat16, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=sp_world)

    # Coalesce directly into symmetric allocation space
    input_tk.data_[:numel].copy_(qkv_tensor.contiguous().view(-1))

    # Single-step P2P gather/scatter
    ext.tk_fused_all_to_all(output_tk, input_tk, barrier_tk, B, S, R, C)

    out_bef_shape = bef_shape.copy()
    out_bef_shape[seq_dim] *= sp_world
    out_bef_shape[-1] //= sp_world

    out_tensor = output_tk.data_[:numel].clone().view(out_bef_shape)

    if restore_shape:
        out_shape = list(orig_shape)
        out_shape[seq_dim] *= sp_world
        out_shape[-1] = qkv_proj_dim // sp_world
        out_tensor = out_tensor.view(out_shape)

    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = out_tensor.size(seq_dim) - unpadded_dim_size
        out_tensor = _unpad_tensor(out_tensor, seq_dim, padding_size)

    return out_tensor.to(original_dtype)