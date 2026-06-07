import os
import torch
import torch.distributed as dist
from typing import Tuple
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source: Fused RoPE + PGL All-Gather 
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace rope_all_gather {

struct config {
    static constexpr int NUM_THREADS = 256;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    // 1D layout is sufficient, we calculate multi-dimensional indices manually
    using parallel_layout = pgl<gl<bf16, -1>, NUM_DEVICES, true>;
    
    parallel_layout q_out;
    parallel_layout k_out;
    const bf16* q_local;
    const bf16* k_local;
    const bf16* cos_local;
    const bf16* sin_local;
    int B, S_local, H, D;
    int dev_idx;
    
    __host__ inline dim3 grid() const {
        return dim3((B * S_local * H * (D / 2) + config::NUM_THREADS - 1) / config::NUM_THREADS);
    }
};

__device__ inline void kernel(const globals &G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = G.B * G.S_local * G.H * (G.D / 2);
    if (idx >= total) return;
    
    // Each thread processes 2 contiguous feature elements (bf16_2) 
    int d_half = idx % (G.D / 2);
    int d = d_half * 2;
    int h = (idx / (G.D / 2)) % G.H;
    int s = (idx / (G.D / 2 * G.H)) % G.S_local;
    int b = idx / (G.D / 2 * G.H * G.S_local);
    
    // Compute exact memory bounds
    int in_idx = b * (G.S_local * G.H * G.D) + s * (G.H * G.D) + h * G.D + d;
    int cos_idx = b * (G.S_local * G.D) + s * G.D + d;
    
    // Partner indices for rotary math
    int pair_d = (d < G.D / 2) ? (d + G.D / 2) : (d - G.D / 2);
    int pair_idx = b * (G.S_local * G.H * G.D) + s * (G.H * G.D) + h * G.D + pair_d;
    
    // Load as bf16_2 mapping strictly to 4-byte aligned blocks
    bf16_2 q_val = *reinterpret_cast<const bf16_2*>(&G.q_local[in_idx]);
    bf16_2 q_pair = *reinterpret_cast<const bf16_2*>(&G.q_local[pair_idx]);
    bf16_2 k_val = *reinterpret_cast<const bf16_2*>(&G.k_local[in_idx]);
    bf16_2 k_pair = *reinterpret_cast<const bf16_2*>(&G.k_local[pair_idx]);
    
    bf16_2 cos_val = *reinterpret_cast<const bf16_2*>(&G.cos_local[cos_idx]);
    bf16_2 sin_val = *reinterpret_cast<const bf16_2*>(&G.sin_local[cos_idx]);
    
    float2 q_f = __bfloat1622float2(q_val);
    float2 q_pair_f = __bfloat1622float2(q_pair);
    float2 k_f = __bfloat1622float2(k_val);
    float2 k_pair_f = __bfloat1622float2(k_pair);
    float2 cos_f = __bfloat1622float2(cos_val);
    float2 sin_f = __bfloat1622float2(sin_val);
    
    float2 q_rot_f, k_rot_f;
    if (d < G.D / 2) {
        q_rot_f.x = -q_pair_f.x; q_rot_f.y = -q_pair_f.y;
        k_rot_f.x = -k_pair_f.x; k_rot_f.y = -k_pair_f.y;
    } else {
        q_rot_f = q_pair_f;
        k_rot_f = k_pair_f;
    }
    
    float2 q_out_f;
    q_out_f.x = q_f.x * cos_f.x + q_rot_f.x * sin_f.x;
    q_out_f.y = q_f.y * cos_f.y + q_rot_f.y * sin_f.y;
    
    float2 k_out_f;
    k_out_f.x = k_f.x * cos_f.x + k_rot_f.x * sin_f.x;
    k_out_f.y = k_f.y * cos_f.y + k_rot_f.y * sin_f.y;
    
    bf16_2 q_out = __float22bfloat162_rn(q_out_f);
    bf16_2 k_out = __float22bfloat162_rn(k_out_f);
    
    // Target global offset for gather-reconstruction
    int s_out = s + G.dev_idx * G.S_local;
    int S_global = G.S_local * globals::NUM_DEVICES;
    int out_idx = b * (S_global * G.H * G.D) + s_out * (G.H * G.D) + h * G.D + d;
    
    // Broadcast computed slice directly to target locations matching sequence shards 
    kittens::multimem<bf16_2>::st(reinterpret_cast<bf16_2*>(&G.q_out.mc_ptr[out_idx]), q_out);
    kittens::multimem<bf16_2>::st(reinterpret_cast<bf16_2*>(&G.k_out.mc_ptr[out_idx]), k_out);
}

} // namespace rope_all_gather

namespace rope_barrier {

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

} // namespace rope_barrier

void entrypoint(
    kittens::py::TKParallelTensor &q_out,
    kittens::py::TKParallelTensor &k_out,
    uintptr_t q_local_ptr,
    uintptr_t k_local_ptr,
    uintptr_t cos_local_ptr,
    uintptr_t sin_local_ptr,
    kittens::py::TKParallelTensor &barrier,
    int B, int S_local, int H, int D
) {
    kittens::py::parallel_tensor_check(q_out, k_out, barrier);

    rope_all_gather::globals rope_G {
        .q_out = kittens::py::parallel_tensor_to_pgl<typename rope_all_gather::globals::parallel_layout>(q_out),
        .k_out = kittens::py::parallel_tensor_to_pgl<typename rope_all_gather::globals::parallel_layout>(k_out),
        .q_local = reinterpret_cast<const bf16*>(q_local_ptr),
        .k_local = reinterpret_cast<const bf16*>(k_local_ptr),
        .cos_local = reinterpret_cast<const bf16*>(cos_local_ptr),
        .sin_local = reinterpret_cast<const bf16*>(sin_local_ptr),
        .B = B,
        .S_local = S_local,
        .H = H,
        .D = D,
        .dev_idx = q_out.local_rank_
    };

    rope_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<rope_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // Synchronize cluster setup -> run async computation & multicast out -> synchronize visibility
    kittens::py::launch_kernel<rope_barrier::config, rope_barrier::globals, rope_barrier::kernel>(barrier_G);
    kittens::py::launch_kernel<rope_all_gather::config, rope_all_gather::globals, rope_all_gather::kernel>(rope_G);
    kittens::py::launch_kernel<rope_barrier::config, rope_barrier::globals, rope_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_rope_all_gather", &entrypoint);
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
NUM_DEVICES = 8


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_rope_allgather_ext",
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
    if not dist.is_initialized():
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
    q_local: torch.Tensor, 
    k_local: torch.Tensor, 
    cos_local: torch.Tensor, 
    sin_local: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, S_local, H, D = q_local.shape
    
    # Kernel requires bfloat16 alignment matching 2 floats layout assumption (D % 4 == 0) and exactly NUM_DEVICES GPUs
    if not dist.is_initialized() or dist.get_world_size() != NUM_DEVICES or D % 4 != 0:
        cos = cos_local.unsqueeze(2)
        sin = sin_local.unsqueeze(2)
        
        half_dim = D // 2
        q_x1, q_x2 = q_local[..., :half_dim], q_local[..., half_dim:]
        q_rot = torch.cat((-q_x2, q_x1), dim=-1)
        k_x1, k_x2 = k_local[..., :half_dim], k_local[..., half_dim:]
        k_rot = torch.cat((-k_x2, k_x1), dim=-1)
        
        q_embed_local = (q_local * cos) + (q_rot * sin)
        k_embed_local = (k_local * cos) + (k_rot * sin)
        
        if not dist.is_initialized():
            return q_embed_local, k_embed_local
            
        world_size = dist.get_world_size()
        q_gather_list = [torch.empty_like(q_embed_local) for _ in range(world_size)]
        k_gather_list = [torch.empty_like(k_embed_local) for _ in range(world_size)]
        
        dist.all_gather(q_gather_list, q_embed_local.contiguous())
        dist.all_gather(k_gather_list, k_embed_local.contiguous())
        
        return torch.cat(q_gather_list, dim=1), torch.cat(k_gather_list, dim=1)

    world = dist.get_world_size()
    S_global = S_local * world
    n_out = B * S_global * H * D

    # Pad parallel dimensions for broker stability requirement
    ALIGNMENT = 4096
    padded_out = ((n_out + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    q_local_c = q_local.to(torch.bfloat16).contiguous()
    k_local_c = k_local.to(torch.bfloat16).contiguous()
    cos_local_c = cos_local.to(torch.bfloat16).contiguous()
    sin_local_c = sin_local.to(torch.bfloat16).contiguous()

    ext = _ensure_ext_jit()

    q_out_tk = get_or_create_parallel_tensor(ext, (padded_out,), torch.bfloat16, multicast=True)
    k_out_tk = get_or_create_parallel_tensor(ext, (padded_out,), torch.bfloat16, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world)

    ext.tk_rope_all_gather(
        q_out_tk,
        k_out_tk,
        q_local_c.data_ptr(),
        k_local_c.data_ptr(),
        cos_local_c.data_ptr(),
        sin_local_c.data_ptr(),
        barrier_tk,
        B, S_local, H, D
    )

    q_global = q_out_tk.data_[:n_out].view(B, S_global, H, D).clone()
    k_global = k_out_tk.data_[:n_out].view(B, S_global, H, D).clone()
    
    # Clean fallback format logic
    orig_dtype = q_local.dtype
    if orig_dtype != torch.bfloat16:
        q_global = q_global.to(orig_dtype)
        k_global = k_global.to(orig_dtype)

    return q_global, k_global