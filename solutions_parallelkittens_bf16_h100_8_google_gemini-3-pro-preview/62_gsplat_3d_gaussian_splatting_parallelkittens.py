import os
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

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

using namespace kittens;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
};

using pgl_i32 = pgl<gl<int32_t, -1, -1, -1, -1>, 8, false>;

struct globals {
    const float* means;
    const float* quats;
    const float* scales;
    const float* opacities;
    const float* colors;
    
    const float* viewmats;
    const float* Ks;
    
    const int* c_world_offsets;
    const int* n_world_offsets;
    int* local_send_counts;
    
    int N;
    int C_total;
    int D;
    int MAX_CAPACITY;
    int WORDS;
    
    int image_width;
    int image_height;
    float eps2d;
    float near_plane;
    float far_plane;
    int dev_idx;
    int num_devices;
    
    pgl_i32 p2p_buffer;
    
    __host__ inline dim3 grid() const {
        return dim3((N + config::NUM_THREADS - 1) / config::NUM_THREADS > 0 ? 
                    (N + config::NUM_THREADS - 1) / config::NUM_THREADS : 1);
    }
};

__device__ inline void kernel(const globals &G) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= G.N) return;
    
    float mean[3] = { G.means[n*3], G.means[n*3+1], G.means[n*3+2] };
    float quat[4] = { G.quats[n*4], G.quats[n*4+1], G.quats[n*4+2], G.quats[n*4+3] };
    float scale[3] = { G.scales[n*3], G.scales[n*3+1], G.scales[n*3+2] };
    float opacity = G.opacities[n];
    
    float q_len = sqrtf(quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]);
    float w = quat[0]/q_len, x = quat[1]/q_len, y = quat[2]/q_len, z = quat[3]/q_len;
    
    float R[3][3] = {
        {1.0f - 2.0f*(y*y + z*z), 2.0f*(x*y - w*z), 2.0f*(x*z + w*y)},
        {2.0f*(x*y + w*z), 1.0f - 2.0f*(x*x + z*z), 2.0f*(y*z - w*x)},
        {2.0f*(x*z - w*y), 2.0f*(y*z + w*x), 1.0f - 2.0f*(x*x + y*y)}
    };
    
    float M[3][3];
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            M[i][j] = R[i][j] * scale[j];
            
    float cov3d[3][3] = {0};
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            for(int k=0; k<3; ++k)
                cov3d[i][k] += M[i][j] * M[k][j];

    for(int c=0; c<G.C_total; ++c) {
        float view[4][4];
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j)
                view[i][j] = G.viewmats[c*16 + i*4 + j];
                
        float mean_c[3] = {0};
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j)
                mean_c[i] += view[i][j] * mean[j];
            mean_c[i] += view[i][3];
        }
        
        float depth = mean_c[2];
        if (depth <= G.near_plane || depth >= G.far_plane) continue;
        
        float cov_c[3][3] = {0};
        for(int i=0; i<3; ++i)
            for(int j=0; j<3; ++j)
                for(int k=0; k<3; ++k)
                    for(int l=0; l<3; ++l)
                        cov_c[i][j] += view[i][k] * cov3d[k][l] * view[j][l];
                        
        float fx = G.Ks[c*9 + 0];
        float cx = G.Ks[c*9 + 2];
        float fy = G.Ks[c*9 + 4];
        float cy = G.Ks[c*9 + 5];
        
        float tx = mean_c[0], ty = mean_c[1], tz = mean_c[2];
        float tz2 = tz * tz;
        
        float tan_fovx = 0.5f * G.image_width / fx;
        float tan_fovy = 0.5f * G.image_height / fy;
        float lim_x_pos = (G.image_width - cx) / fx + 0.3f * tan_fovx;
        float lim_x_neg = cx / fx + 0.3f * tan_fovx;
        float lim_y_pos = (G.image_height - cy) / fy + 0.3f * tan_fovy;
        float lim_y_neg = cy / fy + 0.3f * tan_fovy;
        
        float cl_tx = tz * fmaxf(-lim_x_neg, fminf(tx/tz, lim_x_pos));
        float cl_ty = tz * fmaxf(-lim_y_neg, fminf(ty/tz, lim_y_pos));
        
        float J[2][3] = {
            {fx / tz, 0.0f, -fx * cl_tx / tz2},
            {0.0f, fy / tz, -fy * cl_ty / tz2}
        };
        
        float cov2d[2][2] = {0};
        for(int i=0; i<2; ++i)
            for(int j=0; j<2; ++j)
                for(int k=0; k<3; ++k)
                    for(int l=0; l<3; ++l)
                        cov2d[i][j] += J[i][k] * cov_c[k][l] * J[j][l];
                        
        cov2d[0][0] += G.eps2d;
        cov2d[1][1] += G.eps2d;
        
        float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];
        if (det < 1e-10f) det = 1e-10f;
        
        float conic_x = cov2d[1][1] / det;
        float conic_y = -(cov2d[0][1] + cov2d[1][0]) / 2.0f / det;
        float conic_z = cov2d[0][0] / det;
        
        float radius_x = ceilf(3.33f * sqrtf(cov2d[0][0]));
        float radius_y = ceilf(3.33f * sqrtf(cov2d[1][1]));
        
        float mean2d_x = fx * tx / tz + cx;
        float mean2d_y = fy * ty / tz + cy;
        
        bool inside = (mean2d_x + radius_x > 0.0f) && (mean2d_x - radius_x < G.image_width) &&
                      (mean2d_y + radius_y > 0.0f) && (mean2d_y - radius_y < G.image_height);
                      
        if (inside && radius_x > 0.0f && radius_y > 0.0f) {
            int dest_rank = 0;
            while (dest_rank < G.num_devices - 1 && c >= G.c_world_offsets[dest_rank + 1]) {
                dest_rank++;
            }
            
            int idx = atomicAdd(&G.local_send_counts[dest_rank], 1);
            if (idx < G.MAX_CAPACITY) {
                int offset = (G.dev_idx * G.MAX_CAPACITY + idx) * G.WORDS;
                // Direct NVLink Peer-To-Peer Write inside TK's symmetric memory layout
                int32_t* base_ptr = (int32_t*)&G.p2p_buffer[dest_rank]({0,0,0,0});
                int32_t* ptr = base_ptr + offset;
                
                ptr[0] = c - G.c_world_offsets[dest_rank];     // Local camera id for destination
                ptr[1] = n + G.n_world_offsets[G.dev_idx];     // Global gaussian id
                ptr[2] = (int32_t)radius_x;
                ptr[3] = (int32_t)radius_y;
                
                nv_bfloat16* fptr = (nv_bfloat16*)(ptr + 4);
                fptr[0] = __float2bfloat16(mean2d_x);
                fptr[1] = __float2bfloat16(mean2d_y);
                fptr[2] = __float2bfloat16(depth);
                fptr[3] = __float2bfloat16(conic_x);
                fptr[4] = __float2bfloat16(conic_y);
                fptr[5] = __float2bfloat16(conic_z);
                fptr[6] = __float2bfloat16(opacity);
                for(int d=0; d<G.D; ++d) {
                    fptr[7+d] = __float2bfloat16(G.colors[n * G.D + d]);
                }
            }
        }
    }
}

namespace barrier_ns {
    struct config {
        static constexpr int CLUSTER_SIZE = 1;
        static constexpr int NUM_THREADS = 32;
    };
    struct globals {
        barrier_t<8> bar;
        int dev_idx;
    };
    __device__ inline void kernel(const globals &G) {
        barrier_all(G.bar, {0}, G.dev_idx);
    }
}

void entrypoint(
    torch::Tensor means, torch::Tensor quats, torch::Tensor scales,
    torch::Tensor opacities, torch::Tensor colors,
    torch::Tensor viewmats, torch::Tensor Ks,
    torch::Tensor c_world_offsets, torch::Tensor n_world_offsets,
    torch::Tensor local_send_counts,
    int N, int C_total, int D, int MAX_CAPACITY, int WORDS,
    int image_width, int image_height, float eps2d, float near_plane, float far_plane,
    kittens::py::TKParallelTensor &p2p_buffer,
    kittens::py::TKParallelTensor &barrier
) {
    globals G;
    G.means = means.data_ptr<float>();
    G.quats = quats.data_ptr<float>();
    G.scales = scales.data_ptr<float>();
    G.opacities = opacities.data_ptr<float>();
    G.colors = colors.data_ptr<float>();
    
    G.viewmats = viewmats.data_ptr<float>();
    G.Ks = Ks.data_ptr<float>();
    
    G.c_world_offsets = c_world_offsets.data_ptr<int>();
    G.n_world_offsets = n_world_offsets.data_ptr<int>();
    G.local_send_counts = local_send_counts.data_ptr<int>();
    
    G.N = N;
    G.C_total = C_total;
    G.D = D;
    G.MAX_CAPACITY = MAX_CAPACITY;
    G.WORDS = WORDS;
    
    G.image_width = image_width;
    G.image_height = image_height;
    G.eps2d = eps2d;
    G.near_plane = near_plane;
    G.far_plane = far_plane;
    
    G.dev_idx = p2p_buffer.local_rank_;
    G.num_devices = 8;
    
    G.p2p_buffer = kittens::py::parallel_tensor_to_pgl<pgl_i32>(p2p_buffer);
    
    barrier_ns::globals bG {
        .bar = kittens::py::parallel_tensor_to_pgl<barrier_t<8>>(barrier),
        .dev_idx = barrier.local_rank_
    };
    
    // Barrier setup
    kittens::py::launch_kernel<barrier_ns::config, barrier_ns::globals, barrier_ns::kernel>(bG);
    
    int num_blocks = (N + config::NUM_THREADS - 1) / config::NUM_THREADS;
    if (num_blocks > 0) {
        kittens::py::launch_kernel<config, globals, kernel>(G, {num_blocks, 1, 1});
    }
    
    // Barrier synchronization across stream completions
    kittens::py::launch_kernel<barrier_ns::config, barrier_ns::globals, barrier_ns::kernel>(bG);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_fused_proj", &entrypoint);
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
            "tk_gsplat_ext",
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
    means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, colors: Tensor,
    viewmats: Tensor, Ks: Tensor,
    image_width: int, image_height: int, eps2d: float = 0.3,
    near_plane: float = 0.01, far_plane: float = 1e10, camera_model: str = "pinhole"
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    
    assert dist.is_initialized()
    assert camera_model == "pinhole"
    
    ext = _ensure_ext_jit()
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    device = means.device
    original_dtype = means.dtype

    # Assert exactly 8 peer setup
    assert world_size == 8, f"ThunderKittens layout expects NUM_DEVICES=8, got {world_size}"

    # Enforce input memory layouts for CUDA kernel
    means = means.contiguous().to(torch.float32)
    quats = quats.contiguous().to(torch.float32)
    scales = scales.contiguous().to(torch.float32)
    opacities = opacities.contiguous().to(torch.float32)
    colors = colors.contiguous().to(torch.float32)

    N_local = means.shape[0]
    C_local = viewmats.shape[0]
    D = colors.shape[1]

    # Gather dataset scale
    N_world_tensor = torch.zeros(world_size, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(N_world_tensor, torch.tensor([N_local], dtype=torch.int32, device=device))
    N_world = N_world_tensor.tolist()
    C_world = [C_local] * world_size

    # Only gather camera metrics implicitly (small memory footprint, completely overlap dense projection math)
    viewmats_gather = [torch.empty_like(viewmats) for _ in range(world_size)]
    dist.all_gather(viewmats_gather, viewmats)
    viewmats_all = torch.cat(viewmats_gather, dim=0).contiguous().to(torch.float32)

    Ks_gather = [torch.empty_like(Ks) for _ in range(world_size)]
    dist.all_gather(Ks_gather, Ks)
    Ks_all = torch.cat(Ks_gather, dim=0).contiguous().to(torch.float32)

    c_offsets = [0] + torch.cumsum(torch.tensor(C_world, dtype=torch.int32), dim=0).tolist()
    n_offsets = [0] + torch.cumsum(torch.tensor(N_world, dtype=torch.int32), dim=0).tolist()

    # Pre-calculated structural layout for fused bytes: `[4 ints (IDs/Radii)] + [(7 + D)/2 ints (bfloat16 math params)]`
    WORDS_PER_PROJ = 4 + (7 + D + 1) // 2
    MAX_CAPACITY = min(sum(C_world) * max(N_world) // 4 + 10000, 2000000) 
    shape = (1, 1, 1, world_size * MAX_CAPACITY * WORDS_PER_PROJ)

    # Establish contiguous buffer blocks mapped inside ThunderKittens symmetry
    p2p_buffer = get_or_create_parallel_tensor(ext, shape, torch.int32, multicast=False)
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)
    local_send_counts = torch.zeros(world_size, dtype=torch.int32, device=device)

    # Fire Fused ThunderKittens Projection & Peer Write kernel 
    ext.tk_fused_proj(
        means, quats, scales, opacities, colors,
        viewmats_all, Ks_all,
        torch.tensor(c_offsets, dtype=torch.int32, device=device),
        torch.tensor(n_offsets, dtype=torch.int32, device=device),
        local_send_counts,
        N_local, sum(C_world), D, MAX_CAPACITY, WORDS_PER_PROJ,
        image_width, image_height, eps2d, near_plane, far_plane,
        p2p_buffer, barrier_tk
    )

    all_send_counts = torch.zeros(world_size, world_size, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(all_send_counts, local_send_counts)
    my_recv_counts = all_send_counts[:, world_rank]
    
    # Process written Peer structures inside our slots
    out_cam_ids, out_gauss_ids, out_radii, out_means2d = [], [], [], []
    out_depths, out_conics, out_opacities, out_colors = [], [], [], []
    
    p2p_data = p2p_buffer.data_.view(world_size, MAX_CAPACITY, WORDS_PER_PROJ)

    for i in range(world_size):
        count = my_recv_counts[i].item()
        if count == 0: continue
        
        chunk = p2p_data[i, :count]
        
        cam_ids = chunk[:, 0].contiguous()
        gauss_ids = chunk[:, 1].contiguous()
        radii = torch.stack([chunk[:, 2], chunk[:, 3]], dim=-1)
        
        bf16_data = chunk[:, 4:].contiguous().view(torch.bfloat16)
        means2d = bf16_data[:, 0:2].to(original_dtype)
        depths = bf16_data[:, 2].to(original_dtype)
        conics = bf16_data[:, 3:6].to(original_dtype)
        opacities = bf16_data[:, 6].to(original_dtype)
        out_col = bf16_data[:, 7:7+D].to(original_dtype)
        
        # Enforce exactly the identical sort schema generated by `.where` then `.cat` from PyTorch
        sort_keys = cam_ids.long() * n_offsets[-1] + gauss_ids.long()
        sort_idx = torch.argsort(sort_keys)
        
        out_cam_ids.append(cam_ids[sort_idx])
        out_gauss_ids.append(gauss_ids[sort_idx])
        out_radii.append(radii[sort_idx])
        out_means2d.append(means2d[sort_idx])
        out_depths.append(depths[sort_idx])
        out_conics.append(conics[sort_idx])
        out_opacities.append(opacities[sort_idx])
        out_colors.append(out_col[sort_idx])

    if len(out_cam_ids) == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty((0, 2), dtype=torch.int32, device=device),
            torch.empty((0, 2), dtype=original_dtype, device=device),
            torch.empty(0, dtype=original_dtype, device=device),
            torch.empty((0, 3), dtype=original_dtype, device=device),
            torch.empty(0, dtype=original_dtype, device=device),
            torch.empty((0, D), dtype=original_dtype, device=device),
        )

    return (
        torch.cat(out_cam_ids),
        torch.cat(out_gauss_ids),
        torch.cat(out_radii),
        torch.cat(out_means2d),
        torch.cat(out_depths),
        torch.cat(out_conics),
        torch.cat(out_opacities),
        torch.cat(out_colors)
    )