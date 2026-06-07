import math
from typing import Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

template <typename T>
__device__ __forceinline__ float ld_scalar(const T* p, int64_t i) {
    return static_cast<float>(p[i]);
}
template <>
__device__ __forceinline__ float ld_scalar<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}

template <typename T>
__device__ __forceinline__ void st_scalar(T* p, int64_t i, float v) {
    p[i] = static_cast<T>(v);
}
template <>
__device__ __forceinline__ void st_scalar<__nv_bfloat16>(__nv_bfloat16* p, int64_t i, float v) {
    p[i] = __float2bfloat16(v);
}

struct ProjVals {
    int valid;
    int rx;
    int ry;
    float m2x;
    float m2y;
    float depth;
    float c0;
    float c1;
    float c2;
};

template <typename T>
__device__ __forceinline__ ProjVals project_one(
    const T* __restrict__ means,
    const T* __restrict__ quats,
    const T* __restrict__ scales,
    const T* __restrict__ view,
    const T* __restrict__ K,
    int64_t gi,
    int width,
    int height,
    float eps2d,
    float near_plane,
    float far_plane
) {
    ProjVals o;
    o.valid = 0;
    o.rx = 0;
    o.ry = 0;
    o.m2x = o.m2y = o.depth = o.c0 = o.c1 = o.c2 = 0.0f;

    const float mx_w = ld_scalar<T>(means, gi * 3 + 0);
    const float my_w = ld_scalar<T>(means, gi * 3 + 1);
    const float mz_w = ld_scalar<T>(means, gi * 3 + 2);

    float qw = ld_scalar<T>(quats, gi * 4 + 0);
    float qx = ld_scalar<T>(quats, gi * 4 + 1);
    float qy = ld_scalar<T>(quats, gi * 4 + 2);
    float qz = ld_scalar<T>(quats, gi * 4 + 3);

    const float qn = rsqrtf(fmaxf(qw * qw + qx * qx + qy * qy + qz * qz, 1.0e-20f));
    qw *= qn;
    qx *= qn;
    qy *= qn;
    qz *= qn;

    const float sx = ld_scalar<T>(scales, gi * 3 + 0);
    const float sy = ld_scalar<T>(scales, gi * 3 + 1);
    const float sz = ld_scalar<T>(scales, gi * 3 + 2);
    const float sx2 = sx * sx;
    const float sy2 = sy * sy;
    const float sz2 = sz * sz;

    // Quaternion to rotation matrix, row-major.
    const float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    const float r01 = 2.0f * (qx * qy - qw * qz);
    const float r02 = 2.0f * (qx * qz + qw * qy);
    const float r10 = 2.0f * (qx * qy + qw * qz);
    const float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    const float r12 = 2.0f * (qy * qz - qw * qx);
    const float r20 = 2.0f * (qx * qz - qw * qy);
    const float r21 = 2.0f * (qy * qz + qw * qx);
    const float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);

    // cov = R * diag(scale^2) * R^T.
    const float cov00 = r00 * r00 * sx2 + r01 * r01 * sy2 + r02 * r02 * sz2;
    const float cov01 = r00 * r10 * sx2 + r01 * r11 * sy2 + r02 * r12 * sz2;
    const float cov02 = r00 * r20 * sx2 + r01 * r21 * sy2 + r02 * r22 * sz2;
    const float cov11 = r10 * r10 * sx2 + r11 * r11 * sy2 + r12 * r12 * sz2;
    const float cov12 = r10 * r20 * sx2 + r11 * r21 * sy2 + r12 * r22 * sz2;
    const float cov22 = r20 * r20 * sx2 + r21 * r21 * sy2 + r22 * r22 * sz2;

    const float v00 = ld_scalar<T>(view, 0);
    const float v01 = ld_scalar<T>(view, 1);
    const float v02 = ld_scalar<T>(view, 2);
    const float v03 = ld_scalar<T>(view, 3);
    const float v10 = ld_scalar<T>(view, 4);
    const float v11 = ld_scalar<T>(view, 5);
    const float v12 = ld_scalar<T>(view, 6);
    const float v13 = ld_scalar<T>(view, 7);
    const float v20 = ld_scalar<T>(view, 8);
    const float v21 = ld_scalar<T>(view, 9);
    const float v22c = ld_scalar<T>(view, 10);
    const float v23 = ld_scalar<T>(view, 11);

    const float tx0 = v00 * mx_w + v01 * my_w + v02 * mz_w + v03;
    const float ty0 = v10 * mx_w + v11 * my_w + v12 * mz_w + v13;
    const float tz = v20 * mx_w + v21 * my_w + v22c * mz_w + v23;

    o.depth = tz;

    const float k00 = ld_scalar<T>(K, 0);
    const float k01 = ld_scalar<T>(K, 1);
    const float k02 = ld_scalar<T>(K, 2);
    const float k10 = ld_scalar<T>(K, 3);
    const float k11 = ld_scalar<T>(K, 4);
    const float k12 = ld_scalar<T>(K, 5);

    const float fx = k00;
    const float fy = k11;
    const float cx = k02;
    const float cy = k12;

    const float inv_tz = 1.0f / tz;
    const float tz2 = tz * tz;

    o.m2x = (k00 * tx0 + k01 * ty0 + k02 * tz) * inv_tz;
    o.m2y = (k10 * tx0 + k11 * ty0 + k12 * tz) * inv_tz;

    const float tan_fovx = 0.5f * float(width) / fx;
    const float tan_fovy = 0.5f * float(height) / fy;

    const float lim_x_pos = (float(width) - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = (float(height) - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float nx = tx0 * inv_tz;
    float ny = ty0 * inv_tz;
    nx = fminf(fmaxf(nx, -lim_x_neg), lim_x_pos);
    ny = fminf(fmaxf(ny, -lim_y_neg), lim_y_pos);

    const float tx = tz * nx;
    const float ty = tz * ny;

    // cov_c = V[:3,:3] * cov * V[:3,:3]^T.
    const float a00 = v00 * cov00 + v01 * cov01 + v02 * cov02;
    const float a01 = v00 * cov01 + v01 * cov11 + v02 * cov12;
    const float a02 = v00 * cov02 + v01 * cov12 + v02 * cov22;

    const float a10 = v10 * cov00 + v11 * cov01 + v12 * cov02;
    const float a11 = v10 * cov01 + v11 * cov11 + v12 * cov12;
    const float a12 = v10 * cov02 + v11 * cov12 + v12 * cov22;

    const float a20 = v20 * cov00 + v21 * cov01 + v22c * cov02;
    const float a21 = v20 * cov01 + v21 * cov11 + v22c * cov12;
    const float a22 = v20 * cov02 + v21 * cov12 + v22c * cov22;

    const float cc00 = a00 * v00 + a01 * v01 + a02 * v02;
    const float cc01 = a00 * v10 + a01 * v11 + a02 * v12;
    const float cc02 = a00 * v20 + a01 * v21 + a02 * v22c;
    const float cc11 = a10 * v10 + a11 * v11 + a12 * v12;
    const float cc12 = a10 * v20 + a11 * v21 + a12 * v22c;
    const float cc22 = a20 * v20 + a21 * v21 + a22 * v22c;

    const float j00 = fx * inv_tz;
    const float j02 = -fx * tx / tz2;
    const float j11 = fy * inv_tz;
    const float j12 = -fy * ty / tz2;

    // cov2d = J * cov_c * J^T.
    float c2d00 = j00 * j00 * cc00 + 2.0f * j00 * j02 * cc02 + j02 * j02 * cc22;
    float c2d01 = j00 * j11 * cc01 + j00 * j12 * cc02 + j02 * j11 * cc12 + j02 * j12 * cc22;
    float c2d11 = j11 * j11 * cc11 + 2.0f * j11 * j12 * cc12 + j12 * j12 * cc22;

    c2d00 += eps2d;
    c2d11 += eps2d;

    float det = c2d00 * c2d11 - c2d01 * c2d01;
    det = fmaxf(det, 1.0e-10f);

    o.c0 = c2d11 / det;
    o.c1 = -c2d01 / det;
    o.c2 = c2d00 / det;

    const float radx_f = ceilf(3.33f * sqrtf(fmaxf(c2d00, 0.0f)));
    const float rady_f = ceilf(3.33f * sqrtf(fmaxf(c2d11, 0.0f)));
    o.rx = (int)radx_f;
    o.ry = (int)rady_f;

    const bool valid_depth = (tz > near_plane) && (tz < far_plane);
    const bool inside =
        (o.m2x + radx_f > 0.0f) &&
        (o.m2x - radx_f < float(width)) &&
        (o.m2y + rady_f > 0.0f) &&
        (o.m2y - rady_f < float(height));

    o.valid = (valid_depth && inside && o.rx > 0 && o.ry > 0) ? 1 : 0;
    if (!o.valid) {
        o.rx = 0;
        o.ry = 0;
    }
    return o;
}

template <typename T>
__global__ void count_kernel(
    const T* __restrict__ means,
    const T* __restrict__ quats,
    const T* __restrict__ scales,
    const int64_t* __restrict__ view_ptrs,
    const int64_t* __restrict__ K_ptrs,
    int* __restrict__ camera_counts,
    int N,
    int C_local,
    int width,
    int height,
    float eps2d,
    float near_plane,
    float far_plane
) {
    const int gc = blockIdx.x;
    const int owner = gc / C_local;
    const int lc = gc - owner * C_local;

    const T* view = reinterpret_cast<const T*>((uintptr_t)view_ptrs[owner]) + int64_t(lc) * 16;
    const T* K = reinterpret_cast<const T*>((uintptr_t)K_ptrs[owner]) + int64_t(lc) * 9;

    __shared__ int sh[256];
    int local = 0;
    for (int gi = threadIdx.x; gi < N; gi += blockDim.x) {
        ProjVals p = project_one<T>(
            means, quats, scales, view, K, gi,
            width, height, eps2d, near_plane, far_plane
        );
        local += p.valid;
    }
    sh[threadIdx.x] = local;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            sh[threadIdx.x] += sh[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        camera_counts[gc] = sh[0];
    }
}

__global__ void scan_camera_counts_kernel(
    const int* __restrict__ camera_counts,
    int* __restrict__ camera_offsets,
    int* __restrict__ send_counts,
    int C_local,
    int world_size,
    int cap_per_dest
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int d = 0; d < world_size; ++d) {
        int run = 0;
        for (int lc = 0; lc < C_local; ++lc) {
            const int gc = d * C_local + lc;
            camera_offsets[gc] = d * cap_per_dest + run;
            run += camera_counts[gc];
        }
        send_counts[d] = run;
    }
}

template <typename T>
__global__ void fill_kernel(
    const T* __restrict__ means,
    const T* __restrict__ quats,
    const T* __restrict__ scales,
    const T* __restrict__ opacities,
    const T* __restrict__ colors,
    const int64_t* __restrict__ view_ptrs,
    const int64_t* __restrict__ K_ptrs,
    const int64_t* __restrict__ n_ptrs,
    const int* __restrict__ camera_offsets,
    int* __restrict__ out_camera_ids,
    int* __restrict__ out_gaussian_ids,
    int* __restrict__ out_radii,
    T* __restrict__ out_means2d,
    T* __restrict__ out_depths,
    T* __restrict__ out_conics,
    T* __restrict__ out_opacities,
    T* __restrict__ out_colors,
    int N,
    int C_local,
    int D,
    int rank,
    int width,
    int height,
    float eps2d,
    float near_plane,
    float far_plane
) {
    const int gc = blockIdx.x;
    const int owner = gc / C_local;
    const int lc = gc - owner * C_local;

    const T* view = reinterpret_cast<const T*>((uintptr_t)view_ptrs[owner]) + int64_t(lc) * 16;
    const T* K = reinterpret_cast<const T*>((uintptr_t)K_ptrs[owner]) + int64_t(lc) * 9;

    int global_gaussian_base = 0;
    for (int r = 0; r < rank; ++r) {
        const int* np = reinterpret_cast<const int*>((uintptr_t)n_ptrs[r]);
        global_gaussian_base += np[0];
    }

    __shared__ int scan[256];
    __shared__ int running;

    if (threadIdx.x == 0) running = 0;
    __syncthreads();

    const int camera_base = camera_offsets[gc];

    for (int start = 0; start < N; start += blockDim.x) {
        const int gi = start + threadIdx.x;
        ProjVals p;
        p.valid = 0;
        if (gi < N) {
            p = project_one<T>(
                means, quats, scales, view, K, gi,
                width, height, eps2d, near_plane, far_plane
            );
        }

        const int flag = (gi < N) ? p.valid : 0;
        scan[threadIdx.x] = flag;
        __syncthreads();

        for (int off = 1; off < blockDim.x; off <<= 1) {
            int v = 0;
            if (threadIdx.x >= off) v = scan[threadIdx.x - off];
            __syncthreads();
            scan[threadIdx.x] += v;
            __syncthreads();
        }

        const int active = min(blockDim.x, N - start);
        const int chunk_total = (active > 0) ? scan[active - 1] : 0;

        if (flag) {
            const int local_off = scan[threadIdx.x] - 1;
            const int out_idx = camera_base + running + local_off;

            out_camera_ids[out_idx] = lc;
            out_gaussian_ids[out_idx] = global_gaussian_base + gi;

            out_radii[int64_t(out_idx) * 2 + 0] = p.rx;
            out_radii[int64_t(out_idx) * 2 + 1] = p.ry;

            st_scalar<T>(out_means2d, int64_t(out_idx) * 2 + 0, p.m2x);
            st_scalar<T>(out_means2d, int64_t(out_idx) * 2 + 1, p.m2y);

            st_scalar<T>(out_depths, out_idx, p.depth);

            st_scalar<T>(out_conics, int64_t(out_idx) * 3 + 0, p.c0);
            st_scalar<T>(out_conics, int64_t(out_idx) * 3 + 1, p.c1);
            st_scalar<T>(out_conics, int64_t(out_idx) * 3 + 2, p.c2);

            out_opacities[out_idx] = opacities[gi];

            for (int d = 0; d < D; ++d) {
                out_colors[int64_t(out_idx) * D + d] = colors[int64_t(gi) * D + d];
            }
        }

        __syncthreads();
        if (threadIdx.x == 0) running += chunk_total;
        __syncthreads();
    }
}

__global__ void gather_recv_counts_kernel(
    const int64_t* __restrict__ count_ptrs,
    int* __restrict__ recv_counts,
    int rank,
    int world_size
) {
    int src = threadIdx.x;
    if (src < world_size) {
        const int* counts = reinterpret_cast<const int*>((uintptr_t)count_ptrs[src]);
        recv_counts[src] = counts[rank];
    }
}

template <typename T>
__global__ void copy_records_kernel(
    const int64_t* __restrict__ count_ptrs,
    const int64_t* __restrict__ cam_ptrs,
    const int64_t* __restrict__ gid_ptrs,
    const int64_t* __restrict__ radii_ptrs,
    const int64_t* __restrict__ means2d_ptrs,
    const int64_t* __restrict__ depths_ptrs,
    const int64_t* __restrict__ conics_ptrs,
    const int64_t* __restrict__ opacity_ptrs,
    const int64_t* __restrict__ color_ptrs,
    const int* __restrict__ recv_offsets,
    int* __restrict__ out_camera_ids,
    int* __restrict__ out_gaussian_ids,
    int* __restrict__ out_radii,
    T* __restrict__ out_means2d,
    T* __restrict__ out_depths,
    T* __restrict__ out_conics,
    T* __restrict__ out_opacities,
    T* __restrict__ out_colors,
    int total,
    int D,
    int world_size,
    int rank,
    int cap_per_dest
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int src = 0;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r < world_size) {
            if (idx >= recv_offsets[r] && idx < recv_offsets[r + 1]) {
                src = r;
            }
        }
    }

    const int local_idx = idx - recv_offsets[src];
    const int remote_idx = rank * cap_per_dest + local_idx;

    const int* src_cam = reinterpret_cast<const int*>((uintptr_t)cam_ptrs[src]);
    const int* src_gid = reinterpret_cast<const int*>((uintptr_t)gid_ptrs[src]);
    const int* src_radii = reinterpret_cast<const int*>((uintptr_t)radii_ptrs[src]);
    const T* src_m2d = reinterpret_cast<const T*>((uintptr_t)means2d_ptrs[src]);
    const T* src_depth = reinterpret_cast<const T*>((uintptr_t)depths_ptrs[src]);
    const T* src_conic = reinterpret_cast<const T*>((uintptr_t)conics_ptrs[src]);
    const T* src_opac = reinterpret_cast<const T*>((uintptr_t)opacity_ptrs[src]);
    const T* src_color = reinterpret_cast<const T*>((uintptr_t)color_ptrs[src]);

    out_camera_ids[idx] = src_cam[remote_idx];
    out_gaussian_ids[idx] = src_gid[remote_idx];

    out_radii[int64_t(idx) * 2 + 0] = src_radii[int64_t(remote_idx) * 2 + 0];
    out_radii[int64_t(idx) * 2 + 1] = src_radii[int64_t(remote_idx) * 2 + 1];

    out_means2d[int64_t(idx) * 2 + 0] = src_m2d[int64_t(remote_idx) * 2 + 0];
    out_means2d[int64_t(idx) * 2 + 1] = src_m2d[int64_t(remote_idx) * 2 + 1];

    out_depths[idx] = src_depth[remote_idx];

    out_conics[int64_t(idx) * 3 + 0] = src_conic[int64_t(remote_idx) * 3 + 0];
    out_conics[int64_t(idx) * 3 + 1] = src_conic[int64_t(remote_idx) * 3 + 1];
    out_conics[int64_t(idx) * 3 + 2] = src_conic[int64_t(remote_idx) * 3 + 2];

    out_opacities[idx] = src_opac[remote_idx];

    for (int d = 0; d < D; ++d) {
        out_colors[int64_t(idx) * D + d] = src_color[int64_t(remote_idx) * D + d];
    }
}

void launch_project_pack(
    torch::Tensor means,
    torch::Tensor quats,
    torch::Tensor scales,
    torch::Tensor opacities,
    torch::Tensor colors,
    torch::Tensor view_ptrs,
    torch::Tensor K_ptrs,
    torch::Tensor n_ptrs,
    torch::Tensor camera_counts,
    torch::Tensor camera_offsets,
    torch::Tensor send_counts,
    torch::Tensor out_camera_ids,
    torch::Tensor out_gaussian_ids,
    torch::Tensor out_radii,
    torch::Tensor out_means2d,
    torch::Tensor out_depths,
    torch::Tensor out_conics,
    torch::Tensor out_opacities,
    torch::Tensor out_colors,
    int N,
    int C_local,
    int D,
    int world_size,
    int rank,
    int width,
    int height,
    double eps2d,
    double near_plane,
    double far_plane,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int C_total = C_local * world_size;
    const int cap_per_dest = N * C_local;
    const int threads = 256;

    cudaMemsetAsync(camera_counts.data_ptr<int>(), 0, sizeof(int) * C_total, stream);
    cudaMemsetAsync(camera_offsets.data_ptr<int>(), 0, sizeof(int) * C_total, stream);
    cudaMemsetAsync(send_counts.data_ptr<int>(), 0, sizeof(int) * world_size, stream);

    if (C_total == 0) return;

    const int64_t* d_view_ptrs = view_ptrs.data_ptr<int64_t>();
    const int64_t* d_K_ptrs = K_ptrs.data_ptr<int64_t>();
    const int64_t* d_n_ptrs = n_ptrs.data_ptr<int64_t>();

    if (dtype_enum == 0) {
        const __nv_bfloat16* m = reinterpret_cast<const __nv_bfloat16*>(means.data_ptr<at::BFloat16>());
        const __nv_bfloat16* q = reinterpret_cast<const __nv_bfloat16*>(quats.data_ptr<at::BFloat16>());
        const __nv_bfloat16* s = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
        const __nv_bfloat16* op = reinterpret_cast<const __nv_bfloat16*>(opacities.data_ptr<at::BFloat16>());
        const __nv_bfloat16* col = reinterpret_cast<const __nv_bfloat16*>(colors.data_ptr<at::BFloat16>());

        count_kernel<__nv_bfloat16><<<C_total, threads, 0, stream>>>(
            m, q, s, d_view_ptrs, d_K_ptrs,
            camera_counts.data_ptr<int>(),
            N, C_local, width, height,
            (float)eps2d, (float)near_plane, (float)far_plane
        );

        scan_camera_counts_kernel<<<1, 1, 0, stream>>>(
            camera_counts.data_ptr<int>(),
            camera_offsets.data_ptr<int>(),
            send_counts.data_ptr<int>(),
            C_local,
            world_size,
            cap_per_dest
        );

        fill_kernel<__nv_bfloat16><<<C_total, threads, 0, stream>>>(
            m, q, s, op, col, d_view_ptrs, d_K_ptrs, d_n_ptrs,
            camera_offsets.data_ptr<int>(),
            out_camera_ids.data_ptr<int>(),
            out_gaussian_ids.data_ptr<int>(),
            out_radii.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(out_means2d.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_depths.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_conics.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_opacities.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_colors.data_ptr<at::BFloat16>()),
            N, C_local, D, rank, width, height,
            (float)eps2d, (float)near_plane, (float)far_plane
        );
    } else {
        const float* m = means.data_ptr<float>();
        const float* q = quats.data_ptr<float>();
        const float* s = scales.data_ptr<float>();
        const float* op = opacities.data_ptr<float>();
        const float* col = colors.data_ptr<float>();

        count_kernel<float><<<C_total, threads, 0, stream>>>(
            m, q, s, d_view_ptrs, d_K_ptrs,
            camera_counts.data_ptr<int>(),
            N, C_local, width, height,
            (float)eps2d, (float)near_plane, (float)far_plane
        );

        scan_camera_counts_kernel<<<1, 1, 0, stream>>>(
            camera_counts.data_ptr<int>(),
            camera_offsets.data_ptr<int>(),
            send_counts.data_ptr<int>(),
            C_local,
            world_size,
            cap_per_dest
        );

        fill_kernel<float><<<C_total, threads, 0, stream>>>(
            m, q, s, op, col, d_view_ptrs, d_K_ptrs, d_n_ptrs,
            camera_offsets.data_ptr<int>(),
            out_camera_ids.data_ptr<int>(),
            out_gaussian_ids.data_ptr<int>(),
            out_radii.data_ptr<int>(),
            out_means2d.data_ptr<float>(),
            out_depths.data_ptr<float>(),
            out_conics.data_ptr<float>(),
            out_opacities.data_ptr<float>(),
            out_colors.data_ptr<float>(),
            N, C_local, D, rank, width, height,
            (float)eps2d, (float)near_plane, (float)far_plane
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_recv_counts(
    torch::Tensor count_ptrs,
    torch::Tensor recv_counts,
    int rank,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_recv_counts_kernel<<<1, 32, 0, stream>>>(
        count_ptrs.data_ptr<int64_t>(),
        recv_counts.data_ptr<int>(),
        rank,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_copy_records(
    torch::Tensor count_ptrs,
    torch::Tensor cam_ptrs,
    torch::Tensor gid_ptrs,
    torch::Tensor radii_ptrs,
    torch::Tensor means2d_ptrs,
    torch::Tensor depths_ptrs,
    torch::Tensor conics_ptrs,
    torch::Tensor opacity_ptrs,
    torch::Tensor color_ptrs,
    torch::Tensor recv_offsets,
    torch::Tensor out_camera_ids,
    torch::Tensor out_gaussian_ids,
    torch::Tensor out_radii,
    torch::Tensor out_means2d,
    torch::Tensor out_depths,
    torch::Tensor out_conics,
    torch::Tensor out_opacities,
    torch::Tensor out_colors,
    int total,
    int D,
    int world_size,
    int rank,
    int cap_per_dest,
    int dtype_enum
) {
    if (total <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (dtype_enum == 0) {
        copy_records_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            count_ptrs.data_ptr<int64_t>(),
            cam_ptrs.data_ptr<int64_t>(),
            gid_ptrs.data_ptr<int64_t>(),
            radii_ptrs.data_ptr<int64_t>(),
            means2d_ptrs.data_ptr<int64_t>(),
            depths_ptrs.data_ptr<int64_t>(),
            conics_ptrs.data_ptr<int64_t>(),
            opacity_ptrs.data_ptr<int64_t>(),
            color_ptrs.data_ptr<int64_t>(),
            recv_offsets.data_ptr<int>(),
            out_camera_ids.data_ptr<int>(),
            out_gaussian_ids.data_ptr<int>(),
            out_radii.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(out_means2d.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_depths.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_conics.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_opacities.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out_colors.data_ptr<at::BFloat16>()),
            total, D, world_size, rank, cap_per_dest
        );
    } else {
        copy_records_kernel<float><<<blocks, threads, 0, stream>>>(
            count_ptrs.data_ptr<int64_t>(),
            cam_ptrs.data_ptr<int64_t>(),
            gid_ptrs.data_ptr<int64_t>(),
            radii_ptrs.data_ptr<int64_t>(),
            means2d_ptrs.data_ptr<int64_t>(),
            depths_ptrs.data_ptr<int64_t>(),
            conics_ptrs.data_ptr<int64_t>(),
            opacity_ptrs.data_ptr<int64_t>(),
            color_ptrs.data_ptr<int64_t>(),
            recv_offsets.data_ptr<int>(),
            out_camera_ids.data_ptr<int>(),
            out_gaussian_ids.data_ptr<int>(),
            out_radii.data_ptr<int>(),
            out_means2d.data_ptr<float>(),
            out_depths.data_ptr<float>(),
            out_conics.data_ptr<float>(),
            out_opacities.data_ptr<float>(),
            out_colors.data_ptr<float>(),
            total, D, world_size, rank, cap_per_dest
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_project_pack", &launch_project_pack, "fused gsplat projection + packed partition");
    m.def("launch_gather_recv_counts", &launch_gather_recv_counts, "UVA gather recv counts");
    m.def("launch_copy_records", &launch_copy_records, "UVA all-to-all record copy");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gsplat_projection_symm_uva_bf16_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _ptr_tensor(hdl, device):
    return torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)


def _get_resources(
    N: int,
    C: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
):
    key = (int(N), int(C), int(D), dtype, device.index, int(world_size))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    C_total = C * world_size
    cap_per_dest = N * C
    capacity = cap_per_dest * world_size

    n_buf = symm_mem.empty((1,), device=device, dtype=torch.int32)
    n_hdl = symm_mem.rendezvous(n_buf, dist.group.WORLD)

    view_buf = symm_mem.empty((C, 4, 4), device=device, dtype=dtype)
    view_hdl = symm_mem.rendezvous(view_buf, dist.group.WORLD)

    K_buf = symm_mem.empty((C, 3, 3), device=device, dtype=dtype)
    K_hdl = symm_mem.rendezvous(K_buf, dist.group.WORLD)

    send_counts = symm_mem.empty((world_size,), device=device, dtype=torch.int32)
    counts_hdl = symm_mem.rendezvous(send_counts, dist.group.WORLD)

    send_camera_ids = symm_mem.empty((capacity,), device=device, dtype=torch.int32)
    cam_hdl = symm_mem.rendezvous(send_camera_ids, dist.group.WORLD)

    send_gaussian_ids = symm_mem.empty((capacity,), device=device, dtype=torch.int32)
    gid_hdl = symm_mem.rendezvous(send_gaussian_ids, dist.group.WORLD)

    send_radii = symm_mem.empty((capacity, 2), device=device, dtype=torch.int32)
    radii_hdl = symm_mem.rendezvous(send_radii, dist.group.WORLD)

    send_means2d = symm_mem.empty((capacity, 2), device=device, dtype=dtype)
    means2d_hdl = symm_mem.rendezvous(send_means2d, dist.group.WORLD)

    send_depths = symm_mem.empty((capacity,), device=device, dtype=dtype)
    depths_hdl = symm_mem.rendezvous(send_depths, dist.group.WORLD)

    send_conics = symm_mem.empty((capacity, 3), device=device, dtype=dtype)
    conics_hdl = symm_mem.rendezvous(send_conics, dist.group.WORLD)

    send_opacities = symm_mem.empty((capacity,), device=device, dtype=dtype)
    opac_hdl = symm_mem.rendezvous(send_opacities, dist.group.WORLD)

    send_colors = symm_mem.empty((capacity, D), device=device, dtype=dtype)
    color_hdl = symm_mem.rendezvous(send_colors, dist.group.WORLD)

    camera_counts = torch.empty((C_total,), device=device, dtype=torch.int32)
    camera_offsets = torch.empty((C_total,), device=device, dtype=torch.int32)
    recv_counts = torch.empty((world_size,), device=device, dtype=torch.int32)

    res = {
        "N": N,
        "C": C,
        "D": D,
        "C_total": C_total,
        "cap_per_dest": cap_per_dest,
        "capacity": capacity,
        "n_buf": n_buf,
        "n_hdl": n_hdl,
        "view_buf": view_buf,
        "view_hdl": view_hdl,
        "K_buf": K_buf,
        "K_hdl": K_hdl,
        "send_counts": send_counts,
        "counts_hdl": counts_hdl,
        "send_camera_ids": send_camera_ids,
        "cam_hdl": cam_hdl,
        "send_gaussian_ids": send_gaussian_ids,
        "gid_hdl": gid_hdl,
        "send_radii": send_radii,
        "radii_hdl": radii_hdl,
        "send_means2d": send_means2d,
        "means2d_hdl": means2d_hdl,
        "send_depths": send_depths,
        "depths_hdl": depths_hdl,
        "send_conics": send_conics,
        "conics_hdl": conics_hdl,
        "send_opacities": send_opacities,
        "opac_hdl": opac_hdl,
        "send_colors": send_colors,
        "color_hdl": color_hdl,
        "camera_counts": camera_counts,
        "camera_offsets": camera_offsets,
        "recv_counts": recv_counts,
        "n_ptrs": _ptr_tensor(n_hdl, device),
        "view_ptrs": _ptr_tensor(view_hdl, device),
        "K_ptrs": _ptr_tensor(K_hdl, device),
        "count_ptrs": _ptr_tensor(counts_hdl, device),
        "cam_ptrs": _ptr_tensor(cam_hdl, device),
        "gid_ptrs": _ptr_tensor(gid_hdl, device),
        "radii_ptrs": _ptr_tensor(radii_hdl, device),
        "means2d_ptrs": _ptr_tensor(means2d_hdl, device),
        "depths_ptrs": _ptr_tensor(depths_hdl, device),
        "conics_ptrs": _ptr_tensor(conics_hdl, device),
        "opac_ptrs": _ptr_tensor(opac_hdl, device),
        "color_ptrs": _ptr_tensor(color_hdl, device),
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    image_width: int,
    image_height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    camera_model: str = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert camera_model == "pinhole", "only pinhole camera_model is supported"
    assert means.is_cuda, "inputs must be CUDA tensors"
    assert means.dtype in (torch.bfloat16, torch.float32), "optimized path supports bf16/fp32"
    assert quats.dtype == means.dtype
    assert scales.dtype == means.dtype
    assert opacities.dtype == means.dtype
    assert colors.dtype == means.dtype
    assert viewmats.dtype == means.dtype
    assert Ks.dtype == means.dtype

    ext = _get_ext()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = means.device

    means = means.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    opacities = opacities.contiguous()
    colors = colors.contiguous()
    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()

    N = int(means.shape[0])
    C = int(viewmats.shape[0])
    D = int(colors.shape[1])

    res = _get_resources(N, C, D, means.dtype, device, world_size)

    # Symmetric camera/N publication.  No NCCL collectives: peers consume these
    # buffers directly through UVA pointers in projection kernels.
    res["n_buf"].fill_(N)
    res["view_buf"].copy_(viewmats)
    res["K_buf"].copy_(Ks)

    res["n_hdl"].barrier(channel=0)
    res["view_hdl"].barrier(channel=1)
    res["K_hdl"].barrier(channel=2)

    dtype_enum = 0 if means.dtype == torch.bfloat16 else 1

    ext.launch_project_pack(
        means,
        quats,
        scales,
        opacities,
        colors,
        res["view_ptrs"],
        res["K_ptrs"],
        res["n_ptrs"],
        res["camera_counts"],
        res["camera_offsets"],
        res["send_counts"],
        res["send_camera_ids"],
        res["send_gaussian_ids"],
        res["send_radii"],
        res["send_means2d"],
        res["send_depths"],
        res["send_conics"],
        res["send_opacities"],
        res["send_colors"],
        N,
        C,
        D,
        world_size,
        rank,
        int(image_width),
        int(image_height),
        float(eps2d),
        float(near_plane),
        float(far_plane),
        dtype_enum,
    )

    # Publish packed send buffers and counts.  Destination ranks read their
    # segment rank*C*N : rank*C*N+count from every peer by UVA.
    res["counts_hdl"].barrier(channel=3)

    ext.launch_gather_recv_counts(
        res["count_ptrs"],
        res["recv_counts"],
        rank,
        world_size,
    )

    recv_counts_host = res["recv_counts"].cpu().tolist()
    recv_offsets_host = [0]
    for v in recv_counts_host:
        recv_offsets_host.append(recv_offsets_host[-1] + int(v))
    total = int(recv_offsets_host[-1])

    camera_ids = torch.empty((total,), device=device, dtype=torch.int32)
    gaussian_ids = torch.empty((total,), device=device, dtype=torch.int32)
    radii = torch.empty((total, 2), device=device, dtype=torch.int32)
    means2d = torch.empty((total, 2), device=device, dtype=means.dtype)
    depths = torch.empty((total,), device=device, dtype=means.dtype)
    conics = torch.empty((total, 3), device=device, dtype=means.dtype)
    out_opacities = torch.empty((total,), device=device, dtype=means.dtype)
    out_colors = torch.empty((total, D), device=device, dtype=means.dtype)

    if total > 0:
        recv_offsets = torch.tensor(recv_offsets_host, device=device, dtype=torch.int32)

        ext.launch_copy_records(
            res["count_ptrs"],
            res["cam_ptrs"],
            res["gid_ptrs"],
            res["radii_ptrs"],
            res["means2d_ptrs"],
            res["depths_ptrs"],
            res["conics_ptrs"],
            res["opac_ptrs"],
            res["color_ptrs"],
            recv_offsets,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            out_opacities,
            out_colors,
            total,
            D,
            world_size,
            rank,
            res["cap_per_dest"],
            dtype_enum,
        )

    return (
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        out_opacities,
        out_colors,
    )