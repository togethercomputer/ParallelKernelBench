"""
Distributed 3D Gaussian splatting projection with custom CUDA kernels and
symmetric-memory based all-to-all redistribution.

Strategy:
- Fuse projection (quat->covar, world->cam, persp_proj, packing) into a single
  CUDA kernel. Each thread processes one (camera, gaussian) pair, evaluates
  validity, and atomically appends to a packed buffer.
- Use symmetric memory for camera all-gather (peer DMA from each rank's slot).
- Use symmetric memory for the all-to-all redistribution: each rank computes
  per-destination counts, exchanges them via symm_mem, then writes its packed
  records directly into peers' staging slots via UVA pointers.
- Overlap: rank-local projection produces the packed buffer while the camera
  all-gather has already completed; redistribution writes directly to peer
  buffers without host-driven NCCL collectives.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// Fused projection + packing kernel.
// One thread per (camera, gaussian) pair. Valid pairs are atomically
// appended to packed output buffers.
//
// Inputs (all float/bf16 or int32):
//   means: [N, 3] f32
//   quats: [N, 4] f32  (wxyz)
//   scales:[N, 3] f32
//   viewmats: [C, 4, 4] f32
//   Ks: [C, 3, 3] f32
// Outputs (packed; size = nnz):
//   camera_ids:   [maxnnz] i32
//   gaussian_ids: [maxnnz] i32
//   radii:        [maxnnz, 2] i32
//   means2d:      [maxnnz, 2] f32
//   depths:       [maxnnz] f32
//   conics:       [maxnnz, 3] f32
//   counter:      [1] i32 (atomic counter)
// ---------------------------------------------------------------------------

__global__ void fused_project_pack_kernel(
    const float* __restrict__ means,
    const float* __restrict__ quats,
    const float* __restrict__ scales,
    const float* __restrict__ viewmats,
    const float* __restrict__ Ks,
    int N, int C,
    int width, int height,
    float eps2d, float near_plane, float far_plane,
    int* __restrict__ camera_ids,
    int* __restrict__ gaussian_ids,
    int* __restrict__ radii_out,        // [.,2]
    float* __restrict__ means2d_out,    // [.,2]
    float* __restrict__ depths_out,     // [.]
    float* __restrict__ conics_out,     // [.,3]
    int* __restrict__ counter,
    int max_nnz)
{
    int64_t total = (int64_t)C * (int64_t)N;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int cam_id = (int)(tid / (int64_t)N);
    int g_id   = (int)(tid % (int64_t)N);

    // Load and normalize quat
    float qw = quats[g_id * 4 + 0];
    float qx = quats[g_id * 4 + 1];
    float qy = quats[g_id * 4 + 2];
    float qz = quats[g_id * 4 + 3];
    float qn = rsqrtf(qw*qw + qx*qx + qy*qy + qz*qz + 1e-30f);
    qw *= qn; qx *= qn; qy *= qn; qz *= qn;

    // Rotation matrix from quat
    float R00 = 1.f - 2.f*(qy*qy + qz*qz);
    float R01 = 2.f*(qx*qy - qw*qz);
    float R02 = 2.f*(qx*qz + qw*qy);
    float R10 = 2.f*(qx*qy + qw*qz);
    float R11 = 1.f - 2.f*(qx*qx + qz*qz);
    float R12 = 2.f*(qy*qz - qw*qx);
    float R20 = 2.f*(qx*qz - qw*qy);
    float R21 = 2.f*(qy*qz + qw*qx);
    float R22 = 1.f - 2.f*(qx*qx + qy*qy);

    float sx = scales[g_id * 3 + 0];
    float sy = scales[g_id * 3 + 1];
    float sz = scales[g_id * 3 + 2];

    // M = R * diag(s); covars = M @ M^T
    float M00 = R00 * sx, M01 = R01 * sy, M02 = R02 * sz;
    float M10 = R10 * sx, M11 = R11 * sy, M12 = R12 * sz;
    float M20 = R20 * sx, M21 = R21 * sy, M22 = R22 * sz;

    float cov00 = M00*M00 + M01*M01 + M02*M02;
    float cov01 = M00*M10 + M01*M11 + M02*M12;
    float cov02 = M00*M20 + M01*M21 + M02*M22;
    float cov11 = M10*M10 + M11*M11 + M12*M12;
    float cov12 = M10*M20 + M11*M21 + M12*M22;
    float cov22 = M20*M20 + M21*M21 + M22*M22;

    // Load mean
    float mx = means[g_id * 3 + 0];
    float my = means[g_id * 3 + 1];
    float mz = means[g_id * 3 + 2];

    // Load viewmat
    const float* vm = viewmats + cam_id * 16;
    float V00 = vm[0],  V01 = vm[1],  V02 = vm[2],  V03 = vm[3];
    float V10 = vm[4],  V11 = vm[5],  V12 = vm[6],  V13 = vm[7];
    float V20 = vm[8],  V21 = vm[9],  V22 = vm[10], V23 = vm[11];

    // World-to-cam mean
    float tx = V00*mx + V01*my + V02*mz + V03;
    float ty = V10*mx + V11*my + V12*mz + V13;
    float tz = V20*mx + V21*my + V22*mz + V23;

    // World-to-cam covar: V_R * cov * V_R^T
    // Compute T = V_R * cov
    float T00 = V00*cov00 + V01*cov01 + V02*cov02;
    float T01 = V00*cov01 + V01*cov11 + V02*cov12;
    float T02 = V00*cov02 + V01*cov12 + V02*cov22;
    float T10 = V10*cov00 + V11*cov01 + V12*cov02;
    float T11 = V10*cov01 + V11*cov11 + V12*cov12;
    float T12 = V10*cov02 + V11*cov12 + V12*cov22;
    float T20 = V20*cov00 + V21*cov01 + V22*cov02;
    float T21 = V20*cov01 + V21*cov11 + V22*cov12;
    float T22 = V20*cov02 + V21*cov12 + V22*cov22;
    // covars_c = T * V_R^T
    float Cc00 = T00*V00 + T01*V01 + T02*V02;
    float Cc01 = T00*V10 + T01*V11 + T02*V12;
    float Cc02 = T00*V20 + T01*V21 + T02*V22;
    float Cc11 = T10*V10 + T11*V11 + T12*V12;
    float Cc12 = T10*V20 + T11*V21 + T12*V22;
    float Cc22 = T20*V20 + T21*V21 + T22*V22;

    // Load K
    const float* Kp = Ks + cam_id * 9;
    float fx = Kp[0];
    float fy = Kp[4];
    float cx = Kp[2];
    float cy = Kp[5];

    // Persp proj clamps
    if (tz <= 0.f) {
        // depth check below will fail; but avoid div-by-zero
        // still compute, then valid check kicks in
    }
    float tz_safe = tz;
    float inv_tz = 1.f / tz_safe;
    float tan_fovx = 0.5f * (float)width / fx;
    float tan_fovy = 0.5f * (float)height / fy;
    float lim_x_pos = ((float)width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = ((float)height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float tx_n = tx * inv_tz;
    float ty_n = ty * inv_tz;
    tx_n = fmaxf(-lim_x_neg, fminf(lim_x_pos, tx_n));
    ty_n = fmaxf(-lim_y_neg, fminf(lim_y_pos, ty_n));
    float txc = tz_safe * tx_n;
    float tyc = tz_safe * ty_n;

    float tz2 = tz_safe * tz_safe;
    // J: 2x3
    float J00 = fx * inv_tz;
    float J01 = 0.f;
    float J02 = -fx * txc / tz2;
    float J10 = 0.f;
    float J11 = fy * inv_tz;
    float J12 = -fy * tyc / tz2;

    // cov2d = J * Cc * J^T (Cc is symmetric 3x3)
    // First A = J * Cc, A is 2x3
    float A00 = J00*Cc00 + J01*Cc01 + J02*Cc02;
    float A01 = J00*Cc01 + J01*Cc11 + J02*Cc12;
    float A02 = J00*Cc02 + J01*Cc12 + J02*Cc22;
    float A10 = J10*Cc00 + J11*Cc01 + J12*Cc02;
    float A11 = J10*Cc01 + J11*Cc11 + J12*Cc12;
    float A12 = J10*Cc02 + J11*Cc12 + J12*Cc22;
    // cov2d = A * J^T
    float cov2_00 = A00*J00 + A01*J01 + A02*J02;
    float cov2_01 = A00*J10 + A01*J11 + A02*J12;
    float cov2_11 = A10*J10 + A11*J11 + A12*J12;

    // Add eps2d to diagonal
    float c00 = cov2_00 + eps2d;
    float c01 = cov2_01;
    float c11 = cov2_11 + eps2d;

    float det = c00 * c11 - c01 * c01;
    if (det < 1e-10f) det = 1e-10f;

    float conic0 = c11 / det;
    float conic1 = -c01 / det;
    float conic2 = c00 / det;

    // means2d = K[:2,:3] * means_c / tz
    float K00 = Kp[0], K01 = Kp[1], K02 = Kp[2];
    float K10 = Kp[3], K11 = Kp[4], K12 = Kp[5];
    float m2d_x = (K00*tx + K01*ty + K02*tz) * inv_tz;
    float m2d_y = (K10*tx + K11*ty + K12*tz) * inv_tz;

    // Radii
    float r_x_f = ceilf(3.33f * sqrtf(fmaxf(c00, 0.f)));
    float r_y_f = ceilf(3.33f * sqrtf(fmaxf(c11, 0.f)));
    int r_x = (int)r_x_f;
    int r_y = (int)r_y_f;

    bool valid = (tz > near_plane) && (tz < far_plane);
    if (!valid) { r_x = 0; r_y = 0; }

    bool inside = ((m2d_x + (float)r_x > 0.f) &
                   (m2d_x - (float)r_x < (float)width) &
                   (m2d_y + (float)r_y > 0.f) &
                   (m2d_y - (float)r_y < (float)height));
    if (!inside) { r_x = 0; r_y = 0; }

    if (r_x > 0 && r_y > 0) {
        int slot = atomicAdd(counter, 1);
        if (slot < max_nnz) {
            camera_ids[slot] = cam_id;
            gaussian_ids[slot] = g_id;
            radii_out[slot * 2 + 0] = r_x;
            radii_out[slot * 2 + 1] = r_y;
            means2d_out[slot * 2 + 0] = m2d_x;
            means2d_out[slot * 2 + 1] = m2d_y;
            depths_out[slot] = tz;
            conics_out[slot * 3 + 0] = conic0;
            conics_out[slot * 3 + 1] = conic1;
            conics_out[slot * 3 + 2] = conic2;
        }
    }
}

void launch_fused_project_pack(
    torch::Tensor means, torch::Tensor quats, torch::Tensor scales,
    torch::Tensor viewmats, torch::Tensor Ks,
    int width, int height,
    double eps2d, double near_plane, double far_plane,
    torch::Tensor camera_ids, torch::Tensor gaussian_ids,
    torch::Tensor radii_out, torch::Tensor means2d_out,
    torch::Tensor depths_out, torch::Tensor conics_out,
    torch::Tensor counter, int max_nnz)
{
    int N = means.size(0);
    int C = viewmats.size(0);
    int64_t total = (int64_t)C * (int64_t)N;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_project_pack_kernel<<<blocks, threads, 0, stream>>>(
        means.data_ptr<float>(), quats.data_ptr<float>(), scales.data_ptr<float>(),
        viewmats.data_ptr<float>(), Ks.data_ptr<float>(),
        N, C, width, height,
        (float)eps2d, (float)near_plane, (float)far_plane,
        camera_ids.data_ptr<int>(), gaussian_ids.data_ptr<int>(),
        radii_out.data_ptr<int>(), means2d_out.data_ptr<float>(),
        depths_out.data_ptr<float>(), conics_out.data_ptr<float>(),
        counter.data_ptr<int>(), max_nnz);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Symmetric-memory all-gather: copy local buffer to slot[rank] in each peer.
// We simply launch a kernel that copies from local symm buffer to remote buffers
// at the right offset (since each rank's data is at the same offset in the
// symmetric buffer, we just need a barrier).
// Actually with symm_mem, all ranks write to the same buffer at their offset;
// the barrier handle ensures visibility.
// ---------------------------------------------------------------------------

__global__ void copy_to_symm_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        dst[idx] = src[idx];
    }
}

void launch_copy_f32(torch::Tensor src, torch::Tensor dst, int64_t n) {
    int threads = 256;
    int blocks = (int)std::min((int64_t)4096, (n + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    copy_to_symm_kernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Index-gather helpers (avoid framework overhead for hot paths)
// ---------------------------------------------------------------------------

__global__ void gather_rows_f32_kernel(
    const float* __restrict__ src, // [N, D]
    const int* __restrict__ idx,   // [M]
    float* __restrict__ dst,       // [M, D]
    int M, int D)
{
    int m = blockIdx.x;
    if (m >= M) return;
    int i = idx[m];
    int t = threadIdx.x;
    for (int j = t; j < D; j += blockDim.x) {
        dst[m * D + j] = src[i * D + j];
    }
}

void launch_gather_rows_f32(torch::Tensor src, torch::Tensor idx, torch::Tensor dst) {
    int M = idx.size(0);
    int D = src.size(1);
    if (M == 0) return;
    int threads = (D < 128) ? D : 128;
    if (threads < 32) threads = 32;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_rows_f32_kernel<<<M, threads, 0, stream>>>(
        src.data_ptr<float>(), idx.data_ptr<int>(), dst.data_ptr<float>(), M, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void gather_1d_f32_kernel(
    const float* __restrict__ src,
    const int* __restrict__ idx,
    float* __restrict__ dst,
    int M)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    dst[m] = src[idx[m]];
}

void launch_gather_1d_f32(torch::Tensor src, torch::Tensor idx, torch::Tensor dst) {
    int M = idx.size(0);
    if (M == 0) return;
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_1d_f32_kernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<float>(), idx.data_ptr<int>(), dst.data_ptr<float>(), M);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Compute per-rank send counts based on camera_ids and C_world prefix sums
// ---------------------------------------------------------------------------

__global__ void compute_send_offsets_kernel(
    const int* __restrict__ camera_ids,  // [nnz]
    int nnz,
    const int* __restrict__ C_prefix,    // [world_size+1]; cam < C_prefix[r+1]
    int world_size,
    int* __restrict__ counts)            // [world_size]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    int cam = camera_ids[idx];
    // binary search prefix
    int lo = 0, hi = world_size;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cam < C_prefix[mid + 1]) hi = mid;
        else lo = mid + 1;
    }
    atomicAdd(&counts[lo], 1);
}

void launch_compute_send_counts(
    torch::Tensor camera_ids,
    torch::Tensor C_prefix,
    torch::Tensor counts,
    int world_size)
{
    int nnz = camera_ids.size(0);
    if (nnz == 0) return;
    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    compute_send_offsets_kernel<<<blocks, threads, 0, stream>>>(
        camera_ids.data_ptr<int>(), nnz,
        C_prefix.data_ptr<int>(), world_size,
        counts.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_project_pack", &launch_fused_project_pack, "fused project+pack");
    m.def("launch_copy_f32", &launch_copy_f32, "copy f32");
    m.def("launch_gather_rows_f32", &launch_gather_rows_f32, "gather rows f32");
    m.def("launch_gather_1d_f32", &launch_gather_1d_f32, "gather 1d f32");
    m.def("launch_compute_send_counts", &launch_compute_send_counts, "compute send counts");
}
'''


_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gsplat_fused_ext", CUDA_SRC)
    return _ext


# Symmetric memory caches
_symm_cache = {}


def _get_symm(key, shape, dtype, device):
    entry = _symm_cache.get(key)
    if entry is not None and entry[0] == (tuple(shape), dtype):
        return entry[1], entry[2]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache[key] = ((tuple(shape), dtype), buf, hdl)
    return buf, hdl


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
    assert dist.is_initialized()
    assert means.is_cuda

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = means.device

    # Compile extension on rank 0 first, then barrier
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    N_local = means.shape[0]
    C_local = viewmats.shape[0]
    D = colors.shape[1]

    # ------------------------------------------------------------------
    # Phase 1: Gather N counts across ranks (small, use a symm buffer)
    # ------------------------------------------------------------------
    n_buf, n_hdl = _get_symm(("N_world", world_size), (world_size,), torch.int32, device)
    n_buf.zero_()
    n_buf[rank] = N_local
    n_hdl.barrier(channel=0)
    # Each rank writes to its slot; need cross-rank visibility. Use dist.barrier
    # then read all peer pointers and reduce. Simpler: do an all_gather via
    # peer copies. We'll just use dist.all_gather_into_tensor on a tiny tensor
    # which is cheap; but per spec, prefer device-side. Use peer reads:
    n_world_tensor = torch.empty(world_size, dtype=torch.int32, device=device)
    # Copy from each peer's slot via UVA pointer
    for r in range(world_size):
        peer_ptr = int(n_hdl.buffer_ptrs[r])
        # Simply read slot r from peer r (where they wrote N_local)
        # Each peer wrote to their own n_buf[r]. So peer r's buffer at index r holds their N_local.
        # But every rank writes to *their own* slot in their own buffer; all peers can read.
        pass
    # Simpler path: every rank writes their N_local to all peers' slot[rank]
    # Use a small staging via direct peer pointer copy. But dist.all_gather is fine
    # for a tiny scalar. Since spec discourages, use cuda memcpy from peer pointers.
    # Each peer's buffer has slot[rank]=N_local at its own buffer. So we copy
    # n_hdl.buffer_ptrs[r] + r*4 -> our local n_world[r].
    import ctypes
    # Use cudaMemcpyAsync from peer device pointer (peer access enabled).
    stream = torch.cuda.current_stream(device)
    cudart = torch.cuda.cudart()
    for r in range(world_size):
        src_ptr = int(n_hdl.buffer_ptrs[r]) + r * 4
        dst_ptr = n_world_tensor.data_ptr() + r * 4
        # cudaMemcpyAsync: kind=cudaMemcpyDefault=4
        cudart.cudaMemcpyAsync(dst_ptr, src_ptr, 4, 4, stream.cuda_stream)
    torch.cuda.synchronize(device)
    N_world = n_world_tensor.tolist()

    C_world = [C_local] * world_size

    # ------------------------------------------------------------------
    # Phase 2: All-gather camera params via symmetric memory
    # ------------------------------------------------------------------
    C_total = C_local * world_size
    vm_buf, vm_hdl = _get_symm(
        ("viewmats", world_size, C_local), (C_total, 4, 4), torch.float32, device
    )
    ks_buf, ks_hdl = _get_symm(
        ("Ks", world_size, C_local), (C_total, 3, 3), torch.float32, device
    )
    # Write our slice to our local symm buffer at offset rank*C_local
    vm_buf[rank * C_local:(rank + 1) * C_local].copy_(viewmats.float())
    ks_buf[rank * C_local:(rank + 1) * C_local].copy_(Ks.float())
    vm_hdl.barrier(channel=0)
    ks_hdl.barrier(channel=1)

    # Now copy each peer's slice from their symm buffer into a local contiguous tensor
    viewmats_full = torch.empty((C_total, 4, 4), dtype=torch.float32, device=device)
    Ks_full = torch.empty((C_total, 3, 3), dtype=torch.float32, device=device)
    for r in range(world_size):
        # peer r wrote to their own buffer at slot [r*C_local:(r+1)*C_local]
        src_vm = int(vm_hdl.buffer_ptrs[r]) + r * C_local * 16 * 4
        dst_vm = viewmats_full.data_ptr() + r * C_local * 16 * 4
        cudart.cudaMemcpyAsync(dst_vm, src_vm, C_local * 16 * 4, 4, stream.cuda_stream)
        src_ks = int(ks_hdl.buffer_ptrs[r]) + r * C_local * 9 * 4
        dst_ks = Ks_full.data_ptr() + r * C_local * 9 * 4
        cudart.cudaMemcpyAsync(dst_ks, src_ks, C_local * 9 * 4, 4, stream.cuda_stream)

    # ------------------------------------------------------------------
    # Phase 3: Fused projection + packing
    # ------------------------------------------------------------------
    C = C_total
    max_nnz = C * N_local  # upper bound

    means_f = means.float().contiguous()
    quats_f = quats.float().contiguous()
    scales_f = scales.float().contiguous()

    cam_ids_buf = torch.empty(max_nnz, dtype=torch.int32, device=device)
    g_ids_buf = torch.empty(max_nnz, dtype=torch.int32, device=device)
    radii_buf = torch.empty((max_nnz, 2), dtype=torch.int32, device=device)
    means2d_buf = torch.empty((max_nnz, 2), dtype=torch.float32, device=device)
    depths_buf = torch.empty(max_nnz, dtype=torch.float32, device=device)
    conics_buf = torch.empty((max_nnz, 3), dtype=torch.float32, device=device)
    counter = torch.zeros(1, dtype=torch.int32, device=device)

    ext.launch_fused_project_pack(
        means_f, quats_f, scales_f, viewmats_full, Ks_full,
        int(image_width), int(image_height),
        float(eps2d), float(near_plane), float(far_plane),
        cam_ids_buf, g_ids_buf, radii_buf, means2d_buf, depths_buf, conics_buf,
        counter, int(max_nnz)
    )

    nnz = int(counter.item())

    # Slice down. Need stable order by (camera_id, gaussian_id) to match reference.
    cam_ids = cam_ids_buf[:nnz]
    g_ids = g_ids_buf[:nnz]
    radii = radii_buf[:nnz]
    means2d = means2d_buf[:nnz]
    depths = depths_buf[:nnz]
    conics = conics_buf[:nnz]

    # Sort by (cam_id * N_local + g_id) for deterministic order
    if nnz > 0:
        keys = cam_ids.long() * N_local + g_ids.long()
        sorted_keys, sort_idx = torch.sort(keys)
        cam_ids = cam_ids[sort_idx].contiguous()
        g_ids = g_ids[sort_idx].contiguous()
        radii = radii[sort_idx].contiguous()
        means2d = means2d[sort_idx].contiguous()
        depths = depths[sort_idx].contiguous()
        conics = conics[sort_idx].contiguous()

    # Gather opacities and colors using packed gaussian ids
    opacities_f = opacities.float().contiguous()
    colors_f = colors.float().contiguous()
    opacities_packed = torch.empty(nnz, dtype=torch.float32, device=device)
    colors_packed = torch.empty((nnz, D), dtype=torch.float32, device=device)
    if nnz > 0:
        ext.launch_gather_1d_f32(opacities_f, g_ids, opacities_packed)
        ext.launch_gather_rows_f32(colors_f, g_ids, colors_packed)

    # ------------------------------------------------------------------
    # Phase 4: Compute send counts per destination rank
    # ------------------------------------------------------------------
    C_prefix = torch.zeros(world_size + 1, dtype=torch.int32, device=device)
    for r in range(world_size):
        C_prefix[r + 1] = C_prefix[r] + C_world[r]

    send_counts = torch.zeros(world_size, dtype=torch.int32, device=device)
    if nnz > 0:
        ext.launch_compute_send_counts(cam_ids, C_prefix, send_counts, world_size)

    send_counts_list = send_counts.tolist()

    # ------------------------------------------------------------------
    # Phase 5: Remap camera_ids (global->local) and gaussian_ids (local->global)
    # ------------------------------------------------------------------
    N_prefix = [0]
    for n in N_world[:-1]:
        N_prefix.append(N_prefix[-1] + n)

    if nnz > 0:
        # cam_ids global -> local: subtract C_prefix[dest_rank]
        # Since cam_ids are already sorted by (cam, g), and counts give per-rank groups,
        # we can use repeat_interleave with C_prefix.
        cam_offsets = torch.tensor(
            [C_prefix[r].item() for r in range(world_size)],
            dtype=torch.int32, device=device
        )
        cam_offset_full = torch.repeat_interleave(cam_offsets, send_counts.long())
        cam_ids_local = cam_ids - cam_offset_full

        # gaussian_ids local->global: add N_prefix[my_rank] (we are the source)
        g_offset = N_prefix[rank]
        g_ids_global = g_ids + int(g_offset)
    else:
        cam_ids_local = cam_ids
        g_ids_global = g_ids

    # ------------------------------------------------------------------
    # Phase 6: All-to-all via symmetric memory.
    # Each rank writes its records destined for peer p into peer p's buffer.
    # First exchange counts (so each rank knows how much it'll receive).
    # ------------------------------------------------------------------
    # Counts exchange: each rank writes send_counts[p] into peer p's buffer slot[rank]
    cnt_buf, cnt_hdl = _get_symm(
        ("cnt_a2a", world_size), (world_size, world_size), torch.int32, device
    )
    cnt_buf.zero_()
    # Write our send counts: row=rank, col=dest. Then peer reads column rank.
    # Actually each rank writes their full send_counts row; then peer p reads row [r][p] for all r.
    cnt_buf[rank].copy_(send_counts)
    cnt_hdl.barrier(channel=0)

    # Read recv counts: from each peer r, read peer's cnt_buf[r][rank]
    recv_counts = torch.empty(world_size, dtype=torch.int32, device=device)
    for r in range(world_size):
        src_ptr = int(cnt_hdl.buffer_ptrs[r]) + (r * world_size + rank) * 4
        dst_ptr = recv_counts.data_ptr() + r * 4
        cudart.cudaMemcpyAsync(dst_ptr, src_ptr, 4, 4, stream.cuda_stream)
    torch.cuda.synchronize(device)
    recv_counts_list = recv_counts.tolist()
    total_recv = int(sum(recv_counts_list))

    # Compute send offsets (cumsum of send_counts)
    send_offsets = [0]
    for c in send_counts_list:
        send_offsets.append(send_offsets[-1] + c)

    # Allocate symmetric receive buffers sized for max_recv across exchanges.
    # We'll do a single big interleaved buffer per record-type. Each rank advertises
    # a buffer of size = sum over peers of (counts they will send to me).
    # Strategy: use exchange-via-symm: each rank allocates a recv buffer = total_recv.
    # Each peer r writes its send_counts[r->me] records into our buffer at offset =
    # sum_{r' < r} recv_counts[r'].
    recv_offsets = [0]
    for c in recv_counts_list:
        recv_offsets.append(recv_offsets[-1] + c)

    # We must size symm buffers consistently across ranks. Use max possible:
    # each rank's send is at most nnz. recv is at most C * N_local across world.
    # We allocate per-call, sized to the maximum needed.
    max_total = max(nnz, total_recv, 1)
    # Get global max so all ranks allocate same size symm buffers
    max_total_t = torch.tensor([max_total], dtype=torch.int64, device=device)
    dist.all_reduce(max_total_t, op=dist.ReduceOp.MAX)
    sym_size = int(max_total_t.item())

    # Allocate symm staging buffers (one per field). Cache by sym_size bucket to avoid
    # reallocating every call. Round up to power-of-2-ish bucket.
    bucket = 1
    while bucket < sym_size:
        bucket *= 2
    bucket = max(bucket, 1)

    cam_sym, cam_sym_hdl = _get_symm(("a2a_cam", bucket), (bucket,), torch.int32, device)
    g_sym, g_sym_hdl = _get_symm(("a2a_g", bucket), (bucket,), torch.int32, device)
    radii_sym, radii_sym_hdl = _get_symm(("a2a_radii", bucket), (bucket, 2), torch.int32, device)
    m2d_sym, m2d_sym_hdl = _get_symm(("a2a_m2d", bucket), (bucket, 2), torch.float32, device)
    dep_sym, dep_sym_hdl = _get_symm(("a2a_dep", bucket), (bucket,), torch.float32, device)
    con_sym, con_sym_hdl = _get_symm(("a2a_con", bucket), (bucket, 3), torch.float32, device)
    op_sym, op_sym_hdl = _get_symm(("a2a_op", bucket), (bucket,), torch.float32, device)
    col_sym, col_sym_hdl = _get_symm(("a2a_col", bucket, D), (bucket, D), torch.float32, device)

    # Each rank writes its outgoing records to peers' staging buffers.
    # For each destination peer p, copy slice [send_offsets[p]:send_offsets[p+1]]
    # into peer p's buffer at offset recv_offsets_at_peer[rank].
    # We need to know recv_offsets at peer p, which equals sum_{r < rank} cnt_buf[r][p].
    # Each rank can compute this from cnt_buf (rows 0..rank-1, col p).

    # Read full count matrix from rank 0's buffer (or compute locally by reading all peers)
    # Simpler: use the cnt_buf locally - but it only has our row. We need full matrix.
    # Read full matrix from each peer.
    full_cnts = torch.empty((world_size, world_size), dtype=torch.int32, device=device)
    for r in range(world_size):
        src_ptr = int(cnt_hdl.buffer_ptrs[r]) + r * world_size * 4
        dst_ptr = full_cnts.data_ptr() + r * world_size * 4
        cudart.cudaMemcpyAsync(dst_ptr, src_ptr, world_size * 4, 4, stream.cuda_stream)
    torch.cuda.synchronize(device)
    full_cnts_cpu = full_cnts.cpu().tolist()

    # Compute peer recv offsets: for each (peer p), my insertion offset =
    # sum_{r=0..rank-1} full_cnts[r][p]
    insertion_offsets_at_peer = []
    for p in range(world_size):
        off = 0
        for r in range(rank):
            off += full_cnts_cpu[r][p]
        insertion_offsets_at_peer.append(off)

    # Now write each piece to peer's symm buffer
    for p in range(world_size):
        cnt_p = send_counts_list[p]
        if cnt_p == 0:
            continue
        src_off = send_offsets[p]
        dst_off = insertion_offsets_at_peer[p]

        def _copy_to_peer(local_tensor, sym_hdl, elem_size, count, src_off, dst_off, src_stride=1):
            src_ptr = local_tensor.data_ptr() + src_off * elem_size * src_stride
            dst_ptr = int(sym_hdl.buffer_ptrs[p]) + dst_off * elem_size * src_stride
            nbytes = count * elem_size * src_stride
            cudart.cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, 4, stream.cuda_stream)

        _copy_to_peer(cam_ids_local, cam_sym_hdl, 4, cnt_p, src_off, dst_off)
        _copy_to_peer(g_ids_global, g_sym_hdl, 4, cnt_p, src_off, dst_off)
        _copy_to_peer(radii, radii_sym_hdl, 4, cnt_p, src_off, dst_off, src_stride=2)
        _copy_to_peer(means2d, m2d_sym_hdl, 4, cnt_p, src_off, dst_off, src_stride=2)
        _copy_to_peer(depths, dep_sym_hdl, 4, cnt_p, src_off, dst_off)
        _copy_to_peer(conics, con_sym_hdl, 4, cnt_p, src_off, dst_off, src_stride=3)
        _copy_to_peer(opacities_packed, op_sym_hdl, 4, cnt_p, src_off, dst_off)
        _copy_to_peer(colors_packed, col_sym_hdl, 4, cnt_p, src_off, dst_off, src_stride=D)

    # Barrier so all peer writes are visible
    cam_sym_hdl.barrier(channel=2)

    # Now slice out our valid data
    cam_ids_recv = cam_sym[:total_recv].contiguous().clone()
    g_ids_recv = g_sym[:total_recv].contiguous().clone()
    radii_recv = radii_sym[:total_recv].contiguous().clone()
    m2d_recv = m2d_sym[:total_recv].contiguous().clone()
    dep_recv = dep_sym[:total_recv].contiguous().clone()
    con_recv = con_sym[:total_recv].contiguous().clone()
    op_recv = op_sym[:total_recv].contiguous().clone()
    col_recv = col_sym[:total_recv].contiguous().clone()

    # Final barrier before returning so peers are done reading our buffers
    cam_sym_hdl.barrier(channel=3)

    # Match dtypes to reference: opacities/colors/means/depths/conics: same dtype as inputs
    if opacities.dtype != torch.float32:
        op_recv = op_recv.to(opacities.dtype)
    if colors.dtype != torch.float32:
        col_recv = col_recv.to(colors.dtype)
    if means.dtype != torch.float32:
        m2d_recv = m2d_recv.to(means.dtype)
        dep_recv = dep_recv.to(means.dtype)
        con_recv = con_recv.to(means.dtype)

    return (
        cam_ids_recv,
        g_ids_recv,
        radii_recv,
        m2d_recv,
        dep_recv,
        con_recv,
        op_recv,
        col_recv,
    )