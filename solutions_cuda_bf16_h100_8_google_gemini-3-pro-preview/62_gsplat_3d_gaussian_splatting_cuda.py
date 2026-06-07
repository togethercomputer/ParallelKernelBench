import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple
from torch import Tensor

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ void quat_scale_to_covar(const float* quat, const float* scale, float covar[3][3]) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float norm = sqrtf(w*w + x*x + y*y + z*z);
    w /= norm; x /= norm; y /= norm; z /= norm;

    float R[3][3];
    R[0][0] = 1.0f - 2.0f*(y*y + z*z); R[0][1] = 2.0f*(x*y - w*z); R[0][2] = 2.0f*(x*z + w*y);
    R[1][0] = 2.0f*(x*y + w*z); R[1][1] = 1.0f - 2.0f*(x*x + z*z); R[1][2] = 2.0f*(y*z - w*x);
    R[2][0] = 2.0f*(x*z - w*y); R[2][1] = 2.0f*(y*z + w*x); R[2][2] = 1.0f - 2.0f*(x*x + y*y);

    float M[3][3];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            M[r][c] = R[r][c] * scale[c];
        }
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum += M[r][k] * M[c][k];
            }
            covar[r][c] = sum;
        }
    }
}

__global__ void compute_valid_kernel(
    const float* __restrict__ means, const float* __restrict__ quats, const float* __restrict__ scales,
    const float* __restrict__ viewmats, const float* __restrict__ Ks,
    int width, int height,
    int N_local, int C_total,
    float eps2d, float near_plane, float far_plane,
    int* __restrict__ valid
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N_local) return;

    float mean[3] = {means[n*3], means[n*3+1], means[n*3+2]};
    float quat[4] = {quats[n*4], quats[n*4+1], quats[n*4+2], quats[n*4+3]};
    float scale[3] = {scales[n*3], scales[n*3+1], scales[n*3+2]};

    float covar[3][3];
    quat_scale_to_covar(quat, scale, covar);

    for (int c = 0; c < C_total; ++c) {
        float R[3][3], t[3], K[3][3];
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
                R[i][j] = viewmats[c * 16 + i * 4 + j];
                K[i][j] = Ks[c * 9 + i * 3 + j];
            }
            t[i] = viewmats[c * 16 + i * 4 + 3];
        }
        
        float depth = R[2][0]*mean[0] + R[2][1]*mean[1] + R[2][2]*mean[2] + t[2];
        bool is_valid = true;
        
        if (depth <= near_plane || depth >= far_plane) {
            is_valid = false;
        } else {
            float mean_c[3];
            for(int i=0; i<3; ++i) {
                mean_c[i] = R[i][0]*mean[0] + R[i][1]*mean[1] + R[i][2]*mean[2] + t[i];
            }

            float cov_c[3][3];
            float tmp[3][3];
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    tmp[i][j] = R[i][0]*covar[0][j] + R[i][1]*covar[1][j] + R[i][2]*covar[2][j];
                }
            }
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    cov_c[i][j] = tmp[i][0]*R[j][0] + tmp[i][1]*R[j][1] + tmp[i][2]*R[j][2];
                }
            }

            float tx = mean_c[0], ty = mean_c[1], tz = depth;
            float tz2 = tz * tz;

            float fx = K[0][0]; float fy = K[1][1];
            float cx = K[0][2]; float cy = K[1][2];

            float tan_fovx = 0.5f * width / fx;
            float tan_fovy = 0.5f * height / fy;
            float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
            float lim_x_neg = cx / fx + 0.3f * tan_fovx;
            float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
            float lim_y_neg = cy / fy + 0.3f * tan_fovy;

            float clamp_x = tx / tz;
            clamp_x = fmaxf(-lim_x_neg, fminf(clamp_x, lim_x_pos));
            tx = tz * clamp_x;

            float clamp_y = ty / tz;
            clamp_y = fmaxf(-lim_y_neg, fminf(clamp_y, lim_y_pos));
            ty = tz * clamp_y;

            float J[2][3] = {
                {fx / tz, 0.0f, -fx * tx / tz2},
                {0.0f, fy / tz, -fy * ty / tz2}
            };

            float cov2d[2][2];
            float tmp2[2][3];
            for(int i=0; i<2; ++i) {
                for(int j=0; j<3; ++j) {
                    tmp2[i][j] = J[i][0]*cov_c[0][j] + J[i][1]*cov_c[1][j] + J[i][2]*cov_c[2][j];
                }
            }
            for(int i=0; i<2; ++i) {
                for(int j=0; j<2; ++j) {
                    cov2d[i][j] = tmp2[i][0]*J[j][0] + tmp2[i][1]*J[j][1] + tmp2[i][2]*J[j][2];
                }
            }

            float mean2d[2];
            mean2d[0] = (K[0][0]*mean_c[0] + K[0][1]*mean_c[1] + K[0][2]*mean_c[2]) / tz;
            mean2d[1] = (K[1][0]*mean_c[0] + K[1][1]*mean_c[1] + K[1][2]*mean_c[2]) / tz;

            cov2d[0][0] += eps2d;
            cov2d[1][1] += eps2d;

            int radii[2];
            radii[0] = (int)ceilf(3.33f * sqrtf(cov2d[0][0]));
            radii[1] = (int)ceilf(3.33f * sqrtf(cov2d[1][1]));

            if (mean2d[0] + radii[0] <= 0 || mean2d[0] - radii[0] >= width ||
                mean2d[1] + radii[1] <= 0 || mean2d[1] - radii[1] >= height ||
                radii[0] <= 0 || radii[1] <= 0) {
                is_valid = false;
            }
        }
        valid[c * N_local + n] = is_valid ? 1 : 0;
    }
}

template <typename ColorT>
__global__ void project_and_push_kernel(
    const float* __restrict__ means, const float* __restrict__ quats, const float* __restrict__ scales,
    const float* __restrict__ opacities, const ColorT* __restrict__ colors,
    const float* __restrict__ viewmats, const float* __restrict__ Ks,
    const int* __restrict__ cam_dst_ranks, const int* __restrict__ cam_local_ids,
    int width, int height,
    const int* __restrict__ scan, const int* __restrict__ c_start,
    int N_local, int C_total, int D_colors,
    float eps2d, float near_plane, float far_plane,
    int my_rank, int global_gaussian_offset,
    const int* __restrict__ peer_recv_offsets,
    const int64_t* __restrict__ peer_camera_ids_ptrs,
    const int64_t* __restrict__ peer_gaussian_ids_ptrs,
    const int64_t* __restrict__ peer_radii_ptrs,
    const int64_t* __restrict__ peer_means2d_ptrs,
    const int64_t* __restrict__ peer_depths_ptrs,
    const int64_t* __restrict__ peer_conics_ptrs,
    const int64_t* __restrict__ peer_opac_ptrs,
    const int64_t* __restrict__ peer_colors_ptrs,
    int world_size
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N_local) return;

    float mean[3] = {means[n*3], means[n*3+1], means[n*3+2]};
    float quat[4] = {quats[n*4], quats[n*4+1], quats[n*4+2], quats[n*4+3]};
    float scale[3] = {scales[n*3], scales[n*3+1], scales[n*3+2]};
    float opac = opacities[n];

    float covar[3][3];
    quat_scale_to_covar(quat, scale, covar);

    for (int c = 0; c < C_total; ++c) {
        int idx_1d = c * N_local + n;
        int prev = (idx_1d == 0) ? 0 : scan[idx_1d - 1];
        if (scan[idx_1d] > prev) {
            int local_idx = scan[idx_1d] - 1;
            
            float R[3][3], t[3], K[3][3];
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    R[i][j] = viewmats[c * 16 + i * 4 + j];
                    K[i][j] = Ks[c * 9 + i * 3 + j];
                }
                t[i] = viewmats[c * 16 + i * 4 + 3];
            }
            float fx = K[0][0]; float fy = K[1][1];
            float cx = K[0][2]; float cy = K[1][2];
            
            float depth = R[2][0]*mean[0] + R[2][1]*mean[1] + R[2][2]*mean[2] + t[2];
            
            float mean_c[3];
            for(int i=0; i<3; ++i) {
                mean_c[i] = R[i][0]*mean[0] + R[i][1]*mean[1] + R[i][2]*mean[2] + t[i];
            }

            float cov_c[3][3];
            float tmp[3][3];
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    tmp[i][j] = R[i][0]*covar[0][j] + R[i][1]*covar[1][j] + R[i][2]*covar[2][j];
                }
            }
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    cov_c[i][j] = tmp[i][0]*R[j][0] + tmp[i][1]*R[j][1] + tmp[i][2]*R[j][2];
                }
            }

            float tx = mean_c[0], ty = mean_c[1], tz = depth;
            float tz2 = tz * tz;

            float tan_fovx = 0.5f * width / fx;
            float tan_fovy = 0.5f * height / fy;
            float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
            float lim_x_neg = cx / fx + 0.3f * tan_fovx;
            float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
            float lim_y_neg = cy / fy + 0.3f * tan_fovy;

            float clamp_x = tx / tz;
            clamp_x = fmaxf(-lim_x_neg, fminf(clamp_x, lim_x_pos));
            tx = tz * clamp_x;

            float clamp_y = ty / tz;
            clamp_y = fmaxf(-lim_y_neg, fminf(clamp_y, lim_y_pos));
            ty = tz * clamp_y;

            float J[2][3] = {
                {fx / tz, 0.0f, -fx * tx / tz2},
                {0.0f, fy / tz, -fy * ty / tz2}
            };

            float cov2d[2][2];
            float tmp2[2][3];
            for(int i=0; i<2; ++i) {
                for(int j=0; j<3; ++j) {
                    tmp2[i][j] = J[i][0]*cov_c[0][j] + J[i][1]*cov_c[1][j] + J[i][2]*cov_c[2][j];
                }
            }
            for(int i=0; i<2; ++i) {
                for(int j=0; j<2; ++j) {
                    cov2d[i][j] = tmp2[i][0]*J[j][0] + tmp2[i][1]*J[j][1] + tmp2[i][2]*J[j][2];
                }
            }

            float mean2d[2];
            mean2d[0] = (K[0][0]*mean_c[0] + K[0][1]*mean_c[1] + K[0][2]*mean_c[2]) / tz;
            mean2d[1] = (K[1][0]*mean_c[0] + K[1][1]*mean_c[1] + K[1][2]*mean_c[2]) / tz;

            cov2d[0][0] += eps2d;
            cov2d[1][1] += eps2d;

            float det = cov2d[0][0]*cov2d[1][1] - cov2d[0][1]*cov2d[1][0];
            if (det < 1e-10f) det = 1e-10f;

            float conics[3];
            conics[0] = cov2d[1][1] / det;
            conics[1] = -(cov2d[0][1] + cov2d[1][0]) / 2.0f / det;
            conics[2] = cov2d[0][0] / det;

            int radii[2];
            radii[0] = (int)ceilf(3.33f * sqrtf(cov2d[0][0]));
            radii[1] = (int)ceilf(3.33f * sqrtf(cov2d[1][1]));

            int dst_rank = cam_dst_ranks[c];
            int rank_c_start = c_start[dst_rank];
            int local_start = (rank_c_start == 0) ? 0 : scan[rank_c_start * N_local - 1];
            
            int offset = peer_recv_offsets[dst_rank * world_size + my_rank];
            int peer_idx = offset + (local_idx - local_start);

            int* peer_camera_ids = (int*)peer_camera_ids_ptrs[dst_rank];
            int* peer_gaussian_ids = (int*)peer_gaussian_ids_ptrs[dst_rank];
            int* peer_radii = (int*)peer_radii_ptrs[dst_rank];
            float* peer_means2d = (float*)peer_means2d_ptrs[dst_rank];
            float* peer_depths = (float*)peer_depths_ptrs[dst_rank];
            float* peer_conics = (float*)peer_conics_ptrs[dst_rank];
            float* peer_opacities = (float*)peer_opacities_ptrs[dst_rank];
            ColorT* peer_colors = (ColorT*)peer_colors_ptrs[dst_rank];

            peer_camera_ids[peer_idx] = cam_local_ids[c];
            peer_gaussian_ids[peer_idx] = global_gaussian_offset + n;
            
            peer_radii[peer_idx*2] = radii[0];
            peer_radii[peer_idx*2+1] = radii[1];
            
            peer_means2d[peer_idx*2] = mean2d[0];
            peer_means2d[peer_idx*2+1] = mean2d[1];
            
            peer_depths[peer_idx] = depth;
            
            peer_conics[peer_idx*3] = conics[0];
            peer_conics[peer_idx*3+1] = conics[1];
            peer_conics[peer_idx*3+2] = conics[2];
            
            peer_opacities[peer_idx] = opac;
            
            for(int d=0; d<D_colors; ++d) {
                peer_colors[peer_idx*D_colors + d] = colors[n*D_colors + d];
            }
        }
    }
}

void launch_compute_valid(
    torch::Tensor means, torch::Tensor quats, torch::Tensor scales,
    torch::Tensor viewmats, torch::Tensor Ks,
    int width, int height,
    int N_local, int C_total,
    float eps2d, float near_plane, float far_plane,
    torch::Tensor valid
) {
    int threads = 256;
    int blocks = (N_local + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    compute_valid_kernel<<<blocks, threads, 0, stream>>>(
        means.data_ptr<float>(), quats.data_ptr<float>(), scales.data_ptr<float>(),
        viewmats.data_ptr<float>(), Ks.data_ptr<float>(),
        width, height, N_local, C_total,
        eps2d, near_plane, far_plane, valid.data_ptr<int>()
    );
}

void launch_project_and_push(
    torch::Tensor means, torch::Tensor quats, torch::Tensor scales,
    torch::Tensor opacities, torch::Tensor colors,
    torch::Tensor viewmats, torch::Tensor Ks,
    torch::Tensor cam_dst_ranks, torch::Tensor cam_local_ids,
    int width, int height,
    torch::Tensor scan, torch::Tensor c_start,
    int N_local, int C_total, int D_colors,
    float eps2d, float near_plane, float far_plane,
    int my_rank, int global_gaussian_offset,
    torch::Tensor peer_recv_offsets,
    torch::Tensor peer_cam_ptrs, torch::Tensor peer_gauss_ptrs,
    torch::Tensor peer_radii_ptrs, torch::Tensor peer_means2d_ptrs,
    torch::Tensor peer_depths_ptrs, torch::Tensor peer_conics_ptrs,
    torch::Tensor peer_opac_ptrs, torch::Tensor peer_colors_ptrs,
    int world_size
) {
    int threads = 256;
    int blocks = (N_local + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    #define LAUNCH_PUSH(COLOR_T) \
        project_and_push_kernel<COLOR_T><<<blocks, threads, 0, stream>>>( \
            means.data_ptr<float>(), quats.data_ptr<float>(), scales.data_ptr<float>(), \
            opacities.data_ptr<float>(), (const COLOR_T*)colors.data_ptr(), \
            viewmats.data_ptr<float>(), Ks.data_ptr<float>(), \
            cam_dst_ranks.data_ptr<int>(), cam_local_ids.data_ptr<int>(), \
            width, height, \
            scan.data_ptr<int>(), c_start.data_ptr<int>(), \
            N_local, C_total, D_colors, \
            eps2d, near_plane, far_plane, \
            my_rank, global_gaussian_offset, \
            peer_recv_offsets.data_ptr<int>(), \
            peer_cam_ptrs.data_ptr<int64_t>(), peer_gauss_ptrs.data_ptr<int64_t>(), \
            peer_radii_ptrs.data_ptr<int64_t>(), peer_means2d_ptrs.data_ptr<int64_t>(), \
            peer_depths_ptrs.data_ptr<int64_t>(), peer_conics_ptrs.data_ptr<int64_t>(), \
            peer_opac_ptrs.data_ptr<int64_t>(), peer_colors_ptrs.data_ptr<int64_t>(), \
            world_size \
        )

    if (colors.scalar_type() == at::ScalarType::BFloat16) {
        LAUNCH_PUSH(__nv_bfloat16);
    } else {
        LAUNCH_PUSH(float);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_compute_valid", &launch_compute_valid, "Fused projection pass 1: validity and counting");
    m.def("launch_project_and_push", &launch_project_and_push, "Fused projection pass 2: push to symmetric peer memory");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gsplat_dist_fused_ext", CUDA_SRC)
    return _ext

def _all_gather_tensor_list(world_size: int, tensor_list: list[Tensor]) -> list[Tensor]:
    if world_size == 1: return tensor_list
    N = len(tensor_list[0])
    data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
    sizes = [t.numel() // N for t in tensor_list]
    collected = [torch.empty_like(data) for _ in range(world_size)]
    dist.all_gather(collected, data)
    collected = torch.cat(collected, dim=0)
    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for out_tensor, tensor in zip(out_tensor_tuple, tensor_list):
        out_tensor = out_tensor.view(-1, *tensor.shape[1:])
        out_tensor_list.append(out_tensor)
    return out_tensor_list

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
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = means.device

    N = means.shape[0]
    C = viewmats.shape[0]
    D = colors.shape[1]

    # Quick pre-gather global shapes & configurations
    N_t = torch.tensor([N], dtype=torch.int32, device=device)
    N_world = _all_gather_tensor_list(world_size, [N_t])[0].flatten().tolist()
    C_world = [C] * world_size
    global_gaussian_offset = sum(N_world[:world_rank])
    C_total = sum(C_world)
    
    c_start = [sum(C_world[:r]) for r in range(world_size)]
    c_start_t = torch.tensor(c_start, dtype=torch.int32, device=device)

    all_viewmats = _all_gather_tensor_list(world_size, [viewmats])[0].float().contiguous()
    all_Ks = _all_gather_tensor_list(world_size, [Ks])[0].float().contiguous()

    cam_dst_ranks = torch.zeros(C_total, dtype=torch.int32, device=device)
    cam_local_ids = torch.zeros(C_total, dtype=torch.int32, device=device)
    for r in range(world_size):
        start, end = sum(C_world[:r]), sum(C_world[:r+1])
        cam_dst_ranks[start:end] = r
        cam_local_ids[start:end] = torch.arange(C_world[r], dtype=torch.int32, device=device)

    means_f32 = means.float().contiguous()
    quats_f32 = quats.float().contiguous()
    scales_f32 = scales.float().contiguous()
    opacities_f32 = opacities.float().contiguous()
    colors_c = colors.contiguous()

    ext = _get_ext()
    
    # Pass 1: Local lightweight projection to determine precise required memory per chunk
    valid = torch.zeros(C_total * N, dtype=torch.int32, device=device)
    ext.launch_compute_valid(
        means_f32, quats_f32, scales_f32, all_viewmats, all_Ks,
        image_width, image_height, N, C_total,
        eps2d, near_plane, far_plane, valid
    )

    # Deterministic offsets (maintains ideal grouping and avoids expensive atomic reductions)
    scan = torch.cumsum(valid, dim=0)

    send_counts = torch.zeros(world_size, dtype=torch.int32, device=device)
    for r in range(world_size):
        end_idx = sum(C_world[:r+1]) * N - 1
        start_idx = sum(C_world[:r]) * N - 1
        end_val = scan[end_idx].item() if end_idx >= 0 else 0
        start_val = scan[start_idx].item() if start_idx >= 0 else 0
        send_counts[r] = end_val - start_val

    # Setup the memory structures on peers dynamically
    recv_counts = torch.zeros(world_size, dtype=torch.int32, device=device)
    dist.all_to_all_single(recv_counts, send_counts)
    
    recv_offsets = torch.cumsum(recv_counts, dim=0) - recv_counts
    recv_total = recv_counts.sum().item()

    all_recv_offsets = [torch.zeros(world_size, dtype=torch.int32, device=device) for _ in range(world_size)]
    dist.all_gather(all_recv_offsets, recv_offsets)
    peer_recv_offsets = torch.stack(all_recv_offsets) 

    all_recv_totals = torch.zeros(world_size, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(all_recv_totals, torch.tensor([recv_total], dtype=torch.int32, device=device))

    # Single massive symmetric memory allocation split into typed views
    bytes_per_elem = 44 + D * colors_c.element_size()
    buf = symm_mem.empty(recv_total * bytes_per_elem, dtype=torch.uint8, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    def slice_buffer(base_ptr_int, r_total):
        ptrs = {}
        curr = base_ptr_int
        ptrs['cam'] = curr; curr += r_total * 4
        ptrs['gauss'] = curr; curr += r_total * 4
        ptrs['radii'] = curr; curr += r_total * 8
        ptrs['means2d'] = curr; curr += r_total * 8
        ptrs['depths'] = curr; curr += r_total * 4
        ptrs['conics'] = curr; curr += r_total * 12
        ptrs['opac'] = curr; curr += r_total * 4
        ptrs['colors'] = curr; curr += r_total * D * colors_c.element_size()
        return ptrs

    peer_cam_ptrs, peer_gauss_ptrs, peer_radii_ptrs = [], [], []
    peer_means2d_ptrs, peer_depths_ptrs, peer_conics_ptrs = [], [], []
    peer_opac_ptrs, peer_colors_ptrs = [], []
    
    for p in range(world_size):
        ptrs = slice_buffer(hdl.buffer_ptrs[p], all_recv_totals[p].item())
        peer_cam_ptrs.append(ptrs['cam']); peer_gauss_ptrs.append(ptrs['gauss'])
        peer_radii_ptrs.append(ptrs['radii']); peer_means2d_ptrs.append(ptrs['means2d'])
        peer_depths_ptrs.append(ptrs['depths']); peer_conics_ptrs.append(ptrs['conics'])
        peer_opac_ptrs.append(ptrs['opac']); peer_colors_ptrs.append(ptrs['colors'])

    # Pass 2: Recompute projection and synchronously stream the results over NVLink straight to the receiver
    ext.launch_project_and_push(
        means_f32, quats_f32, scales_f32, opacities_f32, colors_c,
        all_viewmats, all_Ks, cam_dst_ranks, cam_local_ids,
        image_width, image_height,
        scan, c_start_t, N, C_total, D, eps2d, near_plane, far_plane,
        world_rank, global_gaussian_offset, peer_recv_offsets.flatten(),
        torch.tensor(peer_cam_ptrs, dtype=torch.int64, device=device),
        torch.tensor(peer_gauss_ptrs, dtype=torch.int64, device=device),
        torch.tensor(peer_radii_ptrs, dtype=torch.int64, device=device),
        torch.tensor(peer_means2d_ptrs, dtype=torch.int64, device=device),
        torch.tensor(peer_depths_ptrs, dtype=torch.int64, device=device),
        torch.tensor(peer_conics_ptrs, dtype=torch.int64, device=device),
        torch.tensor(peer_opac_ptrs, dtype=torch.int64, device=device),
        torch.tensor(peer_colors_ptrs, dtype=torch.int64, device=device),
        world_size
    )

    hdl.barrier(channel=0)

    # Finally slice out local incoming views
    curr = 0
    cam_ids_out = buf[curr : curr + recv_total * 4].view(torch.int32); curr += recv_total * 4
    gauss_ids_out = buf[curr : curr + recv_total * 4].view(torch.int32); curr += recv_total * 4
    radii_out = buf[curr : curr + recv_total * 8].view(torch.int32).view(recv_total, 2); curr += recv_total * 8
    means2d_out = buf[curr : curr + recv_total * 8].view(torch.float32).view(recv_total, 2); curr += recv_total * 8
    depths_out = buf[curr : curr + recv_total * 4].view(torch.float32); curr += recv_total * 4
    conics_out = buf[curr : curr + recv_total * 12].view(torch.float32).view(recv_total, 3); curr += recv_total * 12
    opacities_out = buf[curr : curr + recv_total * 4].view(torch.float32); curr += recv_total * 4
    colors_out = buf[curr : curr + recv_total * D * colors_c.element_size()].view(colors_c.dtype).view(recv_total, D)

    return (
        cam_ids_out, gauss_ids_out, radii_out,
        means2d_out, depths_out, conics_out,
        opacities_out, colors_out
    )