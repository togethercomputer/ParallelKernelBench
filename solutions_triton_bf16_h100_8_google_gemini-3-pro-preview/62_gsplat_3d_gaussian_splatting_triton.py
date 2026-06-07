"""
Strategy:
1. Eliminated standard PyTorch `all_gather` and `all_to_all` bottlenecks by replacing them 
   with a custom CUDA extension over NVLink via `torch.distributed._symmetric_memory`.
2. Computation-Communication Overlap: The projection of local Gaussians onto cameras is chunked 
   per peer. As soon as projections for a peer are packed locally, a custom asynchronous UVA 
   kernel (`push_data_to_peer`) pushes the valid splats directly into that peer's pre-allocated 
   symmetric memory buffer.
3. Decoupled Memory Allocation: By proving that maximum received projections from a peer is 
   `N_world[peer] * C_local`, each rank statically partitions its receive buffer. This completely 
   removes the need for an AllToAll counts exchange before data transfer.
4. Custom C++ Gather & Compact: Dedicated async CUDA kernels unpack and contiguous-ify the 
   dynamically sized blocks received from peers without Host synchronization.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch import Tensor
from utils.cuda_helpers import compile_cuda_extension

###############################################################################
# Projection helpers reproduced verbatim from gsplat/cuda/_torch_impl.py
###############################################################################

def _quat_to_rotmat(quats: Tensor) -> Tensor:
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


def _quat_scale_to_covar_preci(
    quats: Tensor, scales: Tensor, compute_covar: bool = True, compute_preci: bool = True, triu: bool = False
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    batch_dims = quats.shape[:-1]
    R = _quat_to_rotmat(quats)
    covars = precis = None
    if compute_covar:
        M = R * scales[..., None, :]
        covars = torch.einsum("...ij,...kj -> ...ik", M, M)
        if triu:
            covars = covars.reshape(batch_dims + (9,))
            covars = (covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]) / 2.0
    if compute_preci:
        P = R * (1 / scales[..., None, :])
        precis = torch.einsum("...ij,...kj -> ...ik", P, P)
        if triu:
            precis = precis.reshape(batch_dims + (9,))
            precis = (precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]) / 2.0
    return covars, precis


def _world_to_cam(means: Tensor, covars: Tensor, viewmats: Tensor) -> Tuple[Tensor, Tensor]:
    R = viewmats[..., :3, :3]
    t = viewmats[..., :3, 3]
    means_c = torch.einsum("...cij,...nj->...cni", R, means) + t[..., None, :]
    covars_c = torch.einsum("...cij,...njk,...clk->...cnil", R, covars, R)
    return means_c, covars_c


def _persp_proj(
    means: Tensor, covars: Tensor, Ks: Tensor, width: int, height: int
) -> Tuple[Tensor, Tensor]:
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]

    tx, ty, tz = torch.unbind(means, dim=-1)
    tz2 = tz**2

    fx = Ks[..., 0, 0, None]
    fy = Ks[..., 1, 1, None]
    cx = Ks[..., 0, 2, None]
    cy = Ks[..., 1, 2, None]
    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros(batch_dims + (C, N), device=means.device, dtype=means.dtype)
    J = torch.stack([fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1).reshape(batch_dims + (C, N, 2, 3))

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("...ij,...nj->...ni", Ks[..., :2, :3], means)
    means2d = means2d / tz[..., None]
    return means2d, cov2d


def _fully_fused_projection(
    means: Tensor, covars: Tensor, viewmats: Tensor, Ks: Tensor, width: int, height: int,
    eps2d: float = 0.3, near_plane: float = 0.01, far_plane: float = 1e10,
    calc_compensations: bool = False, camera_model: str = "pinhole"
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    means_c, covars_c = _world_to_cam(means, covars, viewmats)
    means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)

    det_orig = covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    det = det.clamp(min=1e-10)

    compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0)) if calc_compensations else None

    conics = torch.stack([
        covars2d[..., 1, 1] / det,
        -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
        covars2d[..., 0, 0] / det,
    ], dim=-1)

    depths = means_c[..., 2]
    radius_x = torch.ceil(3.33 * torch.sqrt(covars2d[..., 0, 0]))
    radius_y = torch.ceil(3.33 * torch.sqrt(covars2d[..., 1, 1]))
    radius = torch.stack([radius_x, radius_y], dim=-1)

    valid = (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0) & (means2d[..., 0] - radius[..., 0] < width) &
        (means2d[..., 1] + radius[..., 1] > 0) & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0

    return radius.int(), means2d, depths, conics, compensations


def _pack_projection_results(
    radii: Tensor, means2d: Tensor, depths: Tensor, conics: Tensor, compensations: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    C, N = radii.shape[:2]
    valid = (radii > 0).all(dim=-1)
    camera_ids, gaussian_ids = torch.where(valid)
    camera_ids = camera_ids.int()
    gaussian_ids = gaussian_ids.int()

    radii_packed = radii[valid]
    means2d_packed = means2d[valid]
    depths_packed = depths[valid]
    conics_packed = conics[valid]
    compensations_packed = compensations[valid] if compensations is not None else None

    counts = torch.bincount(camera_ids.long(), minlength=C)
    indptr = torch.zeros(C + 1, dtype=torch.int32, device=radii.device)
    indptr[1:] = torch.cumsum(counts, dim=0).int()

    return camera_ids, gaussian_ids, indptr, radii_packed, means2d_packed, depths_packed, conics_packed, compensations_packed


###############################################################################
# Custom CUDA Extension & Symmetric Memory Setup
###############################################################################

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

void gather_meta(torch::Tensor ptrs, torch::Tensor N_world, torch::Tensor C_world) {
    int world_size = ptrs.size(0);
    const int64_t* ptrs_data = ptrs.data_ptr<int64_t>();
    int32_t* N_out = N_world.data_ptr<int32_t>();
    int32_t* C_out = C_world.data_ptr<int32_t>();
    for(int p = 0; p < world_size; ++p) {
        int32_t* remote_meta = reinterpret_cast<int32_t*>(ptrs_data[p]);
        cudaMemcpy(&N_out[p], remote_meta, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&C_out[p], remote_meta + 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }
}

void gather_cams(torch::Tensor ptrs_cpu, torch::Tensor C_world_cpu, torch::Tensor viewmats_out, torch::Tensor Ks_out) {
    int world_size = ptrs_cpu.size(0);
    const int64_t* ptrs_data = ptrs_cpu.data_ptr<int64_t>();
    const int32_t* C_world_data = C_world_cpu.data_ptr<int32_t>();
    float* view_out = viewmats_out.data_ptr<float>();
    float* K_out = Ks_out.data_ptr<float>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int offset_c = 0;
    for(int p = 0; p < world_size; ++p) {
        float* remote_cam = reinterpret_cast<float*>(ptrs_data[p]);
        int C = C_world_data[p];
        if (C > 0) {
            cudaMemcpyAsync(view_out + offset_c * 16, remote_cam, C * 16 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(K_out + offset_c * 9, remote_cam + C * 16, C * 9 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
        offset_c += C;
    }
}

void push_data_to_peer(
    int64_t peer_data_ptr,
    int64_t peer_meta_ptr,
    int my_rank,
    int write_offset,
    torch::Tensor cam_ids,
    torch::Tensor gau_ids,
    torch::Tensor radii,
    torch::Tensor means2d,
    torch::Tensor depths,
    torch::Tensor conics,
    torch::Tensor opacities,
    torch::Tensor colors,
    std::vector<int64_t> peer_offsets,
    torch::Tensor count_tensor
) {
    int count = cam_ids.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t count_ptr = peer_meta_ptr + (2 + my_rank) * 4;
    cudaMemcpyAsync(reinterpret_cast<void*>(count_ptr), count_tensor.data_ptr(), 4, cudaMemcpyDeviceToDevice, stream);

    if (count > 0) {
        auto copy_feat = [&](torch::Tensor src, int64_t base_offset, int elem_size) {
            int64_t dst_ptr = peer_data_ptr + base_offset + write_offset * elem_size;
            cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr), src.data_ptr(), count * elem_size, cudaMemcpyDeviceToDevice, stream);
        };

        copy_feat(cam_ids, peer_offsets[0], 4);
        copy_feat(gau_ids, peer_offsets[1], 4);
        copy_feat(radii, peer_offsets[2], 8);
        copy_feat(means2d, peer_offsets[3], 2 * means2d.element_size());
        copy_feat(depths, peer_offsets[4], 1 * depths.element_size());
        copy_feat(conics, peer_offsets[5], 3 * conics.element_size());
        copy_feat(opacities, peer_offsets[6], 1 * opacities.element_size());
        copy_feat(colors, peer_offsets[7], colors.size(1) * colors.element_size());
    }
}

void compact_recv_data(
    int64_t my_data_ptr,
    std::vector<int64_t> my_offsets,
    std::vector<int> counts,
    std::vector<int> N_w,
    int C_local,
    torch::Tensor out_cam,
    torch::Tensor out_gau,
    torch::Tensor out_rad,
    torch::Tensor out_mea,
    torch::Tensor out_dep,
    torch::Tensor out_con,
    torch::Tensor out_opa,
    torch::Tensor out_col
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int world_size = counts.size();
    int out_offset = 0;
    int N_prefix = 0;

    auto copy_compact = [&](torch::Tensor dst, int64_t base_offset, int elem_size, int count, int src_item_offset, int out_off) {
        int64_t src_ptr = my_data_ptr + base_offset + src_item_offset * elem_size;
        void* dst_ptr = reinterpret_cast<char*>(dst.data_ptr()) + out_off * elem_size;
        cudaMemcpyAsync(dst_ptr, reinterpret_cast<void*>(src_ptr), count * elem_size, cudaMemcpyDeviceToDevice, stream);
    };

    for(int p = 0; p < world_size; ++p) {
        int count = counts[p];
        if (count > 0) {
            int src_item_offset = N_prefix * C_local;
            copy_compact(out_cam, my_offsets[0], 4, count, src_item_offset, out_offset);
            copy_compact(out_gau, my_offsets[1], 4, count, src_item_offset, out_offset);
            copy_compact(out_rad, my_offsets[2], 8, count, src_item_offset, out_offset);
            copy_compact(out_mea, my_offsets[3], 2 * out_mea.element_size(), count, src_item_offset, out_offset);
            copy_compact(out_dep, my_offsets[4], 1 * out_dep.element_size(), count, src_item_offset, out_offset);
            copy_compact(out_con, my_offsets[5], 3 * out_con.element_size(), count, src_item_offset, out_offset);
            copy_compact(out_opa, my_offsets[6], 1 * out_opa.element_size(), count, src_item_offset, out_offset);
            copy_compact(out_col, my_offsets[7], out_col.size(1) * out_col.element_size(), count, src_item_offset, out_offset);
            out_offset += count;
        }
        N_prefix += N_w[p];
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_meta", &gather_meta, "Gather N and C from peers via UVA");
    m.def("gather_cams", &gather_cams, "Gather viewmats and Ks from peers via UVA");
    m.def("push_data_to_peer", &push_data_to_peer, "Push packed projection data to peer via UVA");
    m.def("compact_recv_data", &compact_recv_data, "Compact dynamically received NVLink data blocks");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gsplat_symm_uva_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def get_peer_offsets(peer_C: int, N_global: int, D: int, elem_size: int) -> List[int]:
    cap = int(N_global * peer_C)
    s_cam = cap * 4
    s_gau = cap * 4
    s_rad = cap * 8
    s_mea = cap * 2 * elem_size
    s_dep = cap * elem_size
    s_con = cap * 3 * elem_size
    s_opa = cap * elem_size
    s_col = cap * D * elem_size

    offsets = [0]
    for s in [s_cam, s_gau, s_rad, s_mea, s_dep, s_con, s_opa, s_col]:
        offsets.append((offsets[-1] + s + 7) & ~7)
    return offsets


@torch.no_grad()
def solution(
    means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, colors: Tensor,
    viewmats: Tensor, Ks: Tensor, image_width: int, image_height: int,
    eps2d: float = 0.3, near_plane: float = 0.01, far_plane: float = 1e10, camera_model: str = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = means.device

    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    N_local = means.shape[0]
    C_local = viewmats.shape[0]
    D = colors.shape[1]
    elem_size = means.element_size()

    # 1. Meta Exchange: Share N_local and C_local
    if "meta" not in _symm_cache:
        buf = symm_mem.empty(1024, dtype=torch.int32, device=device)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        _symm_cache["meta"] = (buf, hdl)
    meta_buf, meta_hdl = _symm_cache["meta"]

    meta_buf.zero_()
    meta_buf[0] = N_local
    meta_buf[1] = C_local
    meta_hdl.barrier(channel=0)

    N_world = torch.empty(world_size, dtype=torch.int32, device='cpu')
    C_world = torch.empty(world_size, dtype=torch.int32, device='cpu')
    ptrs_tensor = torch.tensor(meta_hdl.buffer_ptrs, dtype=torch.int64, device='cpu')
    ext.gather_meta(ptrs_tensor, N_world, C_world)

    N_global = int(N_world.sum().item())
    N_offset = int(N_world[:rank].sum().item())

    # 2. Camera Exchange: Share local cameras directly via UVA
    cam_req = int(C_local * 25)
    global_max_cam = max(int(C_world.max().item() * 25), 25)

    if "cam_cap" not in _symm_cache or _symm_cache["cam_cap"] < global_max_cam:
        buf = symm_mem.empty(cam_req, dtype=torch.float32, device=device)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        _symm_cache["cam"] = (buf, hdl)
        _symm_cache["cam_cap"] = global_max_cam
    cam_buf, cam_hdl = _symm_cache["cam"]

    if C_local > 0:
        cam_buf[:C_local*16] = viewmats.float().reshape(-1)
        cam_buf[C_local*16:C_local*25] = Ks.float().reshape(-1)
    cam_hdl.barrier(channel=0)

    C_global = int(C_world.sum().item())
    viewmats_gathered = torch.empty((C_global, 4, 4), dtype=torch.float32, device=device)
    Ks_gathered = torch.empty((C_global, 3, 3), dtype=torch.float32, device=device)
    ext.gather_cams(torch.tensor(cam_hdl.buffer_ptrs, dtype=torch.int64, device='cpu'), C_world, viewmats_gathered, Ks_gathered)

    # 3. Dynamic Data Buffer Setup
    my_req = get_peer_offsets(C_local, N_global, D, elem_size)[-1]
    global_max_req = max([get_peer_offsets(int(c), N_global, D, elem_size)[-1] for c in C_world.tolist()] + [8])

    if "data_cap" not in _symm_cache or _symm_cache["data_cap"] < global_max_req:
        buf = symm_mem.empty(my_req, dtype=torch.uint8, device=device)
        hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
        _symm_cache["data"] = (buf, hdl)
        _symm_cache["data_cap"] = global_max_req
    data_buf, data_hdl = _symm_cache["data"]

    # 4. Overlapped Compute and Communication
    covars, _ = _quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False, triu=False)

    count_tensors = []
    for p in range(world_size):
        peer = (rank + p) % world_size
        C_peer = int(C_world[peer].item())
        if C_peer == 0:
            continue

        c_start = int(C_world[:peer].sum().item())
        p_views = viewmats_gathered[c_start : c_start + C_peer].to(viewmats.dtype)
        p_Ks = Ks_gathered[c_start : c_start + C_peer].to(Ks.dtype)

        radii, means2d, depths, conics, compensations = _fully_fused_projection(
            means, covars, p_views, p_Ks, image_width, image_height,
            eps2d=eps2d, near_plane=near_plane, far_plane=far_plane,
            calc_compensations=False, camera_model=camera_model
        )

        (c_ids, g_ids, _, rad_p, mea_p, dep_p, con_p, _) = _pack_projection_results(radii, means2d, depths, conics, compensations)

        g_ids += N_offset
        opa_p = opacities[g_ids.long() - N_offset]
        col_p = colors[g_ids.long() - N_offset]

        count = c_ids.numel()
        count_t = torch.tensor([count], dtype=torch.int32, device=device)
        count_tensors.append(count_t)

        peer_offsets = get_peer_offsets(C_peer, N_global, D, elem_size)

        ext.push_data_to_peer(
            int(data_hdl.buffer_ptrs[peer]),
            int(meta_hdl.buffer_ptrs[peer]),
            rank,
            N_offset * C_peer,
            c_ids, g_ids, rad_p, mea_p, dep_p, con_p, opa_p, col_p,
            peer_offsets, count_t
        )

    # 5. Receive and Compact
    data_hdl.barrier(channel=0)
    
    recv_counts = meta_buf[2 : 2 + world_size].cpu().tolist()
    total_recv = sum(recv_counts)

    out_cam = torch.empty(total_recv, dtype=torch.int32, device=device)
    out_gau = torch.empty(total_recv, dtype=torch.int32, device=device)
    out_rad = torch.empty((total_recv, 2), dtype=torch.int32, device=device)
    out_mea = torch.empty((total_recv, 2), dtype=means.dtype, device=device)
    out_dep = torch.empty(total_recv, dtype=means.dtype, device=device)
    out_con = torch.empty((total_recv, 3), dtype=means.dtype, device=device)
    out_opa = torch.empty(total_recv, dtype=means.dtype, device=device)
    out_col = torch.empty((total_recv, D), dtype=means.dtype, device=device)

    my_offsets = get_peer_offsets(C_local, N_global, D, elem_size)

    ext.compact_recv_data(
        int(data_hdl.buffer_ptrs[rank]),
        my_offsets,
        recv_counts,
        N_world.tolist(),
        C_local,
        out_cam, out_gau, out_rad, out_mea, out_dep, out_con, out_opa, out_col
    )

    return out_cam, out_gau, out_rad, out_mea, out_dep, out_con, out_opa, out_col