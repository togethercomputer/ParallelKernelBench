import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF
import torch.nn.functional as F
from torch import Tensor


def _all_gather_int32(
    world_size: int, value: Union[int, Tensor], device: Optional[torch.device] = None
) -> List[int]:
    if world_size == 1:
        return [value]

    if isinstance(value, int):
        assert device is not None, "device is required for scalar input"
        value_tensor = torch.tensor(value, dtype=torch.int, device=device)
    else:
        value_tensor = value

    collected = torch.empty(
        world_size, dtype=value_tensor.dtype, device=value_tensor.device
    )
    dist.all_gather_into_tensor(collected, value_tensor)

    if isinstance(value, int):
        return collected.tolist()
    else:
        return collected.unbind()


def _all_to_all_int32(
    world_size: int,
    values: List[Union[int, Tensor]],
    device: Optional[torch.device] = None,
) -> List[int]:
    if world_size == 1:
        return values

    assert len(values) == world_size

    if any(isinstance(v, int) for v in values):
        assert device is not None, "device is required for scalar input"

    values_tensor = [
        (torch.tensor(v, dtype=torch.int, device=device) if isinstance(v, int) else v)
        for v in values
    ]

    collected = [torch.empty_like(v) for v in values_tensor]
    dist.all_to_all(collected, values_tensor)

    return [
        v.item() if isinstance(tensor, int) else v
        for v, tensor in zip(collected, values)
    ]


def _all_gather_tensor_list(world_size: int, tensor_list: List[Tensor]) -> List[Tensor]:
    if world_size == 1:
        return tensor_list

    N = len(tensor_list[0])
    for tensor in tensor_list:
        assert len(tensor) == N, "All tensors should have the same first dimension size"

    data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
    sizes = [t.numel() // N for t in tensor_list]

    if data.requires_grad:
        collected = distF.all_gather(data)
    else:
        collected = [torch.empty_like(data) for _ in range(world_size)]
        dist.all_gather(collected, data)
    collected = torch.cat(collected, dim=0)

    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for out_tensor, tensor in zip(out_tensor_tuple, tensor_list):
        out_tensor = out_tensor.view(-1, *tensor.shape[1:])
        out_tensor_list.append(out_tensor)
    return out_tensor_list


def _all_to_all_tensor_list(
    world_size: int,
    tensor_list: List[Tensor],
    splits: List[Union[int, Tensor]],
    output_splits: Optional[List[Union[int, Tensor]]] = None,
) -> List[Tensor]:
    if world_size == 1:
        return tensor_list

    N = len(tensor_list[0])
    for tensor in tensor_list:
        assert len(tensor) == N, "All tensors should have the same first dimension size"

    assert len(splits) == world_size

    data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
    sizes = [t.numel() // N for t in tensor_list]

    if output_splits is not None:
        collected_splits = output_splits
    else:
        collected_splits = _all_to_all_int32(world_size, splits, device=data.device)
    collected = [
        torch.empty((l, *data.shape[1:]), dtype=data.dtype, device=data.device)
        for l in collected_splits
    ]
    splits = [s.item() if isinstance(s, Tensor) else s for s in splits]
    if data.requires_grad:
        distF.all_to_all(collected, data.split(splits, dim=0))
    else:
        dist.all_to_all(collected, list(data.split(splits, dim=0)))
    collected = torch.cat(collected, dim=0)

    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for out_tensor, tensor in zip(out_tensor_tuple, tensor_list):
        out_tensor = out_tensor.view(-1, *tensor.shape[1:])
        out_tensor_list.append(out_tensor)
    return out_tensor_list


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
    quats: Tensor,
    scales: Tensor,
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape
    R = _quat_to_rotmat(quats)

    if compute_covar:
        M = R * scales[..., None, :]
        covars = torch.einsum("...ij,...kj -> ...ik", M, M)
        if triu:
            covars = covars.reshape(batch_dims + (9,))
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0
    if compute_preci:
        P = R * (1 / scales[..., None, :])
        precis = torch.einsum("...ij,...kj -> ...ik", P, P)
        if triu:
            precis = precis.reshape(batch_dims + (9,))
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0

    return covars if compute_covar else None, precis if compute_preci else None


def _world_to_cam(
    means: Tensor,
    covars: Tensor,
    viewmats: Tensor,
) -> Tuple[Tensor, Tensor]:
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape

    R = viewmats[..., :3, :3]
    t = viewmats[..., :3, 3]
    means_c = (
        torch.einsum("...cij,...nj->...cni", R, means) + t[..., None, :]
    )
    covars_c = torch.einsum(
        "...cij,...njk,...clk->...cnil", R, covars, R
    )
    return means_c, covars_c


def _persp_proj(
    means: Tensor,
    covars: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert covars.shape == batch_dims + (C, N, 3, 3), covars.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

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
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(batch_dims + (C, N, 2, 3))

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum(
        "...ij,...nj->...ni", Ks[..., :2, :3], means
    )
    means2d = means2d / tz[..., None]
    return means2d, cov2d


def _fully_fused_projection(
    means: Tensor,
    covars: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: str = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert camera_model == "pinhole", "only pinhole supported"

    means_c, covars_c = _world_to_cam(means, covars, viewmats)
    means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)

    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )

    depths = means_c[..., 2]

    # CUDA fully_fused_projection can use opacities for tighter radii; this torch
    # implementation follows gsplat/cuda/_torch_impl.py and uses covariance only.
    radius_x = torch.ceil(3.33 * torch.sqrt(covars2d[..., 0, 0]))
    radius_y = torch.ceil(3.33 * torch.sqrt(covars2d[..., 1, 1]))
    radius = torch.stack([radius_x, radius_y], dim=-1)

    valid = (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return radii, means2d, depths, conics, compensations


def _pack_projection_results(
    radii: Tensor,
    means2d: Tensor,
    depths: Tensor,
    conics: Tensor,
    compensations: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    C, N = radii.shape[:2]
    device = radii.device

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
    indptr = torch.zeros(C + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(counts, dim=0).int()

    return (
        camera_ids, gaussian_ids, indptr,
        radii_packed, means2d_packed, depths_packed, conics_packed,
        compensations_packed,
    )


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
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = means.device

    N = means.shape[0]
    C = viewmats.shape[0]
    D = colors.shape[1]

    N_world = _all_gather_int32(world_size, N, device=device)
    C_world = [C] * world_size

    viewmats, Ks = _all_gather_tensor_list(world_size, [viewmats, Ks])

    C = len(viewmats)

    covars, _ = _quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=False
    )

    radii, means2d, depths, conics, compensations = _fully_fused_projection(
        means, covars, viewmats, Ks, image_width, image_height,
        eps2d=eps2d, near_plane=near_plane, far_plane=far_plane,
        calc_compensations=False,
        camera_model=camera_model,
    )

    (
        camera_ids, gaussian_ids, _indptr,
        radii, means2d, depths, conics, _compensations,
    ) = _pack_projection_results(radii, means2d, depths, conics, compensations)

    opacities = opacities[gaussian_ids.long()]
    colors = colors[gaussian_ids.long()]

    cnts = torch.bincount(camera_ids.long(), minlength=C)
    cnts = cnts.split(C_world, dim=0)
    cnts = [cuts.sum() for cuts in cnts]

    collected_splits = _all_to_all_int32(world_size, cnts, device=device)

    (radii,) = _all_to_all_tensor_list(
        world_size, [radii], cnts, output_splits=collected_splits
    )

    (means2d, depths, conics, opacities, colors) = _all_to_all_tensor_list(
        world_size,
        [means2d, depths, conics, opacities, colors],
        cnts,
        output_splits=collected_splits,
    )

    offsets = torch.tensor(
        [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
    )
    offsets = torch.cumsum(offsets, dim=0)
    offsets = offsets.repeat_interleave(torch.stack(cnts))
    camera_ids = camera_ids - offsets

    offsets = torch.tensor(
        [0] + N_world[:-1],
        device=gaussian_ids.device,
        dtype=gaussian_ids.dtype,
    )
    offsets = torch.cumsum(offsets, dim=0)
    offsets = offsets.repeat_interleave(torch.stack(cnts))
    gaussian_ids = gaussian_ids + offsets

    (camera_ids, gaussian_ids) = _all_to_all_tensor_list(
        world_size,
        [camera_ids, gaussian_ids],
        cnts,
        output_splits=collected_splits,
    )

    return (
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        opacities,
        colors,
    )
