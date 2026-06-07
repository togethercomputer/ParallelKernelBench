from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _index_undot(index: int, loop_size: List[int]) -> List[int]:
    out: List[int] = []
    for size in reversed(loop_size):
        out.append(index % size)
        index //= size
    return list(reversed(out))


def _index_dot(index: List[int], loop_size: List[int]) -> int:
    value = 0
    for dim, size in zip(index, loop_size):
        value = value * size + dim
    return value


def _split_tiles(
    tile_numels: List[int],
    group: Optional[dist.ProcessGroup],
) -> Tuple[List[int], List[int]]:
    if group is None:
        tile_indices = list(range(len(tile_numels)))
        return tile_indices, tile_indices

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    sorted_tiles = sorted(
        range(len(tile_numels)),
        key=lambda idx: tile_numels[idx],
        reverse=True,
    )
    per_rank = [sorted_tiles[r::world_size] for r in range(world_size)]
    global_order = [idx for shard in per_rank for idx in shard]
    return per_rank[rank], global_order


def _gather_tiles(
    tiles: List[torch.Tensor],
    global_order: List[int],
    template: torch.Tensor,
    group: Optional[dist.ProcessGroup],
) -> List[torch.Tensor]:
    if group is None:
        return tiles

    world_size = dist.get_world_size(group=group)
    local_shapes = [tuple(tile.shape) for tile in tiles]
    all_shapes: List[List[Tuple[int, ...]]] = [[] for _ in range(world_size)]
    dist.all_gather_object(all_shapes, local_shapes, group=group)

    local_flat = (
        torch.cat([tile.reshape(-1).contiguous() for tile in tiles], dim=0)
        if tiles
        else template.new_empty(0)
    )
    local_size = int(local_flat.numel())
    rank_sizes: List[int] = []
    for shapes in all_shapes:
        total = 0
        for shape in shapes:
            numel = 1
            for size in shape:
                numel *= size
            total += numel
        rank_sizes.append(total)
    send = local_flat.repeat(world_size)
    recv = template.new_empty(sum(rank_sizes))
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=rank_sizes,
        input_split_sizes=[local_size] * world_size,
        group=group,
    )

    gathered: List[torch.Tensor] = []
    offset = 0
    for shapes, total in zip(all_shapes, rank_sizes):
        rank_buf = recv[offset : offset + total]
        rank_offset = 0
        for shape in shapes:
            numel = 1
            for size in shape:
                numel *= size
            gathered.append(rank_buf[rank_offset : rank_offset + numel].view(shape))
            rank_offset += numel
        offset += total

    by_index = {tile_idx: tile for tile_idx, tile in zip(global_order, gathered)}
    return [by_index[idx] for idx in sorted(by_index)]


def _blend_t(prev: torch.Tensor, cur: torch.Tensor, extent: int) -> torch.Tensor:
    extent = min(prev.shape[2], cur.shape[2], extent)
    for idx in range(extent):
        ratio = idx / extent
        cur[:, :, idx] = prev[:, :, -extent + idx] * (1.0 - ratio) + cur[:, :, idx] * ratio
    return cur


def _blend_h(prev: torch.Tensor, cur: torch.Tensor, extent: int) -> torch.Tensor:
    extent = min(prev.shape[3], cur.shape[3], extent)
    for idx in range(extent):
        ratio = idx / extent
        cur[:, :, :, idx] = prev[:, :, :, -extent + idx] * (1.0 - ratio) + cur[:, :, :, idx] * ratio
    return cur


def _blend_w(prev: torch.Tensor, cur: torch.Tensor, extent: int) -> torch.Tensor:
    extent = min(prev.shape[4], cur.shape[4], extent)
    for idx in range(extent):
        ratio = idx / extent
        cur[:, :, :, :, idx] = prev[:, :, :, :, -extent + idx] * (1.0 - ratio) + cur[:, :, :, :, idx] * ratio
    return cur


def _decode_tile(tile: torch.Tensor, spatial_upsample: int, temporal_upsample: int) -> torch.Tensor:
    decoded = F.interpolate(
        tile.float(),
        scale_factor=(temporal_upsample, spatial_upsample, spatial_upsample),
        mode="trilinear",
        align_corners=False,
    )
    if decoded.shape[1] < 3:
        repeats = (3 + decoded.shape[1] - 1) // decoded.shape[1]
        decoded = decoded.repeat(1, repeats, 1, 1, 1)
    return decoded[:, :3].to(torch.bfloat16)


@torch.no_grad()
def solution(
    z: torch.Tensor,
    tile_latent_min_length: int,
    tile_latent_min_height: int,
    tile_latent_min_width: int,
    spatial_tile_overlap_factor: float,
    temporal_tile_overlap_factor: float,
    spatial_upsample: int,
    temporal_upsample: int,
    sr_ratio: int = 1,
    first_frame_as_image: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        group = group or dist.group.WORLD
    else:
        group = None
    tile_latent_min_length = tile_latent_min_length + int(first_frame_as_image)
    spatial_upsample = spatial_upsample * sr_ratio
    stride_h = int(tile_latent_min_height * (1.0 - spatial_tile_overlap_factor))
    stride_w = int(tile_latent_min_width * (1.0 - spatial_tile_overlap_factor))
    stride_t = int(tile_latent_min_length * (1.0 - temporal_tile_overlap_factor))
    if min(stride_t, stride_h, stride_w) <= 0:
        raise ValueError("tile overlap factors must leave a positive stride")

    real_t = tile_latent_min_length * temporal_upsample
    real_h = tile_latent_min_height * spatial_upsample
    real_w = tile_latent_min_width * spatial_upsample
    blend_t = int(real_t * temporal_tile_overlap_factor)
    blend_h = int(real_h * spatial_tile_overlap_factor)
    blend_w = int(real_w * spatial_tile_overlap_factor)
    keep_t = real_t - blend_t
    keep_h = real_h - blend_h
    keep_w = real_w - blend_w

    tiles_t = (z.shape[2] + stride_t - 1) // stride_t
    tiles_h = (z.shape[3] + stride_h - 1) // stride_h
    tiles_w = (z.shape[4] + stride_w - 1) // stride_w
    loop_size = [tiles_t, tiles_h, tiles_w]
    total_tiles = tiles_t * tiles_h * tiles_w

    latent_tiles: List[torch.Tensor] = []
    tile_numels: List[int] = []
    for tile_idx in range(total_tiles):
        t_idx, h_idx, w_idx = _index_undot(tile_idx, loop_size)
        t0 = t_idx * stride_t
        h0 = h_idx * stride_h
        w0 = w_idx * stride_w
        tile = z[
            :,
            :,
            t0 : t0 + tile_latent_min_length,
            h0 : h0 + tile_latent_min_height,
            w0 : w0 + tile_latent_min_width,
        ]
        latent_tiles.append(tile)
        tile_numels.append(int(tile.numel()))

    local_indices, global_order = _split_tiles(tile_numels, group)
    decoded = [
        _decode_tile(latent_tiles[idx], spatial_upsample, temporal_upsample)
        for idx in local_indices
    ]
    template = decoded[0] if decoded else _decode_tile(latent_tiles[0], spatial_upsample, temporal_upsample)
    decoded_all = _gather_tiles(decoded, global_order, template, group)

    blended: List[torch.Tensor] = []
    for tile_idx in local_indices:
        t_idx, h_idx, w_idx = _index_undot(tile_idx, loop_size)
        tile = decoded_all[tile_idx].clone()
        if t_idx > 0:
            prev_idx = _index_dot([t_idx - 1, h_idx, w_idx], loop_size)
            tile = _blend_t(decoded_all[prev_idx], tile, blend_t)
        if h_idx > 0:
            prev_idx = _index_dot([t_idx, h_idx - 1, w_idx], loop_size)
            tile = _blend_h(decoded_all[prev_idx], tile, blend_h)
        if w_idx > 0:
            prev_idx = _index_dot([t_idx, h_idx, w_idx - 1], loop_size)
            tile = _blend_w(decoded_all[prev_idx], tile, blend_w)
        blended.append(tile[:, :, :keep_t, :keep_h, :keep_w].contiguous())

    blended_all = _gather_tiles(blended, global_order, template, group)
    frames_t: List[torch.Tensor] = []
    for t_idx in range(tiles_t):
        rows: List[torch.Tensor] = []
        for h_idx in range(tiles_h):
            row: List[torch.Tensor] = []
            for w_idx in range(tiles_w):
                row.append(blended_all[_index_dot([t_idx, h_idx, w_idx], loop_size)])
            rows.append(torch.cat(row, dim=4))
        frames_t.append(torch.cat(rows, dim=3))
    return torch.cat(frames_t, dim=2)
