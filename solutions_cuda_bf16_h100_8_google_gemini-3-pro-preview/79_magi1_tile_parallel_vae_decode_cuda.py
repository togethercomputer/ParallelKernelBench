import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from typing import List, Optional, Tuple

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void pull_decoded_kernel(
    __nv_bfloat16* __restrict__ local_flat,
    const int64_t* __restrict__ peer_ptrs,
    const int* __restrict__ tile_owners,
    const int64_t* __restrict__ tile_offsets,
    const int64_t* __restrict__ tile_numels,
    int total_tiles,
    int rank
) {
    int threads = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
        int owner = tile_owners[tile_idx];
        if (owner == rank) continue;
        
        int64_t offset = tile_offsets[tile_idx];
        int64_t numel = tile_numels[tile_idx];
        const __nv_bfloat16* src = (const __nv_bfloat16*)peer_ptrs[owner];
        
        for (int64_t i = tid; i < numel; i += threads) {
            local_flat[offset + i] = src[offset + i];
        }
    }
}

__global__ void blend_and_assemble_kernel(
    const __nv_bfloat16* __restrict__ decoded_flat,
    __nv_bfloat16* __restrict__ final_output,
    const int64_t* __restrict__ meta,
    int total_tiles,
    int B, int Total_T, int Total_H, int Total_W
) {
    int threads = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++) {
        int64_t dec_t = meta[tile_idx * 19 + 3];
        int64_t dec_h = meta[tile_idx * 19 + 4];
        int64_t dec_w = meta[tile_idx * 19 + 5];
        int64_t kept_t = meta[tile_idx * 19 + 6];
        int64_t kept_h = meta[tile_idx * 19 + 7];
        int64_t kept_w = meta[tile_idx * 19 + 8];
        int64_t out_t = meta[tile_idx * 19 + 9];
        int64_t out_h = meta[tile_idx * 19 + 10];
        int64_t out_w = meta[tile_idx * 19 + 11];
        int64_t extent_t = meta[tile_idx * 19 + 12];
        int64_t extent_h = meta[tile_idx * 19 + 13];
        int64_t extent_w = meta[tile_idx * 19 + 14];
        int64_t prev_t = meta[tile_idx * 19 + 15];
        int64_t prev_h = meta[tile_idx * 19 + 16];
        int64_t prev_w = meta[tile_idx * 19 + 17];
        int64_t flat_offset = meta[tile_idx * 19 + 18];
        
        int64_t num_kept = (int64_t)B * 3 * kept_t * kept_h * kept_w;
        
        for (int64_t i = tid; i < num_kept; i += threads) {
            int64_t rem = i;
            int64_t w = rem % kept_w; rem /= kept_w;
            int64_t h = rem % kept_h; rem /= kept_h;
            int64_t t = rem % kept_t; rem /= kept_t;
            int64_t c = rem % 3; rem /= 3;
            int64_t b = rem;
            
            int64_t local_idx = (((b * 3 + c) * dec_t + t) * dec_h + h) * dec_w + w;
            float val = __bfloat162float(decoded_flat[flat_offset + local_idx]);
            
            if (prev_t != -1 && t < extent_t) {
                int64_t p_offset = meta[prev_t * 19 + 18];
                int64_t p_dec_t = meta[prev_t * 19 + 3];
                int64_t p_dec_h = meta[prev_t * 19 + 4];
                int64_t p_dec_w = meta[prev_t * 19 + 5];
                
                int64_t p_t = p_dec_t - extent_t + t;
                int64_t p_idx = (((b * 3 + c) * p_dec_t + p_t) * p_dec_h + h) * p_dec_w + w;
                float p_val = __bfloat162float(decoded_flat[p_offset + p_idx]);
                float ratio = (float)t / (float)extent_t;
                val = p_val * (1.0f - ratio) + val * ratio;
            }
            
            if (prev_h != -1 && h < extent_h) {
                int64_t p_offset = meta[prev_h * 19 + 18];
                int64_t p_dec_t = meta[prev_h * 19 + 3];
                int64_t p_dec_h = meta[prev_h * 19 + 4];
                int64_t p_dec_w = meta[prev_h * 19 + 5];
                
                int64_t p_h = p_dec_h - extent_h + h;
                int64_t p_idx = (((b * 3 + c) * p_dec_t + t) * p_dec_h + p_h) * p_dec_w + w;
                float p_val = __bfloat162float(decoded_flat[p_offset + p_idx]);
                float ratio = (float)h / (float)extent_h;
                val = p_val * (1.0f - ratio) + val * ratio;
            }
            
            if (prev_w != -1 && w < extent_w) {
                int64_t p_offset = meta[prev_w * 19 + 18];
                int64_t p_dec_t = meta[prev_w * 19 + 3];
                int64_t p_dec_h = meta[prev_w * 19 + 4];
                int64_t p_dec_w = meta[prev_w * 19 + 5];
                
                int64_t p_w = p_dec_w - extent_w + w;
                int64_t p_idx = (((b * 3 + c) * p_dec_t + t) * p_dec_h + h) * p_dec_w + p_w;
                float p_val = __bfloat162float(decoded_flat[p_offset + p_idx]);
                float ratio = (float)w / (float)extent_w;
                val = p_val * (1.0f - ratio) + val * ratio;
            }
            
            int64_t O_t = out_t + t;
            int64_t O_h = out_h + h;
            int64_t O_w = out_w + w;
            int64_t out_idx = (((b * 3 + c) * Total_T + O_t) * Total_H + O_h) * Total_W + O_w;
            final_output[out_idx] = __float2bfloat16(val);
        }
    }
}

void launch_pull_decoded(
    torch::Tensor local_flat,
    torch::Tensor peer_ptrs,
    torch::Tensor tile_owners,
    torch::Tensor tile_offsets,
    torch::Tensor tile_numels,
    int total_tiles,
    int rank
) {
    int threads = 256;
    int blocks = 2048;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    pull_decoded_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)local_flat.data_ptr<at::BFloat16>(),
        peer_ptrs.data_ptr<int64_t>(),
        tile_owners.data_ptr<int>(),
        tile_offsets.data_ptr<int64_t>(),
        tile_numels.data_ptr<int64_t>(),
        total_tiles,
        rank
    );
}

void launch_blend_and_assemble(
    torch::Tensor decoded_flat,
    torch::Tensor final_output,
    torch::Tensor meta,
    int total_tiles,
    int B, int Total_T, int Total_H, int Total_W
) {
    int threads = 256;
    int blocks = 2048;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    blend_and_assemble_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)decoded_flat.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)final_output.data_ptr<at::BFloat16>(),
        meta.data_ptr<int64_t>(),
        total_tiles,
        B, Total_T, Total_H, Total_W
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pull_decoded", &launch_pull_decoded);
    m.def("launch_blend_and_assemble", &launch_blend_and_assemble);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("magi1_vae_decode_ext", CUDA_SRC)
    return _ext


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
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        if rank == 0:
            _get_ext()
        dist.barrier(group=group)
        _get_ext()
    else:
        group = None
        world_size = 1
        rank = 0
        _get_ext()

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
    B = z.shape[0]

    # Precompute tile shapes, layout, and bounds dynamically
    latent_tiles_shapes = []
    for tile_idx in range(total_tiles):
        t_idx, h_idx, w_idx = _index_undot(tile_idx, loop_size)
        t0 = t_idx * stride_t
        h0 = h_idx * stride_h
        w0 = w_idx * stride_w
        len_t = min(z.shape[2] - t0, tile_latent_min_length)
        len_h = min(z.shape[3] - h0, tile_latent_min_height)
        len_w = min(z.shape[4] - w0, tile_latent_min_width)
        latent_tiles_shapes.append((len_t, len_h, len_w))

    decoded_shapes = []
    kept_shapes = []
    flat_offsets = [0] * total_tiles
    cur_offset = 0
    for tile_idx, (t, h, w) in enumerate(latent_tiles_shapes):
        dec_t = t * temporal_upsample
        dec_h = h * spatial_upsample
        dec_w = w * spatial_upsample
        decoded_shapes.append((dec_t, dec_h, dec_w))
        
        kept_t = min(dec_t, keep_t)
        kept_h = min(dec_h, keep_h)
        kept_w = min(dec_w, keep_w)
        kept_shapes.append((kept_t, kept_h, kept_w))
        
        flat_offsets[tile_idx] = cur_offset
        cur_offset += B * 3 * dec_t * dec_h * dec_w

    total_decoded_numel = cur_offset
    Total_T = sum(kept_shapes[_index_dot([i, 0, 0], loop_size)][0] for i in range(tiles_t))
    Total_H = sum(kept_shapes[_index_dot([0, i, 0], loop_size)][1] for i in range(tiles_h))
    Total_W = sum(kept_shapes[_index_dot([0, 0, i], loop_size)][2] for i in range(tiles_w))

    out_t_offsets = [0] * tiles_t
    cur = 0
    for i in range(tiles_t):
        out_t_offsets[i] = cur
        cur += kept_shapes[_index_dot([i, 0, 0], loop_size)][0]

    out_h_offsets = [0] * tiles_h
    cur = 0
    for i in range(tiles_h):
        out_h_offsets[i] = cur
        cur += kept_shapes[_index_dot([0, i, 0], loop_size)][1]

    out_w_offsets = [0] * tiles_w
    cur = 0
    for i in range(tiles_w):
        out_w_offsets[i] = cur
        cur += kept_shapes[_index_dot([0, 0, i], loop_size)][2]

    # Map rank assignments to tiles
    latent_numels = [B * z.shape[1] * s[0] * s[1] * s[2] for s in latent_tiles_shapes]
    local_indices, _ = _split_tiles(latent_numels, group)
    tile_owners = [0] * total_tiles
    if world_size > 1:
        per_rank, _ = _split_tiles(latent_numels, None)
        sorted_tiles = sorted(range(len(latent_numels)), key=lambda idx: latent_numels[idx], reverse=True)
        assigned = [sorted_tiles[r::world_size] for r in range(world_size)]
        for r in range(world_size):
            for idx in assigned[r]:
                tile_owners[idx] = r

    # Render meta array for CUDA
    meta_list = []
    for tile_idx in range(total_tiles):
        t_idx, h_idx, w_idx = _index_undot(tile_idx, loop_size)
        dec_t, dec_h, dec_w = decoded_shapes[tile_idx]
        kept_t, kept_h, kept_w = kept_shapes[tile_idx]
        
        extent_t = min(decoded_shapes[_index_dot([t_idx - 1, h_idx, w_idx], loop_size)][0], dec_t, blend_t) if t_idx > 0 else 0
        extent_h = min(decoded_shapes[_index_dot([t_idx, h_idx - 1, w_idx], loop_size)][1], dec_h, blend_h) if h_idx > 0 else 0
        extent_w = min(decoded_shapes[_index_dot([t_idx, h_idx, w_idx - 1], loop_size)][2], dec_w, blend_w) if w_idx > 0 else 0
        
        prev_t = _index_dot([t_idx - 1, h_idx, w_idx], loop_size) if t_idx > 0 else -1
        prev_h = _index_dot([t_idx, h_idx - 1, w_idx], loop_size) if h_idx > 0 else -1
        prev_w = _index_dot([t_idx, h_idx, w_idx - 1], loop_size) if w_idx > 0 else -1

        meta_list.append([
            t_idx, h_idx, w_idx, 
            dec_t, dec_h, dec_w, 
            kept_t, kept_h, kept_w, 
            out_t_offsets[t_idx], out_h_offsets[h_idx], out_w_offsets[w_idx], 
            extent_t, extent_h, extent_w, 
            prev_t, prev_h, prev_w, 
            flat_offsets[tile_idx]
        ])
    meta = torch.tensor(meta_list, dtype=torch.int64, device=z.device)

    # Establish memory structures
    hdl = None
    if world_size > 1:
        decoded_all_flat = symm_mem.empty(total_decoded_numel, dtype=torch.bfloat16, device=z.device)
        hdl = symm_mem.rendezvous(decoded_all_flat, group=group)
    else:
        decoded_all_flat = torch.empty(total_decoded_numel, dtype=torch.bfloat16, device=z.device)

    # Decode and fill owned tiles locally
    for tile_idx in local_indices:
        t_idx, h_idx, w_idx = _index_undot(tile_idx, loop_size)
        t0 = t_idx * stride_t
        h0 = h_idx * stride_h
        w0 = w_idx * stride_w
        tile = z[:, :, t0:t0+tile_latent_min_length, h0:h0+tile_latent_min_height, w0:w0+tile_latent_min_width]
        
        dec = _decode_tile(tile, spatial_upsample, temporal_upsample).contiguous()
        offset = flat_offsets[tile_idx]
        decoded_all_flat[offset:offset+dec.numel()].copy_(dec.view(-1))

    # Pull remaining non-owned tiles over NVLink
    if world_size > 1:
        hdl.barrier(channel=0)
        peer_ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=z.device)
        owners_tensor = torch.tensor(tile_owners, dtype=torch.int32, device=z.device)
        offsets_tensor = torch.tensor(flat_offsets, dtype=torch.int64, device=z.device)
        decoded_numels = [B * 3 * s[0] * s[1] * s[2] for s in decoded_shapes]
        numels_tensor = torch.tensor(decoded_numels, dtype=torch.int64, device=z.device)
        
        _get_ext().launch_pull_decoded(
            decoded_all_flat, peer_ptrs, owners_tensor, offsets_tensor, numels_tensor, total_tiles, rank
        )
        hdl.barrier(channel=0)

    # Blend and assemble all final chunks completely fusing operations
    final_output = torch.empty((B, 3, Total_T, Total_H, Total_W), dtype=torch.bfloat16, device=z.device)
    _get_ext().launch_blend_and_assemble(
        decoded_all_flat, final_output, meta, total_tiles, B, Total_T, Total_H, Total_W
    )

    if hdl is not None:
        hdl.barrier(channel=0)

    return final_output