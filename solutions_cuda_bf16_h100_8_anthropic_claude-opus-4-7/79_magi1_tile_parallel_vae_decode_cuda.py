"""
MAGI-1 tile-parallel VAE decode using symmetric memory + custom CUDA kernels.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------- Trilinear upsample: fp(any) input -> bf16 output ----------------
// Input  layout: [B, C, T, H, W] contiguous fp32 (we cast on host before)
// Output layout: [B, 3, T*tu, H*su, W*su] bf16, written into symmetric buffer slot
// Channel 0..min(C,3)-1 from input; if C<3 we replicate channel 0 (matches reference's repeat-then-take-first-3 for C==1; for C==2 reference repeats then takes first 3 -> ch0,ch1,ch0; we approximate by repeat pattern).

extern "C" __global__ void trilinear_decode_kernel(
    const float* __restrict__ inp,    // [B, C, T, H, W]
    __nv_bfloat16* __restrict__ out,  // [B, 3, T*tu, H*su, W*su]
    int B, int C, int T, int H, int W,
    int tu, int su,
    int outT, int outH, int outW
) {
    long long total = (long long)B * 3 * outT * outH * outW;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        long long w_o = idx % outW;
        long long t1 = idx / outW;
        long long h_o = t1 % outH;
        long long t2 = t1 / outH;
        long long t_o = t2 % outT;
        long long t3 = t2 / outT;
        long long c_o = t3 % 3;
        long long b   = t3 / 3;

        // Map to source channel (mimic reference: repeat then take first 3)
        long long c_src = (C >= 3) ? c_o : (c_o % C);

        // align_corners=False mapping
        auto map = [] __device__ (long long o, int out_size, int in_size) {
            float scale = (float)in_size / (float)out_size;
            float x = (((float)o + 0.5f) * scale) - 0.5f;
            return x;
        };
        float ft = map(t_o, outT, T);
        float fh = map(h_o, outH, H);
        float fw = map(w_o, outW, W);

        int t0 = floorf(ft); int t1i = t0 + 1;
        int h0 = floorf(fh); int h1i = h0 + 1;
        int w0 = floorf(fw); int w1i = w0 + 1;
        float dt = ft - (float)t0;
        float dh = fh - (float)h0;
        float dw = fw - (float)w0;
        int t0c = max(0, min(T-1, t0));
        int t1c = max(0, min(T-1, t1i));
        int h0c = max(0, min(H-1, h0));
        int h1c = max(0, min(H-1, h1i));
        int w0c = max(0, min(W-1, w0));
        int w1c = max(0, min(W-1, w1i));

        long long base = (((b * C) + c_src) * T) * H * W;
        #define G(tt,hh,ww) inp[base + ((long long)(tt)*H + (hh))*W + (ww)]
        float c000 = G(t0c,h0c,w0c);
        float c001 = G(t0c,h0c,w1c);
        float c010 = G(t0c,h1c,w0c);
        float c011 = G(t0c,h1c,w1c);
        float c100 = G(t1c,h0c,w0c);
        float c101 = G(t1c,h0c,w1c);
        float c110 = G(t1c,h1c,w0c);
        float c111 = G(t1c,h1c,w1c);
        #undef G
        float c00 = c000*(1-dw) + c001*dw;
        float c01 = c010*(1-dw) + c011*dw;
        float c10 = c100*(1-dw) + c101*dw;
        float c11 = c110*(1-dw) + c111*dw;
        float c0 = c00*(1-dh) + c01*dh;
        float c1 = c10*(1-dh) + c11*dh;
        float v  = c0*(1-dt) + c1*dt;
        out[idx] = __float2bfloat16(v);
    }
}

// ---------------- Blend + crop kernel ----------------
// Reads decoded tile from symm buffer (own slot), reads up to 3 neighbor tiles
// from peer UVA pointers, blends along T/H/W boundaries, writes cropped tile.
extern "C" __global__ void blend_crop_kernel(
    const __nv_bfloat16* __restrict__ cur_tile,   // [B,3,FT,FH,FW]
    const __nv_bfloat16* __restrict__ prev_t,     // may be null
    const __nv_bfloat16* __restrict__ prev_h,
    const __nv_bfloat16* __restrict__ prev_w,
    __nv_bfloat16* __restrict__ out,              // [B,3,KT,KH,KW]
    int B, int FT, int FH, int FW,
    int KT, int KH, int KW,
    int blend_t, int blend_h, int blend_w
) {
    long long total = (long long)B * 3 * KT * KH * KW;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        long long wo = idx % KW;
        long long t1 = idx / KW;
        long long ho = t1 % KH;
        long long t2 = t1 / KH;
        long long to = t2 % KT;
        long long t3 = t2 / KT;
        long long c  = t3 % 3;
        long long b  = t3 / 3;

        long long full_idx = ((((b*3)+c)*FT + to)*FH + ho)*FW + wo;
        float v = __bfloat162float(cur_tile[full_idx]);

        // T blend: positions [0, blend_t)
        if (prev_t != nullptr && to < (long long)blend_t) {
            float ratio = (float)to / (float)blend_t;
            long long pidx = ((((b*3)+c)*FT + (FT - blend_t + to))*FH + ho)*FW + wo;
            float pv = __bfloat162float(prev_t[pidx]);
            v = pv * (1.0f - ratio) + v * ratio;
        }
        // H blend
        if (prev_h != nullptr && ho < (long long)blend_h) {
            float ratio = (float)ho / (float)blend_h;
            long long pidx = ((((b*3)+c)*FT + to)*FH + (FH - blend_h + ho))*FW + wo;
            float pv = __bfloat162float(prev_h[pidx]);
            // After T blend we should re-read v? Reference applies sequentially, overwriting cur.
            v = pv * (1.0f - ratio) + v * ratio;
        }
        // W blend
        if (prev_w != nullptr && wo < (long long)blend_w) {
            float ratio = (float)wo / (float)blend_w;
            long long pidx = ((((b*3)+c)*FT + to)*FH + ho)*FW + (FW - blend_w + wo);
            float pv = __bfloat162float(prev_w[pidx]);
            v = pv * (1.0f - ratio) + v * ratio;
        }
        out[idx] = __float2bfloat16(v);
    }
}

// ---------------- Assemble: copy cropped tile into output video ----------------
extern "C" __global__ void assemble_kernel(
    const __nv_bfloat16* __restrict__ tile,  // [B,3,KT,KH,KW]
    __nv_bfloat16* __restrict__ video,       // [B,3,VT,VH,VW]
    int B, int KT, int KH, int KW,
    int VT, int VH, int VW,
    int off_t, int off_h, int off_w,
    int copy_t, int copy_h, int copy_w
) {
    long long total = (long long)B * 3 * copy_t * copy_h * copy_w;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        long long wo = idx % copy_w;
        long long t1 = idx / copy_w;
        long long ho = t1 % copy_h;
        long long t2 = t1 / copy_h;
        long long to = t2 % copy_t;
        long long t3 = t2 / copy_t;
        long long c  = t3 % 3;
        long long b  = t3 / 3;
        long long sidx = ((((b*3)+c)*KT + to)*KH + ho)*KW + wo;
        long long didx = ((((b*3)+c)*VT + (off_t+to))*VH + (off_h+ho))*VW + (off_w+wo);
        video[didx] = tile[sidx];
    }
}

void launch_trilinear_decode(
    torch::Tensor inp_f32, int64_t out_ptr,
    int B, int C, int T, int H, int W,
    int tu, int su, int outT, int outH, int outW
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    long long total = (long long)B*3*outT*outH*outW;
    int threads = 256;
    int blocks = (int)min((long long)65535, (total + threads - 1)/threads);
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>((uintptr_t)out_ptr);
    trilinear_decode_kernel<<<blocks, threads, 0, stream>>>(
        inp_f32.data_ptr<float>(), out, B, C, T, H, W, tu, su, outT, outH, outW);
}

void launch_blend_crop(
    int64_t cur_ptr, int64_t prev_t_ptr, int64_t prev_h_ptr, int64_t prev_w_ptr,
    int64_t out_ptr,
    int B, int FT, int FH, int FW, int KT, int KH, int KW,
    int blend_t, int blend_h, int blend_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    long long total = (long long)B*3*KT*KH*KW;
    int threads = 256;
    int blocks = (int)min((long long)65535, (total + threads - 1)/threads);
    blend_crop_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>((uintptr_t)cur_ptr),
        reinterpret_cast<const __nv_bfloat16*>((uintptr_t)prev_t_ptr),
        reinterpret_cast<const __nv_bfloat16*>((uintptr_t)prev_h_ptr),
        reinterpret_cast<const __nv_bfloat16*>((uintptr_t)prev_w_ptr),
        reinterpret_cast<__nv_bfloat16*>((uintptr_t)out_ptr),
        B, FT, FH, FW, KT, KH, KW, blend_t, blend_h, blend_w);
}

void launch_assemble(
    int64_t tile_ptr, torch::Tensor video,
    int B, int KT, int KH, int KW, int VT, int VH, int VW,
    int off_t, int off_h, int off_w, int copy_t, int copy_h, int copy_w
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    long long total = (long long)B*3*copy_t*copy_h*copy_w;
    int threads = 256;
    int blocks = (int)min((long long)65535, (total + threads - 1)/threads);
    assemble_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>((uintptr_t)tile_ptr),
        reinterpret_cast<__nv_bfloat16*>(video.data_ptr<at::BFloat16>()),
        B, KT, KH, KW, VT, VH, VW, off_t, off_h, off_w, copy_t, copy_h, copy_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trilinear_decode", &launch_trilinear_decode, "Trilinear decode -> bf16");
    m.def("blend_crop", &launch_blend_crop, "Blend + crop bf16 tile");
    m.def("assemble", &launch_assemble, "Assemble cropped tile into output video");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("magi1_tile_vae_ext", CUDA_SRC)
    return _ext


def _index_undot(index, loop_size):
    out = []
    for size in reversed(loop_size):
        out.append(index % size)
        index //= size
    return list(reversed(out))


def _index_dot(index, loop_size):
    value = 0
    for d, s in zip(index, loop_size):
        value = value * s + d
    return value


# Symmetric buffer cache
_buf_cache = {}

def _get_symm(key, shape, dtype, device, group):
    if key in _buf_cache:
        b, h = _buf_cache[key]
        if tuple(b.shape) == tuple(shape) and b.dtype == dtype:
            return b, h
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _buf_cache[key] = (buf, hdl)
    return buf, hdl


def _split_tiles_rr(tile_numels, world_size, rank):
    if world_size == 1:
        idxs = list(range(len(tile_numels)))
        return idxs, idxs, [idxs]
    sorted_tiles = sorted(range(len(tile_numels)), key=lambda i: tile_numels[i], reverse=True)
    per_rank = [sorted_tiles[r::world_size] for r in range(world_size)]
    global_order = [idx for shard in per_rank for idx in shard]
    return per_rank[rank], global_order, per_rank


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
    else:
        group = None
        world_size = 1
        rank = 0

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
    C = z.shape[1]
    device = z.device

    # Compute each tile's actual latent shape and decoded shape (variable!)
    tile_specs = []  # list of (t0, h0, w0, lt, lh, lw, ft, fh, fw)
    tile_numels = []
    for tile_idx in range(total_tiles):
        ti, hi, wi = _index_undot(tile_idx, loop_size)
        t0 = ti * stride_t; h0 = hi * stride_h; w0 = wi * stride_w
        lt = min(tile_latent_min_length, z.shape[2] - t0)
        lh = min(tile_latent_min_height, z.shape[3] - h0)
        lw = min(tile_latent_min_width,  z.shape[4] - w0)
        ft = lt * temporal_upsample
        fh = lh * spatial_upsample
        fw = lw * spatial_upsample
        tile_specs.append((t0, h0, w0, lt, lh, lw, ft, fh, fw))
        tile_numels.append(B * C * lt * lh * lw)

    local_indices, global_order, per_rank_shards = _split_tiles_rr(tile_numels, world_size, rank)

    # Determine owner rank for each tile
    owner = [0] * total_tiles
    for r, shard in enumerate(per_rank_shards):
        for idx in shard:
            owner[idx] = r

    # We need symmetric buffers holding decoded (full) tile and blended-cropped tile.
    # Tiles have variable size so allocate one big symm buffer per rank with offsets.
    # Layout: each tile occupies element_count = B*3*ft*fh*fw bf16 elements in *its owner*'s slot.
    # We need every rank to know offsets/sizes of every tile -> precompute deterministically.
    full_sizes = [B * 3 * ft * fh * fw for (_,_,_,_,_,_,ft,fh,fw) in tile_specs]
    crop_sizes = []
    for ti in range(tiles_t):
        for hi in range(tiles_h):
            for wi in range(tiles_w):
                idx = _index_dot([ti, hi, wi], loop_size)
                _,_,_,_,_,_,ft,fh,fw = tile_specs[idx]
                kt = min(keep_t, ft); kh = min(keep_h, fh); kw = min(keep_w, fw)
                crop_sizes.append(B * 3 * kt * kh * kw)

    # For each rank compute total full bytes and offsets within its slot
    rank_full_total = [0] * world_size
    rank_crop_total = [0] * world_size
    full_offset = [0] * total_tiles  # offset within owner's slot (bf16 elements)
    crop_offset = [0] * total_tiles
    rank_full_offsets = [[] for _ in range(world_size)]  # not needed, we compute inline
    for r in range(world_size):
        off_f = 0
        off_c = 0
        for idx in per_rank_shards[r]:
            full_offset[idx] = off_f
            crop_offset[idx] = off_c
            off_f += full_sizes[idx]
            off_c += crop_sizes[idx]
        rank_full_total[r] = off_f
        rank_crop_total[r] = off_c

    max_full = max(rank_full_total) if rank_full_total else 1
    max_crop = max(rank_crop_total) if rank_crop_total else 1
    max_full = max(max_full, 1)
    max_crop = max(max_crop, 1)

    if group is not None:
        full_buf, full_hdl = _get_symm(("full", max_full), (max_full,), torch.bfloat16, device, group)
        crop_buf, crop_hdl = _get_symm(("crop", max_crop), (max_crop,), torch.bfloat16, device, group)
    else:
        full_buf = torch.empty(max_full, dtype=torch.bfloat16, device=device)
        crop_buf = torch.empty(max_crop, dtype=torch.bfloat16, device=device)
        full_hdl = None
        crop_hdl = None

    ext = _get_ext()

    # Phase 1: decode local tiles into full_buf
    for idx in local_indices:
        t0, h0, w0, lt, lh, lw, ft, fh, fw = tile_specs[idx]
        latent = z[:, :, t0:t0+lt, h0:h0+lh, w0:w0+lw].contiguous().float()
        # Pointer into our full_buf at offset
        out_ptr = int(full_buf.data_ptr()) + full_offset[idx] * full_buf.element_size()
        ext.trilinear_decode(latent, out_ptr, B, C, lt, lh, lw,
                             temporal_upsample, spatial_upsample, ft, fh, fw)

    # Sync: everyone has decoded their tiles in full_buf
    if full_hdl is not None:
        full_hdl.barrier(channel=0)

    # Helper to get pointer to a tile's full data on its owner's symm slot
    def full_tile_ptr(tile_idx):
        own = owner[tile_idx]
        if full_hdl is not None and own != rank:
            base = int(full_hdl.buffer_ptrs[own])
        else:
            base = int(full_buf.data_ptr())
        return base + full_offset[tile_idx] * 2  # bf16 = 2 bytes

    def crop_tile_ptr(tile_idx):
        own = owner[tile_idx]
        if crop_hdl is not None and own != rank:
            base = int(crop_hdl.buffer_ptrs[own])
        else:
            base = int(crop_buf.data_ptr())
        return base + crop_offset[tile_idx] * 2

    # Phase 2: blend + crop local tiles, writing into crop_buf
    for idx in local_indices:
        ti, hi, wi = _index_undot(idx, loop_size)
        _,_,_,_,_,_,ft,fh,fw = tile_specs[idx]
        kt = min(keep_t, ft); kh = min(keep_h, fh); kw = min(keep_w, fw)

        cur_ptr = full_tile_ptr(idx)
        pt_ptr = 0; ph_ptr = 0; pw_ptr = 0
        bt_use = 0; bh_use = 0; bw_use = 0
        if ti > 0:
            pidx = _index_dot([ti-1, hi, wi], loop_size)
            _,_,_,_,_,_,pft,_,_ = tile_specs[pidx]
            bt_use = min(blend_t, ft, pft)
            if bt_use > 0:
                pt_ptr = full_tile_ptr(pidx)
        if hi > 0:
            pidx = _index_dot([ti, hi-1, wi], loop_size)
            _,_,_,_,_,_,_,pfh,_ = tile_specs[pidx]
            bh_use = min(blend_h, fh, pfh)
            if bh_use > 0:
                ph_ptr = full_tile_ptr(pidx)
        if wi > 0:
            pidx = _index_dot([ti, hi, wi-1], loop_size)
            _,_,_,_,_,_,_,_,pfw = tile_specs[pidx]
            bw_use = min(blend_w, fw, pfw)
            if bw_use > 0:
                pw_ptr = full_tile_ptr(pidx)

        out_ptr = int(crop_buf.data_ptr()) + crop_offset[idx] * 2
        ext.blend_crop(cur_ptr, pt_ptr, ph_ptr, pw_ptr, out_ptr,
                       B, ft, fh, fw, kt, kh, kw, bt_use, bh_use, bw_use)

    if crop_hdl is not None:
        crop_hdl.barrier(channel=0)

    # Phase 3: assemble final output video on every rank by reading cropped tiles
    # via UVA. Compute per-tile output offsets.
    out_t = sum(min(keep_t, tile_specs[_index_dot([ti,0,0], loop_size)][6]) for ti in range(tiles_t))
    out_h = sum(min(keep_h, tile_specs[_index_dot([0,hi,0], loop_size)][7]) for hi in range(tiles_h))
    out_w = sum(min(keep_w, tile_specs[_index_dot([0,0,wi], loop_size)][8]) for wi in range(tiles_w))

    video = torch.empty((B, 3, out_t, out_h, out_w), dtype=torch.bfloat16, device=device)

    # Compute axis offsets
    t_offsets = []
    acc = 0
    for ti in range(tiles_t):
        t_offsets.append(acc)
        ft = tile_specs[_index_dot([ti,0,0], loop_size)][6]
        acc += min(keep_t, ft)
    h_offsets = []
    acc = 0
    for hi in range(tiles_h):
        h_offsets.append(acc)
        fh = tile_specs[_index_dot([0,hi,0], loop_size)][7]
        acc += min(keep_h, fh)
    w_offsets = []
    acc = 0
    for wi in range(tiles_w):
        w_offsets.append(acc)
        fw = tile_specs[_index_dot([0,0,wi], loop_size)][8]
        acc += min(keep_w, fw)

    for ti in range(tiles_t):
        for hi in range(tiles_h):
            for wi in range(tiles_w):
                idx = _index_dot([ti, hi, wi], loop_size)
                _,_,_,_,_,_,ft,fh,fw = tile_specs[idx]
                kt = min(keep_t, ft); kh = min(keep_h, fh); kw = min(keep_w, fw)
                tile_ptr = crop_tile_ptr(idx)
                ext.assemble(tile_ptr, video, B, kt, kh, kw,
                             out_t, out_h, out_w,
                             t_offsets[ti], h_offsets[hi], w_offsets[wi],
                             kt, kh, kw)

    return video