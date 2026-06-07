from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#define META_STRIDE 11
#define META_T0      0
#define META_H0      1
#define META_W0      2
#define META_IN_T    3
#define META_IN_H    4
#define META_IN_W    5
#define META_OUT_T   6
#define META_OUT_H   7
#define META_OUT_W   8
#define META_OFFSET  9
#define META_OWNER   10

__device__ __forceinline__ float read_bf16(const __nv_bfloat16* p, int64_t idx) {
    return __bfloat162float(p[idx]);
}

__device__ __forceinline__ float read_f32(const float* p, int64_t idx) {
    return p[idx];
}

__device__ __forceinline__ float bf16_round_to_float(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ int64_t find_segment(
    const int64_t* __restrict__ prefix,
    int64_t nseg,
    int64_t x
) {
    int64_t lo = 0;
    int64_t hi = nseg;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (prefix[mid + 1] <= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <typename InPtr>
__device__ __forceinline__ float trilinear_sample(
    const InPtr __restrict__ z,
    int64_t B,
    int64_t C,
    int64_t T,
    int64_t H,
    int64_t W,
    int64_t b,
    int64_t c,
    int64_t t0,
    int64_t h0,
    int64_t w0,
    int64_t in_t,
    int64_t in_h,
    int64_t in_w,
    int64_t ot,
    int64_t oh,
    int64_t ow,
    int temporal_up,
    int spatial_up,
    int dtype_enum
) {
    float ft = ((float)ot + 0.5f) / (float)temporal_up - 0.5f;
    float fh = ((float)oh + 0.5f) / (float)spatial_up - 0.5f;
    float fw = ((float)ow + 0.5f) / (float)spatial_up - 0.5f;

    ft = ft < 0.0f ? 0.0f : ft;
    fh = fh < 0.0f ? 0.0f : fh;
    fw = fw < 0.0f ? 0.0f : fw;

    int64_t lt0 = (int64_t)floorf(ft);
    int64_t lh0 = (int64_t)floorf(fh);
    int64_t lw0 = (int64_t)floorf(fw);

    float wt = ft - (float)lt0;
    float wh = fh - (float)lh0;
    float ww = fw - (float)lw0;

    int64_t lt1 = lt0 + 1 < in_t ? lt0 + 1 : lt0;
    int64_t lh1 = lh0 + 1 < in_h ? lh0 + 1 : lh0;
    int64_t lw1 = lw0 + 1 < in_w ? lw0 + 1 : lw0;

    float vt0 = 1.0f - wt;
    float vh0 = 1.0f - wh;
    float vw0 = 1.0f - ww;

    int64_t gt0 = t0 + lt0;
    int64_t gt1 = t0 + lt1;
    int64_t gh0 = h0 + lh0;
    int64_t gh1 = h0 + lh1;
    int64_t gw0 = w0 + lw0;
    int64_t gw1 = w0 + lw1;

    int64_t base000 = (((b * C + c) * T + gt0) * H + gh0) * W;
    int64_t base001 = (((b * C + c) * T + gt0) * H + gh0) * W;
    int64_t base010 = (((b * C + c) * T + gt0) * H + gh1) * W;
    int64_t base011 = (((b * C + c) * T + gt0) * H + gh1) * W;
    int64_t base100 = (((b * C + c) * T + gt1) * H + gh0) * W;
    int64_t base101 = (((b * C + c) * T + gt1) * H + gh0) * W;
    int64_t base110 = (((b * C + c) * T + gt1) * H + gh1) * W;
    int64_t base111 = (((b * C + c) * T + gt1) * H + gh1) * W;

    float v000, v001, v010, v011, v100, v101, v110, v111;
    if (dtype_enum == 0) {
        const __nv_bfloat16* zz = reinterpret_cast<const __nv_bfloat16*>(z);
        v000 = read_bf16(zz, base000 + gw0);
        v001 = read_bf16(zz, base001 + gw1);
        v010 = read_bf16(zz, base010 + gw0);
        v011 = read_bf16(zz, base011 + gw1);
        v100 = read_bf16(zz, base100 + gw0);
        v101 = read_bf16(zz, base101 + gw1);
        v110 = read_bf16(zz, base110 + gw0);
        v111 = read_bf16(zz, base111 + gw1);
    } else {
        const float* zz = reinterpret_cast<const float*>(z);
        v000 = read_f32(zz, base000 + gw0);
        v001 = read_f32(zz, base001 + gw1);
        v010 = read_f32(zz, base010 + gw0);
        v011 = read_f32(zz, base011 + gw1);
        v100 = read_f32(zz, base100 + gw0);
        v101 = read_f32(zz, base101 + gw1);
        v110 = read_f32(zz, base110 + gw0);
        v111 = read_f32(zz, base111 + gw1);
    }

    float v00 = v000 * vw0 + v001 * ww;
    float v01 = v010 * vw0 + v011 * ww;
    float v10 = v100 * vw0 + v101 * ww;
    float v11 = v110 * vw0 + v111 * ww;
    float v0 = v00 * vh0 + v01 * wh;
    float v1 = v10 * vh0 + v11 * wh;
    return v0 * vt0 + v1 * wt;
}

__global__ void decode_tiles_kernel(
    const void* __restrict__ z,
    __nv_bfloat16* __restrict__ symbuf,
    const int64_t* __restrict__ meta,
    const int64_t* __restrict__ local_ids,
    const int64_t* __restrict__ local_prefix,
    int64_t nlocal,
    int64_t total_work,
    int64_t B,
    int64_t C,
    int64_t T,
    int64_t H,
    int64_t W,
    int spatial_up,
    int temporal_up,
    int dtype_enum
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < total_work; idx += stride) {
        int64_t lj = find_segment(local_prefix, nlocal, idx);
        int64_t tile_idx = local_ids[lj];
        int64_t local = idx - local_prefix[lj];

        const int64_t* m = meta + tile_idx * META_STRIDE;
        int64_t t0 = m[META_T0];
        int64_t h0 = m[META_H0];
        int64_t w0 = m[META_W0];
        int64_t in_t = m[META_IN_T];
        int64_t in_h = m[META_IN_H];
        int64_t in_w = m[META_IN_W];
        int64_t out_t = m[META_OUT_T];
        int64_t out_h = m[META_OUT_H];
        int64_t out_w = m[META_OUT_W];
        int64_t dst_offset = m[META_OFFSET];

        int64_t ow = local % out_w;
        local /= out_w;
        int64_t oh = local % out_h;
        local /= out_h;
        int64_t ot = local % out_t;
        local /= out_t;
        int64_t oc = local % 3;
        int64_t b = local / 3;

        int64_t src_c = oc % C;
        float val = trilinear_sample<const void*>(
            z, B, C, T, H, W, b, src_c, t0, h0, w0,
            in_t, in_h, in_w, ot, oh, ow, temporal_up, spatial_up, dtype_enum);

        symbuf[dst_offset + idx - local_prefix[lj]] = __float2bfloat16(val);
    }
}

__device__ __forceinline__ float load_decoded(
    const int64_t* __restrict__ ptrs,
    const int64_t* __restrict__ meta,
    int64_t tile_idx,
    int64_t b,
    int64_t c,
    int64_t t,
    int64_t h,
    int64_t w
) {
    const int64_t* m = meta + tile_idx * META_STRIDE;
    int owner = (int)m[META_OWNER];
    int64_t off = m[META_OFFSET];
    int64_t out_t = m[META_OUT_T];
    int64_t out_h = m[META_OUT_H];
    int64_t out_w = m[META_OUT_W];

    const __nv_bfloat16* base =
        reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[owner]);

    int64_t idx = (((b * 3 + c) * out_t + t) * out_h + h) * out_w + w;
    return __bfloat162float(base[off + idx]);
}

__global__ void assemble_blend_kernel(
    const int64_t* __restrict__ ptrs,
    const int64_t* __restrict__ meta,
    const int64_t* __restrict__ prefix_t,
    const int64_t* __restrict__ prefix_h,
    const int64_t* __restrict__ prefix_w,
    __nv_bfloat16* __restrict__ out,
    int64_t total_out,
    int64_t B,
    int64_t tiles_t,
    int64_t tiles_h,
    int64_t tiles_w,
    int64_t outT_total,
    int64_t outH_total,
    int64_t outW_total,
    int blend_t,
    int blend_h,
    int blend_w
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = tid; linear < total_out; linear += stride) {
        int64_t x = linear;
        int64_t ow_global = x % outW_total;
        x /= outW_total;
        int64_t oh_global = x % outH_total;
        x /= outH_total;
        int64_t ot_global = x % outT_total;
        x /= outT_total;
        int64_t c = x % 3;
        int64_t b = x / 3;

        int64_t ti = find_segment(prefix_t, tiles_t, ot_global);
        int64_t hi = find_segment(prefix_h, tiles_h, oh_global);
        int64_t wi = find_segment(prefix_w, tiles_w, ow_global);

        int64_t lt = ot_global - prefix_t[ti];
        int64_t lh = oh_global - prefix_h[hi];
        int64_t lw = ow_global - prefix_w[wi];

        int64_t tile_idx = (ti * tiles_h + hi) * tiles_w + wi;
        const int64_t* m = meta + tile_idx * META_STRIDE;
        int64_t cur_out_t = m[META_OUT_T];
        int64_t cur_out_h = m[META_OUT_H];
        int64_t cur_out_w = m[META_OUT_W];

        float val = load_decoded(ptrs, meta, tile_idx, b, c, lt, lh, lw);

        if (ti > 0 && blend_t > 0) {
            int64_t prev_idx = ((ti - 1) * tiles_h + hi) * tiles_w + wi;
            const int64_t* pm = meta + prev_idx * META_STRIDE;
            int64_t ext = blend_t;
            if (pm[META_OUT_T] < ext) ext = pm[META_OUT_T];
            if (cur_out_t < ext) ext = cur_out_t;
            if (lt < ext) {
                float r = (float)lt / (float)ext;
                float pv = load_decoded(
                    ptrs, meta, prev_idx, b, c,
                    pm[META_OUT_T] - ext + lt, lh, lw);
                val = bf16_round_to_float(pv * (1.0f - r) + val * r);
            }
        }

        if (hi > 0 && blend_h > 0) {
            int64_t prev_idx = (ti * tiles_h + (hi - 1)) * tiles_w + wi;
            const int64_t* pm = meta + prev_idx * META_STRIDE;
            int64_t ext = blend_h;
            if (pm[META_OUT_H] < ext) ext = pm[META_OUT_H];
            if (cur_out_h < ext) ext = cur_out_h;
            if (lh < ext) {
                float r = (float)lh / (float)ext;
                float pv = load_decoded(
                    ptrs, meta, prev_idx, b, c,
                    lt, pm[META_OUT_H] - ext + lh, lw);
                val = bf16_round_to_float(pv * (1.0f - r) + val * r);
            }
        }

        if (wi > 0 && blend_w > 0) {
            int64_t prev_idx = (ti * tiles_h + hi) * tiles_w + (wi - 1);
            const int64_t* pm = meta + prev_idx * META_STRIDE;
            int64_t ext = blend_w;
            if (pm[META_OUT_W] < ext) ext = pm[META_OUT_W];
            if (cur_out_w < ext) ext = cur_out_w;
            if (lw < ext) {
                float r = (float)lw / (float)ext;
                float pv = load_decoded(
                    ptrs, meta, prev_idx, b, c,
                    lt, lh, pm[META_OUT_W] - ext + lw);
                val = bf16_round_to_float(pv * (1.0f - r) + val * r);
            }
        }

        out[linear] = __float2bfloat16(val);
    }
}

void launch_decode_tiles(
    torch::Tensor z,
    torch::Tensor symbuf,
    torch::Tensor meta,
    torch::Tensor local_ids,
    torch::Tensor local_prefix,
    int64_t total_work,
    int spatial_up,
    int temporal_up,
    int dtype_enum
) {
    if (total_work <= 0) return;

    TORCH_CHECK(z.is_cuda(), "z must be CUDA");
    TORCH_CHECK(symbuf.is_cuda(), "symbuf must be CUDA");
    TORCH_CHECK(symbuf.dtype() == torch::kBFloat16, "symbuf must be bf16");
    TORCH_CHECK(meta.dtype() == torch::kInt64, "meta must be int64");
    TORCH_CHECK(local_ids.dtype() == torch::kInt64, "local_ids must be int64");
    TORCH_CHECK(local_prefix.dtype() == torch::kInt64, "local_prefix must be int64");

    int64_t B = z.size(0);
    int64_t C = z.size(1);
    int64_t T = z.size(2);
    int64_t H = z.size(3);
    int64_t W = z.size(4);
    int64_t nlocal = local_ids.numel();

    int threads = 256;
    int blocks = (int)((total_work + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    decode_tiles_kernel<<<blocks, threads, 0, stream>>>(
        z.data_ptr(),
        reinterpret_cast<__nv_bfloat16*>(symbuf.data_ptr<at::BFloat16>()),
        meta.data_ptr<int64_t>(),
        local_ids.data_ptr<int64_t>(),
        local_prefix.data_ptr<int64_t>(),
        nlocal,
        total_work,
        B, C, T, H, W,
        spatial_up,
        temporal_up,
        dtype_enum);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_assemble_blend(
    torch::Tensor ptrs,
    torch::Tensor meta,
    torch::Tensor prefix_t,
    torch::Tensor prefix_h,
    torch::Tensor prefix_w,
    torch::Tensor out,
    int64_t B,
    int64_t tiles_t,
    int64_t tiles_h,
    int64_t tiles_w,
    int blend_t,
    int blend_h,
    int blend_w
) {
    TORCH_CHECK(ptrs.is_cuda() && meta.is_cuda(), "metadata must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bf16");

    int64_t outT_total = out.size(2);
    int64_t outH_total = out.size(3);
    int64_t outW_total = out.size(4);
    int64_t total_out = B * 3 * outT_total * outH_total * outW_total;
    if (total_out <= 0) return;

    int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    assemble_blend_kernel<<<blocks, threads, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        meta.data_ptr<int64_t>(),
        prefix_t.data_ptr<int64_t>(),
        prefix_h.data_ptr<int64_t>(),
        prefix_w.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        total_out,
        B,
        tiles_t,
        tiles_h,
        tiles_w,
        outT_total,
        outH_total,
        outW_total,
        blend_t,
        blend_h,
        blend_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_decode_tiles", &launch_decode_tiles, "MAGI tile decode to symmetric bf16 buffer");
    m.def("launch_assemble_blend", &launch_assemble_blend, "MAGI UVA assemble/blend from symmetric buffers");
}
'''

_ext = None
_plan_cache: Dict[Tuple, Dict] = {}
_resource_cache: Dict[Tuple, Tuple[torch.Tensor, Optional[object], torch.Tensor]] = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("magi1_tile_parallel_decode_bf16_symm_ext", CUDA_SRC)
    return _ext


def _index_undot(index: int, loop_size: List[int]) -> List[int]:
    out: List[int] = []
    for size in reversed(loop_size):
        out.append(index % size)
        index //= size
    return list(reversed(out))


def _split_tiles(tile_numels: List[int], world_size: int, rank: int) -> Tuple[List[int], List[List[int]]]:
    if world_size == 1:
        all_tiles = list(range(len(tile_numels)))
        return all_tiles, [all_tiles]

    sorted_tiles = sorted(range(len(tile_numels)), key=lambda idx: tile_numels[idx], reverse=True)
    per_rank = [sorted_tiles[r::world_size] for r in range(world_size)]
    return per_rank[rank], per_rank


def _make_plan(
    z_shape: Tuple[int, int, int, int, int],
    tile_latent_min_length: int,
    tile_latent_min_height: int,
    tile_latent_min_width: int,
    spatial_tile_overlap_factor: float,
    temporal_tile_overlap_factor: float,
    spatial_upsample: int,
    temporal_upsample: int,
    world_size: int,
    rank: int,
    device: torch.device,
) -> Dict:
    B, C, T, H, W = z_shape

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

    tiles_t = (T + stride_t - 1) // stride_t
    tiles_h = (H + stride_h - 1) // stride_h
    tiles_w = (W + stride_w - 1) // stride_w
    total_tiles = tiles_t * tiles_h * tiles_w
    loop_size = [tiles_t, tiles_h, tiles_w]

    tile_numels: List[int] = []
    meta_rows: List[List[int]] = []

    for tile_idx in range(total_tiles):
        t_idx, h_idx, w_idx = _index_undot(tile_idx, loop_size)
        t0 = t_idx * stride_t
        h0 = h_idx * stride_h
        w0 = w_idx * stride_w

        in_t = max(0, min(tile_latent_min_length, T - t0))
        in_h = max(0, min(tile_latent_min_height, H - h0))
        in_w = max(0, min(tile_latent_min_width, W - w0))

        out_t = in_t * temporal_upsample
        out_h = in_h * spatial_upsample
        out_w = in_w * spatial_upsample

        tile_numels.append(int(B * C * in_t * in_h * in_w))
        meta_rows.append([t0, h0, w0, in_t, in_h, in_w, out_t, out_h, out_w, 0, 0])

    local_indices, per_rank = _split_tiles(tile_numels, world_size, rank)

    rank_totals: List[int] = []
    for r in range(world_size):
        off = 0
        for tile_idx in per_rank[r]:
            meta_rows[tile_idx][9] = off
            meta_rows[tile_idx][10] = r
            out_t = meta_rows[tile_idx][6]
            out_h = meta_rows[tile_idx][7]
            out_w = meta_rows[tile_idx][8]
            off += int(B * 3 * out_t * out_h * out_w)
        rank_totals.append(off)

    local_prefix: List[int] = [0]
    for tile_idx in local_indices:
        out_t = meta_rows[tile_idx][6]
        out_h = meta_rows[tile_idx][7]
        out_w = meta_rows[tile_idx][8]
        local_prefix.append(local_prefix[-1] + int(B * 3 * out_t * out_h * out_w))

    crop_t: List[int] = []
    for ti in range(tiles_t):
        idx = (ti * tiles_h + 0) * tiles_w + 0
        crop_t.append(min(meta_rows[idx][6], keep_t))

    crop_h: List[int] = []
    for hi in range(tiles_h):
        idx = (0 * tiles_h + hi) * tiles_w + 0
        crop_h.append(min(meta_rows[idx][7], keep_h))

    crop_w: List[int] = []
    for wi in range(tiles_w):
        idx = (0 * tiles_h + 0) * tiles_w + wi
        crop_w.append(min(meta_rows[idx][8], keep_w))

    prefix_t = [0]
    for v in crop_t:
        prefix_t.append(prefix_t[-1] + int(v))
    prefix_h = [0]
    for v in crop_h:
        prefix_h.append(prefix_h[-1] + int(v))
    prefix_w = [0]
    for v in crop_w:
        prefix_w.append(prefix_w[-1] + int(v))

    meta = torch.tensor(meta_rows, device=device, dtype=torch.int64)
    local_ids = torch.tensor(local_indices, device=device, dtype=torch.int64)
    local_prefix_t = torch.tensor(local_prefix, device=device, dtype=torch.int64)
    prefix_t_t = torch.tensor(prefix_t, device=device, dtype=torch.int64)
    prefix_h_t = torch.tensor(prefix_h, device=device, dtype=torch.int64)
    prefix_w_t = torch.tensor(prefix_w, device=device, dtype=torch.int64)

    return {
        "meta": meta,
        "local_ids": local_ids,
        "local_prefix": local_prefix_t,
        "total_local_work": int(local_prefix[-1]),
        "max_rank_elems": max(1, max(rank_totals) if rank_totals else 1),
        "prefix_t": prefix_t_t,
        "prefix_h": prefix_h_t,
        "prefix_w": prefix_w_t,
        "out_shape": (B, 3, int(prefix_t[-1]), int(prefix_h[-1]), int(prefix_w[-1])),
        "tiles_t": tiles_t,
        "tiles_h": tiles_h,
        "tiles_w": tiles_w,
        "blend_t": blend_t,
        "blend_h": blend_h,
        "blend_w": blend_w,
    }


def _get_plan(
    z: torch.Tensor,
    tile_latent_min_length: int,
    tile_latent_min_height: int,
    tile_latent_min_width: int,
    spatial_tile_overlap_factor: float,
    temporal_tile_overlap_factor: float,
    spatial_upsample: int,
    temporal_upsample: int,
    world_size: int,
    rank: int,
) -> Dict:
    key = (
        tuple(z.shape),
        tile_latent_min_length,
        tile_latent_min_height,
        tile_latent_min_width,
        float(spatial_tile_overlap_factor),
        float(temporal_tile_overlap_factor),
        spatial_upsample,
        temporal_upsample,
        world_size,
        rank,
        z.device.index,
    )
    plan = _plan_cache.get(key)
    if plan is None:
        plan = _make_plan(
            tuple(z.shape),
            tile_latent_min_length,
            tile_latent_min_height,
            tile_latent_min_width,
            spatial_tile_overlap_factor,
            temporal_tile_overlap_factor,
            spatial_upsample,
            temporal_upsample,
            world_size,
            rank,
            z.device,
        )
        _plan_cache[key] = plan
    return plan


def _get_resources(
    max_rank_elems: int,
    out_shape: Tuple[int, int, int, int, int],
    device: torch.device,
    group: Optional[dist.ProcessGroup],
    world_size: int,
) -> Tuple[torch.Tensor, Optional[object], torch.Tensor]:
    key = (max_rank_elems, out_shape, device.index, id(group), world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    if group is not None and world_size > 1:
        buf = symm_mem.empty((max_rank_elems,), device=device, dtype=torch.bfloat16)
        hdl = symm_mem.rendezvous(buf, group)
        ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    else:
        buf = torch.empty((max_rank_elems,), device=device, dtype=torch.bfloat16)
        hdl = None
        ptrs = torch.tensor([buf.data_ptr()], device=device, dtype=torch.int64)

    cached = (buf, hdl, ptrs)
    _resource_cache[key] = cached
    return cached


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
    """
    CUDA/symmetric-memory MAGI tile-parallel VAE decode.

    Local ranks decode only their largest-first scheduled tiles into symmetric BF16
    buffers.  After one symmetric-memory device barrier, every rank directly reads
    peer decoded tiles through UVA and fuses boundary blending, crop, and final
    concatenation into one CUDA kernel.
    """
    assert z.is_cuda, "z must be CUDA"
    assert z.dim() == 5, "z must be [B, C, T, H, W]"
    assert z.shape[1] > 0, "channel dimension must be non-empty"

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

    if z.dtype == torch.bfloat16:
        z_work = z.contiguous()
        dtype_enum = 0
    elif z.dtype == torch.float32:
        z_work = z.contiguous()
        dtype_enum = 1
    else:
        z_work = z.contiguous().float()
        dtype_enum = 1

    ext = _get_ext()

    plan = _get_plan(
        z_work,
        tile_latent_min_length,
        tile_latent_min_height,
        tile_latent_min_width,
        spatial_tile_overlap_factor,
        temporal_tile_overlap_factor,
        spatial_upsample,
        temporal_upsample,
        world_size,
        rank,
    )

    symbuf, hdl, ptrs = _get_resources(
        int(plan["max_rank_elems"]),
        tuple(plan["out_shape"]),
        z_work.device,
        group,
        world_size,
    )

    ext.launch_decode_tiles(
        z_work,
        symbuf,
        plan["meta"],
        plan["local_ids"],
        plan["local_prefix"],
        int(plan["total_local_work"]),
        int(spatial_upsample),
        int(temporal_upsample),
        int(dtype_enum),
    )

    if hdl is not None:
        hdl.barrier(channel=0)

    out = torch.empty(tuple(plan["out_shape"]), device=z_work.device, dtype=torch.bfloat16)

    ext.launch_assemble_blend(
        ptrs,
        plan["meta"],
        plan["prefix_t"],
        plan["prefix_h"],
        plan["prefix_w"],
        out,
        int(z_work.shape[0]),
        int(plan["tiles_t"]),
        int(plan["tiles_h"]),
        int(plan["tiles_w"]),
        int(plan["blend_t"]),
        int(plan["blend_h"]),
        int(plan["blend_w"]),
    )

    return out