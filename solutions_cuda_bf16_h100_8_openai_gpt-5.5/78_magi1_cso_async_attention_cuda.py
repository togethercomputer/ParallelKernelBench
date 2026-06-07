from typing import Optional, Tuple, Dict, Any

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

static inline int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

__global__ void pack_kv_bf16_kernel(
    const __nv_bfloat16* __restrict__ kv,
    __nv_bfloat16* __restrict__ send,
    int64_t tokens,
    int orig_heads,
    int eff_heads,
    int local_heads,
    int width,
    int world_size
) {
    int64_t total = (int64_t)world_size * tokens * local_heads * width;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int repeat = (orig_heads < world_size && (world_size % orig_heads) == 0)
        ? (world_size / orig_heads)
        : 1;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int col = (int)(idx % width);
        int64_t t0 = idx / width;
        int lh = (int)(t0 % local_heads);
        int64_t row = t0 / local_heads;
        int tok = (int)(row % tokens);
        int dst = (int)(row / tokens);

        int eff_h = dst * local_heads + lh;
        int src_h = (repeat > 1) ? (eff_h / repeat) : eff_h;
        if (src_h >= orig_heads) src_h = orig_heads - 1;

        send[idx] = kv[((int64_t)tok * orig_heads + src_h) * width + col];
    }
}

__global__ void a2a_read_bf16_kernel(
    const int64_t* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t rows_per_peer,
    int64_t row_elems,
    int rank,
    int world_size
) {
    int64_t total = (int64_t)world_size * rows_per_peer * row_elems;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t col = idx % row_elems;
        int64_t row = idx / row_elems;
        int src_rank = (int)(row / rows_per_peer);
        int64_t r = row - (int64_t)src_rank * rows_per_peer;

        const __nv_bfloat16* remote =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[src_rank]);
        int64_t src_idx = ((int64_t)rank * rows_per_peer + r) * row_elems + col;
        out[idx] = remote[src_idx];
    }
}

__global__ void kv_by_range_split_bf16_kernel(
    const __nv_bfloat16* __restrict__ kv_red,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    int64_t tokens,
    int ranges,
    int spb,
    int clip,
    int local_heads,
    int head_dim,
    int world_size
) {
    int64_t total = (int64_t)ranges * clip * local_heads * head_dim;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % head_dim);
        int64_t t0 = idx / head_dim;
        int h = (int)(t0 % local_heads);
        int64_t pos = t0 / local_heads;
        int c = (int)(pos % clip);
        int r = (int)(pos / clip);

        int src_rank = c / spb;
        int tok_in_rank = c - src_rank * spb;
        if (src_rank >= world_size) {
            key[idx] = __float2bfloat16(0.0f);
            value[idx] = __float2bfloat16(0.0f);
            continue;
        }

        int64_t src_row = (int64_t)src_rank * tokens + (int64_t)r * spb + tok_in_rank;
        int width = head_dim * 2;
        int64_t base = ((src_row * local_heads + h) * width);
        key[idx] = kv_red[base + d];
        value[idx] = kv_red[base + head_dim + d];
    }
}

__global__ void expand_gqa_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t tokens,
    int kv_heads,
    int q_heads,
    int head_dim
) {
    int64_t total = tokens * (int64_t)q_heads * head_dim;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int repeat = q_heads / kv_heads;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % head_dim);
        int64_t t0 = idx / head_dim;
        int qh = (int)(t0 % q_heads);
        int64_t tok = t0 / q_heads;
        int kh = qh / repeat;
        dst[idx] = src[(tok * kv_heads + kh) * head_dim + d];
    }
}

__global__ void pack_q_range_bf16_kernel(
    const __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ send,
    int range_idx,
    int spb,
    int q_heads_total,
    int local_heads,
    int head_dim,
    int world_size
) {
    int64_t total = (int64_t)world_size * spb * local_heads * head_dim;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % head_dim);
        int64_t t0 = idx / head_dim;
        int lh = (int)(t0 % local_heads);
        int64_t row = t0 / local_heads;
        int s = (int)(row % spb);
        int dst = (int)(row / spb);

        int64_t src_tok = (int64_t)range_idx * spb + s;
        int src_h = dst * local_heads + lh;
        send[idx] = query[(src_tok * q_heads_total + src_h) * head_dim + d];
    }
}

__global__ void sdpa_htd_to_thd_bf16_kernel(
    const __nv_bfloat16* __restrict__ attn_htd,
    __nv_bfloat16* __restrict__ send_thd,
    int tokens,
    int heads,
    int head_dim
) {
    int64_t total = (int64_t)tokens * heads * head_dim;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % head_dim);
        int64_t t0 = idx / head_dim;
        int h = (int)(t0 % heads);
        int t = (int)(t0 / heads);
        send_thd[idx] = attn_htd[((int64_t)h * tokens + t) * head_dim + d];
    }
}

__global__ void restore_range_bf16_kernel(
    const __nv_bfloat16* __restrict__ chunk,
    __nv_bfloat16* __restrict__ out,
    int range_idx,
    int spb,
    int local_heads,
    int head_dim,
    int world_size
) {
    int64_t total = (int64_t)world_size * spb * local_heads * head_dim;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % head_dim);
        int64_t t0 = idx / head_dim;
        int lh = (int)(t0 % local_heads);
        int64_t row = t0 / local_heads;
        int s = (int)(row % spb);
        int head_rank = (int)(row / spb);

        int64_t out_tok = (int64_t)range_idx * spb + s;
        int out_h = head_rank * local_heads + lh;
        int q_heads_total = world_size * local_heads;
        out[(out_tok * q_heads_total + out_h) * head_dim + d] = chunk[idx];
    }
}

void launch_pack_kv(
    torch::Tensor kv,
    torch::Tensor send,
    int64_t tokens,
    int orig_heads,
    int eff_heads,
    int local_heads,
    int width,
    int world_size
) {
    int threads = 256;
    int64_t total = (int64_t)world_size * tokens * local_heads * width;
    int blocks = div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_kv_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(send.data_ptr<at::BFloat16>()),
        tokens, orig_heads, eff_heads, local_heads, width, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_a2a_read(
    torch::Tensor ptrs,
    torch::Tensor out,
    int64_t rows_per_peer,
    int64_t row_elems,
    int rank,
    int world_size
) {
    int threads = 256;
    int64_t total = (int64_t)world_size * rows_per_peer * row_elems;
    int blocks = div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    a2a_read_bf16_kernel<<<blocks, threads, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        rows_per_peer, row_elems, rank, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_kv_by_range_split(
    torch::Tensor kv_red,
    torch::Tensor key,
    torch::Tensor value,
    int64_t tokens,
    int ranges,
    int spb,
    int clip,
    int local_heads,
    int head_dim,
    int world_size
) {
    int threads = 256;
    int64_t total = (int64_t)ranges * clip * local_heads * head_dim;
    int blocks = div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    kv_by_range_split_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(kv_red.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
        tokens, ranges, spb, clip, local_heads, head_dim, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_expand_gqa(
    torch::Tensor src,
    torch::Tensor dst,
    int64_t tokens,
    int kv_heads,
    int q_heads,
    int head_dim
) {
    int threads = 256;
    int64_t total = tokens * (int64_t)q_heads * head_dim;
    int blocks = div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    expand_gqa_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
        tokens, kv_heads, q_heads, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pack_q_range(
    torch::Tensor query,
    torch::Tensor send,
    int range_idx,
    int spb,
    int q_heads_total,
    int local_heads,
    int head_dim,
    int world_size
) {
    int threads = 256;
    int64_t total = (int64_t)world_size * spb * local_heads * head_dim;
    int blocks = div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_q_range_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(send.data_ptr<at::BFloat16>()),
        range_idx, spb, q_heads_total, local_heads, head_dim, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_sdpa_htd_to_thd(
    torch::Tensor attn_htd,
    torch::Tensor send_thd,
    int tokens,
    int heads,
    int head_dim
) {
    int threads = 256;
    int64_t total = (int64_t)tokens * heads * head_dim;
    int blocks = div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    sdpa_htd_to_thd_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(attn_htd.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(send_thd.data_ptr<at::BFloat16>()),
        tokens, heads, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_restore_range(
    torch::Tensor chunk,
    torch::Tensor out,
    int range_idx,
    int spb,
    int local_heads,
    int head_dim,
    int world_size
) {
    int threads = 256;
    int64_t total = (int64_t)world_size * spb * local_heads * head_dim;
    int blocks = div_up_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    restore_range_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(chunk.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        range_idx, spb, local_heads, head_dim, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pack_kv", &launch_pack_kv, "pack KV heads for symmetric all-to-all");
    m.def("launch_a2a_read", &launch_a2a_read, "UVA symmetric-memory all-to-all row read");
    m.def("launch_kv_by_range_split", &launch_kv_by_range_split, "KV by-range split into K/V");
    m.def("launch_expand_gqa", &launch_expand_gqa, "expand KV heads for GQA");
    m.def("launch_pack_q_range", &launch_pack_q_range, "pack Q range for symmetric all-to-all");
    m.def("launch_sdpa_htd_to_thd", &launch_sdpa_htd_to_thd, "transpose SDPA H,T,D output to T,H,D send buffer");
    m.def("launch_restore_range", &launch_restore_range, "restore output range to original layout");
}
'''


_ext = None
_jit_ready: Dict[int, bool] = {}
_resource_cache: Dict[Any, Any] = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("magi1_cso_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


def _ensure_jit(group: dist.ProcessGroup):
    gid = id(group)
    if _jit_ready.get(gid, False):
        return
    rank = dist.get_rank(group=group)
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()
    _jit_ready[gid] = True


def _symm_with_ptrs(shape: Tuple[int, ...], device: torch.device, group: dist.ProcessGroup):
    buf = symm_mem.empty(shape, device=device, dtype=torch.bfloat16)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor([int(p) for p in hdl.buffer_ptrs], device=device, dtype=torch.int64)
    return buf, hdl, ptrs


def _get_resources(
    *,
    tokens: int,
    q_heads_total: int,
    kv_heads_orig: int,
    head_dim: int,
    ranges: int,
    spb: int,
    clip: int,
    world_size: int,
    device: torch.device,
    group: dist.ProcessGroup,
):
    if kv_heads_orig < world_size and world_size % kv_heads_orig == 0:
        kv_heads_eff = world_size
    else:
        kv_heads_eff = kv_heads_orig

    if kv_heads_eff % world_size != 0:
        raise ValueError("KV heads must divide evenly across context ranks")
    if q_heads_total % world_size != 0:
        raise ValueError("query heads must divide evenly across context ranks")

    local_kv_heads = kv_heads_eff // world_size
    local_q_heads = q_heads_total // world_size
    width = 2 * head_dim

    key = (
        id(group),
        str(device),
        tokens,
        q_heads_total,
        kv_heads_orig,
        kv_heads_eff,
        head_dim,
        ranges,
        spb,
        clip,
        world_size,
    )
    if key in _resource_cache:
        return _resource_cache[key]

    kv_send, kv_hdl, kv_ptrs = _symm_with_ptrs(
        (world_size * tokens, local_kv_heads, width), device, group
    )
    q_send, q_hdl, q_ptrs = _symm_with_ptrs(
        (world_size * spb, local_q_heads, head_dim), device, group
    )
    o_send, o_hdl, o_ptrs = _symm_with_ptrs(
        (world_size * spb, local_q_heads, head_dim), device, group
    )

    kv_red = torch.empty((world_size * tokens, local_kv_heads, width), device=device, dtype=torch.bfloat16)
    key_t = torch.empty((ranges * clip, local_kv_heads, head_dim), device=device, dtype=torch.bfloat16)
    val_t = torch.empty((ranges * clip, local_kv_heads, head_dim), device=device, dtype=torch.bfloat16)

    if local_kv_heads < local_q_heads:
        key_attn = torch.empty((ranges * clip, local_q_heads, head_dim), device=device, dtype=torch.bfloat16)
        val_attn = torch.empty((ranges * clip, local_q_heads, head_dim), device=device, dtype=torch.bfloat16)
    else:
        key_attn = key_t
        val_attn = val_t

    q_local = torch.empty((world_size * spb, local_q_heads, head_dim), device=device, dtype=torch.bfloat16)
    o_local = torch.empty((world_size * spb, local_q_heads, head_dim), device=device, dtype=torch.bfloat16)

    res = {
        "kv_heads_eff": kv_heads_eff,
        "local_kv_heads": local_kv_heads,
        "local_q_heads": local_q_heads,
        "kv_send": kv_send,
        "kv_hdl": kv_hdl,
        "kv_ptrs": kv_ptrs,
        "q_send": q_send,
        "q_hdl": q_hdl,
        "q_ptrs": q_ptrs,
        "o_send": o_send,
        "o_hdl": o_hdl,
        "o_ptrs": o_ptrs,
        "kv_red": kv_red,
        "key": key_t,
        "value": val_t,
        "key_attn": key_attn,
        "value_attn": val_attn,
        "q_local": q_local,
        "o_local": o_local,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    query: torch.Tensor,
    key_value: torch.Tensor,
    k_ranges: torch.Tensor,
    cp_shuffle_num: int,
    clip_token_nums: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert query.is_cuda and key_value.is_cuda
    assert query.dtype == torch.bfloat16 and key_value.dtype == torch.bfloat16
    assert query.is_contiguous() and key_value.is_contiguous()

    _ensure_jit(group)
    ext = _get_ext()

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    ranges = int(cp_shuffle_num)
    tokens = int(query.shape[0])
    q_heads_total = int(query.shape[1])
    head_dim = int(query.shape[2])
    kv_heads_orig = int(key_value.shape[1])

    if tokens % ranges != 0:
        raise ValueError("query token count must divide cp_shuffle_num")
    spb = tokens // ranges
    clip = int(clip_token_nums or (world_size * spb))

    res = _get_resources(
        tokens=tokens,
        q_heads_total=q_heads_total,
        kv_heads_orig=kv_heads_orig,
        head_dim=head_dim,
        ranges=ranges,
        spb=spb,
        clip=clip,
        world_size=world_size,
        device=query.device,
        group=group,
    )

    local_kv_heads = int(res["local_kv_heads"])
    local_q_heads = int(res["local_q_heads"])
    kv_heads_eff = int(res["kv_heads_eff"])
    width = 2 * head_dim

    # KV head redistribution: local pack -> symmetric P2P all-to-all -> range-major K/V.
    ext.launch_pack_kv(
        key_value,
        res["kv_send"],
        tokens,
        kv_heads_orig,
        kv_heads_eff,
        local_kv_heads,
        width,
        world_size,
    )
    res["kv_hdl"].barrier(channel=0)
    ext.launch_a2a_read(
        res["kv_ptrs"],
        res["kv_red"],
        tokens,
        local_kv_heads * width,
        rank,
        world_size,
    )
    res["kv_hdl"].barrier(channel=1)

    ext.launch_kv_by_range_split(
        res["kv_red"],
        res["key"],
        res["value"],
        tokens,
        ranges,
        spb,
        clip,
        local_kv_heads,
        head_dim,
        world_size,
    )

    key_attn = res["key"]
    value_attn = res["value"]
    attn_heads = local_kv_heads

    if local_kv_heads < local_q_heads:
        if local_q_heads % local_kv_heads != 0:
            raise ValueError("query heads must be an integer multiple of KV heads")
        ext.launch_expand_gqa(
            res["key"],
            res["key_attn"],
            ranges * clip,
            local_kv_heads,
            local_q_heads,
            head_dim,
        )
        ext.launch_expand_gqa(
            res["value"],
            res["value_attn"],
            ranges * clip,
            local_kv_heads,
            local_q_heads,
            head_dim,
        )
        key_attn = res["key_attn"]
        value_attn = res["value_attn"]
        attn_heads = local_q_heads

    out = torch.empty_like(query)
    q_tokens = world_size * spb

    for idx in range(ranges):
        # Query all-to-all for this range.
        ext.launch_pack_q_range(
            query,
            res["q_send"],
            idx,
            spb,
            q_heads_total,
            local_q_heads,
            head_dim,
            world_size,
        )
        res["q_hdl"].barrier(channel=2)
        ext.launch_a2a_read(
            res["q_ptrs"],
            res["q_local"],
            spb,
            local_q_heads * head_dim,
            rank,
            world_size,
        )
        res["q_hdl"].barrier(channel=3)

        start = int(k_ranges[idx, 0].item())
        end = int(k_ranges[idx, 1].item())

        q4 = res["q_local"].unsqueeze(0).transpose(1, 2)
        k4 = key_attn[start:end].unsqueeze(0).transpose(1, 2)
        v4 = value_attn[start:end].unsqueeze(0).transpose(1, 2)

        # H100 Flash/SDPA tensor-core path for BF16 math.
        attn4 = F.scaled_dot_product_attention(q4, k4, v4)

        # Convert [1, H, T, D] into symmetric [T, H, D] send buffer.
        ext.launch_sdpa_htd_to_thd(
            attn4.squeeze(0),
            res["o_send"],
            q_tokens,
            attn_heads,
            head_dim,
        )

        # Output all-to-all back to token owners, then fused restore into query layout.
        res["o_hdl"].barrier(channel=4)
        ext.launch_a2a_read(
            res["o_ptrs"],
            res["o_local"],
            spb,
            local_q_heads * head_dim,
            rank,
            world_size,
        )
        res["o_hdl"].barrier(channel=5)

        ext.launch_restore_range(
            res["o_local"],
            out,
            idx,
            spb,
            local_q_heads,
            head_dim,
            world_size,
        )

    return out