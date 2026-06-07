from typing import List, Optional, Tuple
import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

_NO_OBJ_LOGIT = -10.0

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define DTYPE_F32  0
#define DTYPE_BF16 1
#define DTYPE_F16  2

__device__ __forceinline__ float load_as_f32(const void* base, int64_t idx, int dtype) {
    if (dtype == DTYPE_F32) {
        return reinterpret_cast<const float*>(base)[idx];
    } else if (dtype == DTYPE_BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[idx]);
    } else {
        return __half2float(reinterpret_cast<const __half*>(base)[idx]);
    }
}

__global__ void pack_local_to_sym_kernel(
    const void* __restrict__ masks,
    const void* __restrict__ scores,
    float* __restrict__ sym_flat,
    int64_t global_offset,
    int64_t n_local,
    int64_t pixels,
    int64_t total_objects,
    int mask_dtype,
    int score_dtype
) {
    const int64_t mask_elems = n_local * pixels;
    const int64_t work = mask_elems + n_local;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < work; idx += stride) {
        if (idx < mask_elems) {
            float v = load_as_f32(masks, idx, mask_dtype);
            sym_flat[global_offset * pixels + idx] = v;
        } else {
            int64_t sidx = idx - mask_elems;
            float v = load_as_f32(scores, sidx, score_dtype);
            sym_flat[total_objects * pixels + global_offset + sidx] = v;
        }
    }
}

__global__ void gather_bitpack_area_kernel(
    const int64_t* __restrict__ ptrs,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ counts,
    int world_size,
    float* __restrict__ masks_out,
    float* __restrict__ scores_out,
    uint32_t* __restrict__ bitsets,
    int32_t* __restrict__ areas,
    int64_t total_objects,
    int64_t pixels,
    int64_t words
) {
    int obj = blockIdx.x;
    if (obj >= total_objects) return;

    int owner = 0;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r >= world_size) break;
        int64_t lo = offsets[r];
        int64_t hi = lo + counts[r];
        if ((int64_t)obj >= lo && (int64_t)obj < hi) {
            owner = r;
            break;
        }
    }

    const float* peer_base =
        reinterpret_cast<const float*>(static_cast<uintptr_t>(ptrs[owner]));
    const float* src = peer_base + (int64_t)obj * pixels;
    float* dst = masks_out + (int64_t)obj * pixels;

    for (int64_t p = threadIdx.x; p < pixels; p += blockDim.x) {
        dst[p] = src[p];
    }

    if (threadIdx.x == 0) {
        scores_out[obj] = peer_base[total_objects * pixels + obj];
    }

    __syncthreads();

    int local_area = 0;
    for (int64_t w = threadIdx.x; w < words; w += blockDim.x) {
        uint32_t bits = 0u;
        int64_t base_p = w * 32;
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            int64_t p = base_p + b;
            if (p < pixels && dst[p] > 0.0f) {
                bits |= (1u << b);
            }
        }
        bitsets[(int64_t)obj * words + w] = bits;
        local_area += __popc(bits);
    }

    __shared__ int sh[256];
    sh[threadIdx.x] = local_area;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        areas[obj] = sh[0];
    }
}

__global__ void pair_suppress_bitset_kernel(
    const uint32_t* __restrict__ bitsets,
    const int32_t* __restrict__ areas,
    const int64_t* __restrict__ last_occluded,
    uint8_t* __restrict__ suppress,
    int64_t total_objects,
    int64_t words,
    float iou_threshold,
    bool reverse
) {
    const int j = blockIdx.x * 16 + threadIdx.x;
    const int i = blockIdx.y * 16 + threadIdx.y;

    if ((int64_t)i >= total_objects || (int64_t)j >= total_objects || i >= j) {
        return;
    }

    const uint32_t* bi = bitsets + (int64_t)i * words;
    const uint32_t* bj = bitsets + (int64_t)j * words;

    int inter = 0;
    for (int64_t w = 0; w < words; ++w) {
        inter += __popc(bi[w] & bj[w]);
    }

    int uni = (int)areas[i] + (int)areas[j] - inter;
    if (uni < 1) uni = 1;

    if ((float)inter >= iou_threshold * (float)uni) {
        int64_t li = last_occluded[i];
        int64_t lj = last_occluded[j];

        if (!reverse) {
            if (li > lj && lj > -1) suppress[i] = 1;
            if (lj > li && li > -1) suppress[j] = 1;
        } else {
            if (li < lj && lj > -1) suppress[i] = 1;
            if (lj < li && li > -1) suppress[j] = 1;
        }
    }
}

__global__ void apply_suppression_kernel(
    float* __restrict__ masks_out,
    const uint8_t* __restrict__ suppress,
    int64_t total_objects,
    int64_t pixels
) {
    const int64_t total = total_objects * pixels;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t obj = idx / pixels;
        if (suppress[obj]) {
            masks_out[idx] = -10.0f;
        }
    }
}

static int dtype_enum(torch::Tensor t) {
    if (t.scalar_type() == torch::kFloat32) return DTYPE_F32;
    if (t.scalar_type() == torch::kBFloat16) return DTYPE_BF16;
    if (t.scalar_type() == torch::kFloat16) return DTYPE_F16;
    TORCH_CHECK(false, "unsupported dtype after Python normalization");
}

void pack_local_to_sym(
    torch::Tensor masks,
    torch::Tensor scores,
    torch::Tensor sym_flat,
    int64_t global_offset,
    int64_t n_local,
    int64_t pixels,
    int64_t total_objects
) {
    TORCH_CHECK(masks.is_cuda() && scores.is_cuda() && sym_flat.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(masks.is_contiguous() && scores.is_contiguous() && sym_flat.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(sym_flat.scalar_type() == torch::kFloat32, "sym_flat must be float32");

    const int64_t work = n_local * pixels + n_local;
    if (work <= 0) return;

    int threads = 256;
    int blocks = (int)((work + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_local_to_sym_kernel<<<blocks, threads, 0, stream>>>(
        masks.data_ptr(),
        scores.data_ptr(),
        sym_flat.data_ptr<float>(),
        global_offset,
        n_local,
        pixels,
        total_objects,
        dtype_enum(masks),
        dtype_enum(scores)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_bitpack_area(
    torch::Tensor ptrs,
    torch::Tensor offsets,
    torch::Tensor counts,
    torch::Tensor masks_out,
    torch::Tensor scores_out,
    torch::Tensor bitsets,
    torch::Tensor areas,
    int64_t total_objects,
    int64_t pixels,
    int64_t words,
    int world_size
) {
    TORCH_CHECK(ptrs.is_cuda() && offsets.is_cuda() && counts.is_cuda(), "metadata must be CUDA");
    TORCH_CHECK(masks_out.is_cuda() && scores_out.is_cuda() && bitsets.is_cuda() && areas.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(masks_out.scalar_type() == torch::kFloat32 && scores_out.scalar_type() == torch::kFloat32, "float32 outputs required");
    TORCH_CHECK(bitsets.scalar_type() == torch::kInt32 && areas.scalar_type() == torch::kInt32, "int32 bitsets/areas required");

    if (total_objects <= 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_bitpack_area_kernel<<<(int)total_objects, 256, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(),
        world_size,
        masks_out.data_ptr<float>(),
        scores_out.data_ptr<float>(),
        reinterpret_cast<uint32_t*>(bitsets.data_ptr<int32_t>()),
        areas.data_ptr<int32_t>(),
        total_objects,
        pixels,
        words
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_suppression(
    torch::Tensor bitsets,
    torch::Tensor areas,
    torch::Tensor last_occluded,
    torch::Tensor suppress,
    int64_t total_objects,
    int64_t words,
    double iou_threshold,
    bool reverse
) {
    TORCH_CHECK(bitsets.is_cuda() && areas.is_cuda() && last_occluded.is_cuda() && suppress.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(last_occluded.scalar_type() == torch::kInt64, "last_occluded must be int64");
    TORCH_CHECK(suppress.scalar_type() == torch::kBool, "suppress must be bool");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (total_objects > 0) {
        cudaMemsetAsync(suppress.data_ptr(), 0, (size_t)total_objects, stream);
    }

    if (total_objects <= 1) {
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    dim3 block(16, 16, 1);
    dim3 grid((unsigned int)((total_objects + 15) / 16),
              (unsigned int)((total_objects + 15) / 16),
              1);

    pair_suppress_bitset_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(bitsets.data_ptr<int32_t>()),
        areas.data_ptr<int32_t>(),
        last_occluded.data_ptr<int64_t>(),
        reinterpret_cast<uint8_t*>(suppress.data_ptr<bool>()),
        total_objects,
        words,
        (float)iou_threshold,
        reverse
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void apply_suppression(
    torch::Tensor masks_out,
    torch::Tensor suppress,
    int64_t total_objects,
    int64_t pixels
) {
    TORCH_CHECK(masks_out.is_cuda() && suppress.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(masks_out.scalar_type() == torch::kFloat32, "masks_out must be float32");
    TORCH_CHECK(suppress.scalar_type() == torch::kBool, "suppress must be bool");

    const int64_t work = total_objects * pixels;
    if (work <= 0) return;

    int threads = 256;
    int blocks = (int)((work + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    apply_suppression_kernel<<<blocks, threads, 0, stream>>>(
        masks_out.data_ptr<float>(),
        reinterpret_cast<const uint8_t*>(suppress.data_ptr<bool>()),
        total_objects,
        pixels
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_local_to_sym", &pack_local_to_sym, "pack local masks/scores into symmetric FP32 flat buffer");
    m.def("gather_bitpack_area", &gather_bitpack_area, "UVA gather plus binary bitpack and area");
    m.def("compute_suppression", &compute_suppression, "bitset IoU suppression");
    m.def("apply_suppression", &apply_suppression, "fill suppressed masks with no-object logit");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("sam3_symm_uva_bitset_iou_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _supported_comm_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float32, torch.bfloat16, torch.float16)


def _normalize_local(t: torch.Tensor) -> torch.Tensor:
    if not _supported_comm_dtype(t.dtype):
        t = t.float()
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def _get_resources(
    *,
    trailing_shape: Tuple[int, ...],
    counts: Tuple[int, ...],
    device: torch.device,
    group: dist.ProcessGroup,
):
    total = int(sum(counts))
    pixels = _prod(trailing_shape)
    words = (pixels + 31) // 32
    key = (
        int(device.index) if device.index is not None else torch.cuda.current_device(),
        trailing_shape,
        counts,
        id(group),
    )

    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    sym_elems = total * pixels + total
    sym_flat = symm_mem.empty((sym_elems,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(sym_flat, group)

    masks_out = torch.empty((total, *trailing_shape), device=device, dtype=torch.float32)
    scores_out = torch.empty((total,), device=device, dtype=torch.float32)
    suppress = torch.empty((total,), device=device, dtype=torch.bool)
    bitsets = torch.empty((total, words), device=device, dtype=torch.int32)
    areas = torch.empty((total,), device=device, dtype=torch.int32)

    offsets_list = []
    acc = 0
    for c in counts:
        offsets_list.append(acc)
        acc += int(c)

    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.int64)
    counts_t = torch.tensor(list(counts), device=device, dtype=torch.int64)

    cached = {
        "sym_flat": sym_flat,
        "hdl": hdl,
        "masks_out": masks_out,
        "scores_out": scores_out,
        "suppress": suppress,
        "bitsets": bitsets,
        "areas": areas,
        "ptrs": ptrs,
        "offsets": offsets,
        "counts": counts_t,
        "total": total,
        "pixels": pixels,
        "words": words,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    low_res_masks_local: torch.Tensor,
    obj_scores_local: torch.Tensor,
    num_obj_per_gpu: List[int],
    last_occluded: torch.Tensor,
    iou_threshold: float = 0.7,
    reverse: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    if len(num_obj_per_gpu) != world_size:
        raise ValueError("num_obj_per_gpu length must match group world size")

    expected = int(num_obj_per_gpu[rank])
    if low_res_masks_local.shape[0] != expected:
        raise ValueError("local mask count does not match num_obj_per_gpu")
    if obj_scores_local.shape[0] != expected:
        raise ValueError("local score count does not match num_obj_per_gpu")
    if not low_res_masks_local.is_cuda or not obj_scores_local.is_cuda:
        raise ValueError("CUDA tensors are required")

    total = int(sum(int(x) for x in num_obj_per_gpu))
    trailing_shape = tuple(int(x) for x in low_res_masks_local.shape[1:])
    device = low_res_masks_local.device

    if total == 0:
        return (
            torch.empty((0, *trailing_shape), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.bool),
        )

    masks_local = _normalize_local(low_res_masks_local)
    scores_local = _normalize_local(obj_scores_local)

    counts = tuple(int(x) for x in num_obj_per_gpu)
    res = _get_resources(
        trailing_shape=trailing_shape,
        counts=counts,
        device=device,
        group=group,
    )

    pixels = res["pixels"]
    words = res["words"]

    local_offset = int(res["offsets"].cpu()[rank].item()) if False else sum(counts[:rank])

    ext = _get_ext()

    ext.pack_local_to_sym(
        masks_local,
        scores_local,
        res["sym_flat"],
        int(local_offset),
        int(expected),
        int(pixels),
        int(total),
    )

    # Symmetric-memory device barrier: all ranks' packed slices become visible
    # to peer UVA loads before the fused gather/bitpack kernel starts.
    res["hdl"].barrier(channel=0)

    ext.gather_bitpack_area(
        res["ptrs"],
        res["offsets"],
        res["counts"],
        res["masks_out"],
        res["scores_out"],
        res["bitsets"],
        res["areas"],
        int(total),
        int(pixels),
        int(words),
        int(world_size),
    )

    if last_occluded.device != device or last_occluded.dtype != torch.long:
        last_dev = last_occluded.to(device=device, dtype=torch.long, non_blocking=True)
    else:
        last_dev = last_occluded
    if not last_dev.is_contiguous():
        last_dev = last_dev.contiguous()

    ext.compute_suppression(
        res["bitsets"],
        res["areas"],
        last_dev,
        res["suppress"],
        int(total),
        int(words),
        float(iou_threshold),
        bool(reverse),
    )

    ext.apply_suppression(
        res["masks_out"],
        res["suppress"],
        int(total),
        int(pixels),
    )

    return res["masks_out"], res["scores_out"], res["suppress"]