import os
from typing import List, Optional, Tuple

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

// Fused gather, binarization, and area accumulation.
// Reads bfloat16 patches from peer symmetric memory pointers, converts to f32, and extracts area.
__global__ void fetch_and_prep_kernel(
    const long long* __restrict__ ptrs_masks,
    const long long* __restrict__ ptrs_scores,
    const int* __restrict__ offsets,
    float* __restrict__ masks_global,
    float* __restrict__ scores_global,
    float* __restrict__ binary_masks_flat,
    float* __restrict__ areas,
    int N_total,
    int H_W,
    int world_size
) {
    int i = blockIdx.x; // object index
    if (i >= N_total) return;

    // Find rank owner of the global object i
    int j = 0;
    while (j < world_size - 1 && i >= offsets[j+1]) {
        j++;
    }
    int local_i = i - offsets[j];

    const __nv_bfloat16* src_mask = (const __nv_bfloat16*)ptrs_masks[j] + local_i * H_W;
    float* dst_mask = masks_global + i * H_W;
    float* dst_bin = binary_masks_flat + i * H_W;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local_area = 0.0f;
    
    // Safely vectorize loads if alignment matches
    bool can_vectorize = (((uintptr_t)src_mask) % 16 == 0) && (H_W % 8 == 0);

    if (can_vectorize) {
        int num_vec = H_W / 8;
        for (int k = tid; k < num_vec; k += stride) {
            ulonglong2 vec = *(const ulonglong2*)(src_mask + k * 8);
            const __nv_bfloat16* vals = (const __nv_bfloat16*)&vec;
            
            #pragma unroll
            for (int v = 0; v < 8; v++) {
                float val_f32 = __bfloat162float(vals[v]);
                dst_mask[k * 8 + v] = val_f32;
                float bin_val = val_f32 > 0.0f ? 1.0f : 0.0f;
                dst_bin[k * 8 + v] = bin_val;
                local_area += bin_val;
            }
        }
        for (int k = num_vec * 8 + tid; k < H_W; k += stride) {
            float val_f32 = __bfloat162float(src_mask[k]);
            dst_mask[k] = val_f32;
            float bin_val = val_f32 > 0.0f ? 1.0f : 0.0f;
            dst_bin[k] = bin_val;
            local_area += bin_val;
        }
    } else {
        for (int k = tid; k < H_W; k += stride) {
            float val_f32 = __bfloat162float(src_mask[k]);
            dst_mask[k] = val_f32;
            float bin_val = val_f32 > 0.0f ? 1.0f : 0.0f;
            dst_bin[k] = bin_val;
            local_area += bin_val;
        }
    }

    // Shared-memory block reduction to aggregate object area
    static __shared__ float shared_area[256];
    shared_area[tid] = local_area;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_area[tid] += shared_area[tid + s];
        }
        __syncthreads();
    }

    // Assign area and load matching score sequentially
    if (tid == 0) {
        areas[i] = shared_area[0];
        const __nv_bfloat16* src_score = (const __nv_bfloat16*)ptrs_scores[j] + local_i;
        scores_global[i] = __bfloat162float(*src_score);
    }
}

// Single-pass symmetric compute for boolean suppressions. Overlaps logic mask application.
__global__ void compute_and_apply_suppress_kernel(
    const float* __restrict__ intersection,
    const float* __restrict__ areas,
    const int64_t* __restrict__ last_occluded,
    bool* __restrict__ to_suppress_out,
    float* __restrict__ masks_global,
    float iou_threshold,
    bool reverse,
    int N_total,
    int H_W,
    float no_obj_logit
) {
    int k = blockIdx.x; 
    if (k >= N_total) return;

    __shared__ bool suppress;
    if (threadIdx.x == 0) {
        suppress = false;
        int64_t last_k = last_occluded[k];

        for (int other = 0; other < N_total; other++) {
            if (k == other) continue;

            int i = k < other ? k : other;
            int j = k < other ? other : k;

            float inter = intersection[i * N_total + j];
            float union_area = areas[i] + areas[j] - inter;
            if (union_area < 1.0f) union_area = 1.0f;
            float iou = inter / union_area;

            if (iou >= iou_threshold) {
                int64_t last_other = last_occluded[other];
                bool cmp = reverse ? (last_k < last_other) : (last_k > last_other);
                if (cmp && last_other > -1) {
                    suppress = true;
                    break;
                }
            }
        }
        to_suppress_out[k] = suppress;
    }

    __syncthreads();

    // Mask with No-Object Logit in place across local threads if suppressed
    if (suppress) {
        for (int p = threadIdx.x; p < H_W; p += blockDim.x) {
            masks_global[k * H_W + p] = no_obj_logit;
        }
    }
}

void fetch_and_prep(
    torch::Tensor ptrs_masks,
    torch::Tensor ptrs_scores,
    torch::Tensor offsets,
    torch::Tensor masks_global,
    torch::Tensor scores_global,
    torch::Tensor binary_masks_flat,
    torch::Tensor areas,
    int N_total,
    int H_W,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (N_total > 0) {
        fetch_and_prep_kernel<<<N_total, 256, 0, stream>>>(
            ptrs_masks.data_ptr<int64_t>(),
            ptrs_scores.data_ptr<int64_t>(),
            offsets.data_ptr<int32_t>(),
            masks_global.data_ptr<float>(),
            scores_global.data_ptr<float>(),
            binary_masks_flat.data_ptr<float>(),
            areas.data_ptr<float>(),
            N_total,
            H_W,
            world_size
        );
    }
}

void compute_and_suppress(
    torch::Tensor intersection,
    torch::Tensor areas,
    torch::Tensor last_occluded,
    torch::Tensor to_suppress_out,
    torch::Tensor masks_global,
    float iou_threshold,
    bool reverse,
    int N_total,
    int H_W,
    float no_obj_logit
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (N_total > 0) {
        compute_and_apply_suppress_kernel<<<N_total, 256, 0, stream>>>(
            intersection.data_ptr<float>(),
            areas.data_ptr<float>(),
            last_occluded.data_ptr<int64_t>(),
            to_suppress_out.data_ptr<bool>(),
            masks_global.data_ptr<float>(),
            iou_threshold,
            reverse,
            N_total,
            H_W,
            no_obj_logit
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fetch_and_prep", &fetch_and_prep, "Fetch and prepare inputs");
    m.def("compute_and_suppress", &compute_and_suppress, "Compute IoU logic and suppress");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("sam3_iou_suppress_ext", CUDA_SRC)
    return _ext


_symm_cache = {}

def _get_symm_state(max_N_local, H, W, device, group_id, group):
    key = (max_N_local, H, W, device, group_id)
    if key in _symm_cache:
        return _symm_cache[key]
    
    # BF16 slices to halve the UVA symmetric bandwidth constraints over the fast NVLink loop
    masks_symm = symm_mem.empty((max_N_local, H, W), dtype=torch.bfloat16, device=device)
    scores_symm = symm_mem.empty((max_N_local,), dtype=torch.bfloat16, device=device)
    
    hdl_masks = symm_mem.rendezvous(masks_symm, group=group)
    hdl_scores = symm_mem.rendezvous(scores_symm, group=group)
    
    ptrs_masks_tensor = torch.tensor(hdl_masks.buffer_ptrs, dtype=torch.int64, device=device)
    ptrs_scores_tensor = torch.tensor(hdl_scores.buffer_ptrs, dtype=torch.int64, device=device)
    
    res = (masks_symm, scores_symm, hdl_masks, hdl_scores, ptrs_masks_tensor, ptrs_scores_tensor)
    _symm_cache[key] = res
    return res


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
    device = low_res_masks_local.device

    if rank == 0:
        _get_ext()
    dist.barrier(group=group)

    expected = int(num_obj_per_gpu[rank])
    if low_res_masks_local.shape[0] != expected:
        raise ValueError("local mask count does not match num_obj_per_gpu")
    
    H, W = low_res_masks_local.shape[1:]
    H_W = H * W
    N_total = sum(num_obj_per_gpu)
    max_N_local = max(num_obj_per_gpu) if world_size > 0 else 0

    if N_total == 0:
        return (
            torch.empty((0, H, W), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.bool, device=device)
        )

    masks_symm, scores_symm, hdl_masks, hdl_scores, ptrs_masks, ptrs_scores = _get_symm_state(
        max_N_local, H, W, device, id(group), group
    )

    if expected > 0:
        masks_symm[:expected].copy_(low_res_masks_local)
        scores_symm[:expected].copy_(obj_scores_local)

    # Safe asynchronous block waiting for peer stream writes into symmetrical UVA buffers
    hdl_masks.barrier(channel=0)

    masks_global = torch.empty((N_total, H, W), dtype=torch.float32, device=device)
    scores_global = torch.empty((N_total,), dtype=torch.float32, device=device)
    binary_masks_flat = torch.empty((N_total, H_W), dtype=torch.float32, device=device)
    areas = torch.empty((N_total,), dtype=torch.float32, device=device)
    to_suppress = torch.zeros((N_total,), dtype=torch.bool, device=device)

    offsets = [0] * (world_size + 1)
    for i in range(world_size):
        offsets[i+1] = offsets[i] + num_obj_per_gpu[i]
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)
    
    last_occluded = last_occluded.to(device=device, dtype=torch.int64)

    # Fast UVA read patch overlaps cleanly with format conversion to f32 binarization properties
    _get_ext().fetch_and_prep(
        ptrs_masks, ptrs_scores, offsets_tensor,
        masks_global, scores_global, binary_masks_flat, areas,
        N_total, H_W, world_size
    )

    if N_total > 1:
        # Standard highly optimized cublas handles precision dense pairwise comparisons fast-path 
        intersection = torch.mm(binary_masks_flat, binary_masks_flat.t())
        
        # Inline checks bounding intersections alongside suppression logic assignments
        _get_ext().compute_and_suppress(
            intersection, areas, last_occluded, to_suppress, masks_global,
            float(iou_threshold), bool(reverse), N_total, H_W, _NO_OBJ_LOGIT
        )

    return masks_global, scores_global, to_suppress