"""
SAM3 all-gathered mask IoU suppression with custom CUDA + symmetric memory.

Strategy:
- Use symmetric memory for variable all-gather: each rank writes its local
  shard into its slot of a symm_mem buffer, then all peers read directly via
  UVA pointers (NVLink P2P). One barrier synchronizes producers/consumers.
- Custom CUDA kernel computes binarized mask IoU pairwise using bf16/fp32.
- Suppression decision computed in a separate kernel over the IoU matrix.
- Final scatter of NO_OBJ_LOGIT into masks_global is fused on device.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension

_NO_OBJ_LOGIT = -10.0

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Gather from peer symmetric buffers (bf16 masks) into a local contiguous buffer.
// Each peer's masks_local lives at peer_ptrs[r] with counts[r] rows of HW elements.
__global__ void gather_masks_bf16_kernel(
    const uint64_t* __restrict__ peer_ptrs,
    const int64_t* __restrict__ offsets,   // size = world_size+1, prefix sums of counts
    __nv_bfloat16* __restrict__ out,
    int64_t HW,
    int world_size
) {
    int r = blockIdx.y;
    int64_t start = offsets[r];
    int64_t end = offsets[r + 1];
    int64_t rows = end - start;
    if (rows <= 0) return;

    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[r]);
    __nv_bfloat16* dst = out + start * HW;

    int64_t total = rows * HW;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        dst[idx] = src[idx];
    }
}

// Gather scores (float32)
__global__ void gather_scores_f32_kernel(
    const uint64_t* __restrict__ peer_ptrs,
    const int64_t* __restrict__ offsets,
    float* __restrict__ out,
    int world_size
) {
    int r = blockIdx.y;
    int64_t start = offsets[r];
    int64_t end = offsets[r + 1];
    int64_t rows = end - start;
    if (rows <= 0) return;

    const float* src = reinterpret_cast<const float*>(peer_ptrs[r]);
    float* dst = out + start;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < rows; idx += stride) {
        dst[idx] = src[idx];
    }
}

// Threshold mask logits (bf16) -> binary u8, also compute per-row area.
__global__ void binarize_and_area_kernel(
    const __nv_bfloat16* __restrict__ masks,  // [N, HW]
    uint8_t* __restrict__ binary,             // [N, HW]
    int* __restrict__ areas,                  // [N]
    int N,
    int64_t HW
) {
    int row = blockIdx.y;
    if (row >= N) return;

    const __nv_bfloat16* src = masks + (int64_t)row * HW;
    uint8_t* dst = binary + (int64_t)row * HW;

    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int local_sum = 0;

    for (int64_t i = (int64_t)blockIdx.x * bsz + tid; i < HW;
         i += (int64_t)gridDim.x * bsz) {
        float v = __bfloat162float(src[i]);
        uint8_t b = v > 0.0f ? 1 : 0;
        dst[i] = b;
        local_sum += (int)b;
    }

    // Reduce within block
    __shared__ int sdata[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    // warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    }
    if (lane == 0) sdata[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        int nw = (bsz + 31) >> 5;
        int v = (lane < nw) ? sdata[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        if (lane == 0) atomicAdd(&areas[row], v);
    }
}

// Compute pairwise intersection of binary masks: out[i,j] = sum_k bin[i,k]*bin[j,k]
// One block per (i,j) tile. We use TILE x TILE outputs with TPB threads.
// For simplicity, one block computes a 16x16 tile of (i,j) over HW with reduction.
#define TILE_I 8
#define TILE_J 8
#define TPB 128

__global__ void pairwise_intersect_kernel(
    const uint8_t* __restrict__ binary,  // [N, HW]
    int* __restrict__ inter,             // [N, N]
    int N,
    int64_t HW
) {
    int bi = blockIdx.y * TILE_I;
    int bj = blockIdx.x * TILE_J;
    if (bi >= N || bj >= N) return;
    // Only compute upper triangle (bj >= bi)
    if (bj + TILE_J - 1 < bi) return;

    int tid = threadIdx.x;
    int local_acc[TILE_I][TILE_J];
    #pragma unroll
    for (int ii = 0; ii < TILE_I; ii++)
        #pragma unroll
        for (int jj = 0; jj < TILE_J; jj++)
            local_acc[ii][jj] = 0;

    for (int64_t k = tid; k < HW; k += TPB) {
        uint8_t bvals_i[TILE_I];
        uint8_t bvals_j[TILE_J];
        #pragma unroll
        for (int ii = 0; ii < TILE_I; ii++) {
            int row_i = bi + ii;
            bvals_i[ii] = (row_i < N) ? binary[(int64_t)row_i * HW + k] : 0;
        }
        #pragma unroll
        for (int jj = 0; jj < TILE_J; jj++) {
            int row_j = bj + jj;
            bvals_j[jj] = (row_j < N) ? binary[(int64_t)row_j * HW + k] : 0;
        }
        #pragma unroll
        for (int ii = 0; ii < TILE_I; ii++) {
            #pragma unroll
            for (int jj = 0; jj < TILE_J; jj++) {
                local_acc[ii][jj] += (int)(bvals_i[ii] & bvals_j[jj]);
            }
        }
    }

    // Warp reduce each accumulator
    for (int ii = 0; ii < TILE_I; ii++) {
        for (int jj = 0; jj < TILE_J; jj++) {
            int v = local_acc[ii][jj];
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffff, v, off);
            }
            __shared__ int sdata[TILE_I][TILE_J][TPB / 32];
            int lane = tid & 31;
            int warp = tid >> 5;
            if (lane == 0) sdata[ii][jj][warp] = v;
            __syncthreads();
            if (tid == 0) {
                int sum = 0;
                int nw = TPB / 32;
                for (int w = 0; w < nw; w++) sum += sdata[ii][jj][w];
                int row_i = bi + ii;
                int row_j = bj + jj;
                if (row_i < N && row_j < N && row_j >= row_i) {
                    inter[row_i * N + row_j] = sum;
                }
            }
            __syncthreads();
        }
    }
}

// Compute suppression mask from intersection matrix + areas + last_occluded.
__global__ void suppression_kernel(
    const int* __restrict__ inter,        // [N,N] upper triangle
    const int* __restrict__ areas,        // [N]
    const int64_t* __restrict__ last_occ, // [N]
    bool* __restrict__ suppress,          // [N]
    int N,
    float iou_threshold,
    int reverse
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    bool any_suppress = false;
    int area_i = areas[row];
    int64_t last_i = last_occ[row];

    // Check overlaps where row < other (suppress_i): pair (row, j), j > row
    // and where row > other (suppress_j): pair (i, row), i < row, read inter[i,row]
    for (int j = tid; j < N; j += bsz) {
        if (j == row) continue;
        int i_lo, i_hi;
        if (j > row) { i_lo = row; i_hi = j; }
        else         { i_lo = j; i_hi = row; }
        int it = inter[i_lo * N + i_hi];
        int area_j = areas[j];
        int uni = area_i + area_j - it;
        if (uni < 1) uni = 1;
        float iou = (float)it / (float)uni;
        if (iou < iou_threshold) continue;

        int64_t last_j = last_occ[j];
        bool cmp;
        if (reverse) cmp = (last_i < last_j);
        else         cmp = (last_i > last_j);
        // suppress_i: overlaps & cmp(last_i,last_j) & (last_j > -1)
        // i.e., row should be suppressed if its last_i compares above last_j
        if (cmp && last_j > -1) {
            any_suppress = true;
            break;
        }
    }

    // Reduce OR across threads
    unsigned mask = __ballot_sync(0xffffffff, any_suppress);
    __shared__ int sflag;
    if (tid == 0) sflag = 0;
    __syncthreads();
    if (mask != 0) atomicOr(&sflag, 1);
    __syncthreads();
    if (tid == 0) {
        suppress[row] = sflag != 0;
    }
}

// Apply suppression: set masks[row,*] = NO_OBJ_LOGIT (bf16) for rows where suppress[row]=true
__global__ void apply_suppression_kernel(
    __nv_bfloat16* __restrict__ masks,
    const bool* __restrict__ suppress,
    int N,
    int64_t HW,
    float no_obj_logit
) {
    int row = blockIdx.y;
    if (row >= N) return;
    if (!suppress[row]) return;
    __nv_bfloat16 v = __float2bfloat16(no_obj_logit);
    __nv_bfloat16* dst = masks + (int64_t)row * HW;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < HW; idx += stride) {
        dst[idx] = v;
    }
}

// ---------- Launchers ----------

void launch_gather_masks_bf16(
    torch::Tensor peer_ptrs,    // [W] int64
    torch::Tensor offsets,      // [W+1] int64
    torch::Tensor out,          // [N_total, HW] bf16
    int64_t HW,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks_x = 256;
    dim3 grid(blocks_x, world_size);
    gather_masks_bf16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>()),
        offsets.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        HW,
        world_size);
}

void launch_gather_scores_f32(
    torch::Tensor peer_ptrs,
    torch::Tensor offsets,
    torch::Tensor out,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 128;
    int blocks_x = 32;
    dim3 grid(blocks_x, world_size);
    gather_scores_f32_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>()),
        offsets.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        world_size);
}

void launch_binarize_and_area(
    torch::Tensor masks,    // [N, HW] bf16
    torch::Tensor binary,   // [N, HW] u8
    torch::Tensor areas,    // [N] int32
    int N,
    int64_t HW
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(areas.data_ptr<int>(), 0, N * sizeof(int), stream);
    int threads = 256;
    int blocks_x = 64;
    dim3 grid(blocks_x, N);
    binarize_and_area_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(masks.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint8_t*>(binary.data_ptr<uint8_t>()),
        areas.data_ptr<int>(),
        N, HW);
}

void launch_pairwise_intersect(
    torch::Tensor binary,
    torch::Tensor inter,
    int N,
    int64_t HW
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(inter.data_ptr<int>(), 0, (int64_t)N * N * sizeof(int), stream);
    int gx = (N + TILE_J - 1) / TILE_J;
    int gy = (N + TILE_I - 1) / TILE_I;
    dim3 grid(gx, gy);
    pairwise_intersect_kernel<<<grid, TPB, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(binary.data_ptr<uint8_t>()),
        inter.data_ptr<int>(),
        N, HW);
}

void launch_suppression(
    torch::Tensor inter,
    torch::Tensor areas,
    torch::Tensor last_occ,
    torch::Tensor suppress,
    int N,
    double iou_threshold,
    bool reverse
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(suppress.data_ptr<bool>(), 0, N, stream);
    suppression_kernel<<<N, 32, 0, stream>>>(
        inter.data_ptr<int>(),
        areas.data_ptr<int>(),
        last_occ.data_ptr<int64_t>(),
        suppress.data_ptr<bool>(),
        N,
        (float)iou_threshold,
        reverse ? 1 : 0);
}

void launch_apply_suppression(
    torch::Tensor masks,
    torch::Tensor suppress,
    int N,
    int64_t HW,
    double no_obj_logit
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks_x = 64;
    dim3 grid(blocks_x, N);
    apply_suppression_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(masks.data_ptr<at::BFloat16>()),
        suppress.data_ptr<bool>(),
        N, HW, (float)no_obj_logit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_masks_bf16", &launch_gather_masks_bf16);
    m.def("launch_gather_scores_f32", &launch_gather_scores_f32);
    m.def("launch_binarize_and_area", &launch_binarize_and_area);
    m.def("launch_pairwise_intersect", &launch_pairwise_intersect);
    m.def("launch_suppression", &launch_suppression);
    m.def("launch_apply_suppression", &launch_apply_suppression);
}
"""

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("sam3_iou_suppress_ext", CUDA_SRC)
    return _ext


_mask_buf = None
_mask_hdl = None
_mask_buf_capacity = 0
_mask_buf_HW = 0

_score_buf = None
_score_hdl = None
_score_buf_capacity = 0


def _get_mask_buf(max_n: int, HW: int, device):
    global _mask_buf, _mask_hdl, _mask_buf_capacity, _mask_buf_HW
    if (_mask_buf is None) or (max_n > _mask_buf_capacity) or (HW != _mask_buf_HW):
        _mask_buf = symm_mem.empty((max_n, HW), device=device, dtype=torch.bfloat16)
        _mask_hdl = symm_mem.rendezvous(_mask_buf, dist.group.WORLD)
        _mask_buf_capacity = max_n
        _mask_buf_HW = HW
    return _mask_buf, _mask_hdl


def _get_score_buf(max_n: int, device):
    global _score_buf, _score_hdl, _score_buf_capacity
    if (_score_buf is None) or (max_n > _score_buf_capacity):
        _score_buf = symm_mem.empty((max_n,), device=device, dtype=torch.float32)
        _score_hdl = symm_mem.rendezvous(_score_buf, dist.group.WORLD)
        _score_buf_capacity = max_n
    return _score_buf, _score_hdl


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
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    expected = int(num_obj_per_gpu[rank])
    if low_res_masks_local.shape[0] != expected:
        raise ValueError("local mask count does not match num_obj_per_gpu")
    if obj_scores_local.shape[0] != expected:
        raise ValueError("local score count does not match num_obj_per_gpu")

    device = low_res_masks_local.device

    counts = [int(c) for c in num_obj_per_gpu]
    N_total = sum(counts)
    max_n = max(counts) if counts else 1
    max_n = max(max_n, 1)

    # Determine HW
    if low_res_masks_local.dim() >= 2:
        H = low_res_masks_local.shape[1] if low_res_masks_local.dim() >= 2 else 1
        W = low_res_masks_local.shape[2] if low_res_masks_local.dim() >= 3 else 1
        HW = int(low_res_masks_local.numel() // max(expected, 1)) if expected > 0 else (H * W)
        out_shape_tail = tuple(low_res_masks_local.shape[1:])
    else:
        HW = 1
        out_shape_tail = ()

    if expected == 0:
        # Need HW from broadcast — assume rank 0 has it. Fallback: use 1.
        out_shape_tail = tuple(low_res_masks_local.shape[1:]) if low_res_masks_local.dim() >= 1 else ()

    # Load extension before any peer access (compile once)
    ext = _get_ext()
    dist.barrier(group=group)

    # Allocate symmetric buffers
    mask_buf, mask_hdl = _get_mask_buf(max_n, HW, device)
    score_buf, score_hdl = _get_score_buf(max_n, device)

    # Stage local data into symm_mem (convert masks to bf16)
    if expected > 0:
        masks_local_bf16 = low_res_masks_local.contiguous().to(torch.bfloat16).reshape(expected, HW)
        mask_buf[:expected].copy_(masks_local_bf16)
        score_buf[:expected].copy_(obj_scores_local.contiguous().to(torch.float32))

    # Synchronize across peers
    mask_hdl.barrier(channel=0)
    score_hdl.barrier(channel=1)

    # Compute peer pointers
    mask_peer_ptrs = torch.tensor(
        [int(p) for p in mask_hdl.buffer_ptrs], device=device, dtype=torch.int64
    )
    score_peer_ptrs = torch.tensor(
        [int(p) for p in score_hdl.buffer_ptrs], device=device, dtype=torch.int64
    )

    offsets_list = [0]
    for c in counts:
        offsets_list.append(offsets_list[-1] + c)
    offsets = torch.tensor(offsets_list, device=device, dtype=torch.int64)

    # Allocate outputs
    masks_global_bf16 = torch.empty((max(N_total, 1), HW), device=device, dtype=torch.bfloat16)
    scores_global = torch.empty((max(N_total, 1),), device=device, dtype=torch.float32)

    if N_total > 0:
        ext.launch_gather_masks_bf16(mask_peer_ptrs, offsets, masks_global_bf16, HW, world_size)
        ext.launch_gather_scores_f32(score_peer_ptrs, offsets, scores_global, world_size)

    masks_global_bf16 = masks_global_bf16[:N_total]
    scores_global = scores_global[:N_total]

    # Suppression
    to_suppress = torch.zeros(N_total, dtype=torch.bool, device=device)

    if N_total > 1:
        binary = torch.empty((N_total, HW), device=device, dtype=torch.uint8)
        areas = torch.empty((N_total,), device=device, dtype=torch.int32)
        ext.launch_binarize_and_area(masks_global_bf16, binary, areas, N_total, HW)

        inter = torch.empty((N_total, N_total), device=device, dtype=torch.int32)
        ext.launch_pairwise_intersect(binary, inter, N_total, HW)

        last_occ_long = last_occluded.to(device=device, dtype=torch.int64).contiguous()
        ext.launch_suppression(
            inter, areas, last_occ_long, to_suppress, N_total, float(iou_threshold), bool(reverse)
        )

        # Apply suppression to bf16 masks (then cast back)
        ext.launch_apply_suppression(masks_global_bf16, to_suppress, N_total, HW, _NO_OBJ_LOGIT)

    # Reshape masks back to [N_total, *tail]
    if N_total > 0 and len(out_shape_tail) > 0:
        masks_global = masks_global_bf16.float().reshape((N_total,) + out_shape_tail)
    elif N_total > 0:
        masks_global = masks_global_bf16.float().reshape(N_total, HW)
    else:
        masks_global = torch.empty((0,) + out_shape_tail, device=device, dtype=torch.float32)
        scores_global = torch.empty((0,), device=device, dtype=torch.float32)

    return masks_global, scores_global, to_suppress