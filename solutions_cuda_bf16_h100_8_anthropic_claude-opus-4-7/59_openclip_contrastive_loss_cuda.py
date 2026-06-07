import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// Block computes a BM x BN tile of logits = scale * (img @ txt^T) + bias,
// then accumulates -logsigmoid(label * logit) into a per-block partial sum.
// Each threadblock handles one (bm, bn) tile of one peer block. Grid: (ceil(B/BM), ceil(B/BN), num_blocks_to_process).
// We launch one kernel per peer block (with its own image/text pointers and label mode).

#define BM 64
#define BN 64
#define BK 16
#define WARP_M 16
#define WARP_N 16
#define WARP_K 16

__device__ __forceinline__ float log_sigmoid_neg(float x) {
    // -logsigmoid(x) = log(1+exp(-x)) = softplus(-x)
    // numerically stable: if x >= 0: log1p(exp(-x)); else: -x + log1p(exp(x))
    if (x >= 0.f) {
        return log1pf(__expf(-x));
    } else {
        return -x + log1pf(__expf(x));
    }
}

// label_mode: 0 = diagonal (local block, label=+1 on diag, -1 off-diag)
//             1 = all negative (remote block, label=-1 everywhere)
__global__ void siglip_block_kernel(
    const __nv_bfloat16* __restrict__ img,   // [B, D]
    const __nv_bfloat16* __restrict__ txt,   // [B, D]
    int B,
    int D,
    float scale,
    float bias,
    int label_mode,
    int diag_offset,   // for local: 0; (we pass 0 always since img/txt aligned)
    float* __restrict__ partial_sum  // single float, atomicAdd
) {
    int tile_m = blockIdx.x * BM;
    int tile_n = blockIdx.y * BN;

    __shared__ __nv_bfloat16 As[BM][BK];
    __shared__ __nv_bfloat16 Bs[BN][BK];
    __shared__ float Cs[BM][BN];

    // Each block uses 4 warps = 128 threads. Tile is 64x64 = 4 warp tiles of 32x32... 
    // Use 4 warps each handling a 32x32 sub-tile via 2x2 wmma fragments of 16x16.
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int warp_row = (warp_id / 2) * 32;  // 0 or 32
    int warp_col = (warp_id % 2) * 32;  // 0 or 32

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    for (int k0 = 0; k0 < D; k0 += BK) {
        // Load A tile [BM, BK] from img[tile_m:tile_m+BM, k0:k0+BK]
        // BM*BK = 64*16 = 1024 elements; with 128 threads => 8 elements/thread
        #pragma unroll
        for (int i = tid; i < BM * BK; i += nthreads) {
            int r = i / BK;
            int c = i % BK;
            int gr = tile_m + r;
            int gc = k0 + c;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (gr < B && gc < D) v = img[gr * D + gc];
            As[r][c] = v;
        }
        // Load B tile [BN, BK] from txt[tile_n:tile_n+BN, k0:k0+BK]  (we want txt^T effectively, so we load txt rows directly and use col_major)
        #pragma unroll
        for (int i = tid; i < BN * BK; i += nthreads) {
            int r = i / BK;
            int c = i % BK;
            int gr = tile_n + r;
            int gc = k0 + c;
            __nv_bfloat16 v = __float2bfloat16(0.f);
            if (gr < B && gc < D) v = txt[gr * D + gc];
            Bs[r][c] = v;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
                // A sub-tile rows [warp_row+i*16, +16), cols [0, BK)
                wmma::load_matrix_sync(a_frag, &As[warp_row + i * 16][0], BK);
                // B is laid out as [BN, BK] (row-major). For col_major matrix_b of shape KxN,
                // we want B^T. Treating Bs as col_major with leading dim BK gives us txt rows as columns => correct.
                wmma::load_matrix_sync(b_frag, &Bs[warp_col + j * 16][0], BK);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    // Store to shared Cs
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&Cs[warp_row + i * 16][warp_col + j * 16],
                                    acc[i][j], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Reduce -logsigmoid(label * logit) over tile, with masking for valid (B,B) range
    float local_sum = 0.f;
    int total = BM * BN;
    for (int idx = tid; idx < total; idx += nthreads) {
        int r = idx / BN;
        int c = idx % BN;
        int gr = tile_m + r;
        int gc = tile_n + c;
        if (gr < B && gc < B) {
            float logit = Cs[r][c] * scale + bias;
            float label;
            if (label_mode == 0) {
                label = (gr == gc) ? 1.f : -1.f;
            } else {
                label = -1.f;
            }
            local_sum += log_sigmoid_neg(label * logit);
        }
    }

    // Block reduce
    __shared__ float sdata[32];
    // warp reduce
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, off);
    }
    if (lane == 0) sdata[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < (nthreads / 32)) ? sdata[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(mask, v, off);
        }
        if (lane == 0) {
            atomicAdd(partial_sum, v);
        }
    }
}

void launch_siglip_block(
    int64_t img_ptr,
    int64_t txt_ptr,
    int B,
    int D,
    double scale,
    double bias,
    int label_mode,
    torch::Tensor partial_sum  // float32 [1]
) {
    dim3 grid((B + BM - 1) / BM, (B + BN - 1) / BN);
    dim3 block(128);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    siglip_block_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(img_ptr),
        reinterpret_cast<const __nv_bfloat16*>(txt_ptr),
        B, D, (float)scale, (float)bias, label_mode, 0,
        partial_sum.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_siglip_block", &launch_siglip_block, "SigLIP block loss");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("siglip_ring_ext", CUDA_SRC)
    return _ext


_cache = {}

def _get_resources(B, D, dtype, device):
    key = (B, D, dtype, str(device))
    if key in _cache:
        return _cache[key]
    txt_buf = symm_mem.empty((B, D), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(txt_buf, dist.group.WORLD)
    img_buf = symm_mem.empty((B, D), device=device, dtype=dtype)
    img_hdl = symm_mem.rendezvous(img_buf, dist.group.WORLD)
    partial = torch.zeros(1, device=device, dtype=torch.float32)
    _cache[key] = (txt_buf, hdl, img_buf, img_hdl, partial)
    return _cache[key]


@torch.no_grad()
def solution(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float,
    logit_bias: float = 0.0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert image_features.is_cuda and text_features.is_cuda
    assert image_features.dtype == torch.bfloat16
    assert image_features.is_contiguous() and text_features.is_contiguous()

    grp = group or dist.group.WORLD
    rank = dist.get_rank(grp)
    world_size = dist.get_world_size(grp)

    B, D = image_features.shape
    device = image_features.device

    ext = _get_ext()
    txt_buf, txt_hdl, img_buf, img_hdl, partial = _get_resources(B, D, image_features.dtype, device)

    # Load local features into symmetric buffers
    txt_buf.copy_(text_features)
    img_buf.copy_(image_features)
    partial.zero_()

    # Synchronize across ranks so peer reads see updated buffers
    txt_hdl.barrier(channel=0)

    local_img_ptr = int(img_hdl.buffer_ptrs[rank])

    # Local block: label_mode=0 (diagonal positives)
    ext.launch_siglip_block(
        local_img_ptr,
        int(txt_hdl.buffer_ptrs[rank]),
        B, D, float(logit_scale), float(logit_bias), 0,
        partial,
    )

    # Remote blocks: read peer text via UVA pointer; label_mode=1 (all negatives)
    for offset in range(1, world_size):
        peer = (rank + offset) % world_size
        peer_txt_ptr = int(txt_hdl.buffer_ptrs[peer])
        ext.launch_siglip_block(
            local_img_ptr,
            peer_txt_ptr,
            B, D, float(logit_scale), float(logit_bias), 1,
            partial,
        )

    # Ensure peer reads complete before any rank could overwrite buffers next call
    loss = (partial / float(B)).to(image_features.dtype if image_features.dtype != torch.bfloat16 else torch.bfloat16)
    # Match reference: returns scalar in input dtype context; reference uses logits dtype (bf16)
    result = (partial.squeeze(0) / float(B)).to(torch.bfloat16)

    txt_hdl.barrier(channel=1)
    return result