from typing import Optional, Tuple

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

// Fused causal depthwise conv1d with peer boundary exchange via UVA.
// Layout of symmetric boundary buffer per rank:
//   [2, B, H, P] where slot 0 = chunk_a, slot 1 = chunk_b, P = K-1.
// We need:
//   left_ctx for chunk_a (chunk index 0): peer (rank-1)'s chunk_a (slot 0), or zeros if rank==0
//   left_ctx for chunk_b (chunk index 1): peer (rank+1)'s chunk_b (slot 1), or own chunk_a if last
//
// Output layout matches input x: [B, H, 2*S].
// For chunk c in {0,1}, output positions [c*S .. c*S+S-1] correspond to chunk c.
//
// Conv: y[b,h,t] = sum_{k=0..K-1} weight[h,0,k] * x_eff[b,h,t - (K-1) + k]
// where x_eff is left_ctx (length P) prepended to chunk c (length S).
// So for output position t in [0,S):
//   for k in [0,K-1]:
//     src_pos = t - (K-1) + k   (relative to chunk start)
//     if src_pos < 0:  read left_ctx[P + src_pos]
//     else:            read chunk[src_pos]

extern "C" __global__ void hyena_cp_conv_kernel(
    const __nv_bfloat16* __restrict__ x,        // [B, H, 2S]
    const __nv_bfloat16* __restrict__ weight,   // [H, 1, K]
    __nv_bfloat16* __restrict__ y,              // [B, H, 2S]
    // Boundary symm buffers: each is [2, B, H, P]
    const __nv_bfloat16* __restrict__ left_ctx_a,  // peer (rank-1)'s chunk_a slot, or null
    const __nv_bfloat16* __restrict__ left_ctx_b,  // peer (rank+1)'s chunk_b slot, or null
    const __nv_bfloat16* __restrict__ own_chunk_a, // for last rank: own chunk_a as ctx for b
    int B, int H, int S, int K,
    int has_prev, int has_next
) {
    int P = K - 1;
    int total_S = 2 * S;

    int b = blockIdx.z;
    int h = blockIdx.y;
    // Each block handles a tile of output positions across both chunks.
    int tile = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // We launch enough blocks to cover 2*S outputs per (b,h).
    int t_global = tile * blockSize + tid;
    if (t_global >= total_S) return;

    int c = (t_global >= S) ? 1 : 0;
    int t = (c == 1) ? (t_global - S) : t_global;

    // Load weights for this channel into registers (K small, e.g., <=128).
    // Compute conv.
    float acc = 0.0f;

    // Pointers to chunk c data
    const __nv_bfloat16* x_bh = x + ((int64_t)b * H + h) * total_S;
    const __nv_bfloat16* chunk_ptr = x_bh + c * S;

    // Determine left context pointer for this chunk
    const __nv_bfloat16* lctx = nullptr;
    bool has_ctx = true;
    if (c == 0) {
        if (has_prev) {
            // peer (rank-1)'s chunk_a slot (slot 0) at [b,h,:]
            lctx = left_ctx_a + (((int64_t)0 * B + b) * H + h) * P;
        } else {
            has_ctx = false;  // zeros
        }
    } else { // c == 1
        if (has_next) {
            // peer (rank+1)'s chunk_b slot (slot 1) at [b,h,:]
            lctx = left_ctx_b + (((int64_t)1 * B + b) * H + h) * P;
        } else {
            // last rank: use own chunk_a (slot 0 of own boundary buf)
            lctx = own_chunk_a + (((int64_t)0 * B + b) * H + h) * P;
            has_ctx = true;
        }
    }

    const __nv_bfloat16* w_h = weight + (int64_t)h * K;

    #pragma unroll 1
    for (int k = 0; k < K; ++k) {
        int src_pos = t - (K - 1) + k;
        float xv;
        if (src_pos < 0) {
            if (!has_ctx) {
                xv = 0.0f;
            } else {
                int idx = P + src_pos; // 0..P-1
                xv = __bfloat162float(lctx[idx]);
            }
        } else {
            xv = __bfloat162float(chunk_ptr[src_pos]);
        }
        float wv = __bfloat162float(w_h[k]);
        acc += xv * wv;
    }

    __nv_bfloat16* y_bh = y + ((int64_t)b * H + h) * total_S;
    y_bh[t_global] = __float2bfloat16(acc);
}


void launch_hyena_cp_conv(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor y,
    int64_t left_ctx_a_ptr,   // peer (rank-1) base ptr to symm boundary buf, or 0
    int64_t left_ctx_b_ptr,   // peer (rank+1) base ptr to symm boundary buf, or 0
    int64_t own_boundary_ptr, // own symm boundary buf base ptr
    int64_t B, int64_t H, int64_t S, int64_t K,
    int64_t has_prev, int64_t has_next
) {
    int total_S = 2 * (int)S;
    int blockSize = 128;
    int blocks_x = (total_S + blockSize - 1) / blockSize;
    dim3 grid(blocks_x, (int)H, (int)B);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const __nv_bfloat16* lctx_a = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(left_ctx_a_ptr));
    const __nv_bfloat16* lctx_b = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(left_ctx_b_ptr));
    const __nv_bfloat16* own_ptr = reinterpret_cast<const __nv_bfloat16*>(
        static_cast<uintptr_t>(own_boundary_ptr));

    hyena_cp_conv_kernel<<<grid, blockSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        lctx_a, lctx_b, own_ptr,
        (int)B, (int)H, (int)S, (int)K,
        (int)has_prev, (int)has_next
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// Pack chunk_a and chunk_b boundary patches (last P elements of each chunk) into
// the symmetric memory boundary buffer of layout [2, B, H, P].
extern "C" __global__ void pack_boundary_kernel(
    const __nv_bfloat16* __restrict__ x,  // [B, H, 2S]
    __nv_bfloat16* __restrict__ boundary, // [2, B, H, P]
    int B, int H, int S, int P
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int h = blockIdx.y;
    int b = blockIdx.z;

    int total_S = 2 * S;
    const __nv_bfloat16* x_bh = x + ((int64_t)b * H + h) * total_S;
    // chunk_a: [0..S), last P elements at [S-P..S)
    // chunk_b: [S..2S), last P elements at [2S-P..2S)
    __nv_bfloat16 va = x_bh[S - P + p];
    __nv_bfloat16 vb = x_bh[2 * S - P + p];

    // boundary[0,b,h,p] and boundary[1,b,h,p]
    int64_t stride_slot = (int64_t)B * H * P;
    int64_t off = ((int64_t)b * H + h) * P + p;
    boundary[0 * stride_slot + off] = va;
    boundary[1 * stride_slot + off] = vb;
}


void launch_pack_boundary(
    torch::Tensor x,
    torch::Tensor boundary,
    int64_t B, int64_t H, int64_t S, int64_t P
) {
    int blockSize = 64;
    int gx = ((int)P + blockSize - 1) / blockSize;
    dim3 grid(gx, (int)H, (int)B);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack_boundary_kernel<<<grid, blockSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(boundary.data_ptr<at::BFloat16>()),
        (int)B, (int)H, (int)S, (int)P
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_hyena_cp_conv", &launch_hyena_cp_conv, "Hyena CP conv1d (UVA)");
    m.def("launch_pack_boundary", &launch_pack_boundary, "Pack boundary patches");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("hyena_cp_conv_ext", CUDA_SRC)
    return _ext


_boundary_cache = {}


def _get_boundary_buf(B: int, H: int, P: int, dtype, device, group):
    key = (B, H, P, dtype, device, group)
    if key in _boundary_cache:
        return _boundary_cache[key]
    buf = symm_mem.empty((2, B, H, P), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    _boundary_cache[key] = (buf, hdl)
    return buf, hdl


@torch.no_grad()
def solution(
    x: torch.Tensor,
    weight: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    group_ranks = dist.get_process_group_ranks(group)
    group_rank = dist.get_rank(group)
    group_world_size = len(group_ranks)

    B, H, local_seq = x.shape
    S = local_seq // 2
    K = weight.shape[-1]
    P = K - 1

    x_c = x.contiguous()
    w_c = weight.contiguous()
    y = torch.empty_like(x_c)

    if P == 0 or group_world_size == 1:
        # No boundary needed, but still need causal context for chunk_b from chunk_a
        # when single rank: recv_prev_a = zeros, recv_next_b = chunk_a.
        # Use the kernel anyway with has_prev=0, has_next=0 (so chunk_b uses own chunk_a).
        if P == 0:
            # Pure pointwise; just multiply
            # weight is [H,1,1] -> scale per channel
            return F.conv1d(x_c, w_c, bias=None, stride=1, padding=0, groups=H)

        ext = _get_ext()
        # Need own chunk_a packed for chunk_b's left context
        buf, hdl = _get_boundary_buf(B, H, P, x_c.dtype, x_c.device, group)
        ext.launch_pack_boundary(x_c, buf, B, H, S, P)
        own_ptr = int(buf.data_ptr())
        ext.launch_hyena_cp_conv(
            x_c, w_c, y,
            0, 0, own_ptr,
            B, H, S, K,
            0, 0,
        )
        return y

    ext = _get_ext()

    # Pack boundary patches into symmetric memory.
    buf, hdl = _get_boundary_buf(B, H, P, x_c.dtype, x_c.device, group)
    ext.launch_pack_boundary(x_c, buf, B, H, S, P)

    # Symmetric barrier so all peers' boundary buffers are visible.
    hdl.barrier(channel=0)

    has_prev = 1 if group_rank > 0 else 0
    has_next = 1 if group_rank < group_world_size - 1 else 0

    left_ctx_a_ptr = 0
    left_ctx_b_ptr = 0
    if has_prev:
        # peer index in symm group
        peer_prev = group_rank - 1
        left_ctx_a_ptr = int(hdl.buffer_ptrs[peer_prev])
    if has_next:
        peer_next = group_rank + 1
        left_ctx_b_ptr = int(hdl.buffer_ptrs[peer_next])

    own_ptr = int(hdl.buffer_ptrs[group_rank])

    ext.launch_hyena_cp_conv(
        x_c, w_c, y,
        left_ctx_a_ptr, left_ctx_b_ptr, own_ptr,
        B, H, S, K,
        has_prev, has_next,
    )

    # Ensure remote reads have completed before next iteration overwrites buffers.
    hdl.barrier(channel=1)

    return y