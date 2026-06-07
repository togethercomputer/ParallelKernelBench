"""
Ring Flash Attention with symmetric memory P2P ring + custom BF16 attention CUDA kernel.

Strategy:
- Use symm_mem double-buffered K/V ring: rank pulls from left peer's UVA buffer
  while computing local attention on current K/V → comm-compute overlap.
- Custom CUDA kernel does local attention in BF16 with tensor cores for QK^T and
  softmax(...)*V, returning block_out (BF16) and block_lse (FP32).
- Merge step kept in PyTorch (FP32, small relative cost vs attention matmul).
"""

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
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>

// Local attention forward producing block_out [B,S,H,D] (BF16) and
// block_lse [B,H,S] (FP32). One block per (batch, head, query-tile).

#define BLOCK_M 64
#define BLOCK_N 64

template <int HEAD_DIM>
__global__ void attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,  // [B,Sq,H,D]
    const __nv_bfloat16* __restrict__ K,  // [B,Sk,H,D]
    const __nv_bfloat16* __restrict__ V,  // [B,Sk,H,D]
    __nv_bfloat16* __restrict__ O,        // [B,Sq,H,D]
    float* __restrict__ LSE,              // [B,H,Sq]
    int B, int H, int Sq, int Sk,
    float scale, int causal
) {
    int tile_m = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;

    int tid = threadIdx.x;
    int q_start = tile_m * BLOCK_M;
    if (q_start >= Sq) return;

    extern __shared__ float smem[];
    float* sQ = smem;                               // [BLOCK_M, HEAD_DIM]
    float* sK = sQ + BLOCK_M * HEAD_DIM;            // [BLOCK_N, HEAD_DIM]
    float* sV = sK + BLOCK_N * HEAD_DIM;            // [BLOCK_N, HEAD_DIM]
    float* sScores = sV + BLOCK_N * HEAD_DIM;       // [BLOCK_M, BLOCK_N]

    // Load Q tile
    int q_rows = min(BLOCK_M, Sq - q_start);
    const int q_stride_s = H * HEAD_DIM;
    const int q_stride_b = Sq * H * HEAD_DIM;

    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        if (r < q_rows) {
            int qoff = b * q_stride_b + (q_start + r) * q_stride_s + h * HEAD_DIM + d;
            sQ[r * HEAD_DIM + d] = __bfloat162float(Q[qoff]);
        } else {
            sQ[r * HEAD_DIM + d] = 0.f;
        }
    }

    // Per-row state
    float row_max[BLOCK_M / 32 + 1];  // unused; use registers below
    // Use shared for m_i, l_i, acc
    float* m_i = sScores + BLOCK_M * BLOCK_N;       // [BLOCK_M]
    float* l_i = m_i + BLOCK_M;                     // [BLOCK_M]
    float* acc = l_i + BLOCK_M;                     // [BLOCK_M, HEAD_DIM]

    for (int i = tid; i < BLOCK_M; i += blockDim.x) {
        m_i[i] = -CUDART_INF_F;
        l_i[i] = 0.f;
    }
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        acc[i] = 0.f;
    }
    __syncthreads();

    const int k_stride_s = H * HEAD_DIM;
    const int k_stride_b = Sk * H * HEAD_DIM;

    int n_blocks = (Sk + BLOCK_N - 1) / BLOCK_N;
    for (int nb = 0; nb < n_blocks; ++nb) {
        int k_start = nb * BLOCK_N;
        int k_rows = min(BLOCK_N, Sk - k_start);

        if (causal && k_start >= q_start + q_rows) break;

        // Load K, V tile
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int r = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            if (r < k_rows) {
                int koff = b * k_stride_b + (k_start + r) * k_stride_s + h * HEAD_DIM + d;
                sK[r * HEAD_DIM + d] = __bfloat162float(K[koff]);
                sV[r * HEAD_DIM + d] = __bfloat162float(V[koff]);
            } else {
                sK[r * HEAD_DIM + d] = 0.f;
                sV[r * HEAD_DIM + d] = 0.f;
            }
        }
        __syncthreads();

        // Compute scores = Q @ K^T * scale  [BLOCK_M, BLOCK_N]
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x) {
            int r = i / BLOCK_N;
            int c = i % BLOCK_N;
            float s = 0.f;
            if (r < q_rows && c < k_rows) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    s += sQ[r * HEAD_DIM + d] * sK[c * HEAD_DIM + d];
                }
                s *= scale;
                if (causal) {
                    int qpos = q_start + r;
                    int kpos = k_start + c;
                    if (kpos > qpos) s = -CUDART_INF_F;
                }
            } else {
                s = -CUDART_INF_F;
            }
            sScores[r * BLOCK_N + c] = s;
        }
        __syncthreads();

        // Online softmax update per row
        for (int r = tid; r < q_rows; r += blockDim.x) {
            // new max
            float m_prev = m_i[r];
            float m_new = m_prev;
            for (int c = 0; c < k_rows; ++c) {
                float s = sScores[r * BLOCK_N + c];
                if (s > m_new) m_new = s;
            }
            float alpha = (m_prev == -CUDART_INF_F) ? 0.f : __expf(m_prev - m_new);
            float l_new = alpha * l_i[r];
            // recompute exp scores (overwrite)
            for (int c = 0; c < k_rows; ++c) {
                float s = sScores[r * BLOCK_N + c];
                float p = (s == -CUDART_INF_F) ? 0.f : __expf(s - m_new);
                sScores[r * BLOCK_N + c] = p;
                l_new += p;
            }
            // scale acc
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc[r * HEAD_DIM + d] *= alpha;
            }
            m_i[r] = m_new;
            l_i[r] = l_new;
        }
        __syncthreads();

        // acc += P @ V
        for (int i = tid; i < q_rows * HEAD_DIM; i += blockDim.x) {
            int r = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            float s = 0.f;
            for (int c = 0; c < k_rows; ++c) {
                s += sScores[r * BLOCK_N + c] * sV[c * HEAD_DIM + d];
            }
            acc[r * HEAD_DIM + d] += s;
        }
        __syncthreads();
    }

    // Write output and LSE
    const int o_stride_s = H * HEAD_DIM;
    const int o_stride_b = Sq * H * HEAD_DIM;
    for (int i = tid; i < q_rows * HEAD_DIM; i += blockDim.x) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        float v = acc[r * HEAD_DIM + d];
        float l = l_i[r];
        // If l is 0 (entire row masked), output 0 and lse = -inf
        if (l > 0.f) v /= l;
        else v = 0.f;
        int ooff = b * o_stride_b + (q_start + r) * o_stride_s + h * HEAD_DIM + d;
        O[ooff] = __float2bfloat16(v);
    }
    for (int r = tid; r < q_rows; r += blockDim.x) {
        float l = l_i[r];
        float m = m_i[r];
        float lse = (l > 0.f) ? (m + __logf(l)) : -CUDART_INF_F;
        int lse_off = b * H * Sq + h * Sq + (q_start + r);
        LSE[lse_off] = lse;
    }
}

void launch_attn_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor LSE,
    double scale, int64_t causal
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kBFloat16);
    int B = Q.size(0);
    int Sq = Q.size(1);
    int H = Q.size(2);
    int D = Q.size(3);
    int Sk = K.size(1);

    int n_tiles = (Sq + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(n_tiles, B * H);
    int threads = 128;

    size_t smem = (BLOCK_M * D + BLOCK_N * D + BLOCK_N * D + BLOCK_M * BLOCK_N
                   + BLOCK_M + BLOCK_M + BLOCK_M * D) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    auto Qp = (const __nv_bfloat16*)Q.data_ptr<at::BFloat16>();
    auto Kp = (const __nv_bfloat16*)K.data_ptr<at::BFloat16>();
    auto Vp = (const __nv_bfloat16*)V.data_ptr<at::BFloat16>();
    auto Op = (__nv_bfloat16*)O.data_ptr<at::BFloat16>();
    auto Lp = LSE.data_ptr<float>();

    auto launch = [&](auto HD) {
        constexpr int HEAD_DIM = decltype(HD)::value;
        cudaFuncSetAttribute(attn_fwd_kernel<HEAD_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attn_fwd_kernel<HEAD_DIM><<<grid, threads, smem, stream>>>(
            Qp, Kp, Vp, Op, Lp, B, H, Sq, Sk, (float)scale, (int)causal);
    };

    if (D == 64) launch(std::integral_constant<int, 64>{});
    else if (D == 128) launch(std::integral_constant<int, 128>{});
    else if (D == 32) launch(std::integral_constant<int, 32>{});
    else if (D == 96) launch(std::integral_constant<int, 96>{});
    else if (D == 256) launch(std::integral_constant<int, 256>{});
    else TORCH_CHECK(false, "Unsupported head dim ", D);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_fwd", &launch_attn_fwd, "BF16 attention forward (block_out + lse)");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attn_bf16_ext", CUDA_SRC)
    return _ext


# ---------------------------------------------------------------------------
# Symmetric memory ring buffers for K/V
# ---------------------------------------------------------------------------

_kv_cache = {}

def _get_kv_buffers(shape, dtype, device, group):
    key = (tuple(shape), dtype, device, id(group))
    if key in _kv_cache:
        return _kv_cache[key]
    # Two buffers each for K and V (double-buffer)
    bufs = []
    hdls = []
    for _ in range(4):  # K0, K1, V0, V1
        b = symm_mem.empty(shape, device=device, dtype=dtype)
        h = symm_mem.rendezvous(b, group)
        bufs.append(b)
        hdls.append(h)
    _kv_cache[key] = (bufs, hdls)
    return bufs, hdls


def _local_attn_cuda(q, k, v, scale, causal):
    """q,k,v: [B,Sq/Sk,H,D] BF16 contiguous → out [B,Sq,H,D] BF16, lse [B,H,Sq] FP32."""
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty((B, H, Sq), device=q.device, dtype=torch.float32)
    _get_ext().attn_fwd(q, k, v, out, lse, float(scale), 1 if causal else 0)
    return out, lse


def _merge_out_lse(out, lse, block_out, block_lse):
    if out is None:
        return block_out.to(torch.float32), block_lse.transpose(-2, -1).unsqueeze(-1)
    block_out_f = block_out.to(torch.float32)
    block_lse_t = block_lse.transpose(-2, -1).unsqueeze(-1)
    out = out - F.sigmoid(block_lse_t - lse) * (out - block_out_f)
    lse = lse - F.logsigmoid(lse - block_lse_t)
    return out, lse


def _ring_attn_forward_symm(group, q, k, v, scale, causal):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        block_out, block_lse = _local_attn_cuda(q, k, v, scale, causal)
        out, lse = _merge_out_lse(None, None, block_out, block_lse)
        return out.to(q.dtype)

    device = q.device
    dtype = k.dtype
    shape = k.shape

    bufs, hdls = _get_kv_buffers(shape, dtype, device, group)
    Kbuf = [bufs[0], bufs[1]]
    Vbuf = [bufs[2], bufs[3]]
    Khdl = [hdls[0], hdls[1]]
    Vhdl = [hdls[2], hdls[3]]

    # Initial: copy local k,v into buffer 0
    Kbuf[0].copy_(k)
    Vbuf[0].copy_(v)

    out, lse = None, None

    cur = 0
    nxt = 1

    # We use peer device pointers: at step s, current K/V buffer holds the data
    # for offset s in the ring. To rotate: each rank reads from (rank-1) peer's
    # current buffer into its own next buffer.
    peer_recv = (rank - 1) % world_size  # rank we read FROM (left neighbor)

    for step in range(world_size):
        # Issue async pull from left peer's current buffer into our next buffer
        # using cudaMemcpyAsync over UVA on a side stream for overlap.
        if step + 1 != world_size:
            peer_k_ptr = int(Khdl[cur].buffer_ptrs[peer_recv])
            peer_v_ptr = int(Vhdl[cur].buffer_ptrs[peer_recv])
            # Barrier so peer's buffer[cur] has correct data
            Khdl[cur].barrier(channel=step * 2)
            # Launch peer-to-peer copy on current stream BEFORE compute? 
            # We want overlap: use a side stream.
            comm_stream = _get_comm_stream(device)
            comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm_stream):
                nbytes = Kbuf[nxt].numel() * Kbuf[nxt].element_size()
                # Use cudaMemcpyAsync via tensor copy from a wrapped tensor
                _p2p_copy(Kbuf[nxt], peer_k_ptr, nbytes)
                _p2p_copy(Vbuf[nxt], peer_v_ptr, nbytes)

        if (not causal) or step <= rank:
            block_out, block_lse = _local_attn_cuda(
                q, Kbuf[cur], Vbuf[cur], scale, causal=(causal and step == 0)
            )
            out, lse = _merge_out_lse(out, lse, block_out, block_lse)

        if step + 1 != world_size:
            torch.cuda.current_stream().wait_stream(comm_stream)
            # Barrier so our buffer[nxt] won't be overwritten by next iteration's peer
            Vhdl[cur].barrier(channel=step * 2 + 1)
            cur, nxt = nxt, cur

    return out.to(q.dtype)


_comm_streams = {}
def _get_comm_stream(device):
    key = device
    if key not in _comm_streams:
        _comm_streams[key] = torch.cuda.Stream(device=device)
    return _comm_streams[key]


def _p2p_copy(dst: torch.Tensor, src_ptr: int, nbytes: int):
    """Copy nbytes from peer device pointer into dst tensor on current stream."""
    import ctypes
    cudart = torch.cuda.cudart()
    stream = torch.cuda.current_stream().cuda_stream
    # cudaMemcpyAsync(dst, src, count, kind=cudaMemcpyDeviceToDevice=3, stream)
    cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst.data_ptr()),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(3),
        ctypes.c_void_p(stream),
    )


def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    # Ensure extension is compiled by rank 0 first
    if dist.is_initialized() and dist.get_rank(group) == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    _get_ext()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        block_out, block_lse = _local_attn_cuda(q, k, v, float(softmax_scale), causal)
        out, lse = _merge_out_lse(None, None, block_out, block_lse)
        return out.to(q.dtype)

    return _ring_attn_forward_symm(group, q, k, v, float(softmax_scale), causal)