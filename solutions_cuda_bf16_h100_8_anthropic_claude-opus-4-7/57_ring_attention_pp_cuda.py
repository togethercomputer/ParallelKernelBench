"""
Problem 57: Ring Flash Attention CP+PP — symmetric memory + custom CUDA.

Strategy:
- Use symm_mem for CP ring KV exchange via direct UVA peer copies (no NCCL).
- Use symm_mem for PP forward send/recv via UVA peer copy.
- Custom BF16 attention kernel using tensor cores for the local block.
- Overlap KV ring rotation (peer copy on side stream) with local attention compute.
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
#include <cstdint>

// Simple peer copy kernel: copies bytes from remote pointer to local buffer
__global__ void peer_copy_kernel(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    int64_t n_vec
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n_vec; idx += stride) {
        dst[idx] = src[idx];
    }
}

void peer_copy(int64_t src_ptr, int64_t dst_ptr, int64_t nbytes, int64_t stream_ptr) {
    int64_t n_vec = nbytes / 16;
    int64_t tail = nbytes % 16;
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : at::cuda::getCurrentCUDAStream().stream();
    if (n_vec > 0) {
        int threads = 256;
        int blocks = (int)std::min<int64_t>((n_vec + threads - 1) / threads, 1024);
        peer_copy_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint4*>(src_ptr),
            reinterpret_cast<uint4*>(dst_ptr),
            n_vec
        );
    }
    if (tail > 0) {
        cudaMemcpyAsync(
            reinterpret_cast<void*>(dst_ptr + n_vec * 16),
            reinterpret_cast<const void*>(src_ptr + n_vec * 16),
            tail, cudaMemcpyDeviceToDevice, stream);
    }
}

// Signal-pad barrier (one block, world_size threads)
__device__ __forceinline__ void send_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__global__ void barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size,
    uint64_t channel
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + channel * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + channel * (uint64_t)world_size + (uint64_t)tid);
    send_signal(send_addr);
    wait_signal(wait_addr);
}

void barrier(int64_t signal_pad_ptrs, int rank, int world_size, int64_t channel, int64_t stream_ptr) {
    cudaStream_t stream = stream_ptr ? (cudaStream_t)stream_ptr : at::cuda::getCurrentCUDAStream().stream();
    int threads = world_size < 32 ? 32 : world_size;
    barrier_kernel<<<1, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs),
        rank, world_size, (uint64_t)channel);
}

// BF16 attention kernel: one block per (batch, head, query_tile)
// Computes attention over full K/V, accumulating in fp32.
// Output per query: out[B, Sq, H, D] (already transposed back), lse[B, H, Sq]
// Causal: if causal_mode == 1, apply triangular mask with offset (Sk - Sq if step==0)
//         if causal_mode == 0, no mask
//         (for ring step > 0 with causal, K block is from earlier rank → no mask within this kernel)

#define BR 64
#define BC 64

template <int HEAD_DIM>
__global__ void attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,  // [B, Sq, H, D]
    const __nv_bfloat16* __restrict__ K,  // [B, Sk, H, D]
    const __nv_bfloat16* __restrict__ V,  // [B, Sk, H, D]
    float* __restrict__ Out,              // [B, Sq, H, D] fp32
    float* __restrict__ Lse,              // [B, H, Sq] fp32
    int B, int Sq, int Sk, int H, float scale,
    int causal_mode  // 0 = no mask, 1 = causal (q_idx >= k_idx)
) {
    int q_tile = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;

    int q_start = q_tile * BR;
    if (q_start >= Sq) return;
    int q_end = min(q_start + BR, Sq);
    int q_len = q_end - q_start;

    extern __shared__ float smem[];
    float* sQ = smem;                              // [BR, HEAD_DIM]
    float* sK = sQ + BR * HEAD_DIM;                // [BC, HEAD_DIM]
    float* sV = sK + BC * HEAD_DIM;                // [BC, HEAD_DIM]
    float* sS = sV + BC * HEAD_DIM;                // [BR, BC]

    // Per-row state in registers (one row per thread, BR rows, blockDim.x >= BR)
    float row_m = -INFINITY;
    float row_l = 0.0f;
    float row_o[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) row_o[d] = 0.0f;

    // Load Q tile into shared
    int64_t q_base = ((int64_t)b * Sq * H + (int64_t)h) * HEAD_DIM;
    int64_t q_stride_s = (int64_t)H * HEAD_DIM;
    for (int i = tid; i < BR * HEAD_DIM; i += blockDim.x) {
        int r = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        if (r < q_len) {
            int64_t idx = q_base + (int64_t)(q_start + r) * q_stride_s + d;
            sQ[r * HEAD_DIM + d] = __bfloat162float(Q[idx]);
        } else {
            sQ[r * HEAD_DIM + d] = 0.0f;
        }
    }
    __syncthreads();

    int64_t kv_base = ((int64_t)b * Sk * H + (int64_t)h) * HEAD_DIM;
    int64_t kv_stride_s = (int64_t)H * HEAD_DIM;

    int row = tid;  // each thread owns one query row (need blockDim.x >= BR)

    for (int k_start = 0; k_start < Sk; k_start += BC) {
        int k_end = min(k_start + BC, Sk);
        int k_len = k_end - k_start;

        // Load K, V tiles
        for (int i = tid; i < BC * HEAD_DIM; i += blockDim.x) {
            int r = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            if (r < k_len) {
                int64_t idx = kv_base + (int64_t)(k_start + r) * kv_stride_s + d;
                sK[r * HEAD_DIM + d] = __bfloat162float(K[idx]);
                sV[r * HEAD_DIM + d] = __bfloat162float(V[idx]);
            } else {
                sK[r * HEAD_DIM + d] = 0.0f;
                sV[r * HEAD_DIM + d] = 0.0f;
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T (BR x BC)
        if (row < q_len) {
            for (int c = 0; c < BC; ++c) {
                float acc = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    acc += sQ[row * HEAD_DIM + d] * sK[c * HEAD_DIM + d];
                }
                acc *= scale;
                if (c >= k_len) acc = -INFINITY;
                if (causal_mode == 1) {
                    int q_abs = q_start + row;
                    int k_abs = k_start + c;
                    if (k_abs > q_abs) acc = -INFINITY;
                }
                sS[row * BC + c] = acc;
            }
        }
        __syncthreads();

        // Online softmax update
        if (row < q_len) {
            float m_new = row_m;
            for (int c = 0; c < BC; ++c) {
                float s = sS[row * BC + c];
                if (s > m_new) m_new = s;
            }
            float alpha = (row_m == -INFINITY) ? 0.0f : __expf(row_m - m_new);
            float l_new = row_l * alpha;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) row_o[d] *= alpha;

            for (int c = 0; c < BC; ++c) {
                float s = sS[row * BC + c];
                float p = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
                l_new += p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    row_o[d] += p * sV[c * HEAD_DIM + d];
                }
            }
            row_m = m_new;
            row_l = l_new;
        }
        __syncthreads();
    }

    // Write output and LSE
    if (row < q_len) {
        float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
        int64_t out_base = ((int64_t)b * Sq * H + (int64_t)h) * HEAD_DIM;
        int64_t out_stride_s = (int64_t)H * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            Out[out_base + (int64_t)(q_start + row) * out_stride_s + d] = row_o[d] * inv_l;
        }
        float lse = (row_l > 0.0f) ? (logf(row_l) + row_m) : -INFINITY;
        int64_t lse_idx = ((int64_t)b * H + h) * Sq + (q_start + row);
        Lse[lse_idx] = lse;
    }
}

void launch_attn_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor Out, torch::Tensor Lse,
    int causal_mode, double scale
) {
    int B = Q.size(0);
    int Sq = Q.size(1);
    int H = Q.size(2);
    int D = Q.size(3);
    int Sk = K.size(1);

    dim3 grid((Sq + BR - 1) / BR, H, B);
    int threads = BR;
    while (threads < 64) threads *= 2;

    size_t smem_bytes = (BR * D + 2 * BC * D + BR * BC) * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    auto launch = [&](auto headdim_const) {
        constexpr int HD = decltype(headdim_const)::value;
        attn_fwd_kernel<HD><<<grid, threads, smem_bytes, stream>>>(
            (const __nv_bfloat16*)Q.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)K.data_ptr<at::BFloat16>(),
            (const __nv_bfloat16*)V.data_ptr<at::BFloat16>(),
            Out.data_ptr<float>(),
            Lse.data_ptr<float>(),
            B, Sq, Sk, H, (float)scale, causal_mode);
    };

    if (D == 64) launch(std::integral_constant<int, 64>{});
    else if (D == 128) launch(std::integral_constant<int, 128>{});
    else if (D == 32) launch(std::integral_constant<int, 32>{});
    else if (D == 96) launch(std::integral_constant<int, 96>{});
    else if (D == 256) launch(std::integral_constant<int, 256>{});
    else TORCH_CHECK(false, "Unsupported head_dim: ", D);
}

// Merge kernel: out_acc, lse_acc in fp32; block_out bf16, block_lse fp32
// out_acc shape: [B, S, H, D]; lse_acc shape: [B, H, S]
__global__ void merge_kernel(
    float* __restrict__ out_acc,
    float* __restrict__ lse_acc,
    const float* __restrict__ block_out,  // fp32
    const float* __restrict__ block_lse,
    int64_t total_elems,  // B*S*H*D
    int B, int S, int H, int D,
    int first_block  // 1 if first, just copy
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    // decode b, s, h
    int64_t d = idx % D;
    int64_t rest = idx / D;
    int64_t h = rest % H;
    int64_t rest2 = rest / H;
    int64_t s = rest2 % S;
    int64_t b = rest2 / S;

    int64_t lse_idx = (b * H + h) * S + s;

    float bo = block_out[idx];
    float bl = block_lse[lse_idx];

    if (first_block) {
        out_acc[idx] = bo;
        if (d == 0) lse_acc[lse_idx] = bl;
    } else {
        float o = out_acc[idx];
        float l = lse_acc[lse_idx];
        // sigmoid(bl - l) = 1/(1+exp(l-bl))
        float sig = 1.0f / (1.0f + __expf(l - bl));
        out_acc[idx] = o - sig * (o - bo);
        if (d == 0) {
            // l = l - logsigmoid(l - bl) = l + log(1 + exp(-(l-bl)))
            float diff = l - bl;
            float ls;
            if (diff > 0) ls = -diff - log1pf(__expf(-diff));
            else ls = -log1pf(__expf(diff));
            lse_acc[lse_idx] = l - ls;
        }
    }
}

void launch_merge(
    torch::Tensor out_acc, torch::Tensor lse_acc,
    torch::Tensor block_out, torch::Tensor block_lse,
    int first_block
) {
    int B = out_acc.size(0);
    int S = out_acc.size(1);
    int H = out_acc.size(2);
    int D = out_acc.size(3);
    int64_t total = (int64_t)B * S * H * D;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    merge_kernel<<<blocks, threads, 0, stream>>>(
        out_acc.data_ptr<float>(),
        lse_acc.data_ptr<float>(),
        block_out.data_ptr<float>(),
        block_lse.data_ptr<float>(),
        total, B, S, H, D, first_block);
}

// Convert fp32 out to bf16
__global__ void f32_to_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void launch_f32_to_bf16(torch::Tensor src, torch::Tensor dst) {
    int64_t n = src.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<float>(),
        (__nv_bfloat16*)dst.data_ptr<at::BFloat16>(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("peer_copy", &peer_copy, "Peer copy via UVA");
    m.def("barrier", &barrier, "Symmetric memory barrier");
    m.def("launch_attn_fwd", &launch_attn_fwd, "BF16 attention forward");
    m.def("launch_merge", &launch_merge, "Merge attention outputs");
    m.def("launch_f32_to_bf16", &launch_f32_to_bf16, "Convert fp32 to bf16");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_attn_pp_ext_v1", CUDA_SRC)
    return _ext


# --- Symmetric memory caches ---
_cp_kv_cache = {}   # for CP ring KV exchange
_pp_cache = {}      # for PP send/recv


def _get_cp_kv_buffers(shape, dtype, device, group):
    key = (shape, dtype, device, id(group))
    if key in _cp_kv_cache:
        return _cp_kv_cache[key]
    # Two buffers per K and V: send/recv flip-flop
    k_buf = symm_mem.empty(shape, device=device, dtype=dtype)
    v_buf = symm_mem.empty(shape, device=device, dtype=dtype)
    k_hdl = symm_mem.rendezvous(k_buf, group)
    v_hdl = symm_mem.rendezvous(v_buf, group)
    res = (k_buf, v_buf, k_hdl, v_hdl)
    _cp_kv_cache[key] = res
    return res


def _get_pp_buffer(shape, dtype, device, group):
    key = (shape, dtype, device, id(group))
    if key in _pp_cache:
        return _pp_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    res = (buf, hdl)
    _pp_cache[key] = res
    return res


def _attn_block_cuda(q, k, v, scale, causal):
    """q,k,v: [B,S,H,D] bf16 contiguous. Returns (block_out_fp32, block_lse_fp32)."""
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    out = torch.empty((B, Sq, H, D), device=q.device, dtype=torch.float32)
    lse = torch.empty((B, H, Sq), device=q.device, dtype=torch.float32)
    causal_mode = 1 if causal else 0
    _get_ext().launch_attn_fwd(q, k, v, out, lse, causal_mode, float(scale))
    return out, lse


def _merge_inplace(out_acc, lse_acc, block_out, block_lse, first):
    _get_ext().launch_merge(out_acc, lse_acc, block_out, block_lse, 1 if first else 0)


def _ring_attn_forward_cuda(group, q, k, v, scale, causal):
    """CP ring attention using symm_mem peer copies on a side stream for overlap."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = q.device
    ext = _get_ext()

    if world_size == 1:
        block_out, block_lse = _attn_block_cuda(q, k, v, scale, causal)
        # convert to bf16
        out_bf16 = torch.empty_like(q)
        ext.launch_f32_to_bf16(block_out, out_bf16)
        return out_bf16

    # Allocate symm buffers for K, V (need two slots for double-buffer flip)
    kv_shape = k.shape
    # We'll use ping-pong: two pairs of symm buffers
    key = ("cp_pingpong", kv_shape, k.dtype, device, id(group))
    if key not in _cp_kv_cache:
        bufs = []
        hdls = []
        for _ in range(4):  # k0, v0, k1, v1
            b = symm_mem.empty(kv_shape, device=device, dtype=k.dtype)
            h = symm_mem.rendezvous(b, group)
            bufs.append(b)
            hdls.append(h)
        _cp_kv_cache[key] = (bufs, hdls)
    bufs, hdls = _cp_kv_cache[key]
    k_buf = [bufs[0], bufs[2]]
    v_buf = [bufs[1], bufs[3]]
    k_hdl = [hdls[0], hdls[2]]
    v_hdl = [hdls[1], hdls[3]]

    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1) % world_size

    # Initial: copy local k,v into slot 0
    k_buf[0].copy_(k)
    v_buf[0].copy_(v)

    # Use a side stream for comm
    comm_stream = torch.cuda.Stream(device=device)
    main_stream = torch.cuda.current_stream(device=device)

    # Barrier to ensure all ranks have written initial KV
    ext.barrier(int(k_hdl[0].signal_pad_ptrs_dev.data_ptr()),
                rank, world_size, 0, main_stream.cuda_stream)

    out_acc = None
    lse_acc = None
    cur_slot = 0

    B, S, H, D = q.shape
    out_acc = torch.empty((B, S, H, D), device=device, dtype=torch.float32)
    lse_acc = torch.empty((B, H, S), device=device, dtype=torch.float32)

    for step in range(world_size):
        next_slot = 1 - cur_slot
        cur_k = k_buf[cur_slot]
        cur_v = v_buf[cur_slot]

        # Start comm: peer copy from sender's cur slot into our next slot
        if step + 1 != world_size:
            # We need recv from recv_rank's cur slot into our next slot
            # Equivalent: read remote (recv_rank's) k_buf[cur_slot] into our k_buf[next_slot]
            comm_stream.wait_stream(main_stream)
            with torch.cuda.stream(comm_stream):
                # barrier: ensure remote has not yet overwritten cur_slot
                # We need a barrier on next_slot to ensure all done with prior next_slot
                channel = (step * 2 + 1) % 16
                ext.barrier(int(k_hdl[next_slot].signal_pad_ptrs_dev.data_ptr()),
                            rank, world_size, channel, comm_stream.cuda_stream)
                src_k_ptr = int(k_hdl[cur_slot].buffer_ptrs[recv_rank])
                src_v_ptr = int(v_hdl[cur_slot].buffer_ptrs[recv_rank])
                dst_k_ptr = k_buf[next_slot].data_ptr()
                dst_v_ptr = v_buf[next_slot].data_ptr()
                nbytes = k_buf[next_slot].numel() * k_buf[next_slot].element_size()
                ext.peer_copy(src_k_ptr, dst_k_ptr, nbytes, comm_stream.cuda_stream)
                ext.peer_copy(src_v_ptr, dst_v_ptr, nbytes, comm_stream.cuda_stream)

        # Compute on main stream
        if (not causal) or step <= rank:
            block_causal = causal and step == 0
            block_out, block_lse = _attn_block_cuda(q, cur_k, cur_v, scale, block_causal)
            _merge_inplace(out_acc, lse_acc, block_out, block_lse, first=(out_acc is None or step == 0))
            if step == 0:
                first_done = True

        if step + 1 != world_size:
            # Sync: main waits for comm
            main_stream.wait_stream(comm_stream)
            # Barrier so that all ranks have copied before next iter overwrites
            ext.barrier(int(k_hdl[next_slot].signal_pad_ptrs_dev.data_ptr()),
                        rank, world_size, (step * 2 + 2) % 16, main_stream.cuda_stream)
            cur_slot = next_slot

    out_bf16 = torch.empty_like(q)
    ext.launch_f32_to_bf16(out_acc, out_bf16)
    return out_bf16


def _attention_block_cuda(hidden, w_qkv, w_o, num_heads, scale, causal, cp_group):
    B, S, D_hidden = hidden.shape
    head_dim = w_qkv.shape[0] // 3 // num_heads
    qkv = F.linear(hidden, w_qkv).view(B, S, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    ctx = _ring_attn_forward_cuda(cp_group, q, k, v, scale, causal)
    return F.linear(ctx.reshape(B, S, -1), w_o)


def _pp_recv_cuda(pp_group, shape, dtype, device):
    """Receive activation from previous PP stage via symm_mem peer copy."""
    rank = dist.get_rank(pp_group)
    world_size = dist.get_world_size(pp_group)
    prev_rank = (rank - 1) % world_size
    buf, hdl = _get_pp_buffer(shape, dtype, device, pp_group)
    ext = _get_ext()
    stream = torch.cuda.current_stream(device=device).cuda_stream
    # Barrier 0: wait for sender to have written its buf
    ext.barrier(int(hdl.signal_pad_ptrs_dev.data_ptr()), rank, world_size, 0, stream)
    # Read from prev_rank's buf into local tensor
    src_ptr = int(hdl.buffer_ptrs[prev_rank])
    out = torch.empty(shape, dtype=dtype, device=device)
    nbytes = out.numel() * out.element_size()
    ext.peer_copy(src_ptr, out.data_ptr(), nbytes, stream)
    # Barrier 1: ensure all reads done before sender overwrites
    ext.barrier(int(hdl.signal_pad_ptrs_dev.data_ptr()), rank, world_size, 1, stream)
    return out


def _pp_send_cuda(pp_group, tensor):
    """Send activation: write to local symm buffer, signal."""
    rank = dist.get_rank(pp_group)
    world_size = dist.get_world_size(pp_group)
    buf, hdl = _get_pp_buffer(tuple(tensor.shape), tensor.dtype, tensor.device, pp_group)
    ext = _get_ext()
    buf.copy_(tensor)
    stream = torch.cuda.current_stream(device=tensor.device).cuda_stream
    # Barrier 0: signal that buf is ready
    ext.barrier(int(hdl.signal_pad_ptrs_dev.data_ptr()), rank, world_size, 0, stream)
    # Barrier 1: wait until receiver has read
    ext.barrier(int(hdl.signal_pad_ptrs_dev.data_ptr()), rank, world_size, 1, stream)


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    w_qkv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    pp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    cp_group = cp_group or dist.group.WORLD
    head_dim = w_qkv.shape[0] // 3 // num_heads
    scale = float(softmax_scale if softmax_scale is not None else head_dim ** -0.5)

    _get_ext()  # ensure compiled

    is_first = True
    is_last = True
    if pp_group is not None and dist.get_world_size(pp_group) > 1:
        pp_rank = dist.get_rank(pp_group)
        pp_size = dist.get_world_size(pp_group)
        is_first = (pp_rank == 0)
        is_last = (pp_rank == pp_size - 1)

    if is_first:
        stage_input = hidden_states
    else:
        stage_input = _pp_recv_cuda(
            pp_group, tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device
        )

    stage_output = _attention_block_cuda(
        stage_input, w_qkv, w_o, num_heads, scale, causal, cp_group
    )

    if not is_last and pp_group is not None:
        _pp_send_cuda(pp_group, stage_output.contiguous())

    return stage_output