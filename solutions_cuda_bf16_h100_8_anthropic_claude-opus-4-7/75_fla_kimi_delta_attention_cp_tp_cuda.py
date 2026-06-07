import os
from typing import Optional

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
#include <math.h>

// ---------- Gather: each rank copies remote shards into a local full buffer ----------
// shard layout per rank: [B, T_local, H, D] (contiguous, bf16)
// full layout: [B, T_full, H, D] where T_full = world_size * T_local
// For a given destination index r (rank), elements come from peer r's shard.

template <typename T>
__global__ void symm_gather_kernel(
    const uint64_t* __restrict__ peer_ptrs,
    T* __restrict__ full,
    int world_size,
    int B,
    int T_local,
    int HD       // H * D
) {
    int r = blockIdx.y;
    long long shard_elems = (long long)B * T_local * HD;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shard_elems) return;
    const T* src = reinterpret_cast<const T*>(peer_ptrs[r]);
    // src layout: [B, T_local, HD]
    long long b   = idx / ((long long)T_local * HD);
    long long rem = idx - b * ((long long)T_local * HD);
    long long t   = rem / HD;
    long long hd  = rem - t * HD;
    long long T_full = (long long)world_size * T_local;
    long long dst_t = (long long)r * T_local + t;
    long long dst_idx = b * T_full * HD + dst_t * HD + hd;
    full[dst_idx] = src[idx];
}

void launch_symm_gather_bf16(
    torch::Tensor peer_ptrs,
    torch::Tensor full_out,
    int world_size,
    int B,
    int T_local,
    int HD
) {
    long long shard_elems = (long long)B * T_local * HD;
    int threads = 256;
    int blocks = (int)((shard_elems + threads - 1) / threads);
    dim3 grid(blocks, world_size);
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>());
    symm_gather_kernel<__nv_bfloat16><<<grid, threads, 0, s>>>(
        d_ptrs, (__nv_bfloat16*)full_out.data_ptr<at::BFloat16>(),
        world_size, B, T_local, HD);
}

void launch_symm_gather_f32(
    torch::Tensor peer_ptrs,
    torch::Tensor full_out,
    int world_size,
    int B,
    int T_local,
    int HD
) {
    long long shard_elems = (long long)B * T_local * HD;
    int threads = 256;
    int blocks = (int)((shard_elems + threads - 1) / threads);
    dim3 grid(blocks, world_size);
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>());
    symm_gather_kernel<float><<<grid, threads, 0, s>>>(
        d_ptrs, full_out.data_ptr<float>(),
        world_size, B, T_local, HD);
}

// ---------- Signal pad barrier ----------
__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__global__ void barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int world_size,
    int rank,
    uint64_t channel
) {
    int tid = threadIdx.x;
    if (tid >= world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + channel * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + channel * (uint64_t)world_size + (uint64_t)tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

void launch_barrier(
    torch::Tensor signal_pad_ptrs,
    int world_size,
    int rank,
    int64_t channel
) {
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>());
    int threads = world_size;
    if (threads < 32) threads = 32;
    barrier_kernel<<<1, threads, 0, s>>>(d_ptrs, world_size, rank, (uint64_t)channel);
}

// ---------- KDA forward kernel ----------
// Inputs (bf16): q, k, v, g  shape [B, T, H, D] (D = key_dim) for q,k,g; v: [B,T,H,V]
// beta: [B,T,H] bf16
// a_log: [H] bf16
// dt_bias: [H*D] bf16
// out: [B,T,H,V] bf16
// Each block handles one (batch, head). State [D, V] kept in shared memory (float).

extern "C" __global__ void kda_forward_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ out,
    int B, int T, int H, int D, int V,
    float lower_bound,
    float scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ float smem[];
    // state: D * V
    float* state = smem;                      // size D*V
    float* q_norm = smem + D * V;             // size D
    float* k_norm = q_norm + D;               // size D
    float* decay_s = k_norm + D;              // size D
    float* v_s = decay_s + D;                 // size V
    float* update_s = v_s + V;                // size V
    float* reduce_s = update_s + V;           // size max(D,V) for reductions
    // bias preloaded
    float* bias_s = reduce_s + ((D > V) ? D : V); // size D
    float* a_scale_s = bias_s + D;            // size 1

    // Load dt_bias for this head
    for (int i = tid; i < D; i += blockSize) {
        bias_s[i] = __bfloat162float(dt_bias[h * D + i]);
    }
    if (tid == 0) {
        a_scale_s[0] = expf(__bfloat162float(a_log[h]));
    }
    // Init state
    for (int i = tid; i < D * V; i += blockSize) {
        state[i] = 0.0f;
    }
    __syncthreads();

    float a_scale = a_scale_s[0];

    long long bh_off_qk = ((long long)b * T * H + h) * 0; // unused
    // index helpers: tensor [B,T,H,D] -> b*T*H*D + t*H*D + h*D + d
    long long stride_t_qk = (long long)H * D;
    long long stride_t_v  = (long long)H * V;

    for (int t = 0; t < T; ++t) {
        const __nv_bfloat16* q_ptr = q + ((long long)b * T * H * D + (long long)t * H * D + (long long)h * D);
        const __nv_bfloat16* k_ptr = k + ((long long)b * T * H * D + (long long)t * H * D + (long long)h * D);
        const __nv_bfloat16* g_ptr = g + ((long long)b * T * H * D + (long long)t * H * D + (long long)h * D);
        const __nv_bfloat16* v_ptr = v + ((long long)b * T * H * V + (long long)t * H * V + (long long)h * V);
        __nv_bfloat16* o_ptr = out + ((long long)b * T * H * V + (long long)t * H * V + (long long)h * V);
        float beta_t = __bfloat162float(beta[(long long)b * T * H + (long long)t * H + h]);
        beta_t = 1.0f / (1.0f + expf(-beta_t));

        // load q/k and compute norms
        float q_local_sq = 0.0f, k_local_sq = 0.0f;
        for (int d = tid; d < D; d += blockSize) {
            float qv = __bfloat162float(q_ptr[d]);
            float kv = __bfloat162float(k_ptr[d]);
            q_norm[d] = qv;
            k_norm[d] = kv;
            q_local_sq += qv * qv;
            k_local_sq += kv * kv;
            // decay = exp(lower_bound * sigmoid(a_scale * (g + bias)))
            float gv = __bfloat162float(g_ptr[d]);
            float arg = a_scale * (gv + bias_s[d]);
            float sig = 1.0f / (1.0f + expf(-arg));
            decay_s[d] = expf(lower_bound * sig);
        }
        // load v
        for (int j = tid; j < V; j += blockSize) {
            v_s[j] = __bfloat162float(v_ptr[j]);
        }
        // reduce sums
        // Use reduce_s buffer
        // First put per-thread partial sums into shared
        // Simpler: use atomic via shared via warp shuffles; do block reduction
        // We'll do: write q_local_sq to reduce_s[tid] and reduce
        // But blockSize may exceed reduce_s capacity (we allocated max(D,V)).
        // Use a small extra approach: tree reduce in shared
        __syncthreads();
        // reduce q_local_sq
        // store into reduce_s[tid % size]; instead do warp-level then atomic
        __shared__ float qs_total;
        __shared__ float ks_total;
        if (tid == 0) { qs_total = 0.0f; ks_total = 0.0f; }
        __syncthreads();
        // warp reduce
        unsigned mask = 0xffffffff;
        float qsum = q_local_sq;
        float ksum = k_local_sq;
        for (int off = 16; off > 0; off >>= 1) {
            qsum += __shfl_down_sync(mask, qsum, off);
            ksum += __shfl_down_sync(mask, ksum, off);
        }
        if ((tid & 31) == 0) {
            atomicAdd(&qs_total, qsum);
            atomicAdd(&ks_total, ksum);
        }
        __syncthreads();
        float q_inv = rsqrtf(qs_total + 1e-12f);
        float k_inv = rsqrtf(ks_total + 1e-12f);
        // normalize and apply scale to q
        for (int d = tid; d < D; d += blockSize) {
            q_norm[d] = q_norm[d] * q_inv * scale;
            k_norm[d] = k_norm[d] * k_inv;
        }
        __syncthreads();

        // state = decay * state  (state[d, j])
        // projected[j] = sum_d k_norm[d] * state[d, j]
        // parallelize over j
        for (int j = tid; j < V; j += blockSize) {
            float proj = 0.0f;
            #pragma unroll 1
            for (int d = 0; d < D; ++d) {
                float s = state[d * V + j] * decay_s[d];
                state[d * V + j] = s;
                proj += k_norm[d] * s;
            }
            update_s[j] = (v_s[j] - proj) * beta_t;
        }
        __syncthreads();

        // state += k_norm[d] * update[j]; out[j] = sum_d q_norm[d] * state[d, j]
        for (int j = tid; j < V; j += blockSize) {
            float o_acc = 0.0f;
            #pragma unroll 1
            for (int d = 0; d < D; ++d) {
                float s = state[d * V + j] + k_norm[d] * update_s[j];
                state[d * V + j] = s;
                o_acc += q_norm[d] * s;
            }
            o_ptr[j] = __float2bfloat16(o_acc);
        }
        __syncthreads();
    }
}

void launch_kda_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta, torch::Tensor a_log, torch::Tensor dt_bias,
    torch::Tensor out,
    int B, int T, int H, int D, int V,
    double lower_bound
) {
    float scale = 1.0f / sqrtf((float)D);
    dim3 grid(B, H);
    int threads = 128;
    if (V >= 256) threads = 256;
    int reduce_sz = (D > V) ? D : V;
    size_t smem = sizeof(float) * ((size_t)D * V + 2 * D + D + V + V + reduce_sz + D + 1);
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    auto kernel = kda_forward_kernel;
    cudaFuncSetAttribute((void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 100*1024);
    kernel<<<grid, threads, smem, s>>>(
        (const __nv_bfloat16*)q.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)k.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)v.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)g.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)beta.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)a_log.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)dt_bias.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        B, T, H, D, V, (float)lower_bound, scale
    );
}

// ---------- Peer-pointer all-reduce (bf16 SUM) ----------
__global__ void allreduce_bf16_kernel(
    const long long* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ out,
    int world_size,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float sum = 0.0f;
        for (int r = 0; r < world_size; ++r) {
            const __nv_bfloat16* src = (const __nv_bfloat16*)ptrs[r];
            sum += __bfloat162float(src[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

void launch_allreduce_bf16(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    long long n
) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    allreduce_bf16_kernel<<<blocks, threads, 0, s>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symm_gather_bf16", &launch_symm_gather_bf16, "symm gather bf16");
    m.def("symm_gather_f32",  &launch_symm_gather_f32,  "symm gather f32");
    m.def("barrier", &launch_barrier, "barrier via signal pad");
    m.def("kda_forward", &launch_kda_forward, "KDA forward bf16");
    m.def("allreduce_bf16", &launch_allreduce_bf16, "peer-ptr allreduce bf16");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("kda_cp_tp_ext", CUDA_SRC)
    return _ext


_symm_cache = {}

def _get_symm_buf(numel: int, dtype: torch.dtype, device, group, key: str):
    ws = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    ck = (key, numel, dtype, ws, id(group))
    if ck in _symm_cache:
        return _symm_cache[ck]
    buf = symm_mem.empty(numel, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    sig_ptrs = hdl.signal_pad_ptrs_dev
    entry = (buf, hdl, ptrs, sig_ptrs, rank, ws)
    _symm_cache[ck] = entry
    return entry


_full_cache = {}
def _get_full_tensor(shape, dtype, device, key):
    ck = (key, tuple(shape), dtype)
    if ck in _full_cache:
        return _full_cache[ck]
    t = torch.empty(shape, dtype=dtype, device=device)
    _full_cache[ck] = t
    return t


_channel_counter = [100]
def _next_channel():
    _channel_counter[0] += 1
    return _channel_counter[0]


def _symm_gather(x: torch.Tensor, cp_group, key: str) -> torch.Tensor:
    """Gather sequence shards across cp_group using symmetric memory."""
    ws = dist.get_world_size(group=cp_group)
    if ws == 1:
        return x
    ext = _get_ext()
    B, T_local = x.shape[:2]
    rest = x.shape[2:]
    HD = 1
    for s in rest:
        HD *= s
    numel = B * T_local * HD
    buf, hdl, ptrs, sig_ptrs, rank, _ = _get_symm_buf(numel, x.dtype, x.device, cp_group, key)
    # write local shard into symmetric buffer
    buf.copy_(x.contiguous().view(-1))
    # cross-rank barrier so peers' buffers are populated before we read
    ext.barrier(sig_ptrs, ws, rank, _next_channel())
    full_shape = (B, ws * T_local) + tuple(rest)
    full = _get_full_tensor(full_shape, x.dtype, x.device, key + "_full")
    if x.dtype == torch.bfloat16:
        ext.symm_gather_bf16(ptrs, full, ws, B, T_local, HD)
    else:
        ext.symm_gather_f32(ptrs, full, ws, B, T_local, HD)
    # ensure all reads complete before next write reuses buffer (handled by next barrier)
    return full


@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    cp_group = cp_group or dist.group.WORLD
    cp_size = dist.get_world_size(group=cp_group)
    cp_rank = dist.get_rank(group=cp_group)
    ext = _get_ext()

    if cp_size > 1:
        q_full   = _symm_gather(q,    cp_group, "q")
        k_full   = _symm_gather(k,    cp_group, "k")
        v_full   = _symm_gather(v,    cp_group, "v")
        g_full   = _symm_gather(g,    cp_group, "g")
        beta_full= _symm_gather(beta, cp_group, "beta")
        # final barrier so all gathers are visible (each gather already barriered before read)
        # ensure local writes for next iteration won't overwrite — use one more barrier
        sig = _symm_cache[("q", q.numel(), q.dtype, cp_size, id(cp_group))][3]
        ext.barrier(sig, cp_size, cp_rank, _next_channel())
    else:
        q_full, k_full, v_full, g_full, beta_full = q, k, v, g, beta

    B, T_full, H, D = q_full.shape
    V = v_full.shape[-1]

    # Use custom KDA kernel for bf16; fallback to PyTorch reference for other dtypes
    use_cuda_kda = (
        q_full.dtype == torch.bfloat16
        and k_full.dtype == torch.bfloat16
        and v_full.dtype == torch.bfloat16
        and g_full.dtype == torch.bfloat16
        and beta_full.dtype == torch.bfloat16
    )

    if use_cuda_kda:
        # ensure a_log and dt_bias are bf16
        a_log_bf = a_log.to(torch.bfloat16).contiguous()
        dt_bias_bf = dt_bias.to(torch.bfloat16).contiguous()
        out = torch.empty((B, T_full, H, V), dtype=torch.bfloat16, device=q_full.device)
        ext.kda_forward(
            q_full.contiguous(), k_full.contiguous(), v_full.contiguous(),
            g_full.contiguous(), beta_full.contiguous(),
            a_log_bf, dt_bias_bf, out,
            B, T_full, H, D, V, -5.0
        )
    else:
        out = _kda_forward_ref(q_full, k_full, v_full, g_full, beta_full,
                               a_log, dt_bias, -5.0)

    if tp_group is not None and dist.get_world_size(group=tp_group) > 1:
        # Custom symm-mem all-reduce
        tp_ws = dist.get_world_size(group=tp_group)
        tp_rank = dist.get_rank(group=tp_group)
        n = out.numel()
        buf, hdl, ptrs, sig_ptrs, _, _ = _get_symm_buf(n, out.dtype, out.device, tp_group, "tp_ar")
        buf.copy_(out.view(-1))
        ext.barrier(sig_ptrs, tp_ws, tp_rank, _next_channel())
        if out.dtype == torch.bfloat16:
            ext.allreduce_bf16(ptrs, out, n)
        else:
            # fallback
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
        ext.barrier(sig_ptrs, tp_ws, tp_rank, _next_channel())

    if cp_size == 1:
        return out
    local_seq = q.shape[1]
    start = cp_rank * local_seq
    return out[:, start:start + local_seq].contiguous()


def _kda_forward_ref(q, k, v, g, beta, a_log, dt_bias, lower_bound):
    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    out_dtype = q.dtype
    dt_bias = dt_bias.float().reshape(heads, key_dim)
    a_scale = a_log.float().exp().view(1, 1, heads, 1)
    decay = torch.exp(lower_bound * torch.sigmoid(a_scale * (g.float() + dt_bias)))
    beta = beta.float().sigmoid()
    scale = float(key_dim) ** -0.5
    q_float = F.normalize(q.float(), p=2, dim=-1) * scale
    k_float = F.normalize(k.float(), p=2, dim=-1)
    v_float = v.float()
    q_float = q_float.permute(0, 2, 1, 3).contiguous().reshape(batch*heads, seq_len, key_dim)
    k_float = k_float.permute(0, 2, 1, 3).contiguous().reshape(batch*heads, seq_len, key_dim)
    v_float = v_float.permute(0, 2, 1, 3).contiguous().reshape(batch*heads, seq_len, value_dim)
    decay = decay.permute(0, 2, 1, 3).contiguous().reshape(batch*heads, seq_len, key_dim)
    beta = beta.permute(0, 2, 1).contiguous().reshape(batch*heads, seq_len)
    state = torch.zeros(batch*heads, key_dim, value_dim, dtype=torch.float32, device=q.device)
    output = torch.empty(batch*heads, seq_len, value_dim, dtype=torch.float32, device=q.device)
    for step in range(seq_len):
        q_t = q_float[:, step]; k_t = k_float[:, step]; v_t = v_float[:, step]
        state = decay[:, step].unsqueeze(-1) * state
        projected = torch.bmm(k_t.unsqueeze(1), state).squeeze(1)
        update = (v_t - projected) * beta[:, step].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * update.unsqueeze(1)
        output[:, step] = torch.bmm(q_t.unsqueeze(1), state).squeeze(1)
    output = output.reshape(batch, heads, seq_len, value_dim).permute(0, 2, 1, 3).contiguous()
    return output.to(dtype=out_dtype)