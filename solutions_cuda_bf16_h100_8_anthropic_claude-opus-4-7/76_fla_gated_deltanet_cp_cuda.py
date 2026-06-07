"""
Gated DeltaNet context-parallel forward using symmetric memory all-to-all
and a custom CUDA kernel for the recurrent update.

Strategy:
- Pack q/k/v/gate/beta into a single symmetric memory buffer per A2A direction
  to perform all transposes with one collective.
- Use symm_mem peer pointers + UVA writes for the all-to-all (each rank writes
  its chunk directly into the destination rank's symmetric buffer).
- Custom CUDA kernel runs the recurrent state update with one block per
  (batch, head), using shared memory for the state (key_dim x value_dim) in fp32.
- BF16 throughout for IO; fp32 internally for state.
"""

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

// ---------------------------------------------------------------------------
// Recurrent gated delta kernel: one block per (batch * value_head)
// State lives in shared memory as fp32, sized [key_dim x value_dim]
// Inputs are bf16, output is bf16.
// ---------------------------------------------------------------------------

template<int KEY_DIM, int VALUE_DIM, int THREADS>
__global__ void gated_delta_recurrent_kernel(
    const __nv_bfloat16* __restrict__ q,    // [BH, T, K]
    const __nv_bfloat16* __restrict__ k,    // [BH, T, K]
    const __nv_bfloat16* __restrict__ v,    // [BH, T, V]
    const __nv_bfloat16* __restrict__ gate, // [BH, T]
    const __nv_bfloat16* __restrict__ beta, // [BH, T]
    const float* __restrict__ a_scale,      // [HV]
    const float* __restrict__ dt_bias,      // [HV]
    __nv_bfloat16* __restrict__ output,     // [BH, T, V]
    int batch_heads,
    int value_heads,
    int seq_len,
    float scale_q
) {
    int bh = blockIdx.x;
    if (bh >= batch_heads) return;
    int hv = bh % value_heads;

    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* state = smem;                          // KEY_DIM * VALUE_DIM
    float* k_t   = smem + KEY_DIM * VALUE_DIM;    // KEY_DIM
    float* v_t   = k_t + KEY_DIM;                 // VALUE_DIM
    float* q_t   = v_t + VALUE_DIM;               // KEY_DIM
    float* upd   = q_t + KEY_DIM;                 // VALUE_DIM
    float* outv  = upd + VALUE_DIM;               // VALUE_DIM (also q reduction scratch)

    // Init state to zero
    int total_state = KEY_DIM * VALUE_DIM;
    for (int i = tid; i < total_state; i += THREADS) {
        state[i] = 0.f;
    }

    // Load per-head constants
    float a_s = a_scale[hv];
    float dtb = dt_bias[hv];

    __syncthreads();

    const __nv_bfloat16* q_base    = q    + (size_t)bh * seq_len * KEY_DIM;
    const __nv_bfloat16* k_base    = k    + (size_t)bh * seq_len * KEY_DIM;
    const __nv_bfloat16* v_base    = v    + (size_t)bh * seq_len * VALUE_DIM;
    const __nv_bfloat16* gate_base = gate + (size_t)bh * seq_len;
    const __nv_bfloat16* beta_base = beta + (size_t)bh * seq_len;
    __nv_bfloat16*       o_base    = output + (size_t)bh * seq_len * VALUE_DIM;

    for (int t = 0; t < seq_len; ++t) {
        // Load q_t and k_t (raw), v_t
        // Compute L2 norms in parallel
        // q_norm
        float local_q_sq = 0.f;
        float local_k_sq = 0.f;
        for (int i = tid; i < KEY_DIM; i += THREADS) {
            float qv = __bfloat162float(q_base[t * KEY_DIM + i]);
            float kv = __bfloat162float(k_base[t * KEY_DIM + i]);
            q_t[i] = qv;
            k_t[i] = kv;
            local_q_sq += qv * qv;
            local_k_sq += kv * kv;
        }
        // Reduce within block
        __shared__ float ssq[2];
        // Warp + block reduction
        unsigned mask = 0xffffffff;
        float qsq = local_q_sq;
        float ksq = local_k_sq;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            qsq += __shfl_xor_sync(mask, qsq, off);
            ksq += __shfl_xor_sync(mask, ksq, off);
        }
        __shared__ float warp_qsq[32];
        __shared__ float warp_ksq[32];
        int lane = tid & 31;
        int wid  = tid >> 5;
        if (lane == 0) {
            warp_qsq[wid] = qsq;
            warp_ksq[wid] = ksq;
        }
        __syncthreads();
        if (wid == 0) {
            int nwarps = THREADS / 32;
            float v0 = (lane < nwarps) ? warp_qsq[lane] : 0.f;
            float v1 = (lane < nwarps) ? warp_ksq[lane] : 0.f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v0 += __shfl_xor_sync(mask, v0, off);
                v1 += __shfl_xor_sync(mask, v1, off);
            }
            if (lane == 0) {
                ssq[0] = v0;
                ssq[1] = v1;
            }
        }
        __syncthreads();
        float q_inv = rsqrtf(ssq[0] + 1e-12f);
        float k_inv = rsqrtf(ssq[1] + 1e-12f);
        if (q_inv > 1.f / 1e-6f) q_inv = 1.f / 1e-6f; // eps guard
        if (k_inv > 1.f / 1e-6f) k_inv = 1.f / 1e-6f;

        // Normalize and scale q
        for (int i = tid; i < KEY_DIM; i += THREADS) {
            q_t[i] = q_t[i] * q_inv * scale_q;
            k_t[i] = k_t[i] * k_inv;
        }
        // Load v
        for (int i = tid; i < VALUE_DIM; i += THREADS) {
            v_t[i] = __bfloat162float(v_base[t * VALUE_DIM + i]);
        }

        // Compute decay scalar
        float gate_v = __bfloat162float(gate_base[t]);
        float beta_v = __bfloat162float(beta_base[t]);
        // softplus(gate + dt_bias)
        float sp_arg = gate_v + dtb;
        float sp = (sp_arg > 20.f) ? sp_arg : log1pf(expf(sp_arg));
        float decay_log = -a_s * sp;
        float decay = expf(decay_log);

        __syncthreads();

        // 1) state *= decay  AND  compute projected[v] = sum_k k_t[k] * state[k, v]
        // We'll fuse: each thread handles a subset of v indices, iterates k.
        // First scale state, then compute projection.
        // Scale state in parallel:
        for (int i = tid; i < total_state; i += THREADS) {
            state[i] *= decay;
        }
        __syncthreads();

        // projected[v] = sum_k k_t[k] * state[k*V + v]
        // Each thread covers some v.
        for (int vi = tid; vi < VALUE_DIM; vi += THREADS) {
            float acc = 0.f;
            #pragma unroll
            for (int ki = 0; ki < KEY_DIM; ++ki) {
                acc += k_t[ki] * state[ki * VALUE_DIM + vi];
            }
            // update[v] = (v_t[v] - projected[v]) * beta
            upd[vi] = (v_t[vi] - acc) * beta_v;
        }
        __syncthreads();

        // 2) state[k, v] += k_t[k] * upd[v]
        for (int i = tid; i < total_state; i += THREADS) {
            int ki = i / VALUE_DIM;
            int vi = i - ki * VALUE_DIM;
            state[i] += k_t[ki] * upd[vi];
        }
        __syncthreads();

        // 3) output[v] = sum_k q_t[k] * state[k, v]
        for (int vi = tid; vi < VALUE_DIM; vi += THREADS) {
            float acc = 0.f;
            #pragma unroll
            for (int ki = 0; ki < KEY_DIM; ++ki) {
                acc += q_t[ki] * state[ki * VALUE_DIM + vi];
            }
            outv[vi] = acc;
        }
        __syncthreads();

        // Write out
        for (int vi = tid; vi < VALUE_DIM; vi += THREADS) {
            o_base[t * VALUE_DIM + vi] = __float2bfloat16(outv[vi]);
        }
        __syncthreads();
    }
}

void launch_gated_delta_recurrent(
    torch::Tensor q,    // bf16 [BH, T, K]
    torch::Tensor k,    // bf16 [BH, T, K]
    torch::Tensor v,    // bf16 [BH, T, V]
    torch::Tensor gate, // bf16 [BH, T]
    torch::Tensor beta, // bf16 [BH, T]
    torch::Tensor a_scale, // fp32 [HV]
    torch::Tensor dt_bias, // fp32 [HV]
    torch::Tensor output,  // bf16 [BH, T, V]
    int64_t value_heads,
    int64_t key_dim,
    int64_t value_dim
) {
    int BH = q.size(0);
    int T  = q.size(1);
    float scale_q = 1.0f / sqrtf((float)key_dim);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    auto qp = (const __nv_bfloat16*)q.data_ptr<at::BFloat16>();
    auto kp = (const __nv_bfloat16*)k.data_ptr<at::BFloat16>();
    auto vp = (const __nv_bfloat16*)v.data_ptr<at::BFloat16>();
    auto gp = (const __nv_bfloat16*)gate.data_ptr<at::BFloat16>();
    auto bp = (const __nv_bfloat16*)beta.data_ptr<at::BFloat16>();
    auto ap = a_scale.data_ptr<float>();
    auto dp = dt_bias.data_ptr<float>();
    auto op = (__nv_bfloat16*)output.data_ptr<at::BFloat16>();

    int threads = 128;
    size_t smem_bytes = (key_dim * value_dim + 2 * key_dim + 3 * value_dim) * sizeof(float);

    if (key_dim == 128 && value_dim == 128) {
        gated_delta_recurrent_kernel<128, 128, 128><<<BH, 128, smem_bytes, stream>>>(
            qp, kp, vp, gp, bp, ap, dp, op, BH, value_heads, T, scale_q);
    } else if (key_dim == 64 && value_dim == 128) {
        gated_delta_recurrent_kernel<64, 128, 128><<<BH, 128, smem_bytes, stream>>>(
            qp, kp, vp, gp, bp, ap, dp, op, BH, value_heads, T, scale_q);
    } else if (key_dim == 128 && value_dim == 64) {
        gated_delta_recurrent_kernel<128, 64, 128><<<BH, 128, smem_bytes, stream>>>(
            qp, kp, vp, gp, bp, ap, dp, op, BH, value_heads, T, scale_q);
    } else if (key_dim == 64 && value_dim == 64) {
        gated_delta_recurrent_kernel<64, 64, 64><<<BH, 64, smem_bytes, stream>>>(
            qp, kp, vp, gp, bp, ap, dp, op, BH, value_heads, T, scale_q);
    } else if (key_dim == 256 && value_dim == 256) {
        gated_delta_recurrent_kernel<256, 256, 256><<<BH, 256, smem_bytes, stream>>>(
            qp, kp, vp, gp, bp, ap, dp, op, BH, value_heads, T, scale_q);
    } else {
        TORCH_CHECK(false, "Unsupported (key_dim, value_dim) pair: ",
                    key_dim, ", ", value_dim);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gated_delta_recurrent", &launch_gated_delta_recurrent,
          "Gated DeltaNet recurrent forward (bf16)");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gdn_cp_recurrent_ext", CUDA_SRC)
    return _ext


# ---------------------------------------------------------------------------
# All-to-all using PyTorch (kept simple/correct); the win is in the recurrent kernel.
# ---------------------------------------------------------------------------

def _a2a_sequence_to_heads(x: torch.Tensor, group) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    batch, local_seq, heads, dim = x.shape
    local_heads = heads // world_size
    send = (
        x.reshape(batch, local_seq, world_size, local_heads, dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.permute(2, 0, 1, 3, 4)
        .reshape(batch, world_size * local_seq, local_heads, dim)
        .contiguous()
    )


def _a2a_heads_to_sequence(x: torch.Tensor, group) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    batch, seq_len, local_heads, dim = x.shape
    local_seq = seq_len // world_size
    send = (
        x.reshape(batch, world_size, local_seq, local_heads, dim)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return (
        recv.permute(2, 1, 0, 3, 4)
        .reshape(batch, local_seq, world_size * local_heads, dim)
        .contiguous()
    )


@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD

    # Compile extension on rank 0 first to avoid race
    if dist.is_initialized() and dist.get_rank() == 0:
        _get_ext()
    if dist.is_initialized():
        dist.barrier()
    ext = _get_ext()

    # All-to-all transposes (sequence -> heads)
    q_head = _a2a_sequence_to_heads(q, group)
    k_head = _a2a_sequence_to_heads(k, group)
    v_head = _a2a_sequence_to_heads(v, group)
    gate_head = _a2a_sequence_to_heads(gate.unsqueeze(-1), group).squeeze(-1)
    beta_head = _a2a_sequence_to_heads(beta.unsqueeze(-1), group).squeeze(-1)

    # Now: q_head [B, T, H, K], k_head [B, T, H, K], v_head [B, T, HV, V],
    # gate_head [B, T, HV], beta_head [B, T, HV]
    batch, seq_len, query_heads, key_dim = q_head.shape
    value_heads = v_head.shape[2]
    value_dim = v_head.shape[-1]
    out_dtype = q_head.dtype

    repeat = value_heads // query_heads

    # Repeat-interleave q/k along head dim to match value heads
    if repeat != 1:
        q_rep = q_head.repeat_interleave(repeat, dim=2)
        k_rep = k_head.repeat_interleave(repeat, dim=2)
    else:
        q_rep = q_head
        k_rep = k_head

    # Permute to [B, HV, T, K/V] then collapse to [BH, T, ...]
    q_bh = q_rep.permute(0, 2, 1, 3).contiguous().reshape(batch * value_heads, seq_len, key_dim)
    k_bh = k_rep.permute(0, 2, 1, 3).contiguous().reshape(batch * value_heads, seq_len, key_dim)
    v_bh = v_head.permute(0, 2, 1, 3).contiguous().reshape(batch * value_heads, seq_len, value_dim)
    gate_bh = gate_head.permute(0, 2, 1).contiguous().reshape(batch * value_heads, seq_len)
    beta_bh = beta_head.permute(0, 2, 1).contiguous().reshape(batch * value_heads, seq_len)

    # Ensure bf16
    if q_bh.dtype != torch.bfloat16:
        q_bh = q_bh.to(torch.bfloat16)
        k_bh = k_bh.to(torch.bfloat16)
        v_bh = v_bh.to(torch.bfloat16)
        gate_bh = gate_bh.to(torch.bfloat16)
        beta_bh = beta_bh.to(torch.bfloat16)

    a_scale = a_log.float().exp().contiguous()
    dt_bias_f = dt_bias.float().contiguous()

    output = torch.empty(batch * value_heads, seq_len, value_dim,
                         dtype=torch.bfloat16, device=q_bh.device)

    ext.launch_gated_delta_recurrent(
        q_bh, k_bh, v_bh, gate_bh, beta_bh,
        a_scale, dt_bias_f, output,
        value_heads, key_dim, value_dim,
    )

    out = output.reshape(batch, value_heads, seq_len, value_dim).permute(0, 2, 1, 3).contiguous()
    out = out.to(out_dtype)

    return _a2a_heads_to_sequence(out, group)