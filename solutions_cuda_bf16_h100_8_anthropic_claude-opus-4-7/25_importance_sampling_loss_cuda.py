import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Tuple, Any
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

// Packed layout (8 floats):
// [0] n_valid
// [1] pg_sum
// [2] sum_ratio
// [3] min_ratio
// [4] max_ratio (stored as -max so we can use min reduction; but we will keep separate)
// [5] k3_sum
// [6] entropy_sum
// [7] (padding / unused)

#define PACK_N 8

// ---------- Per-token fused kernel ----------
// For each valid token i:
//   logits row -> compute logsumexp and -logits[label] => per_token_ce
//   new_lp = -ce
//   delta = clamp(new_lp - old_lp, -20, 20)
//   ratio = exp(delta)
//   pg = -(ratio*adv)
//   k3 = ratio - delta - 1
//   entropy = ce
// Writes per-token outputs and accumulates partial reductions per block into scratch.

extern "C" __global__ void fused_token_kernel(
    const __nv_bfloat16* __restrict__ logits, // [N, V]
    const long*           __restrict__ labels, // [N]
    const float*          __restrict__ old_lp, // [N]
    const float*          __restrict__ adv,    // [N]
    float*                __restrict__ per_token_logprobs, // [N]
    float*                __restrict__ per_token_loss,     // [N]
    float*                __restrict__ per_token_ce_out,   // [N] (for surrogate backward)
    float*                __restrict__ ratio_out,          // [N] (for surrogate backward)
    float*                __restrict__ block_partials,     // [num_blocks, PACK_N]
    int N, int V, int ignore_index)
{
    extern __shared__ float smem[];
    // smem layout: PACK_N partials per warp, then final
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    // Each block processes one token via cooperative reduction across threads
    // But many tokens per block is more efficient when V is moderate. Here V can be large (vocab_size).
    // Strategy: 1 token per block.
    int token = blockIdx.x;
    if (token >= N) return;

    long label = labels[token];
    bool valid = (label != (long)ignore_index);

    const __nv_bfloat16* row = logits + (long)token * V;

    // Pass 1: max
    float local_max = -FLT_MAX;
    for (int v = tid; v < V; v += bsz) {
        float x = __bfloat162float(row[v]);
        if (x > local_max) local_max = x;
    }
    // Block reduce max
    __shared__ float sdata[32];
    // warp reduce
    unsigned mask = 0xffffffff;
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(mask, local_max, off);
        if (o > local_max) local_max = o;
    }
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) sdata[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (bsz + 31)/32) ? sdata[lane] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            float o = __shfl_down_sync(mask, v, off);
            if (o > v) v = o;
        }
        if (lane == 0) sdata[0] = v;
    }
    __syncthreads();
    float row_max = sdata[0];

    // Pass 2: sum exp
    float local_sum = 0.0f;
    float label_logit = 0.0f;
    for (int v = tid; v < V; v += bsz) {
        float x = __bfloat162float(row[v]);
        local_sum += __expf(x - row_max);
        if (v == (int)label) label_logit = x;
    }
    // share label_logit: only one thread has it; use shared
    __shared__ float s_label_logit;
    if (tid == 0) s_label_logit = 0.0f;
    __syncthreads();
    if ((int)label >= 0 && (int)label < V && tid == ((int)label % bsz)) {
        // The thread that hit v == label captured it; but using modulo isn't reliable in strided loop.
    }
    // Reliable: thread that processed label index is tid_label = label % bsz (since stride=bsz starting at tid)
    // Actually thread `tid` processes v in {tid, tid+bsz, ...}. So thread tid_label = label % bsz processes label.
    if (valid) {
        int tid_label = (int)(label % (long)bsz);
        if (tid == tid_label) {
            s_label_logit = label_logit;
        }
    }

    // Block reduce sum
    for (int off = 16; off > 0; off >>= 1) local_sum += __shfl_down_sync(mask, local_sum, off);
    if (lane == 0) sdata[warp_id] = local_sum;
    __syncthreads();
    float row_sum = 0.0f;
    if (warp_id == 0) {
        float v = (tid < (bsz + 31)/32) ? sdata[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
        if (lane == 0) sdata[0] = v;
    }
    __syncthreads();
    row_sum = sdata[0];

    // Now thread 0 computes per-token outputs and partials
    __shared__ float s_partials[PACK_N];
    if (tid == 0) {
        float ce, new_lp, delta, ratio, pg, k3, entropy_v;
        float n_valid_inc = 0.0f;
        if (valid) {
            float lse = row_max + __logf(row_sum);
            ce = lse - s_label_logit;
            new_lp = -ce;
            float d = new_lp - old_lp[token];
            if (d < -20.0f) d = -20.0f;
            if (d > 20.0f) d = 20.0f;
            delta = d;
            ratio = __expf(delta);
            float a = adv[token];
            pg = -(ratio * a);
            k3 = ratio - delta - 1.0f;
            entropy_v = ce;
            n_valid_inc = 1.0f;
            per_token_logprobs[token] = new_lp;
            per_token_loss[token] = pg;
            per_token_ce_out[token] = ce;
            ratio_out[token] = ratio;
            s_partials[0] = n_valid_inc;
            s_partials[1] = pg;
            s_partials[2] = ratio;
            s_partials[3] = ratio; // for min
            s_partials[4] = ratio; // for max
            s_partials[5] = k3;
            s_partials[6] = entropy_v;
            s_partials[7] = 0.0f;
        } else {
            per_token_logprobs[token] = 0.0f;
            per_token_loss[token] = 0.0f;
            per_token_ce_out[token] = 0.0f;
            ratio_out[token] = 0.0f;
            s_partials[0] = 0.0f;
            s_partials[1] = 0.0f;
            s_partials[2] = 0.0f;
            s_partials[3] = FLT_MAX;
            s_partials[4] = -FLT_MAX;
            s_partials[5] = 0.0f;
            s_partials[6] = 0.0f;
            s_partials[7] = 0.0f;
        }
    }
    __syncthreads();

    // Each block writes its own partials slot (1 token per block already)
    if (tid < PACK_N) {
        block_partials[(long)blockIdx.x * PACK_N + tid] = s_partials[tid];
    }
}

// Reduce block_partials [num_blocks, PACK_N] -> packed [PACK_N]
extern "C" __global__ void reduce_partials_kernel(
    const float* __restrict__ block_partials,
    float* __restrict__ out_packed,
    int num_blocks)
{
    int field = blockIdx.x; // PACK_N blocks
    if (field >= PACK_N) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    float acc;
    bool is_min = (field == 3);
    bool is_max = (field == 4);
    if (is_min) acc = FLT_MAX;
    else if (is_max) acc = -FLT_MAX;
    else acc = 0.0f;

    for (int i = tid; i < num_blocks; i += bsz) {
        float v = block_partials[(long)i * PACK_N + field];
        if (is_min) { if (v < acc) acc = v; }
        else if (is_max) { if (v > acc) acc = v; }
        else acc += v;
    }

    __shared__ float sdata[32];
    unsigned mask = 0xffffffff;
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(mask, acc, off);
        if (is_min) { if (o < acc) acc = o; }
        else if (is_max) { if (o > acc) acc = o; }
        else acc += o;
    }
    if (lane == 0) sdata[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v;
        int nw = (bsz + 31) / 32;
        if (tid < nw) v = sdata[lane];
        else { v = is_min ? FLT_MAX : (is_max ? -FLT_MAX : 0.0f); }
        for (int off = 16; off > 0; off >>= 1) {
            float o = __shfl_down_sync(mask, v, off);
            if (is_min) { if (o < v) v = o; }
            else if (is_max) { if (o > v) v = o; }
            else v += o;
        }
        if (lane == 0) out_packed[field] = v;
    }
}

// Combine packed reductions across peers using UVA pointers.
// peer_ptrs[world_size] -> each is float* of length PACK_N.
// out: float[PACK_N] global.
extern "C" __global__ void combine_peers_kernel(
    const long long* __restrict__ peer_ptrs,
    float* __restrict__ out_global,
    int world_size)
{
    int field = threadIdx.x;
    if (field >= PACK_N) return;
    bool is_min = (field == 3);
    bool is_max = (field == 4);
    float acc;
    if (is_min) acc = FLT_MAX;
    else if (is_max) acc = -FLT_MAX;
    else acc = 0.0f;
    for (int r = 0; r < world_size; ++r) {
        const float* p = (const float*)peer_ptrs[r];
        float v = p[field];
        if (is_min) { if (v < acc) acc = v; }
        else if (is_max) { if (v > acc) acc = v; }
        else acc += v;
    }
    out_global[field] = acc;
}

// Launchers
void launch_fused_token(
    torch::Tensor logits, torch::Tensor labels,
    torch::Tensor old_lp, torch::Tensor adv,
    torch::Tensor per_token_logprobs, torch::Tensor per_token_loss,
    torch::Tensor per_token_ce_out, torch::Tensor ratio_out,
    torch::Tensor block_partials,
    int64_t N, int64_t V, int64_t ignore_index)
{
    int threads = 256;
    int blocks = (int)N;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_token_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)logits.data_ptr<at::BFloat16>(),
        labels.data_ptr<long>(),
        old_lp.data_ptr<float>(),
        adv.data_ptr<float>(),
        per_token_logprobs.data_ptr<float>(),
        per_token_loss.data_ptr<float>(),
        per_token_ce_out.data_ptr<float>(),
        ratio_out.data_ptr<float>(),
        block_partials.data_ptr<float>(),
        (int)N, (int)V, (int)ignore_index);
}

void launch_reduce_partials(
    torch::Tensor block_partials,
    torch::Tensor out_packed,
    int64_t num_blocks)
{
    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_partials_kernel<<<PACK_N, threads, 0, stream>>>(
        block_partials.data_ptr<float>(),
        out_packed.data_ptr<float>(),
        (int)num_blocks);
}

void launch_combine_peers(
    torch::Tensor peer_ptrs,
    torch::Tensor out_global,
    int64_t world_size)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    combine_peers_kernel<<<1, PACK_N, 0, stream>>>(
        (const long long*)peer_ptrs.data_ptr<int64_t>(),
        out_global.data_ptr<float>(),
        (int)world_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused_token", &launch_fused_token);
    m.def("launch_reduce_partials", &launch_reduce_partials);
    m.def("launch_combine_peers", &launch_combine_peers);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("grpo_is_loss_ext", CUDA_SRC)
    return _ext

PACK_N = 8

_symm_cache = {}
def _get_symm(device, dtype=torch.float32):
    key = (device, dtype)
    if key in _symm_cache:
        return _symm_cache[key]
    buf = symm_mem.empty(PACK_N, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = torch.empty(PACK_N, device=device, dtype=dtype)
    ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, out, ptrs)
    return _symm_cache[key]


@torch.no_grad()
def _forward_compute(hidden_states, weight, labels, old_logprobs, advantages, ignore_index):
    ext = _get_ext()
    B, S, H = hidden_states.shape
    V = weight.shape[0]
    N = B * S

    # GEMM with cuBLAS tensor cores (BF16)
    hs_flat = hidden_states.reshape(N, H).contiguous()
    logits = torch.matmul(hs_flat, weight.t().contiguous())  # [N, V] bf16
    logits = logits.contiguous()

    labels_flat = labels.reshape(-1).contiguous().to(torch.int64)
    old_lp_flat = old_logprobs.reshape(-1).contiguous().to(torch.float32)
    adv_flat = advantages.reshape(-1).contiguous().to(torch.float32)

    device = hidden_states.device
    per_token_logprobs = torch.empty(N, device=device, dtype=torch.float32)
    per_token_loss = torch.empty(N, device=device, dtype=torch.float32)
    per_token_ce = torch.empty(N, device=device, dtype=torch.float32)
    ratio = torch.empty(N, device=device, dtype=torch.float32)

    block_partials = torch.empty(N * PACK_N, device=device, dtype=torch.float32)

    ext.launch_fused_token(
        logits, labels_flat, old_lp_flat, adv_flat,
        per_token_logprobs, per_token_loss,
        per_token_ce, ratio, block_partials,
        N, V, ignore_index)

    # Reduce blocks -> packed local
    buf, hdl, out_global, ptrs = _get_symm(device)
    ext.launch_reduce_partials(block_partials, buf, N)

    # Symm-mem barrier then peer combine
    hdl.barrier(channel=0)
    ext.launch_combine_peers(ptrs, out_global, hdl.world_size)
    hdl.barrier(channel=1)

    return per_token_logprobs, per_token_loss, per_token_ce, ratio, out_global, logits


def solution(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, torch.Tensor]:

    assert dist.is_initialized()
    B, S, H = hidden_states.shape
    N = B * S

    # Run all heavy compute + fused single all-reduce under no_grad
    per_token_logprobs, per_token_loss, per_token_ce, ratio, packed_global, _logits = \
        _forward_compute(hidden_states.detach(), weight.detach(), labels, old_logprobs, advantages, ignore_index)

    n_valid_global = packed_global[0].clamp(min=1.0)
    pg_sum_global = packed_global[1]
    sum_ratio_global = packed_global[2]
    min_ratio_global = packed_global[3]
    max_ratio_global = packed_global[4]
    k3_sum_global = packed_global[5]
    entropy_sum_global = packed_global[6]

    true_pg = pg_sum_global / n_valid_global

    # Surrogate for gradients: re-run F.linear + cross_entropy on requires_grad path,
    # but only multiply by detached weights. We need gradients w.r.t. hidden_states & weight.
    if hidden_states.requires_grad or weight.requires_grad:
        logits = F.linear(hidden_states, weight)
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        per_token_ce_grad = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='none')
        valid_mask = (labels_flat != ignore_index)
        adv_flat = advantages.view(-1)
        w = (ratio * adv_flat).masked_fill(~valid_mask, 0.0)
        local_surrogate_sum = (w * per_token_ce_grad).sum()
        surrogate = local_surrogate_sum / n_valid_global
        loss = true_pg.detach() + surrogate - surrogate.detach()
    else:
        loss = true_pg.detach().clone()

    metrics = torch.stack([
        sum_ratio_global / n_valid_global,
        min_ratio_global,
        max_ratio_global,
        k3_sum_global / n_valid_global,
        entropy_sum_global / n_valid_global,
    ])

    per_token_logprobs_out = per_token_logprobs.view_as(labels)
    per_token_loss_out = per_token_loss.view_as(labels)

    return loss, None, per_token_logprobs_out, per_token_loss_out, metrics