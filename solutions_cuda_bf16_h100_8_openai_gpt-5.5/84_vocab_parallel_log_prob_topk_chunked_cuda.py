from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math_constants.h>

#define BLOCK_THREADS 256

__device__ __forceinline__ float load_scalar(
    const long long* __restrict__ ptrs,
    int shard,
    int64_t offset,
    int dtype_enum
) {
    const char* base = reinterpret_cast<const char*>(ptrs[shard]);
    if (dtype_enum == 0) {  // bf16
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(base);
        return __bfloat162float(p[offset]);
    } else if (dtype_enum == 1) {  // fp32
        const float* p = reinterpret_cast<const float*>(base);
        return p[offset];
    } else {  // fp16
        const __half* p = reinterpret_cast<const __half*>(base);
        return __half2float(p[offset]);
    }
}

__device__ __forceinline__ int64_t load_target_id(
    const void* __restrict__ target,
    int64_t idx,
    int target_dtype_enum
) {
    if (target_dtype_enum == 0) {
        return reinterpret_cast<const int64_t*>(target)[idx];
    } else {
        return static_cast<int64_t>(reinterpret_cast<const int32_t*>(target)[idx]);
    }
}

__device__ float block_reduce_sum(float v) {
    __shared__ float smem[BLOCK_THREADS];
    smem[threadIdx.x] = v;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    return smem[0];
}

__device__ float block_reduce_max(float v) {
    __shared__ float smem[BLOCK_THREADS];
    smem[threadIdx.x] = v;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    return smem[0];
}

__device__ void block_reduce_max_count(float v, int c, float* out_v, int* out_c) {
    __shared__ float sval[BLOCK_THREADS];
    __shared__ int scnt[BLOCK_THREADS];

    sval[threadIdx.x] = v;
    scnt[threadIdx.x] = c;
    __syncthreads();

    #pragma unroll
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float ov = sval[threadIdx.x + s];
            int oc = scnt[threadIdx.x + s];
            float cv = sval[threadIdx.x];
            if (ov > cv) {
                sval[threadIdx.x] = ov;
                scnt[threadIdx.x] = oc;
            } else if (ov == cv) {
                scnt[threadIdx.x] += oc;
            }
        }
        __syncthreads();
    }

    *out_v = sval[0];
    *out_c = scnt[0];
}

__device__ void find_next_distinct_desc(
    const long long* __restrict__ logit_ptrs,
    int64_t token,
    int64_t local_vocab,
    int world_size,
    int dtype_enum,
    float prev,
    float min_allowed,
    float max_for_exp,
    float* best_out,
    int* count_out,
    float* exp_mass_out
) {
    const int64_t total_vocab = local_vocab * (int64_t)world_size;
    float best = -CUDART_INF_F;
    int cnt = 0;

    for (int64_t g = threadIdx.x; g < total_vocab; g += BLOCK_THREADS) {
        int shard = static_cast<int>(g / local_vocab);
        int64_t li = g - (int64_t)shard * local_vocab;
        float x = load_scalar(logit_ptrs, shard, token * local_vocab + li, dtype_enum);

        if (x >= min_allowed && x < prev) {
            if (x > best) {
                best = x;
                cnt = 1;
            } else if (x == best) {
                cnt += 1;
            }
        }
    }

    float rb;
    int rc;
    block_reduce_max_count(best, cnt, &rb, &rc);

    float local_mass = 0.0f;
    if (best == rb && cnt > 0) {
        local_mass = cnt * expf(rb - max_for_exp);
    }
    float mass = block_reduce_sum(local_mass);

    *best_out = rb;
    *count_out = rc;
    *exp_mass_out = mass;
}

__device__ float compute_kth_largest_threshold(
    const long long* __restrict__ logit_ptrs,
    int64_t token,
    int64_t local_vocab,
    int world_size,
    int dtype_enum,
    int top_k
) {
    const int64_t total_vocab = local_vocab * (int64_t)world_size;
    if (top_k <= 0 || top_k >= total_vocab) {
        return -CUDART_INF_F;
    }

    int remaining = top_k;
    float prev = CUDART_INF_F;
    float threshold = -CUDART_INF_F;

    // Exact for BF16/FP16/FP32 values, grouping equal values like torch.topk threshold masking.
    for (int iter = 0; iter < top_k; ++iter) {
        float best;
        int cnt;
        float dummy_mass;
        find_next_distinct_desc(
            logit_ptrs, token, local_vocab, world_size, dtype_enum,
            prev, -CUDART_INF_F, 0.0f, &best, &cnt, &dummy_mass
        );

        if (remaining <= cnt || cnt <= 0) {
            threshold = best;
            break;
        }
        remaining -= cnt;
        prev = best;
    }
    return threshold;
}

__global__ void vocab_logprob_compute_kernel(
    const long long* __restrict__ logit_ptrs,
    const void* __restrict__ target,
    float* __restrict__ partial_out,
    int64_t num_tokens,
    int64_t local_vocab,
    int64_t chunk_tokens,
    int64_t local_chunk_tokens_full,
    int world_size,
    int rank,
    int top_k,
    float top_p,
    int dtype_enum,
    int target_dtype_enum
) {
    const int64_t owned_total = num_tokens / world_size;
    const int64_t total_vocab = local_vocab * (int64_t)world_size;

    for (int64_t owned = blockIdx.x; owned < owned_total; owned += gridDim.x) {
        int64_t chunk_id = owned / local_chunk_tokens_full;
        int64_t off_in_owner = owned - chunk_id * local_chunk_tokens_full;
        int64_t chunk_start = chunk_id * chunk_tokens;
        int64_t current = num_tokens - chunk_start;
        if (current > chunk_tokens) current = chunk_tokens;
        int64_t local_tokens = current / world_size;
        int64_t token = chunk_start + (int64_t)rank * local_tokens + off_in_owner;

        int64_t tgt = load_target_id(target, token, target_dtype_enum);
        int tgt_shard = static_cast<int>(tgt / local_vocab);
        int64_t tgt_local = tgt - (int64_t)tgt_shard * local_vocab;

        top_k = top_k < 0 ? 0 : top_k;
        int effective_k = top_k;
        if (effective_k > total_vocab) effective_k = (int)total_vocab;

        float kth_threshold = compute_kth_largest_threshold(
            logit_ptrs, token, local_vocab, world_size, dtype_enum, effective_k
        );

        float target_logit = -CUDART_INF_F;
        if (tgt >= 0 && tgt < total_vocab) {
            target_logit = load_scalar(
                logit_ptrs, tgt_shard, token * local_vocab + tgt_local, dtype_enum
            );
        }

        float local_max = -CUDART_INF_F;
        for (int64_t g = threadIdx.x; g < total_vocab; g += BLOCK_THREADS) {
            int shard = static_cast<int>(g / local_vocab);
            int64_t li = g - (int64_t)shard * local_vocab;
            float x = load_scalar(logit_ptrs, shard, token * local_vocab + li, dtype_enum);
            if (x >= kth_threshold) local_max = fmaxf(local_max, x);
        }
        float maxv = block_reduce_max(local_max);

        float local_sum = 0.0f;
        for (int64_t g = threadIdx.x; g < total_vocab; g += BLOCK_THREADS) {
            int shard = static_cast<int>(g / local_vocab);
            int64_t li = g - (int64_t)shard * local_vocab;
            float x = load_scalar(logit_ptrs, shard, token * local_vocab + li, dtype_enum);
            if (x >= kth_threshold) local_sum += expf(x - maxv);
        }
        float topk_denom = block_reduce_sum(local_sum);

        float final_threshold = kth_threshold;
        float final_denom = topk_denom;

        if (top_p < 1.0f) {
            float need_mass = top_p * topk_denom;
            float accum = 0.0f;
            float prev = CUDART_INF_F;
            float pth = kth_threshold;

            // Descending exact nucleus threshold over distinct values. Equal-value boundary
            // is kept together; this matches threshold-style filtering and is stable for BF16.
            for (int iter = 0; iter < (int)total_vocab; ++iter) {
                float best;
                int cnt;
                float mass;
                find_next_distinct_desc(
                    logit_ptrs, token, local_vocab, world_size, dtype_enum,
                    prev, kth_threshold, maxv, &best, &cnt, &mass
                );

                if (cnt <= 0) {
                    pth = kth_threshold;
                    final_denom = topk_denom;
                    break;
                }

                accum += mass;
                pth = best;
                if (accum >= need_mass) {
                    final_denom = accum;
                    break;
                }
                prev = best;
            }
            final_threshold = pth;
        }

        if (threadIdx.x == 0) {
            float y = -CUDART_INF_F;
            if (target_logit >= final_threshold) {
                y = target_logit - (logf(final_denom) + maxv);
            }
            partial_out[token] = y;
        }
        __syncthreads();
    }
}

__global__ void vocab_logprob_gather_kernel(
    const long long* __restrict__ partial_ptrs,
    float* __restrict__ out,
    int64_t num_tokens,
    int64_t chunk_tokens,
    int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t token = idx; token < num_tokens; token += stride) {
        int64_t chunk_id = token / chunk_tokens;
        int64_t chunk_start = chunk_id * chunk_tokens;
        int64_t current = num_tokens - chunk_start;
        if (current > chunk_tokens) current = chunk_tokens;
        int64_t local_tokens = current / world_size;
        int64_t off = token - chunk_start;
        int owner = static_cast<int>(off / local_tokens);

        const float* src = reinterpret_cast<const float*>(partial_ptrs[owner]);
        out[token] = src[token];
    }
}

int dtype_enum_from_tensor(torch::Tensor x) {
    if (x.scalar_type() == torch::kBFloat16) return 0;
    if (x.scalar_type() == torch::kFloat32) return 1;
    if (x.scalar_type() == torch::kFloat16) return 2;
    TORCH_CHECK(false, "vocab_parallel_logits must be bf16, fp16, or fp32");
}

int target_dtype_enum_from_tensor(torch::Tensor x) {
    if (x.scalar_type() == torch::kInt64) return 0;
    if (x.scalar_type() == torch::kInt32) return 1;
    TORCH_CHECK(false, "target must be int64 or int32");
}

void launch_vocab_logprob_compute(
    torch::Tensor logit_ptrs,
    torch::Tensor target,
    torch::Tensor partial_out,
    int64_t num_tokens,
    int64_t local_vocab,
    int64_t chunk_tokens,
    int64_t local_chunk_tokens_full,
    int world_size,
    int rank,
    int top_k,
    double top_p,
    int dtype_enum
) {
    TORCH_CHECK(logit_ptrs.is_cuda(), "logit_ptrs must be CUDA");
    TORCH_CHECK(target.is_cuda(), "target must be CUDA");
    TORCH_CHECK(partial_out.is_cuda(), "partial_out must be CUDA");
    TORCH_CHECK(partial_out.scalar_type() == torch::kFloat32, "partial_out must be fp32");

    const long long* ptrs = reinterpret_cast<const long long*>(logit_ptrs.data_ptr<int64_t>());
    const void* tgt = target.data_ptr();
    float* pout = partial_out.data_ptr<float>();

    int64_t owned_total = num_tokens / world_size;
    int blocks = (int)owned_total;
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;

    int target_dtype_enum = target_dtype_enum_from_tensor(target);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    vocab_logprob_compute_kernel<<<blocks, BLOCK_THREADS, 0, stream>>>(
        ptrs,
        tgt,
        pout,
        num_tokens,
        local_vocab,
        chunk_tokens,
        local_chunk_tokens_full,
        world_size,
        rank,
        top_k,
        (float)top_p,
        dtype_enum,
        target_dtype_enum
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_vocab_logprob_gather(
    torch::Tensor partial_ptrs,
    torch::Tensor out,
    int64_t num_tokens,
    int64_t chunk_tokens,
    int world_size
) {
    TORCH_CHECK(partial_ptrs.is_cuda(), "partial_ptrs must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be fp32");

    const long long* ptrs = reinterpret_cast<const long long*>(partial_ptrs.data_ptr<int64_t>());
    float* dst = out.data_ptr<float>();

    int threads = 256;
    int blocks = (int)((num_tokens + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    vocab_logprob_gather_kernel<<<blocks, threads, 0, stream>>>(
        ptrs, dst, num_tokens, chunk_tokens, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dtype_enum_from_tensor", &dtype_enum_from_tensor, "dtype enum");
    m.def("launch_vocab_logprob_compute", &launch_vocab_logprob_compute,
          "Symmetric-memory vocab-parallel target logprob compute");
    m.def("launch_vocab_logprob_gather", &launch_vocab_logprob_gather,
          "Symmetric-memory partial logprob gather");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "vocab_parallel_logprob_symm_bf16_h100_ext",
            CUDA_SRC,
        )
    return _ext


_resource_cache = {}


def _group_key(group):
    if group is None:
        return ("world",)
    return (id(group),)


def _get_resources(
    logits_shape,
    logits_dtype,
    target_shape,
    device,
    group,
):
    key = (tuple(logits_shape), logits_dtype, tuple(target_shape), device, _group_key(group))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    num_tokens = int(logits_shape[0]) * int(logits_shape[1])
    local_vocab = int(logits_shape[2])

    logits_buf = symm_mem.empty(
        (num_tokens, local_vocab),
        device=device,
        dtype=logits_dtype,
    )
    logits_hdl = symm_mem.rendezvous(logits_buf, group)

    partial_buf = symm_mem.empty(
        (num_tokens,),
        device=device,
        dtype=torch.float32,
    )
    partial_hdl = symm_mem.rendezvous(partial_buf, group)

    out = torch.empty((num_tokens,), device=device, dtype=torch.float32)

    logit_ptrs = torch.tensor(
        logits_hdl.buffer_ptrs,
        device=device,
        dtype=torch.int64,
    )
    partial_ptrs = torch.tensor(
        partial_hdl.buffer_ptrs,
        device=device,
        dtype=torch.int64,
    )

    res = {
        "logits_buf": logits_buf,
        "logits_hdl": logits_hdl,
        "partial_buf": partial_buf,
        "partial_hdl": partial_hdl,
        "out": out,
        "logit_ptrs": logit_ptrs,
        "partial_ptrs": partial_ptrs,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    chunk_size: int = 1,
) -> torch.Tensor:
    """
    Device-side replacement for chunked all_to_all + top-k/top-p + log_softmax +
    all_gather. Each rank publishes its local vocab shard in symmetric memory,
    computes the exact reference-owned token subset by directly loading peer UVA
    shards, then all ranks gather scalar fp32 log-probs via a peer-pointer kernel.
    """
    assert vocab_parallel_logits.is_cuda, "vocab_parallel_logits must be CUDA"
    assert target.is_cuda, "target must be CUDA"
    assert dist.is_initialized(), "torch.distributed must be initialized"

    tp_group = tp_group or dist.group.WORLD
    world_size = dist.get_world_size(group=tp_group)
    rank = dist.get_rank(group=tp_group)

    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    num_tokens = batch * seq_len
    chunk_tokens = batch * max(1, int(chunk_size))

    if num_tokens % world_size != 0:
        raise ValueError(
            f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}"
        )
    if chunk_tokens % world_size != 0:
        raise ValueError(
            f"B*chunk_size={chunk_tokens} must be divisible by tp size {world_size}"
        )

    local_chunk_tokens_full = chunk_tokens // world_size
    if local_chunk_tokens_full <= 0:
        raise ValueError("chunk_tokens / world_size must be positive")

    ext = _get_ext()

    logits_2d = vocab_parallel_logits.contiguous().view(num_tokens, local_vocab)
    target_flat = target.contiguous().view(num_tokens)

    res = _get_resources(
        vocab_parallel_logits.shape,
        vocab_parallel_logits.dtype,
        target.shape,
        vocab_parallel_logits.device,
        tp_group,
    )

    logits_buf = res["logits_buf"]
    logits_hdl = res["logits_hdl"]
    partial_buf = res["partial_buf"]
    partial_hdl = res["partial_hdl"]
    out = res["out"]

    logits_buf.copy_(logits_2d)
    logits_hdl.barrier(channel=0)

    k = 0 if top_k is None else int(top_k)
    p = 1.0 if top_p is None else float(top_p)
    dtype_enum = ext.dtype_enum_from_tensor(logits_buf)

    ext.launch_vocab_logprob_compute(
        res["logit_ptrs"],
        target_flat,
        partial_buf,
        num_tokens,
        local_vocab,
        chunk_tokens,
        local_chunk_tokens_full,
        world_size,
        rank,
        k,
        p,
        dtype_enum,
    )

    partial_hdl.barrier(channel=1)

    ext.launch_vocab_logprob_gather(
        res["partial_ptrs"],
        out,
        num_tokens,
        chunk_tokens,
        world_size,
    )

    return out.view(batch, seq_len)