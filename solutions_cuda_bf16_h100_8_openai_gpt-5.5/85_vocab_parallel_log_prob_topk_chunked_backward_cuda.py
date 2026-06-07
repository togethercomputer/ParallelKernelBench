# Strategy:
# - Replace both all_to_all_single calls with symmetric-memory UVA loads/stores.
# - Each rank copies its local BF16 vocab shard once into symmetric memory.
# - For each chunk, rank r owns the same token slice as the reference seq-parallel transpose,
#   computes filtered softmax gradients, then writes each vocab shard directly into the
#   destination rank's symmetric FP32 grad buffer.
# - No NCCL collectives are used on the hot path; synchronization is via symm_mem rendezvous/barriers.

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#ifndef THREADS
#define THREADS 256
#endif

__device__ __forceinline__ uint16_t bf16_to_ordered_key(uint16_t b) {
    // Monotonic key for IEEE-like bf16 values.
    // Negative values are bitwise inverted; positives flip sign bit.
    return (b & 0x8000u) ? (uint16_t)(~b) : (uint16_t)(b ^ 0x8000u);
}

__device__ __forceinline__ float load_bf16_value(
    const long long* __restrict__ ptrs,
    int vocab_rank,
    int64_t row,
    int64_t local_vocab,
    int64_t col
) {
    const __nv_bfloat16* base =
        reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[vocab_rank]);
    return __bfloat162float(base[row * local_vocab + col]);
}

__device__ __forceinline__ uint16_t load_bf16_key(
    const long long* __restrict__ ptrs,
    int vocab_rank,
    int64_t row,
    int64_t local_vocab,
    int64_t col
) {
    const uint16_t* base =
        reinterpret_cast<const uint16_t*>((uintptr_t)ptrs[vocab_rank]);
    return bf16_to_ordered_key(base[row * local_vocab + col]);
}

__global__ void copy_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        dst[idx] = src[idx];
    }
}

__global__ void vocab_parallel_logprob_backward_kernel(
    const long long* __restrict__ input_ptrs,
    const long long* __restrict__ grad_ptrs,
    const int64_t* __restrict__ target,
    const float* __restrict__ grad_output,
    int64_t chunk_start,
    int64_t local_tokens,
    int64_t local_vocab,
    int world_size,
    int rank,
    int top_k,
    float top_p
) {
    const int tid = threadIdx.x;
    const int64_t row_in_owned_slice = (int64_t)blockIdx.x;
    const int64_t row_abs = chunk_start + (int64_t)rank * local_tokens + row_in_owned_slice;
    const int64_t vocab_size = (int64_t)world_size * local_vocab;

    __shared__ float red[THREADS];
    __shared__ uint16_t filter_key;
    __shared__ int need_filter;

    const bool need_k = top_k > 0 && top_k < vocab_size;
    const bool need_p = top_p < 1.0f;

    if (tid == 0) {
        need_filter = (need_k || need_p) ? 1 : 0;
        filter_key = 0;

        uint16_t topk_key = 0;
        if (need_k || need_p) {
            if (need_k) {
                const int64_t k = top_k < vocab_size ? (int64_t)top_k : vocab_size;

                int lo = 0;
                int hi = 65535;
                int ans = 0;

                while (lo <= hi) {
                    int mid = (lo + hi) >> 1;
                    int64_t count_ge = 0;

                    for (int64_t idx = 0; idx < vocab_size; ++idx) {
                        const int vr = (int)(idx / local_vocab);
                        const int64_t col = idx - (int64_t)vr * local_vocab;
                        uint16_t key = load_bf16_key(input_ptrs, vr, row_abs, local_vocab, col);
                        if ((int)key >= mid) {
                            ++count_ge;
                        }
                    }

                    if (count_ge >= k) {
                        ans = mid;
                        lo = mid + 1;
                    } else {
                        hi = mid - 1;
                    }
                }
                topk_key = (uint16_t)ans;
            } else {
                topk_key = 0;
            }

            if (!need_p) {
                filter_key = topk_key;
            } else {
                float maxv = -INFINITY;
                for (int64_t idx = 0; idx < vocab_size; ++idx) {
                    const int vr = (int)(idx / local_vocab);
                    const int64_t col = idx - (int64_t)vr * local_vocab;
                    uint16_t key = load_bf16_key(input_ptrs, vr, row_abs, local_vocab, col);
                    if (key >= topk_key) {
                        float v = load_bf16_value(input_ptrs, vr, row_abs, local_vocab, col);
                        maxv = fmaxf(maxv, v);
                    }
                }

                float total = 0.0f;
                for (int64_t idx = 0; idx < vocab_size; ++idx) {
                    const int vr = (int)(idx / local_vocab);
                    const int64_t col = idx - (int64_t)vr * local_vocab;
                    uint16_t key = load_bf16_key(input_ptrs, vr, row_abs, local_vocab, col);
                    if (key >= topk_key) {
                        float v = load_bf16_value(input_ptrs, vr, row_abs, local_vocab, col);
                        total += expf(v - maxv);
                    }
                }

                const float target_mass = fmaxf(0.0f, top_p) * total;
                float cumulative = 0.0f;
                int upper = 65536;
                uint16_t chosen = topk_key;

                while (upper > 0) {
                    int best_key = -1;
                    for (int64_t idx = 0; idx < vocab_size; ++idx) {
                        const int vr = (int)(idx / local_vocab);
                        const int64_t col = idx - (int64_t)vr * local_vocab;
                        uint16_t key = load_bf16_key(input_ptrs, vr, row_abs, local_vocab, col);
                        if (key >= topk_key && (int)key < upper && (int)key > best_key) {
                            best_key = (int)key;
                        }
                    }

                    if (best_key < 0) {
                        break;
                    }

                    float group_sum = 0.0f;
                    for (int64_t idx = 0; idx < vocab_size; ++idx) {
                        const int vr = (int)(idx / local_vocab);
                        const int64_t col = idx - (int64_t)vr * local_vocab;
                        uint16_t key = load_bf16_key(input_ptrs, vr, row_abs, local_vocab, col);
                        if ((int)key == best_key) {
                            float v = load_bf16_value(input_ptrs, vr, row_abs, local_vocab, col);
                            group_sum += expf(v - maxv);
                        }
                    }

                    cumulative += group_sum;
                    chosen = (uint16_t)best_key;

                    if (cumulative >= target_mass) {
                        break;
                    }
                    upper = best_key;
                }

                filter_key = chosen;
            }
        }
    }

    __syncthreads();

    float local_max = -INFINITY;
    for (int64_t idx = tid; idx < vocab_size; idx += THREADS) {
        const int vr = (int)(idx / local_vocab);
        const int64_t col = idx - (int64_t)vr * local_vocab;

        bool keep = true;
        if (need_filter) {
            uint16_t key = load_bf16_key(input_ptrs, vr, row_abs, local_vocab, col);
            keep = key >= filter_key;
        }

        if (keep) {
            float v = load_bf16_value(input_ptrs, vr, row_abs, local_vocab, col);
            local_max = fmaxf(local_max, v);
        }
    }

    red[tid] = local_max;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] = fmaxf(red[tid], red[tid + stride]);
        }
        __syncthreads();
    }

    const float row_max = red[0];

    float local_sum = 0.0f;
    for (int64_t idx = tid; idx < vocab_size; idx += THREADS) {
        const int vr = (int)(idx / local_vocab);
        const int64_t col = idx - (int64_t)vr * local_vocab;

        bool keep = true;
        if (need_filter) {
            uint16_t key = load_bf16_key(input_ptrs, vr, row_abs, local_vocab, col);
            keep = key >= filter_key;
        }

        if (keep) {
            float v = load_bf16_value(input_ptrs, vr, row_abs, local_vocab, col);
            local_sum += expf(v - row_max);
        }
    }

    red[tid] = local_sum;
    __syncthreads();

    for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }

    const float denom = red[0];
    const int64_t tgt = target[row_abs];
    const float gout = grad_output[row_abs];

    for (int64_t idx = tid; idx < vocab_size; idx += THREADS) {
        const int vr = (int)(idx / local_vocab);
        const int64_t col = idx - (int64_t)vr * local_vocab;

        bool keep = true;
        if (need_filter) {
            uint16_t key = load_bf16_key(input_ptrs, vr, row_abs, local_vocab, col);
            keep = key >= filter_key;
        }

        float outv = 0.0f;
        if (keep) {
            float v = load_bf16_value(input_ptrs, vr, row_abs, local_vocab, col);
            float p = expf(v - row_max) / denom;
            outv = ((idx == tgt) ? (1.0f - p) : (-p)) * gout;
        }

        float* dst =
            reinterpret_cast<float*>((uintptr_t)grad_ptrs[vr]);
        dst[row_abs * local_vocab + col] = outv;
    }
}

void copy_bf16_to_symmetric(torch::Tensor src, torch::Tensor dst, int64_t n) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA");
    TORCH_CHECK(src.scalar_type() == torch::kBFloat16, "src must be bf16");
    TORCH_CHECK(dst.scalar_type() == torch::kBFloat16, "dst must be bf16");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_backward_bf16(
    torch::Tensor input_ptrs_tensor,
    torch::Tensor grad_ptrs_tensor,
    torch::Tensor target,
    torch::Tensor grad_output,
    int64_t num_tokens,
    int64_t local_vocab,
    int64_t chunk_tokens,
    int world_size,
    int rank,
    int top_k,
    double top_p_double
) {
    TORCH_CHECK(input_ptrs_tensor.is_cuda(), "input ptrs must be CUDA");
    TORCH_CHECK(grad_ptrs_tensor.is_cuda(), "grad ptrs must be CUDA");
    TORCH_CHECK(target.is_cuda() && target.is_contiguous(), "target must be contiguous CUDA");
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(), "grad_output must be contiguous CUDA");
    TORCH_CHECK(target.scalar_type() == torch::kLong, "target must be int64");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");

    const long long* input_ptrs =
        reinterpret_cast<const long long*>(input_ptrs_tensor.data_ptr<int64_t>());
    const long long* grad_ptrs =
        reinterpret_cast<const long long*>(grad_ptrs_tensor.data_ptr<int64_t>());

    const int64_t* target_ptr = target.data_ptr<int64_t>();
    const float* grad_ptr = grad_output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const float top_p = (float)top_p_double;

    for (int64_t start = 0; start < num_tokens; start += chunk_tokens) {
        int64_t current = chunk_tokens;
        if (start + current > num_tokens) {
            current = num_tokens - start;
        }

        int64_t local_tokens = current / world_size;
        if (local_tokens <= 0) {
            continue;
        }

        vocab_parallel_logprob_backward_kernel<<<(int)local_tokens, THREADS, 0, stream>>>(
            input_ptrs,
            grad_ptrs,
            target_ptr,
            grad_ptr,
            start,
            local_tokens,
            local_vocab,
            world_size,
            rank,
            top_k,
            top_p
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void sync_current_stream() {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_bf16_to_symmetric", &copy_bf16_to_symmetric,
          "Copy BF16 contiguous tensor to symmetric BF16 buffer");
    m.def("launch_backward_bf16", &launch_backward_bf16,
          "Chunked vocab-parallel target logprob backward using UVA symmetric memory");
    m.def("sync_current_stream", &sync_current_stream,
          "Synchronize current CUDA stream");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "vocab_parallel_logprob_backward_symm_bf16_h100_ext",
            CUDA_SRC,
        )
    return _ext


_resource_cache = {}


def _get_resources(
    num_tokens: int,
    local_vocab: int,
    device: torch.device,
    tp_group: dist.ProcessGroup,
):
    key = (num_tokens, local_vocab, device, id(tp_group))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    in_buf = symm_mem.empty((num_tokens * local_vocab,), device=device, dtype=torch.bfloat16)
    in_hdl = symm_mem.rendezvous(in_buf, tp_group)

    grad_buf = symm_mem.empty((num_tokens * local_vocab,), device=device, dtype=torch.float32)
    grad_hdl = symm_mem.rendezvous(grad_buf, tp_group)

    in_ptrs = torch.tensor(in_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    grad_ptrs = torch.tensor(grad_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "in_buf": in_buf,
        "in_hdl": in_hdl,
        "grad_buf": grad_buf,
        "grad_hdl": grad_hdl,
        "in_ptrs": in_ptrs,
        "grad_ptrs": grad_ptrs,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    chunk_size: int = 1,
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert vocab_parallel_logits.is_cuda, "vocab_parallel_logits must be CUDA"
    assert target.is_cuda and grad_output.is_cuda, "target/grad_output must be CUDA"
    assert vocab_parallel_logits.dtype == torch.bfloat16, "optimized path expects BF16 logits"
    assert target.dtype == torch.long, "target must be int64"
    assert grad_output.dtype == torch.float32, "grad_output must be float32"

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

    if not vocab_parallel_logits.is_contiguous():
        vocab_parallel_logits = vocab_parallel_logits.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    ext = _get_ext()
    resources = _get_resources(
        num_tokens=num_tokens,
        local_vocab=local_vocab,
        device=vocab_parallel_logits.device,
        tp_group=tp_group,
    )

    flat_logits = vocab_parallel_logits.reshape(-1)
    flat_target = target.reshape(-1)
    flat_grad = grad_output.reshape(-1)

    ext.copy_bf16_to_symmetric(
        flat_logits,
        resources["in_buf"],
        flat_logits.numel(),
    )

    # Make the local staging copy visible before peers start UVA reads.
    ext.sync_current_stream()
    resources["in_hdl"].barrier(channel=0)

    k_arg = -1 if top_k is None else int(top_k)
    p_arg = 1.0 if top_p is None else float(top_p)

    ext.launch_backward_bf16(
        resources["in_ptrs"],
        resources["grad_ptrs"],
        flat_target,
        flat_grad,
        int(num_tokens),
        int(local_vocab),
        int(chunk_tokens),
        int(world_size),
        int(rank),
        int(k_arg),
        float(p_arg),
    )

    # Ensure this rank's remote stores are complete, then wait for peer stores
    # into our symmetric grad buffer before returning it.
    ext.sync_current_stream()
    resources["grad_hdl"].barrier(channel=1)

    return resources["grad_buf"].reshape(batch, seq_len, local_vocab)