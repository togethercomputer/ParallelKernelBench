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
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

static constexpr int THREADS = 256;

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : -INFINITY;
    if (wid == 0) v = warp_reduce_max(v);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } x;
    x.u = ((uint32_t)h) << 16;
    return x.f;
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<uint16_t*>(&b);
}

__device__ __forceinline__ float load_value(
    const long long* __restrict__ ptrs,
    int shard,
    int64_t offset,
    int dtype_enum
) {
    const char* base = reinterpret_cast<const char*>(ptrs[shard]);
    if (dtype_enum == 0) {
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(base);
        return __bfloat162float(p[offset]);
    } else if (dtype_enum == 1) {
        const float* p = reinterpret_cast<const float*>(base);
        return p[offset];
    } else {
        const __half* p = reinterpret_cast<const __half*>(base);
        return __half2float(p[offset]);
    }
}

__device__ __forceinline__ uint16_t load_bf16_bits(
    const long long* __restrict__ ptrs,
    int shard,
    int64_t offset
) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(
        reinterpret_cast<const char*>(ptrs[shard]));
    return p[offset];
}

__global__ void logprob_fast_kernel(
    const long long* __restrict__ logits_ptrs,
    const int64_t* __restrict__ target,
    float* __restrict__ out_local,
    int64_t num_tokens,
    int local_vocab,
    int rank,
    int world_size,
    int local_tokens,
    int dtype_enum
) {
    int row = blockIdx.x;
    if (row >= local_tokens) return;

    int64_t global_tok = (int64_t)rank * local_tokens + row;
    int full_vocab = world_size * local_vocab;

    int64_t tgt = target[global_tok];
    int tgt_shard = (int)(tgt / local_vocab);
    int tgt_local = (int)(tgt - (int64_t)tgt_shard * local_vocab);

    float local_max = -INFINITY;
    for (int v = threadIdx.x; v < full_vocab; v += blockDim.x) {
        int shard = v / local_vocab;
        int lv = v - shard * local_vocab;
        float x = load_value(
            logits_ptrs, shard,
            global_tok * (int64_t)local_vocab + lv,
            dtype_enum);
        local_max = fmaxf(local_max, x);
    }

    float maxv = block_reduce_max(local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < full_vocab; v += blockDim.x) {
        int shard = v / local_vocab;
        int lv = v - shard * local_vocab;
        float x = load_value(
            logits_ptrs, shard,
            global_tok * (int64_t)local_vocab + lv,
            dtype_enum);
        local_sum += expf(x - maxv);
    }

    float denom = block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        float tv = load_value(
            logits_ptrs,
            tgt_shard,
            global_tok * (int64_t)local_vocab + tgt_local,
            dtype_enum);
        out_local[row] = tv - maxv - logf(denom);
    }
}

__global__ void logprob_filtered_sort_bf16_kernel(
    const long long* __restrict__ logits_ptrs,
    const int64_t* __restrict__ target,
    float* __restrict__ out_local,
    int local_vocab,
    int rank,
    int world_size,
    int local_tokens,
    int top_k,
    float top_p,
    int n_pow2
) {
    extern __shared__ uint16_t vals[];
    int row = blockIdx.x;
    if (row >= local_tokens) return;

    int full_vocab = world_size * local_vocab;
    int64_t global_tok = (int64_t)rank * local_tokens + row;

    uint16_t pos_inf = 0x7f80u;

    for (int i = threadIdx.x; i < n_pow2; i += blockDim.x) {
        if (i < full_vocab) {
            int shard = i / local_vocab;
            int lv = i - shard * local_vocab;
            vals[i] = load_bf16_bits(
                logits_ptrs, shard,
                global_tok * (int64_t)local_vocab + lv);
        } else {
            vals[i] = pos_inf;
        }
    }
    __syncthreads();

    for (int k = 2; k <= n_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n_pow2; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    float a = bf16_bits_to_float(vals[i]);
                    float b = bf16_bits_to_float(vals[ixj]);
                    bool up = ((i & k) == 0);
                    if ((up && a > b) || (!up && a < b)) {
                        uint16_t tmp = vals[i];
                        vals[i] = vals[ixj];
                        vals[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int64_t tgt = target[global_tok];
        int tgt_shard = (int)(tgt / local_vocab);
        int tgt_local = (int)(tgt - (int64_t)tgt_shard * local_vocab);
        float target_val = bf16_bits_to_float(load_bf16_bits(
            logits_ptrs, tgt_shard,
            global_tok * (int64_t)local_vocab + tgt_local));

        int start = 0;
        float kth_threshold = -INFINITY;
        if (top_k > 0 && top_k < full_vocab) {
            kth_threshold = bf16_bits_to_float(vals[full_vocab - top_k]);
            while (start < full_vocab &&
                   bf16_bits_to_float(vals[start]) < kth_threshold) {
                ++start;
            }
        }

        float maxv = bf16_bits_to_float(vals[full_vocab - 1]);
        float denom = 0.0f;
        for (int i = start; i < full_vocab; ++i) {
            denom += expf(bf16_bits_to_float(vals[i]) - maxv);
        }

        int keep_start = start;
        if (top_p < 1.0f) {
            float cutoff = (1.0f - top_p) * denom;
            float cum = 0.0f;
            keep_start = full_vocab - 1;
            for (int i = start; i < full_vocab; ++i) {
                float e = expf(bf16_bits_to_float(vals[i]) - maxv);
                if (cum + e > cutoff) {
                    keep_start = i;
                    break;
                }
                cum += e;
            }
            denom = 0.0f;
            for (int i = keep_start; i < full_vocab; ++i) {
                denom += expf(bf16_bits_to_float(vals[i]) - maxv);
            }
        }

        float keep_threshold = bf16_bits_to_float(vals[keep_start]);
        bool keep = target_val >= keep_threshold;
        out_local[row] = keep ? (target_val - maxv - logf(denom)) : -INFINITY;
    }
}

__global__ void logprob_filtered_sort_f32_kernel(
    const long long* __restrict__ logits_ptrs,
    const int64_t* __restrict__ target,
    float* __restrict__ out_local,
    int local_vocab,
    int rank,
    int world_size,
    int local_tokens,
    int top_k,
    float top_p,
    int dtype_enum,
    int n_pow2
) {
    extern __shared__ float vals[];
    int row = blockIdx.x;
    if (row >= local_tokens) return;

    int full_vocab = world_size * local_vocab;
    int64_t global_tok = (int64_t)rank * local_tokens + row;

    for (int i = threadIdx.x; i < n_pow2; i += blockDim.x) {
        if (i < full_vocab) {
            int shard = i / local_vocab;
            int lv = i - shard * local_vocab;
            vals[i] = load_value(
                logits_ptrs, shard,
                global_tok * (int64_t)local_vocab + lv,
                dtype_enum);
        } else {
            vals[i] = INFINITY;
        }
    }
    __syncthreads();

    for (int k = 2; k <= n_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n_pow2; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    float a = vals[i];
                    float b = vals[ixj];
                    bool up = ((i & k) == 0);
                    if ((up && a > b) || (!up && a < b)) {
                        vals[i] = b;
                        vals[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        int64_t tgt = target[global_tok];
        int tgt_shard = (int)(tgt / local_vocab);
        int tgt_local = (int)(tgt - (int64_t)tgt_shard * local_vocab);
        float target_val = load_value(
            logits_ptrs,
            tgt_shard,
            global_tok * (int64_t)local_vocab + tgt_local,
            dtype_enum);

        int start = 0;
        if (top_k > 0 && top_k < full_vocab) {
            float kth_threshold = vals[full_vocab - top_k];
            while (start < full_vocab && vals[start] < kth_threshold) ++start;
        }

        float maxv = vals[full_vocab - 1];
        float denom = 0.0f;
        for (int i = start; i < full_vocab; ++i) {
            denom += expf(vals[i] - maxv);
        }

        int keep_start = start;
        if (top_p < 1.0f) {
            float cutoff = (1.0f - top_p) * denom;
            float cum = 0.0f;
            keep_start = full_vocab - 1;
            for (int i = start; i < full_vocab; ++i) {
                float e = expf(vals[i] - maxv);
                if (cum + e > cutoff) {
                    keep_start = i;
                    break;
                }
                cum += e;
            }
            denom = 0.0f;
            for (int i = keep_start; i < full_vocab; ++i) {
                denom += expf(vals[i] - maxv);
            }
        }

        bool keep = target_val >= vals[keep_start];
        out_local[row] = keep ? (target_val - maxv - logf(denom)) : -INFINITY;
    }
}

__global__ void gather_peer_outputs_kernel(
    const long long* __restrict__ out_ptrs,
    float* __restrict__ final_out,
    int64_t num_tokens,
    int local_tokens
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < num_tokens; idx += (int64_t)gridDim.x * blockDim.x) {
        int owner = (int)(idx / local_tokens);
        int off = (int)(idx - (int64_t)owner * local_tokens);
        const float* src = reinterpret_cast<const float*>(out_ptrs[owner]);
        final_out[idx] = src[off];
    }
}

void copy_device_bytes(torch::Tensor src, torch::Tensor dst, int64_t nbytes) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "copy tensors must be CUDA");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), (size_t)nbytes,
                    cudaMemcpyDeviceToDevice, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_compute_logprobs(
    torch::Tensor logits_ptrs,
    torch::Tensor target,
    torch::Tensor out_local,
    int64_t num_tokens,
    int local_vocab,
    int rank,
    int world_size,
    int local_tokens,
    int top_k,
    double top_p,
    int dtype_enum,
    int n_pow2
) {
    TORCH_CHECK(logits_ptrs.is_cuda(), "logits_ptrs must be CUDA");
    TORCH_CHECK(target.is_cuda(), "target must be CUDA");
    TORCH_CHECK(out_local.is_cuda(), "out_local must be CUDA");
    TORCH_CHECK(target.dtype() == torch::kInt64, "target must be int64");
    TORCH_CHECK(out_local.dtype() == torch::kFloat32, "out_local must be fp32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    bool need_filter = (top_k > 0) || (top_p < 1.0);
    if (!need_filter) {
        logprob_fast_kernel<<<local_tokens, THREADS, 0, stream>>>(
            reinterpret_cast<const long long*>(logits_ptrs.data_ptr<int64_t>()),
            target.data_ptr<int64_t>(),
            out_local.data_ptr<float>(),
            num_tokens,
            local_vocab,
            rank,
            world_size,
            local_tokens,
            dtype_enum);
    } else if (dtype_enum == 0) {
        size_t shmem = (size_t)n_pow2 * sizeof(uint16_t);
        cudaFuncSetAttribute(
            logprob_filtered_sort_bf16_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem);
        cudaFuncSetAttribute(
            logprob_filtered_sort_bf16_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100);
        logprob_filtered_sort_bf16_kernel<<<local_tokens, THREADS, shmem, stream>>>(
            reinterpret_cast<const long long*>(logits_ptrs.data_ptr<int64_t>()),
            target.data_ptr<int64_t>(),
            out_local.data_ptr<float>(),
            local_vocab,
            rank,
            world_size,
            local_tokens,
            top_k,
            (float)top_p,
            n_pow2);
    } else {
        size_t shmem = (size_t)n_pow2 * sizeof(float);
        cudaFuncSetAttribute(
            logprob_filtered_sort_f32_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem);
        cudaFuncSetAttribute(
            logprob_filtered_sort_f32_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100);
        logprob_filtered_sort_f32_kernel<<<local_tokens, THREADS, shmem, stream>>>(
            reinterpret_cast<const long long*>(logits_ptrs.data_ptr<int64_t>()),
            target.data_ptr<int64_t>(),
            out_local.data_ptr<float>(),
            local_vocab,
            rank,
            world_size,
            local_tokens,
            top_k,
            (float)top_p,
            dtype_enum,
            n_pow2);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_peer_outputs(
    torch::Tensor out_ptrs,
    torch::Tensor final_out,
    int64_t num_tokens,
    int local_tokens
) {
    TORCH_CHECK(out_ptrs.is_cuda(), "out_ptrs must be CUDA");
    TORCH_CHECK(final_out.is_cuda(), "final_out must be CUDA");
    TORCH_CHECK(final_out.dtype() == torch::kFloat32, "final_out must be fp32");

    int threads = 256;
    int blocks = (int)((num_tokens + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gather_peer_outputs_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(out_ptrs.data_ptr<int64_t>()),
        final_out.data_ptr<float>(),
        num_tokens,
        local_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_device_bytes", &copy_device_bytes, "D2D async copy");
    m.def("launch_compute_logprobs", &launch_compute_logprobs,
          "UVA vocab-parallel filtered target logprobs");
    m.def("launch_gather_peer_outputs", &launch_gather_peer_outputs,
          "UVA all-gather replacement for fp32 token logprobs");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("vp_logprob_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"unsupported logits dtype: {dtype}")


def _get_resources(
    shape,
    dtype: torch.dtype,
    device: torch.device,
    local_tokens: int,
    tp_group,
):
    key = (tuple(shape), dtype, device, local_tokens, id(tp_group))
    res = _resource_cache.get(key)
    if res is not None:
        return res

    logits_buf = symm_mem.empty(shape, device=device, dtype=dtype)
    logits_hdl = symm_mem.rendezvous(logits_buf, tp_group)

    partial_buf = symm_mem.empty((local_tokens,), device=device, dtype=torch.float32)
    partial_hdl = symm_mem.rendezvous(partial_buf, tp_group)

    logits_ptrs = torch.tensor(
        logits_hdl.buffer_ptrs, device=device, dtype=torch.int64
    )
    partial_ptrs = torch.tensor(
        partial_hdl.buffer_ptrs, device=device, dtype=torch.int64
    )
    final_out = torch.empty((shape[0],), device=device, dtype=torch.float32)

    res = {
        "logits_buf": logits_buf,
        "logits_hdl": logits_hdl,
        "partial_buf": partial_buf,
        "partial_hdl": partial_hdl,
        "logits_ptrs": logits_ptrs,
        "partial_ptrs": partial_ptrs,
        "final_out": final_out,
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
) -> torch.Tensor:
    tp_group = tp_group or dist.group.WORLD
    world_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)

    batch, seq_len, local_vocab = vocab_parallel_logits.shape
    num_tokens = batch * seq_len
    if num_tokens % world_size != 0:
        raise ValueError(
            f"B*S={num_tokens} must be divisible by tensor parallel size {world_size}"
        )

    local_tokens = num_tokens // world_size
    full_vocab = world_size * local_vocab

    logits_2d = vocab_parallel_logits.reshape(num_tokens, local_vocab)
    if not logits_2d.is_contiguous():
        logits_2d = logits_2d.contiguous()

    target_flat = target.reshape(-1)
    if not target_flat.is_contiguous():
        target_flat = target_flat.contiguous()
    if target_flat.dtype != torch.long:
        target_flat = target_flat.to(dtype=torch.long)

    dtype_enum = _dtype_enum(logits_2d.dtype)
    ext = _get_ext()

    res = _get_resources(
        (num_tokens, local_vocab),
        logits_2d.dtype,
        logits_2d.device,
        local_tokens,
        tp_group,
    )

    nbytes = logits_2d.numel() * logits_2d.element_size()
    ext.copy_device_bytes(logits_2d, res["logits_buf"], nbytes)

    # Publish this rank's vocab shard before peer UVA reads.
    res["logits_hdl"].barrier(channel=0)

    need_k = top_k is not None and int(top_k) > 0
    need_p = top_p is not None and float(top_p) < 1.0

    top_k_eff = min(int(top_k), full_vocab) if need_k else 0
    top_p_eff = float(top_p) if need_p else 1.0
    n_pow2 = _next_power_of_2(full_vocab) if (need_k or need_p) else 1

    ext.launch_compute_logprobs(
        res["logits_ptrs"],
        target_flat,
        res["partial_buf"],
        num_tokens,
        local_vocab,
        rank,
        world_size,
        local_tokens,
        top_k_eff,
        top_p_eff,
        dtype_enum,
        n_pow2,
    )

    # Publish local token slice log-probs before custom peer-read all-gather.
    res["partial_hdl"].barrier(channel=1)

    ext.launch_gather_peer_outputs(
        res["partial_ptrs"],
        res["final_out"],
        num_tokens,
        local_tokens,
    )

    return res["final_out"].reshape(batch, seq_len)