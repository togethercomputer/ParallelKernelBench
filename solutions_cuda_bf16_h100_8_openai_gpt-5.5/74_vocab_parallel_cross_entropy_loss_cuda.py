"""
Strategy:
- Replace three NCCL collectives with symmetric-memory float stats buffers and UVA peer loads.
- CUDA kernels compute local row max, global max via peer-pointer reads, local exp-sum/predicted logit,
  then final cross-rank sum via peer-pointer reads.
- Only O(tokens * world_size) data crosses NVLink; logits stay local and are scanned by custom BF16 CUDA.
"""

from typing import Optional, Tuple

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

#define DTYPE_BF16 0
#define DTYPE_F32  1
#define DTYPE_F16  2

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
    int nwarp = (blockDim.x + 31) >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    v = (threadIdx.x < nwarp) ? smem[lane] : -INFINITY;
    if (wid == 0) v = warp_reduce_max(v);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarp = (blockDim.x + 31) >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    v = (threadIdx.x < nwarp) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* p) {
    return *p;
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* p) {
    return __half2float(*p);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* p, float v);

template <>
__device__ __forceinline__ void store_from_float<float>(float* p, float v) {
    *p = v;
}

template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

template <>
__device__ __forceinline__ void store_from_float<__half>(__half* p, float v) {
    *p = __float2half(v);
}

// stats layout, symmetric on every rank:
//   stats[0*N + row] = local max
//   stats[1*N + row] = local shifted predicted logit, or 0
//   stats[2*N + row] = local sum(exp(logit - global_max))
template <typename scalar_t>
__global__ void local_max_kernel(
    const scalar_t* __restrict__ logits,
    float* __restrict__ stats,
    int64_t nrow,
    int64_t vocab
) {
    int64_t row = (int64_t)blockIdx.x;
    if (row >= nrow) return;

    const scalar_t* row_ptr = logits + row * vocab;
    float m = -INFINITY;

    for (int64_t c = threadIdx.x; c < vocab; c += blockDim.x) {
        float x = load_as_float(row_ptr + c);
        m = fmaxf(m, x);
    }

    m = block_reduce_max(m);
    if (threadIdx.x == 0) {
        stats[row] = m;
    }
}

template <typename scalar_t>
__global__ void local_exp_sum_pred_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ target,
    const int64_t* __restrict__ peer_ptrs,
    float* __restrict__ stats,
    int64_t nrow,
    int64_t vocab,
    int64_t vocab_start,
    int world_size
) {
    int64_t row = (int64_t)blockIdx.x;
    if (row >= nrow) return;

    float gmax = -INFINITY;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r < world_size) {
            const float* peer_stats =
                reinterpret_cast<const float*>(static_cast<uintptr_t>(peer_ptrs[r]));
            gmax = fmaxf(gmax, peer_stats[row]);
        }
    }

    const scalar_t* row_ptr = logits + row * vocab;
    float sum = 0.0f;

    for (int64_t c = threadIdx.x; c < vocab; c += blockDim.x) {
        float x = load_as_float(row_ptr + c);
        sum += expf(x - gmax);
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        float pred = 0.0f;
        int64_t t = target[row];
        int64_t local = t - vocab_start;
        if (local >= 0 && local < vocab) {
            pred = load_as_float(row_ptr + local) - gmax;
        }
        stats[nrow + row] = pred;
        stats[2 * nrow + row] = sum;
    }
}

template <typename out_t>
__global__ void final_loss_kernel(
    const int64_t* __restrict__ peer_ptrs,
    out_t* __restrict__ out,
    int64_t nrow,
    int world_size
) {
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; row < nrow; row += stride) {
        float pred = 0.0f;
        float sum_exp = 0.0f;

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r < world_size) {
                const float* peer_stats =
                    reinterpret_cast<const float*>(static_cast<uintptr_t>(peer_ptrs[r]));
                pred += peer_stats[nrow + row];
                sum_exp += peer_stats[2 * nrow + row];
            }
        }

        float loss = logf(sum_exp) - pred;
        store_from_float(out + row, loss);
    }
}

template <typename scalar_t>
void launch_local_max_t(torch::Tensor logits, torch::Tensor stats, int64_t nrow, int64_t vocab) {
    int threads = (vocab >= 2048) ? 512 : 256;
    dim3 grid((unsigned int)nrow);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    local_max_kernel<scalar_t><<<grid, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(logits.data_ptr()),
        stats.data_ptr<float>(),
        nrow,
        vocab
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_local_exp_sum_pred_t(
    torch::Tensor logits,
    torch::Tensor target,
    torch::Tensor peer_ptrs,
    torch::Tensor stats,
    int64_t nrow,
    int64_t vocab,
    int64_t vocab_start,
    int world_size
) {
    int threads = (vocab >= 2048) ? 512 : 256;
    dim3 grid((unsigned int)nrow);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    local_exp_sum_pred_kernel<scalar_t><<<grid, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(logits.data_ptr()),
        target.data_ptr<int64_t>(),
        peer_ptrs.data_ptr<int64_t>(),
        stats.data_ptr<float>(),
        nrow,
        vocab,
        vocab_start,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename out_t>
void launch_final_loss_t(torch::Tensor peer_ptrs, torch::Tensor out, int64_t nrow, int world_size) {
    int threads = 256;
    int blocks = (int)((nrow + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    final_loss_kernel<out_t><<<blocks, threads, 0, stream>>>(
        peer_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<out_t*>(out.data_ptr()),
        nrow,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_local_max(torch::Tensor logits, torch::Tensor stats, int64_t nrow, int64_t vocab, int dtype_enum) {
    TORCH_CHECK(logits.is_cuda() && stats.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(stats.dtype() == torch::kFloat32, "stats must be float32");
    if (dtype_enum == DTYPE_BF16) {
        launch_local_max_t<__nv_bfloat16>(logits, stats, nrow, vocab);
    } else if (dtype_enum == DTYPE_F32) {
        launch_local_max_t<float>(logits, stats, nrow, vocab);
    } else {
        launch_local_max_t<__half>(logits, stats, nrow, vocab);
    }
}

void launch_local_exp_sum_pred(
    torch::Tensor logits,
    torch::Tensor target,
    torch::Tensor peer_ptrs,
    torch::Tensor stats,
    int64_t nrow,
    int64_t vocab,
    int64_t vocab_start,
    int world_size,
    int dtype_enum
) {
    TORCH_CHECK(logits.is_cuda() && target.is_cuda() && peer_ptrs.is_cuda() && stats.is_cuda(),
                "CUDA tensors required");
    TORCH_CHECK(target.dtype() == torch::kInt64, "target must be int64");
    TORCH_CHECK(peer_ptrs.dtype() == torch::kInt64, "peer_ptrs must be int64");
    TORCH_CHECK(stats.dtype() == torch::kFloat32, "stats must be float32");

    if (dtype_enum == DTYPE_BF16) {
        launch_local_exp_sum_pred_t<__nv_bfloat16>(
            logits, target, peer_ptrs, stats, nrow, vocab, vocab_start, world_size);
    } else if (dtype_enum == DTYPE_F32) {
        launch_local_exp_sum_pred_t<float>(
            logits, target, peer_ptrs, stats, nrow, vocab, vocab_start, world_size);
    } else {
        launch_local_exp_sum_pred_t<__half>(
            logits, target, peer_ptrs, stats, nrow, vocab, vocab_start, world_size);
    }
}

void launch_final_loss(torch::Tensor peer_ptrs, torch::Tensor out, int64_t nrow, int world_size, int dtype_enum) {
    TORCH_CHECK(peer_ptrs.is_cuda() && out.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(peer_ptrs.dtype() == torch::kInt64, "peer_ptrs must be int64");

    if (dtype_enum == DTYPE_BF16) {
        launch_final_loss_t<__nv_bfloat16>(peer_ptrs, out, nrow, world_size);
    } else if (dtype_enum == DTYPE_F32) {
        launch_final_loss_t<float>(peer_ptrs, out, nrow, world_size);
    } else {
        launch_final_loss_t<__half>(peer_ptrs, out, nrow, world_size);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_local_max", &launch_local_max, "vocab CE local max");
    m.def("launch_local_exp_sum_pred", &launch_local_exp_sum_pred, "vocab CE local exp sum and pred");
    m.def("launch_final_loss", &launch_final_loss, "vocab CE final peer reduction");
}
'''

_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("vocab_parallel_ce_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


def _vocab_range(partition_vocab_size: int, rank: int) -> Tuple[int, int]:
    start = rank * partition_vocab_size
    return start, start + partition_vocab_size


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"unsupported logits dtype: {dtype}; expected bf16/fp16/fp32")


def _get_resources(nrow: int, out_dtype: torch.dtype, device: torch.device, group):
    group_key = id(group)
    dev_key = device.index if device.index is not None else torch.cuda.current_device()
    world_size = dist.get_world_size(group=group)
    key = (nrow, out_dtype, dev_key, group_key, world_size)

    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    stats = symm_mem.empty((3, nrow), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(stats, group)
    out = torch.empty((nrow,), device=device, dtype=out_dtype)
    peer_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (stats, hdl, out, peer_ptrs)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Vocab-parallel cross entropy using custom CUDA + symmetric-memory peer loads.
    Inputs:
      vocab_parallel_logits: [*, V/world_size], optimized for BF16 CUDA contiguous-ish logits.
      target: [*] int64 full-vocab token ids.
      group: model-parallel process group, WORLD when None.
    Returns:
      loss: [*], same dtype as logits for bf16/fp16/fp32.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert vocab_parallel_logits.is_cuda and target.is_cuda, "inputs must be CUDA tensors"
    assert vocab_parallel_logits.dim() >= 1, "logits must have a vocab dimension"
    assert target.dtype == torch.long, "target must be int64/torch.long"

    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    logits = vocab_parallel_logits if vocab_parallel_logits.is_contiguous() else vocab_parallel_logits.contiguous()
    tgt = target if target.is_contiguous() else target.contiguous()

    partition_vocab_size = int(logits.shape[-1])
    nrow = int(logits.numel() // partition_vocab_size)

    if nrow == 0:
        return torch.empty(tgt.shape, device=logits.device, dtype=logits.dtype)

    expected_target_elems = tgt.numel()
    assert expected_target_elems == nrow, "target shape must match logits.shape[:-1]"

    dtype_enum = _dtype_enum(logits.dtype)
    vocab_start, _ = _vocab_range(partition_vocab_size, rank)

    stats, hdl, out, peer_ptrs = _get_resources(nrow, logits.dtype, logits.device, group)
    ext = _get_ext()

    # 1) Per-rank row-wise max into symmetric stats[0].
    ext.launch_local_max(
        logits,
        stats,
        nrow,
        partition_vocab_size,
        dtype_enum,
    )

    # Make every rank's local max visible before peer UVA reads.
    hdl.barrier(channel=0)

    # 2) Read all peer max values, compute local exp sum and shifted predicted logit.
    ext.launch_local_exp_sum_pred(
        logits,
        tgt,
        peer_ptrs,
        stats,
        nrow,
        partition_vocab_size,
        int(vocab_start),
        int(world_size),
        dtype_enum,
    )

    # Make stats[1:3] visible before final peer reductions.
    hdl.barrier(channel=1)

    # 3) Sum tiny per-token stats across ranks with UVA loads; write final loss.
    ext.launch_final_loss(
        peer_ptrs,
        out,
        nrow,
        int(world_size),
        dtype_enum,
    )

    return out.reshape_as(tgt)