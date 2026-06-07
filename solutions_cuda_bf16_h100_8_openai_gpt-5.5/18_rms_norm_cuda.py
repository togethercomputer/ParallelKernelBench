# Device-side RMSNorm for tensor-parallel hidden partitioning.
# Strategy: each persistent CUDA block computes one/more rows' local FP32 square sum,
# publishes it in symmetric memory, uses a lightweight signal-pad GPU barrier, then
# reads peer row sums directly through UVA pointers and writes the locally scaled BF16 output.

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
#include <cstdint>

__device__ __forceinline__ void signal_send_release(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
        if (old != 0u) {
            __nanosleep(32);
        }
    } while (old != 0u);
}

__device__ __forceinline__ void signal_wait_acquire(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
        if (old != 1u) {
            __nanosleep(32);
        }
    } while (old != 1u);
}

__device__ __forceinline__ void blockwise_barrier_acq_rel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t slot,
    int rank,
    int world_size
) {
    const int tid = threadIdx.x;
    if (tid < world_size) {
        const uint64_t local_base = signal_pad_ptrs[rank];
        const uint64_t remote_base = signal_pad_ptrs[tid];

        const uint64_t send_off = (slot * (uint64_t)world_size + (uint64_t)rank) * sizeof(uint32_t);
        const uint64_t wait_off = (slot * (uint64_t)world_size + (uint64_t)tid) * sizeof(uint32_t);

        uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base + send_off);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base + wait_off);

        signal_send_release(send_addr);
        signal_wait_acquire(wait_addr);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, mask);
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float warp_partials[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = warp_sum(v);
    if (lane == 0) {
        warp_partials[warp] = v;
    }
    __syncthreads();

    const int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? warp_partials[lane] : 0.0f;
    if (warp == 0) {
        v = warp_sum(v);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <>
__device__ __forceinline__ float to_float<half>(half x) {
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

template <>
__device__ __forceinline__ half from_float<half>(float x) {
    return __float2half_rn(x);
}

template <typename T>
__global__ void rmsnorm_tp_symm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ out,
    float* __restrict__ local_sums,
    const uint64_t* __restrict__ sum_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t rows,
    int64_t cols,
    float inv_global_hidden,
    float eps,
    int rank,
    int world_size
) {
    __shared__ float inv_rms_s;

    const uint64_t barrier_slot = (uint64_t)blockIdx.x;

    for (int64_t row = (int64_t)blockIdx.x; row < rows; row += (int64_t)gridDim.x) {
        const int64_t base = row * cols;

        float ss = 0.0f;
        for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
            const float v = to_float<T>(x[base + c]);
            ss += v * v;
        }

        const float row_sum = block_sum(ss);

        if (threadIdx.x == 0) {
            local_sums[row] = row_sum;
            __threadfence_system();
        }
        __syncthreads();

        blockwise_barrier_acq_rel(signal_pad_ptrs, barrier_slot, rank, world_size);
        __syncthreads();

        if (threadIdx.x == 0) {
            float global_ss = 0.0f;
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r < world_size) {
                    const float* peer_sums = reinterpret_cast<const float*>(sum_ptrs[r]);
                    global_ss += peer_sums[row];
                }
            }
            inv_rms_s = rsqrtf(global_ss * inv_global_hidden + eps);
        }
        __syncthreads();

        const float inv_rms = inv_rms_s;

        for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
            const float xv = to_float<T>(x[base + c]);

            // Match the reference's ordering:
            // normalized FP32 -> cast to input dtype -> multiply by local_weight.
            const T norm_cast_t = from_float<T>(xv * inv_rms);
            const float y = to_float<T>(norm_cast_t) * to_float<T>(weight[c]);
            out[base + c] = from_float<T>(y);
        }

        __syncthreads();
    }
}

void rmsnorm_tp_symm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    torch::Tensor local_sums,
    torch::Tensor sum_ptrs,
    torch::Tensor signal_pad_ptrs,
    int64_t rows,
    int64_t cols,
    double variance_epsilon,
    int rank,
    int world_size,
    int num_blocks,
    int num_threads,
    int dtype_enum
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(local_sums.is_cuda(), "local_sums must be CUDA");
    TORCH_CHECK(sum_ptrs.is_cuda(), "sum_ptrs must be CUDA");
    TORCH_CHECK(signal_pad_ptrs.is_cuda(), "signal_pad_ptrs must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(local_sums.dtype() == torch::kFloat32, "local_sums must be float32");
    TORCH_CHECK(sum_ptrs.dtype() == torch::kInt64, "sum_ptrs must be int64");
    TORCH_CHECK(signal_pad_ptrs.dtype() == torch::kInt64, "signal_pad_ptrs must be int64");

    if (rows <= 0 || cols <= 0) {
        return;
    }

    const float inv_global_hidden = 1.0f / (float)(cols * (int64_t)world_size);
    const float eps = (float)variance_epsilon;

    const uint64_t* d_sum_ptrs =
        reinterpret_cast<const uint64_t*>(sum_ptrs.data_ptr<int64_t>());
    const uint64_t* d_signal_ptrs =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        rmsnorm_tp_symm_kernel<__nv_bfloat16><<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            local_sums.data_ptr<float>(),
            d_sum_ptrs,
            d_signal_ptrs,
            rows,
            cols,
            inv_global_hidden,
            eps,
            rank,
            world_size
        );
    } else if (dtype_enum == 1) {
        rmsnorm_tp_symm_kernel<float><<<num_blocks, num_threads, 0, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            out.data_ptr<float>(),
            local_sums.data_ptr<float>(),
            d_sum_ptrs,
            d_signal_ptrs,
            rows,
            cols,
            inv_global_hidden,
            eps,
            rank,
            world_size
        );
    } else if (dtype_enum == 2) {
        rmsnorm_tp_symm_kernel<half><<<num_blocks, num_threads, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            local_sums.data_ptr<float>(),
            d_sum_ptrs,
            d_signal_ptrs,
            rows,
            cols,
            inv_global_hidden,
            eps,
            rank,
            world_size
        );
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_tp_symm", &rmsnorm_tp_symm,
          "Tensor-parallel RMSNorm using symmetric memory UVA peer reads and GPU signal barriers");
}
'''


_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("rmsnorm_tp_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"unsupported dtype for custom RMSNorm: {dtype}")


def _threads_for_cols(cols: int) -> int:
    if cols <= 64:
        return 64
    if cols <= 128:
        return 128
    return 256


def _get_resources(shape, dtype, device, rows: int):
    world_size = dist.get_world_size()
    key = (tuple(shape), dtype, device.index, rows, world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    local_sums = symm_mem.empty((rows,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(local_sums, dist.group.WORLD)

    out = torch.empty(shape, device=device, dtype=dtype)
    sum_ptrs = torch.tensor([int(p) for p in hdl.buffer_ptrs], device=device, dtype=torch.int64)

    cached = (local_sums, hdl, out, sum_ptrs)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(local_hidden_states: torch.Tensor, local_weight: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """
    Multi-GPU tensor-parallel RMSNorm over a last-dimension hidden partition.

    BF16 path is the intended fast path.  Communication is implemented by:
      - writing one FP32 sum-of-squares scalar per row into symmetric memory,
      - synchronizing ranks inside the CUDA kernel through symmetric signal pads,
      - reading peer sums directly through UVA pointers,
      - normalizing/scaling locally in the same kernel.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert local_hidden_states.is_cuda, "local_hidden_states must be CUDA"
    assert local_weight.is_cuda, "local_weight must be CUDA"
    assert local_hidden_states.dim() >= 1, "local_hidden_states must have at least one dimension"

    dtype_enum = _dtype_enum(local_hidden_states.dtype)
    assert local_weight.dtype == local_hidden_states.dtype, "weight dtype must match hidden dtype"

    x = local_hidden_states if local_hidden_states.is_contiguous() else local_hidden_states.contiguous()
    w = local_weight if local_weight.is_contiguous() else local_weight.contiguous()

    cols = int(x.shape[-1])
    rows = int(x.numel() // cols) if cols > 0 else 0
    assert int(w.numel()) == cols, "local_weight must have shape (local_hidden_size,)"

    if x.numel() == 0:
        return torch.empty_like(x)

    local_sums, hdl, out, sum_ptrs = _get_resources(x.shape, x.dtype, x.device, rows)

    threads = _threads_for_cols(cols)
    # Persistent blocks reuse signal-pad slots while walking rows grid-stride.
    # Keep this bounded to avoid excessive signal-pad footprint and launch overhead.
    blocks = min(max(rows, 1), 128)

    _get_ext().rmsnorm_tp_symm(
        x,
        w,
        out,
        local_sums,
        sum_ptrs,
        hdl.signal_pad_ptrs_dev,
        rows,
        cols,
        float(variance_epsilon),
        int(dist.get_rank()),
        int(dist.get_world_size()),
        int(blocks),
        int(threads),
        int(dtype_enum),
    )

    return out.reshape_as(local_hidden_states)