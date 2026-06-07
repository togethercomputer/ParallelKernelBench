from typing import List, Optional
import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <vector>

static inline int ceil_div_i64(int64_t a, int b) {
    return static_cast<int>((a + b - 1) / b);
}

void copy_tensor_bytes(torch::Tensor dst, torch::Tensor src, int64_t nbytes) {
    TORCH_CHECK(dst.is_cuda() && src.is_cuda(), "dst/src must be CUDA tensors");
    TORCH_CHECK(dst.device() == src.device(), "dst/src must be on same CUDA device");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (nbytes > 0) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            dst.data_ptr(),
            src.data_ptr(),
            static_cast<size_t>(nbytes),
            cudaMemcpyDeviceToDevice,
            stream));
    }
}

void fill_prefix_meta(
    torch::Tensor meta,
    std::vector<int64_t> counts_sent,
    std::vector<int64_t> counts_received
) {
    TORCH_CHECK(meta.is_cuda(), "meta must be CUDA");
    TORCH_CHECK(meta.scalar_type() == torch::kInt64, "meta must be int64");
    const int64_t world = static_cast<int64_t>(counts_sent.size());
    TORCH_CHECK(static_cast<int64_t>(counts_received.size()) == world,
                "counts_sent/counts_received length mismatch");
    TORCH_CHECK(meta.numel() >= 2 * (world + 1), "meta buffer too small");

    std::vector<int64_t> host(2 * (world + 1), 0);
    int64_t acc = 0;
    host[0] = 0;
    for (int64_t i = 0; i < world; ++i) {
        acc += counts_sent[i];
        host[i + 1] = acc;
    }

    acc = 0;
    const int64_t recv_base = world + 1;
    host[recv_base] = 0;
    for (int64_t i = 0; i < world; ++i) {
        acc += counts_received[i];
        host[recv_base + i + 1] = acc;
    }

    C10_CUDA_CHECK(cudaMemcpy(
        meta.data_ptr<int64_t>(),
        host.data(),
        host.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice));
}

__global__ void zero_bf16_kernel(__nv_bfloat16* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        out[i] = __float2bfloat16(0.0f);
    }
}

__global__ void zero_f16_kernel(__half* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        out[i] = __float2half(0.0f);
    }
}

__global__ void zero_f32_kernel(float* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        out[i] = 0.0f;
    }
}

__device__ __forceinline__ int find_chunk_from_recv_prefix(
    int64_t row,
    const int64_t* __restrict__ recv_prefix,
    int world
) {
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= world) break;
        if (row < recv_prefix[k + 1]) return k;
    }
    return world - 1;
}

__global__ void scatter_pull_bf16_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ meta_ptrs,
    const int64_t* __restrict__ seed_inverse_ids,
    __nv_bfloat16* __restrict__ grad_input,
    int64_t out_rows,
    int64_t feat,
    int world,
    int rank
) {
    const int64_t total = out_rows * feat;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    const int64_t* local_meta = reinterpret_cast<const int64_t*>(
        static_cast<uintptr_t>(meta_ptrs[rank]));
    const int64_t* local_recv_prefix = local_meta + (world + 1);

    for (; linear < total; linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = linear / feat;
        const int64_t h = linear - row * feat;

        const int k = find_chunk_from_recv_prefix(row, local_recv_prefix, world);
        const int64_t intra = row - local_recv_prefix[k];

        const int src_rank = (rank + k) % world;
        const int src_chunk = (rank - src_rank + world) % world;

        const int64_t* src_meta = reinterpret_cast<const int64_t*>(
            static_cast<uintptr_t>(meta_ptrs[src_rank]));
        const int64_t src_row = src_meta[src_chunk] + intra;

        const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(
            static_cast<uintptr_t>(data_ptrs[src_rank]));

        const int64_t dst_row = seed_inverse_ids[row];
        const __nv_bfloat16 v = src[src_row * feat + h];

#if __CUDA_ARCH__ >= 800
        atomicAdd(grad_input + dst_row * feat + h, v);
#else
        float fv = __bfloat162float(v);
        atomicAdd(reinterpret_cast<float*>(grad_input + dst_row * feat + h), fv);
#endif
    }
}

__global__ void scatter_pull_f16_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ meta_ptrs,
    const int64_t* __restrict__ seed_inverse_ids,
    __half* __restrict__ grad_input,
    int64_t out_rows,
    int64_t feat,
    int world,
    int rank
) {
    const int64_t total = out_rows * feat;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    const int64_t* local_meta = reinterpret_cast<const int64_t*>(
        static_cast<uintptr_t>(meta_ptrs[rank]));
    const int64_t* local_recv_prefix = local_meta + (world + 1);

    for (; linear < total; linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = linear / feat;
        const int64_t h = linear - row * feat;

        const int k = find_chunk_from_recv_prefix(row, local_recv_prefix, world);
        const int64_t intra = row - local_recv_prefix[k];

        const int src_rank = (rank + k) % world;
        const int src_chunk = (rank - src_rank + world) % world;

        const int64_t* src_meta = reinterpret_cast<const int64_t*>(
            static_cast<uintptr_t>(meta_ptrs[src_rank]));
        const int64_t src_row = src_meta[src_chunk] + intra;

        const __half* src = reinterpret_cast<const __half*>(
            static_cast<uintptr_t>(data_ptrs[src_rank]));

        const int64_t dst_row = seed_inverse_ids[row];
        atomicAdd(grad_input + dst_row * feat + h, src[src_row * feat + h]);
    }
}

__global__ void scatter_pull_f32_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ meta_ptrs,
    const int64_t* __restrict__ seed_inverse_ids,
    float* __restrict__ grad_input,
    int64_t out_rows,
    int64_t feat,
    int world,
    int rank
) {
    const int64_t total = out_rows * feat;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    const int64_t* local_meta = reinterpret_cast<const int64_t*>(
        static_cast<uintptr_t>(meta_ptrs[rank]));
    const int64_t* local_recv_prefix = local_meta + (world + 1);

    for (; linear < total; linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = linear / feat;
        const int64_t h = linear - row * feat;

        const int k = find_chunk_from_recv_prefix(row, local_recv_prefix, world);
        const int64_t intra = row - local_recv_prefix[k];

        const int src_rank = (rank + k) % world;
        const int src_chunk = (rank - src_rank + world) % world;

        const int64_t* src_meta = reinterpret_cast<const int64_t*>(
            static_cast<uintptr_t>(meta_ptrs[src_rank]));
        const int64_t src_row = src_meta[src_chunk] + intra;

        const float* src = reinterpret_cast<const float*>(
            static_cast<uintptr_t>(data_ptrs[src_rank]));

        const int64_t dst_row = seed_inverse_ids[row];
        atomicAdd(grad_input + dst_row * feat + h, src[src_row * feat + h]);
    }
}

__global__ void scatter_local_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int64_t* __restrict__ seed_inverse_ids,
    __nv_bfloat16* __restrict__ grad_input,
    int64_t rows,
    int64_t feat
) {
    const int64_t total = rows * feat;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear < total; linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = linear / feat;
        const int64_t h = linear - row * feat;
        const int64_t dst_row = seed_inverse_ids[row];
        atomicAdd(grad_input + dst_row * feat + h, src[row * feat + h]);
    }
}

__global__ void scatter_local_f16_kernel(
    const __half* __restrict__ src,
    const int64_t* __restrict__ seed_inverse_ids,
    __half* __restrict__ grad_input,
    int64_t rows,
    int64_t feat
) {
    const int64_t total = rows * feat;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear < total; linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = linear / feat;
        const int64_t h = linear - row * feat;
        const int64_t dst_row = seed_inverse_ids[row];
        atomicAdd(grad_input + dst_row * feat + h, src[row * feat + h]);
    }
}

__global__ void scatter_local_f32_kernel(
    const float* __restrict__ src,
    const int64_t* __restrict__ seed_inverse_ids,
    float* __restrict__ grad_input,
    int64_t rows,
    int64_t feat
) {
    const int64_t total = rows * feat;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; linear < total; linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t row = linear / feat;
        const int64_t h = linear - row * feat;
        const int64_t dst_row = seed_inverse_ids[row];
        atomicAdd(grad_input + dst_row * feat + h, src[row * feat + h]);
    }
}

void launch_scatter_pull(
    torch::Tensor data_ptrs,
    torch::Tensor meta_ptrs,
    torch::Tensor seed_inverse_ids,
    torch::Tensor grad_input,
    int64_t out_rows,
    int64_t feat,
    int world,
    int rank,
    int dtype_enum
) {
    TORCH_CHECK(data_ptrs.is_cuda() && meta_ptrs.is_cuda(), "ptr tensors must be CUDA");
    TORCH_CHECK(seed_inverse_ids.is_cuda() && grad_input.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(seed_inverse_ids.scalar_type() == torch::kInt64, "seed_inverse_ids must be int64");
    TORCH_CHECK(data_ptrs.scalar_type() == torch::kInt64, "data_ptrs must be int64");
    TORCH_CHECK(meta_ptrs.scalar_type() == torch::kInt64, "meta_ptrs must be int64");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int threads = 256;

    const int64_t zero_n = grad_input.numel();
    if (zero_n > 0) {
        int zero_blocks = std::min<int64_t>(65535, ceil_div_i64(zero_n, threads));
        if (dtype_enum == 0) {
            zero_bf16_kernel<<<zero_blocks, threads, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()), zero_n);
        } else if (dtype_enum == 1) {
            zero_f32_kernel<<<zero_blocks, threads, 0, stream>>>(
                grad_input.data_ptr<float>(), zero_n);
        } else if (dtype_enum == 2) {
            zero_f16_kernel<<<zero_blocks, threads, 0, stream>>>(
                reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()), zero_n);
        } else {
            TORCH_CHECK(false, "unsupported dtype_enum");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int64_t total = out_rows * feat;
    if (total <= 0) return;

    int blocks = std::min<int64_t>(65535, ceil_div_i64(total, threads));
    const long long* dptrs = reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>());
    const long long* mptrs = reinterpret_cast<const long long*>(meta_ptrs.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        scatter_pull_bf16_kernel<<<blocks, threads, 0, stream>>>(
            dptrs,
            mptrs,
            seed_inverse_ids.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
            out_rows,
            feat,
            world,
            rank);
    } else if (dtype_enum == 1) {
        scatter_pull_f32_kernel<<<blocks, threads, 0, stream>>>(
            dptrs,
            mptrs,
            seed_inverse_ids.data_ptr<int64_t>(),
            grad_input.data_ptr<float>(),
            out_rows,
            feat,
            world,
            rank);
    } else if (dtype_enum == 2) {
        scatter_pull_f16_kernel<<<blocks, threads, 0, stream>>>(
            dptrs,
            mptrs,
            seed_inverse_ids.data_ptr<int64_t>(),
            reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
            out_rows,
            feat,
            world,
            rank);
    } else {
        TORCH_CHECK(false, "unsupported dtype_enum");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_scatter_local(
    torch::Tensor src,
    torch::Tensor seed_inverse_ids,
    torch::Tensor grad_input,
    int64_t rows,
    int64_t feat,
    int dtype_enum
) {
    TORCH_CHECK(src.is_cuda() && seed_inverse_ids.is_cuda() && grad_input.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(seed_inverse_ids.scalar_type() == torch::kInt64, "seed_inverse_ids must be int64");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int threads = 256;

    const int64_t zero_n = grad_input.numel();
    if (zero_n > 0) {
        int zero_blocks = std::min<int64_t>(65535, ceil_div_i64(zero_n, threads));
        if (dtype_enum == 0) {
            zero_bf16_kernel<<<zero_blocks, threads, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()), zero_n);
        } else if (dtype_enum == 1) {
            zero_f32_kernel<<<zero_blocks, threads, 0, stream>>>(
                grad_input.data_ptr<float>(), zero_n);
        } else if (dtype_enum == 2) {
            zero_f16_kernel<<<zero_blocks, threads, 0, stream>>>(
                reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()), zero_n);
        } else {
            TORCH_CHECK(false, "unsupported dtype_enum");
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int64_t total = rows * feat;
    if (total <= 0) return;

    int blocks = std::min<int64_t>(65535, ceil_div_i64(total, threads));
    if (dtype_enum == 0) {
        scatter_local_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
            seed_inverse_ids.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
            rows,
            feat);
    } else if (dtype_enum == 1) {
        scatter_local_f32_kernel<<<blocks, threads, 0, stream>>>(
            src.data_ptr<float>(),
            seed_inverse_ids.data_ptr<int64_t>(),
            grad_input.data_ptr<float>(),
            rows,
            feat);
    } else if (dtype_enum == 2) {
        scatter_local_f16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
            seed_inverse_ids.data_ptr<int64_t>(),
            reinterpret_cast<__half*>(grad_input.data_ptr<at::Half>()),
            rows,
            feat);
    } else {
        TORCH_CHECK(false, "unsupported dtype_enum");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_tensor_bytes", &copy_tensor_bytes, "D2D copy into symmetric buffer");
    m.def("fill_prefix_meta", &fill_prefix_meta, "Fill symmetric prefix metadata");
    m.def("launch_scatter_pull", &launch_scatter_pull,
          "Fused UVA reverse all-to-all pull + scatter-add");
    m.def("launch_scatter_local", &launch_scatter_local,
          "Local scatter-add without distributed communication");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "gb_coop_backward_symm_uva_bf16_h100_ext",
            CUDA_SRC,
        )
    return _ext


_resource_cache = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"unsupported dtype for CUDA GraphBolt backward fast path: {dtype}")


def _feature_numel(t: torch.Tensor) -> int:
    if t.dim() <= 1:
        return 1
    return int(math.prod(tuple(t.shape[1:])))


def _get_resources(
    grad_shape,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    world: int,
):
    key = (tuple(grad_shape), dtype, device.index, id(group), world)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    data_buf = symm_mem.empty(tuple(grad_shape), device=device, dtype=dtype)
    data_hdl = symm_mem.rendezvous(data_buf, group)

    meta_buf = symm_mem.empty((2 * (world + 1),), device=device, dtype=torch.int64)
    meta_hdl = symm_mem.rendezvous(meta_buf, group)

    data_ptrs = torch.tensor(data_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    meta_ptrs = torch.tensor(meta_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "data_buf": data_buf,
        "data_hdl": data_hdl,
        "meta_buf": meta_buf,
        "meta_hdl": meta_hdl,
        "data_ptrs": data_ptrs,
        "meta_ptrs": meta_ptrs,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    grad_output: torch.Tensor,
    seed_inverse_ids: torch.Tensor,
    seed_size: int,
    counts_sent: List[int],
    counts_received: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    GraphBolt cooperative feature-exchange backward.

    Custom CUDA path:
      1. Place each rank's grad_output and prefix metadata in symmetric memory.
      2. Synchronize with symmetric-memory barrier.
      3. One fused CUDA kernel pulls the reverse all-to-all chunks directly
         through UVA peer pointers and atomically scatter-adds into grad_input.
    """
    ext = _get_ext()

    if not grad_output.is_cuda:
        raise RuntimeError("solution expects CUDA tensors")

    dtype_enum = _dtype_enum(grad_output.dtype)

    if not seed_inverse_ids.is_cuda:
        seed_inverse_ids = seed_inverse_ids.to(device=grad_output.device, non_blocking=True)
    if seed_inverse_ids.dtype != torch.int64:
        seed_inverse_ids = seed_inverse_ids.to(dtype=torch.int64)
    if not seed_inverse_ids.is_contiguous():
        seed_inverse_ids = seed_inverse_ids.contiguous()

    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    feat = _feature_numel(grad_output)
    out_rows = int(sum(counts_received))
    grad_input = torch.empty(
        (int(seed_size),) + tuple(grad_output.shape[1:]),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )

    if not dist.is_available() or not dist.is_initialized():
        rows = int(grad_output.shape[0]) if grad_output.dim() > 0 else 0
        ext.launch_scatter_local(
            grad_output.reshape(-1),
            seed_inverse_ids,
            grad_input.reshape(-1),
            rows,
            feat,
            dtype_enum,
        )
        return grad_input

    group = group or dist.group.WORLD
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world == 1:
        rows = int(grad_output.shape[0]) if grad_output.dim() > 0 else 0
        ext.launch_scatter_local(
            grad_output.reshape(-1),
            seed_inverse_ids,
            grad_input.reshape(-1),
            rows,
            feat,
            dtype_enum,
        )
        return grad_input

    if len(counts_sent) != world or len(counts_received) != world:
        raise RuntimeError("counts_sent/counts_received must have length equal to world size")

    res = _get_resources(
        tuple(grad_output.shape),
        grad_output.dtype,
        grad_output.device,
        group,
        world,
    )

    data_buf = res["data_buf"]
    meta_buf = res["meta_buf"]
    data_hdl = res["data_hdl"]
    meta_hdl = res["meta_hdl"]

    ext.copy_tensor_bytes(
        data_buf,
        grad_output,
        int(grad_output.numel() * grad_output.element_size()),
    )
    ext.fill_prefix_meta(
        meta_buf,
        [int(x) for x in counts_sent],
        [int(x) for x in counts_received],
    )

    # Ensures all symmetric data/meta writes are visible before peer UVA pulls.
    data_hdl.barrier(channel=0)
    meta_hdl.barrier(channel=1)

    ext.launch_scatter_pull(
        res["data_ptrs"],
        res["meta_ptrs"],
        seed_inverse_ids,
        grad_input.reshape(-1),
        out_rows,
        feat,
        world,
        rank,
        dtype_enum,
    )
    return grad_input