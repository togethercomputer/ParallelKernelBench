from typing import Optional, List, Tuple, Dict, Any

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
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <stdint.h>
#include <math.h>

struct Bf16Less {
    __host__ __device__ bool operator()(const __nv_bfloat16& a, const __nv_bfloat16& b) const {
        return __bfloat162float(a) < __bfloat162float(b);
    }
};

struct HalfLess {
    __host__ __device__ bool operator()(const __half& a, const __half& b) const {
        return __half2float(a) < __half2float(b);
    }
};

template <typename T>
__global__ void copy_kernel(T* __restrict__ dst, const T* __restrict__ src, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void write_i64_kernel(long long* __restrict__ dst, int64_t slot, long long value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[slot] = value;
}

__global__ void gather_i64_slots_kernel(
    const long long* __restrict__ ptrs,
    long long* __restrict__ out,
    int world_size,
    int slots
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = world_size * slots;
    for (; idx < total; idx += gridDim.x * blockDim.x) {
        int r = idx / slots;
        int s = idx - r * slots;
        const long long* base = reinterpret_cast<const long long*>((uintptr_t)ptrs[r]);
        out[idx] = base[s];
    }
}

template <typename T>
__global__ void gather_values_kernel(
    const long long* __restrict__ ptrs,
    T* __restrict__ out,
    int world_size,
    int slots
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = world_size * slots;
    for (; idx < total; idx += gridDim.x * blockDim.x) {
        int r = idx / slots;
        int s = idx - r * slots;
        const T* base = reinterpret_cast<const T*>((uintptr_t)ptrs[r]);
        out[idx] = base[s];
    }
}

__global__ void write_samples_bf16_kernel(
    const __nv_bfloat16* __restrict__ sorted,
    __nv_bfloat16* __restrict__ sample_values,
    long long* __restrict__ sample_meta,
    int64_t local_n,
    int sort_rank,
    int n_samples
) {
    int i = threadIdx.x;
    if (i >= n_samples) return;

    if (sort_rank < 0 || local_n <= 0 || i >= local_n) {
        sample_values[i] = __float2bfloat16(INFINITY);
        sample_meta[i] = -1;
        sample_meta[n_samples + i] = -1;
        return;
    }

    int64_t valid_count = local_n < (int64_t)n_samples ? local_n : (int64_t)n_samples;
    if ((int64_t)i >= valid_count) {
        sample_values[i] = __float2bfloat16(INFINITY);
        sample_meta[i] = -1;
        sample_meta[n_samples + i] = -1;
        return;
    }

    int64_t pos;
    if ((int64_t)n_samples < local_n) {
        pos = (((int64_t)i + 1) * local_n) / (int64_t)n_samples - 1;
    } else {
        pos = i;
    }
    sample_values[i] = sorted[pos];
    sample_meta[i] = (long long)sort_rank;
    sample_meta[n_samples + i] = (long long)pos;
}

__global__ void write_samples_f32_kernel(
    const float* __restrict__ sorted,
    float* __restrict__ sample_values,
    long long* __restrict__ sample_meta,
    int64_t local_n,
    int sort_rank,
    int n_samples
) {
    int i = threadIdx.x;
    if (i >= n_samples) return;

    if (sort_rank < 0 || local_n <= 0 || i >= local_n) {
        sample_values[i] = INFINITY;
        sample_meta[i] = -1;
        sample_meta[n_samples + i] = -1;
        return;
    }

    int64_t valid_count = local_n < (int64_t)n_samples ? local_n : (int64_t)n_samples;
    if ((int64_t)i >= valid_count) {
        sample_values[i] = INFINITY;
        sample_meta[i] = -1;
        sample_meta[n_samples + i] = -1;
        return;
    }

    int64_t pos;
    if ((int64_t)n_samples < local_n) {
        pos = (((int64_t)i + 1) * local_n) / (int64_t)n_samples - 1;
    } else {
        pos = i;
    }
    sample_values[i] = sorted[pos];
    sample_meta[i] = (long long)sort_rank;
    sample_meta[n_samples + i] = (long long)pos;
}

__global__ void write_samples_f16_kernel(
    const __half* __restrict__ sorted,
    __half* __restrict__ sample_values,
    long long* __restrict__ sample_meta,
    int64_t local_n,
    int sort_rank,
    int n_samples
) {
    int i = threadIdx.x;
    if (i >= n_samples) return;

    if (sort_rank < 0 || local_n <= 0 || i >= local_n) {
        sample_values[i] = __float2half(INFINITY);
        sample_meta[i] = -1;
        sample_meta[n_samples + i] = -1;
        return;
    }

    int64_t valid_count = local_n < (int64_t)n_samples ? local_n : (int64_t)n_samples;
    if ((int64_t)i >= valid_count) {
        sample_values[i] = __float2half(INFINITY);
        sample_meta[i] = -1;
        sample_meta[n_samples + i] = -1;
        return;
    }

    int64_t pos;
    if ((int64_t)n_samples < local_n) {
        pos = (((int64_t)i + 1) * local_n) / (int64_t)n_samples - 1;
    } else {
        pos = i;
    }
    sample_values[i] = sorted[pos];
    sample_meta[i] = (long long)sort_rank;
    sample_meta[n_samples + i] = (long long)pos;
}

__device__ __forceinline__ float load_as_float_bf16(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}

__device__ __forceinline__ float load_as_float_f16(const __half* p, int64_t i) {
    return __half2float(p[i]);
}

__global__ void compute_boundaries_bf16_kernel(
    const __nv_bfloat16* __restrict__ sorted,
    int64_t local_n,
    const __nv_bfloat16* __restrict__ splitter_values,
    const long long* __restrict__ splitter_ranks,
    const long long* __restrict__ splitter_positions,
    int sort_rank,
    int split_count,
    long long* __restrict__ boundaries
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (sort_rank < 0) {
        for (int i = 0; i <= split_count + 1; ++i) boundaries[i] = 0;
        return;
    }

    long long prev = 0;
    boundaries[0] = 0;
    for (int s = 0; s < split_count; ++s) {
        float v = __bfloat162float(splitter_values[s]);
        long long end = 0;

        if (sort_rank > (int)splitter_ranks[s]) {
            int64_t lo = 0, hi = local_n;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (load_as_float_bf16(sorted, mid) < v) lo = mid + 1;
                else hi = mid;
            }
            end = (long long)lo;
        } else if (sort_rank < (int)splitter_ranks[s]) {
            int64_t lo = 0, hi = local_n;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (load_as_float_bf16(sorted, mid) <= v) lo = mid + 1;
                else hi = mid;
            }
            end = (long long)lo;
        } else {
            end = splitter_positions[s] + 1;
        }

        if (end < prev) end = prev;
        if (end > local_n) end = (long long)local_n;
        boundaries[s + 1] = end;
        prev = end;
    }
    boundaries[split_count + 1] = (long long)local_n;
}

__global__ void compute_boundaries_f32_kernel(
    const float* __restrict__ sorted,
    int64_t local_n,
    const float* __restrict__ splitter_values,
    const long long* __restrict__ splitter_ranks,
    const long long* __restrict__ splitter_positions,
    int sort_rank,
    int split_count,
    long long* __restrict__ boundaries
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (sort_rank < 0) {
        for (int i = 0; i <= split_count + 1; ++i) boundaries[i] = 0;
        return;
    }

    long long prev = 0;
    boundaries[0] = 0;
    for (int s = 0; s < split_count; ++s) {
        float v = splitter_values[s];
        long long end = 0;

        if (sort_rank > (int)splitter_ranks[s]) {
            int64_t lo = 0, hi = local_n;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (sorted[mid] < v) lo = mid + 1;
                else hi = mid;
            }
            end = (long long)lo;
        } else if (sort_rank < (int)splitter_ranks[s]) {
            int64_t lo = 0, hi = local_n;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (sorted[mid] <= v) lo = mid + 1;
                else hi = mid;
            }
            end = (long long)lo;
        } else {
            end = splitter_positions[s] + 1;
        }

        if (end < prev) end = prev;
        if (end > local_n) end = (long long)local_n;
        boundaries[s + 1] = end;
        prev = end;
    }
    boundaries[split_count + 1] = (long long)local_n;
}

__global__ void compute_boundaries_f16_kernel(
    const __half* __restrict__ sorted,
    int64_t local_n,
    const __half* __restrict__ splitter_values,
    const long long* __restrict__ splitter_ranks,
    const long long* __restrict__ splitter_positions,
    int sort_rank,
    int split_count,
    long long* __restrict__ boundaries
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (sort_rank < 0) {
        for (int i = 0; i <= split_count + 1; ++i) boundaries[i] = 0;
        return;
    }

    long long prev = 0;
    boundaries[0] = 0;
    for (int s = 0; s < split_count; ++s) {
        float v = __half2float(splitter_values[s]);
        long long end = 0;

        if (sort_rank > (int)splitter_ranks[s]) {
            int64_t lo = 0, hi = local_n;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (load_as_float_f16(sorted, mid) < v) lo = mid + 1;
                else hi = mid;
            }
            end = (long long)lo;
        } else if (sort_rank < (int)splitter_ranks[s]) {
            int64_t lo = 0, hi = local_n;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (load_as_float_f16(sorted, mid) <= v) lo = mid + 1;
                else hi = mid;
            }
            end = (long long)lo;
        } else {
            end = splitter_positions[s] + 1;
        }

        if (end < prev) end = prev;
        if (end > local_n) end = (long long)local_n;
        boundaries[s + 1] = end;
        prev = end;
    }
    boundaries[split_count + 1] = (long long)local_n;
}

template <typename T>
__global__ void gather_payload_kernel(
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ meta_matrix,
    const long long* __restrict__ recv_offsets,
    T* __restrict__ out,
    int my_rank,
    int world_size,
    int slots
) {
    int peer = blockIdx.x;
    if (peer >= world_size) return;

    long long count = meta_matrix[peer * slots + my_rank];
    long long src_off = meta_matrix[peer * slots + world_size + my_rank];
    long long dst_off = recv_offsets[peer];

    const T* src = reinterpret_cast<const T*>((uintptr_t)data_ptrs[peer]) + src_off;
    T* dst = out + dst_off;

    for (long long i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void sort_inplace(torch::Tensor t, int dtype_enum) {
    int64_t n = t.numel();
    if (n <= 1) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
        thrust::sort(thrust::cuda::par.on(stream), p, p + n, Bf16Less());
    } else if (dtype_enum == 1) {
        float* p = t.data_ptr<float>();
        thrust::sort(thrust::cuda::par.on(stream), p, p + n);
    } else {
        __half* p = reinterpret_cast<__half*>(t.data_ptr<at::Half>());
        thrust::sort(thrust::cuda::par.on(stream), p, p + n, HalfLess());
    }
    C10_CUDA_CHECK(cudaGetLastError());
}

void copy_tensor(torch::Tensor dst, torch::Tensor src, int64_t n, int dtype_enum) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        copy_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
            n);
    } else if (dtype_enum == 1) {
        copy_kernel<<<blocks, threads, 0, stream>>>(
            dst.data_ptr<float>(), src.data_ptr<float>(), n);
    } else {
        copy_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
            n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void write_i64(torch::Tensor dst, int64_t slot, int64_t value) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    write_i64_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<long long*>(dst.data_ptr<int64_t>()),
        slot,
        (long long)value);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_i64_slots(torch::Tensor ptrs, torch::Tensor out, int world_size, int slots) {
    int total = world_size * slots;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gather_i64_slots_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<long long*>(out.data_ptr<int64_t>()),
        world_size,
        slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_values(torch::Tensor ptrs, torch::Tensor out, int world_size, int slots, int dtype_enum) {
    int total = world_size * slots;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        gather_values_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            world_size,
            slots);
    } else if (dtype_enum == 1) {
        gather_values_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
            out.data_ptr<float>(),
            world_size,
            slots);
    } else {
        gather_values_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            world_size,
            slots);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void write_samples(
    torch::Tensor sorted,
    torch::Tensor sample_values,
    torch::Tensor sample_meta,
    int64_t local_n,
    int sort_rank,
    int n_samples,
    int dtype_enum
) {
    if (n_samples <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        write_samples_bf16_kernel<<<1, 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(sorted.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(sample_values.data_ptr<at::BFloat16>()),
            reinterpret_cast<long long*>(sample_meta.data_ptr<int64_t>()),
            local_n,
            sort_rank,
            n_samples);
    } else if (dtype_enum == 1) {
        write_samples_f32_kernel<<<1, 32, 0, stream>>>(
            sorted.data_ptr<float>(),
            sample_values.data_ptr<float>(),
            reinterpret_cast<long long*>(sample_meta.data_ptr<int64_t>()),
            local_n,
            sort_rank,
            n_samples);
    } else {
        write_samples_f16_kernel<<<1, 32, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(sample_values.data_ptr<at::Half>()),
            reinterpret_cast<long long*>(sample_meta.data_ptr<int64_t>()),
            local_n,
            sort_rank,
            n_samples);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_boundaries(
    torch::Tensor sorted,
    torch::Tensor splitter_values,
    torch::Tensor splitter_ranks,
    torch::Tensor splitter_positions,
    int64_t local_n,
    int sort_rank,
    int split_count,
    torch::Tensor boundaries,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        compute_boundaries_bf16_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(sorted.data_ptr<at::BFloat16>()),
            local_n,
            reinterpret_cast<const __nv_bfloat16*>(splitter_values.data_ptr<at::BFloat16>()),
            reinterpret_cast<const long long*>(splitter_ranks.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(splitter_positions.data_ptr<int64_t>()),
            sort_rank,
            split_count,
            reinterpret_cast<long long*>(boundaries.data_ptr<int64_t>()));
    } else if (dtype_enum == 1) {
        compute_boundaries_f32_kernel<<<1, 1, 0, stream>>>(
            sorted.data_ptr<float>(),
            local_n,
            splitter_values.data_ptr<float>(),
            reinterpret_cast<const long long*>(splitter_ranks.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(splitter_positions.data_ptr<int64_t>()),
            sort_rank,
            split_count,
            reinterpret_cast<long long*>(boundaries.data_ptr<int64_t>()));
    } else {
        compute_boundaries_f16_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<const __half*>(sorted.data_ptr<at::Half>()),
            local_n,
            reinterpret_cast<const __half*>(splitter_values.data_ptr<at::Half>()),
            reinterpret_cast<const long long*>(splitter_ranks.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(splitter_positions.data_ptr<int64_t>()),
            sort_rank,
            split_count,
            reinterpret_cast<long long*>(boundaries.data_ptr<int64_t>()));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_payload(
    torch::Tensor data_ptrs,
    torch::Tensor meta_matrix,
    torch::Tensor recv_offsets,
    torch::Tensor out,
    int my_rank,
    int world_size,
    int slots,
    int dtype_enum
) {
    if (world_size <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        gather_payload_kernel<<<world_size, 256, 0, stream>>>(
            reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(meta_matrix.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(recv_offsets.data_ptr<int64_t>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            my_rank,
            world_size,
            slots);
    } else if (dtype_enum == 1) {
        gather_payload_kernel<<<world_size, 256, 0, stream>>>(
            reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(meta_matrix.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(recv_offsets.data_ptr<int64_t>()),
            out.data_ptr<float>(),
            my_rank,
            world_size,
            slots);
    } else {
        gather_payload_kernel<<<world_size, 256, 0, stream>>>(
            reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(meta_matrix.data_ptr<int64_t>()),
            reinterpret_cast<const long long*>(recv_offsets.data_ptr<int64_t>()),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            my_rank,
            world_size,
            slots);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sort_inplace", &sort_inplace, "Thrust in-place sort");
    m.def("copy_tensor", &copy_tensor, "Typed contiguous copy");
    m.def("write_i64", &write_i64, "Write one int64 slot");
    m.def("gather_i64_slots", &gather_i64_slots, "Gather symmetric int64 slots via UVA");
    m.def("gather_values", &gather_values, "Gather symmetric sample values via UVA");
    m.def("write_samples", &write_samples, "Extract end-biased samples");
    m.def("compute_boundaries", &compute_boundaries, "Rank-aware splitter binary searches");
    m.def("gather_payload", &gather_payload, "Variable all-to-all payload gather via UVA");
}
'''


_ext = None
_symm_cache: Dict[Any, Any] = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("sample_sort_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError("optimized sample sort supports bfloat16, float16, and float32")


def _group_key(group: dist.ProcessGroup) -> int:
    return id(group)


def _symm_i64(name: str, n: int, device: torch.device, group: dist.ProcessGroup):
    key = ("i64", name, n, device.index, _group_key(group))
    cached = _symm_cache.get(key)
    if cached is not None:
        return cached
    buf = symm_mem.empty((n,), device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    cached = (buf, hdl, ptrs)
    _symm_cache[key] = cached
    return cached


def _symm_typed(name: str, n: int, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = ("typed", name, n, dtype, device.index, _group_key(group))
    cached = _symm_cache.get(key)
    if cached is not None:
        return cached
    buf = symm_mem.empty((n,), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    cached = (buf, hdl, ptrs)
    _symm_cache[key] = cached
    return cached


def _gather_sizes(meta, hdl, meta_ptrs, world_size: int, channel: int) -> List[int]:
    ext = _get_ext()
    hdl.barrier(channel=channel)
    out = torch.empty((world_size,), device=meta.device, dtype=torch.int64)
    ext.gather_i64_slots(meta_ptrs, out, world_size, 1)
    return [int(x) for x in out.cpu().tolist()]


def _active_rank_info(rank: int, sizes: List[int]) -> Tuple[List[int], int]:
    active = [idx for idx, size in enumerate(sizes) if size > 0]
    return active, (active.index(rank) if rank in active else -1)


def _target_range(rank: int, world_size: int, total: int) -> Tuple[int, int]:
    base = total // world_size
    extra = total % world_size
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end


def _write_meta_counts_offsets(meta: torch.Tensor, counts: List[int], offsets: List[int]) -> None:
    vals = counts + offsets
    tmp = torch.tensor(vals, device=meta.device, dtype=torch.int64)
    meta[: len(vals)].copy_(tmp)


def _gather_meta_matrix(meta_hdl, meta_ptrs, world_size: int, slots: int, device: torch.device, channel: int):
    meta_hdl.barrier(channel=channel)
    gathered = torch.empty((world_size, slots), device=device, dtype=torch.int64)
    _get_ext().gather_i64_slots(meta_ptrs, gathered.reshape(-1), world_size, slots)
    return gathered


def _uvar_alltoall_payload(
    data_ptrs: torch.Tensor,
    meta_matrix: torch.Tensor,
    rank: int,
    world_size: int,
    slots: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    counts = [int(x) for x in meta_matrix[:, rank].cpu().tolist()]
    recv_offsets: List[int] = []
    acc = 0
    for c in counts:
        recv_offsets.append(acc)
        acc += c

    out = torch.empty((acc,), device=device, dtype=dtype)
    recv_offsets_t = torch.tensor(recv_offsets, device=device, dtype=torch.int64)
    _get_ext().gather_payload(
        data_ptrs,
        meta_matrix,
        recv_offsets_t,
        out,
        rank,
        world_size,
        slots,
        _dtype_enum(dtype),
    )
    return out


@torch.no_grad()
def solution(local_shard: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    group = group or dist.group.WORLD
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert local_shard.is_cuda, "local_shard must be CUDA"
    assert local_shard.dim() == 1, "sample sort expects a one-dimensional shard"

    ext = _get_ext()
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    device = local_shard.device
    dtype = local_shard.dtype
    dtype_enum = _dtype_enum(dtype)

    meta_slots = max(2 * world_size, 16)
    meta, meta_hdl, meta_ptrs = _symm_i64("meta", meta_slots, device, group)

    local_flat = local_shard.contiguous()
    sorted_local = torch.empty((local_flat.numel(),), device=device, dtype=dtype)
    ext.copy_tensor(sorted_local, local_flat, local_flat.numel(), dtype_enum)
    ext.sort_inplace(sorted_local, dtype_enum)

    ext.write_i64(meta, 0, int(local_flat.numel()))
    initial_sizes = _gather_sizes(meta, meta_hdl, meta_ptrs, world_size, channel=0)

    active_ranks, sort_rank = _active_rank_info(rank, initial_sizes)
    active_count = len(active_ranks)
    if active_count == 0:
        return torch.empty((0,), device=device, dtype=dtype)

    max_initial = max(1, max(initial_sizes))
    data, data_hdl, data_ptrs = _symm_typed("payload_initial", max_initial, dtype, device, group)

    sample_values, sample_hdl, sample_ptrs = _symm_typed(
        "sample_values", max(world_size, 1), dtype, device, group
    )
    sample_meta, sample_meta_hdl, sample_meta_ptrs = _symm_i64(
        "sample_meta", max(2 * world_size, 2), device, group
    )

    ext.write_samples(
        sorted_local,
        sample_values,
        sample_meta,
        int(sorted_local.numel()),
        int(sort_rank),
        int(active_count),
        dtype_enum,
    )
    sample_hdl.barrier(channel=1)
    sample_meta_hdl.barrier(channel=1)

    gathered_values = torch.empty((world_size * active_count,), device=device, dtype=dtype)
    gathered_sample_meta = torch.empty((world_size, 2 * active_count), device=device, dtype=torch.int64)
    ext.gather_values(sample_ptrs, gathered_values, world_size, active_count, dtype_enum)
    ext.gather_i64_slots(sample_meta_ptrs, gathered_sample_meta.reshape(-1), world_size, 2 * active_count)

    values_cpu = [float(x) for x in gathered_values.cpu().tolist()]
    sm_cpu = gathered_sample_meta.cpu().tolist()

    samples = []
    for src in range(world_size):
        for j in range(active_count):
            sr = int(sm_cpu[src][j])
            pos = int(sm_cpu[src][active_count + j])
            if sr >= 0:
                samples.append((values_cpu[src * active_count + j], sr, pos))
    samples.sort(key=lambda item: (item[0], item[1], item[2]))

    splitters = []
    usable = len(samples)
    for sr in range(active_count - 1):
        idx = (sr + 1) * usable // active_count - 1
        idx = max(0, min(idx, usable - 1))
        splitters.append(samples[idx])

    split_count = active_count - 1
    if split_count > 0:
        split_vals = torch.tensor([x[0] for x in splitters], device=device, dtype=dtype)
        split_ranks = torch.tensor([x[1] for x in splitters], device=device, dtype=torch.int64)
        split_pos = torch.tensor([x[2] for x in splitters], device=device, dtype=torch.int64)
    else:
        split_vals = torch.empty((0,), device=device, dtype=dtype)
        split_ranks = torch.empty((0,), device=device, dtype=torch.int64)
        split_pos = torch.empty((0,), device=device, dtype=torch.int64)

    boundaries_t = torch.empty((active_count + 1,), device=device, dtype=torch.int64)
    ext.compute_boundaries(
        sorted_local,
        split_vals,
        split_ranks,
        split_pos,
        int(sorted_local.numel()),
        int(sort_rank),
        int(split_count),
        boundaries_t,
        dtype_enum,
    )
    boundaries = [int(x) for x in boundaries_t.cpu().tolist()]

    counts = [0] * world_size
    offsets = [0] * world_size
    for bucket, dest_rank in enumerate(active_ranks):
        st = boundaries[bucket]
        en = boundaries[bucket + 1]
        counts[dest_rank] = max(0, en - st)
        offsets[dest_rank] = st

    if sorted_local.numel() > 0:
        ext.copy_tensor(data, sorted_local, sorted_local.numel(), dtype_enum)
    _write_meta_counts_offsets(meta, counts, offsets)

    data_hdl.barrier(channel=2)
    meta_matrix = _gather_meta_matrix(meta_hdl, meta_ptrs, world_size, 2 * world_size, device, channel=2)

    received = _uvar_alltoall_payload(
        data_ptrs,
        meta_matrix,
        rank,
        world_size,
        2 * world_size,
        dtype,
        device,
    )
    ext.sort_inplace(received, dtype_enum)
    merged = received

    ext.write_i64(meta, 0, int(merged.numel()))
    merged_sizes = _gather_sizes(meta, meta_hdl, meta_ptrs, world_size, channel=3)
    total = sum(merged_sizes)

    max_merged = max(1, max(merged_sizes))
    final_data, final_data_hdl, final_data_ptrs = _symm_typed(
        "payload_final", max_merged, dtype, device, group
    )

    if merged.numel() > 0:
        ext.copy_tensor(final_data, merged, merged.numel(), dtype_enum)

    bucket_start = sum(merged_sizes[:rank])
    final_counts = [0] * world_size
    final_offsets = [0] * world_size
    for dest in range(world_size):
        target_start, target_end = _target_range(dest, world_size, total)
        st = max(bucket_start, target_start)
        en = min(bucket_start + int(merged.numel()), target_end)
        if st < en:
            final_counts[dest] = en - st
            final_offsets[dest] = st - bucket_start

    _write_meta_counts_offsets(meta, final_counts, final_offsets)

    final_data_hdl.barrier(channel=4)
    final_meta_matrix = _gather_meta_matrix(
        meta_hdl, meta_ptrs, world_size, 2 * world_size, device, channel=4
    )

    out = _uvar_alltoall_payload(
        final_data_ptrs,
        final_meta_matrix,
        rank,
        world_size,
        2 * world_size,
        dtype,
        device,
    )
    return out