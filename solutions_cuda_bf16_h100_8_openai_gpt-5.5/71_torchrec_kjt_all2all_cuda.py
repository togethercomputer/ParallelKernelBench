# Device-side TorchRec KJT all-to-all using symmetric memory/UVA peer reads.
# Metadata is exchanged through symm_mem, payloads are staged once in symmetric
# buffers, and CUDA kernels pack + recat jagged segments directly from peer UVA
# pointers.  NCCL/torch.distributed collectives are intentionally avoided.

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <stdint.h>

template <typename T>
__global__ void pack_a2a_kernel(
    const long long* __restrict__ ptrs,
    const long long* __restrict__ in_offsets,
    const long long* __restrict__ out_offsets,
    T* __restrict__ out,
    int world,
    long long total
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int src = 0;
        #pragma unroll
        for (int s = 0; s < 16; ++s) {
            if (s + 1 >= world) break;
            if (idx >= out_offsets[s + 1]) src = s + 1;
        }
        long long local = idx - out_offsets[src];
        const T* base = reinterpret_cast<const T*>((uintptr_t)ptrs[src]);
        out[idx] = base[in_offsets[src] + local];
    }
}

template <typename T>
__global__ void permute_segments_kernel(
    const T* __restrict__ data,
    const long long* __restrict__ in_offsets,
    const int* __restrict__ recat,
    const long long* __restrict__ out_offsets,
    T* __restrict__ out,
    int nout_segments
) {
    int seg = blockIdx.x;
    if (seg >= nout_segments) return;
    int in_seg = recat[seg];
    long long in_start = in_offsets[in_seg];
    long long in_end = in_offsets[in_seg + 1];
    long long out_start = out_offsets[seg];
    long long len = in_end - in_start;
    for (long long j = threadIdx.x; j < len; j += blockDim.x) {
        out[out_start + j] = data[in_start + j];
    }
}

template <typename T>
__global__ void permute_fixed_width_kernel(
    const T* __restrict__ data,
    const int* __restrict__ recat,
    T* __restrict__ out,
    int width,
    int nrows
) {
    int row = blockIdx.x;
    if (row >= nrows) return;
    int in_row = recat[row];
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        out[(long long)row * width + j] = data[(long long)in_row * width + j];
    }
}

template <typename T>
__global__ void gather_data_kernel(
    const T* __restrict__ data,
    const int* __restrict__ recat,
    T* __restrict__ out,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        out[idx] = data[recat[idx]];
    }
}

template <typename LenT>
__global__ void key_sums_kernel(
    const LenT* __restrict__ lengths,
    const long long* __restrict__ offsets,
    long long* __restrict__ out,
    int nkeys
) {
    int key = blockIdx.x;
    if (key >= nkeys) return;
    long long start = offsets[key];
    long long end = offsets[key + 1];

    long long local = 0;
    for (long long i = start + threadIdx.x; i < end; i += blockDim.x) {
        local += (long long)lengths[i];
    }

    __shared__ long long smem[256];
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[key] = smem[0];
}

template <typename LenT>
__global__ void row_sums_kernel(
    const LenT* __restrict__ lengths,
    long long* __restrict__ out,
    int width,
    int nrows
) {
    int row = blockIdx.x;
    if (row >= nrows) return;
    long long base = (long long)row * width;

    long long local = 0;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        local += (long long)lengths[base + j];
    }

    __shared__ long long smem[256];
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = smem[0];
}

template <typename LenT>
__global__ void cast_len_i64_kernel(
    const LenT* __restrict__ x,
    long long* __restrict__ y,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) y[idx] = (long long)x[idx];
}

__global__ void gather_i64_kernel(
    const long long* __restrict__ x,
    const int* __restrict__ recat,
    long long* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += gridDim.x * blockDim.x) {
        y[idx] = x[recat[idx]];
    }
}

__global__ void fill_i64_kernel(long long* __restrict__ x, long long v, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) x[idx] = v;
}

template <typename LenT>
__global__ void sum_key_value_lengths_kernel(
    const LenT* __restrict__ lengths,
    const long long* __restrict__ stride_offsets,
    long long* __restrict__ out,
    int nkeys
) {
    int key = blockIdx.x;
    if (key >= nkeys) return;
    long long start = stride_offsets[key];
    long long end = stride_offsets[key + 1];

    long long local = 0;
    for (long long i = start + threadIdx.x; i < end; i += blockDim.x) {
        local += (long long)lengths[i];
    }

    __shared__ long long smem[256];
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[key] = smem[0];
}

__global__ void gather_full_meta_kernel(
    const long long* __restrict__ ptrs,
    long long* __restrict__ out,
    int elems_per_rank,
    int world
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = elems_per_rank * world;
    for (; idx < total; idx += gridDim.x * blockDim.x) {
        int src = idx / elems_per_rank;
        int off = idx - src * elems_per_rank;
        const long long* remote = reinterpret_cast<const long long*>((uintptr_t)ptrs[src]);
        out[idx] = remote[off];
    }
}

__global__ void build_stride_matrix_kernel(
    const long long* __restrict__ recv_strides,
    long long* __restrict__ out,
    int local_split,
    int world,
    int stagger
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = local_split * world;
    for (; idx < total; idx += gridDim.x * blockDim.x) {
        int f = idx / world;
        int col = idx - f * world;
        int groups = world / stagger;
        int rank_idx = (col % stagger) * groups + (col / stagger);
        out[idx] = recv_strides[rank_idx * local_split + f];
    }
}

static inline int blocks_for(long long n, int threads=256) {
    long long b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

void copy_tensor(torch::Tensor src, torch::Tensor dst, long long n) {
    if (n <= 0) return;
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "copy_tensor expects CUDA tensors");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t bytes = (size_t)n * src.element_size();
    cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), bytes, cudaMemcpyDeviceToDevice, stream);
    C10_CUDA_CHECK(cudaGetLastError());
}

void scan_offsets(torch::Tensor lengths, torch::Tensor offsets) {
    TORCH_CHECK(lengths.dtype() == torch::kInt64 && offsets.dtype() == torch::kInt64,
                "scan_offsets expects int64 tensors");
    int64_t n = lengths.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemsetAsync(offsets.data_ptr<long long>(), 0, sizeof(long long), stream));
    if (n > 0) {
        thrust::device_ptr<long long> in(lengths.data_ptr<long long>());
        thrust::device_ptr<long long> out(offsets.data_ptr<long long>() + 1);
        thrust::inclusive_scan(thrust::cuda::par.on(stream), in, in + n, out);
    }
    C10_CUDA_CHECK(cudaGetLastError());
}

void gather_full_meta(torch::Tensor ptrs, torch::Tensor out, int elems_per_rank, int world) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int total = elems_per_rank * world;
    gather_full_meta_kernel<<<blocks_for(total), 256, 0, stream>>>(
        ptrs.data_ptr<long long>(), out.data_ptr<long long>(), elems_per_rank, world);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_a2a(torch::Tensor ptrs, torch::Tensor in_offsets, torch::Tensor out_offsets,
              torch::Tensor out, long long total) {
    if (total <= 0) return;
    int world = ptrs.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = blocks_for(total, threads);

    size_t es = out.element_size();
    if (es == 8) {
        pack_a2a_kernel<unsigned long long><<<blocks, threads, 0, stream>>>(
            ptrs.data_ptr<long long>(), in_offsets.data_ptr<long long>(),
            out_offsets.data_ptr<long long>(), (unsigned long long*)out.data_ptr(), world, total);
    } else if (es == 4) {
        pack_a2a_kernel<unsigned int><<<blocks, threads, 0, stream>>>(
            ptrs.data_ptr<long long>(), in_offsets.data_ptr<long long>(),
            out_offsets.data_ptr<long long>(), (unsigned int*)out.data_ptr(), world, total);
    } else if (es == 2) {
        pack_a2a_kernel<unsigned short><<<blocks, threads, 0, stream>>>(
            ptrs.data_ptr<long long>(), in_offsets.data_ptr<long long>(),
            out_offsets.data_ptr<long long>(), (unsigned short*)out.data_ptr(), world, total);
    } else {
        pack_a2a_kernel<unsigned char><<<blocks, threads, 0, stream>>>(
            ptrs.data_ptr<long long>(), in_offsets.data_ptr<long long>(),
            out_offsets.data_ptr<long long>(), (unsigned char*)out.data_ptr(), world, total);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void permute_segments(torch::Tensor data, torch::Tensor in_offsets, torch::Tensor recat,
                      torch::Tensor out_offsets, torch::Tensor out, int nseg) {
    if (nseg <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    size_t es = data.element_size();
    if (es == 8) {
        permute_segments_kernel<unsigned long long><<<nseg, threads, 0, stream>>>(
            (const unsigned long long*)data.data_ptr(), in_offsets.data_ptr<long long>(),
            recat.data_ptr<int>(), out_offsets.data_ptr<long long>(),
            (unsigned long long*)out.data_ptr(), nseg);
    } else if (es == 4) {
        permute_segments_kernel<unsigned int><<<nseg, threads, 0, stream>>>(
            (const unsigned int*)data.data_ptr(), in_offsets.data_ptr<long long>(),
            recat.data_ptr<int>(), out_offsets.data_ptr<long long>(),
            (unsigned int*)out.data_ptr(), nseg);
    } else if (es == 2) {
        permute_segments_kernel<unsigned short><<<nseg, threads, 0, stream>>>(
            (const unsigned short*)data.data_ptr(), in_offsets.data_ptr<long long>(),
            recat.data_ptr<int>(), out_offsets.data_ptr<long long>(),
            (unsigned short*)out.data_ptr(), nseg);
    } else {
        permute_segments_kernel<unsigned char><<<nseg, threads, 0, stream>>>(
            (const unsigned char*)data.data_ptr(), in_offsets.data_ptr<long long>(),
            recat.data_ptr<int>(), out_offsets.data_ptr<long long>(),
            (unsigned char*)out.data_ptr(), nseg);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void permute_fixed_width(torch::Tensor data, torch::Tensor recat, torch::Tensor out,
                         int width, int nrows) {
    if (nrows <= 0 || width <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    size_t es = data.element_size();
    if (es == 8) {
        permute_fixed_width_kernel<unsigned long long><<<nrows, threads, 0, stream>>>(
            (const unsigned long long*)data.data_ptr(), recat.data_ptr<int>(),
            (unsigned long long*)out.data_ptr(), width, nrows);
    } else if (es == 4) {
        permute_fixed_width_kernel<unsigned int><<<nrows, threads, 0, stream>>>(
            (const unsigned int*)data.data_ptr(), recat.data_ptr<int>(),
            (unsigned int*)out.data_ptr(), width, nrows);
    } else if (es == 2) {
        permute_fixed_width_kernel<unsigned short><<<nrows, threads, 0, stream>>>(
            (const unsigned short*)data.data_ptr(), recat.data_ptr<int>(),
            (unsigned short*)out.data_ptr(), width, nrows);
    } else {
        permute_fixed_width_kernel<unsigned char><<<nrows, threads, 0, stream>>>(
            (const unsigned char*)data.data_ptr(), recat.data_ptr<int>(),
            (unsigned char*)out.data_ptr(), width, nrows);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_data(torch::Tensor data, torch::Tensor recat, torch::Tensor out, long long n) {
    if (n <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256, blocks = blocks_for(n, threads);
    size_t es = data.element_size();
    if (es == 8) {
        gather_data_kernel<unsigned long long><<<blocks, threads, 0, stream>>>(
            (const unsigned long long*)data.data_ptr(), recat.data_ptr<int>(),
            (unsigned long long*)out.data_ptr(), n);
    } else if (es == 4) {
        gather_data_kernel<unsigned int><<<blocks, threads, 0, stream>>>(
            (const unsigned int*)data.data_ptr(), recat.data_ptr<int>(),
            (unsigned int*)out.data_ptr(), n);
    } else if (es == 2) {
        gather_data_kernel<unsigned short><<<blocks, threads, 0, stream>>>(
            (const unsigned short*)data.data_ptr(), recat.data_ptr<int>(),
            (unsigned short*)out.data_ptr(), n);
    } else {
        gather_data_kernel<unsigned char><<<blocks, threads, 0, stream>>>(
            (const unsigned char*)data.data_ptr(), recat.data_ptr<int>(),
            (unsigned char*)out.data_ptr(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_key_sums(torch::Tensor lengths, torch::Tensor offsets, torch::Tensor out) {
    int nkeys = out.numel();
    if (nkeys <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (lengths.dtype() == torch::kInt64) {
        key_sums_kernel<long long><<<nkeys, 256, 0, stream>>>(
            lengths.data_ptr<long long>(), offsets.data_ptr<long long>(), out.data_ptr<long long>(), nkeys);
    } else {
        key_sums_kernel<int><<<nkeys, 256, 0, stream>>>(
            lengths.data_ptr<int>(), offsets.data_ptr<long long>(), out.data_ptr<long long>(), nkeys);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_row_sums(torch::Tensor lengths, torch::Tensor out, int width, int nrows) {
    if (nrows <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (lengths.dtype() == torch::kInt64) {
        row_sums_kernel<long long><<<nrows, 256, 0, stream>>>(
            lengths.data_ptr<long long>(), out.data_ptr<long long>(), width, nrows);
    } else {
        row_sums_kernel<int><<<nrows, 256, 0, stream>>>(
            lengths.data_ptr<int>(), out.data_ptr<long long>(), width, nrows);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cast_lengths_i64(torch::Tensor x, torch::Tensor y) {
    long long n = x.numel();
    if (n <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256, blocks = blocks_for(n, threads);
    if (x.dtype() == torch::kInt64) {
        cudaMemcpyAsync(y.data_ptr(), x.data_ptr(), (size_t)n * sizeof(long long),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        cast_len_i64_kernel<int><<<blocks, threads, 0, stream>>>(
            x.data_ptr<int>(), y.data_ptr<long long>(), n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_i64(torch::Tensor x, torch::Tensor recat, torch::Tensor y) {
    int n = y.numel();
    if (n <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_i64_kernel<<<blocks_for(n), 256, 0, stream>>>(
        x.data_ptr<long long>(), recat.data_ptr<int>(), y.data_ptr<long long>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fill_i64(torch::Tensor x, long long v) {
    long long n = x.numel();
    if (n <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fill_i64_kernel<<<blocks_for(n), 256, 0, stream>>>(x.data_ptr<long long>(), v, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_stride_matrix(torch::Tensor recv_strides, torch::Tensor out,
                         int local_split, int world, int stagger) {
    int total = local_split * world;
    if (total <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    build_stride_matrix_kernel<<<blocks_for(total), 256, 0, stream>>>(
        recv_strides.data_ptr<long long>(), out.data_ptr<long long>(),
        local_split, world, stagger);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_tensor", &copy_tensor, "D2D copy into symmetric buffer");
    m.def("scan_offsets", &scan_offsets, "int64 inclusive scan into exclusive offsets");
    m.def("gather_full_meta", &gather_full_meta, "gather all symmetric metadata");
    m.def("pack_a2a", &pack_a2a, "UVA peer-read all-to-all pack");
    m.def("permute_segments", &permute_segments, "segment recat permutation");
    m.def("permute_fixed_width", &permute_fixed_width, "fixed-width row permutation");
    m.def("gather_data", &gather_data, "dtype-preserving gather by recat");
    m.def("compute_key_sums", &compute_key_sums, "sum jagged lengths per key");
    m.def("compute_row_sums", &compute_row_sums, "sum lengths rows");
    m.def("cast_lengths_i64", &cast_lengths_i64, "cast int32/int64 lengths to int64");
    m.def("gather_i64", &gather_i64, "gather int64");
    m.def("fill_i64", &fill_i64, "fill int64 tensor");
    m.def("build_stride_matrix", &build_stride_matrix, "rank-major strides -> feature-major matrix");
}
'''

_ext = None
_meta_cache = {}
_payload_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("kjt_a2a_symm_uva_h100_ext", CUDA_SRC)
    return _ext


def _group_key(pg):
    return id(pg)


def _get_meta_state(world: int, device: torch.device, pg: dist.ProcessGroup):
    key = (_group_key(pg), world, device)
    cached = _meta_cache.get(key)
    if cached is not None:
        return cached
    buf = symm_mem.empty((4, world), dtype=torch.long, device=device)
    hdl = symm_mem.rendezvous(buf, pg)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.long, device=device)
    meta_all = torch.empty((world, 4, world), dtype=torch.long, device=device)
    cached = (buf, hdl, ptrs, meta_all)
    _meta_cache[key] = cached
    return cached


def _get_payload_state(
    cap: int,
    dtype: torch.dtype,
    device: torch.device,
    pg: dist.ProcessGroup,
    tag: str,
):
    cap_alloc = max(1, int(cap))
    key = (_group_key(pg), tag, cap_alloc, dtype, device)
    cached = _payload_cache.get(key)
    if cached is not None:
        return cached
    buf = symm_mem.empty((cap_alloc,), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, pg)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.long, device=device)
    cached = (buf, hdl, ptrs)
    _payload_cache[key] = cached
    return cached


def _prefix(vals: List[int]) -> List[int]:
    out = [0]
    s = 0
    for v in vals:
        s += int(v)
        out.append(s)
    return out


def _sum_by_splits(values: List[int], splits: List[int]) -> List[int]:
    out: List[int] = []
    off = 0
    for sp in splits:
        out.append(int(sum(values[off : off + sp])))
        off += sp
    return out


def _make_recat(
    local_split: int,
    world: int,
    stagger: int,
    device: torch.device,
    batch_size_per_rank: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    if local_split == 0:
        return None
    feature_order = [
        x + world // stagger * y
        for x in range(world // stagger)
        for y in range(stagger)
    ]
    if batch_size_per_rank is None:
        recat = [
            feature_idx + rank_idx * local_split
            for feature_idx in range(local_split)
            for rank_idx in feature_order
        ]
    else:
        rank_offsets = [0]
        for bs in batch_size_per_rank[:-1]:
            rank_offsets.append(rank_offsets[-1] + local_split * int(bs))
        recat = [
            rank_offsets[rank_idx] + feature_idx * int(batch_size_per_rank[rank_idx]) + b
            for feature_idx in range(local_split)
            for rank_idx in feature_order
            for b in range(int(batch_size_per_rank[rank_idx]))
        ]
    return torch.tensor(recat, dtype=torch.int32, device=device)


def _scan_offsets_i64(lengths_i64: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty((int(lengths_i64.numel()) + 1,), dtype=torch.long, device=lengths_i64.device)
    _get_ext().scan_offsets(lengths_i64.contiguous(), offsets)
    return offsets


def _lengths_to_i64(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.long and x.is_contiguous():
        return x
    y = torch.empty((int(x.numel()),), dtype=torch.long, device=x.device)
    _get_ext().cast_lengths_i64(x.contiguous(), y)
    return y


def _compute_length_per_key(
    lengths: torch.Tensor,
    stride_tensor: torch.Tensor,
    num_features: int,
) -> List[int]:
    if num_features == 0:
        return []
    stride_offsets = _scan_offsets_i64(stride_tensor)
    sums = torch.empty((num_features,), dtype=torch.long, device=lengths.device)
    _get_ext().compute_key_sums(lengths.contiguous(), stride_offsets, sums)
    return [int(x) for x in sums.cpu().tolist()]


def _a2a_pack_from_symm(
    ptrs: torch.Tensor,
    all_meta_cpu: List[List[List[int]]],
    row: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    world = len(all_meta_cpu)
    counts = [int(all_meta_cpu[src][row][rank]) for src in range(world)]
    in_offsets = [int(sum(all_meta_cpu[src][row][:rank])) for src in range(world)]
    out_offsets = _prefix(counts)
    total = out_offsets[-1]
    out = torch.empty((total,), dtype=dtype, device=device)
    if total > 0:
        in_offsets_t = torch.tensor(in_offsets, dtype=torch.long, device=device)
        out_offsets_t = torch.tensor(out_offsets, dtype=torch.long, device=device)
        _get_ext().pack_a2a(ptrs, in_offsets_t, out_offsets_t, out, total)
    return out


def _permute_segments_cuda(
    data: torch.Tensor,
    segment_lengths_i64: torch.Tensor,
    recat: Optional[torch.Tensor],
) -> torch.Tensor:
    if recat is None:
        return data
    nseg_out = int(recat.numel())
    out = torch.empty((int(data.numel()),), dtype=data.dtype, device=data.device)
    if nseg_out == 0:
        return out
    segment_lengths_i64 = segment_lengths_i64.contiguous()
    in_offsets = _scan_offsets_i64(segment_lengths_i64)
    out_lens = torch.empty((nseg_out,), dtype=torch.long, device=data.device)
    _get_ext().gather_i64(segment_lengths_i64, recat, out_lens)
    out_offsets = _scan_offsets_i64(out_lens)
    _get_ext().permute_segments(data.contiguous(), in_offsets, recat, out_offsets, out, nseg_out)
    return out


def _permute_fixed_width_cuda(
    data: torch.Tensor,
    recat: Optional[torch.Tensor],
    width: int,
) -> torch.Tensor:
    if recat is None or width <= 0:
        return data
    nrows = int(recat.numel())
    out = torch.empty_like(data)
    _get_ext().permute_fixed_width(data.contiguous(), recat, out, int(width), nrows)
    return out


def _gather_cuda(data: torch.Tensor, recat: torch.Tensor) -> torch.Tensor:
    out = torch.empty((int(recat.numel()),), dtype=data.dtype, device=data.device)
    _get_ext().gather_data(data.contiguous(), recat, out, int(recat.numel()))
    return out


@torch.no_grad()
def solution(
    lengths: torch.Tensor,
    values: torch.Tensor,
    key_splits: List[int],
    batch_size: int,
    pg: Optional[dist.ProcessGroup] = None,
    weights: Optional[torch.Tensor] = None,
    stride_per_key: Optional[List[int]] = None,
    stagger: int = 1,
) -> Dict[str, torch.Tensor]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert lengths.is_cuda and values.is_cuda
    assert lengths.is_contiguous() and values.is_contiguous()
    if weights is not None:
        assert weights.is_cuda and weights.is_contiguous()

    ext = _get_ext()
    pg = pg or dist.group.WORLD
    world = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    device = lengths.device

    num_features = int(sum(key_splits))
    variable_stride = stride_per_key is not None
    if stride_per_key is None:
        stride_list = [int(batch_size)] * num_features
    else:
        stride_list = [int(x) for x in stride_per_key]

    stride_tensor = torch.tensor(stride_list, dtype=torch.long, device=device)
    length_per_key = _compute_length_per_key(lengths, stride_tensor, num_features)

    length_splits = _sum_by_splits(stride_list, key_splits)
    value_splits = _sum_by_splits(length_per_key, key_splits)

    # Fixed metadata rows:
    # row 0: length splits, row 1: value splits,
    # row 2: variable-stride key_splits OR non-variable batch size per dest,
    # row 3: weight splits if present, else zeros.
    row2 = key_splits if variable_stride else [int(batch_size)] * world
    row3 = value_splits if weights is not None else [0] * world
    local_meta_flat: List[int] = []
    for row_vals in (length_splits, value_splits, row2, row3):
        local_meta_flat.extend([int(x) for x in row_vals])

    meta_buf, meta_hdl, meta_ptrs, meta_all_dev = _get_meta_state(world, device, pg)
    local_meta = torch.tensor(local_meta_flat, dtype=torch.long, device=device).view(4, world)
    ext.copy_tensor(local_meta, meta_buf, 4 * world)
    meta_hdl.barrier(channel=0)

    ext.gather_full_meta(meta_ptrs, meta_all_dev, 4 * world, world)
    all_meta_cpu = meta_all_dev.cpu().tolist()

    len_cap = max(int(sum(all_meta_cpu[src][0])) for src in range(world)) if world else 0
    val_cap = max(int(sum(all_meta_cpu[src][1])) for src in range(world)) if world else 0
    stride_cap = max(int(sum(all_meta_cpu[src][2])) for src in range(world)) if variable_stride else 1
    weight_cap = max(int(sum(all_meta_cpu[src][3])) for src in range(world)) if weights is not None else 1

    len_buf, len_hdl, len_ptrs = _get_payload_state(len_cap, lengths.dtype, device, pg, "lengths")
    val_buf, val_hdl, val_ptrs = _get_payload_state(val_cap, values.dtype, device, pg, "values")
    ext.copy_tensor(lengths, len_buf, int(lengths.numel()))
    ext.copy_tensor(values, val_buf, int(values.numel()))

    stride_buf = stride_hdl = stride_ptrs = None
    if variable_stride:
        stride_buf, stride_hdl, stride_ptrs = _get_payload_state(stride_cap, torch.long, device, pg, "strides")
        ext.copy_tensor(stride_tensor, stride_buf, int(stride_tensor.numel()))

    weight_buf = weight_hdl = weight_ptrs = None
    if weights is not None:
        weight_buf, weight_hdl, weight_ptrs = _get_payload_state(weight_cap, weights.dtype, device, pg, "weights")
        ext.copy_tensor(weights, weight_buf, int(weights.numel()))

    len_hdl.barrier(channel=1)
    val_hdl.barrier(channel=2)
    if variable_stride:
        stride_hdl.barrier(channel=3)
    if weights is not None:
        weight_hdl.barrier(channel=4)

    recv_lengths = _a2a_pack_from_symm(len_ptrs, all_meta_cpu, 0, rank, lengths.dtype, device)
    recv_values = _a2a_pack_from_symm(val_ptrs, all_meta_cpu, 1, rank, values.dtype, device)

    recv_strides: Optional[torch.Tensor] = None
    if variable_stride:
        recv_strides = _a2a_pack_from_symm(stride_ptrs, all_meta_cpu, 2, rank, torch.long, device)

    recv_weights: Optional[torch.Tensor] = None
    if weights is not None:
        recv_weights = _a2a_pack_from_symm(weight_ptrs, all_meta_cpu, 3, rank, weights.dtype, device)

    local_split = int(key_splits[rank])

    if variable_stride:
        assert recv_strides is not None
        recat = _make_recat(local_split, world, stagger, device)
        if recat is not None:
            stride_offsets = _scan_offsets_i64(recv_strides.contiguous())
            value_segment_lengths = torch.empty((int(recv_strides.numel()),), dtype=torch.long, device=device)
            ext.compute_key_sums(recv_lengths.contiguous(), stride_offsets, value_segment_lengths)

            recv_lengths = _permute_segments_cuda(recv_lengths, recv_strides, recat)
            recv_values = _permute_segments_cuda(recv_values, value_segment_lengths, recat)
            if recv_weights is not None:
                recv_weights = _permute_segments_cuda(recv_weights, value_segment_lengths, recat)

        stride_per_key_per_rank = torch.empty((local_split, world), dtype=torch.long, device=device)
        ext.build_stride_matrix(recv_strides.contiguous(), stride_per_key_per_rank, local_split, world, stagger)

        result: Dict[str, torch.Tensor] = {
            "lengths": recv_lengths,
            "values": recv_values,
            "stride_per_key_per_rank": stride_per_key_per_rank,
        }
    else:
        stride_per_rank = [int(all_meta_cpu[src][2][rank]) for src in range(world)]
        single_batch_per_rank = all(s == stride_per_rank[0] for s in stride_per_rank)
        if single_batch_per_rank:
            B = int(stride_per_rank[0])
            recat = _make_recat(local_split, world, stagger, device)
            if recat is not None and B > 0:
                nrows = int(recat.numel())
                row_lengths = torch.empty((nrows,), dtype=torch.long, device=device)
                ext.compute_row_sums(recv_lengths.contiguous(), row_lengths, B, nrows)

                recv_lengths = _permute_fixed_width_cuda(recv_lengths, recat, B)
                recv_values = _permute_segments_cuda(recv_values, row_lengths, recat)
                if recv_weights is not None:
                    recv_weights = _permute_segments_cuda(recv_weights, row_lengths, recat)
        else:
            recat = _make_recat(local_split, world, stagger, device, stride_per_rank)
            if recat is not None:
                seg_lengths = _lengths_to_i64(recv_lengths)
                recv_values = _permute_segments_cuda(recv_values, seg_lengths, recat)
                if recv_weights is not None:
                    recv_weights = _permute_segments_cuda(recv_weights, seg_lengths, recat)
                recv_lengths = _gather_cuda(recv_lengths, recat)

        result = {
            "lengths": recv_lengths,
            "values": recv_values,
            "stride": torch.tensor(sum(stride_per_rank), dtype=torch.long, device=device),
            "stride_per_rank": torch.tensor(stride_per_rank, dtype=torch.long, device=device),
        }

    if recv_weights is not None:
        result["weights"] = recv_weights
    return result