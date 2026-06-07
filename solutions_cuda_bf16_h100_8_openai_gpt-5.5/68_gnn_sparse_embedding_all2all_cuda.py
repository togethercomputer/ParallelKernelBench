from typing import Optional, Tuple
import math

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

#define MAX_W 16

template <typename index_t>
__device__ __forceinline__ int owner_of(index_t x, int world_size) {
    long long v = (long long)x;
    int r = (int)(v % (long long)world_size);
    return r < 0 ? r + world_size : r;
}

template <typename index_t>
__global__ void count_tiles_kernel(
    const index_t* __restrict__ idx,
    long long* __restrict__ block_counts,   // [world_size, num_tiles]
    long long K,
    int world_size,
    int num_tiles
) {
    __shared__ long long counts[MAX_W];

    int tid = threadIdx.x;
    if (tid < world_size) counts[tid] = 0;
    __syncthreads();

    long long i = (long long)blockIdx.x * blockDim.x + tid;
    if (i < K) {
        int o = owner_of<index_t>(idx[i], world_size);
        atomicAdd((unsigned long long*)&counts[o], 1ULL);
    }
    __syncthreads();

    if (tid < world_size) {
        block_counts[(long long)tid * num_tiles + blockIdx.x] = counts[tid];
    }
}

__global__ void prefix_meta_kernel(
    const long long* __restrict__ block_counts,  // [world_size, num_tiles]
    long long* __restrict__ block_offsets,       // [world_size, num_tiles]
    long long* __restrict__ meta,                // counts[W], offsets[W+1]
    int world_size,
    int num_tiles
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    long long base = 0;
    for (int r = 0; r < world_size; ++r) {
        long long sum = 0;
        for (int t = 0; t < num_tiles; ++t) {
            sum += block_counts[(long long)r * num_tiles + t];
        }
        meta[r] = sum;
        meta[world_size + r] = base;

        long long running = base;
        for (int t = 0; t < num_tiles; ++t) {
            block_offsets[(long long)r * num_tiles + t] = running;
            running += block_counts[(long long)r * num_tiles + t];
        }
        base += sum;
    }
    meta[world_size + world_size] = base;
}

template <typename index_t>
__global__ void stable_positions_kernel(
    const index_t* __restrict__ idx,
    index_t* __restrict__ packed_idx,
    long long* __restrict__ pos,
    const long long* __restrict__ block_offsets, // [world_size, num_tiles]
    long long K,
    int world_size,
    int num_tiles
) {
    __shared__ int warp_counts[32 * MAX_W];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = (blockDim.x + 31) >> 5;
    int tile = blockIdx.x;
    long long i = (long long)tile * blockDim.x + tid;

    bool active = i < K;
    int owner = 0;
    if (active) owner = owner_of<index_t>(idx[i], world_size);

    if (lane < world_size) {
        unsigned mask = __ballot_sync(0xffffffff, active && owner == lane);
        warp_counts[warp * MAX_W + lane] = __popc(mask);
    }
    __syncthreads();

    if (!active) return;

    unsigned same_mask = __ballot_sync(0xffffffff, active && owner == owner);
    int local_rank = __popc(same_mask & ((1u << lane) - 1u));

    #pragma unroll
    for (int w = 0; w < 32; ++w) {
        if (w >= warp) break;
        local_rank += warp_counts[w * MAX_W + owner];
    }

    long long dst = block_offsets[(long long)owner * num_tiles + tile] + local_rank;
    packed_idx[dst] = idx[i];
    pos[i] = dst;
}

template <typename unit_t>
__global__ void pack_values_kernel(
    const unit_t* __restrict__ value,
    const long long* __restrict__ pos,
    unit_t* __restrict__ packed_value,
    long long K,
    long long feat_units
) {
    long long total = K * feat_units;
    long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; x < total; x += stride) {
        long long row = x / feat_units;
        long long col = x - row * feat_units;
        long long dst_row = pos[row];
        packed_value[dst_row * feat_units + col] = value[x];
    }
}

__global__ void gather_recv_meta_kernel(
    const long long* __restrict__ meta_ptrs,
    long long* __restrict__ recv_offsets, // [W+1]
    long long* __restrict__ src_offsets,  // [W]
    int rank,
    int world_size
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    recv_offsets[0] = 0;
    long long total = 0;
    for (int s = 0; s < world_size; ++s) {
        const long long* m = reinterpret_cast<const long long*>((uintptr_t)meta_ptrs[s]);
        long long cnt = m[rank];
        long long off = m[world_size + rank];
        src_offsets[s] = off;
        total += cnt;
        recv_offsets[s + 1] = total;
    }
}

template <typename index_t>
__global__ void pull_idx_segments_kernel(
    const long long* __restrict__ idx_ptrs,
    const long long* __restrict__ recv_offsets,
    const long long* __restrict__ src_offsets,
    index_t* __restrict__ recv_idx,
    int world_size
) {
    int src = blockIdx.y;
    if (src >= world_size) return;

    long long dst0 = recv_offsets[src];
    long long dst1 = recv_offsets[src + 1];
    long long n = dst1 - dst0;
    long long src0 = src_offsets[src];

    const index_t* remote = reinterpret_cast<const index_t*>((uintptr_t)idx_ptrs[src]);

    long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; x < n; x += stride) {
        recv_idx[dst0 + x] = remote[src0 + x];
    }
}

template <typename unit_t>
__global__ void pull_value_segments_kernel(
    const long long* __restrict__ val_ptrs,
    const long long* __restrict__ recv_offsets,
    const long long* __restrict__ src_offsets,
    unit_t* __restrict__ recv_value,
    long long feat_units,
    int world_size
) {
    int src = blockIdx.y;
    if (src >= world_size) return;

    long long dst0 = recv_offsets[src];
    long long dst1 = recv_offsets[src + 1];
    long long rows = dst1 - dst0;
    long long total = rows * feat_units;
    long long src0 = src_offsets[src];

    const unit_t* remote = reinterpret_cast<const unit_t*>((uintptr_t)val_ptrs[src]);

    long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; x < total; x += stride) {
        long long row = x / feat_units;
        long long col = x - row * feat_units;
        recv_value[(dst0 + row) * feat_units + col] =
            remote[(src0 + row) * feat_units + col];
    }
}

static inline int ceil_div_ll(long long a, int b) {
    return (int)((a + b - 1) / b);
}

void launch_bucketize_i64(
    torch::Tensor idx,
    torch::Tensor block_counts,
    torch::Tensor block_offsets,
    torch::Tensor meta,
    torch::Tensor packed_idx,
    torch::Tensor pos,
    long long K,
    int world_size,
    int tile
) {
    TORCH_CHECK(world_size <= MAX_W, "world_size > MAX_W");
    int num_tiles = std::max(1, ceil_div_ll(K, tile));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    count_tiles_kernel<long long><<<num_tiles, tile, 0, stream>>>(
        (const long long*)idx.data_ptr<int64_t>(),
        (long long*)block_counts.data_ptr<int64_t>(),
        K, world_size, num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    prefix_meta_kernel<<<1, 1, 0, stream>>>(
        (const long long*)block_counts.data_ptr<int64_t>(),
        (long long*)block_offsets.data_ptr<int64_t>(),
        (long long*)meta.data_ptr<int64_t>(),
        world_size, num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    stable_positions_kernel<long long><<<num_tiles, tile, 0, stream>>>(
        (const long long*)idx.data_ptr<int64_t>(),
        (long long*)packed_idx.data_ptr<int64_t>(),
        (long long*)pos.data_ptr<int64_t>(),
        (const long long*)block_offsets.data_ptr<int64_t>(),
        K, world_size, num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_bucketize_i32(
    torch::Tensor idx,
    torch::Tensor block_counts,
    torch::Tensor block_offsets,
    torch::Tensor meta,
    torch::Tensor packed_idx,
    torch::Tensor pos,
    long long K,
    int world_size,
    int tile
) {
    TORCH_CHECK(world_size <= MAX_W, "world_size > MAX_W");
    int num_tiles = std::max(1, ceil_div_ll(K, tile));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    count_tiles_kernel<int><<<num_tiles, tile, 0, stream>>>(
        (const int*)idx.data_ptr<int32_t>(),
        (long long*)block_counts.data_ptr<int64_t>(),
        K, world_size, num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    prefix_meta_kernel<<<1, 1, 0, stream>>>(
        (const long long*)block_counts.data_ptr<int64_t>(),
        (long long*)block_offsets.data_ptr<int64_t>(),
        (long long*)meta.data_ptr<int64_t>(),
        world_size, num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    stable_positions_kernel<int><<<num_tiles, tile, 0, stream>>>(
        (const int*)idx.data_ptr<int32_t>(),
        (int*)packed_idx.data_ptr<int32_t>(),
        (long long*)pos.data_ptr<int64_t>(),
        (const long long*)block_offsets.data_ptr<int64_t>(),
        K, world_size, num_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename unit_t>
void launch_pack_values_t(torch::Tensor value, torch::Tensor pos, torch::Tensor packed_value,
                          long long K, long long feat_units) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    long long total = K * feat_units;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)std::min<long long>(65535LL, (total + threads - 1) / threads);
    pack_values_kernel<unit_t><<<blocks, threads, 0, stream>>>(
        (const unit_t*)value.data_ptr(),
        (const long long*)pos.data_ptr<int64_t>(),
        (unit_t*)packed_value.data_ptr(),
        K, feat_units);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pack_values(torch::Tensor value, torch::Tensor pos, torch::Tensor packed_value,
                        long long K, long long feat_units, int unit_bytes) {
    if (unit_bytes == 1) {
        launch_pack_values_t<unsigned char>(value, pos, packed_value, K, feat_units);
    } else if (unit_bytes == 2) {
        launch_pack_values_t<unsigned short>(value, pos, packed_value, K, feat_units);
    } else if (unit_bytes == 4) {
        launch_pack_values_t<unsigned int>(value, pos, packed_value, K, feat_units);
    } else if (unit_bytes == 8) {
        launch_pack_values_t<unsigned long long>(value, pos, packed_value, K, feat_units);
    } else {
        TORCH_CHECK(false, "unsupported value element size");
    }
}

void launch_gather_recv_meta(torch::Tensor meta_ptrs, torch::Tensor recv_offsets,
                             torch::Tensor src_offsets, int rank, int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_recv_meta_kernel<<<1, 1, 0, stream>>>(
        (const long long*)meta_ptrs.data_ptr<int64_t>(),
        (long long*)recv_offsets.data_ptr<int64_t>(),
        (long long*)src_offsets.data_ptr<int64_t>(),
        rank, world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pull_idx_i64(torch::Tensor idx_ptrs, torch::Tensor recv_offsets,
                         torch::Tensor src_offsets, torch::Tensor recv_idx,
                         int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    dim3 grid(1024, world_size);
    pull_idx_segments_kernel<long long><<<grid, threads, 0, stream>>>(
        (const long long*)idx_ptrs.data_ptr<int64_t>(),
        (const long long*)recv_offsets.data_ptr<int64_t>(),
        (const long long*)src_offsets.data_ptr<int64_t>(),
        (long long*)recv_idx.data_ptr<int64_t>(),
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pull_idx_i32(torch::Tensor idx_ptrs, torch::Tensor recv_offsets,
                         torch::Tensor src_offsets, torch::Tensor recv_idx,
                         int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    dim3 grid(1024, world_size);
    pull_idx_segments_kernel<int><<<grid, threads, 0, stream>>>(
        (const long long*)idx_ptrs.data_ptr<int64_t>(),
        (const long long*)recv_offsets.data_ptr<int64_t>(),
        (const long long*)src_offsets.data_ptr<int64_t>(),
        (int*)recv_idx.data_ptr<int32_t>(),
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename unit_t>
void launch_pull_values_t(torch::Tensor val_ptrs, torch::Tensor recv_offsets,
                          torch::Tensor src_offsets, torch::Tensor recv_value,
                          long long feat_units, int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    dim3 grid(2048, world_size);
    pull_value_segments_kernel<unit_t><<<grid, threads, 0, stream>>>(
        (const long long*)val_ptrs.data_ptr<int64_t>(),
        (const long long*)recv_offsets.data_ptr<int64_t>(),
        (const long long*)src_offsets.data_ptr<int64_t>(),
        (unit_t*)recv_value.data_ptr(),
        feat_units,
        world_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pull_values(torch::Tensor val_ptrs, torch::Tensor recv_offsets,
                        torch::Tensor src_offsets, torch::Tensor recv_value,
                        long long feat_units, int unit_bytes, int world_size) {
    if (feat_units == 0) return;
    if (unit_bytes == 1) {
        launch_pull_values_t<unsigned char>(val_ptrs, recv_offsets, src_offsets,
                                            recv_value, feat_units, world_size);
    } else if (unit_bytes == 2) {
        launch_pull_values_t<unsigned short>(val_ptrs, recv_offsets, src_offsets,
                                             recv_value, feat_units, world_size);
    } else if (unit_bytes == 4) {
        launch_pull_values_t<unsigned int>(val_ptrs, recv_offsets, src_offsets,
                                           recv_value, feat_units, world_size);
    } else if (unit_bytes == 8) {
        launch_pull_values_t<unsigned long long>(val_ptrs, recv_offsets, src_offsets,
                                                 recv_value, feat_units, world_size);
    } else {
        TORCH_CHECK(false, "unsupported value element size");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_bucketize_i64", &launch_bucketize_i64, "stable remainder bucketize int64");
    m.def("launch_bucketize_i32", &launch_bucketize_i32, "stable remainder bucketize int32");
    m.def("launch_pack_values", &launch_pack_values, "pack values by stable positions");
    m.def("launch_gather_recv_meta", &launch_gather_recv_meta, "gather recv metadata via UVA");
    m.def("launch_pull_idx_i64", &launch_pull_idx_i64, "pull int64 idx via UVA");
    m.def("launch_pull_idx_i32", &launch_pull_idx_i32, "pull int32 idx via UVA");
    m.def("launch_pull_values", &launch_pull_values, "pull values via UVA");
}
'''


_ext = None
_state_cache = {}
TILE = 256


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_sparse_push_symm_uva_bf16_ext", CUDA_SRC)
    return _ext


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _unit_bytes(dtype: torch.dtype) -> int:
    # Payload is copied bitwise, so every fixed-width dtype with element_size in
    # {1,2,4,8} is supported. BF16 is the optimized/common path (2-byte units).
    return torch.empty((), dtype=dtype).element_size()


def _get_state(
    K: int,
    feat_numel: int,
    idx_dtype: torch.dtype,
    value_dtype: torch.dtype,
    device: torch.device,
    world_size: int,
    group,
):
    key = (
        int(K),
        int(feat_numel),
        idx_dtype,
        value_dtype,
        int(device.index if device.index is not None else torch.cuda.current_device()),
        int(world_size),
        id(group),
    )
    st = _state_cache.get(key)
    if st is not None:
        return st

    alloc_K = max(1, int(K))
    alloc_val = max(1, int(K) * int(feat_numel))
    num_tiles = max(1, (int(K) + TILE - 1) // TILE)

    meta = symm_mem.empty((2 * int(world_size) + 1,), dtype=torch.int64, device=device)
    meta_hdl = symm_mem.rendezvous(meta, group)

    packed_idx = symm_mem.empty((alloc_K,), dtype=idx_dtype, device=device)
    idx_hdl = symm_mem.rendezvous(packed_idx, group)

    packed_value = symm_mem.empty((alloc_val,), dtype=value_dtype, device=device)
    value_hdl = symm_mem.rendezvous(packed_value, group)

    block_counts = torch.empty((world_size * num_tiles,), dtype=torch.int64, device=device)
    block_offsets = torch.empty((world_size * num_tiles,), dtype=torch.int64, device=device)
    pos = torch.empty((alloc_K,), dtype=torch.int64, device=device)

    recv_offsets = torch.empty((world_size + 1,), dtype=torch.int64, device=device)
    src_offsets = torch.empty((world_size,), dtype=torch.int64, device=device)

    meta_ptrs = torch.tensor(meta_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    idx_ptrs = torch.tensor(idx_hdl.buffer_ptrs, dtype=torch.int64, device=device)
    value_ptrs = torch.tensor(value_hdl.buffer_ptrs, dtype=torch.int64, device=device)

    st = {
        "meta": meta,
        "meta_hdl": meta_hdl,
        "packed_idx": packed_idx,
        "idx_hdl": idx_hdl,
        "packed_value": packed_value,
        "value_hdl": value_hdl,
        "block_counts": block_counts,
        "block_offsets": block_offsets,
        "pos": pos,
        "recv_offsets": recv_offsets,
        "src_offsets": src_offsets,
        "meta_ptrs": meta_ptrs,
        "idx_ptrs": idx_ptrs,
        "value_ptrs": value_ptrs,
        "num_tiles": num_tiles,
    }
    _state_cache[key] = st
    return st


@torch.no_grad()
def solution(
    idx: torch.Tensor,
    value: torch.Tensor,
    num_nodes: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return idx, value

    assert idx.is_cuda and value.is_cuda, "custom sparse push expects CUDA tensors"
    assert idx.dtype in (torch.int64, torch.int32), "idx must be int64 or int32"
    assert world_size <= 16, "this H100 on-node implementation supports <=16 ranks"

    rank = dist.get_rank(group)
    K = int(idx.numel())
    feat_numel = _prod(value.shape[1:])
    unit = _unit_bytes(value.dtype)
    assert unit in (1, 2, 4, 8), "unsupported value dtype element size"

    idx_c = idx.contiguous()
    value_c = value.contiguous().reshape(-1)

    ext = _get_ext()
    st = _get_state(
        K,
        feat_numel,
        idx_c.dtype,
        value_c.dtype,
        idx_c.device,
        world_size,
        group,
    )

    if idx_c.dtype == torch.int64:
        ext.launch_bucketize_i64(
            idx_c,
            st["block_counts"],
            st["block_offsets"],
            st["meta"],
            st["packed_idx"],
            st["pos"],
            K,
            world_size,
            TILE,
        )
    else:
        ext.launch_bucketize_i32(
            idx_c,
            st["block_counts"],
            st["block_offsets"],
            st["meta"],
            st["packed_idx"],
            st["pos"],
            K,
            world_size,
            TILE,
        )

    ext.launch_pack_values(
        value_c,
        st["pos"],
        st["packed_value"],
        K,
        feat_numel,
        unit,
    )

    # Device-side metadata/payload publication; no NCCL all_to_all/all_reduce.
    st["meta_hdl"].barrier(channel=0)

    ext.launch_gather_recv_meta(
        st["meta_ptrs"],
        st["recv_offsets"],
        st["src_offsets"],
        rank,
        world_size,
    )

    # Only the tiny W+1 offset vector is materialized on host to allocate exact
    # output sizes; all payload transfer remains device-side UVA.
    recv_offsets_cpu = st["recv_offsets"].cpu()
    recv_count = int(recv_offsets_cpu[-1].item())

    recv_idx = torch.empty((recv_count,), dtype=idx.dtype, device=idx.device)
    recv_value = torch.empty(
        (recv_count, *tuple(value.shape[1:])),
        dtype=value.dtype,
        device=value.device,
    )

    if recv_count > 0:
        if idx.dtype == torch.int64:
            ext.launch_pull_idx_i64(
                st["idx_ptrs"],
                st["recv_offsets"],
                st["src_offsets"],
                recv_idx,
                world_size,
            )
        else:
            ext.launch_pull_idx_i32(
                st["idx_ptrs"],
                st["recv_offsets"],
                st["src_offsets"],
                recv_idx,
                world_size,
            )

        ext.launch_pull_values(
            st["value_ptrs"],
            st["recv_offsets"],
            st["src_offsets"],
            recv_value.reshape(-1),
            feat_numel,
            unit,
            world_size,
        )

    return recv_idx, recv_value