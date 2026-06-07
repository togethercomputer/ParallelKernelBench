from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ int64_t read_index(const void* p, int is_i32, int64_t off) {
    if (is_i32) {
        return (int64_t)(reinterpret_cast<const int32_t*>(p)[off]);
    }
    return reinterpret_cast<const int64_t*>(p)[off];
}

/*
  GraphBolt rotated order:
    local send chunk i is destined to rank (rank + i) % world_size.
    On that destination, this source rank's output chunk index is
    (rank - dest + world_size) % world_size.
*/

/* 16-byte vectorized copy path. Best for BF16 with H multiple of 8. */
__global__ void graphbolt_exchange_vec16_kernel(
    const char* __restrict__ local_features,
    const void* __restrict__ seed_ids,
    int seed_is_i32,
    const int64_t* __restrict__ recv_prefix,
    const int64_t* __restrict__ out_ptrs,
    const int64_t* __restrict__ meta_ptrs,
    int64_t row_bytes,
    int64_t vecs_per_row,
    int rank,
    int world_size
) {
    int chunk = blockIdx.y;
    int64_t row_start = recv_prefix[chunk];
    int64_t row_end = recv_prefix[chunk + 1];
    int64_t rows = row_end - row_start;
    if (rows <= 0) return;

    int dest = rank + chunk;
    if (dest >= world_size) dest -= world_size;

    int dst_rot_idx = rank - dest;
    if (dst_rot_idx < 0) dst_rot_idx += world_size;

    const int64_t* __restrict__ remote_prefix =
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(meta_ptrs[dest]));
    int64_t dst_row_start = remote_prefix[dst_rot_idx];

    char* __restrict__ remote_out =
        reinterpret_cast<char*>(static_cast<uintptr_t>(out_ptrs[dest]));

    int64_t work = rows * vecs_per_row;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < work; idx += stride) {
        int64_t local_row_in_chunk = idx / vecs_per_row;
        int64_t v = idx - local_row_in_chunk * vecs_per_row;

        int64_t src_row_pos = row_start + local_row_in_chunk;
        int64_t feature_row = read_index(seed_ids, seed_is_i32, src_row_pos);

        const uint4* __restrict__ src =
            reinterpret_cast<const uint4*>(
                local_features + feature_row * row_bytes + v * 16);

        uint4 val = *src;

        uint4* __restrict__ dst =
            reinterpret_cast<uint4*>(
                remote_out + (dst_row_start + local_row_in_chunk) * row_bytes + v * 16);

        *dst = val;
    }
}

/* Fully general byte-copy fallback for non-16B-aligned rows. */
__global__ void graphbolt_exchange_byte_kernel(
    const char* __restrict__ local_features,
    const void* __restrict__ seed_ids,
    int seed_is_i32,
    const int64_t* __restrict__ recv_prefix,
    const int64_t* __restrict__ out_ptrs,
    const int64_t* __restrict__ meta_ptrs,
    int64_t row_bytes,
    int rank,
    int world_size
) {
    int chunk = blockIdx.y;
    int64_t row_start = recv_prefix[chunk];
    int64_t row_end = recv_prefix[chunk + 1];
    int64_t rows = row_end - row_start;
    if (rows <= 0) return;

    int dest = rank + chunk;
    if (dest >= world_size) dest -= world_size;

    int dst_rot_idx = rank - dest;
    if (dst_rot_idx < 0) dst_rot_idx += world_size;

    const int64_t* __restrict__ remote_prefix =
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(meta_ptrs[dest]));
    int64_t dst_row_start = remote_prefix[dst_rot_idx];

    char* __restrict__ remote_out =
        reinterpret_cast<char*>(static_cast<uintptr_t>(out_ptrs[dest]));

    int64_t work = rows * row_bytes;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < work; idx += stride) {
        int64_t local_row_in_chunk = idx / row_bytes;
        int64_t byte_col = idx - local_row_in_chunk * row_bytes;

        int64_t src_row_pos = row_start + local_row_in_chunk;
        int64_t feature_row = read_index(seed_ids, seed_is_i32, src_row_pos);

        char v = local_features[feature_row * row_bytes + byte_col];
        remote_out[(dst_row_start + local_row_in_chunk) * row_bytes + byte_col] = v;
    }
}

void launch_graphbolt_exchange(
    torch::Tensor local_features,
    torch::Tensor seed_ids,
    torch::Tensor recv_prefix,
    torch::Tensor out_ptrs,
    torch::Tensor meta_ptrs,
    int64_t row_bytes,
    int64_t max_send_rows,
    int rank,
    int world_size,
    bool vectorized
) {
    TORCH_CHECK(local_features.is_cuda(), "local_features must be CUDA");
    TORCH_CHECK(seed_ids.is_cuda(), "seed_ids must be CUDA");
    TORCH_CHECK(recv_prefix.is_cuda(), "recv_prefix must be CUDA");
    TORCH_CHECK(out_ptrs.is_cuda(), "out_ptrs must be CUDA");
    TORCH_CHECK(meta_ptrs.is_cuda(), "meta_ptrs must be CUDA");
    TORCH_CHECK(local_features.is_contiguous(), "local_features must be contiguous");
    TORCH_CHECK(seed_ids.is_contiguous(), "seed_ids must be contiguous");
    TORCH_CHECK(recv_prefix.dtype() == torch::kInt64, "recv_prefix must be int64");
    TORCH_CHECK(out_ptrs.dtype() == torch::kInt64, "out_ptrs must be int64");
    TORCH_CHECK(meta_ptrs.dtype() == torch::kInt64, "meta_ptrs must be int64");
    TORCH_CHECK(seed_ids.dtype() == torch::kInt64 || seed_ids.dtype() == torch::kInt32,
                "seed_ids must be int64 or int32");

    if (max_send_rows <= 0 || row_bytes <= 0 || world_size <= 0) {
        return;
    }

    int seed_is_i32 = (seed_ids.dtype() == torch::kInt32) ? 1 : 0;

    const int threads = 256;
    int64_t units_per_row = vectorized ? (row_bytes / 16) : row_bytes;
    int64_t max_work = max_send_rows * units_per_row;
    if (max_work <= 0) return;

    int64_t bx64 = (max_work + threads - 1) / threads;
    if (bx64 > 65535) bx64 = 65535;
    if (bx64 < 1) bx64 = 1;

    dim3 grid((unsigned int)bx64, (unsigned int)world_size, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const char* lf = reinterpret_cast<const char*>(local_features.data_ptr());
    const void* ids = seed_ids.data_ptr();
    const int64_t* rp = recv_prefix.data_ptr<int64_t>();
    const int64_t* op = out_ptrs.data_ptr<int64_t>();
    const int64_t* mp = meta_ptrs.data_ptr<int64_t>();

    if (vectorized) {
        graphbolt_exchange_vec16_kernel<<<grid, threads, 0, stream>>>(
            lf, ids, seed_is_i32, rp, op, mp,
            row_bytes, row_bytes / 16, rank, world_size);
    } else {
        graphbolt_exchange_byte_kernel<<<grid, threads, 0, stream>>>(
            lf, ids, seed_is_i32, rp, op, mp,
            row_bytes, rank, world_size);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_graphbolt_exchange", &launch_graphbolt_exchange,
          "GraphBolt cooperative feature exchange using symmetric-memory UVA stores");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("graphbolt_symm_uva_exchange_bf16_ext", CUDA_SRC)
    return _ext


def _prefix(xs: List[int]) -> List[int]:
    out = [0]
    s = 0
    for v in xs:
        s += int(v)
        out.append(s)
    return out


@torch.no_grad()
def solution(
    local_features: torch.Tensor,
    seed_inverse_ids: torch.Tensor,
    counts_sent: List[int],
    counts_received: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD

    if not dist.is_initialized():
        idx = seed_inverse_ids.to(device=local_features.device, dtype=torch.long, non_blocking=True)
        return local_features.index_select(0, idx).reshape(
            (sum(counts_sent),) + tuple(local_features.shape[1:])
        )

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    assert len(counts_sent) == world_size
    assert len(counts_received) == world_size
    assert local_features.is_cuda

    device = local_features.device
    lf = local_features if local_features.is_contiguous() else local_features.contiguous()

    if not seed_inverse_ids.is_cuda:
        seed_ids = seed_inverse_ids.to(device=device, dtype=torch.long, non_blocking=True)
    else:
        seed_ids = seed_inverse_ids
        if seed_ids.dtype not in (torch.int64, torch.int32):
            seed_ids = seed_ids.to(dtype=torch.long)
    seed_ids = seed_ids.contiguous()

    out_rows = int(sum(counts_sent))
    send_rows = int(sum(counts_received))
    out_alloc_rows = max(out_rows, 1)

    trailing_shape = tuple(lf.shape[1:])
    out_alloc_shape = (out_alloc_rows,) + trailing_shape
    out_shape = (out_rows,) + trailing_shape

    # Symmetric output buffer: peers write directly into the correct rotated chunks.
    out_buf = symm_mem.empty(out_alloc_shape, device=device, dtype=lf.dtype)
    out_hdl = symm_mem.rendezvous(out_buf, group)

    # Symmetric metadata: each rank exposes prefix offsets for its output chunks.
    meta = symm_mem.empty((world_size + 1,), device=device, dtype=torch.int64)
    meta_hdl = symm_mem.rendezvous(meta, group)

    sent_prefix = _prefix(counts_sent)
    recv_prefix = _prefix(counts_received)

    meta.copy_(torch.tensor(sent_prefix, device=device, dtype=torch.int64))
    recv_prefix_t = torch.tensor(recv_prefix, device=device, dtype=torch.int64)

    out_ptrs_t = torch.tensor(out_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    meta_ptrs_t = torch.tensor(meta_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    # Make all per-rank offset tables visible before any peer reads them.
    meta_hdl.barrier(channel=0)

    if send_rows > 0:
        row_elems = 1
        for d in trailing_shape:
            row_elems *= int(d)
        row_bytes = row_elems * lf.element_size()

        # BF16 fast path when each row is a multiple of 16 bytes
        # (e.g. H multiple of 8 for bfloat16).
        vectorized = (row_bytes % 16 == 0)

        _get_ext().launch_graphbolt_exchange(
            lf,
            seed_ids,
            recv_prefix_t,
            out_ptrs_t,
            meta_ptrs_t,
            int(row_bytes),
            int(max(counts_received) if counts_received else 0),
            int(rank),
            int(world_size),
            bool(vectorized),
        )

    # Ensure all remote UVA stores into this rank's output buffer are complete.
    out_hdl.barrier(channel=1)

    return out_buf.narrow(0, 0, out_rows).reshape(out_shape)