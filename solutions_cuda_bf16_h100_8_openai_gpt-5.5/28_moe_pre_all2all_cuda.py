# Device-side MoE pre-all2all:
# - Build routing_map and stable expert-major permutation with CUDA + CUB DeviceSelect.
# - Stage permuted BF16 tokens in symmetric memory; publish split metadata in symmetric memory.
# - Replace NCCL all_to_all_single with UVA peer reads from remote symmetric buffers.
# - Fuse receive all-to-all layout conversion with final chunk reorder (source-major -> local-expert-major).

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// -----------------------------------------------------------------------------
// routing_map[e, t] = sum_k expert_mask[e, k, t]
// -----------------------------------------------------------------------------

template <typename mask_t>
__global__ void build_routing_kernel(
    const mask_t* __restrict__ mask,
    int64_t* __restrict__ routing,
    int E,
    int K,
    int T
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = (int64_t)E * (int64_t)T;

    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int e = (int)(idx / T);
        int t = (int)(idx - (int64_t)e * T);

        int64_t cnt = 0;
        int64_t base = ((int64_t)e * K) * T + t;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            mask_t v = mask[base + (int64_t)k * T];
            if (v != (mask_t)0) {
                cnt += 1;
            }
        }
        routing[idx] = cnt;
    }
}

// -----------------------------------------------------------------------------
// mapping initially contains selected flat indices from CUB in stable row-major
// order. Convert in-place to token indices and copy hidden[token, :] to send.
// Vectorized BF16/FP16/FP32 row copy when row_bytes is 16-byte aligned.
// -----------------------------------------------------------------------------

__global__ void convert_mapping_copy_vec16_kernel(
    const char* __restrict__ hidden,
    int64_t* __restrict__ mapping,
    char* __restrict__ sendbuf,
    int64_t rows,
    int64_t T,
    int64_t row_bytes,
    int64_t vecs_per_row
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * vecs_per_row;

    const uint4* __restrict__ hidden_v = reinterpret_cast<const uint4*>(hidden);
    uint4* __restrict__ send_v = reinterpret_cast<uint4*>(sendbuf);

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t row = idx / vecs_per_row;
        int64_t v = idx - row * vecs_per_row;

        int64_t flat_or_tok = mapping[row];
        int64_t tok = flat_or_tok % T;
        if (v == 0) {
            mapping[row] = tok;
        }

        const uint4* src_row = reinterpret_cast<const uint4*>(hidden + tok * row_bytes);
        uint4* dst_row = reinterpret_cast<uint4*>(sendbuf + row * row_bytes);
        dst_row[v] = src_row[v];
    }
}

template <typename scalar_t>
__global__ void convert_mapping_copy_scalar_kernel(
    const scalar_t* __restrict__ hidden,
    int64_t* __restrict__ mapping,
    scalar_t* __restrict__ sendbuf,
    int64_t rows,
    int64_t T,
    int64_t H
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * H;

    for (; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t row = idx / H;
        int64_t h = idx - row * H;

        int64_t flat_or_tok = mapping[row];
        int64_t tok = flat_or_tok % T;
        if (h == 0) {
            mapping[row] = tok;
        }
        sendbuf[row * H + h] = hidden[tok * H + h];
    }
}

// -----------------------------------------------------------------------------
// Receive metadata for fused all-to-all + final sort.
// num_global is [world_size, num_local_experts] in source-major order.
// Final output order is local_expert-major, source-minor.
// Remote send buffer order is global expert-major, hence for this rank:
// remote row = prefix(remote input_splits before rank) + prefix(local experts).
// -----------------------------------------------------------------------------

__global__ void build_recv_meta_kernel(
    const int64_t* __restrict__ meta_ptrs,
    const int64_t* __restrict__ num_global,
    int64_t* __restrict__ remote_row_offsets,
    int64_t* __restrict__ dst_row_offsets,
    int64_t* __restrict__ chunk_counts,
    int rank,
    int world_size,
    int num_local_experts
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int chunks = world_size * num_local_experts;
    if (c >= chunks) {
        return;
    }

    int src = c / num_local_experts;
    int le = c - src * num_local_experts;

    int64_t count = num_global[(int64_t)src * num_local_experts + le];

    const int64_t* remote_splits =
        reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(meta_ptrs[src]));

    int64_t remote_base = 0;
    for (int d = 0; d < rank; ++d) {
        remote_base += remote_splits[d];
    }

    int64_t remote_in_chunk = 0;
    for (int j = 0; j < le; ++j) {
        remote_in_chunk += num_global[(int64_t)src * num_local_experts + j];
    }

    int64_t dst = 0;
    for (int j = 0; j < le; ++j) {
        for (int s = 0; s < world_size; ++s) {
            dst += num_global[(int64_t)s * num_local_experts + j];
        }
    }
    for (int s = 0; s < src; ++s) {
        dst += num_global[(int64_t)s * num_local_experts + le];
    }

    remote_row_offsets[c] = remote_base + remote_in_chunk;
    dst_row_offsets[c] = dst;
    chunk_counts[c] = count;
}

// -----------------------------------------------------------------------------
// Fused all-to-all receive via UVA peer reads + chunk reorder.
// -----------------------------------------------------------------------------

__global__ void recv_sort_vec16_kernel(
    const int64_t* __restrict__ send_ptrs,
    const int64_t* __restrict__ remote_row_offsets,
    const int64_t* __restrict__ dst_row_offsets,
    const int64_t* __restrict__ chunk_counts,
    char* __restrict__ out,
    int world_size,
    int num_local_experts,
    int64_t row_bytes,
    int64_t vecs_per_row
) {
    int c = blockIdx.x;
    int lane_block = blockIdx.y;
    int chunks = world_size * num_local_experts;
    if (c >= chunks) {
        return;
    }

    int src_rank = c / num_local_experts;
    int64_t count = chunk_counts[c];
    if (count <= 0) {
        return;
    }

    const char* remote_base =
        reinterpret_cast<const char*>(static_cast<uintptr_t>(send_ptrs[src_rank]));

    int64_t remote_row = remote_row_offsets[c];
    int64_t dst_row = dst_row_offsets[c];
    int64_t total_vec = count * vecs_per_row;

    int64_t idx = (int64_t)lane_block * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.y * blockDim.x;

    for (; idx < total_vec; idx += stride) {
        int64_t r = idx / vecs_per_row;
        int64_t v = idx - r * vecs_per_row;

        const uint4* src =
            reinterpret_cast<const uint4*>(remote_base + (remote_row + r) * row_bytes);
        uint4* dst =
            reinterpret_cast<uint4*>(out + (dst_row + r) * row_bytes);
        dst[v] = src[v];
    }
}

template <typename scalar_t>
__global__ void recv_sort_scalar_kernel(
    const int64_t* __restrict__ send_ptrs,
    const int64_t* __restrict__ remote_row_offsets,
    const int64_t* __restrict__ dst_row_offsets,
    const int64_t* __restrict__ chunk_counts,
    scalar_t* __restrict__ out,
    int world_size,
    int num_local_experts,
    int64_t H
) {
    int c = blockIdx.x;
    int lane_block = blockIdx.y;
    int chunks = world_size * num_local_experts;
    if (c >= chunks) {
        return;
    }

    int src_rank = c / num_local_experts;
    int64_t count = chunk_counts[c];
    if (count <= 0) {
        return;
    }

    const scalar_t* remote_base =
        reinterpret_cast<const scalar_t*>(static_cast<uintptr_t>(send_ptrs[src_rank]));

    int64_t remote_row = remote_row_offsets[c];
    int64_t dst_row = dst_row_offsets[c];
    int64_t total = count * H;

    int64_t idx = (int64_t)lane_block * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.y * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t r = idx / H;
        int64_t h = idx - r * H;
        out[(dst_row + r) * H + h] = remote_base[(remote_row + r) * H + h];
    }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

int64_t cub_select_temp_bytes(
    torch::Tensor routing_map,
    torch::Tensor mapping,
    torch::Tensor selected_count
) {
    CHECK_CUDA(routing_map);
    CHECK_CUDA(mapping);
    CHECK_CUDA(selected_count);
    CHECK_CONTIG(routing_map);
    CHECK_CONTIG(mapping);
    CHECK_CONTIG(selected_count);

    TORCH_CHECK(routing_map.dtype() == torch::kInt64, "routing_map must be int64");
    TORCH_CHECK(mapping.dtype() == torch::kInt64, "mapping must be int64");
    TORCH_CHECK(selected_count.dtype() == torch::kInt64, "selected_count must be int64");

    using CountingIt = cub::CountingInputIterator<int64_t>;

    void* temp_storage = nullptr;
    size_t temp_bytes = 0;
    int64_t n = routing_map.numel();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cub::DeviceSelect::Flagged(
        temp_storage,
        temp_bytes,
        CountingIt(0),
        routing_map.data_ptr<int64_t>(),
        mapping.data_ptr<int64_t>(),
        selected_count.data_ptr<int64_t>(),
        n,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return (int64_t)temp_bytes;
}

void prepare_moe_send(
    torch::Tensor hidden,
    torch::Tensor expert_mask,
    torch::Tensor sendbuf,
    torch::Tensor routing_map,
    torch::Tensor mapping,
    torch::Tensor selected_count,
    torch::Tensor cub_temp,
    int64_t expected_rows
) {
    CHECK_CUDA(hidden);
    CHECK_CUDA(expert_mask);
    CHECK_CUDA(sendbuf);
    CHECK_CUDA(routing_map);
    CHECK_CUDA(mapping);
    CHECK_CUDA(selected_count);
    CHECK_CUDA(cub_temp);
    CHECK_CONTIG(hidden);
    CHECK_CONTIG(expert_mask);
    CHECK_CONTIG(sendbuf);
    CHECK_CONTIG(routing_map);
    CHECK_CONTIG(mapping);
    CHECK_CONTIG(selected_count);
    CHECK_CONTIG(cub_temp);

    TORCH_CHECK(hidden.dim() == 2, "hidden must be [T, H]");
    TORCH_CHECK(expert_mask.dim() == 3, "expert_mask must be [E, K, T]");
    TORCH_CHECK(routing_map.dtype() == torch::kInt64, "routing_map must be int64");
    TORCH_CHECK(mapping.dtype() == torch::kInt64, "mapping must be int64");
    TORCH_CHECK(selected_count.dtype() == torch::kInt64, "selected_count must be int64");

    int E = (int)expert_mask.size(0);
    int K = (int)expert_mask.size(1);
    int T = (int)expert_mask.size(2);
    int64_t H = hidden.size(1);

    TORCH_CHECK(hidden.size(0) == T, "hidden token count mismatch");
    TORCH_CHECK(routing_map.size(0) == E && routing_map.size(1) == T,
                "routing_map shape mismatch");
    TORCH_CHECK(mapping.numel() >= expected_rows, "mapping too small");
    TORCH_CHECK(sendbuf.size(0) >= std::max<int64_t>(expected_rows, 1),
                "sendbuf too small");
    TORCH_CHECK(sendbuf.size(1) == H, "sendbuf hidden dim mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int64_t nroute = (int64_t)E * (int64_t)T;
    int blocks = (int)((nroute + threads - 1) / threads);
    blocks = std::min(blocks, 65535);

    auto mt = expert_mask.scalar_type();
    if (mt == torch::kBool) {
        build_routing_kernel<bool><<<blocks, threads, 0, stream>>>(
            expert_mask.data_ptr<bool>(),
            routing_map.data_ptr<int64_t>(),
            E, K, T);
    } else if (mt == torch::kUInt8) {
        build_routing_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
            expert_mask.data_ptr<uint8_t>(),
            routing_map.data_ptr<int64_t>(),
            E, K, T);
    } else if (mt == torch::kInt8) {
        build_routing_kernel<int8_t><<<blocks, threads, 0, stream>>>(
            expert_mask.data_ptr<int8_t>(),
            routing_map.data_ptr<int64_t>(),
            E, K, T);
    } else if (mt == torch::kInt16) {
        build_routing_kernel<int16_t><<<blocks, threads, 0, stream>>>(
            expert_mask.data_ptr<int16_t>(),
            routing_map.data_ptr<int64_t>(),
            E, K, T);
    } else if (mt == torch::kInt32) {
        build_routing_kernel<int32_t><<<blocks, threads, 0, stream>>>(
            expert_mask.data_ptr<int32_t>(),
            routing_map.data_ptr<int64_t>(),
            E, K, T);
    } else if (mt == torch::kInt64) {
        build_routing_kernel<int64_t><<<blocks, threads, 0, stream>>>(
            expert_mask.data_ptr<int64_t>(),
            routing_map.data_ptr<int64_t>(),
            E, K, T);
    } else if (mt == torch::kFloat32) {
        build_routing_kernel<float><<<blocks, threads, 0, stream>>>(
            expert_mask.data_ptr<float>(),
            routing_map.data_ptr<int64_t>(),
            E, K, T);
    } else {
        TORCH_CHECK(false, "unsupported expert_mask dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    using CountingIt = cub::CountingInputIterator<int64_t>;
    cub::DeviceSelect::Flagged(
        cub_temp.data_ptr(),
        (size_t)cub_temp.numel(),
        CountingIt(0),
        routing_map.data_ptr<int64_t>(),
        mapping.data_ptr<int64_t>(),
        selected_count.data_ptr<int64_t>(),
        nroute,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (expected_rows <= 0) {
        return;
    }

    int64_t row_bytes = H * hidden.element_size();
    if ((row_bytes % 16) == 0) {
        int64_t vecs = row_bytes / 16;
        int64_t total = expected_rows * vecs;
        int cblocks = (int)((total + threads - 1) / threads);
        cblocks = std::min(cblocks, 65535);
        convert_mapping_copy_vec16_kernel<<<cblocks, threads, 0, stream>>>(
            reinterpret_cast<const char*>(hidden.data_ptr()),
            mapping.data_ptr<int64_t>(),
            reinterpret_cast<char*>(sendbuf.data_ptr()),
            expected_rows,
            T,
            row_bytes,
            vecs
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    int64_t total = expected_rows * H;
    int cblocks = (int)((total + threads - 1) / threads);
    cblocks = std::min(cblocks, 65535);

    if (hidden.scalar_type() == torch::kBFloat16) {
        using scalar_t = at::BFloat16;
        recv_sort_scalar_kernel<scalar_t>; // keep nvcc happy with at::BFloat16 linkage
        convert_mapping_copy_scalar_kernel<scalar_t><<<cblocks, threads, 0, stream>>>(
            hidden.data_ptr<scalar_t>(),
            mapping.data_ptr<int64_t>(),
            sendbuf.data_ptr<scalar_t>(),
            expected_rows,
            T,
            H
        );
    } else if (hidden.scalar_type() == torch::kHalf) {
        using scalar_t = at::Half;
        convert_mapping_copy_scalar_kernel<scalar_t><<<cblocks, threads, 0, stream>>>(
            hidden.data_ptr<scalar_t>(),
            mapping.data_ptr<int64_t>(),
            sendbuf.data_ptr<scalar_t>(),
            expected_rows,
            T,
            H
        );
    } else if (hidden.scalar_type() == torch::kFloat32) {
        convert_mapping_copy_scalar_kernel<float><<<cblocks, threads, 0, stream>>>(
            hidden.data_ptr<float>(),
            mapping.data_ptr<int64_t>(),
            sendbuf.data_ptr<float>(),
            expected_rows,
            T,
            H
        );
    } else {
        TORCH_CHECK(false, "unsupported hidden dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void build_recv_meta(
    torch::Tensor meta_ptrs,
    torch::Tensor num_global,
    torch::Tensor remote_row_offsets,
    torch::Tensor dst_row_offsets,
    torch::Tensor chunk_counts,
    int rank,
    int world_size,
    int num_local_experts
) {
    CHECK_CUDA(meta_ptrs);
    CHECK_CUDA(num_global);
    CHECK_CUDA(remote_row_offsets);
    CHECK_CUDA(dst_row_offsets);
    CHECK_CUDA(chunk_counts);
    CHECK_CONTIG(meta_ptrs);
    CHECK_CONTIG(num_global);
    CHECK_CONTIG(remote_row_offsets);
    CHECK_CONTIG(dst_row_offsets);
    CHECK_CONTIG(chunk_counts);

    TORCH_CHECK(meta_ptrs.dtype() == torch::kInt64, "meta_ptrs must be int64");
    TORCH_CHECK(num_global.dtype() == torch::kInt64, "num_global must be int64");
    TORCH_CHECK(remote_row_offsets.dtype() == torch::kInt64, "remote_row_offsets must be int64");
    TORCH_CHECK(dst_row_offsets.dtype() == torch::kInt64, "dst_row_offsets must be int64");
    TORCH_CHECK(chunk_counts.dtype() == torch::kInt64, "chunk_counts must be int64");

    int chunks = world_size * num_local_experts;
    int threads = 128;
    int blocks = (chunks + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    build_recv_meta_kernel<<<blocks, threads, 0, stream>>>(
        meta_ptrs.data_ptr<int64_t>(),
        num_global.data_ptr<int64_t>(),
        remote_row_offsets.data_ptr<int64_t>(),
        dst_row_offsets.data_ptr<int64_t>(),
        chunk_counts.data_ptr<int64_t>(),
        rank,
        world_size,
        num_local_experts
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void recv_sort_alltoall(
    torch::Tensor send_ptrs,
    torch::Tensor remote_row_offsets,
    torch::Tensor dst_row_offsets,
    torch::Tensor chunk_counts,
    torch::Tensor out,
    int world_size,
    int num_local_experts
) {
    CHECK_CUDA(send_ptrs);
    CHECK_CUDA(remote_row_offsets);
    CHECK_CUDA(dst_row_offsets);
    CHECK_CUDA(chunk_counts);
    CHECK_CUDA(out);
    CHECK_CONTIG(send_ptrs);
    CHECK_CONTIG(remote_row_offsets);
    CHECK_CONTIG(dst_row_offsets);
    CHECK_CONTIG(chunk_counts);
    CHECK_CONTIG(out);

    TORCH_CHECK(send_ptrs.dtype() == torch::kInt64, "send_ptrs must be int64");
    TORCH_CHECK(remote_row_offsets.dtype() == torch::kInt64, "remote_row_offsets must be int64");
    TORCH_CHECK(dst_row_offsets.dtype() == torch::kInt64, "dst_row_offsets must be int64");
    TORCH_CHECK(chunk_counts.dtype() == torch::kInt64, "chunk_counts must be int64");
    TORCH_CHECK(out.dim() == 2, "out must be [rows, H]");

    int chunks = world_size * num_local_experts;
    if (chunks == 0 || out.numel() == 0) {
        return;
    }

    int64_t H = out.size(1);
    int64_t row_bytes = H * out.element_size();
    int threads = 256;
    int yblocks = 32;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 grid(chunks, yblocks, 1);

    if ((row_bytes % 16) == 0) {
        recv_sort_vec16_kernel<<<grid, threads, 0, stream>>>(
            send_ptrs.data_ptr<int64_t>(),
            remote_row_offsets.data_ptr<int64_t>(),
            dst_row_offsets.data_ptr<int64_t>(),
            chunk_counts.data_ptr<int64_t>(),
            reinterpret_cast<char*>(out.data_ptr()),
            world_size,
            num_local_experts,
            row_bytes,
            row_bytes / 16
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    if (out.scalar_type() == torch::kBFloat16) {
        using scalar_t = at::BFloat16;
        recv_sort_scalar_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            send_ptrs.data_ptr<int64_t>(),
            remote_row_offsets.data_ptr<int64_t>(),
            dst_row_offsets.data_ptr<int64_t>(),
            chunk_counts.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            world_size,
            num_local_experts,
            H
        );
    } else if (out.scalar_type() == torch::kHalf) {
        using scalar_t = at::Half;
        recv_sort_scalar_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            send_ptrs.data_ptr<int64_t>(),
            remote_row_offsets.data_ptr<int64_t>(),
            dst_row_offsets.data_ptr<int64_t>(),
            chunk_counts.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            world_size,
            num_local_experts,
            H
        );
    } else if (out.scalar_type() == torch::kFloat32) {
        recv_sort_scalar_kernel<float><<<grid, threads, 0, stream>>>(
            send_ptrs.data_ptr<int64_t>(),
            remote_row_offsets.data_ptr<int64_t>(),
            dst_row_offsets.data_ptr<int64_t>(),
            chunk_counts.data_ptr<int64_t>(),
            out.data_ptr<float>(),
            world_size,
            num_local_experts,
            H
        );
    } else {
        TORCH_CHECK(false, "unsupported out dtype");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cub_select_temp_bytes", &cub_select_temp_bytes,
          "Query CUB DeviceSelect temp bytes");
    m.def("prepare_moe_send", &prepare_moe_send,
          "Build routing_map, stable permutation mapping, and symmetric send buffer");
    m.def("build_recv_meta", &build_recv_meta,
          "Build fused receive/sort chunk metadata");
    m.def("recv_sort_alltoall", &recv_sort_alltoall,
          "UVA peer all-to-all receive fused with chunk reorder");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_pre_all2all_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _sum_splits_host(splits: Union[List[int], torch.Tensor]) -> int:
    if isinstance(splits, list):
        return int(sum(int(x) for x in splits))
    return int(splits.sum().item())


def _splits_to_device(
    splits: Union[List[int], torch.Tensor],
    *,
    device: torch.device,
    world_size: int,
) -> torch.Tensor:
    if isinstance(splits, list):
        return torch.tensor(splits, device=device, dtype=torch.int64)
    if splits.device != device or splits.dtype != torch.int64 or not splits.is_contiguous():
        return splits.to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
    return splits


def _num_global_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.device != device or x.dtype != torch.int64 or not x.is_contiguous():
        return x.to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
    return x


def _get_resources(
    *,
    expected_rows: int,
    out_rows: int,
    hidden_dim: int,
    num_tokens: int,
    num_experts: int,
    world_size: int,
    num_local_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
):
    key = (
        int(max(expected_rows, 1)),
        int(max(out_rows, 1)),
        int(hidden_dim),
        int(num_tokens),
        int(num_experts),
        int(world_size),
        int(num_local_experts),
        dtype,
        device,
        id(group),
    )
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    ext = _get_ext()

    send_capacity = max(int(expected_rows), 1)
    out_capacity = max(int(out_rows), 1)
    chunks = int(world_size) * int(num_local_experts)

    sendbuf = symm_mem.empty((send_capacity, hidden_dim), device=device, dtype=dtype)
    send_hdl = symm_mem.rendezvous(sendbuf, group)

    split_meta = symm_mem.empty((world_size,), device=device, dtype=torch.int64)
    meta_hdl = symm_mem.rendezvous(split_meta, group)

    routing_map = torch.empty((num_experts, num_tokens), device=device, dtype=torch.int64)
    mapping = torch.empty((max(expected_rows, 1),), device=device, dtype=torch.int64)
    selected_count = torch.empty((1,), device=device, dtype=torch.int64)

    temp_bytes = int(ext.cub_select_temp_bytes(routing_map, mapping, selected_count))
    cub_temp = torch.empty((max(temp_bytes, 1),), device=device, dtype=torch.uint8)

    send_ptrs = torch.tensor(send_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    meta_ptrs = torch.tensor(meta_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    remote_row_offsets = torch.empty((max(chunks, 1),), device=device, dtype=torch.int64)
    dst_row_offsets = torch.empty((max(chunks, 1),), device=device, dtype=torch.int64)
    chunk_counts = torch.empty((max(chunks, 1),), device=device, dtype=torch.int64)

    out = torch.empty((out_capacity, hidden_dim), device=device, dtype=dtype)

    res = {
        "sendbuf": sendbuf,
        "send_hdl": send_hdl,
        "split_meta": split_meta,
        "meta_hdl": meta_hdl,
        "routing_map": routing_map,
        "mapping": mapping,
        "selected_count": selected_count,
        "cub_temp": cub_temp,
        "send_ptrs": send_ptrs,
        "meta_ptrs": meta_ptrs,
        "remote_row_offsets": remote_row_offsets,
        "dst_row_offsets": dst_row_offsets,
        "chunk_counts": chunk_counts,
        "out": out,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: Union[List[int], torch.Tensor],
    output_splits: Union[List[int], torch.Tensor],
    num_global_tokens_per_local_expert: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    group = group or dist.group.WORLD

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = hidden_states.device
    hidden_dim = hidden_states.size(-1)

    hidden_2d = hidden_states.reshape(-1, hidden_dim)
    if not hidden_2d.is_contiguous():
        hidden_2d = hidden_2d.contiguous()
    if not expert_mask.is_contiguous():
        expert_mask = expert_mask.contiguous()

    org_hidden_states_shape = hidden_2d.shape
    num_tokens = hidden_2d.size(0)

    expected_rows = _sum_splits_host(input_splits)
    out_rows = _sum_splits_host(output_splits)
    num_local_experts = num_experts // world_size

    resources = _get_resources(
        expected_rows=expected_rows,
        out_rows=out_rows,
        hidden_dim=hidden_dim,
        num_tokens=num_tokens,
        num_experts=num_experts,
        world_size=world_size,
        num_local_experts=num_local_experts,
        dtype=hidden_2d.dtype,
        device=device,
        group=group,
    )

    splits_dev = _splits_to_device(input_splits, device=device, world_size=world_size)
    resources["split_meta"].copy_(splits_dev, non_blocking=True)

    ext = _get_ext()
    ext.prepare_moe_send(
        hidden_2d,
        expert_mask,
        resources["sendbuf"],
        resources["routing_map"],
        resources["mapping"],
        resources["selected_count"],
        resources["cub_temp"],
        int(expected_rows),
    )

    # Symmetric-memory rendezvous: publishes both local split metadata and staged tokens.
    resources["send_hdl"].barrier(channel=0)

    num_global = _num_global_to_device(num_global_tokens_per_local_expert, device)

    ext.build_recv_meta(
        resources["meta_ptrs"],
        num_global,
        resources["remote_row_offsets"],
        resources["dst_row_offsets"],
        resources["chunk_counts"],
        int(rank),
        int(world_size),
        int(num_local_experts),
    )

    ext.recv_sort_alltoall(
        resources["send_ptrs"],
        resources["remote_row_offsets"],
        resources["dst_row_offsets"],
        resources["chunk_counts"],
        resources["out"],
        int(world_size),
        int(num_local_experts),
    )

    global_permuted_hidden_states = resources["out"][:out_rows]
    routing_map = resources["routing_map"]
    local_input_permutation_mapping = resources["mapping"][:expected_rows]

    return (
        global_permuted_hidden_states,
        routing_map,
        local_input_permutation_mapping,
        org_hidden_states_shape,
    )