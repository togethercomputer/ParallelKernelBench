from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename scalar_t>
__device__ __forceinline__ long long mask_to_count(scalar_t v) {
    // expert_mask is a binary mask in the MoE preprocess path.
    return v != scalar_t(0);
}

template <typename scalar_t>
__global__ void count_expert_mask_kernel(
    const scalar_t* __restrict__ mask,
    long long* __restrict__ counts,
    int64_t E,
    int64_t K,
    int64_t T,
    int64_t s0,
    int64_t s1,
    int64_t s2
) {
    int e = blockIdx.x;
    int tid = threadIdx.x;

    long long local = 0;
    int64_t n = K * T;
    int64_t base_e = (int64_t)e * s0;

    for (int64_t i = tid; i < n; i += blockDim.x) {
        int64_t k = i / T;
        int64_t t = i - k * T;
        scalar_t v = mask[base_e + k * s1 + t * s2];
        local += mask_to_count<scalar_t>(v);
    }

    extern __shared__ long long smem[];
    smem[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        counts[e] = smem[0];
    }
}

__device__ __forceinline__ const long long* ptr_from_i64(long long p) {
    return reinterpret_cast<const long long*>(static_cast<uintptr_t>(p));
}

__global__ void aggregate_moe_preprocess_kernel(
    const long long* __restrict__ count_ptrs,
    long long* __restrict__ packed,
    int ep_size,
    int rank,
    int num_local_experts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    long long* input_splits = packed;
    long long* output_splits = packed + ep_size;
    long long* local_matrix = packed + 2 * ep_size;
    long long* local_sums = local_matrix + (int64_t)ep_size * num_local_experts;

    const long long* local_counts = ptr_from_i64(count_ptrs[rank]);
    const int local_start = rank * num_local_experts;

    // [ep_size, num_local_experts] for this rank's experts.
    int matrix_elems = ep_size * num_local_experts;
    for (int idx = tid; idx < matrix_elems; idx += stride) {
        int src_rank = idx / num_local_experts;
        int j = idx - src_rank * num_local_experts;
        const long long* src_counts = ptr_from_i64(count_ptrs[src_rank]);
        local_matrix[idx] = src_counts[local_start + j];
    }

    // input_splits[d] = sum over this rank's counts for destination d's experts.
    for (int d = tid; d < ep_size; d += stride) {
        long long s = 0;
        int off = d * num_local_experts;
        #pragma unroll 1
        for (int j = 0; j < num_local_experts; ++j) {
            s += local_counts[off + j];
        }
        input_splits[d] = s;
    }

    // output_splits[src] = tokens this rank receives from src for local experts.
    for (int src = tid; src < ep_size; src += stride) {
        const long long* src_counts = ptr_from_i64(count_ptrs[src]);
        long long s = 0;
        #pragma unroll 1
        for (int j = 0; j < num_local_experts; ++j) {
            s += src_counts[local_start + j];
        }
        output_splits[src] = s;
    }

    // num_global_sum_tokens_per_local_expert[j] = sum over source ranks.
    for (int j = tid; j < num_local_experts; j += stride) {
        long long s = 0;
        #pragma unroll 1
        for (int src = 0; src < ep_size; ++src) {
            const long long* src_counts = ptr_from_i64(count_ptrs[src]);
            s += src_counts[local_start + j];
        }
        local_sums[j] = s;
    }
}

void count_expert_mask_i64(torch::Tensor expert_mask, torch::Tensor counts) {
    TORCH_CHECK(expert_mask.is_cuda(), "expert_mask must be CUDA");
    TORCH_CHECK(counts.is_cuda(), "counts must be CUDA");
    TORCH_CHECK(expert_mask.dim() == 3, "expert_mask must be [num_experts, topk, num_tokens]");
    TORCH_CHECK(counts.dtype() == torch::kInt64, "counts must be int64");
    TORCH_CHECK(counts.is_contiguous(), "counts must be contiguous");

    int64_t E = expert_mask.size(0);
    int64_t K = expert_mask.size(1);
    int64_t T = expert_mask.size(2);
    TORCH_CHECK(counts.numel() >= E, "counts buffer too small");

    int threads = 256;
    dim3 blocks((unsigned int)E);
    size_t shmem = threads * sizeof(long long);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_ALL_TYPES_AND3(
        at::kBool,
        at::kHalf,
        at::kBFloat16,
        expert_mask.scalar_type(),
        "count_expert_mask_i64",
        [&] {
            count_expert_mask_kernel<scalar_t><<<blocks, threads, shmem, stream>>>(
                expert_mask.data_ptr<scalar_t>(),
                reinterpret_cast<long long*>(counts.data_ptr<int64_t>()),
                E,
                K,
                T,
                expert_mask.stride(0),
                expert_mask.stride(1),
                expert_mask.stride(2)
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void aggregate_moe_preprocess(
    torch::Tensor count_ptrs,
    torch::Tensor packed,
    int ep_size,
    int rank,
    int num_local_experts
) {
    TORCH_CHECK(count_ptrs.is_cuda(), "count_ptrs must be CUDA");
    TORCH_CHECK(packed.is_cuda(), "packed must be CUDA");
    TORCH_CHECK(count_ptrs.dtype() == torch::kInt64, "count_ptrs must be int64");
    TORCH_CHECK(packed.dtype() == torch::kInt64, "packed must be int64");
    TORCH_CHECK(count_ptrs.is_contiguous() && packed.is_contiguous(), "tensors must be contiguous");

    int64_t need = 2LL * ep_size + (int64_t)ep_size * num_local_experts + num_local_experts;
    TORCH_CHECK(packed.numel() >= need, "packed buffer too small");

    int threads = 256;
    int work = ep_size * num_local_experts;
    if (work < ep_size) work = ep_size;
    if (work < num_local_experts) work = num_local_experts;
    int blocks = (work + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > 32) blocks = 32;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    aggregate_moe_preprocess_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(count_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<long long*>(packed.data_ptr<int64_t>()),
        ep_size,
        rank,
        num_local_experts
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_expert_mask_i64", &count_expert_mask_i64,
          "Count binary expert mask into int64 counts");
    m.def("aggregate_moe_preprocess", &aggregate_moe_preprocess,
          "Aggregate MoE preprocess counts through symmetric-memory UVA pointers");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_ep_preprocess_symm_cuda_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _group_rank_size(group: Optional[dist.ProcessGroup]) -> Tuple[int, int, dist.ProcessGroup]:
    if group is None:
        group = dist.group.WORLD
    return dist.get_rank(group), dist.get_world_size(group), group


def _get_resources(
    num_experts: int,
    ep_size: int,
    rank: int,
    group: dist.ProcessGroup,
    device: torch.device,
):
    dev_index = device.index
    if dev_index is None:
        dev_index = torch.cuda.current_device()

    key = (id(group), int(dev_index), int(num_experts), int(ep_size), int(rank))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    num_local_experts = num_experts // ep_size
    packed_len = 2 * ep_size + ep_size * num_local_experts + num_local_experts

    counts = symm_mem.empty((num_experts,), device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(counts, group)

    ptrs = torch.tensor([int(p) for p in hdl.buffer_ptrs], device=device, dtype=torch.int64)
    packed = torch.empty((packed_len,), device=device, dtype=torch.int64)

    cached = {
        "counts": counts,
        "hdl": hdl,
        "ptrs": ptrs,
        "packed": packed,
        "num_local_experts": num_local_experts,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    expert_mask: torch.Tensor,
    num_experts: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
    """
    MoE EP preprocess:
      - count local binary expert_mask per expert with CUDA
      - publish counts in symmetric memory
      - read peer count shards directly through UVA, avoiding all_gather/NCCL
      - return the same API objects as the reference implementation
    """
    assert expert_mask.is_cuda, "expert_mask must be a CUDA tensor"
    assert expert_mask.dim() == 3, "expert_mask must be [num_experts, topk, num_tokens]"
    assert expert_mask.size(0) == num_experts, "num_experts must match expert_mask.size(0)"

    ext = _get_ext()
    device = expert_mask.device

    if not dist.is_initialized():
        ep_size = 1
        rank = 0
        num_local_experts = num_experts
        counts = torch.empty((num_experts,), device=device, dtype=torch.int64)
        ptrs = torch.tensor([counts.data_ptr()], device=device, dtype=torch.int64)
        packed = torch.empty((2 + num_experts + num_experts,), device=device, dtype=torch.int64)

        ext.count_expert_mask_i64(expert_mask, counts)
        ext.aggregate_moe_preprocess(ptrs, packed, ep_size, rank, num_local_experts)

        packed_cpu = packed.cpu()
        input_splits = packed_cpu.narrow(0, 0, 1).tolist()
        output_splits = packed_cpu.narrow(0, 1, 1).tolist()
        matrix = packed_cpu.narrow(0, 2, num_experts).view(1, num_experts)
        sums = packed_cpu.narrow(0, 2 + num_experts, num_experts)
        return input_splits, output_splits, matrix, sums

    rank, ep_size, group = _group_rank_size(group)
    assert num_experts % ep_size == 0, "num_experts must be divisible by EP size"

    res = _get_resources(num_experts, ep_size, rank, group, device)
    counts = res["counts"]
    hdl = res["hdl"]
    ptrs = res["ptrs"]
    packed = res["packed"]
    num_local_experts = res["num_local_experts"]

    # Local count directly into this rank's symmetric buffer.
    ext.count_expert_mask_i64(expert_mask, counts)

    # Symmetric-memory synchronization; after this all peer count buffers are visible by UVA.
    hdl.barrier(channel=0)

    # Fill one packed GPU buffer:
    # [input_splits(ep), output_splits(ep), local_matrix(ep*L), local_sums(L)].
    ext.aggregate_moe_preprocess(ptrs, packed, ep_size, rank, num_local_experts)

    # Required API returns Python lists and CPU tensors; keep it to one packed D2H copy.
    packed_cpu = packed.cpu()

    off0 = 0
    off1 = off0 + ep_size
    off2 = off1 + ep_size
    off3 = off2 + ep_size * num_local_experts

    input_splits = packed_cpu.narrow(0, off0, ep_size).tolist()
    output_splits = packed_cpu.narrow(0, off1, ep_size).tolist()
    num_global_tokens_per_local_expert = packed_cpu.narrow(
        0, off2, ep_size * num_local_experts
    ).view(ep_size, num_local_experts)
    num_global_sum_tokens_per_local_expert = packed_cpu.narrow(
        0, off3, num_local_experts
    )

    return (
        input_splits,
        output_splits,
        num_global_tokens_per_local_expert,
        num_global_sum_tokens_per_local_expert,
    )