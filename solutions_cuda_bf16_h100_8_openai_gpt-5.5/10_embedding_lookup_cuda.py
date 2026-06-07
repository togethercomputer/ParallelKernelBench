# solutions_cuda_bf16_h100_8_openai_gpt-5.5/10_embedding_lookup_cuda.py

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_N 256
#define MAX_WORLD 8

__global__ void block_hist_kernel(
    const int64_t* __restrict__ indices,
    int64_t* __restrict__ hist,
    int64_t n,
    int64_t shard_size,
    int world_size
) {
    __shared__ int smem[MAX_WORLD];

    int tid = threadIdx.x;
    if (tid < MAX_WORLD) {
        smem[tid] = 0;
    }
    __syncthreads();

    int64_t i = (int64_t)blockIdx.x * BLOCK_N + tid;
    if (i < n) {
        int64_t idx = indices[i];
        int owner = (int)(idx / shard_size);
        if ((unsigned)owner < (unsigned)world_size) {
            atomicAdd(&smem[owner], 1);
        }
    }

    __syncthreads();

    if (tid < world_size) {
        hist[(int64_t)blockIdx.x * world_size + tid] = (int64_t)smem[tid];
    }
}

__global__ void prefix_blocks_kernel(
    const int64_t* __restrict__ hist,
    int64_t* __restrict__ block_offsets,
    int64_t* __restrict__ owner_offsets,
    int64_t num_blocks,
    int world_size
) {
    __shared__ int64_t totals[MAX_WORLD];

    int r = threadIdx.x;
    if (r < world_size) {
        int64_t run = 0;
        for (int64_t b = 0; b < num_blocks; ++b) {
            block_offsets[b * world_size + r] = run;
            run += hist[b * world_size + r];
        }
        totals[r] = run;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t prefix = 0;
        #pragma unroll
        for (int rr = 0; rr < MAX_WORLD; ++rr) {
            if (rr < world_size) {
                owner_offsets[rr] = prefix;
                prefix += totals[rr];
            }
        }
    }
}

__global__ void stable_group_indices_kernel(
    const int64_t* __restrict__ indices,
    int64_t* __restrict__ grouped,
    const int64_t* __restrict__ block_offsets,
    const int64_t* __restrict__ owner_offsets,
    int64_t n,
    int64_t shard_size,
    int world_size
) {
    __shared__ int warp_counts[8 * MAX_WORLD];
    __shared__ int warp_prefix[8 * MAX_WORLD];

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    if (tid < 8 * MAX_WORLD) {
        warp_counts[tid] = 0;
        warp_prefix[tid] = 0;
    }
    __syncthreads();

    int64_t i = (int64_t)blockIdx.x * BLOCK_N + tid;
    bool valid = i < n;

    int64_t gidx = 0;
    int owner = -1;
    if (valid) {
        gidx = indices[i];
        owner = (int)(gidx / shard_size);
        valid = ((unsigned)owner < (unsigned)world_size);
    }

    int local_rank = 0;
    const unsigned full = 0xffffffffu;
    unsigned lane_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

    #pragma unroll
    for (int r = 0; r < MAX_WORLD; ++r) {
        unsigned mask = __ballot_sync(full, valid && owner == r);
        if (lane == 0) {
            warp_counts[warp * MAX_WORLD + r] = __popc(mask);
        }
        if (valid && owner == r) {
            local_rank = __popc(mask & lane_lt);
        }
    }

    __syncthreads();

    if (tid < MAX_WORLD) {
        int r = tid;
        int run = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            warp_prefix[w * MAX_WORLD + r] = run;
            run += warp_counts[w * MAX_WORLD + r];
        }
    }

    __syncthreads();

    if (valid) {
        int64_t base =
            owner_offsets[owner] +
            block_offsets[(int64_t)blockIdx.x * world_size + owner] +
            (int64_t)warp_prefix[warp * MAX_WORLD + owner] +
            (int64_t)local_rank;
        grouped[base] = gidx;
    }
}

__global__ void peer_embedding_lookup_copy_kernel(
    const int64_t* __restrict__ grouped,
    const uint64_t* __restrict__ shard_ptrs,
    void* __restrict__ out_void,
    int64_t n,
    int64_t embed_dim,
    int64_t shard_size,
    int elem_size
) {
    int64_t total = n * embed_dim;
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    char* __restrict__ out = reinterpret_cast<char*>(out_void);

    for (; linear < total; linear += stride) {
        int64_t row = linear / embed_dim;
        int64_t col = linear - row * embed_dim;

        int64_t global_idx = grouped[row];
        int owner = (int)(global_idx / shard_size);
        int64_t local_row = global_idx - (int64_t)owner * shard_size;

        const char* src_base = reinterpret_cast<const char*>(shard_ptrs[owner]);
        int64_t elem = local_row * embed_dim + col;

        char* dst = out + linear * (int64_t)elem_size;
        const char* src = src_base + elem * (int64_t)elem_size;

        if (elem_size == 2) {
            *reinterpret_cast<uint16_t*>(dst) =
                *reinterpret_cast<const uint16_t*>(src);
        } else if (elem_size == 4) {
            *reinterpret_cast<uint32_t*>(dst) =
                *reinterpret_cast<const uint32_t*>(src);
        } else if (elem_size == 8) {
            *reinterpret_cast<uint64_t*>(dst) =
                *reinterpret_cast<const uint64_t*>(src);
        } else {
            *reinterpret_cast<uint8_t*>(dst) =
                *reinterpret_cast<const uint8_t*>(src);
        }
    }
}

void launch_embedding_lookup(
    torch::Tensor indices,
    torch::Tensor ptrs_tensor,
    torch::Tensor output,
    torch::Tensor grouped,
    torch::Tensor hist,
    torch::Tensor block_offsets,
    torch::Tensor owner_offsets,
    int64_t n,
    int64_t embed_dim,
    int64_t shard_size,
    int world_size,
    int elem_size
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(ptrs_tensor.is_cuda(), "ptrs_tensor must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(grouped.is_cuda(), "grouped must be CUDA");
    TORCH_CHECK(hist.is_cuda(), "hist must be CUDA");
    TORCH_CHECK(block_offsets.is_cuda(), "block_offsets must be CUDA");
    TORCH_CHECK(owner_offsets.is_cuda(), "owner_offsets must be CUDA");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(grouped.dtype() == torch::kInt64, "grouped must be int64");
    TORCH_CHECK(world_size <= MAX_WORLD, "world_size > 8 is not supported by this H100 on-node kernel");

    if (n == 0 || embed_dim == 0) {
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int64_t num_blocks = (n + BLOCK_N - 1) / BLOCK_N;

    block_hist_kernel<<<(unsigned)num_blocks, BLOCK_N, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        hist.data_ptr<int64_t>(),
        n,
        shard_size,
        world_size
    );

    prefix_blocks_kernel<<<1, 32, 0, stream>>>(
        hist.data_ptr<int64_t>(),
        block_offsets.data_ptr<int64_t>(),
        owner_offsets.data_ptr<int64_t>(),
        num_blocks,
        world_size
    );

    stable_group_indices_kernel<<<(unsigned)num_blocks, BLOCK_N, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        grouped.data_ptr<int64_t>(),
        block_offsets.data_ptr<int64_t>(),
        owner_offsets.data_ptr<int64_t>(),
        n,
        shard_size,
        world_size
    );

    int threads = 256;
    int64_t total = n * embed_dim;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535;
    }
    if (blocks < 1) {
        blocks = 1;
    }

    const uint64_t* ptrs = reinterpret_cast<const uint64_t*>(
        ptrs_tensor.data_ptr<int64_t>()
    );

    peer_embedding_lookup_copy_kernel<<<blocks, threads, 0, stream>>>(
        grouped.data_ptr<int64_t>(),
        ptrs,
        output.data_ptr(),
        n,
        embed_dim,
        shard_size,
        elem_size
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "launch_embedding_lookup",
        &launch_embedding_lookup,
        "Stable rank-grouped distributed embedding lookup via symmetric-memory UVA"
    );
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "embedding_lookup_symm_uva_bf16_h100_ext",
            CUDA_SRC,
        )
    return _ext


_shard_cache = {}
_work_cache = {}


def _device_key(device: torch.device):
    return device.index if device.index is not None else torch.cuda.current_device()


def _get_shard_resources(local_shard: torch.Tensor, world_size: int):
    key = (
        tuple(local_shard.shape),
        local_shard.dtype,
        _device_key(local_shard.device),
        world_size,
    )
    cached = _shard_cache.get(key)
    if cached is not None:
        return cached

    symm_shard = symm_mem.empty(
        tuple(local_shard.shape),
        device=local_shard.device,
        dtype=local_shard.dtype,
    )
    hdl = symm_mem.rendezvous(symm_shard, dist.group.WORLD)
    ptrs_tensor = torch.tensor(
        list(hdl.buffer_ptrs),
        device=local_shard.device,
        dtype=torch.int64,
    )

    cached = (symm_shard, hdl, ptrs_tensor)
    _shard_cache[key] = cached
    return cached


def _get_work_buffers(n: int, world_size: int, device: torch.device):
    if n == 0:
        return None

    num_blocks = (n + 255) // 256
    key = (n, num_blocks, world_size, _device_key(device))
    cached = _work_cache.get(key)
    if cached is not None:
        return cached

    grouped = torch.empty((n,), device=device, dtype=torch.long)
    hist = torch.empty((num_blocks, world_size), device=device, dtype=torch.long)
    block_offsets = torch.empty((num_blocks, world_size), device=device, dtype=torch.long)
    owner_offsets = torch.empty((world_size,), device=device, dtype=torch.long)

    cached = (grouped, hist, block_offsets, owner_offsets)
    _work_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    indices: torch.Tensor,
    local_shard: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert indices.is_cuda and local_shard.is_cuda, "Inputs must be CUDA tensors"
    assert indices.dtype == torch.long, "indices must be torch.long"
    assert local_shard.dim() == 2, "local_shard must have shape [ShardSize, D]"

    world_size = dist.get_world_size()
    assert world_size <= 8, "optimized kernel targets the 8-GPU H100 SXM node"

    if not indices.is_contiguous():
        indices = indices.contiguous()

    if not local_shard.is_contiguous():
        shard_src = local_shard.contiguous()
    else:
        shard_src = local_shard

    n = indices.numel()
    shard_size = shard_src.shape[0]
    embed_dim = shard_src.shape[1]

    out = torch.empty(
        (n, embed_dim),
        device=indices.device,
        dtype=shard_src.dtype,
    )

    if n == 0 or embed_dim == 0:
        return out

    symm_shard, hdl, ptrs_tensor = _get_shard_resources(shard_src, world_size)

    # Publish this rank's shard into symmetric memory; peer GPUs read it directly.
    symm_shard.copy_(shard_src)
    hdl.barrier(channel=0)

    grouped, hist, block_offsets, owner_offsets = _get_work_buffers(
        n,
        world_size,
        indices.device,
    )

    _get_ext().launch_embedding_lookup(
        indices,
        ptrs_tensor,
        out,
        grouped,
        hist,
        block_offsets,
        owner_offsets,
        int(n),
        int(embed_dim),
        int(shard_size),
        int(world_size),
        int(shard_src.element_size()),
    )

    return out