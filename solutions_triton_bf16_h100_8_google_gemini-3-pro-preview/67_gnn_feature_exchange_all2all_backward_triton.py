from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <cuda_runtime.h>

// Kernel to push local chunks directly to peer symmetric memory
__global__ void copy_chunk_kernel(
    const at::BFloat16* __restrict__ src,
    int64_t src_offset,
    int64_t size,
    int64_t H,
    const int64_t* __restrict__ dest_meta,
    int m,
    at::BFloat16* __restrict__ dest_out
) {
    int64_t dest_offset = dest_meta[m];
    int64_t total_elements = size * H;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    // Fast path: Vectorized 128-bit loads/stores if pointers are aligned
    bool aligned = ((reinterpret_cast<uintptr_t>(src + src_offset * H) % 16) == 0) &&
                   ((reinterpret_cast<uintptr_t>(dest_out + dest_offset * H) % 16) == 0);

    if (aligned) {
        int64_t vec_elements = total_elements / 8;
        int64_t remainder = total_elements % 8;

        const float4* src_vec = reinterpret_cast<const float4*>(src + src_offset * H);
        float4* dest_vec = reinterpret_cast<float4*>(dest_out + dest_offset * H);

        for (int64_t i = tid; i < vec_elements; i += stride) {
            dest_vec[i] = src_vec[i];
        }

        if (tid == 0 && remainder > 0) {
            for (int64_t i = vec_elements * 8; i < total_elements; ++i) {
                dest_out[dest_offset * H + i] = src[src_offset * H + i];
            }
        }
    } else {
        for (int64_t i = tid; i < total_elements; i += stride) {
            dest_out[dest_offset * H + i] = src[src_offset * H + i];
        }
    }
}

// Kernel to signal a peer that its chunk has fully arrived
__global__ void set_flag_kernel(int* __restrict__ dest_flags, int my_rank) {
    __threadfence_system();
    atomicExch(&dest_flags[my_rank], 1);
}

// Kernel to wait for chunk arrival flags and immediately scatter-add
__global__ void poll_and_scatter_kernel(
    const at::BFloat16* __restrict__ out_symm,
    const int64_t* __restrict__ recv_offsets,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ seed_inverse_ids,
    at::BFloat16* __restrict__ grad_input,
    volatile int* __restrict__ flags_symm,
    int64_t H,
    int W,
    int my_rank
) {
    int r = blockIdx.y; // Iterating over peer rank via block dimension
    int m = (r - my_rank + W) % W; // Compute rotated chunk index for peer r
    int64_t size = sizes[m];
    if (size == 0) return;

    // Block 0 spins on the UVA flag updated by the remote peer
    if (threadIdx.x == 0) {
        while (flags_symm[r] == 0) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(100);
#endif
        }
    }
    __syncthreads(); // Ensure all threads see the flag before reading

    int64_t offset = recv_offsets[m];
    int64_t total_elements = size * H;
    int64_t start_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (int64_t i = start_idx; i < total_elements; i += blockDim.x * gridDim.x) {
        int64_t row = i / H;
        int64_t col = i % H;
        int64_t dest_row = seed_inverse_ids[offset + row];
        // Native BF16 atomicAdd
        gpuAtomicAdd(&grad_input[dest_row * H + col], out_symm[offset * H + i]);
    }
}

void push_chunks(
    torch::Tensor src,
    std::vector<int64_t> src_offsets,
    std::vector<int64_t> sizes,
    int64_t H,
    int64_t W,
    int64_t my_rank,
    std::vector<int64_t> dest_out_ptrs,
    std::vector<int64_t> dest_meta_ptrs,
    std::vector<int64_t> dest_flags_ptrs
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    for (int k = 0; k < W; ++k) {
        int64_t size = sizes[k];
        int P = (my_rank + k) % W;
        int m = (W - k) % W;
        int64_t src_offset = src_offsets[k];

        at::BFloat16* dest_out = reinterpret_cast<at::BFloat16*>(dest_out_ptrs[P]);
        int64_t* dest_meta = reinterpret_cast<int64_t*>(dest_meta_ptrs[P]);
        int* dest_flags = reinterpret_cast<int*>(dest_flags_ptrs[P]);

        if (size > 0) {
            int threads = 256;
            int64_t total = size * H;
            int blocks = std::min((int)((total + 8 * threads - 1) / (8 * threads)), 4096);
            if (blocks == 0) blocks = 1;

            copy_chunk_kernel<<<blocks, threads, 0, stream>>>(
                src.data_ptr<at::BFloat16>(),
                src_offset,
                size,
                H,
                dest_meta,
                m,
                dest_out
            );
        }

        // Emit signal once copy launches are submitted for this chunk
        set_flag_kernel<<<1, 1, 0, stream>>>(dest_flags, my_rank);
    }
}

void poll_and_scatter(
    torch::Tensor out_symm,
    torch::Tensor recv_offsets,
    torch::Tensor sizes,
    torch::Tensor seed_inverse_ids,
    torch::Tensor grad_input,
    torch::Tensor flags_symm,
    int64_t H,
    int64_t W,
    int64_t my_rank
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks_per_chunk = 108; // High occupancy width per chunk
    dim3 blocks(blocks_per_chunk, W);

    poll_and_scatter_kernel<<<blocks, threads, 0, stream>>>(
        out_symm.data_ptr<at::BFloat16>(),
        recv_offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        seed_inverse_ids.data_ptr<int64_t>(),
        grad_input.data_ptr<at::BFloat16>(),
        flags_symm.data_ptr<int>(),
        H,
        W,
        my_rank
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("push_chunks", &push_chunks, "Push local buffers directly to remote peers");
    m.def("poll_and_scatter", &poll_and_scatter, "Wait on flags and scatter add over stream");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_bwd_push_scatter", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(W: int, sum_recv: int, H: int, dtype: torch.dtype, device: torch.device):
    key = (sum_recv, H, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]

    out_symm = symm_mem.empty((sum_recv, H), dtype=dtype, device=device)
    meta_symm = symm_mem.empty((W,), dtype=torch.int64, device=device)
    flags_symm = symm_mem.empty((W,), dtype=torch.int32, device=device)

    hdl_out = symm_mem.rendezvous(out_symm, dist.group.WORLD)
    hdl_meta = symm_mem.rendezvous(meta_symm, dist.group.WORLD)
    hdl_flags = symm_mem.rendezvous(flags_symm, dist.group.WORLD)

    ptrs_out = [int(p) for p in hdl_out.buffer_ptrs]
    ptrs_meta = [int(p) for p in hdl_meta.buffer_ptrs]
    ptrs_flags = [int(p) for p in hdl_flags.buffer_ptrs]

    res = (out_symm, meta_symm, flags_symm, hdl_out, ptrs_out, ptrs_meta, ptrs_flags)
    _symm_cache[key] = res
    return res


@torch.no_grad()
def solution(
    grad_output: torch.Tensor,
    seed_inverse_ids: torch.Tensor,
    seed_size: int,
    counts_sent: List[int],
    counts_received: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    W = dist.get_world_size(group)
    rank = dist.get_rank(group)

    ext = _get_ext()
    dist.barrier(group)

    H = grad_output.numel() // max(1, grad_output.size(0))
    sum_recv = sum(counts_received)

    (out_symm, meta_symm, flags_symm, hdl_out, 
     ptrs_out, ptrs_meta, ptrs_flags) = _get_symm_state(
        W, sum_recv, H, grad_output.dtype, grad_output.device
    )

    # 1. Reset device flags
    flags_symm.zero_()

    # 2. Pre-calculate chunk offsets for inbound payload routing
    recv_offsets = [0] * W
    cum = 0
    for i in range(W):
        recv_offsets[i] = cum
        cum += counts_received[i]

    # Stash the offsets so peers can query them during UVA pushes
    meta_symm.copy_(torch.tensor(recv_offsets, dtype=torch.int64, device=grad_output.device))
    hdl_out.barrier(channel=0)

    # Calculate local start offsets for sending
    sent_offsets = [0] * W
    cum = 0
    for i in range(W):
        sent_offsets[i] = cum
        cum += counts_sent[i]

    # 3. Fire-and-forget chunks via symmetric UVA writes
    ext.push_chunks(
        grad_output,
        sent_offsets,
        counts_sent,
        H,
        W,
        rank,
        ptrs_out,
        ptrs_meta,
        ptrs_flags
    )

    # 4. Allocate outputs entirely on device
    grad_input = torch.zeros((seed_size, H), dtype=grad_output.dtype, device=grad_output.device)
    counts_recv_t = torch.tensor(counts_received, dtype=torch.int64, device=grad_output.device)

    # 5. Overlap scatter reduction ops by polling the sync flags
    ext.poll_and_scatter(
        out_symm,
        meta_symm, 
        counts_recv_t,
        seed_inverse_ids,
        grad_input,
        flags_symm,
        H,
        W,
        rank
    )

    hdl_out.barrier(channel=0)
    return grad_input