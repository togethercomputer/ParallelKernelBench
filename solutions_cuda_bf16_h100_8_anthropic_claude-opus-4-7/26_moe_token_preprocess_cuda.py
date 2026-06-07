"""
MoE EP preprocess with custom CUDA: fused sum reduction over expert_mask + 
symmetric-memory all-gather of token counts via direct peer loads over NVLink.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Reduce expert_mask [E, K, T] -> [E] sum along K,T into symmetric buffer at slot=rank
// expert_mask is bool/uint8.
template <typename T>
__global__ void reduce_expert_mask_kernel(
    const T* __restrict__ mask,    // [E, K*T]
    int64_t* __restrict__ out,      // [E] int64
    int E,
    int64_t KT
) {
    int e = blockIdx.x;
    if (e >= E) return;
    const T* row = mask + (int64_t)e * KT;
    int tid = threadIdx.x;
    int64_t sum = 0;
    for (int64_t i = tid; i < KT; i += blockDim.x) {
        sum += (int64_t)row[i];
    }
    // block reduce
    __shared__ int64_t sdata[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? sdata[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            out[e] = sum;
        }
    }
}

void launch_reduce_expert_mask(
    torch::Tensor mask,
    torch::Tensor out,
    int E,
    int64_t KT
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    if (KT < 256) {
        threads = 64;
    }
    if (mask.scalar_type() == torch::kBool || mask.scalar_type() == torch::kUInt8) {
        reduce_expert_mask_kernel<uint8_t><<<E, threads, 0, stream>>>(
            (const uint8_t*)mask.data_ptr(),
            out.data_ptr<int64_t>(),
            E, KT);
    } else if (mask.scalar_type() == torch::kInt32) {
        reduce_expert_mask_kernel<int32_t><<<E, threads, 0, stream>>>(
            (const int32_t*)mask.data_ptr(),
            out.data_ptr<int64_t>(),
            E, KT);
    } else if (mask.scalar_type() == torch::kInt64) {
        reduce_expert_mask_kernel<int64_t><<<E, threads, 0, stream>>>(
            (const int64_t*)mask.data_ptr(),
            out.data_ptr<int64_t>(),
            E, KT);
    } else {
        TORCH_CHECK(false, "unsupported dtype for expert_mask");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Gather rank-local [E] from each peer symmetric buffer into [ep_size, E]
__global__ void gather_from_peers_kernel(
    const uint64_t* __restrict__ peer_ptrs,  // [ep_size]
    int64_t* __restrict__ out,               // [ep_size, E]
    int ep_size,
    int E
) {
    int r = blockIdx.y;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int64_t* src = reinterpret_cast<const int64_t*>(peer_ptrs[r]);
    out[(int64_t)r * E + e] = src[e];
}

void launch_gather_from_peers(
    torch::Tensor peer_ptrs_tensor,  // int64 [ep_size]
    torch::Tensor out,                // int64 [ep_size, E]
    int ep_size,
    int E
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 block(128);
    dim3 grid((E + 127) / 128, ep_size);
    gather_from_peers_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(peer_ptrs_tensor.data_ptr<int64_t>()),
        out.data_ptr<int64_t>(),
        ep_size, E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_reduce_expert_mask", &launch_reduce_expert_mask);
    m.def("launch_gather_from_peers", &launch_gather_from_peers);
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_preprocess_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_resources(num_experts: int, ep_size: int, device: torch.device, group):
    key = (num_experts, ep_size, device)
    if key in _cache:
        return _cache[key]

    # Symmetric buffer: each rank writes its [num_experts] int64 counts here
    sym_buf = symm_mem.empty(num_experts, device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(sym_buf, group)
    peer_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    gathered = torch.empty(ep_size, num_experts, device=device, dtype=torch.int64)

    res = (sym_buf, hdl, peer_ptrs, gathered)
    _cache[key] = res
    return res


@torch.no_grad()
def solution(
    expert_mask: torch.Tensor,
    num_experts: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[List[int], List[int], torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD
    ep_size = group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(group)
    device = expert_mask.device

    ext = _get_ext()

    E = expert_mask.shape[0]
    KT = expert_mask.shape[1] * expert_mask.shape[2]
    mask_c = expert_mask.contiguous()

    sym_buf, hdl, peer_ptrs, gathered = _get_resources(E, ep_size, device, group)

    # Custom kernel: reduce expert_mask -> sym_buf[E] (int64)
    ext.launch_reduce_expert_mask(mask_c, sym_buf, E, KT)

    # input_splits: sym_buf reshaped [ep_size, num_local_experts].sum(dim=1)
    # Compute on device but we need .tolist(); do small reduction then async copy
    input_splits_dev = sym_buf.view(ep_size, num_local_experts).sum(dim=1)

    # Barrier so all peers have written sym_buf
    hdl.barrier(channel=0)

    # Gather from peer symmetric buffers via UVA
    ext.launch_gather_from_peers(peer_ptrs, gathered, ep_size, E)

    # Slice this rank's experts: [ep_size, num_local_experts]
    start = rank * num_local_experts
    end = start + num_local_experts
    num_global_tokens_per_local_expert_dev = gathered[:, start:end].contiguous()

    output_splits_dev = num_global_tokens_per_local_expert_dev.sum(dim=1)
    num_global_sum_dev = num_global_tokens_per_local_expert_dev.sum(dim=0)

    # Async D2H copies
    input_splits_cpu = input_splits_dev.to("cpu", non_blocking=True)
    output_splits_cpu = output_splits_dev.to("cpu", non_blocking=True)
    num_global_sum_cpu = num_global_sum_dev.to("cpu", non_blocking=True)
    num_global_tokens_cpu = num_global_tokens_per_local_expert_dev.view(-1, num_local_experts).to(
        "cpu", non_blocking=True
    )

    # Final barrier ensures peers don't race-overwrite sym_buf next call
    hdl.barrier(channel=1)

    torch.cuda.current_stream().synchronize()

    return (
        input_splits_cpu.tolist(),
        output_splits_cpu.tolist(),
        num_global_tokens_cpu,
        num_global_sum_cpu,
    )