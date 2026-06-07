from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void uva_write_sizes_kernel(
    int64_t val,
    const int64_t* __restrict__ peer_ptrs,
    int rank,
    int world_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < world_size; ++i) {
            int64_t* peer_buf = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(peer_ptrs[i]));
            peer_buf[rank] = val;
        }
    }
}

__global__ void compute_and_scatter_rankings_kernel(
    const __nv_bfloat16* __restrict__ pos_scores,
    const __nv_bfloat16* __restrict__ neg_scores,
    const int64_t* __restrict__ peer_ptrs,
    int offset,
    int P,
    int K,
    int world_size
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;
    int row = blockIdx.x * num_warps + warp_id;
    
    int count = 0;
    if (row < P) {
        // Reproduce exactly PyTorch's bfloat16 -> float -> sigmoid -> bfloat16 precision boundaries
        float pos_val = __bfloat162float(pos_scores[row]);
        float sig_pos_f = __bfloat162float(__float2bfloat16(sigmoidf(pos_val)));
        
        for (int i = lane; i < K; i += 32) {
            float neg_val = __bfloat162float(neg_scores[row * K + i]);
            float sig_neg_f = __bfloat162float(__float2bfloat16(sigmoidf(neg_val)));
            if (sig_neg_f > sig_pos_f) {
                count++;
            }
        }
        
        // Fast warp-reduction for the count
        #pragma unroll
        for (int offset_shfl = 16; offset_shfl > 0; offset_shfl /= 2) {
            count += __shfl_down_sync(0xffffffff, count, offset_shfl);
        }
    }
    
    // Consolidate values inside the block to allow grouped 64-byte writes over NVLink
    __shared__ int64_t smem_rank[32]; // Accommodates up to 1024 threads/block
    if (lane == 0 && row < P) {
        smem_rank[warp_id] = count + 1;
    }
    __syncthreads();
    
    // Leader warp scatters the block's computed chunk to all remote peers
    if (threadIdx.x < num_warps) {
        int write_row = blockIdx.x * num_warps + threadIdx.x;
        if (write_row < P) {
            int64_t rank_val = smem_rank[threadIdx.x];
            for (int peer = 0; peer < world_size; ++peer) {
                int64_t* peer_buf = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(peer_ptrs[peer]));
                peer_buf[offset + write_row] = rank_val;
            }
        }
    }
}

void uva_write_sizes(
    int64_t val,
    torch::Tensor peer_ptrs_tensor,
    int rank,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    uva_write_sizes_kernel<<<1, 32, 0, stream>>>(
        val,
        peer_ptrs_tensor.data_ptr<int64_t>(),
        rank,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_and_scatter_rankings(
    torch::Tensor pos_scores,
    torch::Tensor neg_scores,
    torch::Tensor peer_ptrs_tensor,
    int offset,
    int world_size
) {
    int P = pos_scores.size(0);
    int K = neg_scores.size(1);
    if (P == 0) return;
    
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (P + warps_per_block - 1) / warps_per_block;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    compute_and_scatter_rankings_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(pos_scores.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(neg_scores.data_ptr()),
        peer_ptrs_tensor.data_ptr<int64_t>(),
        offset, P, K, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_write_sizes", &uva_write_sizes, "UVA write sizes scatter kernel");
    m.def("compute_and_scatter_rankings", &compute_and_scatter_rankings, "Fused ranking and UVA scatter");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_ranking_uva_scatter", CUDA_SRC)
    return _ext

_size_symm_cache = {}
def _get_size_symm_state(world_size: int, device: torch.device, group: dist.ProcessGroup):
    global _size_symm_cache
    if world_size in _size_symm_cache:
        return _size_symm_cache[world_size]
    
    buf = symm_mem.empty(world_size, dtype=torch.long, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.long, device=device)
    state = (buf, hdl, ptrs_tensor)
    _size_symm_cache[world_size] = state
    return state

_data_symm_cache = {}
def _get_data_symm_state(total_size: int, device: torch.device, group: dist.ProcessGroup):
    global _data_symm_cache
    if total_size in _data_symm_cache:
        return _data_symm_cache[total_size]
    
    # Restrict cache to evade OOMs if the workload supplies erratic tensor bounds
    if len(_data_symm_cache) >= 5:
        _data_symm_cache.clear()
        
    buf = symm_mem.empty(total_size, dtype=torch.long, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.long, device=device)
    state = (buf, hdl, ptrs_tensor)
    _data_symm_cache[total_size] = state
    return state


@torch.no_grad()
def solution(
    local_pos_scores: torch.Tensor,
    local_neg_scores: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Per-rank GraphStorm-style link-prediction ranking.
    """
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    device = local_pos_scores.device

    local_pos_scores = local_pos_scores.contiguous()
    local_neg_scores = local_neg_scores.contiguous()

    # Fast path for standalone
    if world_size == 1:
        P = local_pos_scores.shape[0]
        out = torch.empty(P, dtype=torch.long, device=device)
        if P > 0:
            peer_ptrs = torch.tensor([out.data_ptr()], dtype=torch.long, device=device)
            _get_ext().compute_and_scatter_rankings(
                local_pos_scores, local_neg_scores, peer_ptrs, 0, 1
            )
        return out

    ext = _get_ext()
    P = local_pos_scores.shape[0]

    # 1. Pipeline start: Broadcast row counts using UVA over Symmetric memory
    size_buf, size_hdl, size_ptrs_tensor = _get_size_symm_state(world_size, device, group)
    size_hdl.barrier(channel=0)
    
    ext.uva_write_sizes(P, size_ptrs_tensor, rank, world_size)
    size_hdl.barrier(channel=1)

    # Convert implicitly awaits the stream; evaluates global index configuration
    sizes = size_buf.tolist()
    total_size = sum(sizes)
    offset = sum(sizes[:rank])

    # 2. Main computation + UVA Broadcast Scatter over pre-calculated buffers
    data_buf, data_hdl, data_ptrs_tensor = _get_data_symm_state(total_size, device, group)
    data_hdl.barrier(channel=0)
    
    ext.compute_and_scatter_rankings(
        local_pos_scores, local_neg_scores, data_ptrs_tensor, offset, world_size
    )
    data_hdl.barrier(channel=1)

    # Release clone isolating the symmetrical cached buffer from destructive mutation
    return data_buf.clone()