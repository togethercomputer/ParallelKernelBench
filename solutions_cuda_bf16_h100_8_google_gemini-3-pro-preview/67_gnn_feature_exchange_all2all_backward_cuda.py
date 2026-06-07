from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void pull_scatter_add_bf16_kernel(
    const int64_t* __restrict__ ptrs,
    const int32_t* __restrict__ recv_offsets,
    const int32_t* __restrict__ remote_offsets,
    const int32_t* __restrict__ peers,
    const int64_t* __restrict__ seed_inverse_ids,
    __nv_bfloat16* __restrict__ grad_input,
    int total_recv,
    int H,
    int world_size
) {
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= total_recv) return;

    int chunk_i = 0;
    for (int i = 1; i < world_size; i++) {
        if (idx >= recv_offsets[i]) {
            chunk_i = i;
        }
    }

    int offset_in_chunk = idx - recv_offsets[chunk_i];
    int remote_idx = remote_offsets[chunk_i] + offset_in_chunk;
    int peer = peers[chunk_i];

    // Establish mapping to symmetric peer memory
    const __nv_bfloat16* remote_row = (const __nv_bfloat16*)ptrs[peer] + remote_idx * H;
    int dst_row = seed_inverse_ids[idx];
    __nv_bfloat16* dst_ptr = grad_input + dst_row * H;

    // Vectorized path for aligned even-dimension counts (doubles throughput)
    if (H % 2 == 0) {
        int h = threadIdx.x * 2;
        int stride = blockDim.x * 2;
        for (; h < H; h += stride) {
            __nv_bfloat162 val = *(__nv_bfloat162*)(remote_row + h);
            atomicAdd((__nv_bfloat162*)(dst_ptr + h), val);
        }
    } else {
        int h = threadIdx.x;
        int stride = blockDim.x;
        for (; h < H; h += stride) {
            atomicAdd(dst_ptr + h, remote_row[h]);
        }
    }
}

void launch_pull_scatter_add(
    torch::Tensor ptrs_tensor,
    torch::Tensor recv_offsets,
    torch::Tensor remote_offsets,
    torch::Tensor peers,
    torch::Tensor seed_inverse_ids,
    torch::Tensor grad_input,
    int H
) {
    int total_recv = seed_inverse_ids.size(0);
    int world_size = ptrs_tensor.size(0);
    
    dim3 block(32, 8);
    dim3 grid((total_recv + block.y - 1) / block.y);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    pull_scatter_add_bf16_kernel<<<grid, block, 0, stream>>>(
        ptrs_tensor.data_ptr<int64_t>(),
        recv_offsets.data_ptr<int32_t>(),
        remote_offsets.data_ptr<int32_t>(),
        peers.data_ptr<int32_t>(),
        seed_inverse_ids.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr<at::BFloat16>()),
        total_recv,
        H,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pull_scatter_add", &launch_pull_scatter_add, "Pull scatter add kernel");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_pull_scatter_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(max_elements, H, dtype, device, group):
    key = (H, dtype, device)
    if key in _symm_cache:
        c = _symm_cache[key]
        if c['max_elements'] >= max_elements:
            return c['buf'], c['hdl'], c['ptrs']
    
    # Pad to help resist frequent re-allocations as dynamic batches shift sizes
    alloc_elements = max(max_elements, 1024)
    alloc_elements = (alloc_elements + 1023) // 1024 * 1024
    
    buf = symm_mem.empty((alloc_elements, H), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    _symm_cache[key] = {
        'buf': buf,
        'hdl': hdl,
        'ptrs': ptrs,
        'max_elements': alloc_elements
    }
    return buf, hdl, ptrs

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
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = grad_output.device
    dtype = grad_output.dtype
    H = grad_output.shape[1]

    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    
    # Gather structural dimensions globally across all ranks
    local_counts = torch.tensor(counts_sent + counts_received, dtype=torch.int32, device=device)
    all_counts = torch.empty((world_size, len(local_counts)), dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(all_counts, local_counts, group=group)
    
    counts_sent_t = all_counts[:, :world_size]
    max_elements = counts_sent_t.sum(dim=1).max().item()
    local_sent_size = counts_sent_t[rank].sum().item()
    
    buf, hdl, ptrs = _get_symm_state(max_elements, H, dtype, device, group)
    
    # Wait strictly for reads over the last iteration to finish gracefully
    hdl.barrier(channel=1)
    
    # Expose backwards payload into UVA symmetric region
    if local_sent_size > 0:
        buf[:local_sent_size].copy_(grad_output)
        
    # Synchronization before UVA remote pulls map onto memory mapping
    hdl.barrier(channel=0)
    
    # Pure GPU-native prefix sums calculations to derive remote bounds explicitly
    counts_received_t = all_counts[:, world_size:]
    recv_offsets_local = torch.zeros(world_size + 1, dtype=torch.int32, device=device)
    recv_offsets_local[1:] = torch.cumsum(counts_received_t[rank], dim=0)

    sent_offsets = torch.zeros((world_size, world_size + 1), dtype=torch.int32, device=device)
    sent_offsets[:, 1:] = torch.cumsum(counts_sent_t, dim=1)

    i = torch.arange(world_size, dtype=torch.int32, device=device)
    peers = (rank + i) % world_size
    ks = (world_size - i) % world_size
    remote_chunk_offsets = sent_offsets[peers, ks]
    
    grad_input = torch.zeros((seed_size, H), dtype=dtype, device=device)
    total_recv = seed_inverse_ids.size(0)
    
    if total_recv > 0:
        _get_ext().launch_pull_scatter_add(
            ptrs,
            recv_offsets_local,
            remote_chunk_offsets,
            peers,
            seed_inverse_ids,
            grad_input,
            H
        )
        
    return grad_input