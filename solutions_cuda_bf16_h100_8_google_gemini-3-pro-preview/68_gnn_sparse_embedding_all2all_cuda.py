# Strategy:
# - Use PyTorch's native `bincount` and `argsort` for fast local partition calculation and tensor packing.
# - Overlap the metadata exchange (`all_gather_into_tensor` of `send_splits`) on a CUDA side-stream to hide communication latency behind the sorting/packing compute.
# - Maintain automatically cached and dynamically resized `torch.distributed._symmetric_memory` buffers, eliminating `rendezvous` overhead after warmup.
# - Launch custom unified CUDA kernels to push the packed indices and multi-dimensional values directly into peer receive buffers via symmetric memory UVA pointers, yielding maximum NVLink coalesced bandwidth.
# - Use device-side execution barriers (`hdl.barrier()`) for zero-overhead synchronization before and after the direct P2P writes.

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

template<typename T_idx>
__global__ void push_idx_kernel(
    const T_idx* __restrict__ send_idx,
    const int64_t* __restrict__ ptrs_idx,
    const int64_t* __restrict__ send_offsets,
    const int64_t* __restrict__ peer_recv_offsets,
    int world_size
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = send_offsets[world_size];
    if (tid >= total_elements) return;

    int peer = 0;
    while (peer < world_size && tid >= send_offsets[peer + 1]) {
        peer++;
    }

    int64_t local_offset = tid - send_offsets[peer];
    int64_t remote_offset = peer_recv_offsets[peer] + local_offset;

    T_idx* remote_idx = reinterpret_cast<T_idx*>(ptrs_idx[peer]);
    remote_idx[remote_offset] = send_idx[tid];
}

template<typename T_val>
__global__ void push_val_kernel(
    const T_val* __restrict__ send_val,
    const int64_t* __restrict__ ptrs_val,
    const int64_t* __restrict__ send_offsets,
    const int64_t* __restrict__ peer_recv_offsets,
    int64_t D,
    int world_size
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = send_offsets[world_size];
    int64_t total_val_elements = total_elements * D;
    if (tid >= total_val_elements) return;

    int64_t item_idx = tid / D;
    int64_t d_idx = tid % D;

    int peer = 0;
    while (peer < world_size && item_idx >= send_offsets[peer + 1]) {
        peer++;
    }

    int64_t local_offset = item_idx - send_offsets[peer];
    int64_t remote_item_offset = peer_recv_offsets[peer] + local_offset;
    
    int64_t remote_flat_offset = remote_item_offset * D + d_idx;

    T_val* remote_val = reinterpret_cast<T_val*>(ptrs_val[peer]);
    remote_val[remote_flat_offset] = send_val[tid];
}

void launch_push(
    torch::Tensor send_idx,
    torch::Tensor send_val,
    torch::Tensor ptrs_idx,
    torch::Tensor ptrs_val,
    torch::Tensor send_offsets,
    torch::Tensor peer_recv_offsets,
    int world_size
) {
    int64_t total_elements = send_idx.numel();
    if (total_elements == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int blocks_idx = (total_elements + threads - 1) / threads;

    if (send_idx.scalar_type() == torch::kInt64) {
        push_idx_kernel<int64_t><<<blocks_idx, threads, 0, stream>>>(
            send_idx.data_ptr<int64_t>(), ptrs_idx.data_ptr<int64_t>(),
            send_offsets.data_ptr<int64_t>(), peer_recv_offsets.data_ptr<int64_t>(), world_size);
    } else if (send_idx.scalar_type() == torch::kInt32) {
        push_idx_kernel<int32_t><<<blocks_idx, threads, 0, stream>>>(
            send_idx.data_ptr<int32_t>(), ptrs_idx.data_ptr<int64_t>(),
            send_offsets.data_ptr<int64_t>(), peer_recv_offsets.data_ptr<int64_t>(), world_size);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for idx");
    }

    int64_t total_val_elements = send_val.numel();
    int64_t D = total_val_elements / total_elements;
    int blocks_val = (total_val_elements + threads - 1) / threads;

    int elem_size = send_val.element_size();
    if (elem_size == 2) {
        push_val_kernel<uint16_t><<<blocks_val, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(send_val.data_ptr()),
            ptrs_val.data_ptr<int64_t>(), send_offsets.data_ptr<int64_t>(),
            peer_recv_offsets.data_ptr<int64_t>(), D, world_size);
    } else if (elem_size == 4) {
        push_val_kernel<uint32_t><<<blocks_val, threads, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(send_val.data_ptr()),
            ptrs_val.data_ptr<int64_t>(), send_offsets.data_ptr<int64_t>(),
            peer_recv_offsets.data_ptr<int64_t>(), D, world_size);
    } else if (elem_size == 8) {
        push_val_kernel<uint64_t><<<blocks_val, threads, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(send_val.data_ptr()),
            ptrs_val.data_ptr<int64_t>(), send_offsets.data_ptr<int64_t>(),
            peer_recv_offsets.data_ptr<int64_t>(), D, world_size);
    } else if (elem_size == 1) {
        push_val_kernel<uint8_t><<<blocks_val, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(send_val.data_ptr()),
            ptrs_val.data_ptr<int64_t>(), send_offsets.data_ptr<int64_t>(),
            peer_recv_offsets.data_ptr<int64_t>(), D, world_size);
    } else {
        TORCH_CHECK(false, "Unsupported element size for values");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_push", &launch_push, "UVA Custom Push Kernel for AllToAll");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dgl_sparse_push_uva_ext", CUDA_SRC)
    return _ext

_current_capacities = None
_current_D = None
_symm_cache = {}

def get_symm_buffers(recv_counts: torch.Tensor, value_shape_tail: tuple, dtype_idx: torch.dtype, dtype_val: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _current_capacities, _current_D, _symm_cache
    
    world_size = dist.get_world_size(group)
    D = 1
    for d in value_shape_tail:
        D *= d
        
    if _current_capacities is None:
        _current_capacities = torch.zeros(world_size, dtype=torch.long, device=device)
        _symm_cache = {}
        _current_D = D
        
    needs_realloc_tensor = (recv_counts > _current_capacities).any()
    needs_realloc = needs_realloc_tensor.item() or (D != _current_D)
    
    if needs_realloc:
        new_caps = torch.max(_current_capacities, (recv_counts.float() * 1.2).long())
        new_caps = torch.max(new_caps, torch.tensor(1024, dtype=torch.long, device=device))
        
        _current_capacities.copy_(new_caps)
        _current_D = D
        
        my_cap = _current_capacities[dist.get_rank(group)].item()
        
        buf_idx = symm_mem.empty(my_cap, dtype=dtype_idx, device=device)
        hdl_idx = symm_mem.rendezvous(buf_idx, group)
        
        buf_val = symm_mem.empty(my_cap * D, dtype=dtype_val, device=device)
        hdl_val = symm_mem.rendezvous(buf_val, group)
        
        _symm_cache['idx'] = (buf_idx, hdl_idx, torch.tensor(hdl_idx.buffer_ptrs, dtype=torch.int64, device=device))
        _symm_cache['val'] = (buf_val, hdl_val, torch.tensor(hdl_val.buffer_ptrs, dtype=torch.int64, device=device))
        
    return _symm_cache['idx'], _symm_cache['val']

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

    rank = dist.get_rank(group)
    
    # Pre-compile/load kernel cache, sync globally to ensure it's loaded securely
    if rank == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()

    # Calculate partitioned sizes by target bucket
    owner = (idx % world_size).long()
    send_splits = torch.bincount(owner, minlength=world_size)
    
    # Allocate a side stream for communication latency-hiding
    gather_stream = torch.cuda.Stream()
    gather_stream.wait_stream(torch.cuda.current_stream())
    
    all_send_splits_flat = torch.empty(world_size * world_size, dtype=torch.long, device=idx.device)
    
    # Kick off metadata all-gather asynchronously
    with torch.cuda.stream(gather_stream):
        dist.all_gather_into_tensor(all_send_splits_flat, send_splits, group=group)
        
    # Simultaneously locally pack index/value via sort while peers metadata propagates
    perm = torch.argsort(owner, stable=True)
    send_idx = idx[perm]
    send_value = value[perm]
    
    # Wait for completion of parallel all-gather
    torch.cuda.current_stream().wait_stream(gather_stream)
    
    all_send_splits = all_send_splits_flat.view(world_size, world_size)
    recv_counts = all_send_splits.sum(dim=0)
    my_recv_count = recv_counts[rank].item()
    
    # Rendezvous fast-path via symm_mem cache bounds
    idx_res, val_res = get_symm_buffers(recv_counts, value.shape[1:], idx.dtype, value.dtype, idx.device, group)
    buf_idx, hdl_idx, ptrs_idx = idx_res
    buf_val, hdl_val, ptrs_val = val_res
    
    peer_recv_offsets = all_send_splits[:rank, :].sum(dim=0)
    send_offsets = torch.empty(world_size + 1, dtype=torch.long, device=idx.device)
    send_offsets[0] = 0
    torch.cumsum(send_splits, dim=0, out=send_offsets[1:])
    
    # Device-side sync: wait for previous step's peer reads to conclude before overriding buffers
    hdl_idx.barrier(channel=0)
    hdl_val.barrier(channel=0)
    
    # Unified custom push logic - coalesced writes target device-side remote pools natively
    _get_ext().launch_push(
        send_idx, send_value, ptrs_idx, ptrs_val, 
        send_offsets, peer_recv_offsets, world_size
    )
    
    # Device-side sync: enforce flush and wait for incoming peer pushes to land
    hdl_idx.barrier(channel=1)
    hdl_val.barrier(channel=1)
    
    D = 1
    for d in value.shape[1:]:
        D *= d
        
    out_idx = buf_idx[:my_recv_count].clone()
    out_val = buf_val[:my_recv_count * D].view(my_recv_count, *value.shape[1:]).clone()
    
    return out_idx, out_val