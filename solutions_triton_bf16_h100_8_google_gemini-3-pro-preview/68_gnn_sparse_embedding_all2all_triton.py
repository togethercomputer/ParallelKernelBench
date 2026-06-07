import math
from typing import Optional, Tuple
import numpy as np

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

struct PtrArray {
    uintptr_t ptrs[32];
};

__global__ void broadcast_metadata_kernel(
    const int64_t* __restrict__ send_splits,
    int64_t K,
    PtrArray peer_meta_ptrs,
    int rank,
    int world_size
) {
    int tid = threadIdx.x;
    if (tid <= world_size) {
        int64_t val = (tid == 0) ? K : send_splits[tid - 1];
        for (int p = 0; p < world_size; p++) {
            int64_t* dst = (int64_t*)peer_meta_ptrs.ptrs[p];
            dst[rank * (world_size + 1) + tid] = val;
        }
    }
}

__global__ void pack_idx_kernel(
    const int64_t* __restrict__ idx,
    const int64_t* __restrict__ perm,
    int64_t* __restrict__ send_idx,
    int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < K) {
        send_idx[tid] = idx[perm[tid]];
    }
}

__global__ void pack_val_kernel(
    const nv_bfloat16* __restrict__ value,
    const int64_t* __restrict__ perm,
    nv_bfloat16* __restrict__ send_val,
    int K, int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = tid / D;
    int d_idx = tid % D;
    if (k_idx < K) {
        send_val[k_idx * D + d_idx] = value[perm[k_idx] * D + d_idx];
    }
}

__global__ void pull_idx_kernel(
    int64_t* __restrict__ recv_idx,
    PtrArray peer_send_idx_ptrs,
    const int64_t* __restrict__ remote_offsets,
    const int64_t* __restrict__ local_offsets,
    int recv_count,
    int world_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= recv_count) return;

    int p = 0;
    while (p < world_size - 1 && tid >= local_offsets[p + 1]) {
        p++;
    }

    int offset_in_bucket = tid - local_offsets[p];
    int64_t remote_idx = remote_offsets[p] + offset_in_bucket;

    const int64_t* src = (const int64_t*)peer_send_idx_ptrs.ptrs[p];
    recv_idx[tid] = src[remote_idx];
}

__global__ void pull_val_kernel(
    nv_bfloat16* __restrict__ recv_val,
    PtrArray peer_send_val_ptrs,
    const int64_t* __restrict__ remote_offsets,
    const int64_t* __restrict__ local_offsets,
    int recv_count,
    int D,
    int world_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = tid / D;
    int d_idx = tid % D;
    if (k_idx >= recv_count) return;

    int p = 0;
    while (p < world_size - 1 && k_idx >= local_offsets[p + 1]) {
        p++;
    }

    int offset_in_bucket = k_idx - local_offsets[p];
    int64_t remote_idx = remote_offsets[p] + offset_in_bucket;

    const nv_bfloat16* src = (const nv_bfloat16*)peer_send_val_ptrs.ptrs[p];
    recv_val[k_idx * D + d_idx] = src[remote_idx * D + d_idx];
}

void launch_broadcast_metadata(
    torch::Tensor send_splits,
    int64_t K,
    std::vector<int64_t> ptrs,
    int rank,
    int world_size
) {
    PtrArray arr;
    for (int i = 0; i < world_size; i++) arr.ptrs[i] = (uintptr_t)ptrs[i];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    broadcast_metadata_kernel<<<1, 32, 0, stream>>>(
        send_splits.data_ptr<int64_t>(),
        K, arr, rank, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pack(
    torch::Tensor idx,
    torch::Tensor value,
    torch::Tensor perm,
    torch::Tensor send_idx,
    torch::Tensor send_val,
    int K, int D
) {
    if (K == 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    
    int blocks_idx = (K + threads - 1) / threads;
    pack_idx_kernel<<<blocks_idx, threads, 0, stream>>>(
        idx.data_ptr<int64_t>(),
        perm.data_ptr<int64_t>(),
        send_idx.data_ptr<int64_t>(),
        K
    );
    
    int total_val = K * D;
    int blocks_val = (total_val + threads - 1) / threads;
    pack_val_kernel<<<blocks_val, threads, 0, stream>>>(
        reinterpret_cast<const nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
        perm.data_ptr<int64_t>(),
        reinterpret_cast<nv_bfloat16*>(send_val.data_ptr<at::BFloat16>()),
        K, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pull(
    torch::Tensor recv_idx,
    torch::Tensor recv_val,
    std::vector<int64_t> idx_ptrs,
    std::vector<int64_t> val_ptrs,
    torch::Tensor remote_offsets,
    torch::Tensor local_offsets,
    int recv_count, int D, int world_size
) {
    if (recv_count == 0) return;
    PtrArray arr_idx;
    PtrArray arr_val;
    for (int i = 0; i < world_size; i++) {
        arr_idx.ptrs[i] = (uintptr_t)idx_ptrs[i];
        arr_val.ptrs[i] = (uintptr_t)val_ptrs[i];
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    
    int blocks_idx = (recv_count + threads - 1) / threads;
    pull_idx_kernel<<<blocks_idx, threads, 0, stream>>>(
        recv_idx.data_ptr<int64_t>(),
        arr_idx,
        remote_offsets.data_ptr<int64_t>(),
        local_offsets.data_ptr<int64_t>(),
        recv_count, world_size
    );
    
    int total_val = recv_count * D;
    int blocks_val = (total_val + threads - 1) / threads;
    pull_val_kernel<<<blocks_val, threads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(recv_val.data_ptr<at::BFloat16>()),
        arr_val,
        remote_offsets.data_ptr<int64_t>(),
        local_offsets.data_ptr<int64_t>(),
        recv_count, D, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_metadata", &launch_broadcast_metadata, "Broadcast metadata via UVA");
    m.def("pack", &launch_pack, "Pack data into symmetric buffer");
    m.def("pull", &launch_pull, "Pull data from peer symmetric buffers");
}
'''

_ext = None
def get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dgl_sparse_push_bf16_uva", CUDA_SRC)
    return _ext

class DataCache:
    def __init__(self, capacity: int, D: int, dtype_idx: torch.dtype, dtype_val: torch.dtype, device: torch.device, group):
        self.capacity = capacity
        self.D = D
        self.idx_buf = symm_mem.empty((capacity,), dtype=dtype_idx, device=device)
        self.idx_hdl = symm_mem.rendezvous(self.idx_buf, group)
        self.idx_ptrs = [int(p) for p in self.idx_hdl.buffer_ptrs]
        
        self.val_buf = symm_mem.empty((capacity, D), dtype=dtype_val, device=device)
        self.val_hdl = symm_mem.rendezvous(self.val_buf, group)
        self.val_ptrs = [int(p) for p in self.val_hdl.buffer_ptrs]

_symm_meta_buf = None
_symm_meta_hdl = None
_data_cache = None


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

    ext = get_ext()
    rank = dist.get_rank(group)
    
    idx = idx.contiguous()
    value = value.contiguous()
    
    K = idx.numel()
    D = math.prod(value.shape[1:]) if value.ndim > 1 else 1

    global _symm_meta_buf, _symm_meta_hdl, _data_cache

    if _symm_meta_buf is None:
        _symm_meta_buf = symm_mem.empty((world_size * (world_size + 1),), dtype=torch.int64, device=idx.device)
        _symm_meta_hdl = symm_mem.rendezvous(_symm_meta_buf, group)

    # 1. Bucket local updates explicitly per rank.
    owner = (idx % world_size).long()
    send_splits = torch.bincount(owner, minlength=world_size)
    perm = torch.argsort(owner, stable=True).long()

    # 2. UVA Scatter for sending dynamic sizing arrays.
    meta_ptrs = [int(p) for p in _symm_meta_hdl.buffer_ptrs]
    ext.broadcast_metadata(send_splits, K, meta_ptrs, rank, world_size)
    _symm_meta_hdl.barrier(channel=0)

    # Decode payload
    meta_cpu = _symm_meta_buf.cpu().numpy()
    all_K = meta_cpu[0 :: world_size + 1]
    max_K = int(all_K.max())

    all_splits = np.empty((world_size, world_size), dtype=np.int64)
    for r in range(world_size):
        all_splits[r] = meta_cpu[r * (world_size + 1) + 1 : (r + 1) * (world_size + 1)]

    # 3. Synchronized cache reallocation
    capacity = _data_cache.capacity if _data_cache is not None else 0
    if _data_cache is None or max_K > capacity or _data_cache.D != D:
        if _data_cache is not None:
            # Sync explicitly to block drops while any peer might be asynchronously reading
            _symm_meta_hdl.barrier(channel=0) 
        
        new_cap = max(capacity, max_K)
        if _data_cache is not None and max_K > capacity:
            new_cap = max(max_K, capacity * 2)
        new_cap = max(new_cap, 1024)
        
        _data_cache = DataCache(new_cap, D, idx.dtype, value.dtype, idx.device, group)

    # 4. Pack directly onto symmetric sender cache without PyTorch materialization 
    ext.pack(idx, value, perm, _data_cache.idx_buf, _data_cache.val_buf, K, D)

    # 5. Compute fetch instructions
    my_recv_splits = all_splits[:, rank]
    recv_count = int(my_recv_splits.sum())

    remote_offsets = np.zeros((world_size,), dtype=np.int64)
    local_offsets = np.zeros((world_size,), dtype=np.int64)

    for p in range(world_size):
        remote_offsets[p] = all_splits[p, :rank].sum()
        local_offsets[p] = all_splits[:p, rank].sum()

    remote_offsets_t = torch.tensor(remote_offsets, dtype=torch.int64, device=idx.device)
    local_offsets_t = torch.tensor(local_offsets, dtype=torch.int64, device=idx.device)

    # 6. Allocate isolated destination variables
    recv_idx = torch.empty((recv_count,), dtype=idx.dtype, device=idx.device)
    recv_value = torch.empty((recv_count, *value.shape[1:]), dtype=value.dtype, device=value.device)

    # Final sync indicating buffers populated
    _symm_meta_hdl.barrier(channel=0)

    # 7. Dispersed PULL: NVLink-accelerated fetch operation direct from peers
    if recv_count > 0:
        ext.pull(
            recv_idx, recv_value,
            _data_cache.idx_ptrs, _data_cache.val_ptrs,
            remote_offsets_t, local_offsets_t,
            recv_count, D, world_size
        )

    return recv_idx, recv_value