from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void fused_pull_permute_kernel(
    const uint64_t* __restrict__ remote_ptrs,
    const int32_t* __restrict__ permuted_offsets,
    const int32_t* __restrict__ remote_ranks,
    const int32_t* __restrict__ remote_offsets,
    int N,
    int total_elements,
    int element_size,
    void* __restrict__ dest_buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = tid; i < total_elements; i += stride) {
        // Binary search to find which segment this element belongs to
        int L = 0, R = N - 1;
        int idx = 0;
        while (L <= R) {
            int mid = L + (R - L) / 2;
            if (permuted_offsets[mid] <= i) {
                idx = mid;
                L = mid + 1;
            } else {
                R = mid - 1;
            }
        }
        
        int offset_in_segment = i - permuted_offsets[idx];
        int rank = remote_ranks[idx];
        int r_offset = remote_offsets[idx] + offset_in_segment;
        
        const char* src_base = reinterpret_cast<const char*>(remote_ptrs[rank]);
        char* dst_base = reinterpret_cast<char*>(dest_buffer);
        
        // Use naturally aligned memory loads for efficient NVLink transfers
        if (element_size == 4) {
            reinterpret_cast<int32_t*>(dst_base)[i] = reinterpret_cast<const int32_t*>(src_base)[r_offset];
        } else if (element_size == 2) {
            reinterpret_cast<int16_t*>(dst_base)[i] = reinterpret_cast<const int16_t*>(src_base)[r_offset];
        } else if (element_size == 8) {
            reinterpret_cast<int64_t*>(dst_base)[i] = reinterpret_cast<const int64_t*>(src_base)[r_offset];
        } else if (element_size == 1) {
            reinterpret_cast<int8_t*>(dst_base)[i] = reinterpret_cast<const int8_t*>(src_base)[r_offset];
        }
    }
}

void fused_pull_permute(
    torch::Tensor remote_ptrs,
    torch::Tensor permuted_offsets,
    torch::Tensor remote_ranks,
    torch::Tensor remote_offsets,
    int N,
    int total_elements,
    int element_size,
    torch::Tensor dest_buffer
) {
    if (total_elements == 0 || N == 0) return;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; // Limit blocks for grid-stride
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    fused_pull_permute_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(remote_ptrs.data_ptr<int64_t>()),
        permuted_offsets.data_ptr<int32_t>(),
        remote_ranks.data_ptr<int32_t>(),
        remote_offsets.data_ptr<int32_t>(),
        N,
        total_elements,
        element_size,
        dest_buffer.data_ptr()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_pull_permute", &fused_pull_permute, "Fused P2P pull and permute");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_p2p_permute_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_buffer(name, min_size, dtype, device):
    global _symm_cache
    if name in _symm_cache:
        buf, hdl, ptrs = _symm_cache[name]
        if buf.numel() >= min_size and buf.dtype == dtype:
            return buf, hdl, ptrs
    
    alloc_size = min_size + 1024
    buf = symm_mem.empty(alloc_size, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    _symm_cache[name] = (buf, hdl, ptrs)
    return buf, hdl, ptrs

def _lengths_per_key_vectorized(lengths: torch.Tensor, stride_per_key: List[int]) -> torch.Tensor:
    N = len(stride_per_key)
    if N == 0:
        return torch.empty(0, dtype=lengths.dtype, device=lengths.device)
    strides_tensor = torch.tensor(stride_per_key, dtype=torch.long, device=lengths.device)
    indices = torch.repeat_interleave(
        torch.arange(N, device=lengths.device, dtype=torch.long),
        strides_tensor
    )
    res = torch.zeros(N, dtype=lengths.dtype, device=lengths.device)
    res.scatter_add_(0, indices, lengths)
    return res

def _sum_by_splits(values: List[int], splits: List[int]) -> List[int]:
    out: List[int] = []
    offset = 0
    for split in splits:
        out.append(sum(values[offset : offset + split]))
        offset += split
    return out

def _get_recat(
    local_split: int,
    num_splits: int,
    stagger: int = 1,
    device: Optional[torch.device] = None,
    batch_size_per_rank: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    if local_split == 0:
        return None
    feature_order = [
        x + num_splits // stagger * y
        for x in range(num_splits // stagger)
        for y in range(stagger)
    ]
    if batch_size_per_rank is None:
        recat = [
            feature_idx + rank_idx * local_split
            for feature_idx in range(local_split)
            for rank_idx in feature_order
        ]
    else:
        rank_offsets = [0]
        for batch_size in batch_size_per_rank[:-1]:
            rank_offsets.append(rank_offsets[-1] + local_split * batch_size)
        recat = [
            rank_offsets[rank_idx] + feature_idx * batch_size_per_rank[rank_idx] + b
            for feature_idx in range(local_split)
            for rank_idx in feature_order
            for b in range(batch_size_per_rank[rank_idx])
        ]
    return torch.tensor(recat, device=device, dtype=torch.int32)

@torch.no_grad()
def solution(
    lengths: torch.Tensor,
    values: torch.Tensor,
    key_splits: List[int],
    batch_size: int,
    pg: Optional[dist.ProcessGroup] = None,
    weights: Optional[torch.Tensor] = None,
    stride_per_key: Optional[List[int]] = None,
    stagger: int = 1,
) -> Dict[str, torch.Tensor]:
    pg = pg or dist.group.WORLD
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    device = lengths.device

    if rank == 0:
        _get_ext()
    dist.barrier(group=pg)
    
    num_features = sum(key_splits)
    variable_stride = stride_per_key is not None
    if stride_per_key is None:
        stride_per_key = [batch_size] * num_features

    # Completely pure-CUDA metadata preparation to bypass host-device syncs
    length_per_key_tensor = _lengths_per_key_vectorized(lengths, stride_per_key)
    
    key_splits_tensor = torch.tensor(key_splits, dtype=torch.long, device=device)
    indices = torch.repeat_interleave(
        torch.arange(world_size, device=device, dtype=torch.long),
        key_splits_tensor
    )
    value_splits_tensor = torch.zeros(world_size, dtype=lengths.dtype, device=device)
    value_splits_tensor.scatter_add_(0, indices, length_per_key_tensor)

    length_splits = _sum_by_splits(stride_per_key, key_splits)
    length_splits_tensor = torch.tensor(length_splits, dtype=torch.int32, device=device)

    split_tensors = [length_splits_tensor, value_splits_tensor.to(torch.int32)]
    if variable_stride:
        split_tensors.append(key_splits_tensor.to(torch.int32))
    if weights is not None:
        split_tensors.append(value_splits_tensor.to(torch.int32))
    if not variable_stride:
        split_tensors.append(torch.full((world_size,), batch_size, dtype=torch.int32, device=device))

    num_tensors = len(split_tensors)
    meta_local = torch.stack(split_tensors, dim=1).flatten()
    meta_all_flat = torch.empty(world_size, meta_local.numel(), dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(meta_all_flat, meta_local, group=pg)
    
    # meta_all[S, D, T] = Size sent from rank S to rank D for tensor T
    meta_all = meta_all_flat.view(world_size, world_size, num_tensors)

    input_tensors = [lengths, values]
    tensor_names = ["lengths", "values"]
    if variable_stride:
        input_tensors.append(torch.tensor(stride_per_key, dtype=torch.int32, device=device))
        tensor_names.append("strides")
    if weights is not None:
        input_tensors.append(weights)
        tensor_names.append("weights")

    symm_ptrs = []
    for T, (name, tensor) in enumerate(zip(tensor_names, input_tensors)):
        max_send_size = meta_all[:, :, T].sum(dim=1).max().item()
        buf, hdl, ptrs = _get_symm_buffer(name, max_send_size, tensor.dtype, device)
        
        local_size = meta_all[rank, :, T].sum().item()
        if local_size > 0:
            buf[:local_size].copy_(tensor.view(-1))
        symm_ptrs.append(ptrs)

    # Barrier before P2P reading symmetric memory buffers
    dist.barrier(group=pg)

    def pull_and_permute(T: int, seg_sizes: torch.Tensor, recat: Optional[torch.Tensor], dtype: torch.dtype):
        recv_chunk_sizes = meta_all[:, rank, T]
        send_offsets_to_D = meta_all[:, :rank, T].sum(dim=1)
        
        N = seg_sizes.numel()
        total_elements = seg_sizes.sum().item()
        dest_buffer = torch.empty(total_elements, dtype=dtype, device=device)
        if total_elements == 0:
            return dest_buffer
            
        chunk_offsets = torch.zeros(world_size + 1, dtype=torch.int32, device=device)
        chunk_offsets[1:] = torch.cumsum(recv_chunk_sizes, dim=0)
        
        unpermuted_offsets = torch.zeros(N, dtype=torch.int32, device=device)
        if N > 1:
            unpermuted_offsets[1:] = torch.cumsum(seg_sizes[:-1], dim=0)
            
        remote_ranks = torch.bucketize(unpermuted_offsets, chunk_offsets[1:], right=True).to(torch.int32)
        chunk_offset = unpermuted_offsets - chunk_offsets[remote_ranks]
        remote_offsets = send_offsets_to_D[remote_ranks] + chunk_offset
        
        if recat is not None:
            remote_ranks_permuted = remote_ranks[recat].contiguous()
            remote_offsets_permuted = remote_offsets[recat].contiguous()
            seg_sizes_permuted = seg_sizes[recat]
        else:
            remote_ranks_permuted = remote_ranks.contiguous()
            remote_offsets_permuted = remote_offsets.contiguous()
            seg_sizes_permuted = seg_sizes
            
        permuted_offsets = torch.zeros(N + 1, dtype=torch.int32, device=device)
        permuted_offsets[1:] = torch.cumsum(seg_sizes_permuted, dim=0)
        
        _get_ext().fused_pull_permute(
            symm_ptrs[T], permuted_offsets, remote_ranks_permuted, remote_offsets_permuted,
            N, total_elements, dest_buffer.element_size(), dest_buffer
        )
        return dest_buffer

    # Pull structural metadata flatly first
    active_ranks_lengths = (meta_all[:, rank, 0] > 0).nonzero(as_tuple=True)[0]
    recv_lengths_unpermuted = pull_and_permute(
        0, meta_all[active_ranks_lengths, rank, 0], None, lengths.dtype
    )
    
    if variable_stride:
        active_ranks_strides = (meta_all[:, rank, 2] > 0).nonzero(as_tuple=True)[0]
        recv_strides_unpermuted = pull_and_permute(
            2, meta_all[active_ranks_strides, rank, 2], None, torch.int32
        )

    # Derive segment descriptors for payloads using local metadata
    local_split = key_splits[rank]
    if variable_stride:
        recv_strides_list = recv_strides_unpermuted.tolist()
        seg_sizes_lengths = recv_strides_unpermuted
        seg_sizes_values = _lengths_per_key_vectorized(recv_lengths_unpermuted, recv_strides_list).to(torch.int32)
        recat = _get_recat(local_split, world_size, stagger, device=device)
    else:
        stride_per_rank = meta_all[:, rank, -1].tolist()
        single_batch_per_rank = all(s == stride_per_rank[0] for s in stride_per_rank)
        
        if single_batch_per_rank:
            B = stride_per_rank[0]
            if B > 0:
                N = recv_lengths_unpermuted.numel() // B
                seg_sizes_lengths = torch.full((N,), B, dtype=torch.int32, device=device)
                lengths_2d = recv_lengths_unpermuted.view(N, B)
                seg_sizes_values = lengths_2d.sum(dim=1).to(torch.int32)
            else:
                seg_sizes_lengths = torch.empty(0, dtype=torch.int32, device=device)
                seg_sizes_values = torch.empty(0, dtype=torch.int32, device=device)
            recat = _get_recat(local_split, world_size, stagger, device=device)
        else:
            N = recv_lengths_unpermuted.numel()
            seg_sizes_lengths = torch.ones(N, dtype=torch.int32, device=device)
            seg_sizes_values = recv_lengths_unpermuted.to(torch.int32)
            recat = _get_recat(local_split, world_size, stagger, device=device, batch_size_per_rank=stride_per_rank)

    # Fused Permuted P2P Copies for payload tensors
    recv_lengths = pull_and_permute(0, seg_sizes_lengths, recat, lengths.dtype)
    recv_values = pull_and_permute(1, seg_sizes_values, recat, values.dtype)
    
    recv_weights = None
    if weights is not None:
        T_weights = 3 if variable_stride else 2
        recv_weights = pull_and_permute(T_weights, seg_sizes_values, recat, weights.dtype)

    # Barrier before function return to preserve symm_mem valid lifetimes
    dist.barrier(group=pg)

    if variable_stride:
        recv_strides_permuted = recv_strides_unpermuted
        if recat is not None:
            recv_strides_permuted = recv_strides_unpermuted[recat]
        stride_per_key_per_rank = recv_strides_permuted.view(world_size, local_split).T
        if stagger > 1:
            order = torch.arange(world_size, device=device).view(stagger, -1).T.reshape(-1)
            stride_per_key_per_rank = stride_per_key_per_rank[:, order]
            
        result = {
            "lengths": recv_lengths,
            "values": recv_values,
            "stride_per_key_per_rank": stride_per_key_per_rank.to(torch.long),
        }
    else:
        result = {
            "lengths": recv_lengths,
            "values": recv_values,
            "stride": torch.tensor(sum(stride_per_rank), device=device, dtype=torch.long),
            "stride_per_rank": torch.tensor(stride_per_rank, device=device, dtype=torch.long),
        }

    if recv_weights is not None:
        result["weights"] = recv_weights
        
    return result