"""
Optimized TorchRec KeyedJaggedTensor AllToAll using symmetric memory and fused UVA pull.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void copy_segments_kernel(
    const int64_t* __restrict__ dst_offsets,
    const int64_t* __restrict__ src_offsets,
    const int32_t* __restrict__ src_ranks,
    const int64_t* __restrict__ ptrs,
    T* __restrict__ dst,
    int64_t total_elements,
    int num_segments
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Find the segment this element belongs to
    int low = 0, high = num_segments - 1;
    int seg = 0;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (dst_offsets[mid] <= idx) {
            seg = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    
    int64_t offset_in_seg = idx - dst_offsets[seg];
    int32_t rank = src_ranks[seg];
    int64_t src_idx = src_offsets[seg] + offset_in_seg;
    
    const T* src_ptr = reinterpret_cast<const T*>(ptrs[rank]);
    dst[idx] = src_ptr[src_idx];
}

void copy_segments(
    torch::Tensor dst_offsets,
    torch::Tensor src_offsets,
    torch::Tensor src_ranks,
    torch::Tensor ptrs,
    torch::Tensor dst,
    int64_t total_elements,
    int num_segments,
    int elem_size
) {
    if (total_elements == 0) return;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    if (elem_size == 4) {
        copy_segments_kernel<int32_t><<<blocks, threads, 0, stream>>>(
            dst_offsets.data_ptr<int64_t>(),
            src_offsets.data_ptr<int64_t>(),
            src_ranks.data_ptr<int32_t>(),
            ptrs.data_ptr<int64_t>(),
            reinterpret_cast<int32_t*>(dst.data_ptr()),
            total_elements, num_segments
        );
    } else if (elem_size == 2) {
        copy_segments_kernel<int16_t><<<blocks, threads, 0, stream>>>(
            dst_offsets.data_ptr<int64_t>(),
            src_offsets.data_ptr<int64_t>(),
            src_ranks.data_ptr<int32_t>(),
            ptrs.data_ptr<int64_t>(),
            reinterpret_cast<int16_t*>(dst.data_ptr()),
            total_elements, num_segments
        );
    } else if (elem_size == 8) {
        copy_segments_kernel<int64_t><<<blocks, threads, 0, stream>>>(
            dst_offsets.data_ptr<int64_t>(),
            src_offsets.data_ptr<int64_t>(),
            src_ranks.data_ptr<int32_t>(),
            ptrs.data_ptr<int64_t>(),
            reinterpret_cast<int64_t*>(dst.data_ptr()),
            total_elements, num_segments
        );
    } else if (elem_size == 1) {
        copy_segments_kernel<int8_t><<<blocks, threads, 0, stream>>>(
            dst_offsets.data_ptr<int64_t>(),
            src_offsets.data_ptr<int64_t>(),
            src_ranks.data_ptr<int32_t>(),
            ptrs.data_ptr<int64_t>(),
            reinterpret_cast<int8_t*>(dst.data_ptr()),
            total_elements, num_segments
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_segments", &copy_segments, "Direct chunked UVA segment copy over NVLink");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("kjt_uva_all2all", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(name: str, min_size: int, dtype: torch.dtype, device: torch.device, pg: dist.ProcessGroup):
    global _symm_cache
    state = _symm_cache.get(name)
    if state is None or state["size"] < min_size:
        new_size = max(int(min_size * 1.25), 1024)
        buf = symm_mem.empty(new_size, dtype=dtype, device=device)
        hdl = symm_mem.rendezvous(buf, pg)
        state = {"size": new_size, "buf": buf, "hdl": hdl}
        _symm_cache[name] = state
    return state["buf"], state["hdl"]

def _get_recat(
    local_split: int,
    num_splits: int,
    stagger: int = 1,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    if local_split == 0:
        return None
    feature_order = [
        x + num_splits // stagger * y
        for x in range(num_splits // stagger)
        for y in range(stagger)
    ]
    recat = [
        feature_idx + rank_idx * local_split
        for feature_idx in range(local_split)
        for rank_idx in feature_order
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
    W = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    device = lengths.device
    
    if rank == 0:
        _get_ext()
    dist.barrier(pg)
    
    variable_stride = stride_per_key is not None
    num_features = sum(key_splits)
    if not variable_stride:
        stride_per_key = [batch_size] * num_features
        
    stride_tensor = torch.tensor(stride_per_key, dtype=torch.int32, device=device)
    
    # Accelerated segments length collection natively mapped
    if lengths.numel() > 0:
        try:
            length_per_key_tensor = torch.segment_reduce(
                lengths.to(torch.float32) if lengths.dtype in (torch.float16, torch.bfloat16) else lengths,
                reduce="sum",
                lengths=stride_tensor,
                unsafe=True
            ).to(torch.int32)
        except Exception:
            offset = 0
            res_lens = []
            for stride in stride_per_key:
                res_lens.append(int(lengths[offset : offset + stride].sum().item()))
                offset += stride
            length_per_key_tensor = torch.tensor(res_lens, dtype=torch.int32, device=device)
    else:
        length_per_key_tensor = torch.zeros_like(stride_tensor)

    # 1. Collective buffer size sync (fast, tiny elements)
    len_sz = lengths.numel()
    val_sz = values.numel()
    wt_sz = weights.numel() if weights is not None else 0
    local_sz = torch.tensor([len_sz, val_sz, wt_sz], dtype=torch.int64, device=device)
    dist.all_reduce(local_sz, op=dist.ReduceOp.MAX, group=pg)
    
    buf_len, hdl_len = _get_symm_state('lengths', local_sz[0].item(), lengths.dtype, device, pg)
    buf_val, hdl_val = _get_symm_state('values', local_sz[1].item(), values.dtype, device, pg)
    if wt_sz > 0:
        buf_wt, hdl_wt = _get_symm_state('weights', local_sz[2].item(), weights.dtype, device, pg)

    # 2. Asynchronous Overlap: Exchanging meta structure while loading symmetric memory
    meta_tensor = torch.cat([stride_tensor, length_per_key_tensor])
    gathered_meta = torch.empty((W, meta_tensor.numel()), dtype=torch.int32, device=device)
    work = dist.all_gather_into_tensor(gathered_meta, meta_tensor, group=pg, async_op=True)
    
    buf_len[:len_sz].copy_(lengths)
    buf_val[:val_sz].copy_(values)
    if wt_sz > 0:
        buf_wt[:wt_sz].copy_(weights)
        
    work.wait()
    hdl_len.barrier(channel=0)  # Blocks CPU until peers complete D2D local payload copies
    
    local_split = key_splits[rank]
    if local_split == 0:
        # Edge case: No data required by current rank, skip executions but fulfill API signature
        out_lengths = torch.empty(0, dtype=lengths.dtype, device=device)
        out_values = torch.empty(0, dtype=values.dtype, device=device)
        res = {"lengths": out_lengths, "values": out_values}
        if variable_stride:
            res["stride_per_key_per_rank"] = torch.empty((W, 0), dtype=torch.int64, device=device).T
        else:
            res["stride"] = torch.tensor(W * batch_size, device=device)
            res["stride_per_rank"] = torch.tensor([batch_size]*W, device=device)
        if wt_sz > 0:
            res["weights"] = torch.empty(0, dtype=weights.dtype, device=device)
        return res
        
    # 3. Vectorized pre-computation: We determine both permutation & destination positioning
    stride_matrix = gathered_meta[:, :num_features]
    length_matrix = gathered_meta[:, num_features:]
    
    f_start = sum(key_splits[:rank])
    f_end = f_start + local_split
    
    src_len_offsets_all = torch.cumsum(stride_matrix.to(torch.int64), dim=1) - stride_matrix
    src_val_offsets_all = torch.cumsum(length_matrix.to(torch.int64), dim=1) - length_matrix
    
    src_len_seg = src_len_offsets_all[:, f_start:f_end].flatten()
    src_val_seg = src_val_offsets_all[:, f_start:f_end].flatten()
    
    len_size_seg = stride_matrix[:, f_start:f_end].flatten()
    val_size_seg = length_matrix[:, f_start:f_end].flatten()
    
    src_rank_seg = torch.arange(W, device=device, dtype=torch.int32).view(W, 1).expand(W, local_split).flatten()
    
    # 4. Integrate ordering and layout offsets avoiding intermediate permutations entirely
    r_idx = _get_recat(local_split, W, stagger, device).long()
    
    out_len_size = len_size_seg[r_idx]
    out_val_size = val_size_seg[r_idx]
    out_src_len_offset = src_len_seg[r_idx]
    out_src_val_offset = src_val_seg[r_idx]
    out_src_rank = src_rank_seg[r_idx]
    
    out_dst_len_offset = torch.cumsum(out_len_size.to(torch.int64), dim=0) - out_len_size
    out_dst_val_offset = torch.cumsum(out_val_size.to(torch.int64), dim=0) - out_val_size
    
    total_len = out_len_size.sum().item()
    total_val = out_val_size.sum().item()
    num_segments = W * local_split
    
    out_lengths = torch.empty(total_len, dtype=lengths.dtype, device=device)
    out_values = torch.empty(total_val, dtype=values.dtype, device=device)
    out_weights = torch.empty(total_val, dtype=weights.dtype, device=device) if wt_sz > 0 else None
    
    # 5. Direct Fused Pull (Direct NVLink memory reads from remote locations -> Local Output Location)
    ext = _get_ext()
    ptrs_len = torch.tensor(hdl_len.buffer_ptrs, dtype=torch.int64, device=device)
    ext.copy_segments(
        out_dst_len_offset, out_src_len_offset, out_src_rank, ptrs_len,
        out_lengths, total_len, num_segments, lengths.element_size()
    )
    
    ptrs_val = torch.tensor(hdl_val.buffer_ptrs, dtype=torch.int64, device=device)
    ext.copy_segments(
        out_dst_val_offset, out_src_val_offset, out_src_rank, ptrs_val,
        out_values, total_val, num_segments, values.element_size()
    )
    
    if wt_sz > 0:
        ptrs_wt = torch.tensor(hdl_wt.buffer_ptrs, dtype=torch.int64, device=device)
        ext.copy_segments(
            out_dst_val_offset, out_src_val_offset, out_src_rank, ptrs_wt,
            out_weights, total_val, num_segments, weights.element_size()
        )
        
    # Reassemble mapping components corresponding to PyTorch expected signature dict structure
    if variable_stride:
        stride_per_key_per_rank = len_size_seg.view(W, local_split).T
        if stagger > 1:
            order = torch.arange(W, device=device).view(stagger, -1).T.reshape(-1)
            stride_per_key_per_rank = stride_per_key_per_rank[:, order]
        res = {
            "lengths": out_lengths,
            "values": out_values,
            "stride_per_key_per_rank": stride_per_key_per_rank,
        }
    else:
        res = {
            "lengths": out_lengths,
            "values": out_values,
            "stride": torch.tensor(W * batch_size, device=device),
            "stride_per_rank": torch.tensor([batch_size] * W, device=device)
        }
        
    if wt_sz > 0:
        res["weights"] = out_weights
        
    return res