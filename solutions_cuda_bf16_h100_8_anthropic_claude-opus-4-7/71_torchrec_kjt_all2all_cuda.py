"""
KJTAllToAll via torch.distributed._symmetric_memory + custom CUDA UVA kernels.
Replaces dist.all_to_all_single with device-side peer reads through symm_mem
buffer pointers.
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
#include <cstdint>

// Generic per-rank chunked copy: copies `count[r]` elements from peer r at
// offset `src_off[r]` into output at offset `dst_off[r]`.
template <typename T>
__global__ void gather_from_peers_kernel(
    const uint64_t* __restrict__ peer_ptrs,
    const int64_t* __restrict__ src_off,
    const int64_t* __restrict__ dst_off,
    const int64_t* __restrict__ count,
    T* __restrict__ out,
    int world_size
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;
    int64_t n = count[peer];
    if (n <= 0) return;
    const T* src = reinterpret_cast<const T*>(peer_ptrs[peer]) + src_off[peer];
    T* dst = out + dst_off[peer];
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        dst[idx] = src[idx];
    }
}

void launch_gather_peers(
    torch::Tensor peer_ptrs,    // int64 [world_size]
    torch::Tensor src_off,      // int64 [world_size]
    torch::Tensor dst_off,      // int64 [world_size]
    torch::Tensor count,        // int64 [world_size]
    torch::Tensor out,
    int64_t elem_size,
    int world_size,
    int64_t max_count
) {
    if (max_count <= 0) return;
    int threads = 256;
    int blocks_x = (int)((max_count + threads - 1) / threads);
    if (blocks_x > 512) blocks_x = 512;
    if (blocks_x < 1) blocks_x = 1;
    dim3 grid(blocks_x, world_size, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint64_t* p_ptrs = (const uint64_t*)peer_ptrs.data_ptr<int64_t>();
    const int64_t* p_src = src_off.data_ptr<int64_t>();
    const int64_t* p_dst = dst_off.data_ptr<int64_t>();
    const int64_t* p_cnt = count.data_ptr<int64_t>();

    if (elem_size == 8) {
        gather_from_peers_kernel<int64_t><<<grid, threads, 0, stream>>>(
            p_ptrs, p_src, p_dst, p_cnt,
            (int64_t*)out.data_ptr(), world_size);
    } else if (elem_size == 4) {
        gather_from_peers_kernel<int32_t><<<grid, threads, 0, stream>>>(
            p_ptrs, p_src, p_dst, p_cnt,
            (int32_t*)out.data_ptr(), world_size);
    } else if (elem_size == 2) {
        gather_from_peers_kernel<int16_t><<<grid, threads, 0, stream>>>(
            p_ptrs, p_src, p_dst, p_cnt,
            (int16_t*)out.data_ptr(), world_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather_peers", &launch_gather_peers, "gather chunks from peers via UVA");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("kjt_a2a_uva_ext", CUDA_SRC)
    return _ext


# ---------------------------------------------------------------------------
# Symm-mem buffer cache
# ---------------------------------------------------------------------------

_buf_cache: Dict[Tuple, Tuple[torch.Tensor, object, torch.Tensor]] = {}


def _get_symm_buf(numel: int, dtype: torch.dtype, device: torch.device, tag: str):
    """Return (buf, hdl, peer_ptrs_int64). Grows by 2x as needed."""
    key = (tag, dtype)
    if key in _buf_cache:
        buf, hdl, ptrs = _buf_cache[key]
        if buf.numel() >= numel:
            return buf, hdl, ptrs
    # Allocate (or grow). Round up.
    new_size = max(numel, 1)
    # Round to power of 2 to reduce reallocs
    size = 1
    while size < new_size:
        size *= 2
    buf = symm_mem.empty(size, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    _buf_cache[key] = (buf, hdl, ptrs)
    return buf, hdl, ptrs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum_by_splits(values: List[int], splits: List[int]) -> List[int]:
    out: List[int] = []
    offset = 0
    for split in splits:
        out.append(sum(values[offset : offset + split]))
        offset += split
    return out


def _lengths_per_key(lengths: torch.Tensor, stride_per_key: List[int]) -> List[int]:
    out: List[int] = []
    offset = 0
    for stride in stride_per_key:
        out.append(int(lengths[offset : offset + stride].sum().item()))
        offset += stride
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


def _permute_segments(
    data: torch.Tensor,
    segment_lengths: torch.Tensor,
    recat: torch.Tensor,
) -> torch.Tensor:
    segment_lengths = segment_lengths.to(device=data.device, dtype=torch.long)
    offsets = torch.zeros(
        segment_lengths.numel() + 1, dtype=torch.long, device=data.device
    )
    offsets[1:] = torch.cumsum(segment_lengths, dim=0)
    recat_l = recat.long()
    chunks = []
    off_cpu = offsets.cpu().tolist()
    rec_cpu = recat_l.cpu().tolist()
    for idx in rec_cpu:
        chunks.append(data[off_cpu[idx] : off_cpu[idx + 1]])
    return torch.cat(chunks, dim=0) if chunks else data.new_empty((0,))


def _permute_2d_sparse_data(
    recat: torch.Tensor,
    lengths_2d: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    recat = recat.long()
    row_lengths = lengths_2d.sum(dim=1).to(torch.long)
    lengths_out = lengths_2d[recat]
    values_out = _permute_segments(values, row_lengths, recat)
    weights_out = None
    if weights is not None:
        weights_out = _permute_segments(weights, row_lengths, recat)
    return lengths_out, values_out, weights_out


# ---------------------------------------------------------------------------
# Symm-mem based all-to-all for a single 1D tensor
# ---------------------------------------------------------------------------

def _a2a_symm(
    tensor: torch.Tensor,
    input_splits: List[int],
    output_splits: List[int],
    tag: str,
    hdl_barrier_channel: int,
) -> torch.Tensor:
    """Variable-size all-to-all via symmetric memory + UVA peer reads."""
    device = tensor.device
    dtype = tensor.dtype
    world_size = len(input_splits)
    elem_size = tensor.element_size()

    # Stage local input into symmetric buffer (on every rank).
    # Layout: just contiguous = tensor itself; peers index by their input_split offsets,
    # which equal *our* output_split offsets only if their input layout matches.
    # Each rank stages its full contiguous "input_tensors" so peers read the chunk
    # destined for them.
    src_buf, src_hdl, src_ptrs = _get_symm_buf(tensor.numel(), dtype, device, f"src_{tag}")
    if tensor.numel() > 0:
        src_buf[: tensor.numel()].copy_(tensor)

    # Synchronize so peers see our staged data.
    src_hdl.barrier(channel=hdl_barrier_channel)

    total_out = sum(output_splits)
    out = torch.empty(total_out, dtype=dtype, device=device)

    # For each peer r, we read the chunk of length output_splits[r] starting at
    # the offset that r assigned to OUR rank in r's input_splits. We need r's
    # input_splits to find that offset. Equivalently: r's input_splits[my_rank]
    # == output_splits[r] (consistency), and the offset is sum of r's
    # input_splits[0..my_rank-1].
    #
    # We don't have peers' input_splits directly here unless we exchange them.
    # However, by symmetry: receiver's output_splits[r] tells us the size, and
    # we know peer r staged data so that the chunk for our rank starts at peer
    # r's prefix sum. We'll need that prefix sum. We exchange small split
    # metadata once via symm mem (handled by caller passing src_off_per_peer).
    raise RuntimeError("unused path")


# ---------------------------------------------------------------------------
# Bulk symm a2a: exchange multiple tensors using one set of peer src offsets
# ---------------------------------------------------------------------------

def _bulk_a2a_symm(
    tensors: List[torch.Tensor],
    input_splits_list: List[List[int]],
    output_splits_list: List[List[int]],
    peer_input_splits_list: List[List[List[int]]],  # peer_input_splits_list[t][r] = peer r's input_splits[t]
    rank: int,
    world_size: int,
    tag_prefix: str,
) -> List[torch.Tensor]:
    """
    For each tensor t:
      - Stage tensors[t] into symm buf "src_{tag_prefix}_{t}".
      - From each peer r, read length output_splits_list[t][r] starting at
        prefix_sum(peer_input_splits_list[t][r][0..my_rank-1]).
    Returns list of received concatenated tensors.
    """
    device = tensors[0].device
    ext = _get_ext()
    outputs: List[torch.Tensor] = []

    # Stage all
    src_hdls = []
    src_ptrs_list = []
    for t_idx, ten in enumerate(tensors):
        buf, hdl, ptrs = _get_symm_buf(max(ten.numel(), 1), ten.dtype, device,
                                        f"{tag_prefix}_t{t_idx}")
        if ten.numel() > 0:
            buf[: ten.numel()].copy_(ten)
        src_hdls.append(hdl)
        src_ptrs_list.append(ptrs)

    # Single barrier for staging
    src_hdls[0].barrier(channel=0)

    for t_idx, ten in enumerate(tensors):
        out_splits = output_splits_list[t_idx]
        peer_in_splits = peer_input_splits_list[t_idx]  # [r] = list over tensors-of-peer? NO: list of ints of length world_size

        total_out = sum(out_splits)
        out = torch.empty(total_out, dtype=ten.dtype, device=device)

        # src_off[r] = sum over j<rank of peer_in_splits[r][j]
        src_offs = []
        for r in range(world_size):
            psplits = peer_in_splits[r]
            src_offs.append(sum(psplits[:rank]))
        # dst_off[r] = sum over r'<r of out_splits[r']
        dst_offs = []
        running = 0
        for r in range(world_size):
            dst_offs.append(running)
            running += out_splits[r]

        src_off_t = torch.tensor(src_offs, device=device, dtype=torch.int64)
        dst_off_t = torch.tensor(dst_offs, device=device, dtype=torch.int64)
        cnt_t = torch.tensor(out_splits, device=device, dtype=torch.int64)
        max_cnt = max(out_splits) if out_splits else 0

        ext.launch_gather_peers(
            src_ptrs_list[t_idx],
            src_off_t,
            dst_off_t,
            cnt_t,
            out,
            ten.element_size(),
            world_size,
            int(max_cnt),
        )
        outputs.append(out)

    # Trailing barrier so we don't overwrite src bufs before peers finish reading
    src_hdls[0].barrier(channel=1)
    return outputs


# ---------------------------------------------------------------------------
# Metadata exchange via symm mem
# ---------------------------------------------------------------------------

def _exchange_meta(
    meta_input: torch.Tensor,  # [num_meta_tensors, world_size] int64
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Each rank stages its meta row. We then read peer rows. Returns
    `meta_output[t, r]` = peer r's meta_input[t, rank], i.e., size that rank
    receives from r for tensor t.

    Equivalently: meta_output[t, r] = peer_r.meta_input[t, my_rank].
    """
    device = meta_input.device
    n = meta_input.numel()
    buf, hdl, ptrs = _get_symm_buf(n, torch.int64, device, "meta")
    buf[:n].copy_(meta_input.reshape(-1))
    hdl.barrier(channel=2)

    # Read from each peer the entry [t, my_rank] for all t => column my_rank.
    # Build output by copying from each peer's column.
    num_t = meta_input.shape[0]
    out = torch.empty((num_t, world_size), dtype=torch.int64, device=device)

    ext = _get_ext()
    # For each peer r we want num_t elements at offsets [t * world_size + rank].
    # These are strided, not contiguous. Easiest: pull whole peer buffer (small)
    # then index. n is tiny (~ a few * world_size).
    full = torch.empty(n * world_size, dtype=torch.int64, device=device)
    src_offs = torch.zeros(world_size, dtype=torch.int64, device=device)
    dst_offs = torch.arange(0, n * world_size, n, dtype=torch.int64, device=device)
    cnt = torch.full((world_size,), n, dtype=torch.int64, device=device)
    ext.launch_gather_peers(ptrs, src_offs, dst_offs, cnt, full, 8, world_size, n)

    full_view = full.view(world_size, num_t, world_size)
    # peer r row t: full_view[r, t, :]; the entry destined for me is column `rank`.
    # meta_output[t, r] = full_view[r, t, rank]
    out = full_view[:, :, rank].T.contiguous()  # [num_t, world_size]

    hdl.barrier(channel=3)
    return out


# ---------------------------------------------------------------------------
# Main solution
# ---------------------------------------------------------------------------

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

    # Make sure extension is loaded uniformly.
    if rank == 0:
        _get_ext()
    dist.barrier()
    _get_ext()

    num_features = sum(key_splits)
    variable_stride = stride_per_key is not None
    if stride_per_key is None:
        stride_per_key = [batch_size] * num_features

    length_per_key = _lengths_per_key(lengths, stride_per_key)
    length_splits = _sum_by_splits(stride_per_key, key_splits)
    value_splits = _sum_by_splits(length_per_key, key_splits)

    input_splits: List[List[int]] = [length_splits, value_splits]
    input_tensors: List[torch.Tensor] = [lengths, values]
    tensor_kinds: List[str] = ["lengths", "values"]
    if variable_stride:
        input_splits.append(list(key_splits))
        input_tensors.append(
            torch.tensor(stride_per_key, dtype=torch.long, device=device)
        )
        tensor_kinds.append("strides")
    if weights is not None:
        input_splits.append(value_splits)
        input_tensors.append(weights)
        tensor_kinds.append("weights")

    # Build meta tensor [num_meta_rows, world_size].
    # Order: input_splits per tensor, then (if not variable_stride) batch row.
    meta_rows_t = [torch.tensor(s, dtype=torch.long, device=device) for s in input_splits]
    if not variable_stride:
        meta_rows_t.append(
            torch.full((world_size,), batch_size, dtype=torch.long, device=device)
        )
    meta_input = torch.stack(meta_rows_t, dim=0)  # [M, world_size]

    # Exchange meta: meta_output[t, r] = size we receive from peer r for row t.
    meta_output = _exchange_meta(meta_input, rank, world_size)
    meta_rows = [
        [int(item) for item in meta_output[i].tolist()]
        for i in range(meta_output.shape[0])
    ]
    if variable_stride:
        output_splits = meta_rows  # one per tensor
        stride_per_rank = None
    else:
        output_splits = meta_rows[: len(input_tensors)]
        stride_per_rank = meta_rows[-1]

    # We also need each peer's input_splits[t][rank] prefix to compute src_off.
    # For tensor t: peer r's input_splits[t] is what r contributed. We didn't
    # gather entire peer rows. Let's do a second meta exchange to grab full
    # input_splits per peer per tensor. Actually we already have per-peer info
    # via the full buffer - but _exchange_meta only returned the column for
    # this rank. We'll do a dedicated full-exchange for peer input splits.

    # peer_input_splits_full[t][r][j] = peer r's input_splits[t][j]
    # That requires every rank to know peer r's full row for each meta tensor.
    # Stage meta_input again and do an allgather-style read.
    n_meta = meta_input.numel()
    M = meta_input.shape[0]
    buf, hdl, ptrs = _get_symm_buf(n_meta, torch.int64, device, "meta_full")
    buf[:n_meta].copy_(meta_input.reshape(-1))
    hdl.barrier(channel=4)

    full = torch.empty(n_meta * world_size, dtype=torch.int64, device=device)
    src_offs = torch.zeros(world_size, dtype=torch.int64, device=device)
    dst_offs = torch.arange(0, n_meta * world_size, n_meta, dtype=torch.int64, device=device)
    cnt = torch.full((world_size,), n_meta, dtype=torch.int64, device=device)
    _get_ext().launch_gather_peers(ptrs, src_offs, dst_offs, cnt, full, 8, world_size, n_meta)
    hdl.barrier(channel=5)

    full_view = full.view(world_size, M, world_size)  # [peer r, tensor t, j]
    # peer_input_splits_list[t][r] = list of length world_size
    peer_input_splits_list: List[List[List[int]]] = []
    for t in range(len(input_tensors)):
        per_t = []
        for r in range(world_size):
            per_t.append([int(x) for x in full_view[r, t].tolist()])
        peer_input_splits_list.append(per_t)

    # Bulk all-to-all of payload tensors via symm mem
    outputs = _bulk_a2a_symm(
        input_tensors,
        input_splits,
        output_splits,
        peer_input_splits_list,
        rank,
        world_size,
        tag_prefix="payload",
    )

    recv_lengths = outputs[0]
    recv_values = outputs[1]
    recv_strides: Optional[torch.Tensor] = None
    recv_weights: Optional[torch.Tensor] = None
    idx = 2
    if variable_stride:
        recv_strides = outputs[idx]
        idx += 1
    if weights is not None:
        recv_weights = outputs[idx]
        idx += 1

    local_split = key_splits[rank]
    if variable_stride:
        assert recv_strides is not None
        recat = _get_recat(local_split, world_size, stagger, device=device)
        if recat is not None:
            value_segment_lengths = torch.tensor(
                _lengths_per_key(recv_lengths, recv_strides.to(torch.long).tolist()),
                dtype=torch.long,
                device=device,
            )
            recv_lengths = _permute_segments(recv_lengths, recv_strides, recat)
            recv_values = _permute_segments(recv_values, value_segment_lengths, recat)
            if recv_weights is not None:
                recv_weights = _permute_segments(
                    recv_weights, value_segment_lengths, recat
                )
        stride_per_key_per_rank = recv_strides.view(world_size, local_split).T
        if stagger > 1:
            order = (
                torch.arange(world_size, device=device)
                .view(stagger, -1)
                .T.reshape(-1)
            )
            stride_per_key_per_rank = stride_per_key_per_rank[:, order]
        result: Dict[str, torch.Tensor] = {
            "lengths": recv_lengths,
            "values": recv_values,
            "stride_per_key_per_rank": stride_per_key_per_rank,
        }
    else:
        assert stride_per_rank is not None
        single_batch_per_rank = all(
            stride == stride_per_rank[0] for stride in stride_per_rank
        )
        if single_batch_per_rank:
            recat = _get_recat(local_split, world_size, stagger, device=device)
            if recat is not None and stride_per_rank[0] > 0:
                lengths_2d, recv_values, recv_weights = _permute_2d_sparse_data(
                    recat,
                    recv_lengths.view(-1, stride_per_rank[0]),
                    recv_values,
                    recv_weights,
                )
                recv_lengths = lengths_2d.reshape(-1)
        else:
            recat = _get_recat(
                local_split,
                world_size,
                stagger,
                device=device,
                batch_size_per_rank=stride_per_rank,
            )
            if recat is not None:
                recv_values = _permute_segments(recv_values, recv_lengths, recat)
                if recv_weights is not None:
                    recv_weights = _permute_segments(recv_weights, recv_lengths, recat)
                recv_lengths = recv_lengths[recat.long()]
        result = {
            "lengths": recv_lengths,
            "values": recv_values,
            "stride": torch.tensor(sum(stride_per_rank), device=device),
            "stride_per_rank": torch.tensor(stride_per_rank, device=device),
        }

    if recv_weights is not None:
        result["weights"] = recv_weights
    return result