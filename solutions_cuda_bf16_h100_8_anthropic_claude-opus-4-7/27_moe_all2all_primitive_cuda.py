"""
Custom all_to_all_single via symmetric memory + CUDA kernel doing device-side
peer copies over NVLink with UVA pointers. Avoids NCCL.
"""

from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void send_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__global__ void barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size,
    uint64_t block_id
) {
    int tid = threadIdx.x;
    if (tid >= world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal(send_addr);
    wait_signal(wait_addr);
}

// Each block copies one (peer, chunk) tile.
// We pull data from peer's symmetric input buffer to local output.
// Block grid: blocks_per_peer blocks per peer; we map block.x -> peer index,
// block.y -> chunk index within that peer's contribution.
__global__ void all_to_all_pull_kernel(
    const uint64_t* __restrict__ peer_input_ptrs,  // [world_size] symm buffer base on each peer
    uint8_t* __restrict__ local_output,            // local output tensor
    const int64_t* __restrict__ input_offsets_per_peer,  // [world_size]: offset on peer p where peer p has put MY chunk
    const int64_t* __restrict__ input_sizes,             // [world_size]: bytes peer p sends to me
    const int64_t* __restrict__ output_offsets,          // [world_size]: offset in my output for peer p's data
    int world_size,
    int rank
) {
    int peer = blockIdx.x;
    if (peer >= world_size) return;

    int64_t nbytes = input_sizes[peer];
    if (nbytes <= 0) return;

    int64_t src_off = input_offsets_per_peer[peer];
    int64_t dst_off = output_offsets[peer];

    const uint8_t* src = reinterpret_cast<const uint8_t*>(peer_input_ptrs[peer]) + src_off;
    uint8_t* dst = local_output + dst_off;

    // Vectorized copy with uint4 (16 bytes)
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int n_blocks_y = gridDim.y;
    int by = blockIdx.y;

    int64_t n_vec = nbytes / 16;
    int64_t tail_start = n_vec * 16;

    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);

    int64_t total_threads = (int64_t)nthreads * (int64_t)n_blocks_y;
    int64_t global_tid = (int64_t)by * (int64_t)nthreads + (int64_t)tid;

    for (int64_t i = global_tid; i < n_vec; i += total_threads) {
        dst4[i] = src4[i];
    }

    // tail bytes
    if (by == 0) {
        for (int64_t i = tail_start + tid; i < nbytes; i += nthreads) {
            dst[i] = src[i];
        }
    }
}

void launch_barrier(
    torch::Tensor signal_pad_ptrs,
    int64_t rank,
    int64_t world_size,
    int64_t block_id
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_sig = reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>());
    int threads = world_size;
    if (threads < 32) threads = 32;
    barrier_kernel<<<1, threads, 0, stream>>>(d_sig, (int)rank, (int)world_size, (uint64_t)block_id);
}

void launch_all_to_all(
    torch::Tensor peer_input_ptrs,        // int64 [world_size]
    torch::Tensor local_output,           // any dtype
    torch::Tensor input_offsets_per_peer, // int64 [world_size]
    torch::Tensor input_sizes,            // int64 [world_size]
    torch::Tensor output_offsets,         // int64 [world_size]
    int64_t world_size,
    int64_t rank,
    int64_t blocks_per_peer
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_peer = reinterpret_cast<const uint64_t*>(peer_input_ptrs.data_ptr<int64_t>());
    dim3 grid((unsigned)world_size, (unsigned)blocks_per_peer, 1);
    dim3 block(256, 1, 1);
    all_to_all_pull_kernel<<<grid, block, 0, stream>>>(
        d_peer,
        reinterpret_cast<uint8_t*>(local_output.data_ptr()),
        input_offsets_per_peer.data_ptr<int64_t>(),
        input_sizes.data_ptr<int64_t>(),
        output_offsets.data_ptr<int64_t>(),
        (int)world_size,
        (int)rank
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_barrier", &launch_barrier, "barrier via signal pad");
    m.def("launch_all_to_all", &launch_all_to_all, "all-to-all pull kernel");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("a2a_symm_ext", CUDA_SRC)
    return _ext


# Cache: keyed by (group_id, dtype, max_bytes_bucket)
_buf_cache = {}
_block_id_counter = [0]


def _next_block_id(world_size: int) -> int:
    # we only have a fixed signal pad. cycle within range.
    bid = _block_id_counter[0]
    _block_id_counter[0] = (bid + 1) % 64  # signal pad has many slots
    return bid


def _get_symm_buffer(nbytes: int, device, group):
    """Get a symmetric memory buffer >= nbytes (in bytes), as uint8."""
    # Round up to a power-of-two-ish bucket to avoid frequent reallocation.
    bucket = 1 << (max(nbytes, 1) - 1).bit_length()
    bucket = max(bucket, 1 << 20)  # min 1 MB
    key = (id(group), bucket)
    if key in _buf_cache:
        return _buf_cache[key]

    buf = symm_mem.empty(bucket, device=device, dtype=torch.uint8)
    hdl = symm_mem.rendezvous(buf, group)
    peer_ptrs = torch.tensor(
        [int(p) for p in hdl.buffer_ptrs], device=device, dtype=torch.int64
    )
    # signal pad ptrs
    sig_ptrs = torch.tensor(
        [int(p) for p in hdl.signal_pad_ptrs], device=device, dtype=torch.int64
    )
    entry = {
        "buf": buf,
        "hdl": hdl,
        "peer_ptrs": peer_ptrs,
        "sig_ptrs": sig_ptrs,
        "bucket": bucket,
    }
    _buf_cache[key] = entry
    return entry


def _to_list(x, world_size):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)


@torch.no_grad()
def solution(
    local_tensor: torch.Tensor,
    input_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    output_split_sizes: Optional[Union[List[int], torch.Tensor]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    local_tensor = local_tensor.contiguous()
    if world_size == 1:
        return local_tensor

    hidden = local_tensor.size(1)
    elem_size = local_tensor.element_size()
    row_bytes = hidden * elem_size

    n_local = local_tensor.size(0)
    if input_split_sizes is None:
        assert n_local % world_size == 0
        in_splits = [n_local // world_size] * world_size
    else:
        in_splits = _to_list(input_split_sizes, world_size)

    if output_split_sizes is None:
        assert n_local % world_size == 0
        out_splits = [n_local // world_size] * world_size
    else:
        out_splits = _to_list(output_split_sizes, world_size)

    out_rows = sum(out_splits)
    output = torch.empty(
        (out_rows, hidden),
        dtype=local_tensor.dtype,
        device=local_tensor.device,
    )

    # Compute per-peer byte offsets in input (for "what I send to peer p", which lives at offset sum(in_splits[:p]))
    in_offsets_rows = [0]
    for s in in_splits:
        in_offsets_rows.append(in_offsets_rows[-1] + s)
    out_offsets_rows = [0]
    for s in out_splits:
        out_offsets_rows.append(out_offsets_rows[-1] + s)

    total_in_bytes = in_offsets_rows[-1] * row_bytes
    total_out_bytes = out_offsets_rows[-1] * row_bytes

    device = local_tensor.device

    # Need to communicate to each peer: where in their output buffer is "my data for them"?
    # Strategy: each rank places its full input buffer into its own symmetric buffer in canonical
    # (rank-ordered) layout. Then peers PULL the slice destined for them.
    # The slice peer p wants from rank r is at byte offset = in_offsets[p] * row_bytes on rank r.
    #
    # For symmetry, all ranks must agree on layout. Simplest: layout on each rank is the rank's
    # *input* tensor verbatim, with split boundaries given by input_split_sizes.
    #
    # The puller (rank R) needs to know each peer P's input_split layout to compute the offset
    # of "P's chunk for R" within P's symmetric buffer. So we need an all-gather of input_split_sizes.
    #
    # We can use a small device tensor and do this once per call (cheap) -- but to avoid NCCL,
    # we can also exchange via the symmetric buffer itself.

    # Use a separate small symm buffer for split metadata exchange.
    # Format: each rank writes its input_split_sizes (world_size int64) at offset rank*world_size*8
    # in a shared metadata symm buffer.

    meta_bytes = world_size * world_size * 8
    meta_entry = _get_symm_buffer(meta_bytes, device, group)

    ext = _get_ext()

    # Write my input_split_sizes into meta buffer
    my_splits_t = torch.tensor(in_splits, device=device, dtype=torch.int64)
    meta_buf_view = meta_entry["buf"][: meta_bytes].view(torch.int64).view(world_size, world_size)
    meta_buf_view[rank].copy_(my_splits_t)

    # Barrier so everyone has written
    bid = _next_block_id(world_size)
    ext.launch_barrier(meta_entry["sig_ptrs"], rank, world_size, bid)

    # Now read all ranks' splits
    all_splits = meta_buf_view.clone()  # [world_size, world_size], all_splits[p, r] = peer p's input_splits[r]

    # For me (rank R), peer p's "chunk for R" is at offset = sum(all_splits[p, :R]) rows in p's buffer.
    # input_sizes[p] = all_splits[p, R] rows  (== out_splits[p] -- they should match).
    input_offsets_rows_per_peer = torch.zeros(world_size, dtype=torch.int64, device=device)
    cumsum = torch.cumsum(all_splits, dim=1)  # [world_size, world_size]
    # offset of column R in row p = cumsum[p, R-1] for R>=1 else 0
    if rank == 0:
        input_offsets_rows_per_peer.zero_()
    else:
        input_offsets_rows_per_peer = cumsum[:, rank - 1].contiguous()

    input_sizes_rows = all_splits[:, rank].contiguous()  # rows from each peer
    input_offsets_bytes = (input_offsets_rows_per_peer * row_bytes).contiguous()
    input_sizes_bytes = (input_sizes_rows * row_bytes).contiguous()

    out_offsets_t = torch.tensor(
        [o * row_bytes for o in out_offsets_rows[:-1]],
        device=device, dtype=torch.int64,
    )

    # Now prepare data buffer: copy local_tensor into symmetric data buffer
    data_entry = _get_symm_buffer(total_in_bytes, device, group)
    if total_in_bytes > 0:
        data_view = data_entry["buf"][: total_in_bytes]
        # copy local_tensor bytes
        src_bytes = local_tensor.view(torch.uint8).reshape(-1)
        data_view.copy_(src_bytes)

    # Barrier so all peers have written their data
    bid = _next_block_id(world_size)
    ext.launch_barrier(data_entry["sig_ptrs"], rank, world_size, bid)

    # Launch pull kernel
    if total_out_bytes > 0:
        # heuristic: more blocks per peer for larger transfers
        blocks_per_peer = 8
        ext.launch_all_to_all(
            data_entry["peer_ptrs"],
            output.view(torch.uint8).reshape(-1),
            input_offsets_bytes,
            input_sizes_bytes,
            out_offsets_t,
            world_size,
            rank,
            blocks_per_peer,
        )

    # Final barrier so peers don't overwrite the symm buffer before we're done reading
    bid = _next_block_id(world_size)
    ext.launch_barrier(data_entry["sig_ptrs"], rank, world_size, bid)

    return output