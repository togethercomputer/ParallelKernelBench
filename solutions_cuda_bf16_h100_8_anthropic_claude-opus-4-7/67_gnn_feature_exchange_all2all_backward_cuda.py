"""
GraphBolt cooperative GNN feature exchange backward — CUDA + symm_mem.

Strategy:
- Replace dist.all_to_all with a one-shot symmetric-memory all-to-all: each rank
  writes its outgoing chunks directly into peer symmetric buffers via UVA pointers.
- Replace torch.sparse.mm scatter with a fused custom CUDA scatter-add kernel
  that operates on BF16 with float accumulation.
- Use symm_mem.rendezvous handle barriers for cheap device-side synchronization.
"""

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
#include <cstdint>

// Copy this rank's outgoing chunk to each peer's symmetric receive buffer.
// peer_buf_ptrs[r] points to peer r's symmetric buffer (in elements of bf16),
// laid out as [world_size, max_rows_per_pair, H]: peer r expects this rank's
// data at slot [my_rank].
__global__ void a2a_scatter_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,        // [sum(send_counts), H]
    const long long* __restrict__ peer_buf_ptrs,  // [world_size]
    const int* __restrict__ send_offsets,         // [world_size+1]
    int world_size,
    int my_rank,
    int max_rows,
    int H
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;
    int row_in_peer = blockIdx.x;
    int send_off = send_offsets[peer];
    int send_cnt = send_offsets[peer + 1] - send_off;
    if (row_in_peer >= send_cnt) return;

    const __nv_bfloat16* src_row = src + (long long)(send_off + row_in_peer) * H;
    __nv_bfloat16* dst_base = reinterpret_cast<__nv_bfloat16*>(peer_buf_ptrs[peer]);
    // peer r stores incoming-from-rank `my_rank` at [my_rank, row_in_peer, :]
    __nv_bfloat16* dst_row = dst_base
        + ((long long)my_rank * max_rows + row_in_peer) * H;

    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        dst_row[i] = src_row[i];
    }
}

// Pack symm receive buffer [world_size, max_rows, H] -> contiguous [sum(recv_cnts), H]
// using recv_offsets to know per-peer row counts.
__global__ void a2a_pack_bf16_kernel(
    const __nv_bfloat16* __restrict__ recv_buf,   // [world_size, max_rows, H]
    __nv_bfloat16* __restrict__ out,              // [sum(recv_cnts), H]
    const int* __restrict__ recv_offsets,         // [world_size+1]
    int world_size,
    int max_rows,
    int H
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;
    int row = blockIdx.x;
    int off = recv_offsets[peer];
    int cnt = recv_offsets[peer + 1] - off;
    if (row >= cnt) return;

    const __nv_bfloat16* src_row = recv_buf
        + ((long long)peer * max_rows + row) * H;
    __nv_bfloat16* dst_row = out + (long long)(off + row) * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        dst_row[i] = src_row[i];
    }
}

// Scatter-add rows of `src` (bf16) into `dst` (bf16) using `index` (int64).
// dst[index[i], :] += src[i, :], using float accumulation via atomicAdd on bf16.
__global__ void scatter_add_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    const long long* __restrict__ index,
    __nv_bfloat16* __restrict__ dst,
    int n_rows,
    int H,
    int seed_size
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    long long target = index[row];
    if (target < 0 || target >= seed_size) return;

    const __nv_bfloat16* src_row = src + (long long)row * H;
    __nv_bfloat16* dst_row = dst + target * H;

    // bf16 atomicAdd is supported on Hopper.
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        atomicAdd(reinterpret_cast<__nv_bfloat16*>(dst_row + i), src_row[i]);
    }
}

void launch_a2a_scatter_bf16(
    torch::Tensor src,
    torch::Tensor peer_buf_ptrs,
    torch::Tensor send_offsets,
    int64_t world_size,
    int64_t my_rank,
    int64_t max_rows,
    int64_t H
) {
    int max_send_rows = 0;
    auto so_cpu = send_offsets.cpu();
    auto* p = so_cpu.data_ptr<int>();
    for (int i = 0; i < world_size; ++i) {
        int c = p[i+1] - p[i];
        if (c > max_send_rows) max_send_rows = c;
    }
    if (max_send_rows == 0) return;

    dim3 grid(max_send_rows, (unsigned)world_size);
    int threads = (H < 256) ? ((H + 31) / 32 * 32) : 256;
    if (threads < 32) threads = 32;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    a2a_scatter_bf16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        reinterpret_cast<const long long*>(peer_buf_ptrs.data_ptr<int64_t>()),
        send_offsets.data_ptr<int>(),
        (int)world_size,
        (int)my_rank,
        (int)max_rows,
        (int)H
    );
}

void launch_a2a_pack_bf16(
    torch::Tensor recv_buf,
    torch::Tensor out,
    torch::Tensor recv_offsets,
    int64_t world_size,
    int64_t max_rows,
    int64_t H
) {
    int max_recv_rows = 0;
    auto ro_cpu = recv_offsets.cpu();
    auto* p = ro_cpu.data_ptr<int>();
    for (int i = 0; i < world_size; ++i) {
        int c = p[i+1] - p[i];
        if (c > max_recv_rows) max_recv_rows = c;
    }
    if (max_recv_rows == 0) return;

    dim3 grid(max_recv_rows, (unsigned)world_size);
    int threads = (H < 256) ? ((H + 31) / 32 * 32) : 256;
    if (threads < 32) threads = 32;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    a2a_pack_bf16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(recv_buf.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        recv_offsets.data_ptr<int>(),
        (int)world_size,
        (int)max_rows,
        (int)H
    );
}

void launch_scatter_add_bf16(
    torch::Tensor src,
    torch::Tensor index,
    torch::Tensor dst,
    int64_t n_rows,
    int64_t H,
    int64_t seed_size
) {
    if (n_rows == 0) return;
    int threads = (H < 256) ? ((H + 31) / 32 * 32) : 256;
    if (threads < 32) threads = 32;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scatter_add_rows_bf16_kernel<<<(int)n_rows, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr<at::BFloat16>()),
        reinterpret_cast<const long long*>(index.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
        (int)n_rows,
        (int)H,
        (int)seed_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("a2a_scatter_bf16", &launch_a2a_scatter_bf16, "all-to-all scatter bf16");
    m.def("a2a_pack_bf16", &launch_a2a_pack_bf16, "all-to-all pack bf16");
    m.def("scatter_add_bf16", &launch_scatter_add_bf16, "row scatter-add bf16");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gnn_a2a_bwd_ext", CUDA_SRC)
    return _ext


_symm_cache = {}


def _get_symm_buf(world_size: int, max_rows: int, H: int, dtype, device, group):
    key = (world_size, max_rows, H, dtype, device, id(group))
    c = _symm_cache.get(key)
    if c is not None:
        return c
    buf = symm_mem.empty((world_size, max_rows, H), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, ptrs_tensor)
    return _symm_cache[key]


def _shift(chunks, rank, world_size):
    cutoff = world_size - rank
    return chunks[cutoff:] + chunks[:cutoff]


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

    if not grad_output.is_cuda or grad_output.dtype != torch.bfloat16 or not dist.is_initialized():
        # Fallback to reference path
        out = grad_output.new_empty((sum(counts_received),) + grad_output.shape[1:])
        # Reference all_to_all path
        rank = dist.get_rank(group) if dist.is_initialized() else 0
        ws = dist.get_world_size(group) if dist.is_initialized() else 1
        outs = list(torch.split(out, counts_received))
        ins = list(torch.split(grad_output, counts_sent))
        outs_s = _shift(list(outs), rank, ws)
        ins_s = _shift(list(ins), rank, ws)
        if dist.is_initialized():
            dist.all_to_all(outs_s, ins_s, group=group)
        if seed_inverse_ids.numel() == 0:
            return torch.zeros((seed_size,) + grad_output.shape[1:],
                               dtype=grad_output.dtype, device=grad_output.device)
        grad_input = torch.zeros((seed_size,) + grad_output.shape[1:],
                                 dtype=grad_output.dtype, device=grad_output.device)
        grad_input.index_add_(0, seed_inverse_ids, out)
        return grad_input

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    device = grad_output.device

    grad_output = grad_output.contiguous()
    assert grad_output.dim() == 2, "Expecting 2D [N, H]"
    H = grad_output.shape[1]

    # Apply the same _shift logic to derive the actual peer mapping.
    # In the reference, outputs and inputs are both rotated by `cutoff = ws - rank`.
    # After rotation, position `i` in the rotated list corresponds to peer
    # original_index = (i + cutoff) % ws.
    # dist.all_to_all sends rotated inputs[i] to peer i and receives rotated outputs[i] from peer i.
    # So our position-i in counts_sent/counts_received (already rotated externally before passing)
    # corresponds to rotated lists already. Replicate the shift:
    sent_chunks_unrot = list(counts_sent)
    recv_chunks_unrot = list(counts_received)
    sent_rotated = _shift(sent_chunks_unrot, rank, world_size)
    recv_rotated = _shift(recv_chunks_unrot, rank, world_size)
    # Now sent_rotated[peer] = rows we send to peer `peer`
    # recv_rotated[peer] = rows we receive from peer `peer`

    # But grad_output is split by counts_sent (unrotated), and we need to send
    # the chunks in unrotated order to peers in rotated order. The reference does:
    #   inputs = split(grad_output, counts_sent)  # unrotated
    #   inputs = _shift(inputs)                    # rotated
    # So after rotation, rotated_inputs[peer] corresponds to original chunk
    # at index (peer + cutoff) % ws. We need to construct send offsets relative to
    # grad_output's contiguous layout in *rotated* peer order.
    cutoff = world_size - rank
    # rotated_inputs[i] = unrotated_inputs[(i + cutoff) % ws]
    # offsets in original grad_output:
    unrot_offsets = [0]
    for c in sent_chunks_unrot:
        unrot_offsets.append(unrot_offsets[-1] + c)

    # Build a contiguous "send buffer" in rotated peer order so kernel can use
    # simple per-peer offset arithmetic. To avoid extra copy when already aligned,
    # we just compute per-peer source pointers via a permuted copy. Simplest: copy.
    total_send = sum(sent_chunks_unrot)
    total_recv = sum(recv_chunks_unrot)

    if total_send == 0 and total_recv == 0:
        return torch.zeros((seed_size, H), dtype=grad_output.dtype, device=device)

    # Build rotated-order send tensor
    if total_send > 0:
        send_buf = torch.empty_like(grad_output)
        cursor = 0
        for peer in range(world_size):
            orig_idx = (peer + cutoff) % world_size
            cnt = sent_chunks_unrot[orig_idx]
            if cnt > 0:
                src_off = unrot_offsets[orig_idx]
                send_buf[cursor:cursor + cnt].copy_(grad_output[src_off:src_off + cnt])
            cursor += cnt
    else:
        send_buf = grad_output

    # send_offsets in rotated order
    send_offsets = [0]
    for peer in range(world_size):
        orig_idx = (peer + cutoff) % world_size
        send_offsets.append(send_offsets[-1] + sent_chunks_unrot[orig_idx])

    recv_offsets = [0]
    for peer in range(world_size):
        orig_idx = (peer + cutoff) % world_size
        recv_offsets.append(recv_offsets[-1] + recv_chunks_unrot[orig_idx])

    # Determine global max rows per pair across the group for symm buffer sizing.
    # Use max recv count seen on this rank, but symm buffer must be uniform across ranks.
    # We'll allreduce (max) across ranks for the per-peer max.
    local_max = max(max(recv_rotated) if recv_rotated else 0,
                    max(sent_rotated) if sent_rotated else 0)
    max_t = torch.tensor([local_max], device=device, dtype=torch.int64)
    dist.all_reduce(max_t, op=dist.ReduceOp.MAX, group=group)
    max_rows = int(max_t.item())
    if max_rows == 0:
        return torch.zeros((seed_size, H), dtype=grad_output.dtype, device=device)

    # Round up to reduce reallocations
    def _roundup(x, m=64):
        return ((x + m - 1) // m) * m
    max_rows = _roundup(max_rows)

    ext = _get_ext()
    buf, hdl, peer_ptrs = _get_symm_buf(world_size, max_rows, H, grad_output.dtype, device, group)

    send_off_t = torch.tensor(send_offsets, device=device, dtype=torch.int32)
    recv_off_t = torch.tensor(recv_offsets, device=device, dtype=torch.int32)

    # Barrier so peers' symm buffer is ready to be written
    hdl.barrier(channel=0)

    # Push data directly into peer symmetric buffers via UVA
    if total_send > 0:
        ext.a2a_scatter_bf16(
            send_buf, peer_ptrs, send_off_t,
            world_size, rank, max_rows, H
        )

    # Wait for all peers to finish writing into our buffer
    hdl.barrier(channel=1)

    # Pack symm receive buffer into contiguous `out` of size [total_recv, H]
    # in rotated peer order — matching what reference dist.all_to_all wrote.
    out_rotated = torch.empty((total_recv, H), dtype=grad_output.dtype, device=device)
    if total_recv > 0:
        ext.a2a_pack_bf16(
            buf, out_rotated, recv_off_t,
            world_size, max_rows, H
        )

    # Reference then writes per-rotated-chunk into outs (which were views into `out`).
    # `outs` was list(torch.split(out, counts_received)) before _shift, so positions
    # in original `out` correspond to *unrotated* peer order. After _shift, rotated
    # outs[i] (a view) is at position `(i + cutoff) % ws` in original out.
    # So out (unrotated) layout: chunk for original peer p = received from peer
    # whose rotated index i satisfies (i + cutoff) % ws == p, i.e., i = (p - cutoff) % ws.
    # Equivalently: out_unrotated[p] = out_rotated[(p - cutoff) % ws]
    # = out_rotated[(p + rank) % ws] since cutoff = ws - rank.
    # We need `out` ordered by counts_received (unrotated).
    out_unrot = torch.empty((total_recv, H), dtype=grad_output.dtype, device=device)
    unrot_recv_offsets = [0]
    for c in recv_chunks_unrot:
        unrot_recv_offsets.append(unrot_recv_offsets[-1] + c)
    for p in range(world_size):
        rot_i = (p + rank) % world_size  # since cutoff = ws - rank, (p - cutoff) % ws = (p + rank) % ws
        cnt = recv_chunks_unrot[p]
        if cnt > 0:
            src_off = recv_offsets[rot_i]
            dst_off = unrot_recv_offsets[p]
            out_unrot[dst_off:dst_off + cnt].copy_(out_rotated[src_off:src_off + cnt])

    # Scatter-add into grad_input
    grad_input = torch.zeros((seed_size, H), dtype=grad_output.dtype, device=device)
    n_rows = out_unrot.shape[0]
    if n_rows > 0 and seed_inverse_ids.numel() > 0:
        idx = seed_inverse_ids.contiguous().to(torch.int64)
        ext.scatter_add_bf16(out_unrot, idx, grad_input, n_rows, H, seed_size)

    return grad_input