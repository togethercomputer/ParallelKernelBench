"""
Distributed 2D real FFT with all-to-all transpose via symmetric memory.

Strategy:
- Replace dist.all_to_all with a symmetric-memory based all-to-all where each
  rank writes its outgoing chunks directly into peers' UVA buffers via a
  custom CUDA kernel using vectorized loads/stores (bf16 -> uint4).
- FFTs stay on PyTorch (cuFFT) since reimplementing FFT in custom CUDA is
  not productive; the bottleneck we attack is the collective.
- Use symm_mem rendezvous + signal-pad blockwise barrier for device-side sync.
"""

from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

// One-block-per-pair barrier launched at grid level (block 0 only)
__global__ void global_barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank, int world_size, uint64_t block_id, int phase
) {
    int t = threadIdx.x;
    if (t >= world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[t];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)t);
    if (phase == 0) {
        send_signal_relaxed(send_addr);
        wait_signal_relaxed(wait_addr);
    } else {
        send_signal_acq_rel(send_addr);
        wait_signal_acq_rel(wait_addr);
    }
}

// Vectorized copy from local source into peer symm buffer at appropriate offset.
// We treat data as bytes and copy with uint4 (16B) chunks.
__global__ void p2p_put_kernel(
    const uint8_t* __restrict__ src_base,    // local source bytes (the post-FFT tensor, complex64 reinterpreted)
    uint64_t* __restrict__ peer_buf_ptrs,    // [world_size] pointers to symm buffers per peer
    int64_t bytes_per_chunk,                 // bytes for one rank's chunk (out of world_size)
    int64_t src_stride_bytes,                // stride between consecutive chunks in src (= bytes_per_chunk if contiguous split along outermost) -- handled below
    int world_size,
    int my_rank,
    // For non-trivial split_dim, we pass: outer, mid (per-chunk along dim), inner element bytes.
    int64_t outer,
    int64_t mid_per_chunk,
    int64_t inner_bytes,
    int64_t mid_total          // = mid_per_chunk * world_size
) {
    // Each block handles one (peer, outer-row) pair? Simpler: linearize.
    // Total bytes per chunk = outer * mid_per_chunk * inner_bytes
    // For a given peer p, the source slice is:
    //   src[o, p*mid_per_chunk + m, i_byte]  where indexing uses (outer, mid_total, inner_bytes)
    // The destination at peer p is its symm buffer slot for "from my_rank":
    //   peer_buf_ptrs[p] + my_rank * bytes_per_chunk + (o * mid_per_chunk * inner_bytes + m * inner_bytes + i_byte)
    int peer = blockIdx.y;
    if (peer >= world_size) return;

    int64_t total_u4 = bytes_per_chunk / 16;  // assume 16B aligned
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    uint8_t* dst_base = reinterpret_cast<uint8_t*>(peer_buf_ptrs[peer]) + (int64_t)my_rank * bytes_per_chunk;

    // Per-chunk shape: (outer, mid_per_chunk, inner_bytes). Linearize over chunk index k in [0, total_u4).
    // Convert k (units of 16B) back to (o, m, i_u4) where i_u4 covers inner_bytes/16 if inner_bytes is multiple of 16.
    // Simpler: when inner_bytes % 16 == 0, chunk is row-major contiguous within (o,m,i).
    int64_t inner_u4 = inner_bytes / 16;
    int64_t per_o = mid_per_chunk * inner_u4;

    for (int64_t k = tid; k < total_u4; k += stride) {
        int64_t o = k / per_o;
        int64_t rem = k - o * per_o;
        int64_t m = rem / inner_u4;
        int64_t iu = rem - m * inner_u4;

        // Source index: o, peer*mid_per_chunk + m, iu
        int64_t src_off_u4 = (o * mid_total + (int64_t)peer * mid_per_chunk + m) * inner_u4 + iu;
        const uint4* sptr = reinterpret_cast<const uint4*>(src_base) + src_off_u4;
        uint4* dptr = reinterpret_cast<uint4*>(dst_base) + k;
        *dptr = *sptr;
    }
}

void launch_global_barrier(
    torch::Tensor signal_pad_ptrs_dev,
    int rank, int world_size, int64_t block_id, int phase
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = world_size;
    if (threads < 32) threads = 32;
    global_barrier_kernel<<<1, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_dev.data_ptr<int64_t>()),
        rank, world_size, (uint64_t)block_id, phase);
}

void launch_p2p_put(
    torch::Tensor src,
    torch::Tensor peer_buf_ptrs,
    int64_t bytes_per_chunk,
    int world_size,
    int my_rank,
    int64_t outer,
    int64_t mid_per_chunk,
    int64_t inner_bytes,
    int64_t mid_total
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int64_t total_u4 = bytes_per_chunk / 16;
    int blocks_x = (int)std::min<int64_t>((total_u4 + threads - 1) / threads, 1024);
    if (blocks_x < 1) blocks_x = 1;
    dim3 grid(blocks_x, world_size, 1);
    p2p_put_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src.data_ptr()),
        reinterpret_cast<uint64_t*>(peer_buf_ptrs.data_ptr<int64_t>()),
        bytes_per_chunk,
        0,
        world_size, my_rank,
        outer, mid_per_chunk, inner_bytes, mid_total);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_global_barrier", &launch_global_barrier, "Symmetric mem global barrier");
    m.def("launch_p2p_put", &launch_p2p_put, "P2P put for all-to-all transpose");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("distributed_rfft_a2a_ext", CUDA_SRC)
    return _ext


_symm_cache = {}
_barrier_counter = [0]


def _get_symm_buffer(total_bytes: int, device: torch.device):
    """Allocate a symm_mem byte buffer big enough for the all-to-all."""
    key = (total_bytes, device.index)
    if key in _symm_cache:
        return _symm_cache[key]
    # allocate as uint8 buffer
    buf = symm_mem.empty(total_bytes, device=device, dtype=torch.uint8)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    sig_dev = hdl.signal_pad_ptrs_dev
    _symm_cache[key] = (buf, hdl, ptrs_tensor, sig_dev)
    return _symm_cache[key]


def _next_block_id():
    _barrier_counter[0] = (_barrier_counter[0] + 1) % 1024
    return _barrier_counter[0]


def _truncate(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    slices = [slice(None)] * tensor.ndim
    slices[dim % tensor.ndim] = slice(0, size)
    return tensor[tuple(slices)].contiguous()


def _custom_all_to_all_transpose(x1: torch.Tensor, split_dim: int, group) -> torch.Tensor:
    """
    Symm-mem-based all-to-all that splits x1 along split_dim into world_size chunks
    and returns the concatenation along the original 'replicated' dim. Implementation:
      - Each rank writes chunk-for-peer p directly to peer p's symm buffer at slot 'my_rank'.
      - After global barrier, the symm buffer layout on each rank is:
          [from_rank=0 chunk | from_rank=1 chunk | ... | from_rank=W-1 chunk]
        where each chunk has shape == one local chunk.
    The returned tensor concatenates received chunks along dim1 (the dim along which we want
    to be replicated). Caller passes dim1 as the concat dim.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    assert x1.is_cuda
    assert x1.shape[split_dim] % world_size == 0
    x1 = x1.contiguous()

    # Determine chunk shape (split along split_dim into world_size parts).
    chunk_shape = list(x1.shape)
    chunk_shape[split_dim] = x1.shape[split_dim] // world_size
    chunk_numel = 1
    for s in chunk_shape:
        chunk_numel *= s
    elem_size = x1.element_size()
    bytes_per_chunk = chunk_numel * elem_size
    total_bytes = bytes_per_chunk * world_size

    # Compute (outer, mid_per_chunk, inner_bytes) for the source layout per peer.
    # Source x1 shape with split_dim as middle axis:
    outer = 1
    for d in range(split_dim):
        outer *= x1.shape[d]
    mid_total = x1.shape[split_dim]
    mid_per_chunk = mid_total // world_size
    inner = 1
    for d in range(split_dim + 1, x1.ndim):
        inner *= x1.shape[d]
    inner_bytes = inner * elem_size

    # Need alignment for uint4 (16B). If not aligned, fallback to NCCL all_to_all.
    if (inner_bytes % 16) != 0 or (bytes_per_chunk % 16) != 0:
        # Fallback path
        send = [c.contiguous() for c in torch.split(x1, mid_per_chunk, dim=split_dim)]
        recv = [torch.empty_like(send[0]) for _ in range(world_size)]
        dist.all_to_all(recv, send, group=group)
        # We will concat along the "other" dim outside; here just return list-equivalent layout:
        # The caller does torch.cat(x1_recv, dim=dim1). To match, we return None-like tuple plus list.
        return ("fallback", recv)

    device = x1.device
    buf, hdl, peer_ptrs, sig_dev = _get_symm_buffer(total_bytes, device)

    ext = _get_ext()

    # Pre-barrier: ensure all ranks are ready before puts.
    bid = _next_block_id()
    ext.launch_global_barrier(sig_dev, hdl.rank, hdl.world_size, bid, 0)

    # Issue P2P puts (each rank writes its outgoing chunks into peers' buffers).
    ext.launch_p2p_put(
        x1.view(torch.uint8) if False else x1,  # raw pointer used in kernel
        peer_ptrs,
        bytes_per_chunk,
        world_size,
        rank,
        outer,
        mid_per_chunk,
        inner_bytes,
        mid_total,
    )

    # Post-barrier: ensure all writes visible before consumers read.
    bid2 = _next_block_id()
    ext.launch_global_barrier(sig_dev, hdl.rank, hdl.world_size, bid2, 1)

    # Reinterpret buf as the received chunks. Layout: world_size chunks of chunk_shape.
    # We want to torch.cat(recv_chunks, dim=concat_dim). Return as ("ok", buf_view, chunk_shape).
    recv_view = buf.view(x1.dtype if elem_size == buf.element_size() else x1.dtype)
    # buf is uint8; view as x1.dtype:
    recv_view = buf.view(torch.uint8)
    # Reinterpret as x1.dtype:
    recv_typed = torch.empty(0, dtype=x1.dtype, device=device)
    recv_typed = buf  # uint8
    # Use untyped storage trick:
    full = buf
    # Convert via torch.frombuffer-equivalent: use as_strided on a typed view.
    storage_offset = 0
    # Easiest: reinterpret using torch.view of underlying storage via .view(dtype)? Tensor.view(dtype) works:
    typed = full.view(x1.dtype)  # bytes -> elements
    # Now shape = (total_numel,). Reshape to (world_size, *chunk_shape).
    typed = typed.view(world_size, *chunk_shape)
    # Build list of W tensors each of chunk_shape, concat along dim split_dim+? Actually caller wants
    # cat along the dim that becomes "replicated". In the reference, that's dim1. The "split_dim" passed
    # here is dim0 (the dim that was replicated, now becomes sharded). The data that was sharded on dim1
    # needs to be reassembled along dim1. So caller will cat along dim1.
    # Provide list:
    chunks = [typed[i].contiguous() for i in range(world_size)]
    return ("ok", chunks)


@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Sequence[int],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    dim0, dim1 = int(dim[0]), int(dim[1])

    # Warm JIT once on rank 0 then barrier.
    if not hasattr(solution, "_warmed"):
        if dist.get_rank() == 0:
            _get_ext()
        dist.barrier()
        _get_ext()
        solution._warmed = True

    # 1. FFT along dim0 (replicated dim).
    x1 = torch.fft.fft(x, n=int(s[0]), dim=dim0, norm=norm)

    # 2. Custom symm-mem all-to-all transpose: split along dim0, will concat along dim1.
    status, payload = _custom_all_to_all_transpose(x1, split_dim=dim0, group=group)
    if status == "fallback":
        x1_recv = payload
    else:
        x1_recv = payload  # list of tensors of chunk_shape (each split along dim0)

    x1_tran = torch.cat(x1_recv, dim=dim1)

    # 3. FFT along dim1.
    x2 = torch.fft.fft(x1_tran, n=int(s[1]), dim=dim1, norm=norm)

    # 4. Truncate to half spectrum on dim1.
    return _truncate(x2, dim1, x2.shape[dim1] // 2 + 1)