"""
Distributed 2D inverse real FFT using symmetric memory for fast device-side
all-gather and all-to-all transpose. The Hermitian conjugate padding and
transpose phases write directly into peer-mapped symmetric buffers via UVA.
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
#include <cuda_bf16.h>
#include <cstdint>

// Copy a contiguous tile from src into peer's symmetric buffer at byte offset.
__global__ void copy_to_peers_kernel(
    const uint8_t* __restrict__ src,
    const uint64_t* __restrict__ peer_ptrs,
    int64_t bytes,
    int64_t dst_byte_offset,
    int world_size
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;
    uint8_t* dst = reinterpret_cast<uint8_t*>(peer_ptrs[peer]) + dst_byte_offset;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Use uint4 (16-byte) loads when aligned
    if ((bytes % 16) == 0 && (((uintptr_t)src & 15) == 0) && (((uintptr_t)dst & 15) == 0)) {
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        int64_t n4 = bytes / 16;
        for (int64_t i = idx; i < n4; i += stride) {
            d4[i] = s4[i];
        }
    } else {
        for (int64_t i = idx; i < bytes; i += stride) {
            dst[i] = src[i];
        }
    }
}

void launch_copy_to_peers(
    torch::Tensor src,
    torch::Tensor peer_ptrs_tensor,
    int64_t bytes,
    int64_t dst_byte_offset,
    int world_size
) {
    const uint8_t* src_p = reinterpret_cast<const uint8_t*>(src.data_ptr());
    const uint64_t* d_peers = reinterpret_cast<const uint64_t*>(peer_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int64_t n4 = (bytes + 15) / 16;
    int blocks_x = (int)std::min<int64_t>((n4 + threads - 1) / threads, 1024);
    if (blocks_x < 1) blocks_x = 1;
    dim3 grid(blocks_x, world_size);
    copy_to_peers_kernel<<<grid, threads, 0, stream>>>(
        src_p, d_peers, bytes, dst_byte_offset, world_size);
}

// Build full half-spectrum dimension via Hermitian conjugate (bfloat16 complex = 2 bf16)
// Input shard layout: [..., pad_dim=orig_size, ...]. Output: [..., pad_dim=size, ...].
// For indices k in [orig_size, size): out[k] = conj(in_padded_after_gather[size - k])
// But we apply locally first: lhs[k] = conj(in[size - k - orig_size + ... ])
// Mirroring the reference logic exactly.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_copy_to_peers", &launch_copy_to_peers, "Copy buffer into peers via UVA");
}
'''


_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dist_irfft_symm_ext", CUDA_SRC)
    return _ext


_symm_cache = {}

def _get_symm_buffer(nbytes: int, device: torch.device, key: str):
    """Allocate a symmetric memory byte buffer of size nbytes."""
    cache_key = (key, nbytes, device.index)
    if cache_key in _symm_cache:
        return _symm_cache[cache_key]
    # Allocate as int8 for byte-level access
    buf = symm_mem.empty(nbytes, device=device, dtype=torch.int8)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    peer_ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    _symm_cache[cache_key] = (buf, hdl, peer_ptrs)
    return buf, hdl, peer_ptrs


def _symm_all_gather(local: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    """All-gather via symmetric memory: each rank writes its shard into peers' buffers.
    Returns concatenation along dim."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_c = local.contiguous()
    shard_bytes = local_c.numel() * local_c.element_size()
    total_bytes = shard_bytes * world_size

    buf, hdl, peer_ptrs = _get_symm_buffer(total_bytes, local_c.device, f"ag_{tuple(local_c.shape)}_{local_c.dtype}")

    hdl.barrier(channel=0)
    # Each rank writes its shard into slot `rank` of every peer's buffer
    _get_ext().launch_copy_to_peers(
        local_c.view(torch.int8).reshape(-1),
        peer_ptrs,
        shard_bytes,
        rank * shard_bytes,
        world_size,
    )
    hdl.barrier(channel=1)

    # View buf as concat shape and rearrange to gather along dim
    # Concat along dim 0 produces shape [world_size * shard_dim_size, ...other dims...]
    # We need to permute so that dim is the gather dim.
    # buf layout: world_size shards each of local_c.shape, concatenated along dim 0 of "shards"
    full_shape = list(local_c.shape)
    # Construct as [world_size, *local_shape] then move axis
    full = buf.view(local_c.dtype).view(world_size, *local_c.shape)
    # Move axis 0 to position `dim+1` then merge with dim
    # full shape: [W, d0, d1, ..., dN]
    # We want concat along `dim`: shape [d0,...,d_{dim}*W,...,dN]
    # Permute: bring axis 0 to position dim, so axis 0 is adjacent to original dim axis
    perm = list(range(1, full.ndim))
    perm.insert(dim, 0)
    full_perm = full.permute(*perm).contiguous()
    new_shape = list(local_c.shape)
    new_shape[dim] = new_shape[dim] * world_size
    return full_perm.view(*new_shape)


def _symm_all_to_all(send_chunks: list, group: dist.ProcessGroup) -> list:
    """All-to-all of equally sized chunks via symmetric memory."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    chunk = send_chunks[0].contiguous()
    chunk_bytes = chunk.numel() * chunk.element_size()
    total_bytes = chunk_bytes * world_size

    # Need a send buffer (symmetric) and a recv buffer (symmetric)
    send_buf, send_hdl, send_peer_ptrs = _get_symm_buffer(total_bytes, chunk.device, f"a2a_send_{chunk.shape}_{chunk.dtype}")
    recv_buf, recv_hdl, recv_peer_ptrs = _get_symm_buffer(total_bytes, chunk.device, f"a2a_recv_{chunk.shape}_{chunk.dtype}")

    # Pack send buffer
    send_view = send_buf.view(chunk.dtype).view(world_size, *chunk.shape)
    for i, c in enumerate(send_chunks):
        send_view[i].copy_(c.contiguous())

    send_hdl.barrier(channel=0)
    recv_hdl.barrier(channel=0)

    # For each peer p, write our chunk[p] (== send_view[p]) into recv_buf of peer p at slot `rank`
    # We need to launch one copy per peer with different src offset and dst peer.
    # Simpler: launch one kernel per peer.
    ext = _get_ext()
    for p in range(world_size):
        src_chunk = send_view[p].contiguous()  # may already be contiguous
        # Build a tensor with single peer ptr
        single_peer = torch.tensor([int(recv_peer_ptrs[p].item())], device=chunk.device, dtype=torch.int64)
        ext.launch_copy_to_peers(
            src_chunk.view(torch.int8).reshape(-1),
            single_peer,
            chunk_bytes,
            rank * chunk_bytes,
            1,
        )

    recv_hdl.barrier(channel=1)
    send_hdl.barrier(channel=1)

    recv_view = recv_buf.view(chunk.dtype).view(world_size, *chunk.shape)
    return [recv_view[i].clone() for i in range(world_size)]


def _pad_zero(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    dim = dim % tensor.ndim
    if tensor.shape[dim] == size:
        return tensor.contiguous()
    new_shape = list(tensor.shape)
    new_shape[dim] = size
    out = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    sl = [slice(None)] * tensor.ndim
    sl[dim] = slice(0, tensor.shape[dim])
    out[tuple(sl)] = tensor
    return out


def _scatter_dim(tensor: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    chunks = torch.split(tensor, tensor.shape[dim] // world_size, dim=dim)
    return chunks[rank].contiguous()


def _conj_pad_2d_symm(
    tensor: torch.Tensor,
    pad_dim: int,
    other_dim: int,
    size: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    pad_dim = pad_dim % tensor.ndim
    other_dim = other_dim % tensor.ndim
    orig_size = tensor.shape[pad_dim]

    tensor_pad = _pad_zero(tensor, pad_dim, size)
    lhs_slice = [slice(0, s) for s in tensor_pad.shape]
    lhs_slice[pad_dim] = slice(orig_size, size)
    rhs_slice = [slice(0, s) for s in tensor_pad.shape]
    rhs_slice[pad_dim] = slice(1, size - orig_size + 1)
    tensor_pad[tuple(lhs_slice)] = torch.flip(torch.conj(tensor_pad[tuple(rhs_slice)]), dims=[pad_dim])

    # All-gather along other_dim using symm memory
    tensor_pad = _symm_all_gather(tensor_pad, other_dim, group)

    flip_slice = [slice(0, s) for s in tensor_pad.shape]
    flip_slice[pad_dim] = slice(orig_size, size)
    flip_slice[other_dim] = slice(1, tensor_pad.shape[other_dim])
    tensor_pad[tuple(flip_slice)] = torch.flip(tensor_pad[tuple(flip_slice)], dims=[other_dim])
    return _scatter_dim(tensor_pad, other_dim, group)


@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Optional[Sequence[int]],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    dim0, dim1 = int(dim[0]), int(dim[1])
    if s is not None:
        first_dim_size = int(s[0])
        last_dim_size = int(s[1])
    else:
        first_dim_size = int(x.shape[dim0])
        last_dim_size = int(2 * (x.shape[dim1] - 1))

    # Ensure extension compiled
    if dist.get_rank(group) == 0:
        _get_ext()
    dist.barrier(group=group)
    _get_ext()

    # 1. Hermitian-rebuild padding via symmetric all-gather.
    x_pad = _conj_pad_2d_symm(x, pad_dim=dim1, other_dim=dim0, size=last_dim_size, group=group)

    # 2. IFFT along dim1 (now full).
    x1 = torch.fft.ifft(x_pad, n=last_dim_size, dim=dim1, norm=norm)

    # 3. All-to-all transpose along dim1 via symmetric memory.
    world_size = dist.get_world_size(group)
    chunk = x1.shape[dim1] // world_size
    send = [c.contiguous() for c in torch.split(x1, chunk, dim=dim1)]
    recv = _symm_all_to_all(send, group)
    x1_tran = torch.cat(recv, dim=dim0)

    # 4. IFFT along dim0 and take real.
    x2 = torch.fft.ifft(x1_tran, n=first_dim_size, dim=dim0, norm=norm)
    return torch.real(x2).contiguous()