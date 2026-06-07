import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Vectorized copy from a remote (UVA) source into a local destination.
// Uses int4 (16-byte) loads/stores when alignment permits.
__global__ void p2p_copy_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t nbytes
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    int64_t n_vec = nbytes / 16;
    const int4* s4 = reinterpret_cast<const int4*>(src);
    int4* d4 = reinterpret_cast<int4*>(dst);
    for (int64_t i = tid; i < n_vec; i += stride) {
        d4[i] = s4[i];
    }
    int64_t tail_start = n_vec * 16;
    for (int64_t i = tail_start + tid; i < nbytes; i += stride) {
        dst[i] = src[i];
    }
}

void p2p_copy(
    int64_t src_ptr,
    int64_t dst_ptr,
    int64_t nbytes
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint8_t* src = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(src_ptr));
    uint8_t* dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_ptr));
    int threads = 256;
    int64_t n_vec = nbytes / 16;
    int64_t blocks64 = (n_vec + threads - 1) / threads;
    if (blocks64 < 1) blocks64 = 1;
    if (blocks64 > 1024) blocks64 = 1024;
    int blocks = (int)blocks64;
    p2p_copy_kernel<<<blocks, threads, 0, stream>>>(src, dst, nbytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("p2p_copy", &p2p_copy, "P2P UVA copy");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gemm_allgather_p2p_ext", CUDA_SRC)
    return _ext


_cache = {}

def _get_resources(M, K_local, dtype, device, world_size):
    key = (M, K_local, dtype, device, world_size)
    if key in _cache:
        return _cache[key]
    # Symmetric buffer holds this rank's A_local shard, exposed to peers.
    sym_buf = symm_mem.empty((M, K_local), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(sym_buf, dist.group.WORLD)
    # Local assembled A_global buffer.
    A_global = torch.empty((M, K_local * world_size), dtype=dtype, device=device)
    side_stream = torch.cuda.Stream(device=device)
    _cache[key] = (sym_buf, hdl, A_global, side_stream)
    return _cache[key]


@torch.no_grad()
def solution(A_local: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized()
    assert A_local.is_cuda and B.is_cuda

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, K_local = A_local.shape
    K_B, N = B.shape
    dtype = A_local.dtype
    device = A_local.device

    # Trigger compile on all ranks.
    ext = _get_ext()

    sym_buf, hdl, A_global, side_stream = _get_resources(
        M, K_local, dtype, device, world_size
    )

    # Publish our shard into symmetric buffer.
    sym_buf.copy_(A_local)

    # Also place our own shard into the assembled A_global at our slot.
    own_slot = A_global[:, rank * K_local : (rank + 1) * K_local]
    own_slot.copy_(A_local)

    # Cross-rank synchronization: ensure all peers have published before reads.
    hdl.barrier(channel=0)

    main_stream = torch.cuda.current_stream(device)
    side_stream.wait_stream(main_stream)

    elem_size = A_local.element_size()
    shard_bytes = M * K_local * elem_size

    # Issue P2P reads for all peer shards on the side stream (overlap with anything else).
    with torch.cuda.stream(side_stream):
        for offset in range(1, world_size):
            peer = (rank + offset) % world_size
            src_ptr = int(hdl.buffer_ptrs[peer])
            dst_slot = A_global[:, peer * K_local : (peer + 1) * K_local]
            # dst_slot is a view; underlying storage is contiguous along rows of A_global.
            # But the slice along columns is NOT contiguous. We need a contiguous-strided copy.
            # Instead, copy row by row using the kernel: easier to memcpy whole shard into a
            # contiguous staging area then assign? To keep it simple and correct, use
            # cudaMemcpy2DAsync via PyTorch's copy_ with a contiguous temp shard buffer.
            # However, A_global slice is strided. We'll allocate a contiguous staging tensor.
            pass

    # Simpler & correct: stage each peer shard contiguously, then assign into A_global.
    # We'll do the staged copy on side_stream and the assignment on side_stream too.
    staging = []
    with torch.cuda.stream(side_stream):
        for offset in range(1, world_size):
            peer = (rank + offset) % world_size
            src_ptr = int(hdl.buffer_ptrs[peer])
            tmp = torch.empty((M, K_local), dtype=dtype, device=device)
            ext.p2p_copy(src_ptr, tmp.data_ptr(), shard_bytes)
            staging.append((peer, tmp))
        for peer, tmp in staging:
            A_global[:, peer * K_local : (peer + 1) * K_local].copy_(tmp)

    # Wait for all peer shards to be assembled.
    main_stream.wait_stream(side_stream)

    # Single GEMM on assembled A_global.
    C = torch.matmul(A_global, B)

    # Ensure symm buffer isn't reused before peers finish reading.
    hdl.barrier(channel=1)

    return C