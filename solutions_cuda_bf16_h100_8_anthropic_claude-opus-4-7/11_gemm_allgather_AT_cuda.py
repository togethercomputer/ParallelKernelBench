import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Copy from a remote (UVA) device pointer into a local destination buffer.
__global__ void copy_from_peer_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    // vectorized as int4 (8 bf16 per int4)
    int64_t n8 = n / 8;
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    for (int64_t i = idx; i < n8; i += stride) {
        dst4[i] = src4[i];
    }
    int64_t tail_start = n8 * 8;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

void copy_from_peer_bf16(
    int64_t src_ptr,
    torch::Tensor dst,
    int64_t n
) {
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(src_ptr));
    __nv_bfloat16* d = reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>());
    int threads = 256;
    int blocks = (int)std::min<int64_t>((n / 8 + threads - 1) / threads, 1024);
    if (blocks < 1) blocks = 1;
    copy_from_peer_kernel<<<blocks, threads, 0, stream>>>(src, d, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_from_peer_bf16", &copy_from_peer_bf16, "Copy bf16 buffer from peer UVA pointer");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gemm_allgather_at_ext", CUDA_SRC)
    return _ext


_resource_cache = {}

def _get_resources(M, K_local, dtype, device, world_size):
    key = (M, K_local, dtype, device, world_size)
    if key in _resource_cache:
        return _resource_cache[key]

    # Symmetric buffer for A_local^T per rank: shape [K_local, M]
    sym_buf = symm_mem.empty((K_local, M), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(sym_buf, dist.group.WORLD)

    # Streams: one compute stream + one copy stream for double-buffering
    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    # Two staging buffers for double-buffering peer A^T shards
    stage_bufs = [
        torch.empty((K_local, M), device=device, dtype=dtype),
        torch.empty((K_local, M), device=device, dtype=dtype),
    ]

    res = {
        "sym_buf": sym_buf,
        "hdl": hdl,
        "copy_stream": copy_stream,
        "compute_stream": compute_stream,
        "stage_bufs": stage_bufs,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(A_local: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized()
    assert A_local.is_cuda and B.is_cuda

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    M, K_local = A_local.shape
    K_B, N = B.shape
    K_global = world_size * K_local
    assert K_B == K_global

    device = A_local.device
    dtype = A_local.dtype

    # Compile extension on rank 0 first
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    res = _get_resources(M, K_local, dtype, device, world_size)
    sym_buf = res["sym_buf"]
    hdl = res["hdl"]
    copy_stream = res["copy_stream"]
    compute_stream = res["compute_stream"]
    stage_bufs = res["stage_bufs"]

    # Publish A_local^T into symmetric buffer
    A_local_t = A_local.transpose(0, 1).contiguous()
    sym_buf.copy_(A_local_t)
    hdl.barrier(channel=0)

    B_t = B.transpose(0, 1).contiguous()  # [N, K]

    # Allocate output C^T [N, M]; we'll fill it row-strided by writing slices [N, K_local] @ ... no:
    # We compute C^T = B^T @ A_global^T => [N, K] @ [K, M] = [N, M]
    # We split along K: for each peer p, partial = B_t[:, p*Kl:(p+1)*Kl] @ A_p^T  (shape [N, M])
    # Sum over p.
    C_t = torch.zeros((N, M), device=device, dtype=dtype)

    current = torch.cuda.current_stream(device=device)
    # Make compute & copy streams wait for current state
    copy_stream.wait_stream(current)
    compute_stream.wait_stream(current)

    n_chunks = world_size
    copy_done_events = [torch.cuda.Event() for _ in range(n_chunks)]
    compute_done_events = [torch.cuda.Event() for _ in range(n_chunks)]

    # Process peers in a ring starting from local rank to keep first chunk free of P2P
    order = [(rank + i) % world_size for i in range(world_size)]

    for i, p in enumerate(order):
        stage = stage_bufs[i % 2]

        # Issue copy on copy_stream
        with torch.cuda.stream(copy_stream):
            # Prevent overwriting a stage that's still being consumed
            if i >= 2:
                copy_stream.wait_event(compute_done_events[i - 2])

            if p == rank:
                stage.copy_(sym_buf, non_blocking=True)
            else:
                peer_ptr = int(hdl.buffer_ptrs[p])
                ext.copy_from_peer_bf16(peer_ptr, stage, K_local * M)
            copy_done_events[i].record(copy_stream)

        # Compute on compute_stream
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(copy_done_events[i])
            B_slice = B_t[:, p * K_local:(p + 1) * K_local]  # [N, K_local]
            # partial = B_slice @ stage  -> [N, M]
            # Accumulate into C_t
            C_t.addmm_(B_slice, stage)
            compute_done_events[i].record(compute_stream)

    # Wait for all compute to finish on current stream
    current.wait_stream(compute_stream)
    current.wait_stream(copy_stream)

    # Final symmetric barrier so no rank exits before peers finish reading
    hdl.barrier(channel=1)

    C = C_t.transpose(0, 1).contiguous()
    return C