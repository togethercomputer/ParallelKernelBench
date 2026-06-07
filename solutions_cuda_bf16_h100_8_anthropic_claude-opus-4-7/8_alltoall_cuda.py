"""
All-to-all using torch symmetric memory + custom CUDA kernel.

Each rank writes its full input into a symmetric buffer. After a barrier,
every rank reads its assigned chunk directly from each peer's symmetric
buffer via UVA peer pointers, performing the transpose on-device with
no host-side collective calls.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// vectorized copy: 16-byte chunks
__global__ void all_to_all_gather_kernel(
    const long long* __restrict__ peer_ptrs,  // [world_size] base ptrs of peer symm buffers
    char* __restrict__ out,                    // [world_size, chunk_bytes]
    int world_size,
    int rank,
    int64_t chunk_bytes
) {
    int peer = blockIdx.y;
    if (peer >= world_size) return;

    const char* src_base = (const char*)peer_ptrs[peer];
    // chunk index `rank` from peer's buffer
    const char* src = src_base + (int64_t)rank * chunk_bytes;
    char* dst = out + (int64_t)peer * chunk_bytes;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // 16B vectorized
    int64_t n16 = chunk_bytes / 16;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (int64_t i = tid; i < n16; i += stride) {
        d4[i] = s4[i];
    }
    // tail bytes
    int64_t tail_start = n16 * 16;
    for (int64_t i = tail_start + tid; i < chunk_bytes; i += stride) {
        dst[i] = src[i];
    }
}

void launch_all_to_all(
    torch::Tensor peer_ptrs,  // int64 [world_size]
    torch::Tensor out,
    int64_t world_size,
    int64_t rank,
    int64_t chunk_bytes
) {
    const long long* d_ptrs = (const long long*)peer_ptrs.data_ptr<int64_t>();
    char* d_out = (char*)out.data_ptr();

    int threads = 256;
    int64_t n16 = chunk_bytes / 16;
    int blocks_x = (int)std::min<int64_t>((n16 + threads - 1) / threads, 256);
    if (blocks_x < 1) blocks_x = 1;
    dim3 grid(blocks_x, (int)world_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    all_to_all_gather_kernel<<<grid, threads, 0, stream>>>(
        d_ptrs, d_out, (int)world_size, (int)rank, chunk_bytes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_all_to_all", &launch_all_to_all, "Symmetric memory all-to-all");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_all_to_all_ext", CUDA_SRC)
    return _ext


_cache = {}


def _get_resources(shape, dtype, device):
    key = (tuple(shape), dtype, device)
    if key in _cache:
        return _cache[key]

    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    out = torch.empty(shape, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, out, ptrs_tensor)
    _cache[key] = res
    return res


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized()
    inp = tensor.contiguous()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Ensure extension compiled (rank 0 first to avoid race)
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    buf, hdl, out, ptrs_tensor = _get_resources(inp.shape, inp.dtype, inp.device)
    buf.copy_(inp)

    # symmetric barrier ensures all ranks have written their input before reads
    hdl.barrier(channel=0)

    chunk_numel = inp[0].numel()
    chunk_bytes = chunk_numel * inp.element_size()

    ext.launch_all_to_all(ptrs_tensor, out.view(-1), world_size, rank, chunk_bytes)

    # ensure all peer reads complete before any rank's buffer can be reused
    hdl.barrier(channel=1)

    return out