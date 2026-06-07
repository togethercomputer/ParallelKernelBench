"""
All-gather using symmetric memory + custom CUDA kernel that performs
direct peer-to-peer reads via UVA pointers from symm_mem rendezvous.
Each block copies from one peer's symmetric buffer into the output slice.
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

// Each block copies from one peer rank's symmetric buffer into the
// corresponding slice of the output. Uses 16-byte vectorized loads/stores
// when possible.

__global__ void allgather_copy_kernel(
    const long long* __restrict__ peer_ptrs,
    char* __restrict__ out,
    int64_t bytes_per_rank,
    int world_size
) {
    int rank_id = blockIdx.y;
    if (rank_id >= world_size) return;

    const char* src = reinterpret_cast<const char*>(peer_ptrs[rank_id]);
    char* dst = out + (int64_t)rank_id * bytes_per_rank;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // 16-byte vectorized path
    int64_t n_vec = bytes_per_rank / 16;
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);

    for (int64_t i = tid; i < n_vec; i += stride) {
        dst4[i] = src4[i];
    }

    // tail
    int64_t tail_start = n_vec * 16;
    for (int64_t i = tail_start + tid; i < bytes_per_rank; i += stride) {
        dst[i] = src[i];
    }
}

void launch_allgather_copy(
    torch::Tensor peer_ptrs,
    torch::Tensor out,
    int64_t bytes_per_rank,
    int world_size
) {
    const long long* d_ptrs = (const long long*)peer_ptrs.data_ptr<int64_t>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = 256;
    int64_t n_vec = bytes_per_rank / 16;
    int blocks_x = (int)((n_vec + threads - 1) / threads);
    if (blocks_x < 1) blocks_x = 1;
    if (blocks_x > 256) blocks_x = 256;

    dim3 grid(blocks_x, world_size, 1);
    allgather_copy_kernel<<<grid, threads, 0, stream>>>(
        d_ptrs,
        (char*)out.data_ptr(),
        bytes_per_rank,
        world_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_allgather_copy", &launch_allgather_copy,
          "All-gather via P2P UVA reads from symmetric memory");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_allgather_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _get_resources(shape, dtype, device, world_size):
    key = (tuple(shape), dtype, device, world_size)
    if key in _resource_cache:
        return _resource_cache[key]

    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)

    out_shape = (world_size,) + tuple(shape)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    res = (buf, hdl, out, ptrs_tensor)
    _resource_cache[key] = res
    return res


# Warm up extension once
_warmed = False


def _warmup():
    global _warmed
    if not _warmed:
        _get_ext()
        _warmed = True


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda

    _warmup()

    input_tensor = tensor.contiguous()
    world_size = dist.get_world_size()

    buf, hdl, out, ptrs_tensor = _get_resources(
        input_tensor.shape, input_tensor.dtype, input_tensor.device, world_size
    )

    # Stage local input into symmetric buffer
    buf.copy_(input_tensor)

    # Synchronize across ranks: ensure every peer's symmetric buffer holds
    # the new local data before we begin reading.
    hdl.barrier(channel=0)

    bytes_per_rank = input_tensor.numel() * input_tensor.element_size()
    _get_ext().launch_allgather_copy(ptrs_tensor, out, bytes_per_rank, world_size)

    # Ensure peers don't overwrite their buffer until we've finished reading.
    hdl.barrier(channel=1)

    return out