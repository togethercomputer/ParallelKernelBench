"""
Distributed element-wise vector add across two ranks but using a custom CUDA kernel leveraging UVA. Element-wise add of this rank's buffer with one peer's buffer via symmetric memory.
Peer device memory is accessed through a UVA pointer from symm_mem rendezvous. World size must be 2 in this example.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void symmetric_add_kernel(
    const float* __restrict__ local_data,
    const float* __restrict__ remote_data,
    float* __restrict__ out,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = local_data[idx] + remote_data[idx];
    }
}

void symmetric_add_f32(
    torch::Tensor local,
    int64_t remote_ptr,
    torch::Tensor out,
    int64_t n
) {
    TORCH_CHECK(local.is_cuda(), "local must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(local.dtype() == torch::kFloat32, "local must be float32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(local.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");

    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const float* remote = reinterpret_cast<const float*>(static_cast<uintptr_t>(remote_ptr));

    symmetric_add_kernel<<<blocks, threads, 0, stream>>>(
        local.data_ptr<float>(),
        remote,
        out.data_ptr<float>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symmetric_add_f32", &symmetric_add_f32,
          "UVA symmetric add: local + remote float buffers");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symmetric_add_uva_ext", CUDA_SRC)
    return _ext


_symm_cache = None


def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype:
            return c["buf"], c["hdl"], c["out"]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = torch.empty(n, device=device, dtype=dtype)
    _symm_cache = {"n": n, "dtype": dtype, "buf": buf, "hdl": hdl, "out": out}
    return buf, hdl, out


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.is_cuda and tensor.is_contiguous()
    assert tensor.dtype == torch.float32
    assert dist.is_initialized()
    assert dist.get_world_size() == 2

    rank = dist.get_rank()
    peer = 1 - rank
    n = tensor.numel()

    if rank == 0:
        _get_ext()
    dist.barrier()

    buf, hdl, out = _get_symm_state(n, tensor.dtype, tensor.device)
    buf.copy_(tensor.reshape(-1))
    hdl.barrier(channel=0)

    remote_ptr = int(hdl.buffer_ptrs[peer])
    _get_ext().symmetric_add_f32(buf, remote_ptr, out, n)

    return out.reshape_as(tensor)
