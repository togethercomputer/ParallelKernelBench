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

__global__ void cast_complex_to_bf162_kernel(
    const float2* __restrict__ in,
    __nv_bfloat162* __restrict__ out,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = __float22bfloat162_rn(in[idx]);
    }
}

__global__ void conj_pad_2d_bf16_kernel(
    const uint64_t* __restrict__ symm_ptrs,
    float2* __restrict__ out,
    int B, int N0_local, int N1_half, int N1,
    int rank, int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * N0_local * N1;
    if (idx >= total) return;

    int i1 = idx % N1;
    int tmp = idx / N1;
    int i0_local = tmp % N0_local;
    int b = tmp / N0_local;

    int N0 = N0_local * world_size;

    if (i1 < N1_half) {
        const __nv_bfloat162* local_x = reinterpret_cast<const __nv_bfloat162*>(symm_ptrs[rank]);
        __nv_bfloat162 val = local_x[(b * N0_local + i0_local) * N1_half + i1];
        out[idx] = __bfloat1622float2(val);
    } else {
        int i0 = rank * N0_local + i0_local;
        int j0 = (i0 == 0) ? 0 : N0 - i0;
        int j0_rank = j0 / N0_local;
        int j0_local = j0 % N0_local;
        int orig_i1 = N1 - i1;

        const __nv_bfloat162* remote_x = reinterpret_cast<const __nv_bfloat162*>(symm_ptrs[j0_rank]);
        __nv_bfloat162 val = remote_x[(b * N0_local + j0_local) * N1_half + orig_i1];
        
        float2 fval = __bfloat1622float2(val);
        fval.y = -fval.y; // Complex conjugate
        out[idx] = fval;
    }
}

__global__ void transpose_bf16_kernel(
    const uint64_t* __restrict__ symm_ptrs,
    float2* __restrict__ out,
    int B, int N0_local, int N1, int N1_local,
    int rank, int world_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int N0 = N0_local * world_size;
    int64_t total = (int64_t)B * N0 * N1_local;
    if (idx >= total) return;

    int i1_local = idx % N1_local;
    int tmp = idx / N1_local;
    int i0 = tmp % N0;
    int b = tmp / N0;

    int j0_rank = i0 / N0_local;
    int i0_local = i0 % N0_local;
    int i1 = rank * N1_local + i1_local;

    const __nv_bfloat162* remote_x1 = reinterpret_cast<const __nv_bfloat162*>(symm_ptrs[j0_rank]);
    __nv_bfloat162 val = remote_x1[(b * N0_local + i0_local) * N1 + i1];
    out[idx] = __bfloat1622float2(val);
}

void launch_cast_complex_to_bf162(torch::Tensor in, torch::Tensor out) {
    int64_t numel = in.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cast_complex_to_bf162_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(in.data_ptr<c10::complex<float>>()),
        reinterpret_cast<__nv_bfloat162*>(out.data_ptr<at::BFloat16>()),
        numel
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_conj_pad_2d_bf16(
    torch::Tensor symm_ptrs_tensor,
    torch::Tensor out,
    int B, int N0_local, int N1_half, int N1,
    int rank, int world_size
) {
    int64_t total = (int64_t)B * N0_local * N1;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    conj_pad_2d_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(symm_ptrs_tensor.data_ptr<int64_t>()),
        reinterpret_cast<float2*>(out.data_ptr<c10::complex<float>>()),
        B, N0_local, N1_half, N1, rank, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_transpose_bf16(
    torch::Tensor symm_ptrs_tensor,
    torch::Tensor out,
    int B, int N0_local, int N1, int N1_local,
    int rank, int world_size
) {
    int N0 = N0_local * world_size;
    int64_t total = (int64_t)B * N0 * N1_local;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    transpose_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(symm_ptrs_tensor.data_ptr<int64_t>()),
        reinterpret_cast<float2*>(out.data_ptr<c10::complex<float>>()),
        B, N0_local, N1, N1_local, rank, world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_cast_complex_to_bf162", &launch_cast_complex_to_bf162, "Cast complex64 to bfloat162 pairs");
    m.def("launch_conj_pad_2d_bf16", &launch_conj_pad_2d_bf16, "UVA conjugate pad 2d fetching bf16 pairs");
    m.def("launch_transpose_bf16", &launch_transpose_bf16, "UVA all-to-all transpose reading bf16 pairs");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("physicsnemo_irfft_bf16_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(key: str, shape: tuple, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    global _symm_cache
    if key in _symm_cache:
        buf, hdl, ptrs = _symm_cache[key]
        if buf.shape == shape and buf.dtype == dtype:
            return buf, hdl, ptrs

    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _symm_cache[key] = (buf, hdl, ptrs)
    return buf, hdl, ptrs

@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Optional[Sequence[int]],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if x.dtype != torch.complex64:
        x = x.to(torch.complex64)

    dim0, dim1 = int(dim[0]) % x.ndim, int(dim[1]) % x.ndim

    if s is not None:
        first_dim_size = int(s[0])
        last_dim_size = int(s[1])
    else:
        first_dim_size = int(x.shape[dim0])
        last_dim_size = int(2 * (x.shape[dim1] - 1))

    # Permute spatial dimensions to the innermost boundary for contiguous 3D C++ processing 
    perms = [i for i in range(x.ndim) if i not in (dim0, dim1)] + [dim0, dim1]
    x_perm = x.permute(perms).contiguous()

    N0_local = x_perm.shape[-2]
    N1_half = x_perm.shape[-1]
    N0 = N0_local * world_size
    N1 = last_dim_size
    N1_local = N1 // world_size
    B = x_perm.numel() // (N0_local * N1_half)

    ext = _get_ext()

    # Step 1: Push input shard as bfloat16 to Symmetric Memory
    x_symm_shape = (B, N0_local, N1_half, 2)
    x_symm, x_hdl, x_ptrs = _get_symm_state("x", x_symm_shape, torch.bfloat16, x.device, group)
    ext.launch_cast_complex_to_bf162(x_perm, x_symm)
    x_hdl.barrier(channel=0)

    # Step 2: Hermitian Symmetry Rebuilding (Direct remote fetch, fully bypasses gather & flips)
    x_pad_perm = torch.empty((B, N0_local, N1), dtype=torch.complex64, device=x.device)
    ext.launch_conj_pad_2d_bf16(x_ptrs, x_pad_perm, B, N0_local, N1_half, N1, rank, world_size)

    # Step 3: Complex-to-complex IFFT on dim1 
    x1_perm = torch.fft.ifft(x_pad_perm, n=N1, dim=-1, norm=norm)

    # Step 4: Push transformed chunks as bfloat16 back to Symmetric Memory
    x1_symm_shape = (B, N0_local, N1, 2)
    x1_symm, x1_hdl, x1_ptrs = _get_symm_state("x1", x1_symm_shape, torch.bfloat16, x.device, group)
    ext.launch_cast_complex_to_bf162(x1_perm, x1_symm)
    x1_hdl.barrier(channel=0)

    # Step 5: All-to-all spatial transpose via UVA reads
    x1_tran_perm = torch.empty((B, N0, N1_local), dtype=torch.complex64, device=x.device)
    ext.launch_transpose_bf16(x1_ptrs, x1_tran_perm, B, N0_local, N1, N1_local, rank, world_size)

    # Step 6: Complex-to-complex IFFT on dim0 and extraction of reals
    x2_perm = torch.fft.ifft(x1_tran_perm, n=first_dim_size, dim=-2, norm=norm)
    out_perm = torch.real(x2_perm).contiguous()

    # Step 7: Inverse permutation to restore original spatial dimensions positions
    inv_perms = [0] * x.ndim
    for i, p in enumerate(perms):
        inv_perms[p] = i

    out_shape_permuted = [x.shape[i] for i in perms[:-2]] + [first_dim_size, N1_local]
    out_perm = out_perm.view(out_shape_permuted)
    
    return out_perm.permute(inv_perms).contiguous()