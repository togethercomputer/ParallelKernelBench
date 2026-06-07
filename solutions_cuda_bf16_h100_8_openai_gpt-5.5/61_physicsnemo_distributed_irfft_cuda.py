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

static inline int div_up_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

__device__ __forceinline__ float2 c_conj(float2 v) {
    return make_float2(v.x, -v.y);
}

// -----------------------------------------------------------------------------
// Build the post-_conj_pad_2d local shard directly from symmetric peer input.
//
// Input layout after Python canonicalization:
//   x_symm[rank] : [outer, a_local, h_half] complex64 contiguous
//
// Output:
//   x_pad        : [outer, a_local, last_dim_size] complex64 contiguous
//
// Matches reference _conj_pad_2d:
//   - k < h_half: local original row
//   - k >= h_half: conj of column last_dim_size-k from Hermitian partner row
//                  partner row is 0 for global row 0, else n0_comm-global_row.
// -----------------------------------------------------------------------------
__global__ void hermitian_pad_from_symm_c64_kernel(
    const long long* __restrict__ in_ptrs,
    float2* __restrict__ out,
    int64_t outer,
    int a_local,
    int h_half,
    int last_dim_size,
    int world_size,
    int rank
) {
    const int n0_comm = a_local * world_size;
    const int64_t total = outer * (int64_t)a_local * (int64_t)last_dim_size;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int k = (int)(idx % last_dim_size);
        int64_t t = idx / last_dim_size;
        int i = (int)(t % a_local);
        int64_t o = t / a_local;

        float2 v = make_float2(0.0f, 0.0f);

        if (k < h_half) {
            const float2* local =
                reinterpret_cast<const float2*>((uintptr_t)in_ptrs[rank]);
            v = local[(o * a_local + i) * (int64_t)h_half + k];
        } else {
            const int src_col = last_dim_size - k;
            if (src_col >= 0 && src_col < h_half) {
                const int global_row = rank * a_local + i;
                const int partner_global = (global_row == 0) ? 0 : (n0_comm - global_row);
                const int src_rank = partner_global / a_local;
                const int src_i = partner_global - src_rank * a_local;

                const float2* src =
                    reinterpret_cast<const float2*>((uintptr_t)in_ptrs[src_rank]);
                float2 raw = src[(o * a_local + src_i) * (int64_t)h_half + src_col];
                v = c_conj(raw);
            }
        }

        out[idx] = v;
    }
}

// -----------------------------------------------------------------------------
// Symmetric-memory all-to-all transpose.
//
// Reference path:
//   send chunks split along dim1/last FFT dimension
//   all_to_all
//   cat received chunks along dim0/first FFT dimension
//
// Input per source rank:
//   x1_symm[src] : [outer, a_local, last_dim_size]
//
// Output on this rank:
//   x_tran       : [outer, a_local * world_size, b_local]
// where b_local = last_dim_size / world_size and this rank owns columns
// [rank*b_local, (rank+1)*b_local).
// -----------------------------------------------------------------------------
__global__ void alltoall_transpose_from_symm_c64_kernel(
    const long long* __restrict__ x1_ptrs,
    float2* __restrict__ out,
    int64_t outer,
    int a_local,
    int last_dim_size,
    int world_size,
    int rank
) {
    const int n0_comm = a_local * world_size;
    const int b_local = last_dim_size / world_size;
    const int64_t total = outer * (int64_t)n0_comm * (int64_t)b_local;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int j = (int)(idx % b_local);
        int64_t t = idx / b_local;
        int g = (int)(t % n0_comm);
        int64_t o = t / n0_comm;

        const int src_rank = g / a_local;
        const int src_i = g - src_rank * a_local;
        const int src_col = rank * b_local + j;

        const float2* src =
            reinterpret_cast<const float2*>((uintptr_t)x1_ptrs[src_rank]);

        out[idx] = src[(o * a_local + src_i) * (int64_t)last_dim_size + src_col];
    }
}

// -----------------------------------------------------------------------------
// Final real extraction from complex64 to float32.
// -----------------------------------------------------------------------------
__global__ void real_extract_c64_kernel(
    const float2* __restrict__ x,
    float* __restrict__ out,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        out[idx] = x[idx].x;
    }
}

void hermitian_pad_from_symm_c64(
    torch::Tensor in_ptrs,
    torch::Tensor out,
    int64_t outer,
    int a_local,
    int h_half,
    int last_dim_size,
    int world_size,
    int rank
) {
    TORCH_CHECK(in_ptrs.is_cuda(), "in_ptrs must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.scalar_type() == torch::kComplexFloat, "out must be complex64");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int threads = 256;
    int blocks = div_up_i64(outer * (int64_t)a_local * (int64_t)last_dim_size, threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    hermitian_pad_from_symm_c64_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(in_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<float2*>(out.data_ptr<c10::complex<float>>()),
        outer,
        a_local,
        h_half,
        last_dim_size,
        world_size,
        rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void alltoall_transpose_from_symm_c64(
    torch::Tensor x1_ptrs,
    torch::Tensor out,
    int64_t outer,
    int a_local,
    int last_dim_size,
    int world_size,
    int rank
) {
    TORCH_CHECK(x1_ptrs.is_cuda(), "x1_ptrs must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.scalar_type() == torch::kComplexFloat, "out must be complex64");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(last_dim_size % world_size == 0, "last_dim_size must divide world_size");

    const int b_local = last_dim_size / world_size;
    const int n0_comm = a_local * world_size;

    const int threads = 256;
    int blocks = div_up_i64(outer * (int64_t)n0_comm * (int64_t)b_local, threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    alltoall_transpose_from_symm_c64_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(x1_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<float2*>(out.data_ptr<c10::complex<float>>()),
        outer,
        a_local,
        last_dim_size,
        world_size,
        rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void real_extract_c64(
    torch::Tensor x,
    torch::Tensor out,
    int64_t n
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kComplexFloat, "x must be complex64");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int threads = 256;
    int blocks = div_up_i64(n, threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    real_extract_c64_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(x.data_ptr<c10::complex<float>>()),
        out.data_ptr<float>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hermitian_pad_from_symm_c64", &hermitian_pad_from_symm_c64,
          "Hermitian pad directly from symmetric peer shards, complex64");
    m.def("alltoall_transpose_from_symm_c64", &alltoall_transpose_from_symm_c64,
          "All-to-all transpose via symmetric UVA peer reads, complex64");
    m.def("real_extract_c64", &real_extract_c64,
          "Extract real component complex64 -> float32");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "physicsnemo_dist_irfft_symm_cuda_c64_ext",
            CUDA_SRC,
        )
    return _ext


_resource_cache = {}


def _prod(xs):
    p = 1
    for v in xs:
        p *= int(v)
    return p


def _canonicalize_dims(ndim: int, dim: Sequence[int]):
    d0 = int(dim[0]) % ndim
    d1 = int(dim[1]) % ndim
    if d0 == d1:
        raise ValueError("dim entries must be distinct")
    others = [i for i in range(ndim) if i != d0 and i != d1]
    perm = others + [d0, d1]
    inv_perm = [0] * ndim
    for new_i, old_i in enumerate(perm):
        inv_perm[old_i] = new_i
    return d0, d1, perm, inv_perm


def _get_resources(
    x_shape,
    x_dtype,
    device,
    last_dim_size: int,
    world_size: int,
):
    key = (tuple(int(v) for v in x_shape), x_dtype, device, int(last_dim_size), int(world_size))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    outer_shape = tuple(int(v) for v in x_shape[:-2])
    a_local = int(x_shape[-2])

    input_symm = symm_mem.empty(x_shape, device=device, dtype=x_dtype)
    input_hdl = symm_mem.rendezvous(input_symm, dist.group.WORLD)
    input_ptrs = torch.tensor(input_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    x_pad_shape = outer_shape + (a_local, int(last_dim_size))
    x_pad = torch.empty(x_pad_shape, device=device, dtype=x_dtype)

    x1_symm = symm_mem.empty(x_pad_shape, device=device, dtype=x_dtype)
    x1_hdl = symm_mem.rendezvous(x1_symm, dist.group.WORLD)
    x1_ptrs = torch.tensor(x1_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    b_local = int(last_dim_size) // int(world_size)
    x_tran_shape = outer_shape + (a_local * int(world_size), b_local)
    x_tran = torch.empty(x_tran_shape, device=device, dtype=x_dtype)

    res = {
        "input_symm": input_symm,
        "input_hdl": input_hdl,
        "input_ptrs": input_ptrs,
        "x_pad": x_pad,
        "x1_symm": x1_symm,
        "x1_hdl": x1_hdl,
        "x1_ptrs": x1_ptrs,
        "x_tran": x_tran,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Optional[Sequence[int]],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD

    if not dist.is_initialized():
        # Single-rank correctness path: no distributed collectives involved.
        dim0, dim1 = int(dim[0]), int(dim[1])
        if s is not None:
            first_dim_size = int(s[0])
            last_dim_size = int(s[1])
        else:
            first_dim_size = int(x.shape[dim0])
            last_dim_size = int(2 * (x.shape[dim1] - 1))

        full = torch.fft.irfft2(x, s=(first_dim_size, last_dim_size), dim=(dim0, dim1), norm=norm)
        return full.contiguous()

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if group is not dist.group.WORLD:
        # Symmetric-memory rendezvous is performed on WORLD in this implementation.
        # The common benchmark path passes WORLD/None.
        raise RuntimeError("custom symmetric-memory IRFFT currently expects group=WORLD/None")

    if not x.is_cuda:
        raise RuntimeError("x must be a CUDA tensor")
    if x.dtype != torch.complex64:
        raise RuntimeError("custom distributed IRFFT path expects complex64 input")

    _get_ext()

    ndim = x.ndim
    dim0, dim1, perm, inv_perm = _canonicalize_dims(ndim, dim)

    if s is not None:
        first_dim_size = int(s[0])
        last_dim_size = int(s[1])
    else:
        first_dim_size = int(x.shape[dim0])
        last_dim_size = int(2 * (x.shape[dim1] - 1))

    if last_dim_size % world_size != 0:
        raise RuntimeError("last_dim_size must be divisible by world_size for all-to-all transpose")

    # Canonical contiguous layout: [outer..., dim0_local, dim1_half]
    x_c = x.permute(perm).contiguous()

    outer = _prod(x_c.shape[:-2])
    a_local = int(x_c.shape[-2])
    h_half = int(x_c.shape[-1])

    res = _get_resources(
        tuple(x_c.shape),
        x_c.dtype,
        x_c.device,
        last_dim_size,
        world_size,
    )

    input_symm = res["input_symm"]
    input_hdl = res["input_hdl"]
    input_ptrs = res["input_ptrs"]
    x_pad = res["x_pad"]
    x1_symm = res["x1_symm"]
    x1_hdl = res["x1_hdl"]
    x1_ptrs = res["x1_ptrs"]
    x_tran = res["x_tran"]

    # Publish local half-spectrum once; Hermitian completion reads peer rows directly.
    input_symm.copy_(x_c)
    input_hdl.barrier(channel=0)

    _get_ext().hermitian_pad_from_symm_c64(
        input_ptrs,
        x_pad,
        int(outer),
        int(a_local),
        int(h_half),
        int(last_dim_size),
        int(world_size),
        int(rank),
    )

    # First inverse FFT along the now-replicated second transform dimension.
    x1 = torch.fft.ifft(x_pad, n=last_dim_size, dim=-1, norm=norm)

    # Publish x1 for direct peer-read all-to-all transpose.
    x1_symm.copy_(x1)
    x1_hdl.barrier(channel=1)

    _get_ext().alltoall_transpose_from_symm_c64(
        x1_ptrs,
        x_tran,
        int(outer),
        int(a_local),
        int(last_dim_size),
        int(world_size),
        int(rank),
    )

    # Second inverse FFT along first transform dimension.
    x2 = torch.fft.ifft(x_tran, n=first_dim_size, dim=-2, norm=norm).contiguous()

    y_perm = torch.empty(x2.shape, device=x2.device, dtype=torch.float32)
    _get_ext().real_extract_c64(x2, y_perm, int(x2.numel()))

    # Restore original dimension order, with dim0 full and dim1 sharded.
    y = y_perm.permute(inv_perm).contiguous()
    return y