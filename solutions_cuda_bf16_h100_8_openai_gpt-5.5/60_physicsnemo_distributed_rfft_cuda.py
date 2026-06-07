from typing import Optional, Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define MAX_NDIM 16

template <typename T>
__global__ void pack_complex_kernel(const T* __restrict__ src,
                                    T* __restrict__ dst,
                                    int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ src,
                                   float* __restrict__ dst,
                                   int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        dst[i] = __bfloat162float(src[i]);
    }
}

template <typename T>
__global__ void alltoall_transpose_gather_kernel(
    const int64_t* __restrict__ peer_raw_ptrs,
    T* __restrict__ out,
    const int64_t* __restrict__ in_sizes,
    int ndim,
    int dim0,
    int dim1,
    int rank,
    int world_size,
    int64_t n_out
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_grid = (int64_t)gridDim.x * blockDim.x;

    const int64_t n0 = in_sizes[dim0];
    const int64_t n1 = in_sizes[dim1];
    const int64_t chunk0 = n0 / (int64_t)world_size;

    for (int64_t linear = tid; linear < n_out; linear += stride_grid) {
        int64_t tmp = linear;
        int64_t coord[MAX_NDIM];

        #pragma unroll
        for (int d = MAX_NDIM - 1; d >= 0; --d) {
            if (d < ndim) coord[d] = 0;
        }

        for (int d = ndim - 1; d >= 0; --d) {
            int64_t extent = in_sizes[d];
            if (d == dim0) {
                extent = chunk0;
            } else if (d == dim1) {
                extent = n1 * (int64_t)world_size;
            }
            coord[d] = tmp % extent;
            tmp /= extent;
        }

        const int src_rank = (int)(coord[dim1] / n1);
        const int64_t src_dim1 = coord[dim1] - (int64_t)src_rank * n1;
        const int64_t src_dim0 = (int64_t)rank * chunk0 + coord[dim0];

        int64_t src_off = 0;
        int64_t contig_stride = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t c = coord[d];
            if (d == dim0) {
                c = src_dim0;
            } else if (d == dim1) {
                c = src_dim1;
            }
            src_off += c * contig_stride;
            contig_stride *= in_sizes[d];
        }

        const T* __restrict__ remote =
            reinterpret_cast<const T*>(static_cast<uintptr_t>(peer_raw_ptrs[src_rank]));
        out[linear] = remote[src_off];
    }
}

template <typename T>
__global__ void truncate_kernel(const T* __restrict__ src,
                                T* __restrict__ out,
                                const int64_t* __restrict__ src_sizes,
                                int ndim,
                                int dim,
                                int64_t keep,
                                int64_t n_out) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_grid = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = tid; linear < n_out; linear += stride_grid) {
        int64_t tmp = linear;
        int64_t coord[MAX_NDIM];

        #pragma unroll
        for (int d = MAX_NDIM - 1; d >= 0; --d) {
            if (d < ndim) coord[d] = 0;
        }

        for (int d = ndim - 1; d >= 0; --d) {
            int64_t extent = (d == dim) ? keep : src_sizes[d];
            coord[d] = tmp % extent;
            tmp /= extent;
        }

        int64_t src_off = 0;
        int64_t contig_stride = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            src_off += coord[d] * contig_stride;
            contig_stride *= src_sizes[d];
        }

        out[linear] = src[src_off];
    }
}

static inline int launch_blocks(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

void bf16_to_f32(torch::Tensor src, torch::Tensor dst, int64_t n) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(src.dtype() == torch::kBFloat16, "src must be bfloat16");
    TORCH_CHECK(dst.dtype() == torch::kFloat32, "dst must be float32");

    const at::cuda::CUDAGuard guard(dst.device());
    const int threads = 256;
    const int blocks = launch_blocks(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
        dst.data_ptr<float>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_complex(torch::Tensor src, torch::Tensor raw, int64_t n, int dtype_enum) {
    TORCH_CHECK(src.is_cuda() && raw.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(src.is_contiguous() && raw.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(raw.numel() >= 2 * n, "raw symmetric buffer too small");

    const at::cuda::CUDAGuard guard(raw.device());
    const int threads = 256;
    const int blocks = launch_blocks(n, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        TORCH_CHECK(src.dtype() == torch::kComplexFloat, "src must be complex64");
        TORCH_CHECK(raw.dtype() == torch::kFloat32, "raw must be float32 for complex64");
        pack_complex_kernel<float2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float2*>(src.data_ptr()),
            reinterpret_cast<float2*>(raw.data_ptr<float>()),
            n
        );
    } else {
        TORCH_CHECK(src.dtype() == torch::kComplexDouble, "src must be complex128");
        TORCH_CHECK(raw.dtype() == torch::kFloat64, "raw must be float64 for complex128");
        pack_complex_kernel<double2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const double2*>(src.data_ptr()),
            reinterpret_cast<double2*>(raw.data_ptr<double>()),
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void alltoall_transpose_gather(torch::Tensor peer_raw_ptrs,
                               torch::Tensor out,
                               torch::Tensor in_sizes,
                               int ndim,
                               int dim0,
                               int dim1,
                               int rank,
                               int world_size,
                               int64_t n_out,
                               int dtype_enum) {
    TORCH_CHECK(peer_raw_ptrs.is_cuda() && out.is_cuda() && in_sizes.is_cuda(),
                "CUDA tensors required");
    TORCH_CHECK(peer_raw_ptrs.dtype() == torch::kInt64, "peer ptrs must be int64");
    TORCH_CHECK(in_sizes.dtype() == torch::kInt64, "sizes must be int64");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(ndim > 0 && ndim <= MAX_NDIM, "unsupported ndim");

    const at::cuda::CUDAGuard guard(out.device());
    const int threads = 256;
    const int blocks = launch_blocks(n_out, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        TORCH_CHECK(out.dtype() == torch::kComplexFloat, "out must be complex64");
        alltoall_transpose_gather_kernel<float2><<<blocks, threads, 0, stream>>>(
            peer_raw_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<float2*>(out.data_ptr()),
            in_sizes.data_ptr<int64_t>(),
            ndim, dim0, dim1, rank, world_size, n_out
        );
    } else {
        TORCH_CHECK(out.dtype() == torch::kComplexDouble, "out must be complex128");
        alltoall_transpose_gather_kernel<double2><<<blocks, threads, 0, stream>>>(
            peer_raw_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<double2*>(out.data_ptr()),
            in_sizes.data_ptr<int64_t>(),
            ndim, dim0, dim1, rank, world_size, n_out
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void truncate_complex(torch::Tensor src,
                      torch::Tensor out,
                      torch::Tensor src_sizes,
                      int ndim,
                      int dim,
                      int64_t keep,
                      int64_t n_out,
                      int dtype_enum) {
    TORCH_CHECK(src.is_cuda() && out.is_cuda() && src_sizes.is_cuda(),
                "CUDA tensors required");
    TORCH_CHECK(src.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(src_sizes.dtype() == torch::kInt64, "sizes must be int64");
    TORCH_CHECK(ndim > 0 && ndim <= MAX_NDIM, "unsupported ndim");

    const at::cuda::CUDAGuard guard(out.device());
    const int threads = 256;
    const int blocks = launch_blocks(n_out, threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        TORCH_CHECK(src.dtype() == torch::kComplexFloat && out.dtype() == torch::kComplexFloat,
                    "complex64 tensors required");
        truncate_kernel<float2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float2*>(src.data_ptr()),
            reinterpret_cast<float2*>(out.data_ptr()),
            src_sizes.data_ptr<int64_t>(),
            ndim, dim, keep, n_out
        );
    } else {
        TORCH_CHECK(src.dtype() == torch::kComplexDouble && out.dtype() == torch::kComplexDouble,
                    "complex128 tensors required");
        truncate_kernel<double2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const double2*>(src.data_ptr()),
            reinterpret_cast<double2*>(out.data_ptr()),
            src_sizes.data_ptr<int64_t>(),
            ndim, dim, keep, n_out
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_to_f32", &bf16_to_f32, "BF16 to FP32 conversion kernel");
    m.def("pack_complex", &pack_complex, "Pack complex tensor into raw symmetric buffer");
    m.def("alltoall_transpose_gather", &alltoall_transpose_gather,
          "UVA symmetric-memory all-to-all transpose gather");
    m.def("truncate_complex", &truncate_complex, "Contiguous complex truncate kernel");
}
'''


_ext = None
_a2a_cache = {}
_meta_cache = {}
_trunc_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("physicsnemo_dist_rfft_symm_cuda_ext", CUDA_SRC)
    return _ext


def _prod(shape) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return int(n)


def _dtype_enum_complex(dtype: torch.dtype) -> int:
    if dtype == torch.complex64:
        return 0
    if dtype == torch.complex128:
        return 1
    raise TypeError(f"unsupported FFT complex dtype: {dtype}")


def _raw_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise TypeError(f"unsupported FFT complex dtype: {dtype}")


def _meta_tensor(shape, device):
    key = (tuple(int(x) for x in shape), device)
    t = _meta_cache.get(key)
    if t is None:
        t = torch.tensor(list(key[0]), device=device, dtype=torch.int64)
        _meta_cache[key] = t
    return t


def _get_a2a_resources(x1_shape, complex_dtype, device, group, dim0, dim1):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    x1_shape = tuple(int(v) for v in x1_shape)
    key = (x1_shape, complex_dtype, device, id(group), int(dim0), int(dim1), world_size)

    cached = _a2a_cache.get(key)
    if cached is not None:
        return cached

    n0 = x1_shape[dim0]
    n1 = x1_shape[dim1]
    assert n0 % world_size == 0, "dim[0] FFT extent must be divisible by world size"

    raw_dtype = _raw_dtype_for_complex(complex_dtype)
    raw_numel = _prod(x1_shape) * 2
    raw_symm = symm_mem.empty((raw_numel,), device=device, dtype=raw_dtype)
    hdl = symm_mem.rendezvous(raw_symm, group)

    out_shape = list(x1_shape)
    out_shape[dim0] = n0 // world_size
    out_shape[dim1] = n1 * world_size
    out = torch.empty(tuple(out_shape), device=device, dtype=complex_dtype)

    ptrs = torch.tensor(list(hdl.buffer_ptrs), device=device, dtype=torch.int64)
    sizes = _meta_tensor(x1_shape, device)

    cached = {
        "raw": raw_symm,
        "hdl": hdl,
        "out": out,
        "ptrs": ptrs,
        "sizes": sizes,
        "rank": rank,
        "world_size": world_size,
    }
    _a2a_cache[key] = cached
    return cached


def _get_trunc_out(src_shape, dtype, device, dim, keep):
    src_shape = tuple(int(v) for v in src_shape)
    out_shape = list(src_shape)
    out_shape[dim] = int(keep)
    out_shape = tuple(out_shape)
    key = (src_shape, dtype, device, int(dim), int(keep))
    cached = _trunc_cache.get(key)
    if cached is not None:
        return cached

    out = torch.empty(out_shape, device=device, dtype=dtype)
    sizes = _meta_tensor(src_shape, device)
    cached = (out, sizes)
    _trunc_cache[key] = cached
    return cached


def _bf16_to_f32_contiguous(x: torch.Tensor) -> torch.Tensor:
    x_contig = x if x.is_contiguous() else x.contiguous()
    y = torch.empty(x_contig.shape, device=x_contig.device, dtype=torch.float32)
    _get_ext().bf16_to_f32(x_contig, y, x_contig.numel())
    return y


def _real_fft_input(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16:
        return _bf16_to_f32_contiguous(x)
    if x.dtype == torch.float16:
        return x.contiguous().to(torch.float32)
    if x.dtype in (torch.float32, torch.float64):
        return x if x.is_contiguous() else x.contiguous()
    return x.contiguous().to(torch.float32)


@torch.no_grad()
def solution(
    x: torch.Tensor,
    s: Sequence[int],
    dim: Sequence[int],
    norm: str = "ortho",
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    ndim = x.ndim
    dim0 = int(dim[0]) % ndim
    dim1 = int(dim[1]) % ndim
    s0 = int(s[0])
    s1 = int(s[1])

    ext = _get_ext()
    x_fft = _real_fft_input(x)

    # 1. Local FFT over the replicated dimension. cuFFT is retained for FFT math.
    x1 = torch.fft.fft(x_fft, n=s0, dim=dim0, norm=norm)
    if not x1.is_contiguous():
        x1 = x1.contiguous()

    if dist.is_initialized():
        world_size = dist.get_world_size(group)
    else:
        world_size = 1

    # 2. Symmetric-memory all-to-all transpose: direct UVA gathers from peers.
    if world_size > 1:
        dtype_enum = _dtype_enum_complex(x1.dtype)
        res = _get_a2a_resources(x1.shape, x1.dtype, x1.device, group, dim0, dim1)

        raw = res["raw"]
        hdl = res["hdl"]
        out_tran = res["out"]
        ptrs = res["ptrs"]
        sizes = res["sizes"]
        rank = res["rank"]

        ext.pack_complex(x1, raw, x1.numel(), dtype_enum)

        # Publish packed local FFT data before peer UVA loads.
        hdl.barrier(channel=0)

        ext.alltoall_transpose_gather(
            ptrs,
            out_tran,
            sizes,
            ndim,
            dim0,
            dim1,
            rank,
            world_size,
            out_tran.numel(),
            dtype_enum,
        )

        # Do not allow any rank to overwrite its symmetric buffer while peers may
        # still be reading it in the transpose kernel.
        hdl.barrier(channel=1)
        x1_tran = out_tran
    else:
        x1_tran = x1

    # 3. Local FFT over the now-replicated second transform dimension.
    x2 = torch.fft.fft(x1_tran, n=s1, dim=dim1, norm=norm)
    if not x2.is_contiguous():
        x2 = x2.contiguous()

    # 4. Custom contiguous half-spectrum truncation.
    keep = x2.shape[dim1] // 2 + 1
    dtype_enum = _dtype_enum_complex(x2.dtype)
    out, src_sizes = _get_trunc_out(x2.shape, x2.dtype, x2.device, dim1, keep)
    ext.truncate_complex(
        x2,
        out,
        src_sizes,
        ndim,
        dim1,
        int(keep),
        out.numel(),
        dtype_enum,
    )
    return out