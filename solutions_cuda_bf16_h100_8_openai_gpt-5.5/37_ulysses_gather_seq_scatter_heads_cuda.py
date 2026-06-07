from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
import torch.distributed._symmetric_memory as symm_mem

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

static inline int ceil_div_i64(int64_t a, int64_t b) {
    return (int)((a + b - 1) / b);
}

// -----------------------------------------------------------------------------
// Vectorized staging copy: regular CUDA tensor -> symmetric-memory tensor
// -----------------------------------------------------------------------------

__global__ void copy16_kernel(const char* __restrict__ src,
                              char* __restrict__ dst,
                              int64_t n16,
                              int64_t tail_start,
                              int64_t nbytes) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src);
    uint4* __restrict__ d4 = reinterpret_cast<uint4*>(dst);

    for (int64_t i = tid; i < n16; i += stride) {
        d4[i] = s4[i];
    }

    for (int64_t i = tail_start + tid; i < nbytes; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void copy_byte_kernel(const char* __restrict__ src,
                                 char* __restrict__ dst,
                                 int64_t nbytes) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; i < nbytes; i += stride) {
        dst[i] = src[i];
    }
}

void stage_copy(torch::Tensor src, torch::Tensor dst, int64_t nbytes) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "src/dst must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const char* s = reinterpret_cast<const char*>(src.data_ptr());
    char* d = reinterpret_cast<char*>(dst.data_ptr());

    int threads = 256;
    int blocks = (int)min<int64_t>(65535, (nbytes + 4095) / 4096);
    if (blocks < 1) blocks = 1;

    uintptr_t sp = reinterpret_cast<uintptr_t>(s);
    uintptr_t dp = reinterpret_cast<uintptr_t>(d);

    if (((sp | dp) & 15ULL) == 0ULL) {
        int64_t n16 = nbytes / 16;
        int64_t tail_start = n16 * 16;
        copy16_kernel<<<blocks, threads, 0, stream>>>(s, d, n16, tail_start, nbytes);
    } else {
        copy_byte_kernel<<<blocks, threads, 0, stream>>>(s, d, nbytes);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -----------------------------------------------------------------------------
// Fast path for Ulysses common layout:
// input  [B, S_local, H, D]
// output [B, S_out,   H/world, D]
// seq_dim=1, head_dim=2
//
// Each destination rank gathers its own head shard from every source rank.
// -----------------------------------------------------------------------------

__global__ void gather_4d_dim1_dim2_u16_kernel(
    const long long* __restrict__ ptrs,
    uint16_t* __restrict__ out,
    int64_t B,
    int64_t S,
    int64_t H,
    int64_t D,
    int64_t S_out,
    int64_t H_part,
    int rank
) {
    int64_t Dv = (D + 7) >> 3; // vectors of 8 bf16/half elements = 16 bytes
    int64_t total_vec = B * S_out * H_part * Dv;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = tid; linear < total_vec; linear += stride) {
        int64_t t = linear;

        int64_t d_vec = t % Dv;
        t /= Dv;
        int64_t h_out = t % H_part;
        t /= H_part;
        int64_t s_out = t % S_out;
        int64_t b = t / S_out;

        int src_rank = (int)(s_out / S);
        int64_t s_local = s_out - (int64_t)src_rank * S;
        int64_t h_in = (int64_t)rank * H_part + h_out;
        int64_t d = d_vec << 3;

        const uint16_t* __restrict__ src =
            reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(ptrs[src_rank]));

        int64_t in_elem = (((b * S + s_local) * H + h_in) * D + d);
        int64_t out_elem = (((b * S_out + s_out) * H_part + h_out) * D + d);

        if (d + 7 < D && ((in_elem | out_elem) & 7LL) == 0LL) {
            const uint4 v = *reinterpret_cast<const uint4*>(src + in_elem);
            *reinterpret_cast<uint4*>(out + out_elem) = v;
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (d + j < D) {
                    out[out_elem + j] = src[in_elem + j];
                }
            }
        }
    }
}

void launch_gather_4d_dim1_dim2_u16(
    torch::Tensor ptrs,
    torch::Tensor out,
    int64_t B,
    int64_t S,
    int64_t H,
    int64_t D,
    int64_t S_out,
    int64_t H_part,
    int rank
) {
    TORCH_CHECK(ptrs.is_cuda() && out.is_cuda(), "ptrs/out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    int threads = 256;
    int64_t Dv = (D + 7) >> 3;
    int64_t total_vec = B * S_out * H_part * Dv;
    int blocks = (int)min<int64_t>(65535, (total_vec + threads - 1) / threads);
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_4d_dim1_dim2_u16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        B, S, H, D, S_out, H_part, rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -----------------------------------------------------------------------------
// Generic contiguous-layout fallback for arbitrary ndim / seq_dim / head_dim.
// It is still a peer-UVA symmetric-memory all-to-all gather, just scalarized.
// -----------------------------------------------------------------------------

template<int ELEM_SIZE>
__global__ void gather_generic_kernel(
    const long long* __restrict__ ptrs,
    char* __restrict__ out,
    const int64_t* __restrict__ meta,
    int ndim,
    int seq_dim,
    int head_dim,
    int64_t S,
    int64_t H_part,
    int rank,
    int64_t total_out
) {
    const int64_t* in_shape = meta;
    const int64_t* out_shape = meta + ndim;
    const int64_t* in_stride = meta + 2 * ndim;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_grid = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total_out; idx += stride_grid) {
        int64_t tmp = idx;
        int src_rank = 0;
        int64_t in_off = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            int64_t coord = tmp % out_shape[d];
            tmp /= out_shape[d];

            int64_t in_coord = coord;
            if (d == seq_dim) {
                src_rank = (int)(coord / S);
                in_coord = coord - (int64_t)src_rank * S;
            } else if (d == head_dim) {
                in_coord = (int64_t)rank * H_part + coord;
            }
            in_off += in_coord * in_stride[d];
        }

        const char* __restrict__ src =
            reinterpret_cast<const char*>(static_cast<uintptr_t>(ptrs[src_rank]));

        const char* sp = src + in_off * ELEM_SIZE;
        char* dp = out + idx * ELEM_SIZE;

        #pragma unroll
        for (int b = 0; b < ELEM_SIZE; ++b) {
            dp[b] = sp[b];
        }
    }
}

__global__ void gather_generic_dynamic_kernel(
    const long long* __restrict__ ptrs,
    char* __restrict__ out,
    const int64_t* __restrict__ meta,
    int ndim,
    int seq_dim,
    int head_dim,
    int64_t S,
    int64_t H_part,
    int rank,
    int64_t total_out,
    int elem_size
) {
    const int64_t* in_shape = meta;
    const int64_t* out_shape = meta + ndim;
    const int64_t* in_stride = meta + 2 * ndim;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_grid = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total_out; idx += stride_grid) {
        int64_t tmp = idx;
        int src_rank = 0;
        int64_t in_off = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            int64_t coord = tmp % out_shape[d];
            tmp /= out_shape[d];

            int64_t in_coord = coord;
            if (d == seq_dim) {
                src_rank = (int)(coord / S);
                in_coord = coord - (int64_t)src_rank * S;
            } else if (d == head_dim) {
                in_coord = (int64_t)rank * H_part + coord;
            }
            in_off += in_coord * in_stride[d];
        }

        const char* __restrict__ src =
            reinterpret_cast<const char*>(static_cast<uintptr_t>(ptrs[src_rank]));

        const char* sp = src + in_off * elem_size;
        char* dp = out + idx * elem_size;

        for (int b = 0; b < elem_size; ++b) {
            dp[b] = sp[b];
        }
    }
}

void launch_gather_generic(
    torch::Tensor ptrs,
    torch::Tensor out,
    torch::Tensor meta,
    int ndim,
    int seq_dim,
    int head_dim,
    int64_t S,
    int64_t H_part,
    int rank,
    int64_t total_out,
    int elem_size
) {
    TORCH_CHECK(ptrs.is_cuda() && out.is_cuda() && meta.is_cuda(), "ptrs/out/meta must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    int threads = 256;
    int blocks = (int)min<int64_t>(65535, (total_out + threads - 1) / threads);
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* p = reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>());
    char* o = reinterpret_cast<char*>(out.data_ptr());
    const int64_t* m = reinterpret_cast<const int64_t*>(meta.data_ptr<int64_t>());

    if (elem_size == 2) {
        gather_generic_kernel<2><<<blocks, threads, 0, stream>>>(
            p, o, m, ndim, seq_dim, head_dim, S, H_part, rank, total_out
        );
    } else if (elem_size == 4) {
        gather_generic_kernel<4><<<blocks, threads, 0, stream>>>(
            p, o, m, ndim, seq_dim, head_dim, S, H_part, rank, total_out
        );
    } else if (elem_size == 1) {
        gather_generic_kernel<1><<<blocks, threads, 0, stream>>>(
            p, o, m, ndim, seq_dim, head_dim, S, H_part, rank, total_out
        );
    } else if (elem_size == 8) {
        gather_generic_kernel<8><<<blocks, threads, 0, stream>>>(
            p, o, m, ndim, seq_dim, head_dim, S, H_part, rank, total_out
        );
    } else {
        gather_generic_dynamic_kernel<<<blocks, threads, 0, stream>>>(
            p, o, m, ndim, seq_dim, head_dim, S, H_part, rank, total_out, elem_size
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stage_copy", &stage_copy, "Vectorized device copy into symmetric memory");
    m.def("launch_gather_4d_dim1_dim2_u16", &launch_gather_4d_dim1_dim2_u16,
          "Ulysses gather_seq_scatter_heads fast path for 4D dim1/dim2 16-bit tensors");
    m.def("launch_gather_generic", &launch_gather_generic,
          "Generic symmetric-memory UVA all-to-all gather");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_gather_seq_scatter_heads_symm_uva_bf16_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    return dim


def _make_out_shape(in_shape, seq_dim: int, head_dim: int, world: int, unpadded_dim_size: int):
    out_shape = list(in_shape)
    out_shape[head_dim] = in_shape[head_dim] // world
    out_shape[seq_dim] = in_shape[seq_dim] * world
    if unpadded_dim_size and (unpadded_dim_size % world != 0):
        out_shape[seq_dim] = unpadded_dim_size
    return tuple(out_shape)


def _get_resources(x: torch.Tensor, out_shape, seq_dim: int, head_dim: int, group: ProcessGroup):
    key = (
        tuple(x.shape),
        tuple(out_shape),
        x.dtype,
        x.device.index,
        seq_dim,
        head_dim,
        id(group),
    )
    res = _resource_cache.get(key)
    if res is not None:
        return res

    buf = symm_mem.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
    hdl = symm_mem.rendezvous(buf, group)

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    ptrs = torch.tensor([int(p) for p in hdl.buffer_ptrs], device=x.device, dtype=torch.int64)

    ndim = x.dim()
    in_shape = list(x.shape)
    in_stride = list(x.stride())
    meta_vals = in_shape + list(out_shape) + in_stride
    meta = torch.tensor(meta_vals, device=x.device, dtype=torch.int64)

    res = {
        "buf": buf,
        "hdl": hdl,
        "out": out,
        "ptrs": ptrs,
        "meta": meta,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: int = 0,
) -> torch.Tensor:
    if group is None:
        return x

    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert x.is_cuda, "x must be CUDA"

    world = dist.get_world_size(group)
    if world == 1:
        return x

    ndim = x.dim()
    seq_dim = _normalize_dim(seq_dim, ndim)
    head_dim = _normalize_dim(head_dim, ndim)

    assert seq_dim != head_dim, "seq_dim and head_dim must be distinct for Ulysses gather/scatter"
    assert x.size(head_dim) % world == 0, "head_dim must be divisible by world size"

    x_contig = x if x.is_contiguous() else x.contiguous()

    rank = dist.get_rank(group)
    S = int(x_contig.size(seq_dim))
    H = int(x_contig.size(head_dim))
    H_part = H // world

    out_shape = _make_out_shape(tuple(x_contig.shape), seq_dim, head_dim, world, unpadded_dim_size)
    S_out = int(out_shape[seq_dim])

    ext = _get_ext()
    res = _get_resources(x_contig, out_shape, seq_dim, head_dim, group)

    buf = res["buf"]
    hdl = res["hdl"]
    out = res["out"]
    ptrs = res["ptrs"]
    meta = res["meta"]

    # Stage local input into symmetric memory. This is a custom CUDA copy so peer
    # kernels can directly load every rank's data through UVA pointers.
    ext.stage_copy(x_contig, buf, x_contig.numel() * x_contig.element_size())

    # Publish staged data to peers before direct remote loads.
    hdl.barrier(channel=0)

    # Fast BF16/Half 4D path used by the benchmark: [B, S, H, D], seq=1, head=2.
    if (
        x_contig.dim() == 4
        and seq_dim == 1
        and head_dim == 2
        and x_contig.element_size() == 2
    ):
        B = int(x_contig.size(0))
        D = int(x_contig.size(3))
        ext.launch_gather_4d_dim1_dim2_u16(
            ptrs,
            out,
            B,
            S,
            H,
            D,
            S_out,
            H_part,
            rank,
        )
    else:
        ext.launch_gather_generic(
            ptrs,
            out,
            meta,
            ndim,
            seq_dim,
            head_dim,
            S,
            H_part,
            rank,
            out.numel(),
            x_contig.element_size(),
        )

    # Prevent a following invocation from overwriting symmetric buffers while a
    # slower peer may still be reading them.
    hdl.barrier(channel=1)

    return out