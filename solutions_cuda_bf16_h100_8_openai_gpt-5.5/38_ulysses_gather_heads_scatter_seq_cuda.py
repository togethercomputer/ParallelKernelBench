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

#include <cstdint>
#include <vector>
#include <algorithm>
#include <pybind11/stl.h>

#define MAX_NDIM 8
#define MAX_WORLD 16

struct A2AArgs {
    int ndim;
    int rank;
    int world;
    int scatter_dim;
    int gather_dim;
    int elem_size;
    int special_s1_g1;
    int64_t in_dims[MAX_NDIM];
    int64_t out_dims[MAX_NDIM];
    uint64_t ptrs[MAX_WORLD];
};

__device__ __forceinline__ void raw_copy_elem(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    int elem_size
) {
    if (elem_size == 2) {
        *reinterpret_cast<uint16_t*>(dst) = *reinterpret_cast<const uint16_t*>(src);
    } else if (elem_size == 4) {
        *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src);
    } else if (elem_size == 8) {
        *reinterpret_cast<uint64_t*>(dst) = *reinterpret_cast<const uint64_t*>(src);
    } else if (elem_size == 1) {
        *dst = *src;
    } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < elem_size) dst[i] = src[i];
        }
    }
}

__device__ __forceinline__ void raw_zero_elem(
    uint8_t* __restrict__ dst,
    int elem_size
) {
    if (elem_size == 2) {
        *reinterpret_cast<uint16_t*>(dst) = 0;
    } else if (elem_size == 4) {
        *reinterpret_cast<uint32_t*>(dst) = 0;
    } else if (elem_size == 8) {
        *reinterpret_cast<uint64_t*>(dst) = 0;
    } else if (elem_size == 1) {
        *dst = 0;
    } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < elem_size) dst[i] = 0;
        }
    }
}

struct PadArgs {
    int ndim;
    int elem_size;
    int64_t orig_dims[MAX_NDIM];
    int64_t pad_dims[MAX_NDIM];
};

__global__ void prepare_pad_kernel(
    const uint8_t* __restrict__ inp,
    uint8_t* __restrict__ buf,
    PadArgs args,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t t = idx;
        int64_t in_off = 0;
        bool valid = true;

        #pragma unroll
        for (int d = MAX_NDIM - 1; d >= 0; --d) {
            if (d >= args.ndim) continue;
            int64_t c = t % args.pad_dims[d];
            t /= args.pad_dims[d];

            if (c >= args.orig_dims[d]) {
                valid = false;
            }
        }

        if (valid) {
            t = idx;
            int64_t mul = 1;
            in_off = 0;
            #pragma unroll
            for (int d = MAX_NDIM - 1; d >= 0; --d) {
                if (d >= args.ndim) continue;
                int64_t c = t % args.pad_dims[d];
                t /= args.pad_dims[d];
                in_off += c * mul;
                mul *= args.orig_dims[d];
            }
            raw_copy_elem(buf + idx * args.elem_size,
                          inp + in_off * args.elem_size,
                          args.elem_size);
        } else {
            raw_zero_elem(buf + idx * args.elem_size, args.elem_size);
        }
    }
}

// Common Ulysses post-attention BF16 layout:
// input  [B, S, H, D]
// output [B, S/world, H*world, D]
// scatter_dim=1, gather_dim=2.
// Vectorized as 8 BF16 elements = 16 bytes.
__global__ void alltoall_4d_bf16_vec8_kernel(
    uint8_t* __restrict__ out,
    A2AArgs args,
    int64_t total_vec
) {
    const int64_t B = args.in_dims[0];
    const int64_t S = args.in_dims[1];
    const int64_t H = args.in_dims[2];
    const int64_t D = args.in_dims[3];
    const int64_t chunk_s = S / args.world;
    const int64_t Dv = D >> 3;  // /8 BF16 elems

    int64_t vidx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; vidx < total_vec; vidx += stride) {
        int64_t t = vidx;

        int64_t dvec = t % Dv;
        t /= Dv;

        int64_t h_out = t % (H * args.world);
        t /= (H * args.world);

        int64_t s_local = t % chunk_s;
        t /= chunk_s;

        int64_t b = t;

        int peer = (int)(h_out / H);
        int64_t h = h_out - (int64_t)peer * H;

        int64_t in_elem =
            (((b * S + ((int64_t)args.rank * chunk_s + s_local)) * H + h) * D) +
            dvec * 8;

        int64_t out_elem = vidx * 8;

        const uint4* src4 = reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint8_t*>(args.ptrs[peer]) + in_elem * 2);
        uint4* dst4 = reinterpret_cast<uint4*>(out + out_elem * 2);
        *dst4 = *src4;
    }
}

__global__ void alltoall_generic_kernel(
    uint8_t* __restrict__ out,
    A2AArgs args,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    const int sdim = args.scatter_dim;
    const int gdim = args.gather_dim;
    const int64_t chunk_s = args.in_dims[sdim] / args.world;

    for (; idx < total; idx += stride) {
        int64_t coords[MAX_NDIM];
        int64_t t = idx;

        #pragma unroll
        for (int d = MAX_NDIM - 1; d >= 0; --d) {
            if (d >= args.ndim) continue;
            coords[d] = t % args.out_dims[d];
            t /= args.out_dims[d];
        }

        int peer = 0;
        int64_t in_coords[MAX_NDIM];

        #pragma unroll
        for (int d = 0; d < MAX_NDIM; ++d) {
            if (d < args.ndim) in_coords[d] = coords[d];
        }

        if (args.special_s1_g1) {
            // Exact behavior of the reference _all_to_all_single path for
            // scatter_dim == gather_dim == 1 when its reshape is valid.
            const int64_t gather_bef = args.in_dims[1];
            peer = (int)(coords[0] / gather_bef);
            in_coords[0] = coords[0] - (int64_t)peer * gather_bef;
            in_coords[1] = (int64_t)args.rank * chunk_s + coords[1];
        } else if (sdim == gdim) {
            peer = (int)(coords[gdim] / chunk_s);
            in_coords[sdim] = (int64_t)args.rank * chunk_s +
                              (coords[gdim] - (int64_t)peer * chunk_s);
        } else {
            peer = (int)(coords[gdim] / args.in_dims[gdim]);
            in_coords[gdim] = coords[gdim] - (int64_t)peer * args.in_dims[gdim];
            in_coords[sdim] = (int64_t)args.rank * chunk_s + coords[sdim];
        }

        int64_t in_off = 0;
        #pragma unroll
        for (int d = 0; d < MAX_NDIM; ++d) {
            if (d >= args.ndim) continue;
            in_off = in_off * args.in_dims[d] + in_coords[d];
        }

        const uint8_t* src =
            reinterpret_cast<const uint8_t*>(args.ptrs[peer]) + in_off * args.elem_size;
        uint8_t* dst = out + idx * args.elem_size;
        raw_copy_elem(dst, src, args.elem_size);
    }
}

static inline int64_t numel_from_vec(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto v : shape) n *= v;
    return n;
}

void prepare_buffer(
    torch::Tensor input,
    torch::Tensor buffer,
    std::vector<int64_t> orig_shape,
    std::vector<int64_t> padded_shape
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(buffer.is_cuda(), "buffer must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(buffer.is_contiguous(), "buffer must be contiguous");
    TORCH_CHECK(orig_shape.size() == padded_shape.size(), "shape rank mismatch");
    TORCH_CHECK(orig_shape.size() <= MAX_NDIM, "rank > MAX_NDIM unsupported");

    const int ndim = (int)orig_shape.size();
    const int elem_size = (int)input.element_size();
    const int64_t orig_numel = input.numel();
    const int64_t pad_numel = buffer.numel();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    bool same = (orig_numel == pad_numel);
    if (same) {
        for (int i = 0; i < ndim; ++i) {
            if (orig_shape[i] != padded_shape[i]) {
                same = false;
                break;
            }
        }
    }

    if (same) {
        cudaMemcpyAsync(
            buffer.data_ptr(),
            input.data_ptr(),
            (size_t)(orig_numel * elem_size),
            cudaMemcpyDeviceToDevice,
            stream);
        return;
    }

    PadArgs args;
    args.ndim = ndim;
    args.elem_size = elem_size;
    for (int i = 0; i < MAX_NDIM; ++i) {
        args.orig_dims[i] = 1;
        args.pad_dims[i] = 1;
    }
    for (int i = 0; i < ndim; ++i) {
        args.orig_dims[i] = orig_shape[i];
        args.pad_dims[i] = padded_shape[i];
    }

    int threads = 256;
    int blocks = (int)((pad_numel + threads - 1) / threads);
    blocks = std::max(1, std::min(blocks, 65535));

    prepare_pad_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(input.data_ptr()),
        reinterpret_cast<uint8_t*>(buffer.data_ptr()),
        args,
        pad_numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_alltoall_generic(
    torch::Tensor out,
    std::vector<int64_t> ptrs,
    std::vector<int64_t> in_shape,
    std::vector<int64_t> out_shape,
    int scatter_dim,
    int gather_dim,
    int rank,
    int world,
    int special_s1_g1
) {
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(in_shape.size() == out_shape.size(), "rank mismatch");
    TORCH_CHECK(in_shape.size() <= MAX_NDIM, "rank > MAX_NDIM unsupported");
    TORCH_CHECK((int)ptrs.size() == world, "ptr count != world");
    TORCH_CHECK(world <= MAX_WORLD, "world > MAX_WORLD unsupported");

    A2AArgs args;
    args.ndim = (int)in_shape.size();
    args.rank = rank;
    args.world = world;
    args.scatter_dim = scatter_dim;
    args.gather_dim = gather_dim;
    args.elem_size = (int)out.element_size();
    args.special_s1_g1 = special_s1_g1;

    for (int i = 0; i < MAX_NDIM; ++i) {
        args.in_dims[i] = 1;
        args.out_dims[i] = 1;
    }
    for (int i = 0; i < MAX_WORLD; ++i) {
        args.ptrs[i] = 0;
    }
    for (int i = 0; i < args.ndim; ++i) {
        args.in_dims[i] = in_shape[i];
        args.out_dims[i] = out_shape[i];
    }
    for (int i = 0; i < world; ++i) {
        args.ptrs[i] = (uint64_t)ptrs[i];
    }

    const int64_t total = out.numel();
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    blocks = std::max(1, std::min(blocks, 65535));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    alltoall_generic_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint8_t*>(out.data_ptr()),
        args,
        total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_alltoall_4d_bf16_vec8(
    torch::Tensor out,
    std::vector<int64_t> ptrs,
    std::vector<int64_t> in_shape,
    int rank,
    int world
) {
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "optimized path requires BF16");
    TORCH_CHECK(in_shape.size() == 4, "4D shape required");
    TORCH_CHECK((int)ptrs.size() == world, "ptr count != world");
    TORCH_CHECK(world <= MAX_WORLD, "world > MAX_WORLD unsupported");
    TORCH_CHECK((in_shape[3] % 8) == 0, "D must be multiple of 8");

    A2AArgs args;
    args.ndim = 4;
    args.rank = rank;
    args.world = world;
    args.scatter_dim = 1;
    args.gather_dim = 2;
    args.elem_size = 2;
    args.special_s1_g1 = 0;

    for (int i = 0; i < MAX_NDIM; ++i) {
        args.in_dims[i] = 1;
        args.out_dims[i] = 1;
    }
    for (int i = 0; i < MAX_WORLD; ++i) {
        args.ptrs[i] = 0;
    }

    args.in_dims[0] = in_shape[0];
    args.in_dims[1] = in_shape[1];
    args.in_dims[2] = in_shape[2];
    args.in_dims[3] = in_shape[3];

    args.out_dims[0] = in_shape[0];
    args.out_dims[1] = in_shape[1] / world;
    args.out_dims[2] = in_shape[2] * world;
    args.out_dims[3] = in_shape[3];

    for (int i = 0; i < world; ++i) {
        args.ptrs[i] = (uint64_t)ptrs[i];
    }

    const int64_t total_vec = out.numel() / 8;
    int threads = 256;
    int blocks = (int)((total_vec + threads - 1) / threads);
    blocks = std::max(1, std::min(blocks, 65535));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    alltoall_4d_bf16_vec8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint8_t*>(out.data_ptr()),
        args,
        total_vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_buffer", &prepare_buffer, "Copy/pad local tensor into symmetric buffer");
    m.def("launch_alltoall_generic", &launch_alltoall_generic, "UVA all-to-all tensor transform");
    m.def("launch_alltoall_4d_bf16_vec8", &launch_alltoall_4d_bf16_vec8,
          "Vectorized BF16 4D Ulysses gather-heads/scatter-seq");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_gather_heads_scatter_seq_symm_cuda_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    return dim


def _compute_padded_shape(shape, scatter_dim: int, world: int):
    padded = list(shape)
    dim_size = padded[scatter_dim]
    rem = dim_size % world
    if rem != 0:
        padded[scatter_dim] = dim_size + (world - rem)
    return padded


def _compute_out_shape(padded_shape, scatter_dim: int, gather_dim: int, world: int):
    # Exact odd reference branch for scatter_dim == gather_dim == 1:
    # x.reshape([x.shape[1], world, x.shape[1]//world] + x.shape[2:])
    #  .transpose(0,1)
    #  .reshape([x.shape[1]*world, x.shape[1]//world] + x.shape[2:])
    if scatter_dim == 1 and gather_dim == 1:
        out = list(padded_shape)
        out[0] = padded_shape[1] * world
        out[1] = padded_shape[1] // world
        return out

    out = list(padded_shape)
    out[scatter_dim] = padded_shape[scatter_dim] // world
    out[gather_dim] = out[gather_dim] * world
    return out


def _get_resources(padded_shape, dtype, device, group):
    key = (tuple(padded_shape), dtype, device, id(group))
    res = _resource_cache.get(key)
    if res is not None:
        return res

    buf = symm_mem.empty(tuple(padded_shape), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = [int(p) for p in hdl.buffer_ptrs]

    res = (buf, hdl, ptrs)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """
    Ulysses gather_heads_scatter_seq implemented as direct symmetric-memory
    peer reads plus fused CUDA layout transform.
    """
    if group is None:
        return x

    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert x.is_cuda, "input must be CUDA"

    ext = _get_ext()

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    ndim = x.dim()

    scatter_dim = _normalize_dim(seq_dim, ndim)
    gather_dim = _normalize_dim(head_dim, ndim)
    assert 0 <= scatter_dim < ndim
    assert 0 <= gather_dim < ndim
    assert world <= 16, "this CUDA implementation supports world_size <= 16"

    # Reference path materializes contiguous chunks; keep the source layout
    # contiguous before exposing it through symmetric memory.
    xc = x if x.is_contiguous() else x.contiguous()

    orig_shape = list(xc.shape)
    padded_shape = _compute_padded_shape(orig_shape, scatter_dim, world)
    out_shape = _compute_out_shape(padded_shape, scatter_dim, gather_dim, world)

    buf, hdl, ptrs = _get_resources(padded_shape, xc.dtype, xc.device, group)

    ext.prepare_buffer(xc, buf, orig_shape, padded_shape)

    # Symmetric-memory synchronization: all ranks' CUDA writes to their exposed
    # buffers are visible before any rank starts direct UVA peer reads.
    hdl.barrier(channel=0)

    out = torch.empty(tuple(out_shape), dtype=xc.dtype, device=xc.device)

    # Fast path for the benchmark's BF16 post-attention layout.
    if (
        xc.dtype == torch.bfloat16
        and ndim == 4
        and scatter_dim == 1
        and gather_dim == 2
        and padded_shape[1] % world == 0
        and padded_shape[3] % 8 == 0
    ):
        ext.launch_alltoall_4d_bf16_vec8(out, ptrs, padded_shape, rank, world)
    else:
        special = 1 if (scatter_dim == 1 and gather_dim == 1) else 0
        ext.launch_alltoall_generic(
            out,
            ptrs,
            padded_shape,
            out_shape,
            scatter_dim,
            gather_dim,
            rank,
            world,
            special,
        )

    return out