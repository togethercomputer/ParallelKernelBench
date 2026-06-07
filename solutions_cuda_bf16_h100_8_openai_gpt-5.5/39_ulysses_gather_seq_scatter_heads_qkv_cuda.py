from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import ProcessGroup

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>
#include <algorithm>
#include <pybind11/stl.h>

struct QKVMeta {
    int ndim;
    int seq_dim;
    int64_t sizes[8];
    int64_t strides[8];
    int64_t P;
    int64_t H;
    int64_t hc;
    int64_t seq_in;
    int64_t seq_out;
    int64_t last_out;
    int rank;
    int world;
    bool restore;
};

template <typename T>
__global__ void qkv_a2a_generic_kernel(
    const uint64_t* __restrict__ ptrs,
    T* __restrict__ out,
    int64_t total,
    QKVMeta m
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < total; idx += step) {
        int64_t tmp = idx;
        int64_t q, h;

        if (m.restore) {
            const int64_t k = tmp % m.last_out;  // [3 * hc]
            tmp /= m.last_out;
            q = k / m.hc;
            h = k - q * m.hc;
        } else {
            h = tmp % m.hc;
            tmp /= m.hc;
            q = tmp % 3;
            tmp /= 3;
        }

        int64_t input_linear = 0;
        int64_t src_rank = 0;

        // Decode dimensions before the original fused projection dim.
        for (int d = m.ndim - 2; d >= 0; --d) {
            const int64_t odim = (d == m.seq_dim) ? m.seq_out : m.sizes[d];
            const int64_t coord = tmp % odim;
            tmp /= odim;

            int64_t in_coord = coord;
            if (d == m.seq_dim) {
                src_rank = coord / m.seq_in;
                in_coord = coord - src_rank * m.seq_in;
            }
            input_linear += in_coord * m.strides[d];
        }

        const int64_t in_last = q * m.H + (int64_t)m.rank * m.hc + h;
        const int64_t in_idx = input_linear + in_last;

        const T* __restrict__ remote =
            reinterpret_cast<const T*>(ptrs[src_rank]);
        out[idx] = remote[in_idx];
    }
}

// Common hot path: input [B, S, 3*H], seq_dim=1, restore_shape=True.
// Copies 8 bf16/half elements at a time inside each Q/K/V shard.
__global__ void qkv_a2a_3d_vec8_u16_kernel(
    const uint64_t* __restrict__ ptrs,
    uint16_t* __restrict__ out,
    int64_t B,
    int64_t S,
    int64_t Sout,
    int64_t H,
    int64_t hc,
    int rank
) {
    const int64_t P = 3 * H;
    const int64_t K = 3 * hc;
    const int64_t vecs_per_q = (hc + 7) >> 3;
    const int64_t total_vecs = B * Sout * 3 * vecs_per_q;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;

    for (int64_t v = tid; v < total_vecs; v += step) {
        int64_t t = v;
        const int64_t hv = t % vecs_per_q;
        t /= vecs_per_q;
        const int64_t q = t % 3;
        t /= 3;
        const int64_t os = t % Sout;
        const int64_t b = t / Sout;

        const int64_t src_rank = os / S;
        const int64_t ls = os - src_rank * S;
        const int64_t h = hv << 3;

        const uint16_t* __restrict__ remote =
            reinterpret_cast<const uint16_t*>(ptrs[src_rank]);

        const int64_t in_elem =
            (b * S + ls) * P + q * H + (int64_t)rank * hc + h;
        const int64_t out_elem =
            (b * Sout + os) * K + q * hc + h;

        if (h + 7 < hc) {
            const uintptr_t src_addr =
                reinterpret_cast<uintptr_t>(remote + in_elem);
            const uintptr_t dst_addr =
                reinterpret_cast<uintptr_t>(out + out_elem);

            if (((src_addr | dst_addr) & 15ULL) == 0ULL) {
                const uint4 x = *reinterpret_cast<const uint4*>(src_addr);
                *reinterpret_cast<uint4*>(dst_addr) = x;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    out[out_elem + i] = remote[in_elem + i];
                }
            }
        } else {
            for (int i = 0; i < 8 && h + i < hc; ++i) {
                out[out_elem + i] = remote[in_elem + i];
            }
        }
    }
}

static QKVMeta make_meta(
    const std::vector<int64_t>& sizes,
    int64_t seq_dim,
    int rank,
    int world,
    bool restore,
    int64_t seq_out
) {
    TORCH_CHECK(sizes.size() >= 2 && sizes.size() <= 8, "supported ndim is [2, 8]");
    const int ndim = (int)sizes.size();

    if (seq_dim < 0) {
        seq_dim += ndim;
    }
    TORCH_CHECK(seq_dim >= 0 && seq_dim < ndim - 1,
                "seq_dim must address a non-projection dimension");

    QKVMeta m;
    m.ndim = ndim;
    m.seq_dim = (int)seq_dim;
    m.rank = rank;
    m.world = world;
    m.restore = restore;

    for (int i = 0; i < 8; ++i) {
        m.sizes[i] = 1;
        m.strides[i] = 1;
    }
    for (int i = 0; i < ndim; ++i) {
        m.sizes[i] = sizes[i];
    }

    m.strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        m.strides[i] = m.strides[i + 1] * m.sizes[i + 1];
    }

    m.P = sizes[ndim - 1];
    TORCH_CHECK(m.P % 3 == 0, "last dim must be divisible by 3");
    m.H = m.P / 3;
    TORCH_CHECK(m.H % world == 0, "Q/K/V hidden shard must be divisible by world size");
    m.hc = m.H / world;
    m.seq_in = sizes[m.seq_dim];
    m.seq_out = seq_out;
    m.last_out = 3 * m.hc;
    return m;
}

void launch_qkv_a2a(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    std::vector<int64_t> input_sizes,
    int64_t seq_dim,
    int rank,
    int world,
    bool restore,
    int64_t seq_out
) {
    TORCH_CHECK(ptrs_tensor.is_cuda(), "ptrs_tensor must be CUDA");
    TORCH_CHECK(ptrs_tensor.dtype() == torch::kInt64, "ptrs_tensor must be int64");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int64_t total = out.numel();
    if (total == 0) {
        return;
    }

    QKVMeta m = make_meta(input_sizes, seq_dim, rank, world, restore, seq_out);

    const uint64_t* d_ptrs =
        reinterpret_cast<const uint64_t*>(ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int threads = 256;

    // Fast BF16/Half bit-copy path for the dominant Ulysses layout.
    if (out.element_size() == 2 &&
        restore &&
        m.ndim == 3 &&
        m.seq_dim == 1) {
        const int64_t B = m.sizes[0];
        const int64_t S = m.sizes[1];
        const int64_t Sout = m.seq_out;
        const int64_t vecs_per_q = (m.hc + 7) >> 3;
        const int64_t total_vecs = B * Sout * 3 * vecs_per_q;
        int blocks = (int)std::min<int64_t>(
            65535, (total_vecs + threads - 1) / threads);
        qkv_a2a_3d_vec8_u16_kernel<<<blocks, threads, 0, stream>>>(
            d_ptrs,
            reinterpret_cast<uint16_t*>(out.data_ptr()),
            B,
            S,
            Sout,
            m.H,
            m.hc,
            rank
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    int blocks = (int)std::min<int64_t>(
        65535, (total + threads - 1) / threads);

    const size_t elem = out.element_size();
    if (elem == 1) {
        qkv_a2a_generic_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, reinterpret_cast<uint8_t*>(out.data_ptr()), total, m);
    } else if (elem == 2) {
        qkv_a2a_generic_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, reinterpret_cast<uint16_t*>(out.data_ptr()), total, m);
    } else if (elem == 4) {
        qkv_a2a_generic_kernel<uint32_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, reinterpret_cast<uint32_t*>(out.data_ptr()), total, m);
    } else if (elem == 8) {
        qkv_a2a_generic_kernel<uint64_t><<<blocks, threads, 0, stream>>>(
            d_ptrs, reinterpret_cast<uint64_t*>(out.data_ptr()), total, m);
    } else {
        TORCH_CHECK(false, "unsupported element size");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_qkv_a2a", &launch_qkv_a2a,
          "Ulysses fused QKV all-to-all via symmetric-memory UVA peer reads");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ulysses_qkv_symm_uva_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _normalize_seq_dim(seq_dim: int, ndim: int) -> int:
    if seq_dim < 0:
        seq_dim += ndim
    return seq_dim


def _output_shape(
    input_shape,
    seq_dim: int,
    world_size: int,
    unpadded_dim_size: int,
    restore_shape: bool,
):
    ndim = len(input_shape)
    qkv_proj_dim = input_shape[-1]
    h = qkv_proj_dim // 3
    hc = h // world_size

    full_seq = input_shape[seq_dim] * world_size
    seq_out = full_seq
    if unpadded_dim_size and (unpadded_dim_size % world_size != 0):
        seq_out = int(unpadded_dim_size)

    if restore_shape:
        out_shape = list(input_shape)
        out_shape[seq_dim] = seq_out
        out_shape[-1] = qkv_proj_dim // world_size
    else:
        out_shape = list(input_shape[:-1]) + [3, hc]
        out_shape[seq_dim] = seq_out

    return tuple(out_shape), seq_out


def _get_resources(
    input_shape,
    out_shape,
    dtype: torch.dtype,
    device: torch.device,
    group: ProcessGroup,
):
    key = (tuple(input_shape), tuple(out_shape), dtype, device, id(group))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty(tuple(input_shape), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    out = torch.empty(tuple(out_shape), device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (buf, hdl, out, ptrs_tensor)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    qkv_tensor: torch.Tensor,
    seq_dim: int,
    group: Optional[ProcessGroup] = None,
    unpadded_dim_size: Optional[int] = None,
    restore_shape: bool = True,
) -> torch.Tensor:
    """
    Fused Ulysses gather-sequence/scatter-heads QKV transform.

    Communication is implemented as one symmetric-memory exchange:
    every rank publishes its contiguous fused-QKV buffer, then a CUDA kernel
    directly reads peer UVA pointers and writes the final restored/unpadded
    layout, avoiding NCCL all_to_all, tensor_split, cat, and intermediate views.
    """
    if not dist.is_initialized():
        return qkv_tensor

    group = group if group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1 and (not unpadded_dim_size):
        if restore_shape:
            return qkv_tensor

    assert qkv_tensor.is_cuda, "qkv_tensor must be CUDA"
    input_tensor = qkv_tensor if qkv_tensor.is_contiguous() else qkv_tensor.contiguous()

    ndim = input_tensor.dim()
    seq_dim = _normalize_seq_dim(seq_dim, ndim)
    assert 0 <= seq_dim < ndim - 1, "seq_dim must not be the fused QKV projection dim"

    input_shape = tuple(input_tensor.shape)
    qkv_proj_dim = input_shape[-1]
    assert qkv_proj_dim % 3 == 0, "last dim must be divisible by 3"
    assert (qkv_proj_dim // 3) % world_size == 0, (
        "per-Q/K/V projection dim must be divisible by world size"
    )

    unpadded = int(unpadded_dim_size or 0)
    out_shape, seq_out = _output_shape(
        input_shape,
        seq_dim,
        world_size,
        unpadded,
        restore_shape,
    )

    _get_ext()
    buf, hdl, out, ptrs_tensor = _get_resources(
        input_shape,
        out_shape,
        input_tensor.dtype,
        input_tensor.device,
        group,
    )

    # Publish this rank's input into symmetric memory, then all ranks read peer
    # chunks directly from device pointers.  The post barrier protects cached
    # symmetric buffers from being overwritten by a fast rank on the next call.
    buf.copy_(input_tensor)
    hdl.barrier(channel=0)

    _get_ext().launch_qkv_a2a(
        ptrs_tensor,
        out,
        list(input_shape),
        int(seq_dim),
        int(rank),
        int(world_size),
        bool(restore_shape),
        int(seq_out),
    )

    hdl.barrier(channel=1)
    return out