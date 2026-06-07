from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>
#include <algorithm>

static inline uintptr_t uptr(const void* p) {
    return reinterpret_cast<uintptr_t>(p);
}

void write_meta_cuda(torch::Tensor meta, torch::Tensor x, int64_t max_dims) {
    TORCH_CHECK(meta.is_cuda(), "meta must be CUDA");
    TORCH_CHECK(meta.dtype() == torch::kInt64, "meta must be int64");
    TORCH_CHECK(meta.is_contiguous(), "meta must be contiguous");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(max_dims > 0, "max_dims must be positive");
    TORCH_CHECK(x.dim() <= max_dims, "tensor rank exceeds MAX_DIMS");

    const int64_t fields = 2 + max_dims;
    TORCH_CHECK(meta.numel() >= fields, "meta tensor too small");

    std::vector<int64_t> h(fields, 1);
    h[0] = static_cast<int64_t>(x.dim());
    h[1] = static_cast<int64_t>(x.numel());
    for (int64_t i = 0; i < x.dim(); ++i) {
        h[2 + i] = static_cast<int64_t>(x.size(i));
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        meta.data_ptr<int64_t>(),
        h.data(),
        sizeof(int64_t) * fields,
        cudaMemcpyHostToDevice,
        stream));
}

__global__ void collect_meta_kernel(
    const long long* __restrict__ ptrs,
    long long* __restrict__ all_meta,
    int world_size,
    int fields
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = world_size * fields;
    if (idx >= total) return;

    int r = idx / fields;
    int f = idx - r * fields;
    const long long* src = reinterpret_cast<const long long*>(
        static_cast<uintptr_t>(ptrs[r]));
    all_meta[idx] = src[f];
}

void collect_meta_cuda(torch::Tensor ptrs, torch::Tensor all_meta, int64_t world_size, int64_t fields) {
    TORCH_CHECK(ptrs.is_cuda() && all_meta.is_cuda(), "ptrs/all_meta must be CUDA");
    TORCH_CHECK(ptrs.dtype() == torch::kInt64, "ptrs must be int64");
    TORCH_CHECK(all_meta.dtype() == torch::kInt64, "all_meta must be int64");
    TORCH_CHECK(ptrs.is_contiguous() && all_meta.is_contiguous(), "tensors must be contiguous");

    int total = static_cast<int>(world_size * fields);
    int threads = 128;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    collect_meta_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        reinterpret_cast<long long*>(all_meta.data_ptr<int64_t>()),
        static_cast<int>(world_size),
        static_cast<int>(fields));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_to_symm_cuda(torch::Tensor dst_symm, torch::Tensor src, int64_t numel) {
    TORCH_CHECK(dst_symm.is_cuda() && src.is_cuda(), "dst/src must be CUDA");
    TORCH_CHECK(dst_symm.is_contiguous() && src.is_contiguous(), "dst/src must be contiguous");
    TORCH_CHECK(dst_symm.scalar_type() == src.scalar_type(), "dtype mismatch");
    TORCH_CHECK(dst_symm.numel() >= numel, "symmetric buffer too small");

    if (numel <= 0) return;

    const size_t nbytes = static_cast<size_t>(numel) * static_cast<size_t>(src.element_size());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst_symm.data_ptr(),
        src.data_ptr(),
        nbytes,
        cudaMemcpyDeviceToDevice,
        stream));
}

__device__ __forceinline__ void copy_bytes_vec_or_scalar(
    const char* __restrict__ src,
    char* __restrict__ dst,
    long long nbytes,
    long long linear_tid,
    long long linear_stride
) {
    const uintptr_t saddr = reinterpret_cast<uintptr_t>(src);
    const uintptr_t daddr = reinterpret_cast<uintptr_t>(dst);

    if (((saddr | daddr | static_cast<uintptr_t>(nbytes)) & 0xFULL) == 0) {
        const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src);
        uint4* __restrict__ d4 = reinterpret_cast<uint4*>(dst);
        long long n4 = nbytes >> 4;
        for (long long i = linear_tid; i < n4; i += linear_stride) {
            d4[i] = s4[i];
        }
    } else {
        for (long long i = linear_tid; i < nbytes; i += linear_stride) {
            dst[i] = src[i];
        }
    }
}

__global__ void gather_dim0_kernel(
    const long long* __restrict__ ptrs,
    const long long* __restrict__ meta,
    char* __restrict__ out,
    int world_size,
    int fields,
    int elem_size
) {
    int r = blockIdx.y;
    if (r >= world_size) return;

    long long numel = meta[r * fields + 1];
    long long prefix = 0;
    #pragma unroll
    for (int rr = 0; rr < 16; ++rr) {
        if (rr >= r) break;
        prefix += meta[rr * fields + 1];
    }

    const char* src = reinterpret_cast<const char*>(
        static_cast<uintptr_t>(ptrs[r]));
    char* dst = out + prefix * static_cast<long long>(elem_size);
    long long nbytes = numel * static_cast<long long>(elem_size);

    long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    copy_bytes_vec_or_scalar(src, dst, nbytes, tid, stride);
}

__global__ void gather_general_kernel(
    const long long* __restrict__ ptrs,
    const long long* __restrict__ meta,
    char* __restrict__ out,
    int world_size,
    int fields,
    int gather_dim,
    long long outer,
    long long inner,
    long long total_gather,
    int elem_size
) {
    long long segment = static_cast<long long>(blockIdx.x);
    long long r = segment % world_size;
    long long outer_idx = segment / world_size;
    if (outer_idx >= outer) return;

    long long gd = meta[r * fields + 2 + gather_dim];
    if (gd <= 0) return;

    long long prefix_g = 0;
    #pragma unroll
    for (int rr = 0; rr < 16; ++rr) {
        if (rr >= r) break;
        prefix_g += meta[rr * fields + 2 + gather_dim];
    }

    long long seg_elems = gd * inner;
    long long src_elem_off = outer_idx * seg_elems;
    long long dst_elem_off = (outer_idx * total_gather + prefix_g) * inner;
    long long nbytes = seg_elems * static_cast<long long>(elem_size);

    const char* src = reinterpret_cast<const char*>(
        static_cast<uintptr_t>(ptrs[r])) + src_elem_off * static_cast<long long>(elem_size);
    char* dst = out + dst_elem_off * static_cast<long long>(elem_size);

    long long tid =
        (static_cast<long long>(blockIdx.y) * blockDim.x) + threadIdx.x;
    long long stride =
        static_cast<long long>(gridDim.y) * blockDim.x;

    copy_bytes_vec_or_scalar(src, dst, nbytes, tid, stride);
}

void launch_variable_allgather_cuda(
    torch::Tensor data_ptrs,
    torch::Tensor all_meta,
    torch::Tensor out,
    int64_t world_size,
    int64_t fields,
    int64_t gather_dim,
    int64_t outer,
    int64_t inner,
    int64_t total_gather,
    int64_t max_segment_elems
) {
    TORCH_CHECK(data_ptrs.is_cuda() && all_meta.is_cuda() && out.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(data_ptrs.dtype() == torch::kInt64, "data_ptrs must be int64");
    TORCH_CHECK(all_meta.dtype() == torch::kInt64, "all_meta must be int64");
    TORCH_CHECK(data_ptrs.is_contiguous() && all_meta.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");

    if (out.numel() == 0) return;

    int threads = 256;
    int elem_size = static_cast<int>(out.element_size());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* ptrs = reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>());
    const long long* meta = reinterpret_cast<const long long*>(all_meta.data_ptr<int64_t>());
    char* dst = reinterpret_cast<char*>(out.data_ptr());

    if (gather_dim == 0) {
        long long max_bytes = max_segment_elems * static_cast<long long>(elem_size);
        long long units = ((max_bytes + 15) >> 4);
        int blocks_x = static_cast<int>((units + threads - 1) / threads);
        if (blocks_x < 1) blocks_x = 1;
        if (blocks_x > 65535) blocks_x = 65535;

        dim3 grid(blocks_x, static_cast<unsigned int>(world_size), 1);
        gather_dim0_kernel<<<grid, threads, 0, stream>>>(
            ptrs, meta, dst,
            static_cast<int>(world_size),
            static_cast<int>(fields),
            elem_size);
    } else {
        long long max_bytes = max_segment_elems * static_cast<long long>(elem_size);
        long long units = ((max_bytes + 15) >> 4);
        int chunks = static_cast<int>((units + threads - 1) / threads);
        if (chunks < 1) chunks = 1;
        if (chunks > 65535) chunks = 65535;

        unsigned int segments = static_cast<unsigned int>(outer * world_size);
        dim3 grid(segments, static_cast<unsigned int>(chunks), 1);
        gather_general_kernel<<<grid, threads, 0, stream>>>(
            ptrs, meta, dst,
            static_cast<int>(world_size),
            static_cast<int>(fields),
            static_cast<int>(gather_dim),
            static_cast<long long>(outer),
            static_cast<long long>(inner),
            static_cast<long long>(total_gather),
            elem_size);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write_meta_cuda", &write_meta_cuda, "Write local shape metadata");
    m.def("collect_meta_cuda", &collect_meta_cuda, "Collect shape metadata through UVA symmetric pointers");
    m.def("copy_to_symm_cuda", &copy_to_symm_cuda, "Copy local tensor to symmetric buffer");
    m.def("launch_variable_allgather_cuda", &launch_variable_allgather_cuda,
          "Variable-size all-gather concat through UVA symmetric pointers");
}
'''

_EXT = None
MAX_DIMS = 16
FIELDS = 2 + MAX_DIMS

_META_CACHE = {}
_DATA_CACHE = {}


def _get_ext():
    global _EXT
    if _EXT is None:
        _EXT = compile_cuda_extension("ulysses_var_allgather_symm_cuda_ext", CUDA_SRC)
    return _EXT


def _group_key(group):
    return id(group)


def _device_key(device: torch.device):
    return (device.type, device.index if device.index is not None else torch.cuda.current_device())


def _get_meta_resources(group, world_size: int, device: torch.device):
    key = (_group_key(group), world_size, _device_key(device))
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached

    meta = symm_mem.empty((FIELDS,), dtype=torch.int64, device=device)
    hdl = symm_mem.rendezvous(meta, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    all_meta = torch.empty((world_size, FIELDS), dtype=torch.int64, device=device)

    cached = (meta, hdl, ptrs, all_meta)
    _META_CACHE[key] = cached
    return cached


def _get_data_resources(group, dtype: torch.dtype, device: torch.device, capacity: int):
    key = (_group_key(group), dtype, _device_key(device))
    cached = _DATA_CACHE.get(key)
    if cached is not None and cached["capacity"] >= capacity:
        return cached["buf"], cached["hdl"], cached["ptrs"]

    cap = max(int(capacity), 1)
    buf = symm_mem.empty((cap,), dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)

    cached = {"capacity": cap, "buf": buf, "hdl": hdl, "ptrs": ptrs}
    _DATA_CACHE[key] = cached
    return buf, hdl, ptrs


def _prod(vals):
    p = 1
    for v in vals:
        p *= int(v)
    return int(p)


@torch.no_grad()
def solution(
    x: torch.Tensor,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return x.contiguous()

    assert x.is_cuda, "x must be CUDA"
    assert dist.is_initialized(), "torch.distributed must be initialized"

    ext = _get_ext()

    if not x.is_contiguous():
        x = x.contiguous()

    device = x.device
    dtype = x.dtype
    ndim = x.dim()
    assert ndim <= MAX_DIMS, f"tensor dim {ndim} exceeds MAX_DIMS={MAX_DIMS}"

    if gather_dim < 0:
        gather_dim += ndim
    assert 0 <= gather_dim < ndim, "invalid gather_dim"

    meta, meta_hdl, meta_ptrs, all_meta = _get_meta_resources(group, world_size, device)

    ext.write_meta_cuda(meta, x, MAX_DIMS)
    meta_hdl.barrier(channel=0)

    ext.collect_meta_cuda(meta_ptrs, all_meta, world_size, FIELDS)

    # Small control-plane readback only for allocation/shape arithmetic.
    meta_host = all_meta.cpu().tolist()

    sizes = []
    numels = []
    for r in range(world_size):
        r_ndim = int(meta_host[r][0])
        assert r_ndim == ndim, "all ranks must have same tensor rank"
        shape_r = [int(meta_host[r][2 + i]) for i in range(ndim)]
        sizes.append(shape_r)
        numels.append(int(meta_host[r][1]))

    out_shape = list(sizes[0])
    total_gather = sum(s[gather_dim] for s in sizes)
    out_shape[gather_dim] = total_gather

    # torch.cat compatibility: non-gather dimensions must match.
    for r in range(1, world_size):
        for d in range(ndim):
            if d != gather_dim:
                assert sizes[r][d] == out_shape[d], "non-gather dimensions must match"

    total_out_numel = _prod(out_shape)
    out = torch.empty(tuple(out_shape), dtype=dtype, device=device)

    if total_out_numel == 0:
        return out.contiguous()

    max_numel = max(numels) if numels else 0
    data_buf, data_hdl, data_ptrs = _get_data_resources(group, dtype, device, max_numel)

    ext.copy_to_symm_cuda(data_buf, x.reshape(-1), x.numel())
    data_hdl.barrier(channel=0)

    outer = _prod(out_shape[:gather_dim])
    inner = _prod(out_shape[gather_dim + 1:])
    max_segment_elems = max(s[gather_dim] * inner for s in sizes)

    ext.launch_variable_allgather_cuda(
        data_ptrs,
        all_meta,
        out,
        world_size,
        FIELDS,
        gather_dim,
        outer,
        inner,
        total_gather,
        max_segment_elems,
    )

    return out