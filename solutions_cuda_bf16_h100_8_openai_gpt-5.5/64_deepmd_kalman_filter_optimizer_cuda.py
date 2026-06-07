from typing import List, Tuple, Dict, Any

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

#define ROWS_PER_BLOCK 8
#define THREADS_PER_BLOCK 256

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__ float bf16_load(const at::BFloat16* p) {
    const __nv_bfloat16* q = reinterpret_cast<const __nv_bfloat16*>(p);
    return __bfloat162float(*q);
}

__device__ __forceinline__ void bf16_store(at::BFloat16* p, float v) {
    __nv_bfloat16* q = reinterpret_cast<__nv_bfloat16*>(p);
    *q = __float2bfloat16(v);
}

__global__ void matvec_partial_bf16_kernel(
    const at::BFloat16* __restrict__ P,
    const at::BFloat16* __restrict__ H,
    at::BFloat16* __restrict__ K,
    float* __restrict__ partial,
    int64_t partial_offset,
    int64_t n
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t row = (int64_t)blockIdx.x * ROWS_PER_BLOCK + warp;

    __shared__ float row_sums[ROWS_PER_BLOCK];

    float dot = 0.0f;
    if (row < n) {
        const int64_t base = row * n;
        for (int64_t c = lane; c < n; c += 32) {
            dot += bf16_load(P + base + c) * bf16_load(H + c);
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            bf16_store(K + row, dot);
            row_sums[warp] = bf16_load(H + row) * dot;
        }
    } else {
        if (lane == 0 && warp < ROWS_PER_BLOCK) row_sums[warp] = 0.0f;
    }

    __syncthreads();

    if (warp == 0) {
        float s = (lane < ROWS_PER_BLOCK) ? row_sums[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) {
            partial[partial_offset + blockIdx.x] = s;
        }
    }
}

__global__ void matvec_partial_f32_kernel(
    const float* __restrict__ P,
    const float* __restrict__ H,
    float* __restrict__ K,
    float* __restrict__ partial,
    int64_t partial_offset,
    int64_t n
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t row = (int64_t)blockIdx.x * ROWS_PER_BLOCK + warp;

    __shared__ float row_sums[ROWS_PER_BLOCK];

    float dot = 0.0f;
    if (row < n) {
        const int64_t base = row * n;
        for (int64_t c = lane; c < n; c += 32) {
            dot += P[base + c] * H[c];
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            K[row] = dot;
            row_sums[warp] = H[row] * dot;
        }
    } else {
        if (lane == 0 && warp < ROWS_PER_BLOCK) row_sums[warp] = 0.0f;
    }

    __syncthreads();

    if (warp == 0) {
        float s = (lane < ROWS_PER_BLOCK) ? row_sums[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) {
            partial[partial_offset + blockIdx.x] = s;
        }
    }
}

__global__ void reduce_partials_kernel(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int64_t count,
    float lam,
    int num_weight_blocks
) {
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        sum += partial[i];
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[0] = smem[0] + lam * (float)num_weight_blocks;
    }
}

__global__ void sum_peer_scalars_kernel(
    const int64_t* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size
) {
    float sum = 0.0f;
    for (int r = threadIdx.x; r < world_size; r += blockDim.x) {
        const float* p = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
        sum += p[0];
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) out[0] = smem[0];
}

__global__ void update_bf16_kernel(
    const at::BFloat16* __restrict__ Pin,
    const at::BFloat16* __restrict__ Win,
    const at::BFloat16* __restrict__ K,
    const at::BFloat16* __restrict__ err,
    const float* __restrict__ denom,
    at::BFloat16* __restrict__ Pout,
    at::BFloat16* __restrict__ Wout,
    int64_t n,
    float lam
) {
    const int64_t total = n * n;
    const float A = 1.0f / denom[0];
    const float alpha = A * bf16_load(err);
    const float inv_lam = 1.0f / lam;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = idx / n;
        const int64_t c = idx - r * n;
        float p = bf16_load(Pin + idx);
        float kr = bf16_load(K + r);
        float kc = bf16_load(K + c);
        bf16_store(Pout + idx, inv_lam * (p - A * kr * kc));
    }

    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        float w = bf16_load(Win + i);
        float k = bf16_load(K + i);
        bf16_store(Wout + i, w + alpha * k);
    }
}

__global__ void update_f32_kernel(
    const float* __restrict__ Pin,
    const float* __restrict__ Win,
    const float* __restrict__ K,
    const float* __restrict__ err,
    const float* __restrict__ denom,
    float* __restrict__ Pout,
    float* __restrict__ Wout,
    int64_t n,
    float lam
) {
    const int64_t total = n * n;
    const float A = 1.0f / denom[0];
    const float alpha = A * err[0];
    const float inv_lam = 1.0f / lam;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = idx / n;
        const int64_t c = idx - r * n;
        Pout[idx] = inv_lam * (Pin[idx] - A * K[r] * K[c]);
    }

    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        Wout[i] = Win[i] + alpha * K[i];
    }
}

__global__ void fill_meta_kernel(
    int64_t* __restrict__ meta,
    const int64_t* __restrict__ sizes,
    int64_t num_blocks,
    int64_t total,
    int64_t max_meta
) {
    for (int64_t i = threadIdx.x; i < max_meta; i += blockDim.x) {
        meta[i] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        meta[0] = num_blocks;
        meta[1] = total;
    }

    for (int64_t i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        meta[2 + i] = sizes[i];
    }
}

__global__ void collect_meta_kernel(
    const int64_t* __restrict__ ptrs,
    int64_t* __restrict__ out,
    int world_size,
    int64_t max_meta
) {
    const int r = blockIdx.x;
    const int64_t* src = reinterpret_cast<const int64_t*>((uintptr_t)ptrs[r]);
    int64_t* dst = out + (int64_t)r * max_meta;

    for (int64_t i = threadIdx.x; i < max_meta; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void pack_bf16_kernel(
    const at::BFloat16* __restrict__ src,
    at::BFloat16* __restrict__ dst,
    int64_t offset,
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        dst[offset + i] = src[i];
    }
}

__global__ void pack_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t offset,
    int64_t n
) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        dst[offset + i] = src[i];
    }
}

__global__ void copy_remote_bf16_kernel(
    uint64_t remote_base,
    at::BFloat16* __restrict__ dst,
    int64_t offset,
    int64_t n
) {
    const at::BFloat16* src = reinterpret_cast<const at::BFloat16*>((uintptr_t)remote_base);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[offset + i];
    }
}

__global__ void copy_remote_f32_kernel(
    uint64_t remote_base,
    float* __restrict__ dst,
    int64_t offset,
    int64_t n
) {
    const float* src = reinterpret_cast<const float*>((uintptr_t)remote_base);
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        dst[i] = src[offset + i];
    }
}

__global__ void lambda_next_bf16_kernel(
    at::BFloat16* __restrict__ out,
    float lam,
    float nue
) {
    if (threadIdx.x == 0) {
        bf16_store(out, nue * lam + 1.0f - nue);
    }
}

__global__ void lambda_next_f32_kernel(
    float* __restrict__ out,
    float lam,
    float nue
) {
    if (threadIdx.x == 0) {
        out[0] = nue * lam + 1.0f - nue;
    }
}

static inline int blocks_for_elems(int64_t n) {
    int64_t b = (n + 255) / 256;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

void launch_matvec_partial(
    torch::Tensor P,
    torch::Tensor H,
    torch::Tensor K,
    torch::Tensor partial,
    int64_t partial_offset,
    int64_t n,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t grid = (n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (grid < 1) grid = 1;

    if (dtype_enum == 0) {
        matvec_partial_bf16_kernel<<<(int)grid, THREADS_PER_BLOCK, 0, stream>>>(
            P.data_ptr<at::BFloat16>(),
            H.data_ptr<at::BFloat16>(),
            K.data_ptr<at::BFloat16>(),
            partial.data_ptr<float>(),
            partial_offset,
            n
        );
    } else {
        matvec_partial_f32_kernel<<<(int)grid, THREADS_PER_BLOCK, 0, stream>>>(
            P.data_ptr<float>(),
            H.data_ptr<float>(),
            K.data_ptr<float>(),
            partial.data_ptr<float>(),
            partial_offset,
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce_partials(
    torch::Tensor partial,
    torch::Tensor out,
    int64_t count,
    float lam,
    int num_weight_blocks
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_partials_kernel<<<1, 256, 0, stream>>>(
        partial.data_ptr<float>(),
        out.data_ptr<float>(),
        count,
        lam,
        num_weight_blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_sum_peer_scalars(torch::Tensor ptrs, torch::Tensor out, int world_size) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    sum_peer_scalars_kernel<<<1, 256, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_update(
    torch::Tensor Pin,
    torch::Tensor Win,
    torch::Tensor K,
    torch::Tensor err,
    torch::Tensor denom,
    torch::Tensor Pout,
    torch::Tensor Wout,
    int64_t n,
    float lam,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int blocks = blocks_for_elems(n * n);

    if (dtype_enum == 0) {
        update_bf16_kernel<<<blocks, 256, 0, stream>>>(
            Pin.data_ptr<at::BFloat16>(),
            Win.data_ptr<at::BFloat16>(),
            K.data_ptr<at::BFloat16>(),
            err.data_ptr<at::BFloat16>(),
            denom.data_ptr<float>(),
            Pout.data_ptr<at::BFloat16>(),
            Wout.data_ptr<at::BFloat16>(),
            n,
            lam
        );
    } else {
        update_f32_kernel<<<blocks, 256, 0, stream>>>(
            Pin.data_ptr<float>(),
            Win.data_ptr<float>(),
            K.data_ptr<float>(),
            err.data_ptr<float>(),
            denom.data_ptr<float>(),
            Pout.data_ptr<float>(),
            Wout.data_ptr<float>(),
            n,
            lam
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fill_meta(
    torch::Tensor meta,
    torch::Tensor sizes,
    int64_t num_blocks,
    int64_t total,
    int64_t max_meta
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fill_meta_kernel<<<1, 256, 0, stream>>>(
        meta.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        num_blocks,
        total,
        max_meta
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_collect_meta(
    torch::Tensor ptrs,
    torch::Tensor out,
    int world_size,
    int64_t max_meta
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    collect_meta_kernel<<<world_size, 256, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(),
        world_size,
        max_meta
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_pack(
    torch::Tensor src,
    torch::Tensor dst,
    int64_t offset,
    int64_t n,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int blocks = blocks_for_elems(n);

    if (dtype_enum == 0) {
        pack_bf16_kernel<<<blocks, 256, 0, stream>>>(
            src.data_ptr<at::BFloat16>(),
            dst.data_ptr<at::BFloat16>(),
            offset,
            n
        );
    } else {
        pack_f32_kernel<<<blocks, 256, 0, stream>>>(
            src.data_ptr<float>(),
            dst.data_ptr<float>(),
            offset,
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_copy_remote(
    uint64_t remote_base,
    torch::Tensor dst,
    int64_t offset,
    int64_t n,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int blocks = blocks_for_elems(n);

    if (dtype_enum == 0) {
        copy_remote_bf16_kernel<<<blocks, 256, 0, stream>>>(
            remote_base,
            dst.data_ptr<at::BFloat16>(),
            offset,
            n
        );
    } else {
        copy_remote_f32_kernel<<<blocks, 256, 0, stream>>>(
            remote_base,
            dst.data_ptr<float>(),
            offset,
            n
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_lambda_next(
    torch::Tensor out,
    float lam,
    float nue,
    int dtype_enum
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (dtype_enum == 0) {
        lambda_next_bf16_kernel<<<1, 1, 0, stream>>>(
            out.data_ptr<at::BFloat16>(),
            lam,
            nue
        );
    } else {
        lambda_next_f32_kernel<<<1, 1, 0, stream>>>(
            out.data_ptr<float>(),
            lam,
            nue
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_matvec_partial", &launch_matvec_partial, "P@H and local hPh partials");
    m.def("launch_reduce_partials", &launch_reduce_partials, "Reduce hPh partials");
    m.def("launch_sum_peer_scalars", &launch_sum_peer_scalars, "UVA scalar all-reduce sum");
    m.def("launch_update", &launch_update, "Kalman weight/covariance update");
    m.def("launch_fill_meta", &launch_fill_meta, "Fill symmetric gather metadata");
    m.def("launch_collect_meta", &launch_collect_meta, "Collect metadata through UVA");
    m.def("launch_pack", &launch_pack, "Pack local weights to symmetric buffer");
    m.def("launch_copy_remote", &launch_copy_remote, "Copy remote symmetric weight segment");
    m.def("launch_lambda_next", &launch_lambda_next, "Compute next Kalman lambda");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("deepmd_lkf_symm_cuda_ext", CUDA_SRC)
    return _ext


MAX_META = 4096
_tmp_cache: Dict[Any, Any] = {}
_meta_cache: Dict[Any, Any] = {}
_weight_cache: Dict[Any, Any] = {}


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    raise TypeError("optimized DeepMD Kalman path supports torch.bfloat16 and torch.float32")


def _ceil_rows(n: int) -> int:
    return max(1, (int(n) + 7) // 8)


def _get_tmp_resource(device: torch.device):
    key = (device.index, str(device))
    if key in _tmp_cache:
        return _tmp_cache[key]

    buf = symm_mem.empty((1,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    reduced = torch.empty((1,), device=device, dtype=torch.float32)
    res = (buf, hdl, ptrs, reduced)
    _tmp_cache[key] = res
    return res


def _get_meta_resource(device: torch.device, world_size: int):
    key = (device.index, str(device), world_size)
    if key in _meta_cache:
        return _meta_cache[key]

    meta = symm_mem.empty((MAX_META,), device=device, dtype=torch.int64)
    hdl = symm_mem.rendezvous(meta, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    all_meta = torch.empty((world_size, MAX_META), device=device, dtype=torch.int64)
    res = (meta, hdl, ptrs, all_meta)
    _meta_cache[key] = res
    return res


def _get_weight_resource(total: int, dtype: torch.dtype, device: torch.device):
    key = (int(total), dtype, device.index, str(device))
    if key in _weight_cache:
        return _weight_cache[key]

    flat = symm_mem.empty((max(1, int(total)),), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(flat, dist.group.WORLD)
    res = (flat, hdl)
    _weight_cache[key] = res
    return res


@torch.no_grad()
def solution(
    H: List[torch.Tensor],
    error: torch.Tensor,
    weights: List[torch.Tensor],
    P: List[torch.Tensor],
    kalman_lambda: float,
    kalman_nue: float = 0.9987,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    ext = _get_ext()

    weights_num = len(weights)
    if weights_num == 0:
        device = error.device if error.is_cuda else torch.device("cuda", torch.cuda.current_device())
        dtype = torch.bfloat16
        kalman_lambda_next = torch.empty((), device=device, dtype=dtype)
        ext.launch_lambda_next(kalman_lambda_next, float(kalman_lambda), float(kalman_nue), 0)
        return weights, P, kalman_lambda_next

    device = weights[0].device
    dtype = weights[0].dtype
    de = _dtype_enum(dtype)

    Hc = [h.contiguous().reshape(-1, 1) for h in H]
    Wc = [w.contiguous().reshape(-1, 1) for w in weights]
    Pc = [p.contiguous() for p in P]
    err = error.to(device=device, dtype=dtype).contiguous().reshape(1)

    sizes = [int(w.numel()) for w in Wc]
    total_weight = int(sum(sizes))
    partial_counts = [_ceil_rows(n) for n in sizes]
    total_partials = int(sum(partial_counts))

    partial = torch.empty((max(1, total_partials),), device=device, dtype=torch.float32)
    K_list = [torch.empty_like(Hc[i]) for i in range(weights_num)]

    offset = 0
    for i in range(weights_num):
        n = sizes[i]
        ext.launch_matvec_partial(
            Pc[i],
            Hc[i],
            K_list[i],
            partial,
            offset,
            n,
            de,
        )
        offset += partial_counts[i]

    distributed = dist.is_initialized()
    if distributed:
        local_tmp, tmp_hdl, tmp_ptrs, denom = _get_tmp_resource(device)
        ext.launch_reduce_partials(
            partial,
            local_tmp,
            total_partials,
            float(kalman_lambda),
            weights_num,
        )
        tmp_hdl.barrier(channel=0)
        ext.launch_sum_peer_scalars(tmp_ptrs, denom, dist.get_world_size())
    else:
        denom = torch.empty((1,), device=device, dtype=torch.float32)
        ext.launch_reduce_partials(
            partial,
            denom,
            total_partials,
            float(kalman_lambda),
            weights_num,
        )

    out_weights: List[torch.Tensor] = []
    out_P: List[torch.Tensor] = []
    for i in range(weights_num):
        n = sizes[i]
        wout = torch.empty_like(Wc[i])
        pout = torch.empty_like(Pc[i])
        ext.launch_update(
            Pc[i],
            Wc[i],
            K_list[i],
            err,
            denom,
            pout,
            wout,
            n,
            float(kalman_lambda),
            de,
        )
        out_weights.append(wout)
        out_P.append(pout)

    if distributed:
        world = dist.get_world_size()

        if weights_num + 2 > MAX_META:
            raise RuntimeError("too many local DeepMD blocks for fixed symmetric metadata pad")

        sizes_dev = torch.tensor(sizes, device=device, dtype=torch.int64)

        meta, meta_hdl, meta_ptrs, all_meta = _get_meta_resource(device, world)
        ext.launch_fill_meta(meta, sizes_dev, weights_num, total_weight, MAX_META)
        meta_hdl.barrier(channel=1)
        ext.launch_collect_meta(meta_ptrs, all_meta, world, MAX_META)

        meta_cpu = all_meta.cpu()
        shape_list = []
        for r in range(world):
            nb = int(meta_cpu[r, 0].item())
            shape_list.append([int(meta_cpu[r, 2 + j].item()) for j in range(nb)])

        flat, weight_hdl = _get_weight_resource(total_weight, dtype, device)

        off = 0
        for i, n in enumerate(sizes):
            ext.launch_pack(out_weights[i], flat, off, n, de)
            off += n

        weight_hdl.barrier(channel=2)

        gathered: List[torch.Tensor] = []
        for r in range(world):
            remote_base = int(weight_hdl.buffer_ptrs[r])
            roff = 0
            for n in shape_list[r]:
                dst = torch.empty((n, 1), device=device, dtype=dtype)
                if n > 0:
                    ext.launch_copy_remote(remote_base, dst, roff, n, de)
                gathered.append(dst)
                roff += n

        weight_hdl.barrier(channel=3)
        out_weights = gathered

    kalman_lambda_next = torch.empty((), device=device, dtype=dtype)
    ext.launch_lambda_next(
        kalman_lambda_next,
        float(kalman_lambda),
        float(kalman_nue),
        de,
    )

    return out_weights, out_P, kalman_lambda_next