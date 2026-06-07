from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

__device__ __forceinline__ float load_input_as_float(const void* ptr, int64_t idx, int dtype_enum) {
    // dtype_enum: 0=bf16, 1=float32, 2=float16
    if (dtype_enum == 0) {
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(ptr);
        return __bfloat162float(p[idx]);
    } else if (dtype_enum == 2) {
        const __half* p = reinterpret_cast<const __half*>(ptr);
        return __half2float(p[idx]);
    } else {
        const float* p = reinterpret_cast<const float*>(ptr);
        return p[idx];
    }
}

__global__ void init_exp_rows_kernel(
    const void* __restrict__ x,
    float* __restrict__ p,
    float* __restrict__ symm_rows,
    int64_t n,
    int64_t k_cols,
    float inv_temp,
    int dtype_enum
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < n; idx += stride) {
        float v = expf(load_input_as_float(x, idx, dtype_enum) * inv_temp);
        p[idx] = v;
        int64_t k = idx % k_cols;
        atomicAdd(symm_rows + k, v);
    }
}

__global__ void reduce_local_rows_to_total_kernel(
    float* __restrict__ symm_rows,
    int64_t k_cols,
    float local_count
) {
    float local = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t k = tid; k < k_cols; k += stride) {
        local += symm_rows[k];
    }

    __shared__ float smem[256];
    int lane = threadIdx.x;
    smem[lane] = local;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (lane < off) smem[lane] += smem[lane + off];
        __syncthreads();
    }

    if (lane == 0) {
        atomicAdd(symm_rows + k_cols, smem[0]);
        if (blockIdx.x == 0) {
            symm_rows[k_cols + 1] = local_count;
        }
    }
}

__global__ void reduce_global_rows_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ global_rows,
    float* __restrict__ totals,
    int64_t k_cols,
    int world_size
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t k = tid; k < k_cols; k += stride) {
        float s = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const float* base = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
                s += base[k];
            }
        }
        global_rows[k] = s;
    }

    if (tid == 0) {
        float total_mass = 0.0f;
        float total_batch = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const float* base = reinterpret_cast<const float*>((uintptr_t)ptrs[r]);
                total_mass += base[k_cols];
                total_batch += base[k_cols + 1];
            }
        }
        totals[0] = total_mass;
        totals[1] = total_batch;
    }
}

__global__ void row_norm_colsum_kernel(
    float* __restrict__ p,
    const float* __restrict__ global_rows,
    float* __restrict__ colsum,
    int64_t b_rows,
    int64_t k_cols
) {
    int64_t b = (int64_t)blockIdx.x;
    if (b >= b_rows) return;

    float sum = 0.0f;
    float inv_k = 1.0f / (float)k_cols;
    int tid = threadIdx.x;

    for (int64_t k = tid; k < k_cols; k += blockDim.x) {
        int64_t idx = b * k_cols + k;
        float denom = global_rows[k];
        float v = p[idx] * inv_k / denom;
        p[idx] = v;
        sum += v;
    }

    __shared__ float smem[256];
    smem[tid] = sum;
    __syncthreads();

    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }

    if (tid == 0) {
        colsum[b] = smem[0];
    }
}

__global__ void col_norm_kernel(
    float* __restrict__ p,
    const float* __restrict__ colsum,
    float* __restrict__ symm_rows,
    int64_t n,
    int64_t k_cols,
    int accumulate_rows
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < n; idx += stride) {
        int64_t b = idx / k_cols;
        int64_t k = idx - b * k_cols;
        float v = p[idx] / colsum[b];
        p[idx] = v;
        if (accumulate_rows) {
            atomicAdd(symm_rows + k, v);
        }
    }
}

__global__ void scale_zero_iter_kernel(
    float* __restrict__ p,
    const float* __restrict__ totals,
    int64_t n
) {
    float scale = totals[1] / totals[0];
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = tid; idx < n; idx += stride) {
        p[idx] *= scale;
    }
}

void zero_symm_rows(torch::Tensor symm_rows, int64_t k_cols) {
    CHECK_CUDA(symm_rows);
    CHECK_CONTIGUOUS(symm_rows);
    TORCH_CHECK(symm_rows.dtype() == torch::kFloat32, "symm_rows must be float32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(symm_rows.data_ptr<float>(), 0, (size_t)(k_cols + 2) * sizeof(float), stream);
}

void init_exp_rows(
    torch::Tensor x,
    torch::Tensor p,
    torch::Tensor symm_rows,
    int64_t b_rows,
    int64_t k_cols,
    double teacher_temp,
    int dtype_enum,
    double local_count
) {
    CHECK_CUDA(x);
    CHECK_CUDA(p);
    CHECK_CUDA(symm_rows);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(p);
    CHECK_CONTIGUOUS(symm_rows);
    TORCH_CHECK(p.dtype() == torch::kFloat32, "p must be float32");
    TORCH_CHECK(symm_rows.dtype() == torch::kFloat32, "symm_rows must be float32");

    int64_t n = b_rows * k_cols;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (n > 0) {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        float inv_temp = 1.0f / (float)teacher_temp;
        init_exp_rows_kernel<<<blocks, threads, 0, stream>>>(
            x.data_ptr(), p.data_ptr<float>(), symm_rows.data_ptr<float>(),
            n, k_cols, inv_temp, dtype_enum
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int threads = 256;
    int blocks = (int)((k_cols + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;
    reduce_local_rows_to_total_kernel<<<blocks, threads, 0, stream>>>(
        symm_rows.data_ptr<float>(), k_cols, (float)local_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_global_rows(
    torch::Tensor ptrs,
    torch::Tensor global_rows,
    torch::Tensor totals,
    int64_t k_cols,
    int world_size
) {
    CHECK_CUDA(ptrs);
    CHECK_CUDA(global_rows);
    CHECK_CUDA(totals);
    CHECK_CONTIGUOUS(ptrs);
    CHECK_CONTIGUOUS(global_rows);
    CHECK_CONTIGUOUS(totals);
    TORCH_CHECK(ptrs.dtype() == torch::kInt64, "ptrs must be int64");
    TORCH_CHECK(global_rows.dtype() == torch::kFloat32, "global_rows must be float32");
    TORCH_CHECK(totals.dtype() == torch::kFloat32, "totals must be float32");

    int threads = 256;
    int blocks = (int)((k_cols + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_global_rows_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        global_rows.data_ptr<float>(),
        totals.data_ptr<float>(),
        k_cols,
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void row_norm_colsum(
    torch::Tensor p,
    torch::Tensor global_rows,
    torch::Tensor colsum,
    int64_t b_rows,
    int64_t k_cols
) {
    if (b_rows <= 0) return;
    CHECK_CUDA(p);
    CHECK_CUDA(global_rows);
    CHECK_CUDA(colsum);
    CHECK_CONTIGUOUS(p);
    CHECK_CONTIGUOUS(global_rows);
    CHECK_CONTIGUOUS(colsum);
    TORCH_CHECK(p.dtype() == torch::kFloat32, "p must be float32");
    TORCH_CHECK(global_rows.dtype() == torch::kFloat32, "global_rows must be float32");
    TORCH_CHECK(colsum.dtype() == torch::kFloat32, "colsum must be float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    row_norm_colsum_kernel<<<(unsigned int)b_rows, 256, 0, stream>>>(
        p.data_ptr<float>(),
        global_rows.data_ptr<float>(),
        colsum.data_ptr<float>(),
        b_rows,
        k_cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void col_norm(
    torch::Tensor p,
    torch::Tensor colsum,
    torch::Tensor symm_rows,
    int64_t b_rows,
    int64_t k_cols,
    int accumulate_rows
) {
    int64_t n = b_rows * k_cols;
    if (n <= 0) return;
    CHECK_CUDA(p);
    CHECK_CUDA(colsum);
    CHECK_CUDA(symm_rows);
    CHECK_CONTIGUOUS(p);
    CHECK_CONTIGUOUS(colsum);
    CHECK_CONTIGUOUS(symm_rows);
    TORCH_CHECK(p.dtype() == torch::kFloat32, "p must be float32");
    TORCH_CHECK(colsum.dtype() == torch::kFloat32, "colsum must be float32");
    TORCH_CHECK(symm_rows.dtype() == torch::kFloat32, "symm_rows must be float32");

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    col_norm_kernel<<<blocks, threads, 0, stream>>>(
        p.data_ptr<float>(),
        colsum.data_ptr<float>(),
        symm_rows.data_ptr<float>(),
        n,
        k_cols,
        accumulate_rows
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scale_zero_iter(torch::Tensor p, torch::Tensor totals, int64_t n) {
    if (n <= 0) return;
    CHECK_CUDA(p);
    CHECK_CUDA(totals);
    CHECK_CONTIGUOUS(p);
    CHECK_CONTIGUOUS(totals);
    TORCH_CHECK(p.dtype() == torch::kFloat32, "p must be float32");
    TORCH_CHECK(totals.dtype() == torch::kFloat32, "totals must be float32");

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scale_zero_iter_kernel<<<blocks, threads, 0, stream>>>(
        p.data_ptr<float>(), totals.data_ptr<float>(), n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("zero_symm_rows", &zero_symm_rows, "zero symmetric row buffer");
    m.def("init_exp_rows", &init_exp_rows, "exp logits and local prototype sums");
    m.def("reduce_global_rows", &reduce_global_rows, "UVA peer row-sum reduction");
    m.def("row_norm_colsum", &row_norm_colsum, "row normalize and local column sums");
    m.def("col_norm", &col_norm, "column normalize and optionally accumulate next rows");
    m.def("scale_zero_iter", &scale_zero_iter, "n_iterations=0 final scaling");
}
'''

_ext = None
_resource_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dinov2_sinkhorn_symm_cuda_ext", CUDA_SRC)
    return _ext


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float16:
        return 2
    return 1


def _group_key(group):
    return id(group)


def _get_resources(b_rows: int, k_cols: int, device: torch.device, group):
    key = (b_rows, k_cols, device.index, _group_key(group))
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    symm_rows = symm_mem.empty((k_cols + 2,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(symm_rows, group)

    p = torch.empty((b_rows, k_cols), device=device, dtype=torch.float32)
    global_rows = torch.empty((k_cols,), device=device, dtype=torch.float32)
    colsum = torch.empty((b_rows,), device=device, dtype=torch.float32)
    totals = torch.empty((2,), device=device, dtype=torch.float32)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "symm_rows": symm_rows,
        "hdl": hdl,
        "p": p,
        "global_rows": global_rows,
        "colsum": colsum,
        "totals": totals,
        "ptrs": ptrs,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    n_masked_patches_tensor: torch.Tensor,
    n_iterations: int = 3,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert teacher_output.is_cuda, "teacher_output must be CUDA"
    assert teacher_output.dim() == 2, "teacher_output must be [B_local, K]"

    ext = _get_ext()

    x = teacher_output
    if not x.is_contiguous():
        x = x.contiguous()

    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x = x.float().contiguous()

    b_rows = int(x.shape[0])
    k_cols = int(x.shape[1])
    n = b_rows * k_cols

    res = _get_resources(b_rows, k_cols, x.device, group)
    symm_rows = res["symm_rows"]
    hdl = res["hdl"]
    p = res["p"]
    global_rows = res["global_rows"]
    colsum = res["colsum"]
    totals = res["totals"]
    ptrs = res["ptrs"]

    # total_batch only affects the exact n_iterations == 0 path; for >=1 it cancels
    # through Sinkhorn row normalization, so avoid synchronizing this scalar on the hot path.
    if n_iterations == 0:
        local_count = float(n_masked_patches_tensor.detach().cpu().item())
    else:
        local_count = float(b_rows)

    ext.zero_symm_rows(symm_rows, k_cols)
    ext.init_exp_rows(
        x,
        p,
        symm_rows,
        b_rows,
        k_cols,
        float(teacher_temp),
        _dtype_enum(x.dtype),
        local_count,
    )

    # Publish local prototype sums, then reduce peer buffers via UVA loads.
    hdl.barrier(channel=0)
    ext.reduce_global_rows(ptrs, global_rows, totals, k_cols, int(hdl.world_size))
    # Ensure no rank overwrites its symmetric buffer before peers finish reading it.
    hdl.barrier(channel=1)

    if n_iterations == 0:
        ext.scale_zero_iter(p, totals, n)
        return p

    for it in range(int(n_iterations)):
        last = it == int(n_iterations) - 1

        if not last:
            ext.zero_symm_rows(symm_rows, k_cols)

        ext.row_norm_colsum(p, global_rows, colsum, b_rows, k_cols)
        ext.col_norm(p, colsum, symm_rows, b_rows, k_cols, 0 if last else 1)

        if not last:
            hdl.barrier(channel=0)
            ext.reduce_global_rows(ptrs, global_rows, totals, k_cols, int(hdl.world_size))
            hdl.barrier(channel=1)

    return p