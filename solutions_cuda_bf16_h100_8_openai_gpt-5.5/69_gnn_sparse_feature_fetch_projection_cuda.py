from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>

#define CUBLAS_CHECK(cmd) do {                                      \
    cublasStatus_t _status = (cmd);                                 \
    TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                   \
                "cuBLAS failure, status=", (int)_status);           \
} while (0)

template <typename id_t>
__global__ void gather_bf16_kernel(
    const id_t* __restrict__ ids,
    const long long* __restrict__ shard_ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t q_offset,
    int64_t q_count,
    int64_t D,
    int64_t shard_size,
    int world_size
) {
    int64_t total = q_count * D;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = tid; linear < total; linear += stride) {
        int64_t q = linear / D;
        int64_t d = linear - q * D;

        long long gid = (long long)ids[q_offset + q];
        long long owner_ll = gid / shard_size;
        if (owner_ll >= world_size) owner_ll = world_size - 1;
        if (owner_ll < 0) owner_ll = 0;

        long long local_row = gid - owner_ll * shard_size;
        const __nv_bfloat16* base =
            reinterpret_cast<const __nv_bfloat16*>(
                static_cast<uintptr_t>(shard_ptrs[(int)owner_ll])
            );

        out[linear] = base[local_row * D + d];
    }
}

template <typename id_t>
__global__ void gather_f32_kernel(
    const id_t* __restrict__ ids,
    const long long* __restrict__ shard_ptrs,
    float* __restrict__ out,
    int64_t q_offset,
    int64_t q_count,
    int64_t D,
    int64_t shard_size,
    int world_size
) {
    int64_t total = q_count * D;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = tid; linear < total; linear += stride) {
        int64_t q = linear / D;
        int64_t d = linear - q * D;

        long long gid = (long long)ids[q_offset + q];
        long long owner_ll = gid / shard_size;
        if (owner_ll >= world_size) owner_ll = world_size - 1;
        if (owner_ll < 0) owner_ll = 0;

        long long local_row = gid - owner_ll * shard_size;
        const float* base =
            reinterpret_cast<const float*>(
                static_cast<uintptr_t>(shard_ptrs[(int)owner_ll])
            );

        out[linear] = base[local_row * D + d];
    }
}

void launch_gather(
    torch::Tensor input_node_ids,
    torch::Tensor shard_ptrs,
    torch::Tensor gathered,
    int64_t q_offset,
    int64_t q_count,
    int64_t D,
    int64_t shard_size,
    int world_size,
    int dtype_enum,
    int id_dtype_enum
) {
    TORCH_CHECK(input_node_ids.is_cuda(), "input_node_ids must be CUDA");
    TORCH_CHECK(shard_ptrs.is_cuda(), "shard_ptrs must be CUDA");
    TORCH_CHECK(gathered.is_cuda(), "gathered must be CUDA");
    TORCH_CHECK(input_node_ids.is_contiguous(), "input_node_ids must be contiguous");
    TORCH_CHECK(shard_ptrs.is_contiguous(), "shard_ptrs must be contiguous");
    TORCH_CHECK(gathered.is_contiguous(), "gathered must be contiguous");

    if (q_count <= 0 || D <= 0) return;

    int threads = 256;
    int64_t total = q_count * D;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const long long* ptrs =
        reinterpret_cast<const long long*>(shard_ptrs.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        __nv_bfloat16* out =
            reinterpret_cast<__nv_bfloat16*>(gathered.data_ptr<at::BFloat16>());
        if (id_dtype_enum == 0) {
            gather_bf16_kernel<int64_t><<<blocks, threads, 0, stream>>>(
                input_node_ids.data_ptr<int64_t>(), ptrs, out,
                q_offset, q_count, D, shard_size, world_size);
        } else {
            gather_bf16_kernel<int><<<blocks, threads, 0, stream>>>(
                input_node_ids.data_ptr<int>(), ptrs, out,
                q_offset, q_count, D, shard_size, world_size);
        }
    } else {
        float* out = gathered.data_ptr<float>();
        if (id_dtype_enum == 0) {
            gather_f32_kernel<int64_t><<<blocks, threads, 0, stream>>>(
                input_node_ids.data_ptr<int64_t>(), ptrs, out,
                q_offset, q_count, D, shard_size, world_size);
        } else {
            gather_f32_kernel<int><<<blocks, threads, 0, stream>>>(
                input_node_ids.data_ptr<int>(), ptrs, out,
                q_offset, q_count, D, shard_size, world_size);
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_bf16_cublas(
    torch::Tensor gathered,      // [q_count, D], row-major BF16
    torch::Tensor proj,          // [D, O], row-major BF16
    torch::Tensor out,           // [Q, O], row-major BF16
    int64_t q_count,
    int64_t D,
    int64_t O,
    int64_t out_q_offset
) {
    TORCH_CHECK(gathered.is_cuda() && proj.is_cuda() && out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(gathered.dtype() == torch::kBFloat16 &&
                proj.dtype() == torch::kBFloat16 &&
                out.dtype() == torch::kBFloat16,
                "BF16 tensors expected");
    TORCH_CHECK(gathered.is_contiguous() && proj.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");

    if (q_count <= 0 || D <= 0 || O <= 0) return;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const void* A = static_cast<const void*>(proj.data_ptr<at::BFloat16>());
    const void* B = static_cast<const void*>(gathered.data_ptr<at::BFloat16>());
    void* C = static_cast<void*>(out.data_ptr<at::BFloat16>() + out_q_offset * O);

    // Row-major C(q,O) = gathered(q,D) @ proj(D,O)
    // as column-major C^T(O,q) = proj^T(O,D) @ gathered^T(D,q).
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)O,
        (int)q_count,
        (int)D,
        &alpha,
        A,
        CUDA_R_16BF,
        (int)O,
        B,
        CUDA_R_16BF,
        (int)D,
        &beta,
        C,
        CUDA_R_16BF,
        (int)O,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

void project_f32_cublas(
    torch::Tensor gathered,      // [q_count, D], row-major FP32
    torch::Tensor proj,          // [D, O], row-major FP32
    torch::Tensor out,           // [Q, O], row-major FP32
    int64_t q_count,
    int64_t D,
    int64_t O,
    int64_t out_q_offset
) {
    TORCH_CHECK(gathered.is_cuda() && proj.is_cuda() && out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(gathered.dtype() == torch::kFloat32 &&
                proj.dtype() == torch::kFloat32 &&
                out.dtype() == torch::kFloat32,
                "FP32 tensors expected");
    TORCH_CHECK(gathered.is_contiguous() && proj.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");

    if (q_count <= 0 || D <= 0 || O <= 0) return;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const float* A = proj.data_ptr<float>();
    const float* B = gathered.data_ptr<float>();
    float* C = out.data_ptr<float>() + out_q_offset * O;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        (int)O,
        (int)q_count,
        (int)D,
        &alpha,
        A,
        (int)O,
        B,
        (int)D,
        &beta,
        C,
        (int)O
    ));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather", &launch_gather,
          "UVA symmetric sparse embedding gather");
    m.def("project_bf16_cublas", &project_bf16_cublas,
          "BF16 row-major projection via cuBLAS tensor cores");
    m.def("project_f32_cublas", &project_f32_cublas,
          "FP32 row-major projection via cuBLAS");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "gnn_sparse_fetch_project_symm_uva_bf16_h100_ext",
            CUDA_SRC,
        )
    return _ext


# Tuned for H100: enough rows per GEMM to use tensor cores well while keeping
# staging small enough for double-buffered overlap.
_CHUNK_Q = 2048

_symm_cache = {}
_work_cache = {}


def _device_key(device: torch.device):
    return (device.type, device.index if device.index is not None else torch.cuda.current_device())


def _get_symmetric_embedding_resources(
    shard_size: int,
    embed_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    group,
    world_size: int,
):
    key = (shard_size, embed_dim, dtype, _device_key(device), id(group), world_size)
    cached = _symm_cache.get(key)
    if cached is not None:
        return cached

    bufs = []
    hdls = []
    ptr_tensors = []

    # Two symmetric buffers avoid immediate overwrite hazards between consecutive
    # invocations while ranks are still consuming peer data from the previous call.
    for _ in range(2):
        buf = symm_mem.empty((shard_size, embed_dim), device=device, dtype=dtype)
        hdl = symm_mem.rendezvous(buf, group)
        ptrs = torch.tensor(
            [int(p) for p in hdl.buffer_ptrs],
            device=device,
            dtype=torch.int64,
        )
        bufs.append(buf)
        hdls.append(hdl)
        ptr_tensors.append(ptrs)

    cached = {
        "bufs": bufs,
        "hdls": hdls,
        "ptr_tensors": ptr_tensors,
        "counter": 0,
    }
    _symm_cache[key] = cached
    return cached


def _get_work_buffers(
    num_queries: int,
    embed_dim: int,
    out_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    chunk_q = min(_CHUNK_Q, max(1, num_queries))
    key = (chunk_q, num_queries, embed_dim, out_dim, dtype, _device_key(device))
    cached = _work_cache.get(key)
    if cached is not None:
        return cached

    tmp0 = torch.empty((chunk_q, embed_dim), device=device, dtype=dtype)
    tmp1 = torch.empty((chunk_q, embed_dim), device=device, dtype=dtype)
    out = torch.empty((num_queries, out_dim), device=device, dtype=dtype)
    comm_stream = torch.cuda.Stream(device=device)

    cached = {
        "chunk_q": chunk_q,
        "tmp": [tmp0, tmp1],
        "out": out,
        "comm_stream": comm_stream,
    }
    _work_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    local_embedding_shard: torch.Tensor,
    input_node_ids: torch.Tensor,
    proj_matrix: torch.Tensor,
    num_total_nodes: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Distributed sparse feature fetch + projection using symmetric-memory UVA
    peer reads and custom CUDA/cuBLAS kernels.  Optimized BF16 path uses H100
    tensor cores for projection and avoids NCCL all-to-all.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert local_embedding_shard.is_cuda
    assert input_node_ids.is_cuda
    assert proj_matrix.is_cuda
    assert input_node_ids.dtype in (torch.int64, torch.int32)
    assert local_embedding_shard.dtype == proj_matrix.dtype
    assert local_embedding_shard.dtype in (torch.bfloat16, torch.float32)

    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)

    device = local_embedding_shard.device
    shard_size = (num_total_nodes + world_size - 1) // world_size
    embed_dim = int(local_embedding_shard.shape[1])
    num_queries = int(input_node_ids.numel())
    out_dim = int(proj_matrix.shape[1])

    assert int(proj_matrix.shape[0]) == embed_dim

    if num_queries == 0:
        return torch.empty(
            (0, out_dim),
            device=device,
            dtype=local_embedding_shard.dtype,
        )

    ext = _get_ext()

    ids = input_node_ids.contiguous()
    proj = proj_matrix.contiguous()

    symm = _get_symmetric_embedding_resources(
        shard_size,
        embed_dim,
        local_embedding_shard.dtype,
        device,
        group,
        world_size,
    )

    symm_idx = symm["counter"] & 1
    symm["counter"] += 1

    emb_buf = symm["bufs"][symm_idx]
    hdl = symm["hdls"][symm_idx]
    ptr_tensor = symm["ptr_tensors"][symm_idx]

    local_rows = int(local_embedding_shard.shape[0])
    if local_rows == shard_size and local_embedding_shard.is_contiguous():
        emb_buf.copy_(local_embedding_shard)
    else:
        emb_buf[:local_rows, :].copy_(local_embedding_shard)

    # Publish this rank's shard before peers issue UVA loads.
    hdl.barrier(channel=symm_idx)

    work = _get_work_buffers(
        num_queries,
        embed_dim,
        out_dim,
        local_embedding_shard.dtype,
        device,
    )
    chunk_q = work["chunk_q"]
    tmp = work["tmp"]
    out = work["out"]

    dtype_enum = 0 if local_embedding_shard.dtype == torch.bfloat16 else 1
    id_dtype_enum = 0 if ids.dtype == torch.int64 else 1

    cur_stream = torch.cuda.current_stream(device)

    def _launch_gather(q_off: int, q_cnt: int, buf_idx: int):
        ext.launch_gather(
            ids,
            ptr_tensor,
            tmp[buf_idx],
            int(q_off),
            int(q_cnt),
            int(embed_dim),
            int(shard_size),
            int(world_size),
            int(dtype_enum),
            int(id_dtype_enum),
        )

    def _launch_project(q_off: int, q_cnt: int, buf_idx: int):
        if dtype_enum == 0:
            ext.project_bf16_cublas(
                tmp[buf_idx],
                proj,
                out,
                int(q_cnt),
                int(embed_dim),
                int(out_dim),
                int(q_off),
            )
        else:
            ext.project_f32_cublas(
                tmp[buf_idx],
                proj,
                out,
                int(q_cnt),
                int(embed_dim),
                int(out_dim),
                int(q_off),
            )

    # Small/medium batches: one staging buffer, one gather, one GEMM.
    if num_queries <= chunk_q:
        _launch_gather(0, num_queries, 0)
        _launch_project(0, num_queries, 0)
        return out

    # Large batches: gather chunk k+1 on a communication stream while projecting
    # chunk k on the current compute stream.
    comm_stream = work["comm_stream"]
    gather_done = [torch.cuda.Event(blocking=False), torch.cuda.Event(blocking=False)]
    compute_done = [None, None]

    chunks = []
    q = 0
    while q < num_queries:
        qc = min(chunk_q, num_queries - q)
        chunks.append((q, qc))
        q += qc

    prev = None
    for ci, (q_off, q_cnt) in enumerate(chunks):
        buf_idx = ci & 1

        if compute_done[buf_idx] is not None:
            comm_stream.wait_event(compute_done[buf_idx])

        with torch.cuda.stream(comm_stream):
            _launch_gather(q_off, q_cnt, buf_idx)
            gather_done[buf_idx].record(comm_stream)

        if prev is not None:
            p_q_off, p_q_cnt, p_buf_idx = prev
            cur_stream.wait_event(gather_done[p_buf_idx])
            _launch_project(p_q_off, p_q_cnt, p_buf_idx)
            ev = torch.cuda.Event(blocking=False)
            ev.record(cur_stream)
            compute_done[p_buf_idx] = ev

        prev = (q_off, q_cnt, buf_idx)

    p_q_off, p_q_cnt, p_buf_idx = prev
    cur_stream.wait_event(gather_done[p_buf_idx])
    _launch_project(p_q_off, p_q_cnt, p_buf_idx)

    return out