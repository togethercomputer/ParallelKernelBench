from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


# Strategy:
# - Put each rank's train feature shard and labels in symmetric memory once per shape.
# - Each rank computes only its own query shard against all train shards by reading peer UVA pointers directly.
# - BF16 GEMMs are issued from a custom extension through cuBLAS tensor-core kernels; no NCCL broadcast/gather.
# - A custom CUDA merge kernel keeps a running sorted top-k, avoiding materializing/gathering global candidates.


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>
#include <limits>

#define CUDA_CHECK(stmt) do {                                      \
    cudaError_t err__ = (stmt);                                    \
    TORCH_CHECK(err__ == cudaSuccess, cudaGetErrorString(err__));  \
} while (0)

#define CUBLAS_CHECK(stmt) do {                                    \
    cublasStatus_t st__ = (stmt);                                  \
    TORCH_CHECK(st__ == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ",   \
                static_cast<int>(st__));                           \
} while (0)

static thread_local cublasHandle_t tls_handle = nullptr;

static cublasHandle_t get_cublas_handle() {
    if (tls_handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&tls_handle));
        CUBLAS_CHECK(cublasSetMathMode(tls_handle, CUBLAS_TENSOR_OP_MATH));
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    CUBLAS_CHECK(cublasSetStream(tls_handle, stream));
    return tls_handle;
}

__device__ __forceinline__ long long read_label_as_i64(
    const void* __restrict__ labels,
    int dtype_enum,
    int64_t idx
) {
    // dtype_enum: 0=int64, 1=int32, 2=int16, 3=uint8
    if (dtype_enum == 0) {
        return reinterpret_cast<const long long*>(labels)[idx];
    } else if (dtype_enum == 1) {
        return static_cast<long long>(reinterpret_cast<const int*>(labels)[idx]);
    } else if (dtype_enum == 2) {
        return static_cast<long long>(reinterpret_cast<const short*>(labels)[idx]);
    } else {
        return static_cast<long long>(reinterpret_cast<const unsigned char*>(labels)[idx]);
    }
}

__device__ __forceinline__ void write_label_from_i64(
    void* __restrict__ labels,
    int dtype_enum,
    int64_t idx,
    long long v
) {
    if (dtype_enum == 0) {
        reinterpret_cast<long long*>(labels)[idx] = v;
    } else if (dtype_enum == 1) {
        reinterpret_cast<int*>(labels)[idx] = static_cast<int>(v);
    } else if (dtype_enum == 2) {
        reinterpret_cast<short*>(labels)[idx] = static_cast<short>(v);
    } else {
        reinterpret_cast<unsigned char*>(labels)[idx] = static_cast<unsigned char>(v);
    }
}

__global__ void init_topk_kernel(
    __nv_bfloat16* __restrict__ out_sims,
    void* __restrict__ out_labels,
    int64_t total,
    int dtype_enum
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        out_sims[idx] = __float2bfloat16(-INFINITY);
        write_label_from_i64(out_labels, dtype_enum, idx, 0LL);
    }
}

__device__ __forceinline__ bool better_pair(float s0, int id0, float s1, int id1) {
    return (s0 > s1) || ((s0 == s1) && (id0 < id1));
}

__global__ void merge_topk_bf16_kernel(
    const __nv_bfloat16* __restrict__ sims,   // [Q, T], row-major
    const void* __restrict__ shard_labels,    // [T]
    __nv_bfloat16* __restrict__ out_sims,     // [Q, K], running sorted top-k
    void* __restrict__ out_labels,            // [Q, K]
    int64_t Q,
    int64_t T,
    int K,
    int label_dtype_enum
) {
    int64_t q = static_cast<int64_t>(blockIdx.x);
    if (q >= Q) return;

    extern __shared__ unsigned char smem[];
    float* selected_s = reinterpret_cast<float*>(smem);
    long long* selected_l = reinterpret_cast<long long*>(selected_s + K);
    int* selected_id = reinterpret_cast<int*>(selected_l + K);

    float* red_s = reinterpret_cast<float*>(selected_id + K);
    long long* red_l = reinterpret_cast<long long*>(red_s + blockDim.x);
    int* red_id = reinterpret_cast<int*>(red_l + blockDim.x);

    const int tid = threadIdx.x;
    const int64_t total_candidates = T + static_cast<int64_t>(K);

    for (int j = 0; j < K; ++j) {
        float best_s = -INFINITY;
        long long best_l = 0LL;
        int best_id = 0x7fffffff;

        for (int64_t c = tid; c < total_candidates; c += blockDim.x) {
            int cid = static_cast<int>(c);

            bool used = false;
            #pragma unroll 1
            for (int p = 0; p < j; ++p) {
                if (selected_id[p] == cid) {
                    used = true;
                    break;
                }
            }
            if (used) continue;

            float score;
            long long label;
            if (c < K) {
                score = __bfloat162float(out_sims[q * K + c]);
                label = read_label_as_i64(out_labels, label_dtype_enum, q * K + c);
            } else {
                int64_t t = c - K;
                score = __bfloat162float(sims[q * T + t]);
                label = read_label_as_i64(shard_labels, label_dtype_enum, t);
            }

            if (better_pair(score, cid, best_s, best_id)) {
                best_s = score;
                best_l = label;
                best_id = cid;
            }
        }

        red_s[tid] = best_s;
        red_l[tid] = best_l;
        red_id[tid] = best_id;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float os = red_s[tid + stride];
                int oid = red_id[tid + stride];
                long long ol = red_l[tid + stride];
                if (better_pair(os, oid, red_s[tid], red_id[tid])) {
                    red_s[tid] = os;
                    red_l[tid] = ol;
                    red_id[tid] = oid;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            selected_s[j] = red_s[0];
            selected_l[j] = red_l[0];
            selected_id[j] = red_id[0];
        }
        __syncthreads();
    }

    if (tid < K) {
        out_sims[q * K + tid] = __float2bfloat16(selected_s[tid]);
        write_label_from_i64(out_labels, label_dtype_enum, q * K + tid, selected_l[tid]);
    }
}

static int label_dtype_enum(torch::Tensor labels) {
    if (labels.scalar_type() == torch::kInt64) return 0;
    if (labels.scalar_type() == torch::kInt32) return 1;
    if (labels.scalar_type() == torch::kInt16) return 2;
    if (labels.scalar_type() == torch::kUInt8) return 3;
    TORCH_CHECK(false, "train_labels_rank dtype must be int64/int32/int16/uint8");
}

void knn_bf16_uva(
    torch::Tensor queries,                     // bf16 [Q, D]
    std::vector<uint64_t> train_t_ptrs,         // each bf16 [D, T]
    std::vector<uint64_t> label_ptrs,           // each labels [T]
    torch::Tensor workspace,                    // bf16 [Q, T]
    torch::Tensor out_sims,                     // bf16 [Q, K]
    torch::Tensor out_labels,                   // same dtype as labels [Q, K]
    int64_t D,
    int64_t T,
    int64_t K
) {
    TORCH_CHECK(queries.is_cuda(), "queries must be CUDA");
    TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA");
    TORCH_CHECK(out_sims.is_cuda(), "out_sims must be CUDA");
    TORCH_CHECK(out_labels.is_cuda(), "out_labels must be CUDA");

    TORCH_CHECK(queries.scalar_type() == torch::kBFloat16, "queries must be bfloat16");
    TORCH_CHECK(workspace.scalar_type() == torch::kBFloat16, "workspace must be bfloat16");
    TORCH_CHECK(out_sims.scalar_type() == torch::kBFloat16, "out_sims must be bfloat16");

    TORCH_CHECK(queries.is_contiguous(), "queries must be contiguous");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
    TORCH_CHECK(out_sims.is_contiguous(), "out_sims must be contiguous");
    TORCH_CHECK(out_labels.is_contiguous(), "out_labels must be contiguous");

    TORCH_CHECK(train_t_ptrs.size() == label_ptrs.size(), "pointer vector size mismatch");
    TORCH_CHECK(D <= std::numeric_limits<int>::max(), "D too large for cuBLAS int API");
    TORCH_CHECK(T <= std::numeric_limits<int>::max(), "T too large for cuBLAS int API");
    TORCH_CHECK(queries.size(0) <= std::numeric_limits<int>::max(), "Q too large for cuBLAS int API");
    TORCH_CHECK(K > 0 && K <= 1024, "K must be in [1, 1024]");

    const int64_t Q = queries.size(0);
    if (Q == 0) return;

    const int l_dtype = label_dtype_enum(out_labels);

    const int threads_init = 256;
    int blocks_init = static_cast<int>((Q * K + threads_init - 1) / threads_init);
    if (blocks_init > 65535) blocks_init = 65535;

    init_topk_kernel<<<blocks_init, threads_init, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<__nv_bfloat16*>(out_sims.data_ptr<at::BFloat16>()),
        out_labels.data_ptr(),
        Q * K,
        l_dtype
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int m = static_cast<int>(T);
    const int n = static_cast<int>(Q);
    const int k = static_cast<int>(D);

    const void* B_query = static_cast<const void*>(queries.data_ptr<at::BFloat16>());
    void* C_ws = static_cast<void*>(workspace.data_ptr<at::BFloat16>());

    const int merge_threads = 256;
    const size_t shmem =
        static_cast<size_t>(K + merge_threads) *
        (sizeof(float) + sizeof(long long) + sizeof(int));

    CUDA_CHECK(cudaFuncSetAttribute(
        merge_topk_bf16_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        98304
    ));

    for (size_t peer = 0; peer < train_t_ptrs.size(); ++peer) {
        const void* A_train_t =
            reinterpret_cast<const void*>(static_cast<uintptr_t>(train_t_ptrs[peer]));

        // Row-major [Q,D] @ row-major [D,T] -> row-major [Q,T].
        // cuBLAS column-major view:
        //   C_col[T,Q] = A_col[T,D] (train_t) * B_col[D,Q] (queries)
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            A_train_t,
            CUDA_R_16BF,
            m,
            B_query,
            CUDA_R_16BF,
            k,
            &beta,
            C_ws,
            CUDA_R_16BF,
            m,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        const void* peer_labels =
            reinterpret_cast<const void*>(static_cast<uintptr_t>(label_ptrs[peer]));

        merge_topk_bf16_kernel<<<static_cast<unsigned int>(Q), merge_threads, shmem,
                                 at::cuda::getCurrentCUDAStream().stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(workspace.data_ptr<at::BFloat16>()),
            peer_labels,
            reinterpret_cast<__nv_bfloat16*>(out_sims.data_ptr<at::BFloat16>()),
            out_labels.data_ptr(),
            Q,
            T,
            static_cast<int>(K),
            l_dtype
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_bf16_uva", &knn_bf16_uva,
          "DINOv2 distributed kNN over symmetric-memory UVA train shards");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("dinov2_knn_bf16_uva_ext", CUDA_SRC)
    return _ext


_train_cache = {}


def _cache_key(
    train_features_rank_T: torch.Tensor,
    train_labels_rank: torch.Tensor,
    group: dist.ProcessGroup,
):
    return (
        tuple(train_features_rank_T.shape),
        train_features_rank_T.dtype,
        tuple(train_labels_rank.shape),
        train_labels_rank.dtype,
        train_features_rank_T.device.index,
        id(group),
    )


def _get_symmetric_train(
    train_features_rank_T: torch.Tensor,
    train_labels_rank: torch.Tensor,
    group: dist.ProcessGroup,
):
    key = _cache_key(train_features_rank_T, train_labels_rank, group)
    cached = _train_cache.get(key)
    if cached is not None:
        return cached

    feat_symm = symm_mem.empty(
        tuple(train_features_rank_T.shape),
        device=train_features_rank_T.device,
        dtype=train_features_rank_T.dtype,
    )
    label_symm = symm_mem.empty(
        tuple(train_labels_rank.shape),
        device=train_labels_rank.device,
        dtype=train_labels_rank.dtype,
    )

    feat_hdl = symm_mem.rendezvous(feat_symm, group)
    label_hdl = symm_mem.rendezvous(label_symm, group)

    feat_ptrs = [int(p) for p in feat_hdl.buffer_ptrs]
    label_ptrs = [int(p) for p in label_hdl.buffer_ptrs]

    cached = {
        "feat_symm": feat_symm,
        "label_symm": label_symm,
        "feat_hdl": feat_hdl,
        "label_hdl": label_hdl,
        "feat_ptrs": feat_ptrs,
        "label_ptrs": label_ptrs,
    }
    _train_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    test_features_rank: torch.Tensor,
    train_features_rank_T: torch.Tensor,
    train_labels_rank: torch.Tensor,
    max_k: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Distributed DINOv2 k-NN using symmetric-memory UVA train shards and a custom
    BF16 CUDA/cuBLAS top-k pipeline.  Runs on every rank and returns top-k for
    this rank's local query shard.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    group = group or dist.group.WORLD

    if max_k > train_features_rank_T.shape[1]:
        raise ValueError("max_k must not exceed the local train shard size")

    if test_features_rank.dtype != torch.bfloat16 or train_features_rank_T.dtype != torch.bfloat16:
        raise TypeError("This optimized path expects BF16 query/train features")

    assert test_features_rank.is_cuda
    assert train_features_rank_T.is_cuda
    assert train_labels_rank.is_cuda
    assert test_features_rank.device == train_features_rank_T.device
    assert train_labels_rank.device == train_features_rank_T.device

    queries = test_features_rank.contiguous()
    train_t = train_features_rank_T.contiguous()
    labels = train_labels_rank.contiguous().reshape(-1)

    Q = int(queries.shape[0])
    D = int(queries.shape[1])
    T = int(train_t.shape[1])
    K = int(max_k)

    if train_t.shape[0] != D:
        raise ValueError("test feature dimension must match train_features_rank_T.shape[0]")
    if labels.numel() != T:
        raise ValueError("train_labels_rank must contain exactly T_local labels")

    res = _get_symmetric_train(train_t, labels, group)

    res["feat_symm"].copy_(train_t)
    res["label_symm"].reshape(-1).copy_(labels)

    # Symmetric-memory device visibility barrier; no NCCL collectives are used.
    res["feat_hdl"].barrier(channel=0)
    res["label_hdl"].barrier(channel=1)

    out_sims = torch.empty((Q, K), device=queries.device, dtype=torch.bfloat16)
    out_labels = torch.empty((Q, K), device=queries.device, dtype=labels.dtype)
    workspace = torch.empty((Q, T), device=queries.device, dtype=torch.bfloat16)

    _get_ext().knn_bf16_uva(
        queries,
        res["feat_ptrs"],
        res["label_ptrs"],
        workspace,
        out_sims,
        out_labels,
        D,
        T,
        K,
    )

    return out_sims, out_labels