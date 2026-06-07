import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.autograd import Function
from typing import Tuple, Any

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p, int64_t i) {
    return static_cast<float>(p[i]);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* p, int64_t i, float v) {
    p[i] = static_cast<T>(v);
}

template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* p, int64_t i, float v) {
    p[i] = __float2bfloat16(v);
}

__device__ __forceinline__ void atomicMinFloat(float* addr, float value) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int old = *addr_i;
    while (value < __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int old = *addr_i;
    while (value > __int_as_float(old)) {
        int assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__global__ void init_stats_kernel(float* stats) {
    if (threadIdx.x == 0) {
        stats[0] = 0.0f;        // valid count
        stats[1] = 0.0f;        // pg sum
        stats[2] = 0.0f;        // ratio sum
        stats[3] = INFINITY;    // ratio min
        stats[4] = -INFINITY;   // ratio max
        stats[5] = 0.0f;        // k3 sum
        stats[6] = 0.0f;        // entropy sum
        stats[7] = 0.0f;
    }
}

template <typename LogT, typename OldT, typename AdvT>
__global__ void ce_stats_kernel(
    const LogT* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const OldT* __restrict__ old_logprobs,
    const AdvT* __restrict__ advantages,
    float* __restrict__ per_token_logprobs,
    float* __restrict__ per_token_loss,
    float* __restrict__ stats,
    int64_t nrows,
    int64_t vocab,
    int64_t ignore_index
) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int tid = threadIdx.x;
    __shared__ float smem[1024];
    __shared__ float row_max;
    __shared__ float row_sum;
    __shared__ float label_logit;

    int64_t label = labels[row];
    bool valid = (label != ignore_index);

    if (!valid) {
        for (int64_t v = tid; v < vocab; v += blockDim.x) {
            // no-op, just keep block shape identical
        }
        if (tid == 0) {
            per_token_logprobs[row] = 0.0f;
            per_token_loss[row] = 0.0f;
        }
        return;
    }

    float local_max = -INFINITY;
    int64_t base = static_cast<int64_t>(row) * vocab;
    for (int64_t v = tid; v < vocab; v += blockDim.x) {
        float x = load_as_float<LogT>(logits, base + v);
        local_max = fmaxf(local_max, x);
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) {
        row_max = smem[0];
        label_logit = load_as_float<LogT>(logits, base + label);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int64_t v = tid; v < vocab; v += blockDim.x) {
        float x = load_as_float<LogT>(logits, base + v);
        local_sum += expf(x - row_max);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        row_sum = smem[0];
        float ce = logf(row_sum) + row_max - label_logit;
        float new_logp = -ce;

        float oldp = load_as_float<OldT>(old_logprobs, row);
        float adv = load_as_float<AdvT>(advantages, row);
        float delta = fminf(20.0f, fmaxf(-20.0f, new_logp - oldp));
        float ratio = expf(delta);
        float pg = -(ratio * adv);
        float k3 = ratio - delta - 1.0f;

        per_token_logprobs[row] = new_logp;
        per_token_loss[row] = pg;

        atomicAdd(stats + 0, 1.0f);
        atomicAdd(stats + 1, pg);
        atomicAdd(stats + 2, ratio);
        atomicMinFloat(stats + 3, ratio);
        atomicMaxFloat(stats + 4, ratio);
        atomicAdd(stats + 5, k3);
        atomicAdd(stats + 6, ce);
    }
}

__global__ void reduce_global_stats_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ loss,
    float* __restrict__ metrics,
    float* __restrict__ n_global_out,
    int world_size
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float count = 0.0f;
    float pg_sum = 0.0f;
    float ratio_sum = 0.0f;
    float ratio_min = INFINITY;
    float ratio_max = -INFINITY;
    float k3_sum = 0.0f;
    float entropy_sum = 0.0f;

    for (int r = 0; r < world_size; ++r) {
        const float* s = reinterpret_cast<const float*>(ptrs[r]);
        count += s[0];
        pg_sum += s[1];
        ratio_sum += s[2];
        ratio_min = fminf(ratio_min, s[3]);
        ratio_max = fmaxf(ratio_max, s[4]);
        k3_sum += s[5];
        entropy_sum += s[6];
    }

    float denom = fmaxf(count, 1.0f);
    loss[0] = pg_sum / denom;
    n_global_out[0] = denom;

    metrics[0] = ratio_sum / denom;
    metrics[1] = ratio_min;
    metrics[2] = ratio_max;
    metrics[3] = k3_sum / denom;
    metrics[4] = entropy_sum / denom;
}

template <typename LogT, typename OldT, typename AdvT>
__global__ void grad_logits_kernel(
    const LogT* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const OldT* __restrict__ old_logprobs,
    const AdvT* __restrict__ advantages,
    const float* __restrict__ per_token_logprobs,
    const float* __restrict__ n_global,
    const float* __restrict__ grad_loss,
    LogT* __restrict__ grad_logits,
    int64_t nrows,
    int64_t vocab,
    int64_t ignore_index
) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int tid = threadIdx.x;
    int64_t base = static_cast<int64_t>(row) * vocab;
    int64_t label = labels[row];
    bool valid = (label != ignore_index);

    __shared__ float smem[1024];
    __shared__ float row_max;
    __shared__ float row_sum;
    __shared__ float factor;

    if (!valid) {
        for (int64_t v = tid; v < vocab; v += blockDim.x) {
            store_from_float<LogT>(grad_logits, base + v, 0.0f);
        }
        return;
    }

    float local_max = -INFINITY;
    for (int64_t v = tid; v < vocab; v += blockDim.x) {
        local_max = fmaxf(local_max, load_as_float<LogT>(logits, base + v));
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) row_max = smem[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int64_t v = tid; v < vocab; v += blockDim.x) {
        local_sum += expf(load_as_float<LogT>(logits, base + v) - row_max);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        row_sum = smem[0];

        float oldp = load_as_float<OldT>(old_logprobs, row);
        float adv = load_as_float<AdvT>(advantages, row);
        float delta = fminf(20.0f, fmaxf(-20.0f, per_token_logprobs[row] - oldp));
        float ratio = expf(delta);

        factor = grad_loss[0] * ratio * adv / fmaxf(n_global[0], 1.0f);
    }
    __syncthreads();

    for (int64_t v = tid; v < vocab; v += blockDim.x) {
        float x = load_as_float<LogT>(logits, base + v);
        float p = expf(x - row_max) / row_sum;
        float g = factor * (p - (v == label ? 1.0f : 0.0f));
        store_from_float<LogT>(grad_logits, base + v, g);
    }
}

template <typename OldT, typename AdvT>
__global__ void grad_adv_kernel(
    const int64_t* __restrict__ labels,
    const OldT* __restrict__ old_logprobs,
    const AdvT* __restrict__ advantages,
    const float* __restrict__ per_token_logprobs,
    const float* __restrict__ n_global,
    const float* __restrict__ grad_loss,
    AdvT* __restrict__ grad_adv,
    int64_t nrows,
    int64_t ignore_index
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nrows) return;

    float g = 0.0f;
    if (labels[i] != ignore_index) {
        float oldp = load_as_float<OldT>(old_logprobs, i);
        float delta = fminf(20.0f, fmaxf(-20.0f, per_token_logprobs[i] - oldp));
        float ratio = expf(delta);
        float ce = -per_token_logprobs[i];
        g = grad_loss[0] * ratio * ce / fmaxf(n_global[0], 1.0f);
    }
    store_from_float<AdvT>(grad_adv, i, g);
}

static inline cudaDataType_t dtype_to_cuda(torch::ScalarType t) {
    if (t == torch::kBFloat16) return CUDA_R_16BF;
    if (t == torch::kFloat32) return CUDA_R_32F;
    TORCH_CHECK(false, "supported dtypes: bfloat16, float32");
}

void cublas_check(cublasStatus_t st) {
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed");
}

void linear_forward(torch::Tensor hidden, torch::Tensor weight, torch::Tensor logits) {
    CHECK_CUDA(hidden); CHECK_CUDA(weight); CHECK_CUDA(logits);
    CHECK_CONTIG(hidden); CHECK_CONTIG(weight); CHECK_CONTIG(logits);

    int64_t N64 = hidden.size(0);
    int64_t H64 = hidden.size(1);
    int64_t V64 = weight.size(0);

    int N = static_cast<int>(N64);
    int H = static_cast<int>(H64);
    int V = static_cast<int>(V64);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublas_check(cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream()));

    float alpha = 1.0f;
    float beta = 0.0f;
    cudaDataType_t dt = dtype_to_cuda(hidden.scalar_type());

    // Row-major hidden[N,H] @ weight[V,H]^T -> logits[N,V].
    // Interpreted as column-major logits^T[V,N] = weight[V,H] * hidden^T[H,N].
    cublas_check(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        V, N, H,
        &alpha,
        weight.data_ptr(), dt, H,
        hidden.data_ptr(), dt, H,
        &beta,
        logits.data_ptr(), dt, V,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

void linear_backward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor grad_logits,
    torch::Tensor grad_hidden,
    torch::Tensor grad_weight
) {
    CHECK_CUDA(hidden); CHECK_CUDA(weight); CHECK_CUDA(grad_logits);
    CHECK_CUDA(grad_hidden); CHECK_CUDA(grad_weight);
    CHECK_CONTIG(hidden); CHECK_CONTIG(weight); CHECK_CONTIG(grad_logits);
    CHECK_CONTIG(grad_hidden); CHECK_CONTIG(grad_weight);

    int N = static_cast<int>(hidden.size(0));
    int H = static_cast<int>(hidden.size(1));
    int V = static_cast<int>(weight.size(0));

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublas_check(cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream()));

    float alpha = 1.0f;
    float beta = 0.0f;
    cudaDataType_t dt = dtype_to_cuda(hidden.scalar_type());

    // grad_hidden[N,H] = grad_logits[N,V] @ weight[V,H]
    // column-major grad_hidden^T[H,N] = weight^T[H,V] @ grad_logits^T[V,N]
    cublas_check(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        H, N, V,
        &alpha,
        weight.data_ptr(), dt, H,
        grad_logits.data_ptr(), dt, V,
        &beta,
        grad_hidden.data_ptr(), dt, H,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // grad_weight[V,H] = grad_logits[N,V]^T @ hidden[N,H]
    // column-major grad_weight^T[H,V] = hidden^T[H,N] @ grad_logits[N,V]
    cublas_check(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        H, V, N,
        &alpha,
        hidden.data_ptr(), dt, H,
        grad_logits.data_ptr(), dt, V,
        &beta,
        grad_weight.data_ptr(), dt, H,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

void init_stats(torch::Tensor stats) {
    init_stats_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        stats.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename LogT, typename OldT, typename AdvT>
void launch_ce_t(torch::Tensor logits, torch::Tensor labels, torch::Tensor oldp,
                 torch::Tensor adv, torch::Tensor logp_out, torch::Tensor loss_out,
                 torch::Tensor stats, int64_t ignore_index) {
    int64_t nrows = labels.numel();
    int64_t vocab = logits.size(1);
    ce_stats_kernel<LogT, OldT, AdvT><<<static_cast<int>(nrows), 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const LogT*>(logits.data_ptr()),
        labels.data_ptr<int64_t>(),
        reinterpret_cast<const OldT*>(oldp.data_ptr()),
        reinterpret_cast<const AdvT*>(adv.data_ptr()),
        logp_out.data_ptr<float>(),
        loss_out.data_ptr<float>(),
        stats.data_ptr<float>(),
        nrows,
        vocab,
        ignore_index
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ce_stats(torch::Tensor logits, torch::Tensor labels, torch::Tensor oldp,
              torch::Tensor adv, torch::Tensor logp_out, torch::Tensor loss_out,
              torch::Tensor stats, int64_t ignore_index) {
    CHECK_CUDA(logits); CHECK_CUDA(labels); CHECK_CUDA(oldp); CHECK_CUDA(adv);
    CHECK_CONTIG(logits); CHECK_CONTIG(labels); CHECK_CONTIG(oldp); CHECK_CONTIG(adv);

    bool log_bf16 = logits.scalar_type() == torch::kBFloat16;
    bool old_bf16 = oldp.scalar_type() == torch::kBFloat16;
    bool adv_bf16 = adv.scalar_type() == torch::kBFloat16;

    if (log_bf16 && old_bf16 && adv_bf16) {
        launch_ce_t<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(logits, labels, oldp, adv, logp_out, loss_out, stats, ignore_index);
    } else if (log_bf16 && !old_bf16 && !adv_bf16) {
        launch_ce_t<__nv_bfloat16, float, float>(logits, labels, oldp, adv, logp_out, loss_out, stats, ignore_index);
    } else if (log_bf16 && !old_bf16 && adv_bf16) {
        launch_ce_t<__nv_bfloat16, float, __nv_bfloat16>(logits, labels, oldp, adv, logp_out, loss_out, stats, ignore_index);
    } else if (log_bf16 && old_bf16 && !adv_bf16) {
        launch_ce_t<__nv_bfloat16, __nv_bfloat16, float>(logits, labels, oldp, adv, logp_out, loss_out, stats, ignore_index);
    } else {
        launch_ce_t<float, float, float>(logits, labels, oldp, adv, logp_out, loss_out, stats, ignore_index);
    }
}

void reduce_global_stats(torch::Tensor ptrs, torch::Tensor loss,
                         torch::Tensor metrics, torch::Tensor n_global) {
    int world_size = static_cast<int>(ptrs.size(0));
    reduce_global_stats_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        loss.data_ptr<float>(),
        metrics.data_ptr<float>(),
        n_global.data_ptr<float>(),
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename LogT, typename OldT, typename AdvT>
void launch_grad_logits_t(torch::Tensor logits, torch::Tensor labels, torch::Tensor oldp,
                          torch::Tensor adv, torch::Tensor logp, torch::Tensor n_global,
                          torch::Tensor grad_loss, torch::Tensor grad_logits,
                          int64_t ignore_index) {
    int64_t nrows = labels.numel();
    int64_t vocab = logits.size(1);
    grad_logits_kernel<LogT, OldT, AdvT><<<static_cast<int>(nrows), 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const LogT*>(logits.data_ptr()),
        labels.data_ptr<int64_t>(),
        reinterpret_cast<const OldT*>(oldp.data_ptr()),
        reinterpret_cast<const AdvT*>(adv.data_ptr()),
        logp.data_ptr<float>(),
        n_global.data_ptr<float>(),
        grad_loss.data_ptr<float>(),
        reinterpret_cast<LogT*>(grad_logits.data_ptr()),
        nrows,
        vocab,
        ignore_index
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grad_logits(torch::Tensor logits, torch::Tensor labels, torch::Tensor oldp,
                 torch::Tensor adv, torch::Tensor logp, torch::Tensor n_global,
                 torch::Tensor grad_loss, torch::Tensor grad_logits_out,
                 int64_t ignore_index) {
    bool log_bf16 = logits.scalar_type() == torch::kBFloat16;
    bool old_bf16 = oldp.scalar_type() == torch::kBFloat16;
    bool adv_bf16 = adv.scalar_type() == torch::kBFloat16;

    if (log_bf16 && old_bf16 && adv_bf16) {
        launch_grad_logits_t<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(logits, labels, oldp, adv, logp, n_global, grad_loss, grad_logits_out, ignore_index);
    } else if (log_bf16 && !old_bf16 && !adv_bf16) {
        launch_grad_logits_t<__nv_bfloat16, float, float>(logits, labels, oldp, adv, logp, n_global, grad_loss, grad_logits_out, ignore_index);
    } else if (log_bf16 && !old_bf16 && adv_bf16) {
        launch_grad_logits_t<__nv_bfloat16, float, __nv_bfloat16>(logits, labels, oldp, adv, logp, n_global, grad_loss, grad_logits_out, ignore_index);
    } else if (log_bf16 && old_bf16 && !adv_bf16) {
        launch_grad_logits_t<__nv_bfloat16, __nv_bfloat16, float>(logits, labels, oldp, adv, logp, n_global, grad_loss, grad_logits_out, ignore_index);
    } else {
        launch_grad_logits_t<float, float, float>(logits, labels, oldp, adv, logp, n_global, grad_loss, grad_logits_out, ignore_index);
    }
}

template <typename OldT, typename AdvT>
void launch_grad_adv_t(torch::Tensor labels, torch::Tensor oldp, torch::Tensor adv,
                       torch::Tensor logp, torch::Tensor n_global,
                       torch::Tensor grad_loss, torch::Tensor grad_adv_out,
                       int64_t ignore_index) {
    int64_t nrows = labels.numel();
    int threads = 256;
    int blocks = static_cast<int>((nrows + threads - 1) / threads);
    grad_adv_kernel<OldT, AdvT><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        labels.data_ptr<int64_t>(),
        reinterpret_cast<const OldT*>(oldp.data_ptr()),
        reinterpret_cast<const AdvT*>(adv.data_ptr()),
        logp.data_ptr<float>(),
        n_global.data_ptr<float>(),
        grad_loss.data_ptr<float>(),
        reinterpret_cast<AdvT*>(grad_adv_out.data_ptr()),
        nrows,
        ignore_index
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grad_adv(torch::Tensor labels, torch::Tensor oldp, torch::Tensor adv,
              torch::Tensor logp, torch::Tensor n_global,
              torch::Tensor grad_loss, torch::Tensor grad_adv_out,
              int64_t ignore_index) {
    bool old_bf16 = oldp.scalar_type() == torch::kBFloat16;
    bool adv_bf16 = adv.scalar_type() == torch::kBFloat16;

    if (old_bf16 && adv_bf16) {
        launch_grad_adv_t<__nv_bfloat16, __nv_bfloat16>(labels, oldp, adv, logp, n_global, grad_loss, grad_adv_out, ignore_index);
    } else if (!old_bf16 && adv_bf16) {
        launch_grad_adv_t<float, __nv_bfloat16>(labels, oldp, adv, logp, n_global, grad_loss, grad_adv_out, ignore_index);
    } else if (old_bf16 && !adv_bf16) {
        launch_grad_adv_t<__nv_bfloat16, float>(labels, oldp, adv, logp, n_global, grad_loss, grad_adv_out, ignore_index);
    } else {
        launch_grad_adv_t<float, float>(labels, oldp, adv, logp, n_global, grad_loss, grad_adv_out, ignore_index);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "BF16/FP32 linear forward via cuBLAS");
    m.def("linear_backward", &linear_backward, "BF16/FP32 linear backward via cuBLAS");
    m.def("init_stats", &init_stats, "Initialize local symmetric stats");
    m.def("ce_stats", &ce_stats, "Fused CE/logprob/loss/local stats");
    m.def("reduce_global_stats", &reduce_global_stats, "UVA peer-pointer global stats reduce");
    m.def("grad_logits", &grad_logits, "Build surrogate grad logits");
    m.def("grad_adv", &grad_adv, "Build optional advantage gradient");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("grpo_loss_bf16_symm_h100_ext", CUDA_SRC)
    return _ext


_comm_cache = {}


def _get_comm_resources(device: torch.device):
    assert dist.is_initialized(), "torch.distributed must be initialized"
    key = (device.index, dist.get_world_size())
    if key in _comm_cache:
        return _comm_cache[key]

    stats = symm_mem.empty((8,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(stats, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    _comm_cache[key] = (stats, hdl, ptrs)
    return stats, hdl, ptrs


class _GRPOLossCUDA(Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        ignore_index: int,
    ):
        ext = _get_ext()

        hidden_c = hidden_states.contiguous()
        weight_c = weight.contiguous()
        labels_c = labels.contiguous()
        old_c = old_logprobs.contiguous()
        adv_c = advantages.contiguous()

        bsz, seqlen, hidden_dim = hidden_c.shape
        vocab = weight_c.shape[0]
        n_tokens = bsz * seqlen

        hidden_2d = hidden_c.view(n_tokens, hidden_dim)
        labels_1d = labels_c.view(n_tokens)
        old_1d = old_c.view(n_tokens)
        adv_1d = adv_c.view(n_tokens)

        logits = torch.empty((n_tokens, vocab), device=hidden_c.device, dtype=hidden_c.dtype)
        ext.linear_forward(hidden_2d, weight_c, logits)

        per_token_logprobs_1d = torch.empty((n_tokens,), device=hidden_c.device, dtype=torch.float32)
        per_token_loss_1d = torch.empty((n_tokens,), device=hidden_c.device, dtype=torch.float32)

        stats, hdl, ptrs = _get_comm_resources(hidden_c.device)
        ext.init_stats(stats)
        ext.ce_stats(
            logits,
            labels_1d,
            old_1d,
            adv_1d,
            per_token_logprobs_1d,
            per_token_loss_1d,
            stats,
            int(ignore_index),
        )

        hdl.barrier(channel=0)

        loss = torch.empty((), device=hidden_c.device, dtype=torch.float32)
        metrics = torch.empty((5,), device=hidden_c.device, dtype=torch.float32)
        n_global = torch.empty((), device=hidden_c.device, dtype=torch.float32)

        ext.reduce_global_stats(ptrs, loss, metrics, n_global)

        # Prevent stats reuse while a slower peer may still read this rank's symmetric stats.
        hdl.barrier(channel=1)

        per_token_logprobs = per_token_logprobs_1d.view_as(labels_c)
        per_token_loss = per_token_loss_1d.view_as(labels_c)

        ctx.save_for_backward(
            hidden_2d,
            weight_c,
            logits,
            labels_1d,
            old_1d,
            adv_1d,
            per_token_logprobs_1d,
            n_global,
        )
        ctx.hidden_shape = tuple(hidden_c.shape)
        ctx.adv_shape = tuple(adv_c.shape)
        ctx.ignore_index = int(ignore_index)
        ctx.needs_adv_grad = bool(ctx.needs_input_grad[4])

        ctx.mark_non_differentiable(per_token_logprobs, per_token_loss, metrics)
        return loss, per_token_logprobs, per_token_loss, metrics

    @staticmethod
    def backward(ctx, grad_loss, grad_logprobs_unused, grad_ptloss_unused, grad_metrics_unused):
        ext = _get_ext()

        (
            hidden_2d,
            weight,
            logits,
            labels_1d,
            old_1d,
            adv_1d,
            per_token_logprobs_1d,
            n_global,
        ) = ctx.saved_tensors

        grad_loss_c = grad_loss.contiguous()
        if grad_loss_c.dtype != torch.float32:
            grad_loss_c = grad_loss_c.float()

        grad_logits = torch.empty_like(logits)
        ext.grad_logits(
            logits,
            labels_1d,
            old_1d,
            adv_1d,
            per_token_logprobs_1d,
            n_global,
            grad_loss_c,
            grad_logits,
            int(ctx.ignore_index),
        )

        grad_hidden_2d = torch.empty_like(hidden_2d)
        grad_weight = torch.empty_like(weight)
        ext.linear_backward(hidden_2d, weight, grad_logits, grad_hidden_2d, grad_weight)

        grad_adv = None
        if ctx.needs_adv_grad:
            grad_adv_1d = torch.empty_like(adv_1d)
            ext.grad_adv(
                labels_1d,
                old_1d,
                adv_1d,
                per_token_logprobs_1d,
                n_global,
                grad_loss_c,
                grad_adv_1d,
                int(ctx.ignore_index),
            )
            grad_adv = grad_adv_1d.view(ctx.adv_shape)

        return (
            grad_hidden_2d.view(ctx.hidden_shape),
            grad_weight,
            None,
            None,
            grad_adv,
            None,
        )


def solution(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.is_cuda and weight.is_cuda and labels.is_cuda
    assert old_logprobs.is_cuda and advantages.is_cuda
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert hidden_states.dtype in (torch.bfloat16, torch.float32)
    assert weight.dtype == hidden_states.dtype

    loss, per_token_logprobs, per_token_loss, metrics = _GRPOLossCUDA.apply(
        hidden_states,
        weight,
        labels,
        old_logprobs,
        advantages,
        int(ignore_index),
    )
    return loss, None, per_token_logprobs, per_token_loss, metrics