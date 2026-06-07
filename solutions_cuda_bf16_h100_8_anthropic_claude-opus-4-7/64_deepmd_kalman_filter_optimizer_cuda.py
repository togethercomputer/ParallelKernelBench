"""
DeepMD blockwise local Kalman-filter optimizer update with custom CUDA + symmetric memory.

Strategy:
- Compute local tmp_i = lambda + H_i^T P_i H_i with cuBLAS/torch matmul (small blocks).
- All-reduce the scalar `tmp` via symmetric-memory peer-pointer kernel (single fp32 reduce).
- Update weights/P locally with fused custom kernels.
- All-gather weights via symmetric memory: each rank writes its concatenated weight block
  to its symmetric buffer, then every rank reads peers' buffers via UVA pointers.
"""

from typing import List, Tuple

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

// All-reduce SUM for a small fp32 scalar buffer using peer pointers.
__global__ void allreduce_sum_f32_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s = 0.f;
        for (int r = 0; r < world_size; ++r) {
            const float* p = (const float*)ptrs[r];
            s += p[idx];
        }
        out[idx] = s;
    }
}

void allreduce_sum_f32(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int64_t n
) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 64;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_sum_f32_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, out.data_ptr<float>(), world_size, n);
}

// Fused weight update: w[i] = w[i] + scalar * K[i], scalar = A*err
// K = P @ H computed elsewhere (we use torch matmul before this).
__global__ void fused_w_update_bf16_kernel(
    __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ K,
    float scalar,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float wv = __bfloat162float(w[idx]);
        float kv = __bfloat162float(K[idx]);
        w[idx] = __float2bfloat16(wv + scalar * kv);
    }
}

void fused_w_update_bf16(
    torch::Tensor w,
    torch::Tensor K,
    double scalar
) {
    int64_t n = w.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_w_update_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)w.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)K.data_ptr<at::BFloat16>(),
        (float)scalar,
        n);
}

// Fused covariance update: P = (1/lam) * (P - A * K K^T)
// P is [n,n], K is [n,1].
__global__ void fused_p_update_bf16_kernel(
    __nv_bfloat16* __restrict__ P,
    const __nv_bfloat16* __restrict__ K,
    float inv_lam,
    float A,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float pv = __bfloat162float(P[row * n + col]);
        float kr = __bfloat162float(K[row]);
        float kc = __bfloat162float(K[col]);
        float v = inv_lam * (pv - A * kr * kc);
        P[row * n + col] = __float2bfloat16(v);
    }
}

void fused_p_update_bf16(
    torch::Tensor P,
    torch::Tensor K,
    double inv_lam,
    double A
) {
    int n = P.size(0);
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_p_update_bf16_kernel<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)P.data_ptr<at::BFloat16>(),
        (const __nv_bfloat16*)K.data_ptr<at::BFloat16>(),
        (float)inv_lam,
        (float)A,
        n);
}

// Copy peer's contiguous bf16 buffer (UVA) into local destination.
__global__ void copy_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

void copy_bf16_from_ptr(
    int64_t src_ptr,
    torch::Tensor dst,
    int64_t n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)src_ptr,
        (__nv_bfloat16*)dst.data_ptr<at::BFloat16>(),
        n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allreduce_sum_f32", &allreduce_sum_f32, "scalar all-reduce");
    m.def("fused_w_update_bf16", &fused_w_update_bf16, "fused w update");
    m.def("fused_p_update_bf16", &fused_p_update_bf16, "fused P update");
    m.def("copy_bf16_from_ptr", &copy_bf16_from_ptr, "copy bf16 from UVA pointer");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("deepmd_kalman_ext", CUDA_SRC)
    return _ext


_scalar_symm = None  # (buf, hdl, ptrs_tensor, out)


def _get_scalar_symm(device):
    global _scalar_symm
    if _scalar_symm is not None:
        return _scalar_symm
    buf = symm_mem.empty(1, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    out = torch.empty(1, device=device, dtype=torch.float32)
    _scalar_symm = (buf, hdl, ptrs_tensor, out)
    return _scalar_symm


_gather_symm_cache = {}


def _get_gather_symm(total_bytes_numel, device):
    """Symmetric buffer (bf16) sized to the global max of per-rank concatenated weights."""
    key = (total_bytes_numel, device)
    if key in _gather_symm_cache:
        return _gather_symm_cache[key]
    buf = symm_mem.empty(total_bytes_numel, device=device, dtype=torch.bfloat16)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _gather_symm_cache[key] = (buf, hdl)
    return _gather_symm_cache[key]


@torch.no_grad()
def solution(
    H: List[torch.Tensor],
    error: torch.Tensor,
    weights: List[torch.Tensor],
    P: List[torch.Tensor],
    kalman_lambda: float,
    kalman_nue: float = 0.9987,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    weights_num = len(weights)
    device = weights[0].device
    dtype = weights[0].dtype

    lam = torch.as_tensor(kalman_lambda, dtype=dtype, device=device)
    err = error.to(device=device, dtype=dtype)
    lam_f = float(lam.item()) if lam.numel() == 1 else float(lam)

    # 1. Local denominator (compute in fp32 for stability of the reduce).
    tmp_local = torch.zeros(1, dtype=torch.float32, device=device)
    Ks: List[torch.Tensor] = [None] * weights_num
    for i in range(weights_num):
        # K_i = P_i @ H_i  (we'll reuse this in step 2).
        Ki = torch.matmul(P[i], H[i])
        Ks[i] = Ki
        # H_i^T @ K_i is scalar [1,1].
        s = torch.matmul(H[i].transpose(0, 1), Ki).reshape(1).to(torch.float32)
        tmp_local += s
    tmp_local += float(lam_f)

    ext = _get_ext()

    # 2. All-reduce scalar via symmetric memory (only if distributed).
    if dist.is_initialized() and dist.get_world_size() > 1:
        buf, hdl, ptrs_tensor, out_scalar = _get_scalar_symm(device)
        buf.copy_(tmp_local)
        hdl.barrier(channel=0)
        ext.allreduce_sum_f32(ptrs_tensor, out_scalar, 1)
        hdl.barrier(channel=1)
        tmp_global = out_scalar
    else:
        tmp_global = tmp_local

    A = (1.0 / float(tmp_global.item()))
    err_f = float(err.item())
    inv_lam = 1.0 / lam_f
    scalar_w = A * err_f

    # 3. Fused local updates.
    for i in range(weights_num):
        Ki = Ks[i]
        # weights[i] += A*err*K
        ext.fused_w_update_bf16(weights[i], Ki, scalar_w)
        # P[i] = (1/lam) * (P[i] - A*K K^T)
        ext.fused_p_update_bf16(P[i], Ki, inv_lam, A)

    # 4. All-gather weights via symmetric memory if distributed.
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        local_shape = [int(t.shape[0]) for t in weights]
        shape_list = [None] * world_size
        dist.all_gather_object(shape_list, local_shape)

        per_rank_total = [sum(s) for s in shape_list]
        max_total = max(per_rank_total)

        buf, hdl = _get_gather_symm(max_total, device)

        # Pack local weights into symmetric buffer.
        local_total = per_rank_total[rank]
        offset = 0
        for w in weights:
            n = w.numel()
            buf[offset:offset + n].copy_(w.reshape(-1))
            offset += n

        hdl.barrier(channel=0)

        # Pull each peer's buffer and split.
        result: List[torch.Tensor] = []
        for r in range(world_size):
            shapes_r = shape_list[r]
            total_r = per_rank_total[r]
            peer_ptr = int(hdl.buffer_ptrs[r])
            gathered = torch.empty(total_r, dtype=torch.bfloat16, device=device)
            ext.copy_bf16_from_ptr(peer_ptr, gathered, total_r)
            off = 0
            for s in shapes_r:
                result.append(gathered[off:off + s].reshape(-1, 1).to(dtype))
                off += s

        hdl.barrier(channel=1)
        weights = result

    # 5. Decay lambda.
    nue_t = torch.as_tensor(kalman_nue, dtype=lam.dtype, device=device)
    kalman_lambda_next = nue_t * lam + 1 - nue_t

    return weights, P, kalman_lambda_next