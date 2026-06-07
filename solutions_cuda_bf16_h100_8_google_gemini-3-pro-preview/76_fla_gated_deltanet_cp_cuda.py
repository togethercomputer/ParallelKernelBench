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
#include <cstdint>

__global__ void delta_recurrent_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ beta,
    const int64_t* __restrict__ a_log_ptrs,
    const int64_t* __restrict__ dt_bias_ptrs,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ local_state,
    uint32_t* __restrict__ local_flags,
    float* __restrict__ peer_state,
    uint32_t* __restrict__ peer_flags,
    int rank,
    int world_size,
    int B, int T, int QH, int VH, int K, int V
) {
    int b_h = blockIdx.x;
    if (b_h >= B * VH) return;
    int b = b_h / VH;
    int hv = b_h % VH;
    int qh = hv / (VH / QH);
    int tx = threadIdx.x;

    extern __shared__ float smem[];
    float* smem_state = smem;                                    // size K * V
    float* sq = smem + K * V;                                    // size K
    float* sk = smem + K * V + K;                                // size K
    float* r_q = smem + K * V + 2 * K;                           // size blockDim.x
    float* r_k = smem + K * V + 2 * K + blockDim.x;              // size blockDim.x

    // 1) Pipeline Wait: Wait for rank-1 to signal and write the initial state directly into our memory
    if (rank > 0 && local_flags != nullptr && local_state != nullptr) {
        if (tx == 0) {
            volatile uint32_t* flag_ptr = &local_flags[b_h];
            while (*flag_ptr == 0) {
                // busy wait for P2P flag
            }
        }
        __syncthreads();
        for (int i = tx; i < K * V; i += blockDim.x) {
            smem_state[i] = local_state[b_h * K * V + i];
        }
    } else {
        for (int i = tx; i < K * V; i += blockDim.x) {
            smem_state[i] = 0.0f;
        }
    }
    __syncthreads();

    // 2) UVA read of all-gathered 1D tensors (avoid explicit NCCL)
    float a_scale, bias;
    if (a_log_ptrs != nullptr && dt_bias_ptrs != nullptr) {
        int chunk_size = VH / world_size;
        int owner = hv / chunk_size;
        int local_offset = hv % chunk_size;
        const __nv_bfloat16* a_log_owner = (const __nv_bfloat16*)a_log_ptrs[owner];
        const __nv_bfloat16* dt_bias_owner = (const __nv_bfloat16*)dt_bias_ptrs[owner];
        a_scale = expf(__bfloat162float(a_log_owner[local_offset]));
        bias = __bfloat162float(dt_bias_owner[local_offset]);
    } else {
        // Fallback for isolated single-rank execution
        const __nv_bfloat16* a_log_ptr = (const __nv_bfloat16*)a_log_ptrs;
        const __nv_bfloat16* dt_bias_ptr = (const __nv_bfloat16*)dt_bias_ptrs;
        a_scale = expf(__bfloat162float(a_log_ptr[hv]));
        bias = __bfloat162float(dt_bias_ptr[hv]);
    }

    float scale_q_k = 1.0f / sqrtf((float)K);

    // 3) Process chunk elements with fused norms
    for (int t = 0; t < T; ++t) {
        float local_q_sq = 0.0f;
        float local_k_sq = 0.0f;
        for (int i = tx; i < K; i += blockDim.x) {
            float q_val = __bfloat162float(q[b * T * QH * K + t * QH * K + qh * K + i]);
            float k_val = __bfloat162float(k[b * T * QH * K + t * QH * K + qh * K + i]);
            sq[i] = q_val;
            sk[i] = k_val;
            local_q_sq += q_val * q_val;
            local_k_sq += k_val * k_val;
        }
        
        r_q[tx] = local_q_sq;
        r_k[tx] = local_k_sq;
        __syncthreads();

        // Warp reduction for PyTorch equivalent F.normalize L2 norm
        if (tx == 0) {
            float sum_q = 0.0f, sum_k = 0.0f;
            for (int i = 0; i < blockDim.x; ++i) {
                sum_q += r_q[i];
                sum_k += r_k[i];
            }
            float norm_q = sqrtf(sum_q);
            float norm_k = sqrtf(sum_k);
            r_q[0] = 1.0f / (norm_q < 1e-6f ? 1e-6f : norm_q);
            r_k[0] = 1.0f / (norm_k < 1e-6f ? 1e-6f : norm_k);
        }
        __syncthreads();

        float q_scale = r_q[0] * scale_q_k;
        float k_scale = r_k[0];

        for (int i = tx; i < K; i += blockDim.x) {
            sq[i] *= q_scale;
            sk[i] *= k_scale;
        }
        __syncthreads();

        float gate_t = __bfloat162float(gate[b * T * VH + t * VH + hv]);
        float beta_t = __bfloat162float(beta[b * T * VH + t * VH + hv]);
        float sp = (gate_t + bias > 20.0f) ? (gate_t + bias) : logf(1.0f + expf(gate_t + bias));
        float decay_t = expf(-a_scale * sp);

        for (int v_idx = tx; v_idx < V; v_idx += blockDim.x) {
            float v_val = __bfloat162float(v[b * T * VH * V + t * VH * V + hv * V + v_idx]);

            float proj = 0.0f;
            #pragma unroll 4
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                proj += sk[k_idx] * (smem_state[k_idx * V + v_idx] * decay_t);
            }

            float upd = (v_val - proj) * beta_t;

            float out_val = 0.0f;
            #pragma unroll 4
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                float s = smem_state[k_idx * V + v_idx] * decay_t + sk[k_idx] * upd;
                smem_state[k_idx * V + v_idx] = s;
                out_val += sq[k_idx] * s;
            }

            out[b * T * VH * V + t * VH * V + hv * V + v_idx] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }

    // 4) Pipeline Trigger: Write final state to rank+1 using P2P pointers + sync flag
    if (rank < world_size - 1 && peer_state != nullptr && peer_flags != nullptr) {
        for (int i = tx; i < K * V; i += blockDim.x) {
            peer_state[b_h * K * V + i] = smem_state[i];
        }
        __threadfence_system(); // ensure state arrives before flag
        __syncthreads();
        if (tx == 0) {
            atomicExch(&peer_flags[b_h], 1);
        }
    }
}

void launch_delta_recurrent(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor a_log_ptrs,
    torch::Tensor dt_bias_ptrs,
    torch::Tensor out,
    torch::Tensor local_state,
    torch::Tensor local_flags,
    int64_t peer_state_ptr,
    int64_t peer_flags_ptr,
    int rank,
    int world_size
) {
    int B = q.size(0);
    int T = q.size(1);
    int QH = q.size(2);
    int K = q.size(3);
    int VH = v.size(2);
    int V = v.size(3);

    int threads = 256;
    int blocks = B * VH;
    int smem_size = (K * V + 2 * K + 2 * threads) * sizeof(float);

    if (smem_size > 49152) {
        // Boost limits up dynamically for H100
        cudaFuncSetAttribute(delta_recurrent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    float* l_state = local_state.numel() > 0 ? local_state.data_ptr<float>() : nullptr;
    uint32_t* l_flags = local_flags.numel() > 0 ? (uint32_t*)local_flags.data_ptr<int32_t>() : nullptr;
    float* p_state = peer_state_ptr ? reinterpret_cast<float*>(peer_state_ptr) : nullptr;
    uint32_t* p_flags = peer_flags_ptr ? reinterpret_cast<uint32_t*>(peer_flags_ptr) : nullptr;

    const int64_t* a_ptrs = a_log_ptrs.defined() ? a_log_ptrs.data_ptr<int64_t>() : nullptr;
    const int64_t* d_ptrs = dt_bias_ptrs.defined() ? dt_bias_ptrs.data_ptr<int64_t>() : nullptr;

    delta_recurrent_kernel<<<blocks, threads, smem_size, stream>>>(
        (__nv_bfloat16*)q.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)k.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)v.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)gate.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)beta.data_ptr<at::BFloat16>(),
        a_ptrs, d_ptrs,
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        l_state, l_flags, p_state, p_flags,
        rank, world_size,
        B, T, QH, VH, K, V
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_delta_recurrent", &launch_delta_recurrent, "DeltaNet CP kernel");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("deltanet_recurrent_cp_ext", CUDA_SRC)
    return _ext

_symm_cache = {}

def _get_symm_state(B, VH, K, V, dtype, device, group_id, group):
    key = (B, VH, K, V, dtype, device, group_id)
    if key in _symm_cache:
        return _symm_cache[key]
    
    # [batch_heads, K, V] buffer acting as receiver mailbox locally on each rank
    buf_state = symm_mem.empty((B * VH, K, V), dtype=torch.float32, device=device)
    hdl_state = symm_mem.rendezvous(buf_state, group)
    
    # Flags enabling pipelining without NCCL overhead
    buf_flags = symm_mem.empty((B * VH,), dtype=torch.int32, device=device)
    hdl_flags = symm_mem.rendezvous(buf_flags, group)

    world_size = dist.get_world_size(group)
    chunk_size = VH // world_size

    # Used for UVA all-gather of static parameters without collective overhead on hot path
    buf_a = symm_mem.empty((chunk_size,), dtype=dtype, device=device)
    hdl_a = symm_mem.rendezvous(buf_a, group)

    buf_dt = symm_mem.empty((chunk_size,), dtype=dtype, device=device)
    hdl_dt = symm_mem.rendezvous(buf_dt, group)

    res = (buf_state, hdl_state, buf_flags, hdl_flags, buf_a, hdl_a, buf_dt, hdl_dt)
    _symm_cache[key] = res
    return res

@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    
    B, T, QH, K = q.shape
    VH = v.size(2)
    V = v.size(3)

    out = torch.empty((B, T, VH, V), dtype=q.dtype, device=q.device)

    is_dist = dist.is_initialized()
    world_size = dist.get_world_size(group) if is_dist else 1
    rank = dist.get_rank(group) if is_dist else 0

    if world_size > 1:
        buf_state, hdl_state, buf_flags, hdl_flags, buf_a, hdl_a, buf_dt, hdl_dt = _get_symm_state(
            B, VH, K, V, a_log.dtype, q.device, id(group), group
        )

        # Clear flags locally ensuring we don't accidentally match old triggers 
        buf_flags.zero_()
        hdl_flags.barrier(channel=0)

        # Drop 1D tensors to symmetric memory pool for zero-NCCL UVA all-gathers in the kernel
        buf_a.copy_(a_log)
        buf_dt.copy_(dt_bias)
        hdl_a.barrier(channel=0)
        hdl_dt.barrier(channel=0)

        peer_state_ptr = int(hdl_state.buffer_ptrs[rank + 1]) if rank < world_size - 1 else 0
        peer_flags_ptr = int(hdl_flags.buffer_ptrs[rank + 1]) if rank < world_size - 1 else 0

        a_ptrs = torch.tensor(hdl_a.buffer_ptrs, dtype=torch.int64, device=q.device)
        d_ptrs = torch.tensor(hdl_dt.buffer_ptrs, dtype=torch.int64, device=q.device)

        local_state = buf_state
        local_flags = buf_flags
    else:
        peer_state_ptr = 0
        peer_flags_ptr = 0
        a_ptrs = torch.tensor([a_log.data_ptr()], dtype=torch.int64, device=q.device)
        d_ptrs = torch.tensor([dt_bias.data_ptr()], dtype=torch.int64, device=q.device)
        local_state = torch.empty(0, device=q.device)
        local_flags = torch.empty(0, device=q.device)

    _get_ext().launch_delta_recurrent(
        q.contiguous(), k.contiguous(), v.contiguous(),
        gate.contiguous(), beta.contiguous(),
        a_ptrs, d_ptrs, out,
        local_state, local_flags, peer_state_ptr, peer_flags_ptr,
        rank, world_size
    )

    return out