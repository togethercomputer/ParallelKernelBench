"""
Symmetric-memory CUDA implementation for Gated DeltaNet CP forward.

Strategy:
- Replace both all-to-all transposes with symmetric-memory UVA peer loads/stores.
- Each rank computes its local value-head shard over the full sequence by directly
  reading sequence shards from peer symmetric buffers on device.
- The recurrent state stays in shared memory per (batch, local value head).
- Outputs are written to a symmetric full-sequence/head-shard buffer, then each rank
  gathers only its local sequence slice from peer output buffers with a CUDA kernel.
"""

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
#include <cmath>
#include <cstdint>

static inline int param_dtype_enum(torch::Tensor t) {
    if (t.dtype() == torch::kFloat32) return 0;
    if (t.dtype() == torch::kBFloat16) return 1;
    TORCH_CHECK(false, "a_log/dt_bias must be float32 or bfloat16");
}

__device__ __forceinline__ float bf16_load(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

__device__ __forceinline__ __nv_bfloat16 bf16_store(float x) {
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float read_param(const void* base, int dtype, int idx) {
    if (dtype == 0) {
        return reinterpret_cast<const float*>(base)[idx];
    } else {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[idx]);
    }
}

__device__ __forceinline__ float softplus_f32(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ float block_reduce_sum(float x, float* red) {
    int tid = threadIdx.x;
    red[tid] = x;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
    return red[0];
}

__global__ void gated_delta_recurrent_shared_kernel(
    const long long* __restrict__ q_ptrs,
    const long long* __restrict__ k_ptrs,
    const long long* __restrict__ v_ptrs,
    const long long* __restrict__ gate_ptrs,
    const long long* __restrict__ beta_ptrs,
    const void* __restrict__ a_log,
    const void* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ out_sym,
    int a_dtype,
    int dt_dtype,
    int B,
    int T_local,
    int H,
    int K,
    int HV,
    int V,
    int world_size,
    int rank,
    int H_local,
    int HV_local,
    int repeat
) {
    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int hv_l = blockIdx.y;
    const int tid = threadIdx.x;
    const int T_total = T_local * world_size;

    float* state = smem;                         // K * V
    float* q_vec = state + (int64_t)K * V;       // K
    float* k_vec = q_vec + K;                    // K
    float* upd   = k_vec + K;                    // V
    float* red   = upd + V;                      // blockDim.x + scratch

    for (int64_t i = tid; i < (int64_t)K * V; i += blockDim.x) {
        state[i] = 0.0f;
    }
    __syncthreads();

    const int hq_l = hv_l / repeat;
    const int global_h = rank * H_local + hq_l;
    const int global_hv = rank * HV_local + hv_l;

    const float scale = rsqrtf((float)K);
    const float a_scale = expf(read_param(a_log, a_dtype, hv_l));
    const float dt_b = read_param(dt_bias, dt_dtype, hv_l);

    for (int t = 0; t < T_total; ++t) {
        const int src_rank = t / T_local;
        const int ts = t - src_rank * T_local;

        const __nv_bfloat16* q_base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)q_ptrs[src_rank]);
        const __nv_bfloat16* k_base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)k_ptrs[src_rank]);
        const __nv_bfloat16* v_base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)v_ptrs[src_rank]);
        const __nv_bfloat16* gate_base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)gate_ptrs[src_rank]);
        const __nv_bfloat16* beta_base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)beta_ptrs[src_rank]);

        const int64_t qk_base_off =
            (((int64_t)b * T_local + ts) * H + global_h) * K;

        float q_ss = 0.0f;
        float k_ss = 0.0f;
        for (int kk = tid; kk < K; kk += blockDim.x) {
            float qx = bf16_load(q_base + qk_base_off + kk);
            float kx = bf16_load(k_base + qk_base_off + kk);
            q_ss += qx * qx;
            k_ss += kx * kx;
        }

        float q_sum = block_reduce_sum(q_ss, red);
        float k_sum = block_reduce_sum(k_ss, red);

        if (tid == 0) {
            float q_norm = sqrtf(q_sum);
            float k_norm = sqrtf(k_sum);
            red[0] = (q_norm > 1.0e-6f ? 1.0f / q_norm : 1.0e6f) * scale;
            red[1] = (k_norm > 1.0e-6f ? 1.0f / k_norm : 1.0e6f);
        }
        __syncthreads();

        const float q_inv = red[0];
        const float k_inv = red[1];

        for (int kk = tid; kk < K; kk += blockDim.x) {
            q_vec[kk] = bf16_load(q_base + qk_base_off + kk) * q_inv;
            k_vec[kk] = bf16_load(k_base + qk_base_off + kk) * k_inv;
        }

        if (tid == 0) {
            const int64_t gb_off = ((int64_t)b * T_local + ts) * HV + global_hv;
            float gate_x = bf16_load(gate_base + gb_off);
            float beta_x = bf16_load(beta_base + gb_off);
            float decay_log = -a_scale * softplus_f32(gate_x + dt_b);
            red[2] = expf(decay_log);
            red[3] = beta_x;
        }
        __syncthreads();

        const float decay = red[2];
        const float beta_x = red[3];

        for (int64_t i = tid; i < (int64_t)K * V; i += blockDim.x) {
            state[i] *= decay;
        }
        __syncthreads();

        const int64_t v_base_off =
            (((int64_t)b * T_local + ts) * HV + global_hv) * V;

        for (int vv = tid; vv < V; vv += blockDim.x) {
            float projected = 0.0f;
            #pragma unroll 1
            for (int kk = 0; kk < K; ++kk) {
                projected += k_vec[kk] * state[(int64_t)kk * V + vv];
            }
            float v_t = bf16_load(v_base + v_base_off + vv);
            upd[vv] = (v_t - projected) * beta_x;
        }
        __syncthreads();

        for (int64_t i = tid; i < (int64_t)K * V; i += blockDim.x) {
            int kk = (int)(i / V);
            int vv = (int)(i - (int64_t)kk * V);
            state[i] += k_vec[kk] * upd[vv];
        }
        __syncthreads();

        const int64_t out_base_off =
            (((int64_t)b * T_total + t) * HV_local + hv_l) * V;

        for (int vv = tid; vv < V; vv += blockDim.x) {
            float y = 0.0f;
            #pragma unroll 1
            for (int kk = 0; kk < K; ++kk) {
                y += q_vec[kk] * state[(int64_t)kk * V + vv];
            }
            out_sym[out_base_off + vv] = bf16_store(y);
        }
        __syncthreads();
    }
}

__global__ void gather_sequence_slice_kernel(
    const long long* __restrict__ out_ptrs,
    __nv_bfloat16* __restrict__ final_out,
    int64_t n,
    int B,
    int T_local,
    int HV,
    int V,
    int world_size,
    int rank,
    int HV_local
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int T_total = T_local * world_size;

    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        int vv = (int)(idx % V);
        int64_t tmp = idx / V;
        int hv = (int)(tmp % HV);
        tmp /= HV;
        int ts = (int)(tmp % T_local);
        int b = (int)(tmp / T_local);

        int owner = hv / HV_local;
        int hv_l = hv - owner * HV_local;
        int t_global = rank * T_local + ts;

        const __nv_bfloat16* src =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)out_ptrs[owner]);

        int64_t src_off =
            (((int64_t)b * T_total + t_global) * HV_local + hv_l) * V + vv;

        final_out[idx] = src[src_off];
    }
}

void launch_gated_delta_recurrent(
    torch::Tensor q_ptrs,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    torch::Tensor gate_ptrs,
    torch::Tensor beta_ptrs,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    torch::Tensor out_sym,
    int B,
    int T_local,
    int H,
    int K,
    int HV,
    int V,
    int world_size,
    int rank,
    int threads
) {
    TORCH_CHECK(q_ptrs.is_cuda() && k_ptrs.is_cuda() && v_ptrs.is_cuda(), "ptr tensors must be CUDA");
    TORCH_CHECK(gate_ptrs.is_cuda() && beta_ptrs.is_cuda(), "ptr tensors must be CUDA");
    TORCH_CHECK(out_sym.is_cuda(), "out_sym must be CUDA");
    TORCH_CHECK(out_sym.dtype() == torch::kBFloat16, "out_sym must be bfloat16");
    TORCH_CHECK(H % world_size == 0, "H must divide world_size");
    TORCH_CHECK(HV % world_size == 0, "HV must divide world_size");

    int H_local = H / world_size;
    int HV_local = HV / world_size;
    TORCH_CHECK(HV % H == 0, "HV must be divisible by H");
    int repeat = HV / H;
    TORCH_CHECK(HV_local % H_local == 0, "local HV/H mismatch");

    int a_dtype = param_dtype_enum(a_log);
    int dt_dtype = param_dtype_enum(dt_bias);

    dim3 grid(B, HV_local, 1);
    int64_t shared_floats = (int64_t)K * V + 2LL * K + V + threads + 8;
    int64_t shared_bytes = shared_floats * (int64_t)sizeof(float);
    TORCH_CHECK(shared_bytes <= 98304, "K*V too large for shared-memory recurrent kernel");

    cudaFuncSetAttribute(
        gated_delta_recurrent_shared_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        98304
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gated_delta_recurrent_shared_kernel<<<grid, threads, (size_t)shared_bytes, stream>>>(
        reinterpret_cast<const long long*>(q_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(k_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(v_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(gate_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(beta_ptrs.data_ptr<int64_t>()),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        reinterpret_cast<__nv_bfloat16*>(out_sym.data_ptr<at::BFloat16>()),
        a_dtype,
        dt_dtype,
        B,
        T_local,
        H,
        K,
        HV,
        V,
        world_size,
        rank,
        H_local,
        HV_local,
        repeat
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_sequence_slice(
    torch::Tensor out_ptrs,
    torch::Tensor final_out,
    int B,
    int T_local,
    int HV,
    int V,
    int world_size,
    int rank
) {
    TORCH_CHECK(out_ptrs.is_cuda(), "out_ptrs must be CUDA");
    TORCH_CHECK(final_out.is_cuda(), "final_out must be CUDA");
    TORCH_CHECK(final_out.dtype() == torch::kBFloat16, "final_out must be bfloat16");
    TORCH_CHECK(HV % world_size == 0, "HV must divide world_size");

    int HV_local = HV / world_size;
    int64_t n = (int64_t)B * T_local * HV * V;

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gather_sequence_slice_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const long long*>(out_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<__nv_bfloat16*>(final_out.data_ptr<at::BFloat16>()),
        n,
        B,
        T_local,
        HV,
        V,
        world_size,
        rank,
        HV_local
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gated_delta_recurrent", &launch_gated_delta_recurrent,
          "Gated DeltaNet recurrent kernel using symmetric-memory UVA peer loads");
    m.def("launch_gather_sequence_slice", &launch_gather_sequence_slice,
          "Gather local sequence slice from symmetric output shards");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gated_deltanet_cp_symm_bf16_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _ptr_tensor(hdl, device):
    return torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)


def _symm_empty(shape, device, dtype):
    return symm_mem.empty(tuple(shape), device=device, dtype=dtype)


def _get_resources(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    group: dist.ProcessGroup,
):
    world_size = dist.get_world_size(group=group)
    B, T_local, H, K = q.shape
    _, _, HV, V = v.shape
    HV_local = HV // world_size
    T_total = T_local * world_size

    key = (
        id(group),
        q.device.index,
        q.dtype,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(gate.shape),
        tuple(beta.shape),
    )
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    q_buf = _symm_empty(q.shape, q.device, q.dtype)
    k_buf = _symm_empty(k.shape, k.device, k.dtype)
    v_buf = _symm_empty(v.shape, v.device, v.dtype)
    gate_buf = _symm_empty(gate.shape, gate.device, gate.dtype)
    beta_buf = _symm_empty(beta.shape, beta.device, beta.dtype)

    q_hdl = symm_mem.rendezvous(q_buf, group)
    k_hdl = symm_mem.rendezvous(k_buf, group)
    v_hdl = symm_mem.rendezvous(v_buf, group)
    gate_hdl = symm_mem.rendezvous(gate_buf, group)
    beta_hdl = symm_mem.rendezvous(beta_buf, group)

    out_sym = _symm_empty((B, T_total, HV_local, V), q.device, q.dtype)
    out_hdl = symm_mem.rendezvous(out_sym, group)

    q_ptrs = _ptr_tensor(q_hdl, q.device)
    k_ptrs = _ptr_tensor(k_hdl, q.device)
    v_ptrs = _ptr_tensor(v_hdl, q.device)
    gate_ptrs = _ptr_tensor(gate_hdl, q.device)
    beta_ptrs = _ptr_tensor(beta_hdl, q.device)
    out_ptrs = _ptr_tensor(out_hdl, q.device)

    res = {
        "q_buf": q_buf,
        "k_buf": k_buf,
        "v_buf": v_buf,
        "gate_buf": gate_buf,
        "beta_buf": beta_buf,
        "out_sym": out_sym,
        "q_hdl": q_hdl,
        "out_hdl": out_hdl,
        "q_ptrs": q_ptrs,
        "k_ptrs": k_ptrs,
        "v_ptrs": v_ptrs,
        "gate_ptrs": gate_ptrs,
        "beta_ptrs": beta_ptrs,
        "out_ptrs": out_ptrs,
    }
    _resource_cache[key] = res
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
    """
    Per-rank Gated DeltaNet CP forward.

    BF16 fast path:
      - publish local sequence shards into symmetric buffers,
      - compute this rank's value-head shard over the full sequence using peer UVA loads,
      - publish full-sequence local-head outputs,
      - gather this rank's local sequence slice from all peer output shards.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    group = group or dist.group.WORLD

    assert q.is_cuda and k.is_cuda and v.is_cuda and gate.is_cuda and beta.is_cuda
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert gate.dtype == torch.bfloat16
    assert beta.dtype == torch.bfloat16

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    a_log = a_log.contiguous()
    dt_bias = dt_bias.contiguous()

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    B, T_local, H, Kdim = q.shape
    Bv, Tv, HV, Vdim = v.shape

    assert Bv == B and Tv == T_local
    assert gate.shape == (B, T_local, HV)
    assert beta.shape == (B, T_local, HV)
    assert H % world_size == 0
    assert HV % world_size == 0
    assert HV % H == 0
    assert a_log.numel() == HV // world_size
    assert dt_bias.numel() == HV // world_size

    ext = _get_ext()
    res = _get_resources(q, k, v, gate, beta, group)

    res["q_buf"].copy_(q)
    res["k_buf"].copy_(k)
    res["v_buf"].copy_(v)
    res["gate_buf"].copy_(gate)
    res["beta_buf"].copy_(beta)

    # Symmetric-memory stream-aware device synchronization for published inputs.
    res["q_hdl"].barrier(channel=0)

    threads = 256
    ext.launch_gated_delta_recurrent(
        res["q_ptrs"],
        res["k_ptrs"],
        res["v_ptrs"],
        res["gate_ptrs"],
        res["beta_ptrs"],
        a_log,
        dt_bias,
        res["out_sym"],
        B,
        T_local,
        H,
        Kdim,
        HV,
        Vdim,
        world_size,
        rank,
        threads,
    )

    # Make each rank's computed head-shard output visible before peer gather.
    res["out_hdl"].barrier(channel=1)

    out = torch.empty((B, T_local, HV, Vdim), device=q.device, dtype=q.dtype)
    ext.launch_gather_sequence_slice(
        res["out_ptrs"],
        out,
        B,
        T_local,
        HV,
        Vdim,
        world_size,
        rank,
    )
    return out