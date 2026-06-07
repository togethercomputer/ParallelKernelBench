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
#include <cmath>

#define DTYPE_BF16 0
#define DTYPE_F32  1

__device__ __forceinline__ float load_scalar_typed(const void* base, int64_t idx, int dtype) {
    if (dtype == DTYPE_BF16) {
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(base);
        return __bfloat162float(p[idx]);
    } else {
        const float* p = reinterpret_cast<const float*>(base);
        return p[idx];
    }
}

__device__ __forceinline__ void store_scalar_typed(void* base, int64_t idx, float x, int dtype) {
    if (dtype == DTYPE_BF16) {
        __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(base);
        p[idx] = __float2bfloat16_rn(x);
    } else {
        float* p = reinterpret_cast<float*>(base);
        p[idx] = x;
    }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        warp_sums[wid] = v;
    }
    __syncthreads();

    float out = 0.0f;
    if (wid == 0) {
        out = (lane < nwarps) ? warp_sums[lane] : 0.0f;
        out = warp_reduce_sum(out);
    }
    return out;
}

__global__ void kda_cp_uva_kernel(
    const uint64_t* __restrict__ q_ptrs,
    const uint64_t* __restrict__ k_ptrs,
    const uint64_t* __restrict__ v_ptrs,
    const uint64_t* __restrict__ g_ptrs,
    const uint64_t* __restrict__ beta_ptrs,
    const void* __restrict__ a_log,
    const void* __restrict__ dt_bias,
    void* __restrict__ out,
    int B,
    int T_local,
    int H,
    int K,
    int V,
    int cp_world,
    int cp_rank,
    int q_dtype,
    int k_dtype,
    int v_dtype,
    int g_dtype,
    int beta_dtype,
    int a_dtype,
    int dt_dtype,
    int out_dtype
) {
    extern __shared__ float smem[];

    float* state = smem;                              // K * V
    float* qbuf  = state + (int64_t)K * V;             // K
    float* kbuf  = qbuf + K;                          // K
    float* decay = kbuf + K;                          // K
    float* proj  = decay + K;                         // V
    float* upd   = proj + V;                          // V

    __shared__ float s_qinv;
    __shared__ float s_kinv;
    __shared__ float s_beta;
    __shared__ float s_a_scale;

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh - b * H;

    int tid = threadIdx.x;
    int64_t state_elems = (int64_t)K * V;

    for (int64_t i = tid; i < state_elems; i += blockDim.x) {
        state[i] = 0.0f;
    }

    if (tid == 0) {
        float al = load_scalar_typed(a_log, h, a_dtype);
        s_a_scale = expf(al);
    }
    __syncthreads();

    const int T_full = T_local * cp_world;
    const int local_start = cp_rank * T_local;
    const float q_scale = rsqrtf((float)K);

    for (int t = 0; t < T_full; ++t) {
        int owner = t / T_local;
        int lt = t - owner * T_local;

        const void* q_base = reinterpret_cast<const void*>(q_ptrs[owner]);
        const void* k_base = reinterpret_cast<const void*>(k_ptrs[owner]);
        const void* v_base = reinterpret_cast<const void*>(v_ptrs[owner]);
        const void* g_base = reinterpret_cast<const void*>(g_ptrs[owner]);
        const void* beta_base = reinterpret_cast<const void*>(beta_ptrs[owner]);

        float qsum = 0.0f;
        float ksum = 0.0f;

        for (int d = tid; d < K; d += blockDim.x) {
            int64_t qk_idx = (((int64_t)b * T_local + lt) * H + h) * K + d;
            float qv = load_scalar_typed(q_base, qk_idx, q_dtype);
            float kv = load_scalar_typed(k_base, qk_idx, k_dtype);
            qbuf[d] = qv;
            kbuf[d] = kv;
            qsum += qv * qv;
            ksum += kv * kv;

            float gv = load_scalar_typed(g_base, qk_idx, g_dtype);
            float dt = load_scalar_typed(dt_bias, (int64_t)h * K + d, dt_dtype);
            float x = s_a_scale * (gv + dt);
            float sig = 1.0f / (1.0f + expf(-x));
            decay[d] = expf(-5.0f * sig);
        }

        float qred = block_reduce_sum(qsum);
        if (tid == 0) {
            float n = sqrtf(qred);
            n = fmaxf(n, 1.0e-12f);
            s_qinv = q_scale / n;
        }
        __syncthreads();

        float kred = block_reduce_sum(ksum);
        if (tid == 0) {
            float n = sqrtf(kred);
            n = fmaxf(n, 1.0e-12f);
            s_kinv = 1.0f / n;

            int64_t beta_idx = ((int64_t)b * T_local + lt) * H + h;
            float bv = load_scalar_typed(beta_base, beta_idx, beta_dtype);
            s_beta = 1.0f / (1.0f + expf(-bv));
        }
        __syncthreads();

        for (int d = tid; d < K; d += blockDim.x) {
            qbuf[d] *= s_qinv;
            kbuf[d] *= s_kinv;
        }
        __syncthreads();

        for (int64_t i = tid; i < state_elems; i += blockDim.x) {
            int d = (int)(i / V);
            state[i] *= decay[d];
        }
        __syncthreads();

        for (int vv = tid; vv < V; vv += blockDim.x) {
            float s = 0.0f;
            #pragma unroll 1
            for (int d = 0; d < K; ++d) {
                s += kbuf[d] * state[(int64_t)d * V + vv];
            }
            proj[vv] = s;
        }
        __syncthreads();

        for (int vv = tid; vv < V; vv += blockDim.x) {
            int64_t vidx = (((int64_t)b * T_local + lt) * H + h) * V + vv;
            float vv_in = load_scalar_typed(v_base, vidx, v_dtype);
            upd[vv] = (vv_in - proj[vv]) * s_beta;
        }
        __syncthreads();

        for (int64_t i = tid; i < state_elems; i += blockDim.x) {
            int d = (int)(i / V);
            int vv = (int)(i - (int64_t)d * V);
            state[i] += kbuf[d] * upd[vv];
        }
        __syncthreads();

        if (t >= local_start && t < local_start + T_local) {
            int out_t = t - local_start;
            for (int vv = tid; vv < V; vv += blockDim.x) {
                float s = 0.0f;
                #pragma unroll 1
                for (int d = 0; d < K; ++d) {
                    s += qbuf[d] * state[(int64_t)d * V + vv];
                }
                int64_t oidx = (((int64_t)b * T_local + out_t) * H + h) * V + vv;
                store_scalar_typed(out, oidx, s, out_dtype);
            }
        }
        __syncthreads();
    }
}

__global__ void tp_sum_uva_kernel(
    const uint64_t* __restrict__ ptrs,
    void* __restrict__ out,
    int64_t n,
    int tp_world,
    int dtype
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < n; idx += stride) {
        float s = 0.0f;
        #pragma unroll 1
        for (int r = 0; r < tp_world; ++r) {
            const void* base = reinterpret_cast<const void*>(ptrs[r]);
            s += load_scalar_typed(base, idx, dtype);
        }
        store_scalar_typed(out, idx, s, dtype);
    }
}

void launch_kda_cp_uva(
    torch::Tensor q_ptrs,
    torch::Tensor k_ptrs,
    torch::Tensor v_ptrs,
    torch::Tensor g_ptrs,
    torch::Tensor beta_ptrs,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    torch::Tensor out,
    int B,
    int T_local,
    int H,
    int K,
    int V,
    int cp_world,
    int cp_rank,
    int q_dtype,
    int k_dtype,
    int v_dtype,
    int g_dtype,
    int beta_dtype,
    int a_dtype,
    int dt_dtype,
    int out_dtype
) {
    TORCH_CHECK(q_ptrs.is_cuda(), "q_ptrs must be CUDA");
    TORCH_CHECK(k_ptrs.is_cuda(), "k_ptrs must be CUDA");
    TORCH_CHECK(v_ptrs.is_cuda(), "v_ptrs must be CUDA");
    TORCH_CHECK(g_ptrs.is_cuda(), "g_ptrs must be CUDA");
    TORCH_CHECK(beta_ptrs.is_cuda(), "beta_ptrs must be CUDA");
    TORCH_CHECK(a_log.is_cuda(), "a_log must be CUDA");
    TORCH_CHECK(dt_bias.is_cuda(), "dt_bias must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");

    const uint64_t* qp = reinterpret_cast<const uint64_t*>(q_ptrs.data_ptr<int64_t>());
    const uint64_t* kp = reinterpret_cast<const uint64_t*>(k_ptrs.data_ptr<int64_t>());
    const uint64_t* vp = reinterpret_cast<const uint64_t*>(v_ptrs.data_ptr<int64_t>());
    const uint64_t* gp = reinterpret_cast<const uint64_t*>(g_ptrs.data_ptr<int64_t>());
    const uint64_t* bp = reinterpret_cast<const uint64_t*>(beta_ptrs.data_ptr<int64_t>());

    int threads = 256;
    if (K * V <= 1024 && V <= 64) threads = 128;
    if (K * V <= 256 && V <= 32) threads = 64;

    int64_t shmem_elems = (int64_t)K * V + 3LL * K + 2LL * V;
    size_t shmem_bytes = (size_t)shmem_elems * sizeof(float);

    cudaFuncSetAttribute(
        kda_cp_uva_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_bytes
    );

    dim3 grid((unsigned)(B * H));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    kda_cp_uva_kernel<<<grid, threads, shmem_bytes, stream>>>(
        qp, kp, vp, gp, bp,
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        out.data_ptr(),
        B, T_local, H, K, V,
        cp_world, cp_rank,
        q_dtype, k_dtype, v_dtype, g_dtype, beta_dtype,
        a_dtype, dt_dtype, out_dtype
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_tp_sum_uva(
    torch::Tensor ptrs,
    torch::Tensor out,
    int64_t n,
    int dtype
) {
    TORCH_CHECK(ptrs.is_cuda(), "ptrs must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    int tp_world = (int)ptrs.size(0);

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* p = reinterpret_cast<const uint64_t*>(ptrs.data_ptr<int64_t>());

    tp_sum_uva_kernel<<<blocks, threads, 0, stream>>>(
        p, out.data_ptr(), n, tp_world, dtype
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kda_cp_uva", &launch_kda_cp_uva,
          "Kimi Delta Attention CP direct-UVA recurrent forward");
    m.def("launch_tp_sum_uva", &launch_tp_sum_uva,
          "TP symmetric-memory peer-pointer sum");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("kimi_delta_attention_cp_tp_uva_ext", CUDA_SRC)
    return _ext


_DTYPE_BF16 = 0
_DTYPE_F32 = 1


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return _DTYPE_BF16
    if dtype == torch.float32:
        return _DTYPE_F32
    raise TypeError(f"unsupported dtype {dtype}; expected bfloat16 or float32")


_cp_cache = {}
_tp_cache = {}


def _group_key(group: dist.ProcessGroup) -> int:
    return id(group)


def _ptrs_tensor(hdl, device: torch.device) -> torch.Tensor:
    return torch.tensor([int(p) for p in hdl.buffer_ptrs], device=device, dtype=torch.int64)


def _get_cp_resources(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    group: dist.ProcessGroup,
):
    key = (
        _group_key(group),
        q.device,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(g.shape),
        tuple(beta.shape),
        q.dtype,
        k.dtype,
        v.dtype,
        g.dtype,
        beta.dtype,
    )
    res = _cp_cache.get(key)
    if res is not None:
        return res

    q_buf = symm_mem.empty(tuple(q.shape), device=q.device, dtype=q.dtype)
    k_buf = symm_mem.empty(tuple(k.shape), device=k.device, dtype=k.dtype)
    v_buf = symm_mem.empty(tuple(v.shape), device=v.device, dtype=v.dtype)
    g_buf = symm_mem.empty(tuple(g.shape), device=g.device, dtype=g.dtype)
    beta_buf = symm_mem.empty(tuple(beta.shape), device=beta.device, dtype=beta.dtype)

    q_hdl = symm_mem.rendezvous(q_buf, group)
    k_hdl = symm_mem.rendezvous(k_buf, group)
    v_hdl = symm_mem.rendezvous(v_buf, group)
    g_hdl = symm_mem.rendezvous(g_buf, group)
    beta_hdl = symm_mem.rendezvous(beta_buf, group)

    q_ptrs = _ptrs_tensor(q_hdl, q.device)
    k_ptrs = _ptrs_tensor(k_hdl, q.device)
    v_ptrs = _ptrs_tensor(v_hdl, q.device)
    g_ptrs = _ptrs_tensor(g_hdl, q.device)
    beta_ptrs = _ptrs_tensor(beta_hdl, q.device)

    res = {
        "q_buf": q_buf,
        "k_buf": k_buf,
        "v_buf": v_buf,
        "g_buf": g_buf,
        "beta_buf": beta_buf,
        "q_hdl": q_hdl,
        "k_hdl": k_hdl,
        "v_hdl": v_hdl,
        "g_hdl": g_hdl,
        "beta_hdl": beta_hdl,
        "q_ptrs": q_ptrs,
        "k_ptrs": k_ptrs,
        "v_ptrs": v_ptrs,
        "g_ptrs": g_ptrs,
        "beta_ptrs": beta_ptrs,
    }
    _cp_cache[key] = res
    return res


def _get_tp_resources(shape, dtype: torch.dtype, device: torch.device, group: dist.ProcessGroup):
    key = (_group_key(group), device, tuple(shape), dtype)
    res = _tp_cache.get(key)
    if res is not None:
        return res

    buf = symm_mem.empty(tuple(shape), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = _ptrs_tensor(hdl, device)
    out = torch.empty(tuple(shape), device=device, dtype=dtype)

    res = {
        "buf": buf,
        "hdl": hdl,
        "ptrs": ptrs,
        "out": out,
    }
    _tp_cache[key] = res
    return res


def _ensure_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


@torch.no_grad()
def solution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Per-rank Kimi Delta Attention CP/TP forward using symmetric-memory UVA
    instead of NCCL all-gather/all-reduce.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert q.is_cuda and k.is_cuda and v.is_cuda and g.is_cuda and beta.is_cuda
    assert a_log.is_cuda and dt_bias.is_cuda

    _get_ext()

    cp_group = cp_group or dist.group.WORLD
    cp_world = dist.get_world_size(group=cp_group)
    cp_rank = dist.get_rank(group=cp_group)

    q = _ensure_contiguous(q)
    k = _ensure_contiguous(k)
    v = _ensure_contiguous(v)
    g = _ensure_contiguous(g)
    beta = _ensure_contiguous(beta)
    a_log = _ensure_contiguous(a_log)
    dt_bias = _ensure_contiguous(dt_bias)

    B, T_local, H, Kdim = q.shape
    Vdim = v.shape[-1]

    assert k.shape == q.shape
    assert g.shape == q.shape
    assert beta.shape == (B, T_local, H)
    assert v.shape[:3] == (B, T_local, H)
    assert a_log.numel() == H
    assert dt_bias.numel() == H * Kdim

    q_dtype = _dtype_enum(q.dtype)
    k_dtype = _dtype_enum(k.dtype)
    v_dtype = _dtype_enum(v.dtype)
    g_dtype = _dtype_enum(g.dtype)
    beta_dtype = _dtype_enum(beta.dtype)
    a_dtype = _dtype_enum(a_log.dtype)
    dt_dtype = _dtype_enum(dt_bias.dtype)
    out_dtype = _dtype_enum(q.dtype)

    cp_res = _get_cp_resources(q, k, v, g, beta, cp_group)

    cp_res["q_buf"].copy_(q)
    cp_res["k_buf"].copy_(k)
    cp_res["v_buf"].copy_(v)
    cp_res["g_buf"].copy_(g)
    cp_res["beta_buf"].copy_(beta)

    # One symmetric-memory barrier after all local shard publications.  The
    # barrier orders the current stream before peer UVA loads in the recurrent
    # kernel, avoiding host-side NCCL gather.
    cp_res["q_hdl"].barrier(channel=0)

    tp_world = 1
    if tp_group is not None:
        tp_world = dist.get_world_size(group=tp_group)

    out_shape = (B, T_local, H, Vdim)

    if tp_group is not None and tp_world > 1:
        tp_res = _get_tp_resources(out_shape, q.dtype, q.device, tp_group)
        kda_out = tp_res["buf"]
    else:
        tp_res = None
        kda_out = torch.empty(out_shape, device=q.device, dtype=q.dtype)

    _get_ext().launch_kda_cp_uva(
        cp_res["q_ptrs"],
        cp_res["k_ptrs"],
        cp_res["v_ptrs"],
        cp_res["g_ptrs"],
        cp_res["beta_ptrs"],
        a_log,
        dt_bias,
        kda_out,
        int(B),
        int(T_local),
        int(H),
        int(Kdim),
        int(Vdim),
        int(cp_world),
        int(cp_rank),
        int(q_dtype),
        int(k_dtype),
        int(v_dtype),
        int(g_dtype),
        int(beta_dtype),
        int(a_dtype),
        int(dt_dtype),
        int(out_dtype),
    )

    if tp_res is not None:
        # Publish local KDA slice to TP peers, then sum peer symmetric buffers
        # directly on device.
        tp_res["hdl"].barrier(channel=0)
        _get_ext().launch_tp_sum_uva(
            tp_res["ptrs"],
            tp_res["out"],
            int(kda_out.numel()),
            int(out_dtype),
        )
        return tp_res["out"]

    return kda_out