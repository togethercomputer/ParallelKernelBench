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
#include <algorithm>

// ---------------------------------------------------------------------------
// Block-level utilities
// ---------------------------------------------------------------------------

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();
    
    float sum = (threadIdx.x < (blockDim.x + 31) / 32) ? shared_mem[lane] : 0.0f;
    sum = warp_reduce_sum(sum);
    
    if (threadIdx.x == 0) shared_mem[0] = sum;
    __syncthreads();
    return shared_mem[0];
}

// ---------------------------------------------------------------------------
// KDA CP Forward Kernel
// ---------------------------------------------------------------------------

__global__ void kda_forward_kernel(
    const int64_t* __restrict__ cp_ptrs, // Pointers to cp_buf of each CP rank
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ a_log_ptr,
    const __nv_bfloat16* __restrict__ dt_bias_ptr,
    __nv_bfloat16* __restrict__ out_ptr,
    int B, int T_local, int H, int K, int V, int cp_rank
) {
    extern __shared__ float smem[];
    float* S = smem; // [K * V]
    float* sh_k_float = smem + K * V; // [K]
    float* sh_q_float = sh_k_float + K; // [K]
    float* sh_decay = sh_q_float + K; // [K]
    float* sh_reduce = sh_decay + K; // [32]

    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    int tx = threadIdx.x;

    if (tx < V) {
        for (int i = 0; i < K; ++i) {
            S[i * V + tx] = 0.0f;
        }
    }
    __syncthreads();

    float a_scale_val = expf(__bfloat162float(a_log_ptr[h]));
    float dt_b = (tx < K) ? __bfloat162float(dt_bias_ptr[h * K + tx]) : 0.0f;
    int stride_last = 2 * K + V + 1;

    for (int r = 0; r <= cp_rank; ++r) {
        const __nv_bfloat16* peer_buf = (const __nv_bfloat16*)cp_ptrs[r];
        
        for (int t = 0; t < T_local; ++t) {
            int64_t offset = ((int64_t)(b * T_local + t) * H + h) * stride_last;
            const __nv_bfloat16* step_ptr = peer_buf + offset;
            
            float k_val = (tx < K) ? __bfloat162float(step_ptr[tx]) : 0.0f;
            float v_val = (tx < V) ? __bfloat162float(step_ptr[K + tx]) : 0.0f;
            float g_val = (tx < K) ? __bfloat162float(step_ptr[K + V + tx]) : 0.0f;
            float beta_val = __bfloat162float(step_ptr[K + V + K]); 
            
            float k_sq = (tx < K) ? k_val * k_val : 0.0f;
            float norm_sq_k = block_reduce_sum(k_sq, sh_reduce);
            float norm_k = sqrtf(norm_sq_k);
            if (norm_k < 1e-12f) norm_k = 1e-12f;
            if (tx < K) sh_k_float[tx] = k_val / norm_k;

            if (r == cp_rank) {
                int64_t q_offset = ((int64_t)(b * T_local + t) * H + h) * K;
                float q_val = (tx < K) ? __bfloat162float(q_ptr[q_offset + tx]) : 0.0f;
                float q_sq = (tx < K) ? q_val * q_val : 0.0f;
                float norm_sq_q = block_reduce_sum(q_sq, sh_reduce);
                float norm_q = sqrtf(norm_sq_q);
                if (norm_q < 1e-12f) norm_q = 1e-12f;
                if (tx < K) sh_q_float[tx] = (q_val / norm_q) * (1.0f / sqrtf((float)K));
            }

            if (tx < K) {
                float exponent = a_scale_val * (g_val + dt_b);
                float sig = 1.0f / (1.0f + expf(-exponent));
                sh_decay[tx] = expf(-5.0f * sig);
            }
            float beta_sig = 1.0f / (1.0f + expf(-beta_val));

            __syncthreads();

            if (tx < V) {
                float proj = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < K; ++i) {
                    proj += sh_k_float[i] * S[i * V + tx];
                }
                float update = (v_val - proj) * beta_sig;

                float out_val = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < K; ++i) {
                    float new_s = sh_decay[i] * S[i * V + tx] + sh_k_float[i] * update;
                    S[i * V + tx] = new_s;
                    if (r == cp_rank) {
                        out_val += sh_q_float[i] * new_s;
                    }
                }

                if (r == cp_rank) {
                    int64_t out_offset = ((int64_t)(b * T_local + t) * H + h) * V;
                    out_ptr[out_offset + tx] = __float2bfloat16(out_val);
                }
            }
            __syncthreads();
        }
    }
}

void launch_kda_forward(
    torch::Tensor cp_ptrs_tensor,
    torch::Tensor q, torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor out, int cp_rank
) {
    int B = q.size(0);
    int T_local = q.size(1);
    int H = q.size(2);
    int K = q.size(3);
    int V = out.size(3);

    int threads = std::max(K, V);
    threads = ((threads + 31) / 32) * 32;

    int blocks = B * H;
    int smem_size = (K * V + 3 * K + 32) * sizeof(float);

    cudaFuncSetAttribute(kda_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227000);
    const int64_t* cp_ptrs = cp_ptrs_tensor.data_ptr<int64_t>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    kda_forward_kernel<<<blocks, threads, smem_size, stream>>>(
        cp_ptrs,
        (__nv_bfloat16*)q.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)a_log.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)dt_bias.data_ptr<at::BFloat16>(),
        (__nv_bfloat16*)out.data_ptr<at::BFloat16>(),
        B, T_local, H, K, V, cp_rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Multimem TP all-reduce Kernel
// ---------------------------------------------------------------------------

__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp; do { asm volatile("atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;" : "=r"(tmp) : "l"(addr) : "memory"); } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp; do { asm volatile("atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;" : "=r"(tmp) : "l"(addr) : "memory"); } while (tmp != 1u);
}
__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp; do { asm volatile("atom.global.release.sys.cas.b32 %0, [%1], 0, 1;" : "=r"(tmp) : "l"(addr) : "memory"); } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp; do { asm volatile("atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;" : "=r"(tmp) : "l"(addr) : "memory"); } while (tmp != 1u);
}
__device__ void blockwise_barrier_relaxed(const uint64_t* __restrict__ signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(signal_pad_ptrs[flat_tid] + block_id * world_size + rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank] + block_id * world_size + flat_tid);
    send_signal_relaxed(send_addr); wait_signal_relaxed(wait_addr);
}
__device__ void blockwise_barrier_acq_rel(const uint64_t* __restrict__ signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(signal_pad_ptrs[flat_tid] + block_id * world_size + rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank] + block_id * world_size + flat_tid);
    send_signal_acq_rel(send_addr); wait_signal_acq_rel(wait_addr);
}

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(const uint64_t* addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(addr) : "memory");
}
__device__ __forceinline__ void multimem_st_bf16x4(const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base, const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128, int world_size, int rank, int block_stride
) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank = (numel_128 + world_size - 1) / world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = block_id * block_stride; block_start < numel_per_rank; block_start += num_programs * block_stride) {
        const int64_t offsets = block_start + tid;
        if (offsets >= numel_per_rank) continue;
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + (rank * numel_per_rank + offsets) * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr, torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel, int world_size, int rank, int num_blocks, int block_size, int block_stride
) {
    const uint64_t* d_signal = reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_signal, numel, world_size, rank, block_stride);
}

__global__ void allreduce_bf16_kernel(
    const long long* __restrict__ ptrs, __nv_bfloat16* __restrict__ out, int world_size, int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            sum += __bfloat162float(((const __nv_bfloat16*)ptrs[r])[idx]);
        }
        out[idx] = __float2bfloat16(sum);
    }
}

void launch_allreduce(torch::Tensor ptrs_tensor, torch::Tensor out, int64_t n) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = std::min((int)((n + threads - 1) / threads), 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kda_forward", &launch_kda_forward, "KDA Forward Context Kern");
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16, "TP multimem all-reduce");
    m.def("launch_allreduce", &launch_allreduce, "TP peer-pointer all-reduce");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("kda_cp_tp_ext", CUDA_SRC)
    return _ext

_cp_cache = {}
def _get_cp_resources(shape, dtype, device, group):
    key = (shape, dtype, device, group)
    if key in _cp_cache:
        return _cp_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype, group=group)
    hdl = symm_mem.rendezvous(buf, group=group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, ptrs_tensor)
    _cp_cache[key] = res
    return res

_tp_cache = {}
def _get_tp_resources(shape, dtype, device, group):
    key = (shape, dtype, device, group)
    if key in _tp_cache:
        return _tp_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype, group=group)
    hdl = symm_mem.rendezvous(buf, group=group)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, ptrs_tensor)
    _tp_cache[key] = res
    return res

def _multimem_launch_config(numel: int, world_size: int) -> tuple[int, int, int]:
    numel_per_thread = 8 
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < 1024:
        block_size = 1
        while block_size < num_threads: block_size *= 2
        num_blocks = 1
    else:
        block_size = 1024
        num_blocks = min((num_threads + 1023) // 1024, 4)
    return num_blocks, block_size, block_size

@torch.no_grad()
def solution(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
    beta: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    q, k, v, g = q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous()
    beta, a_log, dt_bias = beta.contiguous(), a_log.contiguous(), dt_bias.contiguous()
    
    assert q.dtype == torch.bfloat16, "Hardware bindings exclusively optimized for bfloat16"
    B, T_local, H_local, K = q.shape
    V = v.shape[-1]
    
    cp_group = cp_group or dist.group.WORLD
    cp_size = dist.get_world_size(group=cp_group)
    cp_rank = dist.get_rank(group=cp_group)

    # 1. CP Gathering via Symmetric Memory
    stride_last = 2 * K + V + 1
    cp_buf, cp_hdl, cp_ptrs_tensor = _get_cp_resources(
        (B, T_local, H_local, stride_last), q.dtype, q.device, cp_group
    )
    
    cp_buf[..., :K].copy_(k)
    cp_buf[..., K:K+V].copy_(v)
    cp_buf[..., K+V:2*K+V].copy_(g)
    cp_buf[..., 2*K+V:].copy_(beta.unsqueeze(-1))
    
    if cp_size > 1:
        cp_hdl.barrier(channel=0)

    # 2. Extract specific TP Resources natively resolving buffer placements
    tp_active = tp_group is not None and dist.get_world_size(tp_group) > 1
    if tp_active:
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        out_buf, tp_hdl, tp_ptrs_tensor = _get_tp_resources(
            (B, T_local, H_local, V), q.dtype, q.device, tp_group
        )
    else:
        out_buf = torch.empty((B, T_local, H_local, V), dtype=q.dtype, device=q.device)

    # 3. Custom KDA sequential recurrent kernel executing up to exactly `cp_rank` 
    _get_ext().launch_kda_forward(
        cp_ptrs_tensor, q, a_log, dt_bias, out_buf, cp_rank
    )
    
    # 4. In-switch TP All-reduce
    if tp_active:
        tp_hdl.barrier(channel=0)
        n = out_buf.numel()
        
        # Condition check for Multimem 16-byte hardware constraints 
        if n % 8 == 0:
            num_blocks, block_size, block_stride = _multimem_launch_config(n, tp_size)
            dist.barrier(group=tp_group) # Explicit sync required before multicast manipulation
            
            _get_ext().launch_multimem_allreduce_bf16(
                int(tp_hdl.multicast_ptr), tp_hdl.signal_pad_ptrs_dev,
                n // 8, tp_size, tp_rank, num_blocks, block_size, block_stride
            )
            return out_buf.clone()
        else:
            final_out = torch.empty_like(out_buf)
            _get_ext().launch_allreduce(tp_ptrs_tensor, final_out, n)
            return final_out

    return out_buf