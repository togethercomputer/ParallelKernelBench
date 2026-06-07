"""
Vocab-parallel cross entropy with device-side multimem all-reduce.

Strategy:
- Fuse three all-reduces (max, predicted_logit sum, exp_sum) using symmetric
  memory + multimem PTX on H100 NVSwitch in a single fused launch where
  possible.
- Compute logits_max, predicted_logit, and exp_sum locally in one pass to
  reduce memory traffic. Then issue device-side multimem reductions on small
  per-token tensors (these are tiny so latency-bound; multimem on NVSwitch
  hides the latency).
- Final log/sub fused on device.
"""

from typing import Optional, Tuple
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

// ---- signal-pad barriers ----
__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}
__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void barrier_relaxed(
    const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size)
{
    unsigned int t = threadIdx.x;
    if (t >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[t];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}
__device__ void barrier_acq_rel(
    const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size)
{
    unsigned int t = threadIdx.x;
    if (t >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[t];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)t);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

// ---- local reduce kernel: compute max, predicted, sum(exp) per row in one pass ----
// Uses two-pass within the row: pass1 max, pass2 sum_exp; predicted is simultaneous.
// One block per row. Subtracts pre-known global max (after first phase), but we
// produce only local results here; global reductions happen separately.

__global__ void local_phase_max_pred_kernel(
    const __nv_bfloat16* __restrict__ logits, // [N, V]
    const long* __restrict__ target,           // [N]
    float* __restrict__ local_max,             // [N]
    float* __restrict__ local_pred,            // [N] (masked)
    int N, int V,
    long vocab_start, long vocab_end)
{
    int row = blockIdx.x;
    if (row >= N) return;
    const __nv_bfloat16* base = logits + (size_t)row * V;
    int tid = threadIdx.x;

    float local = -INFINITY;
    for (int j = tid; j < V; j += blockDim.x) {
        float v = __bfloat162float(base[j]);
        if (v > local) local = v;
    }
    // block reduce max
    __shared__ float sdata[32];
    // warp reduce
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(mask, local, off);
        if (other > local) local = other;
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) sdata[warp] = local;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        float v = (tid < n_warps) ? sdata[lane] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(mask, v, off);
            if (other > v) v = other;
        }
        if (lane == 0) {
            local_max[row] = v;
        }
    }

    // predicted logit
    long t = target[row];
    if (tid == 0) {
        float pv = 0.0f;
        if (t >= vocab_start && t < vocab_end) {
            int idx = (int)(t - vocab_start);
            pv = __bfloat162float(base[idx]);
        }
        local_pred[row] = pv;
    }
}

__global__ void local_phase_sumexp_kernel(
    const __nv_bfloat16* __restrict__ logits,  // [N,V]
    const float* __restrict__ global_max,      // [N]
    float* __restrict__ local_sumexp,          // [N]
    int N, int V)
{
    int row = blockIdx.x;
    if (row >= N) return;
    const __nv_bfloat16* base = logits + (size_t)row * V;
    int tid = threadIdx.x;
    float gm = global_max[row];

    float s = 0.0f;
    for (int j = tid; j < V; j += blockDim.x) {
        float v = __bfloat162float(base[j]) - gm;
        s += expf(v);
    }
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(mask, s, off);
    __shared__ float sdata[32];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) sdata[warp] = s;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        float v = (tid < n_warps) ? sdata[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
        if (lane == 0) local_sumexp[row] = v;
    }
}

// ---- subtract max from logits (for output side-effect to match reference) ----
__global__ void sub_max_kernel(
    __nv_bfloat16* logits, const float* gmax, int N, int V)
{
    int row = blockIdx.x;
    if (row >= N) return;
    __nv_bfloat16* base = logits + (size_t)row * V;
    float m = gmax[row];
    int tid = threadIdx.x;
    __nv_bfloat16 mb = __float2bfloat16(m);
    for (int j = tid; j < V; j += blockDim.x) {
        float v = __bfloat162float(base[j]) - m;
        base[j] = __float2bfloat16(v);
    }
}

// ---- multimem all-reduce kernels for f32 ----
// MAX reduction over multimem on f32
__global__ void multimem_allreduce_f32_max_kernel(
    uint64_t mc_base, const uint64_t* sigs,
    int64_t numel, int world_size, int rank)
{
    barrier_relaxed(sigs, blockIdx.x, rank, world_size);
    __syncthreads();

    int64_t numel_per_rank = (numel + world_size - 1) / world_size;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int gdim = gridDim.x;

    for (int64_t i = (int64_t)blockIdx.x * bdim + tid;
         i < numel_per_rank; i += (int64_t)gdim * bdim)
    {
        int64_t idx = (int64_t)rank * numel_per_rank + i;
        if (idx >= numel) continue;
        uint32_t* addr = reinterpret_cast<uint32_t*>(mc_base) + idx;
        uint32_t v;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.max.f32 %0, [%1];"
                     : "=r"(v) : "l"(addr) : "memory");
        asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;"
                     :: "l"(addr), "r"(v) : "memory");
    }

    __syncthreads();
    barrier_acq_rel(sigs, blockIdx.x, rank, world_size);
}

__global__ void multimem_allreduce_f32_sum_kernel(
    uint64_t mc_base, const uint64_t* sigs,
    int64_t numel, int world_size, int rank)
{
    barrier_relaxed(sigs, blockIdx.x, rank, world_size);
    __syncthreads();

    int64_t numel_per_rank = (numel + world_size - 1) / world_size;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int gdim = gridDim.x;

    for (int64_t i = (int64_t)blockIdx.x * bdim + tid;
         i < numel_per_rank; i += (int64_t)gdim * bdim)
    {
        int64_t idx = (int64_t)rank * numel_per_rank + i;
        if (idx >= numel) continue;
        uint32_t* addr = reinterpret_cast<uint32_t*>(mc_base) + idx;
        uint32_t v;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
                     : "=r"(v) : "l"(addr) : "memory");
        asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;"
                     :: "l"(addr), "r"(v) : "memory");
    }

    __syncthreads();
    barrier_acq_rel(sigs, blockIdx.x, rank, world_size);
}

// fallback peer-pointer allreduce f32 max/sum
__global__ void peer_allreduce_f32_kernel(
    const long long* __restrict__ ptrs, float* __restrict__ out,
    int world_size, int64_t n, int op /* 0=sum, 1=max */)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        if (op == 0) {
            float s = 0.0f;
            for (int r = 0; r < world_size; ++r) {
                s += ((const float*)ptrs[r])[idx];
            }
            out[idx] = s;
        } else {
            float m = -INFINITY;
            for (int r = 0; r < world_size; ++r) {
                float v = ((const float*)ptrs[r])[idx];
                if (v > m) m = v;
            }
            out[idx] = m;
        }
    }
}

// final compute: log(sum_exp) - predicted
__global__ void final_loss_kernel(
    const float* sum_exp, const float* pred, float* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = logf(sum_exp[i]) - pred[i];
}

// ---- launchers ----
void launch_local_max_pred(
    torch::Tensor logits, torch::Tensor target,
    torch::Tensor local_max, torch::Tensor local_pred,
    int64_t vocab_start, int64_t vocab_end)
{
    int N = local_max.numel();
    int V = logits.size(-1);
    int threads = 256;
    if (V < 256) {
        threads = 64;
        while (threads < V && threads < 256) threads *= 2;
    }
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    local_phase_max_pred_kernel<<<N, threads, 0, s>>>(
        (const __nv_bfloat16*)logits.data_ptr<at::BFloat16>(),
        target.data_ptr<long>(),
        local_max.data_ptr<float>(),
        local_pred.data_ptr<float>(),
        N, V, vocab_start, vocab_end);
}

void launch_local_sumexp(
    torch::Tensor logits, torch::Tensor gmax, torch::Tensor local_sumexp)
{
    int N = local_sumexp.numel();
    int V = logits.size(-1);
    int threads = 256;
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    local_phase_sumexp_kernel<<<N, threads, 0, s>>>(
        (const __nv_bfloat16*)logits.data_ptr<at::BFloat16>(),
        gmax.data_ptr<float>(),
        local_sumexp.data_ptr<float>(),
        N, V);
}

void launch_sub_max(torch::Tensor logits, torch::Tensor gmax) {
    int N = gmax.numel();
    int V = logits.size(-1);
    int threads = 256;
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    sub_max_kernel<<<N, threads, 0, s>>>(
        (__nv_bfloat16*)logits.data_ptr<at::BFloat16>(),
        gmax.data_ptr<float>(), N, V);
}

void launch_multimem_allreduce_f32(
    uint64_t mc_ptr, torch::Tensor sigs_dev,
    int64_t numel, int world_size, int rank, int op,
    int num_blocks, int block_size)
{
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* sigs = reinterpret_cast<const uint64_t*>(sigs_dev.data_ptr<int64_t>());
    if (op == 1) {
        multimem_allreduce_f32_max_kernel<<<num_blocks, block_size, 0, s>>>(
            mc_ptr, sigs, numel, world_size, rank);
    } else {
        multimem_allreduce_f32_sum_kernel<<<num_blocks, block_size, 0, s>>>(
            mc_ptr, sigs, numel, world_size, rank);
    }
}

void launch_peer_allreduce_f32(
    torch::Tensor ptrs, torch::Tensor out, int64_t n, int op)
{
    int world_size = ptrs.size(0);
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    peer_allreduce_f32_kernel<<<blocks, threads, 0, s>>>(
        (const long long*)ptrs.data_ptr<int64_t>(),
        out.data_ptr<float>(), world_size, n, op);
}

void launch_final_loss(torch::Tensor sum_exp, torch::Tensor pred, torch::Tensor out) {
    int N = out.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    final_loss_kernel<<<blocks, threads, 0, s>>>(
        sum_exp.data_ptr<float>(), pred.data_ptr<float>(),
        out.data_ptr<float>(), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_local_max_pred", &launch_local_max_pred);
    m.def("launch_local_sumexp", &launch_local_sumexp);
    m.def("launch_sub_max", &launch_sub_max);
    m.def("launch_multimem_allreduce_f32", &launch_multimem_allreduce_f32);
    m.def("launch_peer_allreduce_f32", &launch_peer_allreduce_f32);
    m.def("launch_final_loss", &launch_final_loss);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("vocab_par_ce_ext", CUDA_SRC)
    return _ext


_buf_cache = {}

def _get_symm_buf(N, device):
    """Single symmetric buffer of size N (f32) reused for all 3 reductions."""
    key = (N, device)
    if key in _buf_cache:
        return _buf_cache[key]
    buf = symm_mem.empty(N, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    _buf_cache[key] = (buf, hdl, ptrs)
    return _buf_cache[key]


def _can_multimem(N):
    # multimem.f32 needs 4-byte-aligned (always true) and divisible-by-world.
    # We'll require N % world_size == 0; otherwise fallback.
    return True


@torch.no_grad()
def solution(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    ext = _get_ext()

    orig_shape = target.shape
    device = vocab_parallel_logits.device
    partition_vocab_size = vocab_parallel_logits.shape[-1]
    vocab_start = rank * partition_vocab_size
    vocab_end = vocab_start + partition_vocab_size

    logits_2d = vocab_parallel_logits.reshape(-1, partition_vocab_size).contiguous()
    target_1d = target.reshape(-1).contiguous().to(torch.int64)
    N = target_1d.numel()

    # symmetric buffer for reductions (size N, f32). We'll do 3 reductions.
    # To keep things simple and correct, allocate three buffers of size N.
    # Reuse: we cache one buffer per N; for 3 reductions we use it sequentially.
    buf, hdl, ptrs_tensor = _get_symm_buf(N, device)

    # local outputs
    local_max = torch.empty(N, device=device, dtype=torch.float32)
    local_pred = torch.empty(N, device=device, dtype=torch.float32)
    local_sumexp = torch.empty(N, device=device, dtype=torch.float32)

    # ---- pass 1: local max + local pred ----
    ext.launch_local_max_pred(logits_2d, target_1d, local_max, local_pred,
                              vocab_start, vocab_end)

    def _allreduce_inplace(local_tensor: torch.Tensor, op: int) -> torch.Tensor:
        # op: 0=sum, 1=max
        buf.copy_(local_tensor)
        n = local_tensor.numel()
        use_mm = (n % world_size == 0) and (n >= world_size)
        if use_mm:
            # device-side barrier via signal pad
            # ensure writes visible
            num_blocks = min(8, max(1, (n + 255) // 256))
            block_size = 256
            ext.launch_multimem_allreduce_f32(
                int(hdl.multicast_ptr),
                hdl.signal_pad_ptrs_dev,
                n, world_size, rank, op,
                num_blocks, block_size)
            return buf.clone()
        else:
            hdl.barrier(channel=0)
            out = torch.empty(n, device=device, dtype=torch.float32)
            ext.launch_peer_allreduce_f32(ptrs_tensor, out, n, op)
            hdl.barrier(channel=0)
            return out

    # all-reduce max
    global_max = _allreduce_inplace(local_max, op=1)

    # all-reduce predicted (sum)
    global_pred = _allreduce_inplace(local_pred, op=0)

    # subtract max from logits (matches reference side effect)
    ext.launch_sub_max(logits_2d, global_max)

    # ---- pass 2: local sumexp using global max ----
    ext.launch_local_sumexp(logits_2d, global_max, local_sumexp)

    # all-reduce sumexp
    global_sumexp = _allreduce_inplace(local_sumexp, op=0)

    # final: log(sum_exp) - pred
    out = torch.empty(N, device=device, dtype=torch.float32)
    ext.launch_final_loss(global_sumexp, global_pred, out)

    return out.reshape(orig_shape)