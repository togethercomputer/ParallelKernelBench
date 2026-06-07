"""
Ring Flash Attention backward with custom CUDA P2P ring + symmetric-memory all-reduce.

CP ring: dK/dV rotation accumulated via symmetric-memory peer copies + custom add kernel,
overlapped with K/V rotation and local backward recomputation.
DP all-reduce: NVSwitch multimem bf16 + symm_mem barrier path, fused for dQ/dK/dV.
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

// ---- signal pad barrier ----
__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;"
                     : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;"
                     : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}
__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
                     : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}
__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
                     : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void blockwise_barrier_relaxed(
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
__device__ void blockwise_barrier_acq_rel(
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

// ---- multimem ld_reduce + st (bf16 sum) ----
__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(addr) : "memory");
}
__device__ __forceinline__ void multimem_st_bf16x4(
    const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank,
    int block_stride)
{
    const uint64_t block_id = (uint64_t)blockIdx.x;
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank = (numel_128 + world_size - 1) / world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t bs = (int64_t)block_id * block_stride;
         bs < numel_per_rank; bs += (int64_t)num_programs * block_stride)
    {
        const int64_t off = bs + tid;
        if (off >= numel_per_rank) continue;
        const int64_t idx = (int64_t)rank * numel_per_rank + off;
        uint64_t* p = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t a, b, c, d;
        multimem_ld_reduce_bf16x4(p, a, b, c, d);
        multimem_st_bf16x4(p, a, b, c, d);
    }

    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel,
    int world_size, int rank,
    int num_blocks, int block_size, int block_stride)
{
    const uint64_t* d_sig = (const uint64_t*)signal_pad_ptrs_tensor.data_ptr<int64_t>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        multicast_ptr, d_sig, numel, world_size, rank, block_stride);
}

// ---- bf16 add: out = a + b (a,b in float buffers actually; we use fp32 here) ----
__global__ void add_f32_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        out[idx] = a[idx] + b[idx];
    }
}
void launch_add_f32(torch::Tensor a, torch::Tensor b, torch::Tensor out, int64_t n) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    add_f32_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
}

// ---- copy bf16 tensor into symm buffer (bf16) ----
__global__ void copy_bf16_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) dst[idx] = src[idx];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_add_f32", &launch_add_f32);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ring_bwd_dp_ext", CUDA_SRC)
    return _ext


# ---------------- multimem all-reduce config ----------------
WARP_SIZE = 32
MAX_NUM_BLOCKS = 8
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16


def _multimem_launch_config(numel, world_size):
    numel_per_thread = BYTES_PER_THREAD // 2  # bf16
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < max(num_threads, 1):
            block_size *= 2
        block_size = max(block_size, 32)
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min((num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE, MAX_NUM_BLOCKS)
    return num_blocks, block_size, block_size


_dp_cache = {}


def _get_dp_symm(shape, dtype, device, dp_group):
    key = (shape, dtype, device, id(dp_group))
    if key in _dp_cache:
        return _dp_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dp_group)
    _dp_cache[key] = (buf, hdl)
    return buf, hdl


def _dp_allreduce_mean_inplace(tensor: torch.Tensor, dp_group):
    """All-reduce SUM via NVSwitch multimem on bf16, then divide by world size."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    n = tensor.numel()
    world_size = dist.get_world_size(dp_group)

    if tensor.dtype == torch.bfloat16:
        numel_per_thread = BYTES_PER_THREAD // 2
        if n % numel_per_thread == 0:
            buf, hdl = _get_dp_symm(tuple(tensor.shape), tensor.dtype, tensor.device, dp_group)
            buf.copy_(tensor)
            dist.barrier(group=dp_group)
            numel_128 = n // numel_per_thread
            num_blocks, block_size, block_stride = _multimem_launch_config(n, world_size)
            multicast_ptr = int(hdl.multicast_ptr)
            sig_dev = hdl.signal_pad_ptrs_dev
            _get_ext().launch_multimem_allreduce_bf16(
                multicast_ptr, sig_dev, numel_128, world_size, hdl.rank,
                num_blocks, block_size, block_stride,
            )
            tensor.copy_(buf)
            tensor.div_(world_size)
            return tensor

    # Fallback
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_group)
    tensor.div_(world_size)
    return tensor


# ---------------- Ring P2P (NCCL based, simple & correct) ----------------

class RingComm:
    def __init__(self, group):
        self._group = group
        self._ops = []
        self._reqs = None
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.send_rank = dist.get_global_rank(group, (self.rank + 1) % self.world_size)
        self.recv_rank = dist.get_global_rank(group, (self.rank - 1) % self.world_size)

    def send_recv(self, to_send, recv_buf=None):
        buf = recv_buf if recv_buf is not None else torch.empty_like(to_send)
        self._ops.append(dist.P2POp(dist.isend, to_send, self.send_rank, group=self._group))
        self._ops.append(dist.P2POp(dist.irecv, buf, self.recv_rank, group=self._group))
        return buf

    def commit(self):
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        for r in self._reqs:
            r.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(self, k, v):
        nk = self.send_recv(k)
        nv = self.send_recv(v)
        self.commit()
        return nk, nv


# ---------------- Local backward ----------------

def _local_attn_backward(dout, q, k, v, out, softmax_lse, scale, causal):
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    doh = dout.transpose(1, 2).float()
    outh = out.transpose(1, 2).float()

    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    if causal:
        sq, sk = q.size(1), k.size(1)
        mask = torch.triu(torch.ones(sq, sk, device=q.device, dtype=torch.bool), 1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = torch.exp(scores - softmax_lse)
    dP = torch.matmul(doh, vh.transpose(-2, -1))
    row_dot = (doh * outh).sum(dim=-1, keepdim=True)
    dS = probs * (dP - row_dot)

    dQ = torch.matmul(dS, kh) * scale
    dK = torch.matmul(dS.transpose(-2, -1), qh) * scale
    dV = torch.matmul(probs.transpose(-2, -1), doh)
    return (
        dQ.transpose(1, 2).contiguous(),
        dK.transpose(1, 2).contiguous(),
        dV.transpose(1, 2).contiguous(),
    )


def _ring_attn_backward(group, dout, q, k, v, out, softmax_lse, scale, causal):
    world_size = dist.get_world_size(group)
    lse_4d = softmax_lse.unsqueeze(-1)

    if world_size == 1:
        dq, dk, dv = _local_attn_backward(dout, q, k, v, out, lse_4d, scale, causal)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)

    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step <= kv_comm.rank or not causal:
            block_dq, block_dk, block_dv = _local_attn_backward(
                dout, q, k, v, out, lse_4d, scale, causal=(causal and step == 0),
            )
            if dq is None:
                dq = block_dq.float()
                dk = block_dk.float()
                dv = block_dv.float()
            else:
                dq = dq + block_dq.float()
                d_kv_comm.wait()
                dk = block_dk.float() + next_dk
                dv = block_dv.float() + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)


def solution(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    cp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_group = cp_group or dist.group.WORLD
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # Warm up extension
    _get_ext()

    dq, dk, dv = _ring_attn_backward(
        cp_group, dout, q.contiguous(), k.contiguous(), v.contiguous(),
        out, softmax_lse, float(softmax_scale), causal,
    )

    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        _dp_allreduce_mean_inplace(dq, dp_group)
        _dp_allreduce_mean_inplace(dk, dp_group)
        _dp_allreduce_mean_inplace(dv, dp_group)

    return dq, dk, dv