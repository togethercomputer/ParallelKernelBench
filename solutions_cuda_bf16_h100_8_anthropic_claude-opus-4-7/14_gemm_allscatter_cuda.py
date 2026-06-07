import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

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

__device__ __forceinline__ void send_signal_acqrel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acqrel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ __forceinline__ void block_barrier_arrive(
    const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size)
{
    unsigned tid = threadIdx.x;
    if (tid >= (unsigned)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

__device__ __forceinline__ void block_barrier_depart(
    const uint64_t* signal_pad_ptrs, uint64_t block_id, int rank, int world_size)
{
    unsigned tid = threadIdx.x;
    if (tid >= (unsigned)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_acqrel(send_addr);
    wait_signal_acqrel(wait_addr);
}

// Each block copies one peer's shard slab -> our symmetric buffer slice.
// shard_bytes is the byte size of one [M, N_local] slab in bf16.
__global__ void gather_peer_shards_kernel(
    const uint64_t* __restrict__ buffer_ptrs,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size,
    int64_t shard_bytes
) {
    // Arrive: every rank guarantees its local shard is written.
    block_barrier_arrive(signal_pad_ptrs, 0, rank, world_size);
    __syncthreads();

    int peer = blockIdx.y;
    if (peer == rank) {
        __syncthreads();
        block_barrier_depart(signal_pad_ptrs, 1, rank, world_size);
        return;
    }

    const uint64_t local_base = buffer_ptrs[rank];
    const uint64_t remote_base = buffer_ptrs[peer];
    // peer's shard sits at offset peer * shard_bytes in both buffers
    const uint64_t off = (uint64_t)peer * (uint64_t)shard_bytes;

    const int4* src = reinterpret_cast<const int4*>(remote_base + off);
    int4* dst = reinterpret_cast<int4*>(local_base + off);

    int64_t n_vec = shard_bytes / 16;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n_vec; i += stride) {
        dst[i] = src[i];
    }

    // Tail bytes (should be 0 for bf16 with even shapes)
    int64_t tail_start = n_vec * 16;
    int64_t tail = shard_bytes - tail_start;
    if (tail > 0) {
        const char* sb = reinterpret_cast<const char*>(remote_base + off + tail_start);
        char* db = reinterpret_cast<char*>(local_base + off + tail_start);
        for (int64_t i = threadIdx.x; i < tail; i += blockDim.x) {
            if (blockIdx.x == 0) db[i] = sb[i];
        }
    }

    __syncthreads();
    block_barrier_depart(signal_pad_ptrs, 1, rank, world_size);
}

void launch_gather(
    uint64_t buffer_ptrs_dev,
    uint64_t signal_pad_ptrs_dev,
    int rank,
    int world_size,
    int64_t shard_bytes,
    int blocks_x
) {
    dim3 grid(blocks_x, world_size, 1);
    dim3 block(256, 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_peer_shards_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(buffer_ptrs_dev),
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_dev),
        rank, world_size, shard_bytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_gather", &launch_gather, "P2P all-gather of column shards");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("gemm_allscatter_p2p_ext", CUDA_SRC)
    return _ext

_cache = {}

def _get_resources(M, N_total, dtype, device):
    key = (M, N_total, dtype, device)
    if key in _cache:
        return _cache[key]
    # symmetric buffer holds full [M, N_total] in column-major shard order:
    # layout: shard r occupies rows [r*M*N_local : (r+1)*M*N_local) flattened
    # We'll store as [world_size, M, N_local] for simplicity.
    ws = dist.get_world_size()
    N_local = N_total // ws
    buf = symm_mem.empty((ws, M, N_local), device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    buffer_ptrs_dev = hdl.buffer_ptrs_dev
    signal_pad_ptrs_dev = hdl.signal_pad_ptrs_dev
    res = (buf, hdl, buffer_ptrs_dev, signal_pad_ptrs_dev, N_local)
    _cache[key] = res
    return res


@torch.no_grad()
def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized()
    assert A.is_cuda and B.is_cuda
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    _, N_local = B.shape
    N_total = N_local * world_size
    dtype = A.dtype
    device = A.device

    # Make sure extension exists everywhere before first launch
    _get_ext()

    buf, hdl, buf_ptrs, sig_ptrs, _ = _get_resources(M, N_total, dtype, device)

    # Compute local GEMM directly into our slot in the symmetric buffer.
    # buf shape: [world_size, M, N_local]; our slot is buf[rank]
    local_slot = buf[rank]  # [M, N_local], view
    torch.matmul(A, B, out=local_slot)

    # Custom P2P gather: pull each peer's slot into our buffer
    shard_bytes = M * N_local * A.element_size()
    # Choose blocks_x for vectorized copy
    n_vec = (shard_bytes + 15) // 16
    threads = 256
    blocks_x = int(min((n_vec + threads - 1) // threads, 64))
    if blocks_x < 1:
        blocks_x = 1

    _get_ext().launch_gather(
        int(buf_ptrs) if not isinstance(buf_ptrs, torch.Tensor) else int(buf_ptrs.data_ptr()),
        int(sig_ptrs) if not isinstance(sig_ptrs, torch.Tensor) else int(sig_ptrs.data_ptr()),
        rank, world_size, shard_bytes, blocks_x,
    )

    # buf is [world_size, M, N_local]; we need [M, world_size * N_local]
    # That's a permute+reshape (non-contiguous). Materialize into output.
    C = buf.permute(1, 0, 2).contiguous().reshape(M, N_total)
    return C