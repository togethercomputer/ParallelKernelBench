import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

__global__ void global_barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size,
    uint64_t block_id
) {
    unsigned int tid = threadIdx.x;
    if (tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + block_id * (uint64_t)world_size + (uint64_t)tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

// Fused dequant + remote put.
// For each destination peer d, dequantize local_y[d, :] and write to peer d's
// out_buf[rank, :] via UVA pointer.
__global__ void fused_dequant_a2a_kernel(
    const __nv_fp8_e4m3* __restrict__ y,        // [world_size, chunk_numel]
    const float* __restrict__ s,                // [world_size, num_blocks_per_chunk]
    const uint64_t* __restrict__ peer_buf_ptrs, // out symmetric buffer ptrs per peer
    int rank,
    int world_size,
    int64_t chunk_numel,
    int64_t blocks_per_chunk,
    int block_size
) {
    // grid.x: peer (destination)
    // grid.y: blocks within chunk
    int d = blockIdx.x;
    int64_t blk = blockIdx.y;
    if (blk >= blocks_per_chunk) return;

    int64_t chunk_offset = (int64_t)d * chunk_numel;
    int64_t blk_offset = blk * block_size;

    float scale = s[(int64_t)d * blocks_per_chunk + blk];

    const __nv_fp8_e4m3* y_blk = y + chunk_offset + blk_offset;

    // peer d's output buffer; we write into slot [rank, blk_offset:]
    __nv_bfloat16* out_peer = reinterpret_cast<__nv_bfloat16*>(peer_buf_ptrs[d]);
    __nv_bfloat16* out_dst = out_peer + (int64_t)rank * chunk_numel + blk_offset;

    int tid = threadIdx.x;
    int bs = blockDim.x;
    for (int i = tid; i < block_size; i += bs) {
        float v = (float)y_blk[i] * scale;
        out_dst[i] = __float2bfloat16(v);
    }
}

void launch_global_barrier(
    torch::Tensor signal_pad_ptrs,
    int rank,
    int world_size,
    int64_t block_id
) {
    const uint64_t* d_signal =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = world_size;
    if (threads < 32) threads = 32;
    global_barrier_kernel<<<1, threads, 0, stream>>>(d_signal, rank, world_size, (uint64_t)block_id);
}

void launch_fused_dequant_a2a(
    torch::Tensor y,                    // fp8 [world_size, chunk_numel]
    torch::Tensor s,                    // float32 [world_size, blocks_per_chunk]
    torch::Tensor peer_buf_ptrs,        // int64 [world_size]
    int rank,
    int world_size,
    int64_t chunk_numel,
    int64_t blocks_per_chunk,
    int block_size
) {
    const __nv_fp8_e4m3* y_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(y.data_ptr());
    const float* s_ptr = s.data_ptr<float>();
    const uint64_t* peers = reinterpret_cast<const uint64_t*>(peer_buf_ptrs.data_ptr<int64_t>());

    dim3 grid(world_size, (unsigned int)blocks_per_chunk, 1);
    int threads = block_size < 128 ? block_size : 128;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    fused_dequant_a2a_kernel<<<grid, threads, 0, stream>>>(
        y_ptr, s_ptr, peers, rank, world_size, chunk_numel, blocks_per_chunk, block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_global_barrier", &launch_global_barrier, "Global signal-pad barrier");
    m.def("launch_fused_dequant_a2a", &launch_fused_dequant_a2a, "Fused fp8 dequant + all-to-all put");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fused_dequant_a2a_ext", CUDA_SRC)
    return _ext

_resource_cache = {}
_barrier_counter = [0]

def _get_resources(shape, dtype, device, world_size):
    key = (tuple(shape), dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    peer_ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, peer_ptrs)
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    local_y: torch.Tensor,
    local_s: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert local_y.is_contiguous() and local_s.is_contiguous()
    assert local_y.shape[0] == world_size

    chunk_shape = local_y.shape[1:]
    chunk_numel = local_y.numel() // world_size
    assert chunk_numel % block_size == 0
    blocks_per_chunk = chunk_numel // block_size

    device = local_y.device
    out_shape = (world_size, *chunk_shape)

    # Output is bf16 per problem note
    out_dtype = torch.bfloat16
    buf, hdl, peer_ptrs = _get_resources(out_shape, out_dtype, device, world_size)

    ext = _get_ext()
    signal_dev = hdl.signal_pad_ptrs_dev

    # Barrier before writes (ensure all peers ready to receive)
    bid = _barrier_counter[0] % 64
    _barrier_counter[0] += 1
    ext.launch_global_barrier(signal_dev, rank, world_size, bid)

    if local_y.numel() > 0:
        ext.launch_fused_dequant_a2a(
            local_y.view(-1).view(world_size, chunk_numel) if False else local_y,
            local_s.view(world_size, blocks_per_chunk),
            peer_ptrs,
            rank,
            world_size,
            chunk_numel,
            blocks_per_chunk,
            block_size,
        )

    # Barrier after writes (ensure all peers' writes to our buf are visible)
    bid2 = _barrier_counter[0] % 64
    _barrier_counter[0] += 1
    ext.launch_global_barrier(signal_dev, rank, world_size, bid2)

    # Return a float32 copy to match reference dtype
    return buf.to(torch.float32).clone()