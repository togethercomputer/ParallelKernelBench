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

__device__ __forceinline__ void send_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

// Barrier across all ranks using signal pads
__global__ void barrier_kernel(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int world_size,
    uint64_t channel
) {
    int tid = threadIdx.x;
    if (tid >= world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(
        remote_base + channel * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(
        local_base + channel * (uint64_t)world_size + (uint64_t)tid);
    send_signal(send_addr);
    wait_signal(wait_addr);
}

// Gather shards from peer symmetric buffers into a contiguous [M, H] tensor.
// Each rank's shard is at peer_buf[rank], shape [M, H_local].
// Output layout: x_full[m, r*H_local + h] = peer_buf[r][m, h]
__global__ void gather_shards_kernel(
    const uint64_t* __restrict__ peer_ptrs,  // [world_size]
    __nv_bfloat16* __restrict__ x_full,       // [M, H]
    int M,
    int H_local,
    int world_size
) {
    int H = H_local * world_size;
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= H) return;
    int r = col / H_local;
    int h = col - r * H_local;
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[r]);
    x_full[row * H + col] = src[row * H_local + h];
}

// Vectorized gather using float4 (8 bf16 per thread); requires H_local % 8 == 0
__global__ void gather_shards_kernel_vec(
    const uint64_t* __restrict__ peer_ptrs,
    __nv_bfloat16* __restrict__ x_full,
    int M,
    int H_local,
    int world_size
) {
    int H = H_local * world_size;
    int row = blockIdx.y;
    int vec_col = blockIdx.x * blockDim.x + threadIdx.x;  // index in 8-bf16 chunks
    int total_vecs = H / 8;
    if (row >= M || vec_col >= total_vecs) return;
    int col = vec_col * 8;
    int r = col / H_local;
    int h = col - r * H_local;
    const float4* src = reinterpret_cast<const float4*>(
        reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[r]) + row * H_local + h);
    float4* dst = reinterpret_cast<float4*>(x_full + row * H + col);
    *dst = *src;
}

// In-place SiLU on bf16
__global__ void silu_inplace_kernel(__nv_bfloat16* __restrict__ x, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float v = __bfloat162float(x[idx]);
        float s = v / (1.0f + __expf(-v));
        x[idx] = __float2bfloat16(s);
    }
}

// Write 'block' [M_local, H] from this rank into rank r's output slot.
// Specifically, this rank computes block_r and stores it into peer r's
// output buffer at offset 0 (peer r's output is its own [M_local, H]).
// We write into peer_out_ptrs[r] our local 'block' tensor.
__global__ void scatter_block_kernel(
    const __nv_bfloat16* __restrict__ block,  // [M_local, H]
    uint64_t dest_ptr,                         // remote rank r's output buffer
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(dest_ptr);
    const float4* src4 = reinterpret_cast<const float4*>(block);
    float4* dst4 = reinterpret_cast<float4*>(dst);
    int64_t n4 = n / 8;
    for (int64_t i = idx; i < n4; i += stride) {
        dst4[i] = src4[i];
    }
    // tail
    int64_t tail_start = n4 * 8;
    for (int64_t i = tail_start + idx; i < n; i += stride) {
        dst[i] = block[i];
    }
}

void launch_barrier(
    torch::Tensor signal_pad_ptrs,
    int rank,
    int world_size,
    int64_t channel
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_sig = reinterpret_cast<const uint64_t*>(signal_pad_ptrs.data_ptr<int64_t>());
    int threads = world_size;
    if (threads < 32) threads = 32;
    barrier_kernel<<<1, threads, 0, stream>>>(d_sig, rank, world_size, (uint64_t)channel);
}

void launch_gather_shards(
    torch::Tensor peer_ptrs,
    torch::Tensor x_full,
    int M,
    int H_local,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* d_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs.data_ptr<int64_t>());
    int H = H_local * world_size;
    if (H_local % 8 == 0) {
        int total_vecs = H / 8;
        int threads = 128;
        dim3 grid((total_vecs + threads - 1) / threads, M);
        gather_shards_kernel_vec<<<grid, threads, 0, stream>>>(
            d_ptrs, (__nv_bfloat16*)x_full.data_ptr<at::BFloat16>(),
            M, H_local, world_size);
    } else {
        int threads = 256;
        dim3 grid((H + threads - 1) / threads, M);
        gather_shards_kernel<<<grid, threads, 0, stream>>>(
            d_ptrs, (__nv_bfloat16*)x_full.data_ptr<at::BFloat16>(),
            M, H_local, world_size);
    }
}

void launch_silu_inplace(torch::Tensor x) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    silu_inplace_kernel<<<blocks, threads, 0, stream>>>(
        (__nv_bfloat16*)x.data_ptr<at::BFloat16>(), n);
}

void launch_scatter_block(
    torch::Tensor block,
    int64_t dest_ptr,
    int64_t n
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int threads = 256;
    int blocks = (int)((n / 8 + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    scatter_block_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)block.data_ptr<at::BFloat16>(),
        (uint64_t)dest_ptr, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_barrier", &launch_barrier);
    m.def("launch_gather_shards", &launch_gather_shards);
    m.def("launch_silu_inplace", &launch_silu_inplace);
    m.def("launch_scatter_block", &launch_scatter_block);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("tp_mlp_symm_ext", CUDA_SRC)
    return _ext


_cache = {}

def _get_resources(M, H_local, world_size, dtype, device):
    key = (M, H_local, world_size, dtype, device)
    if key in _cache:
        return _cache[key]

    H = H_local * world_size
    M_local = M // world_size

    # Symmetric input buffer for x_local [M, H_local]
    x_symm = symm_mem.empty((M, H_local), device=device, dtype=dtype)
    x_hdl = symm_mem.rendezvous(x_symm, dist.group.WORLD)

    # Symmetric output buffer for y_local [M_local, H]
    y_symm = symm_mem.empty((M_local, H), device=device, dtype=dtype)
    y_hdl = symm_mem.rendezvous(y_symm, dist.group.WORLD)

    x_peer_ptrs = torch.tensor(x_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    y_peer_ptrs = list(y_hdl.buffer_ptrs)

    x_signal = x_hdl.signal_pad_ptrs_dev
    y_signal = y_hdl.signal_pad_ptrs_dev

    x_full = torch.empty((M, H), device=device, dtype=dtype)

    res = {
        'x_symm': x_symm, 'x_hdl': x_hdl, 'x_peer_ptrs': x_peer_ptrs,
        'y_symm': y_symm, 'y_hdl': y_hdl, 'y_peer_ptrs': y_peer_ptrs,
        'x_signal': x_signal, 'y_signal': y_signal,
        'x_full': x_full,
        'rank': x_hdl.rank, 'world_size': x_hdl.world_size,
    }
    _cache[key] = res
    return res


_channel_counter = [0]

@torch.no_grad()
def solution(
    x_local: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized()
    assert x_local.is_cuda and W1.is_cuda and W2.is_cuda

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    M, H_local = x_local.shape
    H, ffn_dim = W1.shape
    M_local = M // world_size

    ext = _get_ext()
    res = _get_resources(M, H_local, world_size, x_local.dtype, x_local.device)

    # Step 1: copy x_local into symmetric buffer
    res['x_symm'].copy_(x_local)

    # Channel for this call (different per phase)
    ch1 = _channel_counter[0] % 8
    ch2 = (_channel_counter[0] + 1) % 8
    _channel_counter[0] = (_channel_counter[0] + 2) % 8

    # Barrier so all ranks have written x_symm
    ext.launch_barrier(res['x_signal'], rank, world_size, ch1)

    # Step 2: gather shards via UVA peer reads
    ext.launch_gather_shards(
        res['x_peer_ptrs'], res['x_full'], M, H_local, world_size
    )

    # Step 3: GEMM up-projection
    z = torch.matmul(res['x_full'], W1)  # [M, F]

    # Step 4: SiLU in place
    ext.launch_silu_inplace(z)

    # Step 5: this rank's row slice
    a_loc = z[rank * M_local : (rank + 1) * M_local].contiguous()

    # Step 6: down-projection
    block = torch.matmul(a_loc, W2)  # [M_local, H]

    # Step 7: scatter block directly into rank `rank`'s y output buffer.
    # Wait — need to think: each rank produces block for its own row slice.
    # In the reference, rank r writes nonzeros at rows [r*M_local:(r+1)*M_local]
    # and reduce_scatter sums over ranks then partitions by row block.
    # rank r receives row-block r; only rank r contributed nonzeros there.
    # So rank r's final output IS its own block. No remote write needed!
    # Just copy block into local y_symm.
    res['y_symm'].copy_(block)

    # Final barrier to ensure all ranks done before returning
    ext.launch_barrier(res['y_signal'], rank, world_size, ch2)

    return res['y_symm'].clone()