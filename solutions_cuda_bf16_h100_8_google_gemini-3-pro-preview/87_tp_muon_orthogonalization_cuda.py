import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Optional, Sequence
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Blockwise Barriers (Device-Side Synchronization)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void send_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.relaxed.sys.cas.b32 %0, [%1], 0, 1;" : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_relaxed(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.sys.relaxed.cas.b32 %0, [%1], 1, 0;" : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ __forceinline__ void send_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.release.sys.cas.b32 %0, [%1], 0, 1;" : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 0u);
}

__device__ __forceinline__ void wait_signal_acq_rel(uint32_t* addr) {
    uint32_t tmp;
    do {
        asm volatile("atom.global.sys.acquire.cas.b32 %0, [%1], 1, 0;" : "=r"(tmp) : "l"(addr) : "memory");
    } while (tmp != 1u);
}

__device__ void blockwise_barrier_relaxed(const uint64_t* __restrict__ signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_relaxed(send_addr);
    wait_signal_relaxed(wait_addr);
}

__device__ void blockwise_barrier_acq_rel(const uint64_t* __restrict__ signal_pad_ptrs, uint64_t block_id, int rank, int world_size) {
    unsigned int flat_tid = threadIdx.x;
    if (flat_tid >= (unsigned int)world_size) return;
    uint64_t local_base = signal_pad_ptrs[rank];
    uint64_t remote_base = signal_pad_ptrs[flat_tid];
    uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base + block_id * (uint64_t)world_size + (uint64_t)rank);
    uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base + block_id * (uint64_t)world_size + (uint64_t)flat_tid);
    send_signal_acq_rel(send_addr);
    wait_signal_acq_rel(wait_addr);
}

// ---------------------------------------------------------------------------
// Multimem operations & Kernels
// ---------------------------------------------------------------------------
__device__ __forceinline__ void multimem_ld_reduce_bf16x4(const uint64_t* addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(addr) : "memory");
}

__device__ __forceinline__ void multimem_st_bf16x4(const uint64_t* addr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" : : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

__global__ void multimem_allreduce_bf16_kernel(
    uint64_t multicast_base,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t numel_128,
    int world_size,
    int rank,
    int block_stride
) {
    const uint64_t block_id = static_cast<uint64_t>(blockIdx.x);
    blockwise_barrier_relaxed(signal_pad_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t numel_per_rank = (numel_128 + (int64_t)world_size - 1) / (int64_t)world_size;
    const int num_programs = gridDim.x;
    const int tid = threadIdx.x;

    for (int64_t block_start = (int64_t)block_id * (int64_t)block_stride; block_start < numel_per_rank; block_start += (int64_t)num_programs * (int64_t)block_stride) {
        const int64_t offsets = block_start + (int64_t)tid;
        if (offsets >= numel_per_rank) continue;
        const int64_t idx = (int64_t)rank * numel_per_rank + offsets;
        uint64_t* ptrs = reinterpret_cast<uint64_t*>(multicast_base) + idx * 2;
        uint32_t x, y, z, w;
        multimem_ld_reduce_bf16x4(ptrs, x, y, z, w);
        multimem_st_bf16x4(ptrs, x, y, z, w);
    }
    __syncthreads();
    blockwise_barrier_acq_rel(signal_pad_ptrs, block_id, rank, world_size);
}

void launch_multimem_allreduce_bf16(
    uint64_t multicast_ptr,
    torch::Tensor signal_pad_ptrs_tensor,
    int64_t numel,
    int world_size,
    int rank,
    int num_blocks,
    int block_size,
    int block_stride
) {
    const uint64_t* d_signal = reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    multimem_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(multicast_ptr, d_signal, numel, world_size, rank, block_stride);
}

// ---------------------------------------------------------------------------
// Fallback & Standard All-Reduce Kernels
// ---------------------------------------------------------------------------
__global__ void allreduce_bf16_kernel(const long long* __restrict__ ptrs, __nv_bfloat16* __restrict__ out, int world_size, int64_t n) {
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

__global__ void allreduce_f32_kernel(const long long* __restrict__ ptrs, float* __restrict__ out, int world_size, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < world_size; ++r) {
            sum += ((const float*)ptrs[r])[idx];
        }
        out[idx] = sum;
    }
}

void launch_allreduce(torch::Tensor ptrs_tensor, torch::Tensor out, int64_t n, int dtype_enum) {
    int world_size = ptrs_tensor.size(0);
    const long long* d_ptrs = (const long long*)ptrs_tensor.data_ptr<int64_t>();
    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (dtype_enum == 0) {
        allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(d_ptrs, (__nv_bfloat16*)out.data_ptr<at::BFloat16>(), world_size, n);
    } else {
        allreduce_f32_kernel<<<blocks, threads, 0, stream>>>(d_ptrs, out.data_ptr<float>(), world_size, n);
    }
}

// ---------------------------------------------------------------------------
// Fused Normalization & Output Cast Kernels
// ---------------------------------------------------------------------------
__global__ void scale_and_cast_bf16_kernel(const float* __restrict__ x, const float* __restrict__ norm_sq, __nv_bfloat16* __restrict__ out, float eps, int64_t n) {
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float n_sq = *norm_sq;
        s_scale = rsqrtf(n_sq < eps ? eps : n_sq);
    }
    __syncthreads();
    float scale = s_scale;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        out[idx] = __float2bfloat16(x[idx] * scale);
    }
}

void launch_scale_and_cast_bf16(torch::Tensor x, torch::Tensor norm_sq, torch::Tensor out, float eps) {
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scale_and_cast_bf16_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), norm_sq.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), eps, n);
}

__global__ void cast_to_f32_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += (int64_t)gridDim.x * blockDim.x) {
        out[idx] = __bfloat162float(x[idx]);
    }
}

__global__ void cast_to_f32_and_transpose_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int64_t rows, int64_t cols) {
    __shared__ float tile[32][33];
    int x_idx = blockIdx.x * 32 + threadIdx.x;
    int y_idx = blockIdx.y * 32 + threadIdx.y;
    if (x_idx < cols && y_idx < rows) {
        tile[threadIdx.y][threadIdx.x] = __bfloat162float(x[y_idx * cols + x_idx]);
    }
    __syncthreads();
    x_idx = blockIdx.y * 32 + threadIdx.x;
    y_idx = blockIdx.x * 32 + threadIdx.y;
    if (x_idx < rows && y_idx < cols) {
        out[y_idx * rows + x_idx] = tile[threadIdx.x][threadIdx.y];
    }
}

void launch_cast_to_f32(torch::Tensor x, torch::Tensor out, bool transpose) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (!transpose) {
        int64_t n = x.numel();
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;
        cast_to_f32_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), out.data_ptr<float>(), n);
    } else {
        int64_t rows = x.size(0);
        int64_t cols = x.size(1);
        dim3 threads(32, 32);
        dim3 blocks((cols + 31) / 32, (rows + 31) / 32);
        cast_to_f32_and_transpose_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), out.data_ptr<float>(), rows, cols);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_multimem_allreduce_bf16", &launch_multimem_allreduce_bf16);
    m.def("launch_allreduce", &launch_allreduce);
    m.def("launch_scale_and_cast_bf16", &launch_scale_and_cast_bf16);
    m.def("launch_cast_to_f32", &launch_cast_to_f32);
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("muon_tp_orthogonalization_ext", CUDA_SRC)
    return _ext

_resource_cache = {}
def _get_resources(shape, dtype, device, group):
    key = (shape, dtype, device)
    if key in _resource_cache:
        return _resource_cache[key]
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, group)
    out = torch.empty(shape, device=device, dtype=dtype)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, out, ptrs_tensor)
    _resource_cache[key] = res
    return res

WARP_SIZE = 32
MAX_NUM_BLOCKS = 4
MAX_BLOCK_SIZE = 1024
BYTES_PER_THREAD = 16

def _multimem_launch_config(numel: int, world_size: int) -> tuple[int, int, int]:
    numel_per_thread = BYTES_PER_THREAD // 2
    num_threads = (numel // numel_per_thread + world_size - 1) // world_size
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads: block_size *= 2
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_blocks = min((num_threads + MAX_BLOCK_SIZE - 1) // MAX_BLOCK_SIZE, MAX_NUM_BLOCKS)
    return num_blocks, block_size, block_size

_COEFFICIENTS: dict[str, Sequence[tuple[float, float, float]]] = {
    "simple": ((3.4445, -4.7750, 2.0315),),
    "quintic": (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ),
    "polar_express": (
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4647, 0.5074),
        (2.2779, -1.6447, 0.4162),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ),
    "aol": (
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ),
}

def _coefficient_at(coefficients: Sequence[tuple[float, float, float]], step: int) -> tuple[float, float, float]:
    return coefficients[step % len(coefficients)]

@torch.no_grad()
def solution(
    x: torch.Tensor,
    steps: int = 5,
    coefficient_type: str = "quintic",
    partition_dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    assert x.ndim == 2 and x.dtype == torch.float32
    coefficients = _COEFFICIENTS[coefficient_type]
    
    if partition_dim == 0:
        x_work = x.mT.contiguous()
    elif partition_dim == 1:
        x_work = x.contiguous()
        
    N, M = x_work.shape
    _get_ext()  # JIT compile

    buf_norm, hdl_norm, out_norm, ptrs_norm = _get_resources((1,), torch.float32, x.device, group)
    buf_gram, hdl_gram, out_gram, ptrs_gram = _get_resources((N, N), torch.bfloat16, x.device, group)
    
    compute_stream = torch.cuda.current_stream()
    comm_stream = torch.cuda.Stream()
    
    x_flat = x_work.flatten()
    buf_norm.copy_(torch.dot(x_flat, x_flat))
    compute_stream.synchronize()
    hdl_norm.barrier(channel=0)
    
    _get_ext().launch_allreduce(ptrs_norm, out_norm, 1, 1)
    
    x_work_bf16 = torch.empty((N, M), dtype=torch.bfloat16, device=x.device)
    _get_ext().launch_scale_and_cast_bf16(x_work, out_norm, x_work_bf16, 1e-7)
    
    update = torch.empty((N, N), dtype=torch.bfloat16, device=x.device)
    x_work_next = torch.empty_like(x_work_bf16)
    
    NUM_CHUNKS = 4
    chunk_size = N // NUM_CHUNKS
    use_multimem = True
    
    if (N * N) % 8 != 0 or not hasattr(hdl_gram, 'multicast_ptr') or hdl_gram.multicast_ptr is None or N % NUM_CHUNKS != 0 or (chunk_size * N) % 8 != 0:
        use_multimem = False
        NUM_CHUNKS = 1
        chunk_size = N
        
    events_comp = [torch.cuda.Event() for _ in range(NUM_CHUNKS)]
    events_comm = [torch.cuda.Event() for _ in range(NUM_CHUNKS)]
    
    for step in range(steps):
        a, b, c = _coefficient_at(coefficients, step)
        
        if use_multimem:
            for i in range(NUM_CHUNKS):
                start = i * chunk_size
                end = start + chunk_size
                torch.matmul(x_work_bf16[start:end], x_work_bf16.mT, out=buf_gram[start:end])
                events_comp[i].record(compute_stream)
                
                comm_stream.wait_event(events_comp[i])
                with torch.cuda.stream(comm_stream):
                    numel_chunk = chunk_size * N
                    numel_128 = numel_chunk // 8
                    num_blocks, block_size, block_stride = _multimem_launch_config(numel_chunk, hdl_gram.world_size)
                    chunk_multicast_ptr = int(hdl_gram.multicast_ptr) + (start * N * 2)
                    
                    _get_ext().launch_multimem_allreduce_bf16(
                        chunk_multicast_ptr, hdl_gram.signal_pad_ptrs_dev, numel_128,
                        hdl_gram.world_size, hdl_gram.rank, num_blocks, block_size, block_stride
                    )
                    events_comm[i].record(comm_stream)

            for i in range(NUM_CHUNKS):
                compute_stream.wait_event(events_comm[i])
            result_gram = buf_gram
        else:
            torch.matmul(x_work_bf16, x_work_bf16.mT, out=buf_gram)
            compute_stream.synchronize()
            hdl_gram.barrier(channel=0)
            _get_ext().launch_allreduce(ptrs_gram, out_gram, N * N, 0)
            compute_stream.synchronize()
            hdl_gram.barrier(channel=0)
            result_gram = out_gram
            
        torch.addmm(result_gram, result_gram, result_gram, beta=b, alpha=c, out=update)
        torch.addmm(x_work_bf16, update, x_work_bf16, beta=a, alpha=1.0, out=x_work_next)
        
        x_work_bf16, x_work_next = x_work_next, x_work_bf16

    out_f32 = torch.empty_like(x)
    _get_ext().launch_cast_to_f32(x_work_bf16, out_f32, partition_dim == 0)
    return out_f32