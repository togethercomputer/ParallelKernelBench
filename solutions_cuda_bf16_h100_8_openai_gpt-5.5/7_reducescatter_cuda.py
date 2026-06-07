# Device-side reduce-scatter for H100/NVLink:
# - Inputs are copied into symmetric memory once, then CUDA kernels read peer UVA pointers.
# - BF16 aligned chunks use NVSwitch multimem.ld_reduce to reduce directly in fabric.
# - Fallback kernels use peer loads with an in-kernel symmetric-memory signal barrier,
#   avoiding NCCL/torch.distributed collectives on the hot path.

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

void copy_bytes(torch::Tensor dst, torch::Tensor src, int64_t nbytes) {
    TORCH_CHECK(dst.is_cuda() && src.is_cuda(), "copy_bytes expects CUDA tensors");
    if (nbytes <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(), src.data_ptr(),
        static_cast<size_t>(nbytes),
        cudaMemcpyDeviceToDevice,
        stream));
}

void memset_zero_i32(torch::Tensor t) {
    TORCH_CHECK(t.is_cuda(), "memset_zero_i32 expects CUDA tensor");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    C10_CUDA_CHECK(cudaMemsetAsync(t.data_ptr(), 0, t.numel() * sizeof(int), stream));
}

// -----------------------------------------------------------------------------
// Device-side symmetric signal barrier.
// Each rank owns int32 signal[grid_blocks, world_size] in symmetric memory.
// For block b, thread peer sends to remote[rank] and waits on local[peer].
// CAS wait resets 1 -> 0, so the same signal storage is reusable.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void send_signal_release(int* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_acquire(int* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1u);
}

__device__ __forceinline__ void block_barrier(
    const long long* __restrict__ signal_ptrs,
    int block_id,
    int rank,
    int world_size
) {
    int t = threadIdx.x;
    if (t < world_size) {
        int* remote_base = reinterpret_cast<int*>(static_cast<uintptr_t>(signal_ptrs[t]));
        int* local_base  = reinterpret_cast<int*>(static_cast<uintptr_t>(signal_ptrs[rank]));

        int* send_addr = remote_base + block_id * world_size + rank;
        int* wait_addr = local_base  + block_id * world_size + t;

        send_signal_release(send_addr);
        wait_signal_acquire(wait_addr);
    }
}

// -----------------------------------------------------------------------------
// BF16 NVSwitch multimem reduce-scatter.
// Reduces exactly this rank's chunk. One multimem op reduces 8 BF16 values.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void multimem_ld_reduce_bf16x4(
    const uint64_t* addr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr)
        : "memory");
}

__global__ void rs_bf16_multimem_kernel(
    uint64_t multicast_base,
    const long long* __restrict__ signal_ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t chunk_elems,
    int64_t chunk_offset_elems,
    int world_size,
    int rank
) {
    const int block_id = blockIdx.x;

    block_barrier(signal_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t n128 = chunk_elems >> 3;  // 8 bf16 = 16 bytes
    const int64_t base128 = chunk_offset_elems >> 3;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    uint4* out128 = reinterpret_cast<uint4*>(out);

    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n128;
         i += stride) {
        const uint64_t* mm_addr =
            reinterpret_cast<const uint64_t*>(multicast_base) + (base128 + i) * 2;

        uint32_t a, b, c, d;
        multimem_ld_reduce_bf16x4(mm_addr, a, b, c, d);
        out128[i] = make_uint4(a, b, c, d);
    }

    __syncthreads();
    block_barrier(signal_ptrs, block_id, rank, world_size);
}

// -----------------------------------------------------------------------------
// Peer-UVA fallback reduce-scatter kernels.
// -----------------------------------------------------------------------------

template <typename T, typename Acc>
__device__ __forceinline__ Acc load_as(const T* p) {
    return static_cast<Acc>(*p);
}

template <>
__device__ __forceinline__ float load_as<__nv_bfloat16, float>(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <>
__device__ __forceinline__ float load_as<__half, float>(const __half* p) {
    return __half2float(*p);
}

template <typename T, typename Acc>
__device__ __forceinline__ void store_as(T* p, Acc v) {
    *p = static_cast<T>(v);
}

template <>
__device__ __forceinline__ void store_as<__nv_bfloat16, float>(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

template <>
__device__ __forceinline__ void store_as<__half, float>(__half* p, float v) {
    *p = __float2half_rn(v);
}

template <typename T, typename Acc>
__global__ void rs_p2p_kernel(
    const long long* __restrict__ input_ptrs,
    const long long* __restrict__ signal_ptrs,
    T* __restrict__ out,
    int64_t chunk_elems,
    int64_t chunk_offset_elems,
    int world_size,
    int rank
) {
    const int block_id = blockIdx.x;

    block_barrier(signal_ptrs, block_id, rank, world_size);
    __syncthreads();

    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < chunk_elems;
         idx += stride) {
        Acc sum = Acc(0);

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < world_size) {
                const T* src = reinterpret_cast<const T*>(
                    static_cast<uintptr_t>(input_ptrs[r]));
                sum += load_as<T, Acc>(src + chunk_offset_elems + idx);
            }
        }

        store_as<T, Acc>(out + idx, sum);
    }

    __syncthreads();
    block_barrier(signal_ptrs, block_id, rank, world_size);
}

// dtype_enum:
// 0 bf16, 1 f32, 2 f16, 3 f64, 4 i32, 5 i64, 6 i16, 7 i8, 8 u8
void launch_rs_p2p(
    torch::Tensor input_ptrs_tensor,
    torch::Tensor signal_ptrs_tensor,
    torch::Tensor out,
    int64_t chunk_elems,
    int64_t chunk_offset_elems,
    int world_size,
    int rank,
    int dtype_enum,
    int blocks,
    int threads
) {
    if (chunk_elems <= 0) return;

    const long long* input_ptrs =
        reinterpret_cast<const long long*>(input_ptrs_tensor.data_ptr<int64_t>());
    const long long* signal_ptrs =
        reinterpret_cast<const long long*>(signal_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    switch (dtype_enum) {
        case 0:
            rs_p2p_kernel<__nv_bfloat16, float><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs,
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        case 1:
            rs_p2p_kernel<float, float><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs, out.data_ptr<float>(),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        case 2:
            rs_p2p_kernel<__half, float><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs,
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        case 3:
            rs_p2p_kernel<double, double><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs, out.data_ptr<double>(),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        case 4:
            rs_p2p_kernel<int32_t, int32_t><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs, out.data_ptr<int32_t>(),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        case 5:
            rs_p2p_kernel<int64_t, int64_t><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs, out.data_ptr<int64_t>(),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        case 6:
            rs_p2p_kernel<int16_t, int32_t><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs,
                reinterpret_cast<int16_t*>(out.data_ptr<short>()),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        case 7:
            rs_p2p_kernel<int8_t, int32_t><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs,
                reinterpret_cast<int8_t*>(out.data_ptr<signed char>()),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        case 8:
            rs_p2p_kernel<uint8_t, int32_t><<<blocks, threads, 0, stream>>>(
                input_ptrs, signal_ptrs,
                reinterpret_cast<uint8_t*>(out.data_ptr<unsigned char>()),
                chunk_elems, chunk_offset_elems, world_size, rank);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype_enum");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_rs_bf16_multimem(
    uint64_t multicast_ptr,
    torch::Tensor signal_ptrs_tensor,
    torch::Tensor out,
    int64_t chunk_elems,
    int64_t chunk_offset_elems,
    int world_size,
    int rank,
    int blocks,
    int threads
) {
    if (chunk_elems <= 0) return;

    TORCH_CHECK((chunk_elems & 7) == 0, "BF16 multimem path requires chunk_elems multiple of 8");
    TORCH_CHECK((chunk_offset_elems & 7) == 0, "BF16 multimem path requires offset multiple of 8");

    const long long* signal_ptrs =
        reinterpret_cast<const long long*>(signal_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    rs_bf16_multimem_kernel<<<blocks, threads, 0, stream>>>(
        multicast_ptr,
        signal_ptrs,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        chunk_elems,
        chunk_offset_elems,
        world_size,
        rank);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_bytes", &copy_bytes, "Async device-to-device byte copy");
    m.def("memset_zero_i32", &memset_zero_i32, "Async zero int32 tensor");
    m.def("launch_rs_p2p", &launch_rs_p2p, "Peer-UVA reduce-scatter");
    m.def("launch_rs_bf16_multimem", &launch_rs_bf16_multimem,
          "BF16 NVSwitch multimem reduce-scatter");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("reducescatter_symm_uva_multimem_ext", CUDA_SRC)
    return _ext


MAX_SIGNAL_BLOCKS = 256
P2P_THREADS = 256
MM_THREADS = 256

_resource_cache = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 0
    if dtype is torch.float32:
        return 1
    if dtype is torch.float16:
        return 2
    if dtype is torch.float64:
        return 3
    if dtype is torch.int32:
        return 4
    if dtype is torch.int64:
        return 5
    if dtype is torch.int16:
        return 6
    if dtype is torch.int8:
        return 7
    if dtype is torch.uint8:
        return 8
    raise TypeError(f"unsupported dtype for custom reduce_scatter: {dtype}")


def _launch_blocks(work_items: int, threads: int) -> int:
    if work_items <= 0:
        return 1
    return max(1, min(MAX_SIGNAL_BLOCKS, _ceil_div(work_items, threads)))


def _get_resources(shape, dtype, device, world_size):
    key = (tuple(shape), dtype, int(device.index if device.index is not None else torch.cuda.current_device()), world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    ext = _get_ext()

    in_buf = symm_mem.empty(shape, device=device, dtype=dtype)
    in_hdl = symm_mem.rendezvous(in_buf, dist.group.WORLD)

    # One int32 flag per (CUDA block, peer rank). CAS wait resets flags to zero.
    signal_buf = symm_mem.empty((MAX_SIGNAL_BLOCKS, world_size), device=device, dtype=torch.int32)
    signal_hdl = symm_mem.rendezvous(signal_buf, dist.group.WORLD)
    ext.memset_zero_i32(signal_buf)
    signal_hdl.barrier(channel=0)

    chunk0 = shape[0] // world_size
    out_shape = (chunk0,) + tuple(shape[1:])
    out = torch.empty(out_shape, device=device, dtype=dtype)

    input_ptrs = torch.tensor(in_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    signal_ptrs = torch.tensor(signal_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "in_buf": in_buf,
        "in_hdl": in_hdl,
        "signal_buf": signal_buf,
        "signal_hdl": signal_hdl,
        "out": out,
        "input_ptrs": input_ptrs,
        "signal_ptrs": signal_ptrs,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda, "input must be CUDA"
    assert tensor.is_contiguous(), "input must be contiguous"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert tensor.shape[0] % world_size == 0, (
        f"First dimension ({tensor.shape[0]}) must be divisible by world_size ({world_size})"
    )

    ext = _get_ext()
    res = _get_resources(tensor.shape, tensor.dtype, tensor.device, world_size)

    total_elems = tensor.numel()
    chunk_elems = total_elems // world_size
    chunk_offset = rank * chunk_elems
    nbytes = total_elems * tensor.element_size()

    # Local D2D copy into symmetric memory; following CUDA kernel performs all
    # inter-rank synchronization and peer/NVSwitch reduction on device.
    ext.copy_bytes(res["in_buf"], tensor, nbytes)

    if chunk_elems == 0:
        return res["out"]

    # Fast path: H100/NVSwitch BF16 fabric reduction on 16-byte vectors.
    if tensor.dtype is torch.bfloat16 and (chunk_elems % 8 == 0):
        work_items = chunk_elems // 8
        blocks = _launch_blocks(work_items, MM_THREADS)
        ext.launch_rs_bf16_multimem(
            int(res["in_hdl"].multicast_ptr),
            res["signal_ptrs"],
            res["out"],
            chunk_elems,
            chunk_offset,
            world_size,
            rank,
            blocks,
            MM_THREADS,
        )
        return res["out"]

    # Generic peer-UVA fallback for tails / non-BF16 dtypes.
    blocks = _launch_blocks(chunk_elems, P2P_THREADS)
    ext.launch_rs_p2p(
        res["input_ptrs"],
        res["signal_ptrs"],
        res["out"],
        chunk_elems,
        chunk_offset,
        world_size,
        rank,
        _dtype_enum(tensor.dtype),
        blocks,
        P2P_THREADS,
    )
    return res["out"]