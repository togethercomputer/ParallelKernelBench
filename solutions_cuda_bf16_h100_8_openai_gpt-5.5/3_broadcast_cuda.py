import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int MAX_SIGNAL_BLOCKS = 4;
static constexpr int THREADS = 256;

// -----------------------------------------------------------------------------
// Symmetric-memory signal slots.
// One uint32 slot per (block_id, src_rank) in each rank's signal pad.
// Source sends one completion flag per CUDA block to every receiver; receiver
// blocks consume their own flag and then copy. This keeps the wait/copy path
// device-side and avoids torch.distributed/NCCL collectives.
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint32_t* signal_slot(
    const uint64_t* __restrict__ signal_pad_ptrs,
    int rank,
    int block_id,
    int world_size,
    int src
) {
    uint32_t* base = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank]);
    return base + (int64_t)block_id * world_size + src;
}

__device__ __forceinline__ void send_signal_release(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_acquire(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1u);
}

// -----------------------------------------------------------------------------
// Vectorized byte copy helpers.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void copy_bytes_grid(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    int64_t nbytes,
    bool aligned16
) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    if (aligned16) {
        const int64_t n16 = nbytes >> 4;
        uint4* __restrict__ d4 = reinterpret_cast<uint4*>(dst);
        const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src);

        for (int64_t i = tid; i < n16; i += stride) {
            d4[i] = s4[i];
        }

        const int64_t tail = n16 << 4;
        for (int64_t i = tail + tid; i < nbytes; i += stride) {
            dst[i] = src[i];
        }
    } else {
        for (int64_t i = tid; i < nbytes; i += stride) {
            dst[i] = src[i];
        }
    }
}

__global__ void direct_copy_kernel(
    const uint8_t* __restrict__ inp,
    uint8_t* __restrict__ out,
    int64_t nbytes,
    bool aligned16
) {
    copy_bytes_grid(out, inp, nbytes, aligned16);
}

__global__ void pack_source_kernel(
    const uint8_t* __restrict__ inp,
    uint8_t* __restrict__ symm_buf,
    int64_t nbytes,
    bool aligned16
) {
    copy_bytes_grid(symm_buf, inp, nbytes, aligned16);
}

// -----------------------------------------------------------------------------
// Hopper/NVSwitch multicast store path for aligned BF16 payloads.
// Source rank writes input once through the multicast UVA pointer; NVSwitch
// broadcasts into every rank's symmetric buffer.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void multimem_st_v4_u32_bits(
    uint64_t* addr,
    uint32_t x,
    uint32_t y,
    uint32_t z,
    uint32_t w
) {
    asm volatile(
        "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(addr), "r"(x), "r"(y), "r"(z), "r"(w)
        : "memory");
}

__global__ void multimem_broadcast_store_kernel(
    const uint4* __restrict__ inp4,
    uint64_t multicast_base,
    int64_t nvec16
) {
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < nvec16; i += stride) {
        uint4 v = inp4[i];
        uint64_t* dst = reinterpret_cast<uint64_t*>(multicast_base) + i * 2;
        multimem_st_v4_u32_bits(dst, v.x, v.y, v.z, v.w);
    }
}

// Source: after pack/multicast kernel has completed in stream order, signal all
// peers block-wise and copy local symmetric buffer to output.
__global__ void signal_and_copy_local_kernel(
    const uint8_t* __restrict__ local_src,
    uint8_t* __restrict__ out,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t nbytes,
    int world_size,
    int src,
    bool aligned16
) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        for (int r = 0; r < world_size; ++r) {
            if (r != src) {
                send_signal_release(signal_slot(
                    signal_pad_ptrs, r, blockIdx.x, world_size, src));
            }
        }
    }

    __syncthreads();
    copy_bytes_grid(out, local_src, nbytes, aligned16);
}

// Receiver: wait for source's per-block signal, then copy either local symmetric
// buffer (multicast path) or source rank's symmetric buffer via UVA (P2P path).
__global__ void wait_and_copy_kernel(
    const uint8_t* __restrict__ src_buf,
    uint8_t* __restrict__ out,
    const uint64_t* __restrict__ signal_pad_ptrs,
    int64_t nbytes,
    int world_size,
    int rank,
    int src,
    bool aligned16
) {
    if (threadIdx.x == 0) {
        wait_signal_acquire(signal_slot(
            signal_pad_ptrs, rank, blockIdx.x, world_size, src));
    }

    __syncthreads();
    copy_bytes_grid(out, src_buf, nbytes, aligned16);
}

// -----------------------------------------------------------------------------
// Launch wrappers.
// -----------------------------------------------------------------------------

static inline int choose_blocks(int64_t nbytes) {
    int64_t n16 = (nbytes + 15) / 16;
    int blocks = (int)((n16 + THREADS - 1) / THREADS);
    if (blocks < 1) blocks = 1;
    if (blocks > MAX_SIGNAL_BLOCKS) blocks = MAX_SIGNAL_BLOCKS;
    return blocks;
}

void launch_direct_copy(torch::Tensor inp, torch::Tensor out, int64_t nbytes) {
    TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uintptr_t a = reinterpret_cast<uintptr_t>(inp.data_ptr());
    const uintptr_t b = reinterpret_cast<uintptr_t>(out.data_ptr());
    const bool aligned16 = (((a | b) & 15ull) == 0ull);

    int blocks = choose_blocks(nbytes);
    direct_copy_kernel<<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(inp.data_ptr()),
        reinterpret_cast<uint8_t*>(out.data_ptr()),
        nbytes,
        aligned16);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_broadcast(
    torch::Tensor inp,
    torch::Tensor symm_buf,
    torch::Tensor out,
    torch::Tensor signal_pad_ptrs_tensor,
    uint64_t src_ptr,
    uint64_t multicast_ptr,
    int64_t nbytes,
    int world_size,
    int rank,
    int src,
    bool use_multimem
) {
    TORCH_CHECK(inp.is_cuda() && symm_buf.is_cuda() && out.is_cuda(),
                "inp/symm_buf/out must be CUDA tensors");
    TORCH_CHECK(inp.is_contiguous() && symm_buf.is_contiguous() && out.is_contiguous(),
                "inp/symm_buf/out must be contiguous");
    TORCH_CHECK(signal_pad_ptrs_tensor.is_cuda(), "signal_pad_ptrs must be CUDA");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint64_t* signal_pad_ptrs =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());

    const int blocks = choose_blocks(nbytes);

    uint8_t* out_u8 = reinterpret_cast<uint8_t*>(out.data_ptr());
    uint8_t* local_u8 = reinterpret_cast<uint8_t*>(symm_buf.data_ptr());
    const uint8_t* inp_u8 = reinterpret_cast<const uint8_t*>(inp.data_ptr());

    const uintptr_t out_addr = reinterpret_cast<uintptr_t>(out.data_ptr());
    const uintptr_t local_addr = reinterpret_cast<uintptr_t>(symm_buf.data_ptr());
    const uintptr_t inp_addr = reinterpret_cast<uintptr_t>(inp.data_ptr());
    const uintptr_t src_addr = static_cast<uintptr_t>(src_ptr);

    if (rank == src) {
        if (use_multimem) {
            int64_t nvec16 = nbytes >> 4;
            multimem_broadcast_store_kernel<<<blocks, THREADS, 0, stream>>>(
                reinterpret_cast<const uint4*>(inp.data_ptr()),
                multicast_ptr,
                nvec16);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            const bool aligned_copy =
                (((local_addr | out_addr) & 15ull) == 0ull);
            signal_and_copy_local_kernel<<<blocks, THREADS, 0, stream>>>(
                local_u8,
                out_u8,
                signal_pad_ptrs,
                nbytes,
                world_size,
                src,
                aligned_copy);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            const bool aligned_pack =
                (((inp_addr | local_addr) & 15ull) == 0ull);
            pack_source_kernel<<<blocks, THREADS, 0, stream>>>(
                inp_u8,
                local_u8,
                nbytes,
                aligned_pack);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            const bool aligned_copy =
                (((local_addr | out_addr) & 15ull) == 0ull);
            signal_and_copy_local_kernel<<<blocks, THREADS, 0, stream>>>(
                local_u8,
                out_u8,
                signal_pad_ptrs,
                nbytes,
                world_size,
                src,
                aligned_copy);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } else {
        const uint8_t* recv_src = use_multimem
            ? local_u8
            : reinterpret_cast<const uint8_t*>(src_addr);

        const uintptr_t recv_src_addr = use_multimem ? local_addr : src_addr;
        const bool aligned_recv =
            (((recv_src_addr | out_addr) & 15ull) == 0ull);

        wait_and_copy_kernel<<<blocks, THREADS, 0, stream>>>(
            recv_src,
            out_u8,
            signal_pad_ptrs,
            nbytes,
            world_size,
            rank,
            src,
            aligned_recv);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_direct_copy", &launch_direct_copy, "Direct CUDA byte copy");
    m.def("launch_broadcast", &launch_broadcast,
          "Symmetric-memory CUDA broadcast with BF16 multicast fast path");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_mem_broadcast_bf16_h100_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _cache_key(tensor: torch.Tensor):
    dev = tensor.device
    return (
        tuple(tensor.shape),
        tensor.dtype,
        dev.type,
        dev.index,
        dist.get_world_size() if dist.is_initialized() else 1,
    )


def _get_resources(tensor: torch.Tensor):
    key = _cache_key(tensor)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty(tuple(tensor.shape), device=tensor.device, dtype=tensor.dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = torch.empty_like(buf)

    cached = (buf, hdl, out)
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda, "input must be CUDA"
    assert tensor.is_contiguous(), "input must be contiguous"

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert 0 <= src < world_size, "invalid broadcast source rank"

    ext = _get_ext()
    nbytes = tensor.numel() * tensor.element_size()

    if nbytes == 0:
        return torch.empty_like(tensor)

    if world_size == 1:
        out = torch.empty_like(tensor)
        ext.launch_direct_copy(tensor, out, nbytes)
        return out.reshape_as(tensor)

    symm_buf, hdl, out = _get_resources(tensor)

    # BF16 fast path: aligned 16-byte chunks are written once by src through the
    # NVSwitch multicast mapping into every rank's symmetric buffer. Other dtypes
    # or tails use direct UVA P2P reads from src's symmetric buffer.
    use_multimem = (
        tensor.dtype is torch.bfloat16
        and (nbytes % 16 == 0)
        and hasattr(hdl, "multicast_ptr")
        and int(hdl.multicast_ptr) != 0
    )

    ext.launch_broadcast(
        tensor,
        symm_buf,
        out,
        hdl.signal_pad_ptrs_dev,
        int(hdl.buffer_ptrs[src]),
        int(hdl.multicast_ptr) if hasattr(hdl, "multicast_ptr") else 0,
        int(nbytes),
        int(world_size),
        int(rank),
        int(src),
        bool(use_multimem),
    )

    return out.reshape_as(tensor)