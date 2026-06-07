import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void store_release_u32(uint32_t* addr, uint32_t value) {
    asm volatile(
        "st.global.release.sys.u32 [%0], %1;"
        :
        : "l"(addr), "r"(value)
        : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_u32(const uint32_t* addr) {
    uint32_t value;
    asm volatile(
        "ld.global.acquire.sys.u32 %0, [%1];"
        : "=r"(value)
        : "l"(addr)
        : "memory");
    return value;
}

__device__ __forceinline__ void wait_eq_u32(const uint32_t* addr, uint32_t token) {
    uint32_t v;
    do {
        v = load_acquire_u32(addr);
    } while (v != token);
}

__global__ void remote_copy_to_dst_kernel(
    const char* __restrict__ src,
    const uint64_t* __restrict__ out_ptrs,
    int src_rank,
    int dst_rank,
    int64_t chunk_bytes
) {
    uint64_t dst_base_u = out_ptrs[dst_rank];
    char* __restrict__ dst =
        reinterpret_cast<char*>(dst_base_u + (uint64_t)src_rank * (uint64_t)chunk_bytes);

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Fast BF16/common path: chunk size is 16B-aligned, so every rank offset is aligned.
    if ((chunk_bytes & 15LL) == 0) {
        int64_t nvec = chunk_bytes >> 4;
        const uint4* __restrict__ src4 = reinterpret_cast<const uint4*>(src);
        uint4* __restrict__ dst4 = reinterpret_cast<uint4*>(dst);
        for (int64_t i = tid; i < nvec; i += stride) {
            dst4[i] = src4[i];
        }
    } else {
        // Correct fallback for odd byte counts / non-16B-aligned chunk offsets.
        for (int64_t i = tid; i < chunk_bytes; i += stride) {
            dst[i] = src[i];
        }
    }
}

__global__ void publish_ready_kernel(
    const uint64_t* __restrict__ sig_ptrs,
    int src_rank,
    int dst_rank,
    uint32_t token
) {
    if (threadIdx.x == 0) {
        uint32_t* dst_sig = reinterpret_cast<uint32_t*>(sig_ptrs[dst_rank]);
        __threadfence_system();
        store_release_u32(dst_sig + src_rank, token);
    }
}

__global__ void wait_all_ready_kernel(
    const uint32_t* __restrict__ local_sig,
    int world_size,
    uint32_t token
) {
    if (threadIdx.x < world_size) {
        wait_eq_u32(local_sig + threadIdx.x, token);
    }
}

__global__ void send_ack_kernel(
    const uint64_t* __restrict__ sig_ptrs,
    int world_size,
    int dst_rank,
    uint32_t token
) {
    int r = threadIdx.x;
    if (r < world_size) {
        uint32_t* peer_sig = reinterpret_cast<uint32_t*>(sig_ptrs[r]);
        __threadfence_system();
        store_release_u32(peer_sig + world_size + dst_rank, token);
    }
}

__global__ void wait_ack_kernel(
    const uint32_t* __restrict__ local_sig,
    int world_size,
    int dst_rank,
    uint32_t token
) {
    if (threadIdx.x == 0) {
        wait_eq_u32(local_sig + world_size + dst_rank, token);
    }
}

static inline int choose_blocks(int64_t chunk_bytes) {
    int64_t units = ((chunk_bytes & 15LL) == 0) ? (chunk_bytes >> 4) : chunk_bytes;
    int blocks = (int)((units + 255) / 256);
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    return blocks;
}

void launch_remote_copy_to_dst(
    torch::Tensor src,
    torch::Tensor out_ptrs_tensor,
    int src_rank,
    int dst_rank,
    int64_t chunk_bytes
) {
    TORCH_CHECK(src.is_cuda(), "src must be CUDA");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(out_ptrs_tensor.is_cuda(), "out_ptrs_tensor must be CUDA");
    TORCH_CHECK(out_ptrs_tensor.dtype() == torch::kInt64, "out_ptrs_tensor must be int64");

    const uint64_t* out_ptrs =
        reinterpret_cast<const uint64_t*>(out_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int blocks = choose_blocks(chunk_bytes);

    remote_copy_to_dst_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const char*>(src.data_ptr()),
        out_ptrs,
        src_rank,
        dst_rank,
        chunk_bytes
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_publish_ready(
    torch::Tensor sig_ptrs_tensor,
    int src_rank,
    int dst_rank,
    int token
) {
    TORCH_CHECK(sig_ptrs_tensor.is_cuda(), "sig_ptrs_tensor must be CUDA");
    TORCH_CHECK(sig_ptrs_tensor.dtype() == torch::kInt64, "sig_ptrs_tensor must be int64");

    const uint64_t* sig_ptrs =
        reinterpret_cast<const uint64_t*>(sig_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    publish_ready_kernel<<<1, 32, 0, stream>>>(
        sig_ptrs,
        src_rank,
        dst_rank,
        (uint32_t)token
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_wait_all_ready(
    torch::Tensor local_sig,
    int world_size,
    int token
) {
    TORCH_CHECK(local_sig.is_cuda(), "local_sig must be CUDA");
    TORCH_CHECK(local_sig.dtype() == torch::kInt32, "local_sig must be int32");
    TORCH_CHECK(local_sig.is_contiguous(), "local_sig must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    wait_all_ready_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(local_sig.data_ptr<int32_t>()),
        world_size,
        (uint32_t)token
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_send_ack(
    torch::Tensor sig_ptrs_tensor,
    int world_size,
    int dst_rank,
    int token
) {
    TORCH_CHECK(sig_ptrs_tensor.is_cuda(), "sig_ptrs_tensor must be CUDA");
    TORCH_CHECK(sig_ptrs_tensor.dtype() == torch::kInt64, "sig_ptrs_tensor must be int64");

    const uint64_t* sig_ptrs =
        reinterpret_cast<const uint64_t*>(sig_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    send_ack_kernel<<<1, 256, 0, stream>>>(
        sig_ptrs,
        world_size,
        dst_rank,
        (uint32_t)token
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_wait_ack(
    torch::Tensor local_sig,
    int world_size,
    int dst_rank,
    int token
) {
    TORCH_CHECK(local_sig.is_cuda(), "local_sig must be CUDA");
    TORCH_CHECK(local_sig.dtype() == torch::kInt32, "local_sig must be int32");
    TORCH_CHECK(local_sig.is_contiguous(), "local_sig must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    wait_ack_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(local_sig.data_ptr<int32_t>()),
        world_size,
        dst_rank,
        (uint32_t)token
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_remote_copy_to_dst", &launch_remote_copy_to_dst,
          "UVA remote-store gather chunk into destination symmetric output");
    m.def("launch_publish_ready", &launch_publish_ready,
          "Publish per-rank ready signal to destination");
    m.def("launch_wait_all_ready", &launch_wait_all_ready,
          "Destination waits for all ready signals");
    m.def("launch_send_ack", &launch_send_ack,
          "Destination sends gather-complete ack to all ranks");
    m.def("launch_wait_ack", &launch_wait_ack,
          "Non-destination waits for destination ack");
}
'''


_ext = None
_resource_cache = {}
_token = 0


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("symm_uva_gather_h100_bf16_ext", CUDA_SRC)
    return _ext


def _next_token() -> int:
    global _token
    _token += 1
    if _token >= 0x7FFFFFF0:
        _token = 1
    return _token


def _get_resources(shape, dtype, device, world_size):
    key = (tuple(shape), dtype, int(device.index) if device.index is not None else torch.cuda.current_device(), world_size)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    # Symmetric destination buffer: every rank allocates the same shape so any rank
    # can directly store into dst's local instance through hdl.buffer_ptrs[dst].
    gather_buf = symm_mem.empty((world_size, *tuple(shape)), device=device, dtype=dtype)

    # Signal layout per rank:
    #   [0:world_size)              ready slots consumed by dst
    #   [world_size:2*world_size)   ack slots consumed by sources, indexed by dst
    sig = symm_mem.empty((2 * world_size,), device=device, dtype=torch.int32)
    sig.zero_()
    torch.cuda.current_stream(device).synchronize()

    gather_hdl = symm_mem.rendezvous(gather_buf, dist.group.WORLD)
    sig_hdl = symm_mem.rendezvous(sig, dist.group.WORLD)

    gather_ptrs = torch.tensor(gather_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    sig_ptrs = torch.tensor(sig_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = {
        "gather_buf": gather_buf,
        "sig": sig,
        "gather_hdl": gather_hdl,
        "sig_hdl": sig_hdl,
        "gather_ptrs": gather_ptrs,
        "sig_ptrs": sig_ptrs,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda, "input tensor must be CUDA"
    assert tensor.is_contiguous(), "input tensor must be contiguous"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert 0 <= dst < world_size, "invalid destination rank"

    ext = _get_ext()
    res = _get_resources(tensor.shape, tensor.dtype, tensor.device, world_size)

    chunk_bytes = tensor.numel() * tensor.element_size()
    token = _next_token()

    # All ranks directly write their local chunk into dst's symmetric gather buffer.
    ext.launch_remote_copy_to_dst(
        tensor,
        res["gather_ptrs"],
        rank,
        dst,
        chunk_bytes,
    )

    # Publish only after the copy kernel has completed in-stream.
    ext.launch_publish_ready(
        res["sig_ptrs"],
        rank,
        dst,
        token,
    )

    if rank == dst:
        # Device-side completion: wait until every source has published readiness,
        # then ack all ranks so later collectives on the same stream are ordered.
        ext.launch_wait_all_ready(
            res["sig"],
            world_size,
            token,
        )
        ext.launch_send_ack(
            res["sig_ptrs"],
            world_size,
            dst,
            token,
        )
        return res["gather_buf"]

    ext.launch_wait_ack(
        res["sig"],
        world_size,
        dst,
        token,
    )
    return tensor