import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

# Strategy:
# - Replace NCCL scatter with a source-rank CUDA kernel that writes chunks directly
#   into each rank's symmetric output buffer through UVA/NVLink peer pointers.
# - Use 128-bit vectorized copies for BF16/aligned payloads; fall back to byte copy.
# - Avoid per-call torch.distributed collectives: receivers wait on device-side
#   release/acquire signal words in symmetric memory.

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void st_release_sys_u32(uint32_t* addr, uint32_t val) {
    asm volatile(
        "st.release.sys.global.u32 [%0], %1;"
        :
        : "l"(addr), "r"(val)
        : "memory");
}

__device__ __forceinline__ uint32_t ld_acquire_sys_u32(const uint32_t* addr) {
    uint32_t val;
    asm volatile(
        "ld.acquire.sys.global.u32 %0, [%1];"
        : "=r"(val)
        : "l"(addr)
        : "memory");
    return val;
}

__global__ void scatter_src_kernel(
    const char* __restrict__ src,
    const long long* __restrict__ data_ptrs,
    const long long* __restrict__ sig_ptrs,
    int* __restrict__ done_counters,
    int64_t chunk_bytes,
    int64_t n_vec16,
    int64_t tail_bytes,
    int blocks_per_rank,
    uint32_t seq,
    bool use_vec16
) {
    const int dst_rank = blockIdx.y;
    const int bx = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    char* dst = reinterpret_cast<char*>(
        static_cast<uintptr_t>(data_ptrs[dst_rank]));
    const char* src_chunk = src + (int64_t)dst_rank * chunk_bytes;

    if (use_vec16) {
        const uint4* __restrict__ src4 =
            reinterpret_cast<const uint4*>(src_chunk);
        uint4* __restrict__ dst4 =
            reinterpret_cast<uint4*>(dst);

        for (int64_t i = (int64_t)bx * blockDim.x + tid;
             i < n_vec16;
             i += stride) {
            dst4[i] = src4[i];
        }

        if (bx == 0 && tail_bytes != 0) {
            const int64_t base = n_vec16 * 16;
            for (int64_t b = tid; b < tail_bytes; b += blockDim.x) {
                dst[base + b] = src_chunk[base + b];
            }
        }
    } else {
        for (int64_t b = (int64_t)bx * blockDim.x + tid;
             b < chunk_bytes;
             b += stride) {
            dst[b] = src_chunk[b];
        }
    }

    // Make all peer writes by this CTA visible before contributing completion.
    __threadfence_system();
    __syncthreads();

    if (tid == 0) {
        int old = atomicAdd(done_counters + dst_rank, 1);
        if (old == blocks_per_rank - 1) {
            done_counters[dst_rank] = 0;

            // Publish completion to the destination rank only after all CTAs
            // assigned to that destination have completed their peer stores.
            __threadfence_system();

            uint32_t* remote_sig = reinterpret_cast<uint32_t*>(
                static_cast<uintptr_t>(sig_ptrs[dst_rank]));
            st_release_sys_u32(remote_sig, seq);
        }
    }
}

__global__ void wait_signal_kernel(
    const int* __restrict__ sig,
    uint32_t seq
) {
    if (threadIdx.x == 0) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(sig);
        uint32_t v = ld_acquire_sys_u32(p);
        int ns = 32;
        while (v != seq) {
            asm volatile("nanosleep.u32 %0;" :: "r"(ns));
            if (ns < 1024) ns <<= 1;
            v = ld_acquire_sys_u32(p);
        }
    }
}

void launch_scatter_src(
    torch::Tensor src,
    torch::Tensor data_ptrs,
    torch::Tensor sig_ptrs,
    torch::Tensor counters,
    int64_t chunk_bytes,
    int world_size,
    uint32_t seq,
    bool use_vec16
) {
    TORCH_CHECK(src.is_cuda(), "src must be CUDA");
    TORCH_CHECK(data_ptrs.is_cuda() && sig_ptrs.is_cuda(), "ptr tensors must be CUDA");
    TORCH_CHECK(counters.is_cuda(), "counters must be CUDA");
    TORCH_CHECK(data_ptrs.dtype() == torch::kInt64, "data_ptrs must be int64");
    TORCH_CHECK(sig_ptrs.dtype() == torch::kInt64, "sig_ptrs must be int64");
    TORCH_CHECK(counters.dtype() == torch::kInt32, "counters must be int32");

    constexpr int threads = 256;
    int64_t n_vec16 = use_vec16 ? (chunk_bytes / 16) : 0;
    int64_t tail = use_vec16 ? (chunk_bytes - n_vec16 * 16) : 0;
    int64_t work_items = use_vec16 ? n_vec16 : chunk_bytes;

    int blocks_per_rank = 1;
    if (work_items > 0) {
        blocks_per_rank = (int)((work_items + threads - 1) / threads);
        if (blocks_per_rank < 1) blocks_per_rank = 1;
        if (blocks_per_rank > 1024) blocks_per_rank = 1024;
    }

    dim3 grid(blocks_per_rank, world_size, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    scatter_src_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const char*>(src.data_ptr()),
        reinterpret_cast<const long long*>(data_ptrs.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(sig_ptrs.data_ptr<int64_t>()),
        counters.data_ptr<int>(),
        chunk_bytes,
        n_vec16,
        tail,
        blocks_per_rank,
        seq,
        use_vec16
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_wait_signal(torch::Tensor sig, uint32_t seq) {
    TORCH_CHECK(sig.is_cuda(), "sig must be CUDA");
    TORCH_CHECK(sig.dtype() == torch::kInt32, "sig must be int32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    wait_signal_kernel<<<1, 32, 0, stream>>>(
        sig.data_ptr<int>(),
        seq
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_scatter_src", &launch_scatter_src,
          "UVA symmetric-memory scatter source kernel");
    m.def("launch_wait_signal", &launch_wait_signal,
          "Device-side wait on symmetric-memory signal");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("scatter_uva_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


_state_cache = {}


def _normalize_device(device: torch.device) -> torch.device:
    if device.type != "cuda":
        return device
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    return torch.device("cuda", idx)


def _get_state(chunk_shape, dtype, device, world_size):
    key = (tuple(chunk_shape), dtype, _normalize_device(device), int(world_size))
    state = _state_cache.get(key)
    if state is not None:
        return state

    out_buf = symm_mem.empty(tuple(chunk_shape), device=device, dtype=dtype)
    out_hdl = symm_mem.rendezvous(out_buf, dist.group.WORLD)

    sig = symm_mem.empty((1,), device=device, dtype=torch.int32)
    sig.zero_()
    sig_hdl = symm_mem.rendezvous(sig, dist.group.WORLD)

    # One-time ordering for signal initialization only. The hot path below uses
    # custom device-side release/acquire signaling instead of distributed scatter.
    sig_hdl.barrier(channel=0)

    data_ptrs_host = [int(p) for p in out_hdl.buffer_ptrs]
    sig_ptrs_host = [int(p) for p in sig_hdl.buffer_ptrs]

    data_ptrs = torch.tensor(data_ptrs_host, device=device, dtype=torch.int64)
    sig_ptrs = torch.tensor(sig_ptrs_host, device=device, dtype=torch.int64)

    counters = torch.empty((world_size,), device=device, dtype=torch.int32)
    counters.zero_()

    state = {
        "out_buf": out_buf,
        "out_hdl": out_hdl,
        "sig": sig,
        "sig_hdl": sig_hdl,
        "data_ptrs": data_ptrs,
        "sig_ptrs": sig_ptrs,
        "data_ptrs_host": data_ptrs_host,
        "counters": counters,
        "seq": 0,
    }
    _state_cache[key] = state
    return state


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert tensor.is_cuda, "tensor must be CUDA"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert 0 <= src < world_size, "invalid src rank"

    if rank == src:
        assert tensor.dim() >= 1, "source tensor must have leading world dimension"
        assert tensor.shape[0] == world_size, (
            f"Source tensor must have {world_size} chunks"
        )
        chunk_shape = tuple(tensor.shape[1:])
        src_tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
    else:
        chunk_shape = tuple(tensor.shape)
        src_tensor = None

    ext = _get_ext()
    state = _get_state(chunk_shape, tensor.dtype, tensor.device, world_size)

    seq = state["seq"] + 1
    if seq >= 0x7FFFFFF0:
        seq = 1
    state["seq"] = seq

    if rank == src:
        out_buf = state["out_buf"]
        chunk_bytes = int(out_buf.numel() * out_buf.element_size())

        src_addr = int(src_tensor.data_ptr())
        aligned16 = (src_addr & 15) == 0 and (chunk_bytes & 15) == 0
        if aligned16:
            for p in state["data_ptrs_host"]:
                if (int(p) & 15) != 0:
                    aligned16 = False
                    break

        ext.launch_scatter_src(
            src_tensor,
            state["data_ptrs"],
            state["sig_ptrs"],
            state["counters"],
            chunk_bytes,
            world_size,
            int(seq),
            bool(aligned16),
        )

    ext.launch_wait_signal(state["sig"], int(seq))
    return state["out_buf"].reshape(chunk_shape)