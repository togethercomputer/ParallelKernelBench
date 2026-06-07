import os
import torch
import torch.distributed as dist
from utils.cuda_helpers import compile_cuda_extension
from utils.parallelkittens_runtime import (
    get_or_create_barrier,
    get_or_create_parallel_tensor,
)

TK_ROOT = os.environ.get("THUNDERKITTENS_ROOT", "/opt/thunderkittens")

# ---------------------------------------------------------------------------
# Embedded .cu source: ThunderKittens Multicast All-Gather
# ---------------------------------------------------------------------------
CUDA_SRC = r'''
#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_gather {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 2;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 8; // float4 -> 8 bf16 elements (16 bytes)
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;

    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true>;

    parallel_layout output;
    size_t out_offset;
    const bf16* input;
    const int dev_idx;
    const size_t numel_per_rank;

    __host__ inline dim3 grid() const {
        return dim3((numel_per_rank + NUM_ELEMS_PER_BLOCK - 1) / NUM_ELEMS_PER_BLOCK);
    }
};

__device__ inline void kernel(const globals &G) {
    const size_t idx = globals::NUM_ELEMS_PER_BLOCK * blockIdx.x +
                       globals::NUM_ELEMS_PER_INST * threadIdx.x;

    if (idx < G.numel_per_rank) {
        // Load 16 bytes from local chunk
        float4 tmp = reinterpret_cast<const float4*>(&G.input[idx])[0];
        
        // Target index in the flat output buffer
        const size_t out_idx = G.out_offset + G.dev_idx * G.numel_per_rank + idx;
        
        // Direct store to multicast pointer broadcasts the write across NVSwitch
        reinterpret_cast<float4*>(&G.output.mc_ptr[out_idx])[0] = tmp;
    }
}

} // namespace all_gather

namespace all_gather_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    barrier_t<NUM_DEVICES> barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    barrier_all(G.barrier, {0}, G.dev_idx);
}

} // namespace all_gather_barrier

void entrypoint(
    kittens::py::TKParallelTensor &output,
    size_t out_offset,
    uintptr_t input_ptr,
    size_t numel_per_rank,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(output, barrier);

    all_gather::globals all_gather_G {
        .output = kittens::py::parallel_tensor_to_pgl<typename all_gather::globals::parallel_layout>(output),
        .out_offset = out_offset,
        .input = reinterpret_cast<const bf16*>(input_ptr),
        .dev_idx = output.local_rank_,
        .numel_per_rank = numel_per_rank
    };

    all_gather_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<barrier_t<all_gather_barrier::globals::NUM_DEVICES>>(barrier),
        .dev_idx = barrier.local_rank_
    };

    // 1. Barrier ensures all devices are ready before overlapping writes
    kittens::py::launch_kernel<all_gather_barrier::config, all_gather_barrier::globals, all_gather_barrier::kernel>(barrier_G);
    
    // 2. Multicast broadcast
    kittens::py::launch_kernel<all_gather::config, all_gather::globals, all_gather::kernel>(all_gather_G);
    
    // 3. Barrier ensures all data landed globally before the stream unblocks cuBLAS
    kittens::py::launch_kernel<all_gather_barrier::config, all_gather_barrier::globals, all_gather_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_gather", &entrypoint);
}
'''

TK_CUDA_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-DKITTENS_HOPPER",
    "-gencode", "arch=compute_90a,code=sm_90a",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-DNDEBUG",
]

_ext = None
_ext_jit_ready = False


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension(
            "tk_pipelined_allgather_ext",
            CUDA_SRC,
            extra_cuda_cflags=TK_CUDA_FLAGS,
            extra_include_paths=[
                os.path.join(TK_ROOT, "include"),
                os.path.join(TK_ROOT, "prototype"),
            ],
            extra_ldflags=["-lcuda"],
        )
    return _ext


def _ensure_ext_jit():
    global _ext_jit_ready
    if _ext_jit_ready:
        return _get_ext()
    rank = dist.get_rank()
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    _ext_jit_ready = True
    return ext


@torch.no_grad()
def solution(A_local: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    world_size = dist.get_world_size()
    M, K_local = A_local.shape
    K_global, N = B.shape
    
    ext = _ensure_ext_jit()

    # We partition M into chunks to overlap ThunderKittens broadcast with cuBLAS Matmul
    NUM_CHUNKS = 4
    if M < NUM_CHUNKS * 8:
        NUM_CHUNKS = 1
        
    pad_M = 0
    align = NUM_CHUNKS * 8 # Guarantee 8 bf16s alignment per chunk for `float4` vectorized loads
    if M % align != 0:
        pad_M = align - (M % align)
        padded_A = torch.zeros((M + pad_M, K_local), dtype=A_local.dtype, device=A_local.device)
        padded_A[:M, :] = A_local
        A_local = padded_A

    M_padded = M + pad_M
    M_chunk = M_padded // NUM_CHUNKS
    numel_per_rank = K_local * M_chunk

    # Pre-slice and transpose the blocks to ensure they're fully contiguous in memory
    A_chunks_t = []
    for c in range(NUM_CHUNKS):
        chunk = A_local[c * M_chunk : (c+1) * M_chunk, :]
        A_chunks_t.append(chunk.transpose(0, 1).contiguous())

    B_t = B.transpose(0, 1).contiguous()
    C_t = torch.empty((N, M_padded), device=A_local.device, dtype=A_local.dtype)
    
    # Pre-allocate TK buffers for a double-buffered pipeline
    NUM_BUFFERS = min(2, NUM_CHUNKS)
    total_buffer_numel = NUM_BUFFERS * world_size * numel_per_rank
    
    # Allocates unified VMM mapping with NVSwitch Multicast capability
    buffer_tk = get_or_create_parallel_tensor(ext, (total_buffer_numel,), A_local.dtype, multicast=True)
    barrier_tk = get_or_create_barrier(ext, num_devices=world_size)

    stream_comm = torch.cuda.Stream()
    stream_comp = torch.cuda.current_stream()
    
    events_comm_done = [torch.cuda.Event() for _ in range(NUM_CHUNKS)]
    events_comp_done = [torch.cuda.Event() for _ in range(NUM_CHUNKS)]

    # Stream schedule loop
    for c in range(NUM_CHUNKS):
        buf_idx = c % NUM_BUFFERS
        out_offset = buf_idx * world_size * numel_per_rank
        
        with torch.cuda.stream(stream_comm):
            if c >= NUM_BUFFERS:
                stream_comm.wait_event(events_comp_done[c - NUM_BUFFERS])
                
            # Launch async TK PGL multicast
            ext.tk_all_gather(
                buffer_tk, 
                out_offset, 
                A_chunks_t[c].data_ptr(), 
                numel_per_rank, 
                barrier_tk
            )
            events_comm_done[c].record(stream_comm)
            
        with torch.cuda.stream(stream_comp):
            stream_comp.wait_event(events_comm_done[c])
            
            # Form standard contiguous PyTorch view spanning the globally gathered block
            gathered_data = buffer_tk.data_[out_offset : out_offset + world_size * numel_per_rank]
            A_global_chunk = gathered_data.view(world_size, K_local, M_chunk).reshape(K_global, M_chunk)
            
            # Standard tensor core matmul overlapping with the subsequent iteration's communication
            C_t_chunk = torch.matmul(B_t, A_global_chunk)
            
            C_t[:, c * M_chunk : (c+1) * M_chunk] = C_t_chunk
            events_comp_done[c].record(stream_comp)

    # Sync pipeline
    stream_comp.wait_stream(stream_comm)
    
    # Strip padding and final transpose
    C = C_t[:, :M].transpose(0, 1).contiguous()
    
    return C