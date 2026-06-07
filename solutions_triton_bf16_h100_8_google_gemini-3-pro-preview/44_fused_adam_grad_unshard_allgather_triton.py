import math
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

// Struct to pass symmetric pointers without allocations
struct PtrArray {
    const void* ptrs[8];
};

template <typename master_t, typename grad_t>
__global__ void adam_kernel(
    const grad_t* __restrict__ grad_shard,
    const master_t* __restrict__ master_shard,
    const master_t* __restrict__ exp_avg,
    const master_t* __restrict__ exp_avg_sq,
    master_t* __restrict__ local_symm_buf,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2,
    int p, int chunk_start, int chunk_end
) {
    int idx = chunk_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_end) {
        float g = static_cast<float>(grad_shard[idx]);
        float m = static_cast<float>(exp_avg[idx]);
        float v = static_cast<float>(exp_avg_sq[idx]);
        float w = static_cast<float>(master_shard[idx]);

        m = m * beta1 + g * (1.0f - beta1);
        v = v * beta2 + g * g * (1.0f - beta2);

        float m_hat = m / bc1;
        float v_hat = v / bc2;

        w += (m_hat / (sqrtf(v_hat) + eps)) * (-lr);

        local_symm_buf[idx] = static_cast<master_t>(w);
    }
}

__global__ void update_flag_kernel(int* sync_flag, int target_flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Guarantee global memory fence before signalling peers
        __threadfence_system();
        *sync_flag = target_flag;
    }
}

template <typename master_t>
__global__ void gather_kernel(
    PtrArray peer_symm_bufs,
    PtrArray peer_sync_flags,
    master_t* __restrict__ out,
    int p, int chunk_start, int chunk_end, int target_flag,
    int world_size
) {
    int blocks_per_peer = gridDim.x / world_size;
    int peer = blockIdx.x / blocks_per_peer;
    int sub_chunk_idx = blockIdx.x % blocks_per_peer;
    
    // Thread 0 spins on peer's progress flag over NVLink
    if (threadIdx.x == 0) {
        volatile const int* flag_ptr = reinterpret_cast<volatile const int*>(peer_sync_flags.ptrs[peer]);
        while (*flag_ptr < target_flag) {
            #if __CUDA_ARCH__ >= 700
                __nanosleep(100);
            #endif
        }
    }
    __syncthreads();
    
    int chunk_size = chunk_end - chunk_start;
    int elements_per_block = (chunk_size + blocks_per_peer - 1) / blocks_per_peer;
    int start_idx = chunk_start + sub_chunk_idx * elements_per_block;
    int end_idx = chunk_start + (sub_chunk_idx + 1) * elements_per_block;
    if (end_idx > chunk_end) end_idx = chunk_end;
    
    const master_t* peer_buf = reinterpret_cast<const master_t*>(peer_symm_bufs.ptrs[peer]);
    int out_offset = peer * p;
    
    // Direct cross-device P2P load
    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        out[out_offset + i] = peer_buf[i];
    }
}

void adam_and_update_flag(
    torch::Tensor grad_shard,
    torch::Tensor master_shard,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    int64_t local_symm_buf_ptr,
    int64_t local_sync_flag_ptr,
    float lr, float beta1, float beta2, float eps, float bc1, float bc2,
    int p, int chunk_start, int chunk_end, int target_flag,
    int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int threads = 256;
    int chunk_size = chunk_end - chunk_start;
    int blocks_adam = (chunk_size + threads - 1) / threads;

    auto master_type = master_shard.scalar_type();
    auto grad_type = grad_shard.scalar_type();

    if (master_type == at::ScalarType::Float && grad_type == at::ScalarType::Float) {
        adam_kernel<float, float><<<blocks_adam, threads, 0, stream>>>(
            grad_shard.data_ptr<float>(), master_shard.data_ptr<float>(),
            exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
            reinterpret_cast<float*>(local_symm_buf_ptr),
            lr, beta1, beta2, eps, bc1, bc2, p, chunk_start, chunk_end
        );
    } else if (master_type == at::ScalarType::BFloat16 && grad_type == at::ScalarType::BFloat16) {
        adam_kernel<at::BFloat16, at::BFloat16><<<blocks_adam, threads, 0, stream>>>(
            grad_shard.data_ptr<at::BFloat16>(), master_shard.data_ptr<at::BFloat16>(),
            exp_avg.data_ptr<at::BFloat16>(), exp_avg_sq.data_ptr<at::BFloat16>(),
            reinterpret_cast<at::BFloat16*>(local_symm_buf_ptr),
            lr, beta1, beta2, eps, bc1, bc2, p, chunk_start, chunk_end
        );
    } else if (master_type == at::ScalarType::Float && grad_type == at::ScalarType::BFloat16) {
        adam_kernel<float, at::BFloat16><<<blocks_adam, threads, 0, stream>>>(
            grad_shard.data_ptr<at::BFloat16>(), master_shard.data_ptr<float>(),
            exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(),
            reinterpret_cast<float*>(local_symm_buf_ptr),
            lr, beta1, beta2, eps, bc1, bc2, p, chunk_start, chunk_end
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination");
    }

    update_flag_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<int*>(local_sync_flag_ptr), target_flag
    );
}

void gather_chunk(
    std::vector<int64_t> peer_symm_bufs_ptrs,
    std::vector<int64_t> peer_sync_flags_ptrs,
    torch::Tensor out,
    int p, int chunk_start, int chunk_end, int target_flag,
    int world_size, int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    PtrArray bufs, flags;
    for (int i = 0; i < world_size; ++i) {
        bufs.ptrs[i] = reinterpret_cast<const void*>(peer_symm_bufs_ptrs[i]);
        flags.ptrs[i] = reinterpret_cast<const void*>(peer_sync_flags_ptrs[i]);
    }

    int blocks_per_peer = 8;
    int total_blocks = world_size * blocks_per_peer;
    int threads = 256;

    auto master_type = out.scalar_type();
    if (master_type == at::ScalarType::Float) {
        gather_kernel<float><<<total_blocks, threads, 0, stream>>>(
            bufs, flags, out.data_ptr<float>(), p, chunk_start, chunk_end, target_flag, world_size
        );
    } else if (master_type == at::ScalarType::BFloat16) {
        gather_kernel<at::BFloat16><<<total_blocks, threads, 0, stream>>>(
            bufs, flags, out.data_ptr<at::BFloat16>(), p, chunk_start, chunk_end, target_flag, world_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam_and_update_flag", &adam_and_update_flag, "Adam compute and flag update");
    m.def("gather_chunk", &gather_chunk, "Gather chunk from all peers");
}
'''

# Process-level state cache
_ext = None
_symm_cache = None
_events = []
_stream_gather = None


@torch.no_grad()
def solution(
    grad_shard: torch.Tensor,
    master_shard: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> torch.Tensor:
    global _ext, _symm_cache, _events, _stream_gather

    assert dist.is_initialized(), "torch.distributed must be initialized"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    p = master_shard.numel()
    dtype = master_shard.dtype
    device = master_shard.device
    
    if _ext is None:
        _ext = compile_cuda_extension("fused_adam_gather_overlap", CUDA_SRC)
        _stream_gather = torch.cuda.Stream()

    # Provision/Maintain symmetric buffers & flags
    if _symm_cache is None or _symm_cache["p"] != p or _symm_cache["dtype"] != dtype:
        symm_data = symm_mem.empty(p, dtype=dtype, device=device)
        symm_flags = symm_mem.empty(1, dtype=torch.int32, device=device)
        symm_flags.zero_()
        
        hdl_data = symm_mem.rendezvous(symm_data, dist.group.WORLD)
        hdl_flags = symm_mem.rendezvous(symm_flags, dist.group.WORLD)
        
        peer_data_ptrs = [int(hdl_data.buffer_ptrs[i]) for i in range(world_size)]
        peer_flag_ptrs = [int(hdl_flags.buffer_ptrs[i]) for i in range(world_size)]
        
        _symm_cache = {
            "p": p, "dtype": dtype,
            "symm_data": symm_data, "symm_flags": symm_flags,
            "hdl_data": hdl_data, "hdl_flags": hdl_flags,
            "peer_data_ptrs": peer_data_ptrs, "peer_flag_ptrs": peer_flag_ptrs,
            "local_data_ptr": peer_data_ptrs[rank], "local_flag_ptr": peer_flag_ptrs[rank],
            "internal_step": 0,
        }
    
    cache = _symm_cache

    # Ensure previous iteration reads have finished before over-writing data chunks locally
    cache["hdl_data"].barrier(channel=0)
    
    cache["internal_step"] += 1
    internal_step = cache["internal_step"]

    out = torch.empty(world_size * p, dtype=dtype, device=device)
    
    # Pre-compute bias corrections
    bc1 = float(1.0 - math.pow(beta1, step))
    bc2 = float(1.0 - math.pow(beta2, step))
    
    num_chunks = 4
    chunk_size = (p + num_chunks - 1) // num_chunks
    stream_adam = torch.cuda.current_stream()
    
    while len(_events) < num_chunks:
        _events.append(torch.cuda.Event())
        
    for c in range(num_chunks):
        chunk_start = c * chunk_size
        chunk_end = min(chunk_start + chunk_size, p)
        if chunk_start >= p:
            break
            
        target_flag = internal_step * num_chunks + c + 1
        
        # 1. Fuse Adam step and dispatch async progress flag on the Compute Stream
        _ext.adam_and_update_flag(
            grad_shard, master_shard, exp_avg, exp_avg_sq,
            cache["local_data_ptr"], cache["local_flag_ptr"],
            lr, beta1, beta2, eps, bc1, bc2,
            p, chunk_start, chunk_end, target_flag,
            stream_adam.cuda_stream
        )
        
        # 2. Barrier event signaling chunk completion to Gather Stream (Local Dependency)
        _events[c].record(stream_adam)
        _stream_gather.wait_event(_events[c])
        
        # 3. Spinlock NVLink flags and directly copy distributed outputs to the final gathered tensor
        _ext.gather_chunk(
            cache["peer_data_ptrs"], cache["peer_flag_ptrs"],
            out, p, chunk_start, chunk_end, target_flag,
            world_size, _stream_gather.cuda_stream
        )

    # 4. Await memory visibility 
    stream_adam.wait_stream(_stream_gather)
    
    return out

__all__ = ["solution"]