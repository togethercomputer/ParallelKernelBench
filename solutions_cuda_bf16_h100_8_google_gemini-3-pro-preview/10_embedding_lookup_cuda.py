import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void lookup_scan_all_vec_kernel(
    const int64_t* __restrict__ ptrs_meta_arr,
    const int64_t* __restrict__ ptrs_queries_arr,
    const int64_t* __restrict__ ptrs_out_arr,
    const T* __restrict__ local_shard,
    int my_rank,
    int world_size,
    int64_t shard_size,
    int64_t embed_dim,
    int vec_size
) {
    // 2D Grid: blockIdx.y = target remote rank, blockIdx.x = query chunk
    int target_rank = blockIdx.y;
    
    // Read the query count (N_A) for the remote rank we are inspecting
    const int64_t* meta_A = reinterpret_cast<const int64_t*>(ptrs_meta_arr[target_rank]);
    int64_t N_A = meta_A[0];
    
    const int64_t* queries_A = reinterpret_cast<const int64_t*>(ptrs_queries_arr[target_rank]);
    T* out_A = reinterpret_cast<T*>(ptrs_out_arr[target_rank]);
    
    int num_blocks_x = gridDim.x;
    int block_x = blockIdx.x;
    
    if (vec_size == 8 && embed_dim % 8 == 0) {
        int vec_dim = embed_dim / 8;
        const uint4* local_shard_vec = reinterpret_cast<const uint4*>(local_shard);
        uint4* out_A_vec = reinterpret_cast<uint4*>(out_A);
        
        for (int64_t q = block_x; q < N_A; q += num_blocks_x) {
            int64_t global_idx = queries_A[q];
            // Mimic Python's floor division `//` for negative indices
            int64_t target = global_idx >= 0 ? (global_idx / shard_size) : ((global_idx - shard_size + 1) / shard_size);
            
            if (target == my_rank) {
                int64_t local_offset = global_idx - my_rank * shard_size;
                // Clamp to safe range, analogous to torch.clamp(..., 0, shard_size - 1)
                if (local_offset < 0) local_offset = 0;
                if (local_offset >= shard_size) local_offset = shard_size - 1;
                
                // Vectorized peer-to-peer write
                for (int d = threadIdx.x; d < vec_dim; d += blockDim.x) {
                    out_A_vec[q * vec_dim + d] = local_shard_vec[local_offset * vec_dim + d];
                }
            }
        }
    } else if (vec_size == 4 && embed_dim % 4 == 0) {
        int vec_dim = embed_dim / 4;
        const uint2* local_shard_vec = reinterpret_cast<const uint2*>(local_shard);
        uint2* out_A_vec = reinterpret_cast<uint2*>(out_A);
        
        for (int64_t q = block_x; q < N_A; q += num_blocks_x) {
            int64_t global_idx = queries_A[q];
            int64_t target = global_idx >= 0 ? (global_idx / shard_size) : ((global_idx - shard_size + 1) / shard_size);
            
            if (target == my_rank) {
                int64_t local_offset = global_idx - my_rank * shard_size;
                if (local_offset < 0) local_offset = 0;
                if (local_offset >= shard_size) local_offset = shard_size - 1;
                
                for (int d = threadIdx.x; d < vec_dim; d += blockDim.x) {
                    out_A_vec[q * vec_dim + d] = local_shard_vec[local_offset * vec_dim + d];
                }
            }
        }
    } else {
        // Scalar fallback
        for (int64_t q = block_x; q < N_A; q += num_blocks_x) {
            int64_t global_idx = queries_A[q];
            int64_t target = global_idx >= 0 ? (global_idx / shard_size) : ((global_idx - shard_size + 1) / shard_size);
            
            if (target == my_rank) {
                int64_t local_offset = global_idx - my_rank * shard_size;
                if (local_offset < 0) local_offset = 0;
                if (local_offset >= shard_size) local_offset = shard_size - 1;
                
                for (int d = threadIdx.x; d < embed_dim; d += blockDim.x) {
                    out_A[q * embed_dim + d] = local_shard[local_offset * embed_dim + d];
                }
            }
        }
    }
}

void launch_lookup(
    torch::Tensor ptrs_meta,
    torch::Tensor ptrs_queries,
    torch::Tensor ptrs_out,
    torch::Tensor local_shard,
    int rank,
    int world_size,
    int64_t shard_size,
    int64_t embed_dim
) {
    int vec_size = 1;
    if (embed_dim % 8 == 0) vec_size = 8;
    else if (embed_dim % 4 == 0) vec_size = 4;
    
    // Distribute queries dynamically across robust grid of 1024 chunks
    int num_blocks_x = 1024;
    dim3 grid(num_blocks_x, world_size);
    dim3 block(128);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    const int64_t* p_meta = ptrs_meta.data_ptr<int64_t>();
    const int64_t* p_queries = ptrs_queries.data_ptr<int64_t>();
    const int64_t* p_out = ptrs_out.data_ptr<int64_t>();
    
    if (local_shard.scalar_type() == torch::kBFloat16) {
        lookup_scan_all_vec_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            p_meta, p_queries, p_out,
            reinterpret_cast<const __nv_bfloat16*>(local_shard.data_ptr()),
            rank, world_size, shard_size, embed_dim, vec_size
        );
    } else if (local_shard.scalar_type() == torch::kHalf) {
        lookup_scan_all_vec_kernel<__half><<<grid, block, 0, stream>>>(
            p_meta, p_queries, p_out,
            reinterpret_cast<const __half*>(local_shard.data_ptr()),
            rank, world_size, shard_size, embed_dim, vec_size
        );
    } else if (local_shard.scalar_type() == torch::kFloat32) {
        lookup_scan_all_vec_kernel<float><<<grid, block, 0, stream>>>(
            p_meta, p_queries, p_out,
            reinterpret_cast<const float*>(local_shard.data_ptr()),
            rank, world_size, shard_size, embed_dim, vec_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Valid types: float32, float16, bfloat16.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_lookup", &launch_lookup, "UVA P2P scan-all embedding lookup");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_lookup_scan_all_ext", CUDA_SRC)
    return _ext


class SymmState:
    def __init__(self):
        # We pre-allocate a generous symmetric buffer (4M queries maximum per rank) to avoid 
        # any host-device synchronization or collectives strictly on the hot path.
        self.capacity = 4194304
        self.embed_dim = -1
        self.dtype = None
        self.buf_meta = None
        self.hdl_meta = None
        self.ptrs_meta = None
        self.buf_queries = None
        self.hdl_queries = None
        self.ptrs_queries = None
        self.buf_out = None
        self.hdl_out = None
        self.ptrs_out = None
        self.is_initialized = False


_STATE = SymmState()

def _ensure_initialized(N: int, embed_dim: int, dtype: torch.dtype, device: torch.device):
    global _STATE
    if _STATE.is_initialized:
        return
        
    _STATE.embed_dim = embed_dim
    _STATE.dtype = dtype
    
    # meta buffer: 1 element to communicate local `N` without host-driven collective
    _STATE.buf_meta = symm_mem.empty((1,), dtype=torch.int64, device=device)
    _STATE.hdl_meta = symm_mem.rendezvous(_STATE.buf_meta, dist.group.WORLD)
    _STATE.ptrs_meta = torch.tensor(_STATE.hdl_meta.buffer_ptrs, dtype=torch.int64, device=device)
    
    # queries buffer (max `self.capacity` elements)
    _STATE.buf_queries = symm_mem.empty((_STATE.capacity,), dtype=torch.int64, device=device)
    _STATE.hdl_queries = symm_mem.rendezvous(_STATE.buf_queries, dist.group.WORLD)
    _STATE.ptrs_queries = torch.tensor(_STATE.hdl_queries.buffer_ptrs, dtype=torch.int64, device=device)
    
    # peer output buffer
    _STATE.buf_out = symm_mem.empty((_STATE.capacity, embed_dim), dtype=dtype, device=device)
    _STATE.hdl_out = symm_mem.rendezvous(_STATE.buf_out, dist.group.WORLD)
    _STATE.ptrs_out = torch.tensor(_STATE.hdl_out.buffer_ptrs, dtype=torch.int64, device=device)
    
    _STATE.is_initialized = True


@torch.no_grad()
def solution(indices: torch.Tensor, local_shard: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return local_shard[torch.clamp(indices, 0, local_shard.shape[0]-1)]

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_size = local_shard.shape[0]
    embed_dim = local_shard.shape[1]
    
    indices = indices.contiguous()
    local_shard = local_shard.contiguous()
    N = indices.numel()
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    _get_ext()
    
    _ensure_initialized(N, embed_dim, local_shard.dtype, indices.device)
    assert N <= _STATE.capacity, f"Query count (N={N}) exceeds generous symmetric capacity limit ({_STATE.capacity})"
    
    # 1. Provide `N` to meta mapping and stage contiguous queries. 
    # fill_() / copy_() occur completely asynchronously on the host stream.
    _STATE.buf_meta.fill_(N)
    if N > 0:
        _STATE.buf_queries[:N].copy_(indices)
        
    # 2. Wait for all ranks to complete uploading queries and configurations.
    _STATE.hdl_out.barrier(channel=0)
    
    # 3. Each rank executes peer evaluation natively pushing resolving elements cleanly backward.
    _get_ext().launch_lookup(
        _STATE.ptrs_meta,
        _STATE.ptrs_queries,
        _STATE.ptrs_out,
        local_shard,
        rank,
        world_size,
        shard_size,
        embed_dim
    )
    
    # 4. Stream safety wait to seal operations across all peer device actions.
    _STATE.hdl_out.barrier(channel=0)
    
    # 5. Provide discrete slice output tensor mimicking standard isolation behavior.
    if N > 0:
        output_vectors = _STATE.buf_out[:N].clone()
    else:
        output_vectors = torch.empty((0, embed_dim), dtype=local_shard.dtype, device=indices.device)
        
    return output_vectors