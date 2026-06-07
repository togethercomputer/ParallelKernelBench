import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

// Vectorized kernel: reads 8 bfloat16 elements (16 bytes) at once over NVLink
__global__ void pull_a_kernel_vec(
    const uint64_t* __restrict__ peer_ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t start_row,
    int64_t num_rows,
    int64_t K_local,
    int world_size
) {
    int peer_idx = blockIdx.y;
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[peer_idx]);
    
    int64_t K_global = K_local * world_size;
    int64_t vec_K = K_local / 8;
    
    int64_t col_vec_start = blockIdx.z * blockDim.x + threadIdx.x;
    int64_t row_start = blockIdx.x * blockDim.y + threadIdx.y;
    
    int64_t row_stride = gridDim.x * blockDim.y;
    int64_t col_stride = gridDim.z * blockDim.x;
    
    for (int64_t row = row_start; row < num_rows; row += row_stride) {
        int64_t src_row = start_row + row;
        int64_t out_row_offset = row * K_global + peer_idx * K_local;
        int64_t src_row_offset = src_row * K_local;
        
        for (int64_t col_vec = col_vec_start; col_vec < vec_K; col_vec += col_stride) {
            int64_t src_idx = src_row_offset + col_vec * 8;
            int64_t out_idx = out_row_offset + col_vec * 8;
            
            float4 val = *reinterpret_cast<const float4*>(&src[src_idx]);
            *reinterpret_cast<float4*>(&out[out_idx]) = val;
        }
    }
}

// Scalar kernel fallback for dimensions not perfectly divisible by 8
__global__ void pull_a_kernel_scalar(
    const uint64_t* __restrict__ peer_ptrs,
    __nv_bfloat16* __restrict__ out,
    int64_t start_row,
    int64_t num_rows,
    int64_t K_local,
    int world_size
) {
    int peer_idx = blockIdx.y;
    const __nv_bfloat16* src = reinterpret_cast<const __nv_bfloat16*>(peer_ptrs[peer_idx]);
    
    int64_t K_global = K_local * world_size;
    
    int64_t col_start = blockIdx.z * blockDim.x + threadIdx.x;
    int64_t row_start = blockIdx.x * blockDim.y + threadIdx.y;
    
    int64_t row_stride = gridDim.x * blockDim.y;
    int64_t col_stride = gridDim.z * blockDim.x;
    
    for (int64_t row = row_start; row < num_rows; row += row_stride) {
        int64_t src_row = start_row + row;
        int64_t out_row_offset = row * K_global + peer_idx * K_local;
        int64_t src_row_offset = src_row * K_local;
        
        for (int64_t col = col_start; col < K_local; col += col_stride) {
            int64_t src_idx = src_row_offset + col;
            int64_t out_idx = out_row_offset + col;
            
            out[out_idx] = src[src_idx];
        }
    }
}

void launch_pull_a(
    torch::Tensor peer_ptrs_tensor,
    torch::Tensor out,
    int64_t start_row,
    int64_t num_rows,
    int64_t K_local,
    int world_size,
    int64_t stream_ptr
) {
    const uint64_t* peer_ptrs = reinterpret_cast<const uint64_t*>(peer_ptrs_tensor.data_ptr<int64_t>());
    __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    // We intentionally cap grid dimensions to only use ~10-20% of the device's SMs. 
    // This leaves the majority of SMs completely free for overlapping Tensor Core matrix math!
    if (K_local % 8 == 0) {
        dim3 threads(32, 8);
        int64_t vec_K = K_local / 8;
        
        int blocks_x = std::min<int>((num_rows + threads.y - 1) / threads.y, 8);
        int blocks_z = std::min<int>((vec_K + threads.x - 1) / threads.x, 4);
        dim3 blocks(blocks_x, world_size, blocks_z);
        
        pull_a_kernel_vec<<<blocks, threads, 0, stream>>>(
            peer_ptrs, out_ptr, start_row, num_rows, K_local, world_size
        );
    } else {
        dim3 threads(32, 8);
        int blocks_x = std::min<int>((num_rows + threads.y - 1) / threads.y, 8);
        int blocks_z = std::min<int>((K_local + threads.x - 1) / threads.x, 16);
        dim3 blocks(blocks_x, world_size, blocks_z);
        
        pull_a_kernel_scalar<<<blocks, threads, 0, stream>>>(
            peer_ptrs, out_ptr, start_row, num_rows, K_local, world_size
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pull_a", &launch_pull_a, "Pull A chunks from peers over NVLink");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("pull_a_allgather_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(shape, dtype, device):
    key = (shape, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    buf = symm_mem.empty(shape, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    res = (buf, hdl, ptrs)
    _symm_cache[key] = res
    return res


@torch.no_grad()
def solution(
    A_local: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    if A_local.dtype != torch.bfloat16:
        # Fallback to reference NCCL implementation for arbitrary unsupported types
        world_size = dist.get_world_size()
        M, K_local = A_local.shape
        K_global = world_size * K_local
        
        A_local_t = A_local.transpose(0, 1).contiguous()
        A_t_gather = torch.empty((world_size, K_local, M), device=A_local.device, dtype=A_local.dtype)
        dist.all_gather_into_tensor(A_t_gather, A_local_t)
        A_global_t = A_t_gather.reshape(K_global, M)
        
        B = B.contiguous()
        C_t = torch.matmul(B.transpose(0, 1), A_global_t)
        return C_t.transpose(0, 1)

    global _ext
    if _ext is None:
        if dist.get_rank() == 0:
            _get_ext()
        dist.barrier()
        
    world_size = dist.get_world_size()
    M, K_local = A_local.shape
    K_global = world_size * K_local
    N = B.shape[1]
    
    A_local = A_local.contiguous()
    B = B.contiguous()
    
    # Expose current rank's input slice via symm_mem
    buf, hdl, peer_ptrs = _get_symm_state(A_local.shape, A_local.dtype, A_local.device)
    buf.copy_(A_local)
    
    # Determine optimized pipeline chunk size (prioritizing max tensor-core efficiency limits)
    chunk_size = 2048
    if M <= chunk_size:
        num_chunks = 1
        chunk_size = M
    else:
        num_chunks = (M + chunk_size - 1) // chunk_size
        if num_chunks > 8:
            num_chunks = 8
            chunk_size = (M + num_chunks - 1) // num_chunks
            
    # Double buffers for pipeline overlap
    A_global_0 = torch.empty((chunk_size, K_global), device=A_local.device, dtype=A_local.dtype)
    A_global_1 = torch.empty((chunk_size, K_global), device=A_local.device, dtype=A_local.dtype)
    C = torch.empty((M, N), device=A_local.device, dtype=A_local.dtype)
    
    stream_0 = torch.cuda.Stream()
    stream_1 = torch.cuda.Stream()
    main_stream = torch.cuda.current_stream()
    
    # Await device-side exposure of remote symm_mems before streams begin NVLink reading
    hdl.barrier(channel=0)
    stream_0.wait_stream(main_stream)
    stream_1.wait_stream(main_stream)
    
    for i in range(num_chunks):
        start_row = i * chunk_size
        end_row = min(start_row + chunk_size, M)
        actual_M = end_row - start_row
        
        is_even = (i % 2 == 0)
        stream_curr = stream_0 if is_even else stream_1
        buf_curr = A_global_0 if is_even else A_global_1
        
        if i == 0:
            with torch.cuda.stream(stream_curr):
                _get_ext().launch_pull_a(
                    peer_ptrs, buf_curr, start_row, actual_M, K_local, world_size, stream_curr.cuda_stream
                )
                
        # Tensor core cuBLAS execution handles chunk i (same stream naturally enforces intra-chunk dependency)
        with torch.cuda.stream(stream_curr):
            out_slice = C[start_row:end_row]
            torch.mm(buf_curr[:actual_M], B, out=out_slice)
            
        # Push the NEXT chunk's NVLink memory pull to the alternate, concurrently executing stream
        if i + 1 < num_chunks:
            next_start = (i + 1) * chunk_size
            next_actual = min(next_start + chunk_size, M) - next_start
            stream_next = stream_1 if is_even else stream_0
            buf_next = A_global_1 if is_even else A_global_0
            
            with torch.cuda.stream(stream_next):
                _get_ext().launch_pull_a(
                    peer_ptrs, buf_next, next_start, next_actual, K_local, world_size, stream_next.cuda_stream
                )

    main_stream.wait_stream(stream_0)
    main_stream.wait_stream(stream_1)
    
    # Ensure no rank destructs/re-enters and alters `buf` while a peer continues chunk retrieval
    hdl.barrier(channel=0)
    
    return C