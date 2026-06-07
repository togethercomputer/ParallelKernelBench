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
#include <vector>

template <typename T>
struct PtrArray {
    const T* ptrs[16];
    int count;
};

// Generic fallback kernel for non-vectorized tails or float32
template <typename T>
__global__ void reduce_scatter_fallback_kernel(
    PtrArray<T> arr,
    T* __restrict__ out,
    int64_t n_elements
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float sum = 0.0f;
        for (int r = 0; r < arr.count; ++r) {
            sum += static_cast<float>(arr.ptrs[r][idx]);
        }
        out[idx] = static_cast<T>(sum);
    }
}

// BFloat16 optimized kernel with uint4 (16 bytes = 8 bf16s) vectorization
__global__ void reduce_scatter_bf16_vec8_kernel(
    PtrArray<__nv_bfloat16> arr,
    __nv_bfloat16* __restrict__ out,
    int64_t n_vecs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vecs) {
        float2 sums[4];
        #pragma unroll
        for(int i=0; i<4; i++) { sums[i].x = 0.0f; sums[i].y = 0.0f; }
        
        for (int r = 0; r < arr.count; ++r) {
            const uint4* ptr = reinterpret_cast<const uint4*>(arr.ptrs[r]);
            uint4 val = ptr[idx];
            
            __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(&val);
            #pragma unroll
            for(int i=0; i<4; ++i) {
                float2 f2 = __bfloat1622float2(v2[i]);
                sums[i].x += f2.x;
                sums[i].y += f2.y;
            }
        }
        
        uint4 out_val;
        __nv_bfloat162* out_v2 = reinterpret_cast<__nv_bfloat162*>(&out_val);
        #pragma unroll
        for(int i=0; i<4; ++i) {
            out_v2[i] = __float22bfloat162_rn(sums[i]);
        }
        reinterpret_cast<uint4*>(out)[idx] = out_val;
    }
}

// Float16 optimized kernel with uint4 vectorization
__global__ void reduce_scatter_fp16_vec8_kernel(
    PtrArray<half> arr,
    half* __restrict__ out,
    int64_t n_vecs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vecs) {
        float2 sums[4];
        #pragma unroll
        for(int i=0; i<4; i++) { sums[i].x = 0.0f; sums[i].y = 0.0f; }
        
        for (int r = 0; r < arr.count; ++r) {
            const uint4* ptr = reinterpret_cast<const uint4*>(arr.ptrs[r]);
            uint4 val = ptr[idx];
            
            half2* v2 = reinterpret_cast<half2*>(&val);
            #pragma unroll
            for(int i=0; i<4; ++i) {
                float2 f2 = __half22float2(v2[i]);
                sums[i].x += f2.x;
                sums[i].y += f2.y;
            }
        }
        
        uint4 out_val;
        half2* out_v2 = reinterpret_cast<half2*>(&out_val);
        #pragma unroll
        for(int i=0; i<4; ++i) {
            out_v2[i] = __float22half2_rn(sums[i]);
        }
        reinterpret_cast<uint4*>(out)[idx] = out_val;
    }
}

void uva_reduce_scatter(
    std::vector<int64_t> remote_ptrs,
    torch::Tensor out,
    int64_t n_elements,
    int64_t stream_ptr
) {
    TORCH_CHECK(out.is_cuda(), "out must be CUDA tensor");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int threads = 256;

    if (out.dtype() == torch::kBFloat16) {
        PtrArray<__nv_bfloat16> arr;
        arr.count = remote_ptrs.size();
        for (int i = 0; i < arr.count; ++i) {
            arr.ptrs[i] = reinterpret_cast<const __nv_bfloat16*>(static_cast<uintptr_t>(remote_ptrs[i]));
        }
        
        if (n_elements % 8 == 0) {
            int64_t n_vecs = n_elements / 8;
            const int blocks = (n_vecs + threads - 1) / threads;
            reduce_scatter_bf16_vec8_kernel<<<blocks, threads, 0, stream>>>(
                arr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), n_vecs);
        } else {
            const int blocks = (n_elements + threads - 1) / threads;
            reduce_scatter_fallback_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                arr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), n_elements);
        }
    } else if (out.dtype() == torch::kHalf) {
        PtrArray<half> arr;
        arr.count = remote_ptrs.size();
        for (int i = 0; i < arr.count; ++i) {
            arr.ptrs[i] = reinterpret_cast<const half*>(static_cast<uintptr_t>(remote_ptrs[i]));
        }
        
        if (n_elements % 8 == 0) {
            int64_t n_vecs = n_elements / 8;
            const int blocks = (n_vecs + threads - 1) / threads;
            reduce_scatter_fp16_vec8_kernel<<<blocks, threads, 0, stream>>>(
                arr, reinterpret_cast<half*>(out.data_ptr<at::Half>()), n_vecs);
        } else {
            const int blocks = (n_elements + threads - 1) / threads;
            reduce_scatter_fallback_kernel<half><<<blocks, threads, 0, stream>>>(
                arr, reinterpret_cast<half*>(out.data_ptr<at::Half>()), n_elements);
        }
    } else if (out.dtype() == torch::kFloat32) {
        PtrArray<float> arr;
        arr.count = remote_ptrs.size();
        for (int i = 0; i < arr.count; ++i) {
            arr.ptrs[i] = reinterpret_cast<const float*>(static_cast<uintptr_t>(remote_ptrs[i]));
        }
        const int blocks = (n_elements + threads - 1) / threads;
        reduce_scatter_fallback_kernel<float><<<blocks, threads, 0, stream>>>(
            arr, reinterpret_cast<float*>(out.data_ptr<float>()), n_elements);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for uva_reduce_scatter");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("uva_reduce_scatter", &uva_reduce_scatter, "UVA reduce scatter supporting FP32, FP16, and BF16");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("uva_reduce_scatter_ext", CUDA_SRC)
    return _ext

_symm_cache = None
def _get_symm_state(n: int, dtype: torch.dtype, device: torch.device):
    global _symm_cache
    if _symm_cache is not None:
        c = _symm_cache
        if c["n"] == n and c["dtype"] == dtype and c["device"] == device:
            return c["buf"], c["hdl"]

    buf = symm_mem.empty(n, device=device, dtype=dtype)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    _symm_cache = {"n": n, "dtype": dtype, "device": device, "buf": buf, "hdl": hdl}
    return buf, hdl

_stream_cache = None
def _get_stream():
    global _stream_cache
    if _stream_cache is None:
        _stream_cache = torch.cuda.Stream()
    return _stream_cache

_event_cache = {}
def _get_event(name: str):
    global _event_cache
    if name not in _event_cache:
        _event_cache[name] = torch.cuda.Event()
    return _event_cache[name]

@torch.no_grad()
def solution(A_local: torch.Tensor, B_local: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B_local.is_cuda, "Inputs must be CUDA tensors"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    M, K_local = A_local.shape
    K_B, N = B_local.shape
    assert K_local == K_B, f"A_local and B_local must have matching K_local dimension: {K_local} != {K_B}"
    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
    
    # 1. Compile extension securely across ranks
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()

    # 2. Setup Symmetric Memory Cache
    M_local = M // world_size
    buf_shape = (world_size, M_local, N)
    buf_numel = world_size * M_local * N
    
    symm_buf, hdl = _get_symm_state(buf_numel, A_local.dtype, A_local.device)
    symm_buf = symm_buf.view(*buf_shape)
    
    C_local = torch.empty((M_local, N), dtype=A_local.dtype, device=A_local.device)
    A_local_chunks = A_local.contiguous().view(world_size, M_local, K_local)
    B_local = B_local.contiguous()

    # 3. Setup Concurrent Stream & Synchronization Context
    compute_stream = torch.cuda.current_stream()
    reduce_stream = _get_stream()
    compute_event = _get_event("compute")
    reduce_event = _get_event("reduce")

    # 4. Compute and P2P Reduce Overlap Pipelining
    # For every chunk 'c', compute locally, then barrier. Afterwards, the designated rank 'c'
    # asynchronously begins its NVLink-bound device-side UVA sum of peers over the `reduce_stream` 
    # whilst all ranks synchronously advance to computing chunk 'c+1' natively on the `compute_stream`.
    for c in range(world_size):
        torch.matmul(A_local_chunks[c], B_local, out=symm_buf[c])
        
        # Hardware device-side sync purely for the current step to guarantee symm_buf writes are visible
        hdl.barrier(channel=c)
        
        if rank == c:
            compute_event.record(compute_stream)
            reduce_stream.wait_event(compute_event)
            with torch.cuda.stream(reduce_stream):
                # Apply address translation via byte offset for this step's chunk pointer offsets
                offset_bytes = c * M_local * N * A_local.element_size()
                ptrs = [int(hdl.buffer_ptrs[r]) + offset_bytes for r in range(world_size)]
                
                # Execute Hopper-optimized UVA accumulation purely device side
                ext.uva_reduce_scatter(ptrs, C_local, M_local * N, reduce_stream.cuda_stream)

    # 5. Pipeline Draining and Global Protection Sink
    # Ensure this rank's compute stream fully awaits its overlapping reduction logic cleanly
    reduce_event.record(reduce_stream)
    compute_stream.wait_event(reduce_event)

    # Enforce global lock step synchronization explicitly avoiding subsequent calls overwriting our symm_buf early
    hdl.barrier(channel=world_size)
    
    return C_local