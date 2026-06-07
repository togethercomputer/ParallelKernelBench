import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
import triton
import triton.language as tl
from utils.cuda_helpers import compile_cuda_extension

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 128-bit aligned structure for 8x bfloat16 vectorization
struct alignas(16) bf16_8 {
    __nv_bfloat16 vals[8];
};

__global__ void pull_gather_dim0_kernel_vec(
    const int64_t* world_ptrs, int64_t offset,
    __nv_bfloat16* out,
    int n_fsdp, int n_tp, int tp_rank,
    int chunk_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int total_elements = chunk_elements * n_fsdp;
    if (idx < total_elements) {
        int peer_fsdp = idx / chunk_elements;
        int elem = idx % chunk_elements;
        int peer_world_rank = peer_fsdp * n_tp + tp_rank;
        
        const __nv_bfloat16* base_ptr = reinterpret_cast<const __nv_bfloat16*>(world_ptrs[peer_world_rank]);
        const bf16_8* peer_ptr = reinterpret_cast<const bf16_8*>(base_ptr + offset);
        
        bf16_8 vals = peer_ptr[elem / 8];
        reinterpret_cast<bf16_8*>(out)[idx / 8] = vals;
    }
}

__global__ void pull_gather_dim1_kernel_vec(
    const int64_t* world_ptrs, int64_t offset,
    __nv_bfloat16* out,
    int n_fsdp, int n_tp, int tp_rank,
    int R_shard, int C_shard
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int total_elements = R_shard * C_shard * n_fsdp;
    if (idx < total_elements) {
        int peer_fsdp = idx / (R_shard * C_shard);
        int elem = idx % (R_shard * C_shard);
        int r = elem / C_shard;
        int c = elem % C_shard;
        
        int peer_world_rank = peer_fsdp * n_tp + tp_rank;
        const __nv_bfloat16* base_ptr = reinterpret_cast<const __nv_bfloat16*>(world_ptrs[peer_world_rank]);
        const bf16_8* peer_ptr = reinterpret_cast<const bf16_8*>(base_ptr + offset);
        
        bf16_8 vals = peer_ptr[elem / 8];
        int out_idx = r * (n_fsdp * C_shard) + peer_fsdp * C_shard + c;
        reinterpret_cast<bf16_8*>(out)[out_idx / 8] = vals;
    }
}

__global__ void tp_allreduce_kernel_vec(
    const int64_t* world_ptrs, int64_t offset,
    __nv_bfloat16* out,
    int fsdp_rank, int n_tp,
    int num_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx < num_elements) {
        float sums[8] = {0.0f};
        for (int i = 0; i < n_tp; ++i) {
            int peer_world_rank = fsdp_rank * n_tp + i;
            const __nv_bfloat16* base_ptr = reinterpret_cast<const __nv_bfloat16*>(world_ptrs[peer_world_rank]);
            const bf16_8* peer_ptr = reinterpret_cast<const bf16_8*>(base_ptr + offset);
            
            bf16_8 vals = peer_ptr[idx / 8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sums[j] += __bfloat162float(vals.vals[j]);
            }
        }
        bf16_8 out_vals;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out_vals.vals[j] = __float2bfloat16(sums[j]);
        }
        reinterpret_cast<bf16_8*>(out)[idx / 8] = out_vals;
    }
}

void pull_gather_dim0(
    torch::Tensor world_ptrs, int64_t offset,
    torch::Tensor out,
    int n_fsdp, int n_tp, int tp_rank, int chunk_elements
) {
    TORCH_CHECK(chunk_elements % 8 == 0, "chunk_elements must be multiple of 8");
    int total_elements = chunk_elements * n_fsdp;
    int threads = 256;
    int blocks = (total_elements / 8 + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    pull_gather_dim0_kernel_vec<<<blocks, threads, 0, stream>>>(
        world_ptrs.data_ptr<int64_t>(), offset,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        n_fsdp, n_tp, tp_rank, chunk_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pull_gather_dim1(
    torch::Tensor world_ptrs, int64_t offset,
    torch::Tensor out,
    int n_fsdp, int n_tp, int tp_rank, int R_shard, int C_shard
) {
    TORCH_CHECK(C_shard % 8 == 0, "C_shard must be multiple of 8");
    int total_elements = R_shard * C_shard * n_fsdp;
    int threads = 256;
    int blocks = (total_elements / 8 + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    pull_gather_dim1_kernel_vec<<<blocks, threads, 0, stream>>>(
        world_ptrs.data_ptr<int64_t>(), offset,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        n_fsdp, n_tp, tp_rank, R_shard, C_shard
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tp_allreduce(
    torch::Tensor world_ptrs, int64_t offset,
    torch::Tensor out,
    int fsdp_rank, int n_tp, int num_elements
) {
    TORCH_CHECK(num_elements % 8 == 0, "num_elements must be multiple of 8");
    int threads = 256;
    int blocks = (num_elements / 8 + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    tp_allreduce_kernel_vec<<<blocks, threads, 0, stream>>>(
        world_ptrs.data_ptr<int64_t>(), offset,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        fsdp_rank, n_tp, num_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pull_gather_dim0", &pull_gather_dim0, "Pull gather dim 0");
    m.def("pull_gather_dim1", &pull_gather_dim1, "Pull gather dim 1");
    m.def("tp_allreduce", &tp_allreduce, "TP allreduce");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("fsdp_tp_opt_ext", CUDA_SRC)
    return _ext

@triton.jit
def swiglu_kernel(
    x1_ptr, x2_ptr, z_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x1 = tl.load(x1_ptr + offsets, mask=mask)
    x2 = tl.load(x2_ptr + offsets, mask=mask)
    
    x1_f32 = x1.to(tl.float32)
    x2_f32 = x2.to(tl.float32)
    
    silu = x1_f32 * tl.sigmoid(x1_f32)
    z = silu * x2_f32
    
    tl.store(z_ptr + offsets, z.to(x1.dtype), mask=mask)


_symm_cache = {}

def _get_symm_state(x_shape, w1_shape, w2_shape, w3_shape, n_fsdp, n_tp, device):
    global _symm_cache
    key = (tuple(x_shape), tuple(w1_shape), tuple(w2_shape), tuple(w3_shape), n_fsdp, n_tp, device)
    if key in _symm_cache:
        return _symm_cache[key]
        
    D_shard, D_FF_TP = w1_shape
    D = D_shard * n_fsdp
    
    size_w1 = w1_shape[0] * w1_shape[1]
    size_w2 = w2_shape[0] * w2_shape[1]
    size_w3 = w3_shape[0] * w3_shape[1]
    size_y = x_shape[0] * x_shape[1]
    
    total_size = size_w1 + size_w2 + size_w3 + size_y
    
    buf = symm_mem.empty(total_size, device=device, dtype=torch.bfloat16)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, dtype=torch.int64, device=device)
    
    W1_gathered = torch.empty((D, D_FF_TP), device=device, dtype=torch.bfloat16)
    W2_gathered = torch.empty((D, D_FF_TP), device=device, dtype=torch.bfloat16)
    W3_gathered = torch.empty((D_FF_TP, D), device=device, dtype=torch.bfloat16)
    z_buf = torch.empty((x_shape[0], D_FF_TP), device=device, dtype=torch.bfloat16)
    y_out = torch.empty((x_shape[0], D), device=device, dtype=torch.bfloat16)
    
    state = {
        "buf": buf,
        "hdl": hdl,
        "ptrs": ptrs_tensor,
        "W1_gathered": W1_gathered,
        "W2_gathered": W2_gathered,
        "W3_gathered": W3_gathered,
        "z_buf": z_buf,
        "y_out": y_out,
        "comm_stream": torch.cuda.Stream(device=device),
        "offsets": (0, size_w1, size_w1 + size_w2, size_w1 + size_w2 + size_w3)
    }
    _symm_cache[key] = state
    return state


@torch.no_grad()
def solution(
    x_local: torch.Tensor,
    W1_shard: torch.Tensor,
    W2_shard: torch.Tensor,
    W3_shard: torch.Tensor,
    n_tp: int,
    n_fsdp: int,
) -> torch.Tensor:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    ext = _get_ext()
    
    fsdp_rank = rank // n_tp
    tp_rank = rank % n_tp
    
    state = _get_symm_state(
        x_local.shape, W1_shard.shape, W2_shard.shape, W3_shard.shape, 
        n_fsdp, n_tp, x_local.device
    )
    
    buf = state["buf"]
    hdl = state["hdl"]
    ptrs = state["ptrs"]
    W1_gathered = state["W1_gathered"]
    W2_gathered = state["W2_gathered"]
    W3_gathered = state["W3_gathered"]
    z = state["z_buf"]
    y_out = state["y_out"]
    comm_stream = state["comm_stream"]
    off_w1, off_w2, off_w3, off_y = state["offsets"]
    
    comp_stream = torch.cuda.current_stream()
    
    # 1. Publish all FSDP Shards to the symmetric memory buffer directly
    buf[off_w1 : off_w2].view(-1).copy_(W1_shard.view(-1))
    buf[off_w2 : off_w3].view(-1).copy_(W2_shard.view(-1))
    buf[off_w3 : off_y].view(-1).copy_(W3_shard.view(-1))
    
    # Fast device-side barrier to ensure local copies are visible globally
    hdl.barrier(channel=0)
    
    # Sync the comm_stream to not start Pull Gathers before the barrier is clear
    comm_stream.wait_stream(comp_stream)
    
    # 2. Overlap Gather for W1 and W2 in the background stream
    with torch.cuda.stream(comm_stream):
        ext.pull_gather_dim0(ptrs, off_w1, W1_gathered, n_fsdp, n_tp, tp_rank, W1_shard.numel())
        ext.pull_gather_dim0(ptrs, off_w2, W2_gathered, n_fsdp, n_tp, tp_rank, W2_shard.numel())
        
    # Main stream waits ONLY for W1 and W2 to be fully gathered
    comp_stream.wait_stream(comm_stream)
    
    # 3. Immediately trigger Gather for W3 in the background to overlap with compute
    with torch.cuda.stream(comm_stream):
        ext.pull_gather_dim1(ptrs, off_w3, W3_gathered, n_fsdp, n_tp, tp_rank, W3_shard.shape[0], W3_shard.shape[1])
        
    # 4. Dense compute (Tensor Cores utilized heavily)
    x1 = torch.matmul(x_local, W1_gathered)
    x2 = torch.matmul(x_local, W2_gathered)
    
    # 5. Fused SwiGLU mapping
    BLOCK_SIZE = 256
    n_elements = z.numel()
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    swiglu_kernel[grid](x1, x2, z, n_elements, BLOCK_SIZE)
    
    # Main stream waits for W3 to finish gathering before the final projection
    comp_stream.wait_stream(comm_stream)
    y_partial = torch.matmul(z, W3_gathered)
    
    # 6. TP sum AllReduce directly using fused symmetric workspace
    buf[off_y :].view(-1).copy_(y_partial.view(-1))
    hdl.barrier(channel=1)
    
    ext.tp_allreduce(ptrs, off_y, y_out, fsdp_rank, n_tp, y_partial.numel())
    
    return y_out