from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension
import triton
import triton.language as tl

# ==============================================================================
# Custom CUDA Collectives via Symmetric Memory
# ==============================================================================

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void symm_allreduce_sum_f32_kernel(
    float* __restrict__ local_buf,
    const int64_t* peer_ptrs,
    int world_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < world_size; i++) {
            const float* peer_ptr = reinterpret_cast<const float*>(peer_ptrs[i]);
            sum += peer_ptr[0];
        }
        local_buf[0] = sum;
    }
}

void symm_allreduce_sum_f32(
    torch::Tensor local_buf,
    torch::Tensor peer_ptrs,
    int world_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    symm_allreduce_sum_f32_kernel<<<1, 1, 0, stream>>>(
        local_buf.data_ptr<float>(),
        peer_ptrs.data_ptr<int64_t>(),
        world_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__global__ void symm_allgather_kernel(
    T* __restrict__ out_global,
    const int64_t* peer_ptrs,
    const int* offsets,
    const int* sizes,
    int world_size,
    int total_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_elements) {
        int rank = 0;
        // Identify which rank owns the element at `tid`
        for (int i = 1; i < world_size; i++) {
            if (tid >= offsets[i]) {
                rank = i;
            }
        }
        int local_idx = tid - offsets[rank];
        const T* peer_ptr = reinterpret_cast<const T*>(peer_ptrs[rank]);
        out_global[tid] = peer_ptr[local_idx];
    }
}

void symm_allgather(
    torch::Tensor out_global,
    torch::Tensor peer_ptrs,
    torch::Tensor offsets,
    torch::Tensor sizes,
    int world_size,
    int total_elements,
    int element_size
) {
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Dynamic dispatch depending on the precision bytes payload (e.g. 2 bytes for BF16/FP16)
    if (element_size == 2) {
        symm_allgather_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<uint16_t*>(out_global.data_ptr()),
            peer_ptrs.data_ptr<int64_t>(),
            offsets.data_ptr<int>(),
            sizes.data_ptr<int>(),
            world_size, total_elements
        );
    } else if (element_size == 4) {
        symm_allgather_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float*>(out_global.data_ptr()),
            peer_ptrs.data_ptr<int64_t>(),
            offsets.data_ptr<int>(),
            sizes.data_ptr<int>(),
            world_size, total_elements
        );
    } else if (element_size == 8) {
        symm_allgather_kernel<double><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<double*>(out_global.data_ptr()),
            peer_ptrs.data_ptr<int64_t>(),
            offsets.data_ptr<int>(),
            sizes.data_ptr<int>(),
            world_size, total_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype size for UVA gather.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symm_allreduce_sum_f32", &symm_allreduce_sum_f32, "UVA rank sum reduce");
    m.def("symm_allgather", &symm_allgather, "UVA global flat tensor gather");
}
'''

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("deepmd_kalman_ext", CUDA_SRC)
    return _ext

# ==============================================================================
# Memory Manager & Workspace Cache
# ==============================================================================

_workspace_cache = None

class Workspace:
    def __init__(self, local_shape, dtype, device):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_shape = local_shape
        self.local_total = sum(local_shape)
        
        # 1. Gather all local totals globally to precalculate buffer offsets
        local_total_t = torch.tensor([self.local_total], dtype=torch.int64, device=device)
        total_tensors = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(self.world_size)]
        dist.all_gather(total_tensors, local_total_t)
        self.world_totals = [t.item() for t in total_tensors]
        self.global_total = sum(self.world_totals)
        
        # 2. Gather structural lists
        self.shape_list = [None] * self.world_size
        dist.all_gather_object(self.shape_list, local_shape)
        
        # 3. Form offsets for the C++ collective kernel
        self.offsets = [0] * self.world_size
        for i in range(1, self.world_size):
            self.offsets[i] = self.offsets[i-1] + self.world_totals[i-1]
            
        self.offsets_tensor = torch.tensor(self.offsets, dtype=torch.int32, device=device)
        self.sizes_tensor = torch.tensor(self.world_totals, dtype=torch.int32, device=device)
        
        # 4. Allocate Symmetric Memory for 32-bit float reductions
        self.tmp_buf = symm_mem.empty(1, dtype=torch.float32, device=device)
        self.tmp_hdl = symm_mem.rendezvous(self.tmp_buf, dist.group.WORLD)
        self.tmp_ptrs_tensor = torch.tensor([int(p) for p in self.tmp_hdl.buffer_ptrs], dtype=torch.int64, device=device)
        
        # 5. Allocate Symmetric Memory for local chunk bfloat16 gathering
        self.weight_buf = symm_mem.empty(self.local_total, dtype=dtype, device=device)
        self.weight_hdl = symm_mem.rendezvous(self.weight_buf, dist.group.WORLD)
        self.weight_ptrs_tensor = torch.tensor([int(p) for p in self.weight_hdl.buffer_ptrs], dtype=torch.int64, device=device)
        
        self.global_weight_buf = torch.empty(self.global_total, dtype=dtype, device=device)
        self.ext = _get_ext()

def get_workspace(local_shape, dtype, device):
    global _workspace_cache
    if _workspace_cache is not None and _workspace_cache.local_shape == local_shape:
        return _workspace_cache
    _workspace_cache = Workspace(local_shape, dtype, device)
    return _workspace_cache

# ==============================================================================
# Fused Triton Kernels for Math Operations
# ==============================================================================

@triton.jit
def update_P_kernel(
    P_ptr, K_ptr, A_ptr, lam, n,
    stride_P_row, stride_P_col,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    
    A = tl.load(A_ptr).to(tl.float32)
    inv_lam = 1.0 / lam
    
    row_offsets = row_pid * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    col_offsets = col_pid * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    
    row_mask = row_offsets < n
    col_mask = col_offsets < n
    
    K_row = tl.load(K_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    K_col = tl.load(K_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    
    P_ptrs = P_ptr + (row_offsets[:, None] * stride_P_row + col_offsets[None, :] * stride_P_col)
    mask_2d = row_mask[:, None] & col_mask[None, :]
    
    P_orig = tl.load(P_ptrs, mask=mask_2d, other=0.0)
    
    # Outer product computed implicitly inside GPU registers, negating expensive external storage
    outer = K_row[:, None] * K_col[None, :]
    new_P = inv_lam * (P_orig.to(tl.float32) - A * outer)
    
    tl.store(P_ptrs, new_P.to(P_orig.dtype), mask=mask_2d)

@triton.jit
def update_weights_kernel(
    weights_ptr, K_ptr, A_ptr, err_ptr, n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    A = tl.load(A_ptr).to(tl.float32)
    err = tl.load(err_ptr).to(tl.float32)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    w_orig = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    k = tl.load(K_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    new_w = w_orig.to(tl.float32) + A * err * k
    tl.store(weights_ptr + offsets, new_w.to(w_orig.dtype), mask=mask)

# ==============================================================================
# Main Optimizer Update Target
# ==============================================================================

@torch.no_grad()
def solution(
    H: List[torch.Tensor],
    error: torch.Tensor,
    weights: List[torch.Tensor],
    P: List[torch.Tensor],
    kalman_lambda: float,
    kalman_nue: float = 0.9987,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:

    weights_num = len(weights)
    device = weights[0].device
    dtype = weights[0].dtype
    lam = kalman_lambda
    
    if dist.is_initialized():
        if dist.get_rank() == 0:
            _get_ext()
        dist.barrier()
        
    K_list = []
    
    # Fully asynchronous GPU-side calculation to prevent CPU blocking operations
    tmp_tensor = torch.full((1,), lam * weights_num, dtype=torch.float32, device=device)
    
    for i in range(weights_num):
        # Using native fast cuBLAS queue execution for memory bound GEMV step
        K = torch.matmul(P[i], H[i])
        K_list.append(K)
        
        # Asynchronously accumulate flat trace / dot products locally on stream
        dot = torch.dot(H[i].view(-1).float(), K.view(-1).float())
        tmp_tensor.add_(dot)

    A_tensor = torch.empty(1, dtype=torch.float32, device=device)

    # Cross-rank Reduction of the global Denominator `tmp`
    if dist.is_initialized():
        local_shape = [w.shape[0] for w in weights]
        workspace = get_workspace(local_shape, dtype, device)
        
        workspace.tmp_buf.copy_(tmp_tensor)
        # CPU waits at this barrier while GPU completes heavy prior queued matmul tasks
        workspace.tmp_hdl.barrier(channel=0) 
        
        workspace.ext.symm_allreduce_sum_f32(
            workspace.tmp_buf, 
            workspace.tmp_ptrs_tensor, 
            workspace.world_size
        )
        A_tensor.copy_(1.0 / workspace.tmp_buf)
    else:
        A_tensor.copy_(1.0 / tmp_tensor)
        
    err_tensor = error.to(device=device, dtype=dtype)
    
    # Fused execution of Local Weights and Covariance blocks (replaces outer-products and multiple elementwise nodes)
    for i in range(weights_num):
        n = weights[i].shape[0]
        
        BLOCK_SIZE_ROW = 32
        BLOCK_SIZE_COL = 32
        grid_P = (triton.cdiv(n, BLOCK_SIZE_ROW), triton.cdiv(n, BLOCK_SIZE_COL))
        
        update_P_kernel[grid_P](
            P[i], K_list[i], A_tensor, lam, n,
            P[i].stride(0), P[i].stride(1),
            BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
            BLOCK_SIZE_COL=BLOCK_SIZE_COL
        )
        
        BLOCK_SIZE_W = 256
        grid_W = (triton.cdiv(n, BLOCK_SIZE_W),)
        
        update_weights_kernel[grid_W](
            weights[i], K_list[i], A_tensor, err_tensor, n,
            BLOCK_SIZE=BLOCK_SIZE_W
        )

    # Gather & distribute the fully updated Parameter blocks across all ranks
    if dist.is_initialized():
        flat_weights = torch.cat([w.view(-1) for w in weights], dim=0)
        workspace.weight_buf.copy_(flat_weights)
        
        # Further overlapping: CPU rests at this barrier while Triton kernels securely resolve the latest weights updates
        workspace.weight_hdl.barrier(channel=1)
        
        workspace.ext.symm_allgather(
            workspace.global_weight_buf,
            workspace.weight_ptrs_tensor,
            workspace.offsets_tensor,
            workspace.sizes_tensor,
            workspace.world_size,
            workspace.global_total,
            flat_weights.element_size()
        )
        
        result = []
        for i in range(workspace.world_size):
            rank_shapes = workspace.shape_list[i]
            start = workspace.offsets[i]
            end = start + workspace.world_totals[i]
            rank_tensor = workspace.global_weight_buf[start:end]
            
            splits = torch.split(rank_tensor, rank_shapes)
            for t in splits:
                # Disconnect explicit views ensuring memory safety against the buffer
                result.append(t.view(-1, 1).clone())
        weights = result

    # Decay Kalman explicitly on the device avoiding a synchronous `.to()` operation
    kalman_lambda_next = (
        torch.as_tensor(kalman_nue, dtype=dtype, device=device) * lam
        + 1.0
        - torch.as_tensor(kalman_nue, dtype=dtype, device=device)
    )

    return weights, P, kalman_lambda_next