import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension
import triton
import triton.language as tl

# Lightweight C++ extension to securely cast integer device pointers to torch.Tensor
# This allows passing peer pointers seamlessly to Triton without pointer hacking.
CUDA_SRC = r'''
#include <torch/extension.h>

torch::Tensor create_tensor_from_ptr(int64_t ptr, torch::Tensor dummy, c10::IntArrayRef sizes, c10::IntArrayRef strides) {
    auto options = dummy.options();
    return torch::from_blob(reinterpret_cast<void*>(ptr), sizes, strides, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_tensor_from_ptr", &create_tensor_from_ptr, "Create tensor from raw pointer");
}
'''

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("ptr_to_tensor_ext", CUDA_SRC)
    return _ext

_symm_cache = {}
def _get_symm_state(M, N, dtype, device):
    global _symm_cache
    key = (M, N, dtype, device)
    if key in _symm_cache:
        return _symm_cache[key]
    
    numel = M * N
    buf = symm_mem.empty(numel, dtype=dtype, device=device)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = buf.view(M, N)
    _symm_cache[key] = (out, hdl)
    return out, hdl

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N_local', 'K']
)
@triton.jit
def fused_gemm_scatter_kernel(
    a_ptr, b_ptr,
    c0_ptr, c1_ptr, c2_ptr, c3_ptr, c4_ptr, c5_ptr, c6_ptr, c7_ptr,
    M, N_local, K, rank_offset_n,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    world_size: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_local, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Trick: wrap dimensions with % M and % N_local to naturally avoid 
    # inner loop out-of-bounds masks while ensuring robustness.
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N_local
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Inner K loop: assumed padded to BLOCK_K multiples in Python to skip heavy masks
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Store out-of-bounds mask strictly required here
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N_local)
    offset_c = offs_cm[:, None] * stride_cm + (offs_cn[None, :] + rank_offset_n) * stride_cn

    # Direct remote NVLink Scatter: fully hidden within the latency of the schedule.
    if world_size >= 1: tl.store(c0_ptr + offset_c, c, mask=c_mask)
    if world_size >= 2: tl.store(c1_ptr + offset_c, c, mask=c_mask)
    if world_size >= 3: tl.store(c2_ptr + offset_c, c, mask=c_mask)
    if world_size >= 4: tl.store(c3_ptr + offset_c, c, mask=c_mask)
    if world_size >= 5: tl.store(c4_ptr + offset_c, c, mask=c_mask)
    if world_size >= 6: tl.store(c5_ptr + offset_c, c, mask=c_mask)
    if world_size >= 7: tl.store(c6_ptr + offset_c, c, mask=c_mask)
    if world_size >= 8: tl.store(c7_ptr + offset_c, c, mask=c_mask)

@torch.no_grad()
def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    M, K_orig = A.shape
    _, N_local = B.shape
    
    if rank == 0:
        _get_ext()
    dist.barrier()
    
    # Target precision mapping
    if A.dtype != torch.bfloat16: A = A.to(torch.bfloat16)
    if B.dtype != torch.bfloat16: B = B.to(torch.bfloat16)

    # Pad K dimension so inner loops skip bounds masking -> drastically boosts triton ops
    PAD_K = 64
    if K_orig % PAD_K != 0:
        pad_len = PAD_K - (K_orig % PAD_K)
        A = torch.nn.functional.pad(A, (0, pad_len)).contiguous()
        B = torch.nn.functional.pad(B, (0, 0, 0, pad_len)).contiguous()
        K = K_orig + pad_len
    else:
        A = A.contiguous()
        B = B.contiguous()
        K = K_orig

    N = world_size * N_local
    # Retrieve symmetric memory for contiguous final C accumulation
    C, hdl = _get_symm_state(M, N, torch.bfloat16, A.device)
    
    # Flush global rendezvous readiness on this rank
    hdl.barrier(channel=0)
    
    ext = _get_ext()
    sizes, strides = C.shape, C.stride()
    peer_tensors = []
    
    for i in range(8):
        if i < world_size:
            ptr = int(hdl.buffer_ptrs[i])
            peer_tensors.append(ext.create_tensor_from_ptr(ptr, C, sizes, strides))
        else:
            # Padding handles any world_size without breaking triton arguments
            peer_tensors.append(C)
            
    rank_offset_n = rank * N_local
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N_local, META['BLOCK_N']),
    )

    if M > 0 and N_local > 0:
        fused_gemm_scatter_kernel[grid](
            A, B,
            peer_tensors[0], peer_tensors[1], peer_tensors[2], peer_tensors[3],
            peer_tensors[4], peer_tensors[5], peer_tensors[6], peer_tensors[7],
            M, N_local, K, rank_offset_n,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            world_size=world_size,
        )

    # 1. Block stream to ensure our SM writes to remote peers complete successfully
    torch.cuda.current_stream().synchronize()
    # 2. Block host ensuring peer stream cycles arrive over our bounds safely
    dist.barrier()
    
    return C