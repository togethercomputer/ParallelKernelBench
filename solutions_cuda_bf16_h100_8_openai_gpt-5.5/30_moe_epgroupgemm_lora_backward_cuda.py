from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int MAX_GRID_BLOCKS = 4;

// -----------------------------------------------------------------------------
// Device-side signal-pad barriers over symmetric memory.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void send_signal_release(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 0u);
}

__device__ __forceinline__ void wait_signal_acquire(uint32_t* addr) {
    uint32_t old;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1u);
}

__device__ __forceinline__ void blockwise_barrier(
    const uint64_t* __restrict__ signal_pad_ptrs,
    uint64_t block_id,
    int rank,
    int world_size
) {
    const int t = threadIdx.x;
    if (t < world_size) {
        const uint64_t local_base = signal_pad_ptrs[rank];
        const uint64_t remote_base = signal_pad_ptrs[t];

        const uint64_t send_off =
            ((block_id * (uint64_t)world_size) + (uint64_t)rank) * sizeof(uint32_t);
        const uint64_t wait_off =
            ((block_id * (uint64_t)world_size) + (uint64_t)t) * sizeof(uint32_t);

        uint32_t* send_addr = reinterpret_cast<uint32_t*>(remote_base + send_off);
        uint32_t* wait_addr = reinterpret_cast<uint32_t*>(local_base + wait_off);

        send_signal_release(send_addr);
        wait_signal_acquire(wait_addr);
    }
}

// -----------------------------------------------------------------------------
// Hopper NVSwitch multimem BF16x8 reduce.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void multimem_ld_reduce_bf16x8(
    const uint64_t* addr,
    uint32_t& r0,
    uint32_t& r1,
    uint32_t& r2,
    uint32_t& r3
) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(addr)
        : "memory");
}

__device__ __forceinline__ void store_bf16x8_packed(
    __nv_bfloat16* dst,
    uint32_t r0,
    uint32_t r1,
    uint32_t r2,
    uint32_t r3
) {
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
    d[0] = r0;
    d[1] = r1;
    d[2] = r2;
    d[3] = r3;
}

__device__ __forceinline__ void copy_bf16x8(
    const __nv_bfloat16* src,
    __nv_bfloat16* dst
) {
    const uint4 v = *reinterpret_cast<const uint4*>(src);
    *reinterpret_cast<uint4*>(dst) = v;
}

// -----------------------------------------------------------------------------
// BF16 fused 3-buffer all-reduce.
// use_multimem requires n1,n2,n3 and tensor data pointers to be 16B aligned.
// Fallback uses UVA peer loads from symmetric buffers.
// -----------------------------------------------------------------------------

__global__ void lora_allreduce_bf16_kernel(
    __nv_bfloat16* __restrict__ g1,
    __nv_bfloat16* __restrict__ g2,
    __nv_bfloat16* __restrict__ g3,
    __nv_bfloat16* __restrict__ symm,
    const uint64_t* __restrict__ signal_pad_ptrs,
    const long long* __restrict__ peer_ptrs,
    uint64_t multicast_base,
    int64_t n1,
    int64_t n2,
    int64_t n3,
    int world_size,
    int rank,
    bool use_multimem
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int64_t grid_stride = (int64_t)gridDim.x * bdim;
    const int64_t linear = (int64_t)blockIdx.x * bdim + tid;
    const int64_t n12 = n1 + n2;
    const int64_t total = n12 + n3;

    if (use_multimem) {
        const int64_t c1 = n1 >> 3;
        const int64_t c2 = n2 >> 3;
        const int64_t c3 = n3 >> 3;
        const int64_t total_chunks = c1 + c2 + c3;

        // Pack this block's future reduce chunks into local symmetric memory.
        for (int64_t ck = linear; ck < total_chunks; ck += grid_stride) {
            if (ck < c1) {
                copy_bf16x8(g1 + (ck << 3), symm + (ck << 3));
            } else if (ck < c1 + c2) {
                const int64_t j = ck - c1;
                copy_bf16x8(g2 + (j << 3), symm + n1 + (j << 3));
            } else {
                const int64_t j = ck - c1 - c2;
                copy_bf16x8(g3 + (j << 3), symm + n12 + (j << 3));
            }
        }

        __syncthreads();
        blockwise_barrier(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
        __syncthreads();

        // In-switch BF16 SUM and write directly back to the original grad tensors.
        for (int64_t ck = linear; ck < total_chunks; ck += grid_stride) {
            int64_t elem_base;
            __nv_bfloat16* dst;

            if (ck < c1) {
                elem_base = ck << 3;
                dst = g1 + elem_base;
            } else if (ck < c1 + c2) {
                const int64_t j = ck - c1;
                elem_base = n1 + (j << 3);
                dst = g2 + (j << 3);
            } else {
                const int64_t j = ck - c1 - c2;
                elem_base = n12 + (j << 3);
                dst = g3 + (j << 3);
            }

            const int64_t chunk_global = elem_base >> 3;
            const uint64_t* mptr =
                reinterpret_cast<const uint64_t*>(multicast_base) + chunk_global * 2;

            uint32_t r0, r1, r2, r3;
            multimem_ld_reduce_bf16x8(mptr, r0, r1, r2, r3);
            store_bf16x8_packed(dst, r0, r1, r2, r3);
        }

        __syncthreads();
        blockwise_barrier(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
        __syncthreads();
        return;
    }

    // Generic BF16 peer-load path for arbitrary sizes/alignment.
    for (int64_t i = linear; i < total; i += grid_stride) {
        if (i < n1) {
            symm[i] = g1[i];
        } else if (i < n12) {
            symm[i] = g2[i - n1];
        } else {
            symm[i] = g3[i - n12];
        }
    }

    __syncthreads();
    blockwise_barrier(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
    __syncthreads();

    for (int64_t i = linear; i < total; i += grid_stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= world_size) break;
            const __nv_bfloat16* p =
                reinterpret_cast<const __nv_bfloat16*>((uintptr_t)peer_ptrs[r]);
            sum += __bfloat162float(p[i]);
        }

        const __nv_bfloat16 v = __float2bfloat16(sum);
        if (i < n1) {
            g1[i] = v;
        } else if (i < n12) {
            g2[i - n1] = v;
        } else {
            g3[i - n12] = v;
        }
    }

    __syncthreads();
    blockwise_barrier(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
    __syncthreads();
}

// -----------------------------------------------------------------------------
// FP32 fallback, still custom UVA + symmetric-memory, no NCCL.
// -----------------------------------------------------------------------------

__global__ void lora_allreduce_f32_kernel(
    float* __restrict__ g1,
    float* __restrict__ g2,
    float* __restrict__ g3,
    float* __restrict__ symm,
    const uint64_t* __restrict__ signal_pad_ptrs,
    const long long* __restrict__ peer_ptrs,
    int64_t n1,
    int64_t n2,
    int64_t n3,
    int world_size,
    int rank
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const int64_t linear = (int64_t)blockIdx.x * bdim + tid;
    const int64_t grid_stride = (int64_t)gridDim.x * bdim;
    const int64_t n12 = n1 + n2;
    const int64_t total = n12 + n3;

    for (int64_t i = linear; i < total; i += grid_stride) {
        if (i < n1) {
            symm[i] = g1[i];
        } else if (i < n12) {
            symm[i] = g2[i - n1];
        } else {
            symm[i] = g3[i - n12];
        }
    }

    __syncthreads();
    blockwise_barrier(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
    __syncthreads();

    for (int64_t i = linear; i < total; i += grid_stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r >= world_size) break;
            const float* p = reinterpret_cast<const float*>((uintptr_t)peer_ptrs[r]);
            sum += p[i];
        }

        if (i < n1) {
            g1[i] = sum;
        } else if (i < n12) {
            g2[i - n1] = sum;
        } else {
            g3[i - n12] = sum;
        }
    }

    __syncthreads();
    blockwise_barrier(signal_pad_ptrs, (uint64_t)blockIdx.x, rank, world_size);
    __syncthreads();
}

void launch_lora_allreduce_bf16(
    torch::Tensor g1,
    torch::Tensor g2,
    torch::Tensor g3,
    torch::Tensor symm,
    torch::Tensor signal_pad_ptrs_tensor,
    torch::Tensor peer_ptrs_tensor,
    uint64_t multicast_ptr,
    int64_t n1,
    int64_t n2,
    int64_t n3,
    int world_size,
    int rank,
    bool use_multimem,
    int num_blocks,
    int block_size
) {
    TORCH_CHECK(g1.is_cuda() && g2.is_cuda() && g3.is_cuda(), "grad tensors must be CUDA");
    TORCH_CHECK(symm.is_cuda(), "symmetric buffer must be CUDA");
    TORCH_CHECK(g1.scalar_type() == torch::kBFloat16, "g1 must be BF16");
    TORCH_CHECK(g2.scalar_type() == torch::kBFloat16, "g2 must be BF16");
    TORCH_CHECK(g3.scalar_type() == torch::kBFloat16, "g3 must be BF16");
    TORCH_CHECK(symm.scalar_type() == torch::kBFloat16, "symm must be BF16");

    const uint64_t* sig =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    const long long* ptrs =
        reinterpret_cast<const long long*>(peer_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    lora_allreduce_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(g1.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(g2.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(g3.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(symm.data_ptr<at::BFloat16>()),
        sig,
        ptrs,
        multicast_ptr,
        n1,
        n2,
        n3,
        world_size,
        rank,
        use_multimem
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_lora_allreduce_f32(
    torch::Tensor g1,
    torch::Tensor g2,
    torch::Tensor g3,
    torch::Tensor symm,
    torch::Tensor signal_pad_ptrs_tensor,
    torch::Tensor peer_ptrs_tensor,
    int64_t n1,
    int64_t n2,
    int64_t n3,
    int world_size,
    int rank,
    int num_blocks,
    int block_size
) {
    TORCH_CHECK(g1.is_cuda() && g2.is_cuda() && g3.is_cuda(), "grad tensors must be CUDA");
    TORCH_CHECK(symm.is_cuda(), "symmetric buffer must be CUDA");
    TORCH_CHECK(g1.scalar_type() == torch::kFloat32, "g1 must be FP32");
    TORCH_CHECK(g2.scalar_type() == torch::kFloat32, "g2 must be FP32");
    TORCH_CHECK(g3.scalar_type() == torch::kFloat32, "g3 must be FP32");
    TORCH_CHECK(symm.scalar_type() == torch::kFloat32, "symm must be FP32");

    const uint64_t* sig =
        reinterpret_cast<const uint64_t*>(signal_pad_ptrs_tensor.data_ptr<int64_t>());
    const long long* ptrs =
        reinterpret_cast<const long long*>(peer_ptrs_tensor.data_ptr<int64_t>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    lora_allreduce_f32_kernel<<<num_blocks, block_size, 0, stream>>>(
        g1.data_ptr<float>(),
        g2.data_ptr<float>(),
        g3.data_ptr<float>(),
        symm.data_ptr<float>(),
        sig,
        ptrs,
        n1,
        n2,
        n3,
        world_size,
        rank
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_lora_allreduce_bf16", &launch_lora_allreduce_bf16,
          "Fused 3-buffer LoRA all-reduce BF16 using symm_mem/UVA/multimem");
    m.def("launch_lora_allreduce_f32", &launch_lora_allreduce_f32,
          "Fused 3-buffer LoRA all-reduce FP32 using symm_mem/UVA");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_lora_ep_grad_sync_symm_cuda", CUDA_SRC)
    return _ext


_resource_cache = {}


def _resource_key(
    group: dist.ProcessGroup,
    grad_fc1_1_lora_A: torch.Tensor,
    grad_fc1_2_lora_A: torch.Tensor,
    grad_fc2_lora_B: torch.Tensor,
):
    return (
        id(group),
        grad_fc1_1_lora_A.device.index,
        grad_fc1_1_lora_A.dtype,
        tuple(grad_fc1_1_lora_A.shape),
        tuple(grad_fc1_2_lora_A.shape),
        tuple(grad_fc2_lora_B.shape),
    )


def _get_resources(
    group: dist.ProcessGroup,
    grad_fc1_1_lora_A: torch.Tensor,
    grad_fc1_2_lora_A: torch.Tensor,
    grad_fc2_lora_B: torch.Tensor,
):
    key = _resource_key(group, grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    total = (
        grad_fc1_1_lora_A.numel()
        + grad_fc1_2_lora_A.numel()
        + grad_fc2_lora_B.numel()
    )
    symm_buf = symm_mem.empty(
        total,
        device=grad_fc1_1_lora_A.device,
        dtype=grad_fc1_1_lora_A.dtype,
    )
    hdl = symm_mem.rendezvous(symm_buf, group)
    peer_ptrs = torch.tensor(hdl.buffer_ptrs, device=grad_fc1_1_lora_A.device, dtype=torch.int64)

    cached = (symm_buf, hdl, peer_ptrs)
    _resource_cache[key] = cached
    return cached


def _launch_config(total_elems: int, use_multimem: bool) -> Tuple[int, int]:
    block_size = 256
    work_items = (total_elems + 7) // 8 if use_multimem else total_elems
    num_blocks = max(1, min(4, (work_items + block_size - 1) // block_size))
    return num_blocks, block_size


@torch.no_grad()
def solution(
    grad_fc1_1_lora_A: torch.Tensor,
    grad_fc1_2_lora_A: torch.Tensor,
    grad_fc2_lora_B: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group = group or dist.group.WORLD

    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B

    assert grad_fc1_1_lora_A.is_cuda
    assert grad_fc1_2_lora_A.is_cuda
    assert grad_fc2_lora_B.is_cuda
    assert grad_fc1_1_lora_A.is_contiguous()
    assert grad_fc1_2_lora_A.is_contiguous()
    assert grad_fc2_lora_B.is_contiguous()
    assert grad_fc1_1_lora_A.device == grad_fc1_2_lora_A.device == grad_fc2_lora_B.device
    assert grad_fc1_1_lora_A.dtype == grad_fc1_2_lora_A.dtype == grad_fc2_lora_B.dtype
    assert grad_fc1_1_lora_A.dtype in (torch.bfloat16, torch.float32)

    ext = _get_ext()

    n1 = grad_fc1_1_lora_A.numel()
    n2 = grad_fc1_2_lora_A.numel()
    n3 = grad_fc2_lora_B.numel()
    total = n1 + n2 + n3

    symm_buf, hdl, peer_ptrs = _get_resources(
        group,
        grad_fc1_1_lora_A,
        grad_fc1_2_lora_A,
        grad_fc2_lora_B,
    )

    world_size = int(hdl.world_size)
    rank = int(hdl.rank)

    multicast_ptr = int(getattr(hdl, "multicast_ptr", 0) or 0)

    use_multimem = (
        grad_fc1_1_lora_A.dtype == torch.bfloat16
        and multicast_ptr != 0
        and (n1 % 8 == 0)
        and (n2 % 8 == 0)
        and (n3 % 8 == 0)
        and (grad_fc1_1_lora_A.data_ptr() % 16 == 0)
        and (grad_fc1_2_lora_A.data_ptr() % 16 == 0)
        and (grad_fc2_lora_B.data_ptr() % 16 == 0)
        and (symm_buf.data_ptr() % 16 == 0)
    )

    num_blocks, block_size = _launch_config(total, use_multimem)

    if grad_fc1_1_lora_A.dtype == torch.bfloat16:
        ext.launch_lora_allreduce_bf16(
            grad_fc1_1_lora_A,
            grad_fc1_2_lora_A,
            grad_fc2_lora_B,
            symm_buf,
            hdl.signal_pad_ptrs_dev,
            peer_ptrs,
            multicast_ptr,
            n1,
            n2,
            n3,
            world_size,
            rank,
            use_multimem,
            num_blocks,
            block_size,
        )
    else:
        ext.launch_lora_allreduce_f32(
            grad_fc1_1_lora_A,
            grad_fc1_2_lora_A,
            grad_fc2_lora_B,
            symm_buf,
            hdl.signal_pad_ptrs_dev,
            peer_ptrs,
            n1,
            n2,
            n3,
            world_size,
            rank,
            num_blocks,
            block_size,
        )

    return grad_fc1_1_lora_A, grad_fc1_2_lora_A, grad_fc2_lora_B