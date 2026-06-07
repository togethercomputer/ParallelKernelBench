import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import List, Optional

from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <vector>

static constexpr int THREADS = 256;
static constexpr int MAX_PARTIAL_BLOCKS = 4096;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, mask);
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    return v;
}

__global__ void set_zero_kernel(float* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 0.0f;
}

__global__ void copy_scalar_kernel(const float* __restrict__ src, float* __restrict__ dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[0] = src[0];
}

__global__ void partial_sum_bf16_kernel(
    __nv_bfloat16* __restrict__ data,
    int64_t n,
    float pre_scale,
    bool do_scale,
    float* __restrict__ scratch,
    int scratch_offset
) {
    float acc = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float x = __bfloat162float(data[i]);
        if (do_scale) {
            __nv_bfloat16 y = __float2bfloat16(x * pre_scale);
            data[i] = y;
            x = __bfloat162float(y);  // match BF16 in-place averaging before norm
        }
        acc += x * x;
    }

    acc = block_sum(acc);
    if (threadIdx.x == 0) scratch[scratch_offset + blockIdx.x] = acc;
}

__global__ void partial_sum_f32_kernel(
    float* __restrict__ data,
    int64_t n,
    float pre_scale,
    bool do_scale,
    float* __restrict__ scratch,
    int scratch_offset
) {
    float acc = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float x = data[i];
        if (do_scale) {
            x *= pre_scale;
            data[i] = x;
        }
        acc += x * x;
    }

    acc = block_sum(acc);
    if (threadIdx.x == 0) scratch[scratch_offset + blockIdx.x] = acc;
}

__global__ void reduce_scratch_kernel(
    const float* __restrict__ scratch,
    int n,
    float* __restrict__ out
) {
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        acc += scratch[i];
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) out[0] = acc;
}

__global__ void reduce_scalar_uva_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size
) {
    float acc = 0.0f;
    for (int r = threadIdx.x; r < world_size; r += blockDim.x) {
        const float* p = reinterpret_cast<const float*>(ptrs[r]);
        acc += p[0];
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) out[0] = acc;
}

__global__ void prepare_clip_kernel(
    const float* __restrict__ non_ep,
    const float* __restrict__ ep,
    float max_norm,
    float* __restrict__ total_out,
    float* __restrict__ coef_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float total = sqrtf(non_ep[0] + ep[0]);
        total_out[0] = total;
        coef_out[0] = (total > max_norm) ? (max_norm / total) : 1.0f;
    }
}

__global__ void scale_bf16_kernel(
    __nv_bfloat16* __restrict__ data,
    int64_t n,
    const float* __restrict__ coef
) {
    float c = coef[0];
    if (c == 1.0f) return;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        float x = __bfloat162float(data[i]);
        data[i] = __float2bfloat16(x * c);
    }
}

__global__ void scale_f32_kernel(
    float* __restrict__ data,
    int64_t n,
    const float* __restrict__ coef
) {
    float c = coef[0];
    if (c == 1.0f) return;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        data[i] *= c;
    }
}

static inline int blocks_for(int64_t n) {
    if (n <= 0) return 0;
    int64_t b = (n + THREADS - 1) / THREADS;
    if (b > MAX_PARTIAL_BLOCKS) b = MAX_PARTIAL_BLOCKS;
    return static_cast<int>(b);
}

void local_sum_list(
    std::vector<torch::Tensor> tensors,
    float pre_scale,
    bool do_scale,
    torch::Tensor scratch,
    torch::Tensor out
) {
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32, "out must be CUDA float32");
    TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == torch::kFloat32, "scratch must be CUDA float32");
    TORCH_CHECK(out.is_contiguous() && scratch.is_contiguous(), "out/scratch must be contiguous");

    int total_blocks = 0;
    for (auto& t : tensors) {
        TORCH_CHECK(t.is_cuda(), "all tensors must be CUDA");
        TORCH_CHECK(t.is_contiguous(), "all tensors must be contiguous");
        total_blocks += blocks_for(t.numel());
    }

    if (total_blocks == 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        set_zero_kernel<<<1, 1, 0, stream>>>(out.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    TORCH_CHECK(scratch.numel() >= total_blocks, "scratch too small");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int off = 0;
    for (auto& t : tensors) {
        int64_t n = t.numel();
        int b = blocks_for(n);
        if (b == 0) continue;

        if (t.scalar_type() == torch::kBFloat16) {
            auto* p = reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
            partial_sum_bf16_kernel<<<b, THREADS, 0, stream>>>(
                p, n, pre_scale, do_scale, scratch.data_ptr<float>(), off);
        } else if (t.scalar_type() == torch::kFloat32) {
            partial_sum_f32_kernel<<<b, THREADS, 0, stream>>>(
                t.data_ptr<float>(), n, pre_scale, do_scale, scratch.data_ptr<float>(), off);
        } else {
            TORCH_CHECK(false, "only bfloat16 and float32 tensors are supported");
        }
        off += b;
    }

    reduce_scratch_kernel<<<1, THREADS, 0, stream>>>(
        scratch.data_ptr<float>(), total_blocks, out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reduce_scalar_uva(torch::Tensor ptrs, torch::Tensor out) {
    TORCH_CHECK(ptrs.is_cuda() && ptrs.scalar_type() == torch::kInt64, "ptrs must be CUDA int64");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32, "out must be CUDA float32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_scalar_uva_kernel<<<1, THREADS, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs.data_ptr<int64_t>()),
        out.data_ptr<float>(),
        static_cast<int>(ptrs.numel()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_scalar(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA");
    TORCH_CHECK(src.scalar_type() == torch::kFloat32 && dst.scalar_type() == torch::kFloat32,
                "src/dst must be float32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    copy_scalar_kernel<<<1, 1, 0, stream>>>(src.data_ptr<float>(), dst.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare_clip(
    torch::Tensor non_ep,
    torch::Tensor ep,
    float max_norm,
    torch::Tensor total_out,
    torch::Tensor coef_out
) {
    TORCH_CHECK(non_ep.is_cuda() && ep.is_cuda() && total_out.is_cuda() && coef_out.is_cuda(),
                "all scalar tensors must be CUDA");
    TORCH_CHECK(non_ep.scalar_type() == torch::kFloat32 && ep.scalar_type() == torch::kFloat32 &&
                total_out.scalar_type() == torch::kFloat32 && coef_out.scalar_type() == torch::kFloat32,
                "all scalar tensors must be float32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    prepare_clip_kernel<<<1, 1, 0, stream>>>(
        non_ep.data_ptr<float>(),
        ep.data_ptr<float>(),
        max_norm,
        total_out.data_ptr<float>(),
        coef_out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scale_list(std::vector<torch::Tensor> tensors, torch::Tensor coef) {
    TORCH_CHECK(coef.is_cuda() && coef.scalar_type() == torch::kFloat32, "coef must be CUDA float32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    for (auto& t : tensors) {
        TORCH_CHECK(t.is_cuda(), "all tensors must be CUDA");
        TORCH_CHECK(t.is_contiguous(), "all tensors must be contiguous");
        int64_t n = t.numel();
        if (n == 0) continue;

        int blocks = static_cast<int>((n + THREADS - 1) / THREADS);
        if (blocks > 65535) blocks = 65535;

        if (t.scalar_type() == torch::kBFloat16) {
            auto* p = reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
            scale_bf16_kernel<<<blocks, THREADS, 0, stream>>>(p, n, coef.data_ptr<float>());
        } else if (t.scalar_type() == torch::kFloat32) {
            scale_f32_kernel<<<blocks, THREADS, 0, stream>>>(t.data_ptr<float>(), n, coef.data_ptr<float>());
        } else {
            TORCH_CHECK(false, "only bfloat16 and float32 tensors are supported");
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("local_sum_list", &local_sum_list, "BF16/F32 local L2 sum with optional in-place prescale");
    m.def("reduce_scalar_uva", &reduce_scalar_uva, "Symmetric-memory UVA scalar sum");
    m.def("copy_scalar", &copy_scalar, "Device scalar copy");
    m.def("prepare_clip", &prepare_clip, "Compute total norm and clipping coefficient");
    m.def("scale_list", &scale_list, "In-place list scale by device scalar");
}
'''


_ext = None
_scratch_cache = {}
_scalar_cache = {}
_stream_cache = {}
_reduce_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("clip_grad_norm_ep_symm_bf16_h100_ext", CUDA_SRC)
    return _ext


def _live_tensors(xs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x for x in xs if x is not None]


def _infer_device(*lists: List[torch.Tensor]) -> torch.device:
    for xs in lists:
        for t in xs:
            if t is not None:
                return t.device
    return torch.device("cuda", torch.cuda.current_device())


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _required_blocks(xs: List[torch.Tensor]) -> int:
    total = 0
    for t in xs:
        if t is None:
            continue
        n = int(t.numel())
        if n > 0:
            total += min((n + 255) // 256, 4096)
    return total


def _get_scratch(name: str, xs: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    need = max(1, _required_blocks(xs))
    key = (name, _device_index(device))
    old = _scratch_cache.get(key)
    if old is None or old.numel() < need:
        old = torch.empty(need, device=device, dtype=torch.float32)
        _scratch_cache[key] = old
    return old


def _get_scalar(name: str, device: torch.device) -> torch.Tensor:
    key = (name, _device_index(device))
    s = _scalar_cache.get(key)
    if s is None:
        s = torch.empty((), device=device, dtype=torch.float32)
        _scalar_cache[key] = s
    return s


def _get_streams(device: torch.device):
    key = _device_index(device)
    pair = _stream_cache.get(key)
    if pair is None:
        with torch.cuda.device(device):
            pair = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
        _stream_cache[key] = pair
    return pair


def _is_non_member_group(group) -> bool:
    gm = getattr(dist, "GroupMember", None)
    return gm is not None and group is getattr(gm, "NON_GROUP_MEMBER", object())


def _get_reduce_state(group, device: torch.device):
    key = (id(group), _device_index(device))
    st = _reduce_cache.get(key)
    if st is not None:
        return st

    buf = symm_mem.empty(1, device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, group)
    ptrs = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)
    st = (buf, hdl, ptrs, group)
    _reduce_cache[key] = st
    return st


def _reduce_scalar(val: torch.Tensor, group) -> torch.Tensor:
    if group is None or (not dist.is_initialized()) or _is_non_member_group(group):
        return val

    ext = _get_ext()
    buf, hdl, ptrs, _ = _get_reduce_state(group, val.device)
    out = torch.empty((), device=val.device, dtype=torch.float32)

    ext.copy_scalar(val, buf)
    hdl.barrier(channel=0)
    ext.reduce_scalar_uva(ptrs, out)
    hdl.barrier(channel=0)
    return out


def _local_sum_pair(
    non_ep: List[torch.Tensor],
    ep: List[torch.Tensor],
    ep_size: int,
    device: torch.device,
):
    ext = _get_ext()
    non_ep = _live_tensors(non_ep)
    ep = _live_tensors(ep)

    non_out = _get_scalar("non_local", device)
    ep_out = _get_scalar("ep_local", device)
    non_scratch = _get_scratch("non_scratch", non_ep, device)
    ep_scratch = _get_scratch("ep_scratch", ep, device)

    cur = torch.cuda.current_stream(device)
    s_non, s_ep = _get_streams(device)

    s_non.wait_stream(cur)
    s_ep.wait_stream(cur)

    with torch.cuda.stream(s_non):
        ext.local_sum_list(non_ep, 1.0, False, non_scratch, non_out)

    ep_scale = 1.0 / float(ep_size) if ep_size > 1 else 1.0
    ep_do_scale = bool(ep_size > 1 and len(ep) > 0)
    with torch.cuda.stream(s_ep):
        ext.local_sum_list(ep, float(ep_scale), ep_do_scale, ep_scratch, ep_out)

    cur.wait_stream(s_non)
    cur.wait_stream(s_ep)
    return non_out, ep_out


@torch.no_grad()
def solution(
    non_ep_grad_tensors: List[torch.Tensor],
    ep_grad_tensors: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    ep_size: int = 1,
    fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_fsdp_group: Optional[dist.ProcessGroup] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    assert float(norm_type) == 2.0, "optimized path implements L2 clip_grad_norm only"

    device = _infer_device(non_ep_grad_tensors, ep_grad_tensors)
    ext = _get_ext()

    non_ep = _live_tensors(non_ep_grad_tensors)
    ep = _live_tensors(ep_grad_tensors)

    # Local BF16/F32 sum-of-squares. EP averaging is fused into the EP local pass.
    non_ep_total, ep_total = _local_sum_pair(non_ep, ep, int(ep_size), device)

    # Replace NCCL all_reduce chains with symmetric-memory scalar reductions.
    non_ep_total = _reduce_scalar(non_ep_total, fsdp_group)

    ep_total = _reduce_scalar(ep_total, ep_fsdp_group)
    ep_total = _reduce_scalar(ep_total, ep_group)

    # Device-side total norm + coefficient; no host scalar sync.
    total_norm = torch.empty((), device=device, dtype=torch.float32)
    coef = _get_scalar("clip_coef", device)
    ext.prepare_clip(non_ep_total, ep_total, float(max_norm), total_norm, coef)

    # In-place clipping of both parameter classes, driven by the device coefficient.
    if non_ep:
        ext.scale_list(non_ep, coef)
    if ep:
        ext.scale_list(ep, coef)

    return total_norm