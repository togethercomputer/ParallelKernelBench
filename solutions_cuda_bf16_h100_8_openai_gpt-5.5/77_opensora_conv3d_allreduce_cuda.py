import math
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


# Strategy:
# - Compute local row-parallel Conv3d directly into a symmetric-memory output shard.
# - Use UVA peer pointers plus a small symmetric int32 signal pad for per-tile device-side barriers.
# - Fuse tile all-reduce and bias add into the Conv3d kernel: each persistent CTA computes a tile,
#   releases it to peers, waits for matching peer tiles, then sums peer symmetric buffers.
# - No NCCL / torch.distributed collectives are used on the hot path.

CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define TILE_ELEMS 256

// -----------------------------------------------------------------------------
// Device-side signal barrier over symmetric signal pads.
// signal layout on each rank: [num_tiles, world_size] int32
// For tile t, rank r sends to peer p by CAS(peer_signal[t, r], 0 -> 1).
// Then waits on local_signal[t, p] by CAS(1 -> 0), resetting for next call.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void send_signal_release(int* addr) {
    int old;
    do {
        asm volatile(
            "atom.global.release.sys.cas.b32 %0, [%1], 0, 1;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 0);
}

__device__ __forceinline__ void wait_signal_acquire(int* addr) {
    int old;
    do {
        asm volatile(
            "atom.global.acquire.sys.cas.b32 %0, [%1], 1, 0;"
            : "=r"(old)
            : "l"(addr)
            : "memory");
    } while (old != 1);
}

__device__ __forceinline__ void tile_barrier(
    const long long* __restrict__ signal_ptrs,
    int64_t tile_id,
    int rank,
    int world_size
) {
    const int tid = threadIdx.x;
    if (tid < world_size) {
        const int peer = tid;
        int* local_base = reinterpret_cast<int*>((uintptr_t)signal_ptrs[rank]);
        int* peer_base  = reinterpret_cast<int*>((uintptr_t)signal_ptrs[peer]);

        int* send_addr = peer_base + tile_id * (int64_t)world_size + rank;
        int* wait_addr = local_base + tile_id * (int64_t)world_size + peer;

        send_signal_release(send_addr);
        wait_signal_acquire(wait_addr);
    }
}

// -----------------------------------------------------------------------------
// Scalar conversion helpers.
// dtype_enum: 0 = bf16, 1 = f32, 2 = f16
// -----------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

template <typename scalar_t>
__device__ __forceinline__ float load_bias_value(
    const void* __restrict__ bias,
    int64_t co,
    int bias_dtype_enum
) {
    if (bias_dtype_enum < 0) {
        return 0.0f;
    }
    if (bias_dtype_enum == 1) {
        const float* b = reinterpret_cast<const float*>(bias);
        return b[co];
    }
    const scalar_t* b = reinterpret_cast<const scalar_t*>(bias);
    return to_float<scalar_t>(b[co]);
}

// -----------------------------------------------------------------------------
// Persistent tiled Conv3d + peer all-reduce + bias.
// One CTA owns one output tile at a time. It computes local partial conv into
// symmetric local_out, device-barriers with peers for that tile, then reads all
// peer symmetric local_out buffers through UVA pointers and writes final output.
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void conv3d_allreduce_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const void* __restrict__ bias,
    int bias_dtype_enum,
    scalar_t* __restrict__ local_out,
    scalar_t* __restrict__ final_out,
    const long long* __restrict__ out_ptrs,
    const long long* __restrict__ signal_ptrs,

    int64_t B,
    int64_t Cin_total,
    int64_t Ti,
    int64_t Hi,
    int64_t Wi,

    int64_t Cout,
    int64_t Cin_per_group,
    int64_t kT,
    int64_t kH,
    int64_t kW,

    int64_t To,
    int64_t Ho,
    int64_t Wo,

    int sT,
    int sH,
    int sW,
    int pT,
    int pH,
    int pW,
    int dT,
    int dH,
    int dW,
    int groups,

    int world_size,
    int rank,
    int64_t total_numel,
    int64_t num_tiles
) {
    for (int64_t tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        const int64_t tile_begin = tile * (int64_t)TILE_ELEMS;
        const int64_t tile_end = min(tile_begin + (int64_t)TILE_ELEMS, total_numel);

        // Local Conv3d partial.
        for (int64_t linear = tile_begin + threadIdx.x;
             linear < tile_end;
             linear += blockDim.x) {
            int64_t x = linear;

            const int64_t wo = x % Wo;
            x /= Wo;
            const int64_t ho = x % Ho;
            x /= Ho;
            const int64_t to = x % To;
            x /= To;
            const int64_t co = x % Cout;
            const int64_t b = x / Cout;

            const int64_t Cout_per_group = Cout / groups;
            const int64_t group_id = co / Cout_per_group;
            const int64_t cin_base = group_id * Cin_per_group;

            float acc = 0.0f;

            #pragma unroll 1
            for (int64_t ci = 0; ci < Cin_per_group; ++ci) {
                const int64_t in_c = cin_base + ci;

                #pragma unroll 1
                for (int64_t kt = 0; kt < kT; ++kt) {
                    const int64_t it = to * (int64_t)sT - (int64_t)pT + kt * (int64_t)dT;
                    if ((uint64_t)it >= (uint64_t)Ti) continue;

                    #pragma unroll 1
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        const int64_t ih = ho * (int64_t)sH - (int64_t)pH + kh * (int64_t)dH;
                        if ((uint64_t)ih >= (uint64_t)Hi) continue;

                        #pragma unroll 1
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            const int64_t iw = wo * (int64_t)sW - (int64_t)pW + kw * (int64_t)dW;
                            if ((uint64_t)iw >= (uint64_t)Wi) continue;

                            const int64_t in_idx =
                                (((b * Cin_total + in_c) * Ti + it) * Hi + ih) * Wi + iw;
                            const int64_t w_idx =
                                (((co * Cin_per_group + ci) * kT + kt) * kH + kh) * kW + kw;

                            acc += to_float<scalar_t>(input[in_idx]) *
                                   to_float<scalar_t>(weight[w_idx]);
                        }
                    }
                }
            }

            local_out[linear] = from_float<scalar_t>(acc);
        }

        __syncthreads();

        // Per-tile device-side cross-rank completion.
        tile_barrier(signal_ptrs, tile, rank, world_size);

        __syncthreads();

        // Peer UVA all-reduce + bias.
        for (int64_t linear = tile_begin + threadIdx.x;
             linear < tile_end;
             linear += blockDim.x) {
            float sum = 0.0f;

            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r >= world_size) break;
                const scalar_t* peer_out =
                    reinterpret_cast<const scalar_t*>((uintptr_t)out_ptrs[r]);
                sum += to_float<scalar_t>(peer_out[linear]);
            }

            int64_t tmp = linear / Wo;
            tmp /= Ho;
            tmp /= To;
            const int64_t co = tmp % Cout;

            sum += load_bias_value<scalar_t>(bias, co, bias_dtype_enum);
            final_out[linear] = from_float<scalar_t>(sum);
        }

        __syncthreads();
    }
}

void zero_i32(torch::Tensor t) {
    TORCH_CHECK(t.is_cuda(), "signal tensor must be CUDA");
    TORCH_CHECK(t.dtype() == torch::kInt32, "signal tensor must be int32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(t.data_ptr<int>(), 0, t.numel() * sizeof(int), stream);
}

void launch_conv3d_allreduce(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool has_bias,
    int dtype_enum,
    int bias_dtype_enum,
    torch::Tensor local_out,
    torch::Tensor final_out,
    torch::Tensor out_ptrs,
    torch::Tensor signal_ptrs,

    int64_t B,
    int64_t Cin_total,
    int64_t Ti,
    int64_t Hi,
    int64_t Wi,

    int64_t Cout,
    int64_t Cin_per_group,
    int64_t kT,
    int64_t kH,
    int64_t kW,

    int64_t To,
    int64_t Ho,
    int64_t Wo,

    int sT,
    int sH,
    int sW,
    int pT,
    int pH,
    int pW,
    int dT,
    int dH,
    int dW,
    int groups,

    int world_size,
    int rank,
    int64_t total_numel,
    int64_t num_tiles,
    int num_blocks
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "input/weight must be CUDA");
    TORCH_CHECK(local_out.is_cuda() && final_out.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(out_ptrs.is_cuda() && signal_ptrs.is_cuda(), "ptr tensors must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(local_out.is_contiguous() && final_out.is_contiguous(), "outputs must be contiguous");

    if (!has_bias) {
        bias_dtype_enum = -1;
    }

    const int threads = TILE_ELEMS;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const long long* out_ptrs_p =
        reinterpret_cast<const long long*>(out_ptrs.data_ptr<int64_t>());
    const long long* signal_ptrs_p =
        reinterpret_cast<const long long*>(signal_ptrs.data_ptr<int64_t>());

    if (dtype_enum == 0) {
        conv3d_allreduce_kernel<__nv_bfloat16><<<num_blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            has_bias ? bias.data_ptr() : nullptr,
            bias_dtype_enum,
            reinterpret_cast<__nv_bfloat16*>(local_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(final_out.data_ptr<at::BFloat16>()),
            out_ptrs_p,
            signal_ptrs_p,
            B, Cin_total, Ti, Hi, Wi,
            Cout, Cin_per_group, kT, kH, kW,
            To, Ho, Wo,
            sT, sH, sW, pT, pH, pW, dT, dH, dW, groups,
            world_size, rank, total_numel, num_tiles);
    } else if (dtype_enum == 1) {
        conv3d_allreduce_kernel<float><<<num_blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            has_bias ? bias.data_ptr() : nullptr,
            bias_dtype_enum,
            local_out.data_ptr<float>(),
            final_out.data_ptr<float>(),
            out_ptrs_p,
            signal_ptrs_p,
            B, Cin_total, Ti, Hi, Wi,
            Cout, Cin_per_group, kT, kH, kW,
            To, Ho, Wo,
            sT, sH, sW, pT, pH, pW, dT, dH, dW, groups,
            world_size, rank, total_numel, num_tiles);
    } else if (dtype_enum == 2) {
        conv3d_allreduce_kernel<__half><<<num_blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            has_bias ? bias.data_ptr() : nullptr,
            bias_dtype_enum,
            reinterpret_cast<__half*>(local_out.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(final_out.data_ptr<at::Half>()),
            out_ptrs_p,
            signal_ptrs_p,
            B, Cin_total, Ti, Hi, Wi,
            Cout, Cin_per_group, kT, kH, kW,
            To, Ho, Wo,
            sT, sH, sW, pT, pH, pW, dT, dH, dW, groups,
            world_size, rank, total_numel, num_tiles);
    } else {
        TORCH_CHECK(false, "unsupported dtype");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("zero_i32", &zero_i32, "async memset int32 signal pad to zero");
    m.def("launch_conv3d_allreduce", &launch_conv3d_allreduce,
          "fused Conv3d + symmetric-memory UVA all-reduce + bias");
}
'''


_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("opensora_conv3d_symm_uva_bf16_ext", CUDA_SRC)
    return _ext


def _to_3tuple(value: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    return (value, value, value) if isinstance(value, int) else value


def _output_shape(
    input_shape: torch.Size,
    out_channels: int,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
):
    b = input_shape[0]
    shape = [b, out_channels]
    for i, size in enumerate(input_shape[-3:]):
        out = size + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1
        shape.append(math.floor(out / stride[i] + 1))
    return tuple(shape)


def _dtype_enum(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float32:
        return 1
    if dtype == torch.float16:
        return 2
    raise TypeError(f"unsupported dtype for CUDA Conv3d all-reduce: {dtype}")


_resource_cache = {}


def _max_blocks(num_tiles: int, device: torch.device) -> int:
    if num_tiles <= 0:
        return 1
    props = torch.cuda.get_device_properties(device)
    # Persistent CTAs; one wave is enough to avoid block-scheduling deadlock while
    # keeping all SMs occupied. H100 SXM is typically 132 SMs.
    return max(1, min(num_tiles, props.multi_processor_count))


def _get_distributed_resources(
    out_shape,
    dtype: torch.dtype,
    device: torch.device,
    group,
    num_tiles: int,
):
    world_size = dist.get_world_size(group)
    key = (
        "dist",
        tuple(out_shape),
        dtype,
        device.index,
        id(group),
        world_size,
        num_tiles,
    )
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    ext = _get_ext()

    local_out = symm_mem.empty(out_shape, device=device, dtype=dtype)
    out_hdl = symm_mem.rendezvous(local_out, group)

    final_out = torch.empty(out_shape, device=device, dtype=dtype)
    out_ptrs = torch.tensor(out_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    signal = symm_mem.empty((num_tiles * world_size,), device=device, dtype=torch.int32)
    sig_hdl = symm_mem.rendezvous(signal, group)
    sig_ptrs = torch.tensor(sig_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    # Initial zeroing only; every tile wait CAS resets its signal slot to zero.
    ext.zero_i32(signal)
    sig_hdl.barrier(channel=0)

    cached = {
        "local_out": local_out,
        "final_out": final_out,
        "out_ptrs": out_ptrs,
        "signal": signal,
        "sig_ptrs": sig_ptrs,
        "rank": out_hdl.rank,
        "world_size": out_hdl.world_size,
    }
    _resource_cache[key] = cached
    return cached


def _get_single_rank_resources(
    out_shape,
    dtype: torch.dtype,
    device: torch.device,
    num_tiles: int,
):
    key = ("single", tuple(out_shape), dtype, device.index, num_tiles)
    cached = _resource_cache.get(key)
    if cached is not None:
        return cached

    ext = _get_ext()

    local_out = torch.empty(out_shape, device=device, dtype=dtype)
    final_out = torch.empty(out_shape, device=device, dtype=dtype)
    signal = torch.empty((num_tiles,), device=device, dtype=torch.int32)

    out_ptrs = torch.tensor([local_out.data_ptr()], device=device, dtype=torch.int64)
    sig_ptrs = torch.tensor([signal.data_ptr()], device=device, dtype=torch.int64)

    ext.zero_i32(signal)

    cached = {
        "local_out": local_out,
        "final_out": final_out,
        "out_ptrs": out_ptrs,
        "signal": signal,
        "sig_ptrs": sig_ptrs,
        "rank": 0,
        "world_size": 1,
    }
    _resource_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Union[int, Tuple[int, int, int]],
    padding: Union[int, Tuple[int, int, int]],
    dilation: Union[int, Tuple[int, int, int]],
    groups: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Row-parallel Conv3d forward with fused symmetric-memory all-reduce.

    Inputs:
      input:  [B, C_in_local, T, H, W], CUDA contiguous preferred
      weight: [C_out, C_in_per_group_local, kT, kH, kW]
      bias:   [C_out] or None

    Returns:
      replicated full output after SUM across tensor-parallel ranks and bias add.
    """
    assert input.is_cuda and weight.is_cuda, "input and weight must be CUDA tensors"
    assert input.dim() == 5 and weight.dim() == 5, "expected 5D input/weight"
    assert input.dtype == weight.dtype, "input and weight dtype must match"
    assert groups >= 1, "groups must be positive"

    dtype_enum = _dtype_enum(input.dtype)

    sT, sH, sW = _to_3tuple(stride)
    pT, pH, pW = _to_3tuple(padding)
    dT, dH, dW = _to_3tuple(dilation)

    x = input.contiguous()
    w = weight.contiguous()

    if bias is not None:
        assert bias.is_cuda, "bias must be CUDA"
        assert bias.numel() == weight.shape[0], "bias shape mismatch"
        if bias.dtype == torch.float32:
            bias_dtype_enum = 1
        else:
            assert bias.dtype == input.dtype, "bias must be input dtype or float32"
            bias_dtype_enum = dtype_enum
        b_arg = bias.contiguous()
        has_bias = True
    else:
        bias_dtype_enum = -1
        b_arg = torch.empty((0,), device=input.device, dtype=input.dtype)
        has_bias = False

    out_shape = _output_shape(
        x.shape,
        int(w.shape[0]),
        (int(w.shape[2]), int(w.shape[3]), int(w.shape[4])),
        (sT, sH, sW),
        (pT, pH, pW),
        (dT, dH, dW),
    )

    if any(dim < 0 for dim in out_shape):
        raise RuntimeError(f"invalid Conv3d output shape: {out_shape}")

    total_numel = math.prod(out_shape)
    if total_numel == 0:
        return torch.empty(out_shape, device=input.device, dtype=input.dtype)

    num_tiles = (total_numel + 255) // 256
    num_blocks = _max_blocks(num_tiles, input.device)

    if dist.is_initialized():
        pg = group if group is not None else dist.group.WORLD
        res = _get_distributed_resources(out_shape, input.dtype, input.device, pg, num_tiles)
    else:
        res = _get_single_rank_resources(out_shape, input.dtype, input.device, num_tiles)

    B = int(x.shape[0])
    Cin_total = int(x.shape[1])
    Ti, Hi, Wi = int(x.shape[2]), int(x.shape[3]), int(x.shape[4])

    Cout = int(w.shape[0])
    Cin_per_group = int(w.shape[1])
    kT, kH, kW = int(w.shape[2]), int(w.shape[3]), int(w.shape[4])

    To, Ho, Wo = int(out_shape[2]), int(out_shape[3]), int(out_shape[4])

    _get_ext().launch_conv3d_allreduce(
        x,
        w,
        b_arg,
        has_bias,
        dtype_enum,
        bias_dtype_enum,
        res["local_out"],
        res["final_out"],
        res["out_ptrs"],
        res["sig_ptrs"],

        B,
        Cin_total,
        Ti,
        Hi,
        Wi,

        Cout,
        Cin_per_group,
        kT,
        kH,
        kW,

        To,
        Ho,
        Wo,

        int(sT),
        int(sH),
        int(sW),
        int(pT),
        int(pH),
        int(pW),
        int(dT),
        int(dH),
        int(dW),
        int(groups),

        int(res["world_size"]),
        int(res["rank"]),
        int(total_numel),
        int(num_tiles),
        int(num_blocks),
    )

    return res["final_out"]