from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/util/complex.h>
#include <stdint.h>

static inline int ceil_div_i64(int64_t a, int b) {
    return (int)((a + b - 1) / b);
}

__device__ __forceinline__ int64_t inv_zigzag_chunk(int64_t logical_chunk, int world_size) {
    // argsort([0, 2w-1, 1, 2w-2, ...])
    if (logical_chunk < world_size) return 2 * logical_chunk;
    return (int64_t)(4 * world_size - 1) - 2 * logical_chunk;
}

__device__ __forceinline__ int64_t zigzag_chunk(int64_t pre_chunk, int world_size) {
    if ((pre_chunk & 1LL) == 0) return pre_chunk >> 1;
    return (int64_t)(2 * world_size - 1) - (pre_chunk >> 1);
}

__device__ __forceinline__ float load_as_float(const void* ptr, int64_t idx, int dtype_code) {
    if (dtype_code == 0) {
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(ptr);
        return __bfloat162float(p[idx]);
    } else {
        const float* p = reinterpret_cast<const float*>(ptr);
        return p[idx];
    }
}

__global__ void pack3_bf16_kernel(
    const __nv_bfloat16* __restrict__ x1,
    const __nv_bfloat16* __restrict__ x2,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ symm,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        symm[idx] = x1[idx];
        symm[n + idx] = x2[idx];
        symm[2 * n + idx] = v[idx];
    }
}

__global__ void gather_x1_and_u_kernel(
    const int64_t* __restrict__ ptrs,
    __nv_bfloat16* __restrict__ x1_full,
    float* __restrict__ u_float,
    int B,
    int D,
    int local_seq,
    int world_size,
    int rank,
    int with_zigzag
) {
    int local_channels = D / world_size;
    int seq_len = local_seq * world_size;
    int64_t total = (int64_t)B * local_channels * seq_len;
    int64_t plane = (int64_t)B * D * local_seq;
    int64_t chunk_len = local_seq / 2;  // seq_len / (2 * world_size)

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int64_t t = idx % seq_len;
        int64_t tmp = idx / seq_len;
        int c = (int)(tmp % local_channels);
        int b = (int)(tmp / local_channels);

        int64_t pre_t = t;
        if (with_zigzag) {
            int64_t ch = t / chunk_len;
            int64_t off = t - ch * chunk_len;
            pre_t = inv_zigzag_chunk(ch, world_size) * chunk_len + off;
        }

        int src_rank = (int)(pre_t / local_seq);
        int sl = (int)(pre_t - (int64_t)src_rank * local_seq);
        int global_c = rank * local_channels + c;

        const __nv_bfloat16* base =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)ptrs[src_rank]);
        int64_t in_off = ((int64_t)b * D + global_c) * local_seq + sl;

        __nv_bfloat16 x1v = base[in_off];
        __nv_bfloat16 x2v = base[plane + in_off];
        __nv_bfloat16 vv  = base[2 * plane + in_off];

        float prod = __bfloat162float(x2v) * __bfloat162float(vv);
        __nv_bfloat16 prod_bf16 = __float2bfloat16(prod);

        x1_full[idx] = x1v;
        u_float[idx] = __bfloat162float(prod_bf16);
    }
}

__global__ void expand_h_kernel(
    const void* __restrict__ h,
    float* __restrict__ h_expanded,
    int filter_len,
    int local_channels,
    int local_groups,
    int group_dim,
    int rank,
    int h_dtype_code
) {
    int64_t total = (int64_t)local_channels * filter_len;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    int group_start = rank * local_groups;

    for (; idx < total; idx += stride) {
        int t = (int)(idx % filter_len);
        int c = (int)(idx / filter_len);
        int g = group_start + c / group_dim;
        float val = load_as_float(h, (int64_t)g * filter_len + t, h_dtype_code);
        h_expanded[idx] = val;
    }
}

__global__ void complex_filter_mul_kernel(
    float2* __restrict__ uf,
    const float2* __restrict__ hf,
    int B,
    int local_channels,
    int freq_len,
    float scale
) {
    int64_t total = (int64_t)B * local_channels * freq_len;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int f = (int)(idx % freq_len);
        int c = (int)((idx / freq_len) % local_channels);

        float2 a = uf[idx];
        float2 b = hf[(int64_t)c * freq_len + f];

        float2 out;
        out.x = (a.x * b.x - a.y * b.y) * scale;
        out.y = (a.x * b.y + a.y * b.x) * scale;
        uf[idx] = out;
    }
}

__global__ void finalize_z_kernel(
    const __nv_bfloat16* __restrict__ x1_full,
    const float* __restrict__ u_float,
    const float* __restrict__ y_full,
    const void* __restrict__ bias,
    __nv_bfloat16* __restrict__ z_symm,
    int B,
    int local_channels,
    int seq_len,
    int fft_size,
    int rank,
    int bias_dtype_code
) {
    int64_t total = (int64_t)B * local_channels * seq_len;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int t = (int)(idx % seq_len);
        int64_t tmp = idx / seq_len;
        int c = (int)(tmp % local_channels);
        int b = (int)(tmp / local_channels);

        float bias_v = load_as_float(bias, (int64_t)rank * local_channels + c, bias_dtype_code);
        float conv = y_full[((int64_t)b * local_channels + c) * fft_size + t]
                   + u_float[idx] * bias_v;

        // Match reference ordering closely: fftconv returns BF16, then x1 * z in BF16.
        __nv_bfloat16 conv_b = __float2bfloat16(conv);
        float prod = __bfloat162float(x1_full[idx]) * __bfloat162float(conv_b);
        z_symm[idx] = __float2bfloat16(prod);
    }
}

__global__ void scatter_final_bsl_kernel(
    const int64_t* __restrict__ z_ptrs,
    __nv_bfloat16* __restrict__ out_bsl,
    int B,
    int D,
    int local_seq,
    int world_size,
    int rank,
    int with_zigzag
) {
    int local_channels = D / world_size;
    int seq_len = local_seq * world_size;
    int64_t total = (int64_t)B * local_seq * D;
    int64_t chunk_len = local_seq / 2;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int d = (int)(idx % D);
        int64_t tmp = idx / D;
        int sl = (int)(tmp % local_seq);
        int b = (int)(tmp / local_seq);

        int src_rank = d / local_channels;
        int c = d - src_rank * local_channels;

        int64_t pre_t = (int64_t)rank * local_seq + sl;
        int64_t logical_t = pre_t;
        if (with_zigzag) {
            int64_t ch = pre_t / chunk_len;
            int64_t off = pre_t - ch * chunk_len;
            logical_t = zigzag_chunk(ch, world_size) * chunk_len + off;
        }

        const __nv_bfloat16* zbase =
            reinterpret_cast<const __nv_bfloat16*>((uintptr_t)z_ptrs[src_rank]);
        out_bsl[idx] = zbase[((int64_t)b * local_channels + c) * seq_len + logical_t];
    }
}

void pack3_bf16(torch::Tensor x1, torch::Tensor x2, torch::Tensor v, torch::Tensor symm) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda() && v.is_cuda() && symm.is_cuda());
    TORCH_CHECK(x1.dtype() == torch::kBFloat16);
    int64_t n = x1.numel();
    int threads = 256;
    int blocks = ceil_div_i64(n, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pack3_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(symm.data_ptr<at::BFloat16>()),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_x1_and_u(
    torch::Tensor ptrs,
    torch::Tensor x1_full,
    torch::Tensor u_float,
    int B,
    int D,
    int local_seq,
    int world_size,
    int rank,
    bool with_zigzag
) {
    int local_channels = D / world_size;
    int seq_len = local_seq * world_size;
    int64_t total = (int64_t)B * local_channels * seq_len;
    int threads = 256;
    int blocks = ceil_div_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gather_x1_and_u_kernel<<<blocks, threads, 0, stream>>>(
        ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(x1_full.data_ptr<at::BFloat16>()),
        u_float.data_ptr<float>(),
        B, D, local_seq, world_size, rank, with_zigzag ? 1 : 0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void expand_h(
    torch::Tensor h,
    torch::Tensor h_expanded,
    int local_channels,
    int local_groups,
    int group_dim,
    int rank
) {
    int filter_len = (int)h.size(1);
    int dtype_code = (h.dtype() == torch::kBFloat16) ? 0 : 1;
    TORCH_CHECK(h.dtype() == torch::kBFloat16 || h.dtype() == torch::kFloat32,
                "h must be bf16 or fp32");
    int64_t total = (int64_t)local_channels * filter_len;
    int threads = 256;
    int blocks = ceil_div_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    expand_h_kernel<<<blocks, threads, 0, stream>>>(
        h.data_ptr(),
        h_expanded.data_ptr<float>(),
        filter_len,
        local_channels,
        local_groups,
        group_dim,
        rank,
        dtype_code
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void complex_filter_mul(torch::Tensor uf, torch::Tensor hf, int B, int local_channels, int freq_len, float scale) {
    int64_t total = (int64_t)B * local_channels * freq_len;
    int threads = 256;
    int blocks = ceil_div_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    complex_filter_mul_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float2*>(uf.data_ptr<c10::complex<float>>()),
        reinterpret_cast<const float2*>(hf.data_ptr<c10::complex<float>>()),
        B, local_channels, freq_len, scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finalize_z(
    torch::Tensor x1_full,
    torch::Tensor u_float,
    torch::Tensor y_full,
    torch::Tensor bias,
    torch::Tensor z_symm,
    int B,
    int local_channels,
    int seq_len,
    int fft_size,
    int rank
) {
    int dtype_code = (bias.dtype() == torch::kBFloat16) ? 0 : 1;
    TORCH_CHECK(bias.dtype() == torch::kBFloat16 || bias.dtype() == torch::kFloat32,
                "bias must be bf16 or fp32");
    int64_t total = (int64_t)B * local_channels * seq_len;
    int threads = 256;
    int blocks = ceil_div_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_z_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x1_full.data_ptr<at::BFloat16>()),
        u_float.data_ptr<float>(),
        y_full.data_ptr<float>(),
        bias.data_ptr(),
        reinterpret_cast<__nv_bfloat16*>(z_symm.data_ptr<at::BFloat16>()),
        B, local_channels, seq_len, fft_size, rank, dtype_code
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_final_bsl(
    torch::Tensor z_ptrs,
    torch::Tensor out_bsl,
    int B,
    int D,
    int local_seq,
    int world_size,
    int rank,
    bool with_zigzag
) {
    int64_t total = (int64_t)B * local_seq * D;
    int threads = 256;
    int blocks = ceil_div_i64(total, threads);
    if (blocks > 65535) blocks = 65535;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scatter_final_bsl_kernel<<<blocks, threads, 0, stream>>>(
        z_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out_bsl.data_ptr<at::BFloat16>()),
        B, D, local_seq, world_size, rank, with_zigzag ? 1 : 0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack3_bf16", &pack3_bf16, "pack x1/x2/v into symmetric BF16 buffer");
    m.def("gather_x1_and_u", &gather_x1_and_u, "UVA all-to-all gather fused with x2*v");
    m.def("expand_h", &expand_h, "expand grouped Hyena filter to per-channel fp32");
    m.def("complex_filter_mul", &complex_filter_mul, "in-place complex spectral multiply");
    m.def("finalize_z", &finalize_z, "bias + BF16 finalize into symmetric z");
    m.def("scatter_final_bsl", &scatter_final_bsl, "UVA all-to-all scatter fused to BSL layout");
}
'''

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("hyena_cp_bf16_symm_cuda_ext", CUDA_SRC)
    return _ext


_resource_cache = {}


def _cache_key(B, D, local_seq, dtype, device, group, world_size):
    return (B, D, local_seq, dtype, int(device.index or 0), id(group), world_size)


def _get_resources(B: int, D: int, local_seq: int, dtype: torch.dtype, device: torch.device, group):
    world_size = dist.get_world_size(group=group)
    key = _cache_key(B, D, local_seq, dtype, device, group, world_size)
    if key in _resource_cache:
        return _resource_cache[key]

    local_channels = D // world_size
    seq_len = local_seq * world_size

    # Input symmetric buffer holds x1, x2, v in one rendezvous.
    inp_symm = symm_mem.empty((3 * B * D * local_seq,), device=device, dtype=dtype)
    inp_hdl = symm_mem.rendezvous(inp_symm, group)

    # z symmetric buffer holds this rank's full-sequence local-channel result.
    z_symm = symm_mem.empty((B * local_channels * seq_len,), device=device, dtype=dtype)
    z_hdl = symm_mem.rendezvous(z_symm, group)

    inp_ptrs = torch.tensor(inp_hdl.buffer_ptrs, device=device, dtype=torch.int64)
    z_ptrs = torch.tensor(z_hdl.buffer_ptrs, device=device, dtype=torch.int64)

    x1_full = torch.empty((B, local_channels, seq_len), device=device, dtype=dtype)
    u_float = torch.empty((B, local_channels, seq_len), device=device, dtype=torch.float32)
    out_bsl = torch.empty((B, local_seq, D), device=device, dtype=dtype)

    res = {
        "inp_symm": inp_symm,
        "inp_hdl": inp_hdl,
        "z_symm": z_symm,
        "z_hdl": z_hdl,
        "inp_ptrs": inp_ptrs,
        "z_ptrs": z_ptrs,
        "x1_full": x1_full,
        "u_float": u_float,
        "out_bsl": out_bsl,
    }
    _resource_cache[key] = res
    return res


@torch.no_grad()
def solution(
    x1_seq: torch.Tensor,
    x2_seq: torch.Tensor,
    v_seq: torch.Tensor,
    h: torch.Tensor,
    conv_bias: torch.Tensor,
    num_groups: int,
    group_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    with_zigzag_splitting: bool = True,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert x1_seq.is_cuda and x2_seq.is_cuda and v_seq.is_cuda
    assert x1_seq.dtype == torch.bfloat16
    assert x2_seq.dtype == torch.bfloat16
    assert v_seq.dtype == torch.bfloat16
    assert x1_seq.is_contiguous() and x2_seq.is_contiguous() and v_seq.is_contiguous()

    ext = _get_ext()

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    B = int(x1_seq.shape[0])
    D = int(x1_seq.shape[1])
    local_seq = int(x1_seq.shape[2])
    local_channels = D // world_size
    seq_len = local_seq * world_size
    local_groups = num_groups // world_size
    fft_size = 2 * seq_len
    freq_len = fft_size // 2 + 1

    res = _get_resources(B, D, local_seq, x1_seq.dtype, x1_seq.device, group)

    # Pack three sequence-sharded activations into one symmetric allocation.
    ext.pack3_bf16(x1_seq, x2_seq, v_seq, res["inp_symm"])
    res["inp_hdl"].barrier(channel=0)

    # Device-side all-to-all gather through UVA; also produces BF16-rounded x2*v as fp32 FFT input.
    ext.gather_x1_and_u(
        res["inp_ptrs"],
        res["x1_full"],
        res["u_float"],
        B,
        D,
        local_seq,
        world_size,
        rank,
        bool(with_zigzag_splitting),
    )

    # Per-channel grouped filter expansion, then cuFFT-backed spectral convolution.
    h_contig = h.contiguous()
    h_expanded = torch.empty(
        (local_channels, int(h_contig.shape[1])),
        device=x1_seq.device,
        dtype=torch.float32,
    )
    ext.expand_h(
        h_contig,
        h_expanded,
        local_channels,
        local_groups,
        group_dim,
        rank,
    )

    u_f = torch.fft.rfft(res["u_float"], n=fft_size)
    h_f = torch.fft.rfft(h_expanded, n=fft_size).contiguous()

    # Reference divides kernel_f by fft_size before irfft(..., norm="forward").
    ext.complex_filter_mul(
        u_f,
        h_f,
        B,
        local_channels,
        freq_len,
        float(1.0 / fft_size),
    )

    y_full = torch.fft.irfft(u_f, n=fft_size, norm="forward")

    # Bias, BF16 cast of fftconv output, multiply by x1, and publish local-channel full-seq z.
    ext.finalize_z(
        res["x1_full"],
        res["u_float"],
        y_full,
        conv_bias.contiguous(),
        res["z_symm"],
        B,
        local_channels,
        seq_len,
        fft_size,
        rank,
    )
    res["z_hdl"].barrier(channel=1)

    # Device-side all-to-all back to sequence-sharded layout, directly returning [B, l, D].
    ext.scatter_final_bsl(
        res["z_ptrs"],
        res["out_bsl"],
        B,
        D,
        local_seq,
        world_size,
        rank,
        bool(with_zigzag_splitting),
    )
    return res["out_bsl"]