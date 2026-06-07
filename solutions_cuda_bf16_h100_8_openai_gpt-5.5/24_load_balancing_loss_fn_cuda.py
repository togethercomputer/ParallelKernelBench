import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from typing import Union, Tuple, Optional
from utils.cuda_helpers import compile_cuda_extension


CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

template <typename T>
__device__ __forceinline__ float val_to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float val_to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float val_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ float read_mask_value(
    const void* __restrict__ mask,
    int dtype_enum,
    int64_t idx
) {
    if (mask == nullptr) return 1.0f;

    switch (dtype_enum) {
        case 0:
            return reinterpret_cast<const bool*>(mask)[idx] ? 1.0f : 0.0f;
        case 1:
            return static_cast<float>(reinterpret_cast<const uint8_t*>(mask)[idx]);
        case 2:
            return static_cast<float>(reinterpret_cast<const int32_t*>(mask)[idx]);
        case 3:
            return static_cast<float>(reinterpret_cast<const int64_t*>(mask)[idx]);
        case 4:
            return reinterpret_cast<const float*>(mask)[idx];
        case 5:
            return __half2float(reinterpret_cast<const __half*>(mask)[idx]);
        case 6:
            return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(mask)[idx]);
        default:
            return 0.0f;
    }
}

template <typename scalar_t>
__global__ void process_gate_tiled_kernel(
    const scalar_t* __restrict__ logits,
    int64_t rows,
    int num_experts,
    int top_k,
    const void* __restrict__ mask,
    int64_t mask_len,
    int mask_dtype_enum,
    int use_mask,
    float* __restrict__ sum_probs,
    float* __restrict__ counts,
    float* __restrict__ denom,
    int rows_per_block
) {
    extern __shared__ float smem[];
    float* red = smem;                       // blockDim.x floats
    float* sh_probs = red + blockDim.x;      // num_experts floats
    float* sh_counts = sh_probs + num_experts; // num_experts floats
    float* sh_denom = sh_counts + num_experts; // 1 float

    int tid = threadIdx.x;

    for (int e = tid; e < num_experts; e += blockDim.x) {
        sh_probs[e] = 0.0f;
        sh_counts[e] = 0.0f;
    }
    if (tid == 0) {
        *sh_denom = 0.0f;
    }
    __syncthreads();

    int64_t start_row = static_cast<int64_t>(blockIdx.x) * rows_per_block;

    for (int rr = 0; rr < rows_per_block; ++rr) {
        int64_t row = start_row + rr;
        if (row >= rows) break;

        float weight = 1.0f;
        if (use_mask) {
            weight = read_mask_value(mask, mask_dtype_enum, row % mask_len);
        }

        if (weight != 0.0f) {
            if (use_mask && tid == 0) {
                *sh_denom += weight;
            }

            const scalar_t* row_ptr = logits + row * static_cast<int64_t>(num_experts);

            float local_max = -INFINITY;
            for (int e = tid; e < num_experts; e += blockDim.x) {
                float x = val_to_float<scalar_t>(row_ptr[e]);
                local_max = fmaxf(local_max, x);
            }

            red[tid] = local_max;
            __syncthreads();

            for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    red[tid] = fmaxf(red[tid], red[tid + offset]);
                }
                __syncthreads();
            }

            float m = red[0];

            float local_sum = 0.0f;
            for (int e = tid; e < num_experts; e += blockDim.x) {
                float x = val_to_float<scalar_t>(row_ptr[e]);
                local_sum += expf(x - m);
            }

            red[tid] = local_sum;
            __syncthreads();

            for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
                if (tid < offset) {
                    red[tid] += red[tid + offset];
                }
                __syncthreads();
            }

            float inv_sum = 1.0f / red[0];

            for (int e = tid; e < num_experts; e += blockDim.x) {
                float x = val_to_float<scalar_t>(row_ptr[e]);
                float p = expf(x - m) * inv_sum;
                sh_probs[e] += weight * p;
            }

            if (tid == 0) {
                int selected[128];

                for (int k = 0; k < top_k; ++k) {
                    int best_idx = 0;
                    float best_val = -INFINITY;

                    for (int e = 0; e < num_experts; ++e) {
                        bool already = false;
                        #pragma unroll
                        for (int j = 0; j < 128; ++j) {
                            if (j >= k) break;
                            if (selected[j] == e) {
                                already = true;
                                break;
                            }
                        }

                        if (!already) {
                            float v = val_to_float<scalar_t>(row_ptr[e]);
                            if (v > best_val) {
                                best_val = v;
                                best_idx = e;
                            }
                        }
                    }

                    selected[k] = best_idx;
                    sh_counts[best_idx] += weight;
                }
            }
            __syncthreads();
        }
    }

    for (int e = tid; e < num_experts; e += blockDim.x) {
        atomicAdd(sum_probs + e, sh_probs[e]);
        atomicAdd(counts + e, sh_counts[e]);
    }

    if (use_mask && tid == 0) {
        atomicAdd(denom, *sh_denom);
    }
}

__global__ void finalize_loss_kernel(
    const float* __restrict__ sum_probs,
    const float* __restrict__ counts,
    const float* __restrict__ denom_ptr,
    float* __restrict__ loss,
    int num_experts,
    int64_t total_rows,
    int use_mask
) {
    extern __shared__ float red[];
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int e = tid; e < num_experts; e += blockDim.x) {
        acc += sum_probs[e] * counts[e];
    }

    red[tid] = acc;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            red[tid] += red[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = use_mask ? denom_ptr[0] : static_cast<float>(total_rows);
        loss[0] = red[0] * static_cast<float>(num_experts) / (denom * denom);
    }
}

__global__ void avg_loss_uva_kernel(
    const long long* __restrict__ ptrs,
    float* __restrict__ out,
    int world_size
) {
    float s = 0.0f;
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r >= world_size) break;
        const float* p = reinterpret_cast<const float*>(static_cast<uintptr_t>(ptrs[r]));
        s += p[0];
    }
    out[0] = s / static_cast<float>(world_size);
}

int next_pow2_threads(int n) {
    int t = 32;
    while (t < n && t < 1024) t <<= 1;
    return t;
}

int mask_dtype_enum(torch::ScalarType st) {
    if (st == torch::kBool) return 0;
    if (st == torch::kUInt8) return 1;
    if (st == torch::kInt32) return 2;
    if (st == torch::kInt64) return 3;
    if (st == torch::kFloat32) return 4;
    if (st == torch::kFloat16) return 5;
    if (st == torch::kBFloat16) return 6;
    TORCH_CHECK(false, "unsupported attention_mask dtype");
}

template <typename ptr_t>
void launch_process_one(
    torch::Tensor gate,
    int num_experts,
    int top_k,
    const void* mask_ptr,
    int64_t mask_len,
    int mask_dtype,
    int use_mask,
    torch::Tensor sum_probs,
    torch::Tensor counts,
    torch::Tensor denom,
    int rows_per_block,
    cudaStream_t stream,
    ptr_t typed_ptr
) {
    int64_t rows = gate.size(0);
    if (rows == 0) return;

    int threads = next_pow2_threads(num_experts);
    int64_t blocks64 = (rows + rows_per_block - 1) / rows_per_block;
    TORCH_CHECK(blocks64 <= INT_MAX, "too many rows");
    int blocks = static_cast<int>(blocks64);

    size_t shmem = static_cast<size_t>(threads + 2 * num_experts + 1) * sizeof(float);

    process_gate_tiled_kernel<<<blocks, threads, shmem, stream>>>(
        typed_ptr,
        rows,
        num_experts,
        top_k,
        mask_ptr,
        mask_len,
        mask_dtype,
        use_mask,
        sum_probs.data_ptr<float>(),
        counts.data_ptr<float>(),
        denom.data_ptr<float>(),
        rows_per_block
    );
}

void compute_local_loss_impl(
    std::vector<torch::Tensor> gates,
    torch::Tensor mask,
    bool has_mask,
    int num_experts,
    int top_k,
    torch::Tensor sum_probs,
    torch::Tensor counts,
    torch::Tensor denom,
    torch::Tensor loss
) {
    TORCH_CHECK(!gates.empty(), "gate_logits must not be empty");
    TORCH_CHECK(num_experts > 0, "num_experts must be positive");
    TORCH_CHECK(top_k > 0 && top_k <= num_experts, "invalid top_k");
    TORCH_CHECK(top_k <= 128, "custom CUDA path supports top_k <= 128");

    CHECK_CUDA(sum_probs);
    CHECK_CUDA(counts);
    CHECK_CUDA(denom);
    CHECK_CUDA(loss);
    CHECK_CONTIGUOUS(sum_probs);
    CHECK_CONTIGUOUS(counts);
    CHECK_CONTIGUOUS(denom);
    CHECK_CONTIGUOUS(loss);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    cudaMemsetAsync(sum_probs.data_ptr<float>(), 0, num_experts * sizeof(float), stream);
    cudaMemsetAsync(counts.data_ptr<float>(), 0, num_experts * sizeof(float), stream);
    cudaMemsetAsync(denom.data_ptr<float>(), 0, sizeof(float), stream);

    const void* mask_ptr = nullptr;
    int64_t mask_len = 1;
    int mask_dtype = 0;

    if (has_mask) {
        CHECK_CUDA(mask);
        CHECK_CONTIGUOUS(mask);
        mask_ptr = mask.data_ptr();
        mask_len = mask.numel();
        mask_dtype = mask_dtype_enum(mask.scalar_type());
        TORCH_CHECK(mask_len > 0, "attention_mask must be non-empty");
    }

    int64_t total_rows = 0;
    int rows_per_block = (num_experts <= 128) ? 8 : 4;

    for (auto& g : gates) {
        CHECK_CUDA(g);
        CHECK_CONTIGUOUS(g);
        TORCH_CHECK(g.dim() == 2, "each gate tensor must have shape [tokens, num_experts]");
        TORCH_CHECK(g.size(1) == num_experts, "gate tensor last dim must equal num_experts");

        total_rows += g.size(0);

        if (g.scalar_type() == torch::kBFloat16) {
            const __nv_bfloat16* p =
                reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
            launch_process_one(g, num_experts, top_k, mask_ptr, mask_len, mask_dtype,
                               has_mask ? 1 : 0, sum_probs, counts, denom,
                               rows_per_block, stream, p);
        } else if (g.scalar_type() == torch::kFloat32) {
            const float* p = g.data_ptr<float>();
            launch_process_one(g, num_experts, top_k, mask_ptr, mask_len, mask_dtype,
                               has_mask ? 1 : 0, sum_probs, counts, denom,
                               rows_per_block, stream, p);
        } else if (g.scalar_type() == torch::kFloat16) {
            const __half* p =
                reinterpret_cast<const __half*>(g.data_ptr<at::Half>());
            launch_process_one(g, num_experts, top_k, mask_ptr, mask_len, mask_dtype,
                               has_mask ? 1 : 0, sum_probs, counts, denom,
                               rows_per_block, stream, p);
        } else {
            TORCH_CHECK(false, "gate_logits dtype must be bfloat16, float16, or float32");
        }
    }

    int threads = next_pow2_threads(num_experts);
    size_t shmem = static_cast<size_t>(threads) * sizeof(float);

    finalize_loss_kernel<<<1, threads, shmem, stream>>>(
        sum_probs.data_ptr<float>(),
        counts.data_ptr<float>(),
        denom.data_ptr<float>(),
        loss.data_ptr<float>(),
        num_experts,
        total_rows,
        has_mask ? 1 : 0
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_local_loss_nomask(
    std::vector<torch::Tensor> gates,
    int num_experts,
    int top_k,
    torch::Tensor sum_probs,
    torch::Tensor counts,
    torch::Tensor denom,
    torch::Tensor loss
) {
    torch::Tensor empty_mask;
    compute_local_loss_impl(
        gates, empty_mask, false, num_experts, top_k, sum_probs, counts, denom, loss);
}

void compute_local_loss_mask(
    std::vector<torch::Tensor> gates,
    torch::Tensor mask,
    int num_experts,
    int top_k,
    torch::Tensor sum_probs,
    torch::Tensor counts,
    torch::Tensor denom,
    torch::Tensor loss
) {
    compute_local_loss_impl(
        gates, mask, true, num_experts, top_k, sum_probs, counts, denom, loss);
}

void launch_avg_loss_uva(
    torch::Tensor ptrs_tensor,
    torch::Tensor out,
    int world_size
) {
    CHECK_CUDA(ptrs_tensor);
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(ptrs_tensor);
    CHECK_CONTIGUOUS(out);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    avg_loss_uva_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const long long*>(ptrs_tensor.data_ptr<int64_t>()),
        out.data_ptr<float>(),
        world_size
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_local_loss_nomask", &compute_local_loss_nomask,
          "Fused local MoE load-balancing loss without attention mask");
    m.def("compute_local_loss_mask", &compute_local_loss_mask,
          "Fused local MoE load-balancing loss with attention mask");
    m.def("launch_avg_loss_uva", &launch_avg_loss_uva,
          "Average scalar loss across ranks via symmetric-memory UVA loads");
}
'''


_ext = None
_scratch_cache = {}
_symm_cache = {}


def _get_ext():
    global _ext
    if _ext is None:
        _ext = compile_cuda_extension("moe_lb_loss_bf16_h100_symm_ext", CUDA_SRC)
    return _ext


def _prepare_gate_list(gate_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
    if isinstance(gate_logits, (tuple, list)):
        assert len(gate_logits) > 0
        device = gate_logits[0].device
        gates = []
        for g in gate_logits:
            if g.device != device:
                g = g.to(device, non_blocking=True)
            if not g.is_contiguous():
                g = g.contiguous()
            gates.append(g)
        return gates, device
    else:
        device = gate_logits.device
        g = gate_logits
        if not g.is_contiguous():
            g = g.contiguous()
        return [g], device


def _get_scratch(num_experts: int, device: torch.device):
    key = (int(num_experts), int(device.index if device.index is not None else torch.cuda.current_device()))
    cached = _scratch_cache.get(key)
    if cached is not None:
        return cached

    sum_probs = torch.empty((num_experts,), device=device, dtype=torch.float32)
    counts = torch.empty((num_experts,), device=device, dtype=torch.float32)
    denom = torch.empty((1,), device=device, dtype=torch.float32)
    local_loss = torch.empty((1,), device=device, dtype=torch.float32)

    cached = (sum_probs, counts, denom, local_loss)
    _scratch_cache[key] = cached
    return cached


def _get_symm_scalar(device: torch.device):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    key = (
        int(device.index if device.index is not None else torch.cuda.current_device()),
        int(world_size),
        int(rank),
    )
    cached = _symm_cache.get(key)
    if cached is not None:
        return cached

    buf = symm_mem.empty((1,), device=device, dtype=torch.float32)
    hdl = symm_mem.rendezvous(buf, dist.group.WORLD)
    out = torch.empty((1,), device=device, dtype=torch.float32)
    ptrs_tensor = torch.tensor(hdl.buffer_ptrs, device=device, dtype=torch.int64)

    cached = (buf, hdl, out, ptrs_tensor)
    _symm_cache[key] = cached
    return cached


@torch.no_grad()
def solution(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    num_experts: int,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gates, compute_device = _prepare_gate_list(gate_logits)

    assert compute_device.type == "cuda", "custom CUDA solution requires CUDA gate_logits"
    assert num_experts > 0
    assert top_k > 0

    ext = _get_ext()
    sum_probs, counts, denom, local_loss_tmp = _get_scratch(num_experts, compute_device)

    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    if distributed:
        symm_loss, hdl, global_out, ptrs_tensor = _get_symm_scalar(compute_device)
        local_loss = symm_loss
    else:
        local_loss = local_loss_tmp

    if attention_mask is None:
        ext.compute_local_loss_nomask(
            gates,
            int(num_experts),
            int(top_k),
            sum_probs,
            counts,
            denom,
            local_loss,
        )
    else:
        mask = attention_mask
        if mask.device != compute_device:
            mask = mask.to(compute_device, non_blocking=True)
        if not mask.is_contiguous():
            mask = mask.contiguous()

        ext.compute_local_loss_mask(
            gates,
            mask,
            int(num_experts),
            int(top_k),
            sum_probs,
            counts,
            denom,
            local_loss,
        )

    if distributed:
        hdl.barrier(channel=0)
        ext.launch_avg_loss_uva(ptrs_tensor, global_out, int(hdl.world_size))
        hdl.barrier(channel=1)
        return global_out.reshape(())

    return local_loss.reshape(())