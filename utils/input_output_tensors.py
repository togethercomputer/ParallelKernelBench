"""
Utility functions for creating input tensors and saving output tensors.

These functions are shared across different worker scripts to ensure
consistent tensor creation and saving behavior.
"""

import os
import json
import copy
import importlib.util
import math

import torch
import torch.distributed as dist

def save_tensor(output, logs_dir: str, rank: int) -> str:
    """
    Save output tensor(s) to file.
    
    Handles:
    - Single tensor: saves as rank_X.pt
    - Tuple/list of tensors: saves as dict with keys 'output_0', 'output_1', etc.
    - Dict: saves as-is
    """
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, f"rank_{rank}.pt")
    
    # Handle different output types
    if isinstance(output, torch.Tensor):
        # Single tensor
        torch.save(output.detach().cpu(), path)
    elif isinstance(output, (tuple, list)):
        # Multiple tensors - save as dict
        output_dict = {f'output_{i}': t.detach().cpu() if isinstance(t, torch.Tensor) else t 
                      for i, t in enumerate(output)}
        torch.save(output_dict, path)
    elif isinstance(output, dict):
        # Dict - convert tensors to CPU
        output_dict = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                      for k, v in output.items()}
        torch.save(output_dict, path)
    else:
        # Fallback: try to save as-is
        torch.save(output, path)
    
    return path

# ---------------------------------------------------------------------------
# INPUT TENSOR STANDARD (tuple-only)
# ---------------------------------------------------------------------------
# create_input_tensor() returns a tuple unpacked as solution_fn(*x). Entries are usually tensors
# but may include Python scalars / dicts / dataclasses (e.g. problem 4, problems 100–105).
#   - solution(tensor) for single-tensor problems: x is (tensor,)
#   - solution(t1, t2) for multi-arg problems: x is (t1, t2, ...)
# Problems 100–105: solution(rank, world_size, cfg, input_ids).
# Output from solution_fn may still be a single tensor or a tuple; save_tensor() handles both.
# ---------------------------------------------------------------------------

def _seed(problem_id: int, rank: int, trial: int = 0) -> None:
    """trial varies RNG across eval runs; trial=0 matches the historical single-run seed."""
    torch.manual_seed(42 + problem_id * 1000 + rank + trial * 1_000_003)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REF_MODULES_CACHE: dict[int, object] = {}

def _round_up_multiple(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m

def _ddp_mlp_shapes_divisible_by_dp(N: int, world_size: int) -> tuple[int, int, int]:
    """Pick (d_in, hidden, d_out) so W1,b1,W2,b2 total numel is divisible by world_size (ZeRO partitions)."""
    d_in = max(16, min(N, 256))
    d_out = max(8, min(N // 4, 256))
    hidden = max(32, min(N // 2, 512))
    for _ in range(1024):
        numel = hidden * d_in + hidden + d_out * hidden + d_out
        if numel % world_size == 0:
            return d_in, hidden, d_out
        hidden += 1
    raise RuntimeError(f"Could not align MLP parameter numel with world_size={world_size}")

def _factor_tp_fsdp(world_size: int) -> tuple[int, int]:
    """Choose ``N_TP × N_FSDP == world_size``, preferring both factors ≥ 2."""
    for n_tp in range(2, world_size):
        if world_size % n_tp == 0:
            n_fsdp = world_size // n_tp
            if n_fsdp >= 2:
                return n_tp, n_fsdp
    return 1, world_size

def _moe_narrow_num_experts(world_size: int) -> int:
    """Largest ``E < world_size`` with ``world_size % E == 0`` (narrow EP / DP-over-EP)."""
    for E in range(world_size // 2, 1, -1):
        if world_size % E == 0:
            return E
    return 1

def _linear(in_features: int, out_features: int, dtype: torch.dtype, device) -> torch.nn.Linear:
    return torch.nn.Linear(in_features, out_features).to(device=device, dtype=dtype)

def _load_reference_module(problem_id: int):
    if problem_id in _REF_MODULES_CACHE:
        return _REF_MODULES_CACHE[problem_id]
    stem = {
        100: "100_deepseek_v3_671b_tp_attn_ep_moe",
        101: "101_gemma3_27b_tp_attn_tp_mlp",
        102: "102_llama32_3b_tp_attn_tp_mlp",
        103: "103_olmo_3_32b_tp_attn_tp_mlp",
        104: "104_qwen3_235b_tp_attn_ep_moe",
        105: "105_qwen3_code_flash_30b_tp_attn_ep_moe",
        106: "106_deepseek_v3_671b_cp_ulysses_attn_ep_moe",
        107: "107_gemma3_27b_cp_ulysses_attn_tp_mlp",
        108: "108_llama32_3b_cp_ulysses_attn_tp_mlp",
        109: "109_olmo_3_32b_cp_ulysses_attn_tp_mlp",
        110: "110_qwen3_235b_cp_ulysses_attn_ep_moe",
        111: "111_qwen3_code_flash_30b_cp_ulysses_attn_ep_moe",
    }[problem_id]
    path = os.path.join(_PROJECT_ROOT, "reference", f"{stem}.py")
    spec = importlib.util.spec_from_file_location(f"ref_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _REF_MODULES_CACHE[problem_id] = mod
    return mod

def _align_model_args_100(cfg, world_size: int) -> None:
    """ModelArgs for reference/100: TP/EP divisibility constraints."""
    cfg.n_layers = 2
    for attr in ("dim", "inter_dim", "moe_inter_dim"):
        v = getattr(cfg, attr)
        if v % world_size:
            setattr(cfg, attr, _round_up_multiple(v, world_size))
    if cfg.vocab_size % world_size:
        cfg.vocab_size = _round_up_multiple(cfg.vocab_size, world_size)
    if cfg.n_heads % world_size:
        cfg.n_heads = _round_up_multiple(cfg.n_heads, world_size)
    if cfg.n_routed_experts % world_size:
        cfg.n_routed_experts = _round_up_multiple(cfg.n_routed_experts, world_size)
    shared = cfg.n_shared_experts * cfg.moe_inter_dim
    guard = 0
    while shared % world_size and guard < 4096:
        cfg.moe_inter_dim += 1
        shared = cfg.n_shared_experts * cfg.moe_inter_dim
        guard += 1

def _common_attn_dims(base_shape, world_size):
    """Shared (B, T, num_heads, head_dim) from base_shape (M, N)."""
    M, N = base_shape
    B, T = max(1, M // 64), max(1, N // 64)
    num_heads = 8
    head_dim = 64
    assert num_heads % world_size == 0, f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"
    return B, T, num_heads, head_dim

def _build_cp_groups():
    """CP-only (problem 54): the CP group is just WORLD."""
    return dist.group.WORLD
 
def _build_tp_cp_groups(tp_size: int):
    """
    Build TP / CP groups for problem 55, following Megatron order='tp-cp'.
    Rank layout: [cp0_tp0, cp0_tp1, ..., cp1_tp0, cp1_tp1, ...]
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cp_size = world_size // tp_size
 
    tp_group = None
    cp_group = None
 
    # TP groups: contiguous blocks of tp_size within each CP index.
    for cp_idx in range(cp_size):
        ranks = list(range(cp_idx * tp_size, (cp_idx + 1) * tp_size))
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            tp_group = g
 
    # CP groups: same TP position across CP partitions.
    for tp_idx in range(tp_size):
        ranks = [tp_idx + cp_idx * tp_size for cp_idx in range(cp_size)]
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            cp_group = g
 
    assert tp_group is not None and cp_group is not None
    return tp_group, cp_group, cp_size
 
def _build_cp_pp_groups(pp_size: int):
    """
    Build CP / PP groups for problem 56, following Megatron order with TP=DP=1.
    Rank layout: [pp0_cp0, pp0_cp1, ..., pp1_cp0, pp1_cp1, ...]
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cp_size = world_size // pp_size
 
    cp_group = None
    pp_group = None
 
    # CP groups: contiguous stage-local blocks.
    for pp_idx in range(pp_size):
        ranks = list(range(pp_idx * cp_size, (pp_idx + 1) * cp_size))
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            cp_group = g
 
    # PP groups: same CP rank across pipeline stages.
    for cp_idx in range(cp_size):
        ranks = [cp_idx + pp_idx * cp_size for pp_idx in range(pp_size)]
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            pp_group = g
 
    assert cp_group is not None and pp_group is not None
    cp_rank = dist.get_rank(cp_group)
    pp_rank = dist.get_rank(pp_group)
    return cp_group, pp_group, cp_rank, pp_rank, cp_size
 
def _build_cp_dp_groups(dp_size: int):
    """
    Build CP / DP / DP-with-CP groups for problem 57 (backward), following Megatron order 'cp-dp'.
    Rank layout: [dp0_cp0, dp0_cp1, ..., dp1_cp0, dp1_cp1, ...]
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    cp_size = world_size // dp_size
 
    dp_cp_group = dist.new_group(ranks=list(range(world_size)))
 
    dp_group = None
    cp_group = None
 
    # DP groups: same CP position across DP replicas.
    for cp_idx in range(cp_size):
        ranks = [cp_idx + dp_idx * cp_size for dp_idx in range(dp_size)]
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            dp_group = g
 
    # CP groups: contiguous CP shards inside one DP replica.
    for dp_idx in range(dp_size):
        ranks = list(range(dp_idx * cp_size, (dp_idx + 1) * cp_size))
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            cp_group = g
 
    assert dp_group is not None and cp_group is not None
    cp_rank = dist.get_rank(cp_group)
    dp_rank = dist.get_rank(dp_group)
    return cp_group, dp_group, dp_cp_group, cp_rank, dp_rank, cp_size

def _build_polar_azimuth_groups(azimuth_size: int):
    """
    Build a 2D polar/azimuth process grid.
    Rank layout: [polar0_az0, polar0_az1, ..., polar1_az0, polar1_az1, ...]
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    polar_size = world_size // azimuth_size

    azimuth_group = None
    polar_group = None

    for polar_idx in range(polar_size):
        ranks = list(range(polar_idx * azimuth_size, (polar_idx + 1) * azimuth_size))
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            azimuth_group = g

    for azimuth_idx in range(azimuth_size):
        ranks = [polar_idx * azimuth_size + azimuth_idx for polar_idx in range(polar_size)]
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            polar_group = g

    assert azimuth_group is not None and polar_group is not None
    azimuth_rank = dist.get_rank(azimuth_group)
    polar_rank = dist.get_rank(polar_group)
    return azimuth_group, polar_group, azimuth_rank, polar_rank, azimuth_size, polar_size

def create_input_tensor(
    rank: int,
    world_size: int,
    problem_id: int,
    base_shape: tuple,
    dtype: torch.dtype,
    trial: int = 0,
    device=None,
):
    """
    Create appropriate input tensors for this problem. Always returns a tuple of tensors.

    base_shape is typically (M, N) from the worker args (e.g. 1024, 1024).
    Derived dimensions are hardcoded where needed for consistency.

    Args:
        rank: Process rank (0..world_size-1)
        world_size: Total number of processes
        problem_id: Problem ID (e.g. 1–105) from reference filename
        base_shape: Base tensor shape tuple (e.g., (M, N))
        dtype: Tensor data type
        trial: Non-negative index; changes RNG for problems that use random inputs (trial=0 is legacy behavior).
        device: PyTorch device or device string. If None, uses torch.device("cuda", rank)
    """
    if device is None:
        dev = torch.device("cuda", rank)
    elif isinstance(device, str):
        dev = device
    else:
        dev = device

    M, N = base_shape
    val = float(rank + 1)

    # 1-8: collectives
    if problem_id in [1, 2, 3, 6]:
        return (torch.full(base_shape, val, dtype=dtype, device=dev),)
    elif problem_id == 4:
        return (torch.full(base_shape, val, dtype=dtype, device=dev), 0)
    elif problem_id == 5:
        src = 0
        if rank == src:
            chunks = [torch.full(base_shape, float(i + 1), dtype=dtype, device=dev) for i in range(world_size)]
            return (torch.stack(chunks, dim=0),)
        return (torch.zeros(base_shape, dtype=dtype, device=dev),)
    elif problem_id == 7:
        return (torch.full((world_size * M,) + base_shape[1:], val, dtype=dtype, device=dev),)
    elif problem_id == 8:
        chunks = [torch.full(base_shape, float(rank * 10 + d), dtype=dtype, device=dev) for d in range(world_size)]
        return (torch.stack(chunks, dim=0),)

    # 9: layernorm_backward
    elif problem_id == 9:
        _seed(problem_id, rank, trial)
        B, H = base_shape
        X_hat = torch.randn((B, H), dtype=dtype, device=dev)
        X_hat = X_hat / (X_hat.norm(dim=-1, keepdim=True) + 1e-5)
        dY = torch.randn((B, H), dtype=dtype, device=dev)
        return (X_hat, dY)

    # 10: embedding_lookup
    elif problem_id == 10:
        _seed(problem_id, rank, trial)
        shard_size, embed_dim = base_shape
        local_shard = torch.randn((shard_size, embed_dim), dtype=dtype, device=dev)
        indices = torch.randint(0, world_size * shard_size, (shard_size,), dtype=torch.long, device=dev)
        return (indices, local_shard)

    # 11: allgather_gemm_AT
    elif problem_id == 11:
        _seed(problem_id, rank, trial)
        K = 512
        K_local = K // world_size
        A_local = torch.randn((M, K_local), dtype=dtype, device=dev)
        B = torch.randn((K, N), dtype=dtype, device=dev)
        return (A_local, B)

    # 12: allgather_gemm
    elif problem_id == 12:
        _seed(problem_id, rank, trial)
        K = 512
        K_local = K // world_size
        A_local = torch.randn((M, K_local), dtype=dtype, device=dev)
        B = torch.randn((K, N), dtype=dtype, device=dev)
        return (A_local, B)

    # 13: gemm_allreduce
    elif problem_id == 13:
        _seed(problem_id, rank, trial)
        K = 512
        A_local = torch.randn((M, K), dtype=dtype, device=dev)
        B_local = torch.randn((K, N), dtype=dtype, device=dev)
        return (A_local, B_local)

    # 14: gemm_allgather
    elif problem_id == 14:
        _seed(problem_id, rank, trial)
        K = 512
        N_local = N // world_size
        A = torch.randn((M, K), dtype=dtype, device=dev)
        B = torch.randn((K, N_local), dtype=dtype, device=dev)
        return (A, B)

    # 15: combined_sharded_gemms
    elif problem_id == 15:
        _seed(problem_id, rank, trial)
        M_rows = _round_up_multiple(M, world_size)
        H = _round_up_multiple(256, world_size)
        H_local = H // world_size
        F = 512
        x_local = torch.randn((M_rows, H_local), dtype=dtype, device=dev)
        W1 = torch.randn((H, F), dtype=dtype, device=dev)
        W2 = torch.randn((F, H), dtype=dtype, device=dev)
        return (x_local, W1, W2)

    # 16: gemm_reducescatter
    elif problem_id == 16:
        _seed(problem_id, rank, trial)
        K = 512
        K_local = K // world_size
        A_local = torch.randn((M, K_local), dtype=dtype, device=dev)
        B_local = torch.randn((K_local, N), dtype=dtype, device=dev)
        return (A_local, B_local)

    # 17: rope_allgather
    elif problem_id == 17:
        _seed(problem_id, rank, trial)
        B, T, num_heads, head_dim = _common_attn_dims(base_shape, world_size)
        S_local = max(1, T // world_size)
        q_local = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)
        k_local = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)
        cos_local = torch.randn((B, S_local, head_dim), dtype=dtype, device=dev)
        sin_local = torch.randn((B, S_local, head_dim), dtype=dtype, device=dev)
        return (q_local, k_local, cos_local, sin_local)

    # 18: rms_norm
    elif problem_id == 18:
        _seed(problem_id, rank, trial)
        hidden = torch.randn(base_shape, dtype=dtype, device=dev)
        weight = torch.randn((N,), dtype=dtype, device=dev)
        return (hidden, weight, 1e-5)

    # 19: blocked_fp8_quantize
    elif problem_id == 19:
        _seed(problem_id, rank, trial)
        return (torch.randn(base_shape, dtype=dtype, device=dev), 128)

    # 20: blocked_fp8_dequantize
    elif problem_id == 20:
        _seed(problem_id, rank, trial)
        chunk_numel = M * N
        block_size = 128
        num_blocks_per_chunk = chunk_numel // block_size
        local_y = torch.randn((world_size, M, N), dtype=dtype, device=dev)
        local_s = torch.randn((world_size, num_blocks_per_chunk), dtype=dtype, device=dev)
        return (local_y, local_s, block_size)

    # 21: clip_grad_norm_no_ep
    elif problem_id == 21:
        _seed(problem_id, rank, trial)
        grad_tensors = [torch.randn(base_shape, dtype=dtype, device=dev) for _ in range(3)]
        return (grad_tensors, 1.0, 2.0, None)

    # 22: clip_grad_norm_ep
    elif problem_id == 22:
        _seed(problem_id, rank, trial)
        non_ep = [torch.randn(base_shape, dtype=dtype, device=dev)]
        ep_size = max(1, world_size // 2)
        ep = [torch.randn(base_shape, dtype=dtype, device=dev)]
        return (non_ep, ep, 1.0, 2.0, ep_size, None, None, None)

    # 23: grad_acc_loss
    elif problem_id == 23:
        _seed(problem_id, rank, trial)
        loss = torch.randn((), dtype=dtype, device=dev)
        local_valid = torch.tensor(M * N, dtype=torch.long, device=dev)
        global_valid = torch.tensor(world_size * M * N, dtype=torch.long, device=dev)
        grad_normalized_loss = torch.ones((), dtype=dtype, device=dev)
        grad_loss_sum = torch.zeros((), dtype=dtype, device=dev)
        return (loss, local_valid, global_valid, grad_normalized_loss, grad_loss_sum)

    # 24: load_balancing_loss_fn
    elif problem_id == 24:
        _seed(problem_id, rank, trial)
        num_experts = 8
        gate_logits = torch.randn((M, num_experts), dtype=dtype, device=dev)
        return (gate_logits, num_experts, 2, None)

    # 25: importance_sampling_loss
    elif problem_id == 25:
        _seed(problem_id, rank, trial)
        vocab_size = 32000
        hidden_states = torch.randn((M, N), dtype=dtype, device=dev)
        weight = torch.randn((vocab_size, N), dtype=dtype, device=dev)
        labels = torch.randint(0, vocab_size, (M,), dtype=torch.long, device=dev)
        old_logprobs = torch.randn((M,), dtype=dtype, device=dev)
        advantages = torch.randn((M,), dtype=dtype, device=dev)
        return (hidden_states, weight, labels, old_logprobs, advantages, -100)

    # 26: moe_token_preprocess
    elif problem_id == 26:
        _seed(problem_id, rank, trial)
        num_experts = 8
        topk = 2
        selected_experts = torch.randint(0, num_experts, (M, topk), device=dev)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts).float().permute(2, 1, 0)
        return (expert_mask, num_experts, None)

    # 27: moe_all2all_primitive
    elif problem_id == 27:
        _seed(problem_id, rank, trial)
        local_tokens = M
        hidden_dim = N
        local_tensor = torch.randn((local_tokens, hidden_dim), dtype=dtype, device=dev)
        chunk = local_tokens // world_size
        input_split_sizes = [chunk] * world_size
        if local_tokens % world_size:
            input_split_sizes[-1] += local_tokens % world_size
        output_split_sizes = list(input_split_sizes)
        return (local_tensor, input_split_sizes, output_split_sizes, None)

    # 28: moe_pre_all2all
    elif problem_id == 28:
        _seed(problem_id, rank, trial)
        num_experts = 8
        assert num_experts % world_size == 0, (
            f"problem 28 needs num_experts ({num_experts}) divisible by world_size ({world_size})"
        )
        topk = 2
        hidden_states = torch.randn((M, N), dtype=dtype, device=dev)
        expert_mask = torch.zeros((num_experts, topk, M), dtype=torch.long, device=dev)
        for j in range(M):
            experts = torch.randperm(num_experts, device=dev)[:topk]
            for i, e in enumerate(experts):
                expert_mask[e, i, j] = 1
        expert_mask = expert_mask.float()
        routing_map_bool = expert_mask.sum(dim=1) > 0
        total_permuted = int(routing_map_bool.sum().item())
        chunk = total_permuted // world_size
        input_splits = [chunk] * world_size
        if total_permuted % world_size:
            input_splits[-1] += total_permuted % world_size
        output_splits = list(input_splits)
        num_local_experts = num_experts // world_size
        n_slots = world_size * num_local_experts
        base = total_permuted // n_slots
        rem_tp = total_permuted % n_slots
        flat = torch.full((n_slots,), base, dtype=torch.long, device=dev)
        flat[:rem_tp] += 1
        num_global_tokens_per_local_expert = flat.view(world_size, num_local_experts)
        return (hidden_states, expert_mask, num_experts, input_splits, output_splits, num_global_tokens_per_local_expert, None)

    # 29: moe_post_all2all
    elif problem_id == 29:
        _seed(problem_id, rank, trial)
        num_experts = 8
        assert num_experts % world_size == 0, (
            f"problem 29 needs num_experts ({num_experts}) divisible by world_size ({world_size})"
        )
        topk = 2
        num_tokens = M
        routing_map = torch.zeros((num_experts, num_tokens), dtype=torch.bool, device=dev)
        for j in range(num_tokens):
            experts = torch.randperm(num_experts, device=dev)[:topk]
            routing_map[experts, j] = True
        num_routed = int(routing_map.sum().item())
        routing_weights = torch.zeros((num_tokens, topk), dtype=dtype, device=dev)
        selected_experts = torch.zeros((num_tokens, topk), dtype=torch.long, device=dev)
        for j in range(num_tokens):
            idx = torch.where(routing_map[:, j])[0][:topk]
            selected_experts[j] = idx
            w = torch.randn((topk,), dtype=dtype, device=dev).softmax(dim=0)
            routing_weights[j, :] = w
        expert_outputs = torch.randn((num_routed, N), dtype=dtype, device=dev)
        chunk = num_routed // world_size
        input_splits = [chunk] * world_size
        if num_routed % world_size:
            input_splits[-1] += num_routed % world_size
        output_splits = list(input_splits)
        num_local_experts = num_experts // world_size
        n_slots = world_size * num_local_experts
        base = num_routed // n_slots
        rem_nr = num_routed % n_slots
        flat = torch.full((n_slots,), base, dtype=torch.long, device=dev)
        flat[:rem_nr] += 1
        num_global_tokens_per_local_expert = flat.view(world_size, num_local_experts)
        perm = torch.zeros(num_routed, dtype=torch.long, device=dev)
        idx = 0
        for e in range(num_experts):
            for t in range(num_tokens):
                if routing_map[e, t]:
                    perm[idx] = t
                    idx += 1
        org_hidden_states_shape = torch.Size([num_tokens, N])
        return (expert_outputs, routing_weights, selected_experts, num_experts, input_splits, output_splits, num_global_tokens_per_local_expert, routing_map, perm, org_hidden_states_shape, None)

    # 30: moe_epgroupgemm_lora_backward
    elif problem_id == 30:
        _seed(problem_id, rank, trial)
        r, in_f, out_f = 8, N, N
        grad_fc1_1 = torch.randn((r, in_f), dtype=dtype, device=dev)
        grad_fc1_2 = torch.randn((r, in_f), dtype=dtype, device=dev)
        grad_fc2 = torch.randn((out_f, r), dtype=dtype, device=dev)
        return (grad_fc1_1, grad_fc1_2, grad_fc2, None)

    # 31: fused_moe_fwd
    elif problem_id == 31:
        _seed(problem_id, rank, trial)
        num_experts = 8
        top_k = 2
        hidden_dim = N
        inter_dim = 128
        hidden_states = torch.randn((M, hidden_dim), dtype=dtype, device=dev)
        gate_weight = torch.randn((num_experts, hidden_dim), dtype=dtype, device=dev)
        gate_bias = torch.randn((num_experts,), dtype=dtype, device=dev)
        gate_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        up_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        down_proj = _linear(inter_dim, hidden_dim, dtype, dev)
        return (hidden_states, gate_weight, gate_bias, gate_proj, up_proj, down_proj, num_experts, top_k, None)

    # 32: fused_moe_fwd_lora
    elif problem_id == 32:
        _seed(problem_id, rank, trial)
        num_experts = 8
        top_k = 2
        hidden_dim = N
        inter_dim = 128
        lora_r = 8
        hidden_states = torch.randn((M, hidden_dim), dtype=dtype, device=dev)
        gate_weight = torch.randn((num_experts, hidden_dim), dtype=dtype, device=dev)
        gate_bias = torch.randn((num_experts,), dtype=dtype, device=dev)
        gate_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        up_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        down_proj = _linear(inter_dim, hidden_dim, dtype, dev)
        lora_gate_A = torch.randn((lora_r, hidden_dim), dtype=dtype, device=dev)
        lora_gate_B = torch.randn((inter_dim, lora_r), dtype=dtype, device=dev)
        lora_up_A = torch.randn((lora_r, hidden_dim), dtype=dtype, device=dev)
        lora_up_B = torch.randn((inter_dim, lora_r), dtype=dtype, device=dev)
        lora_down_A = torch.randn((lora_r, inter_dim), dtype=dtype, device=dev)
        lora_down_B = torch.randn((hidden_dim, lora_r), dtype=dtype, device=dev)
        return (
            hidden_states,
            gate_weight,
            gate_bias,
            gate_proj,
            up_proj,
            down_proj,
            lora_gate_A,
            lora_gate_B,
            lora_up_A,
            lora_up_B,
            lora_down_A,
            lora_down_B,
            num_experts,
            top_k,
            None,
        )

    # 33: ulysses_all_to_all_tensor_primitive
    elif problem_id == 33:
        _seed(problem_id, rank, trial)
        x = torch.randn(base_shape, dtype=dtype, device=dev)
        return (x, 0, 1, None)

    # 34: ulysses_all_gather_into_tensor_primitive
    elif problem_id == 34:
        _seed(problem_id, rank, trial)
        x = torch.randn(base_shape, dtype=dtype, device=dev)
        return (x, None)

    # 35: ulysses_all_gather_variable_primitive
    elif problem_id == 35:
        _seed(problem_id, rank, trial)
        x = torch.randn(base_shape, dtype=dtype, device=dev)
        return (x, 0, None)

    # 36: ulysses_gather_seq_scatter_heads
    elif problem_id == 36:
        _seed(problem_id, rank, trial)
        B, T, num_heads, head_dim = _common_attn_dims(base_shape, world_size)
        x = torch.randn((B, T, num_heads, head_dim), dtype=dtype, device=dev)
        return (x, 1, 2, None, 0)

    # 37: ulysses_gather_heads_scatter_seq
    elif problem_id == 37:
        _seed(problem_id, rank, trial)
        B, T, num_heads, head_dim = _common_attn_dims(base_shape, world_size)
        x = torch.randn((B, T, num_heads, head_dim), dtype=dtype, device=dev)
        return (x, 1, 2, None)

    # 38: ulysses_gather_seq_scatter_heads_qkv
    elif problem_id == 38:
        _seed(problem_id, rank, trial)
        B, T, num_heads, head_dim = _common_attn_dims(base_shape, world_size)
        qkv = torch.randn((B, T, 3 * num_heads * head_dim), dtype=dtype, device=dev)
        return (qkv, 1, None, None, True)

    # 39: ulysses_attention_e2e
    elif problem_id == 39:
        _seed(problem_id, rank, trial)
        B, T, num_heads, head_dim = _common_attn_dims(base_shape, world_size)
        S_local = max(1, T // world_size)
        H = num_heads * head_dim
        hidden_states = torch.randn((B, S_local, H), dtype=dtype, device=dev)
        w_qkv = torch.randn((3 * num_heads * head_dim, H), dtype=dtype, device=dev)
        w_o = torch.randn((H, num_heads * head_dim), dtype=dtype, device=dev)
        return (hidden_states, w_qkv, w_o, None, num_heads, False)

    # 40: ddp
    elif problem_id == 40:
        _seed(problem_id, 0, trial)
        n_total = _round_up_multiple(max(M, world_size), world_size)
        chunk = n_total // world_size
        d_in = max(16, min(N, 256))
        hidden = max(32, min(N // 2, 512))
        d_out = max(8, min(N // 4, 256))

        full_X = torch.randn((n_total, d_in), dtype=dtype, device=dev)
        full_y = torch.randn((n_total, d_out), dtype=dtype, device=dev)
        sl = slice(rank * chunk, (rank + 1) * chunk)
        X_local = full_X[sl].contiguous()
        y_local = full_y[sl].contiguous()

        def _init_param(shape: tuple) -> torch.Tensor:
            if rank == 0:
                return torch.randn(shape, dtype=dtype, device=dev)
            return torch.zeros(shape, dtype=dtype, device=dev)

        W1 = _init_param((hidden, d_in))
        b1 = _init_param((hidden,))
        W2 = _init_param((d_out, hidden))
        b2 = _init_param((d_out,))

        z = torch.zeros
        exp_avg_W1 = z((hidden, d_in), dtype=dtype, device=dev)
        exp_avg_b1 = z((hidden,), dtype=dtype, device=dev)
        exp_avg_W2 = z((d_out, hidden), dtype=dtype, device=dev)
        exp_avg_b2 = z((d_out,), dtype=dtype, device=dev)
        exp_avg_sq_W1 = z((hidden, d_in), dtype=dtype, device=dev)
        exp_avg_sq_b1 = z((hidden,), dtype=dtype, device=dev)
        exp_avg_sq_W2 = z((d_out, hidden), dtype=dtype, device=dev)
        exp_avg_sq_b2 = z((d_out,), dtype=dtype, device=dev)

        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        adam_step = 1 + (trial % 7)
        return (
            X_local,
            y_local,
            W1,
            b1,
            W2,
            b2,
            exp_avg_W1,
            exp_avg_b1,
            exp_avg_W2,
            exp_avg_b2,
            exp_avg_sq_W1,
            exp_avg_sq_b1,
            exp_avg_sq_W2,
            exp_avg_sq_b2,
            lr,
            beta1,
            beta2,
            eps,
            adam_step,
        )

    # 41: zero1_optimizer_shard
    elif problem_id == 41:
        _seed(problem_id, 0, trial)
        n_total = _round_up_multiple(max(M, world_size), world_size)
        chunk = n_total // world_size
        d_in, hidden, d_out = _ddp_mlp_shapes_divisible_by_dp(N, world_size)
        part_numel = (hidden * d_in + hidden + d_out * hidden + d_out) // world_size

        full_X = torch.randn((n_total, d_in), dtype=dtype, device=dev)
        full_y = torch.randn((n_total, d_out), dtype=dtype, device=dev)
        sl = slice(rank * chunk, (rank + 1) * chunk)
        X_local = full_X[sl].contiguous()
        y_local = full_y[sl].contiguous()

        def _init_param(shape: tuple) -> torch.Tensor:
            if rank == 0:
                return torch.randn(shape, dtype=dtype, device=dev)
            return torch.zeros(shape, dtype=dtype, device=dev)

        W1 = _init_param((hidden, d_in))
        b1 = _init_param((hidden,))
        W2 = _init_param((d_out, hidden))
        b2 = _init_param((d_out,))

        z = torch.zeros
        exp_avg_part = z((part_numel,), dtype=dtype, device=dev)
        exp_avg_sq_part = z((part_numel,), dtype=dtype, device=dev)

        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        adam_step = 1 + (trial % 7)
        return (
            X_local,
            y_local,
            W1,
            b1,
            W2,
            b2,
            exp_avg_part,
            exp_avg_sq_part,
            lr,
            beta1,
            beta2,
            eps,
            adam_step,
        )

    # 42: zero2_optimizer_shard_grad
    elif problem_id == 42:
        _seed(problem_id, 0, trial)
        n_total = _round_up_multiple(max(M, world_size), world_size)
        chunk = n_total // world_size
        d_in, hidden, d_out = _ddp_mlp_shapes_divisible_by_dp(N, world_size)
        part_numel = (hidden * d_in + hidden + d_out * hidden + d_out) // world_size

        full_X = torch.randn((n_total, d_in), dtype=dtype, device=dev)
        full_y = torch.randn((n_total, d_out), dtype=dtype, device=dev)
        sl = slice(rank * chunk, (rank + 1) * chunk)
        X_local = full_X[sl].contiguous()
        y_local = full_y[sl].contiguous()

        def _init_param(shape: tuple) -> torch.Tensor:
            if rank == 0:
                return torch.randn(shape, dtype=dtype, device=dev)
            return torch.zeros(shape, dtype=dtype, device=dev)

        W1 = _init_param((hidden, d_in))
        b1 = _init_param((hidden,))
        W2 = _init_param((d_out, hidden))
        b2 = _init_param((d_out,))

        z = torch.zeros
        exp_avg_part = z((part_numel,), dtype=dtype, device=dev)
        exp_avg_sq_part = z((part_numel,), dtype=dtype, device=dev)

        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        adam_step = 1 + (trial % 7)
        return (
            X_local,
            y_local,
            W1,
            b1,
            W2,
            b2,
            exp_avg_part,
            exp_avg_sq_part,
            lr,
            beta1,
            beta2,
            eps,
            adam_step,
        )

    # 43: fused_adam_grad_unshard_allgather
    elif problem_id == 43:
        _seed(problem_id, 0, trial)
        P = max(64, min(M * 64, 4096))
        full_grad = torch.randn(P * world_size, dtype=dtype, device=dev)
        grad_shard = full_grad[rank * P : (rank + 1) * P].contiguous()
        full_master = torch.randn(P * world_size, dtype=dtype, device=dev)
        master_shard = full_master[rank * P : (rank + 1) * P].contiguous()
        exp_avg = torch.zeros(P, dtype=dtype, device=dev)
        exp_avg_sq = torch.zeros(P, dtype=dtype, device=dev)
        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        adam_step = 1 + (trial % 7)
        return (
            grad_shard,
            master_shard,
            exp_avg,
            exp_avg_sq,
            lr,
            beta1,
            beta2,
            eps,
            adam_step,
        )

    # 44: quantized_grad_allreduce
    elif problem_id == 44:
        _seed(problem_id, rank, trial)
        n_el = max(M * N, world_size * 64)
        flat_grad = torch.randn((n_el,), dtype=dtype, device=dev)
        block_size = min(128, max(16, max(N // 4, 16)))
        return (flat_grad, block_size)

    # 45: reducescatter_fused_rmsnorm
    elif problem_id == 45:
        _seed(problem_id, 0, trial)
        hidden = max(32, min(N, 128))
        rows = max(2, max(M, world_size) // max(world_size, 4))
        chunk = rows * hidden
        gamma = torch.randn((hidden,), dtype=dtype, device=dev)
        _seed(problem_id, rank, trial)
        rs_input = torch.randn((chunk * world_size,), dtype=dtype, device=dev)
        eps = 1e-5
        return (rs_input, gamma, eps)

    # 46: fsdp_adamw_sharded
    elif problem_id == 46:
        _seed(problem_id, 0, trial)
        d_in, hidden, d_out = _ddp_mlp_shapes_divisible_by_dp(N, world_size)
        total_numel = hidden * d_in + hidden + d_out * hidden + d_out
        part = total_numel // world_size

        full_param = torch.randn(total_numel, dtype=dtype, device=dev)
        flat_param_shard = full_param[rank * part : (rank + 1) * part].contiguous()
        full_grad = torch.randn(total_numel, dtype=dtype, device=dev)
        flat_grad_shard = full_grad[rank * part : (rank + 1) * part].contiguous()
        exp_avg_shard = torch.zeros(part, dtype=dtype, device=dev)
        exp_avg_sq_shard = torch.zeros(part, dtype=dtype, device=dev)

        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        weight_decay = 0.01
        adam_step = 1 + (trial % 7)
        return (
            flat_param_shard,
            flat_grad_shard,
            exp_avg_shard,
            exp_avg_sq_shard,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            adam_step,
        )

    # 47: fsdp_step_e2e
    elif problem_id == 47:
        _seed(problem_id, 0, trial)
        n_total = _round_up_multiple(max(M, world_size), world_size)
        chunk = n_total // world_size
        d_in, hidden, d_out = _ddp_mlp_shapes_divisible_by_dp(N, world_size)
        total_numel = hidden * d_in + hidden + d_out * hidden + d_out
        part = total_numel // world_size

        full_X = torch.randn((n_total, d_in), dtype=dtype, device=dev)
        full_y = torch.randn((n_total, d_out), dtype=dtype, device=dev)
        sl = slice(rank * chunk, (rank + 1) * chunk)
        X_local = full_X[sl].contiguous()
        y_local = full_y[sl].contiguous()

        def _init_param(shape: tuple) -> torch.Tensor:
            if rank == 0:
                return torch.randn(shape, dtype=dtype, device=dev)
            return torch.zeros(shape, dtype=dtype, device=dev)

        W1 = _init_param((hidden, d_in))
        b1 = _init_param((hidden,))
        W2 = _init_param((d_out, hidden))
        b2 = _init_param((d_out,))

        full_fp = torch.cat([W1.reshape(-1), b1.reshape(-1), W2.reshape(-1), b2.reshape(-1)])
        flat_param_shard = full_fp[rank * part : (rank + 1) * part].contiguous()

        exp_avg_shard = torch.zeros(part, dtype=dtype, device=dev)
        exp_avg_sq_shard = torch.zeros(part, dtype=dtype, device=dev)

        param_shapes = ((hidden, d_in), (hidden,), (d_out, hidden), (d_out,))
        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        weight_decay = 0.01
        adam_step = 1 + (trial % 7)
        return (
            X_local,
            y_local,
            flat_param_shard,
            param_shapes,
            exp_avg_shard,
            exp_avg_sq_shard,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            adam_step,
        )

    # 48: fsdp_and_tp
    elif problem_id == 48:
        _seed(problem_id, 0, trial)
        n_tp, n_fsdp = _factor_tp_fsdp(world_size)
        base_d = max(32, min(N, 256))
        D = _round_up_multiple(base_d, math.lcm(n_tp, n_fsdp))
        D_ff = _round_up_multiple(max(64, M), n_tp)
        B_total = _round_up_multiple(max(M * 2, world_size * 2), n_fsdp)
        B_fsdp = B_total // n_fsdp

        tp_rank = rank % n_tp
        fsdp_rank = rank // n_tp

        full_x = torch.randn(B_total, D, dtype=dtype, device=dev)
        x_local = full_x[fsdp_rank * B_fsdp : (fsdp_rank + 1) * B_fsdp].contiguous()

        full_W1 = torch.randn(D, D_ff, dtype=dtype, device=dev)
        full_W2 = torch.randn(D, D_ff, dtype=dtype, device=dev)
        full_W3 = torch.randn(D_ff, D, dtype=dtype, device=dev)

        dr = D // n_fsdp
        dc = D_ff // n_tp
        rr = D_ff // n_tp
        cr = D // n_fsdp

        W1_shard = full_W1[
            fsdp_rank * dr : (fsdp_rank + 1) * dr, tp_rank * dc : (tp_rank + 1) * dc
        ].contiguous()
        W2_shard = full_W2[
            fsdp_rank * dr : (fsdp_rank + 1) * dr, tp_rank * dc : (tp_rank + 1) * dc
        ].contiguous()
        W3_shard = full_W3[
            tp_rank * rr : (tp_rank + 1) * rr, fsdp_rank * cr : (fsdp_rank + 1) * cr
        ].contiguous()

        return (x_local, W1_shard, W2_shard, W3_shard, n_tp, n_fsdp)

    # 49: moe_ep_balanced
    elif problem_id == 49:
        _seed(problem_id, rank, trial)
        num_experts = max(1, world_size)
        top_k = 2
        hidden_dim = N
        inter_dim = 128
        hidden_states = torch.randn((M, hidden_dim), dtype=dtype, device=dev)
        gate_weight = torch.randn((num_experts, hidden_dim), dtype=dtype, device=dev)
        gate_bias = torch.randn((num_experts,), dtype=dtype, device=dev)
        gate_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        up_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        down_proj = _linear(inter_dim, hidden_dim, dtype, dev)
        return (
            hidden_states,
            gate_weight,
            gate_bias,
            gate_proj,
            up_proj,
            down_proj,
            num_experts,
            top_k,
            None,
        )

    # 50: moe_ep_wide
    elif problem_id == 50:
        _seed(problem_id, rank, trial)
        num_experts = world_size * 2
        top_k = 2
        hidden_dim = N
        inter_dim = 128
        hidden_states = torch.randn((M, hidden_dim), dtype=dtype, device=dev)
        gate_weight = torch.randn((num_experts, hidden_dim), dtype=dtype, device=dev)
        gate_bias = torch.randn((num_experts,), dtype=dtype, device=dev)
        gate_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        up_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        down_proj = _linear(inter_dim, hidden_dim, dtype, dev)
        return (
            hidden_states,
            gate_weight,
            gate_bias,
            gate_proj,
            up_proj,
            down_proj,
            num_experts,
            top_k,
            None,
        )

    # 51: moe_ep_narrow
    elif problem_id == 51:
        _seed(problem_id, rank, trial)
        num_experts = _moe_narrow_num_experts(world_size)
        top_k = min(2, num_experts)
        hidden_dim = N
        inter_dim = 128
        hidden_states = torch.randn((M, hidden_dim), dtype=dtype, device=dev)
        gate_weight = torch.randn((num_experts, hidden_dim), dtype=dtype, device=dev)
        gate_bias = torch.randn((num_experts,), dtype=dtype, device=dev)
        gate_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        up_proj = _linear(hidden_dim, inter_dim, dtype, dev)
        down_proj = _linear(inter_dim, hidden_dim, dtype, dev)
        return (
            hidden_states,
            gate_weight,
            gate_bias,
            gate_proj,
            up_proj,
            down_proj,
            num_experts,
            top_k,
            None,
        )

    # 52: fp8_reduce_scatter_grads
    elif problem_id == 52:
        _seed(problem_id, rank, trial)
        P = max(64, min(M * 64, 4096))
        flat_grads = torch.randn(P * world_size, dtype=dtype, device=dev)
        amax_history = torch.full((16,), 1e-8, dtype=torch.bfloat16, device=dev)
        return (flat_grads, amax_history)

    # 53: fp8_allgather_params
    elif problem_id == 53:
        _seed(problem_id, rank, trial)
        P = max(64, min(M * 64, 4096))
        flat_param_shard = torch.randn(P, dtype=dtype, device=dev)
        amax_history = torch.full((16,), 1e-8, dtype=torch.bfloat16, device=dev)
        return (flat_param_shard, amax_history)

    # 54: ring_attention
    elif problem_id == 54:
        _seed(problem_id, rank, trial)
        B, T, num_heads, head_dim = _common_attn_dims(base_shape, world_size)
        S_local = max(1, T // world_size)
        q = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)
        k = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)
        v = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)
        return (q, k, v, None, True, None)
 
    # 55: ring_attention_tp
    elif problem_id == 55:
        _seed(problem_id, rank, trial)
        num_heads = 8
        head_dim = 64
        hidden_size = num_heads * head_dim
        tp_size = min(2, world_size)
        assert world_size % tp_size == 0
        assert num_heads % tp_size == 0
        tp_group, cp_group, cp_size = _build_tp_cp_groups(tp_size)
        tp_rank = dist.get_rank(tp_group)
        cp_rank = dist.get_rank(cp_group)
 
        B = max(1, M // 64)
        T = max(1, N // 64)
        S_local = max(1, T // cp_size)
        heads_local = num_heads // tp_size
 
        torch.manual_seed(42 + 56 * 1000 + cp_rank + trial * 1_000_003)
        hidden_states = torch.randn((B, S_local, hidden_size), dtype=dtype, device=dev)
        torch.manual_seed(42 + 56 * 1000 + 10000 + tp_rank + trial * 1_000_003)
        w_qkv = torch.randn((3 * heads_local * head_dim, hidden_size), dtype=dtype, device=dev) * 0.02
        w_o = torch.randn((hidden_size, heads_local * head_dim), dtype=dtype, device=dev) * 0.02
 
        return (hidden_states, w_qkv, w_o, num_heads, None, True, tp_group, cp_group)
 
    # 56: ring_attention_pp
    elif problem_id == 56:
        _seed(problem_id, rank, trial)
        num_heads = 8
        head_dim = 64
        hidden_size = num_heads * head_dim
        pp_size = min(2, world_size)
        assert world_size % pp_size == 0
        cp_group, pp_group, cp_rank, pp_rank, cp_size = _build_cp_pp_groups(pp_size)
 
        B = max(1, M // 64)
        T = max(1, N // 64)
        S_local = max(1, T // cp_size)
 
        torch.manual_seed(42 + 57 * 1000 + cp_rank + trial * 1_000_003)
        hidden_states = torch.randn((B, S_local, hidden_size), dtype=dtype, device=dev)
        torch.manual_seed(42 + 57 * 1000 + 20000 + pp_rank + trial * 1_000_003)
        w_qkv = torch.randn((3 * num_heads * head_dim, hidden_size), dtype=dtype, device=dev) * 0.02
        w_o = torch.randn((hidden_size, num_heads * head_dim), dtype=dtype, device=dev) * 0.02
 
        return (hidden_states, w_qkv, w_o, num_heads, None, True, cp_group, pp_group)
 
    # 57: ring_attention_backward_dp
    elif problem_id == 57:
        _seed(problem_id, rank, trial)
        B, T, num_heads, head_dim = _common_attn_dims(base_shape, world_size)
        dp_size = min(2, world_size)
        assert world_size % dp_size == 0
        cp_group, dp_group, _, cp_rank, dp_rank, cp_size = _build_cp_dp_groups(dp_size)
        S_local = max(1, T // cp_size)

        torch.manual_seed(42 + 58 * 1000 + dp_rank * 100 + cp_rank + trial * 1_000_003)
        q = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)
        k = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)
        v = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)
        dout = torch.randn((B, S_local, num_heads, head_dim), dtype=dtype, device=dev)

        scale = head_dim ** -0.5
        qh = q.transpose(1, 2).float()
        kh = k.transpose(1, 2).float()
        vh = v.transpose(1, 2).float()
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        softmax_lse = torch.logsumexp(scores, dim=-1)
        out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2).contiguous()
        out = out.to(dtype)

        return (dout, q, k, v, out, softmax_lse, None, False, cp_group, dp_group)

    # 58: openclip_contrastive_loss
    elif problem_id == 58:
        _seed(problem_id, rank, trial)
        B_local = max(1, M // max(world_size, 1))
        D = max(16, N)
        image_features = torch.randn((B_local, D), dtype=dtype, device=dev)
        text_features = torch.randn((B_local, D), dtype=dtype, device=dev)
        image_features = torch.nn.functional.normalize(image_features, dim=-1).contiguous()
        text_features = torch.nn.functional.normalize(text_features, dim=-1).contiguous()
        logit_scale = 10.0
        logit_bias = -10.0
        return (image_features, text_features, logit_scale, logit_bias, None)

    # 59: physicsnemo_distributed_rfft
    elif problem_id == 59:
        _seed(problem_id, 0, trial)
        B = max(1, M // 64)
        H = _round_up_multiple(max(16, M), world_size)
        W = _round_up_multiple(max(16, N), world_size)
        W_local = W // world_size
        x_full = torch.randn((B, H, W), dtype=torch.float32, device=dev)
        x = x_full[:, :, rank * W_local : (rank + 1) * W_local].contiguous()
        return (x, (H, W), (1, 2), "ortho", None)

    # 60: physicsnemo_distributed_irfft
    elif problem_id == 60:
        _seed(problem_id, 0, trial)
        B = max(1, M // 64)
        H = _round_up_multiple(max(16, M), world_size)
        W = _round_up_multiple(max(16, N), world_size)
        H_local = H // world_size
        x_real = torch.randn((B, H, W), dtype=torch.float32, device=dev)
        x_full = torch.fft.rfft2(x_real, s=(H, W), dim=(1, 2), norm="ortho")
        x = x_full[:, rank * H_local : (rank + 1) * H_local, :].contiguous()
        return (x, (H, W), (1, 2), "ortho", None)

    # 61: gsplat_3d_gaussian_splatting
    elif problem_id == 61:
        _seed(problem_id, rank, trial)
        n_local = max(8, min(M, 256))
        channels = 3
        image_width = int(max(64, min(N, 512)))
        image_height = int(max(64, min(M, 512)))

        means = torch.empty((n_local, 3), dtype=torch.bfloat16, device=dev)
        means[:, 0] = (torch.rand(n_local, dtype=torch.bfloat16, device=dev) - 0.5) * 1.5
        means[:, 1] = (torch.rand(n_local, dtype=torch.bfloat16, device=dev) - 0.5) * 1.5
        means[:, 2] = torch.rand(n_local, dtype=torch.bfloat16, device=dev) * 2.0 + 2.0
        quats = torch.randn((n_local, 4), dtype=torch.bfloat16, device=dev)
        scales = torch.rand((n_local, 3), dtype=torch.bfloat16, device=dev) * 0.04 + 0.02
        opacities = torch.rand((n_local,), dtype=torch.bfloat16, device=dev) * 0.8 + 0.1
        colors = torch.rand((n_local, channels), dtype=torch.bfloat16, device=dev)

        viewmats = torch.eye(4, dtype=torch.bfloat16, device=dev).reshape(1, 4, 4).contiguous()
        Ks = torch.eye(3, dtype=torch.bfloat16, device=dev).reshape(1, 3, 3).contiguous()
        focal = 0.8 * float(min(image_width, image_height))
        Ks[:, 0, 0] = focal
        Ks[:, 1, 1] = focal
        Ks[:, 0, 2] = image_width * 0.5
        Ks[:, 1, 2] = image_height * 0.5
        return (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            image_width,
            image_height,
            0.3,
            0.01,
            1e10,
            "pinhole",
        )

    # 62: torchharmonics_spherical_convolution
    elif problem_id == 62:
        azimuth_size = 2 if world_size % 2 == 0 else 1
        azimuth_group, polar_group, azimuth_rank, polar_rank, _, polar_size = _build_polar_azimuth_groups(
            azimuth_size
        )
        _seed(problem_id, rank, trial)
        batch = max(1, M // 256)
        in_channels = 8
        out_channels = 8
        groups = 1
        kernel_size = 3
        nlat_in = _round_up_multiple(max(8, min(M // 64, 32)), polar_size)
        nlon_in = _round_up_multiple(max(8, min(N // 64, 32)), azimuth_size)
        nlat_out = nlat_in
        nlon_out = nlon_in

        lat_shapes = _round_up_multiple(nlat_in, polar_size) // polar_size
        lon_shapes = _round_up_multiple(nlon_in, azimuth_size) // azimuth_size
        nlat_local = lat_shapes
        nlon_local = lon_shapes

        x = torch.randn((batch, in_channels, nlat_local, nlon_local), dtype=torch.float32, device=dev)
        weight = torch.randn((out_channels, in_channels // groups, kernel_size), dtype=torch.float32, device=dev)
        bias = torch.randn((out_channels,), dtype=torch.float32, device=dev)

        entries_per_row = min(4, nlat_local * nlon_in)
        nnz = kernel_size * nlat_out * entries_per_row
        idx = torch.empty((3, nnz), dtype=torch.long, device=dev)
        vals = torch.randn((nnz,), dtype=torch.float32, device=dev) * 0.05
        cursor = 0
        lat_offset = polar_rank * nlat_local
        for k_idx in range(kernel_size):
            for out_lat in range(nlat_out):
                local_lat = (out_lat - lat_offset) % nlat_local
                for e in range(entries_per_row):
                    lon = (out_lat + e * (k_idx + 1)) % nlon_in
                    idx[0, cursor] = k_idx
                    idx[1, cursor] = out_lat
                    idx[2, cursor] = local_lat * nlon_in + lon
                    cursor += 1
        psi = torch.sparse_coo_tensor(
            idx,
            vals,
            size=(kernel_size, nlat_out, nlat_local * nlon_in),
            device=dev,
        ).coalesce()

        return (x, psi, weight, groups, nlon_out, nlon_in, azimuth_group, polar_group, bias)

    # 63: deepmd_kalman_filter_optimizer
    elif problem_id == 63:
        _seed(problem_id, rank, trial)
        num_blocks = 4
        block = max(8, min(M // 8, 64))
        H = []
        weights = []
        P = []
        for _ in range(num_blocks):
            h = torch.randn((block, 1), dtype=torch.float64, device=dev) * 0.01
            w = torch.randn((block, 1), dtype=torch.float64, device=dev)
            p = torch.eye(block, dtype=torch.float64, device=dev)
            H.append(h)
            weights.append(w)
            P.append(p)
        error = torch.randn((1, 1), dtype=torch.float64, device=dev)
        kalman_lambda = 0.98
        kalman_nue = 0.9987
        return (H, error, weights, P, kalman_lambda, kalman_nue)

    # 64: gnn_neighbor_sampling
    elif problem_id == 64:
        _seed(problem_id, 0, trial)
        num_nodes = _round_up_multiple(max(64, min(M, 1024)), world_size)
        degree = 4
        fanouts = [3, 2]
        node_to_rank = (torch.arange(num_nodes, device=dev, dtype=torch.long) % world_size).contiguous()

        row_chunks = []
        colptr = torch.empty((num_nodes + 1,), dtype=torch.long, device=dev)
        colptr[0] = 0
        for node_idx in range(num_nodes):
            nbrs = (torch.arange(1, degree + 1, device=dev, dtype=torch.long) + node_idx) % num_nodes
            row_chunks.append(nbrs)
            colptr[node_idx + 1] = colptr[node_idx] + degree
        row = torch.cat(row_chunks).contiguous()

        seeds_per_rank = max(4, min(N // max(world_size * 16, 1), 32))
        start = rank * seeds_per_rank
        seed_nodes = (torch.arange(seeds_per_rank, device=dev, dtype=torch.long) + start) % num_nodes
        return (seed_nodes.contiguous(), fanouts, colptr.contiguous(), row, node_to_rank, None, False)

    # 65: gnn_feature_exchange_all2all
    elif problem_id == 65:
        _seed(problem_id, rank, trial)
        rows_per_peer = max(1, min(M // max(world_size * 64, 1), 8))
        hidden = max(8, min(N, 128))
        seed_size = rows_per_peer * world_size
        local_features = torch.randn((seed_size, hidden), dtype=dtype, device=dev)
        seed_inverse_ids = torch.arange(seed_size, dtype=torch.long, device=dev)
        counts_sent = [rows_per_peer for _ in range(world_size)]
        counts_received = [rows_per_peer for _ in range(world_size)]
        return (local_features, seed_inverse_ids, counts_sent, counts_received, None)

    # 66: gnn_feature_exchange_all2all_backward
    elif problem_id == 66:
        _seed(problem_id, rank, trial)
        rows_per_peer = max(1, min(M // max(world_size * 64, 1), 8))
        hidden = max(8, min(N, 128))
        seed_size = rows_per_peer * world_size
        grad_output = torch.randn((seed_size, hidden), dtype=torch.float32, device=dev)
        seed_inverse_ids = torch.arange(seed_size, dtype=torch.long, device=dev)
        counts_sent = [rows_per_peer for _ in range(world_size)]
        counts_received = [rows_per_peer for _ in range(world_size)]
        return (grad_output, seed_inverse_ids, seed_size, counts_sent, counts_received, None)

    # 67: gnn_sparse_embedding_all2all
    elif problem_id == 67:
        _seed(problem_id, rank, trial)
        num_nodes = _round_up_multiple(max(1024, M), world_size)
        nnz = max(16, min(M // max(world_size, 1), 512))
        hidden = max(8, min(N, 128))
        base = torch.arange(nnz, dtype=torch.long, device=dev)
        idx = (base * world_size + rank + torch.randint(0, world_size, (nnz,), device=dev)) % num_nodes
        value = torch.randn((nnz, hidden), dtype=dtype, device=dev)
        return (idx.contiguous(), value.contiguous(), num_nodes, None)

    # 68: gnn_sparse_feature_fetch_projection
    elif problem_id == 68:
        _seed(problem_id, rank, trial)
        num_total_nodes = _round_up_multiple(max(1024, M), world_size)
        shard_size = num_total_nodes // world_size
        embed_dim = max(16, min(N, 128))
        out_dim = max(8, embed_dim // 2)
        local_embedding_shard = torch.randn((shard_size, embed_dim), dtype=dtype, device=dev)
        proj_matrix = torch.randn((embed_dim, out_dim), dtype=dtype, device=dev)

        num_queries = max(16, min(M // max(world_size, 1), 512))
        base = torch.arange(num_queries, dtype=torch.long, device=dev)
        owner = (base + rank) % world_size
        local = (base * 7 + rank) % shard_size
        input_node_ids = owner * shard_size + local
        return (
            local_embedding_shard.contiguous(),
            input_node_ids.contiguous(),
            proj_matrix.contiguous(),
            num_total_nodes,
            None,
        )

    # 69: gnn_negative_scoring
    elif problem_id == 69:
        _seed(problem_id, rank, trial)
        num_pos = max(8, min(M // max(world_size * 4, 1), 256)) + rank % 3
        num_neg = max(4, min(N, 64))
        local_pos_scores = torch.randn((num_pos,), dtype=dtype, device=dev)
        local_neg_scores = torch.randn(
            (num_pos, num_neg), dtype=dtype, device=dev
        )
        return (local_pos_scores.contiguous(), local_neg_scores.contiguous(), None)

    # 70: torchrec_kjt_all2all
    elif problem_id == 70:
        _seed(problem_id, rank, trial)
        key_splits = [1 + (dst % 3) for dst in range(world_size)]
        num_features = sum(key_splits)
        batch_size = max(2, min(M // max(world_size * 64, 1), 16))
        base = torch.arange(
            num_features * batch_size, dtype=torch.long, device=dev
        ).view(num_features, batch_size)
        lengths_2d = ((base + rank) % 4).to(torch.long)
        lengths = lengths_2d.reshape(-1).contiguous()
        values = torch.arange(
            int(lengths.sum().item()), dtype=torch.long, device=dev
        )
        values = values + rank * max(1, values.numel())
        return (lengths, values.contiguous(), key_splits, batch_size, None)

    # 71: hyena_conv1d_boundary_exchange
    elif problem_id == 71:
        _seed(problem_id, rank, trial)
        batch = 1
        channels = 1024
        kernel_size = 7
        local_chunk = 1024
        x = torch.randn((batch, channels, 2 * local_chunk), dtype=dtype, device=dev)
        weight = torch.randn((channels, 1, kernel_size), dtype=dtype, device=dev)
        return (x.contiguous(), weight.contiguous(), None)

    # 72: hyena_forward_cp
    elif problem_id == 72:
        _seed(problem_id, rank, trial)
        batch = 1
        channels = 1024
        group_dim = 1
        num_groups = channels // group_dim
        local_seq = 2048
        filter_len = 4096

        x1_seq = torch.randn((batch, channels, local_seq), dtype=dtype, device=dev)
        x2_seq = torch.randn((batch, channels, local_seq), dtype=dtype, device=dev)
        v_seq = torch.randn((batch, channels, local_seq), dtype=dtype, device=dev)
        h_base = torch.arange(num_groups * filter_len, dtype=torch.bfloat16, device=dev)
        h = (h_base.reshape(num_groups, filter_len) / max(filter_len, 1)).to(dtype)
        bias_base = torch.arange(channels, dtype=torch.bfloat16, device=dev)
        conv_bias = (bias_base / max(channels, 1)).to(dtype)
        return (
            x1_seq.contiguous(),
            x2_seq.contiguous(),
            v_seq.contiguous(),
            h.contiguous(),
            conv_bias.contiguous(),
            num_groups,
            group_dim,
            None,
            True,
        )

    # 73: vocab_parallel_cross_entropy_loss
    elif problem_id == 73:
        _seed(problem_id, rank, trial)
        batch = 8
        seq_len = 1024
        vocab_size = 512
        partition_vocab_size = vocab_size // world_size

        logits = torch.randn(
            (batch, seq_len, partition_vocab_size),
            dtype=dtype,
            device=dev,
        )
        token_ids = torch.arange(batch * seq_len, dtype=torch.long, device=dev)
        target = (token_ids * 13 + 7 + trial).remainder(vocab_size)
        target = target.reshape(batch, seq_len)
        return (logits.contiguous(), target.contiguous(), None)

    # 74: fla_kimi_delta_attention_cp_tp
    elif problem_id == 74:
        _seed(problem_id, rank, trial)
        if world_size >= 4 and world_size % 2 == 0:
            tp_group, cp_group, cp_size = _build_tp_cp_groups(tp_size=2)
            tp_arg = tp_group
        else:
            cp_group = dist.group.WORLD
            cp_size = world_size
            tp_arg = None

        batch = 1
        local_seq = 64
        num_heads = 16
        key_dim = 128
        value_dim = 128

        q = torch.randn((batch, local_seq, num_heads, key_dim), dtype=dtype, device=dev)
        k = torch.randn((batch, local_seq, num_heads, key_dim), dtype=dtype, device=dev)
        v = torch.randn((batch, local_seq, num_heads, value_dim), dtype=dtype, device=dev)
        g = torch.randn((batch, local_seq, num_heads, key_dim), dtype=dtype, device=dev)
        beta = torch.randn((batch, local_seq, num_heads), dtype=dtype, device=dev)
        a_log = torch.linspace(-0.1, 0.1, num_heads, dtype=torch.bfloat16, device=dev)
        dt_bias = torch.linspace(
            -0.1, 0.1, num_heads * key_dim, dtype=torch.bfloat16, device=dev
        )
        return (
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            a_log.contiguous(),
            dt_bias.contiguous(),
            cp_group,
            tp_arg,
        )

    # 75: fla_gated_deltanet_cp
    elif problem_id == 75:
        _seed(problem_id, rank, trial)
        batch = 1
        local_seq = 64
        num_heads = _round_up_multiple(6, world_size)
        num_value_heads = num_heads
        key_dim = 256
        value_dim = 512

        q = torch.randn((batch, local_seq, num_heads, key_dim), dtype=dtype, device=dev)
        k = torch.randn((batch, local_seq, num_heads, key_dim), dtype=dtype, device=dev)
        v = torch.randn(
            (batch, local_seq, num_value_heads, value_dim), dtype=dtype, device=dev
        )
        gate = torch.randn(
            (batch, local_seq, num_value_heads), dtype=dtype, device=dev
        )
        beta = torch.randn(
            (batch, local_seq, num_value_heads), dtype=torch.bfloat16, device=dev
        ).sigmoid().to(dtype)
        local_value_heads = num_value_heads // world_size
        head_start = rank * local_value_heads
        head_end = head_start + local_value_heads
        full_a = torch.linspace(0.0, 0.2, num_value_heads, dtype=torch.bfloat16, device=dev)
        full_dt = torch.linspace(
            -0.1, 0.1, num_value_heads, dtype=torch.bfloat16, device=dev
        )
        local_a = full_a[head_start:head_end]
        local_dt = full_dt[head_start:head_end]
        return (
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            local_a.contiguous(),
            local_dt.contiguous(),
            None,
        )

    # 76: opensora_conv3d_allreduce
    elif problem_id == 76:
        _seed(problem_id, rank, trial)
        batch = 1
        out_channels = 512
        for channels in (512, 256, 128):
            if channels % world_size == 0:
                out_channels = channels
                break
        local_in_channels = out_channels // world_size
        time = 19
        height = 66
        width = 66
        kernel = 3

        x = torch.randn(
            (batch, local_in_channels, time, height, width),
            dtype=dtype,
            device=dev,
        )
        weight = torch.randn(
            (out_channels, local_in_channels, kernel, kernel, kernel),
            dtype=dtype,
            device=dev,
        )
        bias = torch.linspace(-0.1, 0.1, out_channels, dtype=dtype, device=dev)
        return (
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            1,
            0,
            1,
            1,
            None,
        )

    # 77: magi1_cso_async_attention
    elif problem_id == 77:
        _seed(problem_id, rank, trial)
        cp_shuffle_num = 4
        head_dim = 128
        if world_size <= 4:
            total_q_heads = 24
            chunk_token_nums = 12_150
        else:
            total_q_heads = 48
            chunk_token_nums = 21_600
        if total_q_heads % world_size != 0:
            total_q_heads = world_size * max(1, total_q_heads // world_size)
        total_kv_heads = 8
        if total_kv_heads % world_size != 0 and world_size % total_kv_heads != 0:
            total_kv_heads = world_size
        tokens_per_range = (chunk_token_nums + world_size - 1) // world_size
        total_tokens = cp_shuffle_num * tokens_per_range
        attn_dtype = torch.bfloat16 if dtype in (torch.float16, torch.bfloat16, torch.bfloat16) else dtype

        query = torch.randn(
            (total_tokens, total_q_heads, head_dim),
            dtype=attn_dtype,
            device=dev,
        )
        key = torch.randn(
            (total_tokens, total_kv_heads, head_dim),
            dtype=attn_dtype,
            device=dev,
        )
        value = torch.randn_like(key)
        key_value = torch.cat([key, value], dim=-1)
        starts = (
            torch.arange(cp_shuffle_num, dtype=torch.long, device=dev)
            * chunk_token_nums
        )
        ends = starts + chunk_token_nums
        k_ranges = torch.stack([starts, ends], dim=1)
        return (
            query.contiguous(),
            key_value.contiguous(),
            k_ranges.contiguous(),
            cp_shuffle_num,
            chunk_token_nums,
            None,
        )

    # 78: magi1_tile_parallel_vae_decode
    elif problem_id == 78:
        _seed(problem_id, rank, trial)
        batch = 1
        channels = 16
        time = 6
        height = 90
        width = 90
        z = torch.randn(
            (batch, channels, time, height, width),
            dtype=torch.bfloat16,
            device=dev,
        )
        return (
            z.contiguous(),
            3,
            32,
            32,
            0.25,
            0.0,
            8,
            4,
            1,
            False,
            None,
        )

    # 79: dinov2_distributed_knn
    elif problem_id == 79:
        _seed(problem_id, rank, trial)
        local_queries = max(32, min(M // 64, 256))
        local_train = max(1_024, min(N * 16 // max(world_size, 1), 16_384))
        feature_dim = 384
        if N >= 4_096:
            feature_dim = 768
        if N >= 8_192:
            feature_dim = 1_024
        max_k = min(200, local_train)

        test_features = torch.randn(
            (local_queries, feature_dim), dtype=torch.bfloat16, device=dev
        )
        train_features = torch.randn(
            (local_train, feature_dim), dtype=torch.bfloat16, device=dev
        )
        test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)
        train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
        label_ids = rank * local_train + torch.arange(
            local_train, dtype=torch.long, device=dev
        )
        train_labels = label_ids.remainder(1_000).view(1, local_train)
        return (
            test_features.to(dtype=dtype).contiguous(),
            train_features.to(dtype=dtype).t().contiguous(),
            train_labels.contiguous(),
            max_k,
            None,
        )

    # 80: dinov2_distributed_sinkhorn_knopp
    elif problem_id == 80:
        _seed(problem_id, rank, trial)
        local_batch = max(512, min(M, 1_024))
        prototypes = 16_384
        teacher_output = torch.randn(
            (local_batch, prototypes), dtype=torch.float32, device=dev
        ) * 0.01
        teacher_temp = 0.07
        n_masked = torch.full((1,), local_batch, dtype=torch.long, device=dev)
        return (teacher_output.contiguous(), teacher_temp, n_masked, 3, None)

    # 81: sam3_allgather_iou_suppression
    elif problem_id == 81:
        _seed(problem_id, rank, trial)
        height = 256
        width = 256
        counts = [1 + ((idx + trial) % 3) for idx in range(world_size)]
        local_count = counts[rank]
        total_count = sum(counts)

        masks = torch.randn((local_count, height, width), dtype=dtype, device=dev)
        yy = torch.arange(height, device=dev).view(1, height, 1)
        xx = torch.arange(width, device=dev).view(1, 1, width)
        centers = torch.arange(local_count, device=dev).view(local_count, 1, 1)
        pattern = ((yy + xx + centers + rank) % 5 == 0).to(dtype)
        masks = masks + pattern * 4.0 - 1.0
        scores = torch.linspace(-5.0, 5.0, local_count, dtype=dtype, device=dev)
        last_occluded = torch.arange(total_count, dtype=torch.long, device=dev)
        last_occluded = (last_occluded % 5) - 1
        return (
            masks.contiguous(),
            scores.contiguous(),
            counts,
            last_occluded.contiguous(),
            0.7,
            False,
            None,
        )

    # 82: vocab_parallel_log_prob_topk
    elif problem_id == 82:
        _seed(problem_id, rank, trial)
        batch = max(1, min(M // 512, 4))
        seq_len = max(world_size, min(M // 128, 32))
        seq_len = _round_up_multiple(seq_len, world_size)
        vocab_size = 256
        local_vocab = vocab_size // world_size

        logits = torch.randn((batch, seq_len, local_vocab), dtype=dtype, device=dev)
        token_ids = torch.arange(batch * seq_len, dtype=torch.long, device=dev)
        target = (token_ids * 17 + 3 + trial).remainder(vocab_size)
        target = target.reshape(batch, seq_len)
        global_vocab = rank * local_vocab + torch.arange(
            local_vocab, dtype=torch.long, device=dev
        )
        target_mask = global_vocab.view(1, 1, local_vocab) == target.unsqueeze(-1)
        logits = logits + target_mask.to(dtype=logits.dtype) * 8.0
        top_k = 10
        top_p = 0.9
        return (logits.contiguous(), target.contiguous(), None, top_k, top_p)

    # 83: vocab_parallel_log_prob_topk_chunked
    elif problem_id == 83:
        _seed(problem_id, rank, trial)
        batch = max(1, min(M // 512, 4))
        seq_len = max(world_size, min(M // 128, 32))
        seq_len = _round_up_multiple(seq_len, world_size)
        vocab_size = 256
        local_vocab = vocab_size // world_size

        logits = torch.randn((batch, seq_len, local_vocab), dtype=dtype, device=dev)
        token_ids = torch.arange(batch * seq_len, dtype=torch.long, device=dev)
        target = (token_ids * 19 + 5 + trial).remainder(vocab_size)
        target = target.reshape(batch, seq_len)
        global_vocab = rank * local_vocab + torch.arange(
            local_vocab, dtype=torch.long, device=dev
        )
        target_mask = global_vocab.view(1, 1, local_vocab) == target.unsqueeze(-1)
        logits = logits + target_mask.to(dtype=logits.dtype) * 8.0
        top_k = 10
        top_p = 0.9
        chunk_size = _round_up_multiple(max(world_size, seq_len // 4), world_size)
        chunk_size = min(chunk_size, seq_len)
        return (
            logits.contiguous(),
            target.contiguous(),
            None,
            top_k,
            top_p,
            chunk_size,
        )

    # 84: vocab_parallel_log_prob_topk_chunked_backward
    elif problem_id == 84:
        _seed(problem_id, rank, trial)
        batch = max(1, min(M // 512, 4))
        seq_len = max(world_size, min(M // 128, 32))
        seq_len = _round_up_multiple(seq_len, world_size)
        vocab_size = 256
        local_vocab = vocab_size // world_size

        logits = torch.randn((batch, seq_len, local_vocab), dtype=dtype, device=dev)
        token_ids = torch.arange(batch * seq_len, dtype=torch.long, device=dev)
        target = (token_ids * 19 + 5 + trial).remainder(vocab_size)
        target = target.reshape(batch, seq_len)
        global_vocab = rank * local_vocab + torch.arange(
            local_vocab, dtype=torch.long, device=dev
        )
        target_mask = global_vocab.view(1, 1, local_vocab) == target.unsqueeze(-1)
        logits = logits + target_mask.to(dtype=logits.dtype) * 8.0
        grad_output = torch.linspace(
            -1.0, 1.0, batch * seq_len, dtype=torch.bfloat16, device=dev
        ).reshape(batch, seq_len)
        top_k = 10
        top_p = 0.9
        chunk_size = _round_up_multiple(max(world_size, seq_len // 4), world_size)
        chunk_size = min(chunk_size, seq_len)
        return (
            logits.contiguous(),
            target.contiguous(),
            grad_output.contiguous(),
            None,
            top_k,
            top_p,
            chunk_size,
        )

    # 85: distributed_sample_sort
    elif problem_id == 85:
        _seed(problem_id, rank, trial)
        local_n = max(world_size * 4, min(M // max(world_size, 1), 4096))
        if trial % 4 == 3 and rank % 2 == 1:
            local_n = 0
        values = torch.randint(-50, 50, (local_n,), dtype=torch.int64, device=dev)
        values = values.to(dtype) - rank * max(1, local_n)
        return (values.contiguous(), None)

    # 86: tp_muon_orthogonalization
    elif problem_id == 86:
        _seed(problem_id, rank, trial)
        rows = 512
        global_cols = 512 if N < 4096 else 1024
        global_cols = _round_up_multiple(global_cols, world_size)
        local_cols = global_cols // world_size
        x = torch.randn((rows, local_cols), dtype=torch.bfloat16, device=dev)
        x = x + 0.01 * rank
        steps = 5
        coefficient_type = "quintic"
        partition_dim = 1
        return (x.contiguous(), steps, coefficient_type, partition_dim, None)

    # 87: conv2d_boundary_exchange
    elif problem_id == 87:
        _seed(problem_id, rank, trial)
        batch = 1
        channel_choices = (320, 640, 1280)
        in_channels = channel_choices[min(N // 512, 2)]
        out_channels = in_channels
        padding = 1
        kernel = 2 * padding + 1
        latent_h = 128
        latent_w = 128
        n_device_per_batch = max(1, world_size // 2)
        local_h = max(kernel, latent_h // n_device_per_batch)
        width = latent_w

        x = torch.randn((batch, in_channels, local_h, width), dtype=dtype, device=dev)
        weight = torch.randn(
            (out_channels, in_channels, kernel, kernel), dtype=dtype, device=dev
        )
        bias = torch.randn((out_channels,), dtype=dtype, device=dev)
        return (
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            1,
            padding,
            None,
        )

    # Default: standard shape
    return (torch.full(base_shape, val, dtype=dtype, device=dev),)

def save_performance_metrics(metrics: dict, logs_dir: str, rank: int) -> str:
    """Save performance metrics to a JSON file."""
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, f"rank_{rank}_perf.json")
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return path