from typing import Optional, Sequence

import torch
import torch.distributed as dist


_COEFFICIENTS: dict[str, Sequence[tuple[float, float, float]]] = {
    "simple": ((3.4445, -4.7750, 2.0315),),
    "quintic": (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ),
    "polar_express": (
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4647, 0.5074),
        (2.2779, -1.6447, 0.4162),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ),
    "aol": (
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ),
}


def _coefficient_at(
    coefficients: Sequence[tuple[float, float, float]],
    step: int,
) -> tuple[float, float, float]:
    return coefficients[step % len(coefficients)]


def _distributed_normalize(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    eps: float = 1e-7,
) -> torch.Tensor:
    norm_sq = (x * x).sum()
    dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=group)
    return x / torch.sqrt(norm_sq).clamp_min(eps)


def _newton_schulz_step(
    x: torch.Tensor,
    a: float,
    b: float,
    c: float,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    gram = x @ x.mT
    dist.all_reduce(gram, op=dist.ReduceOp.SUM, group=group)
    update = torch.addmm(gram, gram, gram, alpha=c, beta=b)
    return torch.addmm(x, update, x, alpha=1.0, beta=a)


@torch.no_grad()
def solution(
    x: torch.Tensor,
    steps: int = 5,
    coefficient_type: str = "quintic",
    partition_dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    assert x.ndim == 2
    assert coefficient_type in _COEFFICIENTS
    coefficients = _COEFFICIENTS[coefficient_type]
    assert steps % len(coefficients) == 0

    in_dtype = x.dtype

    if partition_dim == 0:
        x_work = x.mT.contiguous()
    elif partition_dim == 1:
        x_work = x
    else:
        raise AssertionError("invalid partition_dim")

    x_work = x_work.to(torch.float32)
    x_work = _distributed_normalize(x_work, group)

    for step in range(steps):
        a, b, c = _coefficient_at(coefficients, step)
        x_work = _newton_schulz_step(x_work, a, b, c, group)

    x_work = x_work.to(in_dtype)
    if partition_dim == 0:
        return x_work.mT.contiguous()
    return x_work.contiguous()
