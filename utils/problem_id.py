"""
Resolve --problem argument to (stem, int_id) for file paths and worker input shape.

Accepts:
  - "1"           -> resolve to first reference file matching 1_*.py (e.g. "1_allreduce")
  - "1_file"      -> stem "1_file"
  - "1_file.py"   -> stem "1_file"

Returns (stem, problem_id_int) where problem_id_int is the leading number from the stem
(for scripts/worker.py --problem_id, which must be int).
"""
from __future__ import annotations
import re
from pathlib import Path
import io


def resolve_problem(
    project_root: Path,
    problem_arg: str,
    solution_type: str,
) -> tuple[str, int]:
    """
    Resolve raw --problem to file stem and integer ID.

    Returns:
        (stem, problem_id_int): stem for paths (e.g. "1_allreduce"), int for worker --problem_id.
    """
    raw = problem_arg.strip().removesuffix(".py").strip()
    ref_dir = project_root / "reference"

    if not raw:
        raise ValueError("Problem cannot be empty")

    # Numeric shorthand: "1" -> first reference file 1_*.py
    if raw.isdigit():
        prefix = raw
        if ref_dir.is_dir():
            matches = sorted(ref_dir.glob(f"{prefix}_*.py"))
            if not matches:
                raise FileNotFoundError(
                    f"No reference problem starting with {prefix}_ (e.g. {prefix}_name.py). "
                    f"Check reference/ for files like 1_allreduce.py"
                )
            stem = matches[0].stem
        else:
            stem = raw
    else:
        stem = raw

    # Integer ID: leading digits from stem (for create_input_tensor / worker --problem_id)
    match = re.match(r"^(\d+)", stem)
    problem_id_int = int(match.group(1)) if match else 1

    return stem, problem_id_int


def resolve_logs_problem_dir(
    project_root: Path,
    logs_base: Path,
    problem_arg: str,
    *,
    solution_type: str = "reference",
) -> tuple[Path, str]:
    """
    Map --problem to the downloaded-artifacts directory used by run_local.py.

    Example: problem_arg ``1`` -> ``logs/problem_1_allreduce/``.
    """
    stem, _ = resolve_problem(project_root, problem_arg, solution_type)
    return logs_base / f"problem_{stem}", stem


def list_problem_log_backends(problem_logs_dir: Path) -> list[str]:
    """Subdirs under logs/problem_<stem>/ that contain rank_*.pt or rank_*_perf.json."""
    if not problem_logs_dir.is_dir():
        return []
    backends: list[str] = []
    for child in sorted(problem_logs_dir.iterdir()):
        if not child.is_dir():
            continue
        has_artifacts = any(child.glob("rank_*.pt")) or any(child.glob("rank_*_perf.json"))
        if has_artifacts:
            backends.append(child.name)
    return backends


def list_reference_stems(
    project_root: Path, *, min_problem_id: int = 1, max_problem_id: int | None = None
) -> list[tuple[str, int]]:
    """
    Return sorted (stem, problem_id_int) for every reference/NN_*.py with NN >= min_problem_id.

    If max_problem_id is not None, only include stems with id <= max_problem_id.
    """
    ref_dir = project_root / "reference"
    if not ref_dir.is_dir():
        return []
    out: list[tuple[str, int]] = []
    for p in ref_dir.glob("*.py"):
        m = re.match(r"^(\d+)", p.stem)
        if not m:
            continue
        pid = int(m.group(1))
        if pid < int(min_problem_id):
            continue
        if max_problem_id is not None and pid > int(max_problem_id):
            continue
        out.append((p.stem, pid))
    return sorted(out, key=lambda t: (t[1], t[0]))


def outputs_match_rank_outputs(
    ref_files: list[tuple[str, bytes]],
    sol_files: list[tuple[str, bytes]],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """
    Compare rank_*.pt payloads from parallel lists of (relpath, bytes).

    Each .pt file may hold a tensor, tuple/list of tensors, or dict of tensors (same
    rules as utils/compare_outputs.py).
    """
    import torch

    def pt_only(items):
        return sorted([(r, b) for r, b in items if r.endswith(".pt")], key=lambda x: x[0])

    ref = pt_only(ref_files)
    sol = pt_only(sol_files)
    if not ref or not sol:
        return False
    if [r[0] for r in ref] != [s[0] for s in sol]:
        return False

    def match_leaf(a, b) -> bool:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape:
                return False
            return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
        try:
            return a == b
        except Exception:
            return False

    def match_recursive(ref_out, sol_out) -> bool:
        if isinstance(ref_out, dict) and isinstance(sol_out, dict):
            if set(ref_out.keys()) != set(sol_out.keys()):
                return False
            return all(match_recursive(ref_out[k], sol_out[k]) for k in sorted(ref_out.keys()))
        if isinstance(ref_out, (tuple, list)) and isinstance(sol_out, (tuple, list)):
            if len(ref_out) != len(sol_out):
                return False
            return all(match_recursive(x, y) for x, y in zip(ref_out, sol_out))
        return match_leaf(ref_out, sol_out)

    for (_r, br), (_s, bs) in zip(ref, sol):
        ref_out = torch.load(io.BytesIO(br), weights_only=True)
        sol_out = torch.load(io.BytesIO(bs), weights_only=True)
        if not match_recursive(ref_out, sol_out):
            return False
    return True


def rank_outputs_compare_details(
    ref_files: list[tuple[str, bytes]],
    sol_files: list[tuple[str, bytes]],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict:
    """
    Same comparison as outputs_match_rank_outputs, but return a JSON-friendly report
    for logging (first concrete failure, per-file load errors).
    """
    import torch

    def pt_only(items):
        return sorted([(r, b) for r, b in items if r.endswith(".pt")], key=lambda x: x[0])

    ref = pt_only(ref_files)
    sol = pt_only(sol_files)
    out: dict = {
        "ok": False,
        "kind": "missing_artifacts",
        "errors": [],
        "per_file": [],
    }
    if not ref or not sol:
        if not ref:
            out["errors"].append("reference job: no rank_*.pt artifacts (after URL download).")
        if not sol:
            out["errors"].append("solution job: no rank_*.pt artifacts (after URL download).")
        return out

    ref_names = [r[0] for r in ref]
    sol_names = [s[0] for s in sol]
    if ref_names != sol_names:
        out["kind"] = "path_mismatch"
        out["errors"].append(f"rank file lists differ.\n  reference: {ref_names}\n  solution:  {sol_names}")
        return out

    def describe_leaf_mismatch(path: str, a, b) -> str:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape:
                return f"{path}: tensor shape ref {tuple(a.shape)} vs sol {tuple(b.shape)}"
            if torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol):
                return ""
            diff = (a.float() - b.float()).abs()
            max_v = float(diff.max().item())
            mean_v = float(diff.mean().item())
            return (
                f"{path}: tensors not allclose (atol={atol}, rtol={rtol}); "
                f"max_abs_diff={max_v:.6g} mean_abs_diff={mean_v:.6g}"
            )
        try:
            same = a == b
        except Exception as exc:
            return f"{path}: non-tensor compare failed: {exc!r}"
        if same:
            return ""
        return f"{path}: values differ ({type(a).__name__} vs {type(b).__name__})"

    def walk_mismatch(path: str, ref_out, sol_out) -> str | None:
        if isinstance(ref_out, dict) and isinstance(sol_out, dict):
            rk, sk = set(ref_out.keys()), set(sol_out.keys())
            if rk != sk:
                return (
                    f"{path}: dict keys differ "
                    f"(only_in_ref={sorted(rk - sk)} only_in_sol={sorted(sk - rk)})"
                )
            for k in sorted(ref_out.keys()):
                sub = f"{path}.{k}" if path else str(k)
                msg = walk_mismatch(sub, ref_out[k], sol_out[k])
                if msg:
                    return msg
            return None
        if isinstance(ref_out, (tuple, list)) and isinstance(sol_out, (tuple, list)):
            if len(ref_out) != len(sol_out):
                return f"{path}: sequence length ref {len(ref_out)} vs sol {len(sol_out)}"
            for i, (x, y) in enumerate(zip(ref_out, sol_out)):
                msg = walk_mismatch(f"{path}[{i}]", x, y)
                if msg:
                    return msg
            return None
        msg = describe_leaf_mismatch(path or "(root)", ref_out, sol_out)
        return msg if msg else None

    out["kind"] = "tensor_mismatch"
    for (rpath, br), (_spath, bs) in zip(ref, sol):
        entry: dict = {"file": rpath, "ok": True, "detail": None}
        try:
            ref_out = torch.load(io.BytesIO(br), weights_only=True)
            sol_out = torch.load(io.BytesIO(bs), weights_only=True)
        except Exception as exc:
            entry["ok"] = False
            entry["detail"] = f"torch.load failed: {exc!r}"
            out["per_file"].append(entry)
            out["errors"].append(f"{rpath}: {entry['detail']}")
            out["kind"] = "load_error"
            return out

        mismatch = walk_mismatch("", ref_out, sol_out)
        if mismatch:
            entry["ok"] = False
            entry["detail"] = mismatch
            out["per_file"].append(entry)
            out["errors"].append(mismatch)
            return out
        out["per_file"].append(entry)

    out["ok"] = True
    out["kind"] = "match"
    out["errors"] = []
    return out
