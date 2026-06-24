#!/usr/bin/env python3
"""
Generate one kernel with Mini-SWE-Agent.

Simple mental model:
1. This script builds the same prompt body as ``generate_kernel.py``.
2. It wraps that prompt in a task that tells an agent which file to write.
3. Mini-SWE-Agent gets a local bash terminal rooted at the repo.
4. The task includes the exact eval command you pass with ``--remote-eval-command`` (a shell command). The agent is in a loop of: it reads failures, edits the solution file, and reruns the command until it is done.

The generated file is saved to ``solutions_<backend>/<problem_stem>_<backend>.py``.

IMPORTANT NOTE: To use this script, you must get the mini-swe-agent github projects into the kernelgen directory:
```
pip install -e kernelgen/mini-swe-agent
cd kernelgen
git clone https://github.com/SWE-agent/mini-swe-agent.git
```

Example usage:
python kernelgen/generate_kernel_agent.py \
  --problem 1 \
  --backend cuda \
  --model gemini-3-flash-preview \
  --step-limit 3 \
  --timeout 600 \
  --remote-dryrun-command \
    'python run_local.py --nproc-per-node 4 --mode dryrun --problem {problem_arg} --solution {backend} --measure-perf' \
  --remote-eval-command \
    'python run_local.py --nproc-per-node 4 --mode eval --problem {problem_arg} --solution {backend} --measure-perf'
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import toml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_KERNELGEN_DIR = Path(__file__).resolve().parent
_MINI_SRC = _KERNELGEN_DIR / "mini-swe-agent" / "src"

for _p in (_PROJECT_ROOT, _MINI_SRC):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from utils.problem_id import resolve_problem

from generate_kernel import (
    _load_reference,
    assemble_user_prompt,
    extract_python_from_response,
)

################################################################################################
# Paths and constants
################################################################################################

_MINI_CONFIG = _MINI_SRC / "minisweagent" / "config" / "mini.yaml"
_PRECISION_CHOICES = ("fp32", "fp16", "bf16")
_HARDWARE_CHOICES = ("h100_8", "b200_72")


################################################################################################
# Log paths and run metadata
################################################################################################

def _sanitize_path_slug(s: str, *, max_len: int = 80) -> str:
    """Make a filesystem-safe fragment for log directory names."""
    t = re.sub(r"[^\w.\-]+", "_", s.strip(), flags=re.ASCII)
    return (t or "run")[:max_len]


def _default_log_dir(args: argparse.Namespace) -> Path:
    """Default per-run log folder for metadata and Mini-SWE-Agent trajectory."""
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    slug = _sanitize_path_slug(f"{args.problem_id}_{args.backend}")
    return _PROJECT_ROOT / "logs" / "kernelgen_agent" / f"{ts}_miniswe_{slug}"


def _resolve_log_paths(args: argparse.Namespace) -> None:
    """Fill args.log_dir and args.trajectory from --trajectory or defaults."""
    if args.trajectory is None:
        args.log_dir = _default_log_dir(args)
        args.trajectory = args.log_dir / "miniswe_agent.traj.json"
        return
    p = Path(args.trajectory).expanduser().resolve()
    if p.is_dir():
        args.log_dir = p
        args.trajectory = args.log_dir / "miniswe_agent.traj.json"
    else:
        args.log_dir = p.parent
        args.trajectory = p


def _write_run_metadata(
    log_dir: Path,
    args: argparse.Namespace,
    *,
    argv: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write lightweight metadata that makes a run reproducible later."""
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = " ".join(shlex.quote(a) for a in argv)
    (log_dir / "command.txt").write_text(cmd + "\n", encoding="utf-8")
    meta: dict[str, Any] = {
        "version": 2,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "argv": argv,
        "command_line": cmd,
        "mode": "miniswe_remote",
        "problem_arg": args.problem_arg,
        "problem_id": args.problem_id,
        "backend": args.backend,
        "model": args.model,
        "together_api_key_configured": bool(_together_api_key_effective()),
        "precision": args.precision,
        "hardware": args.hardware,
        "prompts_path": str(args.prompts),
        "project_root": str(_PROJECT_ROOT),
        "remote_eval_command_template": args.remote_eval_command,
        "remote_dryrun_command_template": args.remote_dryrun_command,
        "cost_limit": args.cost_limit,
        "step_limit": args.step_limit,
        "timeout": args.timeout,
        "bench": {
            "m": args.m,
            "n": args.n,
            "dtype": args.dtype,
            "trials": args.trials,
            "measure_perf": args.measure_perf,
        },
        "artifacts": {
            "miniswe_trajectory": "miniswe_agent.traj.json",
            "note": "mini-swe-agent trajectory (message history and tool calls).",
        },
    }
    if extra:
        meta["extra"] = extra
    (log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _finalize_meta(log_dir: Path, *, exit_code: int, extra: dict[str, Any] | None = None) -> None:
    """Record final status in meta.json even when the agent run fails."""
    p = log_dir / "meta.json"
    if not p.is_file():
        return
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    meta["exit_code"] = exit_code
    if extra:
        meta["result"] = extra
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


################################################################################################
# Model naming and provider authentication
################################################################################################

def _normalize_litellm_model_id(model: str) -> str:
    """LiteLLM provider prefixes; Together uses ``together_ai/`` (not ``together/``)."""
    m = model.strip()
    if m.startswith("together/") and not m.startswith("together_ai/"):
        return "together_ai/" + m[len("together/") :].lstrip("/")
    if m.startswith("google/gemini-"):
        rest = m.removeprefix("google/")
        rest = rest.replace("gemini-2-5-", "gemini-2.5-", 1)
        return "gemini/" + rest
    if m.startswith("gemini-") and not m.startswith("gemini/"):
        return "gemini/" + m
    return m


def _together_api_key_effective() -> str | None:
    """Return the Together key name accepted by LiteLLM, if configured."""
    return os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHERAI_API_KEY")


def _apply_together_api_key(args: argparse.Namespace) -> None:
    """Set env vars LiteLLM reads for Together AI (``litellm.completion(..., model='together_ai/...')``)."""
    key = getattr(args, "together_api_key", None)
    if not key:
        return
    os.environ["TOGETHER_API_KEY"] = key
    # Some LiteLLM builds check this alias
    os.environ.setdefault("TOGETHERAI_API_KEY", key)


def _require_together_key_if_needed(model: str) -> None:
    if not model.startswith("together_ai/"):
        return
    if _together_api_key_effective():
        return
    print(
        "Error: Together AI models need an API key. Set environment variable TOGETHER_API_KEY "
        "or pass --together-api-key (LiteLLM uses it for model names starting with together_ai/).",
        file=sys.stderr,
    )
    raise SystemExit(1)


################################################################################################
# Prompt and task construction
################################################################################################

def _template_agent_kernel(cfg: dict[str, Any], key: str, *, required: bool = True) -> str:
    """Read one [templates.agent_kernel] string from prompts.toml."""
    block = (cfg.get("templates") or {}).get("agent_kernel") or {}
    raw = block.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if required:
        raise KeyError(
            f"prompts.toml: set non-empty [templates.agent_kernel].{key} (missing or blank)"
        )
    return ""


def _agent_system_prompt(cfg: dict[str, Any], backend: str) -> str:
    """Use a backend-specific agent prompt, then fall back to the default."""
    row = (cfg.get("backends") or {}).get(backend)
    if isinstance(row, dict):
        p = row.get("agent_system_prompt")
        if isinstance(p, str) and p.strip():
            return p.strip()
    fb = (cfg.get("agent_kernel") or {}).get("default_system_prompt")
    if isinstance(fb, str) and fb.strip():
        return fb.strip()
    raise KeyError(
        f"prompts.toml: add [backends.{backend}].agent_system_prompt or [agent_kernel].default_system_prompt"
    )


def _placeholder_context(args: argparse.Namespace, *, problem_stem: str, rel_target: str) -> dict[str, Any]:
    """Values available to --remote-eval-command and --remote-dryrun-command."""
    return {
        "project_root": str(_PROJECT_ROOT),
        "problem_arg": args.problem_arg,
        "problem_stem": problem_stem,
        "problem_id": problem_stem,
        "backend": args.backend,
        "rel_target": rel_target,
        "rel": rel_target,
        "ref_rel": f"reference/{problem_stem}.py",
        "m": args.m,
        "n": args.n,
        "dtype": args.dtype,
        "trials": args.trials,
        "measure_perf": "true" if args.measure_perf else "false",
        "measure_perf_flag": "--measure-perf" if args.measure_perf else "",
    }


def _format_remote_template(template: str, ctx: dict[str, Any]) -> str:
    """Render an eval command template and show useful errors for bad placeholders."""
    try:
        return template.format(**ctx)
    except KeyError as e:
        print(
            f"Error: remote command template has unknown placeholder {e!s}. "
            f"Allowed keys: {', '.join(sorted(ctx))}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _remote_verify_section(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    problem_stem: str,
    ctx: dict[str, Any],
) -> str:
    """Build instructions that tell the agent which eval command(s) to run."""
    rel_target = f"solutions_{args.backend}/{problem_stem}_{args.backend}.py"
    pieces: list[str] = []

    if args.remote_dryrun_command and args.remote_dryrun_command.strip():
        dry_command = _format_remote_template(args.remote_dryrun_command.strip(), ctx)
        tpl = _template_agent_kernel(cfg, "remote_dryrun", required=True)
        pieces.append(
            tpl.format(
                dry_command=dry_command,
                rel_target=rel_target,
                problem_arg=args.problem_arg,
                problem_stem=problem_stem,
                backend=args.backend,
            )
        )

    eval_command = _format_remote_template(args.remote_eval_command.strip(), ctx)
    tpl_ev = _template_agent_kernel(cfg, "remote_eval", required=True)
    pieces.append(
        tpl_ev.format(
            eval_command=eval_command,
            dry_command=_format_remote_template(args.remote_dryrun_command.strip(), ctx)
            if (args.remote_dryrun_command and args.remote_dryrun_command.strip())
            else "# (no separate dryrun command — use eval only)",
            rel_target=rel_target,
            problem_arg=args.problem_arg,
            problem_stem=problem_stem,
            backend=args.backend,
        )
    )
    return "\n".join(pieces)


def _build_task(
    cfg: dict[str, Any],
    *,
    project_root: Path,
    problem_id: str,
    backend: str,
    prompt_body: str,
    remote_block: str,
) -> str:
    """Assemble the complete Mini-SWE-Agent user task."""
    rel = f"solutions_{backend}/{problem_id}_{backend}.py"
    ref_rel = f"reference/{problem_id}.py"
    extra = f"\n{remote_block}\n" if remote_block.strip() else ""
    tpl = _template_agent_kernel(cfg, "task")
    return tpl.format(
        project_root=project_root,
        rel=rel,
        ref_rel=ref_rel,
        backend=backend,
        remote_block=extra,
        prompt_body=prompt_body,
    )


################################################################################################
# Mini-SWE-Agent configuration
################################################################################################

def _build_agent_section(
    base_agent: dict[str, Any],
    cfg: dict[str, Any],
    *,
    backend: str,
    cost_limit: float,
    step_limit: int,
    trajectory: Path | None,
) -> dict[str, Any]:
    """Overlay our system prompt, limits, and trajectory path onto mini.yaml."""
    agent = copy.deepcopy(base_agent)
    agent["agent_class"] = "default"
    agent["system_template"] = _agent_system_prompt(cfg, backend)
    agent["cost_limit"] = cost_limit
    agent["step_limit"] = step_limit
    agent["output_path"] = trajectory
    return agent


################################################################################################
# Parse user arguments
################################################################################################

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate solutions_<backend>/<stem>_<backend>.py via mini-swe-agent and "
            "user-defined remote eval commands embedded in the task."
        ),
    )
    p.add_argument(
        "--problem",
        "-p",
        "--problemid",
        dest="problem",
        default="1",
        help="Problem id or stem (e.g. 1, 3_broadcast). Resolved against reference/.",
    )
    p.add_argument(
        "--backend",
        "-b",
        required=True,
        help="prompts.toml [backends.*] key and output folder (cuda, triton, ...).",
    )
    p.add_argument("--precision", default="bf16", choices=_PRECISION_CHOICES)
    p.add_argument("--hardware", default=None, choices=_HARDWARE_CHOICES)
    p.add_argument(
        "--prompts",
        type=Path,
        default=_KERNELGEN_DIR / "prompts.toml",
        help="Path to prompts.toml.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MSWEA_MODEL_NAME", "anthropic/claude-sonnet-4-5-20250929"),
        help=(
            "LiteLLM model id (default: MSWEA_MODEL_NAME). "
            "Together AI: together_ai/<model> (or together/<model>, normalized); "
            "set TOGETHER_API_KEY or --together-api-key."
        ),
    )
    p.add_argument(
        "--together-api-key",
        default=None,
        metavar="KEY",
        help=(
            "Together AI API key for this process (sets TOGETHER_API_KEY). "
            "If omitted, the environment variable TOGETHER_API_KEY is used."
        ),
    )
    p.add_argument(
        "--cost-limit",
        type=float,
        default=15.0,
        help="Agent cost limit (USD).",
    )
    p.add_argument(
        "--step-limit",
        type=int,
        default=0,
        help="Max model calls (0 = unlimited).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-command timeout for LocalEnvironment (seconds; remote eval is often slow).",
    )
    p.add_argument(
        "--remote-eval-command",
        required=True,
        help=(
            "str.format template for the remote eval command the agent must run. "
            "See module docstring for placeholders."
        ),
    )
    p.add_argument(
        "--remote-dryrun-command",
        default=None,
        help="Optional str.format template for a lighter check before full eval.",
    )
    p.add_argument("--m", type=int, default=1024, help="Placeholder {m} for remote command templates.")
    p.add_argument("--n", type=int, default=1024, help="Placeholder {n} for remote command templates.")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        help="Placeholder {dtype} for remote command templates (default bfloat16, matching paper experiments).",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Placeholder {trials} for remote command templates.",
    )
    p.add_argument(
        "--measure-perf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sets {measure_perf} / {measure_perf_flag} placeholders (default: on).",
    )
    p.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Trajectory file or directory (default: logs/kernelgen_agent/<timestamp>_miniswe_.../).",
    )
    p.add_argument(
        "--dry-run-task",
        action="store_true",
        help="Print the task string and exit (no LLM).",
    )
    ns = p.parse_args(argv)
    ns.problem_arg = str(ns.problem)
    try:
        stem, _ = resolve_problem(_PROJECT_ROOT, str(ns.problem), ns.backend)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    ns.problem_id = stem

    prompts_path = ns.prompts.resolve()
    if not prompts_path.is_file():
        print(f"Error: Prompt config not found: {prompts_path}", file=sys.stderr)
        raise SystemExit(1)
    cfg = toml.load(prompts_path)
    if ns.backend not in cfg.get("backends", {}):
        keys = ", ".join(sorted(cfg.get("backends", {})))
        print(f"Error: unknown backend {ns.backend!r}. Known: {keys}", file=sys.stderr)
        raise SystemExit(1)
    if ns.precision not in cfg.get("precision", {}):
        print(f"Error: unknown precision {ns.precision!r}", file=sys.stderr)
        raise SystemExit(1)

    _resolve_log_paths(ns)
    return ns


################################################################################################
# Run Mini-SWE-Agent
################################################################################################

def run_agent(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    """
    Build the task, start Mini-SWE-Agent, and verify it produced the expected file.

    The agent is the process that actually edits files and runs the eval command. This
    wrapper only prepares instructions/config and checks for the final solution file.
    """
    ref_src = _load_reference(_PROJECT_ROOT, args.problem_id)
    rel_out = f"solutions_{args.backend}/{args.problem_id}_{args.backend}.py"
    prompt_body = assemble_user_prompt(
        cfg,
        backend=args.backend,
        precision=args.precision,
        ref_arch_src=ref_src,
        project_root=_PROJECT_ROOT,
        problem_stem=args.problem_id,
        hardware_key=args.hardware,
        solution_relative_path=rel_out,
    )
    ctx = _placeholder_context(args, problem_stem=args.problem_id, rel_target=rel_out)
    remote_block = _remote_verify_section(cfg, args, problem_stem=args.problem_id, ctx=ctx)
    task = _build_task(
        cfg,
        project_root=_PROJECT_ROOT,
        problem_id=args.problem_id,
        backend=args.backend,
        prompt_body=prompt_body,
        remote_block=remote_block,
    )

    if args.dry_run_task:
        print(task)
        return 0

    _require_together_key_if_needed(args.model)

    try:
        from minisweagent.config import get_config_from_spec
        from minisweagent.models import get_model
        from minisweagent.environments import get_environment
        from minisweagent.agents import get_agent
        from minisweagent.utils.serialize import recursive_merge
    except ImportError as e:
        print(
            "Error: mini-swe-agent dependencies missing. From repo root run:\n"
            "  pip install -e kernelgen/mini-swe-agent\n"
            f"Import detail: {e}",
            file=sys.stderr,
        )
        return 1

    mini_cfg = get_config_from_spec(_MINI_CONFIG)
    agent_cfg = _build_agent_section(
        mini_cfg.get("agent", {}),
        cfg,
        backend=args.backend,
        cost_limit=args.cost_limit,
        step_limit=args.step_limit,
        trajectory=args.trajectory,
    )
    config = recursive_merge(
        mini_cfg,
        {
            "model": {"model_name": args.model},
            "environment": {
                "environment_class": "local",
                "cwd": str(_PROJECT_ROOT),
                "timeout": args.timeout,
            },
            "agent": agent_cfg,
        },
    )

    merged_agent = config.get("agent", {})
    for k in ("mode", "whitelist_actions", "confirm_exit"):
        merged_agent.pop(k, None)

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(
        model,
        env,
        merged_agent,
        default_type="default",
    )

    print(
        f"Running mini-swe-agent: problem={args.problem_id} backend={args.backend} -> {rel_out}\n"
        f"Model={args.model!r} trajectory={args.trajectory} cmd_timeout={args.timeout}s",
        flush=True,
    )
    result = agent.run(task)
    out_path = _PROJECT_ROOT / rel_out

    if not out_path.is_file() or out_path.stat().st_size == 0:
        submission = (result or {}).get("submission") or ""
        if submission.strip():
            py = extract_python_from_response(submission)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(py)
            print(f"Wrote from submission fallback: {out_path}")
        else:
            print(
                f"Error: expected file missing or empty: {out_path}\n"
                f"Agent exit: {result!r}\n"
                f"Check trajectory: {args.trajectory}",
                file=sys.stderr,
            )
            return 1
    else:
        print(f"Done: {out_path}\nFull trajectory (mini-swe): {args.trajectory}")
    return 0


################################################################################################
# Entry point
################################################################################################

def main(argv: list[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    _apply_together_api_key(args)
    _raw_model = args.model
    args.model = _normalize_litellm_model_id(args.model)
    if args.model != _raw_model:
        print(
            f"Note: LiteLLM model id {_raw_model!r} -> {args.model!r} "
            "(Gemini: gemini/ prefix; Together: together_ai/; see mini-swe-agent docs/models/troubleshooting.md).",
            file=sys.stderr,
        )
    prompts_path = args.prompts.resolve()
    if not prompts_path.is_file():
        print(f"Error: Prompt config not found: {prompts_path}", file=sys.stderr)
        raise SystemExit(1)
    cfg = toml.load(prompts_path)

    _write_run_metadata(args.log_dir, args, argv=argv_list)
    print(f"Run logs: {args.log_dir}", flush=True)

    code = 0
    try:
        code = run_agent(args, cfg)
    finally:
        _finalize_meta(
            args.log_dir,
            exit_code=code,
            extra={"solution_relative": f"solutions_{args.backend}/{args.problem_id}_{args.backend}.py"},
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
