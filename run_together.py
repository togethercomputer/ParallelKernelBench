#!/usr/bin/env python3
"""
Together launcher for distributed kernel evaluation.

Modes (same idea as run_modal.py):
  dryrun  Submit one job (reference, triton, cuda, parallelkittens, …). Download artifacts
          only if --download. With --measure-perf, prints the same Perf: line as run_modal (from
          rank_*_perf.json fetched in memory when outputs are available, without requiring --download).
  eval    Submit reference, then the given --solution, for each of --trials RNG trials (default 5);
          compare rank_*.pt outputs locally and print per-trial lines (including ref/solution ms and
          speedup when --measure-perf; same perf convention as run_modal: 500/100 iters, L2 input groups,
          500 ms idle between reference and solution) then overall match/mismatch.
          Stops after the first trial when outputs differ (mismatch), or when either job has no rank .pt
          URLs (no further RNG trials). Download only if --download (artifacts under trial_<k>/ per backend).

Requires: TOGETHER_API_KEY, together CLI (pip install together), PyTorch (for --mode eval compare).

Stopping this process (Ctrl+C) does not cancel submitted Together jobs; the CLI has no documented
``cancel`` — use ``together beta jig job-status --request-id …`` and reconcile the queue. On
interrupt, this script prints any request IDs it was still polling.

Usage:
    python run_together.py --mode dryrun --problem 2
    python run_together.py --mode dryrun --problem 2 --solution triton --download
    python run_together.py --mode eval --problem 2 --solution triton
    python run_together.py --mode eval --problem 2 --solution triton --trials 3 --download --measure-perf
    python run_together.py --mode eval --problem 2 --solution triton --measure-perf --measure-warmup-iters 500 --measure-profiling-iters 100
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

from utils.run_utils import (
    check_solution as _check_solution,
    harness_environment_hint as _harness_environment_hint,
    read_solution_source as _read_solution_source,
    summarize_perf_artifacts,
)

# Idle between reference and solution GPU jobs when measuring perf (thermal / power steadying).
_PERF_THERMAL_COOLDOWN_SEC = float(os.environ.get("PKB_PERF_THERMAL_COOLDOWN_SEC", "0.5"))

try:
    import requests as _requests
except ImportError:
    _requests = None

from utils.problem_id import rank_outputs_compare_details, resolve_problem

# Request IDs currently blocked in poll_until_done (Ctrl+C prints these; jobs may remain queued).
_PENDING_TOGETHER_REQUEST_IDS: list[str] = []


def _together_sigint_handler(signum: int, frame: object) -> None:
    """On Ctrl+C, show in-flight Together IDs so you can job-status / reconcile queue manually."""
    ids = list(_PENDING_TOGETHER_REQUEST_IDS)
    if ids:
        print(
            "\n[run_together] Interrupted: these request IDs were still in flight "
            "(Together may keep them queued or running):",
            file=sys.stderr,
        )
        for rid in ids:
            print(f"  {rid}", file=sys.stderr)
        print(
            "[run_together] Check: together beta jig job-status --request-id <id>",
            file=sys.stderr,
        )
        print("[run_together] Backlog: together beta jig queue-status", file=sys.stderr)
    raise KeyboardInterrupt


def _perf_json_urls(urls: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Subset of job output URLs for rank_*_perf.json (avoid downloading .pt when only summarizing perf)."""
    out: list[tuple[str, str]] = []
    for key, url in urls:
        base = os.path.basename(key.replace("\\", "/"))
        if base.startswith("rank_") and base.endswith("_perf.json"):
            out.append((key, url))
    return sorted(out)


def _run_together(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if "TOGETHER_API_KEY" not in env:
        print("Error: TOGETHER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    return subprocess.run(
        ["together", "beta", "jig", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _extract_json_from_stdout(stdout: str) -> dict | None:
    text = stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def submit_job(
    problem_id: str,
    solution_type: str,
    m: int,
    n: int,
    dtype: str,
    measure_perf: bool = False,
    profile: bool = False,
    problem_id_int: int = 1,
    trial: int = 0,
    measure_warmup_iters: int = 500,
    measure_profiling_iters: int = 100,
    solution_source: str | None = None,
) -> str:
    payload_dict = {
        "problem_id": problem_id,
        "problem_id_int": problem_id_int,
        "solution_type": solution_type,
        "m": m,
        "n": n,
        "dtype": dtype,
        "trial": int(trial),
    }
    if solution_source is not None and solution_type != "reference":
        payload_dict["solution_source"] = solution_source
    if measure_perf:
        payload_dict["measure_perf"] = True
        payload_dict["measure_warmup_iters"] = int(measure_warmup_iters)
        payload_dict["measure_profiling_iters"] = int(measure_profiling_iters)
    if profile:
        payload_dict["profile"] = True
    payload = json.dumps(payload_dict)
    result = _run_together(["submit", "--payload", payload])
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        sys.exit(1)
    data = _extract_json_from_stdout(result.stdout)
    if isinstance(data, dict) and data.get("request_id"):
        return data["request_id"]
    print("Could not parse request_id from submit output.", file=sys.stderr)
    if result.stdout:
        print("Stdout:", result.stdout[:500], file=sys.stderr)
    sys.exit(1)


def poll_until_done(request_id: str, poll_interval: float = 5.0) -> dict:
    _PENDING_TOGETHER_REQUEST_IDS.append(request_id)
    try:
        while True:
            result = _run_together(["job-status", "--request-id", request_id])
            if result.returncode != 0:
                print(result.stderr or result.stdout, file=sys.stderr)
                sys.exit(1)
            data = _extract_json_from_stdout(result.stdout)
            if isinstance(data, dict) and "status" in data:
                status = data.get("status")
                if status == "done":
                    return data
                if status == "error":
                    print("Job failed:", data.get("info") or data, file=sys.stderr)
                    sys.exit(1)
                status_str = status
            else:
                text = result.stdout.strip()
                last_brace = text.rfind("{")
                if last_brace != -1:
                    try:
                        data = json.loads(text[last_brace:])
                        if isinstance(data, dict) and "status" in data:
                            status = data.get("status")
                            if status == "done":
                                return data
                            if status == "error":
                                print("Job failed:", data.get("info") or data, file=sys.stderr)
                                sys.exit(1)
                            status_str = status
                        else:
                            status_str = "?"
                    except json.JSONDecodeError:
                        status_str = "?"
                else:
                    status_str = "?"
            print(f"  Waiting for job {request_id[:8]}... (status: {status_str})")
            time.sleep(poll_interval)
    finally:
        try:
            _PENDING_TOGETHER_REQUEST_IDS.remove(request_id)
        except ValueError:
            pass


def _flatten_outputs(outputs: object, prefix: str = "") -> list[tuple[str, object]]:
    """Depth-first leaf listing of API `outputs` (may be nested dicts)."""
    rows: list[tuple[str, object]] = []
    if isinstance(outputs, dict):
        for k, v in outputs.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, dict):
                rows.extend(_flatten_outputs(v, key))
            else:
                rows.append((key, v))
    return rows


def _is_artifact_http_url(key: str, url: str) -> bool:
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    kl = key.replace("\\", "/").lower()
    try:
        path = urlparse(url).path.lower()
    except Exception:
        path = ""
    if kl.endswith((".pt", ".json", ".gz")):
        return True
    if "rank_" in kl and (".pt" in kl or "perf" in kl):
        return True
    if "rank_" in path or path.endswith(".pt") or path.endswith(".json"):
        return True
    return False


def get_output_urls(job_data: dict) -> list[tuple[str, str]]:
    """
    Collect (artifact_name, url) for rank outputs. Flattens nested `outputs` dicts
    and accepts URL paths whose key omits an extension (common API shapes).
    """
    outputs = job_data.get("outputs") or {}
    flat = _flatten_outputs(outputs)
    pairs: list[tuple[str, str]] = []
    for key, value in flat:
        if isinstance(value, str) and _is_artifact_http_url(key, value):
            short = key.split("/")[-1]
            pairs.append((short, value))
    # Deduplicate by artifact name; if collision, keep first (same-rank duplicate URLs).
    by_name: dict[str, str] = {}
    for short, url in pairs:
        by_name.setdefault(short, url)
    return sorted(by_name.items(), key=lambda x: x[0])


def _clip_text_head_tail(
    s: str,
    *,
    max_total: int = 24_000,
    head: int = 8000,
    tail: int = 14_000,
) -> str:
    """Keep start (warnings/context) and end (actual tracebacks / nvcc errors); drop only middle."""
    if not s:
        return s
    if len(s) <= max_total:
        return s
    h = min(head, max_total // 2)
    t = min(tail, max_total - h - 80)
    if h + t >= len(s):
        return s
    omitted = len(s) - h - t
    return s[:h] + f"\n\n... [{omitted} characters omitted from middle] ...\n\n" + s[-t:]


def _summarize_job_for_log(job_data: dict) -> dict:
    """Compact, JSON-serializable job payload slice for debugging (no secrets)."""
    out: dict = {"keys": sorted(job_data.keys())}
    info = job_data.get("info")
    if info is not None:
        out["info"] = str(info)[:4000]
    status = job_data.get("status")
    if status is not None:
        out["status"] = status
    outputs = job_data.get("outputs")
    if outputs is None:
        out["outputs"] = None
        return out
    if isinstance(outputs, dict) and outputs.get("status") == "error":
        out["outputs_status"] = "error"
        for ek in ("error", "message", "stderr", "stdout", "traceback"):
            if ek in outputs and outputs[ek] is not None:
                raw = str(outputs[ek])
                if ek in ("stderr", "stdout", "traceback"):
                    out[f"outputs_{ek}"] = _clip_text_head_tail(raw)
                else:
                    out[f"outputs_{ek}"] = raw[:8000]
    flat = _flatten_outputs(outputs)
    leaves: list[dict] = []
    for k, v in flat[:120]:
        row: dict = {"key": k}
        if isinstance(v, str):
            if v.startswith("http"):
                row["type"] = "url"
                row["preview"] = v[:120] + ("..." if len(v) > 120 else "")
            else:
                row["type"] = "str"
                row["preview"] = v[:1200]
        elif isinstance(v, (int, float, bool)) or v is None:
            row["type"] = type(v).__name__
            row["value"] = v
        else:
            row["type"] = type(v).__name__
            row["preview"] = str(v)[:800]
        leaves.append(row)
    out["output_leaves"] = leaves
    out["output_leaf_count"] = len(flat)
    return out


class _RedirectWithAuth(HTTPRedirectHandler):
    def __init__(self, auth_value: str):
        self._auth = auth_value

    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        new_req = super().redirect_request(req, fp, code, msg, hdrs, newurl)
        if new_req is not None and self._auth:
            new_req.add_header("Authorization", self._auth)
        return new_req


def _download_one_requests(url: str, dest_path: Path, api_key: str) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = _requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    dest_path.write_bytes(r.content)


def _download_one_urllib(url: str, dest_path: Path, api_key: str, auth_in_query: bool) -> None:
    if auth_in_query and api_key:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        qs["api_key"] = [api_key]
        url = parsed._replace(query=urlencode(qs, doseq=True)).geturl()
    auth = f"Bearer {api_key}" if api_key and not auth_in_query else None
    req = Request(url)
    if auth:
        req.add_header("Authorization", auth)
    opener = build_opener(_RedirectWithAuth(auth)) if auth else build_opener()
    with opener.open(req) as resp:
        dest_path.write_bytes(resp.read())


def fetch_url_bytes(url: str, api_key: str) -> bytes:
    """Download URL to bytes (same auth / 403 retry behavior as download_outputs)."""
    if _requests is not None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = _requests.get(url, headers=headers, timeout=120)
        if r.status_code == 403:
            r = _requests.get(url, headers={}, timeout=120)
        r.raise_for_status()
        return r.content

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp = Path(f.name)
    try:
        try:
            _download_one_urllib(url, tmp, api_key, auth_in_query=False)
        except HTTPError as e:
            if e.code == 403 and api_key:
                _download_one_urllib(url, tmp, api_key, auth_in_query=True)
            else:
                raise
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def download_urls_to_bytes_list(urls: list[tuple[str, str]], api_key: str) -> list[tuple[str, bytes]]:
    """(relpath, bytes) for each artifact URL."""
    out: list[tuple[str, bytes]] = []
    for name, url in urls:
        data = fetch_url_bytes(url, api_key)
        out.append((name, data))
    return out


def download_outputs(urls: list[tuple[str, str]], local_logs_dir: Path, api_key: str) -> None:
    local_logs_dir.mkdir(parents=True, exist_ok=True)
    for name, url in urls:
        dest = local_logs_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if _requests is not None:
                _download_one_requests(url, dest, api_key)
            else:
                _download_one_urllib(url, dest, api_key, auth_in_query=False)
        except Exception as e:
            is_403 = getattr(e, "code", None) == 403 or (
                getattr(e, "response", None) is not None
                and getattr(e.response, "status_code", None) == 403
            )
            if is_403:
                try:
                    if _requests is not None:
                        _download_one_requests(url, dest, "")
                    else:
                        _download_one_urllib(url, dest, api_key, auth_in_query=True)
                except Exception:
                    print("  Storage returned 403. Try: pip install requests (sends auth on redirects).", file=sys.stderr)
                    if not api_key:
                        print("  And set TOGETHER_API_KEY.", file=sys.stderr)
                    raise
            else:
                raise
        print(f"  Downloaded: {dest}")


def _run_one_job(
    label: str,
    problem_id: str,
    solution_type: str,
    m: int,
    n: int,
    dtype: str,
    measure_perf: bool,
    profile: bool,
    problem_id_int: int,
    trial: int = 0,
    measure_warmup_iters: int = 500,
    measure_profiling_iters: int = 100,
    solution_source: str | None = None,
) -> dict:
    print(f"Submitting {label} ({solution_type})...")
    request_id = submit_job(
        problem_id,
        solution_type,
        m,
        n,
        dtype,
        measure_perf=measure_perf,
        profile=profile,
        problem_id_int=problem_id_int,
        trial=trial,
        measure_warmup_iters=measure_warmup_iters,
        measure_profiling_iters=measure_profiling_iters,
        solution_source=solution_source,
    )
    print(f"  Request ID: {request_id}")
    job_data = poll_until_done(request_id)
    print("  Job completed.")
    return job_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ParallelKernelBench on Together cluster.")
    parser.add_argument(
        "--mode",
        choices=("dryrun", "eval"),
        default="dryrun",
        help="dryrun: one backend. eval: reference + solution, compare tensors.",
    )
    parser.add_argument("--problem", "-p", default="1", help="Problem: number, stem, or file.py")
    parser.add_argument("--solution", "-s", default="reference", help="Backend: reference, triton, cuda, parallelkittens")
    parser.add_argument("--m", type=int, default=1024, help="Rows")
    parser.add_argument("--n", type=int, default=1024, help="Cols")
    parser.add_argument("--dtype", default="float32", help="Tensor dtype")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Write outputs under logs/problem_<stem>/{reference|<solution>}/ (default: do not download).",
    )
    parser.add_argument("--measure-perf", action="store_true", help="Warmup + timed iterations; save rank_*_perf.json")
    parser.add_argument(
        "--measure-warmup-iters",
        type=int,
        default=500,
        dest="measure_warmup_iters",
        help="Warmup iterations before timed region (with --measure-perf; default 500).",
    )
    parser.add_argument(
        "--measure-profiling-iters",
        type=int,
        default=100,
        dest="measure_profiling_iters",
        help="Profiling iterations inside one CUDA event pair (with --measure-perf; default 100).",
    )
    parser.add_argument("--profile", action="store_true", help="PyTorch profiler traces under traces/")
    parser.add_argument("--atol", type=float, default=1e-5, help="eval mode: abs tolerance for tensor compare")
    parser.add_argument("--rtol", type=float, default=1e-5, help="eval mode: rel tolerance for tensor compare")
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="eval mode only: number of random-input trials (default 5). Ignored for dryrun.",
    )
    parser.add_argument("--logs-dir", type=str, default=None, help="Base logs directory (default: project logs/)")
    parser.add_argument(
        "--solutions-dir",
        type=Path,
        default=None,
        help="Directory for <stem>_<backend>.py when --solution is not reference (default: <project>/solutions_<backend>/).",
    )
    args = parser.parse_args()

    # So Ctrl+C lists any request_id we are polling (Together queue is separate; cannot auto-cancel via CLI).
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _together_sigint_handler)

    project_root = Path(__file__).resolve().parent
    solutions_dir: Path | None = (
        args.solutions_dir.resolve() if args.solutions_dir is not None else None
    )
    try:
        problem_stem, problem_id_int = resolve_problem(project_root, str(args.problem), args.solution)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "eval":
        if args.solution == "reference":
            print("Error: --mode eval requires --solution other than reference.", file=sys.stderr)
            sys.exit(1)
        if args.trials < 1:
            print("Error: --trials must be >= 1", file=sys.stderr)
            sys.exit(1)
        _check_solution(
            project_root, problem_stem, "reference", solutions_dir=solutions_dir
        )
        _check_solution(
            project_root, problem_stem, args.solution, solutions_dir=solutions_dir
        )
    else:
        _check_solution(
            project_root, problem_stem, args.solution, solutions_dir=solutions_dir
        )

    print("Together cluster")
    print(f"  Mode:     {args.mode}")
    print(f"  Problem:  {problem_stem}")
    print(f"  Solution: {args.solution}")
    print(f"  Shape:    ({args.m}, {args.n})  dtype={args.dtype}")
    print(f"  Download: {args.download}")
    if args.mode == "eval":
        print(f"  Trials:   {args.trials}")
    if args.measure_perf:
        print("  measure_perf: yes")
        print(f"  measure_warmup_iters: {args.measure_warmup_iters}")
        print(f"  measure_profiling_iters: {args.measure_profiling_iters}")
    if args.profile:
        print("  profile:      yes")
    print()

    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        print("Warning: TOGETHER_API_KEY not set; API calls may fail.", file=sys.stderr)

    logs_base = Path(args.logs_dir) if args.logs_dir else project_root / "logs"
    solution_payload = _read_solution_source(
        project_root, problem_stem, args.solution, solutions_dir=solutions_dir
    )

    if args.mode == "dryrun":
        job_data = _run_one_job(
            "job",
            problem_stem,
            args.solution,
            args.m,
            args.n,
            args.dtype,
            args.measure_perf,
            args.profile,
            problem_id_int,
            measure_warmup_iters=args.measure_warmup_iters,
            measure_profiling_iters=args.measure_profiling_iters,
            solution_source=solution_payload,
        )
        urls = get_output_urls(job_data)
        perf_urls = _perf_json_urls(urls)
        if args.measure_perf and perf_urls:
            perf_files = download_urls_to_bytes_list(perf_urls, api_key)
            ps = summarize_perf_artifacts(perf_files)
            if ps:
                print(
                    f"Perf:      max wall time (across ranks) {ps['wall_time_ms_max']:.3f} ms "
                    f"(mean {ps['wall_time_ms_mean']:.3f} ms, ranks={ps['n_ranks']})"
                )
            else:
                print("Perf:      (no rank_*_perf.json found)", file=sys.stderr)
        elif args.measure_perf and urls and not perf_urls:
            print("Perf:      (no rank_*_perf.json in job outputs)", file=sys.stderr)
        elif args.measure_perf and not urls:
            print("Perf:      (no output URLs; cannot read rank_*_perf.json)", file=sys.stderr)

        if not args.download:
            print("Skipping download. Pass --download to save .pt/.json under logs/.")
            if args.measure_perf:
                print(
                    "Perf JSON not saved locally. Use --download together with --measure-perf "
                    "to save rank_*_perf.json under logs/."
                )
            if args.profile:
                print(
                    "Profiler traces not saved locally. Use --download together with --profile "
                    "to save traces/ under logs/."
                )
            sys.exit(0)

        if not urls:
            outputs = job_data.get("outputs") or {}
            payload_status = outputs.get("status") if isinstance(outputs, dict) else None
            if payload_status == "error":
                print("Job reported an error (no output files). Check: together beta jig logs --follow", file=sys.stderr)
            else:
                print("No output files to download.", file=sys.stderr)
            sys.exit(1)

        local_logs_dir = logs_base / f"problem_{problem_stem}" / args.solution
        print(f"Downloading {len(urls)} file(s) to {local_logs_dir}")
        download_outputs(urls, local_logs_dir, api_key)
        print(f"All files in: {local_logs_dir}")
        sys.exit(0)

    # eval: one Together job pair per trial (reference then solution) with distinct RNG inputs.
    ref_dir = logs_base / f"problem_{problem_stem}" / "reference"
    sol_dir = logs_base / f"problem_{problem_stem}" / args.solution
    if args.download:
        if ref_dir.is_dir():
            shutil.rmtree(ref_dir)
        if sol_dir.is_dir():
            shutil.rmtree(sol_dir)

    all_ok = True
    trial_reports: list[dict] = []
    for trial in range(args.trials):
        trial_report: dict = {"trial": trial}
        ref_data = _run_one_job(
            f"reference (trial {trial})",
            problem_stem,
            "reference",
            args.m,
            args.n,
            args.dtype,
            args.measure_perf,
            args.profile,
            problem_id_int,
            trial=trial,
            measure_warmup_iters=args.measure_warmup_iters,
            measure_profiling_iters=args.measure_profiling_iters,
        )
        if args.measure_perf and _PERF_THERMAL_COOLDOWN_SEC > 0:
            time.sleep(_PERF_THERMAL_COOLDOWN_SEC)
        sol_data = _run_one_job(
            f"solution (trial {trial})",
            problem_stem,
            args.solution,
            args.m,
            args.n,
            args.dtype,
            args.measure_perf,
            args.profile,
            problem_id_int,
            trial=trial,
            measure_warmup_iters=args.measure_warmup_iters,
            measure_profiling_iters=args.measure_profiling_iters,
            solution_source=solution_payload,
        )

        ref_urls = get_output_urls(ref_data)
        sol_urls = get_output_urls(sol_data)
        ref_rc = 0 if ref_urls else 1
        sol_rc = 0 if sol_urls else 1
        trial_report["reference_url_count"] = len(ref_urls)
        trial_report["solution_url_count"] = len(sol_urls)
        trial_report["reference_artifact_names"] = [k for k, _ in ref_urls]
        trial_report["solution_artifact_names"] = [k for k, _ in sol_urls]

        if not ref_urls or not sol_urls:
            trial_report["reference_job"] = _summarize_job_for_log(ref_data)
            trial_report["solution_job"] = _summarize_job_for_log(sol_data)
            rj, sj = trial_report["reference_job"], trial_report["solution_job"]
            harness = _harness_environment_hint(
                str(rj.get("outputs_stderr") or ""),
                str(rj.get("outputs_stdout") or ""),
                str(sj.get("outputs_stderr") or ""),
                str(sj.get("outputs_stdout") or ""),
            )
            if harness:
                trial_report["result"] = "harness_environment"
                trial_report["harness_hint"] = harness
                tag = "harness_failure (not a kernel correctness issue)"
            else:
                trial_report["result"] = "no_artifacts"
                tag = "cannot_compare (missing artifact URLs)"
            line = f"  trial {trial}: {tag}"
            if args.measure_perf:
                line += " | perf: n/a"
            print(line)
            if harness:
                print(f"    harness: {harness}", file=sys.stderr)
            hint = (
                f"Together eval: trial {trial}: no rank .pt URLs "
                f"(reference={len(ref_urls)}, solution={len(sol_urls)}). "
                "If harness_failure, fix deployment/image/driver; do not refine the solution kernel for this. "
                "Else see PKB_EVAL_JSON. CLI: together beta jig logs --follow"
            )
            print(hint, file=sys.stderr)
            trial_report["early_stop"] = "execution_or_harness_failure"
            if args.trials > 1:
                print(
                    f"  Stopping after trial {trial + 1}/{args.trials}: reference or solution did not "
                    "produce rank .pt URLs (compile/runtime/harness); further RNG trials skipped.",
                    flush=True,
                )
                print(
                    "(Early stop: fix compile/runtime or harness before re-running with --trials > 1.)",
                    file=sys.stderr,
                )
            trial_reports.append(trial_report)
            all_ok = False
            break

        ref_bytes = download_urls_to_bytes_list(ref_urls, api_key)
        sol_bytes = download_urls_to_bytes_list(sol_urls, api_key)
        compare = rank_outputs_compare_details(
            ref_bytes, sol_bytes, atol=args.atol, rtol=args.rtol
        )
        trial_ok = bool(compare.get("ok"))
        trial_report["result"] = "match" if trial_ok else compare.get("kind", "mismatch")
        trial_report["compare"] = compare

        ref_perf = summarize_perf_artifacts(ref_bytes) if args.measure_perf and ref_rc == 0 else None
        sol_perf = summarize_perf_artifacts(sol_bytes) if args.measure_perf and sol_rc == 0 else None
        speedup = None
        if (
            ref_perf is not None
            and sol_perf is not None
            and sol_perf["wall_time_ms_max"] > 0
        ):
            speedup = ref_perf["wall_time_ms_max"] / sol_perf["wall_time_ms_max"]

        line = f"  trial {trial}: {'match' if trial_ok else 'mismatch'}"
        if not trial_ok and compare.get("errors"):
            err0 = str(compare["errors"][0])
            if len(err0) > 220:
                err0 = err0[:217] + "..."
            line += f" | {err0}"
        if args.measure_perf:
            if ref_perf is not None and sol_perf is not None and speedup is not None:
                line += (
                    f" | ref {ref_perf['wall_time_ms_max']:.3f} ms | "
                    f"{args.solution} {sol_perf['wall_time_ms_max']:.3f} ms | speedup {speedup:.2f}x"
                )
            else:
                line += " | perf: n/a"
        print(line)
        if ref_rc != 0 or sol_rc != 0:
            print(
                f"    (reference exit={ref_rc}, solution exit={sol_rc})",
                file=sys.stderr,
            )
        if not trial_ok:
            all_ok = False
            for e in compare.get("errors") or []:
                print(f"    compare: {e}", file=sys.stderr)

        trial_reports.append(trial_report)

        if args.download:
            sub = f"trial_{trial}"
            t_ref = ref_dir / sub
            t_sol = sol_dir / sub
            print(f"Downloading reference ({sub}) -> {t_ref}")
            download_outputs(ref_urls, t_ref, api_key)
            print(f"Downloading {args.solution} ({sub}) -> {t_sol}")
            download_outputs(sol_urls, t_sol, api_key)

        if not trial_ok:
            trial_report["early_stop"] = "mismatch"
            if args.trials > 1:
                print(
                    f"  Stopping after trial {trial + 1}/{args.trials}: tensor mismatch; "
                    "further RNG trials skipped.",
                    flush=True,
                )
            break

    print("all trials match" if all_ok else "mismatch")

    eval_payload = {
        "version": 1,
        "backend": "together",
        "mode": "eval",
        "problem": problem_stem,
        "solution": args.solution,
        "all_ok": all_ok,
        "harness_environment_failed": any(
            isinstance(t, dict) and t.get("result") == "harness_environment" for t in trial_reports
        ),
        "trials": trial_reports,
    }
    eval_line = "PKB_EVAL_JSON:" + json.dumps(eval_payload, default=str, ensure_ascii=True)
    if len(eval_line) > 240_000:
        slim = {
            "version": 1,
            "backend": "together",
            "mode": "eval",
            "problem": problem_stem,
            "solution": args.solution,
            "all_ok": all_ok,
            "trials": [
                {
                    "trial": t.get("trial"),
                    "result": t.get("result"),
                    "compare_errors": (t.get("compare") or {}).get("errors"),
                    "reference_url_count": t.get("reference_url_count"),
                    "solution_url_count": t.get("solution_url_count"),
                }
                for t in trial_reports
            ],
            "_truncated": "full PKB_EVAL_JSON exceeded 240k; inspect run_together stdout or use --download",
        }
        eval_line = "PKB_EVAL_JSON:" + json.dumps(slim, default=str, ensure_ascii=True)
    print(eval_line, flush=True)

    if not args.download:
        if args.measure_perf:
            print(
                "Perf JSON not saved locally. Use --download together with --measure-perf "
                "to save rank_*_perf.json under logs/."
            )
        if args.profile:
            print(
                "Profiler traces not saved locally. Use --download together with --profile "
                "to save traces/ under logs/."
            )

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
