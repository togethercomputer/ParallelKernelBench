#!/usr/bin/env python3
"""
Kernel generation CLI: assembles the user prompt and optionally calls an LLM.

Generated files are written under a run-specific folder at the repo root so runs with different settings do not overwrite each other, e.g.
``solutions_{backend}_{precision}_{hardware|none}_{google|together}_{model_slug}/``

Flags:
  --print-prompt     Print the full assembled prompt to stdout.
  --problem / -p     Reference problem id (e.g. 1), "all" for every stem under PROBLEM_DIR, or a bracket list "[20, 21, 22]"
  --precision        Precision of the input tensors, specified in _PRECISION_CHOICES
  --hardware         Hardware topology (info to pass to the prompt)
  --backend          Backend (e.g. cuda, parallelkittens, triton)
  --paths-to-prompts-template   Path to prompts.toml.
  --model            Must be one of ALLOWED_MODELS (Gemini, Together, Anthropic, or OpenAI).

Environment:
  PROBLEM_DIR                        When --problem all: directory of *.py stems (default: reference/).
  GEMINI_API_KEY or GOOGLE_API_KEY   For Google models.
  TOGETHER_API_KEY                   For Together models.
  ANTHROPIC_API_KEY                  For Anthropic models.
  OPENAI_API_KEY                     For OpenAI models.
  TOGETHER_MAX_TOKENS                Optional max output tokens for Together chat completions (default 100000)
  OPENAI_MAX_TOKENS                  Optional max output tokens for OpenAI chat completions (default 100000)
  ANTHROPIC_MAX_TOKENS               Optional max output tokens for Anthropic chat completions (default 100000)
  LLM_MAX_TOKENS                     Fallback max output tokens for chat providers when a provider-specific var is unset.
  TOGETHER_TIMEOUT_SECS              Optional HTTP read timeout in seconds for each Together chat completion (default 3600).
  OPENAI_TIMEOUT_SECS                Optional HTTP read timeout in seconds for each OpenAI chat completion (default 3600).
  ANTHROPIC_TIMEOUT_SECS             Optional HTTP read timeout in seconds for each Anthropic chat completion (default 3600).
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import toml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.problem_id import resolve_problem

_PROMPTS_TOML = Path(__file__).resolve().parent / "prompts.toml"

_PRECISION_CHOICES = ("fp32", "fp16", "bf16")
_HARDWARE_CHOICES = ("h100_8", "b200_72")

Provider = Literal["google", "together", "anthropic", "openai"]

# Curated allowlist: --model must be a key. Value is the API backend.
ALLOWED_MODELS: dict[str, Provider] = {
    "gemini-3-flash-preview": "google",
    "gemini-3-pro-preview": "google",
    "zai-org/GLM-5.1": "together",
    "deepseek-ai/DeepSeek-V4-Pro": "together",
    "Qwen/Qwen3-Coder-Next-FP8": "together",
    "claude-sonnet-4-20250514": "anthropic",
    "claude-opus-4-20250514": "anthropic",
    "gpt-4.1": "openai",
    "gpt-4o": "openai",
    "o3": "openai",
}

_DEFAULT_MODEL = "gemini-2.5-flash"

# Chat providers apply a server default max output length when ``max_tokens`` is omitted.
# Long CUDA/TK sources then stop mid-file (finish_reason=length).
_CHAT_DEFAULT_MAX_TOKENS = 100000
_CHAT_DEFAULT_TIMEOUT_SEC = 3600.0

# System instruction for the model; task text is the assembled user prompt from TOML.
# Default system instruction for --backend cuda (and any backend without generate_kernel_system_prompt).
SYSTEM_PROMPT = """You are an expert CUDA and distributed systems engineer.

Hard requirements for every answer:
- Replace NCCL / torch.distributed collectives with custom CUDA using torch.distributed._symmetric_memory (symm_mem), UVA device pointers, and utils.cuda_helpers.compile_cuda_extension for JIT compilation.
- Preserve the reference solution() signature and numerical correctness.
- Follow the user prompt below for tone, examples, and output format."""


def _chat_max_tokens(provider_env_var: str) -> int:
    for key in (provider_env_var, "LLM_MAX_TOKENS"):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                raise ValueError(
                    f"{key} must be a positive integer, got {raw!r}"
                ) from None
    return _CHAT_DEFAULT_MAX_TOKENS


def _chat_timeout_sec(provider_env_var: str) -> float:
    raw = os.environ.get(provider_env_var, "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            raise ValueError(
                f"{provider_env_var} must be a positive number, got {raw!r}"
            ) from None
    return _CHAT_DEFAULT_TIMEOUT_SEC


def _together_max_tokens() -> int:
    return _chat_max_tokens("TOGETHER_MAX_TOKENS")


def _together_timeout_sec() -> float:
    return _chat_timeout_sec("TOGETHER_TIMEOUT_SECS")

# helper function: if you want to specify a custom system prompt for a particular backend in your experiments
def system_prompt_for_backend(cfg: dict[str, Any], backend: str) -> str:
    """Use backends.<name>.generate_kernel_system_prompt when set; else SYSTEM_PROMPT."""
    row: dict[str, Any] = (cfg.get("backends") or {}).get(backend) or {}
    custom = row.get("generate_kernel_system_prompt")
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    return SYSTEM_PROMPT

# helper function that extracts the actual python code from a "python```" code block
def extract_python_from_response(text: str) -> str:
    """Extract Python source from model output (handles markdown code blocks)."""
    text = text.strip()
    for pattern in (
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def _read_text(project_root: Path, relative_path: str) -> str:
    p = project_root / relative_path
    if not p.is_file():
        raise FileNotFoundError(f"Referenced file not found: {p}")
    return p.read_text()


def _load_reference(project_root: Path, problem_stem: str) -> str:
    ref_path = project_root / "reference" / f"{problem_stem}.py"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    return ref_path.read_text()


def _default_problem_dir(project_root: Path) -> Path:
    env = os.environ.get("PROBLEM_DIR")
    if env:
        problem_dir = Path(env).expanduser().resolve()
    else:
        problem_dir = project_root / "reference"
    if not problem_dir.is_dir():
        raise FileNotFoundError(f"Problem directory not found: {problem_dir}")
    return problem_dir


def _slug_model_for_path(model: str, *, max_len: int = 72) -> str:
    """Filesystem-safe fragment from a model id (Together ids often contain '/')."""
    t = model.strip().replace("/", "_")
    t = re.sub(r"[^a-zA-Z0-9._\-]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "model"
    return t[:max_len]


def output_run_dir_relative(
    *,
    backend: str,
    precision: str,
    hardware: str | None,
    model: str,
) -> str:
    """
    Directory name (relative to project root) for this run's generated solutions.

    Pattern: solutions_<backend>_<precision>_<hardware|none>_<provider>_<model_slug>
    """
    hw = hardware if hardware else "none"
    prov = ALLOWED_MODELS[model]
    mslug = _slug_model_for_path(model)
    return f"solutions_{backend}_{precision}_{hw}_{prov}_{mslug}"


def _list_problem_stems(project_root: Path) -> list[str]:
    problem_dir = _default_problem_dir(project_root)
    stems = sorted(p.stem for p in problem_dir.glob("*.py") if p.is_file())
    if not stems:
        raise FileNotFoundError(f"No *.py files in {problem_dir}")
    return stems


def _problem_arg_to_stems(project_root: Path, raw_problem: str) -> list[str]:
    """
    Map --problem to a list of reference stems.

    - ``all`` (case-insensitive): every stem under PROBLEM_DIR (same as _list_problem_stems).
    - ``[a, b, c]``: each entry is resolved like a single --problem (numeric id or stem).
    - Otherwise: a single problem via resolve_problem.
    """
    raw = raw_problem.strip()
    if raw.lower() == "all":
        return _list_problem_stems(project_root)

    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            raise ValueError("Empty problem list in [...]")
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if not parts:
            raise ValueError("Empty problem list in [...]")
        stems: list[str] = []
        seen: set[str] = set()
        for part in parts:
            stem, _ = resolve_problem(project_root, part, "cuda")
            if stem not in seen:
                seen.add(stem)
                stems.append(stem)
        return stems

    stem, _ = resolve_problem(project_root, raw, "cuda")
    return [stem]


################################################################################################
# Specifying what's in the LLM prompt
################################################################################################

def _hardware_context_body(cfg: dict[str, Any], hardware_key: str | None) -> str:
    """Concatenate templates.hardware using a [hardware_profiles.*] row, or empty string."""
    if not hardware_key:
        return ""
    profiles = cfg.get("hardware_profiles") or {}
    if hardware_key not in profiles:
        allowed = ", ".join(sorted(profiles.keys()))
        raise KeyError(
            f"hardware_profiles.{hardware_key} missing in prompts.toml (have: {allowed})"
        )
    prof: dict[str, Any] = profiles[hardware_key]
    tpl: dict[str, Any] = cfg["templates"]["hardware"]
    ctx = {k: str(v) for k, v in prof.items()}
    parts = [
        tpl["hardware_header"].strip(),
        tpl["hardware_specs"].format(**ctx),
        tpl["hardware_definitions"].format(**ctx),
        tpl["hardware_best_practices"].format(**ctx),
    ]
    return "\n\n".join(p for p in parts if p)


def build_incontext_examples(
    cfg: dict[str, Any],
    *,
    backend: str,
    precision: str,
    ref_arch_src: str,
    project_root: Path,
    hardware_key: str | None,
) -> dict[str, str]:
    """Build examples_block from the backend's explicit in-context examples."""
    backend_cfg: dict[str, Any] = cfg["backends"][backend]
    prec_cfg: dict[str, Any] = cfg["precision"][precision]
    common: dict[str, Any] = cfg["templates"]["common"]

    ctx: dict[str, str] = {
        "backend_display": backend_cfg["backend_display"],
        "precision_display": prec_cfg["precision_display"],
        "ref_arch_src": ref_arch_src,
        "hardware_context_body": _hardware_context_body(cfg, hardware_key),
    }

    examples = backend_cfg.get("in_context_examples") or []
    if not isinstance(examples, list) or not examples:
        raise ValueError(
            f"backends.{backend}.in_context_examples must be a non-empty list"
        )

    ctx["examples_intro"] = (
        common["example_intro_one_shot"]
        if len(examples) == 1
        else common["example_intro_few_shot"]
    ).format(**ctx)

    entry_parts: list[str] = []
    for i, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(
                f"backends.{backend}.in_context_examples[{i}] must be a table"
            )
        try:
            inp_rel = str(example["pytorch_solution"])
            out_rel = str(example["custom_solution"])
        except KeyError as e:
            raise ValueError(
                f"backends.{backend}.in_context_examples[{i}] missing {e.args[0]!r}"
            ) from None

        block_ctx = {
            **ctx,
            "example_label": str(example.get("label") or f"Example {i + 1}"),
            "input_code": _read_text(project_root, inp_rel),
            "output_code": _read_text(project_root, out_rel),
        }
        entry_parts.append(common["example_entry_template"].format(**block_ctx))

    ctx["examples_entries"] = "\n\n".join(entry_parts)

    return ctx


def assemble_user_prompt(
    cfg: dict[str, Any],
    *,
    backend: str,
    precision: str,
    ref_arch_src: str,
    project_root: Path,
    problem_stem: str,
    hardware_key: str | None,
    solution_relative_path: str | None = None,
) -> str:
    """Build the user message from [options.generate_kernel] components (one-shot layout)."""
    opt_cfg = cfg["options"]["generate_kernel"]
    components: list[str] = opt_cfg["components"]

    ctx = build_incontext_examples(
        cfg,
        backend=backend,
        precision=precision,
        ref_arch_src=ref_arch_src,
        project_root=project_root,
        hardware_key=hardware_key,
    )

    shared: dict[str, Any] = cfg["shared"]
    common: dict[str, Any] = cfg["templates"]["common"]

    def render(name: str) -> str:
        if name in shared:
            return shared[name].format(**ctx)
        if name in common:
            return common[name].format(**ctx)
        raise KeyError(f"Unknown prompt component {name!r}")

    parts = [render(c).strip() for c in components]
    body = "\n\n".join(p for p in parts if p)

    out_rel = solution_relative_path or f"solutions_cuda/{problem_stem}_cuda.py"
    footer = (
        f"\nThe generated file will be saved as: {out_rel}\n"
        f"Produce the complete Python file contents for that path."
    )
    return body + footer

################################################################################################
# Calling the LLM provider API
################################################################################################

def _call_gemini(model: str, user_prompt: str, system_prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("set GEMINI_API_KEY or GOOGLE_API_KEY to call Gemini.")
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise RuntimeError("pip install google-generativeai") from e

    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
    )
    response = m.generate_content(user_prompt)
    raw = response.text or ""
    if not raw:
        raise RuntimeError("Gemini returned empty response.")
    return raw


def _call_together(model: str, user_prompt: str, system_prompt: str) -> str:
    if not os.environ.get("TOGETHER_API_KEY"):
        raise RuntimeError("set TOGETHER_API_KEY for Together models.")
    try:
        from together import Together
    except ImportError as e:
        raise RuntimeError("pip install together") from e

    client = Together(timeout=_together_timeout_sec())
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Required for full-file kernels: Together's implicit default is often too small.
        "max_tokens": _together_max_tokens(),
    }
    try:
        response = client.chat.completions.create(
            **kwargs,
            reasoning={"enabled": False},
        )
    except TypeError:
        response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    msg = choice.message
    raw = (getattr(msg, "content", None) or "") if msg is not None else ""
    raw = raw.strip() if isinstance(raw, str) else ""
    if not raw:
        raise RuntimeError("Together returned empty response.")
    return raw


def _call_anthropic(model: str, user_prompt: str, system_prompt: str) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("set ANTHROPIC_API_KEY for Anthropic models.")
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic") from e

    client = Anthropic(timeout=_chat_timeout_sec("ANTHROPIC_TIMEOUT_SECS"))
    response = client.messages.create(
        model=model,
        max_tokens=_chat_max_tokens("ANTHROPIC_MAX_TOKENS"),
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    raw = "".join(parts).strip()
    if not raw:
        raise RuntimeError("Anthropic returned empty response.")
    return raw


def _call_openai(model: str, user_prompt: str, system_prompt: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("set OPENAI_API_KEY for OpenAI models.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    client = OpenAI(timeout=_chat_timeout_sec("OPENAI_TIMEOUT_SECS"))
    response = client.chat.completions.create(
        model=model,
        max_tokens=_chat_max_tokens("OPENAI_MAX_TOKENS"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    choice = response.choices[0]
    msg = choice.message
    raw = (getattr(msg, "content", None) or "") if msg is not None else ""
    raw = raw.strip() if isinstance(raw, str) else ""
    if not raw:
        raise RuntimeError("OpenAI returned empty response.")
    return raw

def _complete_model(model: str, user_prompt: str, system_prompt: str) -> str:
    provider = ALLOWED_MODELS.get(model)
    if provider is None:
        raise RuntimeError(
            f"unknown model {model!r}. Allowed: {', '.join(sorted(ALLOWED_MODELS))}"
        )
    if provider == "google":
        return _call_gemini(model, user_prompt, system_prompt)
    if provider == "anthropic":
        return _call_anthropic(model, user_prompt, system_prompt)
    if provider == "openai":
        return _call_openai(model, user_prompt, system_prompt)
    return _call_together(model, user_prompt, system_prompt)



################################################################################################
# Parse user arguments and do main generation logic
################################################################################################
@dataclass(frozen=True)
class KernelGenArgs:
    print_prompt: bool
    stems: list[str]
    precision: str
    hardware: str | None
    backend: str
    paths_to_prompts_template: Path
    model: str
    # Relative directory under the project root for this run; see output_run_dir_relative().
    output_dir_rel: str

# Parses all user-given arguments
def parse_args(argv: list[str] | None = None) -> KernelGenArgs:
    parser = argparse.ArgumentParser(
        description="Assemble kernel-generation prompt from prompts.toml and optionally call an LLM."
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the full assembled prompt to stdout.",      # "Testing": allows you to make sure the prompt is what you expect.
    )
    parser.add_argument(
        "--problem",
        "-p",
        default="1",
        metavar="ID",
        help=(
            'Reference problem id (e.g. 1), "all" for every stem under '
            'PROBLEM_DIR, or a list like "[20, 21, 22]" for only those problems.'
        ),
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=_PRECISION_CHOICES,
        metavar="{" + "|".join(_PRECISION_CHOICES) + "}",
        help="Floating-point precision: exact token only (default: bf16).",
    )
    parser.add_argument(
        "--hardware",
        default=None,
        choices=_HARDWARE_CHOICES,
        metavar="{" + "|".join(_HARDWARE_CHOICES) + "}",
        help="Hardware profile; fills [hardware_profiles.*] in prompts.toml. Omit for no hardware section.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Key under [backends] (default: meta.default_backend in prompts.toml).",
    )
    parser.add_argument(
        "--paths-to-prompts-template",
        type=Path,
        default=_PROMPTS_TOML,
        help="Path to prompts.toml.",
    )
    allowed_help = ", ".join(sorted(ALLOWED_MODELS))
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        metavar="NAME",
        help=f"Model id (must be one of: {allowed_help}). Ignored with --print-prompt.",
    )
    ns = parser.parse_args(argv)


    # Check that are args are valid:
    model = str(ns.model)                           # make sure model is allowed
    if model not in ALLOWED_MODELS:
        print(
            f"Error: --model must be one of: {allowed_help}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    raw_problem = str(ns.problem).strip()           # make sure problem file is actually found
    try:
        stems = _problem_arg_to_stems(_PROJECT_ROOT, raw_problem)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    prompts_path = ns.paths_to_prompts_template.resolve()       # verify prompts.toml is actually there
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {prompts_path}")
    cfg = toml.load(prompts_path)
    meta = cfg.get("meta", {})
    backend = ns.backend or meta.get("default_backend", "cuda")     # verify we have an allowed backend
    if backend not in cfg.get("backends", {}):
        print(f"Error: unknown backend {backend!r}", file=sys.stderr)
        raise SystemExit(1)
    if ns.precision not in cfg.get("precision", {}):
        print(f"Error: unknown precision {ns.precision!r}", file=sys.stderr)
        raise SystemExit(1)

    out_dir_rel = output_run_dir_relative(
        backend=backend,
        precision=ns.precision,
        hardware=ns.hardware,
        model=model,
    )

    return KernelGenArgs(
        print_prompt=ns.print_prompt,
        stems=stems,
        precision=ns.precision,
        hardware=ns.hardware,
        backend=backend,
        paths_to_prompts_template=prompts_path,
        model=model,
        output_dir_rel=out_dir_rel,
    )


def generate_kernel(
    cfg: dict[str, Any],
    *,
    stem: str,
    args: KernelGenArgs,
) -> None:
    rel_file = f"{args.output_dir_rel}/{stem}_{args.backend}.py"
    ref_src = _load_reference(_PROJECT_ROOT, stem)
    prompt = assemble_user_prompt(
        cfg,
        backend=args.backend,
        precision=args.precision,
        ref_arch_src=ref_src,
        project_root=_PROJECT_ROOT,
        problem_stem=stem,
        hardware_key=args.hardware,
        solution_relative_path=rel_file,
    )

    if args.print_prompt:
        sep = "=" * 72
        print(f"{sep}\nPROBLEM {stem}\n{sep}\n{prompt}")
        return

    print(
        f"Calling {ALLOWED_MODELS[args.model]} (model={args.model}) for problem {stem}...",
        flush=True,
    )
    sys_prompt = system_prompt_for_backend(cfg, args.backend)
    raw = _complete_model(args.model, prompt, sys_prompt)
    py_content = extract_python_from_response(raw)
    out_path = _PROJECT_ROOT / rel_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(py_content)
    print(f"Wrote: {out_path}")


def main() -> None:
    # load arguments
    args = parse_args()

    # load prompts.toml file
    prompts_path = args.paths_to_prompts_template
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {prompts_path}")
    cfg = toml.load(prompts_path)

    if not args.print_prompt:
        print(f"Saving solutions under: {_PROJECT_ROOT / args.output_dir_rel}", file=sys.stderr, flush=True)

    # generate kernel
    rc = 0
    for stem in args.stems:
        try:
            generate_kernel(cfg, stem=stem, args=args)
        except Exception as e:
            print(f"[{stem}] {e}", file=sys.stderr)
            rc = 1
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
