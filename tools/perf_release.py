#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and report the release TRTMC-versus-reference performance matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = REPOSITORY / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from tensorrt_model_connect.benchmark.catalog import ManifestCatalog  # noqa: E402
from tensorrt_model_connect.python_profiles import (  # noqa: E402
    default_execution_profiles,
    resolve_profile_python,
)


RESULT_SCHEMA = "trtmc.perf-release/v1"
SUITE_SCHEMA = "trtmc.perf-suite/v1"
TERMINAL_STATUSES = {
    "green",
    "yellow",
    "red",
    "unsupported",
    "failed",
    "contract-mismatch",
    "partial",
}
PRIORITIES = {"fast": 0, "normal": 1, "slow": 2}
SEQUENCE_RUNTIME_MARKERS = ("bart_", "marian_", "m2m_100_", "t5_")


class PerfReleaseError(RuntimeError):
    """The release performance request or evidence is invalid."""


@dataclass(frozen=True)
class RunOptions:
    output: Path
    trtmc_bench: str
    trtmc_worker: Path | None
    hf_transformers_runner: Path
    bundle_cache: Path | None
    bundle_roots: tuple[Path, ...]
    runtime_dirs: tuple[Path, ...]
    only: str
    dry_run: bool
    ci: bool
    resume: bool
    rerun_failed: bool
    local_files_only: bool
    timeout_seconds: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="release performance suite YAML")
    parser.add_argument("--output", type=Path, default=Path("artifacts/perf"))
    parser.add_argument("--case", action="append", default=[], help="case id; repeatable")
    parser.add_argument(
        "--priority",
        choices=tuple(PRIORITIES),
        help="run this priority and faster priorities",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--only", choices=("both", "trtmc", "baseline"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ci", action="store_true", help="fail after reporting unexpected rows")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--trtmc-bench", default=os.environ.get("TRTMC_BENCH", "trtmc-bench"))
    parser.add_argument(
        "--trtmc-worker",
        type=Path,
        default=Path(os.environ["TRTMC_PERF_WORKER"])
        if os.environ.get("TRTMC_PERF_WORKER")
        else None,
    )
    parser.add_argument(
        "--hf-transformers-runner",
        type=Path,
        default=REPOSITORY / "benchmarks/performance/baselines/hf_transformers.py",
    )
    parser.add_argument(
        "--bundle-cache",
        type=Path,
        default=(
            Path(os.environ["TRTMC_PERF_BUNDLE_CACHE"])
            if os.environ.get("TRTMC_PERF_BUNDLE_CACHE")
            else None
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--bundle-root",
        action="append",
        default=[
            Path(value)
            for value in os.environ.get("TRTMC_PERF_BUNDLE_ROOTS", "").split(os.pathsep)
            if value
        ],
        type=Path,
    )
    parser.add_argument(
        "--runtime-dir",
        action="append",
        default=[
            Path(value)
            for value in os.environ.get("TRTMC_PERF_RUNTIME_DIRS", "").split(os.pathsep)
            if value
        ],
        type=Path,
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PerfReleaseError(f"cannot read suite {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PerfReleaseError("performance suite must contain a YAML object")
    if value.get("schema_version") != SUITE_SCHEMA:
        raise PerfReleaseError(f"suite schema_version must be {SUITE_SCHEMA!r}")
    return value


def _cases(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    defaults = suite.get("defaults", {})
    configured = suite.get("cases")
    if not isinstance(defaults, Mapping):
        raise PerfReleaseError("suite defaults must be an object")
    if not isinstance(configured, list) or not configured:
        raise PerfReleaseError("suite cases must be a non-empty list")
    cases: list[dict[str, Any]] = []
    for raw in configured:
        if not isinstance(raw, Mapping):
            raise PerfReleaseError("every suite case must be an object")
        merged = _merge_case(defaults, raw)
        _validate_case_shape(merged)
        cases.append(merged)
    _validate_unique_ids(cases)
    return cases


def _merge_case(defaults: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(defaults))
    for key, value in case.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_case_shape(case: Mapping[str, Any]) -> None:
    required = ("id", "family", "operation", "model", "priority", "measurement", "baseline")
    missing = [name for name in required if name not in case]
    if missing:
        raise PerfReleaseError(f"case is missing {', '.join(missing)}: {case}")
    if case["priority"] not in PRIORITIES:
        raise PerfReleaseError(f"case {case['id']} has invalid priority {case['priority']!r}")
    _validate_measurement(case)
    _validate_baseline(case)


def _validate_measurement(case: Mapping[str, Any]) -> None:
    measurement = case["measurement"]
    if not isinstance(measurement, Mapping):
        raise PerfReleaseError(f"case {case['id']} measurement must be an object")
    for name in ("warmup", "iterations"):
        value = measurement.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if name == "warmup" else 1)
        ):
            raise PerfReleaseError(f"case {case['id']} measurement.{name} is invalid")


def _validate_baseline(case: Mapping[str, Any]) -> None:
    baseline = case["baseline"]
    if not isinstance(baseline, Mapping) or baseline.get("runner") not in {
        "hf-transformers",
        "unsupported",
    }:
        raise PerfReleaseError(f"case {case['id']} has an unsupported baseline runner")
    if baseline.get("runner") == "unsupported" and not baseline.get("reason"):
        raise PerfReleaseError(f"case {case['id']} unsupported baseline needs a reason")
    token_policy = baseline.get("output_token_policy", "new-tokens")
    if token_policy not in {"new-tokens", "strip-start", "strip-start-and-eos"}:
        raise PerfReleaseError(f"case {case['id']} baseline output token policy is invalid")
    if baseline.get("padding", "longest") not in {"longest", "max-length"}:
        raise PerfReleaseError(f"case {case['id']} baseline padding is invalid")
    if baseline.get("precision") not in {None, "fp16", "fp32", "bf16"}:
        raise PerfReleaseError(f"case {case['id']} baseline precision is invalid")
    if baseline.get("model_class", "task") not in {"task", "auto"}:
        raise PerfReleaseError(f"case {case['id']} baseline model class is invalid")
    if baseline.get("generation_method", "generate") not in {"generate", "ar-generate"}:
        raise PerfReleaseError(f"case {case['id']} baseline generation method is invalid")
    if baseline.get("experts_implementation") not in {
        None,
        "eager",
        "batched_mm",
        "grouped_mm",
    }:
        raise PerfReleaseError(f"case {case['id']} baseline experts implementation is invalid")
    if baseline.get("output_contract", "exact-token-ids") not in {
        "exact-token-ids",
        "exact-text",
    }:
        raise PerfReleaseError(f"case {case['id']} baseline output contract is invalid")


def _validate_unique_ids(cases: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(case["id"]) for case in cases]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise PerfReleaseError(f"duplicate case ids: {', '.join(duplicates)}")


def _validate_coverage(cases: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (entry.family, entry.operation)
        for entry in ManifestCatalog().entries()
        if entry.status == "ready"
    }
    actual = {(str(case["family"]), str(case["operation"])) for case in cases}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra or len(actual) != len(cases):
        details = _coverage_details(missing, extra, len(actual) != len(cases))
        raise PerfReleaseError("suite coverage does not match ready catalog: " + "; ".join(details))


def _coverage_details(
    missing: Sequence[tuple[str, str]],
    extra: Sequence[tuple[str, str]],
    duplicates: bool,
) -> list[str]:
    details = []
    if missing:
        details.append("missing=" + ",".join(f"{a}.{b}" for a, b in missing))
    if extra:
        details.append("extra=" + ",".join(f"{a}.{b}" for a, b in extra))
    if duplicates:
        details.append("more than one case selects the same family-operation")
    return details


def _selected_cases(
    cases: Sequence[dict[str, Any]],
    requested: Sequence[str],
    priority: str | None,
    max_cases: int | None,
) -> list[dict[str, Any]]:
    requested = [value for value in requested if value]
    _validate_requested_cases(cases, requested)
    selected = [case for case in cases if not requested or case["id"] in requested]
    if priority is not None and not requested:
        selected = _select_through_priority(selected, priority)
    selected.sort(key=lambda case: (PRIORITIES[str(case["priority"])], str(case["id"])))
    if max_cases is not None:
        if max_cases <= 0:
            raise PerfReleaseError("--max-cases must be positive")
        selected = selected[:max_cases]
    return selected


def _validate_requested_cases(cases: Sequence[Mapping[str, Any]], requested: Sequence[str]) -> None:
    known = {str(case["id"]) for case in cases}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise PerfReleaseError(f"unknown case ids: {', '.join(unknown)}")


def _select_through_priority(
    cases: Sequence[dict[str, Any]], priority: str
) -> list[dict[str, Any]]:
    ceiling = PRIORITIES[priority]
    return [case for case in cases if PRIORITIES[str(case["priority"])] <= ceiling]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    configured = os.environ.get("TRTMC_PERF_SOURCE_REVISION") or os.environ.get("GITHUB_SHA")
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_environment() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"gpu": None, "driver": None}
    try:
        output = subprocess.run(
            [executable, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        name, uuid, driver = (part.strip() for part in output.split(",", maxsplit=2))
        return {"gpu": name, "gpu_uuid": uuid, "driver": driver}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"gpu": None, "driver": None}


def _initial_results(
    suite_path: Path,
    suite: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    environment = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        **_gpu_environment(),
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "running",
        "suite": str(suite_path.resolve()),
        "suite_name": str(suite.get("name", suite_path.stem)),
        "suite_sha256": _sha256_file(suite_path),
        "repository_root": str(REPOSITORY),
        "git_commit": _git_commit(),
        "started_at": _now(),
        "environment": environment,
        "cases": [
            {
                "id": case["id"],
                "family": case["family"],
                "operation": case["operation"],
                "model": case["model"],
                "priority": case["priority"],
                "baseline_contract": dict(case["baseline"]),
                "status": "pending",
            }
            for case in cases
        ],
    }


def _load_resume(path: Path, suite_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfReleaseError(f"cannot resume results {path}: {exc}") from exc
    if value.get("schema_version") != RESULT_SCHEMA:
        raise PerfReleaseError(f"cannot resume non-{RESULT_SCHEMA} results")
    if value.get("suite_sha256") != _sha256_file(suite_path):
        raise PerfReleaseError("cannot resume because the suite content changed")
    value["status"] = "running"
    value.pop("finished_at", None)
    return value


def _result_rows(results: MutableMapping[str, Any]) -> dict[str, MutableMapping[str, Any]]:
    return {str(row["id"]): row for row in results["cases"]}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _yaml_cli_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _candidate_base_argv(case: Mapping[str, Any], options: RunOptions) -> list[str]:
    measurement = case["measurement"]
    argv = [
        options.trtmc_bench,
        "run",
        "--model",
        str(case["model"]),
        "--warmup",
        str(measurement["warmup"]),
        "--iterations",
        str(measurement["iterations"]),
        "--telemetry",
        "off",
    ]
    testcase = case.get("testcase")
    if testcase:
        argv.extend(["--case", str(testcase)])
    if options.bundle_cache is not None:
        argv.extend(["--bundle-cache", str(options.bundle_cache.resolve())])
    if options.trtmc_worker is not None:
        argv.extend(["--worker", str(options.trtmc_worker.resolve())])
    for root in options.bundle_roots:
        argv.extend(["--bundle-root", str(root.resolve())])
    for directory in options.runtime_dirs:
        argv.extend(["--runtime-dir", str(directory.resolve())])
    request = case.get("request", {})
    if request and not isinstance(request, Mapping):
        raise PerfReleaseError(f"case {case['id']} request must be an object")
    for name, value in sorted((request or {}).items()):
        argv.extend(["--set", f"request.{name}={_yaml_cli_value(value)}"])
    return argv


def _command_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    paths = [str(PYTHON_SOURCE)]
    if existing:
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _resolve_candidate(
    case: Mapping[str, Any], options: RunOptions
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    argv = [*_candidate_base_argv(case, options), "--dry-run"]
    command = _run_command(argv, _command_environment(), options.timeout_seconds)
    if command["exit_code"] != 0:
        raise PerfReleaseError(
            f"trtmc-bench could not resolve {case['id']}: {command['stderr_tail']}"
        )
    try:
        resolved = json.loads(command["stdout"])
    except json.JSONDecodeError as exc:
        raise PerfReleaseError(f"trtmc-bench dry-run returned invalid JSON: {exc}") from exc
    if not isinstance(resolved, list) or len(resolved) != 1 or not isinstance(resolved[0], dict):
        raise PerfReleaseError(f"case {case['id']} must resolve to exactly one trtmc-bench case")
    value = resolved[0]
    if (value.get("model", {}).get("family"), value.get("operation")) != (
        case["family"],
        case["operation"],
    ):
        raise PerfReleaseError(f"case {case['id']} does not match its resolved family-operation")
    command.pop("stdout", None)
    return value, argv, command


def _workload_digest(resolved: Mapping[str, Any]) -> str:
    model = resolved["model"]
    payload = {
        "schema_version": "trtmc.perf-workload/v1",
        "model": {
            "hf_id": model["hf_id"],
            "family": model["family"],
            "task_strategy": model["task_strategy"],
        },
        "operation": resolved["operation"],
        "request": resolved["request"],
        "precision": model["precision"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _manifest_values(resolved: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(resolved["model"]["manifest_path"]))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfReleaseError(f"cannot read resolved model manifest {path}: {exc}") from exc
    return value


def _baseline_task(resolved: Mapping[str, Any]) -> str:
    strategy = str(resolved["model"]["task_strategy"])
    runtime = str(resolved["model"]["runtime_strategy"])
    if strategy == "encoder_only_nlp":
        return "encoder"
    if strategy != "text_generation_causal":
        raise PerfReleaseError(f"hf-transformers runner does not support task {strategy!r}")
    return "seq2seq-lm" if runtime.startswith(SEQUENCE_RUNTIME_MARKERS) else "causal-lm"


def _reference_python(resolved: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[str, str]:
    explicit = str(baseline.get("python_profile", "") or "").strip()
    model = resolved["model"]
    profile = (
        explicit
        or default_execution_profiles(
            family=str(model["family"]),
            runtime_strategy=str(model["runtime_strategy"]),
            reference_backend="hf_transformers",
        )["reference"]
    )
    return profile, resolve_profile_python(profile, sys.executable)


def _baseline_argv(
    case: Mapping[str, Any],
    resolved: Mapping[str, Any],
    output: Path,
    options: RunOptions,
) -> tuple[list[str], str]:
    baseline = case["baseline"]
    profile, python = _reference_python(resolved, baseline)
    model = resolved["model"]
    manifest = _manifest_values(resolved)
    mode = str(baseline.get("mode", "torch-compile"))
    if mode not in {"torch-compile", "hf-eager"}:
        raise PerfReleaseError(f"case {case['id']} baseline mode is invalid: {mode}")
    build = model.get("build", {})
    max_length = int(baseline.get("max_length", build.get("max_cache_length", 256)))
    argv = [
        python,
        str(options.hf_transformers_runner.resolve()),
        "--model",
        str(model["hf_id"]),
        "--task",
        str(baseline.get("task") or _baseline_task(resolved)),
        "--request-json",
        json.dumps(resolved["request"], ensure_ascii=True, separators=(",", ":")),
        "--precision",
        str(baseline.get("precision", model["precision"])),
        "--max-length",
        str(max_length),
        "--padding",
        str(baseline.get("padding", "longest")),
        "--mode",
        mode,
        "--warmup",
        str(case["measurement"]["warmup"]),
        "--iterations",
        str(case["measurement"]["iterations"]),
        "--workload-digest",
        _workload_digest(resolved),
        "--output-token-policy",
        str(baseline.get("output_token_policy", "new-tokens")),
        "--output",
        str(output),
    ]
    revision = baseline.get("revision", manifest.get("hf_revision"))
    if revision:
        argv.extend(["--revision", str(revision)])
    if bool(manifest.get("trust_remote_code", False)):
        argv.append("--trust-remote-code")
    if baseline.get("experts_implementation"):
        argv.extend(["--experts-implementation", str(baseline["experts_implementation"])])
    if baseline.get("model_class"):
        argv.extend(["--model-class", str(baseline["model_class"])])
    if baseline.get("generation_method"):
        argv.extend(["--generation-method", str(baseline["generation_method"])])
    if options.local_files_only:
        argv.append("--local-files-only")
    if mode == "torch-compile":
        argv.extend(["--compile-mode", str(baseline.get("compile_mode", "default"))])
        if bool(baseline.get("fullgraph", False)):
            argv.append("--compile-fullgraph")
        if bool(baseline.get("dynamic", True)):
            argv.append("--compile-dynamic")
    return argv, profile


def _run_command(
    argv: Sequence[str], environment: Mapping[str, str], timeout_seconds: int
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        process = subprocess.run(
            list(argv),
            cwd=REPOSITORY,
            env=dict(environment),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = (
            _decode_timeout_stream(exc.stderr) + f"\ncommand timed out after {timeout_seconds}s"
        )
    except OSError as exc:
        exit_code = 127
        stdout = ""
        stderr = str(exc)
    elapsed = time.monotonic() - started
    return {
        "argv": list(argv),
        "rendered": shlex.join(argv),
        "cwd": str(REPOSITORY),
        "redacted_environment_names": [
            name for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") if environment.get(name)
        ],
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "stdout": stdout,
        "stdout_tail": stdout[-16000:],
        "stderr_tail": stderr[-16000:],
    }


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _candidate_result(run_dir: Path, workload_digest: str) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfReleaseError(f"cannot read trtmc-bench result: {exc}") from exc
    if result.get("schema_version") != "trtmc.benchmark-run/v1":
        raise PerfReleaseError("trtmc-bench returned an unsupported result schema")
    cells = result.get("cells", [])
    if result.get("status") != "completed" or len(cells) != 1:
        error = cells[0].get("error") if len(cells) == 1 else result.get("status")
        raise PerfReleaseError(f"trtmc-bench did not complete one case: {error}")
    cell = cells[0]
    artifact = run_dir / str(cell["artifact_dir"])
    observations = []
    try:
        for line in (artifact / "observations.jsonl").read_text(encoding="utf-8").splitlines():
            observations.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfReleaseError(f"cannot read trtmc-bench observations: {exc}") from exc
    samples = [float(item["runtime_e2e_wall_ms"]) for item in observations]
    return {
        "schema_version": result["schema_version"],
        "status": "completed",
        "backend": "trtmc-bench",
        "workload_digest": workload_digest,
        "measurement_policy": result.get("measurement_policy", {}),
        "samples_ms": samples,
        "metrics": cell["metrics"],
        "output_summary": cell.get("output_summary", {}),
        "environment": result.get("environment", {}),
    }


def _read_baseline(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfReleaseError(f"cannot read baseline result: {exc}") from exc
    if value.get("schema_version") != "trtmc.perf-baseline/v1":
        raise PerfReleaseError("baseline returned an unsupported result schema")
    return value


def _median(result: Mapping[str, Any]) -> float:
    values = result.get("samples_ms")
    if not isinstance(values, list) or not values:
        raise PerfReleaseError("backend result has no timing samples")
    return float(statistics.median(float(value) for value in values))


def _output_contract(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[bool, str]:
    left = candidate.get("output_summary", {})
    right = baseline.get("output_summary", {})
    operation = str(case["operation"])
    if operation == "generate":
        if case["baseline"].get("output_contract") == "exact-text":
            matched = left.get("text") == right.get("text")
            return matched, "generated text differs" if not matched else ""
        matched = left.get("token_ids") == right.get("token_ids")
        return matched, "generated token ids differ" if not matched else ""
    if operation == "encode":
        left_elements = left.get("element_count")
        right_elements = right.get("embedding_elements")
        matched = left_elements == right_elements and left.get("dim") == right.get("dim")
        return matched, "encoder output shape differs" if not matched else ""
    return True, ""


def _classify(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    mismatch = _baseline_contract_mismatch(case, candidate, baseline)
    if mismatch:
        return "contract-mismatch", {"reason": mismatch}
    outputs_match, reason = _output_contract(case, candidate, baseline)
    if not outputs_match:
        return "contract-mismatch", {"reason": reason}
    candidate_p50 = _median(candidate)
    baseline_p50 = _median(baseline)
    ratio = baseline_p50 / candidate_p50
    margin = float(case.get("equivalence_margin_percent", 5.0)) / 100.0
    status = _comparison_status(ratio, margin)
    return status, {
        "baseline_over_trtmc_p50": ratio,
        "equivalence_margin_percent": margin * 100.0,
    }


def _baseline_contract_mismatch(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    expected_mode = str(case["baseline"].get("mode", "torch-compile"))
    if baseline.get("mode") != expected_mode:
        return "baseline mode differs from the suite"
    expected_precision = case["baseline"].get("precision", candidate.get("precision"))
    if baseline.get("precision") != expected_precision:
        return "baseline precision differs from the suite"
    expected_padding = case["baseline"].get("padding", "longest")
    if baseline.get("padding") != expected_padding:
        return "baseline padding differs from the suite"
    expected_experts = case["baseline"].get("experts_implementation")
    if baseline.get("experts_implementation") != expected_experts:
        return "baseline experts implementation differs from the suite"
    if (
        "model_class" in case["baseline"]
        and baseline.get("model_class") != case["baseline"]["model_class"]
    ):
        return "baseline model class differs from the suite"
    if (
        "generation_method" in case["baseline"]
        and baseline.get("generation_method") != case["baseline"]["generation_method"]
    ):
        return "baseline generation method differs from the suite"
    digest = candidate.get("workload_digest")
    if not digest or baseline.get("workload_digest") != digest:
        return "candidate and baseline workload differ"
    if expected_mode != "torch-compile":
        return ""
    evidence = baseline.get("compile_evidence")
    if not isinstance(evidence, Mapping) or not evidence.get("applied"):
        return "torch.compile evidence is missing"
    if not evidence.get("timed_callable_uses_compiled_target"):
        return "compiled target was not used while timing"
    return ""


def _comparison_status(ratio: float, margin: float) -> str:
    if ratio > 1.0 + margin:
        return "green"
    if ratio < 1.0 - margin:
        return "red"
    return "yellow"


def _case_row(
    case: Mapping[str, Any], resolved: Mapping[str, Any], workload_digest: str
) -> dict[str, Any]:
    return {
        "id": case["id"],
        "family": case["family"],
        "operation": case["operation"],
        "model": case["model"],
        "priority": case["priority"],
        "status": "running",
        "resolved_settings": {
            "model": resolved["model"],
            "testcase": resolved["testcase"],
            "request": resolved["request"],
            "runtime": resolved["runtime"],
            "measurement": case["measurement"],
            "workload_digest": workload_digest,
        },
        "baseline_contract": dict(case["baseline"]),
        "commands": {},
        "started_at": _now(),
    }


def _run_supported_case(
    case: Mapping[str, Any],
    resolved: Mapping[str, Any],
    options: RunOptions,
    case_work: Path,
    row: MutableMapping[str, Any],
) -> None:
    environment = _command_environment()
    digest = _workload_digest(resolved)
    candidate_dir = case_work / "trtmc"
    baseline_path = case_work / "baseline.json"
    candidate_argv = [*_candidate_base_argv(case, options), "--output", str(candidate_dir)]
    baseline_argv, profile = _baseline_argv(case, resolved, baseline_path, options)
    row["resolved_settings"]["baseline_python_profile"] = profile
    commands = {
        "trtmc": {"argv": candidate_argv, "rendered": shlex.join(candidate_argv)},
        "baseline": {"argv": baseline_argv, "rendered": shlex.join(baseline_argv)},
    }
    row["commands"] = commands
    print(f"[{case['id']}] TRTMC: {commands['trtmc']['rendered']}", flush=True)
    print(f"[{case['id']}] baseline: {commands['baseline']['rendered']}", flush=True)
    if options.dry_run:
        row["status"] = "partial"
        row["reason"] = "dry-run: commands resolved but not executed"
        return
    order = ("trtmc", "baseline") if _stable_even(str(case["id"])) else ("baseline", "trtmc")
    for side in order:
        if options.only != "both" and options.only != side:
            continue
        argv = candidate_argv if side == "trtmc" else baseline_argv
        command = _run_command(argv, environment, options.timeout_seconds)
        command.pop("stdout", None)
        commands[side] = command
        if command["exit_code"] != 0:
            row["status"] = "failed"
            row["reason"] = f"{side} command failed with rc={command['exit_code']}"
            return
    if options.only != "both":
        row["status"] = "partial"
        row["reason"] = f"only {options.only} was requested"
        if options.only == "trtmc":
            row["candidate"] = _candidate_result(candidate_dir, digest)
        else:
            row["baseline"] = _read_baseline(baseline_path)
        return
    candidate = _candidate_result(candidate_dir, digest)
    candidate["precision"] = str(resolved["model"]["precision"])
    baseline = _read_baseline(baseline_path)
    status, comparison = _classify(case, candidate, baseline)
    row["candidate"] = candidate
    row["baseline"] = baseline
    row["comparison"] = comparison
    row["status"] = status


def _stable_even(value: str) -> bool:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:2], 16) % 2 == 0


def _run_one(case: Mapping[str, Any], options: RunOptions, work_root: Path) -> dict[str, Any]:
    print(f"[{case['id']}] resolving", flush=True)
    dry_argv = [*_candidate_base_argv(case, options), "--dry-run"]
    try:
        resolved, dry_argv, dry_command = _resolve_candidate(case, options)
    except (PerfReleaseError, OSError, ValueError, RuntimeError) as exc:
        return _resolution_failure_row(case, dry_argv, str(exc))
    digest = _workload_digest(resolved)
    row = _case_row(case, resolved, digest)
    row["commands"]["resolve"] = dry_command
    if case["baseline"]["runner"] == "unsupported":
        row["status"] = "unsupported"
        row["reason"] = str(case["baseline"]["reason"])
        row["commands"]["resolve"]["argv"] = dry_argv
        row["finished_at"] = _now()
        return row
    case_work = work_root / _slug(str(case["id"]))
    if case_work.exists():
        shutil.rmtree(case_work)
    case_work.mkdir(parents=True, exist_ok=True)
    try:
        _run_supported_case(case, resolved, options, case_work, row)
    except (PerfReleaseError, OSError, ValueError, RuntimeError) as exc:
        row["status"] = "failed"
        row["reason"] = str(exc)
    row["finished_at"] = _now()
    return row


def _resolution_failure_row(
    case: Mapping[str, Any], argv: Sequence[str], reason: str
) -> dict[str, Any]:
    return {
        "id": case["id"],
        "family": case["family"],
        "operation": case["operation"],
        "model": case["model"],
        "priority": case["priority"],
        "status": "failed",
        "reason": reason,
        "baseline_contract": dict(case["baseline"]),
        "commands": {
            "resolve": {"argv": list(argv), "rendered": shlex.join(argv)},
        },
        "started_at": _now(),
        "finished_at": _now(),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "case"


def _should_skip(row: Mapping[str, Any], options: RunOptions) -> bool:
    status = row.get("status")
    if not options.resume or status not in TERMINAL_STATUSES:
        return False
    return not options.rerun_failed or status not in {"failed", "contract-mismatch"}


def _final_status(rows: Iterable[Mapping[str, Any]]) -> str:
    statuses = {str(row.get("status")) for row in rows}
    return "completed-with-errors" if statuses & {"failed", "contract-mismatch"} else "completed"


def _light(status: str) -> str:
    return {
        "green": "🟢",
        "yellow": "🟡",
        "red": "🔴",
    }.get(status, "⚪")


def _baseline_label(row: Mapping[str, Any]) -> str:
    contract = row.get("baseline_contract", {})
    if contract.get("runner") == "unsupported":
        return "Unavailable"
    mode = contract.get("mode", "torch-compile")
    scope = contract.get("compile_scope", "model.forward") if mode == "torch-compile" else ""
    details = []
    if contract.get("precision"):
        details.append(str(contract["precision"]))
    if contract.get("experts_implementation"):
        details.append(f"experts={contract['experts_implementation']}")
    if contract.get("model_class"):
        details.append(f"loader={contract['model_class']}")
    if contract.get("generation_method"):
        details.append(f"generate={contract['generation_method']}")
    suffix = f", {', '.join(details)}" if details else ""
    return f"torch.compile ({scope}{suffix})" if mode == "torch-compile" else f"HF eager{suffix}"


def _report_note(row: Mapping[str, Any]) -> str:
    status = row.get("status")
    if status == "failed":
        return "Execution failed; inspect internal results.json."
    comparison = row.get("comparison", {})
    if isinstance(comparison, Mapping) and comparison.get("reason"):
        return str(comparison["reason"])
    return str(row.get("reason", ""))


def _report_html(results: Mapping[str, Any]) -> str:
    rows = list(results["cases"])
    counts = {
        status: sum(row.get("status") == status for row in rows)
        for status in ("green", "yellow", "red")
    }
    body = []
    for row in rows:
        status = str(row.get("status", "pending"))
        reason = html.escape(_report_note(row))
        if any(name in row.get("commands", {}) for name in ("trtmc", "baseline")):
            reproduce = f"python3 reproduce.py {shlex.quote(str(row['id']))}"
        else:
            reproduce = "—"
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['family']))}</td>"
            f"<td>{html.escape(str(row['operation']))}</td>"
            f"<td><code>{html.escape(str(row['model']))}</code></td>"
            f"<td>{html.escape(_baseline_label(row))}</td>"
            f"<td class='light'>{_light(status)}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{reason}</td>"
            f"<td><code>{html.escape(reproduce)}</code></td>"
            "</tr>"
        )
    generated = html.escape(str(results.get("finished_at", results.get("started_at", ""))))
    summary = f"🟢 {counts['green']} &nbsp; 🟡 {counts['yellow']} &nbsp; 🔴 {counts['red']} &nbsp; ⚪ {len(rows) - sum(counts.values())}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRTMC release performance matrix</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#1f2937}} h1{{margin-bottom:4px}}
.meta{{color:#4b5563;margin:4px 0 18px}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #9ca3af;padding:7px;text-align:left;vertical-align:top}}
th{{background:#e5e7eb}} tr:nth-child(even){{background:#f9fafb}} code{{font-size:12px}}
.light{{font-size:20px;text-align:center}}
</style></head><body>
<h1>TRTMC release performance matrix</h1>
<p class="meta">Generated {generated}. Categories compare TRTMC public-pipeline latency with the explicitly named baseline. Raw performance numbers are intentionally omitted.</p>
<p><strong>{summary}</strong></p>
<table><thead><tr><th>Family</th><th>Operation</th><th>Model</th><th>Baseline</th><th>Light</th><th>Status</th><th>Note</th><th>Reproduce</th></tr></thead>
<tbody>{"".join(body)}</tbody></table>
<p class="meta">Green: TRTMC is more than the configured margin faster. Yellow: within the margin. Red: TRTMC is more than the margin slower. White: unsupported, not run, partial, or invalid comparison.</p>
</body></html>"""


REPRODUCE_SOURCE = r'''#!/usr/bin/env python3
"""Replay commands recorded by one TRTMC performance release run."""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent

def replace_output(argv, output):
    values = list(argv)
    for flag in ("--output", "-o"):
        if flag in values:
            values[values.index(flag) + 1] = str(output)
            return values
    return values

def relocate_repo_paths(argv, original_repo, repo):
    values = []
    for value in argv:
        path = pathlib.Path(value)
        if not path.is_absolute():
            values.append(value)
            continue
        try:
            relative = path.relative_to(original_repo)
        except ValueError:
            values.append(value)
        else:
            values.append(str(repo / relative))
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("side", nargs="?", choices=("trtmc", "baseline", "both"), default="both")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("/tmp/trtmc-perf-reproduce"))
    parser.add_argument("--repo", type=pathlib.Path)
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args()
    results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in results["cases"]}
    if args.case not in rows:
        parser.error(f"unknown case {args.case!r}")
    row = rows[args.case]
    original_repo = pathlib.Path(results["repository_root"]).resolve()
    repo = (args.repo or original_repo).resolve()
    selected = ("trtmc", "baseline") if args.side == "both" else (args.side,)
    for side in selected:
        command = row.get("commands", {}).get(side)
        if not command or not command.get("argv"):
            parser.error(f"{args.case} has no recorded {side} command")
        target = args.output / args.case / ("trtmc" if side == "trtmc" else "baseline.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        argv = relocate_repo_paths(command["argv"], original_repo, repo)
        argv = replace_output(argv, target)
        print(subprocess.list2cmdline(argv), flush=True)
        if not args.print_only:
            completed = subprocess.run(argv, cwd=repo, env=os.environ.copy(), check=False)
            if completed.returncode:
                return completed.returncode
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_artifacts(output: Path, results: Mapping[str, Any]) -> None:
    _write_json(output / "results.json", results)
    (output / "report.html").write_text(_report_html(results), encoding="utf-8")
    reproduce = output / "reproduce.py"
    reproduce.write_text(REPRODUCE_SOURCE, encoding="utf-8")
    reproduce.chmod(0o755)


def _prepare_output(
    output: Path,
    suite_path: Path,
    suite: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    options: RunOptions,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if options.resume and result_path.is_file():
        return _load_resume(result_path, suite_path)
    existing = [path for path in output.iterdir() if path.name != ".work"]
    if existing:
        raise PerfReleaseError(
            f"output directory is not empty; use --resume or choose another path: {output}"
        )
    return _initial_results(suite_path, suite, cases)


def _ci_failed(results: Mapping[str, Any], selected_ids: set[str]) -> bool:
    for row in results["cases"]:
        if row["id"] not in selected_ids:
            continue
        if row.get("status") in {"failed", "contract-mismatch", "pending"}:
            return True
    return False


def run(arguments: argparse.Namespace) -> int:
    suite_path = arguments.suite.resolve()
    suite = _read_yaml(suite_path)
    cases = _cases(suite)
    _validate_coverage(cases)
    selected = _selected_cases(cases, arguments.case, arguments.priority, arguments.max_cases)
    if not selected:
        raise PerfReleaseError("selection contains no cases")
    options = RunOptions(
        output=arguments.output.resolve(),
        trtmc_bench=arguments.trtmc_bench,
        trtmc_worker=arguments.trtmc_worker,
        hf_transformers_runner=arguments.hf_transformers_runner,
        bundle_cache=arguments.bundle_cache,
        bundle_roots=tuple(arguments.bundle_root),
        runtime_dirs=tuple(arguments.runtime_dir),
        only=arguments.only,
        dry_run=arguments.dry_run,
        ci=arguments.ci,
        resume=arguments.resume,
        rerun_failed=arguments.rerun_failed,
        local_files_only=arguments.local_files_only,
        timeout_seconds=arguments.timeout_seconds,
    )
    results = _prepare_output(options.output, suite_path, suite, cases, options)
    rows = _result_rows(results)
    work_root = options.output / ".work"
    work_root.mkdir(exist_ok=True)
    _write_artifacts(options.output, results)
    for case in selected:
        existing = rows[str(case["id"])]
        if _should_skip(existing, options):
            print(f"[{case['id']}] resume: keeping {existing['status']}", flush=True)
            continue
        row = _run_one(case, options, work_root)
        rows[str(case["id"])].clear()
        rows[str(case["id"])].update(row)
        _write_artifacts(options.output, results)
    results["finished_at"] = _now()
    results["status"] = _final_status(results["cases"])
    _write_artifacts(options.output, results)
    shutil.rmtree(work_root, ignore_errors=True)
    print(f"Results: {options.output / 'results.json'}")
    print(f"Report: {options.output / 'report.html'}")
    selected_ids = {str(case["id"]) for case in selected}
    return 1 if options.ci and _ci_failed(results, selected_ids) else 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except PerfReleaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
