#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run model-first TRTMC reference validation for Dev and QA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tensorrt_model_connect.python_profiles import (  # noqa: E402
    DEFAULT_PROFILE,
    normalize_execution_profiles,
    profile_env_var,
    resolve_profile_python,
)
from tests.e2e_harness.manifest_loader import load_all_model_manifests  # noqa: E402
from tools import task_eval  # noqa: E402


DEFAULT_CATALOG = REPO_ROOT / "tests" / "validation" / "model_workloads.yaml"
DEFAULT_SUITES = REPO_ROOT / "tests" / "task_eval" / "validation_suites.yaml"
DEFAULT_MODELS = REPO_ROOT / "tests" / "e2e" / "models"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "trtmc-validate"
DEFAULT_ENGINE_DIR = DEFAULT_OUTPUT / "engines"
DEFAULT_REFERENCE_CACHE = DEFAULT_OUTPUT / "references"
COMMON_REFERENCE_PROFILE = "reference_common"


class ValidationError(RuntimeError):
    """The requested validation cannot be resolved or executed."""


@dataclass(frozen=True)
class Binding:
    model: str
    workload: str


@dataclass(frozen=True)
class EnvironmentSelection:
    base_python: str
    names_and_paths: tuple[tuple[str, str], ...]
    overrides: Mapping[str, str]


def _validate_model_spec(path: Path, name: Any, spec: Any) -> None:
    if not isinstance(name, str) or not isinstance(spec, dict):
        raise ValidationError(f"{path}: invalid model binding {name!r}")
    workloads = spec.get("workloads")
    default = spec.get("default")
    valid_workloads = (
        isinstance(workloads, list)
        and bool(workloads)
        and all(isinstance(item, str) and item for item in workloads)
    )
    if not valid_workloads:
        raise ValidationError(f"{path}: {name}.workloads must contain names")
    if default not in workloads:
        raise ValidationError(f"{path}: {name}.default must be one of {name}.workloads")


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValidationError(f"{path}: expected version: 1")
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValidationError(f"{path}: models must be a non-empty mapping")
    for name, spec in models.items():
        _validate_model_spec(path, name, spec)
    return raw


def ready_model_names(models_root: Path = DEFAULT_MODELS) -> tuple[str, ...]:
    models = task_eval.load_manifest_records(models_root)
    return tuple(
        sorted(
            str(model["name"])
            for model in models
            if not model["requires_multi_device"] and not model.get("skip")
        )
    )


def audit_catalog(
    catalog: Mapping[str, Any],
    *,
    ready_models: Iterable[str],
    suite_names: Iterable[str],
) -> None:
    models = catalog["models"]
    ready = set(ready_models)
    configured = set(models)
    missing = sorted(ready - configured)
    stale = sorted(configured - ready)
    if missing or stale:
        details = []
        if missing:
            details.append(f"missing ready models: {', '.join(missing)}")
        if stale:
            details.append(f"non-ready or unknown models: {', '.join(stale)}")
        raise ValidationError("; ".join(details))

    known_workloads = set(suite_names) | {"e2e"}
    unknown = sorted(
        {
            workload
            for spec in models.values()
            for workload in spec["workloads"]
            if workload not in known_workloads
        }
    )
    if unknown:
        raise ValidationError(f"unknown workloads: {', '.join(unknown)}")


def audit_workload_compatibility(
    catalog: Mapping[str, Any],
    *,
    suites: Mapping[str, dict[str, Any]],
    task_models: Mapping[str, dict[str, Any]],
) -> None:
    incompatible = []
    for model_name, spec in catalog["models"].items():
        for workload in spec["workloads"]:
            if workload == "e2e":
                continue
            matched, reason = task_eval.suite_match_reason(
                suites[workload],
                task_models[model_name],
            )
            if not matched:
                incompatible.append(f"{model_name}/{workload}: {reason}")
    if incompatible:
        raise ValidationError("incompatible model/workload bindings: " + "; ".join(incompatible))


def resolve_binding(
    catalog: Mapping[str, Any],
    model: str,
    workload: str | None = None,
) -> Binding:
    models = catalog["models"]
    if model not in models:
        raise ValidationError(f"unknown or unsupported model: {model}")
    spec = models[model]
    selected = workload or spec["default"]
    if selected not in spec["workloads"]:
        available = ", ".join(spec["workloads"])
        raise ValidationError(
            f"model {model} does not declare workload {selected}; available: {available}"
        )
    return Binding(model=model, workload=selected)


def _task_eval_models(models_root: Path) -> dict[str, dict[str, Any]]:
    return {str(model["name"]): model for model in task_eval.load_manifest_records(models_root)}


def _e2e_models(models_root: Path) -> dict[str, Any]:
    return {model.name: model for model in load_all_model_manifests(models_root)}


def _declared_profile(
    *,
    family: str,
    runtime_strategy: str,
    reference_backend: str,
    execution_profiles: Mapping[str, str] | None,
) -> str:
    profiles = normalize_execution_profiles(
        execution_profiles,
        family=family,
        runtime_strategy=runtime_strategy,
        reference_backend=reference_backend,
    )
    profile = profiles["reference"]
    if profile == DEFAULT_PROFILE:
        return COMMON_REFERENCE_PROFILE
    return profile


def _binding_profiles(
    binding: Binding,
    *,
    task_models: Mapping[str, dict[str, Any]],
    e2e_models: Mapping[str, Any],
) -> tuple[str, ...]:
    if binding.workload != "e2e":
        model = task_models[binding.model]
        return (
            _declared_profile(
                family=str(model.get("family", "") or ""),
                runtime_strategy=str(model.get("runtime_strategy", "") or ""),
                reference_backend=str(model.get("reference_backend", "") or ""),
                execution_profiles=model.get("execution_profiles"),
            ),
        )

    profiles = []
    for case in e2e_models[binding.model].testcases:
        profile = _declared_profile(
            family=case.family,
            runtime_strategy=case.runtime_strategy,
            reference_backend=case.reference_backend,
            execution_profiles=case.execution_profiles,
        )
        if profile not in profiles:
            profiles.append(profile)
    return tuple(profiles)


def ensure_environments(
    profile_names: Iterable[str],
    base_python: str,
) -> EnvironmentSelection:
    names_and_paths = []
    overrides = {}
    selected_base = base_python

    def announce_create(name: str) -> None:
        print(f"Creating reference environment: {name}", flush=True)

    for name in profile_names:
        path = resolve_profile_python(
            name,
            base_python,
            on_create=announce_create,
        )
        print(f"Using reference environment: {path}", flush=True)
        names_and_paths.append((name, path))
        if name == COMMON_REFERENCE_PROFILE:
            selected_base = path
        elif name != DEFAULT_PROFILE:
            overrides[profile_env_var(name)] = path
    return EnvironmentSelection(
        base_python=selected_base,
        names_and_paths=tuple(names_and_paths),
        overrides=overrides,
    )


def _dataset_path(suite: Mapping[str, Any], dataset_root: Path | None) -> Path:
    raw = str(suite.get("dataset", {}).get("default_path", "") or "")
    if not raw:
        raise ValidationError(f"workload {suite.get('id')} has no default dataset path")
    path = Path(raw)
    if dataset_root is None:
        return path
    try:
        relative = path.relative_to("/mnt/data")
    except ValueError:
        relative = Path(path.name)
    return dataset_root / relative


def _run_subprocess(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        output.write(f"$ {shlex.join(command)}\n")
        output.flush()
        completed = subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=dict(env),
        )
    return completed.returncode


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    roots = [str(PYTHON_ROOT), str(REPO_ROOT)]
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def _comparison_command(
    binding: Binding,
    *,
    case_dir: Path,
    dataset: Path,
    arguments: argparse.Namespace,
    reference_python: str,
) -> list[str]:
    work_root = case_dir / "validation"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "trtmc_compare.py"),
        "--suite",
        binding.workload,
        "--dataset",
        str(dataset),
        "--model",
        binding.model,
        "--work-root",
        str(work_root),
        "--engine-dir",
        str(arguments.engine_dir),
        "--trtmc-binary",
        str(arguments.trtmc_binary),
        "--benchmark-binary",
        str(arguments.benchmark_binary),
        "--hf-python",
        reference_python,
        "--reference-cache-dir",
        str(arguments.reference_cache_dir),
        "--replace-bundle-on-build",
        "--single-device-only",
        "--include-waived",
        "--fail-fast",
    ]
    if arguments.limit:
        command.extend(["--limit", str(arguments.limit)])
    if arguments.force_hf:
        command.append("--force-hf")
    if arguments.force_build:
        command.append("--force-build")
    if arguments.no_build:
        command.append("--require-prebuilt-bundles")
    if arguments.local_files_only:
        command.append("--local-files-only")
    if arguments.backend_dir:
        command.extend(["--backend-dir", str(arguments.backend_dir)])
    if arguments.model_plugin_dir:
        command.extend(["--model-plugin-dir", str(arguments.model_plugin_dir)])
    if arguments.cuda_visible_devices:
        command.extend(["--cuda-visible-devices", arguments.cuda_visible_devices])
    return command


def _e2e_command(
    binding: Binding,
    *,
    case_dir: Path,
    arguments: argparse.Namespace,
    reference_python: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_e2e.py",
        "-q",
        "--e2e-model",
        binding.model,
        "--e2e-artifacts-dir",
        str(case_dir / "e2e"),
        "--engine-dir",
        str(arguments.engine_dir),
        "--trtmc-binary",
        str(arguments.trtmc_binary),
        "--hf-python",
        reference_python,
    ]
    if arguments.platform:
        command.extend(["--e2e-platform", arguments.platform])
    if arguments.force_build:
        command.append("--rebuild-engines")
    if arguments.model_plugin_dir:
        command.extend(["--model-plugin-dir", str(arguments.model_plugin_dir)])
    return command


def _commands_from_logs(root: Path) -> dict[str, list[str]]:
    commands = {"hf": [], "trtmc": []}
    for path in sorted(root.rglob("*.log")):
        kind = "hf" if "hf" in path.name.lower() else "trtmc"
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            value = _command_from_log_line(line)
            if value and value not in commands[kind]:
                commands[kind].append(value)
    return commands


def _command_from_log_line(line: str) -> str:
    if line.startswith("$ "):
        return line[2:].strip()
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    command = data.get("command")
    if isinstance(command, list) and command:
        return shlex.join(str(token) for token in command)
    if isinstance(command, str):
        return command.strip()
    return ""


def _append_unique(commands: dict[str, list[str]], kind: str, command: str) -> None:
    if command and command not in commands[kind]:
        commands[kind].append(command)


def _e2e_command_kind(command: Sequence[Any]) -> str:
    executable = Path(str(command[0])).name
    is_reference_python = "python" in executable and "-c" in command
    return "hf" if is_reference_python else "trtmc"


def _collect_e2e_result_commands(
    commands: dict[str, list[str]],
    result: Mapping[str, Any],
) -> None:
    for entry in result.get("commands", []):
        command = entry.get("command", []) if isinstance(entry, dict) else []
        if not isinstance(command, list) or not command:
            continue
        _append_unique(
            commands,
            _e2e_command_kind(command),
            shlex.join(str(token) for token in command),
        )
    repro = result.get("repro_commands", {})
    if isinstance(repro, dict):
        _append_unique(
            commands,
            "trtmc",
            str(repro.get("trt_inference", "") or ""),
        )


def _commands_from_e2e_results(results: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    commands = {"hf": [], "trtmc": []}
    for result in results:
        _collect_e2e_result_commands(commands, result)
    return commands


_PRIMARY_COMPARISON_METRICS = (
    "sample_agreement_rate",
    "prediction_agreement_rate",
    "vector_pass_rate",
    "top1_agreement",
    "backend_pixel_agreement",
    "mean_pairwise_ordering_agreement",
    "token_prefix_agreement",
    "token_agreement_rate",
    "exact_match_rate",
)
_PRIMARY_METRIC_BY_MODE = {
    "asr_transcript": "prediction_agreement_rate",
    "continuation": "token_prefix_agreement",
    "diffusion_text_parity": "token_agreement_rate",
    "encoder_embedding_parity": "vector_pass_rate",
    "image_classification_parity": "top1_agreement",
    "ocrbench_v2": "prediction_agreement_rate",
    "reranking_parity": "mean_pairwise_ordering_agreement",
    "semantic_segmentation_parity": "backend_pixel_agreement",
    "time_series_parity": "sample_agreement_rate",
}
_COMPARISON_METRICS = (
    *_PRIMARY_COMPARISON_METRICS,
    "token_id_prefix_agreement",
    "normalized_transcript_exact_agreement_rate",
    "correctness_agreement_rate",
    "divergence_rate",
    "divergent_count",
    "hf_accuracy",
    "trtfb_accuracy",
    "accuracy_delta_trtfb_minus_hf",
    "accuracy_drop_from_hf",
    "hf_top1_accuracy",
    "trtfb_top1_accuracy",
    "top1_accuracy_drop_from_hf",
    "hf_mean_iou",
    "trtfb_mean_iou",
    "backend_mean_iou",
    "mean_iou_drop_from_hf",
    "mean_vector_cosine",
    "min_vector_cosine",
    "mean_pair_cosine_abs_delta",
    "max_pair_cosine_abs_delta",
    "mean_relative_l2",
    "max_relative_l2",
    "max_absolute_error",
)
_EXECUTION_ERROR_FIELDS = ("error", "exception", "traceback", "failure_class")


def _raw_comparison(result: Mapping[str, Any]) -> dict[str, Any]:
    raw_result = result.get("raw_result")
    if isinstance(raw_result, dict) and raw_result:
        return dict(raw_result)
    raw_results = result.get("raw_results")
    if isinstance(raw_results, list) and raw_results:
        passed = all(
            isinstance(item, dict) and item.get("status") in {"pass", "passed"}
            for item in raw_results
        )
        return {"mode": "e2e", "status": "passed" if passed else "failed"}
    status = str(result.get("status", "") or "")
    return {"status": status} if status else {}


def _execution_details(
    result: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    has_error = any(raw_result.get(name) for name in _EXECUTION_ERROR_FIELDS)
    completed = bool(raw_result) and not has_error
    return {
        "status": "completed" if completed else "error",
        "exit_code": result.get("returncode"),
    }


def _comparison_metrics(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: raw_result[name]
        for name in _COMPARISON_METRICS
        if raw_result.get(name) is not None
    }


def _primary_metric(
    mode: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    preferred = _PRIMARY_METRIC_BY_MODE.get(mode)
    if preferred in metrics:
        return {"name": preferred, "value": metrics[preferred]}
    for name in _PRIMARY_COMPARISON_METRICS:
        if name in metrics:
            return {"name": name, "value": metrics[name]}
    return None


def _comparison_details(
    raw_result: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    raw_status = str(raw_result.get("status", "") or "")
    status_by_raw = {
        "pass": "agreement",
        "passed": "agreement",
        "fail": "disagreement",
        "failed": "disagreement",
        "skip": "not_run",
        "skipped": "not_run",
    }
    status = (
        status_by_raw.get(raw_status, "not_run")
        if execution.get("status") == "completed"
        else "not_run"
    )
    metrics = _comparison_metrics(raw_result)
    failures = raw_result.get("gate_failures", [])
    mode = str(raw_result.get("mode", "") or "")
    return {
        "status": status,
        "mode": mode,
        "primary_metric": _primary_metric(mode, metrics),
        "metrics": metrics,
        "failures": failures if isinstance(failures, list) else [],
    }


def _validation_details(
    execution: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, str]:
    if execution.get("status") != "completed":
        return {"status": "failed"}
    status_by_comparison = {
        "agreement": "passed",
        "disagreement": "failed",
        "not_run": "skipped",
    }
    return {"status": status_by_comparison[str(comparison["status"])]}


def _normalize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    raw_result = _raw_comparison(normalized)
    execution = normalized.get("execution")
    if not isinstance(execution, dict):
        execution = _execution_details(normalized, raw_result)
    comparison = normalized.get("comparison")
    if not isinstance(comparison, dict):
        comparison = _comparison_details(raw_result, execution)
    validation = normalized.get("validation")
    if not isinstance(validation, dict):
        validation = _validation_details(execution, comparison)
    reproduce = normalized.get("reproduce", {})
    if not isinstance(reproduce, dict):
        reproduce = {}
    normalized.update(
        {
            "schema_version": "trtmc.validation-result/v2",
            "execution": execution,
            "comparison": comparison,
            "validation": validation,
            "reproduce": {
                "hf": list(reproduce.get("hf", [])),
                "trtmc": list(reproduce.get("trtmc", [])),
            },
        }
    )
    normalized.pop("returncode", None)
    normalized.pop("status", None)
    return normalized


def _comparison_result(
    binding: Binding,
    *,
    case_dir: Path,
    returncode: int,
    reference_environment: EnvironmentSelection,
) -> dict[str, Any]:
    summary_path = case_dir / "validation" / binding.workload / "eval_summary.json"
    raw_result: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for candidate in summary.get("results", []):
            if candidate.get("model") == binding.model:
                raw_result = candidate
                break
        if not raw_result and summary.get("results"):
            raw_result = summary["results"][0]
    status = str(raw_result.get("status", "") or "")
    if status not in {"passed", "failed", "skipped"}:
        status = "passed" if returncode == 0 else "failed"
    work_dir = case_dir / "validation" / binding.workload / binding.model
    return _normalize_result({
        "schema_version": "trtmc.validation-result/v2",
        "model": binding.model,
        "workload": binding.workload,
        "executor": "trtmc_compare",
        "status": status,
        "returncode": returncode,
        "reference_environment": [
            {"name": name, "python": path} for name, path in reference_environment.names_and_paths
        ],
        "reproduce": _commands_from_logs(work_dir),
        "raw_result": raw_result,
        "raw_result_path": str(summary_path),
        "execution_log": str(case_dir / "execution.log"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _e2e_result(
    binding: Binding,
    *,
    case_dir: Path,
    returncode: int,
    reference_environment: EnvironmentSelection,
) -> dict[str, Any]:
    result_paths = sorted((case_dir / "e2e").glob("*/result.json"))
    raw_results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    status = (
        "passed"
        if returncode == 0
        and raw_results
        and all(result.get("status") == "pass" for result in raw_results)
        else "failed"
    )
    return _normalize_result({
        "schema_version": "trtmc.validation-result/v2",
        "model": binding.model,
        "workload": binding.workload,
        "executor": "e2e",
        "status": status,
        "returncode": returncode,
        "reference_environment": [
            {"name": name, "python": path} for name, path in reference_environment.names_and_paths
        ],
        "reproduce": _commands_from_e2e_results(raw_results),
        "raw_result_paths": [str(path) for path in result_paths],
        "raw_results": raw_results,
        "execution_log": str(case_dir / "execution.log"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def run_binding(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    task_models: Mapping[str, dict[str, Any]],
    e2e_models: Mapping[str, Any],
    suites: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    case_dir = Path(arguments.output) / binding.model / binding.workload
    case_dir.mkdir(parents=True, exist_ok=True)
    profiles = _binding_profiles(
        binding,
        task_models=task_models,
        e2e_models=e2e_models,
    )
    environment = ensure_environments(profiles, str(arguments.hf_python))
    process_env = _source_environment()
    process_env.update(environment.overrides)

    if binding.workload == "e2e":
        command = _e2e_command(
            binding,
            case_dir=case_dir,
            arguments=arguments,
            reference_python=environment.base_python,
        )
        returncode = _run_subprocess(command, case_dir / "execution.log", process_env)
        result = _e2e_result(
            binding,
            case_dir=case_dir,
            returncode=returncode,
            reference_environment=environment,
        )
    else:
        suite = suites[binding.workload]
        dataset = (
            Path(arguments.dataset)
            if arguments.dataset
            else _dataset_path(suite, arguments.dataset_root)
        )
        command = _comparison_command(
            binding,
            case_dir=case_dir,
            dataset=dataset,
            arguments=arguments,
            reference_python=environment.base_python,
        )
        returncode = _run_subprocess(command, case_dir / "execution.log", process_env)
        result = _comparison_result(
            binding,
            case_dir=case_dir,
            returncode=returncode,
            reference_environment=environment,
        )

    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _source_revision() -> str:
    configured = os.environ.get("TRTMC_VALIDATION_SOURCE_REVISION", "").strip()
    if configured:
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def write_run_metadata(output: Path) -> Path:
    metadata = {
        "schema_version": "trtmc.validation-run/v1",
        "source_revision": _source_revision(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "command": shlex.join(sys.argv),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output / "run.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _report_provenance(run: Mapping[str, Any]) -> str:
    fields = (
        ("source", run.get("source_revision")),
        ("host", run.get("hostname")),
        ("CUDA_VISIBLE_DEVICES", run.get("cuda_visible_devices")),
    )
    return " · ".join(f"{name}={value}" for name, value in fields if value)


def _merge_commands_from_result_logs(result: dict[str, Any]) -> None:
    raw_result = result.get("raw_result", {})
    work_dir = raw_result.get("work_dir") if isinstance(raw_result, dict) else None
    if not work_dir:
        return
    root = Path(str(work_dir))
    if not root.is_dir():
        return
    discovered = _commands_from_logs(root)
    reproduce = result.setdefault("reproduce", {})
    if not isinstance(reproduce, dict):
        return
    for kind in ("hf", "trtmc"):
        commands = reproduce.setdefault(kind, [])
        if not isinstance(commands, list):
            continue
        for command in discovered[kind]:
            if command not in commands:
                commands.append(command)


def _result_commands(result: Mapping[str, Any], kind: str) -> list[str]:
    reproduce = result.get("reproduce", {})
    if not isinstance(reproduce, dict):
        return []
    commands = reproduce.get(kind, [])
    if not isinstance(commands, list):
        return []
    return [str(command) for command in commands if str(command).strip()]


def _render_command_group(label: str, commands: Sequence[str]) -> str:
    if not commands:
        body = '<span class="unavailable">Not reached; see comparison.json.</span>'
    else:
        shell = "\n".join(f"$ {command}" for command in commands)
        body = f"<pre><code>{html.escape(shell)}</code></pre>"
    return f"<h4>{html.escape(label)}</h4>{body}"


def _render_reproduction(result: Mapping[str, Any]) -> str:
    reference_commands = _result_commands(result, "hf")
    trtmc_commands = _result_commands(result, "trtmc")
    summary = (
        f"Reference {len(reference_commands)} · "
        f"TRTMC {len(trtmc_commands)}"
    )
    return (
        f"<details><summary>{summary}</summary>"
        '<div class="commands">'
        f"{_render_command_group('Reference', reference_commands)}"
        f"{_render_command_group('TRTMC', trtmc_commands)}"
        "</div></details>"
    )


def _reference_result_status(result: Mapping[str, Any]) -> str:
    raw_result = result.get("raw_result", {})
    if not isinstance(raw_result, dict):
        return ""
    return str(
        raw_result.get("hf_cache_status")
        or raw_result.get("hf_reference_status")
        or ""
    )


def _signal(status: str, labels: Mapping[str, str]) -> str:
    label = labels.get(status, status.replace("_", " ").title())
    safe_status = status if status.replace("_", "").isalnum() else "unknown"
    return (
        f'<span class="signal signal-{safe_status}">'
        '<span class="signal-light"></span>'
        f"{html.escape(label)}</span>"
    )


def _render_execution(result: Mapping[str, Any]) -> str:
    execution = result.get("execution", {})
    status = str(execution.get("status", "error")) if isinstance(execution, dict) else "error"
    return _signal(status, {"completed": "Completed", "error": "Error"})


def _render_reference(result: Mapping[str, Any]) -> str:
    status = _reference_result_status(result)
    display_status = {
        "reused": "cached",
        "adopted": "cached",
        "generated": "completed",
        "ran": "completed",
    }.get(status, "not_run")
    label = {
        "reused": "Reused",
        "adopted": "Adopted",
        "generated": "Generated",
        "ran": "Generated",
    }.get(status, "Not recorded")
    environments = ", ".join(
        str(item.get("name", ""))
        for item in result.get("reference_environment", [])
        if isinstance(item, dict)
    )
    detail = f'<div class="detail">{html.escape(environments)}</div>' if environments else ""
    return _signal(display_status, {display_status: label}) + detail


def _render_comparison(result: Mapping[str, Any]) -> str:
    comparison = result.get("comparison", {})
    if not isinstance(comparison, dict):
        return _signal("not_run", {"not_run": "Not compared"})
    status = str(comparison.get("status", "not_run"))
    signal = _signal(
        status,
        {
            "agreement": "Agreement",
            "disagreement": "Disagreement",
            "not_run": "Not compared",
        },
    )
    mode = str(comparison.get("mode", "") or "")
    detail = f'<div class="detail">{html.escape(mode)}</div>' if mode else ""
    return signal + detail


def _format_metric_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    is_ratio = any(
        token in name
        for token in ("accuracy", "agreement", "pass_rate", "exact_match", "divergence_rate")
    )
    if is_ratio:
        return f"{value * 100:.2f}%"
    if value and abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _render_metrics(result: Mapping[str, Any]) -> str:
    comparison = result.get("comparison", {})
    metrics = comparison.get("metrics", {}) if isinstance(comparison, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        return '<span class="unavailable">No metrics</span>'
    primary = comparison.get("primary_metric")
    primary_name = primary.get("name") if isinstance(primary, dict) else None
    ordered = ([primary_name] if primary_name in metrics else []) + [
        name for name in metrics if name != primary_name
    ]
    visible = ordered[:5]
    rows = [
        (
            f'<div class="metric{" primary" if name == primary_name else ""}">'
            f"<span>{html.escape(str(name))}</span>"
            f"<strong>{html.escape(_format_metric_value(str(name), metrics[name]))}</strong>"
            "</div>"
        )
        for name in visible
    ]
    remaining = len(ordered) - len(visible)
    if remaining:
        rows.append(f'<div class="detail">+{remaining} more in comparison.json</div>')
    return "".join(rows)


def _render_validation(result: Mapping[str, Any]) -> str:
    validation = result.get("validation", {})
    status = (
        str(validation.get("status", "failed"))
        if isinstance(validation, dict)
        else "failed"
    )
    return _signal(
        status,
        {"passed": "Pass", "failed": "Fail", "skipped": "Skipped"},
    )


def _normalize_result_files(
    result_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    results = []
    for path in result_paths:
        result = _normalize_result(json.loads(path.read_text(encoding="utf-8")))
        _merge_commands_from_result_logs(result)
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        results.append(result)
    return results


def _report_counts(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int], int]:
    validation_counts = {
        name: sum(result["validation"]["status"] == name for result in results)
        for name in ("passed", "failed", "skipped")
    }
    comparison_counts = {
        name: sum(result["comparison"]["status"] == name for result in results)
        for name in ("agreement", "disagreement", "not_run")
    }
    execution_errors = sum(
        result["execution"]["status"] == "error" for result in results
    )
    return validation_counts, comparison_counts, execution_errors


def _report_rows(
    output: Path,
    results: Sequence[Mapping[str, Any]],
    result_paths: Sequence[Path],
) -> str:
    rows = []
    for result, path in zip(results, result_paths, strict=True):
        relative = path.relative_to(output)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('model', '')))}</td>"
            f"<td>{html.escape(str(result.get('workload', '')))}</td>"
            f"<td>{_render_execution(result)}</td>"
            f"<td>{_render_reference(result)}</td>"
            f"<td>{_render_comparison(result)}</td>"
            f"<td>{_render_metrics(result)}</td>"
            f"<td>{_render_validation(result)}</td>"
            f"<td>{_render_reproduction(result)}</td>"
            f'<td><a href="{html.escape(str(relative))}">comparison.json</a></td>'
            "</tr>"
        )
    return "".join(rows)


def _report_document(
    report: Mapping[str, Any],
    *,
    rows: str,
    comparison_counts: Mapping[str, int],
    execution_errors: int,
) -> str:
    provenance = _report_provenance(report.get("run", {}))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TRTMC Validation Report</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 32px; color: #202124; }}
h1 {{ margin-bottom: 4px; }}
.summary {{ color: #5f6368; margin-bottom: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dadce0; padding: 8px 10px; text-align: left; }}
th {{ background: #f8f9fa; }}
details {{ min-width: 210px; }}
summary {{ cursor: pointer; color: #185abc; }}
.commands {{ min-width: min(760px, 70vw); padding: 4px 0; }}
.commands h4 {{ margin: 12px 0 4px; }}
pre {{ margin: 0; padding: 10px; white-space: pre-wrap; overflow-wrap: anywhere;
       background: #f8f9fa; border: 1px solid #dadce0; border-radius: 4px; }}
.unavailable {{ color: #5f6368; }}
.signal {{ display: inline-flex; align-items: center; gap: 7px; font-weight: 650;
           white-space: nowrap; }}
.signal-light {{ width: 10px; height: 10px; border-radius: 50%;
                 background: #80868b; box-shadow: 0 0 0 3px #eef0f1; }}
.signal-completed, .signal-agreement, .signal-passed {{ color: #137333; }}
.signal-completed .signal-light, .signal-agreement .signal-light,
.signal-passed .signal-light {{ background: #1e8e3e; box-shadow: 0 0 0 3px #e6f4ea; }}
.signal-error, .signal-disagreement, .signal-failed {{ color: #b3261e; }}
.signal-error .signal-light, .signal-disagreement .signal-light,
.signal-failed .signal-light {{ background: #d93025; box-shadow: 0 0 0 3px #fce8e6; }}
.signal-skipped {{ color: #8a4f00; }}
.signal-skipped .signal-light {{ background: #f9ab00; box-shadow: 0 0 0 3px #fef7e0; }}
.signal-cached {{ color: #185abc; }}
.signal-cached .signal-light {{ background: #1a73e8; box-shadow: 0 0 0 3px #e8f0fe; }}
.signal-not_run {{ color: #5f6368; }}
.detail {{ color: #5f6368; font-size: 12px; margin-top: 4px; }}
.metric {{ display: flex; justify-content: space-between; gap: 14px;
           font-variant-numeric: tabular-nums; font-size: 12px; }}
.metric span {{ color: #5f6368; }}
.metric.primary {{ font-size: 13px; }}
.metric.primary span, .metric.primary strong {{ color: #202124; }}
</style></head><body>
<h1>TRTMC Validation Report</h1>
<div class="summary">{report["summary"]["cases"]} cases ·
{comparison_counts["agreement"]} agreements ·
{comparison_counts["disagreement"]} disagreements ·
{execution_errors} execution errors<br>
{html.escape(provenance)}</div>
<table><thead><tr><th>Model</th><th>Workload</th><th>Execution</th>
<th>Reference</th><th>Comparison</th><th>Agreement metrics</th>
<th>Validation</th><th>Vanilla reproduction</th><th>Result</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>
"""


def write_report(output: Path) -> tuple[Path, Path, dict[str, Any]]:
    result_paths = sorted(output.glob("*/*/comparison.json"))
    results = _normalize_result_files(result_paths)
    validation_counts, comparison_counts, execution_errors = _report_counts(results)
    report = {
        "schema_version": "trtmc.validation-report/v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_status": (
            "passed" if results and validation_counts["failed"] == 0 else "failed"
        ),
        "summary": {
            "cases": len(results),
            "execution_completed": len(results) - execution_errors,
            "execution_errors": execution_errors,
            "agreements": comparison_counts["agreement"],
            "disagreements": comparison_counts["disagreement"],
            "not_compared": comparison_counts["not_run"],
            "validation_passed": validation_counts["passed"],
            "validation_failed": validation_counts["failed"],
            "validation_skipped": validation_counts["skipped"],
        },
        "results": results,
    }
    run_path = output / "run.json"
    if run_path.is_file():
        report["run"] = json.loads(run_path.read_text(encoding="utf-8"))
    json_path = output / "report.json"
    html_path = output / "report.html"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    document = _report_document(
        report,
        rows=_report_rows(output, results, result_paths),
        comparison_counts=comparison_counts,
        execution_errors=execution_errors,
    )
    html_path.write_text(document, encoding="utf-8")
    return json_path, html_path, report


def _print_result(result: Mapping[str, Any], comparison: Path, report: Path) -> None:
    reproduce = result.get("reproduce", {})
    hf_commands = reproduce.get("hf", []) if isinstance(reproduce, dict) else []
    trtmc_commands = reproduce.get("trtmc", []) if isinstance(reproduce, dict) else []
    print()
    print("Reproduce HF:")
    if hf_commands:
        for command in hf_commands:
            print(f"  {command}")
    else:
        print("  unavailable; see comparison result")
    print()
    print("Reproduce TRTMC:")
    if trtmc_commands:
        for command in trtmc_commands:
            print(f"  {command}")
    else:
        print("  unavailable; see comparison result")
    print()
    print(f"Compare result: {comparison}")
    print(f"Report:         {report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TRTMC against model reference implementations."
    )
    parser.add_argument("model", nargs="?")
    parser.add_argument("workload", nargs="?")
    parser.add_argument("--all", action="store_true", help="run every ready model")
    parser.add_argument("--list", action="store_true", help="list model-first workloads")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--suites", type=Path, default=DEFAULT_SUITES)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        default=DEFAULT_REFERENCE_CACHE,
    )
    parser.add_argument("--trtmc-binary", type=Path, default=REPO_ROOT / "build" / "trtmc")
    parser.add_argument(
        "--benchmark-binary",
        type=Path,
        default=REPO_ROOT / "build" / "trtmc_dataset_benchmark",
    )
    parser.add_argument("--hf-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--backend-dir", type=Path)
    parser.add_argument("--model-plugin-dir", type=Path)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--platform", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-hf", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_validation_inputs(
    arguments: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    tuple[str, ...],
    dict[str, dict[str, Any]],
]:
    catalog = load_catalog(arguments.catalog)
    suites_list = task_eval.load_suites(arguments.suites)
    suites = {suite["id"]: suite for suite in suites_list}
    ready = ready_model_names(arguments.models_dir)
    task_models = _task_eval_models(arguments.models_dir)
    audit_catalog(catalog, ready_models=ready, suite_names=suites)
    audit_workload_compatibility(
        catalog,
        suites=suites,
        task_models=task_models,
    )
    return catalog, suites, ready, task_models


def _select_bindings(
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    ready_models: Iterable[str],
) -> list[Binding]:
    if arguments.all:
        if arguments.model or arguments.workload or arguments.dataset:
            raise ValidationError("--all cannot be combined with MODEL, WORKLOAD, or --dataset")
        return [resolve_binding(catalog, model) for model in ready_models]
    if not arguments.model:
        raise ValidationError("provide MODEL [WORKLOAD], --all, or --list")
    return [resolve_binding(catalog, arguments.model, arguments.workload)]


def _print_bindings(bindings: Iterable[Binding]) -> None:
    print(
        json.dumps(
            [{"model": binding.model, "workload": binding.workload} for binding in bindings],
            indent=2,
        )
    )


def _run_bindings(
    bindings: Iterable[Binding],
    *,
    arguments: argparse.Namespace,
    task_models: Mapping[str, dict[str, Any]],
    suites: Mapping[str, dict[str, Any]],
) -> int:
    e2e_models = _e2e_models(arguments.models_dir)
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.engine_dir.mkdir(parents=True, exist_ok=True)
    arguments.reference_cache_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(arguments.output)
    failed = False
    for binding in bindings:
        print(f"\n{binding.model} / {binding.workload}", flush=True)
        result = run_binding(
            binding,
            arguments=arguments,
            task_models=task_models,
            e2e_models=e2e_models,
            suites=suites,
        )
        _, report_path, _ = write_report(arguments.output)
        comparison = arguments.output / binding.model / binding.workload / "comparison.json"
        _print_result(result, comparison, report_path)
        failed = failed or result["validation"]["status"] == "failed"
    return 1 if failed else 0


def _main(arguments: argparse.Namespace) -> int:
    catalog, suites, ready, task_models = _load_validation_inputs(arguments)
    if arguments.list:
        for name, spec in catalog["models"].items():
            print(f"{name}: {', '.join(spec['workloads'])}")
        return 0
    bindings = _select_bindings(arguments, catalog, ready)
    if arguments.dry_run:
        _print_bindings(bindings)
        return 0
    return _run_bindings(
        bindings,
        arguments=arguments,
        task_models=task_models,
        suites=suites,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        return _main(parser.parse_args(argv))
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
