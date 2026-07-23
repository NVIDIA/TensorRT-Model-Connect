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


def _task_eval_command(
    binding: Binding,
    *,
    case_dir: Path,
    dataset: Path,
    arguments: argparse.Namespace,
    reference_python: str,
) -> list[str]:
    work_root = case_dir / "task-eval"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "task_eval.py"),
        "eval",
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
            if not line.startswith("$ "):
                continue
            value = line[2:].strip()
            if value and value not in commands[kind]:
                commands[kind].append(value)
    return commands


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


def _task_eval_result(
    binding: Binding,
    *,
    case_dir: Path,
    returncode: int,
    reference_environment: EnvironmentSelection,
    command: Sequence[str],
) -> dict[str, Any]:
    summary_path = case_dir / "task-eval" / binding.workload / "eval_summary.json"
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
    work_dir = case_dir / "task-eval" / binding.workload / binding.model
    return {
        "schema_version": "trtmc.validation-result/v1",
        "model": binding.model,
        "workload": binding.workload,
        "executor": "task_eval",
        "status": status,
        "reference_environment": [
            {"name": name, "python": path} for name, path in reference_environment.names_and_paths
        ],
        "reproduce": {
            **_commands_from_logs(work_dir),
            "validation": shlex.join(command),
        },
        "raw_result": raw_result,
        "raw_result_path": str(summary_path),
        "execution_log": str(case_dir / "execution.log"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _e2e_result(
    binding: Binding,
    *,
    case_dir: Path,
    returncode: int,
    reference_environment: EnvironmentSelection,
    command: Sequence[str],
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
    return {
        "schema_version": "trtmc.validation-result/v1",
        "model": binding.model,
        "workload": binding.workload,
        "executor": "e2e",
        "status": status,
        "reference_environment": [
            {"name": name, "python": path} for name, path in reference_environment.names_and_paths
        ],
        "reproduce": {
            **_commands_from_e2e_results(raw_results),
            "validation": shlex.join(command),
        },
        "raw_result_paths": [str(path) for path in result_paths],
        "raw_results": raw_results,
        "execution_log": str(case_dir / "execution.log"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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
            command=command,
        )
    else:
        suite = suites[binding.workload]
        dataset = (
            Path(arguments.dataset)
            if arguments.dataset
            else _dataset_path(suite, arguments.dataset_root)
        )
        command = _task_eval_command(
            binding,
            case_dir=case_dir,
            dataset=dataset,
            arguments=arguments,
            reference_python=environment.base_python,
        )
        returncode = _run_subprocess(command, case_dir / "execution.log", process_env)
        result = _task_eval_result(
            binding,
            case_dir=case_dir,
            returncode=returncode,
            reference_environment=environment,
            command=command,
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


def write_report(output: Path) -> tuple[Path, Path, dict[str, Any]]:
    result_paths = sorted(output.glob("*/*/comparison.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    counts = {
        name: sum(result.get("status") == name for result in results)
        for name in ("passed", "failed", "skipped")
    }
    report = {
        "schema_version": "trtmc.validation-report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if results and counts["failed"] == 0 else "failed",
        "summary": {"cases": len(results), **counts},
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
    rows = []
    for result, path in zip(results, result_paths, strict=True):
        relative = path.relative_to(output)
        environments = ", ".join(
            str(item.get("name", "")) for item in result.get("reference_environment", [])
        )
        status = str(result.get("status", "failed"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('model', '')))}</td>"
            f"<td>{html.escape(str(result.get('workload', '')))}</td>"
            f'<td class="{html.escape(status)}">{html.escape(status)}</td>'
            f"<td>{html.escape(environments)}</td>"
            f'<td><a href="{html.escape(str(relative))}">comparison.json</a></td>'
            "</tr>"
        )
    provenance = _report_provenance(report.get("run", {}))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TRTMC Validation Report</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 32px; color: #202124; }}
h1 {{ margin-bottom: 4px; }}
.summary {{ color: #5f6368; margin-bottom: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dadce0; padding: 8px 10px; text-align: left; }}
th {{ background: #f8f9fa; }}
.passed {{ color: #137333; font-weight: 600; }}
.failed {{ color: #b3261e; font-weight: 600; }}
.skipped {{ color: #8a4f00; font-weight: 600; }}
</style></head><body>
<h1>TRTMC Validation Report</h1>
<div class="summary">{report["summary"]["cases"]} cases ·
{counts["passed"]} passed · {counts["failed"]} failed · {counts["skipped"]} skipped<br>
{html.escape(provenance)}</div>
<table><thead><tr><th>Model</th><th>Workload</th><th>Status</th>
<th>Reference environment</th><th>Result</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
</body></html>
"""
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
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_OUTPUT / "engines")
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
        failed = failed or result.get("status") == "failed"
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
