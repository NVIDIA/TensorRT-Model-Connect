# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover and run family-owned qualification suites."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import yaml

from .qualification_environment import prepare_family_environment, process_environment


CONFIG_SCHEMA = "trtmc.qualification/v1"
ENVIRONMENT_SCHEMA = "trtmc.qualification-environment/v1"
RUN_CONFIGURATION_SCHEMA = "trtmc.qualification-run/v1"
PLAN_SCHEMA = "trtmc.qualification-plan/v1"
REQUEST_SCHEMA = "trtmc.qualification-executor-request/v1"
RESULT_SCHEMA = "trtmc.qualification-result/v1"
REPORT_SCHEMA = "trtmc.qualification-report/v1"
SUPPORTED_KINDS = ("accuracy", "performance")
GATE_POLICIES = ("blocking", "observation_only")
_FORBIDDEN_MODEL_POLICY_FIELDS = {
    "device",
    "devices",
    "device_include",
    "device_exclude",
    "include_devices",
    "exclude_devices",
    "platforms",
    "resource_class",
}


class QualificationError(RuntimeError):
    """A qualification plan cannot be resolved or executed safely."""


@dataclass(frozen=True)
class PlanItem:
    item_id: str
    family: str
    model: str
    kind: str
    suite_id: str
    case_id: str
    gate_policy: str
    manifest_path: Path
    config_path: Path
    definition_path: Path | None
    executor_path: Path
    definition: Mapping[str, Any]
    case: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "family": self.family,
            "model": self.model,
            "kind": self.kind,
            "suite_id": self.suite_id,
            "case_id": self.case_id,
            "gate_policy": self.gate_policy,
            "manifest_path": str(self.manifest_path),
            "config_path": str(self.config_path),
            "definition_path": str(self.definition_path) if self.definition_path else None,
            "executor_path": str(self.executor_path),
            "definition": dict(self.definition),
            "case": dict(self.case),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "PlanItem":
        required_strings = (
            "id",
            "family",
            "model",
            "kind",
            "suite_id",
            "case_id",
            "gate_policy",
            "manifest_path",
            "config_path",
            "executor_path",
        )
        for field in required_strings:
            _nonempty_string(value.get(field), f"plan item {field}")
        definition = value.get("definition")
        case = value.get("case")
        if not isinstance(definition, Mapping) or not isinstance(case, Mapping):
            raise QualificationError("plan item definition and case must be objects")
        definition_path = value.get("definition_path")
        if definition_path is not None and not isinstance(definition_path, str):
            raise QualificationError("plan item definition_path must be a path or null")
        if value["kind"] not in SUPPORTED_KINDS:
            raise QualificationError(f"unsupported plan item kind {value['kind']!r}")
        if value["gate_policy"] not in GATE_POLICIES:
            raise QualificationError(f"unsupported plan item gate_policy {value['gate_policy']!r}")
        return cls(
            item_id=str(value["id"]),
            family=str(value["family"]),
            model=str(value["model"]),
            kind=str(value["kind"]),
            suite_id=str(value["suite_id"]),
            case_id=str(value["case_id"]),
            gate_policy=str(value["gate_policy"]),
            manifest_path=Path(str(value["manifest_path"])),
            config_path=Path(str(value["config_path"])),
            definition_path=Path(definition_path) if definition_path else None,
            executor_path=Path(str(value["executor_path"])),
            definition=dict(definition),
            case=dict(case),
        )


@dataclass(frozen=True)
class QualificationPlan:
    plan_id: str
    kind: str
    families_root: Path
    items: tuple[PlanItem, ...]
    created_at: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "kind": self.kind,
            "families_root": str(self.families_root),
            "created_at": self.created_at,
            "items": [item.to_json() for item in self.items],
        }


def default_families_root() -> Path:
    packaged = Path(__file__).resolve().parent / "_catalog"
    if packaged.is_dir():
        return packaged
    repository = Path(__file__).resolve().parents[3]
    source = repository / "families"
    if source.is_dir():
        return source
    raise QualificationError("qualification catalog is unavailable; use --families-root")


def default_suites_root() -> Path:
    packaged = Path(__file__).resolve().parent / "_suites"
    if packaged.is_dir():
        return packaged
    repository = Path(__file__).resolve().parents[3]
    source = repository / "apps" / "benchmark" / "qualification" / "suites"
    if source.is_dir():
        return source
    raise QualificationError("shared benchmark definitions are unavailable")


class QualificationCatalog:
    """Read only explicit family-local qualification files."""

    def __init__(
        self,
        families_root: Path | None = None,
        suites_root: Path | None = None,
    ) -> None:
        self.root = (families_root or default_families_root()).expanduser().resolve()
        if not self.root.is_dir():
            raise QualificationError(f"families root does not exist: {self.root}")
        self.suites_root = (suites_root or default_suites_root()).expanduser().resolve()
        if not self.suites_root.is_dir():
            raise QualificationError(
                f"shared benchmark directory does not exist: {self.suites_root}"
            )

    def plan(
        self,
        kind: str,
        *,
        models: Sequence[str] = (),
        suites: Sequence[str] = (),
        cases: Sequence[str] = (),
    ) -> QualificationPlan:
        if kind not in SUPPORTED_KINDS:
            raise QualificationError(f"unsupported qualification kind {kind!r}")
        requested_models = _requested(models, "models")
        requested_suites = _requested(suites, "suites")
        requested_cases = _requested(cases, "cases")
        suffix = f".{kind}.yaml"
        config_paths = tuple(sorted(self.root.glob(f"*/tests/qualification/*{suffix}")))
        if requested_models:
            config_paths = tuple(
                path
                for path in config_paths
                if _model_from_filename(path, kind) in requested_models
            )

        records: list[dict[str, Any]] = []
        discovered_models: set[str] = set()
        discovered_suites: set[str] = set()
        discovered_cases: set[str] = set()
        for path in config_paths:
            parsed = self._load_config(path, kind)
            discovered_models.add(parsed["model"])
            for suite in parsed["suites"]:
                suite_id = str(suite["id"])
                if requested_suites and suite_id not in requested_suites:
                    continue
                discovered_suites.add(suite_id)
                for case in suite["cases"]:
                    case_id = str(case["id"])
                    if requested_cases and case_id not in requested_cases:
                        continue
                    discovered_cases.add(case_id)
                    records.append(
                        {
                            "family": parsed["family"],
                            "model": parsed["model"],
                            "kind": kind,
                            "suite_id": suite_id,
                            "case_id": case_id,
                            "gate_policy": suite["gate_policy"],
                            "manifest_path": parsed["manifest_path"],
                            "config_path": path,
                            "definition_path": suite["definition_path"],
                            "executor_path": parsed["executor_path"],
                            "definition": suite["definition"],
                            "case": case,
                        }
                    )

        _require_requested(requested_models, discovered_models, "models")
        _require_requested(requested_suites, discovered_suites, "suites")
        _require_requested(requested_cases, discovered_cases, "cases")

        items = tuple(self._plan_item(record) for record in records)
        plan_payload = {"kind": kind, "items": [item.to_json() for item in items]}
        return QualificationPlan(
            plan_id=_digest(plan_payload),
            kind=kind,
            families_root=self.root,
            items=items,
            created_at=_now(),
        )

    def _load_config(self, path: Path, kind: str) -> dict[str, Any]:
        raw = _read_yaml(path, "qualification config")
        if raw.get("schema_version") != CONFIG_SCHEMA:
            raise QualificationError(f"{path}: schema_version must be {CONFIG_SCHEMA}")
        model = _nonempty_string(raw.get("model"), f"{path}: model")
        expected_model = _model_from_filename(path, kind)
        if model != expected_model:
            raise QualificationError(
                f"{path}: model {model!r} must match filename {expected_model!r}"
            )
        if raw.get("kind") != kind:
            raise QualificationError(f"{path}: kind must be {kind!r}")
        forbidden = sorted(_find_forbidden_fields(raw))
        if forbidden:
            raise QualificationError(
                f"{path}: device policy belongs to the campaign, not the model config: "
                + ", ".join(forbidden)
            )

        family = path.parents[2].name
        qualification_root = path.parent.resolve()
        executor_path = qualification_root / "executor.py"
        _require_local_file(executor_path, qualification_root, "family executor")
        manifest_path = self._manifest_path(family, model)

        raw_suites = raw.get("suites")
        if not isinstance(raw_suites, list) or not raw_suites:
            raise QualificationError(f"{path}: suites must be a non-empty list")
        suite_ids: set[str] = set()
        parsed_suites = []
        for raw_suite in raw_suites:
            if not isinstance(raw_suite, Mapping):
                raise QualificationError(f"{path}: every suite must be an object")
            benchmark = raw_suite.get("benchmark")
            inline_definition = raw_suite.get("definition")
            if benchmark is not None:
                if raw_suite.get("id") is not None or inline_definition is not None:
                    raise QualificationError(
                        f"{path}: a shared benchmark uses benchmark without id or definition"
                    )
                suite_id = _nonempty_string(benchmark, f"{path}: benchmark")
            else:
                suite_id = _nonempty_string(raw_suite.get("id"), f"{path}: suite id")
            if suite_id in suite_ids:
                raise QualificationError(f"{path}: duplicate suite id {suite_id!r}")
            suite_ids.add(suite_id)
            gate_policy = raw_suite.get("gate_policy")
            if gate_policy not in GATE_POLICIES:
                raise QualificationError(
                    f"{path}: suite {suite_id!r} gate_policy must be blocking or observation_only"
                )
            definition, definition_path = _suite_definition(
                raw_suite,
                self.suites_root,
                path,
                suite_id,
                shared=benchmark is not None,
            )
            raw_cases = raw_suite.get("cases")
            if not isinstance(raw_cases, list) or not raw_cases:
                raise QualificationError(
                    f"{path}: suite {suite_id!r} cases must be a non-empty list"
                )
            case_ids: set[str] = set()
            parsed_cases = []
            for raw_case in raw_cases:
                if not isinstance(raw_case, Mapping):
                    raise QualificationError(f"{path}: suite {suite_id!r} cases must be objects")
                case_id = _nonempty_string(
                    raw_case.get("id"), f"{path}: suite {suite_id!r} case id"
                )
                if case_id in case_ids:
                    raise QualificationError(
                        f"{path}: suite {suite_id!r} has duplicate case id {case_id!r}"
                    )
                case_ids.add(case_id)
                parsed_cases.append(dict(raw_case))
            parsed_suites.append(
                {
                    "id": suite_id,
                    "gate_policy": str(gate_policy),
                    "definition": definition,
                    "definition_path": definition_path,
                    "cases": parsed_cases,
                }
            )
        return {
            "family": family,
            "model": model,
            "manifest_path": manifest_path,
            "executor_path": executor_path,
            "suites": parsed_suites,
        }

    def _manifest_path(self, family: str, model: str) -> Path:
        manifest_root = self.root / family / "tests" / "manifests"
        matches = []
        for path in sorted(manifest_root.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise QualificationError(f"cannot read model manifest {path}: {error}") from error
            if isinstance(value, Mapping) and value.get("name") == model:
                matches.append(path.resolve())
        if len(matches) != 1:
            raise QualificationError(
                f"model {model!r} must have exactly one manifest in family {family!r}"
            )
        try:
            raw = json.loads(matches[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise QualificationError(f"cannot read model manifest {matches[0]}: {error}") from error
        if raw.get("family") != family:
            raise QualificationError(
                f"manifest {matches[0]} declares family {raw.get('family')!r}, expected {family!r}"
            )
        return matches[0]

    @staticmethod
    def _plan_item(record: Mapping[str, Any]) -> PlanItem:
        identity = {
            "family": record["family"],
            "model": record["model"],
            "kind": record["kind"],
            "suite_id": record["suite_id"],
            "case_id": record["case_id"],
            "gate_policy": record["gate_policy"],
            "manifest_sha256": _file_digest(record["manifest_path"]),
            "config_sha256": _file_digest(record["config_path"]),
            "executor_sha256": _file_digest(record["executor_path"]),
            "family_sources": _qualification_sources(Path(record["executor_path"])),
            "definition_sha256": (
                _file_digest(record["definition_path"]) if record["definition_path"] else None
            ),
            "definition": record["definition"],
            "case": record["case"],
        }
        return PlanItem(
            item_id=_digest(identity),
            family=str(record["family"]),
            model=str(record["model"]),
            kind=str(record["kind"]),
            suite_id=str(record["suite_id"]),
            case_id=str(record["case_id"]),
            gate_policy=str(record["gate_policy"]),
            manifest_path=Path(record["manifest_path"]),
            config_path=Path(record["config_path"]),
            definition_path=(
                Path(record["definition_path"]) if record["definition_path"] else None
            ),
            executor_path=Path(record["executor_path"]),
            definition=dict(record["definition"]),
            case=dict(record["case"]),
        )


class QualificationRunner:
    """Execute each plan item in its family-owned Python process."""

    def __init__(self, python: Path | None = None) -> None:
        self.python = (python or Path(sys.executable)).expanduser().absolute()

    def run(
        self,
        plan: QualificationPlan,
        output_dir: Path,
        environment: Mapping[str, Any],
        *,
        resume: bool = False,
        prepare_only: bool = False,
    ) -> dict[str, Any]:
        output_dir = output_dir.expanduser().resolve()
        source = _source_evidence(plan.families_root)
        if resume:
            if not output_dir.is_dir():
                raise QualificationError(f"resume directory does not exist: {output_dir}")
            existing = load_plan(output_dir / "plan.json")
            if existing.plan_id != plan.plan_id:
                raise QualificationError("resume plan does not match the existing run")
            stored_environment = _read_json(
                output_dir / "environment.json", "stored qualification environment"
            )
            if stored_environment != dict(environment):
                raise QualificationError("resume environment does not match the existing run")
            if _read_json(output_dir / "source.json", "run source") != source:
                raise QualificationError("source changed; prepare a new run")
        elif output_dir.exists():
            raise QualificationError(f"output directory already exists: {output_dir}")
        else:
            output_dir.mkdir(parents=True)

        _write_json(output_dir / "plan.json", plan.to_json())
        _write_json(output_dir / "environment.json", dict(environment))
        _write_json(output_dir / "source.json", source)
        items_root = output_dir / "items"
        items_root.mkdir(exist_ok=True)
        timeout = _execution_timeout(environment)
        family_environments: dict[str, Any] = {}
        for family in dict.fromkeys(item.family for item in plan.items):
            selected = [item for item in plan.items if item.family == family]
            try:
                if any(_current_item_id(item) != item.item_id for item in selected):
                    raise QualificationError(
                        "qualification source changed after the plan was created"
                    )
                family_environments[family] = prepare_family_environment(
                    family_root=plan.families_root / family,
                    cases=[item.to_json() for item in selected],
                    environment=environment,
                    common_python=self.python,
                    directory=output_dir / "environments" / family,
                    timeout=timeout,
                    reuse=resume,
                )
            except (
                OSError,
                ValueError,
                KeyError,
                subprocess.SubprocessError,
                QualificationError,
            ) as error:
                family_environments[family] = {"error": str(error)}
        preparations = self._prepare_items(
            plan, output_dir, environment, family_environments, timeout
        )
        if prepare_only:
            receipt = {
                "plan_id": plan.plan_id,
                "items": preparations,
                "status": "prepared"
                if all(value["execution"] == "completed" for value in preparations.values())
                else "error",
            }
            _write_json(output_dir / "preparation.json", receipt)
            return receipt
        for index, item in enumerate(plan.items, start=1):
            item_dir = items_root / _item_directory_name(index, item)
            item_dir.mkdir(exist_ok=True)
            result_path = item_dir / "result.json"
            resolved = family_environments[item.family]
            if "error" in resolved:
                if result_path.exists():
                    _archive_attempt(item_dir)
                _write_json(
                    result_path,
                    _error_result(item, "environment preparation failed: " + resolved["error"]),
                )
                continue
            preparation = preparations[item.item_id]
            if preparation["execution"] != "completed":
                if result_path.exists():
                    _archive_attempt(item_dir)
                _write_json(
                    result_path,
                    _error_result(
                        item,
                        "case preparation failed: "
                        + preparation["details"].get("error", "see preparations/"),
                    ),
                )
                continue
            if resume and result_path.is_file():
                try:
                    existing_result = _read_json(result_path, "qualification result")
                    _validate_result(existing_result, item, item_dir)
                    if existing_result["execution"] == "completed":
                        continue
                    _archive_attempt(item_dir)
                except QualificationError:
                    result_path.unlink()
            item_environment = {
                **environment,
                "tools": {**environment.get("tools", {}), **resolved["interpreters"]},
            }
            _write_json(item_dir / "python-environment.json", resolved)
            self._run_item(
                plan,
                item,
                item_dir,
                item_environment,
                timeout,
                preparation=preparation.get("details", {}).get("prepared", {}),
            )
        return generate_report(plan, output_dir)

    def _prepare_items(self, plan, output_dir, environment, family_environments, timeout):
        results = {}
        for index, item in enumerate(plan.items, start=1):
            resolved = family_environments[item.family]
            if "error" in resolved:
                results[item.item_id] = _error_result(item, resolved["error"])
                continue
            directory = output_dir / "preparations" / _item_directory_name(index, item)
            directory.mkdir(parents=True, exist_ok=True)
            item_environment = {
                **environment,
                "tools": {**environment.get("tools", {}), **resolved["interpreters"]},
            }
            try:
                results[item.item_id] = self._prepare_item(
                    plan, item, directory, item_environment, timeout
                )
            except (QualificationError, OSError, ValueError, KeyError) as error:
                results[item.item_id] = _error_result(item, str(error))
                _write_json(directory / "last-error.json", results[item.item_id])
        return results

    def _prepare_item(self, plan, item, directory, environment, timeout):
        result_path = directory / "result.json"
        if result_path.is_file():
            previous = _read_json(result_path, "preparation result")
            _validate_result(previous, item, directory)
            if previous["execution"] == "completed":
                prepared = previous.get("details", {}).get("prepared", {})
                _prepared_inputs(directory, prepared, reuse=True)
                verification = directory / "verification"
                verification.mkdir(exist_ok=True)
                self._run_item(
                    plan,
                    item,
                    verification,
                    environment,
                    timeout,
                    phase="check",
                    preparation=prepared,
                )
                checked = _read_json(verification / "result.json", "preparation verification")
                return previous if checked["execution"] == "completed" else checked
            _archive_attempt(directory)
        self._run_item(plan, item, directory, environment, timeout, phase="prepare")
        result = _read_json(result_path, "preparation result")
        if result["execution"] == "completed":
            _prepared_inputs(directory, result.get("details", {}).get("prepared", {}), reuse=False)
        return result

    def _run_item(
        self,
        plan: QualificationPlan,
        item: PlanItem,
        item_dir: Path,
        environment: Mapping[str, Any],
        timeout: int,
        *,
        phase: str = "run",
        preparation: Mapping[str, Any] | None = None,
    ) -> None:
        request_path = item_dir / "request.json"
        result_path = item_dir / "result.json"
        stdout_path = item_dir / "executor.stdout.log"
        stderr_path = item_dir / "executor.stderr.log"
        request = {
            "schema_version": REQUEST_SCHEMA,
            "plan_id": plan.plan_id,
            "plan_item": item.to_json(),
            "families_root": str(plan.families_root),
            "environment": dict(environment),
            "phase": phase,
            "preparation": dict(preparation or {}),
        }
        _write_json(request_path, request)
        try:
            if _current_item_id(item) != item.item_id:
                raise QualificationError("qualification source changed after the plan was created")
        except (OSError, QualificationError) as error:
            _write_json(result_path, _error_result(item, str(error)))
            return
        command = [
            str(environment["tools"]["python"]),
            str(item.executor_path),
            "--request",
            str(request_path),
            "--output",
            str(result_path),
        ]
        _write_json(item_dir / "command.json", {"argv": command, "cwd": str(item_dir)})
        process_env = process_environment()
        repository = plan.families_root.parent
        source_paths = [
            repository / "core" / "builder",
            repository / "apps" / "benchmark",
            repository,
        ]
        if (repository / "apps" / "benchmark" / "trtmc_benchmark").is_dir():
            process_env["PYTHONPATH"] = ":".join(str(path) for path in source_paths)
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout,
                stderr_path.open("w", encoding="utf-8") as stderr,
            ):
                completed = subprocess.run(
                    command,
                    cwd=item_dir,
                    env=process_env,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=timeout,
                )
        except (OSError, subprocess.SubprocessError) as error:
            _write_json(result_path, _error_result(item, f"executor failed: {error}"))
            return

        try:
            result = _read_json(result_path, "qualification result")
            _validate_result(result, item, item_dir)
            expected_returncode = 0 if result["execution"] == "completed" else 1
            if completed.returncode != expected_returncode:
                raise QualificationError(
                    f"executor exit code {completed.returncode} disagrees with "
                    f"execution={result['execution']}"
                )
        except QualificationError as error:
            _write_json(result_path, _error_result(item, str(error)))


def load_environment(path: Path) -> dict[str, Any]:
    raw = _read_yaml(path.expanduser().resolve(), "qualification environment")
    if raw.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise QualificationError(f"environment schema_version must be {ENVIRONMENT_SCHEMA}")
    _nonempty_string(raw.get("name"), "environment name")
    for field in ("tools", "storage", "execution"):
        if not isinstance(raw.get(field, {}), Mapping):
            raise QualificationError(f"environment {field} must be an object")
    retired = {"reference_python", "hf_transformers_runner"} & set(raw.get("tools", {}))
    if retired:
        raise QualificationError(
            "reference tools are family-owned; configure tools.python and optional family preparation"
        )
    for field in ("allow_environment_creation", "allow_build", "local_files_only"):
        value = raw.get("execution", {}).get(field, False)
        if not isinstance(value, bool):
            raise QualificationError(f"execution.{field} must be a boolean")
    _performance_target(raw.get("performance_target"))
    return raw


def load_run_configuration(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = _read_yaml(path, "qualification run configuration")
    if raw.get("schema_version") != RUN_CONFIGURATION_SCHEMA:
        raise QualificationError(
            f"run configuration schema_version must be {RUN_CONFIGURATION_SCHEMA}"
        )
    _nonempty_string(raw.get("name"), "run configuration name")
    kind = _nonempty_string(raw.get("kind"), "run configuration kind")
    if kind not in SUPPORTED_KINDS:
        raise QualificationError(f"unsupported run configuration kind {kind!r}")
    environment = Path(_nonempty_string(raw.get("environment"), "run configuration environment"))
    if not environment.is_absolute():
        environment = path.parent / environment
    result = {
        "kind": kind,
        "environment": environment.resolve(),
        "performance": _performance_target(raw.get("performance")),
    }
    for field in ("models", "suites", "cases"):
        values = raw.get(field, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise QualificationError(f"run configuration {field} must be a list of exact names")
        _requested(values, field)
        result[field] = tuple(values)
    return result


def load_plan(path: Path) -> QualificationPlan:
    raw = _read_json(path.expanduser().resolve(), "qualification plan")
    if raw.get("schema_version") != PLAN_SCHEMA:
        raise QualificationError(f"plan schema_version must be {PLAN_SCHEMA}")
    plan_id = _nonempty_string(raw.get("plan_id"), "plan id")
    kind = _nonempty_string(raw.get("kind"), "plan kind")
    if kind not in SUPPORTED_KINDS:
        raise QualificationError(f"unsupported plan kind {kind!r}")
    families_root = _nonempty_string(raw.get("families_root"), "plan families_root")
    created_at = _nonempty_string(raw.get("created_at"), "plan created_at")
    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        raise QualificationError("plan items must be a list")
    items = tuple(PlanItem.from_json(item) for item in raw_items if isinstance(item, Mapping))
    if len(items) != len(raw_items):
        raise QualificationError("plan items must be objects")
    if any(item.kind != kind for item in items):
        raise QualificationError("plan item kind does not match the plan kind")
    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        raise QualificationError("plan contains duplicate item ids")
    expected = _digest({"kind": kind, "items": [item.to_json() for item in items]})
    if expected != plan_id:
        raise QualificationError("plan id does not match its items")
    return QualificationPlan(plan_id, kind, Path(families_root), items, created_at)


def generate_report(plan: QualificationPlan, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    environment_path = output_dir / "environment.json"
    environment = (
        _read_json(environment_path, "run environment") if environment_path.is_file() else {}
    )
    target = _performance_target(environment.get("performance_target"))
    rows = []
    blocking = {"pass": 0, "fail": 0, "error": 0, "missing": 0}
    observations = {"completed": 0, "error": 0, "missing": 0}
    expected_paths: set[Path] = set()
    for index, item in enumerate(plan.items, start=1):
        item_dir = output_dir / "items" / _item_directory_name(index, item)
        result_path = item_dir / "result.json"
        expected_paths.add(result_path.resolve())
        if result_path.is_file():
            try:
                result = _read_json(result_path, "qualification result")
                _validate_result(result, item, item_dir)
            except QualificationError as error:
                result = _error_result(item, str(error))
        else:
            result = _error_result(
                item, f"planned result is missing: {result_path}", execution="missing"
            )
        report_artifacts = list(result.get("artifacts", []))
        result = {
            **result,
            "comparison_valid": result.get("details", {}).get(
                "comparison_valid", result["execution"] == "completed"
            ),
        }
        for label, filename in (
            ("executor stdout", "executor.stdout.log"),
            ("executor stderr", "executor.stderr.log"),
            ("Python environment", "python-environment.json"),
            ("executor command", "command.json"),
        ):
            if not (item_dir / filename).is_symlink() and (item_dir / filename).is_file():
                report_artifacts.append({"label": label, "path": filename})
        rows.append(
            {
                **result,
                "artifacts": report_artifacts,
                "gate_policy": item.gate_policy,
                "result_directory": item_dir.relative_to(output_dir).as_posix(),
                "result_path": result_path.relative_to(output_dir).as_posix(),
                "preparation_path": (
                    "preparations/" + _item_directory_name(index, item) + "/result.json"
                    if (
                        output_dir
                        / "preparations"
                        / _item_directory_name(index, item)
                        / "result.json"
                    ).is_file()
                    else None
                ),
            }
        )
        if item.gate_policy == "blocking":
            if result["execution"] == "missing":
                blocking["missing"] += 1
            elif result["execution"] != "completed":
                blocking["error"] += 1
            else:
                blocking[str(result["verdict"])] += 1
        elif result["execution"] == "missing":
            observations["missing"] += 1
        elif result["execution"] == "completed":
            observations["completed"] += 1
        else:
            observations["error"] += 1

    actual_paths = {
        path.resolve() for path in (output_dir / "items").glob("*/result.json") if path.is_file()
    }
    unexpected = sorted(str(path) for path in actual_paths - expected_paths)
    if (
        unexpected
        or blocking["error"]
        or blocking["missing"]
        or observations["error"]
        or observations["missing"]
    ):
        status = "error"
    elif blocking["fail"]:
        status = "fail"
    elif not plan.items:
        status = "empty"
    elif sum(blocking.values()) == 0:
        status = "observed"
    else:
        status = "pass"
    target_failures = 0
    for row in rows:
        if plan.kind == "performance" and target and row["comparison_valid"]:
            ratio = row.get("details", {}).get("metrics", {}).get("reference_over_candidate_p50")
            valid = (
                isinstance(ratio, (int, float))
                and not isinstance(ratio, bool)
                and math.isfinite(ratio)
                and ratio > 0
            )
            if not valid:
                row["comparison_valid"] = False
                row["execution"] = "error"
                row["verdict"] = None
                row["details"] = {
                    **row.get("details", {}),
                    "error": "Performance target requires a valid speed ratio",
                }
                status = "error"
                continue
            passed = ratio >= target["minimum_speedup"]
            row["performance_target"] = {**target, "passed": passed, "actual_speedup": ratio}
            target_failures += not passed
    if target_failures and target["blocking"] and status != "error":
        status = "fail"
    report = {
        "schema_version": REPORT_SCHEMA,
        "plan_id": plan.plan_id,
        "kind": plan.kind,
        "status": status,
        "generated_at": _now(),
        "source": _read_json(output_dir / "source.json", "source evidence")
        if (output_dir / "source.json").is_file()
        else None,
        "environment": environment,
        "summary": {
            "planned": len(plan.items),
            "results": len(actual_paths & expected_paths),
            "blocking": blocking,
            "observation_only": observations,
            "unexpected_results": len(unexpected),
            "performance_target_failures": target_failures,
            "completed": sum(row["execution"] == "completed" for row in rows),
            "comparable": sum(row["comparison_valid"] is True for row in rows),
            "inconclusive": sum(
                row.get("details", {}).get("error") == "measurement_inconclusive" for row in rows
            ),
            "errors": sum(row["execution"] == "error" for row in rows),
            "failed": sum(row["verdict"] == "fail" for row in rows),
        },
        "unexpected_results": unexpected,
        "items": rows,
    }
    _write_json(output_dir / "report.json", report)
    _write_html_report(report, output_dir / "report.html")
    return report


def _suite_definition(
    suite: Mapping[str, Any],
    suites_root: Path,
    config_path: Path,
    suite_id: str,
    *,
    shared: bool,
) -> tuple[dict[str, Any], Path | None]:
    if not shared:
        inline = suite.get("definition")
        if not isinstance(inline, Mapping) or not inline:
            raise QualificationError(
                f"{config_path}: suite {suite_id!r} definition must be a non-empty object"
            )
        return dict(inline), None
    if Path(suite_id).name != suite_id:
        raise QualificationError(f"{config_path}: benchmark {suite_id!r} must be a plain name")
    definition_path = (suites_root / f"{suite_id}.yaml").resolve()
    _require_local_file(definition_path, suites_root, "shared benchmark definition")
    definition = _read_yaml(definition_path, "suite definition")
    if not definition:
        raise QualificationError(f"suite definition is empty: {definition_path}")
    return definition, definition_path


def _performance_target(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"minimum_speedup", "blocking"}:
        raise QualificationError("performance target requires minimum_speedup and blocking")
    minimum = value["minimum_speedup"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(minimum)
        or minimum <= 0
    ):
        raise QualificationError("minimum_speedup must be a finite positive number")
    if not isinstance(value["blocking"], bool):
        raise QualificationError("performance blocking must be a boolean")
    return dict(value)


def _require_local_file(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise QualificationError(f"{label} escapes its allowed directory: {path}") from error
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} does not exist as a regular file: {path}")


def _find_forbidden_fields(value: Any, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_name, nested in value.items():
            name = str(raw_name)
            field = f"{prefix}.{name}" if prefix else name
            if name in _FORBIDDEN_MODEL_POLICY_FIELDS:
                found.add(field)
            found.update(_find_forbidden_fields(nested, field))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.update(_find_forbidden_fields(nested, f"{prefix}[{index}]"))
    return found


def _validate_result(value: Mapping[str, Any], item: PlanItem, item_dir: Path) -> None:
    if value.get("schema_version") != RESULT_SCHEMA:
        raise QualificationError(f"result schema_version must be {RESULT_SCHEMA}")
    expected = {
        "plan_item_id": item.item_id,
        "family": item.family,
        "model": item.model,
        "kind": item.kind,
        "suite_id": item.suite_id,
        "case_id": item.case_id,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise QualificationError(
                f"result {field} {value.get(field)!r} does not match {expected_value!r}"
            )
    execution = value.get("execution")
    verdict = value.get("verdict")
    if execution not in {"completed", "error"}:
        raise QualificationError("result execution must be completed or error")
    if execution == "error" and verdict is not None:
        raise QualificationError("an error result cannot have a verdict")
    if (
        execution == "completed"
        and item.gate_policy == "blocking"
        and verdict
        not in {
            "pass",
            "fail",
        }
    ):
        raise QualificationError("a completed blocking result must have pass or fail verdict")
    if execution == "completed" and item.gate_policy == "observation_only" and verdict is not None:
        raise QualificationError("an observation-only result cannot claim pass or fail")
    if not isinstance(value.get("details", {}), Mapping):
        raise QualificationError("result details must be an object")
    artifacts = value.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise QualificationError("result artifacts must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
            raise QualificationError("every result artifact requires a relative path")
        relative = PurePosixPath(str(artifact["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise QualificationError(
                f"artifact path must stay inside its item directory: {relative}"
            )
        artifact_path = item_dir / Path(relative)
        resolved = artifact_path.resolve()
        try:
            resolved.relative_to(item_dir.resolve())
        except ValueError as error:
            raise QualificationError(
                f"artifact path escapes its item directory: {relative}"
            ) from error
        if artifact_path.is_symlink() or not resolved.is_file():
            raise QualificationError(f"artifact is missing or not a regular file: {relative}")


def _error_result(item: PlanItem, message: str, *, execution: str = "error") -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "plan_item_id": item.item_id,
        "family": item.family,
        "model": item.model,
        "kind": item.kind,
        "suite_id": item.suite_id,
        "case_id": item.case_id,
        "execution": execution,
        "verdict": None,
        "details": {"error": message},
        "artifacts": [],
    }


def _write_html_report(report: Mapping[str, Any], path: Path) -> None:
    rows = []
    for item in report["items"]:
        detail = item.get("details", {})
        if "performance_target" in item:
            detail = {**detail, "performance_target": item["performance_target"]}
        error = detail.get("error", "") if isinstance(detail, Mapping) else ""
        evidence = html.escape(json.dumps(detail, sort_keys=True, indent=2), quote=False)
        artifact_links = []
        preparation_path = item.get("preparation_path")
        if preparation_path:
            artifact_links.append(
                f'<a href="{html.escape(quote(str(preparation_path), safe="/._-"))}">Preparation</a>'
            )
        for artifact in item.get("artifacts", []):
            if not isinstance(artifact, Mapping):
                continue
            label = html.escape(str(artifact.get("label", artifact.get("path", "artifact"))))
            relative = PurePosixPath(str(item.get("result_directory", ""))) / str(
                artifact.get("path", "")
            )
            artifact_links.append(
                f'<a href="{html.escape(quote(relative.as_posix(), safe="/._-"))}">{label}</a>'
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('family', '')))}</td>"
            f"<td>{html.escape(str(item.get('model', '')))}</td>"
            f"<td>{html.escape(str(item.get('suite_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('case_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('gate_policy', '')))}</td>"
            f"<td>{html.escape(str(item.get('execution', '')))}</td>"
            f"<td>{html.escape(str(item.get('verdict', '') or ''))}</td>"
            f"<td>{html.escape(str(error))}</td>"
            f"<td><details><summary>Metrics and evidence</summary><pre>{evidence}</pre></details></td>"
            f"<td>{'<br>'.join(artifact_links)}</td>"
            "</tr>"
        )
    summary = html.escape(json.dumps(report["summary"], sort_keys=True), quote=False)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TRTMC qualification</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: .5rem; text-align: left; }}
th {{ background: #f3f3f3; }}
code, pre {{ white-space: pre-wrap; word-break: break-word; }}
</style></head><body>
<h1>TRTMC qualification</h1>
<p>Status: {html.escape(str(report["status"]))}</p>
<p><code>{summary}</code></p>
<table><thead><tr><th>Family</th><th>Model</th><th>Suite</th><th>Case</th>
<th>Gate policy</th><th>Execution</th><th>Verdict</th><th>Error</th>
<th>Metrics and gates</th><th>Artifacts</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
</body></html>
"""
    path.write_text(document, encoding="utf-8")


def _execution_timeout(environment: Mapping[str, Any]) -> int:
    execution = environment.get("execution", {})
    if not isinstance(execution, Mapping):
        raise QualificationError("environment execution must be an object")
    timeout = execution.get("timeout_seconds", 7200)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise QualificationError("environment execution.timeout_seconds must be positive")
    return timeout


def _item_directory_name(index: int, item: PlanItem) -> str:
    label = "-".join((item.family, item.model, item.suite_id, item.case_id))
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-") or "item"
    return f"{index:04d}-{slug}"


def _archive_attempt(item_dir: Path) -> None:
    attempts_root = item_dir / "attempts"
    attempts_root.mkdir(exist_ok=True)
    attempt = 1
    while (attempts_root / f"attempt-{attempt}").exists():
        attempt += 1
    archive = attempts_root / f"attempt-{attempt}"
    archive.mkdir()
    for path in tuple(item_dir.iterdir()):
        if path != attempts_root:
            path.replace(archive / path.name)


def _current_item_id(item: PlanItem) -> str:
    return _digest(
        {
            "family": item.family,
            "model": item.model,
            "kind": item.kind,
            "suite_id": item.suite_id,
            "case_id": item.case_id,
            "gate_policy": item.gate_policy,
            "manifest_sha256": _file_digest(item.manifest_path),
            "config_sha256": _file_digest(item.config_path),
            "executor_sha256": _file_digest(item.executor_path),
            "family_sources": _qualification_sources(item.executor_path),
            "definition_sha256": (
                _file_digest(item.definition_path) if item.definition_path else None
            ),
            "definition": item.definition,
            "case": item.case,
        }
    )


def _qualification_sources(executor: Path) -> dict[str, str]:
    sources = sorted(executor.parent.rglob("*.py"))
    requirements = executor.parents[2] / "requirements.txt"
    if requirements.is_file():
        sources.append(requirements)
    return {str(path.relative_to(executor.parents[2])): _file_digest(path) for path in sources}


def _prepared_inputs(directory: Path, prepared: Mapping[str, Any], *, reuse: bool) -> None:
    paths = prepared.get("input_files", [])
    if not isinstance(paths, list) or any(
        not isinstance(path, str) or not Path(path).is_absolute() for path in paths
    ):
        raise QualificationError("prepared input_files must be absolute file paths")
    evidence = {path: _file_digest(Path(path)) for path in paths}
    receipt = directory / "inputs.json"
    if reuse:
        if evidence != _read_json(receipt, "prepared input evidence"):
            raise QualificationError("prepared input files changed; prepare a new run")
    else:
        _write_json(receipt, evidence)


def _source_evidence(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            capture_output=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "source": "installed or non-git catalog"}
    return {
        "revision": revision.stdout.strip(),
        "working_tree_diff": hashlib.sha256(diff.stdout).hexdigest(),
    }


def _requested(values: Sequence[str], label: str) -> set[str]:
    result = {value.strip() for value in values if value.strip()}
    if len(result) != len(values):
        raise QualificationError(f"{label} must be non-empty exact names")
    return result


def _require_requested(requested: set[str], discovered: set[str], label: str) -> None:
    missing = sorted(requested - discovered)
    if missing:
        raise QualificationError(f"unknown {label}: {', '.join(missing)}")


def _model_from_filename(path: Path, kind: str) -> str:
    suffix = f".{kind}.yaml"
    if not path.name.endswith(suffix):
        raise QualificationError(f"invalid qualification filename: {path}")
    return path.name[: -len(suffix)]


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QualificationError(f"{field} must be a non-empty string")
    return value


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise QualificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must contain an object: {path}")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
