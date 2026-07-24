# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose and certify the two-model GB300 dynamic-memory nightly gate.

Boundary: this module only orchestrates existing evidence producers and
validates their durable receipts; runtime, builder, and performance policy
remain owned by those producers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "trtmc.native-dynamic-memory-nightly-request/v1"
PLAN_SCHEMA = "trtmc.native-dynamic-memory-nightly-plan/v1"
GATE_SCHEMA = "trtmc.native-dynamic-memory-nightly-gate/v1"
VERIFICATION_SCHEMA = (
    "trtmc.native-dynamic-memory-nightly-artifact-verification/v1"
)
EXPECTED_MODELS = ("qwen", "tinyllama")
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "native_dynamic_memory_nightly.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_TEST_MANIFEST_COMMAND_LABELS = (
    "build",
    "build_cpp_tests_and_qualifiers",
    "ctest_manifest_all",
    "ctest_all",
    "ctest_manifest_dynamic_memory",
    "ctest_dynamic_memory",
    "pytest_manifest_dynamic_memory",
    "pytest_dynamic_memory",
    "pytest_graph_e2e",
)
_NATIVE_COMPATIBILITY_CTESTS = frozenset(
    {
        "test_bundle_format",
        "test_c_abi_runtime_regression",
        "test_abi_old_consumer",
        "test_abi_current_core_legacy_plugin",
        "test_abi_legacy_core_current_plugin",
        "test_cli_args",
    }
)
_CORRECTNESS_PROMOTION_GATES = (
    "canonical_matrix_complete",
    "case_filter_not_used",
    "case_execution_passed",
    "raw_runner_evidence_passed",
    "hf_parity_executed_and_passed",
    "source_state_unchanged",
    "source_clean_exact_head",
    "base_artifact_binding_passed",
    "runtime_kv_plugin_binding_passed",
    "full_context_memory_coverage",
    "qualified_engine_graph_passed",
    "c_div_2_variant_engine_graph_passed",
    "c_div_2_variant_producer_receipt_passed",
    "source_calibration_evidence_reopened",
    "all_profile_two_sweep_passed",
    "warmup_evidence_passed",
    "admission_rejection_evidence_passed",
)
_EMBEDDED_CALIBRATION_ROOT = "runtime_memory_calibration"
_EMBEDDED_CALIBRATION_EVIDENCE_SECTION = (
    f"{_EMBEDDED_CALIBRATION_ROOT}/evidence.json"
)
_EMBEDDED_CALIBRATION_EVIDENCE_SCHEMA = (
    "trtmc.native-dynamic-memory-build-calibration-evidence/v2"
)
_EMBEDDED_CALIBRATION_ROLES = ("base", "chunk_variant")
_RECEIPT_CONTRACTS: dict[str, dict[str, Any]] = {
    "test-manifest": {
        "schema": "trtmc.dynamic-memory-test-manifest/v2",
        "passed": True,
    },
    "dynamic-build": {
        "schema": "trtmc.native-dynamic-memory-perf-build/v2",
        "fresh_build": True,
    },
    "static-build": {
        "schema": "trtmc.native-dynamic-memory-perf-build/v2",
        "fresh_build": True,
    },
    "chunk-variant-build": {
        "schema": "trtmc.native-dynamic-memory-chunk-variant-build/v2",
        "fresh_build": True,
    },
    "correctness": {
        "schema": 1,
        "status": "passed",
        "passed": True,
        "promotion_eligible": True,
    },
    "policies": {"schema": 1, "passed": True},
    "soak": {"schema": 2, "passed": True},
    "surfaces": {"schema": 1, "passed": True},
    "performance-capture-static-short": {
        "schema": "trtmc.benchmark-worker-result/v1",
        "status": "completed",
    },
    "performance-capture-dynamic-short": {
        "schema": "trtmc.benchmark-worker-result/v1",
        "status": "completed",
    },
    "performance-capture-static-medium": {
        "schema": "trtmc.benchmark-worker-result/v1",
        "status": "completed",
    },
    "performance-capture-dynamic-medium": {
        "schema": "trtmc.benchmark-worker-result/v1",
        "status": "completed",
    },
    "performance-qualification": {
        "schema": "trtmc.native-dynamic-memory-perf-qualification/v1",
        "status": "passed",
    },
    "process-isolation": {
        "schema": "trtmc.native-dynamic-memory-process-isolation/v2",
        "status": "passed",
    },
}


class DynamicMemoryNightlyError(RuntimeError):
    """The nightly orchestration or its evidence failed closed."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DynamicMemoryNightlyError(
            f"{label} is not strict readable JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DynamicMemoryNightlyError(f"{label} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise DynamicMemoryNightlyError(f"evidence is not a file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _require_string(
    value: Mapping[str, Any], key: str, *, label: str
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise DynamicMemoryNightlyError(f"{label}.{key} must be a non-empty string")
    return item


def load_fixture(path: Path) -> dict[str, Any]:
    """Load the reviewed request fixture without accepting partial models."""

    fixture = _read_json(path.expanduser().resolve(strict=True), "request fixture")
    if fixture.get("schema_version") != REQUEST_SCHEMA:
        raise DynamicMemoryNightlyError(
            f"request fixture must use {REQUEST_SCHEMA}"
        )
    models = fixture.get("models")
    if not isinstance(models, dict) or tuple(sorted(models)) != tuple(
        sorted(EXPECTED_MODELS)
    ):
        raise DynamicMemoryNightlyError(
            f"request fixture must define exactly {EXPECTED_MODELS}"
        )
    platform = fixture.get("platform")
    performance = fixture.get("performance")
    embedded_calibration = fixture.get("embedded_automatic_calibration")
    if not isinstance(platform, dict) or not isinstance(performance, dict):
        raise DynamicMemoryNightlyError(
            "request fixture platform/performance sections must be objects"
        )
    if (
        platform.get("cuda_module_loading") != "LAZY"
        or platform.get("gpu_name") != "NVIDIA GB300"
        or platform.get("target") != "gb300-trt-11.2"
    ):
        raise DynamicMemoryNightlyError(
            "request fixture does not select the reviewed GB300/TRT 11.2 tuple"
        )
    if (
        not isinstance(embedded_calibration, dict)
        or embedded_calibration.get("required") is not True
        or embedded_calibration.get("roles") != ["base", "chunk_variant"]
        or embedded_calibration.get("calibration_schema")
        != _EMBEDDED_CALIBRATION_EVIDENCE_SCHEMA
        or embedded_calibration.get("evidence_section")
        != "runtime_memory_calibration/evidence.json"
        or embedded_calibration.get("capture_count") != 2
        or embedded_calibration.get("capture_prefixes")
        != [
            "runtime_memory_calibration/process-00/",
            "runtime_memory_calibration/process-01/",
        ]
        or embedded_calibration.get("require_complete_capture_sections")
        != [
            "command.json",
            "returncode.txt",
            "runner.stdout.log",
            "runner.stderr.log",
            "runner-output.raw.json",
            "runner-trace.json",
            "runner-logits.bin",
            "capture-manifest.json",
        ]
    ):
        raise DynamicMemoryNightlyError(
            "request fixture must require complete in-bundle base and C/2 "
            "automatic calibration evidence"
        )
    for model_key, model in models.items():
        if not isinstance(model, dict):
            raise DynamicMemoryNightlyError(
                f"request fixture model {model_key!r} must be an object"
            )
        for field in (
            "family",
            "model_id",
            "revision",
            "config_sha256",
            "default_bundle_name",
        ):
            _require_string(model, field, label=f"models.{model_key}")
        if not _GIT_SHA_RE.fullmatch(str(model["revision"])):
            raise DynamicMemoryNightlyError(
                f"models.{model_key}.revision must be an immutable commit SHA"
            )
        if not _SHA256_RE.fullmatch(str(model["config_sha256"])):
            raise DynamicMemoryNightlyError(
                f"models.{model_key}.config_sha256 must be SHA-256"
            )
        context_limit = model.get("context_limit")
        soak = model.get("soak")
        if (
            type(context_limit) is not int
            or context_limit <= 0
            or not isinstance(soak, dict)
            or type(soak.get("r1")) is not int
            or type(soak.get("r2")) is not int
            or not (0 < soak["r1"] < soak["r2"] <= context_limit)
        ):
            raise DynamicMemoryNightlyError(
                f"models.{model_key} has an invalid context/soak tuple"
            )
    return fixture


@dataclass(frozen=True)
class ReceiptExpectation:
    """One producer-owned receipt and the fields nightly is allowed to gate."""

    label: str
    path: Path
    schema: str | int
    status: str | None = None
    passed: bool | None = None
    promotion_eligible: bool | None = None
    fresh_build: bool = False


@dataclass(frozen=True)
class PlannedCommand:
    """One existing producer invocation and its primary durable receipt."""

    label: str
    argv: tuple[str, ...]
    receipt: ReceiptExpectation
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class NightlyPlan:
    """Deterministic command graph for one of the two reviewed models."""

    model_key: str
    model_id: str
    model_revision: str
    tested_sha: str
    output_dir: Path
    commands: tuple[PlannedCommand, ...]
    requests: Mapping[str, Mapping[str, Any]]

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tested_sha": self.tested_sha,
            "output_dir": str(self.output_dir),
            "commands": [
                {
                    "label": command.label,
                    "argv": list(command.argv),
                    "receipt": str(command.receipt.path),
                    "expected_schema_version": command.receipt.schema,
                    "expected_status": command.receipt.status,
                    "expected_passed": command.receipt.passed,
                    "expected_promotion_eligible": (
                        command.receipt.promotion_eligible
                    ),
                    "environment": dict(command.environment),
                }
                for command in self.commands
            ],
            "requests": dict(self.requests),
        }


def _benchmark_request(
    *,
    case_name: str,
    bundle: Path,
    prompt: str,
    measurement: Mapping[str, Any],
    semantic_request: Mapping[str, Any],
    context_limit: int,
    dynamic: bool,
) -> dict[str, Any]:
    document = {
        "schema_version": 1,
        "case_name": case_name,
        "bundle": str(bundle),
        "operation": "generate",
        "runtime": (
            {"max_sequence_length": context_limit} if dynamic else {}
        ),
        "measurement": dict(measurement),
        "request": {**semantic_request, "prompt": prompt},
    }
    document["case_digest"] = _canonical_sha(
        {
            "operation": document["operation"],
            "runtime": document["runtime"],
            "measurement": document["measurement"],
            "request": document["request"],
        }
    )
    return document


def create_plan(
    *,
    repo_root: Path,
    build_dir: Path,
    python: Path,
    output_dir: Path,
    fixture: Mapping[str, Any],
    model_key: str,
    model_snapshot: Path,
    tested_sha: str,
    producer_gpu: str,
    runner_gpu: str,
    isolation_gpu_a: str,
    isolation_gpu_b: str,
) -> NightlyPlan:
    """Return producer argv without running or weakening any producer."""

    if model_key not in EXPECTED_MODELS:
        raise DynamicMemoryNightlyError(f"unsupported model key: {model_key}")
    if not _GIT_SHA_RE.fullmatch(tested_sha):
        raise DynamicMemoryNightlyError("--expected-tested-sha must be 40 hex")
    model = fixture["models"][model_key]
    performance = fixture["performance"]
    measurement = performance["measurement"]
    semantic_request = {
        **performance["request"],
        "seed": 0,
    }
    short = (
        str(performance["short_prompt"]["text"])
        * int(performance["short_prompt"]["repeat"])
    )
    medium = (
        str(performance["medium_prompt"]["text"])
        * int(performance["medium_prompt"]["repeat"])
    )

    repo_root = repo_root.expanduser().resolve()
    build_dir = build_dir.expanduser().resolve()
    python = python.expanduser().absolute()
    output_dir = output_dir.expanduser().resolve()
    model_snapshot = model_snapshot.expanduser().resolve()
    work = output_dir / "work"
    dynamic_dir = work / "dynamic-build"
    static_dir = work / "static-build"
    bundles = work / "bundles"
    receipts = work / "receipts"
    requests_dir = work / "requests"
    logs = work / "logs"
    manifest_dir = work / "test-manifest"
    dynamic_bundle = dynamic_dir / str(model["default_bundle_name"])
    static_bundle = bundles / "static-full-context.trtfb"
    variant_bundle = bundles / "dynamic-c-div-2.trtfb"
    manifest = manifest_dir / "test-manifest-report.json"
    plugin = build_dir / "libtrtmc_trt_plugins.so"
    trtmc = build_dir / "trtmc"
    worker = build_dir / "trtmc_benchmark_worker"
    runner = build_dir / "trtmc_dynamic_memory_qualify"
    surfaces_helper = build_dir / "trtmc_dynamic_memory_surfaces"
    model_dirs = (
        build_dir / "models" / "qwen",
        build_dir / "models" / "llama",
    )
    common_paths = [
        "--backend-dir",
        str(build_dir),
        *[
            item
            for model_dir in model_dirs
            for item in ("--model-plugin-dir", str(model_dir))
        ],
    ]
    context_limit = int(model["context_limit"])
    target = str(fixture["platform"]["target"])
    build_id_prefix = (
        f"nightly-{tested_sha[:12]}-{model_key}"
    )
    base_build_receipt = receipts / "dynamic-build.json"
    static_build_receipt = receipts / "static-build.json"
    variant_build_receipt = receipts / "chunk-variant-build.json"
    correctness_report = work / "correctness" / "qualification-report.json"
    policy_report = work / "policies" / "policy-equivalence-report.json"
    soak_report = work / "soak" / "soak-report.json"
    surface_report = work / "surfaces" / "surface-equivalence-report.json"
    perf_report = work / "performance" / "performance-report.json"
    isolation_report = (
        work / "process-isolation" / "process-isolation-report.json"
    )
    requests: dict[str, Mapping[str, Any]] = {}
    request_paths: dict[str, Path] = {}
    for prompt_kind, prompt in (("short", short), ("medium", medium)):
        for role, bundle, dynamic in (
            ("static", static_bundle, False),
            ("dynamic", dynamic_bundle, True),
        ):
            name = f"{role}-{prompt_kind}"
            request_paths[name] = requests_dir / f"{name}.json"
            requests[name] = _benchmark_request(
                case_name=f"{model_key}-{name}",
                bundle=bundle,
                prompt=prompt,
                measurement=measurement,
                semantic_request=semantic_request,
                context_limit=context_limit,
                dynamic=dynamic,
            )

    commands: list[PlannedCommand] = []

    def add(
        label: str,
        argv: Sequence[str | Path],
        *,
        path: Path,
        schema: str | int,
        status: str | None = None,
        passed: bool | None = None,
        promotion: bool | None = None,
        fresh: bool = False,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        commands.append(
            PlannedCommand(
                label=label,
                argv=tuple(str(item) for item in argv),
                receipt=ReceiptExpectation(
                    label=label,
                    path=path,
                    schema=schema,
                    status=status,
                    passed=passed,
                    promotion_eligible=promotion,
                    fresh_build=fresh,
                ),
                environment=tuple(sorted((environment or {}).items())),
            )
        )

    add(
        "test-manifest",
        [
            python,
            repo_root / "tools" / "capture_dynamic_memory_test_manifest.py",
            "--repo-root",
            repo_root,
            "--build-dir",
            build_dir,
            "--python",
            python,
            "--output-dir",
            manifest_dir,
        ],
        path=manifest,
        schema="trtmc.dynamic-memory-test-manifest/v2",
        passed=True,
    )
    add(
        "dynamic-build",
        [
            python,
            repo_root / "tools" / "capture_native_dynamic_memory_perf.py",
            "--repo-root",
            repo_root,
            "build",
            "--bundle",
            dynamic_bundle,
            "--receipt",
            base_build_receipt,
            "--source-artifact-dir",
            work / "source-state" / "dynamic-build",
            "--stdout-output",
            logs / "dynamic-build.stdout",
            "--stderr-output",
            logs / "dynamic-build.stderr",
            "--build-manifest",
            manifest,
            "--plugin-library",
            plugin,
            "--cwd",
            dynamic_dir,
            "--role",
            "native-dynamic",
            "--model-id",
            model["model_id"],
            "--model-revision",
            model["revision"],
            "--precision",
            "bf16",
            "--target",
            target,
            "--bundle-build-id",
            f"{build_id_prefix}-dynamic",
            "--",
            trtmc,
            "build",
            model["model_id"],
        ],
        path=base_build_receipt,
        schema="trtmc.native-dynamic-memory-perf-build/v2",
        fresh=True,
    )
    add(
        "static-build",
        [
            python,
            repo_root / "tools" / "capture_native_dynamic_memory_perf.py",
            "--repo-root",
            repo_root,
            "build",
            "--bundle",
            static_bundle,
            "--receipt",
            static_build_receipt,
            "--source-artifact-dir",
            work / "source-state" / "static-build",
            "--stdout-output",
            logs / "static-build.stdout",
            "--stderr-output",
            logs / "static-build.stderr",
            "--build-manifest",
            manifest,
            "--cwd",
            static_dir,
            "--role",
            "exact-head-static-split",
            "--model-id",
            model["model_id"],
            "--model-revision",
            model["revision"],
            "--precision",
            "bf16",
            "--target",
            target,
            "--bundle-build-id",
            f"{build_id_prefix}-static",
            "--",
            trtmc,
            "build",
            model["model_id"],
            "--model-revision",
            model["revision"],
            "--max-cache-length",
            str(context_limit),
            "--precision",
            "bf16",
            "--output",
            static_bundle,
        ],
        path=static_build_receipt,
        schema="trtmc.native-dynamic-memory-perf-build/v2",
        fresh=True,
    )
    add(
        "chunk-variant-build",
        [
            python,
            repo_root / "tools" / "build_native_dynamic_memory_chunk_variant.py",
            "--model",
            model["model_id"],
            "--model-revision",
            model["revision"],
            "--output",
            variant_bundle,
            "--receipt",
            variant_build_receipt,
            "--plugin-library",
            plugin,
            "--build-manifest",
            manifest,
            "--build-timing-json",
            logs / "chunk-variant-build-timing.json",
        ],
        path=variant_build_receipt,
        schema="trtmc.native-dynamic-memory-chunk-variant-build/v2",
        fresh=True,
        environment={
            "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR": str(runner),
            "TRTMC_DEVELOPER_CHUNK_VARIANT": "C/2",
        },
    )
    add(
        "correctness",
        [
            python,
            repo_root / "tools" / "qualify_native_dynamic_memory.py",
            "--bundle",
            dynamic_bundle,
            "--model",
            model_snapshot,
            "--runner",
            runner,
            "--build-manifest",
            manifest,
            "--base-build-receipt",
            base_build_receipt,
            "--chunk-variant-bundle",
            variant_bundle,
            "--chunk-variant-build-receipt",
            variant_build_receipt,
            "--output-dir",
            correctness_report.parent,
            "--runner-cuda-visible-device",
            runner_gpu,
            "--device",
            "cuda:0",
        ],
        path=correctness_report,
        schema=1,
        status="passed",
        passed=True,
        promotion=True,
        environment={
            "TRTMC_DEVELOPER_CHUNK_VARIANT": "C/2",
        },
    )
    add(
        "policies",
        [
            python,
            repo_root / "tools" / "qualify_native_dynamic_memory_policies.py",
            "--bundle",
            dynamic_bundle,
            "--runner",
            runner,
            "--output-dir",
            policy_report.parent,
            *common_paths,
        ],
        path=policy_report,
        schema=1,
        passed=True,
    )
    soak = model["soak"]
    soak_command: list[str | Path] = [
        python,
        repo_root / "tools" / "qualify_native_dynamic_memory_soak.py",
        "--bundle",
        dynamic_bundle,
        "--runner",
        runner,
        "--output",
        soak_report,
        "--r1",
        str(soak["r1"]),
        "--r2",
        str(soak["r2"]),
    ]
    if soak.get("reservation_target_tokens") is not None:
        soak_command.extend(
            (
                "--reservation-target-tokens",
                str(soak["reservation_target_tokens"]),
            )
        )
    add(
        "soak",
        soak_command,
        path=soak_report,
        schema=2,
        passed=True,
    )
    add(
        "surfaces",
        [
            python,
            repo_root / "tools" / "qualify_native_dynamic_memory_surfaces.py",
            "--bundle",
            dynamic_bundle,
            "--binary",
            trtmc,
            "--helper",
            surfaces_helper,
            "--output-dir",
            surface_report.parent,
            *common_paths,
            "--hf-python",
            python,
        ],
        path=surface_report,
        schema=1,
        passed=True,
    )
    capture_paths: dict[str, Path] = {}
    for prompt_kind in ("short", "medium"):
        for role in ("static", "dynamic"):
            name = f"{role}-{prompt_kind}"
            dynamic = role == "dynamic"
            capture = work / "performance" / f"{name}.json"
            capture_paths[name] = capture
            add(
                f"performance-capture-{name}",
                [
                    python,
                    repo_root
                    / "tools"
                    / "capture_native_dynamic_memory_perf.py",
                    "--repo-root",
                    repo_root,
                    "benchmark",
                    "--bundle",
                    dynamic_bundle if dynamic else static_bundle,
                    "--build-receipt",
                    (
                        base_build_receipt
                        if dynamic
                        else static_build_receipt
                    ),
                    "--request",
                    request_paths[name],
                    "--worker",
                    worker,
                    "--plugin-library",
                    plugin,
                    "--output",
                    capture,
                    "--stderr-output",
                    logs / f"{name}.stderr",
                    "--comparison-sequence-limit",
                    str(context_limit),
                    "--cwd",
                    repo_root,
                    "--role",
                    (
                        "native-dynamic"
                        if dynamic
                        else "exact-head-static-split"
                    ),
                ],
                path=capture,
                schema="trtmc.benchmark-worker-result/v1",
                status="completed",
            )
    add(
        "performance-qualification",
        [
            python,
            repo_root / "tools" / "qualify_native_dynamic_memory_perf.py",
            "--static-short",
            capture_paths["static-short"],
            "--dynamic-short",
            capture_paths["dynamic-short"],
            "--static-medium",
            capture_paths["static-medium"],
            "--dynamic-medium",
            capture_paths["dynamic-medium"],
            "--static-bundle",
            static_bundle,
            "--dynamic-bundle",
            dynamic_bundle,
            "--repo-root",
            repo_root,
            "--output",
            perf_report,
        ],
        path=perf_report,
        schema="trtmc.native-dynamic-memory-perf-qualification/v1",
        status="passed",
    )
    add(
        "process-isolation",
        [
            python,
            repo_root
            / "tools"
            / "capture_native_dynamic_memory_process_isolation.py",
            "--repo-root",
            repo_root,
            "--python",
            python,
            "--bundle",
            dynamic_bundle,
            "--build-receipt",
            base_build_receipt,
            "--request",
            request_paths["dynamic-medium"],
            "--correctness-report",
            correctness_report,
            "--performance-report",
            perf_report,
            "--worker",
            worker,
            "--plugin-library",
            plugin,
            "--comparison-sequence-limit",
            str(context_limit),
            "--gpu-a",
            isolation_gpu_a,
            "--gpu-b",
            isolation_gpu_b,
            "--output-dir",
            isolation_report.parent,
        ],
        path=isolation_report,
        schema="trtmc.native-dynamic-memory-process-isolation/v2",
        status="passed",
    )
    if [command.label for command in commands] != list(_RECEIPT_CONTRACTS):
        raise DynamicMemoryNightlyError(
            "internal producer graph no longer matches its receipt contracts"
        )
    return NightlyPlan(
        model_key=model_key,
        model_id=str(model["model_id"]),
        model_revision=str(model["revision"]),
        tested_sha=tested_sha,
        output_dir=output_dir,
        commands=tuple(commands),
        requests=requests,
    )


def _source_shas(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"git_head", "source_revision"} and isinstance(item, str):
                found.add(item)
            found.update(_source_shas(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_source_shas(item))
    return found


def _boolean_gate_errors(value: Any, *, label: str) -> list[str]:
    """Return every false boolean leaf and reject a gate with no booleans."""

    false_paths: list[str] = []
    boolean_count = 0

    def visit(node: Any, path: str) -> None:
        nonlocal boolean_count
        if isinstance(node, Mapping):
            for key, item in node.items():
                visit(item, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]")
        elif isinstance(node, bool):
            boolean_count += 1
            if not node:
                false_paths.append(path)

    visit(value, label)
    if boolean_count == 0:
        return [f"{label} contains no boolean gate evidence"]
    return [f"{path} is not true" for path in false_paths]


def _test_manifest_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("source_state_unchanged") is not True:
        errors.append("test-manifest: source_state_unchanged is not true")
    commands = payload.get("commands")
    if not isinstance(commands, list):
        return [*errors, "test-manifest: commands must be an array"]
    labels = [
        command.get("label") if isinstance(command, Mapping) else None
        for command in commands
    ]
    if labels != list(_TEST_MANIFEST_COMMAND_LABELS):
        errors.append(
            "test-manifest: command labels do not match the frozen nine-command "
            "manifest"
        )
        return errors
    for command in commands:
        assert isinstance(command, Mapping)
        if (
            command.get("passed") is not True
            or type(command.get("returncode")) is not int
            or command["returncode"] != 0
        ):
            errors.append(
                f"test-manifest: command {command['label']!r} did not pass"
            )
    dynamic_manifest = next(
        command
        for command in commands
        if command["label"] == "ctest_manifest_dynamic_memory"
    )
    entries = dynamic_manifest.get("manifest_entries")
    if not isinstance(entries, list) or not all(
        isinstance(entry, str) and entry for entry in entries
    ):
        errors.append(
            "test-manifest: dynamic-memory CTest manifest is missing"
        )
    else:
        missing = sorted(_NATIVE_COMPATIBILITY_CTESTS.difference(entries))
        if missing:
            errors.append(
                "test-manifest: native compatibility coverage is missing "
                f"{missing}"
            )
    return errors


def _embedded_section_receipt_errors(
    value: Any,
    *,
    expected_section_name: str,
    label: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label} is not a section receipt"]
    if value.get("section_name") != expected_section_name:
        return [
            f"{label}.section_name={value.get('section_name')!r}, expected "
            f"{expected_section_name!r}"
        ]
    size_bytes = value.get("size_bytes")
    if type(size_bytes) is not int or size_bytes <= 0:
        return [f"{label}.size_bytes is not positive"]
    sha256 = value.get("sha256")
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        return [f"{label}.sha256 is not SHA-256"]
    return []


def _embedded_calibration_source_errors(
    value: Any,
    *,
    label: str,
) -> list[str]:
    """Require the correctness producer's reopened in-bundle source binding."""

    if not isinstance(value, Mapping):
        return [f"{label} is absent"]
    errors: list[str] = []
    if value.get("source") != "embedded_bundle_sections":
        errors.append(
            f"{label}.source is not 'embedded_bundle_sections'"
        )
    if (
        value.get("evidence_schema")
        != _EMBEDDED_CALIBRATION_EVIDENCE_SCHEMA
    ):
        errors.append(
            f"{label}.evidence_schema is not "
            f"{_EMBEDDED_CALIBRATION_EVIDENCE_SCHEMA!r}"
        )
    if value.get("passed") is not True:
        errors.append(f"{label}.passed is not true")
    errors.extend(
        _embedded_section_receipt_errors(
            value.get("evidence_section"),
            expected_section_name=_EMBEDDED_CALIBRATION_EVIDENCE_SECTION,
            label=f"{label}.evidence_section",
        )
    )

    bundle = value.get("bundle")
    if (
        not isinstance(bundle, Mapping)
        or type(bundle.get("size_bytes")) is not int
        or bundle["size_bytes"] <= 0
        or not isinstance(bundle.get("sha256"), str)
        or _SHA256_RE.fullmatch(bundle["sha256"]) is None
    ):
        errors.append(f"{label}.bundle identity is invalid")
    runner_sha = value.get("runner_sha256")
    if not isinstance(runner_sha, str) or _SHA256_RE.fullmatch(runner_sha) is None:
        errors.append(f"{label}.runner_sha256 is not SHA-256")

    provenance = value.get("contract_provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance)
        != {
            "qualified_runtime_stack_sha256",
            "plan_set_sha256",
            "cuda_module_loading_mode",
            "plans",
        }
        or provenance.get("cuda_module_loading_mode") != "lazy"
        or any(
            not isinstance(provenance.get(field), str)
            or _SHA256_RE.fullmatch(provenance[field]) is None
            for field in (
                "qualified_runtime_stack_sha256",
                "plan_set_sha256",
            )
        )
    ):
        errors.append(f"{label}.contract_provenance is invalid")
    else:
        plans = provenance.get("plans")
        if (
            not isinstance(plans, list)
            or [
                plan.get("section_name")
                if isinstance(plan, Mapping)
                else None
                for plan in plans
            ]
            != ["engine_plan", "prefill_engine_plan"]
            or any(
                not isinstance(plan.get("section_sha256"), str)
                or _SHA256_RE.fullmatch(plan["section_sha256"]) is None
                for plan in plans
                if isinstance(plan, Mapping)
            )
        ):
            errors.append(f"{label}.contract_provenance.plans is invalid")

    expected_section_sets = {
        "capture_manifests": "capture-manifest.json",
        "raw_captures": "runner-output.raw.json",
        "logits": "runner-logits.bin",
    }
    for field, filename in expected_section_sets.items():
        receipts = value.get(field)
        if not isinstance(receipts, list) or len(receipts) != 2:
            errors.append(f"{label}.{field} does not contain two captures")
            continue
        for process_index, receipt in enumerate(receipts):
            errors.extend(
                _embedded_section_receipt_errors(
                    receipt,
                    expected_section_name=(
                        f"{_EMBEDDED_CALIBRATION_ROOT}/"
                        f"process-{process_index:02d}/{filename}"
                    ),
                    label=f"{label}.{field}[{process_index}]",
                )
            )

    reserves = value.get("recommended_profile_reserves")
    if not isinstance(reserves, list) or not reserves:
        errors.append(f"{label}.recommended_profile_reserves is empty")
    else:
        previous_limit = 0
        previous_reserve = 0
        for index, row in enumerate(reserves):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "covering_profile_limit",
                    "cumulative_reserve_bytes",
                }
                or type(row.get("covering_profile_limit")) is not int
                or row["covering_profile_limit"] <= previous_limit
                or type(row.get("cumulative_reserve_bytes")) is not int
                or row["cumulative_reserve_bytes"] < previous_reserve
                or row["cumulative_reserve_bytes"] <= 0
            ):
                errors.append(
                    f"{label}.recommended_profile_reserves[{index}] is invalid"
                )
                break
            previous_limit = row["covering_profile_limit"]
            previous_reserve = row["cumulative_reserve_bytes"]

    exemption = value.get("bootstrap_cycle_exemption")
    if (
        not isinstance(exemption, Mapping)
        or exemption.get("all_other_receipt_provenance_replayed") is not True
        or exemption.get("field") != "module_residency_evidence_sha256"
        or not isinstance(exemption.get("final_sealed_value"), str)
        or _SHA256_RE.fullmatch(exemption["final_sealed_value"]) is None
        or not isinstance(exemption.get("observed_bootstrap_values"), list)
        or len(exemption["observed_bootstrap_values"]) != 1
        or not isinstance(exemption["observed_bootstrap_values"][0], str)
        or _SHA256_RE.fullmatch(exemption["observed_bootstrap_values"][0])
        is None
    ):
        errors.append(f"{label}.bootstrap_cycle_exemption is invalid")
    return errors


def _receipt_semantic_errors(
    label: str,
    payload: Mapping[str, Any],
    *,
    expected_model_id: str,
    expected_model_revision: str,
) -> list[str]:
    """Gate producer-owned durable fields without reimplementing producers."""

    errors: list[str] = []
    if label == "test-manifest":
        return _test_manifest_errors(payload)

    top_level_model_labels = {
        "dynamic-build",
        "static-build",
        "correctness",
        "policies",
        "soak",
        "surfaces",
        "performance-capture-static-short",
        "performance-capture-dynamic-short",
        "performance-capture-static-medium",
        "performance-capture-dynamic-medium",
    }
    if (
        label in top_level_model_labels
        and payload.get("model_id") != expected_model_id
    ):
        errors.append(
            f"{label}: model_id={payload.get('model_id')!r}, expected "
            f"{expected_model_id!r}"
        )

    if label in {"dynamic-build", "static-build"}:
        expected_role = (
            "native-dynamic"
            if label == "dynamic-build"
            else "exact-head-static-split"
        )
        if payload.get("artifact_role") != expected_role:
            errors.append(
                f"{label}: artifact_role must be {expected_role!r}"
            )
        if payload.get("model_revision") != expected_model_revision:
            errors.append(
                f"{label}: model_revision does not match the pinned fixture"
            )
        if label == "dynamic-build":
            command = payload.get("command")
            if (
                not isinstance(command, list)
                or len(command) != 3
                or command[1:] != ["build", expected_model_id]
            ):
                errors.append(
                    "dynamic-build: product build command must contain only "
                    "`trtmc build <model>`"
                )
            if not isinstance(payload.get("runtime_kv_plugin"), Mapping):
                errors.append(
                    "dynamic-build: runtime-KV plugin binding is absent"
                )
        elif payload.get("runtime_kv_plugin") is not None:
            errors.append(
                "static-build: static baseline unexpectedly binds runtime KV"
            )

    if label == "chunk-variant-build":
        qualified_model = payload.get("qualified_model")
        if (
            not isinstance(qualified_model, Mapping)
            or qualified_model.get("model_id") != expected_model_id
            or qualified_model.get("revision") != expected_model_revision
        ):
            errors.append(
                "chunk-variant-build: qualified model tuple does not match "
                "the pinned fixture"
            )
        if payload.get("developer_only") is not True or payload.get("opt_in") != {
            "environment": "TRTMC_DEVELOPER_CHUNK_VARIANT",
            "value": "C/2",
        }:
            errors.append(
                "chunk-variant-build: the explicit developer C/2 contract is "
                "absent"
            )
        runtime_memory = payload.get("runtime_memory")
        if (
            not isinstance(runtime_memory, Mapping)
            or runtime_memory.get("contract_version") != 2
        ):
            errors.append(
                "chunk-variant-build: sealed runtime-memory contract v2 is "
                "absent"
            )
        if payload.get("source_state_unchanged") is not True:
            errors.append(
                "chunk-variant-build: source_state_unchanged is not true"
            )

    if label == "correctness":
        gates = payload.get("qualification_gates")
        if not isinstance(gates, Mapping):
            errors.append("correctness: qualification_gates is absent")
        else:
            missing_gates = sorted(
                set(_CORRECTNESS_PROMOTION_GATES).difference(gates)
            )
            if missing_gates:
                errors.append(
                    "correctness: reviewed canonical promotion gates are "
                    f"missing {missing_gates}"
                )
            errors.extend(
                _boolean_gate_errors(
                    gates,
                    label="correctness.qualification_gates",
                )
            )
        for field in (
            "source_calibration_evidence",
            "all_profile_two_sweep_evidence",
        ):
            evidence = payload.get(field)
            if (
                not isinstance(evidence, Mapping)
                or evidence.get("status") != "passed"
                or evidence.get("passed") is not True
                or evidence.get("base") is not True
                or evidence.get("chunk_variant") is not True
            ):
                errors.append(
                    f"correctness: {field} does not prove base and C/2"
                )
        profile_sweeps = payload.get("module_residency_profile_sweeps")
        if (
            not isinstance(profile_sweeps, Mapping)
            or set(profile_sweeps) != set(_EMBEDDED_CALIBRATION_ROLES)
        ):
            errors.append(
                "correctness: module_residency_profile_sweeps does not "
                "contain exactly base and chunk_variant"
            )
        else:
            for role in _EMBEDDED_CALIBRATION_ROLES:
                sweep = profile_sweeps.get(role)
                errors.extend(
                    _embedded_calibration_source_errors(
                        (
                            sweep.get("source_calibration_evidence")
                            if isinstance(sweep, Mapping)
                            else None
                        ),
                        label=(
                            "correctness.module_residency_profile_sweeps."
                            f"{role}.source_calibration_evidence"
                        ),
                    )
                )

    if label == "policies":
        required = {
            "bundle_unchanged": True,
            "capacity_sweep_passed": True,
            "all_memory_evidence_passed": True,
            "source_state_unchanged": True,
        }
        for field, expected in required.items():
            if payload.get(field) is not expected:
                errors.append(f"policies: {field} is not true")
        replay = payload.get("raw_runner_replay")
        if not isinstance(replay, Mapping) or replay.get("passed") is not True:
            errors.append("policies: raw_runner_replay.passed is not true")

    if label == "soak":
        errors.extend(
            _boolean_gate_errors(
                payload.get("qualification_gates"),
                label="soak.qualification_gates",
            )
        )

    if label == "surfaces":
        for field in (
            "positive_policy_matrix_complete",
            "negative_policy_matrix_complete",
            "positive_surfaces_passed",
            "negative_surfaces_passed",
            "bundle_unchanged",
            "sealed_calibration_replayed",
            "source_state_unchanged",
        ):
            if payload.get(field) is not True:
                errors.append(f"surfaces: {field} is not true")

    if label.startswith("performance-capture-"):
        role = (
            "native-dynamic"
            if "-dynamic-" in label
            else "exact-head-static-split"
        )
        if payload.get("artifact_role") != role:
            errors.append(
                f"{label}: artifact_role must be {role!r}"
            )
        runtime_receipt = payload.get("runtime_memory_receipt")
        if role == "native-dynamic":
            if (
                not isinstance(runtime_receipt, Mapping)
                or runtime_receipt.get("receipt_schema_version") != 4
                or runtime_receipt.get("contract_version") != 2
            ):
                errors.append(
                    f"{label}: runtime-memory receipt v4/contract v2 is absent"
                )
        elif (
            not isinstance(runtime_receipt, Mapping)
            or set(runtime_receipt)
            != {
                "serialized_plan_bytes",
                "resident_weight_bytes",
                "resident_weight_copy_count",
                "weight_streaming_active",
                "measurement_sources",
            }
        ):
            errors.append(
                f"{label}: static baseline accounting receipt is invalid"
            )

    if label == "performance-qualification":
        errors.extend(
            _boolean_gate_errors(
                payload.get("gates"),
                label="performance-qualification.gates",
            )
        )

    if label == "process-isolation":
        errors.extend(
            _boolean_gate_errors(
                payload.get("gates"),
                label="process-isolation.gates",
            )
        )
        if payload.get("source_state_unchanged") is not True:
            errors.append(
                "process-isolation: source_state_unchanged is not true"
            )
    return errors


def inspect_receipt(
    expectation: ReceiptExpectation,
    *,
    tested_sha: str,
    expected_model_id: str,
    expected_model_revision: str,
) -> tuple[dict[str, Any], list[str]]:
    """Read only producer-owned pass fields; never manufacture an absent one."""

    errors: list[str] = []
    path = expectation.path
    try:
        payload = _read_json(path, expectation.label)
    except DynamicMemoryNightlyError as exc:
        return {
            "label": expectation.label,
            "original_path": str(path),
            "uploaded_path": None,
            "sha256": None,
            "schema_version": None,
            "reported_status": None,
            "reported_passed": None,
            "reported_promotion_eligible": None,
            "validation": {"passed": False, "errors": [str(exc)]},
        }, [str(exc)]

    schema = payload.get("schema_version")
    status = payload.get("status") if "status" in payload else None
    passed = payload.get("passed") if "passed" in payload else None
    promotion = (
        payload.get("promotion_eligible")
        if "promotion_eligible" in payload
        else None
    )
    if schema != expectation.schema:
        errors.append(
            f"{expectation.label}: schema_version={schema!r}, "
            f"expected {expectation.schema!r}"
        )
    if expectation.status is not None and status != expectation.status:
        errors.append(
            f"{expectation.label}: status={status!r}, "
            f"expected {expectation.status!r}"
        )
    if expectation.passed is not None and passed is not expectation.passed:
        errors.append(
            f"{expectation.label}: passed={passed!r}, "
            f"expected {expectation.passed!r}"
        )
    if (
        expectation.promotion_eligible is not None
        and promotion is not expectation.promotion_eligible
    ):
        errors.append(
            f"{expectation.label}: promotion_eligible={promotion!r}, "
            f"expected {expectation.promotion_eligible!r}"
        )
    if expectation.fresh_build:
        if payload.get("fresh_build") is not True:
            errors.append(f"{expectation.label}: fresh_build is not true")
        if payload.get("artifact_reused") is not False:
            errors.append(f"{expectation.label}: artifact_reused is not false")
    source_shas = _source_shas(payload)
    if not source_shas:
        errors.append(
            f"{expectation.label}: receipt has no source revision binding"
        )
    elif source_shas != {tested_sha}:
        errors.append(
            f"{expectation.label}: source revisions {sorted(source_shas)!r} "
            f"do not equal tested SHA {tested_sha}"
        )
    errors.extend(
        _receipt_semantic_errors(
            expectation.label,
            payload,
            expected_model_id=expected_model_id,
            expected_model_revision=expected_model_revision,
        )
    )
    entry = {
        "label": expectation.label,
        "original_path": str(path.resolve()),
        "uploaded_path": None,
        "sha256": _sha256(path),
        "schema_version": schema,
        "reported_status": status,
        "reported_passed": passed,
        "reported_promotion_eligible": promotion,
        "validation": {"passed": not errors, "errors": errors},
    }
    return entry, errors


def _run_command(
    command: PlannedCommand,
    *,
    environment: Mapping[str, str],
    log_dir: Path,
) -> subprocess.CompletedProcess[str]:
    command_environment = dict(environment)
    command_environment.update(dict(command.environment))
    completed = subprocess.run(
        command.argv,
        check=False,
        env=command_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _write_json(
        log_dir / f"{command.label}.command.json",
        {
            "argv": list(command.argv),
            "environment_overrides": dict(command.environment),
            "returncode": completed.returncode,
        },
    )
    (log_dir / f"{command.label}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (log_dir / f"{command.label}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    return completed


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise DynamicMemoryNightlyError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _resolve_model_snapshot(
    model: Mapping[str, Any],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DynamicMemoryNightlyError(
            "huggingface_hub is required to validate the pinned offline cache"
        ) from exc
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=str(model["model_id"]),
                revision=str(model["revision"]),
                local_files_only=True,
            )
        ).resolve(strict=True)
        default_snapshot = Path(
            snapshot_download(
                repo_id=str(model["model_id"]),
                local_files_only=True,
            )
        ).resolve(strict=True)
    except Exception as exc:  # noqa: BLE001 - normalize third-party failures
        raise DynamicMemoryNightlyError(
            "pinned model snapshot is absent from the offline HF cache: "
            f"{model['model_id']}@{model['revision']}: {exc}"
        ) from exc
    if default_snapshot != snapshot:
        raise DynamicMemoryNightlyError(
            "the model-only build would resolve a different cached revision: "
            f"{model['model_id']} default={default_snapshot}, pinned={snapshot}"
        )
    config = snapshot / "config.json"
    if not config.is_file() or _sha256(config) != model["config_sha256"]:
        raise DynamicMemoryNightlyError(
            f"pinned snapshot config hash mismatch: {config}"
        )
    return snapshot


def _validate_gpu_topology(
    selectors: Sequence[str], *, expected_name: str
) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise DynamicMemoryNightlyError(
            f"nvidia-smi topology query failed: {completed.stderr.strip()}"
        )
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise DynamicMemoryNightlyError(
                f"nvidia-smi returned malformed row: {line!r}"
            )
        rows.append(
            {"index": fields[0], "name": fields[1], "uuid": fields[2]}
        )
    if len(rows) != 4 or any(row["name"] != expected_name for row in rows):
        raise DynamicMemoryNightlyError(
            "dynamic-memory nightly requires exactly four NVIDIA GB300 GPUs; "
            f"found {rows!r}"
        )
    if len({row["uuid"] for row in rows}) != 4:
        raise DynamicMemoryNightlyError("GPU topology contains duplicate UUIDs")
    by_selector = {
        selector: row
        for row in rows
        for selector in (row["index"], row["uuid"])
    }
    try:
        selected = [by_selector[selector] for selector in selectors]
    except KeyError as exc:
        raise DynamicMemoryNightlyError(
            f"GPU selector is not present in topology: {exc.args[0]!r}"
        ) from exc
    if len({row["uuid"] for row in selected}) != len(selectors):
        raise DynamicMemoryNightlyError(
            "producer, runner, and isolation GPU selectors must be distinct"
        )
    return rows


def _base_environment(
    *,
    repo_root: Path,
    build_dir: Path,
    output_dir: Path,
    producer_gpu: str,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment.pop("_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR", None)
    environment.pop(
        "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_BUILD_IDENTITY",
        None,
    )
    model_dirs = [
        build_dir / "models" / "qwen",
        build_dir / "models" / "llama",
    ]
    previous_library_path = environment.get("LD_LIBRARY_PATH", "")
    library_paths = [str(build_dir), *(str(path) for path in model_dirs)]
    if previous_library_path:
        library_paths.append(previous_library_path)
    environment.update(
        {
            "CUDA_MODULE_LOADING": "LAZY",
            "CUDA_VISIBLE_DEVICES": producer_gpu,
            "CUDA_CACHE_DISABLE": "0",
            "CUDA_CACHE_PATH": str(output_dir / "work" / "cuda-cache"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(repo_root / "python"), str(repo_root))
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(library_paths),
            "TRTMC_BACKEND_DIR": str(build_dir),
            "TRTMC_MODEL_PLUGIN_DIR": os.pathsep.join(
                str(path) for path in model_dirs
            ),
            "TRTMC_TRT_PLUGIN_LIBRARY": str(
                build_dir / "libtrtmc_trt_plugins.so"
            ),
        }
    )
    return environment


def _write_aggregate(
    *,
    output_dir: Path,
    model_key: str,
    model_id: str,
    tested_sha: str,
    workflow_run_id: str,
    workflow_run_attempt: int,
    fixture: Path,
    plan: NightlyPlan | None,
    topology: list[dict[str, str]] | None,
    receipts: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
) -> Path:
    upload_root = output_dir / "receipt-upload"
    gate_path = upload_root / "dynamic-memory-nightly-gate.json"
    payload = {
        "schema_version": GATE_SCHEMA,
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "model_key": model_key,
        "model_id": model_id,
        "model_revision": (
            plan.model_revision if plan is not None else None
        ),
        "tested_sha": tested_sha,
        "workflow_run_id": workflow_run_id,
        "workflow_run_attempt": workflow_run_attempt,
        "fixture": (
            _file_identity(fixture)
            if fixture.is_file()
            else {"path": str(fixture), "size_bytes": None, "sha256": None}
        ),
        "plan_sha256": (
            _canonical_sha(plan.as_json()) if plan is not None else None
        ),
        "gpu_topology": topology,
        "required_receipt_labels": (
            [command.label for command in plan.commands]
            if plan is not None
            else []
        ),
        "receipts": list(receipts),
        "commands": list(commands),
        "errors": list(errors),
    }
    _write_json(gate_path, payload)
    return gate_path


def run_gate(args: argparse.Namespace) -> int:
    """Execute the reviewed producer graph and preserve partial failure evidence."""

    repo_root = args.repo_root.expanduser().resolve()
    build_dir = args.build_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    fixture_path = args.fixture.expanduser().resolve()
    model_key = args.model
    tested_sha = args.expected_tested_sha
    workflow_run_id = args.workflow_run_id
    workflow_run_attempt = args.workflow_run_attempt
    fixture: dict[str, Any] = {}
    model_id = ""
    plan: NightlyPlan | None = None
    topology: list[dict[str, str]] | None = None
    receipts: list[dict[str, Any]] = []
    command_receipts: list[dict[str, Any]] = []
    errors: list[str] = []

    if output_dir.exists() and any(output_dir.iterdir()):
        raise DynamicMemoryNightlyError(
            f"output directory must start absent or empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "receipt-upload" / "receipts").mkdir(
        parents=True, exist_ok=True
    )
    try:
        fixture = load_fixture(fixture_path)
        model = fixture["models"][model_key]
        model_id = str(model["model_id"])
        if os.environ.get("CUDA_MODULE_LOADING") != "LAZY":
            raise DynamicMemoryNightlyError(
                "CUDA_MODULE_LOADING must be LAZY for the reviewed calibration"
            )
        if os.environ.get("LD_PRELOAD"):
            raise DynamicMemoryNightlyError(
                "LD_PRELOAD must be unset for process-isolation evidence"
            )
        head = _git_output(repo_root, "rev-parse", "HEAD^{commit}")
        if head != tested_sha:
            raise DynamicMemoryNightlyError(
                f"checkout HEAD {head} does not equal tested SHA {tested_sha}"
            )
        dirty = _git_output(
            repo_root, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if dirty:
            raise DynamicMemoryNightlyError(
                "dynamic-memory nightly requires a clean exact HEAD"
            )
        cache = build_dir / "CMakeCache.txt"
        if not cache.is_file() or (
            f"CMAKE_HOME_DIRECTORY:INTERNAL={repo_root}"
            not in cache.read_text(encoding="utf-8", errors="strict")
        ):
            raise DynamicMemoryNightlyError(
                "build directory is not configured from the exact checkout"
            )
        model_snapshot = _resolve_model_snapshot(model)
        topology = _validate_gpu_topology(
            (
                args.producer_gpu,
                args.runner_gpu,
                args.isolation_gpu_a,
                args.isolation_gpu_b,
            ),
            expected_name=str(fixture["platform"]["gpu_name"]),
        )
        plan = create_plan(
            repo_root=repo_root,
            build_dir=build_dir,
            python=args.python,
            output_dir=output_dir,
            fixture=fixture,
            model_key=model_key,
            model_snapshot=model_snapshot,
            tested_sha=tested_sha,
            producer_gpu=args.producer_gpu,
            runner_gpu=args.runner_gpu,
            isolation_gpu_a=args.isolation_gpu_a,
            isolation_gpu_b=args.isolation_gpu_b,
        )
        for directory in (
            output_dir / "work" / "dynamic-build",
            output_dir / "work" / "static-build",
            output_dir / "work" / "bundles",
            output_dir / "work" / "receipts",
            output_dir / "work" / "requests",
            output_dir / "work" / "logs",
            output_dir / "work" / "source-state",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for name, request in plan.requests.items():
            _write_json(
                output_dir / "work" / "requests" / f"{name}.json",
                request,
            )
        environment = _base_environment(
            repo_root=repo_root,
            build_dir=build_dir,
            output_dir=output_dir,
            producer_gpu=args.producer_gpu,
        )
        for index, command in enumerate(plan.commands, start=1):
            completed = _run_command(
                command,
                environment=environment,
                log_dir=output_dir / "work" / "logs",
            )
            entry, receipt_errors = inspect_receipt(
                command.receipt,
                tested_sha=tested_sha,
                expected_model_id=plan.model_id,
                expected_model_revision=plan.model_revision,
            )
            if command.receipt.path.is_file():
                uploaded = (
                    output_dir
                    / "receipt-upload"
                    / "receipts"
                    / f"{index:02d}-{command.label}.json"
                )
                shutil.copy2(command.receipt.path, uploaded)
                entry["uploaded_path"] = str(
                    uploaded.relative_to(output_dir / "receipt-upload")
                )
            receipts.append(entry)
            command_entry = {
                "label": command.label,
                "argv": list(command.argv),
                "returncode": completed.returncode,
                "receipt_path": str(command.receipt.path),
            }
            command_receipts.append(command_entry)
            if completed.returncode:
                receipt_errors.insert(
                    0,
                    f"{command.label}: producer exited {completed.returncode}",
                )
            if receipt_errors:
                errors.extend(receipt_errors)
                break
    except (
        DynamicMemoryNightlyError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        errors.append(str(exc))

    if plan is not None:
        observed = {entry["label"] for entry in receipts}
        missing = [
            command.label
            for command in plan.commands
            if command.label not in observed
        ]
        if missing:
            errors.append(f"producer graph did not emit receipts: {missing}")
    gate_path = _write_aggregate(
        output_dir=output_dir,
        model_key=model_key,
        model_id=model_id,
        tested_sha=tested_sha,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
        fixture=fixture_path,
        plan=plan,
        topology=topology,
        receipts=receipts,
        commands=command_receipts,
        errors=errors,
    )
    print(
        json.dumps(
            {
                "status": "passed" if not errors else "failed",
                "gate": str(gate_path),
                "errors": errors,
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


def _expectations_from_gate(
    gate: Mapping[str, Any], root: Path
) -> dict[str, ReceiptExpectation]:
    labels = gate.get("required_receipt_labels")
    if not isinstance(labels, list) or not all(
        isinstance(label, str) and label for label in labels
    ):
        raise DynamicMemoryNightlyError(
            "aggregate required_receipt_labels is invalid"
        )
    entries = gate.get("receipts")
    if not isinstance(entries, list):
        raise DynamicMemoryNightlyError("aggregate receipts must be an array")
    by_label: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise DynamicMemoryNightlyError("aggregate receipt entry is invalid")
        label = entry.get("label")
        if not isinstance(label, str) or label in by_label:
            raise DynamicMemoryNightlyError(
                f"aggregate has invalid/duplicate receipt label: {label!r}"
            )
        by_label[label] = entry
    if sorted(by_label) != sorted(labels):
        raise DynamicMemoryNightlyError(
            "aggregate receipt labels do not exactly cover its producer plan"
        )
    if labels != list(_RECEIPT_CONTRACTS):
        raise DynamicMemoryNightlyError(
            "aggregate producer plan does not exactly match the reviewed "
            "dynamic-memory producer graph"
        )
    expectations: dict[str, ReceiptExpectation] = {}
    for label in labels:
        entry = by_label[label]
        uploaded = entry.get("uploaded_path")
        if (
            not isinstance(uploaded, str)
            or not uploaded
            or Path(uploaded).is_absolute()
            or ".." in Path(uploaded).parts
        ):
            raise DynamicMemoryNightlyError(
                f"aggregate uploaded path is invalid for {label}"
            )
        path = (root / uploaded).resolve(strict=True)
        if root.resolve() not in path.parents:
            raise DynamicMemoryNightlyError(
                f"aggregate uploaded path escapes artifact root: {uploaded}"
            )
        contract = _RECEIPT_CONTRACTS[label]
        expected_entry = {
            "schema_version": contract["schema"],
            "reported_status": contract.get("status"),
            "reported_passed": contract.get("passed"),
            "reported_promotion_eligible": contract.get(
                "promotion_eligible"
            ),
        }
        mismatches = {
            field: {"expected": value, "actual": entry.get(field)}
            for field, value in expected_entry.items()
            if entry.get(field) != value
        }
        if mismatches:
            raise DynamicMemoryNightlyError(
                f"aggregate producer fields are invalid for {label}: "
                + json.dumps(mismatches, sort_keys=True)
            )
        expectations[label] = ReceiptExpectation(
            label=label,
            path=path,
            schema=contract["schema"],
            status=contract.get("status"),
            passed=contract.get("passed"),
            promotion_eligible=contract.get("promotion_eligible"),
            fresh_build=bool(contract.get("fresh_build", False)),
        )
    return expectations


def _verify_gate(
    path: Path,
    *,
    expected_tested_sha: str,
    expected_run_id: str,
    max_attempt: int,
    fixture: Mapping[str, Any],
    fixture_path: Path,
) -> dict[str, Any]:
    gate = _read_json(path, "dynamic-memory aggregate")
    model_key = gate.get("model_key")
    fixture_model = (
        fixture["models"].get(model_key)
        if isinstance(model_key, str)
        else None
    )
    required = {
        "schema_version": GATE_SCHEMA,
        "status": "passed",
        "passed": True,
        "tested_sha": expected_tested_sha,
        "workflow_run_id": expected_run_id,
        "errors": [],
    }
    if isinstance(fixture_model, Mapping):
        required.update(
            {
                "model_id": fixture_model["model_id"],
                "model_revision": fixture_model["revision"],
            }
        )
    mismatches = {
        key: {"expected": value, "actual": gate.get(key)}
        for key, value in required.items()
        if gate.get(key) != value
    }
    attempt = gate.get("workflow_run_attempt")
    if type(attempt) is not int or not (1 <= attempt <= max_attempt):
        mismatches["workflow_run_attempt"] = {
            "expected": f"integer in [1, {max_attempt}]",
            "actual": attempt,
        }
    if model_key not in EXPECTED_MODELS:
        mismatches["model_key"] = {
            "expected": list(EXPECTED_MODELS),
            "actual": model_key,
        }
    fixture_identity = gate.get("fixture")
    if (
        not isinstance(fixture_identity, Mapping)
        or fixture_identity.get("sha256") != _sha256(fixture_path)
        or fixture_identity.get("size_bytes") != fixture_path.stat().st_size
    ):
        mismatches["fixture"] = {
            "expected": {
                "sha256": _sha256(fixture_path),
                "size_bytes": fixture_path.stat().st_size,
            },
            "actual": fixture_identity,
        }
    plan_sha = gate.get("plan_sha256")
    if not isinstance(plan_sha, str) or _SHA256_RE.fullmatch(plan_sha) is None:
        mismatches["plan_sha256"] = {
            "expected": "lowercase SHA-256",
            "actual": plan_sha,
        }
    commands = gate.get("commands")
    if (
        not isinstance(commands, list)
        or [
            command.get("label")
            if isinstance(command, Mapping)
            else None
            for command in commands
        ]
        != list(_RECEIPT_CONTRACTS)
        or any(
            not isinstance(command, Mapping)
            or type(command.get("returncode")) is not int
            or command["returncode"] != 0
            for command in commands
        )
    ):
        mismatches["commands"] = {
            "expected": "complete ordered producer graph with returncode 0",
            "actual": commands,
        }
    topology = gate.get("gpu_topology")
    if (
        not isinstance(topology, list)
        or len(topology) != 4
        or any(
            not isinstance(row, Mapping)
            or row.get("name") != fixture["platform"]["gpu_name"]
            or not isinstance(row.get("uuid"), str)
            or not row["uuid"]
            for row in topology
        )
        or len(
            {
                row["uuid"]
                for row in topology
                if isinstance(row, Mapping) and isinstance(row.get("uuid"), str)
            }
        )
        != 4
    ):
        mismatches["gpu_topology"] = {
            "expected": "four distinct fixture-matching GPUs",
            "actual": topology,
        }
    if mismatches:
        raise DynamicMemoryNightlyError(
            "dynamic-memory aggregate provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    artifact_root = path.parent
    expectations = _expectations_from_gate(gate, artifact_root)
    entries_by_label = {
        str(entry["label"]): entry for entry in gate["receipts"]
    }
    errors: list[str] = []
    for label, expectation in expectations.items():
        entry = entries_by_label[label]
        if entry.get("sha256") != _sha256(expectation.path):
            errors.append(f"{label}: uploaded receipt SHA differs from aggregate")
            continue
        replayed, replay_errors = inspect_receipt(
            expectation,
            tested_sha=expected_tested_sha,
            expected_model_id=str(fixture_model["model_id"]),
            expected_model_revision=str(fixture_model["revision"]),
        )
        for field in (
            "schema_version",
            "reported_status",
            "reported_passed",
            "reported_promotion_eligible",
        ):
            if replayed.get(field) != entry.get(field):
                replay_errors.append(
                    f"{label}: aggregate {field} differs from uploaded receipt"
                )
        if (
            not isinstance(entry.get("validation"), dict)
            or entry["validation"].get("passed") is not True
        ):
            replay_errors.append(
                f"{label}: aggregate does not record passed receipt validation"
            )
        errors.extend(replay_errors)
    if errors:
        raise DynamicMemoryNightlyError(
            "dynamic-memory uploaded receipts failed replay:\n- "
            + "\n- ".join(errors)
        )
    return {
        "model_key": model_key,
        "model_id": gate.get("model_id"),
        "workflow_run_attempt": attempt,
        "aggregate": _file_identity(path),
        "receipt_count": len(expectations),
    }


def verify_artifacts(args: argparse.Namespace) -> int:
    """Select latest per-model attempts and replay all uploaded receipt gates."""

    parts_dir = args.parts_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    fixture_path = args.fixture.expanduser().resolve(strict=True)
    fixture = load_fixture(fixture_path)
    errors: list[str] = []
    candidates: dict[str, dict[int, list[Path]]] = {
        model: {} for model in EXPECTED_MODELS
    }
    if not parts_dir.is_dir():
        errors.append(f"dynamic-memory artifact parts directory is missing: {parts_dir}")
    else:
        for path in sorted(
            parts_dir.rglob("dynamic-memory-nightly-gate.json")
        ):
            try:
                gate = _read_json(path, "dynamic-memory aggregate identity")
                model_key = gate.get("model_key")
                attempt = gate.get("workflow_run_attempt")
                if gate.get("tested_sha") != args.expected_tested_sha:
                    raise DynamicMemoryNightlyError(
                        "aggregate tested SHA does not match this report"
                    )
                if gate.get("workflow_run_id") != args.expected_run_id:
                    raise DynamicMemoryNightlyError(
                        "aggregate workflow run ID does not match this report"
                    )
                if model_key not in EXPECTED_MODELS:
                    raise DynamicMemoryNightlyError(
                        f"aggregate model key is invalid: {model_key!r}"
                    )
                if type(attempt) is not int or not (
                    1 <= attempt <= args.max_attempt
                ):
                    raise DynamicMemoryNightlyError(
                        f"aggregate attempt is outside [1, {args.max_attempt}]"
                    )
            except (DynamicMemoryNightlyError, OSError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
                continue
            candidates[str(model_key)].setdefault(int(attempt), []).append(path)

    selected: list[dict[str, Any]] = []
    for model_key in EXPECTED_MODELS:
        attempts = candidates[model_key]
        if not attempts:
            errors.append(f"no valid dynamic-memory artifact for {model_key}")
            continue
        attempt = max(attempts)
        paths = attempts[attempt]
        if len(paths) != 1:
            errors.append(
                f"duplicate dynamic-memory artifacts for {model_key} "
                f"attempt {attempt}: {len(paths)}"
            )
            continue
        try:
            selected.append(
                _verify_gate(
                    paths[0],
                    expected_tested_sha=args.expected_tested_sha,
                    expected_run_id=args.expected_run_id,
                    max_attempt=args.max_attempt,
                    fixture=fixture,
                    fixture_path=fixture_path,
                )
            )
        except (DynamicMemoryNightlyError, OSError, ValueError) as exc:
            errors.append(f"{paths[0]}: {exc}")
    status = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "tested_sha": args.expected_tested_sha,
        "workflow_run_id": args.expected_run_id,
        "max_attempt": args.max_attempt,
        "expected_models": list(EXPECTED_MODELS),
        "selected_models": [
            row["model_key"] for row in selected
        ],
        "selected": selected,
        "errors": errors,
    }
    _write_json(output, status)
    print(json.dumps(status, sort_keys=True))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--build-dir", type=Path, required=True)
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--fixture", type=Path, required=True)
    run.add_argument("--model", choices=EXPECTED_MODELS, required=True)
    run.add_argument("--expected-tested-sha", required=True)
    run.add_argument("--workflow-run-id", required=True)
    run.add_argument("--workflow-run-attempt", type=int, required=True)
    run.add_argument("--producer-gpu", default="0")
    run.add_argument("--runner-gpu", default="1")
    run.add_argument("--isolation-gpu-a", default="2")
    run.add_argument("--isolation-gpu-b", default="3")
    run.add_argument("--plan", action="store_true")
    run.add_argument(
        "--model-snapshot",
        type=Path,
        help="Only used by --plan; real runs resolve and hash the pinned cache.",
    )

    verify = subparsers.add_parser("verify-artifacts")
    verify.add_argument("--parts-dir", type=Path, required=True)
    verify.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
    )
    verify.add_argument("--expected-tested-sha", required=True)
    verify.add_argument("--expected-run-id", required=True)
    verify.add_argument("--max-attempt", type=int, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "verify-artifacts":
        return verify_artifacts(args)
    if args.workflow_run_attempt <= 0:
        raise DynamicMemoryNightlyError(
            "--workflow-run-attempt must be positive"
        )
    if not args.plan:
        return run_gate(args)
    fixture = load_fixture(args.fixture)
    snapshot = args.model_snapshot or Path(
        f"/pinned-hf-snapshot/{args.model}"
    )
    plan = create_plan(
        repo_root=args.repo_root,
        build_dir=args.build_dir,
        python=args.python,
        output_dir=args.output_dir,
        fixture=fixture,
        model_key=args.model,
        model_snapshot=snapshot,
        tested_sha=args.expected_tested_sha,
        producer_gpu=args.producer_gpu,
        runner_gpu=args.runner_gpu,
        isolation_gpu_a=args.isolation_gpu_a,
        isolation_gpu_b=args.isolation_gpu_b,
    )
    print(json.dumps(plan.as_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DynamicMemoryNightlyError as exc:
        print(f"dynamic_memory_nightly: {exc}", file=sys.stderr)
        raise SystemExit(1)
