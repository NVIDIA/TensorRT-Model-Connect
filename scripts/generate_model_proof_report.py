#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose matrix model-proof artifacts into one fail-closed HTML report.

Each model matrix job uploads one artifact tree containing
``model-proof-status.json``, ``proof.json``, ``selection.json``, and the raw
``e2e`` result directory.  This command validates the exact expected artifact
set, selects the highest workflow attempt independently for each model, and
renders those proof parts with the established E2E report UI.  Validation
errors, including upstream job failures, are reported *after* both the HTML and
machine-readable status have been written.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import generate_e2e_report as e2e_report


_SAFE_MODEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_ARTIFACT_NAME_RE = re.compile(
    r"model-proof-(?P<model>[a-z0-9][a-z0-9._-]*)-"
    r"(?P<revision>[0-9a-f]{40})-(?P<attempt>[1-9][0-9]*)"
)
_UPSTREAM_NAME_RE = re.compile(r"[a-z][a-z0-9_-]*")
_UPSTREAM_RESULT_RE = re.compile(r"[a-z][a-z0-9_-]*")
_SAFE_INFRA_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_GPU_UUID_RE = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_EMBED_BYTES = 32 * 1024 * 1024


def _read_json(path: Path, label: str, issues: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(f"{label} is missing: {path.name}")
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"{label} is invalid: {path.name}: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"{label} must be a JSON object: {path.name}")
        return {}
    return payload


def _parse_expected_models(raw: str, issues: list[str]) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        issues.append(f"expected models JSON is invalid: {exc}")
        return []
    if not isinstance(payload, list):
        issues.append("expected models must be a JSON array")
        return []

    models: list[str] = []
    for index, value in enumerate(payload):
        if not isinstance(value, str) or not _SAFE_MODEL_RE.fullmatch(value):
            issues.append(f"expected models[{index}] is not a safe model identifier: {value!r}")
            continue
        models.append(value)
    if len(models) != len(set(models)):
        duplicates = sorted({model for model in models if models.count(model) > 1})
        issues.append(f"expected models contain duplicates: {duplicates}")
        models = list(dict.fromkeys(models))
    return models


def _parse_expected_cases_by_model(
    raw: str | None,
    expected_models: list[str],
    issues: list[str],
) -> dict[str, list[str]] | None:
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        issues.append(f"expected cases by model JSON is invalid: {exc}")
        return {}
    if not isinstance(payload, dict):
        issues.append("expected cases by model must be a JSON object")
        return {}

    expected_set = set(expected_models)
    valid_keys = {
        key
        for key in payload
        if isinstance(key, str) and _SAFE_MODEL_RE.fullmatch(key) is not None
    }
    invalid_keys = sorted(repr(key) for key in payload if key not in valid_keys)
    if invalid_keys:
        issues.append(f"expected cases by model has invalid owner keys: {invalid_keys}")
    missing = sorted(expected_set - valid_keys)
    unexpected = sorted(valid_keys - expected_set)
    if missing:
        issues.append(f"expected cases by model is missing owners: {missing}")
    if unexpected:
        issues.append(f"expected cases by model has unexpected owners: {unexpected}")

    parsed: dict[str, list[str]] = {}
    for model in expected_models:
        raw_cases = payload.get(model)
        if not isinstance(raw_cases, list) or not raw_cases:
            issues.append(f"{model}: expected nightly cases must be a non-empty array")
            parsed[model] = []
            continue
        names: list[str] = []
        for index, value in enumerate(raw_cases):
            if (
                not isinstance(value, str)
                or not value
                or "/" in value
                or "\\" in value
                or value in {".", ".."}
            ):
                issues.append(
                    f"{model}: expected nightly case {index} is unsafe: {value!r}"
                )
                continue
            names.append(value)
        if names != sorted(names):
            issues.append(f"{model}: expected nightly cases are not sorted")
        if len(names) != len(set(names)):
            issues.append(f"{model}: expected nightly cases contain duplicates")
        parsed[model] = names
    return parsed


def _parse_upstream_results(raw_results: list[str], issues: list[str]) -> dict[str, str]:
    results: dict[str, str] = {}
    for raw in raw_results:
        name, separator, result = raw.partition("=")
        if (
            not separator
            or _UPSTREAM_NAME_RE.fullmatch(name) is None
            or _UPSTREAM_RESULT_RE.fullmatch(result) is None
        ):
            issues.append(f"invalid upstream result declaration: {raw!r}")
            continue
        if name in results:
            issues.append(f"duplicate upstream result declaration: {name}")
            continue
        results[name] = result
        if result != "success":
            issues.append(f"upstream job {name!r} finished with result {result!r}")
    return results


def _artifact_identity(
    status_path: Path,
    parts_dir: Path,
    expected_revision: str,
    issues: list[str],
) -> tuple[str, int, str] | None:
    try:
        relative = status_path.relative_to(parts_dir)
    except ValueError:
        issues.append(f"model-proof status is outside the parts directory: {status_path}")
        return None
    if len(relative.parts) < 2:
        issues.append(
            f"model-proof status has no downloaded artifact directory: {relative.as_posix()}"
        )
        return None
    artifact_name = relative.parts[0]
    match = _ARTIFACT_NAME_RE.fullmatch(artifact_name)
    if match is None:
        issues.append(f"invalid model-proof artifact name: {artifact_name!r}")
        return None
    artifact_revision = match.group("revision")
    if artifact_revision != expected_revision:
        issues.append(
            f"model-proof artifact {artifact_name!r} targets revision "
            f"{artifact_revision!r}, expected {expected_revision!r}"
        )
    return match.group("model"), int(match.group("attempt")), artifact_name


def _path_within(path: Path, root: Path) -> Optional[Path]:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None
    return resolved


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return str(path)


def _selected_case_names(selection: dict[str, Any], model: str, issues: list[str]) -> list[str]:
    raw_cases = selection.get("e2e_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        issues.append(f"{model}: selection has no E2E cases")
        return []
    names: list[str] = []
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            issues.append(f"{model}: selection E2E case {index} is invalid")
            continue
        name = str(case["name"])
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            issues.append(f"{model}: selection has unsafe E2E case name {name!r}")
            continue
        names.append(name)
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        issues.append(f"{model}: selection contains duplicate E2E cases: {duplicates}")
        names = list(dict.fromkeys(names))
    return names


def _raw_result_cases(
    artifacts_root: Path, model: str, issues: list[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    e2e_root = artifacts_root / "e2e"
    if not e2e_root.is_dir():
        issues.append(f"{model}: raw E2E artifact directory is missing")
        return [], []

    names: list[str] = []
    payloads: list[dict[str, Any]] = []
    direct_result_paths = sorted(e2e_root.glob("*/result.json"))
    for result_path in direct_result_paths:
        payload = _read_json(result_path, f"{model}: E2E result", issues)
        if not payload:
            continue
        case_name = payload.get("case_name")
        if not isinstance(case_name, str) or not case_name:
            issues.append(f"{model}: {result_path.parent.name}/result.json has no case_name")
            continue
        if case_name != result_path.parent.name:
            issues.append(
                f"{model}: result case {case_name!r} does not match directory "
                f"{result_path.parent.name!r}"
            )
        if case_name in names:
            issues.append(f"{model}: duplicate result case {case_name!r}")
        names.append(case_name)
        payloads.append(payload)

    nested_results = sorted(set(e2e_root.rglob("result.json")) - set(direct_result_paths))
    if nested_results:
        issues.append(f"{model}: nested E2E result directories are not allowed")
    if not names:
        issues.append(f"{model}: no result.json was produced")
    return names, payloads


def _validate_nightly_diffusion_assessments(
    results: list[dict[str, Any]], revision: str, issues: list[str]
) -> None:
    diffusion_results = [
        result
        for result in results
        if not result.get("_summary_only")
        and isinstance(result.get("case_config"), dict)
        and result["case_config"].get("task_strategy")
        == "diffusion_media_generation"
    ]
    expected_cases = sorted(
        str(result.get("case_name") or "") for result in diffusion_results
    )
    if not expected_cases:
        return
    if any(not case_name for case_name in expected_cases):
        issues.append("nightly diffusion results contain an empty case name")
        return
    if len(expected_cases) != len(set(expected_cases)):
        issues.append(
            f"nightly diffusion results contain duplicate cases: {expected_cases!r}"
        )
        return

    for result in diffusion_results:
        case_name = str(result.get("case_name"))
        assessment = result.get("vlm_assessment")
        if not isinstance(assessment, dict):
            issues.append(f"{case_name}: required nightly diffusion assessment is missing")
            continue
        judgment = assessment.get("vlm_judgment")
        gate = judgment.get("vlm_gate") if isinstance(judgment, dict) else None
        if not isinstance(gate, dict) or gate.get("failed") is not False:
            issues.append(f"{case_name}: nightly diffusion semantic gate is not passing")

        provenance = assessment.get("_assessment_provenance")
        if not isinstance(provenance, dict):
            issues.append(f"{case_name}: diffusion assessment provenance is missing")
            continue
        required = {
            "source_revision": revision,
            "coverage_complete": True,
            "expected_case_names": expected_cases,
            "assessed_case_names": expected_cases,
        }
        for field, expected in required.items():
            if provenance.get(field) != expected:
                issues.append(
                    f"{case_name}: diffusion assessment {field} is "
                    f"{provenance.get(field)!r}, expected {expected!r}"
                )
        run_id = provenance.get("workflow_run_id")
        if not isinstance(run_id, str) or not run_id.isdigit():
            issues.append(f"{case_name}: diffusion assessment workflow_run_id is invalid")
        run_attempt = provenance.get("workflow_run_attempt")
        if (
            not isinstance(run_attempt, int)
            or isinstance(run_attempt, bool)
            or run_attempt < 1
        ):
            issues.append(
                f"{case_name}: diffusion assessment workflow_run_attempt is invalid"
            )


def _check_equal(issues: list[str], model: str, label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        issues.append(f"{model}: {label} must be {expected!r}, found {actual!r}")


def _parse_lease_timestamp(
    value: object,
    *,
    model: str,
    field: str,
    issues: list[str],
) -> datetime.datetime | None:
    if not isinstance(value, str) or not value:
        issues.append(f"{model}: GPU lease {field} is not a non-empty ISO-8601 timestamp")
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        issues.append(f"{model}: GPU lease {field} is not an ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        issues.append(f"{model}: GPU lease {field} has no timezone")
        return None
    return parsed.astimezone(datetime.timezone.utc)


def _validate_lease_evidence(
    lease: dict[str, Any],
    *,
    model: str,
    revision: str,
    run_id: str,
    expected_job_id: str,
    artifact_attempt: int,
    proof: dict[str, Any],
    issues: list[str],
) -> dict[str, Any] | None:
    """Validate one runner-authored lease and return a normalized receipt."""

    initial_issue_count = len(issues)
    expected = {
        "schema_version": 3,
        "model": model,
        "source_revision": revision,
        "run_id": run_id,
        "run_attempt": str(artifact_attempt),
        "job_id": expected_job_id,
        "gpu_id": proof.get("gpu_id"),
        "resource_class": proof.get("gpu_resource_class"),
        "gpu_resource_class": proof.get("gpu_resource_class"),
        "gpu_slot_ids": proof.get("gpu_slot_ids"),
        "gpu_slots_per_device": proof.get("gpu_slots_per_device"),
    }
    for field, wanted in expected.items():
        _check_equal(issues, model, f"GPU lease {field}", lease.get(field), wanted)

    for field in ("job_id", "runner_name", "node_id", "hostname"):
        value = lease.get(field)
        if not isinstance(value, str) or _SAFE_INFRA_ID_RE.fullmatch(value) is None:
            issues.append(
                f"{model}: GPU lease {field} is not a safe non-empty infrastructure identifier"
            )

    gpu_index = lease.get("gpu_index")
    if (
        not isinstance(gpu_index, str)
        or not gpu_index.isdigit()
        or str(int(gpu_index)) != gpu_index
        or lease.get("gpu_id") != gpu_index
    ):
        issues.append(f"{model}: GPU lease gpu_index is not a canonical GPU index")
    gpu_uuid = lease.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or _GPU_UUID_RE.fullmatch(gpu_uuid) is None:
        issues.append(f"{model}: GPU lease gpu_uuid is missing or malformed")

    lock_namespace = lease.get("lock_namespace")
    if not isinstance(lock_namespace, str) or _SHA256_RE.fullmatch(lock_namespace) is None:
        issues.append(f"{model}: GPU lease lock_namespace is missing or malformed")

    slots_per_gpu = lease.get("slots_per_gpu")
    gpu_slots = lease.get("gpu_slots")
    gpu_slot_ids = lease.get("gpu_slot_ids")
    if (
        not isinstance(slots_per_gpu, int)
        or isinstance(slots_per_gpu, bool)
        or not 1 <= slots_per_gpu <= 16
        or lease.get("gpu_slots_per_device") != slots_per_gpu
    ):
        issues.append(f"{model}: GPU lease slots-per-GPU evidence is inconsistent")
    valid_slots = (
        isinstance(gpu_slot_ids, list)
        and bool(gpu_slot_ids)
        and all(isinstance(slot, int) and not isinstance(slot, bool) for slot in gpu_slot_ids)
        and gpu_slot_ids == sorted(set(gpu_slot_ids))
        and isinstance(slots_per_gpu, int)
        and not isinstance(slots_per_gpu, bool)
        and all(0 <= slot < slots_per_gpu for slot in gpu_slot_ids)
        and gpu_slots == gpu_slot_ids
    )
    if not valid_slots:
        issues.append(f"{model}: GPU lease slot evidence is inconsistent")
    elif lease.get("resource_class") == "shared":
        if len(gpu_slot_ids) != 1 or lease.get("gpu_slot") != gpu_slot_ids[0]:
            issues.append(f"{model}: shared GPU lease does not identify exactly one slot")
    elif lease.get("resource_class") == "exclusive_gpu":
        if gpu_slot_ids != list(range(slots_per_gpu)) or lease.get("gpu_slot") is not None:
            issues.append(f"{model}: exclusive GPU lease does not own every slot")
    else:
        issues.append(f"{model}: GPU lease resource_class is unsupported")

    acquired_at = _parse_lease_timestamp(
        lease.get("acquired_at"), model=model, field="acquired_at", issues=issues
    )
    released_at = _parse_lease_timestamp(
        lease.get("released_at"), model=model, field="released_at", issues=issues
    )
    if acquired_at is not None and released_at is not None:
        if released_at < acquired_at:
            issues.append(f"{model}: GPU lease released_at precedes acquired_at")
        elif released_at == acquired_at:
            issues.append(f"{model}: GPU lease has a zero-length ownership interval")

    if len(issues) != initial_issue_count:
        return None
    assert isinstance(gpu_index, str)
    assert isinstance(gpu_uuid, str)
    assert isinstance(lock_namespace, str)
    assert isinstance(gpu_slot_ids, list)
    assert acquired_at is not None and released_at is not None
    return {
        "model": model,
        "run_id": run_id,
        "run_attempt": artifact_attempt,
        "job_id": lease["job_id"],
        "runner_name": lease["runner_name"],
        "node_id": lease["node_id"],
        "hostname": lease["hostname"],
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "gpu_slot_ids": gpu_slot_ids,
        "slots_per_gpu": slots_per_gpu,
        "resource_class": lease["resource_class"],
        "lock_namespace": lock_namespace,
        "acquired_at": acquired_at.isoformat(),
        "released_at": released_at.isoformat(),
    }


def _intervals_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_start = datetime.datetime.fromisoformat(first["acquired_at"])
    first_end = datetime.datetime.fromisoformat(first["released_at"])
    second_start = datetime.datetime.fromisoformat(second["acquired_at"])
    second_end = datetime.datetime.fromisoformat(second["released_at"])
    return first_start < second_end and second_start < first_end


def _validate_lease_set(receipts: list[dict[str, Any]], issues: list[str]) -> None:
    """Reject internally inconsistent topology and overlapping slot ownership."""

    node_identity: dict[str, tuple[str, str]] = {}
    hostname_nodes: dict[str, str] = {}
    runner_identity: dict[str, tuple[str, str, str]] = {}
    gpu_identity: dict[str, tuple[str, str]] = {}
    node_indices: dict[tuple[str, str], str] = {}
    node_slots_per_gpu: dict[str, int] = {}
    for receipt in receipts:
        model = str(receipt["model"])
        node_id = str(receipt["node_id"])
        hostname = str(receipt["hostname"])
        namespace = str(receipt["lock_namespace"])
        gpu_uuid = str(receipt["gpu_uuid"])
        gpu_index = str(receipt["gpu_index"])
        identity = (hostname, namespace)
        previous_identity = node_identity.setdefault(node_id, identity)
        if previous_identity != identity:
            issues.append(f"{model}: node {node_id!r} maps to multiple hostname/lock namespaces")
        previous_node = hostname_nodes.setdefault(hostname, node_id)
        if previous_node != node_id:
            issues.append(f"{model}: hostname {hostname!r} maps to multiple node IDs")
        runner_name = str(receipt["runner_name"])
        physical_identity = (node_id, hostname, namespace)
        previous_runner_identity = runner_identity.setdefault(runner_name, physical_identity)
        if previous_runner_identity != physical_identity:
            issues.append(
                f"{model}: runner {runner_name!r} maps to multiple node/physical identities"
            )
        previous_gpu = gpu_identity.setdefault(gpu_uuid, (node_id, gpu_index))
        if previous_gpu != (node_id, gpu_index):
            issues.append(f"{model}: GPU UUID {gpu_uuid!r} maps to multiple node/index pairs")
        previous_uuid = node_indices.setdefault((node_id, gpu_index), gpu_uuid)
        if previous_uuid != gpu_uuid:
            issues.append(
                f"{model}: node/index {(node_id, gpu_index)!r} maps to multiple GPU UUIDs"
            )
        slots_per_gpu = int(receipt["slots_per_gpu"])
        previous_slots = node_slots_per_gpu.setdefault(node_id, slots_per_gpu)
        if previous_slots != slots_per_gpu:
            issues.append(f"{model}: node {node_id!r} maps to multiple slots-per-GPU values")

    for index, first in enumerate(receipts):
        for second in receipts[index + 1 :]:
            if not _intervals_overlap(first, second):
                continue
            if first["runner_name"] == second["runner_name"]:
                issues.append(
                    "overlapping GPU leases claim the same runner "
                    f"{first['runner_name']!r}: {first['model']!r}, {second['model']!r}"
                )
            same_gpu = (
                first["node_id"] == second["node_id"] and first["gpu_uuid"] == second["gpu_uuid"]
            )
            shared_slots = sorted(set(first["gpu_slot_ids"]).intersection(second["gpu_slot_ids"]))
            if same_gpu and shared_slots:
                issues.append(
                    "overlapping GPU leases duplicate slot ownership for "
                    f"node={first['node_id']!r}, gpu_uuid={first['gpu_uuid']!r}, "
                    f"slots={shared_slots}: {first['model']!r}, {second['model']!r}"
                )


def _missing_context(model: str, revision: str, suite: str, reason: str) -> dict[str, Any]:
    return {
        "model": model,
        "source_revision": revision,
        "suite": suite,
        "outcome": "missing",
        "load_error": reason,
        "diagnostics": {"Aggregation validation": reason},
    }


def _fallback_html(title: str, issues: list[str]) -> str:
    items = "".join(f"<li>{html.escape(issue)}</li>" for issue in issues)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1><h2>Report composition failed</h2>"
        f"<ul>{items}</ul></body></html>"
    )


def compose(args: argparse.Namespace) -> int:
    issues: list[str] = []
    expected_models = _parse_expected_models(args.expected_models, issues)
    expected_cases_by_model = _parse_expected_cases_by_model(
        args.expected_cases_by_model,
        expected_models,
        issues,
    )
    upstream_results = _parse_upstream_results(args.upstream_result, issues)
    expected_result_count = args.expected_result_count
    if args.suite == "nightly":
        if args.expected_cases_by_model is None:
            issues.append("nightly requires expected cases by model")
        if expected_result_count is None:
            issues.append("nightly requires an expected result count")
    if expected_result_count is not None and expected_result_count < 0:
        issues.append("expected result count must be non-negative")
    if expected_cases_by_model is not None and expected_result_count is not None:
        mapped_result_count = sum(
            len(cases) for cases in expected_cases_by_model.values()
        )
        if mapped_result_count != expected_result_count:
            issues.append(
                "expected cases by model contains "
                f"{mapped_result_count} cases; expected result count is "
                f"{expected_result_count}"
            )
    expected_set = set(expected_models)
    parts_dir = args.parts_dir
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        issues.append(f"revision is not a full lowercase Git SHA: {args.revision!r}")
    if not re.fullmatch(r"[1-9][0-9]*", args.run_id):
        issues.append(f"workflow run ID is not a positive decimal integer: {args.run_id!r}")
    if _SAFE_INFRA_ID_RE.fullmatch(args.expected_job_id) is None:
        issues.append(f"expected workflow job ID is unsafe: {args.expected_job_id!r}")
    if not args.project_dir.is_dir():
        issues.append(f"project directory is missing: {args.project_dir}")

    status_paths: list[Path] = []
    if not parts_dir.is_dir():
        issues.append(f"parts directory is missing: {parts_dir}")
    else:
        for candidate in sorted(parts_dir.rglob("model-proof-status.json")):
            resolved = _path_within(candidate, parts_dir)
            if resolved is None or not resolved.is_file():
                issues.append(
                    f"model-proof status path escapes the downloaded parts directory: {candidate}"
                )
                continue
            status_paths.append(resolved)

    # Artifact names are the immutable workflow identity.  Group every attempt
    # from that identity before reading status content so a truncated status
    # from an older, superseded attempt cannot poison a successful retry.
    discovered: dict[str, list[tuple[Path, Path, int, str]]] = {}
    attempt_roots: dict[tuple[str, int], list[str]] = {}
    invalid_status_roots: list[str] = []
    for status_path in status_paths:
        identity = _artifact_identity(status_path, parts_dir, args.revision, issues)
        if identity is None:
            invalid_status_roots.append(_relative_path(status_path.parent, parts_dir))
            continue
        artifact_model, artifact_attempt, artifact_name = identity
        discovered.setdefault(artifact_model, []).append(
            (status_path.parent, status_path, artifact_attempt, artifact_name)
        )
        attempt_roots.setdefault((artifact_model, artifact_attempt), []).append(
            _relative_path(status_path.parent, parts_dir)
        )

    same_attempt_duplicates = [
        {
            "model": model,
            "attempt": attempt,
            "artifact_roots": roots,
        }
        for (model, attempt), roots in sorted(attempt_roots.items())
        if len(roots) != 1
    ]
    for duplicate in same_attempt_duplicates:
        issues.append(
            "duplicate model-proof artifacts for "
            f"{duplicate['model']!r} at attempt {duplicate['attempt']}: "
            f"{duplicate['artifact_roots']}"
        )

    discovered_models = [model for model in expected_models if model in discovered] + sorted(
        set(discovered) - expected_set
    )
    missing_models = sorted(expected_set - set(discovered))
    unexpected_models = sorted(set(discovered) - expected_set)
    duplicate_models = sorted({str(item["model"]) for item in same_attempt_duplicates})
    artifact_attempts = {
        model: sorted({entry[2] for entry in entries})
        for model, entries in sorted(discovered.items())
    }
    if missing_models:
        issues.append(f"missing model-proof artifacts: {missing_models}")
    if unexpected_models:
        issues.append(f"unexpected model-proof artifacts: {unexpected_models}")
    if duplicate_models:
        issues.append(f"duplicate model-proof artifacts: {duplicate_models}")

    proof_contexts: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    certified_result_count = 0
    case_owners: dict[str, str] = {}
    model_entries: list[dict[str, Any]] = []
    lease_receipts: list[dict[str, Any]] = []

    for model in expected_models:
        entries = discovered.get(model, [])
        latest_attempt = max((entry[2] for entry in entries), default=None)
        latest_entries = (
            [entry for entry in entries if entry[2] == latest_attempt]
            if latest_attempt is not None
            else []
        )
        if len(latest_entries) != 1:
            reason = (
                f"expected exactly one artifact root for {model} at its latest "
                f"attempt; found {len(latest_entries)}"
            )
            proof_contexts.append(_missing_context(model, args.revision, args.suite, reason))
            model_entries.append(
                {
                    "model": model,
                    "status": "missing" if not entries else "duplicate",
                    "artifact_root": None,
                    "artifact_attempt": latest_attempt,
                    "artifact_attempts": artifact_attempts.get(model, []),
                    "selected_cases": [],
                    "result_cases": [],
                    "issues": [reason],
                }
            )
            continue

        artifacts_root, status_path, artifact_attempt, artifact_name = latest_entries[0]
        model_issues: list[str] = []
        status = _read_json(status_path, f"{model}: model-proof status", model_issues)
        declared_model = status.get("model")
        if not isinstance(declared_model, str) or not _SAFE_MODEL_RE.fullmatch(
            declared_model
        ):
            model_issues.append(
                f"{model}: selected model-proof status has no safe model identifier"
            )
        elif declared_model != model:
            model_issues.append(
                f"{model}: model-proof artifact {artifact_name!r} declares model "
                f"{declared_model!r}, expected {model!r} from its name"
            )
        proof = _read_json(artifacts_root / "proof.json", f"{model}: proof", model_issues)
        selection = _read_json(
            artifacts_root / "selection.json", f"{model}: selection", model_issues
        )

        _check_equal(model_issues, model, "status model", status.get("model"), model)
        _check_equal(
            model_issues,
            model,
            "status source revision",
            status.get("source_revision"),
            args.revision,
        )
        _check_equal(model_issues, model, "status suite", status.get("suite"), args.suite)
        _check_equal(model_issues, model, "status outcome", status.get("outcome"), "passed")
        _check_equal(model_issues, model, "status exit code", status.get("exit_code"), 0)
        _check_equal(
            model_issues,
            model,
            "validation exit code",
            status.get("validation_exit_code"),
            0,
        )
        _check_equal(
            model_issues,
            model,
            "report exit code",
            status.get("report_exit_code"),
            0,
        )
        if status.get("report_kind") == "workflow_fallback":
            model_issues.append(f"{model}: status is a workflow fallback")

        _check_equal(model_issues, model, "proof model", proof.get("model"), model)
        _check_equal(
            model_issues,
            model,
            "proof source revision",
            proof.get("source_revision"),
            args.revision,
        )
        _check_equal(model_issues, model, "proof suite", proof.get("suite"), args.suite)
        _check_equal(model_issues, model, "proof outcome", proof.get("passed"), True)
        _check_equal(
            model_issues,
            model,
            "selection model",
            selection.get("requested_model"),
            model,
        )
        _check_equal(model_issues, model, "selection suite", selection.get("suite"), args.suite)

        for proof_issue in e2e_report.validate_proof_context(status, proof, selection):
            model_issues.append(f"{model}: {proof_issue}")

        lease_path = artifacts_root / str(proof.get("gpu_lease_evidence") or "")
        lease = _read_json(lease_path, f"{model}: GPU lease evidence", model_issues)
        lease_receipt = _validate_lease_evidence(
            lease,
            model=model,
            revision=args.revision,
            run_id=args.run_id,
            expected_job_id=args.expected_job_id,
            artifact_attempt=artifact_attempt,
            proof=proof,
            issues=model_issues,
        )
        if lease_receipt is not None:
            lease_receipts.append(lease_receipt)

        report_path = artifacts_root / "model-proof-report.html"
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except OSError as exc:
            model_issues.append(f"{model}: standalone HTML report is missing: {exc}")
            report_text = ""
        if not report_text.strip():
            model_issues.append(f"{model}: standalone HTML report is empty")
        if 'data-report-kind="workflow-fallback"' in report_text or (
            "data-report-kind='workflow-fallback'" in report_text
        ):
            model_issues.append(f"{model}: standalone HTML report is a workflow fallback")

        selected_cases = _selected_case_names(selection, model, model_issues)
        result_cases, raw_results = _raw_result_cases(artifacts_root, model, model_issues)
        certified_result_count += len(raw_results)
        if expected_cases_by_model is not None:
            inventory_cases = expected_cases_by_model.get(model, [])
            if selected_cases != inventory_cases:
                model_issues.append(
                    f"{model}: selected E2E cases {selected_cases!r} do not exactly "
                    f"match inventory cases {inventory_cases!r}"
                )
            if result_cases != inventory_cases:
                model_issues.append(
                    f"{model}: result E2E cases {result_cases!r} do not exactly "
                    f"match inventory cases {inventory_cases!r}"
                )
        if set(selected_cases) != set(result_cases) or len(selected_cases) != len(result_cases):
            model_issues.append(
                f"{model}: selected E2E cases {sorted(selected_cases)!r} do not "
                f"exactly match result.json cases {sorted(result_cases)!r}"
            )
        for payload in raw_results:
            case_name = str(payload.get("case_name") or "unknown")
            if payload.get("status") != "pass":
                model_issues.append(
                    f"{model}: E2E result {case_name!r} has status {payload.get('status')!r}"
                )
            previous_owner = case_owners.get(case_name)
            if previous_owner is not None and previous_owner != model:
                collision = (
                    f"E2E result case {case_name!r} is present in both "
                    f"{previous_owner!r} and {model!r}"
                )
                model_issues.append(f"{model}: {collision}")
                issues.append(collision)
            else:
                case_owners[case_name] = model

        model_results = e2e_report.load_all_results(artifacts_root / "e2e")
        loaded_case_names = [
            str(result.get("case_name") or "")
            for result in model_results
            if not result.get("_summary_only")
        ]
        if set(loaded_case_names) != set(result_cases):
            model_issues.append(
                f"{model}: report loader cases {sorted(loaded_case_names)!r} do not "
                f"match raw result cases {sorted(result_cases)!r}"
            )
        for loaded_result in model_results:
            if loaded_result.get("status") != "pass":
                model_issues.append(
                    f"{model}: report result "
                    f"{loaded_result.get('case_name') or 'unknown'!r} has status "
                    f"{loaded_result.get('status')!r} after JUnit reconciliation"
                )
        for evidence_issue in e2e_report.validate_evidence(model_results, args.project_dir):
            model_issues.append(f"{model}: {evidence_issue}")
        all_results.extend(model_results)

        context = e2e_report._proof_context(status, proof, selection)
        diagnostics = e2e_report._load_proof_diagnostics(artifacts_root / "model-proof-status.json")
        if model_issues:
            diagnostics["Aggregation validation"] = "\n".join(model_issues)
            context["outcome"] = "failed"
        if diagnostics:
            context["diagnostics"] = diagnostics
        proof_contexts.append(context)
        issues.extend(model_issues)
        model_entries.append(
            {
                "model": model,
                "status": "failed" if model_issues else "passed",
                "artifact_root": _relative_path(artifacts_root, parts_dir),
                "artifact_name": artifact_name,
                "artifact_attempt": artifact_attempt,
                "artifact_attempts": artifact_attempts.get(model, []),
                "selected_cases": selected_cases,
                "result_cases": result_cases,
                "gpu_lease": lease_receipt,
                "issues": model_issues,
            }
        )

    for model in unexpected_models:
        reason = f"unexpected artifact for model {model}"
        unexpected_entries = discovered[model]
        latest_attempt = max(entry[2] for entry in unexpected_entries)
        latest_entry = next(entry for entry in unexpected_entries if entry[2] == latest_attempt)
        proof_contexts.append(_missing_context(model, args.revision, args.suite, reason))
        model_entries.append(
            {
                "model": model,
                "status": "unexpected",
                "artifact_root": _relative_path(latest_entry[0], parts_dir),
                "artifact_name": latest_entry[3],
                "artifact_attempt": latest_attempt,
                "artifact_attempts": artifact_attempts.get(model, []),
                "selected_cases": [],
                "result_cases": [],
                "issues": [reason],
            }
        )

    _validate_lease_set(lease_receipts, issues)

    if (
        expected_result_count is not None
        and certified_result_count != expected_result_count
    ):
        issues.append(
            "combined report contains "
            f"{certified_result_count} certified E2E results; "
            f"expected exactly {expected_result_count}"
        )

    if args.suite == "nightly":
        _validate_nightly_diffusion_assessments(all_results, args.revision, issues)

    # The per-model path uses 32 MiB so normal audio/video evidence remains
    # visible.  Apply the same bounded limit to the combined report.
    e2e_report._MAX_EMBED_BYTES = _MAX_EMBED_BYTES
    suite_title = "Nightly" if args.suite == "nightly" else "Premerge"
    title = f"{suite_title} Isolated Model Report: {args.revision[:12]}"
    try:
        html_content = e2e_report.render_report(
            all_results,
            title=title,
            project_dir=args.project_dir,
            proof_contexts=proof_contexts,
            evidence_issues=issues,
        )
    except Exception as exc:  # keep a report even if an asset/render path breaks
        issues.append(f"combined HTML rendering failed: {exc}")
        html_content = _fallback_html(title, issues)

    output_written = False
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(html_content, encoding="utf-8")
        output_written = True
    except OSError as exc:
        issues.append(f"combined HTML could not be written: {exc}")

    status_payload = {
        "schema_version": 1,
        "report_kind": "combined_model_proof",
        "outcome": "failed" if issues or not output_written else "passed",
        "source_revision": args.revision,
        "workflow_run_id": args.run_id,
        "workflow_job_id": args.expected_job_id,
        "suite": args.suite,
        "upstream_results": upstream_results,
        "expected_models": expected_models,
        "discovered_models": discovered_models,
        "missing_models": missing_models,
        "unexpected_models": unexpected_models,
        "duplicate_models": duplicate_models,
        "same_attempt_duplicates": same_attempt_duplicates,
        "artifact_attempts": artifact_attempts,
        "invalid_status_roots": invalid_status_roots,
        "expected_count": len(expected_models),
        "discovered_count": len(discovered_models),
        "artifact_count": len(status_paths),
        "selected_artifact_count": sum(
            entry.get("artifact_root") is not None for entry in model_entries
        ),
        "result_count": certified_result_count,
        "gpu_lease_count": len(lease_receipts),
        "gpu_leases": lease_receipts,
        "issue_count": len(issues),
        "issues": issues,
        "report": args.output.name,
        "report_exists": output_written,
        "models": model_entries,
    }
    if expected_result_count is not None:
        status_payload["expected_result_count"] = expected_result_count
    if expected_cases_by_model is not None:
        status_payload["expected_cases_by_model"] = expected_cases_by_model
    status_written = False
    try:
        args.status_output.parent.mkdir(parents=True, exist_ok=True)
        args.status_output.write_text(
            json.dumps(status_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status_written = True
    except OSError as exc:
        print(f"ERROR: combined status could not be written: {exc}", file=sys.stderr)

    for issue in issues:
        print(f"ERROR: {issue}", file=sys.stderr)
    if output_written:
        print(f"Combined model proof report written to {args.output}", file=sys.stderr)
    return 0 if not issues and output_written and status_written else 2


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose isolated model matrix artifacts into one HTML report."
    )
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--expected-models", required=True)
    parser.add_argument(
        "--expected-cases-by-model",
        help="Exact sorted nightly E2E case names keyed by ownership model.",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Exact positive GitHub Actions run ID that produced every lease receipt.",
    )
    parser.add_argument(
        "--expected-job-id",
        required=True,
        help="Exact reusable-workflow job ID that produced every lease receipt.",
    )
    parser.add_argument("--suite", choices=("premerge", "nightly"), required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument(
        "--expected-result-count",
        type=int,
        help="Require exactly this many certified raw E2E result records.",
    )
    parser.add_argument(
        "--upstream-result",
        action="append",
        default=[],
        metavar="JOB=RESULT",
        help="Record an upstream job result; every declared result must be success.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    return compose(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
