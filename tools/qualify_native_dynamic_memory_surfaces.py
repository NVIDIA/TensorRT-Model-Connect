#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove equivalent runtime-memory policy resolution across public surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import qualify_native_dynamic_memory as boundary

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from tensorrt_model_connect.pipeline import (  # noqa: E402
    Pipeline,
    _memory_receipt_from_stderr,
)


RECEIPT_EQUIVALENCE_FIELDS = (
    "receipt_schema_version",
    "contract_version",
    "policy",
    "policy_fraction",
    "requested_kv_bytes",
    "kv_budget_bytes",
    "safety_reserve_bytes",
    "module_residency_reserve_bytes",
    "module_residency_reserve_profile_limit",
    "module_residency_plan_set_sha256",
    "module_residency_evidence_sha256",
    "module_residency_cuda_module_loading_mode",
    "capacity_decision_resident_overhead_bytes",
    "final_non_kv_overhead_delta_bytes",
    "model_context_limit",
    "prefill_chunk_limit",
    "request_context_limit",
    "runtime_kv_capacity_tokens",
    "effective_request_limit",
    "kv_bytes_per_token",
    "serialized_plan_bytes",
    "resident_weight_bytes",
    "resident_weight_copy_count",
    "engine_weight_bytes",
    "weight_streaming_active",
    "context_device_memory_bytes",
    "ordinary_device_input_bytes",
    "ordinary_device_output_bytes",
    "external_device_output_bytes",
    "host_staging_bytes",
    "graph_private_device_bytes",
    "kv_reserved_bytes",
    "kv_committed_bytes",
    "kv_metadata_bytes",
    "backend_owned_cache_input_bytes",
    "backend_owned_cache_output_bytes",
)

POSITIVE_POLICY_CASE_NAMES = frozenset(
    {
        "bytes_plus_u",
        "explicit_auto_plus_u",
        "fraction_plus_u",
        "u_only",
    }
)
NEGATIVE_POLICY_ERRORS = {
    "over_model_context": "model_context_limit_exceeded",
    "conflicting_policy_fields": "conflicting_memory_policy",
}


def _schema_v4_receipt_errors(
    receipt: dict[str, Any],
    *,
    expected_policy: str = "bytes",
    expected_fraction: float = 0.0,
    expected_requested_bytes: int = 2_048,
) -> list[str]:
    """Validate per-process schema-v4 invariants without comparing free memory."""

    errors: list[str] = []

    def typed_int(field: str, *, positive: bool = False) -> int | None:
        value = receipt.get(field)
        if type(value) is not int or (value <= 0 if positive else value < 0):
            qualifier = "positive" if positive else "nonnegative"
            errors.append(f"{field} must be a typed {qualifier} integer")
            return None
        return value

    if (
        type(receipt.get("receipt_schema_version")) is not int
        or receipt.get("receipt_schema_version") != 4
    ):
        errors.append("receipt_schema_version must be integer 4")

    capacity_free = typed_int("capacity_decision_free_bytes", positive=True)
    capacity_total = typed_int("capacity_decision_total_bytes", positive=True)
    capacity_used = typed_int("capacity_decision_device_used_bytes")
    settled_free = typed_int("settled_free_bytes", positive=True)
    settled_total = typed_int("settled_total_bytes", positive=True)
    settled_used = typed_int("settled_device_used_bytes")
    final_free = typed_int("final_free_bytes", positive=True)
    final_total = typed_int("final_total_bytes", positive=True)
    final_used = typed_int("final_device_used_bytes")
    context_overhead = typed_int("context_device_memory_bytes", positive=True)
    graph_private = typed_int("graph_private_device_bytes")
    device_overheads: list[int | None] = []
    for field in (
        "ordinary_device_input_bytes",
        "ordinary_device_output_bytes",
        "external_device_output_bytes",
    ):
        device_overheads.append(typed_int(field))
    resident_overhead = typed_int(
        "capacity_decision_resident_overhead_bytes"
    )
    final_overhead_delta = typed_int("final_non_kv_overhead_delta_bytes")
    kv_budget = typed_int("kv_budget_bytes", positive=True)
    requested_bytes = typed_int(
        "requested_kv_bytes",
        positive=expected_policy == "bytes",
    )
    capacity_tokens = typed_int("runtime_kv_capacity_tokens", positive=True)
    bytes_per_token = typed_int("kv_bytes_per_token", positive=True)
    kv_reserved = typed_int("kv_reserved_bytes", positive=True)
    module_residency_reserve = typed_int(
        "module_residency_reserve_bytes",
        positive=True,
    )
    module_residency_profile = typed_int(
        "module_residency_reserve_profile_limit",
        positive=True,
    )
    for field in (
        "module_residency_plan_set_sha256",
        "module_residency_evidence_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            errors.append(f"{field} must be a lowercase SHA256")
    if receipt.get("module_residency_cuda_module_loading_mode") not in {
        "lazy",
        "eager",
    }:
        errors.append(
            "module_residency_cuda_module_loading_mode must be lazy or eager"
        )
    if (
        module_residency_profile is not None
        and capacity_tokens is not None
        and module_residency_profile < capacity_tokens
    ):
        errors.append(
            "module residency profile limit does not cover runtime KV capacity"
        )
    if (
        context_overhead is not None
        and graph_private is not None
        and resident_overhead is not None
        and final_overhead_delta is not None
        and all(value is not None for value in device_overheads)
    ):
        final_overhead = (
            context_overhead
            + graph_private
            + sum(int(value) for value in device_overheads)
        )
        if final_overhead_delta != max(
            0,
            final_overhead - resident_overhead,
        ):
            errors.append(
                "final non-KV overhead delta does not replay "
                "O(final)-O(resident)"
            )

    if (
        capacity_free is not None
        and capacity_total is not None
        and capacity_used is not None
        and (capacity_free > capacity_total or capacity_used != capacity_total - capacity_free)
    ):
        errors.append("capacity-decision snapshot accounting is inconsistent")
    if (
        settled_free is not None
        and settled_total is not None
        and settled_used is not None
        and (
            settled_free > settled_total
            or settled_used != settled_total - settled_free
            or (capacity_total is not None and settled_total != capacity_total)
        )
    ):
        errors.append("settled snapshot accounting is inconsistent")
    if "settled_snapshot_unavailable_reason" not in receipt:
        errors.append("settled_snapshot_unavailable_reason must be present")
    elif receipt.get("settled_snapshot_unavailable_reason") is not None:
        errors.append("settled snapshot is unavailable")
    if (
        capacity_free is not None
        and capacity_total is not None
        and capacity_used is not None
        and (
            final_free != capacity_free
            or final_total != capacity_total
            or final_used != capacity_used
        )
    ):
        errors.append("deprecated final snapshot must exactly alias capacity decision")
    if receipt.get("policy") != expected_policy:
        errors.append(
            "surface qualification policy mismatch: "
            f"expected {expected_policy}, got {receipt.get('policy')}"
        )
    if (
        type(receipt.get("policy_fraction")) not in {int, float}
        or float(receipt.get("policy_fraction", -1.0)) != expected_fraction
    ):
        errors.append("policy_fraction does not match the requested policy")
    if requested_bytes is not None and requested_bytes != expected_requested_bytes:
        errors.append("requested_kv_bytes does not match the requested policy")
    if (
        expected_policy == "bytes"
        and kv_budget is not None
        and requested_bytes is not None
        and kv_budget != requested_bytes
    ):
        errors.append("bytes policy budget must equal requested bytes")
    if (
        capacity_tokens is not None
        and bytes_per_token is not None
        and kv_reserved is not None
        and kv_reserved != capacity_tokens * bytes_per_token
    ):
        errors.append("KV reservation must equal R times bytes per token")
    if kv_reserved is not None and kv_budget is not None and kv_reserved > kv_budget:
        errors.append("KV reservation exceeds bytes-policy budget")
    if (
        expected_policy == "bytes"
        and kv_reserved is not None
        and module_residency_reserve is not None
        and capacity_free is not None
    ):
        safety_reserve = typed_int("safety_reserve_bytes")
        if (
            safety_reserve is not None
            and final_overhead_delta is not None
            and kv_reserved
            > max(
                0,
                capacity_free
                - safety_reserve
                - module_residency_reserve
                - final_overhead_delta,
            )
        ):
            errors.append(
                "KV reservation exceeds safe memory after schema-v4 reserves"
            )
    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _sealed_runtime_memory_contract(
    bundle: Path,
    header: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen split plan bytes and validate the exact sealed v2 calibration."""

    resolved_header = (
        boundary._read_bundle_header(bundle)
        if header is None
        else header
    )
    spec = boundary._resolve_spec(resolved_header)
    try:
        return dict(
            boundary._sealed_profile_sweep_contract(
                bundle,
                resolved_header,
                expected_model_id=spec.model_id,
                expected_context_limit=spec.context_limit,
                expected_profile_limits=tuple(spec.buckets),
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "surface qualification could not replay the sealed v2 "
            f"module-residency calibration: {exc}"
        ) from exc


def _module_residency_receipt_errors(
    receipt: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    calibration = contract.get("module_residency_calibration")
    if not isinstance(calibration, dict):
        return ["sealed contract has no module_residency_calibration"]
    capacity = receipt.get("runtime_kv_capacity_tokens")
    reserves = calibration.get("profile_reserves")
    if type(capacity) is not int or capacity <= 0 or not isinstance(
        reserves,
        list,
    ):
        return ["cannot select a sealed module-residency reserve row"]
    selected = next(
        (
            row
            for row in reserves
            if isinstance(row, dict)
            and type(row.get("covering_profile_limit")) is int
            and row["covering_profile_limit"] >= capacity
        ),
        None,
    )
    if selected is None:
        return [f"sealed calibration does not cover runtime R={capacity}"]
    expected = {
        "receipt_schema_version": 4,
        "contract_version": 2,
        "module_residency_reserve_bytes": selected.get(
            "cumulative_reserve_bytes"
        ),
        "module_residency_reserve_profile_limit": selected.get(
            "covering_profile_limit"
        ),
        "module_residency_plan_set_sha256": calibration.get(
            "plan_set_sha256"
        ),
        "module_residency_evidence_sha256": calibration.get(
            "evidence_sha256"
        ),
        "module_residency_cuda_module_loading_mode": calibration.get(
            "cuda_module_loading_mode"
        ),
    }
    return [
        f"{field} does not match sealed bundle calibration"
        for field, value in expected.items()
        if receipt.get(field) != value
    ]


def _source_state_snapshot(artifact_dir: Path, *, label: str) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    try:
        relative = artifact_dir.relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        top_level = relative.parts[0] if relative.parts else ""
        if not (top_level == "artifacts" or top_level == "build" or top_level.startswith("build-")):
            raise ValueError(
                "qualification output inside the repository must be under "
                "artifacts/, build/, or build-* so source snapshots exclude it"
            )
    return boundary.source_state_provenance(
        REPO_ROOT,
        Path(__file__),
        artifact_dir,
        label=label,
    )


def apply_source_state_gate(
    report: dict[str, Any],
    source_state_pre: dict[str, Any],
    source_state_post: dict[str, Any],
) -> bool:
    pre_sha = source_state_pre.get("source_state_sha256")
    post_sha = source_state_post.get("source_state_sha256")
    unchanged = bool(
        isinstance(pre_sha, str)
        and pre_sha
        and pre_sha == post_sha
        and source_state_pre.get("git_head") == source_state_post.get("git_head")
    )
    report["source_state_pre"] = source_state_pre
    report["source_state_post"] = source_state_post
    report["source_state_unchanged"] = unchanged
    report["passed"] = bool(report.get("passed") is True and unchanged)
    return unchanged


def _parse_final_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("surface helper produced no JSON")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("surface helper JSON is not an object")
    return payload


def _request_peak_is_complete(receipt: dict[str, Any]) -> bool:
    peak = receipt.get("peak_device_bytes")
    boundaries = receipt.get("peak_device_sample_boundaries")
    return (
        isinstance(peak, int)
        and peak >= 0
        and receipt.get("peak_device_bytes_scope") == "device_wide"
        and isinstance(boundaries, list)
        and "after_runtime_kv_allocation" in boundaries
        and "after_successful_request_completion" in boundaries
        and int(receipt.get("peak_device_sample_count", 0)) >= 2
    )


def compare_surface_receipts(
    surfaces: list[dict[str, Any]],
    *,
    expected_capacity: int = 512,
    expected_policy: str = "bytes",
    expected_fraction: float = 0.0,
    expected_requested_bytes: int = 2_048,
    runtime_memory_contract: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    if not surfaces:
        raise ValueError("surface comparison requires at least one result")
    reference = surfaces[0]
    reference_receipt = reference["runtime_memory_receipt"]
    comparisons: dict[str, Any] = {}
    all_passed = True
    for surface in surfaces:
        receipt = surface["runtime_memory_receipt"]
        equivalence_fields = tuple(
            field
            for field in RECEIPT_EQUIVALENCE_FIELDS
            if expected_policy == "bytes" or field != "kv_budget_bytes"
        )
        mismatches = {
            field: {
                "reference": reference_receipt.get(field),
                "candidate": receipt.get(field),
            }
            for field in equivalence_fields
            if receipt.get(field) != reference_receipt.get(field)
        }
        accepted = surface.get("status") == "accepted"
        capacity_matches = (
            int(receipt.get("runtime_kv_capacity_tokens", 0)) == expected_capacity
        )
        request_peak_complete = _request_peak_is_complete(receipt)
        schema_v4_errors = _schema_v4_receipt_errors(
            receipt,
            expected_policy=expected_policy,
            expected_fraction=expected_fraction,
            expected_requested_bytes=expected_requested_bytes,
        )
        calibration_errors = (
            _module_residency_receipt_errors(
                receipt,
                runtime_memory_contract,
            )
            if runtime_memory_contract is not None
            else []
        )
        passed = (
            accepted
            and capacity_matches
            and request_peak_complete
            and not schema_v4_errors
            and not calibration_errors
            and not mismatches
        )
        comparisons[surface["surface"]] = {
            "accepted": accepted,
            f"resolved_R_is_{expected_capacity}": capacity_matches,
            "request_complete_peak": request_peak_complete,
            "schema_v4_complete": not schema_v4_errors,
            "schema_v4_errors": schema_v4_errors,
            "sealed_calibration_matches": not calibration_errors,
            "sealed_calibration_errors": calibration_errors,
            "receipt_mismatches": mismatches,
            "passed": passed,
        }
        all_passed = all_passed and passed
    return comparisons, all_passed


def positive_policy_cases(kv_bytes: int) -> tuple[dict[str, Any], ...]:
    if type(kv_bytes) is not int or kv_bytes <= 0:
        raise ValueError("positive policy matrix requires positive KV bytes")
    return (
        {
            "name": "bytes_plus_u",
            "helper_policy": "bytes",
            "helper_bytes": kv_bytes,
            "helper_fraction": None,
            "cli_memory": f"{kv_bytes}B",
            "python_memory": kv_bytes,
            "expected_policy": "bytes",
            "expected_fraction": 0.0,
            "expected_requested_bytes": kv_bytes,
            "max_sequence_length": 512,
        },
        {
            "name": "explicit_auto_plus_u",
            "helper_policy": "auto",
            "helper_bytes": None,
            "helper_fraction": None,
            "cli_memory": "auto",
            "python_memory": "auto",
            "expected_policy": "auto",
            "expected_fraction": 0.90,
            "expected_requested_bytes": 0,
            "max_sequence_length": 512,
        },
        {
            "name": "fraction_plus_u",
            "helper_policy": "fraction",
            "helper_bytes": None,
            "helper_fraction": 1.0,
            "cli_memory": "100%",
            "python_memory": "100%",
            "expected_policy": "fraction",
            "expected_fraction": 1.0,
            "expected_requested_bytes": 0,
            "max_sequence_length": 512,
        },
        {
            "name": "u_only",
            "helper_policy": "u_only",
            "helper_bytes": None,
            "helper_fraction": None,
            "cli_memory": None,
            "python_memory": None,
            "expected_policy": "auto",
            "expected_fraction": 0.90,
            "expected_requested_bytes": 0,
            "max_sequence_length": 512,
        },
    )


def negative_policy_cases(
    *,
    model_context_limit: int,
    kv_bytes: int,
) -> tuple[dict[str, Any], ...]:
    if (
        type(model_context_limit) is not int
        or model_context_limit <= 0
        or type(kv_bytes) is not int
        or kv_bytes <= 0
    ):
        raise ValueError("negative policy matrix inputs must be positive integers")
    return (
        {
            "name": "over_model_context",
            "normalized_error": NEGATIVE_POLICY_ERRORS["over_model_context"],
            "helper_policy": "auto",
            "helper_bytes": None,
            "helper_fraction": None,
            "cli_memory_values": ["auto"],
            "python_memory": "auto",
            "max_sequence_length": model_context_limit + 1,
            "error_needles": {
                "cli": ("exceeds the model context limit",),
                "cpp": ("exceeds the model context limit",),
                "cabi": ("exceeds the model context limit",),
                "python": ("exceeds the model context limit",),
            },
        },
        {
            "name": "conflicting_policy_fields",
            "normalized_error": NEGATIVE_POLICY_ERRORS[
                "conflicting_policy_fields"
            ],
            "helper_policy": "conflict",
            "helper_bytes": kv_bytes,
            "helper_fraction": 1.0,
            "cli_memory_values": ["100%", f"{kv_bytes}B"],
            # Python deliberately encodes two mutually exclusive choices in
            # its one policy parameter.  Its typed API makes two independent
            # fields unrepresentable, and the delegated CLI must reject the
            # combined value before pipeline construction.
            "python_memory": f"100%,{kv_bytes}B",
            "max_sequence_length": 512,
            "error_needles": {
                "cli": ("may be specified only once",),
                "cpp": ("zero fraction",),
                "cabi": ("zero fraction",),
                "python": (
                    "--kv-cache-memory expects auto",
                    "trtmc run failed",
                ),
            },
        },
    )


def _normalized_rejection_error(
    *,
    surface: str,
    message: str,
    case: dict[str, Any],
) -> str | None:
    needles = case["error_needles"][surface]
    if not all(needle in message for needle in needles):
        return None
    normalized = case.get("normalized_error")
    return normalized if isinstance(normalized, str) and normalized else None


def validate_rejection_matrix(
    cases: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], bool]:
    expected_surfaces = {"cli", "cpp", "cabi", "python"}
    validations: dict[str, Any] = {}
    all_passed = True
    for case_name, results in cases.items():
        expected_error = NEGATIVE_POLICY_ERRORS.get(case_name)
        surfaces = {result.get("surface") for result in results}
        rows: dict[str, Any] = {}
        case_passed = (
            expected_error is not None
            and surfaces == expected_surfaces
            and len(results) == 4
        )
        normalized_errors = {
            result.get("normalized_error")
            for result in results
            if isinstance(result.get("normalized_error"), str)
        }
        case_passed = case_passed and normalized_errors == {expected_error}
        for result in results:
            surface = str(result.get("surface"))
            passed = bool(
                result.get("status") == "rejected"
                and result.get("returncode", 0) != 0
                and result.get("normalized_error") == expected_error
                and result.get("runtime_memory_receipt_present") is False
                and result.get("request_started") is False
                and result.get("attention_launch_observed") is False
                and isinstance(result.get("message"), str)
                and bool(result["message"])
            )
            rows[surface] = {
                "rejected": result.get("status") == "rejected",
                "normalized_error": result.get("normalized_error"),
                "runtime_memory_receipt_absent": (
                    result.get("runtime_memory_receipt_present") is False
                ),
                "request_not_started": result.get("request_started") is False,
                "attention_launch_count": 0
                if result.get("attention_launch_observed") is False
                else 1,
                "passed": passed,
            }
            case_passed = case_passed and passed
        validations[case_name] = {
            "surfaces": rows,
            "expected_normalized_error": expected_error,
            "normalized_error_consistent": normalized_errors == {expected_error},
            "passed": case_passed,
        }
        all_passed = all_passed and case_passed
    return validations, all_passed


def qualification_gate(
    *,
    policy_matrix: dict[str, Any],
    positive_surfaces_passed: bool,
    rejection_matrix: dict[str, Any],
    negative_surfaces_passed: bool,
    bundle_unchanged: bool,
    sealed_calibration_replayed: bool,
) -> dict[str, bool]:
    positive_complete = set(policy_matrix) == POSITIVE_POLICY_CASE_NAMES
    negative_complete = set(rejection_matrix) == set(NEGATIVE_POLICY_ERRORS)
    return {
        "positive_policy_matrix_complete": positive_complete,
        "negative_policy_matrix_complete": negative_complete,
        "positive_surfaces_passed": positive_surfaces_passed,
        "negative_surfaces_passed": negative_surfaces_passed,
        "bundle_unchanged": bundle_unchanged,
        "sealed_calibration_replayed": sealed_calibration_replayed,
        "passed": bool(
            positive_complete
            and negative_complete
            and positive_surfaces_passed
            and negative_surfaces_passed
            and bundle_unchanged
            and sealed_calibration_replayed
        ),
    }


def _run_helper(
    *,
    surface: str,
    helper: Path,
    bundle: Path,
    policy_case: dict[str, Any],
    backend_dirs: list[Path],
    model_plugin_dirs: list[Path],
    hf_python: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        str(helper),
        "--surface",
        surface,
        "--bundle",
        str(bundle),
        "--policy",
        str(policy_case["helper_policy"]),
        "--max-sequence-length",
        str(policy_case["max_sequence_length"]),
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "2",
    ]
    if policy_case["helper_bytes"] is not None:
        command.extend(["--kv-cache-bytes", str(policy_case["helper_bytes"])])
    if policy_case["helper_fraction"] is not None:
        command.extend(
            ["--kv-cache-fraction", str(policy_case["helper_fraction"])]
        )
    if hf_python:
        command.extend(["--hf-python", hf_python])
    for directory in backend_dirs:
        command.extend(["--backend-dir", str(directory)])
    for directory in model_plugin_dirs:
        command.extend(["--model-plugin-dir", str(directory)])
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = _parse_final_json(completed.stdout)
    label = str(policy_case["name"])
    (output_dir / f"{label}.{surface}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_dir / f"{label}.{surface}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0 or payload.get("status") != "accepted":
        raise RuntimeError(f"{surface}: helper failed ({completed.returncode}): {payload}")
    payload["command"] = command
    return payload


def _run_cli(
    *,
    binary: Path,
    bundle: Path,
    policy_case: dict[str, Any],
    backend_dirs: list[Path],
    model_plugin_dirs: list[Path],
    hf_python: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        str(binary),
        "run",
        str(bundle),
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "2",
        "--greedy",
        "--max-sequence-length",
        str(policy_case["max_sequence_length"]),
    ]
    if policy_case["cli_memory"] is not None:
        command.extend(["--kv-cache-memory", str(policy_case["cli_memory"])])
    if hf_python:
        command.extend(["--hf-python", hf_python])
    for directory in backend_dirs:
        command.extend(["--backend-dir", str(directory)])
    for directory in model_plugin_dirs:
        command.extend(["--model-plugin-dir", str(directory)])
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    receipt = _memory_receipt_from_stderr(completed.stderr)
    label = str(policy_case["name"])
    (output_dir / f"{label}.cli.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_dir / f"{label}.cli.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0 or receipt is None:
        raise RuntimeError(
            f"CLI surface failed ({completed.returncode}): {completed.stderr[-4000:]}"
        )
    return {
        "status": "accepted",
        "surface": "cli",
        "generated_text": completed.stdout.strip(),
        "runtime_memory_receipt": receipt,
        "command": command,
    }


def _run_python(
    *,
    binary: Path,
    bundle: Path,
    policy_case: dict[str, Any],
    model_plugin_dirs: list[Path],
    hf_python: str | None,
) -> dict[str, Any]:
    old_plugin_path = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    try:
        if model_plugin_dirs:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = os.pathsep.join(
                str(path) for path in model_plugin_dirs
            )
        pipeline = Pipeline(
            str(bundle),
            binary=str(binary),
            hf_python=hf_python,
            kv_cache_memory=policy_case["python_memory"],
            max_sequence_length=policy_case["max_sequence_length"],
        )
        generated = pipeline("Hello", max_new_tokens=2)
        receipt = pipeline.last_memory_receipt
    finally:
        if old_plugin_path is None:
            os.environ.pop("TRTMC_MODEL_PLUGIN_DIR", None)
        else:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = old_plugin_path
    if receipt is None:
        raise RuntimeError("Python surface did not parse a runtime receipt")
    return {
        "status": "accepted",
        "surface": "python",
        "generated_text": generated,
        "runtime_memory_receipt": receipt,
        "api_call": {
            "bundle": str(bundle),
            "binary": str(binary),
            "kv_cache_memory": policy_case["python_memory"],
            "max_sequence_length": policy_case["max_sequence_length"],
            "prompt": "Hello",
            "max_new_tokens": 2,
        },
    }


def _run_helper_rejection(
    *,
    surface: str,
    helper: Path,
    bundle: Path,
    case: dict[str, Any],
    backend_dirs: list[Path],
    model_plugin_dirs: list[Path],
    hf_python: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        str(helper),
        "--surface",
        surface,
        "--bundle",
        str(bundle),
        "--policy",
        str(case["helper_policy"]),
        "--max-sequence-length",
        str(case["max_sequence_length"]),
    ]
    if case["helper_bytes"] is not None:
        command.extend(["--kv-cache-bytes", str(case["helper_bytes"])])
    if case["helper_fraction"] is not None:
        command.extend(["--kv-cache-fraction", str(case["helper_fraction"])])
    if hf_python:
        command.extend(["--hf-python", hf_python])
    for directory in backend_dirs:
        command.extend(["--backend-dir", str(directory)])
    for directory in model_plugin_dirs:
        command.extend(["--model-plugin-dir", str(directory)])
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = _parse_final_json(completed.stdout)
    label = str(case["name"])
    (output_dir / f"{label}.{surface}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_dir / f"{label}.{surface}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    message = str(payload.get("message", ""))
    rejected = completed.returncode != 0 and payload.get("status") == "error"
    normalized_error = _normalized_rejection_error(
        surface=surface,
        message=message,
        case=case,
    )
    return {
        "surface": surface,
        "status": "rejected" if rejected else "accepted",
        "normalized_error": normalized_error,
        "message": message,
        "returncode": completed.returncode,
        "runtime_memory_receipt_present": (
            payload.get("runtime_memory_receipt") is not None
        ),
        "request_started": payload.get("request_started") is not False,
        "attention_launch_observed": payload.get("attention_launch_observed")
        is not False,
        "command": command,
    }


def _run_cli_rejection(
    *,
    binary: Path,
    bundle: Path,
    case: dict[str, Any],
    backend_dirs: list[Path],
    model_plugin_dirs: list[Path],
    hf_python: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        str(binary),
        "run",
        str(bundle),
        "--prompt",
        "Hello",
        "--max-new-tokens",
        "2",
        "--greedy",
        "--max-sequence-length",
        str(case["max_sequence_length"]),
    ]
    for value in case["cli_memory_values"]:
        command.extend(["--kv-cache-memory", str(value)])
    if hf_python:
        command.extend(["--hf-python", hf_python])
    for directory in backend_dirs:
        command.extend(["--backend-dir", str(directory)])
    for directory in model_plugin_dirs:
        command.extend(["--model-plugin-dir", str(directory)])
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    label = str(case["name"])
    (output_dir / f"{label}.cli.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output_dir / f"{label}.cli.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    normalized_error = _normalized_rejection_error(
        surface="cli",
        message=completed.stderr,
        case=case,
    )
    receipt = _memory_receipt_from_stderr(completed.stderr)
    pre_request_rejection_proven = bool(
        completed.returncode != 0
        and normalized_error is not None
        and receipt is None
        and not completed.stdout.strip()
    )
    request_started = not pre_request_rejection_proven
    return {
        "surface": "cli",
        "status": "rejected" if completed.returncode != 0 else "accepted",
        "normalized_error": normalized_error,
        "message": completed.stderr.strip(),
        "returncode": completed.returncode,
        "runtime_memory_receipt_present": receipt is not None,
        "request_started": request_started,
        "attention_launch_observed": request_started,
        "command": command,
    }


def _run_python_rejection(
    *,
    binary: Path,
    bundle: Path,
    case: dict[str, Any],
    model_plugin_dirs: list[Path],
    hf_python: str | None,
) -> dict[str, Any]:
    old_plugin_path = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    pipeline: Pipeline | None = None
    message = ""
    rejected = False
    try:
        if model_plugin_dirs:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = os.pathsep.join(
                str(path) for path in model_plugin_dirs
            )
        try:
            pipeline = Pipeline(
                str(bundle),
                binary=str(binary),
                hf_python=hf_python,
                kv_cache_memory=case["python_memory"],
                max_sequence_length=case["max_sequence_length"],
            )
            pipeline("Hello", max_new_tokens=2)
        except Exception as error:
            message = str(error)
            rejected = True
    finally:
        if old_plugin_path is None:
            os.environ.pop("TRTMC_MODEL_PLUGIN_DIR", None)
        else:
            os.environ["TRTMC_MODEL_PLUGIN_DIR"] = old_plugin_path
    normalized_error = _normalized_rejection_error(
        surface="python",
        message=message,
        case=case,
    )
    receipt = pipeline.last_memory_receipt if pipeline is not None else None
    pre_request_rejection_proven = bool(
        rejected and normalized_error is not None and receipt is None
    )
    request_started = not pre_request_rejection_proven
    return {
        "surface": "python",
        "status": "rejected" if rejected else "accepted",
        "normalized_error": normalized_error,
        "message": message,
        "returncode": 1 if rejected else 0,
        "runtime_memory_receipt_present": receipt is not None,
        "request_started": request_started,
        "attention_launch_observed": request_started,
        "api_call": {
            "kv_cache_memory": case["python_memory"],
            "max_sequence_length": case["max_sequence_length"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, action="append", default=[])
    parser.add_argument("--model-plugin-dir", type=Path, action="append", default=[])
    parser.add_argument("--hf-python")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    binary = args.binary.resolve()
    helper = args.helper.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_state_pre = _source_state_snapshot(output_dir, label="pre")
    backend_dirs = [path.resolve() for path in args.backend_dir]
    model_plugin_dirs = [path.resolve() for path in args.model_plugin_dir]

    header = boundary._read_bundle_header(bundle)
    spec = boundary._resolve_spec(header)
    runtime_memory_contract = _sealed_runtime_memory_contract(
        bundle,
        header,
    )
    bytes_per_token = int(header["runtime_memory"]["kv_bytes_per_token"])
    kv_bytes = 512 * bytes_per_token
    bundle_before = _bundle_identity(bundle)

    policy_matrix: dict[str, Any] = {}
    surfaces_passed = True
    for policy_case in positive_policy_cases(kv_bytes):
        case_surfaces = [
            _run_cli(
                binary=binary,
                bundle=bundle,
                policy_case=policy_case,
                backend_dirs=backend_dirs,
                model_plugin_dirs=model_plugin_dirs,
                hf_python=args.hf_python,
                output_dir=output_dir,
            ),
            _run_helper(
                surface="cpp",
                helper=helper,
                bundle=bundle,
                policy_case=policy_case,
                backend_dirs=backend_dirs,
                model_plugin_dirs=model_plugin_dirs,
                hf_python=args.hf_python,
                output_dir=output_dir,
            ),
            _run_helper(
                surface="cabi",
                helper=helper,
                bundle=bundle,
                policy_case=policy_case,
                backend_dirs=backend_dirs,
                model_plugin_dirs=model_plugin_dirs,
                hf_python=args.hf_python,
                output_dir=output_dir,
            ),
            _run_python(
                binary=binary,
                bundle=bundle,
                policy_case=policy_case,
                model_plugin_dirs=model_plugin_dirs,
                hf_python=args.hf_python,
            ),
        ]
        comparisons, case_passed = compare_surface_receipts(
            case_surfaces,
            expected_capacity=int(policy_case["max_sequence_length"]),
            expected_policy=str(policy_case["expected_policy"]),
            expected_fraction=float(policy_case["expected_fraction"]),
            expected_requested_bytes=int(
                policy_case["expected_requested_bytes"]
            ),
            runtime_memory_contract=runtime_memory_contract,
        )
        policy_matrix[str(policy_case["name"])] = {
            "policy": policy_case,
            "surfaces": case_surfaces,
            "comparisons": comparisons,
            "passed": case_passed,
        }
        surfaces_passed = surfaces_passed and case_passed

    rejection_results: dict[str, list[dict[str, Any]]] = {}
    rejection_cases = negative_policy_cases(
        model_context_limit=spec.context_limit,
        kv_bytes=kv_bytes,
    )
    for case in rejection_cases:
        rejection_results[str(case["name"])] = [
            _run_cli_rejection(
                binary=binary,
                bundle=bundle,
                case=case,
                backend_dirs=backend_dirs,
                model_plugin_dirs=model_plugin_dirs,
                hf_python=args.hf_python,
                output_dir=output_dir,
            ),
            _run_helper_rejection(
                surface="cpp",
                helper=helper,
                bundle=bundle,
                case=case,
                backend_dirs=backend_dirs,
                model_plugin_dirs=model_plugin_dirs,
                hf_python=args.hf_python,
                output_dir=output_dir,
            ),
            _run_helper_rejection(
                surface="cabi",
                helper=helper,
                bundle=bundle,
                case=case,
                backend_dirs=backend_dirs,
                model_plugin_dirs=model_plugin_dirs,
                hf_python=args.hf_python,
                output_dir=output_dir,
            ),
            _run_python_rejection(
                binary=binary,
                bundle=bundle,
                case=case,
                model_plugin_dirs=model_plugin_dirs,
                hf_python=args.hf_python,
            ),
        ]
    rejection_validations, rejection_surfaces_passed = (
        validate_rejection_matrix(rejection_results)
    )
    rejection_matrix = {
        str(case["name"]): {
            "policy": case,
            "surfaces": rejection_results[str(case["name"])],
            "validation": rejection_validations[str(case["name"])],
            "passed": rejection_validations[str(case["name"])]["passed"],
        }
        for case in rejection_cases
    }

    bundle_after = _bundle_identity(bundle)
    bundle_unchanged = bundle_before == bundle_after
    gate = qualification_gate(
        policy_matrix=policy_matrix,
        positive_surfaces_passed=surfaces_passed,
        rejection_matrix=rejection_matrix,
        negative_surfaces_passed=rejection_surfaces_passed,
        bundle_unchanged=bundle_unchanged,
        sealed_calibration_replayed=True,
    )
    report = {
        "schema_version": 1,
        "gate": "UX-05",
        "model_id": spec.model_id,
        "binary": {"path": str(binary), "sha256": _sha256(binary)},
        "helper": {"path": str(helper), "sha256": _sha256(helper)},
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "policy_matrix": policy_matrix,
        "rejection_matrix": rejection_matrix,
        "request": {"prompt": "Hello", "max_new_tokens": 2},
        "c_abi_scope_note": (
            "The current versioned C ABI returns IPipeline*; the qualification "
            "uses that documented handle for the positive text request."
        ),
        "receipt_equivalence_fields": list(RECEIPT_EQUIVALENCE_FIELDS),
        "bundle_runtime_memory_contract": runtime_memory_contract,
        "bundle_runtime_memory_contract_sha256": (
            hashlib.sha256(
                json.dumps(
                    runtime_memory_contract,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        ),
        **gate,
        "bundle_before": bundle_before,
        "bundle_after": bundle_after,
    }
    source_state_post = _source_state_snapshot(output_dir, label="post")
    apply_source_state_gate(report, source_state_pre, source_state_post)
    report_path = output_dir / "surface-equivalence-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str(report_path),
                "bundle_sha256": bundle_before["sha256"],
                "policy_cases": sorted(policy_matrix),
                "rejection_cases": sorted(rejection_matrix),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
