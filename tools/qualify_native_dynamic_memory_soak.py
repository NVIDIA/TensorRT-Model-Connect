#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and allocation-slope qualification for native dynamic KV memory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from collections.abc import Mapping
from typing import Any


def _load_boundary_module():
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location("_trtmc_dynamic_boundary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import qualification helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


boundary = _load_boundary_module()
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_RESERVATION_ALIGNMENT_BYTES = 2 * 1024 * 1024
CONTROLLED_PREPLANNING_HEADROOM_BYTES = 32 * CONTROLLED_RESERVATION_ALIGNMENT_BYTES
CONTROLLED_MAX_CORRECTION_ATTEMPTS = 64
CONTROLLED_TARGET_TOLERANCE_ROWS = 19
MEMORY_ATTRIBUTION_FLOOR_BYTES = 64 * 1024 * 1024
COLD_PERSISTENT_UNLISTED_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
_PLAN_BOUND_RESIDENCY_STABLE_RECEIPT_FIELDS = (
    "receipt_schema_version",
    "contract_version",
    "module_residency_reserve_bytes",
    "module_residency_reserve_profile_limit",
    "module_residency_plan_set_sha256",
    "module_residency_evidence_sha256",
    "module_residency_cuda_module_loading_mode",
    "capacity_decision_resident_overhead_bytes",
    "final_non_kv_overhead_delta_bytes",
)
_SAMPLER_FIELDS = {
    "source",
    "pid",
    "cuda_logical_device_index",
    "physical_device_index",
    "pci_bus_id",
    "gpu_uuid",
    "captures_all_compute_processes",
    "device_memory_source",
}
_MEMORY_SAMPLE_FIELDS = {
    "free_bytes",
    "total_bytes",
    "used_bytes",
    "process_used_bytes",
    "all_compute_process_used_bytes",
    "other_compute_process_used_bytes",
    "nvml_device_total_bytes",
    "nvml_device_reserved_bytes",
    "nvml_device_free_bytes",
    "nvml_device_used_bytes",
    "post_nvml_free_bytes",
    "post_nvml_total_bytes",
    "compute_processes",
}
_RUNTIME_PHASES_AFTER_BASELINE = (
    "before runtime KV planning",
    "after shared context and output allocation",
    "after runtime KV allocation",
    "after successful runtime-memory request completion",
)
_QWEN_PRESSURE_MODEL_ID = "Qwen/Qwen3-0.6B"
_TINY_PRESSURE_NOT_APPLICABLE_REASON = (
    "TinyLlama's full 2,048-row KV slab is too small for controlled pressure "
    "on a large-memory GPU to be a deterministic planner test"
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


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


def _sealed_runtime_memory_contract(
    bundle: Path,
    header: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash the actual engine sections and replay sealed contract v2."""

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
            "soak qualification could not replay the sealed v2 "
            f"module-residency calibration: {exc}"
        ) from exc


def _runtime_receipts(value: Any) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            receipt = node.get("runtime_memory_receipt")
            if isinstance(receipt, dict):
                receipts.append(receipt)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for receipt in receipts:
        identity = id(receipt)
        if identity not in seen:
            seen.add(identity)
            unique.append(receipt)
    return unique


def validate_trace_module_residency(
    trace: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Bind every trace receipt to the independently reopened bundle row."""

    calibration = contract.get("module_residency_calibration")
    if not isinstance(calibration, Mapping):
        raise RuntimeError(
            f"{label}: sealed contract has no module-residency calibration"
        )
    reserves = calibration.get("profile_reserves")
    if not isinstance(reserves, list):
        raise RuntimeError(
            f"{label}: sealed calibration has no profile reserve table"
        )
    receipts = _runtime_receipts(trace)
    if not receipts:
        raise RuntimeError(
            f"{label}: trace contains no runtime-memory receipt to bind"
        )
    covered_profiles: set[int] = set()
    for index, receipt in enumerate(receipts):
        capacity = receipt.get("runtime_kv_capacity_tokens")
        if type(capacity) is not int or capacity <= 0:
            raise RuntimeError(
                f"{label}: receipt {index} has no valid runtime KV capacity"
            )
        selected = next(
            (
                row
                for row in reserves
                if isinstance(row, Mapping)
                and type(row.get("covering_profile_limit")) is int
                and int(row["covering_profile_limit"]) >= capacity
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(
                f"{label}: sealed calibration does not cover R={capacity}"
            )
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
        mismatches = {
            field: {"expected": value, "actual": receipt.get(field)}
            for field, value in expected.items()
            if receipt.get(field) != value
        }
        if mismatches:
            raise RuntimeError(
                f"{label}: receipt {index} does not match the sealed "
                f"module-residency calibration: {mismatches}"
            )
        covered_profiles.add(int(selected["covering_profile_limit"]))
    return {
        "receipt_count": len(receipts),
        "covering_profile_limits": sorted(covered_profiles),
        "plan_set_sha256": calibration["plan_set_sha256"],
        "evidence_sha256": calibration["evidence_sha256"],
        "cuda_module_loading_mode": calibration[
            "cuda_module_loading_mode"
        ],
        "passed": True,
    }


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


def controlled_reservation_requirement(
    model_id: str,
    reservation_target_tokens: int | None,
) -> dict[str, Any]:
    """Resolve the model-specific MEM-07 promotion requirement.

    This soak tool is a promotion producer, not a best-effort diagnostic.  Qwen
    must therefore carry the controlled-pressure proof.  TinyLlama records an
    explicit N/A result because allocator tail granularity can exceed its
    entire KV slab on the qualification device.
    """

    if model_id == _QWEN_PRESSURE_MODEL_ID:
        if reservation_target_tokens is None:
            raise ValueError(
                "Qwen full soak qualification requires "
                "--reservation-target-tokens for the MEM-07 controlled "
                "external-reservation proof"
            )
        return {
            "required": True,
            "status": "pending",
            "reservation_target_tokens": reservation_target_tokens,
            "not_applicable_reason": None,
        }
    if reservation_target_tokens is not None:
        raise ValueError(
            "controlled external reservation is assigned only to "
            f"{_QWEN_PRESSURE_MODEL_ID}; omit --reservation-target-tokens "
            f"for {model_id}"
        )
    return {
        "required": False,
        "status": "not_applicable",
        "reservation_target_tokens": None,
        "not_applicable_reason": _TINY_PRESSURE_NOT_APPLICABLE_REASON,
    }


def _run(
    runner: Path,
    bundle: Path,
    token_file: Path,
    logits_file: Path,
    runtime_tokens: int | None,
    *,
    repeat: int = 1,
    load_cycles: int = 1,
    second_runtime_tokens: int | None = None,
    controlled_reservation_target_tokens: int | None = None,
) -> dict[str, Any]:
    command = [
        str(runner),
        "--bundle",
        str(bundle),
        "--tokens",
        str(token_file),
        "--logits",
        str(logits_file),
        "--max-new-tokens",
        "2",
        "--repeat",
        str(repeat),
        "--load-cycles",
        str(load_cycles),
    ]
    if runtime_tokens is not None:
        command.extend(
            [
                "--max-sequence-length",
                str(runtime_tokens),
            ]
        )
    if second_runtime_tokens is not None:
        command.extend(
            [
                "--second-max-sequence-length",
                str(second_runtime_tokens),
            ]
        )
    if controlled_reservation_target_tokens is not None:
        command.extend(
            [
                "--controlled-reservation-target-tokens",
                str(controlled_reservation_target_tokens),
            ]
        )
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = boundary._parse_runner_json(completed.stdout)
    if completed.returncode != 0 or payload.get("status") != "ok":
        raise RuntimeError(
            f"memory qualification runner failed ({completed.returncode}): "
            f"{payload}; stderr={completed.stderr[-4000:]}"
        )
    return payload


def _process_used(sample: Mapping[str, Any]) -> int:
    value = sample.get("process_used_bytes")
    if type(value) is not int or value < 0:
        raise RuntimeError("memory sample has no valid per-process NVML usage")
    return value


def _align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise RuntimeError("controlled reservation alignment inputs are invalid")
    return ((value + alignment - 1) // alignment) * alignment


def _fraction_budget_bytes(fraction: float, available_bytes: int) -> int:
    """Scale bytes by the exact binary64 value, without Python float multiplication."""

    if (
        type(fraction) is not float
        or not math.isfinite(fraction)
        or not 0.0 < fraction <= 1.0
        or type(available_bytes) is not int
        or available_bytes < 0
    ):
        raise RuntimeError("runtime fraction-budget inputs are invalid")
    numerator, denominator = fraction.as_integer_ratio()
    return (available_bytes * numerator) // denominator


def _ceil_divided_by_fraction(bytes_value: int, fraction: float) -> int:
    """Compute ceil(bytes/fraction) from the exact binary64 ratio."""

    if type(bytes_value) is not int or bytes_value < 0:
        raise RuntimeError("runtime inverse-fraction byte input is invalid")
    if type(fraction) is not float or not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise RuntimeError("runtime inverse-fraction input is invalid")
    numerator, denominator = fraction.as_integer_ratio()
    scaled = bytes_value * denominator
    return (scaled + numerator - 1) // numerator


def validate_nvml_sampler(trace: dict[str, Any]) -> dict[str, Any]:
    metadata = trace.get("memory_sampler")
    if not isinstance(metadata, dict) or set(metadata) != _SAMPLER_FIELDS:
        raise RuntimeError("runner output has no memory-sampler provenance")
    pid = metadata.get("pid")
    logical_device = metadata.get("cuda_logical_device_index")
    physical_device = metadata.get("physical_device_index")
    pci_bus_id = metadata.get("pci_bus_id")
    gpu_uuid = metadata.get("gpu_uuid")
    if (
        metadata.get("source") != "nvmlDeviceGetComputeRunningProcesses_v3"
        or metadata.get("captures_all_compute_processes") is not True
        or metadata.get("device_memory_source") != "nvmlDeviceGetMemoryInfo_v2"
        or type(pid) is not int
        or pid <= 0
        or type(logical_device) is not int
        or logical_device < 0
        or type(physical_device) is not int
        or physical_device < 0
        or not isinstance(pci_bus_id, str)
        or re.fullmatch(
            r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",
            pci_bus_id,
        )
        is None
        or not isinstance(gpu_uuid, str)
        or re.fullmatch(r"GPU-[0-9a-fA-F-]{16,}", gpu_uuid) is None
    ):
        raise RuntimeError("memory-sampler provenance has an invalid identity or type")
    return dict(metadata)


def _validate_typed_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise RuntimeError("lifecycle qualification policy is not an object")
    kind = policy.get("kind")
    if kind == "auto":
        valid = policy == {"kind": "auto"}
    elif kind == "fraction":
        fraction = policy.get("requested_fraction")
        valid = (
            set(policy) == {"kind", "requested_fraction"}
            and type(fraction) is float
            and math.isfinite(fraction)
            and 0.0 < fraction <= 1.0
        )
    elif kind == "bytes":
        requested_bytes = policy.get("requested_bytes")
        valid = (
            set(policy) == {"kind", "requested_bytes"}
            and type(requested_bytes) is int
            and requested_bytes > 0
        )
    elif kind == "max_sequence_length":
        requested_tokens = policy.get("requested_tokens")
        valid = (
            set(policy) == {"kind", "requested_tokens"}
            and type(requested_tokens) is int
            and requested_tokens > 0
        )
    else:
        valid = False
    if not valid:
        raise RuntimeError(f"lifecycle qualification policy is invalid: {policy!r}")
    return dict(policy)


def _parse_memory_sample(
    sample: Any,
    *,
    sampler: Mapping[str, Any],
    role: str,
    require_phase: bool = False,
) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise RuntimeError(f"{role} memory sample is not an object")
    expected_fields = set(_MEMORY_SAMPLE_FIELDS)
    if require_phase:
        expected_fields.update({"phase", "device"})
    if set(sample) != expected_fields:
        raise RuntimeError(f"{role} memory sample has an incomplete NVML/CUDA ledger")

    if require_phase:
        phase = sample.get("phase")
        device = sample.get("device")
        if (
            not isinstance(phase, str)
            or not phase
            or type(device) is not int
            or device != sampler["cuda_logical_device_index"]
        ):
            raise RuntimeError(f"{role} phase sample does not match the sampler identity")

    integer_fields = _MEMORY_SAMPLE_FIELDS - {"compute_processes"}
    if any(type(sample.get(field)) is not int for field in integer_fields):
        raise RuntimeError(f"{role} memory sample has an invalid scalar type")
    free_bytes = sample["free_bytes"]
    total_bytes = sample["total_bytes"]
    used_bytes = sample["used_bytes"]
    process_used_bytes = sample["process_used_bytes"]
    all_process_used_bytes = sample["all_compute_process_used_bytes"]
    other_process_used_bytes = sample["other_compute_process_used_bytes"]
    nvml_total_bytes = sample["nvml_device_total_bytes"]
    nvml_reserved_bytes = sample["nvml_device_reserved_bytes"]
    nvml_free_bytes = sample["nvml_device_free_bytes"]
    nvml_used_bytes = sample["nvml_device_used_bytes"]
    post_nvml_free_bytes = sample["post_nvml_free_bytes"]
    post_nvml_total_bytes = sample["post_nvml_total_bytes"]
    processes = sample["compute_processes"]
    if (
        total_bytes <= 0
        or free_bytes < 0
        or free_bytes > total_bytes
        or used_bytes != total_bytes - free_bytes
        or process_used_bytes < 0
        or all_process_used_bytes < process_used_bytes
        or other_process_used_bytes != all_process_used_bytes - process_used_bytes
        or nvml_total_bytes <= 0
        or min(nvml_reserved_bytes, nvml_free_bytes, nvml_used_bytes) < 0
        or nvml_reserved_bytes + nvml_free_bytes + nvml_used_bytes != nvml_total_bytes
        or nvml_total_bytes != total_bytes + nvml_reserved_bytes
        or post_nvml_total_bytes != total_bytes
        or post_nvml_free_bytes < 0
        or post_nvml_free_bytes > post_nvml_total_bytes
        or not isinstance(processes, list)
    ):
        raise RuntimeError(f"{role} memory sample has inconsistent CUDA/NVML accounting")

    process_sum = 0
    current_process_sum = 0
    observed_pids: set[int] = set()
    parsed_processes: list[dict[str, int]] = []
    for process in processes:
        if not isinstance(process, dict) or set(process) != {"pid", "used_bytes"}:
            raise RuntimeError(f"{role} NVML process ledger row is malformed")
        pid = process.get("pid")
        process_bytes = process.get("used_bytes")
        if (
            type(pid) is not int
            or pid <= 0
            or pid in observed_pids
            or type(process_bytes) is not int
            or process_bytes < 0
        ):
            raise RuntimeError(f"{role} NVML process ledger row has an invalid type")
        observed_pids.add(pid)
        process_sum += process_bytes
        if pid == sampler["pid"]:
            current_process_sum += process_bytes
        parsed_processes.append({"pid": pid, "used_bytes": process_bytes})
    if sampler["pid"] not in observed_pids:
        raise RuntimeError(f"{role} NVML process ledger does not contain the sampler PID")
    if process_sum != all_process_used_bytes or current_process_sum != process_used_bytes:
        raise RuntimeError(f"{role} NVML process ledger disagrees with aggregate fields")

    parsed = dict(sample)
    parsed["compute_processes"] = parsed_processes
    return parsed


def _signed_memory_attribution(
    baseline: Mapping[str, Any],
    boundary_sample: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline["total_bytes"] != boundary_sample["total_bytes"]:
        raise RuntimeError("memory attribution samples disagree on CUDA device total")
    device_growth = int(baseline["free_bytes"]) - int(boundary_sample["free_bytes"])
    process_growth = int(boundary_sample["process_used_bytes"]) - int(
        baseline["process_used_bytes"]
    )
    visible_other_growth = int(boundary_sample["other_compute_process_used_bytes"]) - int(
        baseline["other_compute_process_used_bytes"]
    )
    baseline_non_current = int(baseline["nvml_device_used_bytes"]) - int(
        baseline["process_used_bytes"]
    )
    boundary_non_current = int(boundary_sample["nvml_device_used_bytes"]) - int(
        boundary_sample["process_used_bytes"]
    )
    external_growth = boundary_non_current - baseline_non_current
    unexplained_growth = device_growth - process_growth - external_growth
    unlisted_external_growth = external_growth - visible_other_growth
    baseline_bracket = abs(int(baseline["post_nvml_free_bytes"]) - int(baseline["free_bytes"]))
    boundary_bracket = abs(
        int(boundary_sample["post_nvml_free_bytes"]) - int(boundary_sample["free_bytes"])
    )
    tolerance = max(
        MEMORY_ATTRIBUTION_FLOOR_BYTES,
        math.ceil(
            0.02
            * max(
                abs(device_growth),
                abs(process_growth),
                abs(external_growth),
                1,
            )
        ),
    )
    components_passed = bool(
        baseline_bracket <= tolerance
        and boundary_bracket <= tolerance
        and abs(unexplained_growth) <= tolerance
    )
    unlisted_passed = abs(unlisted_external_growth) <= tolerance
    return {
        "cuda_device_growth_bytes": device_growth,
        "nvml_current_process_growth_bytes": process_growth,
        "nvml_visible_other_process_growth_bytes": visible_other_growth,
        "nvml_non_current_device_growth_bytes": external_growth,
        "unlisted_external_growth_bytes": unlisted_external_growth,
        "unexplained_growth_bytes": unexplained_growth,
        "baseline_cuda_nvml_bracket_difference_bytes": baseline_bracket,
        "boundary_cuda_nvml_bracket_difference_bytes": boundary_bracket,
        "tolerance_bytes": tolerance,
        "tolerance_rule": "max(64MiB,2pct)",
        "reconciliation_formula": "U = D - P - X",
        "components_passed": components_passed,
        "unlisted_external_passed": unlisted_passed,
        "passed": bool(components_passed and unlisted_passed),
    }


def _independent_unlisted_tolerance_bytes(*signed_deltas: int) -> int:
    if not signed_deltas or any(type(value) is not int for value in signed_deltas):
        raise RuntimeError("unlisted-delta tolerance inputs are invalid")
    largest_absolute_delta = max(abs(value) for value in signed_deltas)
    return max(
        MEMORY_ATTRIBUTION_FLOOR_BYTES,
        math.ceil(0.02 * max(1, largest_absolute_delta)),
    )


def _validate_positive_growth_envelope(
    baseline: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    label: str,
    positive_limit_bytes: int,
    allow_process_release: bool = False,
) -> dict[str, Any]:
    if type(positive_limit_bytes) is not int or positive_limit_bytes < 0:
        raise RuntimeError(f"{label} has an invalid positive-growth limit")
    attribution = _signed_memory_attribution(baseline, sample)
    process_delta = int(attribution["nvml_current_process_growth_bytes"])
    device_delta = int(attribution["cuda_device_growth_bytes"])
    visible_other_delta = int(attribution["nvml_visible_other_process_growth_bytes"])
    unlisted_delta = int(attribution["unlisted_external_growth_bytes"])
    ordinary_unlisted_limit = _independent_unlisted_tolerance_bytes(
        unlisted_delta,
    )
    device_positive_limit = positive_limit_bytes
    unlisted_positive_limit = ordinary_unlisted_limit
    process_passed = bool(
        process_delta <= positive_limit_bytes
        if allow_process_release
        else abs(process_delta) <= positive_limit_bytes
    )
    passed = bool(
        attribution["components_passed"]
        and process_passed
        and device_delta <= device_positive_limit
        and unlisted_delta <= unlisted_positive_limit
    )
    result = {
        "label": label,
        "process_signed_delta_bytes": process_delta,
        "device_wide_signed_delta_bytes": device_delta,
        "visible_other_process_signed_delta_bytes": visible_other_delta,
        "unlisted_external_signed_delta_bytes": unlisted_delta,
        "positive_process_limit_bytes": positive_limit_bytes,
        "positive_device_limit_bytes": device_positive_limit,
        "positive_unlisted_limit_bytes": unlisted_positive_limit,
        "negative_external_release_allowed": True,
        "current_process_release_allowed": allow_process_release,
        "process_gate_passed": process_passed,
        "signed_attribution": attribution,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"{label} exceeded its fixed positive-growth envelope: {result}")
    return result


def validate_receipt(
    trace: dict[str, Any],
    expected_r: int,
    *,
    expected_policy: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    receipt = trace.get("runtime_memory_receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("runner output has no runtime memory receipt")
    required = (
        "receipt_schema_version",
        "contract_version",
        "policy",
        "policy_fraction",
        "requested_kv_bytes",
        "safety_reserve_bytes",
        "module_residency_reserve_bytes",
        "module_residency_reserve_profile_limit",
        "module_residency_plan_set_sha256",
        "module_residency_evidence_sha256",
        "module_residency_cuda_module_loading_mode",
        "model_context_limit",
        "request_context_limit",
        "effective_request_limit",
        "kv_budget_bytes",
        "serialized_plan_bytes",
        "resident_weight_bytes",
        "resident_weight_copy_count",
        "engine_weight_bytes",
        "capacity_decision_free_bytes",
        "capacity_decision_total_bytes",
        "capacity_decision_device_used_bytes",
        "capacity_decision_resident_overhead_bytes",
        "final_non_kv_overhead_delta_bytes",
        "settled_free_bytes",
        "settled_total_bytes",
        "settled_device_used_bytes",
        "settled_snapshot_unavailable_reason",
        "final_free_bytes",
        "final_total_bytes",
        "final_device_used_bytes",
        "context_device_memory_bytes",
        "ordinary_device_input_bytes",
        "ordinary_device_output_bytes",
        "external_device_output_bytes",
        "host_staging_bytes",
        "graph_private_device_bytes",
        "kv_reserved_bytes",
        "kv_committed_bytes",
        "kv_metadata_bytes",
        "peak_device_bytes",
        "backend_owned_cache_input_bytes",
        "backend_owned_cache_output_bytes",
        "kv_allocation_id",
        "kv_bytes_per_token",
        "runtime_kv_capacity_tokens",
    )
    missing = [field for field in required if field not in receipt]
    if missing:
        raise RuntimeError(f"runtime memory receipt misses fields: {missing}")

    typed_nonnegative = (
        "requested_kv_bytes",
        "safety_reserve_bytes",
        "serialized_plan_bytes",
        "resident_weight_bytes",
        "engine_weight_bytes",
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
        "capacity_decision_resident_overhead_bytes",
        "final_non_kv_overhead_delta_bytes",
    )
    invalid_nonnegative = [
        field
        for field in typed_nonnegative
        if type(receipt[field]) is not int or receipt[field] < 0
    ]
    if invalid_nonnegative:
        raise RuntimeError(
            f"runtime memory receipt has invalid typed nonnegative fields: {invalid_nonnegative}"
        )
    if (
        type(receipt["receipt_schema_version"]) is not int
        or receipt["receipt_schema_version"] != 4
        or type(receipt["contract_version"]) is not int
        or receipt["contract_version"] != 2
        or type(receipt["resident_weight_copy_count"]) is not int
        or receipt["resident_weight_copy_count"] <= 0
        or type(receipt["kv_allocation_id"]) is not int
        or receipt["kv_allocation_id"] <= 0
        or type(receipt["model_context_limit"]) is not int
        or receipt["model_context_limit"] <= 0
        or type(receipt["request_context_limit"]) is not int
        or receipt["request_context_limit"] < 0
        or type(receipt["effective_request_limit"]) is not int
        or receipt["effective_request_limit"] <= 0
        or type(receipt["runtime_kv_capacity_tokens"]) is not int
        or receipt["runtime_kv_capacity_tokens"] <= 0
        or type(receipt["kv_bytes_per_token"]) is not int
        or receipt["kv_bytes_per_token"] <= 0
        or type(receipt["kv_budget_bytes"]) is not int
        or receipt["kv_budget_bytes"] <= 0
        or type(receipt["module_residency_reserve_bytes"]) is not int
        or receipt["module_residency_reserve_bytes"] <= 0
        or type(receipt["module_residency_reserve_profile_limit"]) is not int
        or receipt["module_residency_reserve_profile_limit"]
        < receipt["runtime_kv_capacity_tokens"]
        or receipt["module_residency_reserve_profile_limit"]
        > receipt["model_context_limit"]
        or not isinstance(receipt["module_residency_plan_set_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            receipt["module_residency_plan_set_sha256"],
        )
        is None
        or not isinstance(receipt["module_residency_evidence_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            receipt["module_residency_evidence_sha256"],
        )
        is None
        or receipt["module_residency_cuda_module_loading_mode"]
        not in {"lazy", "eager"}
    ):
        raise RuntimeError(
            "runtime memory receipt does not use the complete typed schema-v4 contract"
        )
    final_non_kv_overhead = sum(
        int(receipt[field])
        for field in (
            "context_device_memory_bytes",
            "ordinary_device_input_bytes",
            "ordinary_device_output_bytes",
            "external_device_output_bytes",
            "graph_private_device_bytes",
        )
    )
    expected_final_non_kv_overhead_delta = max(
        0,
        final_non_kv_overhead
        - int(receipt["capacity_decision_resident_overhead_bytes"]),
    )
    if (
        receipt["final_non_kv_overhead_delta_bytes"]
        != expected_final_non_kv_overhead_delta
    ):
        raise RuntimeError(
            "runtime memory receipt final non-KV overhead delta does not "
            "replay O(final)-O(resident)"
        )

    capacity_free = receipt["capacity_decision_free_bytes"]
    capacity_total = receipt["capacity_decision_total_bytes"]
    capacity_used = receipt["capacity_decision_device_used_bytes"]
    settled_free = receipt["settled_free_bytes"]
    settled_total = receipt["settled_total_bytes"]
    settled_used = receipt["settled_device_used_bytes"]
    if (
        type(capacity_free) is not int
        or capacity_free <= 0
        or type(capacity_total) is not int
        or capacity_total <= 0
        or capacity_free > capacity_total
        or type(capacity_used) is not int
        or capacity_used != capacity_total - capacity_free
        or type(settled_free) is not int
        or settled_free <= 0
        or type(settled_total) is not int
        or settled_total != capacity_total
        or settled_free > settled_total
        or type(settled_used) is not int
        or settled_used != settled_total - settled_free
        or receipt["settled_snapshot_unavailable_reason"] is not None
    ):
        raise RuntimeError(
            "runtime memory receipt has no valid capacity-decision and settled snapshots"
        )
    if (
        type(receipt["final_free_bytes"]) is not int
        or receipt["final_free_bytes"] != capacity_free
        or type(receipt["final_total_bytes"]) is not int
        or receipt["final_total_bytes"] != capacity_total
        or type(receipt["final_device_used_bytes"]) is not int
        or receipt["final_device_used_bytes"] != capacity_used
    ):
        raise RuntimeError("deprecated final snapshot is not an exact capacity-decision alias")

    r = receipt["runtime_kv_capacity_tokens"]
    b = receipt["kv_bytes_per_token"]
    reserved = receipt["kv_reserved_bytes"]
    if r != expected_r:
        raise RuntimeError(f"runtime allocated R={r}, expected exactly {expected_r}")
    if b <= 0 or reserved != r * b:
        raise RuntimeError(f"KV reservation is not contiguous R*B: R={r}, B={b}, bytes={reserved}")
    if reserved > receipt["kv_budget_bytes"]:
        raise RuntimeError(
            "runtime R exceeds the capacity-decision policy budget and "
            "conservative monotonic solve ceiling"
        )
    exact_zero = (
        "kv_metadata_bytes",
        "backend_owned_cache_input_bytes",
        "backend_owned_cache_output_bytes",
    )
    for field in exact_zero:
        if receipt[field] != 0:
            raise RuntimeError(f"contiguous runtime requires {field}=0")
    if receipt["kv_committed_bytes"] != reserved:
        raise RuntimeError("KV committed bytes differ from reserved bytes")

    if expected_policy is not None:
        policy = _validate_typed_policy(dict(expected_policy))
        kind = policy["kind"]
        expected_receipt_policy = "auto" if kind == "max_sequence_length" else kind
        expected_fraction = (
            float(policy["requested_fraction"])
            if kind == "fraction"
            else 0.0
            if kind == "bytes"
            else 0.9
        )
        expected_requested_bytes = int(policy["requested_bytes"]) if kind == "bytes" else 0
        expected_request_limit = (
            int(policy["requested_tokens"]) if kind == "max_sequence_length" else 0
        )
        fraction = receipt["policy_fraction"]
        if (
            receipt["policy"] != expected_receipt_policy
            or type(fraction) not in {int, float}
            or not math.isfinite(float(fraction))
            or float(fraction) != expected_fraction
            or receipt["requested_kv_bytes"] != expected_requested_bytes
            or receipt["request_context_limit"] != expected_request_limit
            or receipt["effective_request_limit"] != r
        ):
            raise RuntimeError("runtime memory receipt does not bind the expected typed policy")
        safely_available = max(
            0,
            capacity_free
            - receipt["safety_reserve_bytes"]
            - receipt["module_residency_reserve_bytes"]
            - receipt["final_non_kv_overhead_delta_bytes"],
        )
        if kind == "bytes":
            expected_budget = expected_requested_bytes
            if reserved > safely_available:
                raise RuntimeError("explicit byte policy exceeds capacity-decision free memory")
        else:
            expected_budget = _fraction_budget_bytes(
                expected_fraction,
                safely_available,
            )
        if receipt["kv_budget_bytes"] != expected_budget:
            raise RuntimeError("KV budget does not resolve from the capacity-decision snapshot")
        semantic_limit = min(
            receipt["model_context_limit"],
            expected_request_limit if expected_request_limit else receipt["model_context_limit"],
        )
        conservative_capacity_ceiling = min(
            semantic_limit,
            expected_budget // b,
        )
        if r > conservative_capacity_ceiling:
            raise RuntimeError(
                "runtime R exceeds the capacity-decision policy budget's "
                "conservative monotonic solve ceiling"
            )
    return {
        "R": r,
        "B": b,
        "kv_reserved_bytes": reserved,
        "kv_allocation_id": int(receipt["kv_allocation_id"]),
    }


def _validate_receipt_phase_binding(
    receipt: Mapping[str, Any],
    phase_samples: list[dict[str, Any]],
    *,
    role: str,
) -> None:
    """Bind schema-v4 snapshots to their synchronized runtime phase samples."""

    expected_snapshot_fields = (
        ("pre_load_free_bytes", phase_samples[0]["free_bytes"]),
        ("pre_load_total_bytes", phase_samples[0]["total_bytes"]),
        ("post_load_free_bytes", phase_samples[1]["free_bytes"]),
        ("post_load_total_bytes", phase_samples[1]["total_bytes"]),
        ("capacity_decision_free_bytes", phase_samples[2]["free_bytes"]),
        ("capacity_decision_total_bytes", phase_samples[2]["total_bytes"]),
        ("capacity_decision_device_used_bytes", phase_samples[2]["used_bytes"]),
        ("final_free_bytes", phase_samples[2]["free_bytes"]),
        ("final_total_bytes", phase_samples[2]["total_bytes"]),
        ("final_device_used_bytes", phase_samples[2]["used_bytes"]),
        ("settled_free_bytes", phase_samples[3]["free_bytes"]),
        ("settled_total_bytes", phase_samples[3]["total_bytes"]),
        ("settled_device_used_bytes", phase_samples[3]["used_bytes"]),
    )
    for field, expected in expected_snapshot_fields:
        if receipt.get(field) != expected:
            raise RuntimeError(
                f"{role} receipt {field} does not bind its synchronized runtime phase"
            )
    if receipt.get("settled_snapshot_unavailable_reason") is not None:
        raise RuntimeError(f"{role} receipt has no qualified settled memory snapshot")


def _validate_runtime_phase_samples(
    lifetime: Mapping[str, Any],
    *,
    sampler: Mapping[str, Any],
    role: str,
) -> list[dict[str, Any]]:
    samples = lifetime.get("runtime_phase_memory_samples")
    if not isinstance(samples, list) or len(samples) != 5:
        raise RuntimeError(f"{role} lifetime requires exactly five runtime phase samples")
    parsed = [
        _parse_memory_sample(
            sample,
            sampler=sampler,
            role=f"{role} runtime phase {index}",
            require_phase=True,
        )
        for index, sample in enumerate(samples)
    ]
    baseline_phase = parsed[0]["phase"]
    phases = tuple(sample["phase"] for sample in parsed[1:])
    if (
        not baseline_phase.startswith("before runtime-memory ")
        or not baseline_phase.endswith(" engine deserialization")
        or phases != _RUNTIME_PHASES_AFTER_BASELINE
    ):
        raise RuntimeError(f"{role} lifetime runtime phases are incomplete or out of order")
    totals = {sample["total_bytes"] for sample in parsed}
    if len(totals) != 1:
        raise RuntimeError(f"{role} lifetime runtime phases changed CUDA device identity")
    return parsed


def _validate_lifetime_memory(
    lifetime: Mapping[str, Any],
    *,
    sampler: Mapping[str, Any],
    role: str,
    expected_policy: Mapping[str, Any],
    expected_r: int,
    cold_start: bool,
    cold_driver_budget_bytes: int = 0,
    expected_markers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(lifetime, Mapping):
        raise RuntimeError(f"{role} lifetime is not an object")
    if expected_markers is not None:
        for field, expected in expected_markers.items():
            actual = lifetime.get(field)
            if type(expected) is bool:
                matches = actual is expected
            elif type(expected) is int:
                matches = type(actual) is int and actual == expected
            else:
                matches = actual == expected
            if not matches:
                raise RuntimeError(f"{role} lifetime {field}={actual!r}, expected {expected!r}")

    policy = _validate_typed_policy(lifetime.get("policy"))
    if policy != dict(expected_policy):
        raise RuntimeError(f"{role} lifetime used the wrong typed policy")
    capacity = lifetime.get("runtime_kv_capacity_tokens")
    if type(capacity) is not int or capacity != expected_r:
        raise RuntimeError(f"{role} lifetime used the wrong runtime R")
    phase_samples = _validate_runtime_phase_samples(
        lifetime,
        sampler=sampler,
        role=role,
    )
    receipt = validate_receipt(
        dict(lifetime),
        expected_r,
        expected_policy=expected_policy,
    )
    allocation_id = lifetime.get("kv_allocation_id")
    if (
        type(allocation_id) is not int
        or allocation_id <= 0
        or allocation_id != receipt["kv_allocation_id"]
    ):
        raise RuntimeError(f"{role} lifetime allocation identity is inconsistent")
    raw_receipt = lifetime.get("runtime_memory_receipt")
    if not isinstance(raw_receipt, Mapping) or raw_receipt.get("policy") != (
        "auto" if expected_policy["kind"] == "max_sequence_length" else expected_policy["kind"]
    ):
        raise RuntimeError(f"{role} lifetime receipt policy is inconsistent")

    before_load = _parse_memory_sample(
        lifetime.get("before_load"),
        sampler=sampler,
        role=f"{role} before-load",
    )
    after_requests = _parse_memory_sample(
        lifetime.get("after_requests"),
        sampler=sampler,
        role=f"{role} after-requests",
    )
    after_unload = _parse_memory_sample(
        lifetime.get("after_unload"),
        sampler=sampler,
        role=f"{role} after-unload",
    )
    if (
        len(
            {
                before_load["total_bytes"],
                after_requests["total_bytes"],
                after_unload["total_bytes"],
            }
        )
        != 1
    ):
        raise RuntimeError(f"{role} lifetime boundary samples changed CUDA device identity")

    retained_process = after_unload["process_used_bytes"] - before_load["process_used_bytes"]
    retained_device = after_unload["used_bytes"] - before_load["used_bytes"]
    process_growth = after_requests["process_used_bytes"] - before_load["process_used_bytes"]
    device_growth = after_requests["used_bytes"] - before_load["used_bytes"]
    if (
        type(lifetime.get("retained_bytes")) is not int
        or lifetime.get("retained_bytes") != retained_process
        or type(lifetime.get("device_wide_retained_bytes")) is not int
        or lifetime.get("device_wide_retained_bytes") != retained_device
        or type(lifetime.get("process_growth_bytes")) is not int
        or lifetime.get("process_growth_bytes") != process_growth
        or type(lifetime.get("device_wide_growth_bytes")) is not int
        or lifetime.get("device_wide_growth_bytes") != device_growth
    ):
        raise RuntimeError(f"{role} lifetime growth/retention receipts are inconsistent")

    _validate_receipt_phase_binding(
        raw_receipt,
        phase_samples,
        role=role,
    )
    endpoint_rows = {
        "before_load_to_pre_engine": _signed_memory_attribution(
            before_load,
            phase_samples[0],
        ),
        "request_completion_to_after_requests": _signed_memory_attribution(
            phase_samples[4],
            after_requests,
        ),
    }
    endpoint_fields = (
        "cuda_device_growth_bytes",
        "nvml_current_process_growth_bytes",
        "unlisted_external_growth_bytes",
        "unexplained_growth_bytes",
    )
    for endpoint_name, endpoint in endpoint_rows.items():
        endpoint["passed"] = bool(
            endpoint["passed"]
            and all(
                abs(int(endpoint[field])) <= MEMORY_ATTRIBUTION_FLOOR_BYTES
                for field in endpoint_fields
            )
        )
        if not endpoint["passed"]:
            raise RuntimeError(
                f"{role} lifetime {endpoint_name} endpoint binding failed: {endpoint}"
            )
    attribution_rows = {
        "request_completion": _signed_memory_attribution(
            before_load,
            after_requests,
        ),
        "runtime_kv_allocation": _signed_memory_attribution(
            phase_samples[0],
            phase_samples[3],
        ),
        "runtime_request_completion": _signed_memory_attribution(
            phase_samples[0],
            phase_samples[4],
        ),
        "unload": _signed_memory_attribution(
            before_load,
            after_unload,
        ),
    }

    measured_driver_release = (
        max(
            0,
            -int(attribution_rows["unload"]["unlisted_external_growth_bytes"]),
        )
        if not cold_start
        else 0
    )
    cold_allowance_needed = bool(
        cold_start
        and any(
            abs(row["unlisted_external_growth_bytes"]) > row["tolerance_bytes"]
            for row in attribution_rows.values()
        )
    )
    unload_attribution = attribution_rows["unload"]
    unload_unlisted = unload_attribution["unlisted_external_growth_bytes"]
    all_samples = [
        before_load,
        after_requests,
        after_unload,
        *phase_samples,
    ]
    no_visible_other_process = all(
        sample["other_compute_process_used_bytes"] == 0
        and len(sample["compute_processes"]) == 1
        and sample["compute_processes"][0]["pid"] == sampler["pid"]
        for sample in all_samples
    )
    cold_persistent_boundary_tolerances = {
        name: _independent_unlisted_tolerance_bytes(
            int(row["unlisted_external_growth_bytes"]),
            int(unload_unlisted),
        )
        for name, row in attribution_rows.items()
    }
    cold_persistent_boundary_matches = {
        name: bool(
            0 < int(row["unlisted_external_growth_bytes"]) <= COLD_PERSISTENT_UNLISTED_LIMIT_BYTES
            and abs(int(row["unlisted_external_growth_bytes"]) - unload_unlisted)
            <= cold_persistent_boundary_tolerances[name]
        )
        for name, row in attribution_rows.items()
    }
    persistent_unlisted = bool(
        cold_allowance_needed
        and no_visible_other_process
        and 0 < unload_unlisted <= COLD_PERSISTENT_UNLISTED_LIMIT_BYTES
        and all(cold_persistent_boundary_matches.values())
    )
    for name, row in attribution_rows.items():
        persistent_match_tolerance = cold_persistent_boundary_tolerances[name]
        special_unlisted_passed = bool(
            persistent_unlisted
            and row["unlisted_external_growth_bytes"] > 0
            and abs(row["unlisted_external_growth_bytes"] - unload_unlisted)
            <= persistent_match_tolerance
        )
        measured_negative_unlisted_passed = bool(
            not cold_start
            and -max(
                row["tolerance_bytes"],
                cold_driver_budget_bytes,
            )
            <= row["unlisted_external_growth_bytes"]
            <= row["tolerance_bytes"]
        )
        measured_release_match_tolerance = _independent_unlisted_tolerance_bytes(
            int(row["unlisted_external_growth_bytes"]),
            -measured_driver_release,
        )
        measured_persistent_release_matches = bool(
            cold_start
            or name
            not in {
                "request_completion",
                "runtime_kv_allocation",
                "runtime_request_completion",
                "unload",
            }
            or abs(int(row["unlisted_external_growth_bytes"]) + measured_driver_release)
            <= measured_release_match_tolerance
        )
        row["cold_persistent_unlisted_allowance_used"] = special_unlisted_passed
        row["measured_cold_driver_release_allowance_used"] = bool(
            measured_negative_unlisted_passed and not row["unlisted_external_passed"]
        )
        row["persistent_release_matches_unload"] = measured_persistent_release_matches
        row["persistent_release_match_tolerance_bytes"] = measured_release_match_tolerance
        row["passed"] = bool(
            row["components_passed"]
            and (
                row["unlisted_external_passed"]
                or special_unlisted_passed
                or measured_negative_unlisted_passed
            )
            and measured_persistent_release_matches
        )
        if not row["passed"]:
            raise RuntimeError(f"{role} lifetime {name} CUDA/NVML signed attribution failed: {row}")

    resident_weight_bytes = raw_receipt.get("resident_weight_bytes")
    if type(resident_weight_bytes) is not int or resident_weight_bytes <= 0:
        raise RuntimeError(f"{role} lifetime has no valid resident-weight receipt")
    process_limit = (
        int(raw_receipt["module_residency_reserve_bytes"])
        if cold_start
        else MEMORY_ATTRIBUTION_FLOOR_BYTES
    )
    device_positive_limit = process_limit
    device_negative_limit = process_limit
    if persistent_unlisted:
        device_positive_limit += COLD_PERSISTENT_UNLISTED_LIMIT_BYTES
        device_negative_limit += COLD_PERSISTENT_UNLISTED_LIMIT_BYTES
    if not cold_start:
        # A proven cold driver/JIT allocation can only justify a later
        # device-wide release.  It must never enlarge the allowance for new
        # positive retention or for unrelated visible-process disappearance.
        device_negative_limit += measured_driver_release
    process_retention_passed = abs(retained_process) <= process_limit
    device_retention_passed = -device_negative_limit <= retained_device <= device_positive_limit
    if not process_retention_passed:
        raise RuntimeError(
            f"{role} process retention exceeded its bound: "
            f"delta={retained_process}, limit={process_limit}"
        )
    if not device_retention_passed:
        raise RuntimeError(
            f"{role} device-wide retention exceeded its bound: "
            f"delta={retained_device}, positive_limit={device_positive_limit}, "
            f"negative_limit={device_negative_limit}"
        )

    return {
        "role": role,
        "cold_start": cold_start,
        "policy": policy,
        "runtime_kv_capacity_tokens": expected_r,
        "kv_allocation_id": allocation_id,
        "process_growth_bytes": process_growth,
        "device_wide_growth_bytes": device_growth,
        "process_retention_gate": {
            "signed_delta_bytes": retained_process,
            "absolute_delta_bytes": abs(retained_process),
            "limit_bytes": process_limit,
            "limit_rule": (
                "plan_bound_profile_calibration"
                if cold_start
                else "64MiB"
            ),
            "passed": process_retention_passed,
        },
        "device_retention_gate": {
            "signed_delta_bytes": retained_device,
            "absolute_delta_bytes": abs(retained_device),
            "positive_limit_bytes": device_positive_limit,
            "negative_limit_bytes": device_negative_limit,
            "limit_rule": (
                "cold_process_limit_plus_persistent_unlisted_allowance"
                if persistent_unlisted
                else "measured_positive_process_limit_negative_plus_cold_driver_release"
                if not cold_start
                else "same_as_process_retention_limit"
            ),
            "passed": device_retention_passed,
        },
        "cold_persistent_unlisted_gate": {
            "needed": cold_allowance_needed,
            "used": persistent_unlisted,
            "persistent_until_unload": (persistent_unlisted if cold_allowance_needed else True),
            "no_visible_other_compute_process": no_visible_other_process,
            "required_boundaries": list(attribution_rows),
            "boundary_matches_unload": cold_persistent_boundary_matches,
            "boundary_match_tolerance_bytes": cold_persistent_boundary_tolerances,
            "unload_unlisted_external_growth_bytes": unload_unlisted,
            "limit_bytes": COLD_PERSISTENT_UNLISTED_LIMIT_BYTES,
            "passed": bool(not cold_allowance_needed or persistent_unlisted),
        },
        "measured_cold_driver_release_bytes": measured_driver_release,
        "cold_driver_budget_bytes": cold_driver_budget_bytes,
        "signed_attribution_gates": attribution_rows,
        "endpoint_binding_gates": endpoint_rows,
        "before_load": before_load,
        "after_requests": after_requests,
        "after_unload": after_unload,
        "runtime_phase_samples": phase_samples,
        "receipt": receipt,
        "passed": True,
    }


def _validate_sample_continuity(
    baseline: Mapping[str, Any],
    boundary_sample: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    attribution = _signed_memory_attribution(baseline, boundary_sample)
    process_delta = boundary_sample["process_used_bytes"] - baseline["process_used_bytes"]
    device_delta = boundary_sample["used_bytes"] - baseline["used_bytes"]
    process_passed = abs(process_delta) <= MEMORY_ATTRIBUTION_FLOOR_BYTES
    device_passed = abs(device_delta) <= MEMORY_ATTRIBUTION_FLOOR_BYTES
    if not attribution["passed"] or not process_passed or not device_passed:
        raise RuntimeError(
            f"{label} inter-lifetime continuity failed: "
            f"process_delta={process_delta}, device_delta={device_delta}, "
            f"attribution={attribution}"
        )
    return {
        "label": label,
        "process_signed_delta_bytes": process_delta,
        "device_wide_signed_delta_bytes": device_delta,
        "absolute_limit_bytes": MEMORY_ATTRIBUTION_FLOOR_BYTES,
        "process_gate_passed": process_passed,
        "device_gate_passed": device_passed,
        "signed_attribution": attribution,
        "passed": True,
    }


def _validate_lifetime_continuity(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    return _validate_sample_continuity(
        previous["after_unload"],
        current["before_load"],
        label=label,
    )


def _validate_cold_release_budget(
    cold_start: Mapping[str, Any],
    measured_lifetimes: list[Mapping[str, Any]],
) -> dict[str, Any]:
    cold_unload = cold_start["signed_attribution_gates"]["unload"]
    cold_persistent_delta = max(
        0,
        int(cold_unload["unlisted_external_growth_bytes"]),
    )
    measured_unlisted_deltas = [
        int(row["unlisted_external_growth_bytes"])
        for lifetime in measured_lifetimes
        for row in lifetime["signed_attribution_gates"].values()
    ]
    release_tolerance = _independent_unlisted_tolerance_bytes(
        int(cold_unload["unlisted_external_growth_bytes"]),
        *measured_unlisted_deltas,
    )
    effective_budget = cold_persistent_delta + release_tolerance
    cumulative_release = 0
    lifetime_gates: list[dict[str, Any]] = []
    for index, lifetime in enumerate(measured_lifetimes):
        rows = lifetime["signed_attribution_gates"]
        released = max(
            0,
            max(-int(row["unlisted_external_growth_bytes"]) for row in rows.values()),
        )
        cumulative_release += released
        passed = cumulative_release <= effective_budget
        gate = {
            "measured_lifetime_index": index,
            "maximum_unlisted_release_bytes": released,
            "cumulative_unlisted_release_bytes": cumulative_release,
            "cold_persistent_unlisted_budget_bytes": cold_persistent_delta,
            "signed_attribution_tolerance_bytes": release_tolerance,
            "effective_release_budget_bytes": effective_budget,
            "passed": passed,
        }
        lifetime_gates.append(gate)
        if not passed:
            raise RuntimeError(
                "measured lifetime released more unlisted device memory than "
                f"the cold-start persistent delta: {gate}"
            )
    return {
        "cold_persistent_unlisted_budget_bytes": cold_persistent_delta,
        "signed_attribution_tolerance_bytes": release_tolerance,
        "effective_release_budget_bytes": effective_budget,
        "cumulative_measured_unlisted_release_bytes": cumulative_release,
        "measured_lifetime_gates": lifetime_gates,
        "passed": True,
    }


def validate_sequential_requests(
    trace: dict[str, Any],
    *,
    expected_count: int,
    tolerance_bytes: int,
) -> dict[str, Any]:
    if (
        type(expected_count) is not int
        or expected_count <= 0
        or type(tolerance_bytes) is not int
        or tolerance_bytes < 0
    ):
        raise RuntimeError("sequential request validation inputs are invalid")
    sampler = validate_nvml_sampler(trace)
    samples = trace.get("sequential_requests")
    if (
        type(trace.get("sequential_request_count")) is not int
        or trace.get("sequential_request_count") != expected_count
        or not isinstance(samples, list)
        or len(samples) != expected_count
    ):
        raise RuntimeError("sequential request sample count mismatch")
    expected_row_fields = {
        "request_index",
        "before",
        "after",
        "kv_allocation_id",
        "final_kv_position",
    }
    allocation_ids: set[int] = set()
    positions: set[int] = set()
    parsed_requests: list[dict[str, Any]] = []
    for index, item in enumerate(samples):
        if not isinstance(item, dict) or set(item) != expected_row_fields:
            raise RuntimeError("sequential request evidence row has an invalid schema")
        if type(item.get("request_index")) is not int or item["request_index"] != index:
            raise RuntimeError("sequential request indices are not contiguous")
        allocation_id = item.get("kv_allocation_id")
        final_position = item.get("final_kv_position")
        if type(allocation_id) is not int or allocation_id <= 0:
            raise RuntimeError("sequential request allocation identity is invalid")
        if type(final_position) is not int or final_position <= 0:
            raise RuntimeError("sequential request final KV position is invalid")
        allocation_ids.add(allocation_id)
        positions.add(final_position)
        before = _parse_memory_sample(
            item.get("before"),
            sampler=sampler,
            role=f"sequential request {index} before",
        )
        after = _parse_memory_sample(
            item.get("after"),
            sampler=sampler,
            role=f"sequential request {index} after",
        )
        parsed_requests.append({"before": before, "after": after})
    if len(allocation_ids) != 1:
        raise RuntimeError("sequential requests did not reuse one KV allocation")
    if len(positions) != 1:
        raise RuntimeError("sequential requests produced inconsistent final positions")

    first_request_attribution = _signed_memory_attribution(
        parsed_requests[0]["before"],
        parsed_requests[0]["after"],
    )
    first_request_no_visible_other_process = all(
        int(sample["other_compute_process_used_bytes"]) == 0
        and len(sample["compute_processes"]) == 1
        for sample in (
            parsed_requests[0]["before"],
            parsed_requests[0]["after"],
        )
    )
    first_request_unlisted = int(first_request_attribution["unlisted_external_growth_bytes"])
    first_request_cold_growth_used = bool(
        first_request_attribution["components_passed"]
        and first_request_no_visible_other_process
        and 0 < first_request_unlisted <= COLD_PERSISTENT_UNLISTED_LIMIT_BYTES
    )
    first_request_external_release_used = bool(
        first_request_attribution["components_passed"] and first_request_unlisted < 0
    )
    first_request_transition_gate = {
        **first_request_attribution,
        "one_time_cold_growth_used": first_request_cold_growth_used,
        "external_release_used": first_request_external_release_used,
        "no_visible_other_compute_process": first_request_no_visible_other_process,
        "passed": bool(
            first_request_attribution["passed"]
            or first_request_cold_growth_used
            or first_request_external_release_used
        ),
    }
    if not first_request_transition_gate["passed"]:
        raise RuntimeError(
            "sequential request 0 has invalid one-time cold/JIT attribution: "
            f"{first_request_transition_gate}"
        )
    fixed_baseline = parsed_requests[0]["after"]
    continuity_gates: list[dict[str, Any]] = []
    fixed_baseline_gates: list[dict[str, Any]] = []
    for index, request in enumerate(parsed_requests):
        fixed_baseline_gates.append(
            _validate_positive_growth_envelope(
                fixed_baseline,
                request["after"],
                label=f"sequential_fixed_baseline_to_request_{index}_after",
                positive_limit_bytes=tolerance_bytes,
                allow_process_release=True,
            )
        )
        if index == 0:
            continue
        continuity_gates.append(
            _validate_positive_growth_envelope(
                parsed_requests[index - 1]["after"],
                request["before"],
                label=f"sequential_request_{index - 1}_to_{index}",
                positive_limit_bytes=tolerance_bytes,
                allow_process_release=True,
            )
        )
        fixed_baseline_gates.append(
            _validate_positive_growth_envelope(
                fixed_baseline,
                request["before"],
                label=f"sequential_fixed_baseline_to_request_{index}_before",
                positive_limit_bytes=tolerance_bytes,
                allow_process_release=True,
            )
        )

    process_after_used = [_process_used(item["after"]) for item in parsed_requests]
    device_after_used = [int(item["after"]["used_bytes"]) for item in parsed_requests]
    window = min(10, expected_count)
    trend_gates: dict[str, dict[str, Any]] = {}
    for scope, after_used in (
        ("current_process", process_after_used),
        ("device_wide", device_after_used),
    ):
        first_mean = statistics.fmean(after_used[:window])
        last_mean = statistics.fmean(after_used[-window:])
        delta = last_mean - first_mean
        monotonic_growth_steps = sum(
            current > previous + tolerance_bytes
            for previous, current in zip(after_used, after_used[1:])
        )
        passed = bool(delta <= tolerance_bytes and monotonic_growth_steps != expected_count - 1)
        trend_gates[scope] = {
            "first_window_used_mean": first_mean,
            "last_window_used_mean": last_mean,
            "delta_used_bytes": delta,
            "tolerance_bytes": tolerance_bytes,
            "monotonic_growth_steps": monotonic_growth_steps,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(
                f"sequential requests show retained {scope} memory growth: "
                f"first_mean={first_mean:.0f}, last_mean={last_mean:.0f}, "
                f"delta={delta:.0f}, monotonic_steps={monotonic_growth_steps}, "
                f"tolerance={tolerance_bytes}"
            )
    process_gate = trend_gates["current_process"]
    device_gate = trend_gates["device_wide"]
    return {
        "request_count": expected_count,
        "kv_allocation_id": next(iter(allocation_ids)),
        "final_kv_position": next(iter(positions)),
        "first_window_used_mean": process_gate["first_window_used_mean"],
        "last_window_used_mean": process_gate["last_window_used_mean"],
        "delta_used_bytes": process_gate["delta_used_bytes"],
        "tolerance_bytes": tolerance_bytes,
        "monotonic_growth_steps": process_gate["monotonic_growth_steps"],
        "current_process_memory_trend": process_gate,
        "device_wide_memory_trend": device_gate,
        "memory_sampler": sampler,
        "first_request_transition_gate": first_request_transition_gate,
        "inter_request_continuity_gates": continuity_gates,
        "fixed_baseline_memory_sample": fixed_baseline,
        "fixed_baseline_memory_gates": fixed_baseline_gates,
        "passed": True,
    }


def validate_load_cycles(
    trace: dict[str, Any],
    *,
    expected_count: int,
    expected_r: int,
    tolerance_bytes: int,
) -> dict[str, Any]:
    sampler = validate_nvml_sampler(trace)
    expected_policy = {
        "kind": "max_sequence_length",
        "requested_tokens": expected_r,
    }
    warmup = trace.get("load_cycle_warmup")
    if not isinstance(warmup, dict):
        raise RuntimeError("load/unload trace has no explicit unmeasured warm-up")
    cycles = trace.get("load_cycles")
    if (
        type(trace.get("load_cycle_count")) is not int
        or trace.get("load_cycle_count") != expected_count
        or not isinstance(cycles, list)
        or len(cycles) != expected_count
    ):
        raise RuntimeError("load/unload cycle sample count mismatch")

    warmup_evidence = _validate_lifetime_memory(
        warmup,
        sampler=sampler,
        role="load_cycle_cold_start",
        expected_policy=expected_policy,
        expected_r=expected_r,
        cold_start=True,
        expected_markers={
            "label": "unmeasured-load-cycle-warmup",
            "measured": False,
            "execution_ordinal": 0,
            "role": "warmup",
        },
    )
    warmup_receipt = warmup.get("runtime_memory_receipt")
    if not isinstance(warmup_receipt, Mapping):
        raise RuntimeError("cold-start lifetime has no stable runtime-memory receipt")
    cold_driver_budget = max(
        0,
        int(
            warmup_evidence["signed_attribution_gates"]["unload"]["unlisted_external_growth_bytes"]
        ),
    )
    deltas: list[int] = []
    allocation_ids: set[int] = set()
    measured_cycles: list[dict[str, Any]] = []
    measured_memory_evidence: list[dict[str, Any]] = []
    external_pressure_deltas: list[int] = []
    continuity_gates: list[dict[str, Any]] = []
    common_baseline_gates: list[dict[str, Any]] = []
    common_measured_baseline: Mapping[str, Any] | None = None
    previous_evidence = warmup_evidence
    for index, cycle in enumerate(cycles):
        if (
            not isinstance(cycle, dict)
            or type(cycle.get("cycle_index")) is not int
            or cycle.get("cycle_index") != index
        ):
            raise RuntimeError("load/unload cycle indices are not contiguous")
        evidence = _validate_lifetime_memory(
            cycle,
            sampler=sampler,
            role=f"load_cycle_measured_{index}",
            expected_policy=expected_policy,
            expected_r=expected_r,
            cold_start=False,
            cold_driver_budget_bytes=cold_driver_budget,
            expected_markers={
                "label": "measured-load-cycle",
                "measured": True,
                "execution_ordinal": index + 1,
                "role": "measured",
            },
        )
        cycle_receipt = cycle.get("runtime_memory_receipt")
        if not isinstance(cycle_receipt, Mapping):
            raise RuntimeError(
                f"load_cycle_measured_{index} has no stable runtime-memory receipt"
            )
        for field in _PLAN_BOUND_RESIDENCY_STABLE_RECEIPT_FIELDS:
            if cycle_receipt.get(field) != warmup_receipt.get(field):
                raise RuntimeError(
                    "cold/measured receipts disagree on stable plan-bound "
                    f"module residency field {field}"
                )
        continuity_gates.append(
            _validate_positive_growth_envelope(
                previous_evidence["after_unload"],
                evidence["before_load"],
                label=(
                    f"load_cycle_{index - 1}_to_{index}" if index else "cold_start_to_load_cycle_0"
                ),
                positive_limit_bytes=MEMORY_ATTRIBUTION_FLOOR_BYTES,
            )
        )
        if common_measured_baseline is None:
            common_measured_baseline = evidence["before_load"]
        common_baseline_gates.extend(
            [
                _validate_positive_growth_envelope(
                    common_measured_baseline,
                    evidence["before_load"],
                    label=f"load_cycle_common_baseline_to_{index}_before_load",
                    positive_limit_bytes=MEMORY_ATTRIBUTION_FLOOR_BYTES,
                ),
                _validate_positive_growth_envelope(
                    common_measured_baseline,
                    evidence["after_unload"],
                    label=f"load_cycle_common_baseline_to_{index}_after_unload",
                    positive_limit_bytes=MEMORY_ATTRIBUTION_FLOOR_BYTES,
                ),
            ]
        )
        previous_evidence = evidence
        measured_memory_evidence.append(evidence)
        delta = evidence["process_retention_gate"]["signed_delta_bytes"]
        deltas.append(delta)
        device_wide_delta = evidence["device_retention_gate"]["signed_delta_bytes"]
        external_pressure_delta = device_wide_delta - delta
        external_pressure_deltas.append(external_pressure_delta)
        allocation_ids.add(evidence["kv_allocation_id"])
        measured_cycles.append(
            {
                "cycle_index": index,
                "retained_bytes": delta,
                "device_wide_retained_bytes": device_wide_delta,
                "external_pressure_delta_bytes": external_pressure_delta,
                "policy": expected_policy,
                "runtime_kv_capacity_tokens": expected_r,
                "runtime_memory_receipt": cycle["runtime_memory_receipt"],
                "kv_allocation_id": evidence["kv_allocation_id"],
                "memory_evidence": evidence,
            }
        )
    if len(allocation_ids) != expected_count:
        raise RuntimeError("load cycles reused an allocation identity across lifetimes")
    if warmup_evidence["kv_allocation_id"] in allocation_ids:
        raise RuntimeError("load cycle cold start reused a measured allocation identity")
    cold_release_budget = _validate_cold_release_budget(
        warmup_evidence,
        measured_memory_evidence,
    )
    return {
        "memory_sampler": sampler,
        "warmup": warmup,
        "cold_start_evidence": warmup_evidence,
        "cycle_count": expected_count,
        "measured_cycle_indices": list(range(expected_count)),
        "measured_cycles": measured_cycles,
        "continuity_gates": continuity_gates,
        "common_measured_baseline": common_measured_baseline,
        "common_baseline_gates": common_baseline_gates,
        "cold_release_budget_gate": cold_release_budget,
        "max_retained_bytes": max(deltas),
        "min_retained_bytes": min(deltas),
        "max_external_pressure_delta_bytes": max(external_pressure_deltas),
        "min_external_pressure_delta_bytes": min(external_pressure_deltas),
        "tolerance_bytes": tolerance_bytes,
        "unique_allocation_ids": len(allocation_ids),
        "passed": True,
    }


def validate_two_r_slope(small: dict[str, int], large: dict[str, int]) -> dict[str, Any]:
    if large["R"] <= small["R"]:
        raise RuntimeError("two-R qualification requires R2 > R1")
    if small["B"] != large["B"]:
        raise RuntimeError("KV bytes per token changed between R1 and R2")
    expected = (large["R"] - small["R"]) * small["B"]
    actual = large["kv_reserved_bytes"] - small["kv_reserved_bytes"]
    if actual != expected:
        raise RuntimeError(
            f"KV allocation slope mismatch: actual delta={actual}, expected={expected}"
        )
    if small["kv_allocation_id"] == large["kv_allocation_id"]:
        raise RuntimeError("independent R1/R2 lifetimes reused an allocation identity")
    return {
        "R1": small["R"],
        "R2": large["R"],
        "B": small["B"],
        "actual_delta_bytes": actual,
        "expected_delta_bytes": expected,
        "passed": True,
    }


def validate_same_process_two_r(
    trace: dict[str, Any],
    *,
    r1: int,
    r2: int,
    tolerance_bytes: int,
) -> dict[str, Any]:
    sampler = validate_nvml_sampler(trace)
    if type(r1) is not int or type(r2) is not int or not (0 < r1 < r2):
        raise RuntimeError("two-R qualification requires typed positive R1 < R2")
    if trace.get("mode") != "same_process_two_r_allocation_slope":
        raise RuntimeError("runner did not use same-process two-R mode")
    warmup = trace.get("allocation_slope_warmup")
    if not isinstance(warmup, dict):
        raise RuntimeError("two-R trace has no explicit unmeasured R2 warm-up")
    lifetimes = trace.get("allocation_slope_lifetimes")
    if not isinstance(lifetimes, list) or len(lifetimes) != 2:
        raise RuntimeError("two-R trace must contain exactly two measured lifetimes")

    r1_policy = {"kind": "max_sequence_length", "requested_tokens": r1}
    r2_policy = {"kind": "max_sequence_length", "requested_tokens": r2}
    warmup_evidence = _validate_lifetime_memory(
        warmup,
        sampler=sampler,
        role="two_r_cold_start",
        expected_policy=r2_policy,
        expected_r=r2,
        cold_start=True,
        expected_markers={
            "label": "unmeasured-r2-warmup",
            "measured": False,
        },
    )
    cold_driver_budget = max(
        0,
        int(
            warmup_evidence["signed_attribution_gates"]["unload"]["unlisted_external_growth_bytes"]
        ),
    )
    receipts: list[dict[str, int]] = []
    expected_rs = (r1, r2)
    expected_policies = (r1_policy, r2_policy)
    labels = ("measured-r1", "measured-r2")
    measured_evidence: list[dict[str, Any]] = []
    continuity_gates: list[dict[str, Any]] = []
    previous_evidence = warmup_evidence
    for index, (lifetime, expected_r, policy, label) in enumerate(
        zip(lifetimes, expected_rs, expected_policies, labels)
    ):
        evidence = _validate_lifetime_memory(
            lifetime,
            sampler=sampler,
            role=f"two_r_measured_{index}",
            expected_policy=policy,
            expected_r=expected_r,
            cold_start=False,
            cold_driver_budget_bytes=cold_driver_budget,
            expected_markers={
                "label": label,
                "measured": True,
            },
        )
        measured_evidence.append(evidence)
        receipts.append(evidence["receipt"])
        continuity_gates.append(
            _validate_lifetime_continuity(
                previous_evidence,
                evidence,
                label="two_r_cold_start_to_r1" if index == 0 else "two_r_r1_to_r2",
            )
        )
        previous_evidence = evidence

    plan_slope = validate_two_r_slope(receipts[0], receipts[1])
    if warmup_evidence["kv_allocation_id"] in {
        evidence["kv_allocation_id"] for evidence in measured_evidence
    }:
        raise RuntimeError("two-R cold start reused a measured allocation identity")
    cold_release_budget = _validate_cold_release_budget(
        warmup_evidence,
        measured_evidence,
    )
    expected_delta = int(plan_slope["expected_delta_bytes"])
    process_growth = [item["process_growth_bytes"] for item in measured_evidence]
    nvml_delta = process_growth[1] - process_growth[0]
    if abs(nvml_delta - expected_delta) > tolerance_bytes:
        raise RuntimeError(
            "same-process NVML allocation slope mismatch: "
            f"actual delta={nvml_delta}, expected={expected_delta}, "
            f"tolerance={tolerance_bytes}"
        )
    return {
        **plan_slope,
        "memory_sampler": sampler,
        "warmup": warmup,
        "measured_lifetimes": lifetimes,
        "cold_start_evidence": warmup_evidence,
        "measured_lifetime_evidence": measured_evidence,
        "continuity_gates": continuity_gates,
        "cold_release_budget_gate": cold_release_budget,
        "nvml_process_growth_r1_bytes": process_growth[0],
        "nvml_process_growth_r2_bytes": process_growth[1],
        "nvml_actual_delta_bytes": nvml_delta,
        "nvml_delta_tolerance_bytes": tolerance_bytes,
        "passed": True,
    }


def validate_controlled_reservation(
    trace: dict[str, Any],
    *,
    tolerance_bytes: int,
) -> dict[str, Any]:
    if type(tolerance_bytes) is not int or tolerance_bytes < 0:
        raise RuntimeError("controlled reservation tolerance is invalid")
    sampler = validate_nvml_sampler(trace)
    if trace.get("mode") != "same_process_controlled_external_reservation":
        raise RuntimeError("runner did not use controlled-reservation mode")
    proof = trace.get("controlled_reservation")
    if not isinstance(proof, dict):
        raise RuntimeError("runner output has no controlled-reservation receipt")
    if proof.get("passed") is not True:
        raise RuntimeError("runner reported that controlled reservation failed")
    warmup = proof.get("warmup")
    calibration = proof.get("calibration")
    baseline = proof.get("baseline")
    constrained = proof.get("constrained")
    if (
        not isinstance(warmup, dict)
        or not isinstance(calibration, dict)
        or not isinstance(baseline, dict)
        or not isinstance(constrained, dict)
    ):
        raise RuntimeError("controlled reservation misses lifetime receipts")

    def parse_lifetime_samples(
        lifetime: Mapping[str, Any],
        *,
        role: str,
    ) -> dict[str, Any]:
        before_load = _parse_memory_sample(
            lifetime.get("before_load"),
            sampler=sampler,
            role=f"{role} before-load",
        )
        after_requests = _parse_memory_sample(
            lifetime.get("after_requests"),
            sampler=sampler,
            role=f"{role} after-requests",
        )
        runtime_phase_samples = _validate_runtime_phase_samples(
            lifetime,
            sampler=sampler,
            role=role,
        )
        endpoint_rows = {
            "before_load_to_pre_engine": _signed_memory_attribution(
                before_load,
                runtime_phase_samples[0],
            ),
            "request_completion_to_after_requests": _signed_memory_attribution(
                runtime_phase_samples[4],
                after_requests,
            ),
        }
        endpoint_sample_pairs = {
            "before_load_to_pre_engine": (
                before_load,
                runtime_phase_samples[0],
            ),
            "request_completion_to_after_requests": (
                runtime_phase_samples[4],
                after_requests,
            ),
        }
        for endpoint_name, endpoint in endpoint_rows.items():
            baseline_sample, boundary_sample = endpoint_sample_pairs[endpoint_name]
            no_visible_other_process = all(
                int(sample["other_compute_process_used_bytes"]) == 0
                and len(sample["compute_processes"]) == 1
                and sample["compute_processes"][0]["pid"] == sampler["pid"]
                for sample in (baseline_sample, boundary_sample)
            )
            visible_other_unchanged = int(endpoint["nvml_visible_other_process_growth_bytes"]) == 0
            process_passed = (
                abs(int(endpoint["nvml_current_process_growth_bytes"]))
                <= MEMORY_ATTRIBUTION_FLOOR_BYTES
            )
            brackets_passed = bool(
                int(endpoint["baseline_cuda_nvml_bracket_difference_bytes"])
                <= MEMORY_ATTRIBUTION_FLOOR_BYTES
                and int(endpoint["boundary_cuda_nvml_bracket_difference_bytes"])
                <= MEMORY_ATTRIBUTION_FLOOR_BYTES
            )
            unexplained_passed = (
                abs(int(endpoint["unexplained_growth_bytes"])) <= MEMORY_ATTRIBUTION_FLOOR_BYTES
            )
            generic_attribution_passed = bool(endpoint["passed"])
            endpoint.update(
                {
                    "external_driver_delta_bytes": int(endpoint["unlisted_external_growth_bytes"]),
                    "external_driver_delta_is_runtime_action_bytes": False,
                    "runtime_phase_boundary_is_authoritative": True,
                    "generic_attribution_passed": generic_attribution_passed,
                    "independent_attribution_tolerance_bytes": (MEMORY_ATTRIBUTION_FLOOR_BYTES),
                    "no_visible_other_compute_process": (no_visible_other_process),
                    "visible_other_process_unchanged": (visible_other_unchanged),
                    "current_process_gate_passed": process_passed,
                    "cuda_nvml_bracket_gate_passed": brackets_passed,
                    "unexplained_residual_gate_passed": unexplained_passed,
                    "passed": bool(
                        no_visible_other_process
                        and visible_other_unchanged
                        and process_passed
                        and brackets_passed
                        and unexplained_passed
                    ),
                }
            )
            if not endpoint["passed"]:
                raise RuntimeError(f"{role} {endpoint_name} endpoint binding failed: {endpoint}")
        parsed = {
            "before_load": before_load,
            "after_requests": after_requests,
            "after_unload": _parse_memory_sample(
                lifetime.get("after_unload"),
                sampler=sampler,
                role=f"{role} after-unload",
            ),
            "runtime_phase_samples": runtime_phase_samples,
            "endpoint_binding_gates": endpoint_rows,
        }
        return parsed

    controlled_lifetime_samples = {
        "warmup": parse_lifetime_samples(warmup, role="controlled warmup"),
        "calibration": parse_lifetime_samples(
            calibration,
            role="controlled calibration",
        ),
        "baseline": parse_lifetime_samples(baseline, role="controlled baseline"),
        "constrained": parse_lifetime_samples(
            constrained,
            role="controlled constrained",
        ),
    }
    target_tokens = int(proof["target_tokens"])
    if (
        warmup.get("policy") != {"kind": "auto"}
        or warmup.get("measured") is not False
        or calibration.get("policy")
        != {
            "kind": "max_sequence_length",
            "requested_tokens": target_tokens,
        }
        or calibration.get("measured") is not True
        or baseline.get("policy") != {"kind": "auto"}
        or constrained.get("policy") != {"kind": "auto"}
        or baseline.get("measured") is not True
        or constrained.get("measured") is not True
    ):
        raise RuntimeError(
            "controlled reservation did not use warmup, explicit target "
            "calibration, and measured auto lifetimes in order"
        )

    baseline_r = int(proof["baseline_r"])
    constrained_r = int(proof["constrained_r"])
    sizing = proof.get("sizing")
    if not isinstance(sizing, dict):
        raise RuntimeError("controlled reservation misses sizing components")
    alignment = int(sizing.get("reservation_alignment_bytes", 0))
    if alignment != CONTROLLED_RESERVATION_ALIGNMENT_BYTES:
        raise RuntimeError("controlled reservation used the wrong alignment")
    rounding_rows = int(sizing.get("max_capacity_rounding_rows", -1))
    target_tolerance_rows = int(sizing.get("target_tolerance_rows", -1))
    if not (
        rounding_rows == CONTROLLED_TARGET_TOLERANCE_ROWS
        and target_tolerance_rows == CONTROLLED_TARGET_TOLERANCE_ROWS
        and baseline_r > constrained_r >= int(trace["final_kv_position"])
        and target_tokens <= constrained_r <= target_tokens + CONTROLLED_TARGET_TOLERANCE_ROWS
    ):
        raise RuntimeError(
            "controlled reservation did not resolve near target while reducing "
            "R and fitting the request"
        )
    calibration_receipt = validate_receipt(
        calibration,
        target_tokens,
        expected_policy={
            "kind": "max_sequence_length",
            "requested_tokens": target_tokens,
        },
    )
    baseline_receipt = validate_receipt(
        baseline,
        baseline_r,
        expected_policy={"kind": "auto"},
    )
    constrained_receipt = validate_receipt(
        constrained,
        constrained_r,
        expected_policy={"kind": "auto"},
    )
    calibration_receipt_raw = calibration["runtime_memory_receipt"]
    baseline_receipt_raw = baseline["runtime_memory_receipt"]
    constrained_receipt_raw = constrained["runtime_memory_receipt"]
    for role, receipt, samples in (
        (
            "controlled calibration",
            calibration_receipt_raw,
            controlled_lifetime_samples["calibration"]["runtime_phase_samples"],
        ),
        (
            "controlled baseline",
            baseline_receipt_raw,
            controlled_lifetime_samples["baseline"]["runtime_phase_samples"],
        ),
        (
            "controlled constrained",
            constrained_receipt_raw,
            controlled_lifetime_samples["constrained"]["runtime_phase_samples"],
        ),
    ):
        _validate_receipt_phase_binding(receipt, samples, role=role)
    if not (calibration_receipt["B"] == baseline_receipt["B"] == constrained_receipt["B"]):
        raise RuntimeError("controlled reservation changed KV bytes per token")
    if any(
        receipt.get("policy") != "auto"
        for receipt in (
            calibration_receipt_raw,
            baseline_receipt_raw,
            constrained_receipt_raw,
        )
    ):
        raise RuntimeError("controlled reservation receipt policy is not auto")
    expected_warmup_retained = _process_used(
        controlled_lifetime_samples["warmup"]["after_unload"]
    ) - _process_used(controlled_lifetime_samples["warmup"]["before_load"])
    if int(sizing["warmup_retained_process_bytes"]) != expected_warmup_retained:
        raise RuntimeError("controlled reservation warmup-retained receipt mismatch")

    before_planning_phase = "before runtime KV planning"
    after_overhead_phase = "after shared context and output allocation"
    after_kv_phase = "after runtime KV allocation"
    after_request_phase = "after successful runtime-memory request completion"
    calibration_phases = controlled_lifetime_samples["calibration"]["runtime_phase_samples"]
    constrained_phases = controlled_lifetime_samples["constrained"]["runtime_phase_samples"]
    calibration_before_planning = calibration_phases[1]
    calibration_after_overhead = calibration_phases[2]
    calibration_after_kv = calibration_phases[3]
    calibration_after_request = calibration_phases[4]
    measured_context_output_bytes = int(calibration_before_planning["free_bytes"]) - int(
        calibration_after_overhead["free_bytes"]
    )
    if measured_context_output_bytes < 0:
        raise RuntimeError("calibration context/output delta is negative")
    request_completion_device_bytes = max(
        0,
        int(calibration_after_kv["free_bytes"]) - int(calibration_after_request["free_bytes"]),
    )
    request_completion_process_bytes = max(
        0,
        _process_used(calibration_after_request) - _process_used(calibration_after_kv),
    )
    request_completion_headroom_bytes = max(
        request_completion_device_bytes,
        request_completion_process_bytes,
    )
    request_completion_external_delta_bytes = (
        request_completion_device_bytes - request_completion_process_bytes
    )
    request_completion_guard_basis = (
        "max(calibration device-wide free delta, calibration per-process NVML delta)"
    )
    attribution_fields = (
        "request_completion_external_delta_bytes",
        "request_completion_guard_basis",
    )
    missing_attribution_fields = [field for field in attribution_fields if field not in sizing]
    if missing_attribution_fields:
        raise RuntimeError(
            "controlled reservation sizing misses request-attribution fields: "
            f"{missing_attribution_fields}"
        )
    logical_non_kv_fields = (
        "calibration_context_device_memory_bytes",
        "calibration_ordinary_device_input_bytes",
        "calibration_ordinary_device_output_bytes",
        "calibration_external_device_output_bytes",
        "calibration_graph_private_device_bytes",
        "logical_context_output_bytes",
    )
    missing_logical_non_kv_fields = [
        field for field in logical_non_kv_fields if field not in sizing
    ]
    if missing_logical_non_kv_fields:
        raise RuntimeError(
            "controlled reservation sizing misses logical non-KV fields: "
            f"{missing_logical_non_kv_fields}"
        )
    target_kv_bytes = calibration_receipt["kv_reserved_bytes"]
    auto_fraction = float(sizing.get("auto_fraction", 0.0))
    if (
        not math.isclose(auto_fraction, 0.9, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            float(calibration_receipt_raw.get("policy_fraction", 0.0)),
            auto_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(baseline_receipt_raw.get("policy_fraction", 0.0)),
            auto_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(constrained_receipt_raw.get("policy_fraction", 0.0)),
            auto_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("controlled reservation auto fraction mismatch")
    policy_safe_bytes = _ceil_divided_by_fraction(target_kv_bytes, auto_fraction)
    policy_fraction_headroom_bytes = policy_safe_bytes - target_kv_bytes
    safety_reserve_bytes = int(baseline_receipt_raw["safety_reserve_bytes"])
    if any(
        int(receipt.get("safety_reserve_bytes", -1)) != safety_reserve_bytes
        for receipt in (calibration_receipt_raw, constrained_receipt_raw)
    ):
        raise RuntimeError("controlled reservation changed the safety reserve")
    module_residency_reserve_bytes = int(
        baseline_receipt_raw["module_residency_reserve_bytes"]
    )
    if any(
        int(receipt.get("module_residency_reserve_bytes", -1))
        != module_residency_reserve_bytes
        for receipt in (calibration_receipt_raw, constrained_receipt_raw)
    ):
        raise RuntimeError(
            "controlled reservation changed the plan-bound module residency reserve"
        )
    logical_context_output_bytes = sum(
        int(calibration_receipt_raw[field])
        for field in (
            "context_device_memory_bytes",
            "ordinary_device_input_bytes",
            "ordinary_device_output_bytes",
            "external_device_output_bytes",
            "graph_private_device_bytes",
        )
    )
    final_free_lower_bound = (
        safety_reserve_bytes
        + module_residency_reserve_bytes
        + policy_safe_bytes
    )
    final_free_upper_bound = final_free_lower_bound + alignment
    required_visible_post_load_free = (
        logical_context_output_bytes
        + final_free_upper_bound
        + CONTROLLED_PREPLANNING_HEADROOM_BYTES
    )
    expected_guard_bytes = _align_up(
        target_kv_bytes + request_completion_headroom_bytes,
        alignment,
    )
    expected_engine_load_device_bytes = max(
        0,
        int(sizing["baseline_pre_load_free_bytes"]) - int(sizing["baseline_post_load_free_bytes"]),
    )
    if (
        sizing.get("required_free_basis")
        != (
            "calibration receipt logical context/output bytes plus exact final target "
            "window and preplanning headroom"
        )
        or int(sizing["target_kv_bytes"]) != target_kv_bytes
        or int(sizing["policy_safe_bytes"]) != policy_safe_bytes
        or int(sizing["policy_fraction_headroom_bytes"]) != policy_fraction_headroom_bytes
        or int(sizing["measured_context_output_bytes"]) != measured_context_output_bytes
        or int(sizing["logical_context_output_bytes"]) != logical_context_output_bytes
        or int(sizing["request_completion_device_bytes"]) != request_completion_device_bytes
        or int(sizing["request_completion_process_bytes"]) != request_completion_process_bytes
        or int(sizing["request_completion_external_delta_bytes"])
        != request_completion_external_delta_bytes
        or int(sizing["request_completion_headroom_bytes"]) != request_completion_headroom_bytes
        or sizing["request_completion_guard_basis"] != request_completion_guard_basis
        or int(sizing["guard_bytes"]) != expected_guard_bytes
        or int(sizing["safety_reserve_bytes"]) != safety_reserve_bytes
        or int(sizing["required_visible_post_load_free_bytes"]) != required_visible_post_load_free
        or int(sizing["final_free_lower_bound_bytes"]) != final_free_lower_bound
        or int(sizing["final_free_upper_bound_bytes"]) != final_free_upper_bound
        or int(sizing["preplanning_headroom_bytes"]) != CONTROLLED_PREPLANNING_HEADROOM_BYTES
        or sizing.get("visible_free_formula")
        != (
            "logical_context_output_bytes + final_free_upper_bound_bytes + "
            "preplanning_headroom_bytes"
        )
        or int(sizing["baseline_engine_load_device_bytes"]) != expected_engine_load_device_bytes
        or int(sizing["calibration_context_device_memory_bytes"])
        != int(calibration_receipt_raw["context_device_memory_bytes"])
        or int(sizing["calibration_ordinary_device_input_bytes"])
        != int(calibration_receipt_raw["ordinary_device_input_bytes"])
        or int(sizing["calibration_ordinary_device_output_bytes"])
        != int(calibration_receipt_raw["ordinary_device_output_bytes"])
        or int(sizing["calibration_external_device_output_bytes"])
        != int(calibration_receipt_raw["external_device_output_bytes"])
        or int(sizing["calibration_graph_private_device_bytes"])
        != int(calibration_receipt_raw["graph_private_device_bytes"])
    ):
        raise RuntimeError("controlled reservation sizing formula mismatch")

    expected_snapshot_r = min(
        baseline_r,
        _fraction_budget_bytes(
            auto_fraction,
            max(
                0,
                int(constrained_receipt_raw["capacity_decision_free_bytes"])
                - safety_reserve_bytes
                - module_residency_reserve_bytes
                - int(
                    constrained_receipt_raw[
                        "final_non_kv_overhead_delta_bytes"
                    ]
                ),
            ),
        )
        // baseline_receipt["B"],
    )
    if constrained_r != expected_snapshot_r:
        raise RuntimeError(
            "controlled R does not match the recorded final memory snapshot: "
            f"actual={constrained_r}, expected={expected_snapshot_r}"
        )

    expected_kv_delta = (baseline_r - constrained_r) * baseline_receipt["B"]
    actual_kv_delta = (
        baseline_receipt["kv_reserved_bytes"] - constrained_receipt["kv_reserved_bytes"]
    )
    if actual_kv_delta != expected_kv_delta:
        raise RuntimeError(
            "controlled reservation KV slope mismatch: "
            f"actual={actual_kv_delta}, expected={expected_kv_delta}"
        )

    guard = proof.get("guard")
    bulk = proof.get("bulk")
    if not isinstance(guard, dict) or not isinstance(bulk, dict):
        raise RuntimeError("controlled reservation misses guard or bulk receipt")
    guard_bytes = int(guard.get("bytes", 0))
    guard_allocations = guard.get("allocations")
    guard_allocation_count = int(guard.get("allocation_count", 0))
    if (
        guard.get("allocation_phase") != before_planning_phase
        or guard.get("release_after_snapshot_phase") != after_overhead_phase
        or guard_bytes != expected_guard_bytes
        or int(guard.get("address", 0)) == 0
        or guard_allocation_count != 1
        or not isinstance(guard_allocations, list)
        or len(guard_allocations) != 1
        or int(guard_allocations[0].get("address", 0)) != int(guard["address"])
        or int(guard_allocations[0].get("bytes", 0)) != guard_bytes
        or guard_bytes < target_kv_bytes + request_completion_headroom_bytes
    ):
        raise RuntimeError("controlled contiguous guard receipt is inconsistent")

    def parse_controlled_sample(
        value: Any,
        *,
        role: str,
    ) -> dict[str, Any]:
        return _parse_memory_sample(
            value,
            sampler=sampler,
            role=role,
        )

    def same_sample(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
        return all(lhs[field] == rhs[field] for field in _MEMORY_SAMPLE_FIELDS)

    def validate_action_delta(
        action_before: Mapping[str, Any],
        action_after: Mapping[str, Any],
        *,
        expected_process_and_device_growth_bytes: int,
        label: str,
    ) -> dict[str, Any]:
        attribution = _signed_memory_attribution(action_before, action_after)
        process_delta = int(attribution["nvml_current_process_growth_bytes"])
        device_delta = int(attribution["cuda_device_growth_bytes"])
        device_tolerance = max(
            tolerance_bytes,
            int(attribution["tolerance_bytes"]),
        )
        passed = bool(
            attribution["passed"]
            and abs(process_delta - expected_process_and_device_growth_bytes) <= tolerance_bytes
            and abs(device_delta - expected_process_and_device_growth_bytes) <= device_tolerance
        )
        if not passed:
            raise RuntimeError(
                f"{label} memory action delta mismatch: "
                f"process={process_delta}, device={device_delta}, "
                f"expected={expected_process_and_device_growth_bytes}, "
                f"process_tolerance={tolerance_bytes}, "
                f"device_tolerance={device_tolerance}, attribution={attribution}"
            )
        return {
            "label": label,
            "expected_signed_delta_bytes": expected_process_and_device_growth_bytes,
            "process_signed_delta_bytes": process_delta,
            "device_wide_signed_delta_bytes": device_delta,
            "process_tolerance_bytes": tolerance_bytes,
            "device_tolerance_bytes": device_tolerance,
            "signed_attribution": attribution,
            "passed": True,
        }

    before = parse_controlled_sample(
        proof.get("before_reservation"),
        role="controlled before-reservation",
    )
    after = parse_controlled_sample(
        proof.get("after_reservation"),
        role="controlled after-reservation",
    )
    guard_before_allocation = parse_controlled_sample(
        guard.get("before_allocation"),
        role="controlled guard before-allocation",
    )
    after_guard = parse_controlled_sample(
        guard.get("after_allocation"),
        role="controlled guard after-allocation",
    )
    bulk_before_allocation = parse_controlled_sample(
        bulk.get("before_allocation"),
        role="controlled bulk before-allocation",
    )
    bulk_after_allocation = parse_controlled_sample(
        bulk.get("after_allocation"),
        role="controlled bulk after-allocation",
    )
    constrained_before_planning = constrained_phases[1]
    constrained_after_overhead = constrained_phases[2]
    constrained_after_kv = constrained_phases[3]
    constrained_after_request = constrained_phases[4]
    if (
        not same_sample(guard_before_allocation, before)
        or not same_sample(bulk_before_allocation, after_guard)
        or not same_sample(bulk_after_allocation, after)
        or not same_sample(constrained_before_planning, after)
    ):
        raise RuntimeError("controlled reservation allocation samples are inconsistent")
    guard_allocation_gate = validate_action_delta(
        before,
        after_guard,
        expected_process_and_device_growth_bytes=guard_bytes,
        label="controlled contiguous-guard allocation",
    )
    observed_guard_allocation = guard_allocation_gate["process_signed_delta_bytes"]
    initial_bulk_allocation_gate = validate_action_delta(
        after_guard,
        after,
        expected_process_and_device_growth_bytes=int(bulk.get("initial_bytes", 0)),
        label="controlled initial bulk allocation",
    )
    guard_before_release = parse_controlled_sample(
        guard.get("before_release"),
        role="controlled guard before-release",
    )
    guard_after_release = parse_controlled_sample(
        guard.get("after_release"),
        role="controlled guard after-release",
    )
    if not same_sample(guard_before_release, constrained_after_overhead):
        raise RuntimeError("guard release did not follow the recorded final snapshot")
    guard_release_gate = validate_action_delta(
        guard_before_release,
        guard_after_release,
        expected_process_and_device_growth_bytes=-guard_bytes,
        label="controlled contiguous-guard release",
    )
    observed_guard_release = -guard_release_gate["process_signed_delta_bytes"]
    runtime_kv_allocation_gate = validate_action_delta(
        guard_after_release,
        constrained_after_kv,
        expected_process_and_device_growth_bytes=constrained_receipt["kv_reserved_bytes"],
        label="controlled runtime KV allocation",
    )
    observed_kv_device_bytes = runtime_kv_allocation_gate["device_wide_signed_delta_bytes"]
    observed_kv_process_bytes = runtime_kv_allocation_gate["process_signed_delta_bytes"]
    constrained_request_device_bytes = max(
        0,
        int(constrained_after_kv["free_bytes"]) - int(constrained_after_request["free_bytes"]),
    )
    constrained_request_process_bytes = max(
        0,
        _process_used(constrained_after_request) - _process_used(constrained_after_kv),
    )
    constrained_request_external_delta_bytes = (
        constrained_request_device_bytes - constrained_request_process_bytes
    )
    if constrained_request_process_bytes > request_completion_process_bytes + tolerance_bytes:
        raise RuntimeError(
            "controlled request per-process NVML growth exceeded calibrated "
            "process growth plus tolerance"
        )

    phases = [
        sample.get("phase")
        for sample in constrained["runtime_phase_memory_samples"]
        if isinstance(sample, dict)
    ]
    if not (
        phases.index(after_overhead_phase)
        < phases.index(after_kv_phase)
        < phases.index(after_request_phase)
    ):
        raise RuntimeError("controlled runtime phase order is invalid")
    if int(constrained_receipt_raw["post_load_free_bytes"]) != int(after["free_bytes"]) or int(
        constrained_receipt_raw["capacity_decision_free_bytes"]
    ) != int(guard_before_release["free_bytes"]):
        raise RuntimeError("controlled receipt does not bind planner snapshots")

    bulk_bytes = int(bulk.get("bytes", 0))
    initial_bulk_bytes = int(bulk.get("initial_bytes", 0))
    bulk_correction_bytes = int(bulk.get("correction_bytes", -1))
    bulk_released_correction_bytes = int(bulk.get("released_correction_bytes", -1))
    bulk_correction_attempts = int(bulk.get("correction_attempts", -1))
    max_correction_attempts = int(bulk.get("max_correction_attempts", -1))
    corrections = bulk.get("corrections")
    final_feedback = bulk.get("final_feedback")
    bulk_allocations = bulk.get("allocations")
    bulk_allocation_count = int(bulk.get("allocation_count", 0))
    expected_initial_bulk_bytes = (
        (int(after_guard["free_bytes"]) - required_visible_post_load_free) // alignment
    ) * alignment
    if (
        bulk.get("allocation_phase") != before_planning_phase
        or bulk.get("final_feedback_phase") != after_overhead_phase
        or bulk.get("release_phase") != "after constrained pipeline unload"
        or initial_bulk_bytes != expected_initial_bulk_bytes
        or bulk_bytes != initial_bulk_bytes + bulk_correction_bytes - bulk_released_correction_bytes
        or bulk_bytes <= 0
        or bulk_correction_bytes < 0
        or bulk_correction_bytes % alignment != 0
        or bulk_released_correction_bytes < 0
        or bulk_released_correction_bytes % alignment != 0
        or bulk_correction_attempts < 0
        or bulk_correction_attempts > CONTROLLED_MAX_CORRECTION_ATTEMPTS
        or max_correction_attempts != CONTROLLED_MAX_CORRECTION_ATTEMPTS
        or not isinstance(corrections, list)
        or len(corrections) != bulk_correction_attempts
        or not isinstance(final_feedback, dict)
        or int(bulk.get("address", 0)) == 0
        or not isinstance(bulk_allocations, list)
        or len(bulk_allocations) != bulk_allocation_count
        or bulk_allocation_count <= 0
        or any(int(item.get("address", 0)) == 0 for item in bulk_allocations)
        or sum(int(item.get("bytes", 0)) for item in bulk_allocations) != bulk_bytes
    ):
        raise RuntimeError("controlled bulk reservation receipt is inconsistent")
    total_preplanning_reservation_bytes = guard_bytes + initial_bulk_bytes
    total_reservation_bytes = guard_bytes + bulk_bytes
    observed_process_reservation = _process_used(after) - _process_used(before)
    if abs(observed_process_reservation - total_preplanning_reservation_bytes) > tolerance_bytes:
        raise RuntimeError(
            "NVML controlled-reservation delta mismatch: "
            f"actual={observed_process_reservation}, "
            f"expected={total_preplanning_reservation_bytes}, "
            f"tolerance={tolerance_bytes}"
        )
    observed_free_delta = int(before["free_bytes"]) - int(after["free_bytes"])
    minimum_initial_free = logical_context_output_bytes + final_free_upper_bound
    if int(after["free_bytes"]) < minimum_initial_free:
        raise RuntimeError("controlled preplanning memory misses conservative headroom")

    if not isinstance(final_feedback, dict):
        raise RuntimeError("controlled final-feedback receipt is not an object")
    controller_final_sample = parse_controlled_sample(
        final_feedback.get("controller_final_sample"),
        role="controlled feedback controller-final",
    )
    actual_final_snapshot = parse_controlled_sample(
        final_feedback.get("actual_final_snapshot"),
        role="controlled feedback actual-final",
    )
    if (
        final_feedback.get("phase") != after_overhead_phase
        or int(final_feedback.get("lower_bound_bytes", -1)) != final_free_lower_bound
        or int(final_feedback.get("upper_bound_bytes", -1)) != final_free_upper_bound
        or int(final_feedback.get("max_attempts", -1)) != CONTROLLED_MAX_CORRECTION_ATTEMPTS
        or int(final_feedback.get("attempts", -1)) != bulk_correction_attempts
        or int(final_feedback.get("allocated_bytes", -1)) != bulk_correction_bytes
        or int(final_feedback.get("released_bytes", -1)) != bulk_released_correction_bytes
        or final_feedback.get("converged") is not True
        or not same_sample(actual_final_snapshot, guard_before_release)
    ):
        raise RuntimeError("controlled final-feedback receipt is inconsistent")

    running_reserved_bytes = initial_bulk_bytes
    previous_after: dict[str, Any] | None = None
    allocated_sum = 0
    released_sum = 0
    correction_action_gates: list[dict[str, Any]] = []
    controller_start_sample: dict[str, Any] | None = None
    for index, correction in enumerate(corrections):
        if not isinstance(correction, dict):
            raise RuntimeError("controlled correction evidence row is not an object")
        before_correction = parse_controlled_sample(
            correction.get("before"),
            role=f"controlled correction {index} before",
        )
        after_correction = parse_controlled_sample(
            correction.get("after"),
            role=f"controlled correction {index} after",
        )
        if controller_start_sample is None:
            controller_start_sample = before_correction
        if previous_after is not None and not same_sample(previous_after, before_correction):
            raise RuntimeError("controlled correction evidence samples are not contiguous")
        direction = correction.get("direction")
        allocated_bytes = int(correction.get("allocated_bytes", -1))
        released_bytes = int(correction.get("released_bytes", -1))
        deficit_bytes = int(correction.get("deficit_bytes", -1))
        excess_bytes = int(correction.get("excess_bytes", -1))
        reserved_before = int(correction.get("cumulative_reserved_bytes_before", -1))
        reserved_after = int(correction.get("cumulative_reserved_bytes_after", -1))
        before_free = int(before_correction["free_bytes"])
        if (
            int(correction.get("attempt_index", -1)) != index
            or correction.get("status") != "completed"
            or reserved_before != running_reserved_bytes
            or int(after_correction["free_bytes"]) == before_free
        ):
            raise RuntimeError("controlled correction evidence row is inconsistent")
        if direction == "allocate":
            expected_excess = before_free - final_free_upper_bound + 1
            expected_allocation = _align_up(expected_excess, alignment)
            if (
                before_free < final_free_upper_bound
                or excess_bytes != expected_excess
                or deficit_bytes != 0
                or allocated_bytes != expected_allocation
                or released_bytes != 0
                or allocated_bytes <= 0
                or allocated_bytes % alignment != 0
                or reserved_after != reserved_before + allocated_bytes
            ):
                raise RuntimeError("controlled allocate-correction evidence is inconsistent")
            allocated_sum += allocated_bytes
            expected_action_delta = allocated_bytes
        elif direction == "release":
            expected_deficit = final_free_lower_bound - before_free
            if (
                before_free >= final_free_lower_bound
                or deficit_bytes != expected_deficit
                or excess_bytes != 0
                or allocated_bytes != 0
                or released_bytes <= 0
                or released_bytes % alignment != 0
                or reserved_after != reserved_before - released_bytes
            ):
                raise RuntimeError("controlled release-correction evidence is inconsistent")
            released_sum += released_bytes
            expected_action_delta = -released_bytes
        else:
            raise RuntimeError("controlled correction evidence has unknown direction")
        correction_action_gates.append(
            validate_action_delta(
                before_correction,
                after_correction,
                expected_process_and_device_growth_bytes=expected_action_delta,
                label=f"controlled correction {index} {direction}",
            )
        )
        running_reserved_bytes = reserved_after
        previous_after = after_correction

    if controller_start_sample is None:
        controller_start_sample = controller_final_sample
    if previous_after is not None and not same_sample(previous_after, controller_final_sample):
        raise RuntimeError("controlled correction evidence misses its final sample")
    constrained_context_output_bytes = sum(
        int(constrained_receipt_raw[field])
        for field in (
            "context_device_memory_bytes",
            "ordinary_device_input_bytes",
            "ordinary_device_output_bytes",
            "external_device_output_bytes",
            "graph_private_device_bytes",
        )
    )
    feedback_start_gate = validate_action_delta(
        constrained_before_planning,
        controller_start_sample,
        expected_process_and_device_growth_bytes=constrained_context_output_bytes,
        label="controlled context/output allocation before feedback",
    )
    final_snapshot_continuity = _validate_sample_continuity(
        controller_final_sample,
        actual_final_snapshot,
        label="controlled feedback to actual final snapshot",
    )
    if abs(int(final_snapshot_continuity["process_signed_delta_bytes"])) > tolerance_bytes or abs(
        int(final_snapshot_continuity["device_wide_signed_delta_bytes"])
    ) > max(tolerance_bytes, MEMORY_ATTRIBUTION_FLOOR_BYTES):
        raise RuntimeError(
            "controlled feedback and actual final snapshot are not contiguous "
            f"within tolerance: {final_snapshot_continuity}"
        )
    if (
        running_reserved_bytes != bulk_bytes
        or allocated_sum != bulk_correction_bytes
        or released_sum != bulk_released_correction_bytes
        or not (
            final_free_lower_bound
            <= int(controller_final_sample["free_bytes"])
            < final_free_upper_bound
        )
        or not (
            final_free_lower_bound
            <= int(actual_final_snapshot["free_bytes"])
            < final_free_upper_bound
        )
    ):
        raise RuntimeError("controlled final feedback did not reach the exact window")

    constrained_before_load = controlled_lifetime_samples["constrained"]["before_load"]
    constrained_after_unload = controlled_lifetime_samples["constrained"]["after_unload"]
    constrained_raw_retained = _process_used(constrained_after_unload) - _process_used(
        constrained_before_load
    )
    constrained_retained = constrained_raw_retained - bulk_bytes
    if abs(constrained_retained) > tolerance_bytes:
        raise RuntimeError(
            "constrained pipeline lifetime retained process memory after "
            f"excluding held bulk reservation: {constrained_retained} bytes"
        )
    after_constrained = parse_controlled_sample(
        proof.get("after_constrained_unload"),
        role="controlled after-constrained-unload",
    )
    after_release = parse_controlled_sample(
        proof.get("after_release"),
        role="controlled after-release",
    )
    bulk_before_release = parse_controlled_sample(
        bulk.get("before_release"),
        role="controlled bulk before-release",
    )
    bulk_after_release = parse_controlled_sample(
        bulk.get("after_release"),
        role="controlled bulk after-release",
    )
    if (
        not same_sample(bulk_before_release, after_constrained)
        or not same_sample(bulk_after_release, after_release)
        or not same_sample(after_constrained, constrained_after_unload)
    ):
        raise RuntimeError("bulk reservation release samples are inconsistent")
    bulk_release_gate = validate_action_delta(
        after_constrained,
        after_release,
        expected_process_and_device_growth_bytes=-bulk_bytes,
        label="controlled bulk release",
    )
    observed_release = -bulk_release_gate["process_signed_delta_bytes"]
    recovery_attribution = _signed_memory_attribution(
        constrained_before_load,
        after_release,
    )
    recovery = int(recovery_attribution["nvml_current_process_growth_bytes"])
    recovery_device = int(recovery_attribution["cuda_device_growth_bytes"])
    recovery_positive_limit = max(
        tolerance_bytes,
        MEMORY_ATTRIBUTION_FLOOR_BYTES,
    )
    recovery_passed = bool(
        recovery_attribution["components_passed"]
        and abs(recovery) <= tolerance_bytes
        and recovery_device <= recovery_positive_limit
        and int(recovery_attribution["unlisted_external_growth_bytes"]) <= recovery_positive_limit
    )
    recovery_gate = {
        "label": "controlled final recovery",
        "process_signed_delta_bytes": recovery,
        "device_wide_signed_delta_bytes": recovery_device,
        "positive_retention_limit_bytes": recovery_positive_limit,
        "negative_device_release_allowed": True,
        "signed_attribution": recovery_attribution,
        "passed": recovery_passed,
    }
    if not recovery_passed:
        raise RuntimeError(
            "controlled reservation did not return to process/device baseline: "
            f"process_delta={recovery}, device_delta={recovery_device}, "
            f"tolerance={tolerance_bytes}"
        )
    return {
        "memory_sampler": sampler,
        "warmup": warmup,
        "calibration": calibration,
        "warmup_retained_process_bytes": expected_warmup_retained,
        "target_tokens": target_tokens,
        "sizing": sizing,
        "guard": guard,
        "bulk": bulk,
        "initial_bulk_reservation_bytes": initial_bulk_bytes,
        "bulk_correction_bytes": bulk_correction_bytes,
        "bulk_released_correction_bytes": bulk_released_correction_bytes,
        "bulk_correction_attempts": bulk_correction_attempts,
        "controlled_lifetime_memory_evidence": controlled_lifetime_samples,
        "guard_allocation_gate": guard_allocation_gate,
        "initial_bulk_allocation_gate": initial_bulk_allocation_gate,
        "guard_release_gate": guard_release_gate,
        "runtime_kv_allocation_gate": runtime_kv_allocation_gate,
        "feedback_start_gate": feedback_start_gate,
        "correction_action_gates": correction_action_gates,
        "final_snapshot_continuity_gate": final_snapshot_continuity,
        "bulk_release_gate": bulk_release_gate,
        "recovery_gate": recovery_gate,
        "final_free_lower_bound_bytes": final_free_lower_bound,
        "final_free_upper_bound_bytes": final_free_upper_bound,
        "total_preplanning_reservation_bytes": total_preplanning_reservation_bytes,
        "total_reservation_bytes": total_reservation_bytes,
        "free_before_reservation_bytes": int(before["free_bytes"]),
        "free_after_reservation_bytes": int(after["free_bytes"]),
        "observed_device_free_delta_bytes": observed_free_delta,
        "observed_process_reservation_bytes": observed_process_reservation,
        "observed_guard_allocation_bytes": observed_guard_allocation,
        "observed_guard_release_bytes": observed_guard_release,
        "observed_runtime_kv_device_bytes": observed_kv_device_bytes,
        "observed_runtime_kv_process_bytes": observed_kv_process_bytes,
        "constrained_request_device_bytes": constrained_request_device_bytes,
        "constrained_request_process_bytes": constrained_request_process_bytes,
        "constrained_request_external_delta_bytes": (constrained_request_external_delta_bytes),
        "calibration_request_device_bytes": request_completion_device_bytes,
        "calibration_request_process_bytes": request_completion_process_bytes,
        "calibration_request_external_delta_bytes": (request_completion_external_delta_bytes),
        "request_completion_guard_basis": request_completion_guard_basis,
        "request_completion_hard_gate_basis": (
            "per-process NVML constrained growth <= calibration per-process growth + tolerance"
        ),
        "baseline_r": baseline_r,
        "constrained_r": constrained_r,
        "r_delta": baseline_r - constrained_r,
        "kv_bytes_per_token": baseline_receipt["B"],
        "actual_kv_delta_bytes": actual_kv_delta,
        "expected_kv_delta_bytes": expected_kv_delta,
        "final_kv_position": int(trace["final_kv_position"]),
        "constrained_lifetime_raw_retained_bytes": constrained_raw_retained,
        "constrained_lifetime_retained_bytes": constrained_retained,
        "observed_bulk_release_bytes": observed_release,
        "release_recovery_process_bytes": recovery,
        "release_recovery_device_bytes": recovery_device,
        "tolerance_bytes": tolerance_bytes,
        "baseline": baseline,
        "constrained": constrained,
        "passed": True,
    }


def _write_raw_trace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("build-dynkv/trtmc_dynamic_memory_qualify"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r1", type=int)
    parser.add_argument("--r2", type=int)
    parser.add_argument("--reservation-target-tokens", type=int)
    parser.add_argument("--tolerance-bytes", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    runner = args.runner.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_state_pre = _source_state_snapshot(output.parent, label="pre")
    raw_dir = output.parent / "soak-raw"
    header = boundary._read_bundle_header(bundle)
    spec = boundary._resolve_spec(header)
    runtime_memory_contract = _sealed_runtime_memory_contract(
        bundle,
        header,
    )
    controlled_requirement = controlled_reservation_requirement(
        spec.model_id,
        args.reservation_target_tokens,
    )
    r1 = args.r1 or spec.chunk_limit
    r2 = args.r2 or min(spec.context_limit, 2 * r1)
    if not (4 <= r1 < r2 <= spec.context_limit):
        raise ValueError("require 4 <= R1 < R2 <= model context limit")
    if controlled_requirement["required"] and not (
        66 <= args.reservation_target_tokens < spec.context_limit
    ):
        raise ValueError(
            "reservation target must fit the qualification request and be "
            "below the model context limit"
        )

    vocab_size = int(header["vocab_size"])
    with tempfile.TemporaryDirectory(prefix="trtmc-memory-soak-") as temporary:
        work = Path(temporary)
        token_file = work / "tokens.txt"
        token_count = min(64, r1 - 2)
        boundary._write_tokens(
            token_file, boundary.deterministic_token_ids(token_count, vocab_size)
        )
        sequential = _run(
            runner,
            bundle,
            token_file,
            work / "sequential.bin",
            r1,
            repeat=100,
        )
        _write_raw_trace(raw_dir / "sequential-100.json", sequential)
        load_cycles = _run(
            runner,
            bundle,
            token_file,
            work / "load-cycles.bin",
            r1,
            load_cycles=20,
        )
        _write_raw_trace(raw_dir / "load-unload-20.json", load_cycles)
        two_r = _run(
            runner,
            bundle,
            token_file,
            work / "two-r.bin",
            r1,
            second_runtime_tokens=r2,
        )
        _write_raw_trace(raw_dir / "same-process-two-r.json", two_r)
        controlled_reservation = None
        if controlled_requirement["required"]:
            controlled_reservation = _run(
                runner,
                bundle,
                token_file,
                work / "controlled-reservation.bin",
                None,
                controlled_reservation_target_tokens=(args.reservation_target_tokens),
            )
            _write_raw_trace(
                raw_dir / "controlled-reservation.json",
                controlled_reservation,
            )

    raw_traces = {
        "sequential_100": str(raw_dir / "sequential-100.json"),
        "load_unload_20": str(raw_dir / "load-unload-20.json"),
        "same_process_two_r": str(raw_dir / "same-process-two-r.json"),
    }
    if controlled_reservation is not None:
        raw_traces["controlled_reservation"] = str(raw_dir / "controlled-reservation.json")
    module_residency_replays = {
        "sequential_100": validate_trace_module_residency(
            sequential,
            contract=runtime_memory_contract,
            label="sequential_100",
        ),
        "load_unload_20": validate_trace_module_residency(
            load_cycles,
            contract=runtime_memory_contract,
            label="load_unload_20",
        ),
        "same_process_two_r": validate_trace_module_residency(
            two_r,
            contract=runtime_memory_contract,
            label="same_process_two_r",
        ),
    }
    if controlled_reservation is not None:
        module_residency_replays["controlled_reservation"] = (
            validate_trace_module_residency(
                controlled_reservation,
                contract=runtime_memory_contract,
                label="controlled_reservation",
            )
        )
    sequential_result = validate_sequential_requests(
        sequential,
        expected_count=100,
        tolerance_bytes=args.tolerance_bytes,
    )
    load_cycles_result = validate_load_cycles(
        load_cycles,
        expected_count=20,
        expected_r=r1,
        tolerance_bytes=args.tolerance_bytes,
    )
    two_r_result = validate_same_process_two_r(
        two_r,
        r1=r1,
        r2=r2,
        tolerance_bytes=args.tolerance_bytes,
    )
    if controlled_requirement["required"]:
        if controlled_reservation is None:
            raise RuntimeError("required controlled-reservation trace was not produced")
        controlled_result = validate_controlled_reservation(
            controlled_reservation,
            tolerance_bytes=args.tolerance_bytes,
        )
        controlled_gate = {
            **controlled_requirement,
            "status": "passed" if controlled_result.get("passed") is True else "failed",
            "passed": controlled_result.get("passed") is True,
            "evidence": controlled_result,
        }
    else:
        controlled_gate = {
            **controlled_requirement,
            "passed": True,
            "evidence": None,
        }
    qualification_gates = {
        "sequential_100_passed": sequential_result.get("passed") is True,
        "load_unload_20_passed": load_cycles_result.get("passed") is True,
        "same_process_two_r_passed": two_r_result.get("passed") is True,
        "sealed_module_residency_replayed_for_every_trace": all(
            replay.get("passed") is True
            for replay in module_residency_replays.values()
        ),
        "controlled_external_reservation_present_and_passed_or_not_applicable": (
            controlled_gate["passed"] is True
            and (
                controlled_gate["status"] == "passed"
                if controlled_gate["required"]
                else controlled_gate["status"] == "not_applicable"
            )
        ),
    }
    report = {
        "schema_version": 2,
        "model_id": spec.model_id,
        "bundle": str(bundle),
        "bundle_sha256": boundary._sha256(bundle),
        "bundle_runtime_memory_contract": runtime_memory_contract,
        "bundle_runtime_memory_contract_sha256": _canonical_sha256(
            runtime_memory_contract
        ),
        "runner": str(runner),
        "runner_sha256": boundary._sha256(runner),
        "raw_traces": raw_traces,
        "module_residency_replays": module_residency_replays,
        "memory_sampler": validate_nvml_sampler(sequential),
        "sequential_requests": sequential_result,
        "load_unload_cycles": load_cycles_result,
        "two_r_allocation_slope": two_r_result,
        "controlled_external_reservation_gate": controlled_gate,
        "qualification_gates": qualification_gates,
        "passed": all(qualification_gates.values()),
    }
    if controlled_gate["required"]:
        report["controlled_external_reservation"] = controlled_gate["evidence"]
    else:
        report["controlled_external_reservation"] = {
            "status": "not_applicable",
            "required": False,
            "passed": True,
            "not_applicable_reason": controlled_gate["not_applicable_reason"],
        }
    source_state_post = _source_state_snapshot(output.parent, label="post")
    source_state_unchanged = apply_source_state_gate(
        report,
        source_state_pre,
        source_state_post,
    )
    report["qualification_gates"]["source_state_unchanged"] = source_state_unchanged
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
