# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "qualify_native_dynamic_memory_soak.py"
SPEC = importlib.util.spec_from_file_location("qualify_native_dynamic_memory_soak", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
soak = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = soak
SPEC.loader.exec_module(soak)

pytestmark = pytest.mark.dynamic_memory


_TEST_DEVICE_TOTAL_BYTES = 16 * 1024 * 1024 * 1024


def _sample(
    process_used: int,
    *,
    device_used: int | None = None,
    other_process_used: int = 0,
) -> dict:
    if device_used is None:
        device_used = process_used + other_process_used
    assert device_used >= process_used + other_process_used
    processes = [{"pid": 123, "used_bytes": process_used}]
    if other_process_used:
        processes.append({"pid": 999, "used_bytes": other_process_used})
    return {
        "free_bytes": _TEST_DEVICE_TOTAL_BYTES - device_used,
        "total_bytes": _TEST_DEVICE_TOTAL_BYTES,
        "used_bytes": device_used,
        "process_used_bytes": process_used,
        "all_compute_process_used_bytes": process_used + other_process_used,
        "other_compute_process_used_bytes": other_process_used,
        "nvml_device_total_bytes": _TEST_DEVICE_TOTAL_BYTES,
        "nvml_device_reserved_bytes": 0,
        "nvml_device_free_bytes": _TEST_DEVICE_TOTAL_BYTES - device_used,
        "nvml_device_used_bytes": device_used,
        "post_nvml_free_bytes": _TEST_DEVICE_TOTAL_BYTES - device_used,
        "post_nvml_total_bytes": _TEST_DEVICE_TOTAL_BYTES,
        "compute_processes": processes,
    }


def _device_sample(*, total: int, free: int, process_used: int) -> dict:
    device_used = total - free
    assert device_used >= process_used
    return {
        "free_bytes": free,
        "total_bytes": total,
        "used_bytes": device_used,
        "process_used_bytes": process_used,
        "all_compute_process_used_bytes": process_used,
        "other_compute_process_used_bytes": 0,
        "nvml_device_total_bytes": total,
        "nvml_device_reserved_bytes": 0,
        "nvml_device_free_bytes": free,
        "nvml_device_used_bytes": device_used,
        "post_nvml_free_bytes": free,
        "post_nvml_total_bytes": total,
        "compute_processes": [{"pid": 123, "used_bytes": process_used}],
    }


def _phase(phase: str, sample: dict) -> dict:
    return {"phase": phase, "device": 0, **copy.deepcopy(sample)}


def _sampler() -> dict:
    return {
        "source": "nvmlDeviceGetComputeRunningProcesses_v3",
        "pid": 123,
        "cuda_logical_device_index": 0,
        "physical_device_index": 1,
        "pci_bus_id": "0000:01:00.0",
        "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "captures_all_compute_processes": True,
        "device_memory_source": "nvmlDeviceGetMemoryInfo_v2",
    }


def _receipt(r: int, b: int, allocation_id: int) -> dict:
    safety_reserve = 64 * 1024 * 1024
    capacity_free = safety_reserve + max(2 * r * b, 1)
    capacity_total = max(_TEST_DEVICE_TOTAL_BYTES, capacity_free + 1)
    settled_free = max(1, capacity_free - r * b)
    kv_budget = soak._fraction_budget_bytes(
        0.9,
        capacity_free - safety_reserve,
    )
    return {
        "receipt_schema_version": 3,
        "policy": "auto",
        "policy_fraction": 0.9,
        "requested_kv_bytes": 0,
        "safety_reserve_bytes": safety_reserve,
        "model_context_limit": r,
        "request_context_limit": 0,
        "effective_request_limit": r,
        "kv_budget_bytes": kv_budget,
        "pre_load_free_bytes": 9_000,
        "pre_load_total_bytes": capacity_total,
        "post_load_free_bytes": 8_000,
        "post_load_total_bytes": capacity_total,
        "capacity_decision_free_bytes": capacity_free,
        "capacity_decision_total_bytes": capacity_total,
        "capacity_decision_device_used_bytes": (capacity_total - capacity_free),
        "settled_free_bytes": settled_free,
        "settled_total_bytes": capacity_total,
        "settled_device_used_bytes": capacity_total - settled_free,
        "settled_snapshot_unavailable_reason": None,
        "final_free_bytes": capacity_free,
        "final_total_bytes": capacity_total,
        "final_device_used_bytes": capacity_total - capacity_free,
        "serialized_plan_bytes": 1,
        "resident_weight_bytes": 2,
        "resident_weight_copy_count": 1,
        "engine_weight_bytes": 2,
        "context_device_memory_bytes": 3,
        "ordinary_device_input_bytes": 6,
        "ordinary_device_output_bytes": 7,
        "external_device_output_bytes": 4,
        "host_staging_bytes": 5,
        "graph_private_device_bytes": 0,
        "kv_reserved_bytes": r * b,
        "kv_committed_bytes": r * b,
        "kv_metadata_bytes": 0,
        "peak_device_bytes": None,
        "backend_owned_cache_input_bytes": 0,
        "backend_owned_cache_output_bytes": 0,
        "kv_allocation_id": allocation_id,
        "kv_bytes_per_token": b,
        "runtime_kv_capacity_tokens": r,
    }


def _lifetime(
    *,
    r: int,
    b: int,
    allocation_id: int,
    label: str,
    measured: bool,
    before_load: dict,
    after_requests: dict,
    after_unload: dict,
    execution_ordinal: int | None = None,
    role: str | None = None,
    policy: dict | None = None,
) -> dict:
    if policy is None:
        policy = {"kind": "max_sequence_length", "requested_tokens": r}
    phases = [
        _phase("before runtime-memory test engine deserialization", copy.deepcopy(before_load)),
        _phase("before runtime KV planning", copy.deepcopy(before_load)),
        _phase("after shared context and output allocation", copy.deepcopy(before_load)),
        _phase("after runtime KV allocation", copy.deepcopy(after_requests)),
        _phase(
            "after successful runtime-memory request completion",
            copy.deepcopy(after_requests),
        ),
    ]
    receipt = _receipt(r, b, allocation_id)
    lifetime = {
        "label": label,
        "measured": measured,
        "policy": policy,
        "runtime_kv_capacity_tokens": r,
        "runtime_memory_receipt": receipt,
        "kv_allocation_id": allocation_id,
        "before_load": before_load,
        "after_requests": after_requests,
        "after_unload": after_unload,
        "process_growth_bytes": (
            after_requests["process_used_bytes"] - before_load["process_used_bytes"]
        ),
        "device_wide_growth_bytes": (after_requests["used_bytes"] - before_load["used_bytes"]),
        "retained_bytes": (after_unload["process_used_bytes"] - before_load["process_used_bytes"]),
        "device_wide_retained_bytes": (after_unload["used_bytes"] - before_load["used_bytes"]),
        "runtime_phase_memory_samples": phases,
    }
    if execution_ordinal is not None:
        lifetime["execution_ordinal"] = execution_ordinal
    if role is not None:
        lifetime["role"] = role
    _sync_receipt_snapshots(lifetime)
    return lifetime


def _load_cycle_trace(count: int = 2) -> dict:
    warmup = _lifetime(
        r=512,
        b=4,
        allocation_id=1,
        label="unmeasured-load-cycle-warmup",
        measured=False,
        before_load=_sample(1_000),
        after_requests=_sample(5_000),
        after_unload=_sample(1_500),
        execution_ordinal=0,
        role="warmup",
    )
    cycles = [
        {
            **_lifetime(
                r=512,
                b=4,
                allocation_id=index + 2,
                label="measured-load-cycle",
                measured=True,
                before_load=_sample(1_500),
                after_requests=_sample(4_500),
                after_unload=_sample(1_510),
                execution_ordinal=index + 1,
                role="measured",
            ),
            "cycle_index": index,
        }
        for index in range(count)
    ]
    return {
        "memory_sampler": _sampler(),
        "load_cycle_warmup": warmup,
        "load_cycle_count": count,
        "load_cycles": cycles,
    }


def _add_unlisted_device_bytes(sample: dict, amount: int) -> None:
    sample["free_bytes"] -= amount
    sample["used_bytes"] += amount
    sample["nvml_device_free_bytes"] -= amount
    sample["nvml_device_used_bytes"] += amount
    sample["post_nvml_free_bytes"] -= amount


def _add_current_process_bytes(sample: dict, amount: int) -> None:
    _add_unlisted_device_bytes(sample, amount)
    _add_current_process_ledger_bytes(sample, amount)


def _add_current_process_ledger_bytes(sample: dict, amount: int) -> None:
    sample["process_used_bytes"] += amount
    sample["all_compute_process_used_bytes"] += amount
    current = [row for row in sample["compute_processes"] if row["pid"] == 123]
    assert len(current) == 1
    current[0]["used_bytes"] += amount


def _sync_receipt_snapshots(lifetime: dict) -> None:
    phases = lifetime["runtime_phase_memory_samples"]
    receipt = lifetime["runtime_memory_receipt"]
    policy = lifetime["policy"]
    kind = policy["kind"]
    decision_free = phases[2]["free_bytes"]
    decision_total = phases[2]["total_bytes"]
    settled_free = phases[3]["free_bytes"]
    settled_total = phases[3]["total_bytes"]
    fraction = (
        float(policy["requested_fraction"])
        if kind == "fraction"
        else 0.0
        if kind == "bytes"
        else 0.9
    )
    requested_bytes = int(policy["requested_bytes"]) if kind == "bytes" else 0
    request_limit = int(policy["requested_tokens"]) if kind == "max_sequence_length" else 0
    safely_available = max(
        0,
        decision_free - int(receipt["safety_reserve_bytes"]),
    )
    kv_budget = (
        requested_bytes
        if kind == "bytes"
        else soak._fraction_budget_bytes(fraction, safely_available)
    )
    receipt.update(
        {
            "policy": "auto" if kind == "max_sequence_length" else kind,
            "policy_fraction": fraction,
            "requested_kv_bytes": requested_bytes,
            "request_context_limit": request_limit,
            "effective_request_limit": (receipt["runtime_kv_capacity_tokens"]),
            "kv_budget_bytes": kv_budget,
            "pre_load_free_bytes": phases[0]["free_bytes"],
            "pre_load_total_bytes": phases[0]["total_bytes"],
            "post_load_free_bytes": phases[1]["free_bytes"],
            "post_load_total_bytes": phases[1]["total_bytes"],
            "capacity_decision_free_bytes": decision_free,
            "capacity_decision_total_bytes": decision_total,
            "capacity_decision_device_used_bytes": (decision_total - decision_free),
            "settled_free_bytes": settled_free,
            "settled_total_bytes": settled_total,
            "settled_device_used_bytes": settled_total - settled_free,
            "settled_snapshot_unavailable_reason": None,
            "final_free_bytes": decision_free,
            "final_total_bytes": decision_total,
            "final_device_used_bytes": decision_total - decision_free,
        }
    )


def test_two_r_slope_requires_exact_contiguous_delta() -> None:
    small = {
        "R": 512,
        "B": 22_528,
        "kv_reserved_bytes": 512 * 22_528,
        "kv_allocation_id": 1,
    }
    large = {
        "R": 1_024,
        "B": 22_528,
        "kv_reserved_bytes": 1_024 * 22_528,
        "kv_allocation_id": 2,
    }

    assert soak.validate_two_r_slope(small, large)["passed"]
    large["kv_reserved_bytes"] += 1
    with pytest.raises(RuntimeError, match="slope mismatch"):
        soak.validate_two_r_slope(small, large)


def test_same_process_two_r_requires_nvml_and_distinct_lifetimes() -> None:
    r1, r2, b = 512, 1_024, 4
    cold_baseline = _sample(1_000)
    warm_baseline = _sample(1_010)
    warmup = _lifetime(
        r=r2,
        b=b,
        allocation_id=1,
        label="unmeasured-r2-warmup",
        measured=False,
        before_load=cold_baseline,
        after_requests=_sample(6_000),
        after_unload=warm_baseline,
    )
    lifetimes = [
        _lifetime(
            r=r1,
            b=b,
            allocation_id=2,
            label="measured-r1",
            measured=True,
            before_load=warm_baseline,
            after_requests=_sample(4_010),
            after_unload=warm_baseline,
        ),
        _lifetime(
            r=r2,
            b=b,
            allocation_id=3,
            label="measured-r2",
            measured=True,
            before_load=warm_baseline,
            after_requests=_sample(6_058),
            after_unload=warm_baseline,
        ),
    ]
    trace = {
        "mode": "same_process_two_r_allocation_slope",
        "memory_sampler": _sampler(),
        "allocation_slope_warmup": warmup,
        "allocation_slope_lifetimes": lifetimes,
    }

    result = soak.validate_same_process_two_r(trace, r1=r1, r2=r2, tolerance_bytes=32)
    assert result["passed"]
    assert result["nvml_actual_delta_bytes"] == (r2 - r1) * b
    assert result["cold_start_evidence"]["passed"]
    assert len(result["continuity_gates"]) == 2

    lifetimes[1]["runtime_memory_receipt"]["kv_allocation_id"] = 2
    lifetimes[1]["kv_allocation_id"] = 2
    with pytest.raises(RuntimeError, match="reused an allocation identity"):
        soak.validate_same_process_two_r(trace, r1=r1, r2=r2, tolerance_bytes=32)


def _controlled_reservation_trace() -> dict:
    total = 4_000_000_000
    alignment = soak.CONTROLLED_RESERVATION_ALIGNMENT_BYTES
    safety = 64 * 1024 * 1024
    target_tokens, baseline_r, b = 512, 1_024, 114_688
    target_kv_bytes = target_tokens * b
    measured_context_output_bytes = 4 * alignment
    logical_context_output_bytes = measured_context_output_bytes
    context_device_memory_bytes = alignment
    ordinary_device_input_bytes = alignment
    ordinary_device_output_bytes = alignment
    external_device_output_bytes = 0
    graph_private_device_bytes = alignment
    request_completion_headroom_bytes = 0
    constrained_request_device_bytes = 4 * 1024
    constrained_request_process_bytes = 16
    policy_safe_bytes = soak._ceil_divided_by_fraction(target_kv_bytes, 0.9)
    final_free_lower_bound = safety + policy_safe_bytes
    final_free_upper_bound = final_free_lower_bound + alignment
    required_visible_free = (
        logical_context_output_bytes
        + final_free_upper_bound
        + soak.CONTROLLED_PREPLANNING_HEADROOM_BYTES
    )
    guard_bytes = (
        math.ceil((target_kv_bytes + request_completion_headroom_bytes) / alignment) * alignment
    )

    process_baseline = _device_sample(
        total=total,
        free=3_000_000_000,
        process_used=10_000_000,
    )
    before_reservation = _device_sample(
        total=total,
        free=1_800_000_000,
        process_used=100_000_000,
    )
    after_guard = _device_sample(
        total=total,
        free=before_reservation["free_bytes"] - guard_bytes,
        process_used=before_reservation["process_used_bytes"] + guard_bytes,
    )
    initial_bulk_bytes = (
        (after_guard["free_bytes"] - required_visible_free) // alignment
    ) * alignment
    after_reservation = _device_sample(
        total=total,
        free=after_guard["free_bytes"] - initial_bulk_bytes,
        process_used=after_guard["process_used_bytes"] + initial_bulk_bytes,
    )
    final_feedback_before = _device_sample(
        total=total,
        free=(after_reservation["free_bytes"] - measured_context_output_bytes),
        process_used=(after_reservation["process_used_bytes"] + measured_context_output_bytes),
    )
    final_feedback_excess = final_feedback_before["free_bytes"] - final_free_upper_bound + 1
    final_feedback_allocation = math.ceil(final_feedback_excess / alignment) * alignment
    bulk_bytes = initial_bulk_bytes + final_feedback_allocation
    guard_before_release = _device_sample(
        total=total,
        free=(final_feedback_before["free_bytes"] - final_feedback_allocation),
        process_used=(final_feedback_before["process_used_bytes"] + final_feedback_allocation),
    )
    guard_after_release = _device_sample(
        total=total,
        free=guard_before_release["free_bytes"] + guard_bytes,
        process_used=guard_before_release["process_used_bytes"] - guard_bytes,
    )
    constrained_r = (
        soak._fraction_budget_bytes(
            0.9,
            max(
                0,
                guard_before_release["free_bytes"] - safety,
            ),
        )
        // b
    )
    constrained_kv_bytes = constrained_r * b
    constrained_after_kv = _device_sample(
        total=total,
        free=guard_after_release["free_bytes"] - constrained_kv_bytes,
        process_used=(guard_after_release["process_used_bytes"] + constrained_kv_bytes),
    )
    constrained_after_request = _device_sample(
        total=total,
        free=(constrained_after_kv["free_bytes"] - constrained_request_device_bytes),
        process_used=(
            constrained_after_kv["process_used_bytes"] + constrained_request_process_bytes
        ),
    )
    after_constrained_unload = _device_sample(
        total=total,
        free=process_baseline["free_bytes"] - bulk_bytes,
        process_used=process_baseline["process_used_bytes"] + bulk_bytes,
    )

    pre_engine_phase = "before runtime-memory controlled engine deserialization"
    before_planning_phase = "before runtime KV planning"
    after_overhead_phase = "after shared context and output allocation"
    after_kv_phase = "after runtime KV allocation"
    after_request_phase = "after successful runtime-memory request completion"

    calibration_before_planning = _device_sample(
        total=total,
        free=2_500_000_000,
        process_used=100_000_000,
    )
    calibration_after_overhead = _device_sample(
        total=total,
        free=(calibration_before_planning["free_bytes"] - measured_context_output_bytes),
        process_used=(
            calibration_before_planning["process_used_bytes"] + measured_context_output_bytes
        ),
    )
    calibration_after_kv = _device_sample(
        total=total,
        free=calibration_after_overhead["free_bytes"] - target_kv_bytes,
        process_used=(calibration_after_overhead["process_used_bytes"] + target_kv_bytes),
    )
    calibration_after_request = _device_sample(
        total=total,
        free=(calibration_after_kv["free_bytes"] - request_completion_headroom_bytes),
        process_used=(
            calibration_after_kv["process_used_bytes"] + request_completion_headroom_bytes
        ),
    )

    calibration_receipt = _receipt(target_tokens, b, 10)
    calibration_receipt.update(
        {
            "pre_load_free_bytes": process_baseline["free_bytes"],
            "post_load_free_bytes": calibration_before_planning["free_bytes"],
            "final_free_bytes": calibration_after_overhead["free_bytes"],
            "context_device_memory_bytes": context_device_memory_bytes,
            "ordinary_device_input_bytes": ordinary_device_input_bytes,
            "ordinary_device_output_bytes": ordinary_device_output_bytes,
            "external_device_output_bytes": (external_device_output_bytes),
            "graph_private_device_bytes": graph_private_device_bytes,
        }
    )
    baseline_receipt = _receipt(baseline_r, b, 11)
    baseline_receipt.update(
        {
            "pre_load_free_bytes": 3_100_000_000,
            "post_load_free_bytes": 1_900_000_000,
            "final_free_bytes": (1_900_000_000 - measured_context_output_bytes),
        }
    )
    constrained_receipt = _receipt(constrained_r, b, 12)
    constrained_receipt.update(
        {
            "pre_load_free_bytes": process_baseline["free_bytes"],
            "post_load_free_bytes": after_reservation["free_bytes"],
            "final_free_bytes": guard_before_release["free_bytes"],
            "context_device_memory_bytes": context_device_memory_bytes,
            "ordinary_device_input_bytes": ordinary_device_input_bytes,
            "ordinary_device_output_bytes": ordinary_device_output_bytes,
            "external_device_output_bytes": (external_device_output_bytes),
            "graph_private_device_bytes": graph_private_device_bytes,
        }
    )

    warmup_before_load = _device_sample(
        total=total,
        free=3_100_000_000,
        process_used=9_000_000,
    )
    warmup_after_requests = _device_sample(
        total=total,
        free=1_600_000_000,
        process_used=220_000_000,
    )
    warmup = {
        "policy": {"kind": "auto"},
        "measured": False,
        "before_load": warmup_before_load,
        "after_requests": warmup_after_requests,
        "after_unload": copy.deepcopy(process_baseline),
        "runtime_phase_memory_samples": [
            _phase(pre_engine_phase, warmup_before_load),
            _phase(before_planning_phase, before_reservation),
            _phase(after_overhead_phase, before_reservation),
            _phase(after_kv_phase, warmup_after_requests),
            _phase(after_request_phase, warmup_after_requests),
        ],
    }
    calibration = {
        "policy": {
            "kind": "max_sequence_length",
            "requested_tokens": target_tokens,
        },
        "measured": True,
        "runtime_memory_receipt": calibration_receipt,
        "before_load": copy.deepcopy(process_baseline),
        "after_requests": copy.deepcopy(calibration_after_request),
        "after_unload": copy.deepcopy(process_baseline),
        "runtime_phase_memory_samples": [
            _phase(pre_engine_phase, process_baseline),
            _phase(before_planning_phase, calibration_before_planning),
            _phase(after_overhead_phase, calibration_after_overhead),
            _phase(after_kv_phase, calibration_after_kv),
            _phase(after_request_phase, calibration_after_request),
        ],
    }
    baseline_after_requests = _device_sample(
        total=total,
        free=1_600_000_000,
        process_used=(process_baseline["process_used_bytes"] + baseline_r * b + 120_000_000),
    )
    baseline = {
        "policy": {"kind": "auto"},
        "measured": True,
        "runtime_memory_receipt": baseline_receipt,
        "before_load": copy.deepcopy(process_baseline),
        "after_requests": baseline_after_requests,
        "after_unload": copy.deepcopy(process_baseline),
        "runtime_phase_memory_samples": [
            _phase(pre_engine_phase, process_baseline),
            _phase(
                before_planning_phase,
                _device_sample(total=total, free=1_900_000_000, process_used=100_000_000),
            ),
            _phase(
                after_overhead_phase,
                _device_sample(
                    total=total,
                    free=1_900_000_000 - measured_context_output_bytes,
                    process_used=100_000_000 + measured_context_output_bytes,
                ),
            ),
            _phase(after_kv_phase, baseline_after_requests),
            _phase(after_request_phase, baseline_after_requests),
        ],
    }
    constrained = {
        "policy": {"kind": "auto"},
        "measured": True,
        "runtime_memory_receipt": constrained_receipt,
        "before_load": copy.deepcopy(process_baseline),
        "after_requests": copy.deepcopy(constrained_after_request),
        "after_unload": copy.deepcopy(after_constrained_unload),
        "runtime_phase_memory_samples": [
            _phase(pre_engine_phase, process_baseline),
            _phase(before_planning_phase, after_reservation),
            _phase(after_overhead_phase, guard_before_release),
            _phase(after_kv_phase, constrained_after_kv),
            _phase(after_request_phase, constrained_after_request),
        ],
    }
    for lifetime in (calibration, baseline, constrained):
        lifetime["runtime_memory_receipt"]["model_context_limit"] = baseline_r
        _sync_receipt_snapshots(lifetime)
    return {
        "mode": "same_process_controlled_external_reservation",
        "memory_sampler": _sampler(),
        "final_kv_position": 66,
        "controlled_reservation": {
            "target_tokens": target_tokens,
            "sizing": {
                "baseline_process_growth_bytes": (
                    baseline["after_requests"]["process_used_bytes"]
                    - baseline["before_load"]["process_used_bytes"]
                ),
                "baseline_kv_reserved_bytes": baseline_r * b,
                "estimated_non_kv_growth_bytes": 120_000_000,
                "baseline_pre_load_free_bytes": baseline_receipt["pre_load_free_bytes"],
                "baseline_post_load_free_bytes": baseline_receipt["post_load_free_bytes"],
                "baseline_engine_load_device_bytes": (
                    baseline_receipt["pre_load_free_bytes"]
                    - baseline_receipt["post_load_free_bytes"]
                ),
                "warmup_retained_process_bytes": 1_000_000,
                "warmup_retained_device_wide_bytes": 100_000_000,
                "required_free_basis": (
                    "calibration receipt logical context/output bytes plus exact final "
                    "target window and preplanning headroom"
                ),
                "auto_fraction": 0.9,
                "calibration_context_device_memory_bytes": (context_device_memory_bytes),
                "calibration_ordinary_device_input_bytes": (ordinary_device_input_bytes),
                "calibration_ordinary_device_output_bytes": (ordinary_device_output_bytes),
                "calibration_external_device_output_bytes": (external_device_output_bytes),
                "calibration_graph_private_device_bytes": (graph_private_device_bytes),
                "logical_context_output_bytes": logical_context_output_bytes,
                "measured_context_output_bytes": (measured_context_output_bytes),
                "request_completion_device_bytes": (request_completion_headroom_bytes),
                "request_completion_process_bytes": (request_completion_headroom_bytes),
                "request_completion_external_delta_bytes": 0,
                "request_completion_headroom_bytes": (request_completion_headroom_bytes),
                "request_completion_guard_basis": (
                    "max(calibration device-wide free delta, calibration per-process NVML delta)"
                ),
                "target_kv_bytes": target_kv_bytes,
                "safety_reserve_bytes": safety,
                "policy_safe_bytes": policy_safe_bytes,
                "policy_fraction_headroom_bytes": (policy_safe_bytes - target_kv_bytes),
                "reservation_alignment_bytes": alignment,
                "max_capacity_rounding_rows": (soak.CONTROLLED_TARGET_TOLERANCE_ROWS),
                "target_tolerance_rows": soak.CONTROLLED_TARGET_TOLERANCE_ROWS,
                "final_free_lower_bound_bytes": final_free_lower_bound,
                "final_free_upper_bound_bytes": final_free_upper_bound,
                "preplanning_headroom_bytes": (soak.CONTROLLED_PREPLANNING_HEADROOM_BYTES),
                "guard_bytes": guard_bytes,
                "required_visible_post_load_free_bytes": (required_visible_free),
                "visible_free_formula": (
                    "logical_context_output_bytes + final_free_upper_bound_bytes + "
                    "preplanning_headroom_bytes"
                ),
            },
            "before_reservation": copy.deepcopy(before_reservation),
            "after_reservation": copy.deepcopy(after_reservation),
            "guard": {
                "allocation_phase": before_planning_phase,
                "release_after_snapshot_phase": after_overhead_phase,
                "bytes": guard_bytes,
                "address": 4_096,
                "allocation_count": 1,
                "allocations": [
                    {
                        "index": 0,
                        "address": 4_096,
                        "bytes": guard_bytes,
                    }
                ],
                "before_allocation": copy.deepcopy(before_reservation),
                "after_allocation": copy.deepcopy(after_guard),
                "before_release": copy.deepcopy(guard_before_release),
                "after_release": copy.deepcopy(guard_after_release),
            },
            "bulk": {
                "allocation_phase": before_planning_phase,
                "final_feedback_phase": after_overhead_phase,
                "release_phase": "after constrained pipeline unload",
                "bytes": bulk_bytes,
                "initial_bytes": initial_bulk_bytes,
                "correction_bytes": final_feedback_allocation,
                "released_correction_bytes": 0,
                "correction_attempts": 1,
                "max_correction_attempts": (soak.CONTROLLED_MAX_CORRECTION_ATTEMPTS),
                "corrections": [
                    {
                        "attempt_index": 0,
                        "direction": "allocate",
                        "before": copy.deepcopy(final_feedback_before),
                        "after": copy.deepcopy(guard_before_release),
                        "deficit_bytes": 0,
                        "excess_bytes": final_feedback_excess,
                        "allocated_bytes": final_feedback_allocation,
                        "released_bytes": 0,
                        "cumulative_reserved_bytes_before": initial_bulk_bytes,
                        "cumulative_reserved_bytes_after": bulk_bytes,
                        "status": "completed",
                    }
                ],
                "final_feedback": {
                    "phase": after_overhead_phase,
                    "lower_bound_bytes": final_free_lower_bound,
                    "upper_bound_bytes": final_free_upper_bound,
                    "max_attempts": soak.CONTROLLED_MAX_CORRECTION_ATTEMPTS,
                    "attempts": 1,
                    "allocated_bytes": final_feedback_allocation,
                    "released_bytes": 0,
                    "converged": True,
                    "controller_final_sample": copy.deepcopy(guard_before_release),
                    "actual_final_snapshot": copy.deepcopy(guard_before_release),
                },
                "address": 8_192,
                "allocation_count": 2,
                "allocations": [
                    {
                        "index": 0,
                        "address": 8_192,
                        "bytes": initial_bulk_bytes,
                    },
                    {
                        "index": 1,
                        "address": 12_288,
                        "bytes": final_feedback_allocation,
                    },
                ],
                "before_allocation": copy.deepcopy(after_guard),
                "after_allocation": copy.deepcopy(after_reservation),
                "before_release": copy.deepcopy(after_constrained_unload),
                "after_release": copy.deepcopy(process_baseline),
            },
            "warmup": warmup,
            "calibration": calibration,
            "after_constrained_unload": copy.deepcopy(after_constrained_unload),
            "after_release": copy.deepcopy(process_baseline),
            "baseline_r": baseline_r,
            "constrained_r": constrained_r,
            "r_delta": baseline_r - constrained_r,
            "passed": True,
            "baseline": baseline,
            "constrained": constrained,
        },
    }


def test_controlled_reservation_reduces_auto_r_and_recovers() -> None:
    trace = _controlled_reservation_trace()
    result = soak.validate_controlled_reservation(trace, tolerance_bytes=32)
    assert result["passed"]
    assert result["target_tokens"] == 512
    assert result["constrained_r"] == 515
    assert result["observed_guard_allocation_bytes"] == 56 * 1024 * 1024
    assert result["observed_runtime_kv_device_bytes"] > result["observed_guard_allocation_bytes"]
    assert result["constrained_request_device_bytes"] == 4 * 1024
    assert result["constrained_request_process_bytes"] == 16
    assert result["constrained_request_external_delta_bytes"] == 4 * 1024 - 16
    assert result["request_completion_hard_gate_basis"].startswith("per-process NVML")


def test_qwen_promotion_requires_controlled_reservation() -> None:
    with pytest.raises(
        ValueError,
        match="Qwen full soak qualification requires",
    ):
        soak.controlled_reservation_requirement(
            "Qwen/Qwen3-0.6B",
            None,
        )

    requirement = soak.controlled_reservation_requirement(
        "Qwen/Qwen3-0.6B",
        512,
    )
    assert requirement == {
        "required": True,
        "status": "pending",
        "reservation_target_tokens": 512,
        "not_applicable_reason": None,
    }


def test_tiny_controlled_reservation_is_explicitly_not_applicable() -> None:
    requirement = soak.controlled_reservation_requirement(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        None,
    )
    assert requirement["required"] is False
    assert requirement["status"] == "not_applicable"
    assert "2,048-row KV slab" in requirement["not_applicable_reason"]

    with pytest.raises(
        ValueError,
        match="assigned only",
    ):
        soak.controlled_reservation_requirement(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            512,
        )


@pytest.mark.parametrize(
    "external_driver_delta",
    [100 * 1024 * 1024, -100 * 1024 * 1024],
)
def test_controlled_endpoint_records_signed_external_driver_delta(
    external_driver_delta: int,
) -> None:
    trace = _controlled_reservation_trace()
    _add_unlisted_device_bytes(
        trace["controlled_reservation"]["constrained"]["after_requests"],
        external_driver_delta,
    )

    result = soak.validate_controlled_reservation(trace, tolerance_bytes=32)
    gate = result["controlled_lifetime_memory_evidence"]["constrained"]["endpoint_binding_gates"][
        "request_completion_to_after_requests"
    ]
    assert gate["passed"]
    assert gate["external_driver_delta_bytes"] == external_driver_delta
    assert gate["external_driver_delta_is_runtime_action_bytes"] is False
    assert gate["runtime_phase_boundary_is_authoritative"] is True


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("calibration_r", "expected exactly"),
        ("guard_release_phase", "guard receipt"),
        ("bulk_bytes", "bulk reservation receipt"),
        ("bulk_correction_receipt", "bulk reservation receipt"),
        ("bulk_correction_row", "allocate-correction evidence"),
        ("bulk_correction_status", "correction evidence row"),
        ("bulk_correction_action_delta", "memory action delta mismatch"),
        ("controlled_phase_pid", "does not contain the sampler PID"),
        ("final_feedback_window", "exact window"),
        ("target_window", "near target"),
        ("planner_snapshot", "recorded final snapshot"),
        ("runtime_kv_malloc", "runtime KV allocation"),
        ("request_process_growth", "per-process NVML growth"),
        ("calibration_external_delta", "sizing formula"),
        ("missing_calibration_external_delta", "request-attribution fields"),
        ("missing_guard_basis", "request-attribution fields"),
        ("ordinary_device_bytes_omitted", "sizing formula"),
        ("missing_ordinary_device_bytes", "logical non-KV fields"),
        ("controlled_after_requests_endpoint", "endpoint binding failed"),
        ("release_recovery", "memory action delta"),
    ],
)
def test_controlled_reservation_rejects_tampered_receipts(
    tamper: str,
    match: str,
) -> None:
    trace = copy.deepcopy(_controlled_reservation_trace())
    proof = trace["controlled_reservation"]
    if tamper == "calibration_r":
        proof["calibration"]["runtime_memory_receipt"]["runtime_kv_capacity_tokens"] += 1
    elif tamper == "guard_release_phase":
        proof["guard"]["release_after_snapshot_phase"] = "after runtime KV allocation"
    elif tamper == "bulk_bytes":
        proof["bulk"]["bytes"] += soak.CONTROLLED_RESERVATION_ALIGNMENT_BYTES
    elif tamper == "bulk_correction_receipt":
        proof["bulk"]["correction_bytes"] += soak.CONTROLLED_RESERVATION_ALIGNMENT_BYTES
    elif tamper == "bulk_correction_row":
        proof["bulk"]["corrections"][0]["allocated_bytes"] += (
            soak.CONTROLLED_RESERVATION_ALIGNMENT_BYTES
        )
    elif tamper == "bulk_correction_status":
        proof["bulk"]["corrections"][0]["status"] = "applying"
    elif tamper == "bulk_correction_action_delta":
        _add_current_process_ledger_bytes(
            proof["bulk"]["corrections"][0]["before"],
            1024 * 1024 * 1024,
        )
    elif tamper == "controlled_phase_pid":
        phase = proof["constrained"]["runtime_phase_memory_samples"][2]
        phase["compute_processes"][0]["pid"] = 999
    elif tamper == "final_feedback_window":
        controller_sample = proof["bulk"]["final_feedback"]["controller_final_sample"]
        controller_sample["free_bytes"] = proof["sizing"]["final_free_upper_bound_bytes"]
        controller_sample["used_bytes"] = (
            controller_sample["total_bytes"] - controller_sample["free_bytes"]
        )
        proof["bulk"]["corrections"][0]["after"] = copy.deepcopy(controller_sample)
    elif tamper == "target_window":
        proof["constrained_r"] = (
            proof["target_tokens"] + proof["sizing"]["max_capacity_rounding_rows"] + 1
        )
    elif tamper == "planner_snapshot":
        _add_unlisted_device_bytes(proof["guard"]["before_release"], -1)
    elif tamper == "runtime_kv_malloc":
        _add_current_process_bytes(
            proof["constrained"]["runtime_phase_memory_samples"][3],
            -64,
        )
        _sync_receipt_snapshots(proof["constrained"])
    elif tamper == "request_process_growth":
        _add_current_process_bytes(
            proof["constrained"]["runtime_phase_memory_samples"][4],
            17,
        )
    elif tamper == "calibration_external_delta":
        proof["sizing"]["request_completion_external_delta_bytes"] += 1
    elif tamper == "missing_calibration_external_delta":
        proof["sizing"].pop("request_completion_external_delta_bytes")
    elif tamper == "missing_guard_basis":
        proof["sizing"].pop("request_completion_guard_basis")
    elif tamper == "ordinary_device_bytes_omitted":
        proof["sizing"]["logical_context_output_bytes"] -= (
            proof["sizing"]["calibration_ordinary_device_input_bytes"]
            + proof["sizing"]["calibration_ordinary_device_output_bytes"]
        )
    elif tamper == "missing_ordinary_device_bytes":
        proof["sizing"].pop("calibration_ordinary_device_input_bytes")
    elif tamper == "controlled_after_requests_endpoint":
        _add_current_process_bytes(
            proof["constrained"]["after_requests"],
            100 * 1024 * 1024,
        )
    elif tamper == "release_recovery":
        _add_current_process_bytes(proof["after_release"], 1_024)
        _add_current_process_bytes(proof["bulk"]["after_release"], 1_024)
    else:
        raise AssertionError(f"unknown tamper case: {tamper}")

    with pytest.raises(RuntimeError, match=match):
        soak.validate_controlled_reservation(trace, tolerance_bytes=32)


def test_sequential_validator_requires_stable_allocation_and_memory() -> None:
    samples = [
        {
            "request_index": index,
            "before": _sample(1_000),
            "after": _sample(1_100 + index),
            "kv_allocation_id": 9,
            "final_kv_position": 66,
        }
        for index in range(100)
    ]
    trace = {
        "memory_sampler": _sampler(),
        "sequential_request_count": 100,
        "sequential_requests": samples,
    }
    assert soak.validate_sequential_requests(trace, expected_count=100, tolerance_bytes=128)[
        "passed"
    ]

    samples[-1]["kv_allocation_id"] = 10
    with pytest.raises(RuntimeError, match="reuse one KV allocation"):
        soak.validate_sequential_requests(trace, expected_count=100, tolerance_bytes=128)


def test_sequential_validator_rejects_device_wide_growth_with_stable_process() -> None:
    step = 1024 * 1024
    process_used = 1_100
    samples = [
        {
            "request_index": index,
            "before": _sample(
                process_used,
                device_used=process_used + index * step,
            ),
            "after": _sample(
                process_used,
                device_used=process_used + (index + 1) * step,
            ),
            "kv_allocation_id": 9,
            "final_kv_position": 66,
        }
        for index in range(100)
    ]
    trace = {
        "memory_sampler": _sampler(),
        "sequential_request_count": 100,
        "sequential_requests": samples,
    }

    with pytest.raises(RuntimeError, match="fixed positive-growth envelope"):
        soak.validate_sequential_requests(
            trace,
            expected_count=100,
            tolerance_bytes=128,
        )


def test_sequential_validator_rejects_cumulative_growth_hidden_by_late_reset() -> None:
    step = 32 * 1024 * 1024
    base = 1_100
    current = base
    samples = []
    for index in range(100):
        before = current
        if index < 80:
            current += step
        elif index == 80:
            current = base
        samples.append(
            {
                "request_index": index,
                "before": _sample(before),
                "after": _sample(current),
                "kv_allocation_id": 9,
                "final_kv_position": 66,
            }
        )
    trace = {
        "memory_sampler": _sampler(),
        "sequential_request_count": 100,
        "sequential_requests": samples,
    }

    with pytest.raises(RuntimeError, match="fixed positive-growth envelope"):
        soak.validate_sequential_requests(
            trace,
            expected_count=100,
            tolerance_bytes=step,
        )


@pytest.mark.parametrize("release_index", [None, 50])
def test_sequential_fixed_baseline_allows_stable_and_external_release(
    release_index: int | None,
) -> None:
    base = 1_100
    external = 512 * 1024 * 1024
    current_external = external
    samples = []
    for index in range(100):
        before_external = current_external
        if index == release_index:
            current_external = 0
        samples.append(
            {
                "request_index": index,
                "before": _sample(
                    base,
                    device_used=base + before_external,
                ),
                "after": _sample(
                    base,
                    device_used=base + current_external,
                ),
                "kv_allocation_id": 9,
                "final_kv_position": 66,
            }
        )
    trace = {
        "memory_sampler": _sampler(),
        "sequential_request_count": 100,
        "sequential_requests": samples,
    }

    result = soak.validate_sequential_requests(
        trace,
        expected_count=100,
        tolerance_bytes=32 * 1024 * 1024,
    )
    assert result["passed"]
    assert all(gate["passed"] for gate in result["fixed_baseline_memory_gates"])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("request_index", "0", "indices"),
        ("kv_allocation_id", "9", "allocation identity"),
        ("final_kv_position", 66.0, "final KV position"),
    ],
)
def test_sequential_validator_requires_typed_rows(
    field: str,
    value: object,
    match: str,
) -> None:
    trace = {
        "memory_sampler": _sampler(),
        "sequential_request_count": 1,
        "sequential_requests": [
            {
                "request_index": 0,
                "before": _sample(1_000),
                "after": _sample(1_100),
                "kv_allocation_id": 9,
                "final_kv_position": 66,
            }
        ],
    }
    trace["sequential_requests"][0][field] = value

    with pytest.raises(RuntimeError, match=match):
        soak.validate_sequential_requests(trace, expected_count=1, tolerance_bytes=128)


def test_load_cycle_validator_rejects_retained_memory() -> None:
    trace = _load_cycle_trace(20)
    cycles = trace["load_cycles"]
    result = soak.validate_load_cycles(trace, expected_count=20, expected_r=512, tolerance_bytes=32)
    assert result["passed"]
    assert result["warmup"]["retained_bytes"] == 500
    assert result["cold_start_evidence"]["passed"]
    assert result["measured_cycle_indices"] == list(range(20))
    assert len(result["measured_cycles"]) == 20
    assert len(result["continuity_gates"]) == 20

    retained = soak.MEMORY_ATTRIBUTION_FLOOR_BYTES + 1
    cycles[7]["after_unload"] = _sample(1_500 + retained)
    cycles[7]["retained_bytes"] = retained
    cycles[7]["device_wide_retained_bytes"] = retained
    with pytest.raises(RuntimeError, match="process retention exceeded"):
        soak.validate_load_cycles(trace, expected_count=20, expected_r=512, tolerance_bytes=32)


def test_load_cycle_rejects_cumulative_retention_against_common_baseline() -> None:
    trace = _load_cycle_trace(20)
    step = 32 * 1024 * 1024
    base = 1_500
    trace["load_cycles"] = [
        {
            **_lifetime(
                r=512,
                b=4,
                allocation_id=index + 2,
                label="measured-load-cycle",
                measured=True,
                before_load=_sample(base + index * step),
                after_requests=_sample(base + index * step + 3_000),
                after_unload=_sample(base + (index + 1) * step),
                execution_ordinal=index + 1,
                role="measured",
            ),
            "cycle_index": index,
        }
        for index in range(20)
    ]

    with pytest.raises(RuntimeError, match="fixed positive-growth envelope"):
        soak.validate_load_cycles(
            trace,
            expected_count=20,
            expected_r=512,
            tolerance_bytes=32,
        )


def test_load_cycle_rejects_cold_retention_above_explicit_bound() -> None:
    trace = _load_cycle_trace()
    warmup = trace["load_cycle_warmup"]
    retained = soak.COLD_PROCESS_RETENTION_FLOOR_BYTES + 1
    warmup["after_unload"] = _sample(1_000 + retained)
    warmup["retained_bytes"] = retained
    warmup["device_wide_retained_bytes"] = retained

    with pytest.raises(RuntimeError, match="cold_start process retention exceeded"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)


def test_load_cycle_rejects_device_retention_when_process_retention_is_small() -> None:
    trace = _load_cycle_trace()
    cycle = trace["load_cycles"][0]
    external = soak.MEMORY_ATTRIBUTION_FLOOR_BYTES + 1
    cycle["after_unload"] = _sample(
        1_510,
        device_used=1_510 + external,
        other_process_used=external,
    )
    cycle["retained_bytes"] = 10
    cycle["device_wide_retained_bytes"] = 10 + external

    with pytest.raises(RuntimeError, match="device-wide retention exceeded"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)


def test_load_cycle_rejects_large_negative_measured_retention() -> None:
    trace = _load_cycle_trace(1)
    offset = 2 * soak.MEMORY_ATTRIBUTION_FLOOR_BYTES
    trace["load_cycle_warmup"] = _lifetime(
        r=512,
        b=4,
        allocation_id=1,
        label="unmeasured-load-cycle-warmup",
        measured=False,
        before_load=_sample(1_000 + offset),
        after_requests=_sample(5_000 + offset),
        after_unload=_sample(1_500 + offset),
        execution_ordinal=0,
        role="warmup",
    )
    trace["load_cycles"][0] = {
        **_lifetime(
            r=512,
            b=4,
            allocation_id=2,
            label="measured-load-cycle",
            measured=True,
            before_load=_sample(1_500 + offset),
            after_requests=_sample(4_500 + offset),
            after_unload=_sample(1_510),
            execution_ordinal=1,
            role="measured",
        ),
        "cycle_index": 0,
    }

    with pytest.raises(RuntimeError, match="process retention exceeded"):
        soak.validate_load_cycles(trace, expected_count=1, expected_r=512, tolerance_bytes=32)


def test_measured_unlisted_release_is_bounded_by_cold_delta_plus_tolerance() -> None:
    trace = _load_cycle_trace(1)
    warmup = trace["load_cycle_warmup"]
    cold_baseline_unlisted = 100 * 1024 * 1024
    cold_persistent_unlisted = 110 * 1024 * 1024
    for sample in [
        warmup["before_load"],
        *warmup["runtime_phase_memory_samples"][:3],
    ]:
        _add_unlisted_device_bytes(sample, cold_baseline_unlisted)
    for sample in [
        warmup["after_requests"],
        warmup["after_unload"],
        *warmup["runtime_phase_memory_samples"][3:],
    ]:
        _add_unlisted_device_bytes(sample, cold_persistent_unlisted)
    warmup["device_wide_growth_bytes"] += cold_persistent_unlisted - cold_baseline_unlisted
    warmup["device_wide_retained_bytes"] += cold_persistent_unlisted - cold_baseline_unlisted
    _sync_receipt_snapshots(warmup)

    cycle = trace["load_cycles"][0]
    measured_unlisted = 50 * 1024 * 1024
    for sample in [
        cycle["before_load"],
        *cycle["runtime_phase_memory_samples"][:3],
    ]:
        _add_unlisted_device_bytes(sample, cold_persistent_unlisted)
    for sample in [
        cycle["after_requests"],
        cycle["after_unload"],
        *cycle["runtime_phase_memory_samples"][3:],
    ]:
        _add_unlisted_device_bytes(sample, measured_unlisted)
    released = cold_persistent_unlisted - measured_unlisted
    cycle["device_wide_growth_bytes"] -= released
    cycle["device_wide_retained_bytes"] -= released
    _sync_receipt_snapshots(cycle)

    result = soak.validate_load_cycles(
        trace,
        expected_count=1,
        expected_r=512,
        tolerance_bytes=32,
    )
    gate = result["cold_release_budget_gate"]
    assert gate["cumulative_measured_unlisted_release_bytes"] == released
    assert (
        gate["cumulative_measured_unlisted_release_bytes"] <= gate["effective_release_budget_bytes"]
    )


def test_load_cycle_rejects_missing_process_memory_without_used_fallback() -> None:
    trace = _load_cycle_trace()
    trace["load_cycle_warmup"]["before_load"].pop("process_used_bytes")

    with pytest.raises(RuntimeError, match="incomplete NVML/CUDA ledger"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)


def test_load_cycle_rejects_unlisted_external_growth() -> None:
    trace = _load_cycle_trace()
    cycle = trace["load_cycles"][0]
    unlisted = 2 * soak.MEMORY_ATTRIBUTION_FLOOR_BYTES
    after_requests = _sample(4_500, device_used=4_500 + unlisted)
    cycle["after_requests"] = after_requests
    cycle["runtime_phase_memory_samples"][3] = _phase(
        "after runtime KV allocation",
        copy.deepcopy(after_requests),
    )
    cycle["runtime_phase_memory_samples"][4] = _phase(
        "after successful runtime-memory request completion",
        copy.deepcopy(after_requests),
    )
    cycle["device_wide_growth_bytes"] = (
        after_requests["used_bytes"] - cycle["before_load"]["used_bytes"]
    )
    _sync_receipt_snapshots(cycle)

    with pytest.raises(RuntimeError, match="signed attribution failed"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)


def test_cold_persistent_unlisted_driver_growth_is_bounded_and_carried_forward() -> None:
    trace = _load_cycle_trace()
    unlisted = 1024 * 1024 * 1024
    warmup = trace["load_cycle_warmup"]
    _add_unlisted_device_bytes(warmup["after_requests"], unlisted)
    _add_unlisted_device_bytes(warmup["after_unload"], unlisted)
    for sample in warmup["runtime_phase_memory_samples"][3:]:
        _add_unlisted_device_bytes(sample, unlisted)
    warmup["device_wide_growth_bytes"] += unlisted
    warmup["device_wide_retained_bytes"] += unlisted
    _sync_receipt_snapshots(warmup)
    for cycle in trace["load_cycles"]:
        _add_unlisted_device_bytes(cycle["before_load"], unlisted)
        _add_unlisted_device_bytes(cycle["after_requests"], unlisted)
        _add_unlisted_device_bytes(cycle["after_unload"], unlisted)
        for sample in cycle["runtime_phase_memory_samples"]:
            _add_unlisted_device_bytes(sample, unlisted)
        _sync_receipt_snapshots(cycle)

    result = soak.validate_load_cycles(
        trace,
        expected_count=2,
        expected_r=512,
        tolerance_bytes=32,
    )
    cold_gate = result["cold_start_evidence"]["cold_persistent_unlisted_gate"]
    assert cold_gate["used"]
    assert cold_gate["persistent_until_unload"]
    assert cold_gate["no_visible_other_compute_process"]


def test_cold_driver_allowance_requires_every_peak_to_match_unload() -> None:
    trace = _load_cycle_trace(1)
    unlisted = 1024 * 1024 * 1024
    warmup = trace["load_cycle_warmup"]
    for sample in [
        warmup["after_requests"],
        warmup["after_unload"],
        # Deliberately omit the "after runtime KV allocation" peak.
        warmup["runtime_phase_memory_samples"][4],
    ]:
        _add_unlisted_device_bytes(sample, unlisted)
    warmup["device_wide_growth_bytes"] += unlisted
    warmup["device_wide_retained_bytes"] += unlisted

    cycle = trace["load_cycles"][0]
    for sample in [
        cycle["before_load"],
        cycle["after_requests"],
        cycle["after_unload"],
        *cycle["runtime_phase_memory_samples"],
    ]:
        _add_unlisted_device_bytes(sample, unlisted)
    _sync_receipt_snapshots(cycle)

    with pytest.raises(RuntimeError, match="signed attribution failed"):
        soak.validate_load_cycles(
            trace,
            expected_count=1,
            expected_r=512,
            tolerance_bytes=32,
        )


def test_cold_driver_persistence_tolerance_is_independent_of_process_peak() -> None:
    trace = _load_cycle_trace(1)
    warmup = trace["load_cycle_warmup"]
    peak_process = 14 * 1024 * 1024 * 1024
    unload_unlisted = 256 * 1024 * 1024
    peak = _sample(peak_process, device_used=peak_process + 1)
    unload = _sample(1_500, device_used=1_500 + unload_unlisted)
    warmup["after_requests"] = copy.deepcopy(peak)
    warmup["after_unload"] = copy.deepcopy(unload)
    warmup["runtime_phase_memory_samples"][3] = _phase(
        "after runtime KV allocation",
        peak,
    )
    warmup["runtime_phase_memory_samples"][4] = _phase(
        "after successful runtime-memory request completion",
        peak,
    )
    warmup["process_growth_bytes"] = (
        warmup["after_requests"]["process_used_bytes"] - warmup["before_load"]["process_used_bytes"]
    )
    warmup["device_wide_growth_bytes"] = (
        warmup["after_requests"]["used_bytes"] - warmup["before_load"]["used_bytes"]
    )
    warmup["retained_bytes"] = (
        warmup["after_unload"]["process_used_bytes"] - warmup["before_load"]["process_used_bytes"]
    )
    warmup["device_wide_retained_bytes"] = (
        warmup["after_unload"]["used_bytes"] - warmup["before_load"]["used_bytes"]
    )
    _sync_receipt_snapshots(warmup)

    cycle = trace["load_cycles"][0]
    for sample in [
        cycle["before_load"],
        cycle["after_requests"],
        cycle["after_unload"],
        *cycle["runtime_phase_memory_samples"],
    ]:
        _add_unlisted_device_bytes(sample, unload_unlisted)
    _sync_receipt_snapshots(cycle)

    with pytest.raises(RuntimeError, match="signed attribution failed"):
        soak.validate_load_cycles(
            trace,
            expected_count=1,
            expected_r=512,
            tolerance_bytes=32,
        )


def test_cold_driver_budget_does_not_allow_positive_measured_device_retention() -> None:
    trace = _load_cycle_trace(1)
    cold_driver = 1024 * 1024 * 1024
    warmup = trace["load_cycle_warmup"]
    for sample in [
        warmup["after_requests"],
        warmup["after_unload"],
        *warmup["runtime_phase_memory_samples"][3:],
    ]:
        _add_unlisted_device_bytes(sample, cold_driver)
    warmup["device_wide_growth_bytes"] += cold_driver
    warmup["device_wide_retained_bytes"] += cold_driver
    _sync_receipt_snapshots(warmup)

    cycle = trace["load_cycles"][0]
    for sample in [
        cycle["before_load"],
        cycle["after_requests"],
        *cycle["runtime_phase_memory_samples"],
    ]:
        _add_unlisted_device_bytes(sample, cold_driver)
    positive_retention = 512 * 1024 * 1024
    cycle["after_unload"] = _sample(
        1_510,
        device_used=1_510 + cold_driver + positive_retention,
        other_process_used=positive_retention,
    )
    cycle["device_wide_growth_bytes"] = (
        cycle["after_requests"]["used_bytes"] - cycle["before_load"]["used_bytes"]
    )
    cycle["device_wide_retained_bytes"] = (
        cycle["after_unload"]["used_bytes"] - cycle["before_load"]["used_bytes"]
    )
    _sync_receipt_snapshots(cycle)

    with pytest.raises(RuntimeError, match="device-wide retention exceeded"):
        soak.validate_load_cycles(
            trace,
            expected_count=1,
            expected_r=512,
            tolerance_bytes=32,
        )


def test_cold_unlisted_allowance_requires_persistence_and_no_visible_process() -> None:
    trace = _load_cycle_trace()
    unlisted = 1024 * 1024 * 1024
    warmup = trace["load_cycle_warmup"]
    _add_unlisted_device_bytes(warmup["after_requests"], unlisted)
    for sample in warmup["runtime_phase_memory_samples"][3:]:
        _add_unlisted_device_bytes(sample, unlisted)
    warmup["device_wide_growth_bytes"] += unlisted
    _sync_receipt_snapshots(warmup)
    with pytest.raises(RuntimeError, match="signed attribution failed"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)

    trace = _load_cycle_trace()
    warmup = trace["load_cycle_warmup"]
    oversized = soak.COLD_PERSISTENT_UNLISTED_LIMIT_BYTES + 1
    for sample in [
        warmup["after_requests"],
        warmup["after_unload"],
        *warmup["runtime_phase_memory_samples"][3:],
    ]:
        _add_unlisted_device_bytes(sample, oversized)
    warmup["device_wide_growth_bytes"] += oversized
    warmup["device_wide_retained_bytes"] += oversized
    _sync_receipt_snapshots(warmup)
    with pytest.raises(RuntimeError, match="signed attribution failed"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)

    trace = _load_cycle_trace()
    warmup = trace["load_cycle_warmup"]
    for sample in [
        warmup["after_requests"],
        warmup["after_unload"],
        *warmup["runtime_phase_memory_samples"][3:],
    ]:
        _add_unlisted_device_bytes(sample, unlisted)
        sample["compute_processes"].append({"pid": 999, "used_bytes": 1})
        sample["all_compute_process_used_bytes"] += 1
        sample["other_compute_process_used_bytes"] += 1
        sample["nvml_device_free_bytes"] -= 1
        sample["nvml_device_used_bytes"] += 1
    warmup["device_wide_growth_bytes"] += unlisted
    warmup["device_wide_retained_bytes"] += unlisted
    _sync_receipt_snapshots(warmup)
    with pytest.raises(RuntimeError, match="signed attribution failed"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)


def test_load_cycle_rejects_inter_lifetime_continuity_drift_in_either_direction() -> None:
    trace = _load_cycle_trace()
    drift = soak.MEMORY_ATTRIBUTION_FLOOR_BYTES + 1_024
    trace["load_cycles"][1] = {
        **_lifetime(
            r=512,
            b=4,
            allocation_id=3,
            label="measured-load-cycle",
            measured=True,
            before_load=_sample(1_500 + drift),
            after_requests=_sample(4_500 + drift),
            after_unload=_sample(1_510 + drift),
            execution_ordinal=2,
            role="measured",
        ),
        "cycle_index": 1,
    }

    with pytest.raises(RuntimeError, match="fixed positive-growth envelope"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)

    trace = _load_cycle_trace()
    limit = soak.MEMORY_ATTRIBUTION_FLOOR_BYTES
    trace["load_cycles"][0] = {
        **_lifetime(
            r=512,
            b=4,
            allocation_id=2,
            label="measured-load-cycle",
            measured=True,
            before_load=_sample(1_500 + limit),
            after_requests=_sample(4_500 + limit),
            after_unload=_sample(1_500 + 2 * limit),
            execution_ordinal=1,
            role="measured",
        ),
        "cycle_index": 0,
    }
    with pytest.raises(RuntimeError, match="fixed positive-growth envelope"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)


@pytest.mark.parametrize(
    "policy",
    [
        {"kind": "max_sequence_length", "requested_tokens": "512"},
        {"kind": "max_sequence_length", "requested_tokens": 512.0},
        {"kind": "max_sequence_length", "requested_tokens": True},
        {"kind": "max_sequence_length", "requested_tokens": 512, "extra": 1},
        {"kind": "fraction", "requested_fraction": 1},
    ],
)
def test_load_cycle_rejects_malformed_typed_policy(policy: dict) -> None:
    trace = _load_cycle_trace()
    trace["load_cycles"][0]["policy"] = policy

    with pytest.raises(RuntimeError, match="policy is invalid|wrong typed policy"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)


def test_sampler_and_sample_identity_are_strict_and_pid_must_be_listed() -> None:
    trace = _load_cycle_trace()
    trace["memory_sampler"]["captures_all_compute_processes"] = False
    with pytest.raises(RuntimeError, match="invalid identity or type"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)

    trace = _load_cycle_trace()
    phase = trace["load_cycle_warmup"]["runtime_phase_memory_samples"][0]
    phase["device"] = 1
    with pytest.raises(RuntimeError, match="sampler identity"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)

    trace = _load_cycle_trace()
    sample = trace["load_cycle_warmup"]["before_load"]
    sample["compute_processes"] = [{"pid": 999, "used_bytes": sample["process_used_bytes"]}]
    with pytest.raises(RuntimeError, match="does not contain the sampler PID"):
        soak.validate_load_cycles(trace, expected_count=2, expected_r=512, tolerance_bytes=32)


def test_receipt_requires_complete_contiguous_accounting() -> None:
    receipt = _receipt(512, 22_528, 4)
    assert (
        soak.validate_receipt({"runtime_memory_receipt": receipt}, 512)["kv_reserved_bytes"]
        == 512 * 22_528
    )

    receipt["backend_owned_cache_output_bytes"] = 1
    with pytest.raises(RuntimeError, match="requires.*=0"):
        soak.validate_receipt({"runtime_memory_receipt": receipt}, 512)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "ordinary_device_input_bytes",
            None,
            "typed nonnegative fields",
        ),
        (
            "ordinary_device_output_bytes",
            -1,
            "typed nonnegative fields",
        ),
        (
            "final_free_bytes",
            1,
            "exact capacity-decision alias",
        ),
        (
            "settled_snapshot_unavailable_reason",
            "cudaMemGetInfo failed",
            "capacity-decision and settled snapshots",
        ),
    ],
)
def test_receipt_schema_v3_fails_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    receipt = _receipt(512, 22_528, 4)
    receipt[field] = value

    with pytest.raises(RuntimeError, match=match):
        soak.validate_receipt({"runtime_memory_receipt": receipt}, 512)


def test_receipt_schema_v3_rejects_missing_ordinary_field() -> None:
    receipt = _receipt(512, 22_528, 4)
    receipt.pop("ordinary_device_input_bytes")

    with pytest.raises(RuntimeError, match="misses fields"):
        soak.validate_receipt({"runtime_memory_receipt": receipt}, 512)


def test_fraction_policy_uses_exact_binary64_floor() -> None:
    fraction = 0.9
    safely_available = 9_007_199_254_740_994
    exact_budget = soak._fraction_budget_bytes(
        fraction,
        safely_available,
    )
    rounded_multiply_budget = math.floor(fraction * safely_available)
    assert rounded_multiply_budget == exact_budget + 1

    receipt = _receipt(1, 1, 1)
    total = safely_available + 1_000_000
    receipt.update(
        {
            "policy": "fraction",
            "policy_fraction": fraction,
            "model_context_limit": 1,
            "capacity_decision_free_bytes": safely_available,
            "capacity_decision_total_bytes": total,
            "capacity_decision_device_used_bytes": (total - safely_available),
            "final_free_bytes": safely_available,
            "final_total_bytes": total,
            "final_device_used_bytes": total - safely_available,
            "settled_free_bytes": safely_available - 1,
            "settled_total_bytes": total,
            "settled_device_used_bytes": total - safely_available + 1,
            "safety_reserve_bytes": 0,
            "kv_budget_bytes": rounded_multiply_budget,
        }
    )

    with pytest.raises(
        RuntimeError,
        match="capacity-decision snapshot",
    ):
        soak.validate_receipt(
            {"runtime_memory_receipt": receipt},
            1,
            expected_policy={
                "kind": "fraction",
                "requested_fraction": fraction,
            },
        )


def test_lifetime_rejects_swapped_decision_and_settled_snapshots() -> None:
    trace = _load_cycle_trace(1)
    lifetime = trace["load_cycle_warmup"]
    receipt = lifetime["runtime_memory_receipt"]
    for suffix in ("free_bytes", "total_bytes", "device_used_bytes"):
        decision = f"capacity_decision_{suffix}"
        settled = f"settled_{suffix}"
        receipt[decision], receipt[settled] = (
            receipt[settled],
            receipt[decision],
        )
    receipt["final_free_bytes"] = receipt["capacity_decision_free_bytes"]
    receipt["final_total_bytes"] = receipt["capacity_decision_total_bytes"]
    receipt["final_device_used_bytes"] = receipt["capacity_decision_device_used_bytes"]
    receipt["kv_budget_bytes"] = soak._fraction_budget_bytes(
        0.9,
        max(
            0,
            receipt["capacity_decision_free_bytes"] - receipt["safety_reserve_bytes"],
        ),
    )
    with pytest.raises(RuntimeError, match="synchronized runtime phase"):
        soak.validate_load_cycles(
            trace,
            expected_count=1,
            expected_r=512,
            tolerance_bytes=32,
        )


def test_lifetime_never_uses_legacy_final_snapshot_for_policy_r() -> None:
    trace = _load_cycle_trace(1)
    lifetime = trace["load_cycles"][0]
    receipt = lifetime["runtime_memory_receipt"]
    receipt["final_free_bytes"] = receipt["settled_free_bytes"]
    receipt["final_total_bytes"] = receipt["settled_total_bytes"]
    receipt["final_device_used_bytes"] = receipt["settled_device_used_bytes"]

    with pytest.raises(RuntimeError, match="exact capacity-decision alias"):
        soak.validate_load_cycles(
            trace,
            expected_count=1,
            expected_r=512,
            tolerance_bytes=32,
        )


def test_source_state_gate_is_fail_closed() -> None:
    pre = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    post = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    report = {"passed": True}

    assert soak.apply_source_state_gate(report, pre, post)
    assert report["source_state_pre"] is pre
    assert report["source_state_post"] is post
    assert report["source_state_unchanged"] is True
    assert report["passed"] is True

    changed = {"source_state_sha256": "c" * 64, "git_head": "b" * 40}
    report = {"passed": True}
    assert not soak.apply_source_state_gate(report, pre, changed)
    assert report["source_state_unchanged"] is False
    assert report["passed"] is False

    report = {"passed": False}
    assert soak.apply_source_state_gate(report, pre, post)
    assert report["passed"] is False


def test_source_snapshot_excludes_artifact_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, str]] = []

    def snapshot(
        repo_root: Path,
        tool_path: Path,
        artifact_dir: Path,
        *,
        label: str,
    ) -> dict:
        assert repo_root == soak.REPO_ROOT
        assert tool_path == Path(soak.__file__)
        calls.append((artifact_dir, label))
        return {"source_state_sha256": "a" * 64, "git_head": "b" * 40}

    monkeypatch.setattr(soak.boundary, "source_state_provenance", snapshot)
    external = tmp_path / "proof"
    soak._source_state_snapshot(external, label="pre")
    artifact = soak.REPO_ROOT / "artifacts" / "unit-soak-proof"
    soak._source_state_snapshot(artifact, label="post")

    assert calls == [
        (external.resolve(), "pre"),
        (artifact.resolve(), "post"),
    ]
    with pytest.raises(ValueError, match="source snapshots exclude it"):
        soak._source_state_snapshot(
            soak.REPO_ROOT / "unit-soak-proof",
            label="post",
        )
