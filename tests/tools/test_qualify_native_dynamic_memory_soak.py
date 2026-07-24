# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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


def _sample(used: int) -> dict:
    return {
        "free_bytes": 10_000 - used,
        "total_bytes": 10_000,
        "used_bytes": used,
        "process_used_bytes": used,
    }


def _device_sample(*, total: int, free: int, process_used: int) -> dict:
    return {
        "free_bytes": free,
        "total_bytes": total,
        "used_bytes": total - free,
        "process_used_bytes": process_used,
    }


def _phase(phase: str, sample: dict) -> dict:
    return {"phase": phase, **sample}


def _sampler() -> dict:
    return {
        "source": "nvmlDeviceGetComputeRunningProcesses_v3",
        "pid": 123,
        "cuda_logical_device_index": 0,
        "physical_device_index": 1,
        "pci_bus_id": "00000000:01:00.0",
        "gpu_uuid": "GPU-test",
    }


def _receipt(r: int, b: int, allocation_id: int) -> dict:
    return {
        "policy": "auto",
        "policy_fraction": 0.9,
        "safety_reserve_bytes": 64 * 1024 * 1024,
        "pre_load_free_bytes": 9_000,
        "post_load_free_bytes": 8_000,
        "final_free_bytes": 7_000,
        "serialized_plan_bytes": 1,
        "resident_weight_bytes": 2,
        "resident_weight_copy_count": 1,
        "engine_weight_bytes": 2,
        "context_device_memory_bytes": 3,
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
    warmup = {
        "measured": False,
        "runtime_kv_capacity_tokens": r2,
    }
    lifetimes = [
        {
            "measured": True,
            "policy": {
                "kind": "max_sequence_length",
                "requested_tokens": r1,
            },
            "runtime_memory_receipt": _receipt(r1, b, 2),
            "before_load": _sample(1_000),
            "after_requests": _sample(4_000),
            "after_unload": _sample(1_010),
        },
        {
            "measured": True,
            "policy": {
                "kind": "max_sequence_length",
                "requested_tokens": r2,
            },
            "runtime_memory_receipt": _receipt(r2, b, 3),
            "before_load": _sample(1_000),
            "after_requests": _sample(6_048),
            "after_unload": _sample(1_010),
        },
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

    lifetimes[1]["runtime_memory_receipt"]["kv_allocation_id"] = 2
    with pytest.raises(RuntimeError, match="reused an allocation identity"):
        soak.validate_same_process_two_r(trace, r1=r1, r2=r2, tolerance_bytes=32)


def _controlled_reservation_trace() -> dict:
    total = 4_000_000_000
    alignment = soak.CONTROLLED_RESERVATION_ALIGNMENT_BYTES
    safety = 64 * 1024 * 1024
    target_tokens, baseline_r, b = 512, 1_024, 114_688
    target_kv_bytes = target_tokens * b
    measured_context_output_bytes = 4 * alignment
    request_completion_headroom_bytes = 0
    constrained_request_device_bytes = 4 * 1024
    constrained_request_process_bytes = 16
    policy_safe_bytes = math.ceil(target_kv_bytes / 0.9)
    required_visible_free = measured_context_output_bytes + safety + policy_safe_bytes
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
    bulk_bytes = ((after_guard["free_bytes"] - required_visible_free) // alignment) * alignment
    after_reservation = _device_sample(
        total=total,
        free=after_guard["free_bytes"] - bulk_bytes,
        process_used=after_guard["process_used_bytes"] + bulk_bytes,
    )
    guard_before_release = _device_sample(
        total=total,
        free=(after_reservation["free_bytes"] - measured_context_output_bytes),
        process_used=(after_reservation["process_used_bytes"] + measured_context_output_bytes),
    )
    guard_after_release = _device_sample(
        total=total,
        free=guard_before_release["free_bytes"] + guard_bytes,
        process_used=guard_before_release["process_used_bytes"] - guard_bytes,
    )
    constrained_r = (
        int(
            0.9
            * max(
                0,
                guard_before_release["free_bytes"] - safety,
            )
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
        }
    )

    warmup = {
        "policy": {"kind": "auto"},
        "measured": False,
        "before_load": _device_sample(
            total=total,
            free=3_100_000_000,
            process_used=9_000_000,
        ),
        "after_unload": dict(process_baseline),
    }
    calibration = {
        "policy": {
            "kind": "max_sequence_length",
            "requested_tokens": target_tokens,
        },
        "measured": True,
        "runtime_memory_receipt": calibration_receipt,
        "before_load": dict(process_baseline),
        "after_requests": dict(calibration_after_request),
        "after_unload": dict(process_baseline),
        "runtime_phase_memory_samples": [
            _phase(before_planning_phase, calibration_before_planning),
            _phase(after_overhead_phase, calibration_after_overhead),
            _phase(after_kv_phase, calibration_after_kv),
            _phase(after_request_phase, calibration_after_request),
        ],
    }
    baseline = {
        "policy": {"kind": "auto"},
        "measured": True,
        "runtime_memory_receipt": baseline_receipt,
        "before_load": dict(process_baseline),
        "after_requests": _device_sample(
            total=total,
            free=1_600_000_000,
            process_used=(process_baseline["process_used_bytes"] + baseline_r * b + 120_000_000),
        ),
        "after_unload": dict(process_baseline),
    }
    constrained = {
        "policy": {"kind": "auto"},
        "measured": True,
        "runtime_memory_receipt": constrained_receipt,
        "before_load": dict(process_baseline),
        "after_requests": dict(constrained_after_request),
        "after_unload": dict(after_constrained_unload),
        "runtime_phase_memory_samples": [
            _phase(before_planning_phase, after_reservation),
            _phase(after_overhead_phase, guard_before_release),
            _phase(after_kv_phase, constrained_after_kv),
            _phase(after_request_phase, constrained_after_request),
        ],
    }
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
                "baseline_engine_load_device_bytes": 1_200_000_000,
                "warmup_retained_process_bytes": 1_000_000,
                "warmup_retained_device_wide_bytes": 100_000_000,
                "required_free_basis": (
                    "measured target context/output delta plus safety and "
                    "ceil(target KV / auto fraction)"
                ),
                "auto_fraction": 0.9,
                "calibration_context_device_memory_bytes": 3,
                "calibration_external_device_output_bytes": 4,
                "calibration_graph_private_device_bytes": 0,
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
                "max_capacity_rounding_rows": (alignment + b - 1) // b,
                "guard_bytes": guard_bytes,
                "required_visible_post_load_free_bytes": (required_visible_free),
                "visible_free_formula": (
                    "measured_context_output_bytes + safety_reserve_bytes + "
                    "ceil(target_kv_bytes / auto_fraction)"
                ),
            },
            "before_reservation": dict(before_reservation),
            "after_reservation": dict(after_reservation),
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
                "before_allocation": dict(before_reservation),
                "after_allocation": dict(after_guard),
                "before_release": dict(guard_before_release),
                "after_release": dict(guard_after_release),
            },
            "bulk": {
                "allocation_phase": before_planning_phase,
                "release_phase": "after constrained pipeline unload",
                "bytes": bulk_bytes,
                "initial_bytes": bulk_bytes,
                "correction_bytes": 0,
                "correction_attempts": 0,
                "address": 8_192,
                "allocation_count": 1,
                "allocations": [
                    {
                        "index": 0,
                        "address": 8_192,
                        "bytes": bulk_bytes,
                    }
                ],
                "before_allocation": dict(after_guard),
                "after_allocation": dict(after_reservation),
                "before_release": dict(after_constrained_unload),
                "after_release": dict(process_baseline),
            },
            "warmup": warmup,
            "calibration": calibration,
            "after_constrained_unload": dict(after_constrained_unload),
            "after_release": dict(process_baseline),
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


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("calibration_r", "expected exactly"),
        ("guard_release_phase", "guard receipt"),
        ("bulk_bytes", "bulk reservation receipt"),
        ("bulk_correction_receipt", "bulk reservation receipt"),
        ("target_window", "near target"),
        ("planner_snapshot", "recorded final snapshot"),
        ("runtime_kv_malloc", "runtime KV allocation"),
        ("request_process_growth", "per-process NVML growth"),
        ("calibration_external_delta", "sizing formula"),
        ("missing_calibration_external_delta", "request-attribution fields"),
        ("missing_guard_basis", "request-attribution fields"),
        ("release_recovery", "bulk release delta"),
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
        proof["bulk"]["correction_attempts"] = 1
    elif tamper == "target_window":
        proof["constrained_r"] = (
            proof["target_tokens"] + proof["sizing"]["max_capacity_rounding_rows"] + 1
        )
    elif tamper == "planner_snapshot":
        proof["guard"]["before_release"]["free_bytes"] += 1
    elif tamper == "runtime_kv_malloc":
        proof["constrained"]["runtime_phase_memory_samples"][2]["free_bytes"] += 64
    elif tamper == "request_process_growth":
        proof["constrained"]["runtime_phase_memory_samples"][3]["process_used_bytes"] += 17
    elif tamper == "calibration_external_delta":
        proof["sizing"]["request_completion_external_delta_bytes"] += 1
    elif tamper == "missing_calibration_external_delta":
        proof["sizing"].pop("request_completion_external_delta_bytes")
    elif tamper == "missing_guard_basis":
        proof["sizing"].pop("request_completion_guard_basis")
    elif tamper == "release_recovery":
        proof["after_release"]["process_used_bytes"] += 1_024
        proof["bulk"]["after_release"]["process_used_bytes"] += 1_024
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
        "sequential_request_count": 100,
        "sequential_requests": samples,
    }
    assert soak.validate_sequential_requests(trace, expected_count=100, tolerance_bytes=128)[
        "passed"
    ]

    samples[-1]["kv_allocation_id"] = 10
    with pytest.raises(RuntimeError, match="reuse one KV allocation"):
        soak.validate_sequential_requests(trace, expected_count=100, tolerance_bytes=128)


def test_load_cycle_validator_rejects_retained_memory() -> None:
    warmup = {
        "measured": False,
        "before_load": _sample(1_000),
        "after_requests": _sample(5_000),
        "after_unload": _sample(1_500),
        "retained_bytes": 500,
        "device_wide_retained_bytes": 500,
        "policy": {"kind": "max_sequence_length", "requested_tokens": 512},
        "runtime_kv_capacity_tokens": 512,
        "runtime_memory_receipt": {},
        "kv_allocation_id": 1,
    }
    cycles = [
        {
            "cycle_index": index,
            "before_load": _sample(1_000),
            "after_requests": _sample(4_000),
            "after_unload": _sample(1_010),
            "retained_bytes": 10,
            "device_wide_retained_bytes": 10,
            "policy": {
                "kind": "max_sequence_length",
                "requested_tokens": 512,
            },
            "runtime_kv_capacity_tokens": 512,
            "runtime_memory_receipt": {},
            "kv_allocation_id": index + 1,
        }
        for index in range(20)
    ]
    trace = {
        "memory_sampler": _sampler(),
        "load_cycle_warmup": warmup,
        "load_cycle_count": 20,
        "load_cycles": cycles,
    }
    result = soak.validate_load_cycles(trace, expected_count=20, expected_r=512, tolerance_bytes=32)
    assert result["passed"]
    assert result["warmup"]["retained_bytes"] == 500
    assert result["measured_cycle_indices"] == list(range(20))
    assert len(result["measured_cycles"]) == 20

    cycles[7]["after_unload"] = _sample(1_100)
    with pytest.raises(RuntimeError, match="retained"):
        soak.validate_load_cycles(trace, expected_count=20, expected_r=512, tolerance_bytes=32)


def test_receipt_requires_complete_contiguous_accounting() -> None:
    receipt = {
        "serialized_plan_bytes": 1,
        "resident_weight_bytes": 2,
        "resident_weight_copy_count": 1,
        "engine_weight_bytes": 2,
        "context_device_memory_bytes": 3,
        "external_device_output_bytes": 4,
        "host_staging_bytes": 5,
        "graph_private_device_bytes": 0,
        "kv_reserved_bytes": 512 * 22_528,
        "kv_committed_bytes": 512 * 22_528,
        "kv_metadata_bytes": 0,
        "peak_device_bytes": None,
        "backend_owned_cache_input_bytes": 0,
        "backend_owned_cache_output_bytes": 0,
        "kv_allocation_id": 4,
        "kv_bytes_per_token": 22_528,
        "runtime_kv_capacity_tokens": 512,
    }
    assert (
        soak.validate_receipt({"runtime_memory_receipt": receipt}, 512)["kv_reserved_bytes"]
        == 512 * 22_528
    )

    receipt["backend_owned_cache_output_bytes"] = 1
    with pytest.raises(RuntimeError, match="requires.*=0"):
        soak.validate_receipt({"runtime_memory_receipt": receipt}, 512)


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
