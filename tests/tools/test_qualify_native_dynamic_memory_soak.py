# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "qualify_native_dynamic_memory_soak.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_native_dynamic_memory_soak", MODULE_PATH
)
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

    result = soak.validate_same_process_two_r(
        trace, r1=r1, r2=r2, tolerance_bytes=32
    )
    assert result["passed"]
    assert result["nvml_actual_delta_bytes"] == (r2 - r1) * b

    lifetimes[1]["runtime_memory_receipt"]["kv_allocation_id"] = 2
    with pytest.raises(RuntimeError, match="reused an allocation identity"):
        soak.validate_same_process_two_r(
            trace, r1=r1, r2=r2, tolerance_bytes=32
        )


def test_controlled_reservation_reduces_auto_r_and_recovers() -> None:
    baseline_r, constrained_r, b = 1_024, 512, 4
    process_baseline = _sample(1_000)
    before_reservation = _sample(4_000)
    after_reservation = _sample(6_048)
    warmup = {
        "policy": {"kind": "auto"},
        "measured": False,
        "before_load": _sample(800),
        "after_unload": _sample(1_000),
    }
    baseline = {
        "policy": {"kind": "auto"},
        "measured": True,
        "runtime_memory_receipt": _receipt(baseline_r, b, 1),
        "before_load": _sample(1_000),
        "after_requests": _sample(7_000),
        "after_unload": process_baseline,
    }
    constrained = {
        "policy": {"kind": "auto"},
        "measured": True,
        "runtime_memory_receipt": _receipt(constrained_r, b, 2),
        "before_load": process_baseline,
        "after_requests": _sample(6_500),
        "after_unload": _sample(3_048),
    }
    trace = {
        "mode": "same_process_controlled_external_reservation",
        "memory_sampler": _sampler(),
        "final_kv_position": 66,
        "controlled_reservation": {
            "target_tokens": constrained_r,
            "reservation_phase": "before runtime KV planning",
            "sizing": {
                "estimated_non_kv_growth_bytes": 100,
                "baseline_pre_load_free_bytes": 5_000,
                "baseline_post_load_free_bytes": 4_900,
                "baseline_engine_load_device_bytes": 100,
                "warmup_retained_process_bytes": 200,
                "warmup_retained_device_wide_bytes": 200,
                "required_free_basis": (
                    "post-load runtime overhead before KV planning"
                ),
                "context_device_memory_bytes": 3,
                "external_device_output_bytes": 4,
                "graph_private_device_bytes": 0,
                "post_load_runtime_overhead_bytes": 7,
                "target_kv_bytes": constrained_r * b,
                "safety_reserve_bytes": 64,
                "target_fraction_denominator": 8,
                "target_fraction_headroom_bytes": (
                    constrained_r * b + 7
                )
                // 8,
                "target_headroom_bytes": 8 * 1024 * 1024,
                "required_post_load_free_bytes": (
                    constrained_r * b
                    + 64
                    + 7
                    + 8 * 1024 * 1024
                ),
            },
            "reservation_bytes": 2_048,
            "reservation_address": 4_096,
            "reservation_allocation_count": 1,
            "reservation_allocations": [
                {"index": 0, "address": 4_096, "bytes": 2_048}
            ],
            "before_reservation": before_reservation,
            "after_reservation": after_reservation,
            "warmup": warmup,
            "after_constrained_unload": _sample(3_048),
            "after_release": _sample(1_002),
            "baseline_r": baseline_r,
            "constrained_r": constrained_r,
            "baseline": baseline,
            "constrained": constrained,
        },
    }

    result = soak.validate_controlled_reservation(
        trace, tolerance_bytes=32
    )
    assert result["passed"]
    assert result["r_delta"] == baseline_r - constrained_r
    assert result["actual_kv_delta_bytes"] == (baseline_r - constrained_r) * b

    trace["controlled_reservation"]["after_release"] = _sample(1_100)
    with pytest.raises(RuntimeError, match="release delta mismatch|process baseline"):
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
    assert soak.validate_sequential_requests(
        trace, expected_count=100, tolerance_bytes=128
    )["passed"]

    samples[-1]["kv_allocation_id"] = 10
    with pytest.raises(RuntimeError, match="reuse one KV allocation"):
        soak.validate_sequential_requests(
            trace, expected_count=100, tolerance_bytes=128
        )


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
    result = soak.validate_load_cycles(
        trace, expected_count=20, expected_r=512, tolerance_bytes=32
    )
    assert result["passed"]
    assert result["warmup"]["retained_bytes"] == 500
    assert result["measured_cycle_indices"] == list(range(20))
    assert len(result["measured_cycles"]) == 20

    cycles[7]["after_unload"] = _sample(1_100)
    with pytest.raises(RuntimeError, match="retained"):
        soak.validate_load_cycles(
            trace, expected_count=20, expected_r=512, tolerance_bytes=32
        )


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
    assert soak.validate_receipt(
        {"runtime_memory_receipt": receipt}, 512
    )["kv_reserved_bytes"] == 512 * 22_528

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
