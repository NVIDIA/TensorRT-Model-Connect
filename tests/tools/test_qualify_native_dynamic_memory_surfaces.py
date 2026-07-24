# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))
MODULE_PATH = TOOLS_DIR / "qualify_native_dynamic_memory_surfaces.py"
SPEC = importlib.util.spec_from_file_location("qualify_native_dynamic_memory_surfaces", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
surfaces = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = surfaces
SPEC.loader.exec_module(surfaces)

pytestmark = pytest.mark.dynamic_memory


def _receipt(
    *,
    capacity_free: int = 12_000,
    settled_free: int = 9_000,
    **overrides: object,
) -> dict:
    receipt = {field: index + 1 for index, field in enumerate(surfaces.RECEIPT_EQUIVALENCE_FIELDS)}
    total = 20_000
    receipt.update(
        {
            "receipt_schema_version": 4,
            "contract_version": 2,
            "policy": "bytes",
            "policy_fraction": 0.0,
            "requested_kv_bytes": 2_048,
            "kv_budget_bytes": 2_048,
            "runtime_kv_capacity_tokens": 512,
            "kv_bytes_per_token": 4,
            "kv_reserved_bytes": 2_048,
            "module_residency_reserve_bytes": 1,
            "module_residency_reserve_profile_limit": 512,
            "module_residency_plan_set_sha256": "a" * 64,
            "module_residency_evidence_sha256": "b" * 64,
            "module_residency_cuda_module_loading_mode": "lazy",
            "ordinary_device_input_bytes": 128,
            "ordinary_device_output_bytes": 256,
            "external_device_output_bytes": 512,
            "context_device_memory_bytes": 1_024,
            "graph_private_device_bytes": 0,
            "capacity_decision_resident_overhead_bytes": 1_920,
            "final_non_kv_overhead_delta_bytes": 0,
            "capacity_decision_free_bytes": capacity_free,
            "capacity_decision_total_bytes": total,
            "capacity_decision_device_used_bytes": total - capacity_free,
            "settled_free_bytes": settled_free,
            "settled_total_bytes": total,
            "settled_device_used_bytes": total - settled_free,
            "settled_snapshot_unavailable_reason": None,
            "final_free_bytes": capacity_free,
            "final_total_bytes": total,
            "final_device_used_bytes": total - capacity_free,
            "peak_device_bytes": 1234,
            "peak_device_bytes_scope": "device_wide",
            "peak_device_sample_count": 2,
            "peak_device_sample_boundaries": [
                "after_runtime_kv_allocation",
                "after_successful_request_completion",
            ],
        }
    )
    receipt.update(overrides)
    return receipt


def _runtime_memory_contract() -> dict:
    return {
        "module_residency_calibration": {
            "profile_reserves": [
                {
                    "covering_profile_limit": 512,
                    "cumulative_reserve_bytes": 1,
                }
            ],
            "plan_set_sha256": "a" * 64,
            "evidence_sha256": "b" * 64,
            "cuda_module_loading_mode": "lazy",
        }
    }


def _rejection(
    surface: str,
    normalized_error: str,
    **overrides: object,
) -> dict:
    result = {
        "surface": surface,
        "status": "rejected",
        "normalized_error": normalized_error,
        "message": "pre-request policy rejection",
        "returncode": 1,
        "runtime_memory_receipt_present": False,
        "request_started": False,
        "attention_launch_observed": False,
    }
    result.update(overrides)
    return result


def test_surface_comparison_accepts_equal_resolution_with_different_peaks() -> None:
    results = [
        {
            "status": "accepted",
            "surface": "cli",
            "runtime_memory_receipt": _receipt(
                capacity_free=12_000,
                settled_free=9_000,
                peak_device_bytes=1000,
            ),
        },
        {
            "status": "accepted",
            "surface": "cpp",
            "runtime_memory_receipt": _receipt(
                capacity_free=11_500,
                settled_free=8_500,
                peak_device_bytes=2000,
            ),
        },
        {
            "status": "accepted",
            "surface": "cabi",
            "runtime_memory_receipt": _receipt(
                capacity_free=11_000,
                settled_free=8_000,
                peak_device_bytes=3000,
            ),
        },
        {
            "status": "accepted",
            "surface": "python",
            "runtime_memory_receipt": _receipt(
                capacity_free=10_500,
                settled_free=7_500,
                peak_device_bytes=4000,
            ),
        },
    ]

    comparison, passed = surfaces.compare_surface_receipts(results)

    assert passed
    assert set(comparison) == {"cli", "cpp", "cabi", "python"}
    assert all(item["passed"] for item in comparison.values())
    assert all(item["schema_v4_complete"] for item in comparison.values())


def test_surface_comparison_replays_sealed_module_residency_contract() -> None:
    results = [
        {
            "status": "accepted",
            "surface": surface,
            "runtime_memory_receipt": _receipt(),
        }
        for surface in ("cli", "cpp", "cabi", "python")
    ]

    comparison, passed = surfaces.compare_surface_receipts(
        results,
        runtime_memory_contract=_runtime_memory_contract(),
    )

    assert passed
    assert all(
        item["sealed_calibration_matches"]
        for item in comparison.values()
    )


def test_surface_comparison_fails_closed_on_sealed_calibration_drift() -> None:
    results = [
        {
            "status": "accepted",
            "surface": "cli",
            "runtime_memory_receipt": _receipt(
                module_residency_evidence_sha256="c" * 64,
            ),
        }
    ]

    comparison, passed = surfaces.compare_surface_receipts(
        results,
        runtime_memory_contract=_runtime_memory_contract(),
    )

    assert not passed
    assert not comparison["cli"]["sealed_calibration_matches"]
    assert comparison["cli"]["sealed_calibration_errors"] == [
        "module_residency_evidence_sha256 does not match sealed bundle calibration"
    ]


def test_surface_schema_replays_positive_final_overhead_delta() -> None:
    receipt = _receipt(
        capacity_free=12_016,
        capacity_decision_resident_overhead_bytes=1_904,
        final_non_kv_overhead_delta_bytes=16,
    )
    errors = surfaces._schema_v4_receipt_errors(receipt)
    assert not errors

    receipt["final_non_kv_overhead_delta_bytes"] = 0
    errors = surfaces._schema_v4_receipt_errors(receipt)
    assert any("O(final)-O(resident)" in error for error in errors)


def test_positive_policy_matrix_covers_auto_fraction_bytes_and_u() -> None:
    cases = surfaces.positive_policy_cases(2_048)
    assert [case["name"] for case in cases] == [
        "bytes_plus_u",
        "explicit_auto_plus_u",
        "fraction_plus_u",
        "u_only",
    ]

    for case in cases:
        results = []
        for index, surface in enumerate(("cli", "cpp", "cabi", "python")):
            receipt = _receipt(
                policy=case["expected_policy"],
                policy_fraction=case["expected_fraction"],
                requested_kv_bytes=case["expected_requested_bytes"],
                kv_budget_bytes=(
                    2_048
                    if case["expected_policy"] == "bytes"
                    else 10_000 + index
                ),
            )
            results.append(
                {
                    "status": "accepted",
                    "surface": surface,
                    "runtime_memory_receipt": receipt,
                }
            )
        comparison, passed = surfaces.compare_surface_receipts(
            results,
            expected_capacity=case["max_sequence_length"],
            expected_policy=case["expected_policy"],
            expected_fraction=case["expected_fraction"],
            expected_requested_bytes=case["expected_requested_bytes"],
        )
        assert passed, (case["name"], comparison)


def test_positive_policy_matrix_rejects_policy_drift() -> None:
    case = surfaces.positive_policy_cases(2_048)[1]
    results = [
        {
            "status": "accepted",
            "surface": "cli",
            "runtime_memory_receipt": _receipt(
                policy="auto",
                policy_fraction=0.90,
                requested_kv_bytes=0,
                kv_budget_bytes=10_000,
            ),
        },
        {
            "status": "accepted",
            "surface": "python",
            "runtime_memory_receipt": _receipt(
                policy="fraction",
                policy_fraction=0.90,
                requested_kv_bytes=0,
                kv_budget_bytes=10_001,
            ),
        },
    ]
    comparison, passed = surfaces.compare_surface_receipts(
        results,
        expected_capacity=512,
        expected_policy=case["expected_policy"],
        expected_fraction=case["expected_fraction"],
        expected_requested_bytes=case["expected_requested_bytes"],
    )
    assert not passed
    assert not comparison["python"]["schema_v4_complete"]


def test_negative_policy_matrix_covers_u_over_m_and_conflicting_policy() -> None:
    cases = surfaces.negative_policy_cases(
        model_context_limit=2_048,
        kv_bytes=2_048,
    )

    assert [case["name"] for case in cases] == [
        "over_model_context",
        "conflicting_policy_fields",
    ]
    assert cases[0]["max_sequence_length"] == 2_049
    assert cases[0]["normalized_error"] == "model_context_limit_exceeded"
    assert cases[1]["normalized_error"] == "conflicting_memory_policy"
    assert len(cases[1]["cli_memory_values"]) == 2
    assert cases[1]["helper_bytes"] == 2_048
    assert cases[1]["helper_fraction"] == 1.0


def test_rejection_error_is_normalized_only_from_expected_message() -> None:
    case = surfaces.negative_policy_cases(
        model_context_limit=2_048,
        kv_bytes=2_048,
    )[0]

    for surface in ("cli", "cpp", "cabi", "python"):
        message = " ".join(case["error_needles"][surface])
        assert (
            surfaces._normalized_rejection_error(
                surface=surface,
                message=message,
                case=case,
            )
            == "model_context_limit_exceeded"
        )
        assert (
            surfaces._normalized_rejection_error(
                surface=surface,
                message="unrelated runtime failure",
                case=case,
            )
            is None
        )


def test_rejection_matrix_requires_all_four_pre_request_rejections() -> None:
    case_name = "over_model_context"
    normalized_error = surfaces.NEGATIVE_POLICY_ERRORS[case_name]
    results = [
        _rejection(surface, normalized_error)
        for surface in ("cli", "cpp", "cabi", "python")
    ]

    validations, passed = surfaces.validate_rejection_matrix(
        {case_name: results}
    )

    assert passed
    assert validations[case_name]["normalized_error_consistent"]
    assert all(
        row["attention_launch_count"] == 0 and row["passed"]
        for row in validations[case_name]["surfaces"].values()
    )


@pytest.mark.parametrize(
    "tamper",
    [
        {"normalized_error": None},
        {"runtime_memory_receipt_present": True},
        {"request_started": True},
        {"attention_launch_observed": True},
        {"returncode": 0, "status": "accepted"},
    ],
)
def test_rejection_matrix_fails_closed_on_negative_evidence_drift(
    tamper: dict,
) -> None:
    case_name = "conflicting_policy_fields"
    normalized_error = surfaces.NEGATIVE_POLICY_ERRORS[case_name]
    results = [
        _rejection(surface, normalized_error)
        for surface in ("cli", "cpp", "cabi", "python")
    ]
    results[-1].update(tamper)

    validations, passed = surfaces.validate_rejection_matrix(
        {case_name: results}
    )

    assert not passed
    assert not validations[case_name]["passed"]
    assert not validations[case_name]["surfaces"]["python"]["passed"]


def test_report_gate_requires_complete_positive_and_negative_matrices() -> None:
    positive = {name: {} for name in surfaces.POSITIVE_POLICY_CASE_NAMES}
    negative = {name: {} for name in surfaces.NEGATIVE_POLICY_ERRORS}

    gate = surfaces.qualification_gate(
        policy_matrix=positive,
        positive_surfaces_passed=True,
        rejection_matrix=negative,
        negative_surfaces_passed=True,
        bundle_unchanged=True,
        sealed_calibration_replayed=True,
    )
    assert gate["passed"]

    for overrides in (
        {"rejection_matrix": {}},
        {"negative_surfaces_passed": False},
        {"policy_matrix": {}},
        {"positive_surfaces_passed": False},
        {"bundle_unchanged": False},
        {"sealed_calibration_replayed": False},
    ):
        arguments = {
            "policy_matrix": positive,
            "positive_surfaces_passed": True,
            "rejection_matrix": negative,
            "negative_surfaces_passed": True,
            "bundle_unchanged": True,
            "sealed_calibration_replayed": True,
        }
        arguments.update(overrides)
        assert not surfaces.qualification_gate(**arguments)["passed"]


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        (
            lambda receipt: receipt.pop("ordinary_device_input_bytes"),
            "ordinary_device_input_bytes",
        ),
        (
            lambda receipt: receipt.__setitem__("ordinary_device_output_bytes", -1),
            "ordinary_device_output_bytes",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "final_free_bytes",
                receipt["settled_free_bytes"],
            ),
            "deprecated final snapshot",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "settled_snapshot_unavailable_reason",
                "cudaMemGetInfo failed",
            ),
            "settled snapshot is unavailable",
        ),
        (
            lambda receipt: receipt.pop("settled_snapshot_unavailable_reason"),
            "settled_snapshot_unavailable_reason must be present",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "module_residency_reserve_bytes",
                0,
            ),
            "module_residency_reserve_bytes",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "module_residency_reserve_profile_limit",
                511,
            ),
            "does not cover",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "module_residency_plan_set_sha256",
                "A" * 64,
            ),
            "lowercase SHA256",
        ),
        (
            lambda receipt: receipt.__setitem__(
                "module_residency_cuda_module_loading_mode",
                "unknown",
            ),
            "must be lazy or eager",
        ),
    ],
)
def test_surface_comparison_rejects_incomplete_schema_v4(
    tamper,
    expected_error: str,
) -> None:
    candidate = _receipt()
    tamper(candidate)
    results = [
        {
            "status": "accepted",
            "surface": "cli",
            "runtime_memory_receipt": _receipt(),
        },
        {
            "status": "accepted",
            "surface": "cpp",
            "runtime_memory_receipt": candidate,
        },
    ]

    comparison, passed = surfaces.compare_surface_receipts(results)

    assert not passed
    assert not comparison["cpp"]["schema_v4_complete"]
    assert any(expected_error in error for error in comparison["cpp"]["schema_v4_errors"])


def test_surface_comparison_requires_ordinary_allocation_equivalence() -> None:
    results = [
        {
            "status": "accepted",
            "surface": "cli",
            "runtime_memory_receipt": _receipt(),
        },
        {
            "status": "accepted",
            "surface": "cpp",
            "runtime_memory_receipt": _receipt(
                ordinary_device_input_bytes=129,
                capacity_decision_resident_overhead_bytes=1_921,
            ),
        },
    ]

    comparison, passed = surfaces.compare_surface_receipts(results)

    assert not passed
    assert comparison["cpp"]["schema_v4_complete"]
    assert "ordinary_device_input_bytes" in (comparison["cpp"]["receipt_mismatches"])


def test_surface_comparison_rejects_R_receipt_or_request_peak_drift() -> None:
    results = [
        {
            "status": "accepted",
            "surface": "cli",
            "runtime_memory_receipt": _receipt(),
        },
        {
            "status": "accepted",
            "surface": "cpp",
            "runtime_memory_receipt": _receipt(runtime_kv_capacity_tokens=511),
        },
        {
            "status": "accepted",
            "surface": "python",
            "runtime_memory_receipt": _receipt(
                peak_device_sample_boundaries=["after_runtime_kv_allocation"]
            ),
        },
    ]

    comparison, passed = surfaces.compare_surface_receipts(results)

    assert not passed
    assert not comparison["cpp"]["resolved_R_is_512"]
    assert "runtime_kv_capacity_tokens" in comparison["cpp"]["receipt_mismatches"]
    assert not comparison["python"]["request_complete_peak"]


def test_source_state_gate_is_fail_closed() -> None:
    pre = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    post = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    report = {"passed": True}

    assert surfaces.apply_source_state_gate(report, pre, post)
    assert report["source_state_pre"] is pre
    assert report["source_state_post"] is post
    assert report["source_state_unchanged"] is True
    assert report["passed"] is True

    changed = {"source_state_sha256": "c" * 64, "git_head": "b" * 40}
    report = {"passed": True}
    assert not surfaces.apply_source_state_gate(report, pre, changed)
    assert report["source_state_unchanged"] is False
    assert report["passed"] is False

    report = {"passed": False}
    assert surfaces.apply_source_state_gate(report, pre, post)
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
        assert repo_root == surfaces.REPO_ROOT
        assert tool_path == Path(surfaces.__file__)
        calls.append((artifact_dir, label))
        return {"source_state_sha256": "a" * 64, "git_head": "b" * 40}

    monkeypatch.setattr(surfaces.boundary, "source_state_provenance", snapshot)
    external = tmp_path / "proof"
    surfaces._source_state_snapshot(external, label="pre")
    artifact = surfaces.REPO_ROOT / "artifacts" / "unit-surface-proof"
    surfaces._source_state_snapshot(artifact, label="post")

    assert calls == [
        (external.resolve(), "pre"),
        (artifact.resolve(), "post"),
    ]
    with pytest.raises(ValueError, match="source snapshots exclude it"):
        surfaces._source_state_snapshot(
            surfaces.REPO_ROOT / "unit-surface-proof",
            label="post",
        )
