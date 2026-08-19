# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tools.validation.gate_policy import (
    describe_shadow_gate_policy,
    evaluate_shadow_gates,
)
from tools.validation.catalog import load_suites


def test_rate_gate_expands_against_actual_sample_count() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"prediction_agreement_rate": 0.95},
        configured_gates={"min_prediction_agreement": 0.98},
        sample_count=20,
    )

    assert evaluation == {
        "schema_version": "trtmc.validation-gate-evaluation/v1",
        "status": "fail",
        "sample_count": 20,
        "checks": [
            {
                "gate": "min_prediction_agreement",
                "metric": "prediction_agreement_rate",
                "operator": ">=",
                "actual": 0.95,
                "required": 0.98,
                "verdict": "fail",
                "effective": {
                    "kind": "proportion",
                    "required_passes": 20,
                    "allowed_failures": 0,
                    "observed_passes": 19,
                    "observed_failures": 1,
                    "resolution": 0.05,
                },
            }
        ],
        "issues": [],
    }


def test_policy_description_expands_threshold_without_runtime_metrics() -> None:
    description = describe_shadow_gate_policy(
        configured_gates={"min_prediction_agreement": 0.98},
        sample_count=20,
    )

    assert description == {
        "schema_version": "trtmc.validation-gate-policy-description/v1",
        "policy_mode": "blocking",
        "sample_count": 20,
        "gates": [
            {
                "gate": "min_prediction_agreement",
                "metric": "prediction_agreement_rate",
                "operator": ">=",
                "required": 0.98,
                "effective": {
                    "kind": "proportion",
                    "required_passes": 20,
                    "allowed_failures": 0,
                    "resolution": 0.05,
                },
            }
        ],
        "issues": [],
    }


def test_policy_description_preserves_continuous_override() -> None:
    description = describe_shadow_gate_policy(
        configured_gates={"min_prediction_agreement": 0.9},
        sample_count=10,
        metric_kinds={"min_prediction_agreement": "continuous"},
    )

    assert description["gates"][0]["effective"] == {
        "kind": "continuous",
        "sample_count": 10,
    }


def test_policy_description_keeps_gate_visible_without_sample_count() -> None:
    description = describe_shadow_gate_policy(
        configured_gates={"min_prediction_agreement": 0.9},
        sample_count=None,
    )

    assert description["gates"] == [
        {
            "gate": "min_prediction_agreement",
            "metric": "prediction_agreement_rate",
            "operator": ">=",
            "required": 0.9,
            "effective": {"kind": "proportion", "sample_count": None},
        }
    ]
    assert description["issues"] == [{"code": "sample_count_unavailable"}]


def test_accuracy_drop_gate_exposes_discrete_loss_budget() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"accuracy_drop_from_hf": 0.05},
        configured_gates={"max_accuracy_drop_from_hf": 0.01},
        sample_count=20,
    )

    assert evaluation["status"] == "fail"
    assert evaluation["checks"] == [
        {
            "gate": "max_accuracy_drop_from_hf",
            "metric": "accuracy_drop_from_hf",
            "operator": "<=",
            "actual": 0.05,
            "required": 0.01,
            "verdict": "fail",
            "effective": {
                "kind": "proportion_drop",
                "allowed_drop_count": 0,
                "observed_drop_count": 1,
                "resolution": 0.05,
            },
        }
    ]


def test_unknown_gate_is_an_invalid_shadow_policy() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"prediction_agreement_rate": 1.0},
        configured_gates={"prediction_agreement": 0.98},
        sample_count=20,
    )

    assert evaluation["status"] == "invalid"
    assert evaluation["checks"] == []
    assert evaluation["issues"] == [
        {
            "code": "unsupported_gate",
            "gate": "prediction_agreement",
        }
    ]


def test_missing_metric_is_an_invalid_shadow_policy() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={},
        configured_gates={"min_prediction_agreement": 0.98},
        sample_count=20,
    )

    assert evaluation["status"] == "invalid"
    assert evaluation["checks"] == []
    assert evaluation["issues"] == [
        {
            "code": "metric_unavailable",
            "gate": "min_prediction_agreement",
            "metric": "prediction_agreement_rate",
        }
    ]


def test_continuous_gate_does_not_claim_an_integer_failure_budget() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"backend_mask_iou": 0.754},
        configured_gates={"min_backend_mask_iou": 0.70},
        sample_count=5,
    )

    assert evaluation["status"] == "pass"
    assert evaluation["checks"][0]["effective"] == {
        "kind": "continuous",
        "sample_count": 5,
    }


def test_tts_rate_gate_exposes_three_sample_zero_failure_smoke() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"correctness_agreement_rate": 2 / 3},
        configured_gates={"min_correctness_agreement": 0.95},
        sample_count=3,
    )

    assert evaluation["status"] == "fail"
    assert evaluation["checks"][0]["effective"] == {
        "kind": "proportion",
        "required_passes": 3,
        "allowed_failures": 0,
        "observed_passes": 2,
        "observed_failures": 1,
        "resolution": 1 / 3,
    }


def test_blocking_policy_with_no_gates_is_invalid() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={},
        configured_gates={},
        sample_count=20,
        policy_mode="blocking",
    )

    assert evaluation["status"] == "invalid"
    assert evaluation["issues"] == [{"code": "empty_gate_policy"}]


def test_observation_only_policy_can_explicitly_have_no_gates() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={},
        configured_gates={},
        sample_count=3,
        policy_mode="observation_only",
    )

    assert evaluation["status"] == "observation_only"
    assert evaluation["checks"] == []
    assert evaluation["issues"] == []


def test_blocking_policy_requires_an_actual_sample_count() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"prediction_agreement_rate": 1.0},
        configured_gates={"min_prediction_agreement": 0.98},
        sample_count=None,
    )

    assert evaluation["status"] == "invalid"
    assert evaluation["checks"] == []
    assert evaluation["issues"] == [{"code": "sample_count_unavailable"}]


def test_rate_gate_recalculates_for_each_actual_sample_count() -> None:
    expected = {
        3: (3, 0),
        5: (5, 0),
        10: (10, 0),
        20: (20, 0),
        50: (49, 1),
        100: (98, 2),
    }

    for sample_count, (required_passes, allowed_failures) in expected.items():
        evaluation = evaluate_shadow_gates(
            metrics={"prediction_agreement_rate": 1.0},
            configured_gates={"min_prediction_agreement": 0.98},
            sample_count=sample_count,
        )

        effective = evaluation["checks"][0]["effective"]
        assert effective["required_passes"] == required_passes
        assert effective["allowed_failures"] == allowed_failures


def test_direct_minimum_gate_uses_its_declared_metric() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"temporal_consistency": 0.72},
        configured_gates={"temporal_consistency": 0.6},
        sample_count=5,
    )

    assert evaluation["status"] == "pass"
    assert evaluation["checks"][0] == {
        "gate": "temporal_consistency",
        "metric": "temporal_consistency",
        "operator": ">=",
        "actual": 0.72,
        "required": 0.6,
        "verdict": "pass",
        "effective": {"kind": "continuous", "sample_count": 5},
    }


def test_direct_reranking_gate_uses_worst_sample_not_mean() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={
            "pairwise_ordering_agreement": 1.0,
            "min_pairwise_ordering_agreement": 0.5,
        },
        configured_gates={"pairwise_ordering_agreement": 1.0},
        sample_count=2,
    )

    assert evaluation["status"] == "fail"
    assert evaluation["checks"][0]["metric"] == "min_pairwise_ordering_agreement"
    assert evaluation["checks"][0]["actual"] == 0.5


def test_exact_gate_uses_equality_instead_of_a_range() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"num_frames": 120},
        configured_gates={"exact_num_frames": 121},
        sample_count=1,
    )

    assert evaluation["status"] == "fail"
    assert evaluation["checks"][0]["operator"] == "=="


def test_non_numeric_threshold_is_an_invalid_shadow_policy() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"prediction_agreement_rate": 1.0},
        configured_gates={"min_prediction_agreement": "strict"},
        sample_count=20,
    )

    assert evaluation["status"] == "invalid"
    assert evaluation["issues"] == [
        {
            "code": "invalid_threshold",
            "gate": "min_prediction_agreement",
            "value": "strict",
        }
    ]


def test_every_configured_suite_gate_is_understood_by_shadow_analysis() -> None:
    class AvailableMetrics(dict):
        def get(self, key, default=None):  # noqa: ANN001, ANN201
            return 1.0

        def __getitem__(self, key):  # noqa: ANN001, ANN201
            return 1.0

    unsupported: list[tuple[str, str]] = []
    for suite in load_suites():
        gates = suite.get("gates", {})
        if not gates:
            continue
        evaluation = evaluate_shadow_gates(
            metrics=AvailableMetrics(),
            configured_gates=gates,
            sample_count=20,
        )
        unsupported.extend(
            (str(suite["id"]), str(issue.get("gate", "")))
            for issue in evaluation["issues"]
            if issue["code"] == "unsupported_gate"
        )

    assert unsupported == []


def test_unknown_policy_mode_is_invalid() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"prediction_agreement_rate": 1.0},
        configured_gates={"min_prediction_agreement": 0.98},
        sample_count=20,
        policy_mode="optional",
    )

    assert evaluation["status"] == "invalid"
    assert evaluation["issues"] == [
        {"code": "unsupported_policy_mode", "value": "optional"}
    ]


def test_non_finite_metric_is_invalid() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"prediction_agreement_rate": float("nan")},
        configured_gates={"min_prediction_agreement": 0.98},
        sample_count=20,
    )

    assert evaluation["status"] == "invalid"
    assert evaluation["issues"] == [
        {
            "code": "invalid_metric",
            "gate": "min_prediction_agreement",
            "metric": "prediction_agreement_rate",
            "value": "nan",
        }
    ]


def test_workload_can_mark_rate_named_metric_as_continuous() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"prediction_agreement_rate": 0.8953},
        configured_gates={"min_prediction_agreement": 0.9},
        sample_count=20,
        metric_kinds={"min_prediction_agreement": "continuous"},
    )

    assert evaluation["status"] == "fail"
    assert evaluation["checks"][0]["effective"] == {
        "kind": "continuous",
        "sample_count": 20,
    }


def test_metric_kind_cannot_target_an_unconfigured_gate() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"prediction_agreement_rate": 1.0},
        configured_gates={"min_prediction_agreement": 0.98},
        sample_count=20,
        metric_kinds={"min_typo": "continuous"},
    )

    assert evaluation["status"] == "invalid"
    assert evaluation["issues"] == [
        {"code": "metric_kind_without_gate", "gate": "min_typo"}
    ]


def test_continuous_gate_keeps_its_configured_aggregate_semantics() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"pixel_mean": 0.5},
        configured_gates={"min_pixel_mean": 0.15, "max_pixel_mean": 0.85},
        sample_count=5,
    )

    assert evaluation["status"] == "pass"
    assert [check["actual"] for check in evaluation["checks"]] == [0.5, 0.5]
    assert [check["verdict"] for check in evaluation["checks"]] == ["pass", "pass"]


def test_exact_gate_fails_when_any_observed_value_differs() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"min_num_frames": 120, "max_num_frames": 121},
        configured_gates={"exact_num_frames": 121},
        sample_count=5,
    )

    assert evaluation["status"] == "fail"
    assert evaluation["checks"][0]["actual"] == {"min": 120.0, "max": 121.0}
    assert evaluation["checks"][0]["effective"] == {
        "kind": "exact",
        "sample_count": 5,
    }


def test_continuation_smoke_exposes_one_failure_budget_at_ten_samples() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"tie_adjusted_exact_match_rate": 0.9},
        configured_gates={"min_tie_adjusted_exact_match_rate": 0.9},
        sample_count=10,
    )

    assert evaluation["status"] == "pass"
    assert evaluation["checks"][0]["effective"] == {
        "kind": "proportion",
        "required_passes": 9,
        "allowed_failures": 1,
        "observed_passes": 9,
        "observed_failures": 1,
        "resolution": 0.1,
    }


def test_codegen_zero_failure_gate_rejects_one_divergence() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"exact_match_rate": 0.9},
        configured_gates={"min_exact_match_rate": 1.0},
        sample_count=10,
    )

    assert evaluation["status"] == "fail"
    assert evaluation["checks"][0]["effective"]["allowed_failures"] == 0
    assert evaluation["checks"][0]["effective"]["observed_failures"] == 1


def test_sam_shadow_does_not_invent_an_uncalibrated_tail_gate() -> None:
    evaluation = evaluate_shadow_gates(
        metrics={"mean_backend_mask_iou": 0.754, "min_mean_backend_mask_iou": 0.2},
        configured_gates={"min_backend_mask_iou": 0.7},
        sample_count=5,
    )

    assert evaluation["status"] == "pass"
    assert evaluation["checks"][0]["metric"] == "mean_backend_mask_iou"
    assert evaluation["checks"][0]["actual"] == 0.754
