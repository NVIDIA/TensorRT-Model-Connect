# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TIMM ViT-owned image classification contract plugin."""

from __future__ import annotations

from tests.e2e_harness.contracts import CompareResult, MetricResult


def _make_pass(stage_name: str, metrics, rule: str) -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="passed",
        metrics=metrics,
        composite_rule=rule,
        message="TIMM ViT image classification contract verified",
    )


def _make_fail(stage_name: str, metrics, rule: str, message: str) -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="failed",
        metrics=metrics,
        composite_rule=rule,
        message=message,
    )


class TimmRegnetImageClassificationPlugin:
    reference_families = ["image_classification"]
    user_contract = "image_classification"

    def configure_reference(self, case):
        del case
        return {}

    def verify(self, trt_output, ref_output, case, threshold):
        del case
        stage = trt_output.stage_name or "full_inference"
        metrics: dict[str, MetricResult] = {}
        trt_top = trt_output.data.get("top_class")
        ref_top = ref_output.data.get("top_class")
        if trt_top is None:
            return _make_fail(stage, metrics, "top-1 class must match", "TRT output missing top_class")
        if ref_top is None:
            return _make_fail(
                stage,
                metrics,
                "top-1 class must match",
                "Reference output missing top_class",
            )

        top1_match = int(trt_top) == int(ref_top)
        metrics["top1_match"] = MetricResult(
            value=1.0 if top1_match else 0.0,
            threshold=1.0,
            operator="==",
            passed=top1_match,
        )

        if "top_score" in trt_output.data and "top_score" in ref_output.data:
            diff = abs(float(trt_output.data["top_score"]) - float(ref_output.data["top_score"]))
            score_atol = threshold.metrics.get("top_score_atol")
            metrics["top_score_abs_diff"] = MetricResult(
                value=diff,
                threshold=score_atol,
                operator="<=" if score_atol is not None else "informational",
                passed=True if score_atol is None else diff <= score_atol,
            )

        passed = all(metric.passed for metric in metrics.values())
        rule = "top-1 class must match"
        if passed:
            return _make_pass(stage, metrics, rule)
        return _make_fail(
            stage,
            metrics,
            rule,
            f"TIMM ViT classification mismatch: TRT top={trt_top}, reference top={ref_top}",
        )


plugin = TimmRegnetImageClassificationPlugin()
