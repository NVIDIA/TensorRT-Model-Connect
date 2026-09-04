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


class TimmEfficientnetImageClassificationPlugin:
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

        # When the reference is itself close to a tie, top-1 equality stops
        # being a meaningful assertion: the harness resizes with stb while the
        # reference resizes with PIL, and an fp16 engine carries its own error,
        # so either can pick the runner-up without the model being wrong. If the
        # case declares top1_margin_atol and the reference's own top-1/top-2 gap
        # is inside it, accept the runner-up.
        margin_atol = threshold.metrics.get("top1_margin_atol")
        if not top1_match and margin_atol is not None:
            ref_margin = ref_output.data.get("top1_margin")
            runner_up = ref_output.data.get("second_class")
            if (
                ref_margin is not None
                and runner_up is not None
                and float(ref_margin) <= float(margin_atol)
                and int(trt_top) == int(runner_up)
            ):
                top1_match = True
                metrics["top1_margin"] = MetricResult(
                    value=float(ref_margin),
                    threshold=float(margin_atol),
                    operator="<=",
                    passed=True,
                )

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
        rule = "top-1 class must match, or the reference must be within top1_margin_atol of a tie"
        if passed:
            return _make_pass(stage, metrics, rule)
        return _make_fail(
            stage,
            metrics,
            rule,
            f"TIMM ViT classification mismatch: TRT top={trt_top}, reference top={ref_top}",
        )


plugin = TimmEfficientnetImageClassificationPlugin()
