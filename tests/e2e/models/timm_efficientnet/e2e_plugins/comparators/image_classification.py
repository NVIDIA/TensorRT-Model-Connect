# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-classification comparator."""

from __future__ import annotations

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


class ImageClassificationComparator:
    @property
    def task_strategy(self) -> str:
        return "image_classification"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: dict[str, MetricResult] = {}

        trt_top = trt.data.get("top_class")
        ref_top = ref.data.get("top_class")
        if trt_top is None:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message="TRT classification output missing top_class",
            )
        if ref_top is None:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics=metrics,
                message="Reference classification output missing top_class",
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
            ref_margin = ref.data.get("top1_margin")
            runner_up = ref.data.get("second_class")
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
                    passed=True,
                    message=(
                        "reference top-1 and top-2 are within "
                        f"{float(margin_atol)}; accepted the runner-up"
                    ),
                )

        metrics["top1_match"] = MetricResult(
            value=1.0 if top1_match else 0.0,
            threshold=1.0,
            operator="==",
            passed=top1_match,
        )

        if "top_score" in trt.data and "top_score" in ref.data:
            diff = abs(float(trt.data["top_score"]) - float(ref.data["top_score"]))
            score_atol = threshold.metrics.get("top_score_atol")
            metrics["top_score_abs_diff"] = MetricResult(
                value=diff,
                threshold=score_atol,
                operator="<=" if score_atol is not None else "informational",
                passed=True if score_atol is None else diff <= score_atol,
            )

        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="top-1 class must match",
            message=(
                f"Image classification: TRT top={int(trt_top)}, "
                f"reference top={int(ref_top)}"
            ),
        )


plugin = ImageClassificationComparator()
