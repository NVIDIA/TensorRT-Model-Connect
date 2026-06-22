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
