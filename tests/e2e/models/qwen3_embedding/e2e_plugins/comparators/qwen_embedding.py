# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-family embedding parity comparator."""

from __future__ import annotations

import numpy as np

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


class EmbeddingComparator:
    @property
    def task_strategy(self) -> str:
        return "embedding"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        trt_vector = np.asarray(trt.data.get("embedding", []), dtype=np.float64)
        ref_vector = np.asarray(ref.data.get("embedding", []), dtype=np.float64)
        if trt_vector.ndim != 1 or ref_vector.ndim != 1 or trt_vector.size == 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="Missing Qwen embedding vector",
            )
        if trt_vector.shape != ref_vector.shape:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=(
                    "Qwen embedding dimension mismatch: "
                    f"TRT={trt_vector.size}, reference={ref_vector.size}"
                ),
            )

        trt_norm = float(np.linalg.norm(trt_vector))
        ref_norm = float(np.linalg.norm(ref_vector))
        cosine = float(np.dot(trt_vector, ref_vector) / (trt_norm * ref_norm))
        l2_distance = float(np.linalg.norm(trt_vector - ref_vector))
        metrics_config = threshold.metrics
        cosine_threshold = float(metrics_config.get("cosine_similarity", 0.99))
        l2_threshold = float(metrics_config.get("l2_distance", 0.1))
        norm_tolerance = float(metrics_config.get("embedding_norm_tolerance", 0.001))
        metrics = {
            "cosine_similarity": MetricResult(
                value=cosine,
                threshold=cosine_threshold,
                operator=">=",
                passed=cosine >= cosine_threshold,
            ),
            "l2_distance": MetricResult(
                value=l2_distance,
                threshold=l2_threshold,
                operator="<=",
                passed=l2_distance <= l2_threshold,
            ),
            "trt_l2_norm_error": MetricResult(
                value=abs(trt_norm - 1.0),
                threshold=norm_tolerance,
                operator="<=",
                passed=abs(trt_norm - 1.0) <= norm_tolerance,
            ),
            "reference_l2_norm_error": MetricResult(
                value=abs(ref_norm - 1.0),
                threshold=norm_tolerance,
                operator="<=",
                passed=abs(ref_norm - 1.0) <= norm_tolerance,
            ),
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Qwen embedding cosine={cosine:.6f}, L2={l2_distance:.6f}",
        )


plugin = EmbeddingComparator()
