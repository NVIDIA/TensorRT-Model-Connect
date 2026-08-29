# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BERT-owned embedding comparator with family-specific vector checks."""

from __future__ import annotations

import numpy as np

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile
from ._helpers import cosine_similarity


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
        trt_emb = trt.data.get("embedding", [])
        ref_emb = ref.data.get("embedding", [])
        if not trt_emb or not ref_emb:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message="Missing embedding data from TRT or reference",
            )
        if len(trt_emb) != len(ref_emb):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"Embedding dimension mismatch: TRT={len(trt_emb)}, ref={len(ref_emb)}",
            )

        trt_arr = np.asarray(trt_emb, dtype=np.float32)
        ref_arr = np.asarray(ref_emb, dtype=np.float32)
        cosine = cosine_similarity(trt_arr, ref_arr)
        l2 = float(np.linalg.norm(trt_arr - ref_arr))
        trt_norm_error = abs(float(np.linalg.norm(trt_arr)) - 1.0)
        ref_norm_error = abs(float(np.linalg.norm(ref_arr)) - 1.0)
        limits = threshold.metrics
        values = {
            "cosine_similarity": (cosine, limits.get("cosine_similarity", 0.99), ">="),
            "l2_distance": (l2, limits.get("l2_distance", 0.1), "<="),
            "trt_unit_norm_error": (
                trt_norm_error,
                limits.get("trt_unit_norm_error", 0.001),
                "<=",
            ),
            "reference_unit_norm_error": (
                ref_norm_error,
                limits.get("reference_unit_norm_error", 0.001),
                "<=",
            ),
        }
        metrics = {
            name: MetricResult(
                value=value,
                threshold=limit,
                operator=operator,
                passed=value >= limit if operator == ">=" else value <= limit,
            )
            for name, (value, limit, operator) in values.items()
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Embedding comparison: cosine={cosine:.6f}, L2={l2:.6f}",
        )


plugin = EmbeddingComparator()
