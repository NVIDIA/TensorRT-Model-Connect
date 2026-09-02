# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Embedding external-reference acceptance contract."""

from __future__ import annotations

import numpy as np

from tests.e2e_harness.contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    StageOutput,
    ThresholdProfile,
)


class Qwen3EmbeddingContract:
    reference_families = ["qwen3_embedding"]
    user_contract = "embedding_vector"

    def configure_reference(self, case: E2ECase) -> dict:
        del case
        return {}

    def verify(
        self,
        trt_output: StageOutput,
        ref_output: StageOutput,
        case: E2ECase,
        threshold: ThresholdProfile,
    ) -> CompareResult:
        del case
        trt_vector = np.asarray(trt_output.data.get("embedding", []), dtype=np.float64)
        ref_vector = np.asarray(ref_output.data.get("embedding", []), dtype=np.float64)
        if trt_vector.ndim != 1 or ref_vector.ndim != 1 or trt_vector.size == 0:
            return CompareResult(
                stage_name="full_inference",
                status="error",
                message="Missing Qwen3-Embedding vector",
            )
        if trt_vector.shape != ref_vector.shape:
            return CompareResult(
                stage_name="full_inference",
                status="error",
                message=(
                    "Qwen3-Embedding dimensions differ: "
                    f"{trt_vector.size} != {ref_vector.size}"
                ),
            )

        trt_norm = float(np.linalg.norm(trt_vector))
        ref_norm = float(np.linalg.norm(ref_vector))
        if trt_norm <= 0.0 or ref_norm <= 0.0:
            return CompareResult(
                stage_name="full_inference",
                status="error",
                message="Qwen3-Embedding vector has zero norm",
            )
        cosine = float(np.dot(trt_vector, ref_vector) / (trt_norm * ref_norm))
        l2_distance = float(np.linalg.norm(trt_vector - ref_vector))
        cosine_threshold = float(threshold.metrics.get("cosine_similarity", 0.99))
        l2_threshold = float(threshold.metrics.get("l2_distance", 0.1))
        norm_tolerance = float(
            threshold.metrics.get("embedding_norm_tolerance", 0.001)
        )
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
            stage_name="full_inference",
            status="passed" if passed else "failed",
            metrics=metrics,
            composite_rule="cosine AND L2 distance AND both unit norms",
            message=(
                "Qwen3-Embedding parity "
                f"cosine={cosine:.6f}, L2={l2_distance:.6f}"
            ),
        )


plugin = Qwen3EmbeddingContract()
