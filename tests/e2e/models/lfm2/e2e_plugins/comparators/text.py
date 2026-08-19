# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict text/token comparator for dense LFM2 continuation parity."""

from __future__ import annotations

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, ThresholdProfile
from ._helpers import normalized_edit_distance


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


class TextComparator:
    @property
    def task_strategy(self) -> str:
        return "text_generation_causal"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        trt_tokens = [int(token) for token in (trt.data or {}).get("token_ids", [])]
        ref_tokens = [int(token) for token in (ref.data or {}).get("token_ids", [])]
        tokens_available = bool(trt_tokens) and bool(ref_tokens)
        token_exact = tokens_available and trt_tokens == ref_tokens

        trt_text = _normalize(trt.text or "")
        ref_text = _normalize(ref.text or "")
        text_ned = normalized_edit_distance(trt_text, ref_text)
        max_ned = float(threshold.metrics.get("normalized_text_edit_distance", 0.0))
        text_ok = bool(trt_text) and text_ned <= max_ned

        metrics = {
            "token_exact": MetricResult(
                value=1.0 if token_exact else 0.0,
                threshold=1.0,
                operator="==",
                passed=token_exact,
                note=(
                    f"TRT={trt_tokens}, HF={ref_tokens}"
                    if tokens_available
                    else "generated token IDs missing"
                ),
            ),
            "normalized_text_edit_distance": MetricResult(
                value=text_ned,
                threshold=max_ned,
                operator="<=",
                passed=text_ok,
            ),
        }
        passed = token_exact and text_ok
        return CompareResult(
            stage_name=stage.name,
            status="passed" if passed else "failed",
            metrics=metrics,
            composite_rule="token_exact AND normalized_text_edit_distance <= threshold",
            message=(
                "Exact LFM2 continuation parity verified"
                if passed
                else "LFM2 continuation diverged from the pinned Transformers oracle"
            ),
        )


plugin = TextComparator()
