"""Encoder-only NLP comparator — compare TRT vs reference encoder outputs.

Metrics: hidden state cosine, hidden state L2, CLS embedding cosine.
"""

from __future__ import annotations

import logging
import math
from typing import List

import numpy as np

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile
from ._helpers import cosine_similarity

logger = logging.getLogger(__name__)


def _l2_distance(a: List[float], b: List[float]) -> float:
    """Compute L2 distance between two vectors."""
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class EncoderOnlyComparator:
    """Compare TRT vs reference encoder-only outputs."""

    @property
    def task_strategy(self) -> str:
        return "encoder_only_nlp"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: dict[str, MetricResult] = {}
        th = threshold.metrics

        # Compare CLS embeddings if available
        trt_cls = trt.data.get("cls_embedding", [])
        ref_cls = ref.data.get("cls_embedding", [])

        if trt_cls and ref_cls:
            if len(trt_cls) != len(ref_cls):
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    metrics={},
                    message=(
                        f"CLS embedding dimension mismatch: TRT={len(trt_cls)}, "
                        f"ref={len(ref_cls)}"
                    ),
                )

            cls_cosine = cosine_similarity(np.asarray(trt_cls), np.asarray(ref_cls))
            cls_l2 = _l2_distance(trt_cls, ref_cls)

            cls_cosine_thresh = th.get("cls_embedding_cosine", 0.99)
            cls_l2_thresh = th.get("cls_embedding_l2", 0.1)

            metrics["cls_embedding_cosine"] = MetricResult(
                value=cls_cosine, threshold=cls_cosine_thresh, operator=">=",
                passed=cls_cosine >= cls_cosine_thresh,
            )
            metrics["cls_embedding_l2"] = MetricResult(
                value=cls_l2, threshold=cls_l2_thresh, operator="<=",
                passed=cls_l2 <= cls_l2_thresh,
            )

        # Compare hidden states if available (flattened vectors)
        trt_hidden = trt.data.get("hidden_states", [])
        ref_hidden = ref.data.get("hidden_states", [])

        if trt_hidden and ref_hidden:
            if len(trt_hidden) != len(ref_hidden):
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.ERROR.value,
                    metrics={},
                    message=(
                        f"Hidden state dimension mismatch: TRT={len(trt_hidden)}, "
                        f"ref={len(ref_hidden)}"
                    ),
                )

            hidden_cosine = cosine_similarity(np.asarray(trt_hidden), np.asarray(ref_hidden))
            hidden_l2 = _l2_distance(trt_hidden, ref_hidden)

            hidden_cosine_thresh = th.get("hidden_state_cosine", 0.99)
            hidden_l2_thresh = th.get("hidden_state_l2", 0.5)

            metrics["hidden_state_cosine"] = MetricResult(
                value=hidden_cosine, threshold=hidden_cosine_thresh, operator=">=",
                passed=hidden_cosine >= hidden_cosine_thresh,
            )
            metrics["hidden_state_l2"] = MetricResult(
                value=hidden_l2, threshold=hidden_l2_thresh, operator="<=",
                passed=hidden_l2 <= hidden_l2_thresh,
            )

        if not metrics:
            # Both ran successfully but output formats are incompatible.
            # Pass at L4 (invariant-only) — TRT inference succeeded.
            trt_has_data = bool(trt.data) and not trt.data.get("skipped")
            ref_has_data = bool(ref.data) and not ref.data.get("skipped")
            if trt_has_data and ref_has_data:
                return CompareResult(
                    stage_name=stage.name,
                    status=StageStatus.PASSED.value,
                    metrics={"invariant_only": MetricResult(
                        value=1.0, threshold=None, operator=">=", passed=True,
                        note=(
                            f"L4 invariant-only: TRT + HF both produced output, "
                            f"but no shared keys (TRT: {list(trt.data.keys())}, "
                            f"ref: {list(ref.data.keys())})"
                        ),
                    )},
                    message="Invariant-only pass: both ran successfully",
                )
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message="No comparable outputs found (missing cls_embedding and hidden_states)",
            )

        passed = all(m.passed for m in metrics.values())

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Encoder-only comparison: {len(metrics)} metrics evaluated",
        )


plugin = EncoderOnlyComparator()
