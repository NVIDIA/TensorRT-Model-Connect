"""Embedding comparator — compare TRT vs reference embedding outputs.

Metrics: cosine similarity, top-k neighborhood overlap, L2 distance.
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


def _topk_overlap(
    trt_emb: List[float], ref_emb: List[float], k: int
) -> float:
    """Compute overlap of top-k dimensions by absolute magnitude.

    Returns fraction of top-k dimensions that appear in both sets.
    """
    if len(trt_emb) != len(ref_emb) or len(trt_emb) == 0:
        return 0.0
    k = min(k, len(trt_emb))
    trt_topk = set(
        sorted(range(len(trt_emb)), key=lambda i: abs(trt_emb[i]), reverse=True)[:k]
    )
    ref_topk = set(
        sorted(range(len(ref_emb)), key=lambda i: abs(ref_emb[i]), reverse=True)[:k]
    )
    return len(trt_topk & ref_topk) / k


class EmbeddingComparator:
    """Compare TRT vs reference embedding outputs."""

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
            missing = []
            if not trt_emb:
                missing.append("TRT")
            if not ref_emb:
                missing.append("ref")
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"Missing embedding data from {', '.join(missing)}",
            )

        if len(trt_emb) != len(ref_emb):
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=(
                    f"Embedding dimension mismatch: TRT={len(trt_emb)}, "
                    f"ref={len(ref_emb)}"
                ),
            )

        cosine = cosine_similarity(np.asarray(trt_emb), np.asarray(ref_emb))
        l2 = _l2_distance(trt_emb, ref_emb)
        topk_10 = _topk_overlap(trt_emb, ref_emb, k=10)
        topk_100 = _topk_overlap(trt_emb, ref_emb, k=100)

        # Gate on thresholds
        th = threshold.metrics

        cosine_thresh = th.get("cosine_similarity", 0.99)
        l2_thresh = th.get("l2_distance", 0.1)
        topk10_thresh = th.get("topk_neighborhood_overlap_10", 0.8)
        topk100_thresh = th.get("topk_neighborhood_overlap_100", 0.7)

        metrics: dict[str, MetricResult] = {
            "cosine_similarity": MetricResult(
                value=cosine, threshold=cosine_thresh, operator=">=",
                passed=cosine >= cosine_thresh,
            ),
            "l2_distance": MetricResult(
                value=l2, threshold=l2_thresh, operator="<=",
                passed=l2 <= l2_thresh,
            ),
            "topk_neighborhood_overlap_10": MetricResult(
                value=topk_10, threshold=topk10_thresh, operator=">=",
                passed=topk_10 >= topk10_thresh,
            ),
            "topk_neighborhood_overlap_100": MetricResult(
                value=topk_100, threshold=topk100_thresh, operator=">=",
                passed=topk_100 >= topk100_thresh,
            ),
        }

        passed = all(m.passed for m in metrics.values())

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Embedding comparison: cosine={cosine:.6f}, L2={l2:.6f}",
        )


plugin = EmbeddingComparator()
