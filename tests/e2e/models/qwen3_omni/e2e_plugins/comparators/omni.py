"""Omni-multimodal comparator — compare TRT vs reference multi-branch outputs.

Metrics per branch: thinker token agreement, vision/audio embedding cosine,
talker token match, code2wav spectral distance, e2e text edit distance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile
from ._helpers import cosine_similarity, normalized_edit_distance

logger = logging.getLogger(__name__)


def _is_invariant_only(ref: StageOutput) -> bool:
    return (
        bool((ref.data or {}).get("_invariant_only"))
        or ref.metadata.get("source") == "invariant_only"
    )


def _token_agreement(a: List[int], b: List[int]) -> float:
    """Fraction of tokens that match between two sequences."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    matches = sum(1 for i in range(min_len) if a[i] == b[i])
    return matches / max(len(a), len(b))


class OmniComparator:
    """Compare TRT vs reference omni-multimodal outputs.

    Evaluates stage-specific metrics depending on the stage name.
    """

    @property
    def task_strategy(self) -> str:
        return "omni_multimodal"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        stage_name = stage.name
        th = threshold.metrics

        if _is_invariant_only(ref):
            return self._compare_invariants(trt, stage)

        # Dispatch to stage-specific comparison
        if stage_name == "thinker_decode":
            return self._compare_thinker(trt, ref, th, stage)
        elif stage_name in ("vision_encode", "audio_encode"):
            return self._compare_encoder(trt, ref, th, stage)
        elif stage_name == "talker_decode":
            return self._compare_talker(trt, ref, th, stage)
        elif stage_name == "end_to_end":
            return self._compare_e2e(trt, ref, th, stage)
        else:
            return self._compare_generic(trt, ref, th, stage)

    def _compare_thinker(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Compare thinker text decoding output."""
        trt_tokens = trt.data.get("token_ids", [])
        ref_tokens = ref.data.get("token_ids", [])

        metrics: Dict[str, MetricResult] = {}

        if trt_tokens and ref_tokens:
            agreement = _token_agreement(trt_tokens, ref_tokens)
            thresh = th.get("thinker_token_agreement", 0.8)
            metrics["thinker_token_agreement"] = MetricResult(
                value=agreement, threshold=thresh, operator=">=", passed=agreement >= thresh)

        trt_text = trt.text or ""
        ref_text = ref.text or ""
        if trt_text or ref_text:
            ned = normalized_edit_distance(trt_text, ref_text)
            thresh = th.get("thinker_text_edit_distance", 0.3)
            metrics["thinker_text_edit_distance"] = MetricResult(
                value=ned, threshold=thresh, operator="<=", passed=ned <= thresh)

        passed = all(m.passed for m in metrics.values()) if metrics else False
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Thinker comparison: {len(metrics)} metrics",
        )

    def _compare_encoder(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Compare vision/audio encoder embedding output."""
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
                message=f"Missing embedding for {stage.name} from {', '.join(missing)}",
            )

        cosine = cosine_similarity(np.asarray(trt_emb), np.asarray(ref_emb))
        # Use canonical name from threshold defaults (e.g. "vision_embedding_cosine")
        branch = stage.name.replace("_encode", "")
        metric_name = f"{branch}_embedding_cosine"
        # Look up threshold: canonical name -> stage-based name -> generic fallback
        thresh = th.get(metric_name, th.get(
            f"{stage.name}_embedding_cosine", th.get("encoder_embedding_cosine", 0.95)))
        metrics: Dict[str, MetricResult] = {
            metric_name: MetricResult(
                value=cosine, threshold=thresh, operator=">=", passed=cosine >= thresh),
        }

        passed = all(m.passed for m in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"{stage.name} embedding cosine={cosine:.6f}",
        )

    def _compare_talker(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Compare talker decoding output (tokens and/or audio)."""
        metrics: Dict[str, MetricResult] = {}

        trt_tokens = trt.data.get("token_ids", [])
        ref_tokens = ref.data.get("token_ids", [])

        if trt_tokens and ref_tokens:
            agreement = _token_agreement(trt_tokens, ref_tokens)
            thresh = th.get("talker_token_match", 0.7)
            metrics["talker_token_match"] = MetricResult(
                value=agreement, threshold=thresh, operator=">=", passed=agreement >= thresh)

        passed = all(m.passed for m in metrics.values()) if metrics else True
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"Talker comparison: {len(metrics)} metrics",
        )

    def _compare_e2e(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Compare end-to-end omni output (text edit distance)."""
        trt_text = trt.text or ""
        ref_text = ref.text or ""

        ned = normalized_edit_distance(trt_text, ref_text)
        thresh = th.get("e2e_text_edit_distance", 0.3)
        metrics: Dict[str, MetricResult] = {
            "e2e_text_edit_distance": MetricResult(
                value=ned, threshold=thresh, operator="<=", passed=ned <= thresh),
        }

        passed = all(m.passed for m in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"E2E text edit distance={ned:.4f}",
        )

    def _compare_generic(
        self, trt: StageOutput, ref: StageOutput,
        th: Dict[str, float], stage: StageSpec,
    ) -> CompareResult:
        """Fallback: compare any stage with available data."""
        # Try embedding comparison
        trt_emb = trt.data.get("embedding", [])
        ref_emb = ref.data.get("embedding", [])
        if trt_emb and ref_emb:
            return self._compare_encoder(trt, ref, th, stage)

        # Try text comparison
        if trt.text and ref.text:
            return self._compare_e2e(trt, ref, th, stage)

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.SKIPPED.value,
            metrics={},
            message=f"No comparable data for stage {stage.name} (skipped)",
        )

    def _compare_invariants(
        self,
        trt: StageOutput,
        stage: StageSpec,
    ) -> CompareResult:
        """L4 invariant checks for omni stages without a strong reference."""
        metrics: Dict[str, MetricResult] = {}

        if stage.name == "talker_decode":
            audio_path = str(trt.metadata.get("audio_output_path", "") or "")
            audio_size = 0
            if audio_path:
                path = Path(audio_path)
                if path.is_file():
                    audio_size = path.stat().st_size
            has_audio = audio_size > 0
            metrics["audio_artifact_bytes"] = MetricResult(
                value=float(audio_size),
                threshold=1.0,
                operator=">=",
                passed=has_audio,
                note="generated audio artifact is non-empty",
            )
        elif stage.name in ("thinker_decode", "end_to_end"):
            text = trt.text or str(trt.data.get("text", "") or "")
            if not text:
                text = str(trt.data.get("raw_output", "") or "")
            has_text = bool(text.strip())
            metrics["non_empty_text"] = MetricResult(
                value=1.0 if has_text else 0.0,
                threshold=1.0,
                operator="==",
                passed=has_text,
                note="generated text/raw output is non-empty",
            )
        else:
            has_output = bool(trt.text) or any(
                value not in ("", None, [], {})
                for value in (trt.data or {}).values()
            )
            metrics["non_empty_output"] = MetricResult(
                value=1.0 if has_output else 0.0,
                threshold=1.0,
                operator="==",
                passed=has_output,
                note="stage produced observable output",
            )

        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all invariant metrics must pass",
            message=f"{stage.name} invariant check",
        )


plugin = OmniComparator()
