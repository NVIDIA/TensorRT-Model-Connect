# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech-to-text comparator.

Compares TRT Whisper-style transcription output against reference with metrics:
- Decoder token agreement rate
- Word Error Rate (WER)
- Character Error Rate (CER)
- Timestamp sanity (if timestamps are available)
"""

from __future__ import annotations

import logging

from ..contracts import CompareResult, MetricResult, StageOutput, StageSpec, StageStatus, ThresholdProfile
from ._helpers import levenshtein_distance

logger = logging.getLogger(__name__)


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def _compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    distance = levenshtein_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def _token_agreement_rate(trt_tokens: list, ref_tokens: list) -> float:
    """Compute token-level agreement rate between two token sequences."""
    if not ref_tokens:
        return 1.0 if not trt_tokens else 0.0
    n = min(len(trt_tokens), len(ref_tokens))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if trt_tokens[i] == ref_tokens[i])
    return matches / max(len(trt_tokens), len(ref_tokens))


class SpeechToTextComparator:
    """Compares TRT transcription against reference transcription."""

    @property
    def task_strategy(self) -> str:
        return "speech_to_text"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        metrics: dict[str, MetricResult] = {}
        thresholds = threshold.metrics
        all_pass = True

        # Check TRT returncode
        if trt.data.get("returncode", -1) != 0:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                metrics={},
                message=f"TRT transcription failed (rc={trt.data.get('returncode')})",
            )

        trt_transcript = trt.data.get("transcript", trt.text or "")
        ref_transcript = ref.data.get("transcript", ref.text or "")

        # Token agreement
        trt_tokens = trt.data.get("token_ids", [])
        ref_tokens = ref.data.get("token_ids", [])
        if trt_tokens and ref_tokens:
            agreement = _token_agreement_rate(trt_tokens, ref_tokens)
            thresh = thresholds.get("token_agreement_rate", 0.8)
            ok = agreement >= thresh
            metrics["token_agreement_rate"] = MetricResult(
                value=agreement, threshold=thresh, operator=">=", passed=ok)
            if not ok:
                all_pass = False

        # Word Error Rate
        if trt_transcript and ref_transcript:
            wer = _compute_wer(ref_transcript, trt_transcript)
            wer_thresh = thresholds.get("wer", 0.1)
            wer_ok = wer <= wer_thresh
            metrics["wer"] = MetricResult(
                value=wer, threshold=wer_thresh, operator="<=", passed=wer_ok)
            if not wer_ok:
                all_pass = False

            # Character Error Rate
            cer = _compute_cer(ref_transcript, trt_transcript)
            cer_thresh = thresholds.get("cer", 0.05)
            cer_ok = cer <= cer_thresh
            metrics["cer"] = MetricResult(
                value=cer, threshold=cer_thresh, operator="<=", passed=cer_ok)
            if not cer_ok:
                all_pass = False

        # Timestamp sanity (if available)
        trt_timestamps = trt.data.get("timestamps", [])
        if trt_timestamps:
            ts_sane = self._check_timestamp_sanity(
                trt_timestamps,
                thresholds.get("timestamp_tolerance_s", 0.5),
            )
            metrics["timestamp_sanity"] = MetricResult(
                value=1.0 if ts_sane else 0.0, threshold=1.0, operator=">=", passed=ts_sane)
            if not ts_sane:
                all_pass = False

        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if all_pass else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all metrics must pass",
            message=f"{'PASS' if all_pass else 'FAIL'}: "
                    f"trt='{trt_transcript[:80]}' ref='{ref_transcript[:80]}'",
        )

    @staticmethod
    def _check_timestamp_sanity(timestamps: list, tolerance_s: float) -> bool:
        """Check that timestamps are monotonically increasing within tolerance."""
        if len(timestamps) < 2:
            return True
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1] - tolerance_s:
                return False
        return True


plugin = SpeechToTextComparator()
