"""Canary-owned speech-to-text contract plugin."""

from __future__ import annotations

import re

from tests.e2e_harness.contracts import MetricResult
from tests.e2e_harness.plugins.base import (
    levenshtein_ned,
    make_error,
    make_fail,
    make_pass,
    normalize_text,
)

_NO_SPEECH_STATE_VALUE = {
    "speech": 0.0,
    "empty": 1.0,
    "blank_audio_token": 2.0,
}


def _edit_breakdown(ref_items, hyp_items):
    rows = len(ref_items) + 1
    cols = len(hyp_items) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            sub_cost = 0 if ref_items[i - 1] == hyp_items[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    i = len(ref_items)
    j = len(hyp_items)
    matches = substitutions = insertions = deletions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_items[i - 1] == hyp_items[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            matches += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return {
        "matches": matches,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
    }


def _word_error_rate(ref_words, hyp_words):
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    breakdown = _edit_breakdown(ref_words, hyp_words)
    errors = breakdown["substitutions"] + breakdown["insertions"] + breakdown["deletions"]
    return errors / len(ref_words)


def _wer_words(text):
    words = []
    for word in str(text or "").split():
        stripped = re.sub(r"^[^\w]+|[^\w]+$", "", word).lower()
        if stripped:
            words.append(stripped)
    return words


def _character_error_rate(ref_text, hyp_text):
    if not ref_text:
        return 0.0 if not hyp_text else 1.0
    breakdown = _edit_breakdown(list(ref_text), list(hyp_text))
    errors = breakdown["substitutions"] + breakdown["insertions"] + breakdown["deletions"]
    return errors / len(ref_text)


def _no_speech_state(text):
    stripped = str(text or "").strip()
    if not stripped:
        return "empty"
    if re.fullmatch(r"\[?\s*blank[\s_-]*audio\s*\]?", stripped, flags=re.IGNORECASE):
        return "blank_audio_token"
    return "speech"


class CanaryASRPlugin:
    reference_families = ["asr_canary"]
    user_contract = "exact_transcript"

    def configure_reference(self, case):
        return {"auto_class": "AutoModelForSpeechSeq2Seq", "canary": True}

    def verify(self, trt_output, ref_output, case, threshold):
        raw_trt_text = trt_output.data.get("transcript", trt_output.text or "")
        raw_ref_text = ref_output.data.get("transcript", ref_output.text or "")
        trt_no_speech_state = _no_speech_state(raw_trt_text)
        ref_no_speech_state = _no_speech_state(raw_ref_text)
        no_speech_state_match = trt_no_speech_state == ref_no_speech_state

        trt_text = normalize_text(raw_trt_text)
        ref_text = normalize_text(raw_ref_text)

        if not ref_text:
            return make_error("full_generation", "Reference produced empty transcript")

        ned = levenshtein_ned(trt_text, ref_text)

        trt_words = _wer_words(trt_text)
        ref_words = _wer_words(ref_text)
        wer = _word_error_rate(ref_words, trt_words)
        wer_breakdown = _edit_breakdown(ref_words, trt_words)
        cer = _character_error_rate(ref_text, trt_text)

        ned_threshold = threshold.metrics.get(
            "contract_ned_threshold",
            threshold.metrics.get("normalized_text_edit_distance", 0.1),
        )
        wer_threshold = threshold.metrics.get(
            "contract_wer_threshold",
            threshold.metrics.get("wer", 0.1),
        )
        cer_threshold = threshold.metrics.get(
            "contract_cer_threshold",
            threshold.metrics.get("cer", 0.1),
        )

        metrics = {
            "ned": MetricResult(value=ned, threshold=ned_threshold, operator="<=", passed=ned <= ned_threshold),
            "wer": MetricResult(value=wer, threshold=wer_threshold, operator="<=", passed=wer <= wer_threshold),
            "cer": MetricResult(
                value=cer,
                threshold=cer_threshold,
                operator="<=",
                passed=cer <= cer_threshold,
                note="ASR-specific informational metric; not part of composite gate yet",
            ),
            "wer_substitutions": MetricResult(
                value=float(wer_breakdown["substitutions"]),
                note="ASR-specific informational WER breakdown",
            ),
            "wer_insertions": MetricResult(
                value=float(wer_breakdown["insertions"]),
                note="ASR-specific informational WER breakdown",
            ),
            "wer_deletions": MetricResult(
                value=float(wer_breakdown["deletions"]),
                note="ASR-specific informational WER breakdown",
            ),
            "trt_no_speech_state": MetricResult(
                value=_NO_SPEECH_STATE_VALUE[trt_no_speech_state],
                note=f"0=speech, 1=empty, 2=blank_audio_token; observed={trt_no_speech_state}",
            ),
            "reference_no_speech_state": MetricResult(
                value=_NO_SPEECH_STATE_VALUE[ref_no_speech_state],
                note=f"0=speech, 1=empty, 2=blank_audio_token; observed={ref_no_speech_state}",
            ),
            "no_speech_state_match": MetricResult(
                value=1.0 if no_speech_state_match else 0.0,
                note="Informational only; gate after silence/blank-audio user contract is agreed",
            ),
        }

        passed = ned <= ned_threshold and wer <= wer_threshold
        rule = "ned <= threshold AND wer <= threshold"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"Transcript diverged: WER={wer:.3f} NED={ned:.3f} CER={cer:.3f}",
        )


plugin = CanaryASRPlugin()
