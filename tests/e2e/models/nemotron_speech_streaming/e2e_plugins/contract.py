"""Nemotron Speech Streaming-owned speech-to-text contract plugin."""

from __future__ import annotations

import re

from tests.e2e_harness.contracts import CompareResult, MetricResult


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip().lower()


def levenshtein_ned(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        curr = [i + 1]
        for j, c2 in enumerate(b):
            curr.append(
                min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (0 if c1 == c2 else 1),
                )
            )
        prev = curr
    return prev[-1] / max_len


def make_pass(stage_name: str, metrics, rule: str = "") -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="passed",
        metrics=metrics,
        composite_rule=rule,
        message="Nemotron Speech Streaming ASR contract verified",
    )


def make_fail(stage_name: str, metrics, rule: str = "", message: str = "") -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="failed",
        metrics=metrics,
        composite_rule=rule,
        message=message or "Nemotron Speech Streaming ASR contract failed",
    )


def make_error(stage_name: str, error: str) -> CompareResult:
    return CompareResult(
        stage_name=stage_name,
        status="error",
        message=f"Contract verification error: {error}",
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
        if (
            i > 0
            and j > 0
            and ref_items[i - 1] == hyp_items[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
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


class NemotronSpeechStreamingASRPlugin:
    reference_families = ["asr_canary"]
    user_contract = "exact_transcript"

    def configure_reference(self, case):
        config = case.metadata.get("contract_config", {})
        return dict(config) if isinstance(config, dict) else {}

    def verify(self, trt_output, ref_output, case, threshold):
        del case
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
            threshold.metrics.get("cer", 0.05),
        )
        metrics = {
            "no_speech_state_match": MetricResult(
                value=1.0 if no_speech_state_match else 0.0,
                threshold=1.0,
                operator="==",
                passed=no_speech_state_match,
                note=f"trt={trt_no_speech_state} ref={ref_no_speech_state}",
            ),
            "trt_no_speech_state": MetricResult(
                value=_NO_SPEECH_STATE_VALUE[trt_no_speech_state],
                threshold=None,
                operator="",
                passed=True,
                note=trt_no_speech_state,
            ),
            "reference_no_speech_state": MetricResult(
                value=_NO_SPEECH_STATE_VALUE[ref_no_speech_state],
                threshold=None,
                operator="",
                passed=True,
                note=ref_no_speech_state,
            ),
            "normalized_text_edit_distance": MetricResult(
                value=ned,
                threshold=ned_threshold,
                operator="<=",
                passed=ned <= ned_threshold,
            ),
            "wer": MetricResult(
                value=wer,
                threshold=wer_threshold,
                operator="<=",
                passed=wer <= wer_threshold,
                note=(
                    f"matches={wer_breakdown['matches']} "
                    f"subs={wer_breakdown['substitutions']} "
                    f"ins={wer_breakdown['insertions']} "
                    f"del={wer_breakdown['deletions']}"
                ),
            ),
            "cer": MetricResult(
                value=cer,
                threshold=cer_threshold,
                operator="<=",
                passed=cer <= cer_threshold,
            ),
        }

        passed = (
            no_speech_state_match
            and ned <= ned_threshold
            and wer <= wer_threshold
            and cer <= cer_threshold
        )
        rule = "no-speech state match AND NED/WER/CER <= thresholds"
        if passed:
            return make_pass("full_generation", metrics, rule)
        return make_fail(
            "full_generation",
            metrics,
            rule,
            f"ASR transcript diverged: NED={ned:.3f} WER={wer:.3f} CER={cer:.3f}",
        )


plugin = NemotronSpeechStreamingASRPlugin()
