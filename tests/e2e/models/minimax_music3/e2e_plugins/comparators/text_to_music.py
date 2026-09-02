# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional comparator for MiniMax-Music3 audio output.

The gates are the ones tts_audio already defines for the repository's two
speech families: the waveform exists, carries signal, runs about as long as
asked, and transcribes back to the text it was given. Here that text is the
lyrics, which is what turns the transcription score into the
lyric-intelligibility check the onboarding request asks for.

The sung-lyric edit distance is looser than the 0.15 speech uses. That number
is a starting point, not a measurement: the first qualified run should replace
it with what this model actually achieves rather than leave a guess in place.
"""

from __future__ import annotations

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)

REQUIRED_THRESHOLDS = frozenset({
    "contract_min_rms",
    "contract_min_duration_s",
    "contract_max_duration_s",
    "contract_asr_ned_threshold",
    "sampling_rate",
})


def _metric(value, threshold, operator, passed, note=""):
    return MetricResult(
        value=value, threshold=threshold, operator=operator, passed=passed, note=note
    )


class TextToMusicComparator:
    @property
    def task_strategy(self) -> str:
        return "text_to_audio"

    def required_thresholds(self) -> frozenset[str]:
        return REQUIRED_THRESHOLDS

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        missing = sorted(REQUIRED_THRESHOLDS - set(threshold.metrics))
        if missing:
            raise ValueError(
                "minimax_music3 comparator needs thresholds: " + ", ".join(missing)
            )

        metrics: dict[str, MetricResult] = {}
        data = trt.data or {}

        rate = int(data.get("sampling_rate", 0))
        expected_rate = int(threshold.metrics["sampling_rate"])
        metrics["sampling_rate"] = _metric(
            float(rate), float(expected_rate), "==", rate == expected_rate,
            "the vocoder's rate, not the model card's prose",
        )

        channels = int(data.get("channels", 0))
        metrics["channels"] = _metric(
            float(channels), 2.0, "==", channels == 2,
            "two folded streams through one mono decoder",
        )

        rms = data.get("rms")
        if rms is not None:
            floor = float(threshold.metrics["contract_min_rms"])
            metrics["rms"] = _metric(float(rms), floor, ">=", float(rms) >= floor,
                                     "non-silence")

        duration = data.get("duration_s")
        if duration is not None:
            low = float(threshold.metrics["contract_min_duration_s"])
            high = float(threshold.metrics["contract_max_duration_s"])
            metrics["duration_s"] = _metric(
                float(duration), low, ">=", low <= float(duration) <= high,
                f"range [{low}, {high}]",
            )

        ned = data.get("asr_ned")
        if ned is not None:
            limit = float(threshold.metrics["contract_asr_ned_threshold"])
            metrics["asr_ned"] = _metric(
                float(ned), limit, "<=", float(ned) <= limit,
                "transcribed lyrics against the prompt",
            )

        gated = [m for m in metrics.values() if m.threshold is not None]
        passed = bool(gated) and all(m.passed for m in gated)
        rule = (
            "audio exists at the vocoder's rate in stereo AND carries signal "
            "AND runs the requested length AND its transcript matches the lyrics"
        )
        return CompareResult(
            stage_name=stage.name,
            status="passed" if passed else "failed",
            metrics=metrics,
            composite_rule=rule,
            message="Audio contract verified"
            if passed
            else "MiniMax-Music3 audio contract failed",
        )


plugin = TextToMusicComparator()
