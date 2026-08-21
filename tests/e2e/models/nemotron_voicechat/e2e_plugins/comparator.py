# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict model-card contract for native Nemotron VoiceChat output."""

from __future__ import annotations

import re

from tests.e2e_harness.contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _similarity(left: str, right: str) -> float:
    left = _normalized(left)
    right = _normalized(right)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _metric(value: float, threshold: float, operator: str, passed: bool) -> MetricResult:
    return MetricResult(value=value, threshold=threshold, operator=operator, passed=passed)


class VoiceChatModelCardComparator:
    @property
    def task_strategy(self) -> str:
        return "speech_to_speech"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        actual = trt.data
        expected = ref.data
        source = actual.get("source_stats", {})
        output = actual.get("output_stats", {})
        metrics: dict[str, MetricResult] = {}

        def exact(name: str, value: object, target: object) -> None:
            passed = value == target
            numeric = float(value) if isinstance(value, (int, float, bool)) else float(passed)
            expected_numeric = float(target) if isinstance(target, (int, float, bool)) else 1.0
            metrics[name] = _metric(numeric, expected_numeric, "==", passed)

        exact(
            "source_sha256_match",
            actual.get("source_sha256"),
            expected.get("speech_source_sha256"),
        )
        exact("source_channels", source.get("channels"), 1)
        exact(
            "source_sample_rate",
            source.get("sample_rate"),
            expected.get("speech_source_sample_rate"),
        )
        exact(
            "source_num_samples",
            source.get("num_samples"),
            expected.get("speech_source_num_samples"),
        )
        exact("output_channels", output.get("channels"), 1)
        exact("output_encoding", output.get("encoding"), "ieee_float32le")
        exact("output_all_finite", output.get("all_finite"), True)
        exact(
            "output_sample_rate",
            output.get("sample_rate"),
            expected.get("expected_output_sample_rate"),
        )
        exact(
            "output_num_samples",
            output.get("num_samples"),
            expected.get("expected_output_num_samples"),
        )
        exact("cli_generated_count", actual.get("generated_count"), output.get("num_samples"))

        samples_per_frame = int(expected["expected_output_samples_per_frame"])
        output_samples = int(output.get("num_samples", 0) or 0)
        codec_frames = output_samples // samples_per_frame if samples_per_frame else 0
        exact("codec_frame_alignment", output_samples % samples_per_frame, 0)
        exact("codec_frame_count", codec_frames, expected["expected_output_codec_frames"])

        input_frames = (int(source.get("num_samples", 0) or 0) + 1280 - 1) // 1280 + int(
            actual.get("tail_frames", 0) or 0
        )
        exact("session_frame_mapping", codec_frames, input_frames)

        rms = float(output.get("rms", 0.0) or 0.0)
        rms_floor = float(threshold.metrics.get("audio_min_rms", 0.001))
        metrics["audio_rms"] = _metric(rms, rms_floor, ">=", rms >= rms_floor)
        peak = float(output.get("peak", 0.0) or 0.0)
        peak_floor = float(threshold.metrics.get("audio_min_peak", 0.01))
        metrics["audio_peak"] = _metric(peak, peak_floor, ">=", peak >= peak_floor)

        agent_text = str(actual.get("agent_text", ""))
        exact("agent_text_stdout_line_count", actual.get("agent_text_line_count"), 1)
        required = [str(term).lower() for term in expected["required_response_terms"]]
        normalized_agent_text = _normalized(agent_text)
        terms_found = sum(term in normalized_agent_text for term in required)
        metrics["agent_required_response_terms"] = _metric(
            float(terms_found), float(len(required)), "==", terms_found == len(required)
        )
        agent_similarity = _similarity(agent_text, str(expected["expected_response_text"]))
        agent_similarity_floor = float(threshold.metrics.get("agent_text_min_similarity", 0.75))
        metrics["agent_text_similarity"] = _metric(
            agent_similarity,
            agent_similarity_floor,
            ">=",
            agent_similarity >= agent_similarity_floor,
        )

        transcript = str(actual.get("transcript", ""))
        exact("transcript_stdout_line_count", actual.get("transcript_line_count"), 1)
        words = _normalized(transcript).split()
        min_words = float(threshold.metrics.get("transcript_min_words", 8))
        metrics["transcript_word_count"] = _metric(
            float(len(words)), min_words, ">=", len(words) >= min_words
        )
        similarity = _similarity(transcript, str(expected["expected_response_text"]))
        similarity_floor = float(threshold.metrics.get("transcript_min_similarity", 0.35))
        metrics["transcript_similarity"] = _metric(
            similarity, similarity_floor, ">=", similarity >= similarity_floor
        )

        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all model-card audio, text, and session-frame gates must pass",
            message=f"VoiceChat model-card contract: {sum(m.passed for m in metrics.values())}/"
            f"{len(metrics)} gates passed",
        )


comparator = VoiceChatModelCardComparator()
