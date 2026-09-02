# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict model-card contract for native Nemotron VoiceChat output."""

from __future__ import annotations

import re
from typing import Any

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
        if stage.name == "native_full_duplex_lifecycle":
            return self._compare_lifecycle(trt, ref, stage)
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
        exact(
            "function_source_sha256_match",
            actual.get("function_source_sha256"),
            expected.get("function_speech_source_sha256"),
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

    def _compare_lifecycle(
        self, trt: StageOutput, ref: StageOutput, stage: StageSpec
    ) -> CompareResult:
        actual = trt.data
        expected = ref.data
        receipt_value = actual.get("receipt")
        receipt = receipt_value if isinstance(receipt_value, dict) else {}
        metrics: dict[str, MetricResult] = {}

        def exact(name: str, value: object, target: object) -> None:
            passed = value == target
            numeric = float(value) if isinstance(value, (int, float, bool)) else float(passed)
            target_numeric = float(target) if isinstance(target, (int, float, bool)) else 1.0
            metrics[name] = _metric(numeric, target_numeric, "==", passed)

        def at_least(name: str, value: object, minimum: float) -> None:
            numeric = (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else minimum - 1.0
            )
            metrics[name] = _metric(numeric, minimum, ">=", numeric >= minimum)

        def at_most(name: str, value: object, maximum: float) -> None:
            numeric = (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else maximum + 1.0
            )
            metrics[name] = _metric(numeric, maximum, "<=", numeric <= maximum)

        def greater(name: str, value: object, lower: object) -> None:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(lower, (int, float))
                and not isinstance(lower, bool)
            )
            numeric = float(value) if valid else -1.0
            lower_numeric = float(lower) if valid else 0.0
            metrics[name] = _metric(numeric, lower_numeric, ">", valid and numeric > lower_numeric)

        def section(name: str) -> dict[str, Any]:
            value = receipt.get(name)
            exact(f"section_{name}_present", isinstance(value, dict), True)
            return value if isinstance(value, dict) else {}

        exact("receipt_is_object", isinstance(receipt_value, dict), True)
        exact(
            "receipt_schema_version", receipt.get("schema_version"), expected.get("schema_version")
        )
        exact(
            "runtime_identity",
            receipt.get("runtime"),
            "C++ ISpeechSession with TensorRT backend",
        )
        exact("runtime_path_confirmed", actual.get("runtime_path_confirmed"), True)
        exact(
            "source_sha256_match",
            actual.get("source_sha256"),
            expected.get("speech_source_sha256"),
        )
        exact(
            "function_source_sha256_match",
            actual.get("function_source_sha256"),
            expected.get("function_speech_source_sha256"),
        )

        required_sections = expected.get("required_sections", [])
        if not isinstance(required_sections, list):
            required_sections = []
        sections = {name: section(str(name)) for name in required_sections}
        baseline = sections.get("baseline", {})
        irregular = sections.get("irregular_chunking", {})
        barge = sections.get("barge_in", {})
        cancel = sections.get("cancel", {})
        reset = sections.get("reset_vs_fresh", {})
        clear = sections.get("processed_input_clear", {})
        response_cancel = sections.get("response_cancel_recovery", {})
        response_truncate = sections.get("response_truncate_recovery", {})
        tail = sections.get("partial_finish_tail", {})
        sequence = sections.get("sequence_continuity", {})
        media = sections.get("media_continuity", {})
        multiturn = sections.get("normal_multiturn", {})
        function = sections.get("function_channel", {})
        concurrency = sections.get("backpressure_concurrency", {})

        frame_samples = int(expected.get("expected_output_samples_per_frame", 0) or 0)
        codec_frames = int(expected.get("expected_output_codec_frames", 0) or 0)
        expected_samples = int(expected.get("expected_output_num_samples", 0) or 0)
        audio_rms_floor = float(expected.get("audio_min_rms", 0.001))
        audio_peak_floor = float(expected.get("audio_min_peak", 0.01))
        exact("baseline_output_samples", baseline.get("output_samples"), expected_samples)
        exact("baseline_audio_events", baseline.get("audio_events"), codec_frames)
        exact(
            "baseline_agent_text",
            baseline.get("agent_text"),
            expected.get("expected_response_text"),
        )
        exact("baseline_input_finished", baseline.get("input_finished_events"), 1)
        exact("baseline_audio_hash_present", bool(baseline.get("audio_fnv1a64")), True)

        at_least("irregular_append_calls", irregular.get("append_calls"), 2.0)
        at_least(
            "irregular_audio_events_before_finish",
            irregular.get("audio_events_before_finish"),
            1.0,
        )
        at_most(
            "irregular_max_append_call_ms",
            irregular.get("max_append_call_ms"),
            float(expected.get("control_latency_limit_ms", 500.0)),
        )
        exact("irregular_output_samples", irregular.get("output_samples"), expected_samples)
        exact(
            "irregular_audio_hash_match",
            irregular.get("audio_fnv1a64"),
            baseline.get("audio_fnv1a64"),
        )
        exact(
            "irregular_bitwise_audio_equal",
            irregular.get("bitwise_audio_equal_to_one_shot"),
            True,
        )
        exact("irregular_text_equal", irregular.get("text_equal_to_one_shot"), True)
        exact("irregular_input_finished", irregular.get("input_finished_events"), 1)

        exact("barge_interrupted_audio", barge.get("interrupted_audio_before_yield"), True)
        exact(
            "barge_interrupted_partial_text",
            barge.get("interrupted_partial_text_before_yield"),
            True,
        )
        exact("barge_yield_event", barge.get("barge_in_yield_events"), 1)
        greater("barge_yield_epoch", barge.get("yielded_epoch"), barge.get("interrupted_epoch"))
        exact("barge_stale_payloads", barge.get("stale_agent_payloads_after_yield"), 0)
        greater("barge_recovered_epoch", barge.get("recovered_epoch"), barge.get("yielded_epoch"))
        exact("barge_recovery_audio", barge.get("recovery_audio_before_finish"), True)
        exact(
            "barge_recovery_partial_text",
            barge.get("recovery_partial_text_before_finish"),
            True,
        )
        exact("barge_input_finished", barge.get("input_finished_events"), 1)

        exact("cancel_event_count", cancel.get("cancel_events"), 1)
        exact("cancel_append_rejected", cancel.get("append_after_cancel_rejected"), True)
        exact("cancel_late_events", cancel.get("late_events"), 0)
        at_most(
            "cancel_append_call_ms",
            cancel.get("append_call_ms"),
            float(expected.get("control_latency_limit_ms", 500.0)),
        )
        at_most(
            "cancel_call_ms",
            cancel.get("cancel_call_ms"),
            float(expected.get("control_latency_limit_ms", 500.0)),
        )

        exact("reset_event_count", reset.get("reset_events"), 1)
        exact("reset_output_samples", reset.get("output_samples"), 3 * frame_samples)
        exact(
            "reset_audio_hash_match",
            reset.get("reset_audio_fnv1a64"),
            reset.get("fresh_audio_fnv1a64"),
        )
        exact("reset_bitwise_audio_equal", reset.get("bitwise_audio_equal"), True)
        exact("reset_text_equal", reset.get("text_equal"), True)
        exact("reset_input_finished", reset.get("reset_input_finished_events"), 1)
        exact("fresh_input_finished", reset.get("fresh_input_finished_events"), 1)

        exact("clear_implemented", clear.get("implemented"), True)
        exact("clear_succeeded", clear.get("clear_succeeded"), True)
        at_least("clear_processed_append_calls", clear.get("processed_append_calls"), 2.0)
        at_least("clear_processed_input_samples", clear.get("processed_input_samples"), 2560.0)
        at_least(
            "clear_processed_transcript_deltas",
            clear.get("transcript_delta_events_before_clear"),
            1.0,
        )
        at_most(
            "clear_call_ms",
            clear.get("clear_call_ms"),
            float(expected.get("control_latency_limit_ms", 500.0)),
        )
        exact("clear_completion_event", clear.get("clear_completion_events"), 1)
        at_least("clear_output_samples", clear.get("cleared_output_samples"), 1.0)
        at_least("clear_fresh_output_samples", clear.get("fresh_output_samples"), 1.0)
        exact(
            "clear_output_samples_fresh_equivalent",
            clear.get("cleared_output_samples"),
            clear.get("fresh_output_samples"),
        )
        at_least("clear_audio_rms", clear.get("cleared_audio_rms"), audio_rms_floor)
        at_least("clear_fresh_audio_rms", clear.get("fresh_audio_rms"), audio_rms_floor)
        at_least("clear_audio_peak", clear.get("cleared_audio_peak"), audio_peak_floor)
        at_least("clear_fresh_audio_peak", clear.get("fresh_audio_peak"), audio_peak_floor)
        exact("clear_audio_hash_present", bool(clear.get("cleared_audio_fnv1a64")), True)
        exact("clear_fresh_audio_hash_present", bool(clear.get("fresh_audio_fnv1a64")), True)
        exact(
            "clear_audio_hash_fresh_equivalent",
            clear.get("cleared_audio_fnv1a64"),
            clear.get("fresh_audio_fnv1a64"),
        )
        exact("clear_bitwise_audio_fresh_equivalent", clear.get("bitwise_audio_equal"), True)
        exact("clear_agent_text_fresh_equivalent", clear.get("agent_text_equal"), True)
        exact(
            "clear_user_transcript_fresh_equivalent",
            clear.get("user_transcript_equal"),
            True,
        )
        exact("clear_turn_finished", clear.get("cleared_turn_finished_events"), 1)
        exact("clear_fresh_turn_finished", clear.get("fresh_turn_finished_events"), 1)
        exact("clear_input_finished", clear.get("cleared_input_finished_events"), 1)
        exact("clear_fresh_input_finished", clear.get("fresh_input_finished_events"), 1)

        def response_recovery(prefix: str, evidence: dict[str, Any], *, truncate: bool) -> None:
            exact(f"{prefix}_implemented", evidence.get("implemented"), True)
            exact(
                f"{prefix}_commit_without_response",
                evidence.get("commit_without_response"),
                True,
            )
            at_least(f"{prefix}_interrupted_epoch", evidence.get("interrupted_epoch"), 1.0)
            greater(
                f"{prefix}_yielded_epoch",
                evidence.get("yielded_epoch"),
                evidence.get("interrupted_epoch"),
            )
            greater(
                f"{prefix}_replacement_epoch",
                evidence.get("replacement_epoch"),
                evidence.get("yielded_epoch"),
            )
            at_least(
                f"{prefix}_old_audio_before_control",
                evidence.get("old_audio_events_before_control"),
                2.0 if truncate else 1.0,
            )
            at_least(
                f"{prefix}_old_partial_text_before_control",
                evidence.get("old_partial_text_events_before_control"),
                1.0,
            )
            exact(f"{prefix}_yield_event", evidence.get("control_yield_events"), 1)
            at_most(
                f"{prefix}_control_call_ms",
                evidence.get("control_call_ms"),
                float(expected.get("control_latency_limit_ms", 500.0)),
            )
            at_least(
                f"{prefix}_observed_output_span",
                evidence.get("observed_output_span_samples"),
                1.0,
            )
            exact(
                f"{prefix}_generated_output_span",
                evidence.get("generated_output_samples"),
                evidence.get("observed_output_span_samples"),
            )
            if truncate:
                at_least(
                    f"{prefix}_played_output_samples",
                    evidence.get("played_output_samples"),
                    1.0,
                )
                greater(
                    f"{prefix}_played_before_generated",
                    evidence.get("generated_output_samples"),
                    evidence.get("played_output_samples"),
                )
                played = evidence.get("played_output_samples")
                generated = evidence.get("generated_output_samples")
                retained = evidence.get("retained_output_samples")
                discarded = evidence.get("discarded_output_samples")
                valid_samples = all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in (played, generated, retained, discarded)
                )
                exact(
                    f"{prefix}_played_frame_alignment",
                    played % frame_samples if valid_samples and frame_samples > 0 else -1,
                    0,
                )
                exact(
                    f"{prefix}_retained_played_boundary",
                    evidence.get("retained_output_samples"),
                    evidence.get("played_output_samples"),
                )
                exact(
                    f"{prefix}_discarded_sample_accounting",
                    discarded if valid_samples else None,
                    generated - retained if valid_samples else 0,
                )
                at_least(
                    f"{prefix}_discarded_output_samples",
                    discarded,
                    float(expected.get("response_truncate_min_discarded_samples", 0) or 0),
                )
            else:
                exact(f"{prefix}_played_output_samples", evidence.get("played_output_samples"), 0)
                exact(
                    f"{prefix}_retained_generated_boundary",
                    evidence.get("retained_output_samples"),
                    evidence.get("generated_output_samples"),
                )
                exact(
                    f"{prefix}_discarded_output_samples",
                    evidence.get("discarded_output_samples"),
                    0,
                )
            exact(
                f"{prefix}_stale_agent_payloads",
                evidence.get("stale_agent_payloads_after_control"),
                0,
            )
            at_least(
                f"{prefix}_replacement_audio_events",
                evidence.get("replacement_audio_events"),
                1.0,
            )
            at_least(
                f"{prefix}_replacement_audio_samples",
                evidence.get("replacement_audio_samples"),
                1.0,
            )
            at_least(
                f"{prefix}_replacement_audio_rms",
                evidence.get("replacement_audio_rms"),
                audio_rms_floor,
            )
            at_least(
                f"{prefix}_replacement_audio_peak",
                evidence.get("replacement_audio_peak"),
                audio_peak_floor,
            )
            exact(
                f"{prefix}_replacement_final_text_events",
                evidence.get("replacement_final_text_events"),
                1,
            )
            exact(
                f"{prefix}_replacement_final_text_nonempty",
                isinstance(evidence.get("replacement_final_text"), str)
                and bool(evidence["replacement_final_text"].strip()),
                True,
            )
            exact(
                f"{prefix}_replacement_turn_finished",
                evidence.get("replacement_turn_finished_events"),
                1,
            )
            exact(f"{prefix}_input_finished", evidence.get("input_finished_events"), 1)

        response_recovery("response_cancel", response_cancel, truncate=False)
        response_recovery("response_truncate", response_truncate, truncate=True)

        minimum_tail_events = 3
        maximum_tail_events = 4
        exact("tail_partial_samples", tail.get("partial_input_samples"), 317)
        at_least(
            "tail_pre_finish_committed_audio",
            tail.get("pre_finish_committed_audio_events"),
            1.0,
        )
        exact("tail_configured_frames", tail.get("configured_tail_frames"), 3)
        exact(
            "tail_minimum_audio_events",
            tail.get("minimum_audio_events_after_finish"),
            minimum_tail_events,
        )
        exact(
            "tail_maximum_audio_events",
            tail.get("maximum_audio_events_after_finish"),
            maximum_tail_events,
        )
        at_least(
            "tail_audio_events_min", tail.get("audio_events_after_finish"), minimum_tail_events
        )
        at_most("tail_audio_events_max", tail.get("audio_events_after_finish"), maximum_tail_events)
        exact(
            "tail_output_samples",
            tail.get("output_samples_after_finish"),
            int(tail.get("audio_events_after_finish", 0) or 0) * frame_samples,
        )
        at_most(
            "tail_completion_ms",
            tail.get("completion_ms"),
            float(expected.get("tail_completion_limit_ms", 15000.0)),
        )
        exact("tail_input_finished", tail.get("input_finished_events"), 1)

        exact(
            "sequence_sessions_checked",
            sequence.get("sessions_checked"),
            expected.get("expected_sequence_sessions_checked"),
        )
        at_least("sequence_events_checked", sequence.get("events_checked"), 1.0)
        exact("sequence_violations", sequence.get("violations"), 0)
        exact("sequence_monotonic", sequence.get("pass"), True)
        at_least("media_audio_events_checked", media.get("audio_events_checked"), 1.0)
        exact(
            "media_segments_checked",
            media.get("segments_checked"),
            expected.get("expected_media_segments_checked"),
        )
        exact("media_violations", media.get("violations"), 0)
        exact("media_contiguous", media.get("pass"), True)

        exact("multiturn_implemented", multiturn.get("implemented"), True)
        exact("multiturn_same_session", multiturn.get("same_session"), True)
        exact("multiturn_started", multiturn.get("turn_started_events"), 3)
        exact("multiturn_finished", multiturn.get("turn_finished_events"), 3)
        exact("multiturn_distinct_epochs", multiturn.get("distinct_turn_epochs"), 3)
        exact("multiturn_every_turn_completed", multiturn.get("every_turn_completed"), True)
        exact("multiturn_final_agent_text", multiturn.get("final_agent_text_events"), 3)
        exact(
            "multiturn_final_user_transcript",
            multiturn.get("final_user_transcript_events"),
            2,
        )
        exact("multiturn_input_finished", multiturn.get("input_finished_events"), 1)
        exact("multiturn_no_yield", multiturn.get("yield_events"), 0)
        exact("multiturn_no_reset", multiturn.get("reset_events"), 0)

        exact("function_channel_implemented", function.get("implemented"), True)
        at_least("function_sotc", function.get("sotc_events"), 1.0)
        at_least("function_eotc", function.get("eotc_events"), 1.0)
        at_least("function_eotr", function.get("eotr_events"), 1.0)
        at_least("function_completed_calls", function.get("completed_calls"), 1.0)
        at_least(
            "function_tool_response_injections",
            function.get("tool_response_injections"),
            1.0,
        )
        at_least(
            "function_agent_resumed_audio",
            function.get("agent_resumed_audio_events"),
            1.0,
        )
        at_least(
            "function_agent_resumed_text",
            function.get("agent_resumed_text_events"),
            1.0,
        )
        exact("function_expected_tool", function.get("expected_tool_name_match"), True)
        exact("function_response_submitted", function.get("tool_response_submitted"), True)
        exact("function_stale_response_rejected", function.get("stale_response_rejected"), True)
        exact("function_stale_payloads", function.get("stale_function_payloads"), 0)

        exact("concurrency_implemented", concurrency.get("implemented"), True)
        exact(
            "concurrency_producer_completed",
            concurrency.get("producer_thread_completed"),
            True,
        )
        exact(
            "concurrency_consumer_completed",
            concurrency.get("consumer_thread_completed"),
            True,
        )
        exact(
            "concurrency_events_while_producing",
            concurrency.get("events_observed_while_producing"),
            True,
        )
        exact("concurrency_bounded_queue", concurrency.get("bounded_queue"), True)
        exact("concurrency_overflow_observed", concurrency.get("overflow_error_observed"), True)
        exact("concurrency_no_deadlock", concurrency.get("no_deadlock"), True)
        at_least("concurrency_producer_appends", concurrency.get("producer_append_calls"), 2.0)
        exact("concurrency_finish_calls", concurrency.get("finish_input_calls"), 1)
        greater(
            "concurrency_overflow_attempt",
            concurrency.get("overflow_attempt_samples"),
            concurrency.get("live_capacity_samples"),
        )
        at_most(
            "concurrency_max_append_call_ms",
            concurrency.get("max_append_call_ms"),
            float(expected.get("control_latency_limit_ms", 500.0)),
        )
        at_most(
            "concurrency_overflow_call_ms",
            concurrency.get("overflow_call_ms"),
            float(expected.get("control_latency_limit_ms", 500.0)),
        )
        exact("concurrency_input_finished", concurrency.get("input_finished_events"), 1)

        # Keep the producer's summary as a consistency gate, but derive every
        # behavior gate above from primitive receipt fields.
        exact("probe_reported_pass", receipt.get("pass"), True)
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all native lifecycle primitive gates must pass",
            message=f"VoiceChat native lifecycle contract: "
            f"{sum(metric.passed for metric in metrics.values())}/{len(metrics)} gates passed",
        )


comparator = VoiceChatModelCardComparator()
