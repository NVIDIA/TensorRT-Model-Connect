# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict model-card contract for native Nemotron VoiceChat output."""

from __future__ import annotations

import json
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
        if stage.name == "realtime_websocket_interop":
            return self._compare_realtime(trt, ref, stage)
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

        required_sections = expected.get("required_sections", [])
        if not isinstance(required_sections, list):
            required_sections = []
        sections = {name: section(str(name)) for name in required_sections}
        baseline = sections.get("baseline", {})
        irregular = sections.get("irregular_chunking", {})
        barge = sections.get("barge_in", {})
        cancel = sections.get("cancel", {})
        reset = sections.get("reset_vs_fresh", {})
        tail = sections.get("partial_finish_tail", {})
        sequence = sections.get("sequence_continuity", {})
        media = sections.get("media_continuity", {})
        multiturn = sections.get("normal_multiturn", {})
        function = sections.get("function_channel", {})
        concurrency = sections.get("backpressure_concurrency", {})

        frame_samples = int(expected.get("expected_output_samples_per_frame", 0) or 0)
        codec_frames = int(expected.get("expected_output_codec_frames", 0) or 0)
        expected_samples = int(expected.get("expected_output_num_samples", 0) or 0)
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

        at_least("sequence_sessions_checked", sequence.get("sessions_checked"), 6.0)
        at_least("sequence_events_checked", sequence.get("events_checked"), 1.0)
        exact("sequence_violations", sequence.get("violations"), 0)
        exact("sequence_monotonic", sequence.get("pass"), True)
        at_least("media_audio_events_checked", media.get("audio_events_checked"), 1.0)
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

    def _compare_realtime(
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

        raw_scenarios = receipt.get("scenarios")
        scenario_values = raw_scenarios if isinstance(raw_scenarios, list) else []

        def scenario(name: str) -> dict[str, Any]:
            matches = [
                value
                for value in scenario_values
                if isinstance(value, dict) and value.get("name") == name
            ]
            return matches[0] if len(matches) == 1 else {}

        def trace(value: dict[str, Any]) -> list[dict[str, Any]]:
            events = value.get("timeline")
            if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
                return []
            return events

        def indices(events: list[dict[str, Any]], event_type: str, direction: str) -> list[int]:
            return [
                index
                for index, event in enumerate(events)
                if event.get("type") == event_type and event.get("direction") == direction
            ]

        def ordered(events: list[dict[str, Any]], sequence: list[tuple[str, str]]) -> bool:
            cursor = -1
            for direction, event_type in sequence:
                match = next(
                    (
                        index
                        for index, event in enumerate(events[cursor + 1 :], cursor + 1)
                        if event.get("direction") == direction and event.get("type") == event_type
                    ),
                    -1,
                )
                if match < 0:
                    return False
                cursor = match
            return True

        def valid_sha(value: object) -> bool:
            return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

        def non_silent_audio(event: dict[str, Any], response_id: object) -> bool:
            rms = event.get("audio_rms")
            peak = event.get("audio_peak")
            return (
                isinstance(response_id, str)
                and bool(response_id)
                and event.get("direction") == "server"
                and event.get("type") == "response.output_audio.delta"
                and event.get("response_id") == response_id
                and isinstance(event.get("audio_samples"), int)
                and not isinstance(event.get("audio_samples"), bool)
                and event["audio_samples"] > 0
                and valid_sha(event.get("audio_sha256"))
                and isinstance(rms, (int, float))
                and not isinstance(rms, bool)
                and 0.001 <= rms <= 1.0
                and isinstance(peak, (int, float))
                and not isinstance(peak, bool)
                and 0.01 <= peak <= 1.0
            )

        def non_empty_transcript_done(event: dict[str, Any], response_id: object) -> bool:
            text = event.get("text")
            return (
                isinstance(response_id, str)
                and bool(response_id)
                and event.get("direction") == "server"
                and event.get("type") == "response.output_audio_transcript.done"
                and event.get("response_id") == response_id
                and isinstance(text, str)
                and bool(text.strip())
            )

        def recovery_evidence(
            events: list[dict[str, Any]],
            creates: list[int],
            created: list[int],
            interrupted_done: list[int],
            interrupted_response_id: object,
        ) -> dict[str, object]:
            created_index = created[1] if len(created) == 2 else -1
            response_id = events[created_index].get("response_id") if created_index >= 0 else None
            done = [
                index
                for index, event in enumerate(events)
                if event.get("direction") == "server"
                and event.get("type") == "response.done"
                and event.get("response_id") == response_id
                and event.get("response_status") == "completed"
            ]
            done_index = done[0] if len(done) == 1 else -1
            media_types = {
                "response.output_audio.delta",
                "response.output_audio.done",
                "response.output_audio_transcript.delta",
                "response.output_audio_transcript.done",
            }
            media = [
                (index, event)
                for index, event in enumerate(events)
                if event.get("direction") == "server"
                and event.get("type") in media_types
                and event.get("response_id") == response_id
            ]
            item_ids = [event.get("item_id") for _, event in media]
            audio = [
                index
                for index, event in media
                if event.get("type") == "response.output_audio.delta"
            ]
            non_silent = [index for index, event in media if non_silent_audio(event, response_id)]
            audio_done = [
                index for index, event in media if event.get("type") == "response.output_audio.done"
            ]
            transcript_done = [
                index for index, event in media if non_empty_transcript_done(event, response_id)
            ]
            return {
                "identity": isinstance(response_id, str)
                and bool(response_id)
                and response_id != interrupted_response_id
                and len(creates) == 2
                and len(created) == 2
                and len(interrupted_done) == 1
                and interrupted_done[0] < creates[1] < created_index,
                "item_binding": bool(media)
                and all(isinstance(item_id, str) and item_id for item_id in item_ids)
                and len(set(item_ids)) == 1
                and done_index >= 0
                and all(created_index < index < done_index for index, _ in media),
                "audio_completion": bool(non_silent)
                and len(audio_done) == 1
                and done_index >= 0
                and created_index < min(non_silent)
                and all(index < audio_done[0] for index in audio)
                and audio_done[0] < done_index,
                "transcript_done": len(transcript_done) == 1
                and done_index >= 0
                and created_index < transcript_done[0] < done_index,
                "late_media": (
                    sum(index > done_index for index, _ in media) if done_index >= 0 else -1
                ),
            }

        exact("receipt_is_object", isinstance(receipt_value, dict), True)
        exact(
            "receipt_schema_version", receipt.get("schema_version"), expected.get("schema_version")
        )
        exact(
            "runtime_identity",
            receipt.get("runtime"),
            "Python /v1/realtime host with native JSONL worker",
        )
        exact("wire_sample_rate", receipt.get("wire_sample_rate"), expected.get("wire_sample_rate"))
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
        exact("probe_returncode", actual.get("probe_returncode"), 0)
        exact("failure_code_empty", receipt.get("failure_code"), "")

        required_names = expected.get("required_scenarios", [])
        names = required_names if isinstance(required_names, list) else []
        scenarios = {str(name): scenario(str(name)) for name in names}
        traces = {name: trace(value) for name, value in scenarios.items()}
        exact(
            "scenario_names",
            [value.get("name") for value in scenario_values if isinstance(value, dict)],
            names,
        )
        all_events = [event for events in traces.values() for event in events]
        exact(
            "timeline_shape",
            all(events for events in traces.values())
            and all(
                event.get("ordinal") == index
                and event.get("direction") in {"client", "server"}
                and isinstance(event.get("type"), str)
                and bool(event.get("type"))
                for events in traces.values()
                for index, event in enumerate(events)
            ),
            True,
        )
        exact(
            "server_error_events",
            sum(
                event.get("direction") == "server" and event.get("type") == "error"
                for event in all_events
            ),
            0,
        )
        client_ids = [
            event.get("event_id") for event in all_events if event.get("direction") == "client"
        ]
        exact(
            "client_event_ids",
            all(isinstance(value, str) and value for value in client_ids)
            and len(set(client_ids)) == len(client_ids),
            True,
        )

        required_client = expected.get("required_client_events", [])
        required_client = required_client if isinstance(required_client, list) else []
        client_types = {
            event.get("type") for event in all_events if event.get("direction") == "client"
        }
        exact("required_client_events", all(name in client_types for name in required_client), True)
        input_chunks = [
            event
            for event in all_events
            if event.get("direction") == "client"
            and event.get("type") == "input_audio_buffer.append"
        ]
        input_samples = sum(int(event.get("audio_samples", 0) or 0) for event in input_chunks)
        exact(
            "streamed_input_samples",
            input_samples,
            expected.get("expected_input_wire_samples"),
        )
        at_least(
            "streamed_input_chunks",
            len(input_chunks),
            3.0,
        )
        exact(
            "streamed_input_chunk_bounds",
            all(
                isinstance(event.get("audio_samples"), int)
                and 1 <= event["audio_samples"] <= 2400
                and valid_sha(event.get("audio_sha256"))
                for event in input_chunks
            ),
            True,
        )

        scenario_audio_samples = 0
        expected_scenario_inputs = expected.get("expected_input_wire_samples_by_scenario")
        expected_scenario_inputs = (
            expected_scenario_inputs if isinstance(expected_scenario_inputs, dict) else {}
        )
        for name, value in scenarios.items():
            events = traces[name]
            exact(
                f"{name}_session_and_close",
                value.get("clean_close") is True
                and value.get("close_code") == 1000
                and ordered(
                    events,
                    [
                        ("server", "session.created"),
                        ("client", "session.update"),
                        ("server", "session.updated"),
                        ("client", "session.close"),
                    ],
                )
                and len(indices(events, "session.close", "client")) == 1
                and not any(
                    event.get("direction") == "client"
                    for event in events[indices(events, "session.close", "client")[0] + 1 :]
                ),
                True,
            )
            created = indices(events, "response.created", "server")
            exact(
                f"{name}_response_shape",
                bool(created)
                and all(
                    events[index].get("response_object") == "realtime.response"
                    and events[index].get("response_status") == "in_progress"
                    and events[index].get("response_status_details_is_null") is True
                    and events[index].get("response_output_count") == 0
                    for index in created
                ),
                True,
            )
            scenario_input_samples = sum(
                int(event.get("audio_samples", 0) or 0)
                for event in events
                if event.get("direction") == "client"
                and event.get("type") == "input_audio_buffer.append"
            )
            exact(
                f"{name}_streamed_input_samples",
                scenario_input_samples,
                expected_scenario_inputs.get(name),
            )
            audio = value.get("audio") if isinstance(value.get("audio"), dict) else {}
            output_chunks = [
                event
                for event in events
                if event.get("direction") == "server"
                and event.get("type") == "response.output_audio.delta"
            ]
            output_samples = sum(int(event.get("audio_samples", 0) or 0) for event in output_chunks)
            scenario_audio_samples += output_samples
            exact(
                f"{name}_audio_evidence",
                audio.get("encoding") == "pcm_s16le"
                and audio.get("sample_rate") == expected.get("wire_sample_rate")
                and audio.get("num_samples") == output_samples
                and output_samples > 0
                and isinstance(audio.get("rms"), (int, float))
                and not isinstance(audio.get("rms"), bool)
                and audio["rms"] >= 0.001
                and isinstance(audio.get("peak"), (int, float))
                and not isinstance(audio.get("peak"), bool)
                and audio["peak"] >= 0.01
                and valid_sha(audio.get("sha256"))
                and all(
                    isinstance(event.get("audio_samples"), int)
                    and event["audio_samples"] > 0
                    and valid_sha(event.get("audio_sha256"))
                    for event in output_chunks
                ),
                True,
            )

        function = traces.get("function", [])
        function_creates = indices(function, "response.create", "client")
        call_indices = indices(function, "response.function_call_arguments.done", "server")
        output_indices = indices(function, "conversation.item.create", "client")
        added_indices = indices(function, "conversation.item.added", "server")
        done_indices = indices(function, "conversation.item.done", "server")
        call = function[call_indices[0]] if len(call_indices) == 1 else {}
        output = function[output_indices[0]] if len(output_indices) == 1 else {}
        added = function[added_indices[0]] if len(added_indices) == 1 else {}
        done = function[done_indices[0]] if len(done_indices) == 1 else {}
        exact(
            "function_event_order",
            len(function_creates) == 1
            and ordered(
                function,
                [
                    ("client", "input_audio_buffer.append"),
                    ("client", "input_audio_buffer.clear"),
                    ("server", "input_audio_buffer.cleared"),
                    ("client", "input_audio_buffer.append"),
                    ("server", "response.created"),
                    ("server", "response.function_call_arguments.done"),
                    ("server", "response.done"),
                    ("client", "conversation.item.create"),
                    ("server", "conversation.item.added"),
                    ("server", "conversation.item.done"),
                    ("client", "response.create"),
                    ("server", "response.created"),
                    ("server", "response.output_audio.delta"),
                    ("server", "response.output_audio_transcript.done"),
                    ("server", "response.output_audio.done"),
                    ("server", "response.done"),
                    ("client", "session.close"),
                ],
            ),
            True,
        )
        clear_requests = indices(function, "input_audio_buffer.clear", "client")
        clear_acks = indices(function, "input_audio_buffer.cleared", "server")
        clear_request = clear_requests[0] if len(clear_requests) == 1 else -1
        clear_ack = clear_acks[0] if len(clear_acks) == 1 else -1
        completed_after_clear = [
            str(event.get("text", "")).lower()
            for event in function[clear_ack + 1 :]
            if event.get("direction") == "server"
            and event.get("type") == "conversation.item.input_audio_transcription.completed"
        ]
        exact(
            "function_processed_input_clear",
            clear_request > 0
            and clear_ack > clear_request
            and any(
                event.get("direction") == "server"
                and event.get("type") == "conversation.item.input_audio_transcription.delta"
                for event in function[:clear_request]
            )
            and len(completed_after_clear) == 1
            and "random number" in completed_after_clear[0]
            and completed_after_clear[0].count("random") == 1,
            True,
        )
        try:
            arguments = json.loads(str(call.get("arguments", "")))
        except json.JSONDecodeError:
            arguments = None
        exact(
            "function_call_and_output",
            call.get("name") == expected.get("expected_tool_name")
            and isinstance(call.get("call_id"), str)
            and bool(call.get("call_id"))
            and isinstance(arguments, dict)
            and set(arguments) == {"min", "max"}
            and all(
                isinstance(arguments[name], int) and not isinstance(arguments[name], bool)
                for name in ("min", "max")
            )
            and arguments["min"] <= arguments["max"]
            and output.get("call_id") == call.get("call_id")
            and output.get("item_type") == "function_call_output"
            and output.get("output_sha256") == expected.get("tool_result_sha256")
            and all(
                item.get("call_id") == call.get("call_id")
                and item.get("item_type") == "function_call_output"
                and item.get("item_object") == "realtime.item"
                and item.get("item_status") == "completed"
                and isinstance(item.get("item_id"), str)
                and bool(item.get("item_id"))
                and item.get("output_sha256") == expected.get("tool_result_sha256")
                for item in (added, done)
            )
            and added.get("item_id") == done.get("item_id")
            and added.get("previous_item_id") == done.get("previous_item_id"),
            True,
        )
        continuation = function_creates[0] if len(function_creates) == 1 else len(function)
        call_response_id = call.get("response_id")
        call_done = [
            event
            for event in function[:continuation]
            if event.get("type") == "response.done"
            and event.get("response_status") == "completed"
            and event.get("response_id") == call_response_id
        ]
        resumed_created = [
            event
            for event in function[continuation + 1 :]
            if event.get("type") == "response.created"
        ]
        resumed_response_id = (
            resumed_created[0].get("response_id") if len(resumed_created) == 1 else None
        )
        continued = function[continuation + 1 :]
        exact(
            "function_response_identity",
            isinstance(call_response_id, str)
            and bool(call_response_id)
            and len(call_done) == 1
            and any(
                event.get("type") == "response.created"
                and event.get("response_id") == call_response_id
                for event in function[:continuation]
            )
            and isinstance(resumed_response_id, str)
            and bool(resumed_response_id)
            and resumed_response_id != call_response_id,
            True,
        )
        exact(
            "function_resumed_audio_text",
            any(
                event.get("type") == "response.output_audio.delta"
                and event.get("response_id") == resumed_response_id
                for event in continued
            )
            and any(
                event.get("type")
                in {
                    "response.output_audio_transcript.delta",
                    "response.output_audio_transcript.done",
                }
                and isinstance(event.get("text"), str)
                and event.get("text")
                and event.get("response_id") == resumed_response_id
                for event in continued
            )
            and any(
                event.get("type") == "response.done"
                and event.get("response_status") == "completed"
                and event.get("response_id") == resumed_response_id
                for event in continued
            ),
            True,
        )
        function_text = " ".join(str(event.get("text", "")) for event in continued).lower()
        exact("function_result_spoken", "20" in function_text or "twenty" in function_text, True)

        manual_commits = [
            traces[name][matches[0]]
            for name in ("truncate", "cancel")
            if len(
                matches := indices(traces.get(name, []), "input_audio_buffer.committed", "server")
            )
            == 1
        ]
        exact(
            "manual_commit_ack_shapes",
            len(manual_commits) == 2
            and all(
                isinstance(event.get("item_id"), str)
                and bool(event.get("item_id"))
                and event.get("previous_item_id") is None
                for event in manual_commits
            ),
            True,
        )

        truncate = traces.get("truncate", [])
        truncate_creates = indices(truncate, "response.create", "client")
        truncate_created = indices(truncate, "response.created", "server")
        truncate_audio = indices(truncate, "response.output_audio.delta", "server")
        truncate_request = indices(truncate, "conversation.item.truncate", "client")
        truncate_ack = indices(truncate, "conversation.item.truncated", "server")
        truncate_done = [
            index
            for index, event in enumerate(truncate)
            if event.get("direction") == "server"
            and event.get("type") == "response.done"
            and event.get("response_status") == "cancelled"
        ]
        exact(
            "truncate_event_order",
            ordered(
                truncate,
                [
                    ("client", "input_audio_buffer.commit"),
                    ("server", "input_audio_buffer.committed"),
                    ("client", "response.create"),
                    ("server", "response.created"),
                    ("server", "response.output_audio.delta"),
                    ("client", "conversation.item.truncate"),
                    ("server", "conversation.item.truncated"),
                    ("server", "response.output_audio.done"),
                    ("server", "response.done"),
                    ("client", "response.create"),
                    ("server", "response.created"),
                    ("server", "response.output_audio.delta"),
                    ("server", "response.output_audio.done"),
                    ("server", "response.done"),
                    ("client", "session.close"),
                ],
            )
            and len(truncate_creates) == 2
            and len(truncate_created) == 2
            and all(len(values) == 1 for values in (truncate_request, truncate_ack, truncate_done)),
            True,
        )
        truncated_response = (
            truncate[truncate_created[0]].get("response_id") if len(truncate_created) == 2 else None
        )
        truncated_audio = [
            index
            for index in truncate_audio
            if truncate[index].get("response_id") == truncated_response
        ]
        truncated_item = truncate[truncated_audio[0]].get("item_id") if truncated_audio else None
        truncate_public = truncate[truncate_ack[0]] if truncate_ack else {}
        exact(
            "truncate_target_ack",
            bool(truncated_response)
            and bool(truncated_item)
            and truncate[truncate_request[0]].get("item_id") == truncated_item
            and truncate[truncate_request[0]].get("content_index") == 0
            and truncate[truncate_request[0]].get("audio_end_ms") == 80
            and truncate_public.get("item_id") == truncated_item
            and truncate_public.get("content_index") == 0
            and truncate_public.get("audio_end_ms") == 80
            and truncate[truncate_done[0]].get("response_id") == truncated_response
            if truncate_request and truncate_ack and truncate_done
            else False,
            True,
        )
        truncate_stale_audio = (
            sum(
                event.get("direction") == "server"
                and event.get("type") == "response.output_audio.delta"
                and event.get("response_id") == truncated_response
                for event in truncate[truncate_ack[0] + 1 :]
            )
            if truncate_request and truncate_ack and truncate_done
            else -1
        )
        exact(
            "truncate_stale_audio_after_ack",
            truncate_stale_audio,
            0,
        )
        truncate_recovery = recovery_evidence(
            truncate,
            truncate_creates,
            truncate_created,
            truncate_done,
            truncated_response,
        )
        exact(
            "truncate_recovery_identity",
            truncate_recovery["identity"],
            True,
        )
        exact(
            "truncate_recovery_item_binding",
            truncate_recovery["item_binding"],
            True,
        )
        exact(
            "truncate_recovery_audio_completion",
            truncate_recovery["audio_completion"],
            True,
        )
        exact(
            "truncate_recovery_transcript_done",
            truncate_recovery["transcript_done"],
            True,
        )
        exact(
            "truncate_recovery_late_media_after_done",
            truncate_recovery["late_media"],
            0,
        )

        cancel = traces.get("cancel", [])
        cancel_creates = indices(cancel, "response.create", "client")
        cancel_created = indices(cancel, "response.created", "server")
        cancel_request = indices(cancel, "response.cancel", "client")
        cancel_done = [
            index
            for index, event in enumerate(cancel)
            if event.get("direction") == "server"
            and event.get("type") == "response.done"
            and event.get("response_status") == "cancelled"
            and event.get("response_reason") == "client_cancelled"
        ]
        exact(
            "cancel_event_order",
            ordered(
                cancel,
                [
                    ("client", "input_audio_buffer.commit"),
                    ("server", "input_audio_buffer.committed"),
                    ("client", "response.create"),
                    ("server", "response.created"),
                    ("server", "response.output_audio.delta"),
                    ("client", "response.cancel"),
                    ("server", "response.output_audio.done"),
                    ("server", "response.done"),
                    ("client", "response.create"),
                    ("server", "response.created"),
                    ("server", "response.output_audio.delta"),
                    ("server", "response.output_audio.done"),
                    ("server", "response.done"),
                    ("client", "session.close"),
                ],
            )
            and len(cancel_creates) == 2
            and len(cancel_created) == 2
            and all(len(values) == 1 for values in (cancel_request, cancel_done)),
            True,
        )
        cancelled_response = (
            cancel[cancel_created[0]].get("response_id") if len(cancel_created) == 2 else None
        )
        exact(
            "cancel_target_done",
            bool(cancelled_response)
            and cancel[cancel_request[0]].get("response_id") == cancelled_response
            and cancel[cancel_done[0]].get("response_id") == cancelled_response
            if cancel_request and cancel_done
            else False,
            True,
        )
        cancel_stale_audio = (
            sum(
                event.get("direction") == "server"
                and event.get("type") == "response.output_audio.delta"
                and event.get("response_id") == cancelled_response
                for event in cancel[cancel_done[0] + 1 :]
            )
            if cancel_request and cancel_done
            else -1
        )
        exact(
            "cancel_stale_audio_after_done",
            cancel_stale_audio,
            0,
        )
        cancel_recovery = recovery_evidence(
            cancel,
            cancel_creates,
            cancel_created,
            cancel_done,
            cancelled_response,
        )
        exact(
            "cancel_recovery_identity",
            cancel_recovery["identity"],
            True,
        )
        exact(
            "cancel_recovery_item_binding",
            cancel_recovery["item_binding"],
            True,
        )
        exact(
            "cancel_recovery_audio_completion",
            cancel_recovery["audio_completion"],
            True,
        )
        exact(
            "cancel_recovery_transcript_done",
            cancel_recovery["transcript_done"],
            True,
        )
        exact(
            "cancel_recovery_late_media_after_done",
            cancel_recovery["late_media"],
            0,
        )

        combined = receipt.get("combined_audio")
        combined_audio = combined if isinstance(combined, dict) else {}
        exact(
            "combined_audio_shape",
            combined_audio.get("encoding") == "pcm_s16le"
            and combined_audio.get("sample_rate") == expected.get("wire_sample_rate")
            and combined_audio.get("num_samples") == scenario_audio_samples
            and scenario_audio_samples > 0
            and valid_sha(combined_audio.get("sha256")),
            True,
        )
        at_least("combined_audio_rms", combined_audio.get("rms"), 0.001)
        at_least("combined_audio_peak", combined_audio.get("peak"), 0.01)

        exact("probe_reported_pass", receipt.get("pass"), True)
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all realtime WebSocket transport primitive gates must pass",
            message=f"VoiceChat realtime transport contract: "
            f"{sum(metric.passed for metric in metrics.values())}/{len(metrics)} gates passed",
        )


comparator = VoiceChatModelCardComparator()
