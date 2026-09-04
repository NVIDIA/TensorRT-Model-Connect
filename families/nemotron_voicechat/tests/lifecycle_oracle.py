# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Primitive receipt checks for the native VoiceChat lifecycle probe."""

from __future__ import annotations

REQUIRED_SECTIONS = (
    "baseline",
    "irregular_chunking",
    "barge_in",
    "cancel",
    "reset_vs_fresh",
    "processed_input_clear",
    "response_cancel_recovery",
    "response_truncate_recovery",
    "partial_finish_tail",
    "sequence_continuity",
    "media_continuity",
    "normal_multiturn",
    "function_channel",
    "backpressure_concurrency",
)

_FRAME_SAMPLES = 1764
_OUTPUT_SAMPLES = 345744
_CODEC_FRAMES = 196
_CONTROL_LIMIT_MS = 500.0


def _section(receipt: dict, name: str) -> dict:
    value = receipt.get(name)
    assert isinstance(value, dict), f"lifecycle receipt missing section {name}"
    return value


def _at_least(value, minimum: float, name: str) -> None:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), name
    assert float(value) >= minimum, name


def _at_most(value, maximum: float, name: str) -> None:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), name
    assert float(value) <= maximum, name


def _greater(value, lower, name: str) -> None:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), name
    assert isinstance(lower, (int, float)) and not isinstance(lower, bool), name
    assert float(value) > float(lower), name


def _response_recovery(section: dict, *, truncate: bool) -> None:
    assert section.get("implemented") is True
    assert section.get("commit_without_response") is True
    _at_least(section.get("interrupted_epoch"), 1, "interrupted epoch")
    _greater(section.get("yielded_epoch"), section.get("interrupted_epoch"), "yielded epoch")
    _greater(section.get("replacement_epoch"), section.get("yielded_epoch"), "replacement epoch")
    _at_least(section.get("old_audio_events_before_control"), 2 if truncate else 1, "old audio")
    _at_least(section.get("old_partial_text_events_before_control"), 1, "old partial text")
    assert section.get("control_yield_events") == 1
    _at_most(section.get("control_call_ms"), _CONTROL_LIMIT_MS, "response control latency")
    _at_least(section.get("observed_output_span_samples"), 1, "observed output span")
    assert section.get("generated_output_samples") == section.get("observed_output_span_samples")
    generated = section.get("generated_output_samples")
    retained = section.get("retained_output_samples")
    discarded = section.get("discarded_output_samples")
    assert all(type(value) is int for value in (generated, retained, discarded))
    if truncate:
        played = section.get("played_output_samples")
        assert type(played) is int and played > 0
        assert generated > played and played % _FRAME_SAMPLES == 0
        assert retained == played
        assert discarded == generated - retained
        assert discarded >= _FRAME_SAMPLES
    else:
        assert section.get("played_output_samples") == 0
        assert retained == generated
        assert discarded == 0
    assert section.get("stale_agent_payloads_after_control") == 0
    _at_least(section.get("replacement_audio_events"), 1, "replacement audio events")
    _at_least(section.get("replacement_audio_samples"), 1, "replacement audio samples")
    _at_least(section.get("replacement_audio_rms"), 0.001, "replacement audio RMS")
    _at_least(section.get("replacement_audio_peak"), 0.01, "replacement audio peak")
    assert section.get("replacement_final_text_events") == 1
    assert isinstance(section.get("replacement_final_text"), str)
    assert section["replacement_final_text"].strip()
    assert section.get("replacement_turn_finished_events") == 1
    assert section.get("input_finished_events") == 1


def assert_lifecycle_receipt(receipt: dict, expected_text: str) -> None:
    assert isinstance(receipt, dict)
    assert receipt.get("schema_version") == 3
    assert receipt.get("runtime") == "C++ ISpeechSession with TensorRT backend"
    sections = {name: _section(receipt, name) for name in REQUIRED_SECTIONS}

    baseline = sections["baseline"]
    assert baseline.get("output_samples") == _OUTPUT_SAMPLES
    assert baseline.get("audio_events") == _CODEC_FRAMES
    assert baseline.get("agent_text") == expected_text
    assert baseline.get("input_finished_events") == 1

    irregular = sections["irregular_chunking"]
    _at_least(irregular.get("append_calls"), 2, "irregular append calls")
    _at_least(irregular.get("audio_events_before_finish"), 1, "pre-finish audio")
    _at_most(irregular.get("max_append_call_ms"), _CONTROL_LIMIT_MS, "append latency")
    assert irregular.get("output_samples") == _OUTPUT_SAMPLES
    assert irregular.get("bitwise_audio_equal_to_one_shot") is True
    assert irregular.get("text_equal_to_one_shot") is True
    assert irregular.get("input_finished_events") == 1

    barge = sections["barge_in"]
    assert barge.get("interrupted_audio_before_yield") is True
    assert barge.get("interrupted_partial_text_before_yield") is True
    assert barge.get("barge_in_yield_events") == 1
    _greater(barge.get("yielded_epoch"), barge.get("interrupted_epoch"), "barge yield epoch")
    assert barge.get("stale_agent_payloads_after_yield") == 0
    _greater(barge.get("recovered_epoch"), barge.get("yielded_epoch"), "barge recovery epoch")
    assert barge.get("recovery_audio_before_finish") is True
    assert barge.get("recovery_partial_text_before_finish") is True
    assert barge.get("input_finished_events") == 1

    cancel = sections["cancel"]
    assert cancel.get("cancel_events") == 1
    assert cancel.get("append_after_cancel_rejected") is True
    assert cancel.get("late_events") == 0
    _at_most(cancel.get("append_call_ms"), _CONTROL_LIMIT_MS, "cancel append latency")
    _at_most(cancel.get("cancel_call_ms"), _CONTROL_LIMIT_MS, "cancel latency")

    reset = sections["reset_vs_fresh"]
    assert reset.get("reset_events") == 1
    assert reset.get("output_samples") == 3 * _FRAME_SAMPLES
    assert reset.get("bitwise_audio_equal") is True
    assert reset.get("text_equal") is True
    assert reset.get("reset_input_finished_events") == 1
    assert reset.get("fresh_input_finished_events") == 1

    clear = sections["processed_input_clear"]
    assert clear.get("implemented") is True and clear.get("clear_succeeded") is True
    _at_least(clear.get("processed_append_calls"), 2, "clear append calls")
    _at_least(clear.get("processed_input_samples"), 2560, "processed samples")
    _at_least(clear.get("transcript_delta_events_before_clear"), 1, "transcript delta")
    _at_most(clear.get("clear_call_ms"), _CONTROL_LIMIT_MS, "clear latency")
    assert clear.get("clear_completion_events") == 1
    assert clear.get("cleared_output_samples") == clear.get("fresh_output_samples")
    _at_least(clear.get("cleared_output_samples"), 1, "cleared output")
    _at_least(clear.get("cleared_audio_rms"), 0.001, "cleared audio RMS")
    _at_least(clear.get("fresh_audio_rms"), 0.001, "fresh audio RMS")
    _at_least(clear.get("cleared_audio_peak"), 0.01, "cleared audio peak")
    _at_least(clear.get("fresh_audio_peak"), 0.01, "fresh audio peak")
    assert clear.get("bitwise_audio_equal") is True
    assert clear.get("agent_text_equal") is True
    assert clear.get("user_transcript_equal") is True
    assert clear.get("cleared_turn_finished_events") == 1
    assert clear.get("fresh_turn_finished_events") == 1
    assert clear.get("cleared_input_finished_events") == 1
    assert clear.get("fresh_input_finished_events") == 1

    _response_recovery(sections["response_cancel_recovery"], truncate=False)
    _response_recovery(sections["response_truncate_recovery"], truncate=True)

    tail = sections["partial_finish_tail"]
    assert tail.get("partial_input_samples") == 317
    _at_least(tail.get("pre_finish_committed_audio_events"), 1, "pre-finish commit")
    assert tail.get("configured_tail_frames") == 3
    assert tail.get("minimum_audio_events_after_finish") == 3
    assert tail.get("maximum_audio_events_after_finish") == 4
    assert 3 <= int(tail.get("audio_events_after_finish", 0)) <= 4
    assert (
        tail.get("output_samples_after_finish")
        == tail.get("audio_events_after_finish") * _FRAME_SAMPLES
    )
    _at_most(tail.get("completion_ms"), 15000, "tail completion")
    assert tail.get("input_finished_events") == 1

    sequence = sections["sequence_continuity"]
    assert sequence.get("sessions_checked") == 13
    _at_least(sequence.get("events_checked"), 1, "sequence events")
    assert sequence.get("violations") == 0 and sequence.get("pass") is True
    media = sections["media_continuity"]
    assert media.get("segments_checked") == 16
    _at_least(media.get("audio_events_checked"), 1, "media audio events")
    assert media.get("violations") == 0 and media.get("pass") is True

    multiturn = sections["normal_multiturn"]
    assert multiturn.get("implemented") is True and multiturn.get("same_session") is True
    assert multiturn.get("turn_started_events") == 3
    assert multiturn.get("turn_finished_events") == 3
    assert multiturn.get("distinct_turn_epochs") == 3
    assert multiturn.get("every_turn_completed") is True
    assert multiturn.get("final_agent_text_events") == 3
    assert multiturn.get("final_user_transcript_events") == 2
    assert multiturn.get("input_finished_events") == 1
    assert multiturn.get("yield_events") == 0 and multiturn.get("reset_events") == 0

    function = sections["function_channel"]
    assert function.get("implemented") is True
    for name in (
        "sotc_events",
        "eotc_events",
        "eotr_events",
        "completed_calls",
        "tool_response_injections",
        "agent_resumed_audio_events",
        "agent_resumed_text_events",
    ):
        _at_least(function.get(name), 1, name)
    assert function.get("expected_tool_name_match") is True
    assert function.get("tool_response_submitted") is True
    assert function.get("stale_response_rejected") is True
    assert function.get("stale_function_payloads") == 0

    concurrency = sections["backpressure_concurrency"]
    for name in (
        "implemented",
        "producer_thread_completed",
        "consumer_thread_completed",
        "events_observed_while_producing",
        "bounded_queue",
        "overflow_error_observed",
        "no_deadlock",
    ):
        assert concurrency.get(name) is True, name
    _at_least(concurrency.get("producer_append_calls"), 2, "producer append calls")
    assert concurrency.get("finish_input_calls") == 1
    _greater(
        concurrency.get("overflow_attempt_samples"),
        concurrency.get("live_capacity_samples"),
        "overflow attempt",
    )
    _at_most(concurrency.get("max_append_call_ms"), _CONTROL_LIMIT_MS, "producer latency")
    _at_most(concurrency.get("overflow_call_ms"), _CONTROL_LIMIT_MS, "overflow latency")
    assert concurrency.get("input_finished_events") == 1
    assert receipt.get("pass") is True
