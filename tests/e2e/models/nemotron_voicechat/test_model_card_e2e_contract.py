# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the real native VoiceChat E2E contract."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
import struct
import subprocess
import sys
import tomllib
from array import array
from pathlib import Path
from types import SimpleNamespace

from tests.e2e.models.nemotron_voicechat.e2e_plugins.comparator import (
    VoiceChatModelCardComparator,
)
from tests.e2e.models.nemotron_voicechat.e2e_plugins.runner import (
    VoiceChatModelCardRunner,
)
from tests.e2e.models.nemotron_voicechat.e2e_plugins.reference import (
    VoiceChatLifecycleInvariantReference,
    VoiceChatPinnedModelCardReference,
)
from tests.e2e.models.nemotron_voicechat.realtime_transport_probe import (
    _audio_stats,
    _event_summary,
    _resample_linear,
)
from tests.e2e_harness.contracts import StageOutput, StageSpec, ThresholdProfile

_ROOT = Path(__file__).resolve().parents[4]
_MODEL_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _MODEL_DIR / "manifests/nemotron-voicechat-11b.json"
_INPUT_AUDIO = _MODEL_DIR / "assets/sample_general_input.flac"
_REFERENCE_AUDIO = _MODEL_DIR / "assets/sample_general_reference.flac"


def _import_report():
    scripts_dir = str(_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("generate_e2e_report")


def _write_wav(path: Path, samples: list[float], *, sample_rate: int, float32: bool) -> None:
    if float32:
        encoding = 3
        bits = 32
        data = struct.pack(f"<{len(samples)}f", *samples)
    else:
        encoding = 1
        bits = 16
        data = struct.pack(f"<{len(samples)}h", *(int(value * 32767) for value in samples))
    block_align = bits // 8
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH",
            16,
            encoding,
            1,
            sample_rate,
            sample_rate * block_align,
            block_align,
            bits,
        )
        + b"data"
        + struct.pack("<I", len(data))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data)


def _expected_reference() -> dict:
    return {
        "speech_source_sha256": "source-sha",
        "speech_source_sample_rate": 16000,
        "speech_source_num_samples": 249734,
        "expected_output_sample_rate": 22050,
        "expected_output_num_samples": 345744,
        "expected_output_samples_per_frame": 1764,
        "expected_output_codec_frames": 196,
        "expected_response_text": (
            "Hi there! How can you? How can I help you today? The sky is blue. "
            "That blue color is because of something called Rayleigh scattering."
        ),
        "required_response_terms": ["rayleigh", "scattering"],
    }


def _actual_output() -> dict:
    return {
        "source_sha256": "source-sha",
        "source_stats": {"channels": 1, "sample_rate": 16000, "num_samples": 249734},
        "output_stats": {
            "encoding": "ieee_float32le",
            "channels": 1,
            "sample_rate": 22050,
            "num_samples": 345744,
            "all_finite": True,
            "rms": 0.009,
            "peak": 0.25,
        },
        "generated_count": 345744,
        "tail_frames": 0,
        "agent_text_line_count": 1,
        "agent_text": (
            "Hi there! How can you? How can I help you today? The sky is blue. "
            "That blue color is because of something called Rayleigh scattering."
        ),
        "transcript_line_count": 1,
        "transcript": (
            "Hi there how can I help you today the sky is blue because light scatters "
            "through the atmosphere"
        ),
    }


def _lifecycle_reference() -> dict:
    return {
        "schema_version": 3,
        "speech_source_sha256": "source-sha",
        "function_speech_source_sha256": "function-source-sha",
        "expected_output_sample_rate": 22050,
        "expected_output_num_samples": 345744,
        "expected_output_samples_per_frame": 1764,
        "expected_output_codec_frames": 196,
        "expected_response_text": (
            "Hi there! How can you? How can I help you today? The sky is blue. "
            "That blue color is because of something called Rayleigh scattering."
        ),
        "control_latency_limit_ms": 500.0,
        "tail_completion_limit_ms": 15000.0,
        "required_sections": [
            "baseline",
            "irregular_chunking",
            "barge_in",
            "cancel",
            "reset_vs_fresh",
            "partial_finish_tail",
            "sequence_continuity",
            "media_continuity",
            "normal_multiturn",
            "function_channel",
            "backpressure_concurrency",
        ],
    }


def _lifecycle_receipt() -> dict:
    agent_text = _lifecycle_reference()["expected_response_text"]
    return {
        "schema_version": 3,
        "pass": True,
        "runtime": "C++ ISpeechSession with TensorRT backend",
        "baseline": {
            "output_samples": 345744,
            "audio_events": 196,
            "audio_fnv1a64": "baseline-hash",
            "agent_text": agent_text,
            "input_finished_events": 1,
        },
        "irregular_chunking": {
            "append_calls": 347,
            "max_append_call_ms": 0.02,
            "audio_events_before_finish": 1,
            "output_samples": 345744,
            "audio_fnv1a64": "baseline-hash",
            "bitwise_audio_equal_to_one_shot": True,
            "text_equal_to_one_shot": True,
            "input_finished_events": 1,
        },
        "barge_in": {
            "interrupted_epoch": 2,
            "interrupted_audio_before_yield": True,
            "interrupted_partial_text_before_yield": True,
            "yielded_epoch": 3,
            "barge_in_yield_events": 1,
            "stale_agent_payloads_after_yield": 0,
            "recovered_epoch": 4,
            "recovery_audio_before_finish": True,
            "recovery_partial_text_before_finish": True,
            "input_finished_events": 1,
        },
        "cancel": {
            "append_call_ms": 0.02,
            "cancel_call_ms": 0.01,
            "cancel_events": 1,
            "append_after_cancel_rejected": True,
            "late_events": 0,
        },
        "reset_vs_fresh": {
            "reset_events": 1,
            "output_samples": 5292,
            "reset_audio_fnv1a64": "reset-hash",
            "fresh_audio_fnv1a64": "reset-hash",
            "bitwise_audio_equal": True,
            "text_equal": True,
            "reset_input_finished_events": 1,
            "fresh_input_finished_events": 1,
        },
        "partial_finish_tail": {
            "partial_input_samples": 317,
            "pre_finish_committed_audio_events": 1,
            "configured_tail_frames": 3,
            "minimum_audio_events_after_finish": 3,
            "maximum_audio_events_after_finish": 4,
            "audio_events_after_finish": 3,
            "output_samples_after_finish": 5292,
            "completion_ms": 200.0,
            "input_finished_events": 1,
        },
        "sequence_continuity": {
            "sessions_checked": 6,
            "events_checked": 500,
            "violations": 0,
            "pass": True,
        },
        "media_continuity": {
            "audio_events_checked": 400,
            "violations": 0,
            "pass": True,
        },
        "normal_multiturn": {
            "implemented": True,
            "same_session": True,
            "turn_started_events": 3,
            "turn_finished_events": 3,
            "distinct_turn_epochs": 3,
            "every_turn_completed": True,
            "final_agent_text_events": 3,
            "final_user_transcript_events": 2,
            "yield_events": 0,
            "reset_events": 0,
            "input_finished_events": 1,
            "pass": True,
        },
        "function_channel": {
            "implemented": True,
            "sotc_events": 1,
            "eotc_events": 1,
            "eotr_events": 1,
            "completed_calls": 1,
            "tool_response_injections": 1,
            "agent_resumed_audio_events": 2,
            "agent_resumed_text_events": 1,
            "expected_tool_name_match": True,
            "tool_response_submitted": True,
            "stale_response_rejected": True,
            "stale_function_payloads": 0,
            "pass": True,
        },
        "backpressure_concurrency": {
            "implemented": True,
            "producer_thread_completed": True,
            "consumer_thread_completed": True,
            "events_observed_while_producing": True,
            "bounded_queue": True,
            "overflow_error_observed": True,
            "no_deadlock": True,
            "producer_append_calls": 196,
            "finish_input_calls": 1,
            "live_capacity_samples": 480000,
            "overflow_attempt_samples": 480001,
            "max_append_call_ms": 0.03,
            "overflow_call_ms": 0.04,
            "input_finished_events": 1,
            "pass": True,
        },
    }


def _realtime_reference() -> dict:
    return {
        "schema_version": 1,
        "speech_source_sha256": "source-sha",
        "function_speech_source_sha256": "function-source-sha",
        "wire_sample_rate": 24000,
        "expected_input_wire_samples": 429_417,
        "expected_input_wire_samples_by_scenario": {
            "function": 333_417,
            "truncate": 48_000,
            "cancel": 48_000,
        },
        "expected_tool_name": "generate_random_number",
        "tool_result_sha256": hashlib.sha256(b'{"result":20}').hexdigest(),
        "required_scenarios": ["function", "truncate", "cancel"],
        "required_client_events": [
            "session.update",
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "input_audio_buffer.clear",
            "response.create",
            "response.cancel",
            "conversation.item.truncate",
            "conversation.item.create",
            "session.close",
        ],
    }


def _trace(name: str, input_samples: int) -> tuple[list[dict], dict]:
    events: list[dict] = []

    def add(direction: str, event_type: str, **fields) -> int:
        events.append(
            {
                "ordinal": len(events),
                "direction": direction,
                "type": event_type,
                **fields,
            }
        )
        return len(events) - 1

    add("server", "session.created")
    add("client", "session.update", event_id=f"{name}_update")
    add("server", "session.updated")
    if name == "function":
        clear_remaining = 48_000
        clear_chunk = 0
        while clear_remaining:
            count = min(2400, clear_remaining)
            add(
                "client",
                "input_audio_buffer.append",
                event_id=f"function_clear_probe_audio_{clear_chunk}",
                audio_samples=count,
                audio_sha256=f"{clear_chunk + 100:064x}",
            )
            clear_remaining -= count
            clear_chunk += 1
        add(
            "server",
            "conversation.item.input_audio_transcription.delta",
            item_id="item_cleared_input",
            text="temporary speech",
        )
        add("client", "input_audio_buffer.clear", event_id="function_clear")
        add("server", "input_audio_buffer.cleared")
    remaining = input_samples - (48_000 if name == "function" else 0)
    chunk = 0
    while remaining:
        count = min(2400, remaining)
        add(
            "client",
            "input_audio_buffer.append",
            event_id=f"{name}_audio_{chunk}",
            audio_samples=count,
            audio_sha256=f"{chunk + 1:064x}",
        )
        remaining -= count
        chunk += 1
    if name == "function":
        add(
            "server",
            "conversation.item.input_audio_transcription.completed",
            item_id="item_function_input",
            text="Okay, can you give me a random number between one and fifty",
        )
    if name != "function":
        add("client", "input_audio_buffer.commit", event_id=f"{name}_commit")
        add(
            "server",
            "input_audio_buffer.committed",
            previous_item_id=None,
            item_id=f"user_{name}",
        )
        add("client", "response.create", event_id=f"{name}_create")
    response_id = f"resp_{name}"
    item_id = f"item_{name}"
    add(
        "server",
        "response.created",
        response_id=response_id,
        response_object="realtime.response",
        response_status="in_progress",
        response_status_details_is_null=True,
        response_output_count=0,
    )
    add(
        "server",
        "response.output_audio.delta",
        response_id=response_id,
        item_id=item_id,
        audio_samples=2,
        audio_sha256="a" * 64,
        audio_rms=0.1,
        audio_peak=0.25,
    )

    if name == "function":
        add(
            "server",
            "response.function_call_arguments.done",
            response_id=response_id,
            call_id="call_1",
            name="generate_random_number",
            arguments='{"min":1,"max":100}',
        )
        add("server", "response.output_audio.done", response_id=response_id, item_id=item_id)
        add("server", "response.done", response_id=response_id, response_status="completed")
        add(
            "client",
            "conversation.item.create",
            event_id="function_output",
            call_id="call_1",
            item_type="function_call_output",
            output_sha256=hashlib.sha256(b'{"result":20}').hexdigest(),
        )
        for event_type in ("conversation.item.added", "conversation.item.done"):
            add(
                "server",
                event_type,
                call_id="call_1",
                previous_item_id=None,
                item_id="item_function_output",
                item_type="function_call_output",
                item_object="realtime.item",
                item_status="completed",
                output_sha256=hashlib.sha256(b'{"result":20}').hexdigest(),
            )
        add("client", "response.create", event_id="function_continue")
        response_id = "resp_function_continuation"
        item_id = "item_function_continuation"
        add(
            "server",
            "response.created",
            response_id=response_id,
            response_object="realtime.response",
            response_status="in_progress",
            response_status_details_is_null=True,
            response_output_count=0,
        )
        add(
            "server",
            "response.output_audio.delta",
            response_id=response_id,
            item_id=item_id,
            audio_samples=2,
            audio_sha256="b" * 64,
            audio_rms=0.1,
            audio_peak=0.25,
        )
        add(
            "server",
            "response.output_audio_transcript.done",
            response_id=response_id,
            item_id=item_id,
            text="The result is 20.",
        )
        add("server", "response.output_audio.done", response_id=response_id, item_id=item_id)
        add("server", "response.done", response_id=response_id, response_status="completed")
    elif name == "truncate":
        add(
            "client",
            "conversation.item.truncate",
            event_id="truncate_control",
            item_id=item_id,
            content_index=0,
            audio_end_ms=80,
        )
        add(
            "server",
            "conversation.item.truncated",
            item_id=item_id,
            content_index=0,
            audio_end_ms=80,
        )
        add("server", "response.output_audio.done", response_id=response_id, item_id=item_id)
        add(
            "server",
            "response.done",
            response_id=response_id,
            response_status="cancelled",
            response_reason="client_cancelled",
        )
        add("client", "response.create", event_id="truncate_recovery_create")
        response_id = "resp_truncate_recovery"
        item_id = "item_truncate_recovery"
        add(
            "server",
            "response.created",
            response_id=response_id,
            response_object="realtime.response",
            response_status="in_progress",
            response_status_details_is_null=True,
            response_output_count=0,
        )
        add(
            "server",
            "response.output_audio.delta",
            response_id=response_id,
            item_id=item_id,
            audio_samples=2,
            audio_sha256="c" * 64,
            audio_rms=0.1,
            audio_peak=0.25,
        )
        add(
            "server",
            "response.output_audio_transcript.done",
            response_id=response_id,
            item_id=item_id,
            text="Recovered after truncation.",
        )
        add("server", "response.output_audio.done", response_id=response_id, item_id=item_id)
        add("server", "response.done", response_id=response_id, response_status="completed")
    else:
        add(
            "client",
            "response.cancel",
            event_id="cancel_control",
            response_id=response_id,
        )
        add("server", "response.output_audio.done", response_id=response_id, item_id=item_id)
        add(
            "server",
            "response.done",
            response_id=response_id,
            response_status="cancelled",
            response_reason="client_cancelled",
        )
        add("client", "response.create", event_id="cancel_recovery_create")
        response_id = "resp_cancel_recovery"
        item_id = "item_cancel_recovery"
        add(
            "server",
            "response.created",
            response_id=response_id,
            response_object="realtime.response",
            response_status="in_progress",
            response_status_details_is_null=True,
            response_output_count=0,
        )
        add(
            "server",
            "response.output_audio.delta",
            response_id=response_id,
            item_id=item_id,
            audio_samples=2,
            audio_sha256="d" * 64,
            audio_rms=0.1,
            audio_peak=0.25,
        )
        add(
            "server",
            "response.output_audio_transcript.done",
            response_id=response_id,
            item_id=item_id,
            text="Recovered after cancellation.",
        )
        add("server", "response.output_audio.done", response_id=response_id, item_id=item_id)
        add("server", "response.done", response_id=response_id, response_status="completed")
    add("client", "session.close", event_id=f"{name}_close")
    audio_samples = sum(
        event.get("audio_samples", 0)
        for event in events
        if event["direction"] == "server" and event["type"] == "response.output_audio.delta"
    )
    return events, {
        "encoding": "pcm_s16le",
        "sample_rate": 24000,
        "num_samples": audio_samples,
        "rms": 0.1,
        "peak": 0.25,
        "sha256": {"function": "a", "truncate": "b", "cancel": "c"}[name] * 64,
    }


def _realtime_receipt() -> dict:
    scenarios = []
    for name, samples in (("function", 333_417), ("truncate", 48_000), ("cancel", 48_000)):
        timeline, audio = _trace(name, samples)
        scenarios.append(
            {
                "name": name,
                "clean_close": True,
                "close_code": 1000,
                "timeline": timeline,
                "audio": audio,
            }
        )
    combined_samples = sum(scenario["audio"]["num_samples"] for scenario in scenarios)
    return {
        "schema_version": 1,
        "pass": True,
        "runtime": "Python /v1/realtime host with native JSONL worker",
        "wire_sample_rate": 24000,
        "failure_code": "",
        "failure_message": "",
        "scenarios": scenarios,
        "combined_audio": {
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
            "num_samples": combined_samples,
            "rms": 0.1,
            "peak": 0.25,
            "sha256": "d" * 64,
        },
    }


def _compare(actual: dict):
    comparator = VoiceChatModelCardComparator()
    return comparator.compare(
        StageOutput(stage_name="model_card_general_conversation", data=actual),
        StageOutput(stage_name="model_card_general_conversation", data=_expected_reference()),
        ThresholdProfile(
            task_strategy="speech_to_speech",
            metrics={
                "audio_min_rms": 0.001,
                "audio_min_peak": 0.01,
                "agent_text_min_similarity": 0.75,
                "transcript_min_words": 8,
                "transcript_min_similarity": 0.35,
            },
        ),
        StageSpec(name="model_card_general_conversation"),
    )


def _compare_lifecycle(receipt: dict):
    comparator = VoiceChatModelCardComparator()
    return comparator.compare(
        StageOutput(
            stage_name="native_full_duplex_lifecycle",
            data={
                "receipt": receipt,
                "source_sha256": "source-sha",
                "function_source_sha256": "function-source-sha",
                "runtime_path_confirmed": True,
            },
        ),
        StageOutput(stage_name="native_full_duplex_lifecycle", data=_lifecycle_reference()),
        ThresholdProfile(task_strategy="speech_to_speech"),
        StageSpec(name="native_full_duplex_lifecycle"),
    )


def _compare_realtime(receipt: dict):
    comparator = VoiceChatModelCardComparator()
    return comparator.compare(
        StageOutput(
            stage_name="realtime_websocket_interop",
            data={
                "receipt": receipt,
                "source_sha256": "source-sha",
                "function_source_sha256": "function-source-sha",
                "probe_returncode": 0,
            },
        ),
        StageOutput(stage_name="realtime_websocket_interop", data=_realtime_reference()),
        ThresholdProfile(task_strategy="speech_to_speech"),
        StageSpec(name="realtime_websocket_interop"),
    )


def test_manifest_pins_public_model_card_identity_and_exact_receipt() -> None:
    owner = tomllib.loads((_MODEL_DIR / "MODEL.toml").read_text(encoding="utf-8"))
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    case = manifest["testcases"][0]

    assert owner["model_reference_cache"] == {
        "repository": "https://github.com/NVIDIA%2DNeMo/Speech.git",
        "revision": "097dfe9e2f55baf653b83035868bdc89849f1b47",
        "relative_path": "nemotron_voicechat/reference/Speech-097dfe9e2f55",
        "entrypoint": "examples/speechlm2/sample_audio/sample_general.wav",
        "environment_variable": "NEMOTRON_VOICECHAT_SPEECH_REPO",
    }
    assert manifest["hf_revision"] == "359ada7b1c60851e40ff08065f9b0340244f27e0"
    assert manifest["bundle"] == "nemotron-voicechat-11b.bundle"
    assert manifest["runtime_strategy"] == "nemotron_voicechat_full_duplex"
    assert manifest["task_strategy"] == "speech_to_speech"
    assert manifest["execution_profiles"] == {
        "build": "base",
        "runtime": "base",
        "reference": "base",
    }
    assert case["speech_source_sha256"] == (
        "481f422a961fb160ddeba9824d55cb7c190c57acb7dc1730a2d595fd078dcb04"
    )
    assert case["inputs"]["audio"] == (
        "tests/e2e/models/nemotron_voicechat/assets/sample_general_input.flac"
    )
    assert case["inputs"]["reference_audio"] == (
        "tests/e2e/models/nemotron_voicechat/assets/sample_general_reference.flac"
    )
    assert (
        hashlib.sha256(_INPUT_AUDIO.read_bytes()).hexdigest() == (case["report_input_audio_sha256"])
    )
    assert (
        hashlib.sha256(_REFERENCE_AUDIO.read_bytes()).hexdigest()
        == (case["reference_audio_sha256"])
    )
    assert case["text_model_revision"] == "6533e8de2c68e4536bf7c411d7a3ce5734111476"
    assert (case["expected_output_sample_rate"], case["expected_output_num_samples"]) == (
        22050,
        345744,
    )
    assert case["expected_output_num_samples"] == (
        case["expected_output_codec_frames"] * case["expected_output_samples_per_frame"]
    )
    assert "runtime_cli_requires_hf_python" not in case.get("metadata", {})

    lifecycle = next(
        testcase
        for testcase in manifest["testcases"]
        if testcase["name"] == "nemotron-voicechat-11b-full-duplex-lifecycle"
    )
    assert lifecycle["reference_backend"] == "voicechat_lifecycle_invariants"
    assert lifecycle["oracle_level"] == "L4_invariants"
    assert lifecycle["test_category"] == "regression"
    assert lifecycle["preflight_requirements"] == [
        {
            "kind": "python_module_available",
            "args": {"module": "websockets", "phase": "runtime"},
            "gating": True,
        }
    ]
    assert lifecycle["stages"] == [
        {
            "name": "native_full_duplex_lifecycle",
            "required": True,
            "artifact_type": "waveform",
            "comparison_mode": "invariant_check",
        },
        {
            "name": "realtime_websocket_interop",
            "required": True,
            "artifact_type": "waveform",
            "comparison_mode": "invariant_check",
        },
    ]
    assert lifecycle["inputs"]["function_speech_source_relative_path"].endswith("/sample_fc.wav")
    assert lifecycle["function_speech_source_sha256"] == (
        "265f9e5f58bff1e71f4354f7d83e2ff405a8405b29c2cfeb50eb9085042e9136"
    )
    assert lifecycle["function_speech_source_num_samples"] == 190278
    assert lifecycle["inputs"]["realtime_runtime_timeout_s"] == 1800


def test_runner_uses_native_speak_then_native_transcribe(monkeypatch, tmp_path: Path) -> None:
    speech = tmp_path / "Speech"
    source = speech / "examples/speechlm2/sample_audio/sample_general.wav"
    _write_wav(source, [0.25, -0.25, 0.125, -0.125], sample_rate=16000, float32=False)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setenv("NEMOTRON_VOICECHAT_SPEECH_REPO", str(speech))

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        if command[1] == "speak":
            output = Path(command[command.index("--audio-out") + 1])
            _write_wav(output, [0.2, -0.2, 0.1, -0.1], sample_rate=22050, float32=True)
            return subprocess.CompletedProcess(
                command,
                0,
                "Agent text: The blue sky is explained by Rayleigh scattering.\n"
                f"Generated 4 audio samples -> {output}\n",
                "",
            )
        assert command[1] == "transcribe"
        return subprocess.CompletedProcess(
            command,
            0,
            "The blue sky is explained by light scattering through the atmosphere.\n",
            "",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    case = SimpleNamespace(
        name="nemotron-voicechat-11b",
        bundle="nemotron-voicechat-11b.bundle",
        inputs={
            "speech_source_relative_path": ("examples/speechlm2/sample_audio/sample_general.wav"),
            "runtime_timeout_s": 10,
            "transcribe_timeout_s": 10,
            "tail_frames": 0,
            "max_new_tokens": 256,
        },
        metadata={"speech_source_sha256": source_sha},
    )
    context = SimpleNamespace(
        binary_path="/opt/trtmc/bin/trtmc",
        engine_dir=str(tmp_path / "engines"),
        model_plugin_dir=str(tmp_path / "plugins"),
        ld_library_path="/opt/tensorrt/lib",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    output = VoiceChatModelCardRunner().run_stage(
        case, StageSpec(name="model_card_general_conversation"), context
    )

    assert [command[1] for command in commands] == ["speak", "transcribe"]
    assert all("--hf-python" not in command for command in commands)
    assert commands[0][commands[0].index("--tail-frames") + 1] == "0"
    assert commands[0][commands[0].index("--seed") + 1] == "0"
    assert commands[1][commands[1].index("--audio") + 1] == output.data["wav_path"]
    assert output.data["generated_count"] == output.data["output_stats"]["num_samples"] == 4
    assert output.data["output_stats"]["encoding"] == "ieee_float32le"
    assert output.data["agent_text_line_count"] == 1
    assert "Rayleigh scattering" in output.data["agent_text"]
    assert output.data["transcript_line_count"] == 1
    assert "Rayleigh" not in output.text


def test_runner_builds_and_executes_native_lifecycle_target(monkeypatch, tmp_path: Path) -> None:
    speech = tmp_path / "Speech"
    source = speech / "examples/speechlm2/sample_audio/sample_general.wav"
    _write_wav(source, [0.25, -0.25, 0.125, -0.125], sample_rate=16000, float32=False)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    function_source = speech / "examples/speechlm2/sample_audio/sample_fc.wav"
    _write_wav(function_source, [0.1, -0.1], sample_rate=16000, float32=False)
    function_source_sha = hashlib.sha256(function_source.read_bytes()).hexdigest()
    monkeypatch.setenv("NEMOTRON_VOICECHAT_SPEECH_REPO", str(speech))

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("configured\n", encoding="utf-8")
    binary = build_dir / "trtmc"
    binary.touch()
    probe = build_dir / "test_nemotron_voicechat_native_lifecycle"
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    (engine_dir / "nemotron-voicechat-11b.bundle").touch()
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        if command[0] == "cmake":
            probe.touch()
            return subprocess.CompletedProcess(command, 0, "built\n", "")
        assert command[0] == str(probe)
        output_wav = Path(command[-2])
        receipt_path = Path(command[-1])
        _write_wav(output_wav, [0.2, -0.2], sample_rate=22050, float32=True)
        receipt_path.write_text(json.dumps(_lifecycle_receipt()), encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            0,
            f"receipt={receipt_path}\n",
            "[trtmc] Pipeline loaded (strategy=nemotron_voicechat_full_duplex, "
            "backend=trt_new_runtime)\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    case = SimpleNamespace(
        name="nemotron-voicechat-11b-full-duplex-lifecycle",
        bundle="nemotron-voicechat-11b.bundle",
        inputs={
            "speech_source_relative_path": "examples/speechlm2/sample_audio/sample_general.wav",
            "function_speech_source_relative_path": "examples/speechlm2/sample_audio/sample_fc.wav",
            "lifecycle_build_timeout_s": 10,
            "lifecycle_runtime_timeout_s": 10,
        },
        metadata={
            "speech_source_sha256": source_sha,
            "function_speech_source_sha256": function_source_sha,
        },
    )
    context = SimpleNamespace(
        binary_path=str(binary),
        engine_dir=str(engine_dir),
        model_plugin_dir=str(plugin_dir),
        ld_library_path="/opt/tensorrt/lib",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    output = VoiceChatModelCardRunner().run_stage(
        case, StageSpec(name="native_full_duplex_lifecycle"), context
    )

    assert commands[0] == [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "test_nemotron_voicechat_native_lifecycle",
    ]
    assert commands[1][0] == str(probe)
    assert output.data["receipt"]["schema_version"] == 3
    assert output.data["runtime_path_confirmed"] is True
    assert output.data["function_source_sha256"] == function_source_sha
    assert Path(output.data["receipt_path"]).is_file()
    assert Path(output.data["wav_path"]).is_file()


def test_runner_builds_and_executes_realtime_websocket_probe(monkeypatch, tmp_path: Path) -> None:
    speech = tmp_path / "Speech"
    source = speech / "examples/speechlm2/sample_audio/sample_general.wav"
    _write_wav(source, [0.25, -0.25], sample_rate=16000, float32=False)
    function_source = speech / "examples/speechlm2/sample_audio/sample_fc.wav"
    _write_wav(function_source, [0.1, -0.1], sample_rate=16000, float32=False)
    monkeypatch.setenv("NEMOTRON_VOICECHAT_SPEECH_REPO", str(speech))

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("configured\n", encoding="utf-8")
    binary = build_dir / "trtmc"
    binary.touch()
    worker = build_dir / "trtmc_realtime_worker"
    engine_dir = tmp_path / "engines"
    engine_dir.mkdir()
    (engine_dir / "nemotron-voicechat-11b.bundle").touch()
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        if command[0] == "cmake":
            worker.touch()
            return subprocess.CompletedProcess(command, 0, "built\n", "")
        output_wav = Path(command[command.index("--output-wav") + 1])
        receipt_path = Path(command[command.index("--receipt") + 1])
        _write_wav(output_wav, [0.2, -0.2], sample_rate=24000, float32=False)
        receipt_path.write_text(json.dumps(_realtime_receipt()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    case = SimpleNamespace(
        name="nemotron-voicechat-11b-full-duplex-lifecycle",
        bundle="nemotron-voicechat-11b.bundle",
        inputs={
            "speech_source_relative_path": "examples/speechlm2/sample_audio/sample_general.wav",
            "function_speech_source_relative_path": "examples/speechlm2/sample_audio/sample_fc.wav",
            "realtime_build_timeout_s": 10,
            "realtime_event_timeout_s": 10,
            "realtime_runtime_timeout_s": 10,
        },
        metadata={
            "speech_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "function_speech_source_sha256": hashlib.sha256(
                function_source.read_bytes()
            ).hexdigest(),
        },
    )
    context = SimpleNamespace(
        binary_path=str(binary),
        engine_dir=str(engine_dir),
        model_plugin_dir=str(plugin_dir),
        ld_library_path="/opt/tensorrt/lib",
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    output = VoiceChatModelCardRunner().run_stage(
        case, StageSpec(name="realtime_websocket_interop"), context
    )

    assert commands[0] == [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "trtmc_realtime_worker",
    ]
    assert commands[1][0] == sys.executable
    assert commands[1][commands[1].index("--worker") + 1] == str(worker)
    assert commands[1][commands[1].index("--bundle") + 1].endswith("nemotron-voicechat-11b.bundle")
    assert output.data["receipt"]["schema_version"] == 1
    assert output.data["source_sha256"] == case.metadata["speech_source_sha256"]
    assert output.data["function_source_sha256"] == case.metadata["function_speech_source_sha256"]
    assert Path(output.data["receipt_path"]).is_file()
    assert Path(output.data["wav_path"]).is_file()


def test_realtime_probe_records_bounded_pcm_evidence_without_base64() -> None:
    wire = _resample_linear(array("h", [0, 16384, -16384, 0]), 16000)
    assert len(wire) == 12
    encoded = base64.b64encode(wire).decode("ascii")
    summary = _event_summary(
        "server",
        {"type": "response.output_audio.delta", "delta": encoded},
        7,
    )
    stats = _audio_stats(wire)

    assert summary == {
        "ordinal": 7,
        "direction": "server",
        "type": "response.output_audio.delta",
        "audio_samples": 6,
        "audio_sha256": hashlib.sha256(wire).hexdigest(),
        "audio_rms": stats["rms"],
        "audio_peak": stats["peak"],
    }
    assert stats["encoding"] == "pcm_s16le"
    assert stats["sample_rate"] == 24000
    assert stats["num_samples"] == 6
    assert stats["rms"] > 0
    assert stats["peak"] > 0
    assert encoded not in json.dumps(summary)


def test_reference_persists_pinned_audio_for_the_standalone_report(tmp_path: Path) -> None:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    testcase = manifest["testcases"][0]
    case = SimpleNamespace(
        name=manifest["name"],
        inputs=testcase["inputs"],
        metadata=testcase,
        hf_id=manifest["hf_id"],
        hf_revision=manifest["hf_revision"],
    )
    context = SimpleNamespace(artifacts_dir=str(tmp_path))

    output = VoiceChatPinnedModelCardReference().run_stage(
        case, StageSpec(name="model_card_general_conversation"), context
    )

    persisted = Path(output.data["audio_output_path"])
    assert persisted.parent == tmp_path / manifest["name"]
    assert persisted.read_bytes() == _REFERENCE_AUDIO.read_bytes()


def test_lifecycle_reference_declares_every_fail_closed_section(tmp_path: Path) -> None:
    case = SimpleNamespace(
        name="nemotron-voicechat-11b-full-duplex-lifecycle",
        inputs={
            "reference_audio": (
                "tests/e2e/models/nemotron_voicechat/assets/sample_general_reference.flac"
            )
        },
        metadata={
            **_lifecycle_reference(),
            "reference_audio_sha256": hashlib.sha256(_REFERENCE_AUDIO.read_bytes()).hexdigest(),
        },
    )
    output = VoiceChatLifecycleInvariantReference().run_stage(
        case,
        StageSpec(name="native_full_duplex_lifecycle"),
        SimpleNamespace(artifacts_dir=str(tmp_path)),
    )

    expected = _lifecycle_reference()
    persisted = Path(output.data.pop("audio_output_path"))
    assert output.data == expected
    assert persisted.read_bytes() == _REFERENCE_AUDIO.read_bytes()
    assert output.metadata["source"] == "model_owned_l4_lifecycle_invariants"


def test_realtime_reference_declares_transport_primitives(tmp_path: Path) -> None:
    case = SimpleNamespace(
        name="nemotron-voicechat-11b-full-duplex-lifecycle",
        inputs={
            "reference_audio": (
                "tests/e2e/models/nemotron_voicechat/assets/sample_general_reference.flac"
            )
        },
        metadata={
            **_realtime_reference(),
            "speech_source_num_samples": 249734,
            "function_speech_source_num_samples": 190278,
            "reference_audio_sha256": hashlib.sha256(_REFERENCE_AUDIO.read_bytes()).hexdigest(),
        },
    )
    output = VoiceChatLifecycleInvariantReference().run_stage(
        case,
        StageSpec(name="realtime_websocket_interop"),
        SimpleNamespace(artifacts_dir=str(tmp_path)),
    )

    persisted = Path(output.data.pop("audio_output_path"))
    assert output.data == _realtime_reference()
    assert persisted.read_bytes() == _REFERENCE_AUDIO.read_bytes()
    assert output.metadata["source"] == "model_owned_l4_realtime_transport_invariants"


def test_standalone_report_embeds_input_trt_and_reference_audio(tmp_path: Path) -> None:
    generate_e2e_report = _import_report()
    trt_audio = tmp_path / "trt.wav"
    _write_wav(trt_audio, [0.2, -0.2, 0.1, -0.1], sample_rate=22050, float32=True)
    reference_audio = tmp_path / "reference.flac"
    reference_audio.write_bytes(_REFERENCE_AUDIO.read_bytes())
    result = {
        "status": "pass",
        "case_name": "nemotron-voicechat-11b",
        "oracle_level": "L3_snapshot_regression",
        "case_config": {
            "task_strategy": "speech_to_speech",
            "reference_backend": "voicechat_pinned_model_card",
            "inputs": {
                "audio": ("tests/e2e/models/nemotron_voicechat/assets/sample_general_input.flac")
            },
        },
        "artifacts": {"trt_wav": trt_audio.name},
        "stage_outputs": {
            "ref_model_card_general_conversation": {
                "data": {"audio_output_path": str(reference_audio)}
            }
        },
        "_artifact_dir": str(tmp_path),
    }

    assert generate_e2e_report.validate_evidence([result], project_dir=_ROOT) == []
    rendered = generate_e2e_report.render_audio_model(result, project_dir=_ROOT)
    assert rendered.count("<audio controls") == 3

    lifecycle = copy.deepcopy(result)
    lifecycle["case_name"] = "nemotron-voicechat-11b-full-duplex-lifecycle"
    lifecycle["oracle_level"] = "L4_invariants"
    lifecycle["case_config"]["reference_backend"] = "voicechat_lifecycle_invariants"
    lifecycle["stage_outputs"] = {
        "ref_native_full_duplex_lifecycle": {"data": {"audio_output_path": str(reference_audio)}}
    }
    assert generate_e2e_report.validate_evidence([lifecycle], project_dir=_ROOT) == []


def test_comparator_requires_every_audio_text_and_session_gate() -> None:
    result = _compare(_actual_output())
    assert result.status == "passed"
    assert all(metric.passed for metric in result.metrics.values())
    assert result.metrics["codec_frame_count"].value == 196
    assert result.metrics["session_frame_mapping"].passed

    off_by_one = copy.deepcopy(_actual_output())
    off_by_one["output_stats"]["num_samples"] -= 1
    off_by_one["generated_count"] -= 1
    failed_audio = _compare(off_by_one)
    assert failed_audio.status == "failed"
    assert not failed_audio.metrics["output_num_samples"].passed
    assert not failed_audio.metrics["codec_frame_alignment"].passed

    missing_semantics = copy.deepcopy(_actual_output())
    missing_semantics["agent_text"] = "Hello, I can help with that today."
    failed_text = _compare(missing_semantics)
    assert failed_text.status == "failed"
    assert not failed_text.metrics["agent_required_response_terms"].passed

    unintelligible_audio = copy.deepcopy(_actual_output())
    unintelligible_audio["transcript"] = "noise only"
    failed_transcript = _compare(unintelligible_audio)
    assert failed_transcript.status == "failed"
    assert failed_transcript.metrics["agent_required_response_terms"].passed
    assert not failed_transcript.metrics["transcript_word_count"].passed


def test_lifecycle_comparator_recomputes_primitives_and_fails_closed() -> None:
    passed = _compare_lifecycle(_lifecycle_receipt())
    assert passed.status == "passed"
    assert all(metric.passed for metric in passed.metrics.values())

    stale = copy.deepcopy(_lifecycle_receipt())
    stale["pass"] = True
    stale["barge_in"]["stale_agent_payloads_after_yield"] = 1
    stale_result = _compare_lifecycle(stale)
    assert stale_result.status == "failed"
    assert not stale_result.metrics["barge_stale_payloads"].passed

    unsupported = copy.deepcopy(_lifecycle_receipt())
    unsupported["pass"] = True
    unsupported["function_channel"]["implemented"] = False
    unsupported_result = _compare_lifecycle(unsupported)
    assert unsupported_result.status == "failed"
    assert not unsupported_result.metrics["function_channel_implemented"].passed

    missing = copy.deepcopy(_lifecycle_receipt())
    missing["pass"] = True
    missing.pop("backpressure_concurrency")
    missing_result = _compare_lifecycle(missing)
    assert missing_result.status == "failed"
    assert not missing_result.metrics["section_backpressure_concurrency_present"].passed


def test_realtime_comparator_recomputes_websocket_trace_and_fails_closed() -> None:
    passed = _compare_realtime(_realtime_receipt())
    assert passed.status == "passed"
    assert all(metric.passed for metric in passed.metrics.values())
    assert len(passed.metrics) == 55
    assert passed.message.endswith("55/55 gates passed")

    stale = copy.deepcopy(_realtime_receipt())
    truncate = next(item for item in stale["scenarios"] if item["name"] == "truncate")
    ack = next(
        index
        for index, event in enumerate(truncate["timeline"])
        if event["type"] == "conversation.item.truncated"
    )
    truncate["timeline"].insert(
        ack + 1,
        {
            "ordinal": ack + 1,
            "direction": "server",
            "type": "response.output_audio.delta",
            "response_id": "resp_truncate",
            "item_id": "item_truncate",
            "audio_samples": 2,
            "audio_sha256": "e" * 64,
            "audio_rms": 0.1,
            "audio_peak": 0.25,
        },
    )
    truncate["audio"]["num_samples"] += 2
    stale["combined_audio"]["num_samples"] += 2
    for ordinal, event in enumerate(truncate["timeline"]):
        event["ordinal"] = ordinal
    stale_result = _compare_realtime(stale)
    assert stale_result.status == "failed"
    assert not stale_result.metrics["truncate_stale_audio_after_ack"].passed

    missing_terminal = copy.deepcopy(_realtime_receipt())
    cancel = next(item for item in missing_terminal["scenarios"] if item["name"] == "cancel")
    cancel["timeline"] = [
        event
        for event in cancel["timeline"]
        if not (
            event["type"] == "response.done" and event.get("response_id") == "resp_cancel_recovery"
        )
    ]
    for ordinal, event in enumerate(cancel["timeline"]):
        event["ordinal"] = ordinal
    missing_terminal_result = _compare_realtime(missing_terminal)
    assert missing_terminal_result.status == "failed"
    assert not missing_terminal_result.metrics["cancel_recovery_audio_completion"].passed

    silent = copy.deepcopy(_realtime_receipt())
    silent_cancel = next(item for item in silent["scenarios"] if item["name"] == "cancel")
    recovery_audio = next(
        event
        for event in silent_cancel["timeline"]
        if event["type"] == "response.output_audio.delta"
        and event.get("response_id") == "resp_cancel_recovery"
    )
    recovery_audio["audio_rms"] = 0.0
    recovery_audio["audio_peak"] = 0.0
    silent_result = _compare_realtime(silent)
    assert silent_result.status == "failed"
    assert silent_result.metrics["cancel_audio_evidence"].passed
    assert not silent_result.metrics["cancel_recovery_audio_completion"].passed

    reused_identity = copy.deepcopy(_realtime_receipt())
    truncate = next(item for item in reused_identity["scenarios"] if item["name"] == "truncate")
    for event in truncate["timeline"]:
        if event.get("response_id") == "resp_truncate_recovery":
            event["response_id"] = "resp_truncate"
    reused_identity_result = _compare_realtime(reused_identity)
    assert reused_identity_result.status == "failed"
    assert not reused_identity_result.metrics["truncate_recovery_identity"].passed

    missing_transcript = copy.deepcopy(_realtime_receipt())
    truncate = next(item for item in missing_transcript["scenarios"] if item["name"] == "truncate")
    truncate["timeline"] = [
        event
        for event in truncate["timeline"]
        if not (
            event["type"].startswith("response.output_audio_transcript.")
            and event.get("response_id") == "resp_truncate_recovery"
        )
    ]
    for ordinal, event in enumerate(truncate["timeline"]):
        event["ordinal"] = ordinal
    missing_transcript_result = _compare_realtime(missing_transcript)
    assert missing_transcript_result.status == "failed"
    assert not missing_transcript_result.metrics["truncate_recovery_transcript_done"].passed

    empty_transcript = copy.deepcopy(_realtime_receipt())
    cancel = next(item for item in empty_transcript["scenarios"] if item["name"] == "cancel")
    transcript = next(
        event
        for event in cancel["timeline"]
        if event["type"].startswith("response.output_audio_transcript.")
        and event.get("response_id") == "resp_cancel_recovery"
    )
    transcript["text"] = "  "
    empty_transcript_result = _compare_realtime(empty_transcript)
    assert empty_transcript_result.status == "failed"
    assert not empty_transcript_result.metrics["cancel_recovery_transcript_done"].passed

    cross_item = copy.deepcopy(_realtime_receipt())
    truncate = next(item for item in cross_item["scenarios"] if item["name"] == "truncate")
    transcript = next(
        event
        for event in truncate["timeline"]
        if event["type"] == "response.output_audio_transcript.done"
        and event.get("response_id") == "resp_truncate_recovery"
    )
    transcript["item_id"] = "item_from_another_response"
    cross_item_result = _compare_realtime(cross_item)
    assert cross_item_result.status == "failed"
    assert not cross_item_result.metrics["truncate_recovery_item_binding"].passed

    delta_only = copy.deepcopy(_realtime_receipt())
    cancel = next(item for item in delta_only["scenarios"] if item["name"] == "cancel")
    transcript = next(
        event
        for event in cancel["timeline"]
        if event["type"] == "response.output_audio_transcript.done"
        and event.get("response_id") == "resp_cancel_recovery"
    )
    transcript["type"] = "response.output_audio_transcript.delta"
    delta_only_result = _compare_realtime(delta_only)
    assert delta_only_result.status == "failed"
    assert not delta_only_result.metrics["cancel_recovery_transcript_done"].passed

    late_media = copy.deepcopy(_realtime_receipt())
    truncate = next(item for item in late_media["scenarios"] if item["name"] == "truncate")
    recovery_done = next(
        index
        for index, event in enumerate(truncate["timeline"])
        if event["type"] == "response.done" and event.get("response_id") == "resp_truncate_recovery"
    )
    truncate["timeline"].insert(
        recovery_done + 1,
        {
            "ordinal": recovery_done + 1,
            "direction": "server",
            "type": "response.output_audio_transcript.delta",
            "response_id": "resp_truncate_recovery",
            "item_id": "item_truncate_recovery",
            "text": "late",
        },
    )
    for ordinal, event in enumerate(truncate["timeline"]):
        event["ordinal"] = ordinal
    late_media_result = _compare_realtime(late_media)
    assert late_media_result.status == "failed"
    assert not late_media_result.metrics["truncate_recovery_late_media_after_done"].passed

    cancel_stale = copy.deepcopy(_realtime_receipt())
    cancel = next(item for item in cancel_stale["scenarios"] if item["name"] == "cancel")
    cancelled_done = next(
        index
        for index, event in enumerate(cancel["timeline"])
        if event["type"] == "response.done" and event.get("response_id") == "resp_cancel"
    )
    cancel["timeline"].insert(
        cancelled_done + 1,
        {
            "ordinal": cancelled_done + 1,
            "direction": "server",
            "type": "response.output_audio.delta",
            "response_id": "resp_cancel",
            "item_id": "item_cancel",
            "audio_samples": 2,
            "audio_sha256": "f" * 64,
            "audio_rms": 0.1,
            "audio_peak": 0.25,
        },
    )
    cancel["audio"]["num_samples"] += 2
    cancel_stale["combined_audio"]["num_samples"] += 2
    for ordinal, event in enumerate(cancel["timeline"]):
        event["ordinal"] = ordinal
    cancel_stale_result = _compare_realtime(cancel_stale)
    assert cancel_stale_result.status == "failed"
    assert not cancel_stale_result.metrics["cancel_stale_audio_after_done"].passed


def test_e2e_files_stay_model_owned() -> None:
    tracked = {
        path.relative_to(_ROOT).as_posix()
        for path in _MODEL_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert "tests/e2e/models/nemotron_voicechat/runner.py" in tracked
    assert "tests/e2e/models/nemotron_voicechat/test_nemotron_voicechat_e2e.py" in tracked
