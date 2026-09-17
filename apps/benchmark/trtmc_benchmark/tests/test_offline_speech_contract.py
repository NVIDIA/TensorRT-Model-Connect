# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline speech representation checks leave model accuracy to family tests."""

from copy import deepcopy
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

import tools.perf_matrix as perf


def _entry(task="offline_speech_dialogue", *, primary="duplex_speech_dialogue", explicit=False):
    return perf.ResolvedEntry(
        spec={
            "id": "offline-speech-fixture",
            "operation": "speech_dialogue",
            "baseline": {"output_contract": "offline-speech-shape"} if explicit else {},
        },
        model=SimpleNamespace(task=primary),
        case=SimpleNamespace(
            selected_task=task,
            request={},
            measurement=SimpleNamespace(asset_loading_included=False),
        ),
        manifest={},
        reference_precision="fp32",
        baseline_timing={
            "timing_scope": "task-pipeline-call-wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
        },
    )


def _chunk(kind, payload):
    return kind + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")


def _wav(path: Path, values, *, rate=22050, channels=1, format_tag=3, extra=True):
    # Real IEEE_FLOAT WAV bytes, not a mocked file reader. Non-PCM fact and an
    # odd-sized auxiliary chunk exercise the layout emitted by WAV writers.
    samples = struct.pack(f"<{len(values)}f", *values)
    fmt = struct.pack("<HHIIHH", format_tag, channels, rate, rate * channels * 4, channels * 4, 32)
    chunks = _chunk(b"fmt ", fmt)
    if extra:
        chunks += _chunk(b"fact", struct.pack("<I", len(values) // channels))
        chunks += _chunk(b"JUNK", b"aux")
    chunks += _chunk(b"data", samples)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks)
    return path


def _event(
    kind="agent_audio",
    *,
    audio=(),
    text="",
    final=False,
    epoch=1,
    sequence=0,
    rate=22050,
    channels=1,
):
    return {
        "kind": kind,
        "epoch": epoch,
        "sequence": sequence,
        "frame_index": -1,
        "media_start_sample": -1,
        "media_end_sample": -1,
        "text": text,
        "is_final": final,
        "audio": list(audio),
        "audio_samples": len(audio),
        "sample_rate": rate if kind == "agent_audio" else None,
        "channels": channels,
    }


def _candidate(values=(0.25, -0.5, 0.75, -1.0), *, rate=22050, channels=1):
    return {
        "events": [
            _event("agent_text", text="complete native text", final=True),
            _event(audio=values, sequence=1, rate=rate, channels=channels),
            _event("input_finished", final=True, sequence=2),
        ],
        "read_states": [1, 2],
        "output_format": {"sample_rate": rate, "channels": channels},
        "output_audio_seconds": len(values) / channels / rate,
        "input_samples": 8,
        "input_frames": 8,
        "input_sample_rate": 16000,
        "input_channels": 1,
    }


def _reference(path: Path, values=(1.0, 0.5, -0.25, 0.0), *, rate=22050, channels=1):
    return {
        "text": "different complete reference text",
        "audio_samples": len(values),
        "num_samples": len(values),
        "sample_rate": rate,
        "channels": channels,
        "audio_seconds": len(values) / channels / rate,
        "input_samples": 8,
        "input_sample_rate": 16000,
        "input_channels": 1,
        "finite": True,
        "audio_artifact": str(_wav(path, values, rate=rate, channels=channels)),
    }


def _compare(left, right, entry=None):
    entry = _entry() if entry is None else entry
    return perf.compare(
        entry,
        {
            "output_summary": left,
            "timing_scope": "public_task_call_wall",
            "asset_loading_included": False,
            "metrics": {"latency_ms": {"p50": 1.0}},
        },
        {
            "output_summary": right,
            "measurement_policy": entry.baseline_timing,
            "precision": "fp32",
            "metrics": {"latency_ms": {"p50": 1.0}},
        },
    )


def test_real_float_wav_and_complete_events_reach_shape_comparison_without_quality_gate(tmp_path):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    original = deepcopy(left)
    artifact = Path(right["audio_artifact"]).read_bytes()
    assert perf._contract_name(_entry()) == "offline-speech-shape"
    assert "offline-speech-shape" in perf.OUTPUT_CONTRACTS
    status, result = _compare(left, right)
    assert status == "yellow"
    evidence = result["output_contract"]
    assert evidence["numerical_parity_checked"] is False
    assert evidence["text_parity_checked"] is False
    assert evidence["candidate"]["text"] == "complete native text"
    assert evidence["reference"]["text"] == "different complete reference text"
    assert evidence["candidate"]["audio_samples"] == evidence["reference"]["audio_samples"] == 4
    assert left == original and Path(right["audio_artifact"]).read_bytes() == artifact


def test_chunk_order_and_final_replacement_are_preserved_per_epoch(tmp_path):
    left = _candidate()
    left["events"] = [
        _event("agent_text", text="par", epoch=9),
        _event(audio=(0.25, -0.5), epoch=9, sequence=1),
        _event("agent_text", text="tial", epoch=9, sequence=2),
        _event("agent_text", text="old complete", final=True, epoch=9, sequence=3),
        _event("agent_text", text="second", epoch=3, sequence=4),
        _event(audio=(0.75, -1.0), epoch=3, sequence=5),
        _event("agent_text", text="complete", final=True, epoch=9, sequence=6),
        _event("agent_text", text="complete", final=True, epoch=9, sequence=7),
        _event("agent_text", text=" ignored after final", epoch=9, sequence=8),
        _event("agent_text", text=" turn", epoch=3, sequence=9),
    ]
    parsed = perf._offline_speech_candidate(left)
    assert parsed["text"] == "complete second turn"
    assert parsed["audio"] == [0.25, -0.5, 0.75, -1.0]
    assert _compare(left, _reference(tmp_path / "reference.wav"))[0] == "yellow"


def test_empty_final_replaces_partial_and_empty_pcm_is_a_valid_complete_result(tmp_path):
    left, right = _candidate(()), _reference(tmp_path / "empty.wav", ())
    left["events"] = [
        _event("agent_text", text="discarded partial"),
        _event("agent_text", text="", final=True, sequence=1),
        _event(audio=(), sequence=2),
    ]
    right["text"] = ""
    assert perf._offline_speech_candidate(left)["text"] == ""
    assert _compare(left, right)[0] == "yellow"
    # A complete empty event list plus explicit format/duration is zero output,
    # not a missing events field; the reference still needs an empty data chunk.
    left["events"] = []
    assert _compare(left, right)[0] == "yellow"


def test_float_wav_retains_stereo_order_and_does_not_downmix_or_resample(tmp_path):
    values = [0.25, -0.5, 0.75, -1.0]
    right = _reference(tmp_path / "stereo.wav", values, rate=8000, channels=2)
    parsed = perf._offline_speech_reference(right)
    assert parsed["audio"] == values and parsed["sample_rate"] == 8000 and parsed["channels"] == 2
    assert _compare(_candidate(values, rate=8000, channels=2), right)[0] == "yellow"


def test_oversized_output_rate_is_mismatch_instead_of_arithmetic_overflow(tmp_path):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    left["output_format"]["sample_rate"] = 10**1000
    left["events"][1]["sample_rate"] = 10**1000
    status, value = _compare(left, right)
    assert status == "contract-mismatch"
    assert value["reason"].startswith("candidate ")


@pytest.mark.parametrize(
    "field,value",
    (
        ("input_samples", 1 << 64),
        ("input_sample_rate", 1 << 32),
        ("input_channels", 1 << 32),
    ),
)
def test_input_bookkeeping_retains_public_integer_ranges(tmp_path, field, value):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    left[field] = value
    right[field] = value
    assert _compare(left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "task", ("duplex_speech_dialogue", "tool_speech_dialogue", "speech_to_speech_response")
)
def test_offline_contract_is_not_a_default_or_explicit_qualification_for_other_tasks(
    tmp_path, task
):
    with pytest.raises(perf.PerfMatrixError, match="unsupported output contract"):
        perf._contract_name(_entry(task))
    status, value = _compare(
        _candidate(), _reference(tmp_path / "reference.wav"), _entry(task, explicit=True)
    )
    assert status == "contract-mismatch"
    assert "cannot qualify" in value["reason"]


@pytest.mark.parametrize(
    "field",
    (
        "events",
        "output_format",
        "output_audio_seconds",
        "input_samples",
        "input_sample_rate",
        "input_channels",
    ),
)
def test_candidate_missing_complete_output_or_input_metadata_fails(tmp_path, field):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    del left[field]
    status, value = _compare(left, right)
    assert status == "contract-mismatch" and value["reason"].startswith("candidate ")


@pytest.mark.parametrize(
    "field",
    (
        "text",
        "audio_samples",
        "num_samples",
        "sample_rate",
        "channels",
        "audio_seconds",
        "finite",
        "audio_artifact",
        "input_samples",
        "input_sample_rate",
        "input_channels",
    ),
)
def test_reference_missing_complete_output_or_input_metadata_fails(tmp_path, field):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    del right[field]
    status, value = _compare(left, right)
    assert status == "contract-mismatch" and value["reason"].startswith("reference ")


@pytest.mark.parametrize(
    "field",
    (
        "kind",
        "epoch",
        "sequence",
        "text",
        "is_final",
        "audio",
        "audio_samples",
        "channels",
        "sample_rate",
    ),
)
def test_agent_audio_event_must_retain_its_complete_typed_payload(tmp_path, field):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    del left["events"][1][field]
    assert _compare(left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field,value",
    (
        ("kind", "new_unknown_event"),
        ("kind", "error"),
        ("kind", "cancelled"),
        ("kind", "function_call"),
        ("kind", []),
        ("epoch", True),
        ("epoch", -1),
        ("sequence", "1"),
        ("text", None),
        ("is_final", 1),
        ("audio_samples", None),
        ("audio_samples", True),
        ("audio_samples", 3),
        ("audio", [0.25, -0.5, 0.75]),
        ("audio", [True, -0.5, 0.75, -1]),
        ("audio", [float("nan"), -0.5, 0.75, -1]),
        ("audio", [float("inf"), -0.5, 0.75, -1]),
        ("audio", [1e100, -0.5, 0.75, -1]),
        ("audio", [10**1000, -0.5, 0.75, -1]),
        ("channels", True),
        ("channels", 2),
        ("sample_rate", None),
        ("sample_rate", 16000),
    ),
)
def test_malformed_unknown_or_failed_event_cannot_be_a_successful_shape(tmp_path, field, value):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    left["events"][1][field] = value
    status, result = _compare(left, right)
    assert status == "contract-mismatch" and result["reason"].startswith("candidate ")


@pytest.mark.parametrize(
    "field,value",
    (
        ("events", None),
        ("events", [None]),
        ("output_format", None),
        ("output_format", {"sample_rate": 22050, "channels": False}),
        ("output_audio_seconds", float("nan")),
        ("output_audio_seconds", 1),
        ("input_samples", 0),
        ("input_samples", True),
        ("input_frames", 2),
        ("input_sample_rate", 0),
        ("input_channels", 3),
    ),
)
def test_invalid_candidate_summary_is_a_contract_mismatch(tmp_path, field, value):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    left[field] = value
    assert _compare(left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field,value",
    (
        ("text", None),
        ("audio_samples", 3),
        ("num_samples", True),
        ("num_samples", 5),
        ("sample_rate", 8000),
        ("channels", 2),
        ("finite", False),
        ("audio_seconds", None),
        ("audio_seconds", 1),
        ("audio_artifact", None),
        ("input_samples", 0),
        ("input_sample_rate", False),
        ("input_channels", None),
    ),
)
def test_invalid_reference_summary_is_a_contract_mismatch(tmp_path, field, value):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    right[field] = value
    assert _compare(left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field,value", (("input_samples", 16), ("input_sample_rate", 8000), ("input_channels", 2))
)
def test_well_formed_but_different_input_counts_or_format_do_not_compare(tmp_path, field, value):
    right = _reference(tmp_path / "reference.wav")
    right[field] = value
    assert _compare(_candidate(), right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "malformation",
    (
        "missing",
        "truncated",
        "trailing",
        "integer_pcm",
        "missing_data",
        "repeated_data",
        "bad_alignment",
        "incomplete_frame",
        "nonfinite",
    ),
)
def test_actual_float_wav_artifact_must_be_complete_and_valid(tmp_path, malformation):
    left, right = _candidate(), _reference(tmp_path / "reference.wav")
    path = Path(right["audio_artifact"])
    payload = path.read_bytes()
    if malformation == "missing":
        right["audio_artifact"] = str(tmp_path / "missing.wav")
    elif malformation == "truncated":
        path.write_bytes(payload[:-1])
    elif malformation == "trailing":
        path.write_bytes(payload + b"\x00")
    elif malformation == "integer_pcm":
        _wav(path, [1, 2, 3, 4], format_tag=1)
    elif malformation == "missing_data":
        path.write_bytes(payload.replace(b"data", b"JUNK"))
    elif malformation == "repeated_data":
        body = payload[8:] + _chunk(b"data", b"")
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    elif malformation == "bad_alignment":
        changed = bytearray(payload)
        struct.pack_into("<H", changed, 32, 8)
        path.write_bytes(changed)
    elif malformation == "incomplete_frame":
        _wav(path, [1, 2, 3], channels=2)
    else:
        _wav(path, [float("inf"), 2, 3, 4])
    status, result = _compare(left, right)
    assert status == "contract-mismatch" and result["reason"].startswith("reference ")
    assert result["output_contract"]["numerical_parity_checked"] is False
