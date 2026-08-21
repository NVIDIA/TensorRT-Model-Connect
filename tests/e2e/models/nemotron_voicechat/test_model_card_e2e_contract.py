# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the real native VoiceChat E2E contract."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

from tests.e2e.models.nemotron_voicechat.e2e_plugins.comparator import (
    VoiceChatModelCardComparator,
)
from tests.e2e.models.nemotron_voicechat.e2e_plugins.runner import (
    VoiceChatModelCardRunner,
)
from tests.e2e_harness.contracts import StageOutput, StageSpec, ThresholdProfile

_ROOT = Path(__file__).resolve().parents[4]
_MODEL_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _MODEL_DIR / "manifests/nemotron-voicechat-11b.json"


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
    assert case["text_model_revision"] == "6533e8de2c68e4536bf7c411d7a3ce5734111476"
    assert (case["expected_output_sample_rate"], case["expected_output_num_samples"]) == (
        22050,
        345744,
    )
    assert case["expected_output_num_samples"] == (
        case["expected_output_codec_frames"] * case["expected_output_samples_per_frame"]
    )
    assert "runtime_cli_requires_hf_python" not in case.get("metadata", {})


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


def test_e2e_files_stay_model_owned() -> None:
    tracked = {
        path.relative_to(_ROOT).as_posix()
        for path in _MODEL_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert "tests/e2e/models/nemotron_voicechat/runner.py" in tracked
    assert "tests/e2e/models/nemotron_voicechat/test_nemotron_voicechat_e2e.py" in tracked
