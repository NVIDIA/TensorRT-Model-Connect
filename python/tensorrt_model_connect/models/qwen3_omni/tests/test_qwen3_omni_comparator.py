# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni-owned comparator tests."""

from __future__ import annotations

import math
import struct
import wave

from tensorrt_model_connect.models.qwen3_omni.tests.e2e_plugins.comparators.omni import OmniComparator
from tests.e2e_harness.contracts import (
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


def _invariant_ref(stage_name: str, *, num_samples: int = 37_845) -> StageOutput:
    return StageOutput(
        stage_name=stage_name,
        data={
            "_invariant_only": True,
            "sample_rate": 24_000,
            "num_samples": num_samples,
        },
        metadata={"source": "invariant_only"},
    )


def _write_wav(path, *, num_samples: int, amplitude: float = 0.2) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        samples = (
            int(amplitude * 32767 * math.sin(2 * math.pi * 440 * i / 24_000))
            for i in range(num_samples)
        )
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _write_float32_wav(path, *, num_samples: int, amplitude: float = 0.2) -> None:
    samples = b"".join(
        struct.pack("<f", amplitude * math.sin(2 * math.pi * 440 * i / 24_000))
        for i in range(num_samples)
    )
    data_size = len(samples)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 3, 1, 24_000, 24_000 * 4, 4, 32)
        + b"data"
        + struct.pack("<I", data_size)
        + samples
    )


def _threshold() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="omni_multimodal",
        metrics={
            "audio_artifact_bytes_min": 44.0,
            "audio_duration_s_min": 0.5,
            "audio_reference_duration_ratio_min": 0.5,
            "audio_rms_min": 0.005,
            "audio_peak_min": 0.02,
            "audio_reference_waveform_cosine_min": 0.25,
        },
    )


def test_omni_invariant_talker_requires_meaningful_audio(tmp_path) -> None:
    audio = tmp_path / "talker.wav"
    _write_wav(audio, num_samples=24_000)
    reference = _invariant_ref("talker_decode", num_samples=24_000)
    reference.data["wav_path"] = str(audio)

    result = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            metadata={"audio_output_path": str(audio)},
        ),
        reference,
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["audio_artifact_bytes"].passed is True
    assert result.metrics["audio_wav_valid"].passed is True
    assert result.metrics["audio_reference_duration_ratio"].passed is True
    assert result.metrics["audio_rms"].passed is True


def test_omni_invariant_talker_accepts_product_float32_wav(tmp_path) -> None:
    audio = tmp_path / "talker.wav"
    reference_audio = tmp_path / "reference.wav"
    _write_float32_wav(audio, num_samples=24_000)
    _write_wav(reference_audio, num_samples=24_000)
    reference = _invariant_ref("talker_decode", num_samples=24_000)
    reference.data["wav_path"] = str(reference_audio)

    result = OmniComparator().compare(
        StageOutput(stage_name="talker_decode", metadata={"audio_output_path": str(audio)}),
        reference,
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["audio_wav_valid"].passed is True
    assert result.metrics["audio_encoding_supported"].passed is True


def test_omni_talker_requires_exact_thinker_text_and_audio_shape(tmp_path) -> None:
    audio = tmp_path / "talker.wav"
    short_audio = tmp_path / "short.wav"
    _write_float32_wav(audio, num_samples=37_845)
    _write_float32_wav(short_audio, num_samples=35_925)
    reference = _invariant_ref("talker_decode")
    reference.data.update(
        {
            "decoded_text": "Hello from Qwen-Omni!",
            "wav_path": str(audio),
        }
    )

    matching = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            data={"thinker_text": "Hello from Qwen-Omni!"},
            metadata={"audio_output_path": str(audio)},
        ),
        reference,
        _threshold(),
        StageSpec(name="talker_decode"),
    )
    wrong_text = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            data={"thinker_text": "Hello from Qwen-Omni"},
            metadata={"audio_output_path": str(audio)},
        ),
        reference,
        _threshold(),
        StageSpec(name="talker_decode"),
    )
    wrong_shape = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            data={"thinker_text": "Hello from Qwen-Omni!"},
            metadata={"audio_output_path": str(short_audio)},
        ),
        reference,
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert matching.status == StageStatus.PASSED.value
    assert matching.metrics["thinker_text_exact"].passed is True
    assert matching.metrics["audio_num_samples_exact"].passed is True
    assert wrong_text.metrics["thinker_text_exact"].passed is False
    assert wrong_shape.metrics["audio_num_samples_exact"].passed is False


def test_omni_waveform_oracle_fails_closed_without_reference_audio(tmp_path) -> None:
    audio = tmp_path / "talker.wav"
    _write_wav(audio, num_samples=24_000)

    result = OmniComparator().compare(
        StageOutput(stage_name="talker_decode", metadata={"audio_output_path": str(audio)}),
        _invariant_ref("talker_decode"),
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["audio_reference_waveform_cosine"].passed is False


def test_omni_invariant_talker_fails_without_audio(tmp_path) -> None:
    result = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            metadata={"audio_output_path": str(tmp_path / "missing.wav")},
        ),
        _invariant_ref("talker_decode"),
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["audio_artifact_bytes"].passed is False


def test_omni_invariant_talker_rejects_non_wav_bytes(tmp_path) -> None:
    audio = tmp_path / "talker.wav"
    audio.write_bytes(b"RIFFaudio")

    result = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            metadata={"audio_output_path": str(audio)},
        ),
        _invariant_ref("talker_decode"),
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["audio_wav_valid"].passed is False


def test_omni_invariant_talker_rejects_short_fallback_waveform(tmp_path) -> None:
    audio = tmp_path / "talker.wav"
    _write_wav(audio, num_samples=5_120)

    result = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            metadata={
                "audio_output_path": str(audio),
                "stderr": ("[trtmc] Omni: no Code2Wav engine, generating simple waveform\n"),
            },
        ),
        _invariant_ref("talker_decode"),
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["simple_waveform_fallback_absent"].passed is False
    assert result.metrics["audio_duration_s"].passed is False
    assert result.metrics["audio_reference_duration_ratio"].passed is False


def test_omni_invariant_talker_rejects_silent_audio(tmp_path) -> None:
    audio = tmp_path / "talker.wav"
    _write_wav(audio, num_samples=24_000, amplitude=0.0)

    result = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            metadata={"audio_output_path": str(audio)},
        ),
        _invariant_ref("talker_decode"),
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["audio_rms"].passed is False
    assert result.metrics["audio_peak"].passed is False


def test_omni_invariant_talker_compares_pinned_reference_waveform(tmp_path) -> None:
    reference = tmp_path / "reference.wav"
    matching = tmp_path / "matching.wav"
    unrelated = tmp_path / "unrelated.wav"
    _write_wav(reference, num_samples=24_000)
    _write_wav(matching, num_samples=24_000)
    with wave.open(str(unrelated), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        samples = (
            int(0.2 * 32767 * math.sin(2 * math.pi * 880 * i / 24_000)) for i in range(24_000)
        )
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))

    reference_output = _invariant_ref("talker_decode", num_samples=24_000)
    reference_output.data["wav_path"] = str(reference)
    matching_result = OmniComparator().compare(
        StageOutput(stage_name="talker_decode", metadata={"audio_output_path": str(matching)}),
        reference_output,
        _threshold(),
        StageSpec(name="talker_decode"),
    )
    unrelated_result = OmniComparator().compare(
        StageOutput(stage_name="talker_decode", metadata={"audio_output_path": str(unrelated)}),
        reference_output,
        _threshold(),
        StageSpec(name="talker_decode"),
    )

    assert matching_result.metrics["audio_reference_waveform_cosine"].passed is True
    assert unrelated_result.metrics["audio_reference_waveform_cosine"].passed is False


def test_omni_invariant_text_stage_requires_non_empty_output() -> None:
    result = OmniComparator().compare(
        StageOutput(stage_name="end_to_end", text="hello"),
        _invariant_ref("end_to_end"),
        ThresholdProfile(task_strategy="omni_multimodal"),
        StageSpec(name="end_to_end"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["non_empty_text"].passed is True
