"""VoxCPM2 audio E2E contract support tests."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

from tests.e2e_harness.comparators.text_to_audio import TextToAudioComparator
from tests.e2e_harness.contracts import StageOutput, StageSpec, StageStatus, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.registry import get_reference, reset


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests" / "e2e" / "models" / "voxcpm2.json"


def _write_pcm16_wav(path: Path, samples: list[int], *, sample_rate: int = 48000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _output(path: Path) -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        data={
            "returncode": 0,
            "wav_exists": True,
            "wav_path": str(path),
            "rms": 0.1,
            "duration_s": 4 / 48000,
            "sample_rate": 48000,
        },
    )


def test_voxcpm2_manifest_enforces_exact_reference_waveform_contract():
    case = load_manifest(MANIFEST_PATH)

    assert case.name == "voxcpm2"
    assert case.hf_id == "openbmb/VoxCPM2"
    assert case.family == "voxcpm2"
    assert case.runtime_strategy == "text_to_audio_voxcpm2"
    assert case.task_strategy == "text_to_audio"
    assert case.reference_backend == "voxcpm"
    assert case.reference_family == "tts_voxcpm2"
    assert case.user_contract == "tts_audio"
    assert case.inputs["cfg_value"] == 2.0
    assert case.inputs["inference_timesteps"] == 10
    assert case.threshold_overrides["exact_waveform_match"] == 1.0
    assert case.stages[0].artifact_type == "waveform"
    assert case.stages[0].comparison_mode == "waveform_exact"


def test_voxcpm_reference_backend_is_discoverable():
    reset()
    assert get_reference("voxcpm") is not None


def test_exact_waveform_mode_passes_identical_wavs(tmp_path):
    trt_wav = tmp_path / "trt.wav"
    ref_wav = tmp_path / "ref.wav"
    _write_pcm16_wav(trt_wav, [0, 64, -64, 128])
    _write_pcm16_wav(ref_wav, [0, 64, -64, 128])

    result = TextToAudioComparator().compare(
        _output(trt_wav),
        _output(ref_wav),
        ThresholdProfile(
            task_strategy="text_to_audio",
            metrics={"exact_waveform_match": 1.0},
        ),
        StageSpec(
            name="full_generation",
            artifact_type="waveform",
            comparison_mode="waveform_exact",
        ),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["sample_rate_exact"].passed
    assert result.metrics["waveform_exact_match"].passed


def test_exact_waveform_mode_fails_sample_mismatch(tmp_path):
    trt_wav = tmp_path / "trt.wav"
    ref_wav = tmp_path / "ref.wav"
    _write_pcm16_wav(trt_wav, [0, 64, -64, 128])
    _write_pcm16_wav(ref_wav, [0, 64, -64, 129])

    result = TextToAudioComparator().compare(
        _output(trt_wav),
        _output(ref_wav),
        ThresholdProfile(
            task_strategy="text_to_audio",
            metrics={"exact_waveform_match": 1.0},
        ),
        StageSpec(
            name="full_generation",
            artifact_type="waveform",
            comparison_mode="waveform_exact",
        ),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["waveform_exact_match"].passed
