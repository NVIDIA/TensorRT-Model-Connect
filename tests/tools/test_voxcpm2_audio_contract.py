"""VoxCPM2 audio E2E contract support tests."""

from __future__ import annotations

import json
import sys
import subprocess
import struct
import wave
from pathlib import Path

import tools.compare_wav_exact as compare_wav_exact
from tests.e2e_harness.comparators.text_to_audio import TextToAudioComparator
from tests.e2e_harness.contracts import (
    E2ECase,
    RunContext,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.orchestrator import _build_repro_commands
from tests.e2e_harness.registry import get_reference, reset
from tests.e2e_harness.references import voxcpm as voxcpm_reference
from tests.e2e_harness.runners import audio_speech


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
    assert case.inputs["prompt_wav_path"] is None
    assert case.inputs["prompt_text"] is None
    assert case.inputs["cfg_value"] == 2.0
    assert case.inputs["inference_timesteps"] == 10
    assert case.threshold_overrides["exact_waveform_match"] == 1.0
    assert case.stages[0].artifact_type == "waveform"
    assert case.stages[0].comparison_mode == "waveform_exact"
    reference_modules = {
        req.args.get("module")
        for req in case.preflight
        if req.kind == "python_module_available"
        and req.args.get("phase") == "reference"
        and req.gating
    }
    assert {"torch", "numpy", "voxcpm", "soundfile"} <= reference_modules


def test_voxcpm_reference_backend_is_discoverable():
    reset()
    assert get_reference("voxcpm") is not None


def test_voxcpm_reference_uses_model_card_params_and_float_wav(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "cfg_value": 2.0,
                    "duration_s": 0.1,
                    "inference_timesteps": 10,
                    "num_samples": 4800,
                    "rms": 0.1,
                    "sample_rate": 48000,
                    "wav_path": str(tmp_path / "voxcpm2" / "hf_reference.wav"),
                },
                sort_keys=True,
            )
            + "\n",
            stderr="",
        )

    monkeypatch.setattr(voxcpm_reference.subprocess, "run", _fake_run)

    case = E2ECase(
        name="voxcpm2",
        hf_id="openbmb/VoxCPM2",
        family="voxcpm2",
        runtime_strategy="text_to_audio_voxcpm2",
        reference_backend="voxcpm",
        inputs={
            "prompt": "Hello, this is the VoxCPM2 TensorRT Model Connect parity test.",
            "prompt_wav_path": None,
            "prompt_text": None,
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        },
    )
    ctx = RunContext(case=case, artifacts_dir=str(tmp_path), reference_python="/opt/ref-python")

    out = voxcpm_reference.VoxCPMReference().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    assert captured["cmd"][0] == "/opt/ref-python"
    script = captured["cmd"][2]
    assert "VoxCPM.from_pretrained('openbmb/VoxCPM2')" in script
    assert "prompt_wav_path=None" in script
    assert "prompt_text=None" in script
    assert "cfg_value=2.0" in script
    assert "inference_timesteps=10" in script
    assert 'getattr(model, "tts_model", model)' in script
    assert '"sample_rate", 48000' in script
    assert 'subtype="FLOAT"' in script
    assert out.data["returncode"] == 0
    assert out.data["sample_rate"] == 48000


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


def test_compare_wav_exact_cli_payload_matches_comparator_contract(tmp_path):
    trt_wav = tmp_path / "trt_output.wav"
    ref_wav = tmp_path / "hf_reference.wav"
    _write_pcm16_wav(trt_wav, [0, 64, -64, 128])
    _write_pcm16_wav(ref_wav, [0, 64, -64, 128])

    result = compare_wav_exact.compare_wavs(trt_wav, ref_wav)

    assert result["passed"] is True
    assert result["metrics"]["sample_rate_exact"] is True
    assert result["metrics"]["waveform_exact_match"] is True


def test_compare_wav_exact_cli_payload_fails_sample_mismatch(tmp_path):
    trt_wav = tmp_path / "trt_output.wav"
    ref_wav = tmp_path / "hf_reference.wav"
    _write_pcm16_wav(trt_wav, [0, 64, -64, 128])
    _write_pcm16_wav(ref_wav, [0, 64, -64, 129])

    result = compare_wav_exact.compare_wavs(trt_wav, ref_wav)

    assert result["passed"] is False
    assert result["metrics"]["sample_rate_exact"] is True
    assert result["metrics"]["waveform_exact_match"] is False


def test_compare_wav_exact_cli_reports_json_and_status(tmp_path):
    trt_wav = tmp_path / "trt_output.wav"
    ref_wav = tmp_path / "hf_reference.wav"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "compare_wav_exact.py"),
        str(trt_wav),
        str(ref_wav),
    ]

    _write_pcm16_wav(trt_wav, [0, 64, -64, 128])
    _write_pcm16_wav(ref_wav, [0, 64, -64, 128])
    passed = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert passed.returncode == 0
    passed_payload = json.loads(passed.stdout)
    assert passed_payload["passed"] is True
    assert passed_payload["metrics"]["waveform_exact_match"] is True

    _write_pcm16_wav(ref_wav, [0, 64, -64, 129])
    failed = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert failed.returncode == 1
    failed_payload = json.loads(failed.stdout)
    assert failed_payload["passed"] is False
    assert failed_payload["metrics"]["waveform_exact_match"] is False


def test_voxcpm2_trt_runner_preserves_required_output_wav(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        wav_path = Path(cmd[cmd.index("--output") + 1])
        _write_pcm16_wav(wav_path, [0, 64, -64, 128])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio_speech.subprocess, "run", _fake_run)

    case = E2ECase(
        name="voxcpm2",
        hf_id="openbmb/VoxCPM2",
        family="voxcpm2",
        runtime_strategy="text_to_audio_voxcpm2",
        reference_backend="voxcpm",
        bundle="voxcpm2.trtfb",
        inputs={
            "prompt": "Hello, this is the VoxCPM2 TensorRT Model Connect parity test.",
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        },
        metadata={
            "runtime_config": {
                "audio_voxcpm2": {
                    "cfg_value": 2.0,
                    "inference_timesteps": 10,
                }
            }
        },
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        binary_path="/tmp/trtmc",
        engine_dir=str(tmp_path / "engines"),
    )

    out = audio_speech.TextToAudioRunner().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    assert "audio_voxcpm2.cfg_value=2.0" in captured["cmd"]
    assert "audio_voxcpm2.inference_timesteps=10" in captured["cmd"]
    wav_path = Path(out.data["wav_path"])
    assert wav_path == tmp_path / "artifacts" / "voxcpm2" / "trt_output.wav"
    assert wav_path.is_file()
    assert out.data["wav_exists"] is True


def test_voxcpm2_repro_commands_preserve_audio_artifacts_and_exact_compare(tmp_path):
    case = load_manifest(MANIFEST_PATH)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        binary_path="/tmp/trtmc",
        hf_python=sys.executable,
        engine_dir=str(tmp_path / "engines"),
    )
    bundle_path = str(Path(ctx.engine_dir) / "voxcpm2.trtfb")

    repro = _build_repro_commands(case, ctx, bundle_path, {})

    trt_inference = repro["trt_inference"]
    assert "/tmp/trtmc generate-audio" in trt_inference
    assert bundle_path in trt_inference
    assert "--output" in trt_inference
    assert str(tmp_path / "artifacts" / "voxcpm2" / "trt_output.wav") in trt_inference
    assert "--set audio_voxcpm2.cfg_value=2.0" in trt_inference
    assert "--set audio_voxcpm2.inference_timesteps=10" in trt_inference

    compare_cmd = repro["compare_audio_exact"]
    assert "tools/compare_wav_exact.py" in compare_cmd
    assert str(tmp_path / "artifacts" / "voxcpm2" / "trt_output.wav") in compare_cmd
    assert str(tmp_path / "artifacts" / "voxcpm2" / "hf_reference.wav") in compare_cmd
