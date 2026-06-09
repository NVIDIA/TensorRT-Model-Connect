"""VoxCPM2 audio E2E contract support tests."""

from __future__ import annotations

import hashlib
import json
import sys
import subprocess
import struct
import wave
from pathlib import Path

import pytest
import tools.compare_wav_exact as compare_wav_exact
from tests.e2e_harness.comparators.text_to_audio import TextToAudioComparator
from tests.e2e_harness.contracts import (
    CompareResult,
    E2ECase,
    MetricResult,
    RunContext,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.orchestrator import (
    _build_repro_commands,
    _auto_register_artifacts,
    _register_compare_artifacts,
)
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
    assert {
        "torch",
        "numpy",
        "voxcpm",
        "wetext",
        "modelscope",
        "soundfile",
    } <= reference_modules
    build_modules = {
        req.args.get("module")
        for req in case.preflight
        if req.kind == "python_module_available"
        and req.args.get("phase") == "build"
        and req.gating
    }
    assert "tensorrt" in build_modules
    assert any(
        req.kind == "gpu_count_min"
        and req.args.get("count") == 1
        and req.gating
        for req in case.preflight
    )


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
    assert "TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR" in script
    assert "install_voxcpm2_tensor_dump(model)" in script
    assert out.data["returncode"] == 0
    assert out.data["sample_rate"] == 48000
    assert out.data["result_json_path"] == str(
        tmp_path / "voxcpm2" / "hf_reference_result.json"
    )


def test_voxcpm_reference_forwards_shared_locdit_noise_when_dumping(
    monkeypatch, tmp_path
):
    captured: dict[str, dict[str, str]] = {}
    noise_path = tmp_path / "voxcpm2" / "locdit_noise.raw"
    noise_path.parent.mkdir(parents=True)
    noise_path.write_bytes(struct.pack("<12f", *[float(i) for i in range(12)]))
    monkeypatch.setenv(
        "TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR", str(tmp_path / "hf_tensor_dump")
    )

    def _fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
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
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        },
    )
    ctx = RunContext(case=case, artifacts_dir=str(tmp_path), reference_python="/opt/ref-python")

    out = voxcpm_reference.VoxCPMReference().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    assert captured["env"]["TRTMC_VOXCPM2_HF_NOISE_RAW"] == str(noise_path)
    assert out.data["locdit_noise_raw"] == str(noise_path)


def test_voxcpm_reference_tensor_dump_hook_writes_trt_compatible_manifest(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    from tests.e2e_harness.references.voxcpm_debug import (
        install_voxcpm2_tensor_dump,
    )

    class Projection:
        def forward(self, value):
            return value

    class IdentityLM:
        def forward(self, *, inputs_embeds, is_causal=True):
            return inputs_embeds + 1.0, []

        def forward_step(self, inputs_embeds, position_id):
            return inputs_embeds + position_id.reshape(1, 1).to(inputs_embeds.dtype)

    class FsqLayer:
        def forward(self, hidden):
            return hidden + 2.0

    class Decoder:
        in_channels = 3

        def solve_euler(self, *, x, **_kwargs):
            return x + 1.0

    class TTSModel:
        patch_size = 4
        feat_dim = 3

        def __init__(self):
            self.enc_to_lm_proj = Projection()
            self.base_lm = IdentityLM()
            self.fsq_layer = FsqLayer()
            self.fusion_concat_proj = Projection()
            self.residual_lm = IdentityLM()
            self.lm_to_dit_proj = Projection()
            self.res_to_dit_proj = Projection()
            self.feat_decoder = Decoder()

        def _dtype(self):
            return torch.bfloat16

        def _inference(self, text, text_mask, feat, feat_mask, **_kwargs):
            local = self.enc_to_lm_proj.forward(
                torch.ones((1, 2, 5), dtype=torch.bfloat16)
            )
            enc_outputs, _ = self.base_lm.forward(
                inputs_embeds=local, is_causal=True
            )
            semantic = self.fsq_layer.forward(enc_outputs) * feat_mask.unsqueeze(
                -1
            ).to(torch.bfloat16) + enc_outputs * text_mask.unsqueeze(-1).to(
                torch.bfloat16
            )
            residual_inputs = self.fusion_concat_proj.forward(
                torch.cat((semantic, feat_mask.unsqueeze(-1).to(torch.bfloat16) * local), dim=-1)
            )
            residual_outputs, _ = self.residual_lm.forward(
                inputs_embeds=residual_inputs, is_causal=True
            )
            curr = self.enc_to_lm_proj.forward(
                torch.ones((1, 1, 5), dtype=torch.bfloat16)
            )
            lm_step = self.base_lm.forward_step(
                curr[:, 0, :], torch.tensor([2], dtype=torch.int32)
            )
            lm_step = self.fsq_layer.forward(lm_step)
            residual_step_input = self.fusion_concat_proj.forward(
                torch.cat((lm_step, curr[:, 0, :]), dim=-1)
            )
            self.residual_lm.forward_step(
                residual_step_input, torch.tensor([2], dtype=torch.int32)
            )
            self.lm_to_dit_proj.forward(semantic[:, -1, :])
            self.res_to_dit_proj.forward(residual_outputs[:, -1, :])
            yield torch.zeros((1,), dtype=torch.float32), None, None

    class Model:
        def __init__(self):
            self.tts_model = TTSModel()

    dump_dir = tmp_path / "hf_dump"
    noise_path = tmp_path / "noise.raw"
    noise_path.write_bytes(struct.pack("<12f", *[float(i) for i in range(12)]))
    monkeypatch.setenv("TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("TRTMC_VOXCPM2_HF_NOISE_RAW", str(noise_path))

    model = Model()
    assert install_voxcpm2_tensor_dump(model) is True
    tts = model.tts_model
    list(
        tts._inference(
            torch.tensor([[11, 101]], dtype=torch.int64),
            torch.tensor([[1, 1]], dtype=torch.int32),
            torch.zeros((1, 2, 4, 3), dtype=torch.bfloat16),
            torch.tensor([[0, 0]], dtype=torch.int32),
        )
    )
    tts.lm_to_dit_proj.forward(
        torch.arange(5, dtype=torch.float32).reshape(1, 5).to(torch.bfloat16)
    )
    tts.res_to_dit_proj.forward(
        torch.arange(5, 10, dtype=torch.float32).reshape(1, 5).to(torch.bfloat16)
    )
    out = tts.feat_decoder.forward(
        mu=torch.zeros((1, 6), dtype=torch.bfloat16),
        n_timesteps=10,
        patch_size=4,
        cond=torch.zeros((1, 3, 4), dtype=torch.bfloat16),
        cfg_value=2.0,
    )

    assert tuple(out.shape) == (1, 3, 4)
    records = [
        json.loads(line)
        for line in (dump_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    record_keys = [
        (record["phase"], record["step"], record["direction"], record["name"])
        for record in records
    ]
    assert ("tslm_prefill", 0, "input", "text_tokens") in record_keys
    assert ("tslm_prefill", 1, "output", "semantic_lm_states") in record_keys
    assert ("ralm_prefill", 1, "input", "semantic_lm_states") in record_keys
    assert ("ralm_prefill", 1, "output", "residual_hidden") in record_keys
    assert ("tslm_refresh", 0, "input", "local_text_features") in record_keys
    assert ("tslm_refresh", 0, "output", "lm_hidden") in record_keys
    assert ("ralm_refresh", 0, "input", "local_text_features") in record_keys
    assert ("ralm_refresh", 0, "output", "residual_hidden") in record_keys
    assert ("locdit", 0, "input", "locdit_noise") in record_keys
    assert ("locdit", 0, "output", "audio_vae_latents") in record_keys

    by_name = {
        (record["phase"], record["step"], record["direction"], record["name"]): record
        for record in records
    }
    assert by_name[("tslm_prefill", 0, "input", "text_tokens")]["dtype"] == "int32"
    assert by_name[("tslm_prefill", 0, "input", "text_tokens")]["shape"] == [1]
    assert by_name[("tslm_prefill", 0, "input", "local_text_features")]["dtype"] == "bfloat16"
    assert by_name[("tslm_prefill", 0, "input", "local_text_features")]["shape"] == [1, 5]
    assert by_name[("ralm_prefill", 1, "output", "residual_hidden")]["shape"] == [1, 10]
    assert by_name[("locdit", 0, "input", "locdit_noise")]["dtype"] == "bfloat16"
    assert by_name[("locdit", 0, "input", "locdit_noise")]["shape"] == [4, 3]
    assert by_name[("locdit", 0, "input", "locdit_noise")]["nbytes"] == 24
    assert by_name[("locdit", 0, "input", "lm_hidden")]["shape"] == [1, 5]
    assert by_name[("locdit", 0, "output", "audio_vae_latents")]["dtype"] == "float32"
    assert by_name[("locdit", 0, "output", "audio_vae_latents")]["shape"] == [4, 3]
    for record in records:
        assert Path(record["path"]).is_file()


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
    sidecar = tmp_path / "compare_wav_exact.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["command"] == [
        "python",
        "tools/compare_wav_exact.py",
        str(trt_wav),
        str(ref_wav),
    ]
    assert payload["result"]["metrics"]["waveform_exact_match"] is True
    assert (
        payload["result"]["trt"]["data_sha256"]
        == payload["result"]["ref"]["data_sha256"]
    )
    assert result.metrics["waveform_exact_match"].note == (
        f"exact_compare_result={sidecar}"
    )


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
    expected_digest = hashlib.sha256(
        compare_wav_exact.read_wav_payload(trt_wav)["data"]
    ).hexdigest()
    assert result["trt"]["data_sha256"] == expected_digest
    assert result["trt"]["data_sha256"] == result["ref"]["data_sha256"]


def test_exact_compare_sidecar_is_registered_as_report_artifact(tmp_path):
    sidecar = tmp_path / "compare_wav_exact.json"
    sidecar.write_text(
        json.dumps({"result": {"passed": True}}, sort_keys=True),
        encoding="utf-8",
    )

    class Sink:
        base_dir = tmp_path

        def __init__(self) -> None:
            self.artifacts: dict[str, str] = {}

        def register_artifact(self, key: str, rel_path: str) -> None:
            self.artifacts[key] = rel_path

    sink = Sink()
    result = CompareResult(
        stage_name="full_generation",
        metrics={
            "waveform_exact_match": MetricResult(
                value=1.0,
                threshold=1.0,
                operator="==",
                passed=True,
                note=f"exact_compare_result={sidecar}",
            )
        },
    )

    _register_compare_artifacts(sink, result)

    assert sink.artifacts["compare_wav_exact"] == "compare_wav_exact.json"


def test_voxcpm_reference_result_json_is_registered_as_report_artifact(tmp_path):
    result_json = tmp_path / "voxcpm2" / "hf_reference_result.json"
    result_json.parent.mkdir()
    result_json.write_text(json.dumps({"sample_rate": 48000}), encoding="utf-8")

    class Sink:
        base_dir = tmp_path

        def __init__(self) -> None:
            self.artifacts: dict[str, str] = {}

        def register_artifact(self, key: str, rel_path: str) -> None:
            self.artifacts[key] = rel_path

    sink = Sink()
    out = StageOutput(
        stage_name="full_generation",
        data={"result_json_path": str(result_json)},
    )

    _auto_register_artifacts(sink, out, "ref")

    assert sink.artifacts["ref_result_json"] == "voxcpm2/hf_reference_result.json"


def test_voxcpm2_locdit_noise_raw_is_registered_as_report_artifact(tmp_path):
    noise_path = tmp_path / "voxcpm2" / "locdit_noise.raw"
    noise_path.parent.mkdir()
    noise_path.write_bytes(b"noise")

    class Sink:
        base_dir = tmp_path

        def __init__(self) -> None:
            self.artifacts: dict[str, str] = {}

        def register_artifact(self, key: str, rel_path: str) -> None:
            self.artifacts[key] = rel_path

    sink = Sink()
    out = StageOutput(
        stage_name="full_generation",
        data={"locdit_noise_raw": str(noise_path)},
    )

    _auto_register_artifacts(sink, out, "trt")

    assert sink.artifacts["trt_locdit_noise_raw"] == "voxcpm2/locdit_noise.raw"


def test_compare_wav_exact_cli_payload_fails_sample_mismatch(tmp_path):
    trt_wav = tmp_path / "trt_output.wav"
    ref_wav = tmp_path / "hf_reference.wav"
    _write_pcm16_wav(trt_wav, [0, 64, -64, 128])
    _write_pcm16_wav(ref_wav, [0, 64, -64, 129])

    result = compare_wav_exact.compare_wavs(trt_wav, ref_wav)

    assert result["passed"] is False
    assert result["metrics"]["sample_rate_exact"] is True
    assert result["metrics"]["waveform_exact_match"] is False
    assert result["trt"]["data_sha256"] != result["ref"]["data_sha256"]


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
    assert captured["cmd"][captured["cmd"].index("--cfg-scale") + 1] == "2.0"
    assert captured["cmd"][captured["cmd"].index("--num-steps") + 1] == "10"
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

    hf_reference = repro["hf_reference_audio"]
    assert "from voxcpm import VoxCPM" in hf_reference
    assert "VoxCPM.from_pretrained('openbmb/VoxCPM2')" in hf_reference
    assert "cfg_value=2.0" in hf_reference
    assert "inference_timesteps=10" in hf_reference
    assert "TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR" in hf_reference
    assert "TRTMC_VOXCPM2_HF_NOISE_RAW" in hf_reference
    assert str(tmp_path / "artifacts" / "voxcpm2" / "hf_tensor_dump") in hf_reference
    assert str(tmp_path / "artifacts" / "voxcpm2" / "locdit_noise.raw") in hf_reference
    assert "install_voxcpm2_tensor_dump(model)" in hf_reference
    assert str(tmp_path / "artifacts" / "voxcpm2" / "hf_reference.wav") in hf_reference
    assert (
        str(tmp_path / "artifacts" / "voxcpm2" / "hf_reference_result.json")
        in hf_reference
    )

    trt_inference = repro["trt_inference"]
    assert "/tmp/trtmc generate-audio" in trt_inference
    assert bundle_path in trt_inference
    assert "--output" in trt_inference
    assert str(tmp_path / "artifacts" / "voxcpm2" / "trt_output.wav") in trt_inference
    assert "--set audio_voxcpm2.cfg_value=2.0" in trt_inference
    assert "--set audio_voxcpm2.inference_timesteps=10" in trt_inference
    assert "--cfg-scale 2.0" in trt_inference
    assert "--num-steps 10" in trt_inference

    compare_cmd = repro["compare_audio_exact"]
    assert "tools/compare_wav_exact.py" in compare_cmd
    assert str(tmp_path / "artifacts" / "voxcpm2" / "trt_output.wav") in compare_cmd
    assert str(tmp_path / "artifacts" / "voxcpm2" / "hf_reference.wav") in compare_cmd
    assert ">" in compare_cmd
    assert (
        str(tmp_path / "artifacts" / "voxcpm2" / "compare_wav_exact.json")
        in compare_cmd
    )
