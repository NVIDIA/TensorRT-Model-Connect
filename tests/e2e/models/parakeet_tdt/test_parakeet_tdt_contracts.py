# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import io
import importlib
import importlib.util
import json
import re
import runpy
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
FAMILY_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect" / "families" / "parakeet_tdt"
PINNED_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"


def _load_module(name: str, filename: str):
    path = FAMILY_ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hf_config() -> dict:
    return {
        "architectures": ["ParakeetForTDT"],
        "model_type": "parakeet_tdt",
        "vocab_size": 8193,
        "blank_token_id": 8192,
        "decoder_hidden_size": 640,
        "num_decoder_layers": 2,
        "durations": [0, 1, 2, 3, 4],
        "max_symbols_per_step": 10,
        "hidden_act": "relu",
        "pad_token_id": 2,
        "encoder_config": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_mel_bins": 128,
            "conv_kernel_size": 9,
            "subsampling_conv_channels": 256,
            "subsampling_factor": 8,
            "max_position_embeddings": 5000,
            "attention_bias": False,
            "convolution_bias": False,
            "scale_input": False,
        },
    }


def test_config_parses_the_published_architecture_and_rejects_drift() -> None:
    config_mod = _load_module("parakeet_tdt_config", "config.py")
    cfg = config_mod.ParakeetTDTConfig.from_json(json.dumps(_hf_config()))

    assert cfg.encoder_hidden_size == 1024
    assert cfg.encoder_layers == 24
    assert cfg.encoder_heads == 8
    assert cfg.decoder_hidden_size == 640
    assert cfg.decoder_layers == 2
    assert cfg.vocab_size == 8193
    assert cfg.blank_id == 8192
    assert cfg.durations == (0, 1, 2, 3, 4)
    cfg.validate_supported_checkpoint()

    bad = _hf_config()
    bad["durations"] = [0, 1, 2, 4]
    with pytest.raises(ValueError, match="durations"):
        config_mod.ParakeetTDTConfig.from_json(json.dumps(bad)).validate_supported_checkpoint()


def test_config_from_dir_reports_the_missing_family_contract(tmp_path: Path) -> None:
    config_mod = _load_module("parakeet_tdt_config_missing", "config.py")

    with pytest.raises(FileNotFoundError, match="Parakeet TDT model directory"):
        config_mod.ParakeetTDTConfig.from_dir(tmp_path)


def test_family_private_builder_config_only_projects_parakeet_fields() -> None:
    model_config = _load_module("parakeet_tdt_model_config", "model_config.py")

    cfg = model_config.ModelConfig.from_json(json.dumps(_hf_config()))

    assert cfg.model_type == "parakeet_tdt"
    assert cfg.hidden_size == 640
    assert cfg.num_hidden_layers == 2
    assert cfg.vocab_size == 8193
    assert cfg.raw["encoder_config"]["hidden_size"] == 1024


def test_nemo_checkpoint_load_uses_safe_weight_only_mode(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = _load_module("parakeet_tdt_checkpoint_safe", "checkpoint.py")
    archive_path = tmp_path / "model.nemo"
    with tarfile.open(archive_path, "w") as archive:
        for name, payload in (
            ("model_config.yaml", b"target: test\n"),
            ("model_weights.ckpt", b"checkpoint"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    captured: dict[str, object] = {}
    fake_torch = ModuleType("torch")

    def _load(_stream, **kwargs):
        captured.update(kwargs)
        return {"state_dict": {"weight": np.array([1.0], dtype=np.float32)}}

    fake_torch.load = _load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    state, _ = checkpoint.load_nemo_archive(tmp_path)

    assert captured == {"map_location": "cpu", "weights_only": True}
    assert "weight" in state


def test_tdt_policy_uses_duration_values_and_forces_zero_duration_progress() -> None:
    policy = _load_module("parakeet_tdt_decode_policy", "decode_policy.py")

    emitted = policy.make_tdt_greedy_decision(
        token_id=7,
        duration_index=2,
        durations=(0, 1, 3),
        blank_id=9,
    )
    assert emitted.emit_token is True
    assert emitted.frame_advance == 3

    blank_zero = policy.make_tdt_greedy_decision(
        token_id=9,
        duration_index=0,
        durations=(0, 1, 3),
        blank_id=9,
    )
    assert blank_zero.emit_token is False
    assert blank_zero.frame_advance == 1

    nonblank_zero = policy.make_tdt_greedy_decision(
        token_id=7,
        duration_index=0,
        durations=(0, 1, 3),
        blank_id=9,
    )
    assert nonblank_zero.emit_token is True
    assert nonblank_zero.frame_advance == 0


def test_hf_tensor_mapper_keeps_token_and_duration_heads_separate() -> None:
    checkpoint = _load_module("parakeet_tdt_checkpoint", "checkpoint.py")
    state = {
        "decoder.embedding.weight": np.zeros((6, 4), dtype=np.float32),
        "decoder.lstm.weight_ih_l0": np.zeros((16, 4), dtype=np.float32),
        "decoder.lstm.weight_hh_l0": np.zeros((16, 4), dtype=np.float32),
        "decoder.lstm.bias_ih_l0": np.ones((16,), dtype=np.float32),
        "decoder.lstm.bias_hh_l0": np.ones((16,), dtype=np.float32),
        "decoder.decoder_projector.weight": np.zeros((4, 4), dtype=np.float32),
        "decoder.decoder_projector.bias": np.zeros((4,), dtype=np.float32),
        "encoder_projector.weight": np.zeros((4, 8), dtype=np.float32),
        "encoder_projector.bias": np.zeros((4,), dtype=np.float32),
        "joint.head.weight": np.arange(36, dtype=np.float32).reshape(9, 4),
        "joint.head.bias": np.arange(9, dtype=np.float32),
    }
    mapped = checkpoint.map_transducer_weights(
        state,
        vocab_size=6,
        duration_count=3,
        decoder_layers=1,
        decoder_hidden_size=4,
        encoder_hidden_size=8,
    )

    assert mapped["joint_token_w"].shape == (4, 6)
    assert mapped["joint_duration_w"].shape == (4, 3)
    np.testing.assert_array_equal(mapped["joint_token_b"], np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(mapped["joint_duration_b"], np.arange(6, 9, dtype=np.float32))
    np.testing.assert_array_equal(mapped["pred.0.bias"], np.full((16,), 2.0, dtype=np.float32))


def test_nemo_classifier_uses_tdt_architecture_fields_not_checkpoint_name(tmp_path: Path) -> None:
    archive = _load_module("parakeet_tdt_nemo_archive", "nemo_archive.py")
    parakeet = {
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "decoding": {"model_type": "tdt", "durations": [0, 1, 2, 3, 4]},
        "joint": {"num_extra_outputs": 5},
    }
    ordinary_rnnt = {
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "joint": {"num_extra_outputs": 0},
    }
    assert archive._matches_parakeet_tdt(parakeet)
    assert not archive._matches_parakeet_tdt(ordinary_rnnt)

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "parakeet_tdt"}), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"header-only-test-placeholder")
    (tmp_path / "legacy.nemo").write_bytes(b"not-read-when-hf-is-present")
    assert archive.resolve_model_dir(tmp_path) == tmp_path


def test_nemo_resolution_emits_the_complete_typed_checkpoint_schema(
    monkeypatch, tmp_path: Path
) -> None:
    archive = _load_module("parakeet_tdt_nemo_archive_schema", "nemo_archive.py")
    config_mod = _load_module("parakeet_tdt_config_schema", "config.py")
    nemo_cfg = {
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "encoder": {
            "d_model": 1024,
            "n_layers": 24,
            "n_heads": 8,
            "ff_expansion_factor": 4,
            "conv_kernel_size": 9,
            "subsampling_conv_channels": 256,
            "subsampling_factor": 8,
            "max_len": 5000,
        },
        "preprocessor": {"features": 128},
        "decoder": {
            "blank_idx": 8192,
            "prednet": {"pred_hidden": 640, "pred_rnn_layers": 2},
        },
        "joint": {
            "num_extra_outputs": 5,
            "jointnet": {"activation": "relu"},
        },
        "decoding": {"durations": [0, 1, 2, 3, 4], "max_symbols_per_step": 10},
        "tdt_durations": [0, 1, 2, 3, 4],
        "vocab_size": 8193,
    }
    import yaml

    nemo_path = tmp_path / "checkpoint.nemo"
    payload = yaml.safe_dump(nemo_cfg).encode()
    with tarfile.open(nemo_path, "w") as tar:
        info = tarfile.TarInfo("model_config.yaml")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    staged = tmp_path / "staged"
    staged.mkdir()
    monkeypatch.setattr(archive.tempfile, "mkdtemp", lambda **_kwargs: str(staged))
    monkeypatch.setattr(archive, "_symlink_archive", lambda *_args: None)

    resolved = Path(archive.resolve_nemo_archive(nemo_path))
    cfg = config_mod.ParakeetTDTConfig.from_dir(resolved)

    cfg.validate_supported_checkpoint()


def test_manifest_pins_exact_hugging_face_revision_and_strict_asr_oracle() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "parakeet-tdt-0.6b-v3.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert raw["hf_id"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert raw["hf_revision"] == PINNED_REVISION
    assert raw["family"] == "parakeet_tdt"
    assert raw["runtime_strategy"] == "parakeet_tdt_speech_to_text"
    assert raw["task_strategy"] == "speech_to_text"
    assert raw["testcases"]
    assert all(case["reference_family"] == "asr_parakeet_tdt" for case in raw["testcases"])
    assert all(case["user_contract"] == "exact_transcript" for case in raw["testcases"])
    assert len(raw["testcases"]) == 8
    family_root = manifest_path.parents[1]
    for case in raw["testcases"]:
        assert (family_root / case["test_input_audio"]).is_file()
        threshold = family_root / "thresholds" / f"{case['name']}.json"
        values = json.loads(threshold.read_text(encoding="utf-8"))["threshold_overrides"]
        assert values["wer"] <= 0.3
        assert values["cer"] <= 0.2


def test_ffi_kernel_tempfiles_are_explicitly_disabled_on_windows() -> None:
    source = (
        REPO_ROOT / "src/runtime/models/parakeet_tdt/plugin_helpers.cpp"
    ).read_text(encoding="utf-8")

    assert "#if TRTMC_HAS_TVM_FFI && !defined(_WIN32)" in source
    assert "TVM-FFI bundle kernels are not supported on Windows" in source


def test_hf_reference_propagates_the_manifest_revision(monkeypatch, tmp_path: Path) -> None:
    reference = importlib.import_module(
        "tests.e2e.models.parakeet_tdt.e2e_plugins.references.parakeet_tdt_hf"
    )
    captured: dict = {}

    def _fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        script = command[2]
        output_match = re.search(r"^output_path = (.+)$", script, re.MULTILINE)
        assert output_match is not None
        output_path = ast.literal_eval(output_match.group(1))
        Path(output_path).write_text(json.dumps({"text": "reference text"}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(reference.subprocess, "run", _fake_run)
    case = SimpleNamespace(
        task_strategy="speech_to_text",
        metadata={},
        hf_revision=PINNED_REVISION,
        hf_id="nvidia/parakeet-tdt-0.6b-v3",
        inputs={"audio": "missing-test-audio.wav"},
        name="pinned-reference",
    )
    stage = SimpleNamespace(name="full_inference")
    ctx = SimpleNamespace(
        artifacts_dir=str(tmp_path),
        ld_library_path="",
        reference_python_path=lambda: None,
    )

    output = reference.HfTransformersReference().run_stage(case, stage, ctx)

    script = captured["command"][2]
    assert f"revision = {PINNED_REVISION!r}" in script
    assert "target_sample_rate = int(asr.feature_extractor.sampling_rate)" in script
    assert "resample_poly(" in script
    assert output.metadata["revision"] == PINNED_REVISION


def test_reference_profile_pins_transformers_with_tdt_support() -> None:
    from tensorrt_model_connect.python_profiles import (
        default_execution_profiles,
        load_python_profile_registry,
    )

    profiles = load_python_profile_registry()["profiles"]

    assert default_execution_profiles(family="parakeet_tdt")["reference"] == (
        "parakeet_tdt_reference"
    )
    spec = profiles["parakeet_tdt_reference"]
    requirements = (FAMILY_ROOT.parents[1] / str(spec["requirements"])).read_text(
        encoding="utf-8"
    )
    assert "librosa==0.11.0" in requirements.splitlines()
    assert "scipy==1.18.1" in requirements.splitlines()
    assert "transformers==5.9.0" in requirements.splitlines()


def test_release_reference_uses_the_tdt_auto_model_contract(monkeypatch) -> None:
    runner = runpy.run_path(
        str(REPO_ROOT / "benchmarks/performance/baselines/task_reference.py")
    )
    captured: dict[str, object] = {}

    class FakeTensor:
        def is_floating_point(self):
            return True

        def to(self, *args, **kwargs):
            captured.setdefault("input_to", []).append((args, kwargs))
            return self

    class FakeSequences:
        def __getitem__(self, _index):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [4, 8, 15]

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["processor_load"] = (model, kwargs)
            return cls()

        def __call__(self, audio, **kwargs):
            captured["processor_call"] = (audio.tolist(), kwargs)
            return {"input_features": FakeTensor()}

        def decode(self, sequences, **kwargs):
            captured["decode"] = (sequences, kwargs)
            return ["Parakeet transcript"]

    class FakeModel:
        config = SimpleNamespace(_commit_hash=PINNED_REVISION)

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["model_load"] = (model, kwargs)
            return cls()

        def eval(self):
            return self

        def to(self, device):
            captured["model_device"] = device
            return self

        def parameters(self):
            return iter([SimpleNamespace(dtype="fp32")])

        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return SimpleNamespace(sequences=FakeSequences())

    class WrongModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("Parakeet TDT must not use AutoModelForSpeechSeq2Seq")

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    fake_torch.bfloat16 = "bf16"
    fake_torch.device = lambda value: value
    fake_torch.inference_mode = InferenceMode
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForSpeechSeq2Seq = WrongModel
    fake_transformers.AutoModelForTDT = FakeModel
    fake_transformers.AutoProcessor = FakeProcessor
    fake_engine = ModuleType("tools.validation.engine")
    fake_engine._read_wav_float32 = lambda _path: (
        np.array([0.25], dtype=np.float32),
        16_000,
    )
    fake_engine._resample_audio = lambda audio, _source_rate, _target_rate: audio
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "tools.validation.engine", fake_engine)

    manifest = Path(__file__).with_name("manifests") / "parakeet-tdt-0.6b-v3.json"
    arguments = SimpleNamespace(
        family="parakeet_tdt",
        manifest=manifest,
        model="nvidia/parakeet-tdt-0.6b-v3",
        revision=PINNED_REVISION,
        precision="fp32",
        trust_remote_code=False,
        local_files_only=True,
    )
    session = runner["_load_asr"](
        arguments,
        {"audio_path": "data/Recording.wav", "max_new_tokens": 50},
        {"auto_model_class": "AutoModelForTDT"},
    )

    assert session.invoke() == {
        "text": "Parakeet transcript",
        "token_ids": [4, 8, 15],
        "output_tokens": 3,
    }
    assert captured["model_load"] == (
        "nvidia/parakeet-tdt-0.6b-v3",
        {
            "trust_remote_code": False,
            "local_files_only": True,
            "torch_dtype": "fp32",
            "revision": PINNED_REVISION,
        },
    )
    assert captured["generate"]["max_new_tokens"] == 50
    assert captured["generate"]["return_dict_in_generate"] is True
    assert isinstance(captured["decode"][0], FakeSequences)
    assert captured["decode"][1] == {"skip_special_tokens": True}
    assert session.resolved_revision == PINNED_REVISION


def test_validation_reference_decodes_tdt_generation_output(monkeypatch) -> None:
    from tools.reference import speech

    captured: dict[str, object] = {}

    class FakeSequence:
        def tolist(self):
            return [16, 23, 42]

    sequence = FakeSequence()

    class FakeBatch:
        def __getitem__(self, index):
            assert index == 0
            return sequence

    class FakeModel:
        def parameters(self):
            return iter([SimpleNamespace(dtype="fp32")])

        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return SimpleNamespace(sequences=FakeBatch())

    class FakeProcessor:
        feature_extractor = SimpleNamespace(sampling_rate=16_000)

        def decode(self, sequences, **kwargs):
            captured["decode"] = (sequences, kwargs)
            return ["Validation transcript"]

        def batch_decode(self, *_args, **_kwargs):
            raise AssertionError("Parakeet TDT must use its processor decode contract")

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    fake_torch = SimpleNamespace(
        inference_mode=InferenceMode,
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    monkeypatch.setattr(
        speech,
        "_load_whisper_runtime",
        lambda _arguments: (fake_torch, FakeProcessor(), FakeModel(), "cuda"),
    )
    monkeypatch.setattr(
        speech,
        "_audio_for_prompt",
        lambda _prompt, _target_rate: (
            np.array([0.25], dtype=np.float32),
            Path("input.wav"),
        ),
    )
    monkeypatch.setattr(speech, "_whisper_inputs", lambda *_args: {"input_features": "x"})

    responses = speech._run_whisper_asr(
        SimpleNamespace(
            reference_family="asr_parakeet_tdt",
            family="parakeet_tdt",
            max_new_tokens=50,
            seed=-1,
        ),
        [{"sample_id": "sample-1", "eval_index": 0, "audio": "input.wav"}],
        {},
    )

    assert responses[0]["output_text"] == "Validation transcript"
    assert responses[0]["generated_token_ids"] == [16, 23, 42]
    assert captured["generate"] == {
        "input_features": "x",
        "max_new_tokens": 50,
        "return_dict_in_generate": True,
    }
    assert captured["decode"][0] is sequence
    assert captured["decode"][1] == {"skip_special_tokens": True}


def test_exact_transcript_contract_rejects_any_normalized_difference() -> None:
    contract = importlib.import_module(
        "tests.e2e.models.parakeet_tdt.e2e_plugins.contract"
    )
    threshold = SimpleNamespace(
        metrics={
            "contract_ned_threshold": 1.0,
            "contract_wer_threshold": 1.0,
            "contract_cer_threshold": 1.0,
        }
    )
    case = SimpleNamespace(metadata={})
    reference = SimpleNamespace(data={"transcript": "the exact transcript"}, text="")
    different = SimpleNamespace(data={"transcript": "the wrong transcript"}, text="")

    result = contract.plugin.verify(different, reference, case, threshold)

    assert result.status == "failed"


def test_asr_comparator_rejects_one_sided_empty_transcript() -> None:
    comparator = importlib.import_module(
        "tests.e2e.models.parakeet_tdt.e2e_plugins.comparators.parakeet_tdt_asr"
    )
    trt = SimpleNamespace(data={"returncode": 0, "transcript": ""}, text="")
    ref = SimpleNamespace(data={"transcript": "expected speech"}, text="")
    threshold = SimpleNamespace(metrics={"wer": 0.1, "cer": 0.1})
    stage = SimpleNamespace(name="full_inference")

    result = comparator.plugin.compare(trt, ref, threshold, stage)

    assert result.status == "failed"
    assert result.metrics["wer"].value == 1.0


def test_reference_resolves_audio_from_the_owning_model_directory() -> None:
    reference = importlib.import_module(
        "tests.e2e.models.parakeet_tdt.e2e_plugins.references.parakeet_tdt_hf"
    )

    resolved = Path(reference._resolve_audio_path("data/Recording.wav"))

    assert resolved == (Path(__file__).parent / "data" / "Recording.wav").resolve()
