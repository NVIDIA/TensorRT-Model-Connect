# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned reference parity through the public speech transcription SDK."""

import json
import math
import os
from pathlib import Path
import subprocess
import wave

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "parakeet_tdt"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"

def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            (
                line.strip()
                for line in Path(models_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and (not line.lstrip().startswith("#"))
            )
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    if not model_filters and (not testcase_filters):
        return (sorted(CASES), False)
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or (manifest["name"] in model_filters)
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return (sorted(selected), True)


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _audio(case, generated_root):
    path = TEST_ROOT / case["test_input_audio"]
    if not path.is_file():
        from families.parakeet_tdt.tests.data.asr_probes.generate_asr_probe_inputs import main
        path = generated_root / path.name
        if not path.is_file():
            main(generated_root)
    with wave.open(str(path), "rb") as source:
        assert source.getsampwidth() == 2, "Parakeet fixtures must use PCM16"
        rate, channels = source.getframerate(), source.getnchannels()
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return audio.astype(np.float32) / 32768.0, rate, channels


def _assert_parity(actual, expected):
    hypothesis = " ".join(actual["text"].split()).lower()
    reference = " ".join(expected["text"].split()).lower()
    assert reference, "official reference produced an empty transcript"
    assert hypothesis == reference, f"transcript mismatch: {hypothesis!r} != {reference!r}"


def _reference(model_dir, audio, rate, channels, max_new_tokens):
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModelForTDT, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTDT.from_pretrained(
        model_dir, dtype=torch.float32, local_files_only=True,
    ).cpu().eval()
    mono = audio.reshape(-1, channels).mean(axis=1)
    target = int(processor.feature_extractor.sampling_rate)
    if rate != target:
        divisor = math.gcd(rate, target)
        mono = resample_poly(mono, target // divisor, rate // divisor).astype(np.float32)
    inputs = processor(mono, sampling_rate=target, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
    decoded = processor.decode(output.sequences, skip_special_tokens=True)
    return {"text": decoded[0] if isinstance(decoded, (list, tuple)) else str(decoded)}


def _sdk_consumer_binary() -> Path:
    native = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(native), "--parallel", "8", "--target", "test_parakeet_tdt_sdk_cpp"],
        check=True,
        timeout=600,
    )
    consumer = native / "test_parakeet_tdt_sdk_cpp"
    assert consumer.is_file(), "native build did not produce the family-owned SDK consumer"
    return consumer


def test_official_checkpoint_e2e(case_name, tmp_path):
    _, manifest, case = CASES[case_name]
    runtime = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    consumer = _sdk_consumer_binary()
    assert (runtime / "libtrtmc_model_parakeet_tdt.so").is_file()
    assert (runtime / "libtrtmc_backend_trt.so").is_file()
    import torch
    assert torch.cuda.is_available(), "selected Parakeet E2E requires CUDA"
    model_dir = _model_dir(manifest)
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_revision": manifest["hf_revision"]})
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(model_dir=model_dir, output_path=bundle, family=FAMILY,
                           task=manifest["task"], precision=manifest["precision"],
                           tensor_parallel_size=int(manifest["tensor_parallel_size"])))
    audio, rate, channels = _audio(case, tmp_path / "audio-probes")
    pcm = tmp_path / "input.f32"
    audio.astype("<f4").tofile(pcm)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(runtime) + ":" + env.get("LD_LIBRARY_PATH", "")
    with evidence_stage("native"):
        result = subprocess.run([str(consumer), str(bundle), str(runtime), str(pcm),
                                 str(rate), str(channels), str(case["max_new_tokens"])],
                                check=True, capture_output=True, text=True, env=env, timeout=1800)
        record_evidence("commands", {"argv": result.args})
        record_evidence("native_process", {"stdout": result.stdout, "stderr": result.stderr})
        outputs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        assert len(outputs) == 1, "SDK consumer must return exactly one transcript"
        actual = outputs[0]
    with evidence_stage("reference"):
        expected = _reference(model_dir, audio, rate, channels, int(case["max_new_tokens"]))
    record_evidence("native", actual)
    record_evidence("reference", expected)
    with evidence_stage("compare"):
        _assert_parity(actual, expected)


def test_exact_transcript_contract():
    _assert_parity({"text": "Hello  WORLD"}, {"text": "hello world"})
    for actual, expected in [("", "hello"), ("", ""), ("hello", "hello there")]:
        with pytest.raises(AssertionError):
            _assert_parity({"text": actual}, {"text": expected})


def test_all_audio_cases_have_valid_pcm(tmp_path):
    for _, _, case in CASES.values():
        audio, rate, channels = _audio(case, tmp_path / "audio-probes")
        assert audio.size and audio.size % channels == 0
        assert np.isfinite(audio).all()
        assert audio.size / channels / rate <= 30


def test_reference_uses_tdt_generation_contract(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys

    class Processor:
        feature_extractor = SimpleNamespace(sampling_rate=16000)

        @classmethod
        def from_pretrained(cls, path, **options):
            assert path == tmp_path and options == {"local_files_only": True}
            return cls()

        def __call__(self, audio, **options):
            assert audio.shape == (4,)
            assert options == {"sampling_rate": 16000, "return_tensors": "pt"}
            return {"input_features": "features"}

        def decode(self, sequences, **options):
            assert sequences == [[1, 2]] and options == {"skip_special_tokens": True}
            return ["test transcript"]

    class Model:
        @classmethod
        def from_pretrained(cls, path, **options):
            assert path == tmp_path
            assert options == {"dtype": "fp32", "local_files_only": True}
            return cls()

        def cpu(self):
            return self

        def eval(self):
            return self

        def generate(self, **options):
            assert options == {"input_features": "features", "max_new_tokens": 50,
                               "return_dict_in_generate": True}
            return SimpleNamespace(sequences=[[1, 2]])

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="fp32", inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForTDT=Model, AutoProcessor=Processor))
    monkeypatch.setitem(sys.modules, "scipy.signal", SimpleNamespace(resample_poly=None))
    assert _reference(tmp_path, np.zeros(8, dtype=np.float32), 16000, 2, 50) == {"text": "test transcript"}


def test_missing_probes_are_generated_outside_source_tree(monkeypatch, tmp_path):
    from families.parakeet_tdt.tests.data.asr_probes import generate_asr_probe_inputs as probes

    source_root = tmp_path / "read-only-source"
    source_root.mkdir()
    monkeypatch.setattr(probes, "ROOT", source_root)
    monkeypatch.setitem(globals(), "TEST_ROOT", source_root)
    output = tmp_path / "generated"
    for _, _, case in CASES.values():
        if "asr_probes" in case["test_input_audio"]:
            audio, rate, channels = _audio(case, output)
            assert audio.size > 0 and rate > 0 and channels > 0
    assert list(source_root.iterdir()) == []
    assert len(list(output.glob("*.wav"))) == 7
