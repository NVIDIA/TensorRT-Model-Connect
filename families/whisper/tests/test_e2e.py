# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for whisper."""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path
import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "whisper"
TASKS = frozenset({"transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


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
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime(manifest: dict) -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    required_gpus = int(manifest["tensor_parallel_size"])
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= required_gpus, (
        f"selected {FAMILY} E2E requires {required_gpus} GPUs, found {torch.cuda.device_count()}"
    )
    return (binary, runtime_root)


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            video_num_frames=manifest.get("video_num_frames"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    command: str,
    *arguments: str,
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    if int(manifest["tensor_parallel_size"]) > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "TRTMC_NCCL_RENDEZVOUS",
            "-np",
            str(manifest["tensor_parallel_size"]),
            *invocation,
        ]
    env = os.environ.copy()
    env["TRTMC_NCCL_RENDEZVOUS"] = str(bundle.with_suffix(".nccl-rendezvous"))
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _read_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    assert channels > 0, f"WAV has no channels: {path}"
    assert sample_width == 2, f"WAV must be 16-bit PCM: {path}"
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    assert audio.size % channels == 0, f"WAV payload is not channel aligned: {path}"
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def _distance(left: list[str], right: list[str]) -> int:
    a = left
    b = right
    previous = list(range(len(b) + 1))
    for index, char_a in enumerate(a, start=1):
        current = [index]
        for offset, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (char_a != char_b)
                )
            )
        previous = current
    return previous[-1]


def _normalized_text_edit_distance(reference: str, hypothesis: str) -> float:
    reference_text = " ".join(reference.lower().split())
    hypothesis_text = " ".join(hypothesis.lower().split())
    return _distance(list(reference_text), list(hypothesis_text)) / max(
        len(reference_text), len(hypothesis_text), 1
    )


def _wer_words(text: str) -> list[str]:
    return [
        stripped
        for word in text.split()
        if (stripped := re.sub(r"^[^\w]+|[^\w]+$", "", word).lower())
    ]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = _wer_words(reference)
    hypothesis_words = _wer_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _distance(reference_words, hypothesis_words) / len(reference_words)


def test_word_error_rate_ignores_edge_punctuation() -> None:
    assert _word_error_rate("one two", "one three") == 0.5
    assert _word_error_rate("hello, world!", "hello world") == 0.0


def _torch_dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    manifest["task"]
    audio = _asset(case["test_input_audio"])
    return _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "transcribe",
        "--input",
        str(audio),
        "--max-output-tokens",
        str(int(case["max_new_tokens"])),
    )


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    import torch
    from scipy.signal import resample
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    audio_path = _asset(case["test_input_audio"])
    audio, sample_rate = _read_pcm16_wav(audio_path)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=_torch_dtype(case["reference_precision"]),
        )
        .to(torch.device("cuda"))
        .eval()
    )
    target_rate = int(processor.feature_extractor.sampling_rate)
    if sample_rate != target_rate:
        audio = resample(audio, round(audio.size * target_rate / sample_rate)).astype(np.float32)
        sample_rate = target_rate
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    inputs = {
        key: (
            value.to(device=model.device, dtype=model_dtype)
            if value.is_floating_point()
            else value.to(device=model.device)
        )
        for key, value in inputs.items()
    }
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=int(case["max_new_tokens"]))
    return {"text": processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()}


def _assert_contract(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    del manifest, case
    reference = str(expected["text"])
    hypothesis = str(actual["text"])
    assert " ".join(reference.lower().split())
    ned_threshold = float(
        thresholds.get(
            "contract_ned_threshold",
            thresholds.get("normalized_text_edit_distance", 0.1),
        )
    )
    wer_threshold = float(thresholds.get("contract_wer_threshold", thresholds.get("wer", 0.1)))
    assert _normalized_text_edit_distance(reference, hypothesis) <= ned_threshold
    assert _word_error_rate(reference, hypothesis) <= wer_threshold


def test_reference_wav_reader_uses_pcm_without_ffmpeg() -> None:
    audio, sample_rate = _read_pcm16_wav(
        TEST_ROOT / "data/librispeech-test-clean-6930-75918-0003.wav"
    )
    assert sample_rate == 16000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert audio.size > 0


def test_official_reference_uses_local_processor_and_model(monkeypatch, tmp_path: Path) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    import torch
    import transformers

    calls = {}

    class Value:
        def __init__(self, floating: bool):
            self.floating = floating

        def is_floating_point(self):
            return self.floating

        def to(self, **kwargs):
            calls.setdefault("input_devices", []).append(kwargs)
            return self

    class Processor:
        feature_extractor = SimpleNamespace(sampling_rate=16000)

        def __call__(self, audio, **kwargs):
            calls["processor_input"] = (audio, kwargs)
            return {"input_features": Value(True), "attention_mask": Value(False)}

        def batch_decode(self, token_ids, **kwargs):
            calls["decode"] = (token_ids, kwargs)
            return [" transcript "]

    class Model:
        device = torch.device("cuda")

        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([SimpleNamespace(dtype=torch.float32)])

        def generate(self, **kwargs):
            calls["generation"] = kwargs
            return [[1, 2]]

    processor = Processor()
    model = Model()

    def load_processor(model_dir, **kwargs):
        calls["processor_load"] = (model_dir, kwargs)
        return processor

    def load_model(model_dir, **kwargs):
        calls["model_load"] = (model_dir, kwargs)
        return model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(
        transformers.AutoModelForSpeechSeq2Seq,
        "from_pretrained",
        load_model,
    )
    scipy = ModuleType("scipy")
    scipy_signal = ModuleType("scipy.signal")
    scipy_signal.resample = lambda values, _size: values
    scipy.signal = scipy_signal
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.signal", scipy_signal)
    result = _official_reference(
        tmp_path,
        {"task": "transcription"},
        {
            "test_input_audio": "data/librispeech-test-clean-6930-75918-0003.wav",
            "max_new_tokens": 17,
            "reference_precision": "fp32",
        },
        tmp_path,
    )

    assert calls["processor_load"] == (tmp_path, {"local_files_only": True})
    assert calls["model_load"] == (
        tmp_path,
        {"local_files_only": True, "torch_dtype": torch.float32},
    )
    assert calls["model_device"] == torch.device("cuda")
    assert calls["processor_input"][1] == {"sampling_rate": 16000, "return_tensors": "pt"}
    assert calls["generation"]["max_new_tokens"] == 17
    assert calls["decode"][1] == {"skip_special_tokens": True}
    assert result == {"text": "transcript"}


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_contract(actual, expected, manifest, case, _thresholds(case_name))
