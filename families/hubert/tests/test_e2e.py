# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for hubert."""

from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "hubert"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
# The files the builder and the official reference read; the checkpoint's
# duplicate TensorFlow weights are not needed.
CHECKPOINT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "pytorch_model.bin",
)


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
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return sorted(selected), True


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
            repo_id=manifest["hf_id"],
            revision=manifest.get("hf_revision"),
            local_files_only=True,
            allow_patterns=list(CHECKPOINT_FILES),
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= 1, f"selected {FAMILY} E2E requires one GPU"
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    # The builder reads the official single-file pytorch_model.bin in place,
    # with weights-only loading; the cached checkpoint is not modified.
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _run_json(binary: Path, runtime_root: Path, bundle: Path, case: dict, command: str,
              *arguments: str) -> dict:
    invocation = [str(binary), command, str(bundle), "--runtime-root", str(runtime_root),
                  *arguments]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
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
    assert len(payloads) == 1, f"native {command} returned {len(payloads)} JSON payloads: " + (
        completed.stdout[-1000:])
    return payloads[0]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    return _run_json(binary, runtime_root, bundle, case, "transcribe", "--input",
                     str(_asset(case["test_input_audio"])))


def _sdk_consumer() -> Path:
    """The family-owned direct C++ SDK consumer, built in the native build tree."""
    native = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    target = f"test_{FAMILY}_sdk_cpp"
    subprocess.run(["cmake", "--build", str(native), "--parallel", "8", "--target", target],
                   check=True, timeout=600)
    consumer = native / target
    assert consumer.is_file(), "native build did not produce the family-owned SDK consumer"
    return consumer


def _sdk(consumer: Path, runtime_root: Path, bundle: Path, case: dict, scratch: Path) -> dict:
    """The same samples through the public C++ SpeechTranscription Task."""
    waveform, rate = _read_pcm16(_asset(case["test_input_audio"]))
    pcm = scratch / "input.f32"
    waveform.astype("<f4").tofile(pcm)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [str(consumer), str(bundle), str(runtime_root), str(rate), "1", str(pcm)],
        check=True, capture_output=True, text=True, env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    outputs = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(outputs) == 1, "the SDK consumer must return exactly one transcript"
    return outputs[0]


def _read_pcm16(path: Path):
    """The waveform the native CLI reads: 16-bit PCM scaled by 1/32768, channels averaged."""
    import numpy as np

    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2, "the reference path expects 16-bit PCM"
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return raw.astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768.0, rate


def _official_reference(model_dir: Path, case: dict) -> tuple[str, list[int]]:
    """Transformers' HubertForCTC and processor, in fp32 on the CPU, fed the same samples.

    The assets are mono at the checkpoint's rate, so nothing resamples; the
    processor's only work is the normalisation the runtime performs itself.
    """
    import torch
    from transformers import AutoProcessor, HubertForCTC

    waveform, rate = _read_pcm16(_asset(case["test_input_audio"]))
    processor = AutoProcessor.from_pretrained(str(model_dir))
    assert rate == processor.feature_extractor.sampling_rate
    model = HubertForCTC.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    values = processor(waveform, sampling_rate=rate, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(values).logits[0]

    predicted = logits.argmax(dim=-1).tolist()
    blank = model.config.pad_token_id
    collapsed: list[int] = []
    previous = None
    for token_id in predicted:
        if token_id != previous and token_id != blank:
            collapsed.append(int(token_id))
        previous = token_id
    return processor.decode(predicted), collapsed


def _assert_parity(native: dict, reference_text: str, reference_ids: list[int]) -> None:
    assert reference_text, "the reference produced an empty transcript, so the gate proves nothing"
    # Exact on both the transcript and the CTC ids behind it. This model
    # degrades gently: a positional convolution trimmed on the wrong side or
    # built without its weight normalisation still reads as plausible English,
    # so a tolerance wide enough for float noise would also pass those errors.
    text = str(native["text"])
    assert text == reference_text, f"transcript mismatch: native={text!r} reference={reference_text!r}"
    token_ids = [int(value) for value in native["token_ids"]]
    assert token_ids == reference_ids, (
        f"token id mismatch: native has {len(token_ids)} ids, reference {len(reference_ids)}"
    )
    print(f"hubert parity: {len(token_ids)} CTC tokens, transcript {text!r}")


def test_e2e(tmp_path: Path, case_name: str) -> None:
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / str(manifest["bundle"])
    _build(model_dir, bundle, manifest)
    assert bundle.is_file(), f"{FAMILY} bundle was not produced: {bundle}"

    native = _native(binary, runtime_root, bundle, case)
    sdk = _sdk(_sdk_consumer(), runtime_root, bundle, case, tmp_path)
    reference_text, reference_ids = _official_reference(model_dir, case)
    _assert_parity(native, reference_text, reference_ids)
    _assert_parity(sdk, reference_text, reference_ids)
