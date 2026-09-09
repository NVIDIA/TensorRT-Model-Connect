# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for glmasr."""

from __future__ import annotations
import json
import os
import re
import subprocess
import wave
from pathlib import Path
import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "glmasr"
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
    del manifest
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
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
        )
    )


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
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
    env = os.environ.copy()
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


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    audio = _asset(case["test_input_audio"])
    return _run_json(
        binary,
        runtime_root,
        bundle,
        case,
        "transcribe",
        "--input",
        str(audio),
        "--max-output-tokens",
        str(int(case["max_new_tokens"])),
    )


def _official_reference(model_dir: Path, case: dict) -> dict:
    import torch
    from transformers import GlmAsrForConditionalGeneration, GlmAsrProcessor

    audio_path = _asset(case["test_input_audio"])
    audio, sample_rate = _read_pcm16_wav(audio_path)
    processor = GlmAsrProcessor.from_pretrained(model_dir, local_files_only=True)
    model = (
        GlmAsrForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=_torch_dtype(case["reference_precision"]),
        )
        .to(torch.device("cuda"))
        .eval()
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Please transcribe this audio into text"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        sampling_rate=sample_rate,
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=int(case["max_new_tokens"]), do_sample=False
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {"text": text}


def _assert_contract(actual: dict, expected: dict, thresholds: dict) -> None:
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


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    # The prompt carries one audio placeholder per post-merge encoder frame,
    # so its length grows with the clip: the 23.3 s librispeech sample needs
    # 291 placeholders and a 306 token prompt. max_sequence_length is 384
    # because the decoder engine fails to build at 512 with a TensorRT Myelin
    # internal error, and max_new_tokens is 72 so the case stays inside that
    # cache (306 + 72 = 378) instead of relying on an early end-of-sequence.
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, case)
    expected = _official_reference(model_dir, case)
    _assert_contract(actual, expected, _thresholds(case_name))
