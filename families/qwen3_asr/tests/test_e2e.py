# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, SDK and official-reference E2E for Qwen3-ASR.

Each case builds the manifest's bundle, transcribes the case's WAV with the native CLI and with
the public C and C++ SDK consumers, and runs the official ``Qwen3ASRForConditionalGeneration``
greedy ``generate`` on the same waveform:

* the three native paths must return the same transcript and token ids; the C++ consumer
  transcribes a longer clip first on the same loaded model, so a request after a longer one is
  checked too;
* the native prompt must be the official processor's prompt, id for id;
* ``oracle: "exact"`` cases require the official token ids and transcript;
* ``oracle: "transcript"`` cases bound the word error rate of the native transcript against the
  official one (``max_word_error_rate``), and report how many leading token ids agree.
"""

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

FAMILY = "qwen3_asr"
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
            assert case["oracle"] in {"exact", "transcript"}
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
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root, native_build


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
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


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """PCM16 to float32 exactly as the native CLI reads it: sample / 32768, channels averaged."""
    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2, "the reference path expects 16-bit PCM"
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    samples = raw.astype(np.float32) / np.float32(32768.0)
    return samples.reshape(-1, channels).mean(axis=1, dtype=np.float32), rate


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    return env


def _json_lines(stdout: str, label: str) -> list[dict]:
    payloads = []
    for line in stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"{label} returned no JSON: {stdout[-1000:]}"
    return payloads


def _last_json(stdout: str, label: str) -> dict:
    return _json_lines(stdout, label)[-1]


def _native_cli(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    arguments = [str(binary), "transcribe", str(bundle), "--runtime-root", str(runtime_root),
                 "--input", str(_asset(case["test_input_audio"])),
                 "--max-output-tokens", str(case["max_output_tokens"])]
    if case.get("source_language"):
        arguments += ["--source-language", case["source_language"]]
    completed = subprocess.run(arguments, check=True, capture_output=True, text=True,
                               env=_environment(runtime_root), timeout=1800)
    return _last_json(completed.stdout, "native transcribe")


def _sdk(native_build: Path, runtime_root: Path, bundle: Path, case: dict, audio: np.ndarray,
         rate: int, tmp_path: Path) -> dict[str, dict]:
    pcm = tmp_path / "audio.f32"
    audio.astype("<f4").tofile(pcm)
    # A longer request first (the clip plus 10 s of silence) leaves cache rows past the case's
    # own; the case's result on the same model must not depend on them.
    longer = tmp_path / "longer.f32"
    np.concatenate([audio, np.zeros(10 * rate, dtype=np.float32)]).astype("<f4").tofile(longer)
    inputs = {"c": str(pcm), "cpp": f"{longer},{pcm}"}
    results = {}
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"missing SDK consumer {consumer}"
        arguments = [str(consumer), str(bundle), str(runtime_root), inputs[language], str(rate), "1",
                     str(case["max_output_tokens"])]
        if case.get("source_language"):
            arguments.append(case["source_language"])
        completed = subprocess.run(arguments, check=True, capture_output=True, text=True,
                                   env=_environment(runtime_root), timeout=1800)
        payloads = _json_lines(completed.stdout, f"{language} SDK consumer")
        assert len(payloads) == len(inputs[language].split(",")), f"{language} SDK result count"
        assert all(payload["text"].strip() for payload in payloads), f"{language} SDK empty result"
        results[language] = payloads[-1]
    return results


def _official_reference(model_dir: Path, case: dict, audio: np.ndarray) -> dict:
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    from families.qwen3_asr import tokens
    from families.qwen3_asr.config import load_config

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[case["reference_precision"]]
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = Qwen3ASRForConditionalGeneration.from_pretrained(str(model_dir), dtype=dtype).to("cuda")
    model.eval()
    inputs = processor.apply_transcription_request(audio, language=case.get("source_language"))
    prompt = inputs["input_ids"][0].tolist()

    # The native prompt plan, built from the same tokenizer files as the bundle.
    config = load_config(model_dir)
    tokenizer = tokens.load_tokenizer(model_dir)
    prefix, suffix = tokens.prompt_pieces(tokenizer, config.audio_token_id)
    native = prefix + [config.audio_token_id] * prompt.count(config.audio_token_id) + suffix
    if case.get("source_language"):
        language = next(entry for entry in tokens.language_suffixes(tokenizer)
                        if case["source_language"].lower() in (entry["code"], entry["name"].lower()))
        native += language["ids"]
    assert native == prompt, "the native prompt plan differs from the official processor's prompt"

    inputs = inputs.to("cuda")
    inputs["input_features"] = inputs["input_features"].to(dtype)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=int(case["max_output_tokens"]), do_sample=False
        )[0, len(prompt):].tolist()
    stop = set(config.eos_token_ids)
    if generated and generated[-1] in stop:
        generated = generated[:-1]
    text = processor.decode(generated, return_format="transcription_only")
    del model
    torch.cuda.empty_cache()
    return {"text": text, "token_ids": generated, "prompt_tokens": len(prompt)}


def _words(text: str) -> list[str]:
    return [word for word in (re.sub(r"^\W+|\W+$", "", item).lower() for item in text.split()) if word]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    expected, actual = _words(reference), _words(hypothesis)
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(expected, 1):
        current = [i] + [0] * len(actual)
        for j, other in enumerate(actual, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other))
        previous = current
    return previous[-1] / max(len(expected), 1)


def test_word_error_rate_ignores_case_and_edge_punctuation() -> None:
    assert _word_error_rate("Hello, world!", "hello world") == 0.0
    assert _word_error_rate("one two", "one three") == 0.5


def test_e2e(tmp_path: Path, case_name: str) -> None:
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root, native_build = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / str(manifest["bundle"])
    _build(model_dir, bundle, manifest)
    assert bundle.is_file(), f"{FAMILY} bundle was not produced: {bundle}"

    audio, rate = _read_wav(_asset(case["test_input_audio"]))
    native = _native_cli(binary, runtime_root, bundle, case)
    sdk = _sdk(native_build, runtime_root, bundle, case, audio, rate, tmp_path)
    for language, result in sdk.items():
        assert result["text"] == native["text"], f"{language} SDK transcript differs from the CLI"
        assert result["token_ids"] == native["token_ids"], f"{language} SDK token ids differ"

    reference = _official_reference(model_dir, case, audio)
    assert reference["text"].strip(), "the reference produced an empty transcript"
    agree = 0
    for mine, theirs in zip(native["token_ids"], reference["token_ids"]):
        if mine != theirs:
            break
        agree += 1
    error = _word_error_rate(reference["text"], native["text"])
    print(
        f"{case_name}: {len(native['token_ids'])} native / {len(reference['token_ids'])} reference "
        f"tokens, {agree} leading ids agree, WER {error:.4f}, prefill {native.get('prefill_ms')} ms, "
        f"decode {native.get('decode_ms')} ms\nnative:    {native['text']!r}\n"
        f"reference: {reference['text']!r}"
    )
    if case["oracle"] == "exact":
        assert native["token_ids"] == reference["token_ids"], "token ids differ from the reference"
        assert native["text"] == reference["text"], "transcript differs from the reference"
    else:
        assert error <= float(case["max_word_error_rate"]), f"word error rate {error:.4f}"
