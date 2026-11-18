# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-checkpoint native CLI / SDK proof against the official Arabic reference."""

import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import soundfile as sf
from tensorrt_model_connect import BuildRequest, build
from .reference import load_reference, transcribe

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/cohere-transcribe-arabic-fp32.json").read_text())


def _required(name):
    value = os.environ.get(name)
    assert value, f"selected Cohere E2E requires {name}"
    path = Path(value)
    assert path.exists(), path
    return path


def _json_command(command):
    completed = subprocess.run(
        [str(x) for x in command], check=True, capture_output=True, text=True, timeout=600
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]


def test_official_arabic_transcription(pytestconfig, tmp_path):
    selected = []
    for option in ["--e2e-model", "--e2e-testcase"]:
        selected.extend(
            x.strip() for raw in (pytestconfig.getoption(option) or []) for x in str(raw).split(",")
        )
    models_file = pytestconfig.getoption("--e2e-models-file")
    if models_file:
        selected.extend(Path(models_file).read_text().splitlines())
    if not set(selected) & {"cohere_asr", MANIFEST["name"]}:
        pytest.skip("real Cohere E2E requires an explicit model or testcase selection")
    if os.environ.get("TRTMC_COHERE_ASR_MODEL_DIR"):
        model_dir = _required("TRTMC_COHERE_ASR_MODEL_DIR")
    else:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
            )
        )
    binary = _required("TRTMC_BINARY")
    runtime = _required("TRTMC_RUNTIME_ROOT")
    native = _required("TRTMC_NATIVE_BUILD_DIR")
    if os.environ.get("TRTMC_COHERE_ASR_BUNDLE"):
        bundle = _required("TRTMC_COHERE_ASR_BUNDLE")
    else:
        bundle = tmp_path / MANIFEST["bundle"]
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=MANIFEST["family"],
                task=MANIFEST["task"],
                precision=MANIFEST["precision"],
                max_sequence_length=int(MANIFEST["max_sequence_length"]),
                tensor_parallel_size=int(MANIFEST["tensor_parallel_size"]),
            )
        )
    source_language = MANIFEST["testcases"][0]["source_language"]
    assert source_language == "ar", "this case declares the Arabic reference and SDK prompts"
    model, frontend, tokenizer = load_reference(model_dir)
    frontend.window.numpy().tofile(tmp_path / "window.f32")
    frontend.fb.numpy().tofile(tmp_path / "filters.f32")
    audio_paths = []
    expected = []
    evidence = {}
    for audio_file in MANIFEST["testcases"][0]["audio_files"]:
        path = model_dir / audio_file
        audio, rate = sf.read(path, dtype="float32")
        assert rate == 16000 and audio.ndim == 1
        name = path.stem
        pcm = tmp_path / (name + ".f32")
        audio.tofile(pcm)
        audio_paths.append(pcm)
        reference = transcribe(model, frontend, tokenizer, audio)
        output = tmp_path / (name + ".features.f32")
        subprocess.run(
            [
                str(native / "test_cohere_asr_frontend_consumer"),
                str(pcm),
                str(tmp_path / "window.f32"),
                str(tmp_path / "filters.f32"),
                str(output),
            ],
            check=True,
            timeout=60,
        )
        features = np.fromfile(output, dtype=np.float32).reshape(reference["features"].shape)
        difference = features - reference["features"]
        relative = float(np.linalg.norm(difference) / np.linalg.norm(reference["features"]))
        maximum = float(np.max(np.abs(difference)))
        assert relative < 1e-4 and maximum < 1e-3, (relative, maximum)
        actual = _json_command(
            [
                binary,
                "transcribe",
                bundle,
                "--runtime-root",
                runtime,
                "--input",
                path,
                "--source-language",
                source_language,
            ]
        )[0]
        assert actual["token_ids"] == reference["token_ids"], (actual, reference["token_ids"])
        assert actual["text"] == reference["text"], (actual, reference["text"])
        expected.append({"text": reference["text"], "token_ids": reference["token_ids"]})
        evidence[name] = {
            "text": actual["text"],
            "token_ids": actual["token_ids"],
            "frontend_relative_l2": relative,
            "frontend_max_abs": maximum,
            "exact_reference_tokens": True,
        }
    for language in ["c", "cpp"]:
        results = _json_command(
            [native / f"test_cohere_asr_sdk_{language}", bundle, runtime, *audio_paths]
        )
        if language == "cpp":
            assert results[-1] == results[0], "loaded Task changed output after another request"
            results = results[:-1]
        assert len(results) == len(expected)
        for actual, wanted in zip(results, expected):
            assert bytes.fromhex(actual["text_hex"]).decode("utf-8") == wanted["text"]
            assert actual["token_ids"] == wanted["token_ids"]
    (tmp_path / "parity.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
