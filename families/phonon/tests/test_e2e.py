# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, public native Task execution and official Phonon reference."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence
from .engine_probe import execute, section

ROOT = Path(__file__).resolve().parent
FAMILY = "phonon"
CASES = {}
for path in (ROOT / "manifests").glob("*.json"):
    manifest = json.loads(path.read_text())
    assert manifest["family"] == FAMILY
    assert manifest["task"] == "text_to_speech"
    for case in manifest["testcases"]:
        CASES[case["name"]] = (manifest, case)


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    models, cases = set(), set()
    for option, target in (("--e2e-model", models), ("--e2e-testcase", cases)):
        for raw in metafunc.config.getoption(option) or []:
            target.update(str(raw).split(","))
    file = metafunc.config.getoption("--e2e-models-file")
    if file:
        models.update(
            line.strip()
            for line in Path(file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    selected = []
    for name, (manifest, _) in CASES.items():
        if models and not ({FAMILY, name, manifest["name"]} & models):
            continue
        if cases and name not in cases:
            continue
        selected.append(
            name
            if models or cases
            else pytest.param(name, marks=pytest.mark.skip(reason="select this GPU E2E explicitly"))
        )
    metafunc.parametrize("case_name", selected)


def _checkpoint(manifest):
    explicit = os.environ.get("TRTMC_PHONON_MODEL_DIR")
    if explicit:
        root = Path(explicit)
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
            )
        )
    assert (root / "model.safetensors").is_file()
    return root.resolve()


def _waveform(path):
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    offset, audio, format_ = 12, None, None
    while offset + 8 <= len(data):
        kind, size = struct.unpack_from("<4sI", data, offset)
        payload = data[offset + 8 : offset + 8 + size]
        if kind == b"fmt ":
            format_ = struct.unpack_from("<HHIIHH", payload)
        elif kind == b"data":
            audio = payload
        offset += 8 + size + size % 2
    assert format_ is not None and format_[0:3] == (3, 1, 24000) and format_[-1] == 32
    assert audio
    return np.frombuffer(audio, dtype="<f4")


def test_direct_e2e(case_name, tmp_path):
    manifest, case = CASES[case_name]
    record_evidence("manifest", manifest)
    model_dir = _checkpoint(manifest)
    binary = Path(os.environ["TRTMC_BINARY"]).resolve()
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"]).resolve()
    assert binary.is_file() and (runtime / "libtrtmc_model_phonon.so").is_file()
    assert (runtime / "libtrtmc_backend_trt.so").is_file()
    text = case["inputs"]["text"]
    limits = {
        "backbone_atol": 1e-4,
        "flow_atol": 1e-4,
        "codec_rmse": 1e-4,
        "waveform_rmse": 0.01,
        "waveform_correlation": 0.98,
    }
    threshold = ROOT / "thresholds" / (case_name + ".json")
    if threshold.is_file():
        limits.update(json.loads(threshold.read_text())["threshold_overrides"])
    record_evidence("thresholds", limits)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                max_sequence_length=manifest["max_sequence_length"],
                tensor_parallel_size=manifest["tensor_parallel_size"],
            )
        )
    with evidence_stage("reference"):
        reference = tmp_path / "official.npz"
        interpreter = os.environ.get("TRTMC_PHONON_REFERENCE_PYTHON", sys.executable)
        subprocess.run(
            [
                interpreter,
                str(ROOT / "reference.py"),
                "--model-dir",
                str(model_dir),
                "--text",
                text,
                "--output",
                str(reference),
            ],
            check=True,
        )
        ref = np.load(reference)
    with evidence_stage("engine_comparison"):
        first = execute(section(bundle, "backbone.plan"), {"embeddings": ref["embeddings"]})
        for name in ("condition", "eos"):
            error = float(np.max(np.abs(first[name] - ref[name])))
            record_evidence("backbone_" + name + "_max_error", error)
            assert error <= limits["backbone_atol"]
        sample = execute(
            section(bundle, "flow.plan"), {"noise": ref["noise"], "condition": ref["condition"]}
        )
        error = float(np.max(np.abs(sample["latent"] - ref["latent"])))
        record_evidence("flow_max_error", error)
        assert error <= limits["flow_atol"]
        codec = execute(section(bundle, "codec.plan"), {"latents": ref["normalized_latents"]})[
            "pcm"
        ]
        error = float(np.sqrt(np.mean((codec - ref["pcm"]) ** 2)))
        record_evidence("codec_rmse", error)
        assert error <= limits["codec_rmse"]
    with evidence_stage("native"):
        audio = tmp_path / "native.wav"
        native = subprocess.run(
            [
                str(binary),
                "generate-audio",
                str(bundle),
                "--runtime-root",
                str(runtime),
                "--prompt",
                text,
                "--language",
                "en",
                "--set",
                "temperature=0",
                "--output",
                str(audio),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        result = json.loads(native.stdout)
        record_evidence("native_result", result)
        actual, expected = _waveform(audio), ref["pcm"].reshape(-1)
        assert actual.shape == expected.shape
        assert np.isfinite(actual).all() and np.max(np.abs(actual)) > 0.01
        assert result["num_samples"] == actual.size and result["sample_rate"] == 24000
        rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
        correlation = float(np.corrcoef(actual, expected)[0, 1])
        record_evidence("waveform_rmse", rmse)
        record_evidence("waveform_correlation", correlation)
        record_evidence("native_waveform", audio)
        assert rmse <= limits["waveform_rmse"]
        assert correlation >= limits["waveform_correlation"]
        # The selected reference must terminate through EOS, rather than its frame cap.
        assert np.any(ref["eos_trace"] > 4.0)
    with evidence_stage("sdk"):
        native_build = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"])
        encoded = subprocess.run(
            [str(native_build / "test_phonon_tokenizer"), str(model_dir / "tokenizer.json"), text],
            check=True,
            text=True,
            capture_output=True,
        )
        assert [int(item) for item in encoded.stdout.split()] == ref["token_ids"].tolist()
        for language in ("c", "cpp"):
            subprocess.run(
                [str(native_build / ("test_phonon_sdk_" + language)), str(bundle), str(runtime)],
                check=True,
            )
