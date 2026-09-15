# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native orchestration contract for the experimental CosyVoice3 family."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


FAMILY = "cosyvoice3"
MANIFEST = Path(__file__).parent / "manifests" / "fun-cosyvoice3-0.5b-2512.json"


def _selected(config, case="cosyvoice3-native-contract") -> bool:
    requested = {
        item.strip()
        for value in config.getoption("--e2e-model") or []
        for item in str(value).split(",")
        if item.strip()
    }
    requested.update(
        item.strip()
        for value in config.getoption("--e2e-testcase") or []
        for item in str(value).split(",")
        if item.strip()
    )
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        requested.update(
            line.strip()
            for line in Path(models_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return bool(requested & {FAMILY, "fun-cosyvoice3-0.5b-2512", case})


def test_cosyvoice3_native_contract(request) -> None:
    from families.cosyvoice3.config import MODEL_ID, MODEL_REVISION

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["family"] == FAMILY
    assert manifest["task"] == "audio_generation"
    assert manifest["hf_id"] == MODEL_ID
    assert manifest["hf_revision"] == MODEL_REVISION
    assert {"name": "cosyvoice3-native-contract", "premerge": True} in manifest["testcases"]
    if not _selected(request.config):
        pytest.skip("direct E2E requires one of the three explicit E2E selectors")

    build_dir = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    assert build_dir, "selected CosyVoice3 E2E requires TRTMC_NATIVE_BUILD_DIR"
    binary = Path(build_dir) / "families" / FAMILY / "test_cosyvoice3_runtime"
    assert binary.is_file(), f"missing CosyVoice3 native contract executable: {binary}"
    subprocess.run([str(binary)], check=True, timeout=60)


@pytest.mark.e2e
def test_cosyvoice3_build_to_reference_audio(request, tmp_path):
    """Real standard build and native CLI smoke; not a perceptual-quality oracle."""
    if not _selected(request.config, "cosyvoice3-reference-audio"):
        pytest.skip("select cosyvoice3-reference-audio or cosyvoice3 to run checkpoint E2E")
    import numpy as np
    import soundfile as sf
    from families.cosyvoice3.config import MODEL_ID, MODEL_REVISION

    binary = os.environ.get("TRTMC_BINARY")
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    assert binary and Path(binary).is_file(), "selected E2E requires TRTMC_BINARY"
    assert runtime and Path(runtime).is_dir(), "selected E2E requires TRTMC_RUNTIME_ROOT"
    model = os.environ.get("TRTMC_COSYVOICE3_MODEL_DIR")
    if not model:
        from huggingface_hub import snapshot_download

        model = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = Path(model)
    bundle = tmp_path / "model.bundle"
    with (tmp_path / "build.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tensorrt_model_connect",
                "build",
                str(model),
                "-o",
                str(bundle),
            ],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=14400,
        )
    assert bundle.is_file()
    samples, rate = sf.read(model / "zero_shot_prompt.wav", dtype="float32")
    # Use the full published reference with the default build profile.
    reference = tmp_path / "reference.wav"
    sf.write(reference, samples, rate, subtype="FLOAT")
    output = tmp_path / "audio.wav"
    command = [
        binary,
        "generate-audio",
        str(bundle),
        "--runtime-root",
        runtime,
        "--reference-audio",
        str(reference),
        "--prompt",
        "你好，欢迎。",
        "--max-new-tokens",
        "100",
        "--seed",
        "2512",
        "--output",
        str(output),
    ]
    with (tmp_path / "runtime.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    actual, sr = sf.read(output, dtype="float32")
    assert sr == 24000 and actual.ndim == 1 and len(actual) > 0
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) > 1e-5
    assert np.max(np.abs(actual)) <= 1 and len(actual) % 960 == 0
    # A voice-independent bundle must not silently substitute a fixed voice.
    missing = command.copy()
    position = missing.index("--reference-audio")
    del missing[position : position + 2]
    missing[-1] = str(tmp_path / "missing-reference.wav")
    result = subprocess.run(missing, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert not (tmp_path / "missing-reference.wav").exists()

    # Obsolete fixed-voice bundles fail at load, before any engine is created.
    from tensorrt_model_connect.bundle_writer import BundleWriter

    obsolete = tmp_path / "obsolete.bundle"
    writer = BundleWriter(obsolete)
    writer.set_header(family=FAMILY, task="audio_generation", backend="trt")
    writer.add_json("config.json", {"cosyvoice3_schema": 1, "precision": "fp32"})
    writer.finish()
    rejected = command.copy()
    rejected[2] = str(obsolete)
    rejected[-1] = str(tmp_path / "obsolete.wav")
    result = subprocess.run(rejected, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Unsupported CosyVoice3 bundle" in result.stdout + result.stderr
    assert not (tmp_path / "obsolete.wav").exists()
