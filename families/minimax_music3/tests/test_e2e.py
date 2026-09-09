# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build and native-runtime E2E for MiniMax-Music3."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build


FAMILY = "minimax_music3"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = TEST_ROOT / "manifests" / "minimax-music3-l0.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _selected(config) -> bool:
    requested = set()
    for raw in config.getoption("--e2e-model") or []:
        requested.update(item.strip() for item in str(raw).split(",") if item.strip())
    cases = set()
    for raw in config.getoption("--e2e-testcase") or []:
        cases.update(item.strip() for item in str(raw).split(",") if item.strip())
    return FAMILY in requested or MANIFEST["name"] in requested or bool(
        cases.intersection(case["name"] for case in MANIFEST["testcases"])
    )


def _required_env(name: str) -> Path:
    value = os.environ.get(name)
    assert value, f"selected {FAMILY} E2E requires {name}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E path does not exist: {path}"
    return path


def _model_dir() -> Path:
    explicit = os.environ.get("TRTMC_MINIMAX_MUSIC3_MODEL_DIR")
    if explicit:
        return _required_env("TRTMC_MINIMAX_MUSIC3_MODEL_DIR")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=MANIFEST["hf_id"],
            revision=MANIFEST["hf_revision"],
            local_files_only=True,
        )
    )


def _read_float_wav(path: Path) -> tuple[np.ndarray, int, int]:
    import soundfile as sf

    info = sf.info(path)
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    assert info.format == "WAV"
    assert samples.shape[1] == 2
    assert np.isfinite(samples).all()
    return samples, int(sample_rate), int(info.channels)


@pytest.mark.gpu
def test_minimax_music3_native_audio(case_name, request, tmp_path: Path) -> None:
    if not _selected(request.config):
        pytest.skip("direct E2E requires an explicit MiniMax-Music3 selector")
    case = next(case for case in MANIFEST["testcases"] if case["name"] == case_name)
    binary = _required_env("TRTMC_BINARY")
    runtime_root = _required_env("TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_model_minimax_music3.so").is_file()

    bundle = tmp_path / MANIFEST["bundle"]
    build(
        BuildRequest(
            model_dir=_model_dir(),
            output_path=bundle,
            family=FAMILY,
            task="audio_generation",
            precision=MANIFEST["precision"],
            max_sequence_length=MANIFEST["max_sequence_length"],
        )
    )
    output = tmp_path / "native.wav"
    command = [
        str(binary),
        "generate-audio",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        case["prompt"],
        "--description",
        case["description"],
        "--max-new-tokens",
        str(case["max_new_tokens"]),
        "--seed",
        str(case["seed"]),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if part
    )
    subprocess.run(command, check=True, env=environment, timeout=3600)

    samples, sample_rate, channels = _read_float_wav(output)
    thresholds = json.loads(
        (TEST_ROOT / "thresholds" / f"{case_name}.json").read_text(encoding="utf-8")
    )["threshold_overrides"]
    duration = samples.shape[0] / sample_rate
    rms = float(np.sqrt(np.mean(samples**2)))
    assert channels == 2
    assert sample_rate == int(thresholds["sampling_rate"])
    assert float(thresholds["contract_min_duration_s"]) <= duration
    assert duration <= float(thresholds["contract_max_duration_s"])
    assert rms >= float(thresholds["contract_min_rms"])


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        metafunc.parametrize(
            "case_name", [case["name"] for case in MANIFEST["testcases"]]
        )
