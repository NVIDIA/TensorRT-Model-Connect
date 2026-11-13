# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned full-image parity through the native text-to-image Task."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "p_h3_image"
TASKS = frozenset({"text_to_image"})
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
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
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


def test_native_tokenizer(case_name):
    from transformers import AutoTokenizer

    _, manifest, _ = CASES[case_name]
    directory = _model_dir(manifest)
    reference = AutoTokenizer.from_pretrained(directory / "tokenizer")
    root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    runner = root / "test_p_h3_image_tokenizer"
    for text in (
        "",
        "A watercolor painting of Berlin.",
        "Grüße 世界 😀",
        "café cafe\u0301",
        "1234567890",
        "We're ready.\t\n",
        "<|im_start|>user\nHello<|im_end|>",
    ):
        completed = subprocess.run(
            [str(runner), str(directory / "tokenizer/tokenizer.json"), text],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert json.loads(completed.stdout) == reference.encode(text, add_special_tokens=False)


def test_official_checkpoint_e2e(case_name, tmp_path):
    import gc
    import numpy as np
    import torch

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    import sys

    height, width = manifest["image_height"], manifest["image_width"]
    input_path = tmp_path / "reference-input.json"
    input_path.write_text(json.dumps([manifest, case]))
    oracle = subprocess.run(
        [
            sys.executable,
            str(TEST_ROOT / "reference.py"),
            str(directory),
            str(input_path),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=case["runtime_timeout_s"],
    )
    assert oracle.returncode == 0, oracle.stderr + oracle.stdout
    print(oracle.stdout, end="", flush=True)
    requests = json.loads((tmp_path / "reference-requests.json").read_text())
    expected = [
        np.fromfile(tmp_path / f"reference-{index}.f32", dtype="<f4").reshape(height, width, 3)
        for index in range(len(case["prompts"]))
    ]
    bundle = tmp_path / manifest["bundle"]
    _build(directory, bundle, manifest)
    gc.collect()
    torch.cuda.empty_cache()
    completed = subprocess.run(
        [str(runtime_root / "test_p_h3_image_runtime"), str(bundle), str(runtime_root)],
        input=json.dumps(requests),
        capture_output=True,
        text=True,
        check=False,
        timeout=case["runtime_timeout_s"],
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    results = json.loads(completed.stdout.splitlines()[-1])
    assert len(results) == len(expected)
    for index, (result, oracle) in enumerate(zip(results, expected)):
        assert result == dict(height=height, width=width, channels=3, count=height * width * 3)
        actual = np.fromfile(requests[index]["output"], dtype="<f4").reshape(oracle.shape)
        assert np.isfinite(actual).all()
        assert actual.min() >= 0 and actual.max() <= 1
        difference = actual.astype(np.float64) - oracle.astype(np.float64)
        rmse = float(np.sqrt(np.mean(difference**2)))
        maximum = float(np.abs(difference).max())
        a, b = actual.astype(np.float64).ravel(), oracle.astype(np.float64).ravel()
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(
            f"P-H3-Image prompt {index}: RMSE={rmse}, max_abs={maximum}, cosine={cosine}",
            flush=True,
        )
        assert rmse <= case["rmse_max"]
        assert maximum <= case["max_abs_error"]
        assert cosine >= case["cosine_min"]
