# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for depth_anything."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "depth_anything"
TASKS = frozenset({"monocular_depth"})
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


def _asset(name: str) -> Path:
    path = TEST_ROOT / "data" / str(name)
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
            max_sequence_length=manifest["max_sequence_length"],
        )
    )


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict):
    import numpy as np

    invocation = [str(binary), "depth", str(bundle), "--runtime-root", str(runtime_root),
                  "--image", str(_asset(case["test_image"]))]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation, check=True, capture_output=True, text=True, env=env, timeout=3600
    )
    payload = None
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            payload = json.loads(line[start:])
    assert payload is not None, f"native {FAMILY} returned no JSON: {completed.stdout[-1000:]}"
    depth = np.asarray(payload["depth"], dtype=np.float64)
    assert depth.size == int(payload["height"]) * int(payload["width"]), (
        f"native {FAMILY} depth map does not cover the image"
    )
    return depth, int(payload["height"]), int(payload["width"])


def _official_reference(model_dir: Path, case: dict, height: int, width: int):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    dtype = torch.float32 if case["reference_precision"] == "fp32" else torch.float16
    model = AutoModelForDepthEstimation.from_pretrained(str(model_dir), dtype=dtype).eval()
    size = int(model.config.backbone_config.image_size)
    # The engine builds the checkpoint's native square and resizes bilinearly, so
    # the reference is configured the same way. Left to its own defaults this
    # processor keeps the aspect ratio, producing a size the engine cannot accept.
    encoding = processor(
        images=image, return_tensors="pt", size={"height": size, "width": size},
        keep_aspect_ratio=False, resample=2,
    )
    with torch.no_grad():
        outputs = model(**encoding)
    resampled = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1).float(), size=(height, width), mode="nearest"
    )
    return resampled.squeeze().numpy().astype(np.float64).ravel()


def _assert_parity(native, reference, case: dict) -> None:
    import numpy as np

    assert native.shape == reference.shape, (
        f"depth map shape mismatch: native={native.shape} reference={reference.shape}"
    )
    assert np.isfinite(native).all(), f"native {FAMILY} depth map is not finite"
    correlation = float(np.corrcoef(native, reference)[0, 1])
    scale = float(reference.max()) if float(reference.max()) > 0.0 else 1.0
    relative_error = float(np.abs(native - reference).mean() / scale)
    assert correlation >= float(case["min_depth_correlation"]), (
        f"depth correlation {correlation:.6f} is below {case['min_depth_correlation']}"
    )
    assert relative_error <= float(case["max_depth_relative_error"]), (
        f"depth relative error {relative_error:.6f} exceeds {case['max_depth_relative_error']}"
    )


def test_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    assert manifest["trust_remote_code"] is False
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]

    _build(model_dir, bundle, manifest)
    native, height, width = _native(binary, runtime_root, bundle, case)
    reference = _official_reference(model_dir, case, height, width)
    _assert_parity(native, reference, case)
