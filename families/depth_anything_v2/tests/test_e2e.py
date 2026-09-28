# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof for Depth Anything V2.

Unlike MoGe, this checkpoint is a standard `transformers`
`DepthAnythingForDepthEstimation` model, so the reference run below calls
`transformers` directly in-process rather than a vendored upstream repository.
The engine this family builds fixes the input to one square resolution (see
`model.py`); the reference preprocessing below matches that exactly so the
comparison is apples to apples, not a comparison against the (aspect-ratio
preserving) default `DPTImageProcessor` output.
"""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_OPERATORS = {
    "depth_mae_normalized": "<=",
    "depth_rel_l2": "<=",
    "depth_pearson_correlation": ">=",
}


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "monocular_geometry", path
        assert manifest["tensor_parallel_size"] == 1, path
        for case in manifest["testcases"]:
            name = case["name"]
            assert name not in cases, name
            cases[name] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def _selection(config) -> set[str]:
    selected = _csv_values(config.getoption("--e2e-model", default=[]) or [])
    selected |= _csv_values(config.getoption("--e2e-testcase", default=[]) or [])
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        path = Path(models_file)
        assert path.is_file(), f"E2E models file does not exist: {path}"
        selected |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return selected


def _require_selected(case_name: str, manifest: dict, config) -> None:
    selected = _selection(config)
    if os.environ.get("TRTMC_E2E") != "1" and not selected:
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit E2E selection")
    if selected and not ({_FAMILY, manifest["name"], case_name} & selected):
        pytest.skip(f"{case_name} was not selected")


def _required_environment() -> tuple[Path, Path]:
    binary_value = os.environ.get("TRTMC_BINARY")
    runtime_value = os.environ.get("TRTMC_RUNTIME_ROOT")
    assert binary_value, "selected E2E requires TRTMC_BINARY"
    assert runtime_value, "selected E2E requires TRTMC_RUNTIME_ROOT"

    binary = Path(binary_value)
    runtime_root = Path(runtime_value)
    assert binary.is_file() and os.access(binary, os.X_OK), binary
    assert runtime_root.is_dir(), runtime_root
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), runtime_root
    assert (runtime_root / f"libtrtmc_model_{_FAMILY}.so").is_file(), runtime_root

    import torch

    assert torch.cuda.is_available(), "selected E2E requires CUDA"
    return binary, runtime_root


def _checkpoint(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"]))
    assert (path / "config.json").is_file(), path
    assert (path / "model.safetensors").is_file(), path
    return path


def _build_bundle(manifest: dict, model_dir: Path, bundle: Path) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=_FAMILY,
            task="monocular_geometry",
            precision=manifest["precision"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
        )
    )
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle


def _inspect_bundle(binary: Path, bundle: Path) -> None:
    completed = subprocess.run(
        [str(binary), "inspect", str(bundle)], check=True, capture_output=True, text=True, timeout=30
    )
    record_evidence(
        "native_process", {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr}
    )
    payload = json.loads(completed.stdout)
    assert payload["family"] == _FAMILY
    assert payload["task"] == "monocular_geometry"
    assert "engine.plan" in payload["sections"]


def _run_native(binary: Path, runtime_root: Path, bundle: Path, image: Path, output_dir: Path) -> np.ndarray:
    completed = subprocess.run(
        [
            str(binary), "geometry", str(bundle),
            "--runtime-root", str(runtime_root),
            "--image", str(image),
            "--output", str(output_dir),
        ],
        check=True, capture_output=True, text=True, timeout=1800,
    )
    record_evidence(
        "native_process", {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr}
    )
    payload = json.loads(completed.stdout)
    height, width = int(payload["height"]), int(payload["width"])
    assert height > 0 and width > 0
    depth = np.fromfile(output_dir / "depth.f32", dtype="<f4")
    assert depth.size == height * width
    return depth.reshape(height, width)


def _reference_depth(model_dir: Path, image: Path, image_size: int) -> np.ndarray:
    """Run the real `transformers` model with the same fixed-square resize this family's engine uses."""
    import torch
    from PIL import Image
    from transformers import AutoModelForDepthEstimation
    import torchvision.transforms.functional as functional_transforms

    model = AutoModelForDepthEstimation.from_pretrained(model_dir)
    model.eval()

    source = Image.open(image).convert("RGB")
    resized = functional_transforms.resize(
        source, [image_size, image_size],
        interpolation=functional_transforms.InterpolationMode.BILINEAR, antialias=True,
    )
    array = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    pixel_values = torch.from_numpy(array.transpose(2, 0, 1)[None]).float()

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    return outputs.predicted_depth[0].numpy().astype(np.float32)


def _metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Depth-map error metrics that stay meaningful when much of the frame is
    background at (or near) zero relative depth.

    A per-pixel relative error (`|delta| / |reference|`) blows up on this
    output: Depth Anything's "relative" depth is ReLU-clipped, so roughly a
    quarter of a typical frame's reference pixels sit near zero, and dividing
    a tiny absolute difference by a near-zero reference produces a huge ratio
    that reflects the metric's denominator, not the model. MAE normalized by
    the reference's own dynamic range, the global L2 ratio, and correlation
    all stay well-behaved under that distribution.
    """
    assert actual.shape == reference.shape
    actual64 = actual.astype(np.float64).ravel()
    reference64 = reference.astype(np.float64).ravel()
    delta = actual64 - reference64
    depth_range = max(float(reference64.max() - reference64.min()), 1.0e-6)
    correlation = float(np.corrcoef(actual64, reference64)[0, 1])
    return {
        "depth_mae_normalized": float(np.mean(np.abs(delta)) / depth_range),
        "depth_rel_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(reference64)), 1.0e-12)),
        "depth_pearson_correlation": correlation,
    }


def _thresholds(case_name: str) -> dict[str, float]:
    path = _TEST_DIR / "thresholds" / f"{case_name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload["threshold_overrides"]
    assert set(thresholds) == set(_OPERATORS)
    return thresholds


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config)
    record_evidence("inputs", {"manifest": manifest, "case": case})
    binary, runtime_root = _required_environment()
    model_dir = _checkpoint(manifest)
    record_evidence(
        "checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"), "hf_revision": manifest.get("hf_revision")}
    )
    image = _TEST_DIR / case["test_image"]
    record_evidence("inputs", {"image": image})
    with evidence_stage("compare"):
        assert image.is_file(), image
    bundle = tmp_path / manifest["bundle"]

    with evidence_stage("build"):
        _build_bundle(manifest, model_dir, bundle)
    with evidence_stage("inspect"):
        _inspect_bundle(binary, bundle)
    with evidence_stage("native"):
        actual = _run_native(binary, runtime_root, bundle, image, tmp_path / "native")
    record_evidence("native", {"shape": actual.shape})
    image_size = actual.shape[0]
    assert actual.shape == (image_size, image_size)
    with evidence_stage("reference"):
        reference = _reference_depth(model_dir, image, image_size)
    record_evidence("reference", {"shape": reference.shape})
    thresholds = record_evidence("thresholds", _thresholds(case_name))
    for name, value in _metrics(actual, reference).items():
        threshold = float(thresholds[name])
        with evidence_stage("compare"):
            assert np.isfinite(value), name
        if _OPERATORS[name] == "<=":
            with evidence_stage("compare"):
                assert value <= threshold, f"{name}: {value} > {threshold}"
        else:
            with evidence_stage("compare"):
                assert value >= threshold, f"{name}: {value} < {threshold}"
