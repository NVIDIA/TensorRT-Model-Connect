# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof for MoGe."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_OPERATORS = {
    "mask_iou": ">=",
    "depth_absrel_mean": "<=",
    "depth_rel_l2": "<=",
    "points_rel_l2": "<=",
    "points_cosine": ">=",
    "intrinsics_max_relative_error": "<=",
    "point_depth_consistency": "<=",
}


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "monocular_geometry", path
        assert manifest["precision"] == "fp32", path
        assert manifest["tensor_parallel_size"] == 1, path
        for case in manifest["testcases"]:
            name = case["name"]
            assert name not in cases, name
            cases[name] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values: list[str]) -> set[str]:
    return {
        item.strip()
        for value in values
        for item in str(value).split(",")
        if item.strip()
    }


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


def _required_environment() -> tuple[Path, Path, Path]:
    binary_value = os.environ.get("TRTMC_BINARY")
    runtime_value = os.environ.get("TRTMC_RUNTIME_ROOT")
    source_value = os.environ.get("TRTMC_REFERENCE_SOURCE_DIR")
    assert binary_value, "selected E2E requires TRTMC_BINARY"
    assert runtime_value, "selected E2E requires TRTMC_RUNTIME_ROOT"
    assert source_value, "selected E2E requires TRTMC_REFERENCE_SOURCE_DIR"

    binary = Path(binary_value)
    runtime_root = Path(runtime_value)
    source_root = Path(source_value)
    assert binary.is_file() and os.access(binary, os.X_OK), binary
    assert runtime_root.is_dir(), runtime_root
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), runtime_root
    assert (runtime_root / "libtrtmc_model_moge.so").is_file(), runtime_root
    assert (source_root / "moge/model/v2.py").is_file(), source_root

    import torch

    assert torch.cuda.is_available(), "selected E2E requires CUDA"
    return binary, runtime_root, source_root


def _checkpoint(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            allow_patterns=["model.pt"],
        )
    )
    assert (path / "model.pt").is_file(), path
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
        [str(binary), "inspect", str(bundle)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    assert payload["family"] == _FAMILY
    assert payload["task"] == "monocular_geometry"
    assert "engine.plan" in payload["sections"]


def _require_shape(array: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    assert array.size == int(np.prod(shape, dtype=np.int64)), (
        f"{label} has {array.size} values, expected shape {shape}"
    )
    return array.reshape(shape)


def _load_native_geometry(output_dir: Path, stdout: str) -> dict[str, np.ndarray]:
    payload = json.loads(stdout)
    height = int(payload["height"])
    width = int(payload["width"])
    assert height > 0 and width > 0
    points = _require_shape(
        np.fromfile(output_dir / "points.f32", dtype="<f4"),
        (height, width, 3),
        "points",
    )
    depth = _require_shape(
        np.fromfile(output_dir / "depth.f32", dtype="<f4"),
        (height, width),
        "depth",
    )
    mask = _require_shape(
        np.fromfile(output_dir / "mask.u8", dtype=np.uint8),
        (height, width),
        "mask",
    )
    intrinsics_payload = json.loads(
        (output_dir / "intrinsics.json").read_text(encoding="utf-8")
    )
    assert intrinsics_payload["normalized"] is True
    return {
        "points": points,
        "depth": depth,
        "mask": mask,
        "intrinsics": np.asarray(intrinsics_payload["intrinsics"], dtype=np.float32),
    }


def _run_native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    image: Path,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    completed = subprocess.run(
        [
            str(binary),
            "geometry",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--image",
            str(image),
            "--output",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return _load_native_geometry(output_dir, completed.stdout)


def _run_reference(
    source_root: Path,
    model_dir: Path,
    image: Path,
    output: Path,
    num_tokens: int,
) -> dict[str, np.ndarray]:
    completed = subprocess.run(
        [
            sys.executable,
            str(_TEST_DIR / "official_reference.py"),
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(model_dir / "model.pt"),
            "--image",
            str(image),
            "--output",
            str(output),
            "--num-tokens",
            str(num_tokens),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert json.loads(completed.stdout)["num_tokens"] == 1800
    with np.load(output, allow_pickle=False) as payload:
        return {
            name: np.array(payload[name], copy=True)
            for name in ("points", "depth", "mask", "intrinsics")
        }


def _geometry(data: dict[str, np.ndarray], label: str) -> tuple[np.ndarray, ...]:
    points = np.asarray(data["points"], dtype=np.float32)
    depth = np.asarray(data["depth"], dtype=np.float32)
    mask = np.asarray(data["mask"])
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
    assert depth.ndim == 2, label
    assert points.shape == (*depth.shape, 3), label
    assert mask.shape == depth.shape, label
    assert intrinsics.shape == (3, 3), label
    assert mask.size and np.isin(mask, (0, 1)).all() and np.any(mask), label
    valid = mask.astype(bool, copy=False)
    assert np.isfinite(points[valid]).all(), label
    assert np.isfinite(depth[valid]).all() and np.all(depth[valid] > 0.0), label
    invalid = ~valid
    if np.any(invalid):
        assert np.isposinf(points[invalid]).all(), label
        assert np.isposinf(depth[invalid]).all(), label
    assert np.isfinite(intrinsics).all(), label
    assert np.array_equal(intrinsics[2], np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    assert intrinsics[0, 0] > 0.0 and intrinsics[1, 1] > 0.0
    assert intrinsics[0, 2] == 0.5 and intrinsics[1, 2] == 0.5
    return points, depth, mask, intrinsics


def _metrics(actual: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    actual_points, actual_depth, actual_mask, actual_intrinsics = _geometry(actual, "TRT")
    ref_points, ref_depth, ref_mask, ref_intrinsics = _geometry(reference, "reference")
    assert actual_depth.shape == ref_depth.shape
    actual_valid = actual_mask.astype(bool, copy=False)
    ref_valid = ref_mask.astype(bool, copy=False)
    common = actual_valid & ref_valid
    union = actual_valid | ref_valid
    assert np.any(common)

    actual_depth_valid = actual_depth[common].astype(np.float64)
    ref_depth_valid = ref_depth[common].astype(np.float64)
    depth_delta = actual_depth_valid - ref_depth_valid
    actual_points_valid = actual_points[common].astype(np.float64)
    ref_points_valid = ref_points[common].astype(np.float64)
    point_delta = actual_points_valid - ref_points_valid
    cosine_denominator = np.maximum(
        np.linalg.norm(actual_points_valid, axis=-1)
        * np.linalg.norm(ref_points_valid, axis=-1),
        1.0e-12,
    )
    ref_intrinsics64 = ref_intrinsics.astype(np.float64)
    intrinsics_delta = actual_intrinsics.astype(np.float64) - ref_intrinsics64
    nonzero = np.abs(ref_intrinsics64) > 1.0e-12
    return {
        "mask_iou": float(common.sum() / union.sum()),
        "depth_absrel_mean": float(
            np.mean(np.abs(depth_delta) / np.maximum(np.abs(ref_depth_valid), 1.0e-12))
        ),
        "depth_rel_l2": float(
            np.linalg.norm(depth_delta) / max(float(np.linalg.norm(ref_depth_valid)), 1.0e-12)
        ),
        "points_rel_l2": float(
            np.linalg.norm(point_delta) / max(float(np.linalg.norm(ref_points_valid)), 1.0e-12)
        ),
        "points_cosine": float(
            np.mean(
                np.sum(actual_points_valid * ref_points_valid, axis=-1)
                / cosine_denominator
            )
        ),
        "intrinsics_max_relative_error": float(
            np.max(np.abs(intrinsics_delta[nonzero]) / np.abs(ref_intrinsics64[nonzero]))
        ),
        "point_depth_consistency": float(
            np.max(np.abs(actual_points[..., 2][actual_valid] - actual_depth[actual_valid]))
        ),
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
    binary, runtime_root, source_root = _required_environment()
    model_dir = _checkpoint(manifest)
    image = _TEST_DIR / case["image"]
    num_tokens = int(case["num_tokens"])
    assert image.is_file(), image
    assert num_tokens == 1800
    bundle = tmp_path / manifest["bundle"]

    _build_bundle(manifest, model_dir, bundle)
    _inspect_bundle(binary, bundle)
    actual = _run_native(binary, runtime_root, bundle, image, tmp_path / "native")
    reference = _run_reference(
        source_root,
        model_dir,
        image,
        tmp_path / "reference.npz",
        num_tokens,
    )
    thresholds = _thresholds(case_name)
    for name, value in _metrics(actual, reference).items():
        threshold = float(thresholds[name])
        assert np.isfinite(value), name
        if _OPERATORS[name] == "<=":
            assert value <= threshold, f"{name}: {value} > {threshold}"
        else:
            assert value >= threshold, f"{name}: {value} < {threshold}"
