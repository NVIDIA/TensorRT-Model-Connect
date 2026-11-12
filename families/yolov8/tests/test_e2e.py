# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime, and PyTorch reference proof for YOLOv8."""

from __future__ import annotations

from tools.e2e_evidence import record_evidence, record_reference_comparison
from families.yolov8.tests.reporting import record_detection_views

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "yolov8"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == "image_to_boxes"
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
    assert result
    return result


CASES = _cases()


def _selection(config) -> set[str]:
    selected = set()
    for option in ("--e2e-model", "--e2e-testcase"):
        for raw in config.getoption(option, default=[]) or []:
            selected.update(value.strip() for value in str(raw).split(",") if value.strip())
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        selected.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return selected


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selection(metafunc.config)
    names = [
        name
        for name, (manifest, _) in CASES.items()
        if not selected or selected & {FAMILY, name, manifest["name"]}
    ]
    if not selected:
        names = [
            pytest.param(
                name,
                marks=pytest.mark.skip(reason="real YOLOv8 E2E requires explicit selection"),
                id=name,
            )
            for name in names
        ]
    metafunc.parametrize("case_name", names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected YOLOv8 E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected YOLOv8 E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_YOLOV8_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_YOLOV8_MODEL_DIR")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=True,
        )
    )


def _asset(case: dict) -> Path:
    path = TEST_ROOT / str(case["test_image"])
    assert path.is_file(), f"selected YOLOv8 E2E image is missing: {path}"
    return path


def _assert_matches_reference(
    actual: dict, kept: np.ndarray, scale: float, pad_x: float, pad_y: float
) -> None:
    """The original native-vs-PyTorch oracle, applied unchanged to every path."""
    record_evidence("native", actual)
    assert len(kept), "the reference found nothing above the threshold to compare against"
    assert len(actual["scores"]) == len(kept), (
        f"engine returned {len(actual['scores'])} detections, reference {len(kept)}"
    )
    record_evidence("reference", {"detections_letterbox": kept, "scale": scale, "pad_x": pad_x, "pad_y": pad_y})
    comparison_checks = []
    for index, row in enumerate(kept):
        x0, y0, x1, y1, score, label = row
        assert int(actual["classes"][index]) == int(label)
        assert abs(float(actual["scores"][index]) - float(score)) < 5e-3
        # The engine reports source-image pixels, so the reference box has to
        # come back out of the letterbox before they can be compared.
        expected = (
            (float(x0) - pad_x) / scale,
            (float(y0) - pad_y) / scale,
            (float(x1) - pad_x) / scale,
            (float(y1) - pad_y) / scale,
        )
        for axis, want in enumerate(expected):
            # Each path writes the boxes as one flat array, four per detection.
            assert abs(float(actual["boxes"][index * 4 + axis]) - want) < 1.0
        comparison_checks.extend([
            {"name": f"class_{index}", "scope": "independent_reference",
             "actual": int(actual["classes"][index]), "operator": "==", "expected": int(label), "passed": True},
            {"name": f"score_error_{index}", "scope": "independent_reference",
             "actual": abs(float(actual["scores"][index]) - float(score)), "operator": "<", "expected": 5e-3, "passed": True},
            {"name": f"box_error_{index}", "scope": "independent_reference",
             "actual": max(abs(float(actual["boxes"][index * 4 + axis]) - want) for axis, want in enumerate(expected)),
             "operator": "<", "expected": 1.0, "passed": True},
        ])
    record_reference_comparison(
        "Detections against the independent reference", actual,
        {"detections_letterbox": kept, "scale": scale, "pad_x": pad_x, "pad_y": pad_y},
        comparison_checks,
    )


def _sdk_detections(runtime_root: Path, bundle: Path, image: Path, tmp_path: Path) -> list[dict]:
    """Run the public C and C++ consumers on the reference's own decoded pixels."""
    record_evidence("inputs", {"operation": "sdk", "sdk_language": None, "image": image})
    record_evidence("native", None)
    record_evidence("reference", None)
    record_evidence("views", [])
    record_evidence("detection_view_summary", None)
    from PIL import Image

    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    pixels = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) / np.float32(255.0)
    raw = tmp_path / "sdk-input.rgb.f32"
    np.ascontiguousarray(pixels).tofile(raw)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    outputs = []
    for language in ("c", "cpp"):
        record_evidence("inputs", {"sdk_language": language})
        record_evidence("native", None)
        record_evidence("reference", None)
        record_evidence("views", [])
        record_evidence("detection_view_summary", None)
        consumer = native_build / f"test_yolov8_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        completed = subprocess.run(
            [
                str(consumer),
                str(bundle),
                str(runtime_root),
                str(raw),
                str(pixels.shape[0]),
                str(pixels.shape[1]),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        payloads = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        assert len(payloads) == 1, f"the {language} consumer must print one result"
        result = json.loads(payloads[0])
        record_evidence("native", result)
        assert result["task"] == "image_to_boxes"
        assert (result["image_height"], result["image_width"]) == pixels.shape[:2]
        outputs.append(result)
    # Same pixels, same bundle, same runtime: the two public bindings agree exactly.
    assert outputs[0] == outputs[1], "the C and C++ SDK consumers disagree"
    return outputs


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence("checkpoint", {"hf_id": manifest["hf_id"], "hf_revision": manifest.get("hf_revision")})
    record_evidence("thresholds", {"class_ids": "exact equality", "score_absolute_difference_strict_max": 5e-3, "box_coordinate_difference_strict_max_px": 1.0})
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )
    image = _asset(case)
    record_evidence("inputs", {"image": image})
    completed = subprocess.run(
        [
            str(binary),
            "detect",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--image",
            str(image),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    actual = json.loads(completed.stdout)
    record_evidence("native", actual)

    reference_output = tmp_path / "reference.npz"
    subprocess.run(
        [sys.executable, str(TEST_ROOT / "reference/e2e.py"),
         str(model_dir), str(image), str(reference_output)],
        check=True, timeout=600,
        env={**os.environ, "PYTHONPATH": str(TEST_ROOT.parents[2]) + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    with np.load(reference_output, allow_pickle=False) as result:
        kept = result["kept"]
        scale, pad_x, pad_y = (float(result[name]) for name in ("scale", "pad_x", "pad_y"))
    record_detection_views(actual, kept, image, tmp_path / "detection-views",
                           scale=scale, pad_x=pad_x, pad_y=pad_y)
    _assert_matches_reference(actual, kept, scale, pad_x, pad_y)
    # The public SDK reads the same pixels the reference decoded, so it is held to
    # the same oracle; the CLI's own JPEG decoder is not claimed bitwise equal.
    for sample_index, sdk in enumerate(_sdk_detections(runtime_root, bundle, image, tmp_path), start=1):
        language = ("c", "cpp")[sample_index - 1]
        record_evidence("inputs", {"operation": "sdk", "sdk_language": language})
        record_evidence("native", sdk)
        record_evidence("views", [])
        record_evidence("detection_view_summary", None)
        record_evidence("reference", {"detections_letterbox": kept, "scale": scale, "pad_x": pad_x, "pad_y": pad_y})
        record_detection_views(sdk, kept, image, tmp_path / "sdk-detection-views",
                               scale=scale, pad_x=pad_x, pad_y=pad_y, sample_index=sample_index,
                               operation="SDK C" if language == "c" else "SDK C++")
        _assert_matches_reference(sdk, kept, scale, pad_x, pad_y)
