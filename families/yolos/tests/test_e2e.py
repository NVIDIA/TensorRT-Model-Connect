# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for yolos."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "yolos"
TASKS = frozenset({"object_detection"})
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


def _run_json(binary: Path, runtime_root: Path, bundle: Path, case: dict, *arguments) -> dict:
    invocation = [str(binary), *arguments[:1], str(bundle), "--runtime-root", str(runtime_root),
                  *arguments[1:]]
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
    return payload


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> list[dict]:
    payload = _run_json(
        binary, runtime_root, bundle, case, "detect", "--image", str(_asset(case["test_image"]))
    )
    boxes = payload["boxes"]
    scores = payload["scores"]
    classes = payload["classes"]
    assert len(boxes) == 4 * len(scores) and len(scores) == len(classes), (
        f"native {FAMILY} returned inconsistent detection arrays"
    )
    return [
        {
            "box": [float(value) for value in boxes[index * 4 : (index + 1) * 4]],
            "score": float(scores[index]),
            "label": int(classes[index]),
        }
        for index in range(len(scores))
    ]


def _official_reference(model_dir: Path, manifest: dict, case: dict) -> list[dict]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    dtype = torch.float32 if case["reference_precision"] == "fp32" else torch.float16
    model = AutoModelForObjectDetection.from_pretrained(str(model_dir), dtype=dtype).eval()
    # This family builds the checkpoint's native resolution and does not
    # interpolate position embeddings, so the reference is fed that same fixed
    # size rather than the processor's aspect-preserving default.
    height, width = model.config.image_size
    encoding = processor(images=image, return_tensors="pt",
                         size={"height": int(height), "width": int(width)})
    with torch.no_grad():
        outputs = model(**encoding)
    results = processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=[(image.height, image.width)]
    )[0]
    return [
        {"box": [float(value) for value in box], "score": float(score), "label": int(label)}
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"])
    ]


def _box_iou(left: list[float], right: list[float]) -> float:
    x_min = max(left[0], right[0])
    y_min = max(left[1], right[1])
    x_max = min(left[2], right[2])
    y_max = min(left[3], right[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _assert_parity(native: list[dict], reference: list[dict], case: dict) -> None:
    assert native, f"native {FAMILY} returned no detections"
    assert reference, f"official {FAMILY} reference returned no detections"
    assert len(native) == len(reference), (
        f"detection count mismatch: native={len(native)} reference={len(reference)}"
    )
    minimum_iou = float(case["min_box_iou"])
    maximum_score_difference = float(case["max_score_abs_diff"])
    for detection in native:
        candidates = [item for item in reference if item["label"] == detection["label"]]
        assert candidates, (
            f"native detection label={detection['label']} has no reference match"
        )
        best = max(candidates, key=lambda item: _box_iou(detection["box"], item["box"]))
        overlap = _box_iou(detection["box"], best["box"])
        assert overlap >= minimum_iou, (
            f"native detection label={detection['label']} box={detection['box']} "
            f"overlaps its reference by {overlap:.4f}"
        )
        assert abs(detection["score"] - best["score"]) <= maximum_score_difference, (
            f"native detection label={detection['label']} score={detection['score']:.4f} "
            f"differs from reference {best['score']:.4f}"
        )


def test_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    assert manifest["trust_remote_code"] is False
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / manifest["bundle"]

    _build(model_dir, bundle, manifest)
    native = _native(binary, runtime_root, bundle, case)
    reference = _official_reference(model_dir, manifest, case)
    _assert_parity(native, reference, case)
