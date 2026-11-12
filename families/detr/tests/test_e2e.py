# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for detr."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence, record_reference_comparison
from families.detr.tests.reporting import record_detection_views

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "detr"
TASKS = frozenset({"image_to_boxes"})
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


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    case: dict,
    command: str,
    *arguments: str,
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    return _run_json(
        binary,
        runtime_root,
        bundle,
        case,
        "detect",
        "--image",
        str(_asset(case["test_image"])),
    )


def _official_reference(model_dir: Path, case: dict) -> list[dict]:
    import torch
    from PIL import Image
    from transformers import DetrForObjectDetection, DetrImageProcessor

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    processor = DetrImageProcessor.from_pretrained(str(model_dir))
    model = DetrForObjectDetection.from_pretrained(str(model_dir)).eval()
    encoding = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    target_sizes = torch.tensor([[image.height, image.width]])
    results = processor.post_process_object_detection(
        outputs, threshold=float(case.get("score_threshold", 0.5)), target_sizes=target_sizes
    )[0]
    return [
        {
            "score": float(score),
            "label": int(label),
            "box": [float(value) for value in box.tolist()],
        }
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
    ]


def _box_iou(left: list[float], right: list[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union_area = left_area + right_area - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


def _assert_parity(actual: dict, expected: list[dict]) -> None:
    boxes = actual.get("boxes", [])
    scores = actual.get("scores", [])
    classes = actual.get("classes", [])
    assert boxes and scores and classes, "native DETR returned no detections"
    assert len(boxes) == len(scores) * 4 == len(classes) * 4, (
        "native DETR returned inconsistent detection arrays"
    )
    assert int(actual["image_height"]) > 0 and int(actual["image_width"]) > 0
    native = [
        {
            "score": float(scores[index]),
            "label": int(classes[index]),
            "box": [float(value) for value in boxes[index * 4 : index * 4 + 4]],
        }
        for index in range(len(scores))
    ]
    assert expected, "official DETR reference returned no detections"
    native = sorted(native, key=lambda item: -item["score"])
    reference = sorted(expected, key=lambda item: -item["score"])
    assert len(native) == len(reference), (
        f"detection count mismatch: native={len(native)} reference={len(reference)}"
    )
    matched = set()
    for detection in native:
        best_match = None
        best_iou = 0.0
        for index, ref_detection in enumerate(reference):
            if index in matched:
                continue
            if int(detection["label"]) != int(ref_detection["label"]):
                continue
            iou = _box_iou(detection["box"], ref_detection["box"])
            if iou > best_iou:
                best_iou = iou
                best_match = index
        assert best_match is not None, (
            f"native detection label={detection['label']} has no reference match"
        )
        assert best_iou >= 0.5, (
            f"native detection label={detection['label']} box={detection['box']} "
            f"does not overlap reference box={reference[best_match]['box']}"
        )
        assert abs(float(detection["score"]) - float(reference[best_match]["score"])) <= 0.05
        matched.add(best_match)
        record_evidence("thresholds", {"min_box_iou": 0.5, "max_score_error": 0.05})
        record_reference_comparison(
            "Matched detection",
            detection,
            reference[best_match],
            [
                {
                    "name": "box_iou",
                    "scope": "independent_reference",
                    "actual": best_iou,
                    "operator": ">=",
                    "expected": 0.5,
                    "passed": best_iou >= 0.5,
                },
                {
                    "name": "score_error",
                    "scope": "independent_reference",
                    "actual": abs(float(detection["score"]) - float(reference[best_match]["score"])),
                    "operator": "<=",
                    "expected": 0.05,
                    "passed": abs(float(detection["score"]) - float(reference[best_match]["score"]))
                    <= 0.05,
                },
            ],
        )


def _sdk_detections(runtime_root: Path, bundle: Path, image: Path, tmp_path: Path) -> list[dict]:
    """Run the public C and C++ consumers on the reference's own decoded pixels."""
    import numpy as np
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
        consumer = native_build / f"test_detr_sdk_{language}"
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
        assert result["task"] == "image_to_boxes"
        assert (result["image_height"], result["image_width"]) == pixels.shape[:2]
        outputs.append(result)
    assert outputs[0] == outputs[1], "the C and C++ SDK consumers disagree"
    return outputs


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    """Verify end-to-end detection output against official PyTorch reference and SDK consumers."""
    _, manifest, case = CASES[case_name]
    record_evidence(
        "inputs", {"manifest": manifest, "case": case, "image": TEST_ROOT / case["test_image"]}
    )
    record_evidence("thresholds", {"min_box_iou": 0.5, "max_score_error": 0.05})
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, case)
    record_evidence("native", actual)
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, case)
    record_evidence("reference", {"detections": expected})
    record_detection_views(actual, expected, _asset(case["test_image"]), tmp_path / "detection-views")
    with evidence_stage("compare"):
        _assert_parity(actual, expected)
    for sample_index, sdk in enumerate(_sdk_detections(runtime_root, bundle, _asset(case["test_image"]), tmp_path), start=1):
        record_evidence("native", sdk)
        record_detection_views(sdk, expected, _asset(case["test_image"]),
                               tmp_path / "sdk-detection-views", sample_index=sample_index)
        with evidence_stage("compare"):
            _assert_parity(sdk, expected)

    # The same bundle must accept images outside the original single fixture's
    # aspect ratio, with the official resize and unchanged parity criteria.
    from PIL import Image, ImageOps

    with Image.open(_asset(case["test_image"])) as image:
        variants = {
            "portrait": image.transpose(Image.Transpose.ROTATE_90),
            "square": ImageOps.pad(image, (max(image.size), max(image.size))),
        }
        for sample_index, (name, variant) in enumerate(variants.items(), start=3):
            path = tmp_path / f"{name}.png"
            variant.save(path)
            variant_case = {**case, "test_image": str(path)}
            record_evidence("inputs", {"variant": name, "case": variant_case, "image": path})
            with evidence_stage("compare"):
                variant_actual = record_evidence("native", _native(binary, runtime_root, bundle, variant_case))
                variant_reference = record_evidence("reference", _official_reference(model_dir, variant_case))
                record_detection_views(variant_actual, variant_reference, path,
                                       tmp_path / f"{name}-detection-views", sample_index=sample_index)
                _assert_parity(variant_actual, variant_reference)
