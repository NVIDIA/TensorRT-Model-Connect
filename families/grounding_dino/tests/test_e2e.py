# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for grounding_dino."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "grounding_dino"
TASKS = frozenset({"image_text_to_boxes"})
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
        "--prompt",
        str(case["prompt"]),
        "--set",
        f"box_threshold={float(case['box_threshold'])}",
        "--set",
        f"text_threshold={float(case['text_threshold'])}",
    )


def _square(image, manifest: dict):
    """The engine's fixed input, applied before the reference preprocessor.

    This family does not preserve the aspect ratio: shifted-window attention
    needs a static partition, so the bundle is built for one square size and
    the runtime resizes to it. The reference has to see the same pixels or the
    comparison measures the resize rather than the model.
    """
    from PIL import Image

    return image.resize(
        (int(manifest["image_width"]), int(manifest["image_height"])), Image.BILINEAR
    )


def _official_reference(model_dir: Path, manifest: dict, case: dict) -> list[dict]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    resized = _square(image, manifest)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = GroundingDinoForObjectDetection.from_pretrained(str(model_dir)).eval()
    encoding = processor(
        images=resized, text=str(case["prompt"]), return_tensors="pt", do_resize=False
    )
    with torch.no_grad():
        outputs = model(**encoding)
    results = processor.post_process_grounded_object_detection(
        outputs,
        encoding["input_ids"],
        threshold=float(case["box_threshold"]),
        text_threshold=float(case["text_threshold"]),
        target_sizes=[(image.height, image.width)],
    )[0]
    return [
        {
            "score": float(score),
            "label": str(label),
            "box": [float(value) for value in box.tolist()],
        }
        for score, label, box in zip(results["scores"], results["text_labels"], results["boxes"])
    ]


def _box_iou(left: list[float], right: list[float]) -> float:
    inter_x1 = max(left[0], right[0])
    inter_y1 = max(left[1], right[1])
    inter_x2 = min(left[2], right[2])
    inter_y2 = min(left[3], right[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union_area = left_area + right_area - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


def _assert_parity(actual: dict, expected: list[dict]) -> None:
    boxes = actual.get("boxes", [])
    scores = actual.get("scores", [])
    labels = actual.get("labels", [])
    assert boxes and scores and labels, f"native {FAMILY} returned no detections"
    assert len(boxes) == len(scores) * 4 == len(labels) * 4, (
        f"native {FAMILY} returned inconsistent detection arrays"
    )
    assert actual.get("box_format") == "xyxy"
    assert actual.get("coordinate_space") == "original_image_pixels"
    assert int(actual["image_height"]) > 0 and int(actual["image_width"]) > 0
    native = [
        {
            "score": float(scores[index]),
            "label": str(labels[index]),
            "box": [float(value) for value in boxes[index * 4 : index * 4 + 4]],
        }
        for index in range(len(scores))
    ]
    assert expected, f"official {FAMILY} reference returned no detections"
    native = sorted(native, key=lambda item: -item["score"])
    reference = sorted(expected, key=lambda item: -item["score"])
    assert len(native) == len(reference), (
        f"detection count mismatch: native={len(native)} reference={len(reference)}"
    )

    # Matched on geometry, not on identity, because two-stage query selection
    # is tie-sensitive: the top-900 cut sits on score gaps of about 1e-6, so a
    # low-scoring query can legitimately swap between runs. The label is then
    # required to agree on whatever pair the geometry picked.
    matched: set[int] = set()
    worst_iou = 1.0
    worst_score = 0.0
    for detection in native:
        best_match = None
        best_iou = 0.0
        for index, candidate in enumerate(reference):
            if index in matched:
                continue
            iou = _box_iou(detection["box"], candidate["box"])
            if iou > best_iou:
                best_iou = iou
                best_match = index
        assert best_match is not None, (
            f"native detection {detection['label']} has no reference match"
        )
        assert best_iou >= 0.5, (
            f"native box={detection['box']} does not overlap "
            f"reference box={reference[best_match]['box']}"
        )
        assert detection["label"].strip() == reference[best_match]["label"].strip(), (
            f"phrase mismatch: native={detection['label']!r} "
            f"reference={reference[best_match]['label']!r}"
        )
        assert abs(detection["score"] - reference[best_match]["score"]) <= 0.05
        worst_iou = min(worst_iou, best_iou)
        worst_score = max(worst_score, abs(detection["score"] - reference[best_match]["score"]))
        matched.add(best_match)
    print(
        f"{FAMILY} parity: {len(native)} detections, worst IoU {worst_iou:.4f}, "
        f"worst score delta {worst_score:.4f}"
    )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, case)
    expected = _official_reference(model_dir, manifest, case)
    _assert_parity(actual, expected)
