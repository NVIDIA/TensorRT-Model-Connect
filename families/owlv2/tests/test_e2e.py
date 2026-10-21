# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for owlv2."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "owlv2"
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


def _phrases(prompt: str) -> list[str]:
    return [phrase.strip() for phrase in prompt.split(".") if phrase.strip()]


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
        f"score_threshold={float(case['score_threshold'])}",
    )


def _official_reference(model_dir: Path, case: dict) -> list[dict]:
    """The reference, on the original image.

    No pre-resize here, unlike the other family on this task: OWLv2 pads to a
    square before it resizes, so the aspect ratio survives and the runtime and
    the reference can both start from the same original pixels.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Owlv2ForObjectDetection

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    phrases = _phrases(str(case["prompt"]))
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = Owlv2ForObjectDetection.from_pretrained(str(model_dir)).eval()
    encoding = processor(images=image, text=[phrases], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    # The boxes are scaled by the side of the padded square, which is the
    # longer edge of the original image, not its height and width.
    side = max(image.height, image.width)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=float(case["score_threshold"]),
        target_sizes=torch.tensor([[side, side]]),
    )[0]
    return [
        {
            "score": float(score),
            "label": phrases[int(label)],
            "box": [float(value) for value in box.tolist()],
        }
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
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


_IOU_FLOOR = 0.5
_SCORE_TOLERANCE = 0.05


def _assert_parity(actual: dict, expected: list[dict], case: dict) -> None:
    boxes = actual.get("boxes", [])
    scores = actual.get("scores", [])
    labels = actual.get("labels", [])
    assert boxes and scores and labels, f"native {FAMILY} returned no detections"
    assert len(boxes) == len(scores) * 4 == len(labels) * 4, (
        f"native {FAMILY} returned inconsistent detection arrays"
    )
    assert actual.get("box_format") == "xyxy"
    assert actual.get("coordinate_space") == "original_image_pixels"
    native = sorted(
        (
            {
                "score": float(scores[index]),
                "label": str(labels[index]),
                "box": [float(value) for value in boxes[index * 4 : index * 4 + 4]],
            }
            for index in range(len(scores))
        ),
        key=lambda item: -item["score"],
    )
    reference = sorted(expected, key=lambda item: -item["score"])
    assert reference, f"official {FAMILY} reference returned no detections"
    assert len(native) == len(reference), (
        f"detection count mismatch: native={len(native)} reference={len(reference)}"
    )

    # Every phrase the reference reported must also be a phrase the prompt
    # offered, so a label can never be an index that drifted.
    allowed = set(_phrases(str(case["prompt"])))

    # Pair on geometry, not on score order. Several patches sit within a few
    # thousandths of each other - 0.1697 and 0.1612 on this case - and fp16
    # moves them by about that much, so two detections legitimately swap rank
    # between the reference and the runtime. Zipping the two score-sorted
    # lists then compares a box against its neighbour and reports an IoU of
    # 0.26 for a model that is actually right. The label still has to agree on
    # whichever pair the geometry picked.
    remaining = list(range(len(native)))
    worst_iou = 1.0
    worst_score = 0.0
    for theirs in reference:
        best_index, best_iou = None, -1.0
        for index in remaining:
            iou = _box_iou(native[index]["box"], theirs["box"])
            if iou > best_iou:
                best_index, best_iou = index, iou
        assert best_index is not None
        mine = native[best_index]
        remaining.remove(best_index)
        assert mine["label"] in allowed, f"native label is not a prompt phrase: {mine['label']}"
        assert mine["label"] == theirs["label"], (
            f"label mismatch: native={mine['label']} reference={theirs['label']}"
        )
        worst_iou = min(worst_iou, best_iou)
        worst_score = max(worst_score, abs(mine["score"] - theirs["score"]))
    assert worst_iou >= _IOU_FLOOR, f"worst box IoU {worst_iou:.4f} below {_IOU_FLOOR}"
    assert worst_score <= _SCORE_TOLERANCE, (
        f"worst score delta {worst_score:.4f} above {_SCORE_TOLERANCE}"
    )
    print(
        f"owlv2 parity: {len(native)} detections, worst IoU {worst_iou:.4f}, "
        f"worst score delta {worst_score:.4f}"
    )


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / str(manifest["bundle"])
    _build(model_dir, bundle, manifest)
    assert bundle.is_file()

    actual = _native(binary, runtime_root, bundle, case)
    expected = _official_reference(model_dir, case)
    _assert_parity(actual, expected, case)
