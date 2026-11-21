# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for detr."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "rt_detr"
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


def _model_dir(manifest: dict) -> tuple[Path, str | None]:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR"), None
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot), manifest["hf_revision"]


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


def _native(binary: Path, runtime_root: Path, bundle: Path, case: dict, tmp_path: Path) -> dict:
    from PIL import Image

    # The Task consumes RGB pixels. Decode the source fixture once with the
    # reference decoder, then transport those exact pixels through the CLI
    # losslessly; STB and Pillow JPEG decoding can differ by several levels.
    image_path = tmp_path / "native-input.png"
    Image.open(_asset(case["test_image"])).convert("RGB").save(image_path)
    return _run_json(
        binary,
        runtime_root,
        bundle,
        case,
        "detect",
        "--image",
        str(image_path),
    )


def _official_reference(model_dir: Path, case: dict) -> list[dict]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, RTDetrForObjectDetection

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = RTDetrForObjectDetection.from_pretrained(str(model_dir)).eval()
    encoding = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    target_sizes = torch.tensor([[image.height, image.width]])
    results = processor.post_process_object_detection(
        outputs, threshold=float(case["score_threshold"]), target_sizes=target_sizes
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


def _assert_parity(actual: dict, expected: list[dict]) -> list[dict]:
    boxes = actual.get("boxes", [])
    scores = actual.get("scores", [])
    classes = actual.get("classes", [])
    assert boxes and scores and classes, "native rt_detr returned no detections"
    assert len(boxes) == len(scores) * 4 == len(classes) * 4, (
        "native rt_detr returned inconsistent detection arrays"
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
    assert expected, "official rt_detr reference returned no detections"
    native = sorted(native, key=lambda item: -item["score"])
    reference = sorted(expected, key=lambda item: -item["score"])
    assert len(native) == len(reference), (
        f"detection count mismatch: native={len(native)} reference={len(reference)}"
    )
    matched = set()
    measurements = []
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
        measurements.append(
            {
                "native": detection,
                "reference": reference[best_match],
                "iou": best_iou,
                "score_error": abs(
                    float(detection["score"]) - float(reference[best_match]["score"])
                ),
            }
        )
        matched.add(best_match)
    return measurements


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    if evidence_enabled():
        for key in ("image", "test_image", "audio"):
            if key in case:
                for source in (TEST_ROOT / case[key], TEST_ROOT / "data" / case[key]):
                    if source.is_file():
                        record_evidence("inputs", {key: source})
                        break
    record_evidence(
        "checkpoint",
        {
            "hf_id": manifest["hf_id"],
            "expected_hf_revision": manifest["hf_revision"],
            "hf_revision": None,
            "revision_status": "unresolved",
            "precision": manifest["precision"],
        },
    )
    record_evidence(
        "thresholds",
        {
            key: value
            for key, value in case.items()
            if any(word in key for word in ("error", "cosine", "atol", "rtol", "snr", "rmse"))
        },
    )
    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        model_dir, checkpoint_revision = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_revision": checkpoint_revision,
            "revision_status": "unverified" if checkpoint_revision is None else "pinned_cache_lookup",
        },
    )
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    record_evidence(
        "thresholds",
        {"box_iou_min": 0.5, "score_error_max": 0.05, "score_threshold": case["score_threshold"]},
    )
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, case, tmp_path)
    record_evidence("native", actual)
    record_evidence("inputs", {"image": tmp_path / "native-input.png"})
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, case)
    record_evidence("reference", {"mode": "hf_transformers", "detections": expected})
    with evidence_stage("compare"):
        measurements = _assert_parity(actual, expected)
    record_evidence("metrics", measurements)
    if evidence_enabled():
        from PIL import Image, ImageDraw

        records = {
            "native": [
                {"box": actual["boxes"][4 * i : 4 * i + 4], "label": label}
                for i, label in enumerate(actual["classes"])
            ],
            "reference": expected,
        }
        views, pair = [], {}
        for role, detections in records.items():
            raw = tmp_path / f"{role}-detections.json"
            raw.write_text(json.dumps(actual if role == "native" else expected, indent=2))
            pair[role] = raw
            image = Image.open(tmp_path / "native-input.png").convert("RGB")
            draw = ImageDraw.Draw(image)
            for detection in detections:
                draw.rectangle(detection["box"], outline="red", width=2)
                draw.text(tuple(detection["box"][:2]), str(detection["label"]), fill="red")
            preview = tmp_path / f"{role}-detections.png"
            image.save(preview)
            views.append({"title": f"Sample 0 / {role}", "image": preview})
        record_evidence("views", views)
        minimum = min(item["iou"] for item in measurements)
        maximum = max(item["score_error"] for item in measurements)
        record_evidence(
            "reference_comparison",
            {
                "label": "Matched detection boxes and scores",
                "scope": "independent_reference",
                "enforced": True,
                **pair,
                "checks": [
                    {
                        "name": "box_iou",
                        "scope": "independent_reference",
                        "actual": minimum,
                        "operator": ">=",
                        "expected": 0.5,
                        "passed": minimum >= 0.5,
                    },
                    {
                        "name": "score_error",
                        "scope": "independent_reference",
                        "actual": maximum,
                        "operator": "<=",
                        "expected": 0.05,
                        "passed": maximum <= 0.05,
                    },
                ],
            },
        )
