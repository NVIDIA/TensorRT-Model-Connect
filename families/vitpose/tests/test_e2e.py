# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI and SDK runs, and the Transformers reference for ViTPose.

The reference is the checkpoint's own processor (`AutoProcessor`, the fast VitPose processor in Transformers 5.2.0)
and `VitPoseForPoseEstimation` in float32 on the GPU with TF32 disabled, decoded by `post_process_pose_estimation`
with the same person box. The test image is lossless, so the CLI's image loader, the C and C++ SDK consumers and the
reference all see the same 8-bit pixels. Every request of a case is held to the same bounds on every native path.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, mark_loaded, record_evidence

FAMILY = "vitpose"
TASK = "image_text_to_points"
QUERY = "person"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == TASK
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
    assert result
    return result


CASES = _cases()


def _values(raw_values) -> set[str]:
    return {
        value.strip() for raw in raw_values or [] for value in str(raw).split(",") if value.strip()
    }


def _selection(config) -> tuple[set[str], set[str]]:
    models = _values(config.getoption("--e2e-model", default=[]))
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        models.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return models, _values(config.getoption("--e2e-testcase", default=[]))


def _selected_cases(config) -> list[str] | None:
    """Return the selected cases, or None when no E2E selector is given.

    A case is selected when it matches the model filters (`--e2e-model`, `--e2e-models-file`: the family, a manifest
    or a case name) and the testcase filters (`--e2e-testcase`); a missing kind of filter matches every case. A
    selection that names this family, one of its manifests or one of its cases but matches no case is a usage error,
    so that a misspelt name cannot pass without testing anything; a selection of other families selects nothing here
    (`test_a_testcase_selection_matches_a_collected_e2e_case` checks it against the whole session).
    """

    models, testcases = _selection(config)
    if not models and not testcases:
        return None
    selected = [
        name
        for name, (manifest, _) in CASES.items()
        if (not models or models & {FAMILY, name, manifest["name"]})
        and (not testcases or name in testcases)
    ]
    own_names = {FAMILY, *CASES, *(manifest["name"] for manifest, _ in CASES.values())}
    if not selected and (models | testcases) & own_names:
        raise pytest.UsageError(
            f"the E2E selection names {FAMILY} but matches none of its testcases "
            f"(models: {sorted(models)}, testcases: {sorted(testcases)}; available: {sorted(CASES)})"
        )
    return selected


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selected_cases(metafunc.config)
    if selected is None:
        selected = [
            pytest.param(
                name,
                marks=pytest.mark.skip(reason="the real ViTPose E2E requires explicit selection"),
                id=name,
            )
            for name in CASES
        ]
    metafunc.parametrize("case_name", selected)


def _collected_cases(items) -> set[str]:
    """The E2E cases of every family collected in this session: the `case_name` parameter of each `test_*e2e` item."""

    cases = set()
    for item in items:
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        if (
            re.fullmatch(r"test_.*e2e", getattr(item, "originalname", "") or "")
            and "case_name" in parameters
        ):
            cases.add(str(parameters["case_name"]))
    return cases


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected ViTPose E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected ViTPose E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    # The pinned snapshot comes from the local cache only: the premerge runs offline with a cache that holds just
    # the manifests' snapshots.
    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
        )
    )


def _asset(case: dict) -> Path:
    path = TEST_ROOT / str(case["test_image"])
    assert path.is_file(), f"selected ViTPose E2E image is missing: {path}"
    return path


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected ViTPose E2E requires its thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _checkpoint_config(model_dir: Path) -> tuple[list[str], list[tuple[int, int]]]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    labels = [config["id2label"][str(index)] for index in range(len(config["id2label"]))]
    return labels, [(int(start), int(end)) for start, end in config["edges"]]


def _coco_box(box, image) -> list[float]:
    """The reference processor's COCO box for a request: the person box, or the whole image."""

    if box is None:
        return [0.0, 0.0, float(image.width), float(image.height)]
    x_min, y_min, x_max, y_max = (float(value) for value in box)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _points(payload: dict, labels: list[str]) -> dict:
    """A native result as label -> (x, y, score), checked for shape and order before any comparison."""

    assert payload["raw_response"] == "" and payload["parse_complete"] is True, payload
    names, scores, coordinates = payload["labels"], payload["scores"], payload["points"]
    assert len(coordinates) == 2 * len(names) and len(scores) == len(names), payload
    assert len(set(names)) == len(names) and [name for name in labels if name in names] == names, (
        names
    )
    points = {
        name: (
            float(coordinates[2 * index]),
            float(coordinates[2 * index + 1]),
            float(scores[index]),
        )
        for index, name in enumerate(names)
    }
    assert np.isfinite(list(points.values())).all(), points
    return points


def _native_cli(
    binary: Path, runtime_root: Path, bundle: Path, image: Path, box, labels: list[str]
) -> dict:
    command = [
        str(binary),
        "detect",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--image",
        str(image),
        "--prompt",
        QUERY,
    ]
    if box is not None:
        command += ["--set", "person_box=" + json.dumps([float(value) for value in box])]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    record_evidence(
        "native_process", {"command": shlex.join(completed.args), "stderr": completed.stderr}
    )
    payload = json.loads(completed.stdout)
    assert payload["coordinate_space"] == "original_image_pixels", payload
    return _points(payload, labels)


def _native_sdk(
    native_build: Path,
    runtime_root: Path,
    bundle: Path,
    pixels: np.ndarray,
    box,
    labels: list[str],
    tmp_path: Path,
) -> dict:
    raw_image = tmp_path / "sdk-input.rgb.f32"
    (pixels.astype(np.float32) / np.float32(255.0)).tofile(raw_image)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    payloads = []
    for language in ("c", "cpp"):
        consumer = native_build / f"test_vitpose_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        command = [
            str(consumer),
            str(bundle),
            str(runtime_root),
            str(raw_image),
            *map(str, pixels.shape[:2]),
        ]
        if box is not None:
            command += [repr(float(value)) for value in box]
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, env=env, timeout=900
        )
        payload = json.loads(completed.stdout)
        assert payload["task"] == TASK
        record_evidence(f"sdk_{language}", payload)
        payloads.append(payload)
    assert payloads[0] == payloads[1], "the C and C++ SDK consumers must return the same keypoints"
    return _points(payloads[0], labels)


def _reference(model_dir: Path, image, boxes: list) -> list[dict]:
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    labels, _ = _checkpoint_config(model_dir)
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    results = []
    try:
        # The checkpoint and the engine are float32; TF32 would round every reference product to 10 bits.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.get_float32_matmul_precision() == "highest"
        processor = AutoProcessor.from_pretrained(model_dir)
        model = (
            VitPoseForPoseEstimation.from_pretrained(model_dir, dtype=torch.float32)
            .to("cuda")
            .eval()
        )
        for box in boxes:
            coco = [_coco_box(box, image)]
            inputs = processor(image, boxes=[coco], return_tensors="pt")
            with torch.no_grad():
                outputs = model(pixel_values=inputs["pixel_values"].to("cuda"))
            pose = processor.post_process_pose_estimation(outputs, boxes=[coco])[0][0]
            keypoints, scores = pose["keypoints"].numpy(), pose["scores"].numpy()
            assert keypoints.shape == (len(labels), 2) and scores.shape == (len(labels),)
            assert np.isfinite(keypoints).all() and np.isfinite(scores).all(), (keypoints, scores)
            # The reference marks a keypoint whose heatmap has no positive value -1; the family leaves it out.
            results.append(
                {
                    labels[index]: (
                        float(keypoints[index, 0]),
                        float(keypoints[index, 1]),
                        float(scores[index]),
                    )
                    for index in range(len(labels))
                    if scores[index] > 0.0
                }
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
    return results


def _agreement(actual: dict, expected: dict) -> dict:
    """The located keypoints of both sides, and the largest distance and score difference over them."""

    measured = {"located_keypoints": len(actual), "reference_keypoints": len(expected)}
    if list(actual) != list(expected):
        measured.update(max_keypoint_error_px=float("inf"), max_score_error=float("inf"))
        return measured
    distances = [
        np.hypot(actual[name][0] - expected[name][0], actual[name][1] - expected[name][1])
        for name in actual
    ]
    score_errors = [abs(actual[name][2] - expected[name][2]) for name in actual]
    # numpy's max propagates a NaN, where Python's max() keeps the first value a NaN does not beat.
    measured["max_keypoint_error_px"] = float(np.max(distances, initial=0.0))
    measured["max_score_error"] = float(np.max(score_errors, initial=0.0))
    return measured


_CHECKS = {
    "max_keypoint_error_px": ("Largest keypoint distance (pixels)", "<="),
    "max_score_error": ("Largest heatmap score difference", "<="),
}
_PATH_LABELS = {"cli": "CLI", "sdk": "C and C++ SDK"}


def _comparison(
    label: str, native: Path, reference: Path, measured: dict, thresholds: dict
) -> dict:
    """One `reference_comparison` record: both keypoint sets and every bound the test asserts."""

    checks = [
        {
            "name": "reference_keypoints",
            "label": "Keypoints the reference locates",
            "scope": "independent_reference",
            "actual": measured["reference_keypoints"],
            "operator": ">=",
            "expected": 1,
            "passed": bool(measured["reference_keypoints"] >= 1),
        },
        {
            "name": "located_keypoints",
            "label": "Located keypoints (native = reference)",
            "scope": "independent_reference",
            "actual": measured["located_keypoints"],
            "operator": "==",
            "expected": measured["reference_keypoints"],
            "passed": bool(measured["located_keypoints"] == measured["reference_keypoints"]),
        },
    ]
    for name, (text, operator) in _CHECKS.items():
        actual, expected = float(measured[name]), float(thresholds[name])
        checks.append(
            {
                "name": name,
                "label": text,
                "scope": "independent_reference",
                "actual": actual,
                "operator": operator,
                "expected": expected,
                "passed": bool(actual <= expected),
            }
        )
    return {
        "label": label,
        "scope": "independent_reference",
        "enforced": True,
        "native": native,
        "reference": reference,
        "checks": checks,
    }


def _assert_agreement(measured: dict, thresholds: dict) -> None:
    # Bounds (tests/thresholds/<case>.json): 0.005 px and 1e-4. Measured on one B300 (TensorRT 11.4, torch 2.15) over 3
    # fresh bundles x 2 requests x the CLI and the SDK (12 comparisons): every keypoint within 6.2e-5 px of the
    # reference and every score within 6.6e-7. One defect at a time injected into the reference moves the keypoints and
    # scores by at least 0.0126 px and 3.5e-4 (a tanh GELU; position row 0 not added), 5 px (no DARK step; a box
    # padding of 1.0) or 15 px (corner-aligned upsampling); a layer-norm epsilon of 1e-6 stays within both bounds.
    # Two empty keypoint sets would agree perfectly: the fixture's person must be located.
    assert measured["reference_keypoints"] >= 1, measured
    assert measured["located_keypoints"] == measured["reference_keypoints"], measured
    assert measured["max_keypoint_error_px"] <= float(thresholds["max_keypoint_error_px"]), measured
    assert measured["max_score_error"] <= float(thresholds["max_score_error"]), measured


def _view(
    image, points: dict, edges: list[tuple[int, int]], labels: list[str], box, path: Path
) -> dict:
    """The image with the person box, the keypoints and the skeleton between located keypoints."""

    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if box is not None:
        draw.rectangle([float(value) for value in box], outline=(255, 255, 0), width=2)
    for start, end in edges:
        if labels[start] in points and labels[end] in points:
            draw.line(
                [points[labels[start]][:2], points[labels[end]][:2]], fill=(0, 255, 255), width=2
            )
    for x, y, _ in points.values():
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0), outline=(255, 255, 255))
    canvas.save(path)
    return {
        "image": path,
        "keypoints": {name: [x, y, score] for name, (x, y, score) in points.items()},
    }


def _standing_facing_camera(points: dict, image, labels: list[str]) -> dict[str, bool]:
    """What any keypoints of one adult standing upright and facing the camera must satisfy, whatever the numbers.

    Native and reference read the same input, so they would agree on a wrong one too; these facts tie the keypoints to
    the scene the report shows: every joint is located inside the image, the nose is above the shoulders, each side's
    shoulder, hip, knee and ankle go down the image in that order, and the person's left side is on the image's right.
    """

    def at(name: str, axis: int) -> float:
        return points[name][axis] if name in points else float("nan")

    return {
        "every keypoint located": list(points) == labels,
        "inside the image": all(
            0.0 <= x <= image.width and 0.0 <= y <= image.height for x, y, _ in points.values()
        ),
        "nose above the shoulders": at("Nose", 1) < min(at("L_Shoulder", 1), at("R_Shoulder", 1)),
        "shoulder above hip above knee above ankle": all(
            at(f"{side}_Shoulder", 1)
            < at(f"{side}_Hip", 1)
            < at(f"{side}_Knee", 1)
            < at(f"{side}_Ankle", 1)
            for side in ("L", "R")
        ),
        "left side on the image's right": all(
            at(f"L_{joint}", 0) > at(f"R_{joint}", 0)
            for joint in ("Shoulder", "Hip", "Wrist", "Ankle")
        ),
    }


_SCENES = {"standing_facing_camera": _standing_facing_camera}


def _compare(
    image,
    requests: list[dict],
    native: list[dict],
    expected: list[dict],
    model_info,
    thresholds: dict,
    tmp_path: Path,
    scene: str,
) -> None:
    """Record every request's keypoints, overlays, scene facts and comparisons with the reference, then assert them.

    `native[i]` maps a path name (CLI, SDK) to the keypoints of request i. Everything is recorded before the
    assertions, so a failing case keeps its measured values.
    """

    labels, edges = model_info
    measured = []
    facts = {}
    for index, request in enumerate(requests):
        box = request.get("person_box")
        stem = f"request-{index}"
        for name, points in {
            **{f"native-{path}": value for path, value in native[index].items()},
            "reference": expected[index],
        }.items():
            np.save(
                tmp_path / f"{stem}-{name}.npy",
                np.asarray([points[label] for label in points], dtype=np.float64),
            )
        record_evidence(
            "native",
            {
                f"{request['label']} ({_PATH_LABELS[path]})": _view(
                    image, value, edges, labels, box, tmp_path / f"{stem}-native-{path}.png"
                )
                for path, value in native[index].items()
            },
        )
        record_evidence(
            "reference",
            {
                request["label"]: _view(
                    image, expected[index], edges, labels, box, tmp_path / f"{stem}-reference.png"
                )
            },
        )
        for name, points in {**native[index], "reference": expected[index]}.items():
            facts[f"{request['label']} ({_PATH_LABELS.get(name, 'Transformers')})"] = _SCENES[
                scene
            ](points, image, labels)
        for path, value in native[index].items():
            result = _agreement(value, expected[index])
            measured.append(result)
            record_evidence(
                "reference_comparison",
                _comparison(
                    f"{request['label']}: {_PATH_LABELS[path]} keypoints vs Transformers",
                    tmp_path / f"{stem}-native-{path}.npy",
                    tmp_path / f"{stem}-reference.npy",
                    result,
                    thresholds,
                ),
            )
    record_evidence("scene", {"description": scene, "facts": facts})
    for result in measured:
        _assert_agreement(result, thresholds)
    for label, checks in facts.items():
        assert all(checks.values()), f"{label} does not fit the shown {scene} scene: {checks}"


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    from PIL import Image

    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    thresholds = record_evidence("thresholds", _thresholds(case_name))
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest["hf_id"],
            "hf_revision": manifest["hf_revision"],
        },
    )
    model_info = _checkpoint_config(model_dir)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            )
        )
    image_path = _asset(case)
    image = Image.open(image_path).convert("RGB")
    requests = case["requests"]
    assert requests and all(isinstance(request["label"], str) for request in requests)
    record_evidence(
        "inputs",
        {
            "image": image_path,
            "prompt": QUERY,
            "person_boxes": [request.get("person_box") for request in requests],
        },
    )
    native = []
    with evidence_stage("native"):
        for request in requests:
            box = request.get("person_box")
            native.append(
                {
                    "cli": _native_cli(
                        binary, runtime_root, bundle, image_path, box, model_info[0]
                    ),
                    "sdk": _native_sdk(
                        native_build,
                        runtime_root,
                        bundle,
                        np.asarray(image),
                        box,
                        model_info[0],
                        tmp_path,
                    ),
                }
            )
    mark_loaded()
    with evidence_stage("reference"):
        expected = _reference(model_dir, image, [request.get("person_box") for request in requests])
    with evidence_stage("compare"):
        _compare(image, requests, native, expected, model_info, thresholds, tmp_path, case["scene"])


def test_a_testcase_selection_matches_a_collected_e2e_case(request) -> None:
    """A testcase-only selection that matches no E2E case of this session must fail, not test nothing.

    The model filters are checked at collection (`_selected_cases`); a testcase of another collected family passes.
    """

    models, testcases = _selection(request.config)
    if models or not testcases:
        pytest.skip("only a testcase-only E2E selection is checked against the session")
    collected = _collected_cases(request.session.items)
    assert testcases & collected, (
        f"--e2e-testcase {sorted(testcases)} matches no E2E case collected in this session: {sorted(collected)}"
    )


def test_agreement_requires_the_same_keypoints_before_it_measures() -> None:
    expected = {"Nose": (10.0, 20.0, 0.9), "L_Eye": (12.0, 18.0, 0.8)}
    assert _agreement(dict(expected), expected) == {
        "located_keypoints": 2,
        "reference_keypoints": 2,
        "max_keypoint_error_px": 0.0,
        "max_score_error": 0.0,
    }
    moved = {"Nose": (13.0, 24.0, 0.85), "L_Eye": (12.0, 18.0, 0.8)}
    result = _agreement(moved, expected)
    assert result["max_keypoint_error_px"] == pytest.approx(5.0) and result[
        "max_score_error"
    ] == pytest.approx(0.05)
    missing = _agreement({"Nose": expected["Nose"]}, expected)
    assert missing["located_keypoints"] == 1 and missing["max_keypoint_error_px"] == float("inf")
    reordered = _agreement({"L_Eye": expected["L_Eye"], "Nose": expected["Nose"]}, expected)
    assert reordered["max_keypoint_error_px"] == float("inf")
    thresholds = {"max_keypoint_error_px": 0.5, "max_score_error": 0.01}
    for result in (_agreement(moved, expected), missing):
        with pytest.raises(AssertionError):
            _assert_agreement(result, thresholds)
    _assert_agreement(_agreement(dict(expected), expected), thresholds)


def test_a_nan_after_a_finite_error_fails_the_comparison() -> None:
    expected = {"Nose": (10.0, 20.0, 0.9), "L_Eye": (12.0, 18.0, 0.8)}
    thresholds = {"max_keypoint_error_px": 0.5, "max_score_error": 0.01}
    for later in ((float("nan"), 18.0, 0.8), (12.0, 18.0, float("nan"))):
        measured = _agreement({"Nose": expected["Nose"], "L_Eye": later}, expected)
        with pytest.raises(AssertionError):
            _assert_agreement(measured, thresholds)


def test_two_empty_keypoint_sets_do_not_agree() -> None:
    measured = _agreement({}, {})
    assert measured["reference_keypoints"] == 0
    with pytest.raises(AssertionError):
        _assert_agreement(measured, {"max_keypoint_error_px": 0.5, "max_score_error": 0.01})


def test_the_comparison_record_holds_every_asserted_bound(tmp_path: Path) -> None:
    native, reference = tmp_path / "native.npy", tmp_path / "reference.npy"
    record = _comparison(
        "label",
        native,
        reference,
        {
            "located_keypoints": 17,
            "reference_keypoints": 17,
            "max_keypoint_error_px": 0.75,
            "max_score_error": 1e-4,
        },
        {"max_keypoint_error_px": 0.5, "max_score_error": 1e-3},
    )
    assert record["native"] == native and record["reference"] == reference
    assert [
        (check["name"], check["actual"], check["expected"], check["passed"])
        for check in record["checks"]
    ] == [
        ("reference_keypoints", 17, 1, True),
        ("located_keypoints", 17, 17, True),
        ("max_keypoint_error_px", 0.75, 0.5, False),
        ("max_score_error", 1e-4, 1e-3, True),
    ]
    assert all(type(check["passed"]) is bool for check in record["checks"])


def test_native_results_are_checked_before_they_are_compared() -> None:
    labels = ["Nose", "L_Eye", "R_Eye"]
    payload = {
        "points": [1.0, 2.0, 3.0, 4.0],
        "labels": ["Nose", "R_Eye"],
        "scores": [0.5, 0.25],
        "raw_response": "",
        "parse_complete": True,
    }
    assert _points(payload, labels) == {"Nose": (1.0, 2.0, 0.5), "R_Eye": (3.0, 4.0, 0.25)}
    for broken in (
        {**payload, "points": [1.0, 2.0, 3.0]},
        {**payload, "scores": [0.5]},
        {**payload, "labels": ["R_Eye", "Nose"]},
        {**payload, "labels": ["Nose", "Nose"]},
        {**payload, "parse_complete": False},
        {**payload, "raw_response": "text"},
        {**payload, "points": [1.0, 2.0, 3.0, float("nan")]},
        {**payload, "scores": [0.5, float("inf")]},
    ):
        with pytest.raises(AssertionError):
            _points(broken, labels)


def test_the_scene_facts_reject_keypoints_that_do_not_fit_the_image() -> None:
    from PIL import Image

    labels = [
        "Nose",
        "L_Shoulder",
        "R_Shoulder",
        "L_Wrist",
        "R_Wrist",
        "L_Hip",
        "R_Hip",
        "L_Knee",
        "R_Knee",
        "L_Ankle",
        "R_Ankle",
    ]
    standing = {
        "Nose": (50.0, 10.0, 0.9),
        "L_Shoulder": (60.0, 20.0, 0.9),
        "R_Shoulder": (40.0, 20.0, 0.9),
        "L_Wrist": (95.0, 20.0, 0.9),
        "R_Wrist": (5.0, 20.0, 0.9),
        "L_Hip": (56.0, 50.0, 0.9),
        "R_Hip": (44.0, 50.0, 0.9),
        "L_Knee": (70.0, 70.0, 0.9),
        "R_Knee": (30.0, 70.0, 0.9),
        "L_Ankle": (80.0, 95.0, 0.9),
        "R_Ankle": (20.0, 95.0, 0.9),
    }
    image = Image.new("RGB", (100, 100))
    assert all(_standing_facing_camera(standing, image, labels).values())
    mirrored = {name: (100.0 - x, y, score) for name, (x, y, score) in standing.items()}
    upside_down = {name: (x, 100.0 - y, score) for name, (x, y, score) in standing.items()}
    missing = {name: value for name, value in standing.items() if name != "L_Knee"}
    outside = {**standing, "R_Wrist": (-3.0, 20.0, 0.9)}
    for wrong, fact in (
        (mirrored, "left side on the image's right"),
        (upside_down, "shoulder above hip above knee above ankle"),
        (missing, "every keypoint located"),
        (outside, "inside the image"),
    ):
        assert not _standing_facing_camera(wrong, image, labels)[fact], fact


def test_a_request_box_is_the_reference_coco_box() -> None:
    from PIL import Image

    image = Image.new("RGB", (40, 30))
    assert _coco_box(None, image) == [0.0, 0.0, 40.0, 30.0]
    assert _coco_box([4, 5.5, 20, 25], image) == [4.0, 5.5, 16.0, 19.5]


class _Config:
    def __init__(self, options: dict) -> None:
        self._options = options

    def getoption(self, name: str, default=None):
        return self._options.get(name, default)


class _Metafunc:
    fixturenames = ("case_name", "tmp_path")

    def __init__(self, options: dict) -> None:
        self.config = _Config(options)
        self.values = None

    def parametrize(self, name: str, values) -> None:
        assert name == "case_name"
        self.values = list(values)


def _parametrized(options: dict) -> list:
    metafunc = _Metafunc(options)
    pytest_generate_tests(metafunc)
    return metafunc.values


def test_an_explicit_selection_of_this_family_that_matches_no_case_is_an_error(
    tmp_path: Path,
) -> None:
    case = next(iter(CASES))
    models_file = tmp_path / "models.txt"
    models_file.write_text(f"# selected models\n{CASES[case][0]['name']}\n", encoding="utf-8")
    unselected = _parametrized({})
    assert [value.values for value in unselected] == [(name,) for name in CASES]
    assert all([mark.name for mark in value.marks] == ["skip"] for value in unselected)
    assert _parametrized({"--e2e-model": ["another-family"]}) == []
    assert _parametrized({"--e2e-testcase": ["another-case"]}) == []
    assert _parametrized({"--e2e-testcase": [case]}) == [case]
    assert _parametrized(
        {"--e2e-model": [f"another-family,{FAMILY}"], "--e2e-testcase": [case]}
    ) == [case]
    assert _parametrized({"--e2e-models-file": str(models_file)}) == list(CASES)
    for options in (
        {"--e2e-model": [FAMILY], "--e2e-testcase": ["no-such-case"]},
        {"--e2e-models-file": str(models_file), "--e2e-testcase": ["another-case"]},
        {"--e2e-model": ["another-family"], "--e2e-testcase": [case]},
        {"--e2e-testcase": [FAMILY]},
    ):
        with pytest.raises(pytest.UsageError, match="matches none of its testcases"):
            _parametrized(options)


class _Item:
    def __init__(self, name: str, case: str | None) -> None:
        self.originalname = name
        self.callspec = type(
            "CallSpec", (), {"params": {} if case is None else {"case_name": case}}
        )()


def test_the_session_cases_are_the_e2e_items_with_a_case() -> None:
    items = [
        _Item("test_official_checkpoint_e2e", "vitpose-base-simple"),
        _Item("test_e2e", "another-family-case"),
        _Item("test_agreement_requires_the_same_keypoints_before_it_measures", None),
        _Item("test_e2e_helper", "not-an-e2e-case"),
    ]
    assert _collected_cases(items) == {"vitpose-base-simple", "another-family-case"}
