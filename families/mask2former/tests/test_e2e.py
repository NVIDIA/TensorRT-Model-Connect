# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI and SDK runs, and the official Transformers reference for Mask2Former.

The reference is `Mask2FormerImageProcessor` (the checkpoint's own processor, PIL resize) and
`Mask2FormerForUniversalSegmentation` in float32 on the GPU with TF32 disabled, post-processed by
`post_process_semantic_segmentation` at the original image size. The test image is lossless, so the CLI's image
loader, the C and C++ SDK consumers and the reference all see the same 8-bit pixels; every label map is held to the
same agreement with the reference.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "mask2former"
TASK = "image_to_semantic_segmentation"
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


def _selected_cases(config) -> list[str] | None:
    """Return the selected cases, or None when no E2E selector is given.

    A case is selected when it matches the model filters (`--e2e-model`, `--e2e-models-file`: the family, a manifest
    or a case name) and the testcase filters (`--e2e-testcase`); a missing kind of filter matches every case. A
    selection that names this family, one of its manifests or one of its cases but matches no case is a usage error,
    so that a misspelt name cannot pass without testing anything; a selection of other families selects nothing here.
    """

    models = _values(config.getoption("--e2e-model", default=[]))
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        models.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcases = _values(config.getoption("--e2e-testcase", default=[]))
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
                marks=pytest.mark.skip(
                    reason="the real Mask2Former E2E requires explicit selection"
                ),
                id=name,
            )
            for name in CASES
        ]
    metafunc.parametrize("case_name", selected)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected Mask2Former E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected Mask2Former E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
        )
    )


def _asset(case: dict) -> Path:
    path = TEST_ROOT / str(case["test_image"])
    assert path.is_file(), f"selected Mask2Former E2E image is missing: {path}"
    record_evidence("inputs", {"image": path})
    return path


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected Mask2Former E2E requires its thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _labels(model_dir: Path) -> list[str]:
    id2label = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["id2label"]
    return [id2label[str(index)] for index in range(len(id2label))]


def _reference(model_dir: Path, image) -> np.ndarray:
    import torch
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(model_dir, dtype=torch.float32)
        .to("cuda")
        .eval()
    )
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(pixel_values=inputs["pixel_values"].to("cuda"))
    labels = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]
    return labels.cpu().numpy().astype(np.int32)


def _agreement(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Pixel accuracy and the mean IoU over every class either map uses."""

    assert actual.shape == expected.shape and actual.ndim == 2
    ious = []
    for label in np.union1d(actual, expected):
        left, right = actual == label, expected == label
        ious.append(np.logical_and(left, right).sum() / np.logical_or(left, right).sum())
    return {"pixel_accuracy": float(np.mean(actual == expected)), "mIoU": float(np.mean(ious))}


_CHECK_LABELS = {"pixel_accuracy": "Pixel accuracy", "mIoU": "Mean IoU"}
_PATH_LABELS = {"cli": "CLI", "sdk": "C and C++ SDK"}


def _comparison(
    label: str, native: Path, reference: Path, measured: dict, thresholds: dict
) -> dict:
    """One `reference_comparison` record: both label maps and every bound the test asserts."""

    checks = []
    for name, text in _CHECK_LABELS.items():
        actual, expected = float(measured[name]), float(thresholds[name])
        checks.append(
            {
                "name": name,
                "label": text,
                "scope": "independent_reference",
                "actual": actual,
                "operator": ">=",
                "expected": expected,
                "passed": bool(actual >= expected),
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
    # Calibrated on the test image (382 x 640 = 244,480 pixels): every healthy run of the CLI and of the C and C++
    # SDKs (2 E2E runs) agreed with the reference on every pixel (1.0 / 1.0); the reference map holds 7 classes, and
    # 17 of the 384 x 384 score pixels have a top-2 margin below 1e-3 against an engine deviation of at most 7.8e-6.
    # A defective input path (the JPEG of this image decoded by another decoder, up to 3 levels off in 4.7 % of the
    # values) gave pixel accuracy 0.9986 and mIoU 0.859; both bounds reject it.
    assert measured["pixel_accuracy"] >= float(thresholds["pixel_accuracy"]), measured
    assert measured["mIoU"] >= float(thresholds["mIoU"]), measured


def _view(image, mask: np.ndarray, names: list[str], path: Path) -> dict:
    """Blend one fixed color per class over the image and count the pixels of every class present."""

    from PIL import Image

    palette = np.random.default_rng(0).integers(0, 256, size=(len(names), 3), dtype=np.uint8)
    colors = Image.fromarray(palette[mask])
    Image.blend(image, colors, alpha=0.5).save(path)
    classes, counts = np.unique(mask, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return {
        "image": path,
        "class_pixels": {names[int(classes[index])]: int(counts[index]) for index in order},
    }


def _compare(
    image, native: dict, expected: np.ndarray, labels: list[str], thresholds: dict, tmp_path: Path
) -> None:
    """Record every label map, its overlay and its comparison with the reference, then assert the bounds.

    `native` maps a path name (CLI, SDK) to its label map. The comparisons are recorded before the assertions, so a
    failing case keeps its measured values.
    """

    maps = {**{f"native-{name}": mask for name, mask in native.items()}, "reference": expected}
    for name, mask in maps.items():
        assert mask.shape == (image.height, image.width), name
        assert mask.min() >= 0 and mask.max() < len(labels), name
        np.save(tmp_path / f"{name}.labels.npy", mask, allow_pickle=False)
    record_evidence(
        "native",
        {
            name: _view(image, mask, labels, tmp_path / f"native-{name}.png")
            for name, mask in native.items()
        },
    )
    record_evidence("reference", _view(image, expected, labels, tmp_path / "reference.png"))
    measured = {name: _agreement(mask, expected) for name, mask in native.items()}
    for name in native:
        record_evidence(
            "reference_comparison",
            _comparison(
                f"{_PATH_LABELS[name]} label map vs Transformers",
                tmp_path / f"native-{name}.labels.npy",
                tmp_path / "reference.labels.npy",
                measured[name],
                thresholds,
            ),
        )
    for name in native:
        _assert_agreement(measured[name], thresholds)


def _native(
    binary: Path, runtime_root: Path, bundle: Path, image: Path, labels: list[str]
) -> np.ndarray:
    completed = subprocess.run(
        [
            str(binary),
            "segment",
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            "--image",
            str(image),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    record_evidence(
        "native_process", {"command": shlex.join(completed.args), "stderr": completed.stderr}
    )
    payload = json.loads(completed.stdout)
    height, width = int(payload["height"]), int(payload["width"])
    assert payload["class_ids"] == list(range(len(labels)))
    assert payload["class_names"] == labels
    assert payload["vocabulary_id"] == ""
    assert payload["ignore_label"] is None and payload["background_label"] is None
    assert (
        payload["class_scores"] == []
        and payload["score_height"] == 0
        and payload["score_width"] == 0
    )
    mask = np.asarray(payload["mask"], dtype=np.int32)
    assert mask.size == height * width
    return mask.reshape(height, width)


def _sdk_masks(
    runtime_root: Path, bundle: Path, pixels: np.ndarray, labels: list[str], tmp_path: Path
) -> list:
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    raw_image = tmp_path / "sdk-input.rgb.f32"
    (pixels.astype(np.float32) / np.float32(255.0)).tofile(raw_image)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    masks = []
    for language in ("c", "cpp"):
        consumer = native_build / f"test_mask2former_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        output = tmp_path / f"sdk-{language}.labels.i32"
        completed = subprocess.run(
            [
                str(consumer),
                str(bundle),
                str(runtime_root),
                str(raw_image),
                *map(str, pixels.shape[:2]),
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        summary = json.loads(completed.stdout)
        record_evidence(f"sdk_{language}", summary)
        assert summary["task"] == TASK
        assert (summary["height"], summary["width"]) == pixels.shape[:2]
        assert summary["pixel_count"] == pixels.shape[0] * pixels.shape[1]
        assert summary["class_ids"] == list(range(len(labels))) and summary["class_names"] == labels
        assert summary["vocabulary_id"] == "" and summary["score_count"] == 0
        assert summary["has_ignore_label"] == 0 and summary["has_background_label"] == 0
        masks.append(np.fromfile(output, dtype=np.int32).reshape(pixels.shape[:2]))
    assert np.array_equal(masks[0], masks[1]), "the C and C++ SDKs must return the same labels"
    return masks


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    from PIL import Image

    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    thresholds = record_evidence("thresholds", _thresholds(case_name))
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
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
    labels = _labels(model_dir)
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
    with evidence_stage("native"):
        native = _native(binary, runtime_root, bundle, image_path, labels)
        image = Image.open(image_path).convert("RGB")
        sdk = _sdk_masks(runtime_root, bundle, np.asarray(image), labels, tmp_path)
    with evidence_stage("reference"):
        expected = _reference(model_dir, image)
    with evidence_stage("compare"):
        _compare(image, {"cli": native, "sdk": sdk[0]}, expected, labels, thresholds, tmp_path)


def test_agreement_metrics_count_every_used_class() -> None:
    expected = np.asarray([[0, 0], [1, 1]], dtype=np.int32)
    assert _agreement(expected, expected) == {"pixel_accuracy": 1.0, "mIoU": 1.0}
    measured = _agreement(np.asarray([[0, 0], [1, 2]], dtype=np.int32), expected)
    assert measured["pixel_accuracy"] == 0.75
    assert measured["mIoU"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    with pytest.raises(AssertionError):
        _assert_agreement(measured, {"pixel_accuracy": 0.9, "mIoU": 0.5})


def test_compare_saves_every_map_and_its_overlay_before_it_asserts(tmp_path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (4, 2))
    expected = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    thresholds = {"pixel_accuracy": 0.999, "mIoU": 0.98}
    (tmp_path / "equal").mkdir()
    _compare(image, {"cli": expected.copy()}, expected, ["a", "b"], thresholds, tmp_path / "equal")
    with pytest.raises(AssertionError):
        _compare(
            image,
            {"cli": expected, "sdk": 1 - expected},
            expected,
            ["a", "b"],
            thresholds,
            tmp_path,
        )
    for name in ("native-cli", "native-sdk", "reference"):
        assert (tmp_path / f"{name}.png").is_file()
        assert np.array_equal(
            np.load(tmp_path / f"{name}.labels.npy"),
            1 - expected if name == "native-sdk" else expected,
        )
    with pytest.raises(AssertionError, match="native-cli"):
        _compare(image, {"cli": expected + 1}, expected, ["a", "b"], thresholds, tmp_path)


def test_the_comparison_record_holds_every_asserted_bound(tmp_path: Path) -> None:
    native, reference = tmp_path / "native.labels.npy", tmp_path / "reference.labels.npy"
    record = _comparison(
        "label",
        native,
        reference,
        {"pixel_accuracy": 0.75, "mIoU": 0.99},
        {"pixel_accuracy": 0.9, "mIoU": 0.98},
    )
    assert record["native"] == native and record["reference"] == reference
    assert [
        (check["name"], check["actual"], check["expected"], check["passed"])
        for check in record["checks"]
    ] == [
        ("pixel_accuracy", 0.75, 0.9, False),
        ("mIoU", 0.99, 0.98, True),
    ]
    assert all(type(check["passed"]) is bool for check in record["checks"])


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
