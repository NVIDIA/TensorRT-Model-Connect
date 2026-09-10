# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime, and transformers reference proof for DETR."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "detr"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == "object_detection"
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
                marks=pytest.mark.skip(reason="real DETR E2E requires explicit selection"),
                id=name,
            )
            for name in names
        ]
    metafunc.parametrize("case_name", names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected DETR E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected DETR E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_DETR_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_DETR_MODEL_DIR")
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
    assert path.is_file(), f"selected DETR E2E image is missing: {path}"
    return path


def _resize_like_runtime(image, size: int) -> np.ndarray:
    """Resize the way families/detr/runtime/image_preprocess_seam.cpp does.

    The engine sees whatever the runtime seam produced, so the reference has to
    see the same pixels or the comparison measures the gap between two bilinear
    implementations instead of the engine. PIL's resize differs from the seam's
    by about 1e-3 per pixel, which the detector amplifies into roughly two
    pixels of box. The seam's own resize is covered by its C++ test.
    """
    source = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = source.shape
    rows = np.clip((np.arange(size) + 0.5) * (height / size) - 0.5, 0.0, height - 1)
    columns = np.clip((np.arange(size) + 0.5) * (width / size) - 0.5, 0.0, width - 1)
    row0 = rows.astype(np.int32)
    column0 = columns.astype(np.int32)
    row1 = np.minimum(row0 + 1, height - 1)
    column1 = np.minimum(column0 + 1, width - 1)
    weight_y = (rows - row0)[:, None, None]
    weight_x = (columns - column0)[None, :, None]
    top = source[row0[:, None], column0[None, :]] * (1.0 - weight_x) + (
        source[row0[:, None], column1[None, :]] * weight_x
    )
    bottom = source[row1[:, None], column0[None, :]] * (1.0 - weight_x) + (
        source[row1[:, None], column1[None, :]] * weight_x
    )
    return top * (1.0 - weight_y) + bottom * weight_y


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
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
            # The engine has one fixed input size, so the manifest names it and
            # the reference below is fed the very same resize.
            image_height=int(manifest["input_size"]),
            image_width=int(manifest["input_size"]),
        )
    )
    image = _asset(case)
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

    import torch
    from PIL import Image
    from transformers import DetrForObjectDetection

    reference = DetrForObjectDetection.from_pretrained(str(model_dir)).eval()

    # Feed the reference exactly what the engine saw: the same plain resize and
    # the same normalisation the runtime seam applies.
    size = int(manifest["input_size"])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    source = Image.open(image).convert("RGB")
    resized = _resize_like_runtime(source, size)
    pixels = ((resized - mean) / std).transpose(2, 0, 1)[None]
    with torch.no_grad():
        outputs = reference(pixel_values=torch.from_numpy(pixels.astype(np.float32)))

    # DETR scores a query by its best real class; the last logit means "no object".
    probabilities = torch.softmax(outputs.logits[0], -1)[:, :-1].numpy()
    scores = probabilities.max(1)
    labels = probabilities.argmax(1)
    boxes = outputs.pred_boxes[0].numpy()
    order = np.argsort(-scores)

    threshold = 0.5
    kept = [index for index in order if scores[index] >= threshold]
    assert kept, "the reference found nothing above the threshold to compare against"
    assert len(actual["scores"]) == len(kept), (
        f"engine returned {len(actual['scores'])} detections, reference {len(kept)}"
    )
    width, height = source.size
    for position, index in enumerate(kept):
        assert int(actual["classes"][position]) == int(labels[index])
        assert abs(float(actual["scores"][position]) - float(scores[index])) < 5e-3
        centre_x, centre_y, box_w, box_h = (float(value) for value in boxes[index])
        expected_box = (
            (centre_x - box_w / 2) * width,
            (centre_y - box_h / 2) * height,
            (centre_x + box_w / 2) * width,
            (centre_y + box_h / 2) * height,
        )
        for axis, want in enumerate(expected_box):
            # The engine reports source-image pixels; preprocessing is a plain
            # resize, so the reference box scales by one factor per axis.
            # The CLI writes the boxes as one flat array, four per detection.
            assert abs(float(actual["boxes"][position * 4 + axis]) - want) < 1.0
