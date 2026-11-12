# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build -> native Task/CLI -> official reference, with layout geometry."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "pp_doclayout_v3"
TASKS = frozenset({"document_layout"})
MANIFEST = json.loads((Path(__file__).parent / "manifests/pp-doclayout-v3.json").read_text())


def _selected(config):
    models = set()
    for raw in config.getoption("--e2e-model") or []:
        models.update(x.strip() for x in raw.split(",") if x.strip())
    model_file = config.getoption("--e2e-models-file")
    if model_file:
        models.update(
            x.strip()
            for x in Path(model_file).read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        )
    cases = set()
    for raw in config.getoption("--e2e-testcase") or []:
        cases.update(x.strip() for x in raw.split(",") if x.strip())
    return (
        bool(models or cases)
        and (not models or bool(models & {FAMILY, "pp-doclayout-v3"}))
        and (not cases or "pp-doclayout-v3" in cases)
    )


def _path(name):
    value = os.environ.get(name)
    assert value, f"selected {FAMILY} E2E requires {name}"
    path = Path(value).resolve()
    assert path.exists(), f"{name} does not exist: {path}"
    return path


def _iou(left, right):
    lo = np.maximum(left[:2], right[:2])
    hi = np.minimum(left[2:], right[2:])
    intersection = float(np.maximum(hi - lo, 0).prod())
    union = float(
        np.prod(np.asarray(left[2:]) - left[:2])
        + np.prod(np.asarray(right[2:]) - right[:2])
        - intersection
    )
    return intersection / union if union > 0 else 0


def _polygon_iou(left, right, height, width):
    import cv2

    masks = []
    for polygon in (left, right):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
        masks.append(mask.astype(bool))
    union = np.logical_or(*masks).sum()
    return float(np.logical_and(*masks).sum() / union) if union else 0


def test_official_checkpoint_e2e(request, tmp_path):
    if not _selected(request.config):
        pytest.skip("direct E2E requires an explicit model or testcase selector")
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from transformers.utils import logging

    logging.disable_progress_bar()
    torch.set_num_threads(8)
    model_dir = _path("TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR")
    with (model_dir / "model.safetensors").open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == MANIFEST["weights_sha256"]
    image_path = _path("TRTMC_PP_DOCLAYOUT_V3_TEST_IMAGE")
    binary, runtime = _path("TRTMC_BINARY"), _path("TRTMC_RUNTIME_ROOT")
    assert (runtime / "libtrtmc_model_pp_doclayout_v3.so").is_file()
    assert (runtime / "libtrtmc_backend_trt.so").is_file()
    bundle = tmp_path / MANIFEST["bundle"]
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task="document_layout",
            precision="fp32",
        )
    )
    model = AutoModelForObjectDetection.from_pretrained(model_dir, local_files_only=True).eval()
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    original = Image.open(image_path).convert("RGB")
    for index, image in enumerate((original, ImageOps.mirror(original))):
        path = tmp_path / f"document-{index}.png"
        image.save(path)
        completed = subprocess.run(
            [
                str(binary),
                "layout",
                str(bundle),
                "--runtime-root",
                str(runtime),
                "--image",
                str(path),
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
        actual = json.loads(completed.stdout)
        with torch.inference_mode():
            outputs = model(**processor(images=image, return_tensors="pt"))
        for name in ("logits", "pred_boxes", "order_logits", "out_masks"):
            assert torch.isfinite(getattr(outputs, name)).all(), name
        expected = processor.post_process_object_detection(
            outputs, target_sizes=[image.size[::-1]], threshold=0.5
        )[0]
        regions = actual["regions"]
        assert regions and len(regions) == len(expected["scores"])
        assert actual["image_height"] == image.height and actual["image_width"] == image.width
        # Compare by reading order. Matching only by score would miss an order-head defect.
        for region, score, label, box, polygon in zip(
            regions,
            expected["scores"],
            expected["labels"],
            expected["boxes"],
            expected["polygon_points"],
        ):
            assert region["class_id"] == int(label)
            assert abs(region["score"] - float(score)) <= 0.02
            assert _iou(region["box"], box.tolist()) >= 0.98
            assert len(region["polygon"]) >= 4 and polygon is not None
            assert np.isfinite(region["polygon"]).all()
            assert _polygon_iou(region["polygon"], polygon, image.height, image.width) >= 0.98
