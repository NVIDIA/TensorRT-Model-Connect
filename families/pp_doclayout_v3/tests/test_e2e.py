# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build -> native Task/CLI -> official reference, with layout geometry."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "pp_doclayout_v3"
TASKS = frozenset({"document_layout"})
MANIFEST = json.loads((Path(__file__).parent / "manifests/pp-doclayout-v3.json").read_text())
(DOCUMENT,) = MANIFEST["external_files"]
CI_ASSET_DIR = Path("/opt/trtmc-ci")


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


def _model_dir():
    name = "TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR"
    if name in os.environ:
        path = _path(name)
        assert path.is_dir(), f"selected {FAMILY} E2E model path is not a directory: {path}"
        if (path / "config.json").is_file():
            return path
        # Internal CI points MODEL_DIR at external_files, not the Hub snapshot.
        assert (path / DOCUMENT["path"]).is_file(), (
            f"{name} contains neither checkpoint config.json nor declared layout_demo.jpg: {path}"
        )
    from huggingface_hub import snapshot_download

    try:
        path = Path(
            snapshot_download(
                repo_id=MANIFEST["hf_id"],
                revision=MANIFEST["hf_revision"],
                local_files_only=True,
            )
        )
    except OSError as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint "
            f"{MANIFEST['hf_id']}@{MANIFEST['hf_revision']}"
        ) from error
    assert path.is_dir(), f"selected {FAMILY} E2E model path is not a directory: {path}"
    return path


def _image_path(model_dir):
    from PIL import Image

    name = "TRTMC_PP_DOCLAYOUT_V3_TEST_IMAGE"
    if name in os.environ:
        path = _path(name)
        assert path.is_file(), f"selected {FAMILY} E2E image is not a file: {path}"
        return path
    roots = []
    if "TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR" in os.environ:
        roots.append(_path("TRTMC_PP_DOCLAYOUT_V3_MODEL_DIR"))
    roots.extend((model_dir, CI_ASSET_DIR))
    for root in roots:
        path = root / DOCUMENT["path"]
        if path.is_file():
            try:
                with Image.open(path) as image:
                    image.verify()
            except OSError as error:
                raise AssertionError(
                    f"selected {FAMILY} E2E invalid layout document: {path}"
                ) from error
            return path
    raise AssertionError(
        f"selected {FAMILY} E2E requires layout_demo.jpg; provision the official "
        "document declared in the manifest's external_files before offline execution"
    )


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


@pytest.mark.parametrize("case_name", ["pp-doclayout-v3"])
def test_official_checkpoint_e2e(request, tmp_path, case_name):
    if not _selected(request.config):
        pytest.skip("direct E2E requires an explicit model or testcase selector")
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from transformers.utils import logging

    logging.disable_progress_bar()
    torch.set_num_threads(8)
    record_evidence("inputs", {"manifest": MANIFEST, "variants": ["original", "mirrored"]})
    record_evidence(
        "checkpoint", {"repo_id": MANIFEST["hf_id"], "revision": MANIFEST["hf_revision"]}
    )
    record_evidence(
        "thresholds",
        {
            "score_absolute_error_max": 0.02,
            "box_iou_min": 0.98,
            "polygon_iou_min": 0.98,
            "class_id": "exact reading order",
        },
    )
    with evidence_stage("checkpoint"):
        model_dir = _model_dir()
        image_path = _image_path(model_dir)
    binary, runtime = _path("TRTMC_BINARY"), _path("TRTMC_RUNTIME_ROOT")
    assert (runtime / "libtrtmc_model_pp_doclayout_v3.so").is_file()
    assert (runtime / "libtrtmc_backend_trt.so").is_file()
    bundle = tmp_path / MANIFEST["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=MANIFEST["task"],
                precision=MANIFEST["precision"],
                image_height=MANIFEST["image_height"],
                image_width=MANIFEST["image_width"],
                tensor_parallel_size=MANIFEST["tensor_parallel_size"],
            )
        )
    record_evidence("bundle", {"path": str(bundle)})
    with evidence_stage("reference"):
        model = AutoModelForObjectDetection.from_pretrained(model_dir, local_files_only=True).eval()
        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    original = Image.open(image_path).convert("RGB")
    native_records, reference_records, metrics, views = [], [], [], []
    for index, image in enumerate((original, ImageOps.mirror(original))):
        path = tmp_path / f"document-{index}.png"
        image.save(path)
        record_evidence("inputs", {f"document_{index}": path})
        with evidence_stage("native"):
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
        record_evidence(
            f"native_process_{index}",
            {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr},
        )
        native_records.append({"variant": index, **actual})
        record_evidence("native", {"variants": native_records})
        with evidence_stage("reference"):
            with torch.inference_mode():
                outputs = model(**processor(images=image, return_tensors="pt"))
            for name in ("logits", "pred_boxes", "order_logits", "out_masks"):
                assert torch.isfinite(getattr(outputs, name)).all(), name
            expected = processor.post_process_object_detection(
                outputs, target_sizes=[image.size[::-1]], threshold=0.5
            )[0]
        reference_regions = [
            {
                "class_id": int(label),
                "score": float(score),
                "box": box.tolist(),
                "polygon": np.asarray(polygon).tolist() if polygon is not None else None,
            }
            for score, label, box, polygon in zip(
                expected["scores"],
                expected["labels"],
                expected["boxes"],
                expected["polygon_points"],
                strict=True,
            )
        ]
        reference_records.append(
            {
                "variant": index,
                "image_height": image.height,
                "image_width": image.width,
                "regions": reference_regions,
            }
        )
        record_evidence("reference", {"mode": "hf_transformers", "variants": reference_records})
        with evidence_stage("compare"):
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
                strict=True,
            ):
                measurement = {
                    "variant": index,
                    "region": len(metrics),
                    "score_absolute_error": abs(region["score"] - float(score)),
                    "box_iou": _iou(region["box"], box.tolist()),
                }
                metrics.append(measurement)
                record_evidence("metrics", metrics)
                assert region["class_id"] == int(label)
                assert abs(region["score"] - float(score)) <= 0.02
                assert _iou(region["box"], box.tolist()) >= 0.98
                assert len(region["polygon"]) >= 4 and polygon is not None
                assert np.isfinite(region["polygon"]).all()
                measurement["polygon_iou"] = _polygon_iou(
                    region["polygon"], polygon, image.height, image.width
                )
                record_evidence("metrics", metrics)
                assert _polygon_iou(region["polygon"], polygon, image.height, image.width) >= 0.98
        if evidence_enabled():
            # Explicit comparison records retain the enforced geometry limits;
            # report readers need not infer them from family helper names.
            paired = {}
            for role, record in (("native", actual), ("reference", reference_records[-1])):
                geometry = tmp_path / f"document-{index}-{role}.json"
                geometry.write_text(json.dumps(record, indent=2))
                paired[role] = geometry
                preview = image.copy()
                drawing = ImageDraw.Draw(preview)
                for order, region in enumerate(record["regions"]):
                    drawing.polygon(
                        [tuple(point) for point in region["polygon"]], outline="red", width=2
                    )
                    drawing.text(
                        tuple(region["box"][:2]), f"{order}: {region['class_id']}", fill="red"
                    )
                preview_path = tmp_path / f"document-{index}-{role}.png"
                preview.save(preview_path)
                views.append(
                    {
                        "title": f"Sample {index} / {role}",
                        "image": preview_path,
                        "caption": "Original" if index == 0 else "Mirrored",
                    }
                )
            record_evidence("views", views)
            measured = [item for item in metrics if item["variant"] == index]
            limits = [
                ("score_error", max(item["score_absolute_error"] for item in measured), "<=", 0.02),
                ("box_iou", min(item["box_iou"] for item in measured), ">=", 0.98),
                ("polygon_iou", min(item["polygon_iou"] for item in measured), ">=", 0.98),
            ]
            record_evidence(
                "reference_comparison",
                {
                    "label": f"Document {index}: regions in reading order",
                    "scope": "independent_reference",
                    "enforced": True,
                    **paired,
                    "checks": [
                        {
                            "name": "ordered_class_ids",
                            "scope": "independent_reference",
                            "actual": sum(
                                left["class_id"] != right["class_id"]
                                for left, right in zip(
                                    actual["regions"], reference_regions, strict=True
                                )
                            ),
                            "operator": "==",
                            "expected": 0,
                            "passed": all(
                                left["class_id"] == right["class_id"]
                                for left, right in zip(
                                    actual["regions"], reference_regions, strict=True
                                )
                            ),
                        }
                    ]
                    + [
                        {
                            "name": name,
                            "scope": "independent_reference",
                            "actual": value,
                            "operator": operator,
                            "expected": limit,
                            "passed": value <= limit if operator == "<=" else value >= limit,
                        }
                        for name, value, operator, limit in limits
                    ],
                },
            )
