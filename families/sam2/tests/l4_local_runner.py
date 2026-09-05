# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit local-only SAM2 five-frame golden qualification."""

from __future__ import annotations

import argparse
import json
import math
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


_LOCAL_ROOT = Path(__file__).resolve().parent / "local"
_MANIFEST = _LOCAL_ROOT / "sam2-l4-local.json"
_THRESHOLDS = _LOCAL_ROOT / "sam2-l4-local.thresholds.json"
_FRAME_COUNT = 5
_HEIGHT = 1280
_WIDTH = 1088
_FRAME_PIXELS = _HEIGHT * _WIDTH
_UNPACK_LSB = tuple(bytes((value >> bit) & 1 for bit in range(8)) for value in range(256))


class QualificationError(RuntimeError):
    """The explicit local qualification inputs or results are incomplete."""


def _regular(path: Path, label: str, *, size: int | None = None) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise QualificationError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise QualificationError(f"{label} must be a regular non-symlink: {path}")
    if size is not None and metadata.st_size != size:
        raise QualificationError(f"{label} has size {metadata.st_size}, expected {size}")
    return path


def _directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise QualificationError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise QualificationError(f"{label} must be a directory non-symlink: {path}")
    return path


def _json(path: Path, label: str) -> dict:
    try:
        value = json.loads(_regular(path, label).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must contain one JSON object")
    return value


def _contract() -> tuple[dict, dict]:
    manifest = _json(_MANIFEST, "SAM2 L4 local manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("name") != "sam2-l4-local"
        or manifest.get("family") != "sam2"
        or manifest.get("task") != "video_segmentation"
        or manifest.get("qualification") != "local_only"
        or manifest.get("bundle") != "sam2-l4-local.bundle"
    ):
        raise QualificationError("SAM2 L4 local manifest identity is invalid")
    fixture = manifest.get("fixture")
    if not isinstance(fixture, dict) or fixture != {
        "frame_directory": "rgb8",
        "frame_count": _FRAME_COUNT,
        "height": _HEIGHT,
        "width": _WIDTH,
        "golden_manifest": "golden/manifest.json",
        "golden_masks": "golden/masks.bitpack",
        "mask_bit_order": "least_significant_bit_first",
    }:
        raise QualificationError("SAM2 L4 local fixture contract is invalid")
    raw_thresholds = _json(_THRESHOLDS, "SAM2 L4 local thresholds")
    thresholds = raw_thresholds.get("threshold_overrides")
    names = {
        "minimum_frame_mask_iou",
        "minimum_macro_mask_iou",
        "minimum_global_mask_iou",
        "minimum_bbox_iou",
        "maximum_bbox_coordinate_error",
        "maximum_bbox_score_error",
        "label_exact",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != names:
        raise QualificationError("SAM2 L4 local threshold contract is incomplete")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in thresholds.values()
    ):
        raise QualificationError("SAM2 L4 local thresholds must be finite numbers")
    return manifest, thresholds


def _frames(fixture_dir: Path) -> list[Path]:
    root = _directory(fixture_dir / "rgb8", "SAM2 RGB8 frame directory")
    expected = {f"{index:06d}.rgb8" for index in range(_FRAME_COUNT)}
    observed = {path.name for path in root.iterdir()}
    if observed != expected:
        raise QualificationError("SAM2 RGB8 frame directory must contain exactly five frames")
    return [
        _regular(
            root / f"{index:06d}.rgb8",
            f"SAM2 RGB8 frame {index}",
            size=_FRAME_PIXELS * 3,
        )
        for index in range(_FRAME_COUNT)
    ]


def _golden(fixture_dir: Path) -> tuple[tuple[tuple[float, ...], float, int], bytes]:
    root = _directory(fixture_dir / "golden", "SAM2 golden directory")
    manifest = _json(root / "manifest.json", "SAM2 golden manifest")
    try:
        bbox = manifest["frame_zero_bbox"]
        coordinates = tuple(float(value) for value in bbox["original_image_xyxy"])
        score = float(bbox["score"])
        label = bbox["label"]
    except (KeyError, TypeError, ValueError) as error:
        raise QualificationError("SAM2 golden bbox contract is invalid") from error
    if (
        len(coordinates) != 4
        or not all(math.isfinite(value) for value in coordinates)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
        or not isinstance(label, int)
        or isinstance(label, bool)
    ):
        raise QualificationError("SAM2 golden bbox values are invalid")
    packed_size = (_FRAME_COUNT * _FRAME_PIXELS + 7) // 8
    packed = _regular(
        root / "masks.bitpack", "SAM2 golden packed masks", size=packed_size
    ).read_bytes()
    masks = b"".join(_UNPACK_LSB[value] for value in packed)[: _FRAME_COUNT * _FRAME_PIXELS]
    return (coordinates, score, label), masks


def _mask_accuracy(candidate: bytes, reference: bytes) -> tuple[list[float], float, float]:
    expected_size = _FRAME_COUNT * _FRAME_PIXELS
    if len(candidate) != expected_size or len(reference) != expected_size:
        raise QualificationError("SAM2 mask evidence has the wrong size")
    if not set(candidate) <= {0, 1}:
        raise QualificationError("SAM2 candidate masks are not binary")
    frame_iou = []
    total_intersection = 0
    total_union = 0
    for index in range(_FRAME_COUNT):
        begin = index * _FRAME_PIXELS
        left = candidate[begin : begin + _FRAME_PIXELS]
        right = reference[begin : begin + _FRAME_PIXELS]
        intersection = sum(a & b for a, b in zip(left, right, strict=True))
        union = sum(left) + sum(right) - intersection
        frame_iou.append(1.0 if union == 0 else intersection / union)
        total_intersection += intersection
        total_union += union
    return (
        frame_iou,
        sum(frame_iou) / _FRAME_COUNT,
        1.0 if total_union == 0 else total_intersection / total_union,
    )


def _bbox_accuracy(candidate: tuple, reference: tuple) -> tuple[float, float, float, bool]:
    left, left_score, left_label = candidate
    right, right_score, right_label = reference
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return (
        intersection / union if union else 0.0,
        max(abs(a - b) for a, b in zip(left, right, strict=True)),
        abs(left_score - right_score),
        left_label == right_label,
    )


def _enforce(metrics: dict[str, float], thresholds: dict[str, float]) -> None:
    failures = []
    for name, limit in thresholds.items():
        passed = metrics[name] <= limit if name.startswith("maximum_") else metrics[name] >= limit
        if not passed:
            failures.append(f"{name}={metrics[name]} threshold={limit}")
    if failures:
        raise QualificationError("SAM2 L4 local accuracy failed: " + "; ".join(failures))


def run(probe: Path, bundle: Path, runtime_root: Path, fixture_dir: Path) -> dict:
    manifest, thresholds = _contract()
    probe = _regular(probe, "SAM2 local probe")
    bundle = _regular(bundle, "SAM2 local bundle")
    if bundle.name != manifest["bundle"]:
        raise QualificationError(f"SAM2 local bundle must be named {manifest['bundle']}")
    runtime_root = _directory(runtime_root, "SAM2 runtime root")
    _regular(runtime_root / "libtrtmc_backend_trt.so", "TensorRT backend")
    _regular(runtime_root / "libtrtmc_model_sam2.so", "SAM2 family library")
    fixture_dir = _directory(fixture_dir, "SAM2 fixture root")
    frames = _frames(fixture_dir)
    golden_bbox, golden_masks = _golden(fixture_dir)
    with tempfile.TemporaryDirectory(prefix="trtmc-sam2-l4-local-") as directory:
        masks = Path(directory) / "candidate.u8"
        completed = subprocess.run(
            [
                str(probe),
                str(bundle),
                str(runtime_root),
                str(masks),
                *(str(path) for path in frames),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        receipt = json.loads(completed.stdout)
        candidate_masks = _regular(
            masks, "SAM2 candidate masks", size=_FRAME_COUNT * _FRAME_PIXELS
        ).read_bytes()
    required = {
        "same_session_repeat_exact",
        "bbox_xyxy",
        "detector_score",
        "label",
        "binary_masks",
        "mask_foreground_pixels",
        "temporally_distinct_masks",
        "metadata_exact",
        "device_mask_ordinal",
        "device_metadata_exact",
        "device_masks_match_host",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise QualificationError("SAM2 local probe returned an invalid receipt")
    coordinates = receipt["bbox_xyxy"]
    if not isinstance(coordinates, list) or len(coordinates) != 4:
        raise QualificationError("SAM2 local probe returned an invalid bbox")
    candidate_bbox = (
        tuple(float(value) for value in coordinates),
        float(receipt["detector_score"]),
        receipt["label"],
    )
    frame_iou, macro_iou, global_iou = _mask_accuracy(candidate_masks, golden_masks)
    bbox_iou, coordinate_error, score_error, label_exact = _bbox_accuracy(
        candidate_bbox, golden_bbox
    )
    metrics = {
        "minimum_frame_mask_iou": min(frame_iou),
        "minimum_macro_mask_iou": macro_iou,
        "minimum_global_mask_iou": global_iou,
        "minimum_bbox_iou": bbox_iou,
        "maximum_bbox_coordinate_error": coordinate_error,
        "maximum_bbox_score_error": score_error,
        "label_exact": 1.0 if label_exact else 0.0,
    }
    if (
        receipt["same_session_repeat_exact"] is not True
        or receipt["binary_masks"] is not True
        or receipt["temporally_distinct_masks"] is not True
        or type(receipt["device_mask_ordinal"]) is not int
        or receipt["device_mask_ordinal"] < 0
        or receipt["device_metadata_exact"] is not True
        or receipt["device_masks_match_host"] is not True
    ):
        raise QualificationError("SAM2 local probe runtime invariants failed")
    _enforce(metrics, thresholds)
    return {"name": manifest["name"], "status": "passed", "metrics": metrics}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                arguments.probe,
                arguments.bundle,
                arguments.runtime_root,
                arguments.fixture_dir,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
