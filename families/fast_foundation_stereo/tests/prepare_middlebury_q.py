# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned Middlebury-v3 trainingQ data for the 700x700 profile."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


DATA_URL = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip"
GROUND_TRUTH_URL = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip"
SCENES = (
    "Adirondack",
    "ArtL",
    "Jadeplant",
    "Motorcycle",
    "MotorcycleE",
    "Piano",
    "PianoL",
    "Pipes",
    "Playroom",
    "Playtable",
    "PlaytableP",
    "Recycle",
    "Shelves",
    "Teddy",
    "Vintage",
)
TARGET_HEIGHT = 700
TARGET_WIDTH = 700


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)
    return destination


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"archive contains unsafe path: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, target.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)


def read_pfm(path: Path) -> np.ndarray:
    """Read one grayscale PFM file and return top-to-bottom float32 pixels."""
    with path.open("rb") as source:
        if source.readline().strip() != b"Pf":
            raise ValueError(f"{path} is not a grayscale PFM")
        dimensions = source.readline().split()
        if len(dimensions) != 2:
            raise ValueError(f"{path} has invalid PFM dimensions")
        width, height = (int(value) for value in dimensions)
        scale = float(source.readline().strip())
        endian = "<" if scale < 0 else ">"
        payload = source.read()
    expected_bytes = width * height * 4
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{path} has invalid PFM payload size: expected {expected_bytes}, got {len(payload)}"
        )
    values = np.frombuffer(payload, dtype=f"{endian}f4").reshape(height, width)
    return np.flipud(values).astype(np.float32, copy=True)


def _transform_plan(
    height: int,
    width: int,
    *,
    target_height: int = TARGET_HEIGHT,
    target_width: int = TARGET_WIDTH,
) -> dict[str, int]:
    if height > target_height:
        raise ValueError(
            f"Middlebury-Q height {height} exceeds fixed profile height {target_height}"
        )
    crop_total = max(width - target_width, 0)
    crop_left = crop_total // 2
    crop_right = crop_total - crop_left
    cropped_width = width - crop_total
    pad_width = target_width - cropped_width
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_height = target_height - height
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    return {
        "crop_left": crop_left,
        "crop_right": crop_right,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }


def transform_scene(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    nonocc_mask: np.ndarray,
    *,
    target_height: int = TARGET_HEIGHT,
    target_width: int = TARGET_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Apply the approved no-resize crop/pad transform to one stereo scene."""
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ValueError("stereo inputs must be equal-shape HWC RGB arrays")
    height, width = left.shape[:2]
    if disparity.shape != (height, width) or nonocc_mask.shape != (height, width):
        raise ValueError("ground truth and non-occluded mask must match stereo dimensions")
    plan = _transform_plan(
        height,
        width,
        target_height=target_height,
        target_width=target_width,
    )
    start = plan["crop_left"]
    stop = width - plan["crop_right"]
    image_pad = (
        (plan["pad_top"], plan["pad_bottom"]),
        (plan["pad_left"], plan["pad_right"]),
        (0, 0),
    )
    field_pad = image_pad[:2]
    left_out = np.pad(left[:, start:stop], image_pad, mode="edge")
    right_out = np.pad(right[:, start:stop], image_pad, mode="edge")
    disparity_out = np.pad(
        disparity[:, start:stop].astype(np.float32, copy=False),
        field_pad,
        mode="constant",
        constant_values=0,
    )
    valid = nonocc_mask[:, start:stop].astype(bool, copy=False)
    valid &= np.isfinite(disparity[:, start:stop])
    valid_out = np.pad(valid, field_pad, mode="constant", constant_values=False)
    expected = (target_height, target_width)
    if left_out.shape[:2] != expected or disparity_out.shape != expected:
        raise RuntimeError("Middlebury transform did not produce the fixed profile shape")
    return left_out, right_out, disparity_out, valid_out, plan


def prepare_archives(
    data_archive: Path,
    ground_truth_archive: Path,
    output: Path,
    *,
    target_height: int = TARGET_HEIGHT,
    target_width: int = TARGET_WIDTH,
) -> Path:
    """Extract, transform, and describe all 15 official trainingQ scenes."""
    source_root = output / "source"
    _safe_extract(data_archive, source_root)
    _safe_extract(ground_truth_archive, source_root)
    training_root = source_root / "MiddEval3" / "trainingQ"
    discovered = tuple(sorted(path.name for path in training_root.iterdir() if path.is_dir()))
    if discovered != tuple(sorted(SCENES)):
        raise ValueError(
            "Middlebury trainingQ scene set differs from the pinned 15-scene contract: "
            f"expected={list(sorted(SCENES))}, actual={list(discovered)}"
        )

    requests: list[dict[str, Any]] = []
    prepared_root = output / "prepared"
    for scene in sorted(SCENES):
        source_scene = training_root / scene
        with Image.open(source_scene / "im0.png") as image:
            left = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(source_scene / "im1.png") as image:
            right = np.asarray(image.convert("RGB"), dtype=np.uint8)
        disparity = read_pfm(source_scene / "disp0GT.pfm")
        with Image.open(source_scene / "mask0nocc.png") as image:
            # MiddEval3 SDK evaldisp.cpp evaluates exactly mask value 255;
            # value 128 is deliberately outside the non-occluded mask.
            nonocc = np.asarray(image.convert("L"), dtype=np.uint8) == 255
        original_height, original_width = left.shape[:2]
        left, right, disparity, valid, transform = transform_scene(
            left,
            right,
            disparity,
            nonocc,
            target_height=target_height,
            target_width=target_width,
        )
        destination = prepared_root / scene
        destination.mkdir(parents=True, exist_ok=True)
        left_path = destination / "im0.png"
        right_path = destination / "im1.png"
        disparity_path = destination / "disp0GT.npy"
        mask_path = destination / "valid_nonocc.npy"
        Image.fromarray(left, mode="RGB").save(left_path)
        Image.fromarray(right, mode="RGB").save(right_path)
        np.save(disparity_path, disparity, allow_pickle=False)
        np.save(mask_path, valid, allow_pickle=False)
        requests.append(
            {
                "sample_id": f"middlebury-q-{scene.casefold()}",
                "testcase": "fast-foundation-stereo-middlebury-q",
                "stage": "full_inference",
                "category": "middlebury-v3-trainingQ-profile-700x700",
                "inputs": {
                    "fixture": "middlebury-v3-trainingQ",
                    "scene": scene,
                    "left_image": str(left_path.resolve()),
                    "right_image": str(right_path.resolve()),
                    "ground_truth_disparity": str(disparity_path.resolve()),
                    "valid_nonocc_mask": str(mask_path.resolve()),
                    "original_height": original_height,
                    "original_width": original_width,
                    "target_height": target_height,
                    "target_width": target_width,
                    "transform": transform,
                },
            }
        )
    manifest = output / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "trtmc.model-plugin-validation/v1",
                "dataset": "Middlebury Stereo Evaluation v3 quarter-resolution training set",
                "version": "MiddEval3-trainingQ-profile-700x700-v1",
                "license": (
                    "Middlebury Stereo Vision website permission to use and publish its "
                    "images and numerical results with citation; no SPDX dataset license stated"
                ),
                "source": "https://vision.middlebury.edu/stereo/submit3/",
                "archives": {
                    "data": {
                        "url": DATA_URL,
                        "path": str(data_archive.resolve()),
                    },
                    "ground_truth": {
                        "url": GROUND_TRUTH_URL,
                        "path": str(ground_truth_archive.resolve()),
                    },
                },
                "preparation": {
                    "resize": False,
                    "width_crop": "symmetric-center",
                    "image_padding": "symmetric-edge",
                    "ground_truth_padding": "zero-invalid",
                    "target_shape": [target_height, target_width],
                    "scene_order": "lexical",
                },
                "requests": requests,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-archive", type=Path)
    parser.add_argument("--ground-truth-archive", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    archive_root = arguments.output / "archives"
    data = arguments.data_archive or archive_root / "MiddEval3-data-Q.zip"
    ground_truth = arguments.ground_truth_archive or archive_root / "MiddEval3-GT0-Q.zip"
    if arguments.data_archive is None and not data.is_file():
        _download(DATA_URL, data)
    if arguments.ground_truth_archive is None and not ground_truth.is_file():
        _download(GROUND_TRUTH_URL, ground_truth)
    manifest = prepare_archives(data, ground_truth, arguments.output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
