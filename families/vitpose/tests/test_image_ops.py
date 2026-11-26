# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native box crop and heatmap decoding against the reference image processor, on the CPU.

The family's `vitpose_image_ops_tool` (built with the CTest targets) runs the runtime's own code on raw files. The
crop and its pixel values must match `VitPoseImageProcessorFast` (the processor `AutoProcessor` loads for this
checkpoint) bit for bit; the decoded keypoints must match `post_process_pose_estimation` to float32 rounding.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

# The checkpoint's preprocessor_config.json: a 256 x 192 crop, 1/255 rescale and the ImageNet statistics.
_HEIGHT, _WIDTH = 256, 192
_RESCALE = 0.00392156862745098
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
# box_to_center_and_scale's fixed padding and the DARK kernel of post_process_pose_estimation.
_PADDING = 1.25
_KERNEL = 11
# (image height, image width, COCO box): the whole image, boxes inside it, boxes that reach past its edges, a box
# wider and one taller than the crop's 3:4 aspect, sub-pixel and one-pixel images.
_CROPS = (
    (480, 640, (0.0, 0.0, 640.0, 480.0)),
    (480, 640, (412.8, 157.61, 53.05, 138.01)),
    (480, 640, (384.43, 172.21, 15.12, 35.74)),
    (300, 200, (0.0, 0.0, 200.0, 300.0)),
    (256, 192, (0.0, 0.0, 192.0, 256.0)),
    (97, 1203, (5.5, 1.25, 1000.0, 90.0)),
    (1203, 97, (0.0, 3.0, 97.0, 1100.5)),
    (64, 64, (60.0, 60.0, 4.0, 4.0)),
    (7, 5, (0.0, 0.0, 5.0, 7.0)),
    (1, 1, (0.0, 0.0, 1.0, 1.0)),
)


def _tool() -> Path:
    build = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    assert build, "the image-ops parity test requires TRTMC_NATIVE_BUILD_DIR"
    tool = Path(build) / "vitpose_image_ops_tool"
    assert tool.is_file(), f"build the family's CTest targets first: {tool}"
    return tool


def _run(*arguments) -> str:
    completed = subprocess.run(
        [str(_tool()), *(str(value) for value in arguments)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.stdout


def _processor():
    from transformers import VitPoseImageProcessorFast

    return VitPoseImageProcessorFast(
        do_affine_transform=True,
        size={"height": _HEIGHT, "width": _WIDTH},
        do_rescale=True,
        rescale_factor=_RESCALE,
        do_normalize=True,
        image_mean=list(_MEAN),
        image_std=list(_STD),
    )


def _image(seed: int, height: int, width: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _preprocess(image: np.ndarray, box, tmp_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    image.tofile(tmp_path / "in.u8")
    stdout = _run(
        "preprocess",
        tmp_path / "in.u8",
        *image.shape[:2],
        *map(repr, box),
        _HEIGHT,
        _WIDTH,
        repr(_PADDING),
        repr(_RESCALE),
        *map(repr, _MEAN),
        *map(repr, _STD),
        tmp_path / "crop.u8",
        tmp_path / "pixels.f32",
    )
    crop = np.fromfile(tmp_path / "crop.u8", dtype=np.uint8).reshape(_HEIGHT, _WIDTH, 3)
    pixels = np.fromfile(tmp_path / "pixels.f32", dtype=np.float32).reshape(3, _HEIGHT, _WIDTH)
    return json.loads(stdout), crop, pixels


@pytest.mark.parametrize(
    "height,width,box", _CROPS, ids=[f"{h}x{w}-{i}" for i, (h, w, _) in enumerate(_CROPS)]
)
def test_preprocess_matches_the_reference_processor(height, width, box, tmp_path: Path) -> None:
    import torch
    from PIL import Image
    from transformers.image_utils import SizeDict
    from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale

    image = _image(height * 7919 + width, height, width)
    geometry, crop, pixels = _preprocess(image, box, tmp_path)

    center, scale = box_to_center_and_scale(list(box), image_width=_WIDTH, image_height=_HEIGHT)
    assert np.array_equal(np.asarray(geometry["center"], dtype=np.float32), center)
    assert np.array_equal(np.asarray(geometry["scale"], dtype=np.float32), scale)

    processor = _processor()
    expected = processor(Image.fromarray(image), boxes=[[list(box)]], return_tensors="pt")
    expected = expected["pixel_values"][0].numpy()
    assert expected.dtype == np.float32 and expected.shape == pixels.shape
    assert np.array_equal(pixels, expected)
    # The 8-bit crop is the processor's own warp of the same box.
    warped = processor.torch_affine_transform(
        torch.from_numpy(image).permute(2, 0, 1),
        center,
        scale,
        rotation=0,
        size=SizeDict(height=_HEIGHT, width=_WIDTH),
    )
    assert np.array_equal(crop, warped.permute(1, 2, 0).numpy())


def test_preprocess_rounds_a_half_up_like_the_reference(tmp_path: Path) -> None:
    # On a 0/1 checkerboard this box puts samples exactly half way between a 0 and a 1. scipy stores the 0.5 as 1
    # (half up); rounding half to even or truncating would store 0.
    from PIL import Image
    from scipy.linalg import inv
    from scipy.ndimage import affine_transform
    from transformers.models.vitpose.image_processing_vitpose import (
        box_to_center_and_scale,
        get_warp_matrix,
    )

    image = np.repeat((np.indices((16, 16)).sum(axis=0) % 2).astype(np.uint8)[..., None], 3, axis=2)
    box = (0.0, 0.0, 3.0, 4.0)
    _, crop, pixels = _preprocess(image, box, tmp_path)
    expected = _processor()(Image.fromarray(image), boxes=[[list(box)]], return_tensors="pt")
    assert np.array_equal(pixels, expected["pixel_values"][0].numpy())

    # The same warp left unrounded, as scipy_warp_affine builds it: the exact half-way samples exist, and became 1.
    center, scale = box_to_center_and_scale(list(box), image_width=_WIDTH, image_height=_HEIGHT)
    matrix = inv(
        np.vstack(
            [
                get_warp_matrix(0, center * 2.0, np.array((_WIDTH, _HEIGHT)) - 1.0, scale * 200.0),
                [0, 0, 1],
            ]
        )
    )
    matrix = matrix[[1, 0, 2]][:, [1, 0, 2]]
    unrounded = affine_transform(
        image[..., 0].astype(np.float64), matrix, output_shape=(_HEIGHT, _WIDTH), order=1
    )
    halves = unrounded == 0.5
    assert halves.sum() >= 10
    assert np.all(crop[..., 0][halves] == 1)


def _heatmaps(seed: int, keypoints: int, height: int, width: int) -> np.ndarray:
    """Gaussian peaks of trained-heatmap width (sigma 2) at sub-pixel centres, plus noise and negative valleys."""

    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float64)
    maps = np.empty((keypoints, height, width), dtype=np.float32)
    for index in range(keypoints):
        y, x = rng.uniform(-2.0, height + 1.0), rng.uniform(-2.0, width + 1.0)
        peak = rng.uniform(0.05, 1.0)
        blob = peak * np.exp(-((rows - y) ** 2 + (columns - x) ** 2) / (2.0 * 2.0**2))
        maps[index] = (blob + rng.normal(0.0, 0.01, size=blob.shape) - 0.02).astype(np.float32)
    return maps


@pytest.mark.parametrize("seed", range(6))
def test_decode_matches_the_reference_postprocessing(seed: int, tmp_path: Path) -> None:
    import torch
    from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale
    from transformers.models.vitpose.modeling_vitpose import VitPoseEstimatorOutput

    keypoints, height, width = 17, 64, 48
    maps = _heatmaps(seed, keypoints, height, width)
    if seed == 0:
        maps[3] = -np.abs(maps[3]) - 0.5  # no positive value: the reference marks the keypoint -1
        maps[4] = 0.25  # flat: the first cell, refined by a zero gradient
    box = [(0.0, 0.0, 640.0, 480.0), (412.8, 157.61, 53.05, 138.01), (3.0, 4.0, 5.0, 900.0)][
        seed % 3
    ]
    center, scale = box_to_center_and_scale(list(box), image_width=_WIDTH, image_height=_HEIGHT)
    maps.tofile(tmp_path / "heatmaps.f32")
    _run(
        "decode",
        tmp_path / "heatmaps.f32",
        keypoints,
        height,
        width,
        *(repr(float(value)) for value in (*center, *scale)),
        _KERNEL,
        tmp_path / "keypoints.f32",
    )
    actual = np.fromfile(tmp_path / "keypoints.f32", dtype=np.float32).reshape(keypoints, 4)

    outputs = VitPoseEstimatorOutput(heatmaps=torch.from_numpy(maps.copy())[None])
    expected = _processor().post_process_pose_estimation(
        outputs, boxes=[[list(box)]], kernel_size=_KERNEL
    )[0][0]
    scores = expected["scores"].numpy()
    located = actual[:, 3] == 1.0
    assert np.array_equal(located, scores > 0.0)
    assert np.array_equal(actual[:, 2], scores)
    if seed == 0:
        assert not located[3] and located[4]
    reference = expected["keypoints"].numpy()[located]
    # Equal but for the last bits: numpy's SIMD exp and log and its LAPACK inverse may round differently from the C
    # library. Measured on 200 such heatmap sets (3,400 keypoints): 3,302 equal, the rest within 4.2e-6 relative.
    np.testing.assert_allclose(actual[located, :2], reference, rtol=2e-5, atol=1e-4)
