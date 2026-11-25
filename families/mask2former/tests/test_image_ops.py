# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native image operations against Pillow, the reference image processor and torch, on the CPU.

The family's `mask2former_image_ops_tool` (built with the CTest targets) runs the runtime's own code on raw files.
Resizing and preprocessing must match bit for bit; the resized argmax must match wherever the reference's best
class is not a near-tie.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

# The checkpoint's preprocessor_config.json: a fixed 384 x 384 bilinear resize, 1/255 rescale, ImageNet statistics.
_SIZE = 384
_RESCALE = 0.00392156862745098
_MEAN = (0.48500001430511475, 0.4560000002384186, 0.4059999883174896)
_STD = (0.2290000021457672, 0.2239999920129776, 0.22499999403953552)
# (height, width) -> (height, width): Pillow 13 runs the horizontal pass first unless the height shrinks by more
# than twice the width's reduction; both orders, growth, shrinkage, identity and one-pixel edges are covered.
_RESIZES = (
    ((382, 640), (_SIZE, _SIZE)),
    ((1000, 600), (_SIZE, _SIZE)),
    ((900, 300), (_SIZE, _SIZE)),
    ((_SIZE, _SIZE), (_SIZE, _SIZE)),
    ((97, 1203), (_SIZE, _SIZE)),
    ((1, 1), (_SIZE, _SIZE)),
    ((5, 3), (2, 7)),
    ((2, 400), (1, 9)),
)


def _tool() -> Path:
    build = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    assert build, "the image-ops parity test requires TRTMC_NATIVE_BUILD_DIR"
    tool = Path(build) / "mask2former_image_ops_tool"
    assert tool.is_file(), f"build the family's CTest targets first: {tool}"
    return tool


def _run(*arguments) -> None:
    subprocess.run([str(_tool()), *(str(value) for value in arguments)], check=True, timeout=300)


def _image(seed: int, height: int, width: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "source,target", _RESIZES, ids=[f"{s[0]}x{s[1]}-{t[0]}x{t[1]}" for s, t in _RESIZES]
)
def test_resize_matches_pillow(source, target, tmp_path: Path) -> None:
    from PIL import Image

    image = _image(source[0] * 7919 + source[1], *source)
    image.tofile(tmp_path / "in.u8")
    _run("resize", tmp_path / "in.u8", *source, *target, tmp_path / "out.u8")
    actual = np.fromfile(tmp_path / "out.u8", dtype=np.uint8).reshape(target + (3,))
    expected = np.asarray(
        Image.fromarray(image).resize((target[1], target[0]), Image.Resampling.BILINEAR)
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("source", [source for source, _ in _RESIZES[:5]])
def test_preprocess_matches_the_reference_processor(source, tmp_path: Path) -> None:
    from PIL import Image
    from transformers import Mask2FormerImageProcessor

    processor = Mask2FormerImageProcessor(
        do_resize=True,
        size={"height": _SIZE, "width": _SIZE},
        size_divisor=32,
        resample=Image.Resampling.BILINEAR,
        do_rescale=True,
        rescale_factor=_RESCALE,
        do_normalize=True,
        image_mean=list(_MEAN),
        image_std=list(_STD),
    )
    image = _image(source[0] * 31 + source[1], *source)
    expected = processor(images=Image.fromarray(image), return_tensors="np")["pixel_values"][0]
    image.tofile(tmp_path / "in.u8")
    _run(
        "preprocess",
        tmp_path / "in.u8",
        *source,
        _SIZE,
        _SIZE,
        repr(_RESCALE),
        *map(repr, _MEAN),
        *map(repr, _STD),
        tmp_path / "out.f32",
    )
    actual = np.fromfile(tmp_path / "out.f32", dtype=np.float32).reshape(3, _SIZE, _SIZE)
    assert expected.dtype == np.float32 and expected.shape == actual.shape
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("target", [(382, 640), (_SIZE, _SIZE), (17, 23), (1, 5), (1000, 3)])
def test_resize_argmax_matches_torch(target, tmp_path: Path) -> None:
    import torch

    classes, height, width = 7, 13, 11
    scores = np.random.default_rng(target[0] * 13 + target[1]).random(
        (classes, height, width), dtype=np.float32
    )
    scores.tofile(tmp_path / "scores.f32")
    _run(
        "argmax", tmp_path / "scores.f32", classes, height, width, *target, tmp_path / "labels.i32"
    )
    actual = np.fromfile(tmp_path / "labels.i32", dtype=np.int32).reshape(target)
    resized = torch.nn.functional.interpolate(
        torch.from_numpy(scores)[None], size=target, mode="bilinear", align_corners=False
    )[0]
    expected = resized.argmax(dim=0).numpy()
    top2 = torch.topk(resized, 2, dim=0).values
    decided = (top2[0] - top2[1]).numpy() > 1e-5
    assert decided.mean() > 0.99
    assert np.array_equal(actual[decided], expected[decided])
