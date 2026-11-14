# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native pre- and post-processing against the official fast OneFormer processor.

The native runtime reproduces the processor's torchvision steps instead of
approximating them. Its preprocessing must equal ATen's uint8 antialiased
bilinear kernel plus the fused normalization bit for bit on every host, and
the official processor must produce the same ``pixel_values`` here: torchvision
0.27 and later run that kernel for uint8 CPU tensors on AVX2 / AVX-512 x86 and
on aarch64 (elsewhere it resizes in float32, and the second check reports the
host). The float antialiased resize plus argmax must give the processor's
semantic labels exactly. Requires the native build (``TRTMC_NATIVE_BUILD_DIR``
with ``test_oneformer_image_ops``), torch, torchvision and transformers; no GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

TEST_ROOT = Path(__file__).resolve().parent
IMAGE = TEST_ROOT / "data" / "test_img.jpeg"
SIZE = {"shortest_edge": 1024, "longest_edge": 2048}
MEAN = [0.48500001430511475, 0.4560000002384186, 0.4059999883174896]
STD = [0.2290000021457672, 0.2239999920129776, 0.22499999403953552]
RESCALE = 0.00392156862745098


def _tool() -> Path:
    build_dir = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if not build_dir:
        pytest.skip("set TRTMC_NATIVE_BUILD_DIR to a native build with TRTMC_BUILD_TESTS=ON")
    tool = Path(build_dir) / "test_oneformer_image_ops"
    assert tool.is_file(), f"build the family test target {tool.name}"
    return tool


def _processor():
    pytest.importorskip("torchvision")
    from transformers.models.oneformer.image_processing_oneformer_fast import (
        OneFormerImageProcessorFast,
    )

    return OneFormerImageProcessorFast(
        size=SIZE, image_mean=MEAN, image_std=STD, rescale_factor=RESCALE
    )


def _variants():
    from PIL import Image

    with Image.open(IMAGE) as opened:
        base = opened.convert("RGB")
    return {
        "photo": base,
        "portrait": base.transpose(Image.Transpose.ROTATE_90),
        "downscale": base.resize((base.width * 3, base.height * 3), Image.Resampling.BICUBIC),
        "small": base.resize((base.width // 4, base.height // 4), Image.Resampling.BICUBIC),
        "wide": base.crop((0, 100, base.width, 300)),
    }


def _aten_uint8(image, height: int, width: int):
    """ATen's uint8 antialiased bilinear resize plus the processor's fused normalization."""
    import numpy as np
    import torch

    pixels = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1)[None].contiguous()
    resized = torch.nn.functional.interpolate(
        pixels, size=(height, width), mode="bilinear", align_corners=False, antialias=True
    )[0]
    assert resized.dtype == torch.uint8
    mean = torch.tensor(MEAN) * (1.0 / RESCALE)
    std = torch.tensor(STD) * (1.0 / RESCALE)
    return ((resized.to(torch.float32) - mean[:, None, None]) / std[:, None, None]).numpy()


def test_preprocessing_matches_the_official_processor_bit_for_bit(tmp_path: Path) -> None:
    import numpy as np

    tool, processor = _tool(), _processor()
    for name, image in _variants().items():
        expected = processor(images=image, task_inputs=["semantic"], return_tensors="pt")[
            "pixel_values"
        ][0].numpy()
        kernel = _aten_uint8(image, *expected.shape[1:])
        raw = tmp_path / f"{name}.rgb"
        np.ascontiguousarray(np.asarray(image, dtype=np.uint8)).tofile(raw)
        out = tmp_path / f"{name}.f32"
        completed = subprocess.run(
            [
                str(tool),
                "preprocess",
                str(raw),
                str(image.height),
                str(image.width),
                str(SIZE["shortest_edge"]),
                str(SIZE["longest_edge"]),
                *map(repr, MEAN),
                *map(repr, STD),
                repr(RESCALE),
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        size = json.loads(completed.stdout)
        assert (size["height"], size["width"]) == expected.shape[1:], name
        actual = np.fromfile(out, dtype=np.float32).reshape(expected.shape)
        assert np.array_equal(actual, kernel), (
            f"{name}: {int((actual != kernel).sum())} values differ from ATen's uint8 kernel"
        )
        assert np.array_equal(expected, kernel), (
            f"{name}: torchvision on this host does not resize uint8 with ATen's kernel "
            f"({int((expected != kernel).sum())} values differ), so the official processor "
            "differs from the native runtime here"
        )


def test_score_resize_and_argmax_match_the_official_post_processing(tmp_path: Path) -> None:
    import numpy as np
    import torch

    tool, processor = _tool(), _processor()
    generator = torch.Generator().manual_seed(254)
    cases = {
        "upsample": ((96, 160), (382, 640)),
        "downsample": ((256, 429), (200, 300)),
        "mixed": ((64, 512), (300, 100)),
    }
    for name, ((height, width), target) in cases.items():
        logits = torch.randn(1, 50, 20, generator=generator)
        masks = torch.randn(1, 50, height, width, generator=generator) * 4
        outputs = SimpleNamespace(class_queries_logits=logits, masks_queries_logits=masks)
        expected = processor.post_process_semantic_segmentation(outputs, target_sizes=[target])[
            0
        ].numpy()
        scores = torch.einsum(
            "bqc, bqhw -> bchw", logits.softmax(dim=-1)[..., :-1], masks.sigmoid()
        )[0]
        raw = tmp_path / f"{name}.scores"
        scores.contiguous().numpy().astype(np.float32).tofile(raw)
        out = tmp_path / f"{name}.labels"
        subprocess.run(
            [
                str(tool),
                "argmax",
                str(raw),
                "19",
                str(height),
                str(width),
                str(target[0]),
                str(target[1]),
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        actual = np.fromfile(out, dtype=np.int32).reshape(target)
        assert np.array_equal(actual, expected), (
            f"{name}: {int((actual != expected).sum())} labels differ"
        )
