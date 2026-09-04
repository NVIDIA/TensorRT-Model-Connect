# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from PIL import Image

from families.qwen_vl.tests.vision_oracle import (
    VISION_FEATURE_COSINE,
    _bundle_section,
    _merge_group_pixels,
    assert_vision_parity,
)


def _bundle(path, sections: dict[str, bytes]) -> None:
    offset = 0
    table = {}
    for name, data in sections.items():
        table[name] = {"offset": offset, "length": len(data)}
        offset += len(data)
    header = json.dumps(
        {
            "format": 1,
            "family": "qwen_vl",
            "task": "vision_language_generation",
            "backend": "trt",
            "sections": table,
        },
        separators=(",", ":"),
    ).encode()
    path.write_bytes(
        b"BUNDLE\x01\x00" + struct.pack("<Q", len(header)) + header + b"".join(sections.values())
    )


def test_bundle_reader_selects_the_family_vision_plan(tmp_path) -> None:
    bundle = tmp_path / "model.bundle"
    _bundle(bundle, {"vision.plan": b"VISION", "runtime.json": b"{}"})
    assert _bundle_section(bundle, "vision.plan") == b"VISION"


def test_merge_group_pixels_follow_qwen_vision_order(tmp_path) -> None:
    red = np.arange(16, dtype=np.uint8).reshape(4, 4)
    image = np.stack((red, np.zeros_like(red), np.zeros_like(red)), axis=-1)
    image_path = tmp_path / "image.png"
    Image.fromarray(image).save(image_path)
    pixels = _merge_group_pixels(
        image_path,
        {
            "fixed_image_size": 4,
            "fixed_image_height": 4,
            "fixed_image_width": 4,
            "patch_size": 1,
            "merge_size": 2,
            "temporal_patch_size": 1,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [1.0, 1.0, 1.0],
        },
    )
    assert pixels.shape == (3, 4, 4)
    expected = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    assert np.allclose(pixels[0] * 255.0, expected)


def test_vision_gate_is_the_active_half_cosine_contract() -> None:
    assert VISION_FEATURE_COSINE == 0.5
    assert assert_vision_parity([[1.0, 1.0]], [[1.0, 0.0]]) > 0.5
    with pytest.raises(AssertionError):
        assert_vision_parity([[1.0, 0.0]], [[-1.0, 0.0]])


@pytest.mark.parametrize("features", [[0.0, 0.0], [float("nan"), 1.0]])
def test_vision_gate_rejects_invalid_native_features(features) -> None:
    with pytest.raises(AssertionError):
        assert_vision_parity([features], [[1.0, 1.0]])
