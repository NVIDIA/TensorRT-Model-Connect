# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from PIL import Image

from families.deepseek_ocr.tests.test_e2e import CASES
from families.deepseek_ocr.tests.vision_oracle import (
    VISION_FEATURE_COSINE,
    _bundle_section,
    _native_pixels,
    assert_vision_parity,
)


def test_every_deepseek_ocr_case_requires_the_vision_oracle() -> None:
    assert set(CASES) == {"deepseek-ocr", "deepseek-ocr-l0", "deepseek-ocr-l0-tp2"}


def _bundle(path, sections: dict[str, bytes]) -> None:
    offset = 0
    table = {}
    for name, data in sections.items():
        table[name] = {"offset": offset, "length": len(data)}
        offset += len(data)
    header = json.dumps(
        {
            "format": 1,
            "family": "deepseek_ocr",
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


def test_native_pixels_follow_the_family_runtime_contract() -> None:
    image = Image.fromarray(np.full((2, 4, 3), 255, dtype=np.uint8))
    pixels = _native_pixels(
        image,
        {
            "fixed_image_size": 4,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
    )
    assert pixels.shape == (1, 3, 4, 4)
    assert np.allclose(pixels, 1.0)


def test_vision_gate_is_the_active_half_cosine_contract() -> None:
    assert VISION_FEATURE_COSINE == 0.5
    assert assert_vision_parity([[1.0, 1.0]], [[1.0, 0.0]]) > 0.5
    with pytest.raises(AssertionError):
        assert_vision_parity([[1.0, 0.0]], [[-1.0, 0.0]])


@pytest.mark.parametrize("features", [[0.0, 0.0], [float("inf"), 1.0]])
def test_vision_gate_rejects_invalid_native_features(features) -> None:
    with pytest.raises(AssertionError):
        assert_vision_parity([features], [[1.0, 1.0]])
