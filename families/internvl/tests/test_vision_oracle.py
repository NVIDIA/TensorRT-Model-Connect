# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from PIL import Image

from families.internvl.tests.test_e2e import CASES
from families.internvl.tests.vision_oracle import (
    VISION_FEATURE_COSINE,
    _bundle_section,
    _native_pixels,
    assert_vision_parity,
)


def test_every_internvl_case_requires_the_vision_oracle() -> None:
    assert set(CASES) == {
        "internvl3-2b",
        "internvl3-2b-tp2",
        "internvl3-8b",
        "internvl3-8b-tp4",
    }


def _bundle(path, sections: dict[str, bytes]) -> None:
    offset = 0
    table = {}
    for name, data in sections.items():
        table[name] = {"offset": offset, "length": len(data)}
        offset += len(data)
    header = json.dumps(
        {
            "format": 1,
            "family": "internvl",
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


def test_native_pixels_follow_the_family_runtime_contract(tmp_path) -> None:
    image = tmp_path / "image.png"
    Image.fromarray(np.full((2, 4, 3), 255, dtype=np.uint8)).save(image)
    pixels = _native_pixels(
        image,
        {
            "fixed_image_size": 4,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
    )
    assert pixels.shape == (3, 4, 4)
    assert np.allclose(pixels, 1.0)


def test_vision_gate_is_the_active_half_cosine_contract() -> None:
    assert VISION_FEATURE_COSINE == 0.5
    assert assert_vision_parity([[1.0, 1.0]], [[1.0, 0.0]]) > 0.5
    with pytest.raises(AssertionError):
        assert_vision_parity([[1.0, 0.0]], [[-1.0, 0.0]])


@pytest.mark.parametrize("features", [[0.0, 0.0], [float("nan"), 1.0]])
def test_vision_gate_rejects_invalid_native_features(features) -> None:
    with pytest.raises(AssertionError):
        assert_vision_parity([features], [[1.0, 1.0]])
