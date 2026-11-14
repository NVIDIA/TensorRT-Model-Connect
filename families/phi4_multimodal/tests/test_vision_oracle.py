# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from families.phi4_multimodal.tests.test_e2e import CASES
from families.phi4_multimodal.tests.vision_oracle import (
    VISION_FEATURE_COSINE,
    _bundle_section,
    _native_dynamic_hd_inputs,
    assert_vision_parity,
)


def test_phi4_multimodal_case_requires_the_vision_oracle() -> None:
    assert set(CASES) == {"phi4-multimodal"}


def test_checkpoint_reference_declares_scipy() -> None:
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    packages = {
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "scipy" in packages


def _bundle(path, sections: dict[str, bytes]) -> None:
    offset = 0
    table = {}
    for name, data in sections.items():
        table[name] = {"offset": offset, "length": len(data)}
        offset += len(data)
    header = json.dumps(
        {
            "format": 1,
            "family": "phi4_multimodal",
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


_DYNAMIC_HD = Path(__file__).resolve().parent / "data/dynamic-hd"
_DYNAMIC_HD_CASES = json.loads((_DYNAMIC_HD / "hf-dynamic-hd.json").read_text(encoding="utf-8"))[
    "cases"
]


@pytest.mark.parametrize("case", _DYNAMIC_HD_CASES, ids=lambda case: case["image"])
def test_native_inputs_match_the_checkpoint_dynamic_hd(case) -> None:
    with Image.open(_DYNAMIC_HD / case["image"]) as image:
        inputs = _native_dynamic_hd_inputs(
            image,
            {
                "fixed_image_size": 448,
                "preprocessor_type": "phi4_hd_chw",
                "interpolation": "bilinear",
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            },
        )
    cols, rows = case["grid"]
    entries = 1 + cols * rows
    pixels = inputs["pixel_values"]
    assert pixels.shape == (entries, 3, 448, 448) and pixels.dtype == np.float32
    assert inputs["crop_valid"].dtype == np.int32
    assert inputs["crop_valid"].tolist() == case["crop_valid"]
    for entry, channel, y, x, expected in case["pixel_samples"]:
        # The crops are 8-bit resizes: exact. The global view is a float bicubic resize.
        tolerance = 1e-4 if entry == 0 else 0.0
        assert abs(float(pixels[entry, channel, y, x]) - expected) <= tolerance

    token_rows = inputs["image_token_rows"]
    assert token_rows.dtype == np.int32
    assert token_rows.shape == (case["num_image_tokens"],)
    features = token_rows[token_rows >= 2]
    assert len(np.unique(features)) == len(features)
    assert features.max() < 2 + entries * 16 * 16
    assert np.count_nonzero(token_rows == 1) == 1


def test_vision_gate_is_the_active_half_cosine_contract() -> None:
    assert VISION_FEATURE_COSINE == 0.5
    assert assert_vision_parity([[1.0, 1.0]], [[1.0, 0.0]]) > 0.5
    with pytest.raises(AssertionError):
        assert_vision_parity([[1.0, 0.0]], [[-1.0, 0.0]])


@pytest.mark.parametrize("features", [[0.0, 0.0], [float("nan"), 1.0]])
def test_vision_gate_rejects_invalid_native_features(features) -> None:
    with pytest.raises(AssertionError):
        assert_vision_parity([features], [[1.0, 1.0]])
