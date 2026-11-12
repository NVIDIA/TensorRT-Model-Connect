# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native byte-rounding and pass order with the pinned Pillow reference."""

import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest


@pytest.mark.parametrize("height,width", [(4000, 1000), (1000, 4000), (2400, 1200), (2401, 1200)])
def test_native_resize_matches_pillow(height, width, tmp_path):
    root = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if not root:
        pytest.skip("requires the family-owned preprocessing consumer")
    consumer = Path(root) / "test_table_transformer_preprocess_consumer"
    assert consumer.is_file()
    pixels = np.random.default_rng(2305).integers(0, 256, (height, width, 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    raw, output = tmp_path / "input.f32", tmp_path / "output.f32"
    (pixels.astype(np.float32) / np.float32(255)).tofile(raw)
    subprocess.run(
        [str(consumer), str(raw), str(output), str(height), str(width)], check=True, timeout=60
    )
    from transformers import DetrImageProcessor

    processor = DetrImageProcessor(size={"shortest_edge": 800, "longest_edge": 1000})
    expected = processor(images=source, return_tensors="np")["pixel_values"][0]
    actual = np.fromfile(output, dtype=np.float32).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
