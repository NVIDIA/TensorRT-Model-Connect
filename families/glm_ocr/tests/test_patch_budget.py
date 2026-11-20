# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vision patch budget of a bundle built for a declared image size."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from families.glm_ocr.image_budget import patch_budget, smart_resize

# The checkpoint's processor (preprocessor_config.json of zai-org/GLM-OCR).
CONFIG = SimpleNamespace(
    processor=SimpleNamespace(patch_size=14, merge_size=2, temporal_patch_size=2, min_pixels=12544,
                              max_pixels=9633792),
    max_image_patches=24576,
)

# Transformers 5.22.2 smart_resize of the checkpoint's processor (the runtime's unit test uses the same cases);
# None where the processor refuses the aspect ratio.
_RESIZE_CASES = [
    ((56, 840), (56, 840)), ((448, 924), (448, 924)), ((468, 931), (476, 924)), ((14, 14), (84, 84)),
    ((27, 300), (28, 308)), ((70, 70), (84, 84)), ((42, 42), (84, 84)), ((1000, 3), None),
    ((5000, 5000), (2184, 2184)), ((3000, 2000), (2688, 1792)), ((2193, 2193), (2184, 2184)),
    ((150, 150), (140, 140)), ((98, 182), (112, 168)), ((1, 250), None), ((300, 1), None), ((12, 2500), None),
    ((140, 210), (140, 224)), ((500, 30), (504, 28)),
]


@pytest.mark.parametrize(("size", "expected"), _RESIZE_CASES)
def test_smart_resize_matches_the_processor(size, expected) -> None:
    assert smart_resize(*size, CONFIG) == expected


def _patches(height: int, width: int) -> int:
    size = smart_resize(height, width, CONFIG)
    return 0 if size is None else (size[0] // 14) * (size[1] // 14)


@pytest.mark.parametrize("declared", [(28, 28), (14, 14), (56, 840), (30, 400), (100, 60), (230, 41)])
def test_every_image_up_to_the_declared_size_fits_the_budget(declared) -> None:
    budget = patch_budget(*declared, CONFIG)
    largest = max(_patches(h, w) for h in range(1, declared[0] + 1) for w in range(1, declared[1] + 1))
    assert largest <= budget <= CONFIG.max_image_patches


def test_small_declared_sizes_include_the_min_pixels_enlargement() -> None:
    # A 28 x 28 image is enlarged to 84 x 84 = 36 patches; the plain rounding would allow 4.
    assert _patches(28, 28) == 36
    assert patch_budget(28, 28, CONFIG) >= 36


def test_large_declared_sizes_are_capped() -> None:
    assert patch_budget(5000, 5000, CONFIG) == CONFIG.max_image_patches
