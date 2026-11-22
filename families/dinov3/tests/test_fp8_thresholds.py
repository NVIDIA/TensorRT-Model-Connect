# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FP8 feature criteria: the full-feature cosine floor of each FP8 case."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

THRESHOLDS = Path(__file__).resolve().parent / "thresholds"


def _criteria(case: str) -> dict:
    return json.loads((THRESHOLDS / f"{case}.json").read_text())["threshold_overrides"]


def test_convnext_fp8_full_cosine_is_the_cosine_its_relative_error_limit_allows() -> None:
    # The recipe's error spreads over all 18 blocks: on 200 held-out ImageNet images the full cosine has median
    # 0.992 and 1st percentile 0.980 (a PyTorch emulation of the recipe gives the same), and the kNN accuracy of the
    # features is unchanged (see the README). The floor is the cosine the relative Frobenius limit allows at equal
    # norms, so neither criterion is looser than the other.
    criteria = _criteria("dinov3-convnext-tiny-pretrain-lvd1689m-fp8")
    assert criteria["full_cosine"] == pytest.approx(1 - criteria["relative_frobenius"] ** 2 / 2)
    assert criteria["relative_frobenius"] == 0.2
    assert {key: criteria[key] for key in ("cls_cosine", "pooler_cosine", "mean_patch_cosine")} == {
        "cls_cosine": 0.98,
        "pooler_cosine": 0.98,
        "mean_patch_cosine": 0.98,
    }
    assert criteria["p01_patch_cosine"] == 0.9


def test_vit_fp8_keeps_its_full_cosine_floor() -> None:
    assert _criteria("dinov3-vits16-pretrain-lvd1689m-fp8")["full_cosine"] == 0.99
