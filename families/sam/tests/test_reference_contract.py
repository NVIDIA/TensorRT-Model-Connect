# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from families.sam.tests.test_e2e import (
    _assert_parity,
    _official_masks,
    _official_processor,
    _semantic_masks,
)


def test_official_reference_keeps_the_checkpoint_slow_processor(monkeypatch) -> None:
    calls = []
    expected = object()

    class SamProcessor:
        @staticmethod
        def from_pretrained(model_dir, **kwargs):
            calls.append((model_dir, kwargs))
            return expected

    transformers = ModuleType("transformers")
    transformers.SamProcessor = SamProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model_dir = Path("/checkpoint")
    assert _official_processor(model_dir) is expected
    assert calls == [(model_dir, {"use_fast": False})]


def test_official_reference_preserves_mask_logits() -> None:
    calls = []
    expected = np.array([[-1.0, 1.0]], dtype=np.float32)

    class ImageProcessor:
        @staticmethod
        def post_process_masks(*args, **kwargs):
            calls.append((args, kwargs))
            return [expected]

    tensor = SimpleNamespace(cpu=lambda: tensor)
    processor = SimpleNamespace(image_processor=ImageProcessor())
    outputs = SimpleNamespace(pred_masks=tensor)
    encoded = {"original_sizes": tensor, "reshaped_input_sizes": tensor}

    assert _official_masks(processor, outputs, encoded) is expected
    assert calls == [((tensor, tensor, tensor), {"binarize": False})]


def test_mask_parity_uses_the_sam_logit_boundary() -> None:
    actual = {
        "masks": np.array([-4.0, 2.0, -0.5, 0.25, -2.0, 3.0], dtype=np.float32),
        "iou_scores": [0.1, 0.9, 0.5],
        "num_masks": 3,
    }
    expected = {
        "masks": np.array([-3.0, 1.0, -0.1, 4.0, -1.0, 2.0], dtype=np.float32),
        "iou_scores": [0.2, 0.8, 0.4],
    }

    assert _semantic_masks(actual["masks"]).tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]

    _assert_parity(
        actual,
        expected,
        {"task": "prompted_segmentation"},
        {},
        {
            "iou_per_prompt": 1.0,
            "mask_rank_consistency": 1.0,
            "num_expected_masks": 3,
            "num_masks_consistency": True,
        },
    )


def test_mask_parity_rejects_a_missing_mask() -> None:
    actual = {
        "masks": [-1.0, 1.0],
        "iou_scores": [0.1, 0.9],
        "num_masks": 2,
    }
    expected = {
        "masks": [-1.0, 1.0],
        "iou_scores": [0.2, 0.8, 0.4],
    }

    with pytest.raises(AssertionError):
        _assert_parity(
            actual,
            expected,
            {"task": "prompted_segmentation"},
            {},
            {
                "iou_per_prompt": 0.7,
                "mask_rank_consistency": 0.8,
                "num_expected_masks": 3,
                "num_masks_consistency": True,
            },
        )
