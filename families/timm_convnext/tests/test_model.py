# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm ConvNeXt builds."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.timm_convnext import model  # noqa: E402
from families.timm_convnext.checkpoint import Checkpoint  # noqa: E402
from families.timm_convnext.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _checkpoint(tmp_path: Path, depths: tuple[int, ...] = (1, 2), gamma: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "convnext_tiny",
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.95,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, np.ndarray] = {
        "stem.0.weight": _random(8, 3, 4, 4),
        "stem.0.bias": _random(8),
        "stem.1.weight": _random(8),
        "stem.1.bias": _random(8),
        "head.norm.weight": _random(8),
        "head.norm.bias": _random(8),
        "head.fc.weight": _random(5, 8),
        "head.fc.bias": _random(5),
    }
    for stage, depth in enumerate(depths):
        if stage > 0:
            tensors[f"stages.{stage}.downsample.0.weight"] = _random(8)
            tensors[f"stages.{stage}.downsample.0.bias"] = _random(8)
            tensors[f"stages.{stage}.downsample.1.weight"] = _random(8, 8, 2, 2)
            tensors[f"stages.{stage}.downsample.1.bias"] = _random(8)
        for index in range(depth):
            prefix = f"stages.{stage}.blocks.{index}"
            tensors[f"{prefix}.conv_dw.weight"] = _random(8, 1, 7, 7)
            tensors[f"{prefix}.conv_dw.bias"] = _random(8)
            tensors[f"{prefix}.norm.weight"] = _random(8)
            tensors[f"{prefix}.norm.bias"] = _random(8)
            tensors[f"{prefix}.mlp.fc1.weight"] = _random(32, 8)
            tensors[f"{prefix}.mlp.fc1.bias"] = _random(32)
            tensors[f"{prefix}.mlp.fc2.weight"] = _random(8, 32)
            tensors[f"{prefix}.mlp.fc2.bias"] = _random(8)
            if gamma:
                tensors[f"{prefix}.gamma"] = _random(8)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("convnext_tiny")) is not None
    assert describe(_metadata("repvgg_a2")) is None


def test_layout_reads_depth_per_stage(tmp_path: Path) -> None:
    assert model._layout(Checkpoint.open(_checkpoint(tmp_path, (3, 3, 9, 3)))) == [3, 3, 9, 3]


def test_layout_rejects_non_contiguous_block_indices(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "stages.0.blocks.0.norm.weight": _random(4),
        "stages.0.blocks.2.norm.weight": _random(4),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous"):
        model._layout(Checkpoint.open(tmp_path))


def test_weights_keep_the_downsample_norm_ahead_of_its_convolution(tmp_path: Path) -> None:
    """ConvNeXt normalises before the strided convolution, not after."""
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weights = model._weights(checkpoint, model._layout(checkpoint), np.float32)
    # The norm parameters are one-dimensional; the reducer is a real kernel.
    assert weights["stages.1.downsample.norm.weight"].ndim == 1
    assert weights["stages.1.downsample.weight"].ndim == 4
    np.testing.assert_allclose(
        weights["stages.1.downsample.norm.weight"],
        checkpoint.tensor("stages.1.downsample.0.weight"),
    )


def test_layer_scale_is_optional(tmp_path: Path) -> None:
    """A checkpoint trained without layer scale must still build."""
    with_gamma = Checkpoint.open(_checkpoint(tmp_path / "a", gamma=True))
    without = Checkpoint.open(_checkpoint(tmp_path / "b", gamma=False))
    assert "stages.0.blocks.0.gamma" in model._weights(
        with_gamma, model._layout(with_gamma), np.float32
    )
    assert "stages.0.blocks.0.gamma" not in model._weights(
        without, model._layout(without), np.float32
    )


def test_layer_norm_uses_the_convnext_epsilon() -> None:
    assert model._LAYER_NORM_EPSILON == 1e-6


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "repvgg_a2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm ConvNeXt model identity"):
        model._read_config(tmp_path)
