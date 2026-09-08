# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm GhostNet builds."""

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

from families.timm_ghostnet import model  # noqa: E402
from families.timm_ghostnet.checkpoint import Checkpoint  # noqa: E402
from families.timm_ghostnet.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _ghost(tensors: dict[str, np.ndarray], prefix: str, half: int) -> None:
    tensors[f"{prefix}.primary_conv.0.weight"] = _random(half, 8, 1, 1)
    _norm(tensors, f"{prefix}.primary_conv.1", half)
    tensors[f"{prefix}.cheap_operation.0.weight"] = _random(half, 1, 3, 3)
    _norm(tensors, f"{prefix}.cheap_operation.1", half)


def _checkpoint(tmp_path: Path, *, stride: bool = True, se: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "ghostnet_100",
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, np.ndarray] = {}
    tensors["conv_stem.weight"] = _random(8, 3, 3, 3)
    _norm(tensors, "bn1", 8)
    prefix = "blocks.0.0"
    _ghost(tensors, f"{prefix}.ghost1", 4)
    _ghost(tensors, f"{prefix}.ghost2", 4)
    if stride:
        tensors[f"{prefix}.conv_dw.weight"] = _random(8, 1, 3, 3)
        _norm(tensors, f"{prefix}.bn_dw", 8)
        tensors[f"{prefix}.shortcut.0.weight"] = _random(8, 1, 3, 3)
        _norm(tensors, f"{prefix}.shortcut.1", 8)
        tensors[f"{prefix}.shortcut.2.weight"] = _random(8, 8, 1, 1)
        _norm(tensors, f"{prefix}.shortcut.3", 8)
    if se:
        tensors[f"{prefix}.se.conv_reduce.weight"] = _random(2, 8, 1, 1)
        tensors[f"{prefix}.se.conv_reduce.bias"] = _random(2)
        tensors[f"{prefix}.se.conv_expand.weight"] = _random(8, 2, 1, 1)
        tensors[f"{prefix}.se.conv_expand.bias"] = _random(8)
    tensors["blocks.1.0.conv.weight"] = _random(8, 8, 1, 1)
    _norm(tensors, "blocks.1.0.bn1", 8)
    tensors["conv_head.weight"] = _random(8, 8, 1, 1)
    tensors["conv_head.bias"] = _random(8)
    tensors["classifier.weight"] = _random(5, 8)
    tensors["classifier.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("ghostnet_100")) is not None
    assert describe(_metadata("repvgg_a2")) is None


def test_layout_separates_bottlenecks_from_convolution_blocks(tmp_path: Path) -> None:
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path)))
    assert [block["kind"] for block in blocks] == ["bottleneck", "convolution"]


def test_layout_takes_stride_from_the_depthwise_convolution(tmp_path: Path) -> None:
    """A bottleneck reduces exactly when it carries a depthwise convolution."""
    strided = model._layout(Checkpoint.open(_checkpoint(tmp_path / "a", stride=True)))
    plain = model._layout(Checkpoint.open(_checkpoint(tmp_path / "b", stride=False)))
    assert strided[0]["stride"] == 2
    assert plain[0]["stride"] == 1


def test_layout_reads_the_gate_from_the_checkpoint(tmp_path: Path) -> None:
    gated = model._layout(Checkpoint.open(_checkpoint(tmp_path / "a", se=True)))
    plain = model._layout(Checkpoint.open(_checkpoint(tmp_path / "b", se=False)))
    assert gated[0]["has_se"] is True
    assert plain[0]["has_se"] is False


def test_layout_rejects_a_block_of_neither_shape(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors = {"blocks.0.0.mystery.weight": _random(4, 4, 1, 1)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="neither a Ghost bottleneck"):
        model._layout(Checkpoint.open(tmp_path))


def test_fold_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weight, bias = model._fold(checkpoint, "conv_stem.weight", "bn1", np.float32)
    raw = checkpoint.tensor("conv_stem.weight")
    gamma = checkpoint.tensor("bn1.weight")
    beta = checkpoint.tensor("bn1.bias")
    mean = checkpoint.tensor("bn1.running_mean")
    variance = checkpoint.tensor("bn1.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    np.testing.assert_allclose(bias, beta - mean * scale, rtol=1e-6)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "repvgg_a2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm GhostNet model identity"):
        model._read_config(tmp_path)
