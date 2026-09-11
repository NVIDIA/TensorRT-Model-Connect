# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm RegNet builds."""

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

from families.timm_regnet import model  # noqa: E402
from families.timm_regnet.checkpoint import Checkpoint  # noqa: E402
from families.timm_regnet.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _conv_bn(tensors: dict[str, np.ndarray], prefix: str, out_ch: int, in_ch: int, kernel: int):
    tensors[f"{prefix}.conv.weight"] = _random(out_ch, in_ch, kernel, kernel)
    tensors[f"{prefix}.bn.weight"] = _random(out_ch)
    tensors[f"{prefix}.bn.bias"] = _random(out_ch)
    tensors[f"{prefix}.bn.running_mean"] = _random(out_ch)
    tensors[f"{prefix}.bn.running_var"] = np.abs(_random(out_ch)) + 1.0


def _checkpoint(tmp_path: Path, stages: tuple[int, ...] = (2, 1), se: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "regnety_040",
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
    tensors: dict[str, np.ndarray] = {}
    _conv_bn(tensors, "stem", 8, 3, 3)
    for stage, count in enumerate(stages, start=1):
        for index in range(1, count + 1):
            prefix = f"s{stage}.b{index}"
            _conv_bn(tensors, f"{prefix}.conv1", 8, 8, 1)
            _conv_bn(tensors, f"{prefix}.conv2", 8, 4, 3)
            _conv_bn(tensors, f"{prefix}.conv3", 8, 8, 1)
            if index == 1:
                _conv_bn(tensors, f"{prefix}.downsample", 8, 8, 1)
            if se:
                tensors[f"{prefix}.se.fc1.weight"] = _random(2, 8, 1, 1)
                tensors[f"{prefix}.se.fc1.bias"] = _random(2)
                tensors[f"{prefix}.se.fc2.weight"] = _random(8, 2, 1, 1)
                tensors[f"{prefix}.se.fc2.bias"] = _random(8)
    tensors["head.fc.weight"] = _random(5, 8)
    tensors["head.fc.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("regnety_040")) is not None
    assert describe(_metadata("repvgg_a2")) is None


def test_layout_strides_only_at_the_head_of_each_stage(tmp_path: Path) -> None:
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path, (2, 1))))
    assert [block["prefix"] for block in blocks] == ["s1.b1", "s1.b2", "s2.b1"]
    assert [block["stride"] for block in blocks] == [2, 1, 2]


def test_layout_reads_squeeze_excitation_from_the_checkpoint(tmp_path: Path) -> None:
    """Squeeze-excitation is optional and is detected, never assumed."""
    with_se = model._layout(Checkpoint.open(_checkpoint(tmp_path / "a", se=True)))
    without = model._layout(Checkpoint.open(_checkpoint(tmp_path / "b", se=False)))
    assert all(block["has_se"] for block in with_se)
    assert not any(block["has_se"] for block in without)


def test_layout_rejects_stage_numbering_that_does_not_start_at_one(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {}
    for leaf in ("conv1", "conv2", "conv3"):
        _conv_bn(tensors, f"s2.b1.{leaf}", 4, 4, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous from 1"):
        model._layout(Checkpoint.open(tmp_path))


def test_layout_rejects_a_block_missing_a_bottleneck_convolution(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {}
    _conv_bn(tensors, "s1.b1.conv1", 4, 4, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="missing a bottleneck convolution"):
        model._layout(Checkpoint.open(tmp_path))


def test_fold_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weight, bias = model._fold(checkpoint, "stem", np.float32)
    raw = checkpoint.tensor("stem.conv.weight")
    gamma = checkpoint.tensor("stem.bn.weight")
    beta = checkpoint.tensor("stem.bn.bias")
    mean = checkpoint.tensor("stem.bn.running_mean")
    variance = checkpoint.tensor("stem.bn.running_var")
    scale = gamma / np.sqrt(variance + model._BATCH_NORM_EPSILON)
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    np.testing.assert_allclose(bias, beta - mean * scale, rtol=1e-6)


def test_preprocess_config_rejects_a_degenerate_std() -> None:
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.0, 0.225],
            "crop_pct": 0.95,
            "interpolation": "bicubic",
        },
    }
    with pytest.raises(ValueError, match="preprocessing or classifier config is invalid"):
        model._preprocess_config(raw)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architecture": "repvgg_a2"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported timm RegNet model identity"):
        model._read_config(tmp_path)
