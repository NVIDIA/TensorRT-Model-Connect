# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm SENet builds."""

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

from families.timm_senet import model  # noqa: E402
from families.timm_senet.checkpoint import Checkpoint  # noqa: E402
from families.timm_senet.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _checkpoint(
    tmp_path: Path,
    depths: tuple[int, ...] = (1, 1, 1, 1),
    bottleneck: bool = True,
    se: bool = True,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "senet154",
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
    tensors["conv1.0.weight"] = _random(8, 3, 3, 3)
    _norm(tensors, "conv1.1", 8)
    tensors["conv1.3.weight"] = _random(8, 8, 3, 3)
    _norm(tensors, "conv1.4", 8)
    tensors["conv1.6.weight"] = _random(8, 8, 3, 3)
    _norm(tensors, "bn1", 8)
    for stage, depth in zip(("layer1", "layer2", "layer3", "layer4"), depths):
        for index in range(depth):
            prefix = f"{stage}.{index}"
            if bottleneck:
                tensors[f"{prefix}.conv1.weight"] = _random(8, 8, 1, 1)
                _norm(tensors, f"{prefix}.bn1", 8)
                tensors[f"{prefix}.conv2.weight"] = _random(8, 8, 3, 3)
                _norm(tensors, f"{prefix}.bn2", 8)
                tensors[f"{prefix}.conv3.weight"] = _random(8, 8, 1, 1)
                _norm(tensors, f"{prefix}.bn3", 8)
            else:
                tensors[f"{prefix}.conv1.weight"] = _random(8, 8, 3, 3)
                _norm(tensors, f"{prefix}.bn1", 8)
                tensors[f"{prefix}.conv2.weight"] = _random(8, 8, 3, 3)
                _norm(tensors, f"{prefix}.bn2", 8)
            if se:
                tensors[f"{prefix}.se.fc1.weight"] = _random(2, 8, 1, 1)
                tensors[f"{prefix}.se.fc1.bias"] = _random(2)
                tensors[f"{prefix}.se.fc2.weight"] = _random(8, 2, 1, 1)
                tensors[f"{prefix}.se.fc2.bias"] = _random(8)
    tensors["fc.weight"] = _random(5, 8)
    tensors["fc.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("senet154")) is not None
    assert describe(_metadata("resnet50")) is None


def test_layout_reads_depth_per_stage(tmp_path: Path) -> None:
    layout = model._layout(Checkpoint.open(_checkpoint(tmp_path, (3, 4, 6, 3))))
    assert layout["depths"] == [3, 4, 6, 3]


def test_layout_detects_the_block_shape_from_a_third_convolution(tmp_path: Path) -> None:
    """A third convolution is the only thing separating the two block shapes."""
    deep = model._layout(Checkpoint.open(_checkpoint(tmp_path / "a", bottleneck=True)))
    shallow = model._layout(Checkpoint.open(_checkpoint(tmp_path / "b", bottleneck=False)))
    assert deep["bottleneck"] is True
    assert shallow["bottleneck"] is False


def test_layout_rejects_a_checkpoint_without_the_gate(tmp_path: Path) -> None:
    """A plain ResNet must not be built silently without its gate."""
    with pytest.raises(ValueError, match="no squeeze-excitation gate"):
        model._layout(Checkpoint.open(_checkpoint(tmp_path, se=False)))


def test_layout_rejects_non_contiguous_block_indices(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for index in (0, 2):
        tensors[f"layer1.{index}.conv1.weight"] = _random(4, 4, 1, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous"):
        model._layout(Checkpoint.open(tmp_path))


def test_weights_pick_stem_convolutions_by_rank(tmp_path: Path) -> None:
    """Norms interleaved in the stem Sequential also carry a `.weight`."""
    weights = model._weights(
        Checkpoint.open(_checkpoint(tmp_path)),
        model._layout(Checkpoint.open(_checkpoint(tmp_path))),
        np.float32,
    )
    assert int(weights["stem_depth"]) == 3
    for position in range(3):
        assert weights[f"stem.{position}.weight"].ndim == 4


def test_fold_reproduces_the_batch_norm_it_replaces(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weight, bias = model._fold_norm(checkpoint, "conv1.6.weight", "bn1", np.float32)
    raw = checkpoint.tensor("conv1.6.weight")
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
    with pytest.raises(ValueError, match="unsupported timm SENet model identity"):
        model._read_config(tmp_path)
