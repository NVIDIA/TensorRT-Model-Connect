# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for DETR builds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.detr import model  # noqa: E402
from families.detr.checkpoint import Checkpoint  # noqa: E402
from families.detr.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


BACKBONE = "model.backbone.conv_encoder.model"


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(11).randn(*shape).astype(np.float32)


def _norm(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _random(channels)
    tensors[f"{prefix}.bias"] = _random(channels)
    tensors[f"{prefix}.running_mean"] = _random(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_random(channels)) + 1.0


def _checkpoint(tmp_path: Path, depths: tuple[int, ...] = (2, 1)) -> Path:
    """A tiny DETR-shaped checkpoint: only the keys the layout reads."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    tensors[f"{BACKBONE}.conv1.weight"] = _random(4, 3, 7, 7)
    _norm(tensors, f"{BACKBONE}.bn1", 4)
    for stage, count in enumerate(depths, start=1):
        for index in range(count):
            prefix = f"{BACKBONE}.layer{stage}.{index}"
            for leaf in ("conv1", "conv2", "conv3"):
                tensors[f"{prefix}.{leaf}.weight"] = _random(4, 4, 1, 1)
                _norm(tensors, f"{prefix}.{leaf.replace('conv', 'bn')}", 4)
            if index == 0:
                tensors[f"{prefix}.downsample.0.weight"] = _random(4, 4, 1, 1)
                _norm(tensors, f"{prefix}.downsample.1", 4)
    for kind, count in (("encoder", 2), ("decoder", 2)):
        for index in range(count):
            tensors[f"model.{kind}.layers.{index}.fc1.weight"] = _random(8, 4)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("detr")) is not None
    assert describe(_metadata("yolov10")) is None


def test_backbone_layout_counts_blocks_per_stage(tmp_path: Path) -> None:
    assert model._backbone_layout(Checkpoint.open(_checkpoint(tmp_path, (2, 1)))) == [2, 1]


def test_backbone_layout_rejects_a_gap_in_the_stages(tmp_path: Path) -> None:
    tensors = {f"{BACKBONE}.layer2.0.conv1.weight": _random(4, 4, 1, 1)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous from 1"):
        model._backbone_layout(Checkpoint.open(tmp_path))


def test_transformer_depth_counts_each_stack(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    assert model._transformer_depth(checkpoint, "model.encoder") == 2
    assert model._transformer_depth(checkpoint, "model.decoder") == 2


def test_transformer_depth_rejects_a_gap_in_the_layers(tmp_path: Path) -> None:
    tensors = {"model.encoder.layers.1.fc1.weight": _random(4, 4)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous from 0"):
        model._transformer_depth(Checkpoint.open(tmp_path), "model.encoder")


def test_frozen_batch_norm_uses_its_own_epsilon() -> None:
    """DetrFrozenBatchNorm2d hard-codes 1e-5; it is never read from the config."""
    assert model._FROZEN_BATCH_NORM_EPSILON == 1e-5


def test_fold_reproduces_the_frozen_norm_it_replaces(tmp_path: Path) -> None:
    checkpoint = Checkpoint.open(_checkpoint(tmp_path))
    weight, bias = model._fold(
        checkpoint, f"{BACKBONE}.conv1.weight", f"{BACKBONE}.bn1", np.float32
    )
    raw = checkpoint.tensor(f"{BACKBONE}.conv1.weight")
    gamma = checkpoint.tensor(f"{BACKBONE}.bn1.weight")
    beta = checkpoint.tensor(f"{BACKBONE}.bn1.bias")
    mean = checkpoint.tensor(f"{BACKBONE}.bn1.running_mean")
    variance = checkpoint.tensor(f"{BACKBONE}.bn1.running_var")
    scale = gamma / np.sqrt(variance + model._FROZEN_BATCH_NORM_EPSILON)
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    np.testing.assert_allclose(bias, beta - mean * scale, rtol=1e-6)


def test_position_embedding_has_one_row_per_feature_pixel() -> None:
    embedding = model.sine_position_embedding(4, 6, 8)
    assert embedding.shape == (1, 24, 8)
    assert np.all(np.abs(embedding) <= 1.0)


def test_position_embedding_puts_the_row_lanes_first() -> None:
    """Transposing a square grid must swap the two halves of every vector.

    The row half is built from the y coordinate and the column half from x, so
    this fails if the two are concatenated the other way round.
    """
    size, channels = 5, 8
    grid = model.sine_position_embedding(size, size, channels).reshape(size, size, channels)
    half = channels // 2
    rows, columns = grid[..., :half], grid[..., half:]
    np.testing.assert_allclose(rows, columns.transpose(1, 0, 2), rtol=1e-6, atol=1e-6)


def test_position_embedding_rejects_an_odd_channel_count() -> None:
    with pytest.raises(ValueError, match="even channel count"):
        model.sine_position_embedding(2, 2, 7)


def test_read_config_rejects_another_family(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "yolov10"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported DETR model identity"):
        model._read_config(tmp_path)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dilation": True}, "dilated backbones"),
        ({"position_embedding_type": "learned"}, "sine position embedding"),
        ({"activation_function": "gelu"}, "relu activation"),
    ],
)
def test_read_config_rejects_variants_the_builder_does_not_cover(
    tmp_path: Path, override: dict, message: str
) -> None:
    raw = {
        "model_type": "detr",
        "architectures": ["DetrForObjectDetection"],
        **override,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(NotImplementedError, match=message):
        model._read_config(tmp_path)


def test_preprocess_config_requires_a_size_the_backbone_can_halve(tmp_path: Path) -> None:
    raw = {"trtmc_input_height": 800, "trtmc_input_width": 801}
    with pytest.raises(ValueError, match="must be divisible by 32"):
        model._preprocess_config(raw, tmp_path)


def test_preprocess_config_rejects_a_degenerate_std(tmp_path: Path) -> None:
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.0, 0.5]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="three non-zero values"):
        model._preprocess_config({}, tmp_path)
