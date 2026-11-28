# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint tensors against the configuration: names, dtypes and shapes are checked before any tensor loads."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.mask2former import weights
from families.mask2former.config import Mask2FormerConfig

# A miniature of the released structure: every width differs, so a check that reads the wrong field fails.
_CONFIG = Mask2FormerConfig(
    image_height=8,
    image_width=8,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
    rescale_factor=1 / 255,
    patch_size=1,
    embed_dim=2,
    depths=(1, 1, 2, 1),
    num_heads=(1, 1, 2, 2),
    window_size=1,
    mlp_ratio=3.0,
    backbone_eps=1e-5,
    feature_size=4,
    mask_feature_size=5,
    hidden_dim=4,
    num_attention_heads=2,
    encoder_layers=2,
    encoder_feedforward_dim=6,
    decoder_layers=3,
    dim_feedforward=7,
    num_queries=2,
    labels=("a", "b"),
)


def _checkpoint(directory: Path, config: Mask2FormerConfig = _CONFIG, **changes) -> Path:
    tensors = {
        name: np.zeros(shape, dtype=np.int64 if dtype == "I64" else np.float32)
        for name, (shape, dtype) in weights.expected_tensors(config).items()
    }
    tensors["criterion.empty_weight"] = np.ones(config.num_labels + 1, dtype=np.float32)
    for name, value in changes.items():
        if value is None:
            del tensors[name]
        else:
            tensors[name] = value
    save_file(tensors, str(directory / "model.safetensors"))
    return directory


def test_the_released_structure_reads_every_checkpoint_tensor() -> None:
    released = dataclasses.replace(
        _CONFIG,
        image_height=384,
        image_width=384,
        patch_size=4,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=12,
        mlp_ratio=4.0,
        feature_size=256,
        mask_feature_size=256,
        hidden_dim=256,
        num_attention_heads=8,
        encoder_layers=6,
        encoder_feedforward_dim=1024,
        decoder_layers=10,
        dim_feedforward=2048,
        num_queries=100,
        labels=tuple(str(index) for index in range(150)),
    )
    # facebook/mask2former-swin-large-ade-semantic holds 782 tensors: these and criterion.empty_weight.
    tensors = weights.expected_tensors(released)
    assert len(tensors) == 781
    assert tensors["class_predictor.weight"] == ((151, 256), "F32")
    assert tensors[
        "model.pixel_level_module.encoder.encoder.layers.2.blocks.17.intermediate.dense.weight"
    ] == ((3072, 768), "F32")
    assert tensors["model.transformer_module.decoder.layers.8.fc1.weight"] == ((2048, 256), "F32")


def test_a_checkpoint_that_matches_its_configuration_loads(tmp_path: Path) -> None:
    loaded = weights.load(_checkpoint(tmp_path), _CONFIG)
    assert set(loaded) == set(weights.expected_tensors(_CONFIG))
    assert loaded["model.transformer_module.decoder.layers.1.fc1.weight"].shape == (7, 4)


@pytest.mark.parametrize(
    ("field", "value", "tensor"),
    [
        ("dim_feedforward", 8, "model.transformer_module.decoder.layers.0.fc1.weight"),
        (
            "encoder_feedforward_dim",
            8,
            "model.pixel_level_module.decoder.encoder.layers.0.fc1.weight",
        ),
        ("mask_feature_size", 4, "model.pixel_level_module.decoder.mask_projection.weight"),
        ("mlp_ratio", 4.0, "intermediate.dense.weight"),
        ("num_queries", 3, "queries_embedder.weight"),
    ],
)
def test_a_configuration_that_disagrees_with_its_weights_is_rejected(
    tmp_path: Path, field: str, value, tensor: str
) -> None:
    _checkpoint(tmp_path)
    with pytest.raises(ValueError, match=f"tensor .*{tensor}.* configuration implies"):
        weights.load(tmp_path, dataclasses.replace(_CONFIG, **{field: value}))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"class_predictor.bias": None}, "1 missing .*class_predictor.bias"),
        (
            {"model.extra.weight": np.zeros(2, dtype=np.float32)},
            "1 unexpected .*model.extra.weight",
        ),
        (
            {
                "model.transformer_module.decoder.layers.2.fc1.weight": np.zeros(
                    (7, 4), dtype=np.float32
                )
            },
            "unexpected .*decoder.layers.2.fc1.weight",
        ),
        (
            {
                "model.pixel_level_module.encoder.encoder.layers.0.blocks.0.attention.self.relative_position_index": (
                    np.zeros((1, 1), dtype=np.float32)
                )
            },
            "relative_position_index is F32 \\[1, 1\\], but the configuration implies I64",
        ),
    ],
)
def test_missing_unexpected_and_mistyped_tensors_are_rejected(
    tmp_path: Path, changes: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        weights.load(_checkpoint(tmp_path, **changes), _CONFIG)
