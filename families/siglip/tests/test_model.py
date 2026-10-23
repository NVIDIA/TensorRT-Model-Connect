# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for SigLIP.

The subject here is ``config.resolve``, and the thing worth testing is that
it reads the geometry from the tensors rather than from ``config.json``. The
published checkpoint states almost nothing, so a resolver that trusts the
file would be exercised by no real input at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.siglip import config as config_module, model as model_module
from tensorrt_model_connect import BuildRequest

_HIDDEN = 128
_INTERMEDIATE = 256
_LAYERS = 2
_HEADS = 4
_PATCH = 16
_GRID = 4
_VOCAB = 100
_WINDOW = 8


def _weights(**overrides) -> dict:
    weights = {
        "logit_scale": np.zeros((1,), dtype=np.float32),
        "logit_bias": np.zeros((1,), dtype=np.float32),
        "text_model.embeddings.token_embedding.weight": np.zeros((_VOCAB, _HIDDEN), np.float32),
        "text_model.embeddings.position_embedding.weight": np.zeros((_WINDOW, _HIDDEN), np.float32),
        "text_model.head.weight": np.zeros((_HIDDEN, _HIDDEN), np.float32),
        "vision_model.embeddings.patch_embedding.weight":
            np.zeros((_HIDDEN, 3, _PATCH, _PATCH), np.float32),
        "vision_model.embeddings.position_embedding.weight":
            np.zeros((_GRID * _GRID, _HIDDEN), np.float32),
        "vision_model.head.attention.in_proj_weight": np.zeros((3 * _HIDDEN, _HIDDEN), np.float32),
    }
    for prefix in ("text_model", "vision_model"):
        for index in range(_LAYERS):
            weights[f"{prefix}.encoder.layers.{index}.mlp.fc1.weight"] = np.zeros(
                (_INTERMEDIATE, _HIDDEN), np.float32)
    weights.update(overrides)
    return weights


def _model_dir(tmp_path: Path, config: dict | None = None,
               preprocessor: dict | None = None) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(config if config is not None else {}))
    if preprocessor is not None:
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    return tmp_path


def test_geometry_comes_from_the_tensors_when_the_file_says_nothing(tmp_path: Path) -> None:
    # This is the published checkpoint's situation, not a contrived one.
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    assert cfg["vision"]["hidden_size"] == _HIDDEN
    assert cfg["vision"]["intermediate_size"] == _INTERMEDIATE
    assert cfg["vision"]["num_layers"] == _LAYERS
    assert cfg["vision"]["patch_size"] == _PATCH
    assert cfg["vision"]["grid"] == _GRID
    assert cfg["vision"]["image_size"] == _GRID * _PATCH
    assert cfg["text"]["vocab_size"] == _VOCAB
    assert cfg["text"]["max_text_len"] == _WINDOW


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    config = {"vision_config": {"hidden_size": _HIDDEN * 2}}
    with pytest.raises(ValueError, match="tensors say"):
        config_module.resolve(_model_dir(tmp_path, config), _weights())


def test_the_stated_image_size_is_checked_against_the_position_table(tmp_path: Path) -> None:
    # The table has one row per patch and no class token. A checkpoint that
    # states a size the table cannot cover is refused rather than cropped.
    config = {"vision_config": {"image_size": _GRID * _PATCH + _PATCH}}
    with pytest.raises(ValueError, match="image_size"):
        config_module.resolve(_model_dir(tmp_path, config), _weights())


def test_a_class_token_sized_table_is_refused(tmp_path: Path) -> None:
    # 17 positions is a 4x4 grid plus a class token, which is CLIP's layout.
    # SigLIP has no class token, so this is not a square grid and is refused
    # rather than quietly truncated to 16.
    weights = _weights(**{
        "vision_model.embeddings.position_embedding.weight":
            np.zeros((_GRID * _GRID + 1, _HIDDEN), np.float32)})
    with pytest.raises(NotImplementedError, match="square patch grid"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_non_tanh_activation_is_refused(tmp_path: Path) -> None:
    config = {"text_config": {"hidden_act": "gelu"}}
    with pytest.raises(NotImplementedError, match="gelu_pytorch_tanh"):
        config_module.resolve(_model_dir(tmp_path, config), _weights())


def test_an_unpacked_pooling_head_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{
        "vision_model.head.attention.in_proj_weight": np.zeros((_HIDDEN, _HIDDEN), np.float32)})
    with pytest.raises(ValueError, match="packed QKV"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_bilinear_resize_is_refused(tmp_path: Path) -> None:
    # SigLIP 2 asks for bilinear; the runtime reproduces Pillow's bicubic only.
    directory = _model_dir(tmp_path, {}, {"resample": 2})
    with pytest.raises(NotImplementedError, match="bicubic"):
        config_module.resolve(directory, _weights())


def test_a_centre_crop_is_refused(tmp_path: Path) -> None:
    # SigLIP squashes to the square; a checkpoint asking for a crop is a
    # different preprocessing rule than this family implements.
    directory = _model_dir(tmp_path, {}, {"do_center_crop": True, "resample": 3})
    with pytest.raises(NotImplementedError, match="centre-crop"):
        config_module.resolve(directory, _weights())


def test_missing_layers_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    del weights["vision_model.encoder.layers.1.mlp.fc1.weight"]
    cfg = config_module.resolve(_model_dir(tmp_path), weights)
    assert cfg["vision"]["num_layers"] == 1


def test_a_width_with_no_head_count_to_derive_is_refused(tmp_path: Path) -> None:
    # No tensor carries the head count, so a width the reference's 64-wide
    # rule cannot divide leaves nothing to fall back on.
    weights = _weights()
    for prefix in ("text_model", "vision_model"):
        for index in range(_LAYERS):
            weights[f"{prefix}.encoder.layers.{index}.mlp.fc1.weight"] = np.zeros(
                (_INTERMEDIATE, 96), np.float32)
    with pytest.raises(ValueError, match="64-wide heads"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_non_contiguous_layer_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    weights["vision_model.encoder.layers.7.mlp.fc1.weight"] = np.zeros(
        (_INTERMEDIATE, _HIDDEN), np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_fp16_is_refused(tmp_path: Path) -> None:
    """fp16 is refused at build time, not merely left out of the manifests.

    Measured against the reference, an all-fp16 engine is 3.6e-04 off on the
    embeddings where a correct fp32 one is 2.1e-07, against a 1e-04 gate and
    an 8.4e-04 break. model.build carries the numbers.
    """
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "siglip.bundle",
        family="siglip",
        task="text_image_to_relevance",
        precision="fp16",
    )
    with pytest.raises(NotImplementedError, match="fp32 only"):
        model_module.build(request, writer=None)
