# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for DINOv2: geometry reading and the position table.

The position table is the subject worth most of this file. The checkpoint's
table never matches the grid the engine runs at, so it is resized at build
time, and a resize that is merely close produces features that are merely
plausible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.dinov2 import config as config_module, model as model_module, position
from tensorrt_model_connect import BuildRequest

_HIDDEN = 128
_INTERMEDIATE = 512
_LAYERS = 2
_HEADS = 2
_PATCH = 14
_TRAINED_GRID = 37
_CROP = 224


def _weights(**overrides) -> dict:
    weights = {
        "embeddings.cls_token": np.zeros((1, 1, _HIDDEN), np.float32),
        "embeddings.position_embeddings":
            np.zeros((1, _TRAINED_GRID * _TRAINED_GRID + 1, _HIDDEN), np.float32),
        "embeddings.patch_embeddings.projection.weight":
            np.zeros((_HIDDEN, 3, _PATCH, _PATCH), np.float32),
        "embeddings.patch_embeddings.projection.bias": np.zeros((_HIDDEN,), np.float32),
        "layernorm.weight": np.zeros((_HIDDEN,), np.float32),
        "layernorm.bias": np.zeros((_HIDDEN,), np.float32),
    }
    for index in range(_LAYERS):
        weights[f"encoder.layer.{index}.mlp.fc1.weight"] = np.zeros(
            (_INTERMEDIATE, _HIDDEN), np.float32)
    weights.update(overrides)
    return weights


_BASE_CONFIG = {"num_attention_heads": _HEADS}


def _model_dir(tmp_path: Path, config: dict | None = None,
               preprocessor: dict | None = None) -> Path:
    merged = dict(_BASE_CONFIG)
    merged.update(config or {})
    (tmp_path / "config.json").write_text(json.dumps(merged))
    if preprocessor is not None:
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    return tmp_path


def test_geometry_comes_from_the_tensors(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    assert cfg["hidden_size"] == _HIDDEN
    assert cfg["intermediate_size"] == _INTERMEDIATE
    assert cfg["num_layers"] == _LAYERS
    assert cfg["patch_size"] == _PATCH
    assert cfg["trained_grid"] == _TRAINED_GRID
    # The grid the engine runs at follows from the CROP, not from image_size.
    assert cfg["grid"] == _CROP // _PATCH
    assert cfg["image_size"] == _CROP


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tensors say"):
        config_module.resolve(_model_dir(tmp_path, {"hidden_size": _HIDDEN * 2}), _weights())


def test_an_mlp_ratio_that_disagrees_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mlp_ratio"):
        config_module.resolve(_model_dir(tmp_path, {"mlp_ratio": 3}), _weights())


def test_the_stated_image_size_is_checked_against_the_table(tmp_path: Path) -> None:
    # The table has one row per patch of the trained grid; a stated size that
    # the table cannot cover is a different checkpoint.
    with pytest.raises(ValueError, match="image_size"):
        config_module.resolve(_model_dir(tmp_path, {"image_size": 512}), _weights())


def test_a_table_without_a_class_row_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"embeddings.position_embeddings":
                          np.zeros((1, _TRAINED_GRID * _TRAINED_GRID, _HIDDEN), np.float32)})
    with pytest.raises(NotImplementedError, match="class row"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_the_swiglu_and_register_variants_are_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="SwiGLU"):
        config_module.resolve(_model_dir(tmp_path, {"use_swiglu_ffn": True}), _weights())
    with pytest.raises(NotImplementedError, match="register"):
        config_module.resolve(_model_dir(tmp_path, {"num_register_tokens": 4}), _weights())


def test_a_tanh_activation_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="exact gelu"):
        config_module.resolve(
            _model_dir(tmp_path, {"hidden_act": "gelu_pytorch_tanh"}), _weights())


def test_bilinear_resize_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path, {}, {"resample": 2})
    with pytest.raises(NotImplementedError, match="bicubic"):
        config_module.resolve(directory, _weights())


def test_a_crop_that_is_not_whole_patches_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path, {}, {"resample": 3, "crop_size": {"height": 225,
                                                                      "width": 225}})
    with pytest.raises(NotImplementedError, match="whole number"):
        config_module.resolve(directory, _weights())


def test_non_contiguous_layer_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    weights["encoder.layer.7.mlp.fc1.weight"] = np.zeros((_INTERMEDIATE, _HIDDEN), np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_the_position_table_keeps_its_class_row_and_resizes_the_rest() -> None:
    rows = _TRAINED_GRID * _TRAINED_GRID + 1
    table = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    out = position.interpolate_position_table(table, _TRAINED_GRID, 16)
    assert out.shape == (16 * 16 + 1, 4)
    # The class row is carried across untouched, not resampled with the rest.
    assert np.array_equal(out[0], table[0])


def test_the_position_table_is_left_alone_when_the_grid_already_matches() -> None:
    rows = 4 * 4 + 1
    table = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    out = position.interpolate_position_table(table, 4, 4)
    assert np.array_equal(out, table)


def test_a_table_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="rows"):
        position.interpolate_position_table(np.zeros((10, 4), np.float32), _TRAINED_GRID, 16)


def test_fp16_is_refused(tmp_path: Path) -> None:
    """fp16 is refused at build time, not merely left out of the manifests.

    Measured against the reference, an all-fp16 engine is about 1.0 off on
    features whose peak is 25.6, where a correct fp32 one is 4.5e-04 and the
    weakest defect the E2E gate catches is 6.2e-02. fp16's error is the
    larger of the two, so an fp16 gate would pass a broken build.
    model.build carries the numbers.
    """
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "dinov2.bundle",
        family="dinov2",
        task="image_to_token_and_pooled_features",
        precision="fp16",
    )
    with pytest.raises(NotImplementedError, match="fp32 only"):
        model_module.build(request, writer=None)
