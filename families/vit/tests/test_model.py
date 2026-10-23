# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for the transformers ViT.

The subject is ``config.resolve``: what it reads, what it refuses, and what
it declines to invent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.vit import config as config_module, model as model_module
from tensorrt_model_connect import BuildRequest

_HIDDEN = 128
_INTERMEDIATE = 512
_LAYERS = 2
_HEADS = 2
_PATCH = 16
_GRID = 14
_CLASSES = 5


def _weights(prefix: str = "vit.", **overrides) -> dict:
    weights = {
        f"{prefix}embeddings.cls_token": np.zeros((1, 1, _HIDDEN), np.float32),
        f"{prefix}embeddings.position_embeddings":
            np.zeros((1, _GRID * _GRID + 1, _HIDDEN), np.float32),
        f"{prefix}embeddings.patch_embeddings.projection.weight":
            np.zeros((_HIDDEN, 3, _PATCH, _PATCH), np.float32),
        f"{prefix}embeddings.patch_embeddings.projection.bias": np.zeros((_HIDDEN,), np.float32),
        f"{prefix}layernorm.weight": np.zeros((_HIDDEN,), np.float32),
        f"{prefix}layernorm.bias": np.zeros((_HIDDEN,), np.float32),
        "classifier.weight": np.zeros((_CLASSES, _HIDDEN), np.float32),
        "classifier.bias": np.zeros((_CLASSES,), np.float32),
    }
    for index in range(_LAYERS):
        weights[f"{prefix}encoder.layer.{index}.intermediate.dense.weight"] = np.zeros(
            (_INTERMEDIATE, _HIDDEN), np.float32)
    weights.update(overrides)
    return weights


_BASE = {"num_attention_heads": _HEADS}


def _model_dir(tmp_path: Path, config: dict | None = None,
               preprocessor: dict | None = None) -> Path:
    merged = dict(_BASE)
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
    assert cfg["grid"] == _GRID
    assert cfg["image_size"] == _GRID * _PATCH
    assert cfg["num_classes"] == _CLASSES
    assert cfg["prefix"] == "vit."


def test_an_unprefixed_checkpoint_also_resolves(tmp_path: Path) -> None:
    # ViTModel checkpoints carry no "vit." prefix; the classifier ones do.
    cfg = config_module.resolve(_model_dir(tmp_path), _weights(prefix=""))
    assert cfg["prefix"] == ""
    assert cfg["hidden_size"] == _HIDDEN


def test_the_epsilon_is_read_from_the_file(tmp_path: Path) -> None:
    # 1e-12 is this checkpoint's value and nothing like the other ViTs' here.
    cfg = config_module.resolve(_model_dir(tmp_path, {"layer_norm_eps": 1e-12}), _weights())
    assert cfg["eps"] == 1e-12


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tensors say"):
        config_module.resolve(_model_dir(tmp_path, {"hidden_size": _HIDDEN * 2}), _weights())


def test_the_stated_image_size_is_checked_against_the_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image_size"):
        config_module.resolve(_model_dir(tmp_path, {"image_size": 512}), _weights())


def test_a_table_without_a_class_row_is_refused(tmp_path: Path) -> None:
    weights = _weights(**{"vit.embeddings.position_embeddings":
                          np.zeros((1, _GRID * _GRID, _HIDDEN), np.float32)})
    with pytest.raises(NotImplementedError, match="class row"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_a_processed_size_the_table_cannot_cover_is_refused(tmp_path: Path) -> None:
    # The table has one row per patch of the trained grid, so a preprocessor
    # that resizes elsewhere would need it interpolated.
    directory = _model_dir(tmp_path, {}, {"size": 256, "resample": 2})
    with pytest.raises(NotImplementedError, match="position table is sized"):
        config_module.resolve(directory, _weights())


def test_both_spellings_of_size_are_accepted(tmp_path: Path) -> None:
    side = _GRID * _PATCH
    for size in (side, {"height": side, "width": side}):
        directory = _model_dir(tmp_path, {}, {"size": size, "resample": 2})
        cfg = config_module.resolve(directory, _weights())
        assert cfg["image_size"] == side


def test_a_centre_crop_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path, {}, {"do_center_crop": True, "resample": 2})
    with pytest.raises(NotImplementedError, match="squashes"):
        config_module.resolve(directory, _weights())


def test_an_unsupported_resize_filter_is_refused(tmp_path: Path) -> None:
    directory = _model_dir(tmp_path, {}, {"resample": 0})
    with pytest.raises(NotImplementedError, match="bilinear and bicubic"):
        config_module.resolve(directory, _weights())


def test_a_tanh_activation_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="exact gelu"):
        config_module.resolve(_model_dir(tmp_path, {"hidden_act": "gelu_pytorch_tanh"}),
                              _weights())


def test_a_label_set_that_does_not_match_the_classifier_is_refused(tmp_path: Path) -> None:
    labels = {str(i): f"class {i}" for i in range(_CLASSES - 1)}
    with pytest.raises(ValueError, match="labels"):
        config_module.resolve(_model_dir(tmp_path, {"id2label": labels}), _weights())


def test_a_label_set_with_a_hole_is_refused(tmp_path: Path) -> None:
    labels = {str(i): f"class {i}" for i in range(_CLASSES)}
    labels.pop("2")
    labels["99"] = "stray"
    with pytest.raises(ValueError, match="no entry for class"):
        config_module.resolve(_model_dir(tmp_path, {"id2label": labels}), _weights())


def test_labels_are_ordered_by_class_index(tmp_path: Path) -> None:
    labels = {str(i): f"class {i}" for i in range(_CLASSES)}
    cfg = config_module.resolve(_model_dir(tmp_path, {"id2label": labels}), _weights())
    assert cfg["labels"] == [f"class {i}" for i in range(_CLASSES)]


def test_an_unnamed_vocabulary_stays_unnamed(tmp_path: Path) -> None:
    # A thousand classes is not proof of ImageNet-1k. An invented vocabulary
    # name claims the scores are comparable with another model's.
    cfg = config_module.resolve(_model_dir(tmp_path), _weights())
    assert cfg["labels"] == []
    assert cfg["vocabulary_id"] == ""


def test_non_contiguous_layer_indices_are_refused(tmp_path: Path) -> None:
    weights = _weights()
    weights["vit.encoder.layer.7.intermediate.dense.weight"] = np.zeros(
        (_INTERMEDIATE, _HIDDEN), np.float32)
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_model_dir(tmp_path), weights)


def test_fp16_is_refused(tmp_path: Path) -> None:
    """fp16 is refused at build time, not merely left out of the manifests.

    Measured against the reference, an all-fp16 engine is 6.9e-03 off where a
    correct fp32 one is 1.0e-05, and the weakest defect the E2E gate catches
    is 1.9e-02. A bundle built at a precision whose own error is within 3x of
    the break is a bundle nothing meaningfully checks. model.build carries the
    numbers.
    """
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "vit.bundle",
        family="vit",
        task="image_to_class_scores",
        precision="fp16",
    )
    with pytest.raises(NotImplementedError, match="fp32 only"):
        model_module.build(request, writer=None)
