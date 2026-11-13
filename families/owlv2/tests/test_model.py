# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for owlv2: geometry from shapes, and the box bias."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.owlv2 import config as config_module
from families.owlv2 import head_builder

_GRID = 60
_PATCH = 16


def _shapes(layers: int = 12) -> dict:
    shapes = {
        "owlv2.text_model.embeddings.token_embedding.weight": (49408, 512),
        "owlv2.text_model.embeddings.position_embedding.weight": (16, 512),
        "owlv2.vision_model.embeddings.patch_embedding.weight": (768, 3, _PATCH, _PATCH),
        "owlv2.vision_model.embeddings.position_embedding.weight": (_GRID * _GRID + 1, 768),
        "class_head.dense0.weight": (512, 768),
    }
    for index in range(layers):
        shapes[f"owlv2.text_model.encoder.layers.{index}.layer_norm1.weight"] = (512,)
        shapes[f"owlv2.text_model.encoder.layers.{index}.mlp.fc1.weight"] = (2048, 512)
        shapes[f"owlv2.vision_model.encoder.layers.{index}.layer_norm1.weight"] = (768,)
        shapes[f"owlv2.vision_model.encoder.layers.{index}.mlp.fc1.weight"] = (3072, 768)
    return shapes


def _model_dir(tmp_path: Path, config: dict | None = None) -> Path:
    raw = config if config is not None else {
        "model_type": "owlv2",
        "architectures": ["Owlv2ForObjectDetection"],
        "text_config": {"model_type": "owlv2_text_model"},
        "vision_config": {"model_type": "owlv2_vision_model",
                          "image_size": 960, "patch_size": _PATCH},
    }
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return tmp_path


def test_geometry_comes_from_the_tensor_shapes(tmp_path: Path) -> None:
    # This checkpoint's config.json states no width, depth or vocabulary at
    # all, so every one of these came off a tensor rather than out of a field.
    cfg = config_module.resolve(_model_dir(tmp_path), _shapes())
    assert cfg["vision"]["hidden_size"] == 768
    assert cfg["text"]["hidden_size"] == 512
    assert cfg["vision"]["num_layers"] == 12
    assert cfg["text"]["num_layers"] == 12
    assert cfg["text"]["vocab_size"] == 49408
    assert cfg["query_dim"] == 512


def test_prompt_length_is_sixteen_not_seventy_seven(tmp_path: Path) -> None:
    # The text position table is the prompt length. CLIP's is 77; this one is
    # 16, and a phrase is padded to it.
    cfg = config_module.resolve(_model_dir(tmp_path), _shapes())
    assert cfg["text"]["max_position_embeddings"] == 16


def test_patch_grid_is_derived_from_the_position_table(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path), _shapes())
    assert cfg["vision"]["grid"] == _GRID
    assert cfg["vision"]["num_patches"] == _GRID * _GRID
    assert cfg["vision"]["num_positions"] == _GRID * _GRID + 1
    assert cfg["vision"]["image_size"] == _GRID * _PATCH


def test_a_config_that_contradicts_the_tensors_is_refused(tmp_path: Path) -> None:
    config = {
        "model_type": "owlv2",
        "text_config": {},
        "vision_config": {"image_size": 768, "patch_size": _PATCH},
    }
    with pytest.raises(ValueError, match="image_size"):
        config_module.resolve(_model_dir(tmp_path, config), _shapes())


def test_a_gap_in_the_encoder_layers_is_refused(tmp_path: Path) -> None:
    shapes = _shapes()
    del shapes["owlv2.vision_model.encoder.layers.5.layer_norm1.weight"]
    del shapes["owlv2.vision_model.encoder.layers.5.mlp.fc1.weight"]
    with pytest.raises(ValueError, match="contiguous"):
        config_module.resolve(_model_dir(tmp_path), shapes)


def _recovered(coordinate: float) -> float:
    """What a sigmoid gives back from the bias of ``coordinate``.

    The bias is ``log(c + 1e-4) - log1p(-c + 1e-4)``, so a sigmoid returns
    ``(c + 1e-4) / (1 + 2e-4)`` rather than ``c`` itself. At the right and
    bottom edges that is 0.99990, not 1.0, and asserting the round number with
    a 1e-4 tolerance fails by exactly the epsilon.
    """
    return (coordinate + 1e-4) / (1.0 + 2e-4)


def test_box_bias_centres_each_patch_on_its_own_cell() -> None:
    bias = head_builder.box_bias(_GRID)
    assert bias.shape == (_GRID * _GRID, 4)
    centres = 1.0 / (1.0 + np.exp(-bias[:, :2]))
    near, far = _recovered(1.0 / _GRID), _recovered(1.0)
    # Row-major over the grid with x varying fastest: patch 0 sits at the top
    # left, patch 59 at the top right, patch 3599 at the bottom right. Using
    # the other meshgrid order transposes the whole image silently, and the
    # transpose is invisible on the first and last patch alone.
    assert centres[0] == pytest.approx([near, near], abs=1e-6)
    assert centres[_GRID - 1] == pytest.approx([far, near], abs=1e-6)
    assert centres[_GRID] == pytest.approx([near, _recovered(2.0 / _GRID)], abs=1e-6)
    assert centres[-1] == pytest.approx([far, far], abs=1e-6)


def test_box_bias_starts_one_patch_wide() -> None:
    bias = head_builder.box_bias(_GRID)
    sizes = 1.0 / (1.0 + np.exp(-bias[:, 2:]))
    assert np.allclose(sizes, _recovered(1.0 / _GRID), atol=1e-6)
