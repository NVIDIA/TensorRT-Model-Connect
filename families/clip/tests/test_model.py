# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-side unit tests for clip: config reading and the request contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.clip import config as config_module

_MINIMAL = {
    "model_type": "clip",
    "architectures": ["CLIPModel"],
    "projection_dim": 512,
    "text_config": {
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "intermediate_size": 2048,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-05,
        "vocab_size": 49408,
        "max_position_embeddings": 77,
    },
    "vision_config": {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-05,
        "image_size": 224,
        "patch_size": 32,
    },
}


def _model_dir(tmp_path: Path, **overrides) -> Path:
    raw = json.loads(json.dumps(_MINIMAL))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key].update(value)
        else:
            raw[key] = value
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    return tmp_path


def test_resolve_keeps_the_two_tower_widths_apart(tmp_path: Path) -> None:
    # The towers disagree on width - 768 vision, 512 text - and both project to
    # 512. Reading one width for both builds an engine that loads and returns
    # the wrong vector.
    cfg = config_module.resolve(_model_dir(tmp_path))
    assert cfg["vision"]["hidden_size"] == 768
    assert cfg["text"]["hidden_size"] == 512
    assert cfg["projection_dim"] == 512
    assert cfg["vision"]["head_dim"] == 64
    assert cfg["text"]["head_dim"] == 64


def test_position_table_covers_the_class_token(tmp_path: Path) -> None:
    # 224 / 32 gives a 7x7 grid; the table has 50 rows, not 49, because the
    # class token is prepended.
    cfg = config_module.resolve(_model_dir(tmp_path))
    assert cfg["vision"]["num_positions"] == 50


def test_image_statistics_come_from_the_checkpoint(tmp_path: Path) -> None:
    model_dir = _model_dir(tmp_path)
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps({"image_mean": [0.1, 0.2, 0.3], "image_std": [0.4, 0.5, 0.6]}),
        encoding="utf-8",
    )
    cfg = config_module.resolve(model_dir)
    assert cfg["image_mean"] == [0.1, 0.2, 0.3]
    assert cfg["image_std"] == [0.4, 0.5, 0.6]


def test_missing_preprocessor_falls_back_to_the_published_statistics(tmp_path: Path) -> None:
    cfg = config_module.resolve(_model_dir(tmp_path))
    assert cfg["image_mean"][0] == pytest.approx(0.48145466)
    assert cfg["image_std"][0] == pytest.approx(0.26862954)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vision_config": {"hidden_act": "gelu"}},
        {"text_config": {"hidden_act": "gelu"}},
    ],
)
def test_other_activations_are_refused(tmp_path: Path, overrides) -> None:
    # QuickGELU and the exact GELU agree closely enough to pass a careless
    # eyeball and not closely enough to keep the logits.
    with pytest.raises(NotImplementedError):
        config_module.resolve(_model_dir(tmp_path, **overrides))


def test_indivisible_width_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        config_module.resolve(_model_dir(tmp_path, vision_config={"num_attention_heads": 7}))


def test_patch_grid_must_divide_the_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        config_module.resolve(_model_dir(tmp_path, vision_config={"image_size": 225}))


def test_missing_section_names_itself(tmp_path: Path) -> None:
    raw = json.loads(json.dumps(_MINIMAL))
    del raw["vision_config"]
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(KeyError, match="vision_config"):
        config_module.resolve(tmp_path)
