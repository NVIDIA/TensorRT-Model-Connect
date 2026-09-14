# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the DETR object-detection family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

try:
    from families.detr.config import ModelConfig
    from families.detr.model import _DetrModel
    from tensorrt_model_connect.model_support import load_model_metadata
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _write_config(tmp_path: Path) -> Path:
    config = {
        "model_type": "detr",
        "architectures": ["DetrForObjectDetection"],
        "num_queries": 100,
        "num_labels": 91,
        "d_model": 256,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "encoder_attention_heads": 8,
        "decoder_attention_heads": 8,
        "encoder_ffn_dim": 2048,
        "decoder_ffn_dim": 2048,
        "position_embedding_type": "sine",
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_model_config_reads_detr_identity(tmp_path: Path):
    _write_config(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "detr"
    assert cfg.architecture == "DetrForObjectDetection"


def test_model_config_requires_model_type(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DetrForObjectDetection"]}))
    with pytest.raises(ValueError, match="model_type"):
        ModelConfig.from_dir(tmp_path)


def test_bundle_config_includes_detr_runtime_fields(tmp_path: Path):
    _write_config(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    model = _DetrModel()

    bundle_config = model.get_bundle_config_overrides(cfg)

    assert bundle_config["input_image_h"] == 800
    assert bundle_config["input_image_w"] == 800
    assert bundle_config["num_queries"] == 100
    assert bundle_config["num_labels"] == 91
    assert bundle_config["image_mean"] == [0.485, 0.456, 0.406]


def test_support_resolves_facebook_detr_model(tmp_path: Path):
    _write_config(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"")
    metadata = load_model_metadata(tmp_path)
    from families.detr.support import describe

    support = describe(metadata)

    assert support is not None
    assert support.tasks == ("object_detection",)
    assert support.default_task == "object_detection"
