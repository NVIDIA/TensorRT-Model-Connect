# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm VGG image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_vgg import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _write_tiny_vgg(
    tmp_path: Path,
    *,
    conv_indices: tuple[int, ...] = (0, 2, 5, 7, 10),
) -> dict[str, np.ndarray]:
    """Write a miniature VGG whose Sequential indices imply the pool positions."""
    classes = 5
    config = {
        "architecture": "vgg16",
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bilinear",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors: dict[str, np.ndarray] = {}
    in_ch = 3
    for index in conv_indices:
        out_ch = 8
        tensors[f"features.{index}.weight"] = _rand(out_ch, in_ch, 3, 3)
        tensors[f"features.{index}.bias"] = _rand(out_ch)
        in_ch = out_ch
    tensors["pre_logits.fc1.weight"] = _rand(16, in_ch, 7, 7)
    tensors["pre_logits.fc1.bias"] = _rand(16)
    tensors["pre_logits.fc2.weight"] = _rand(16, 16, 1, 1)
    tensors["pre_logits.fc2.bias"] = _rand(16)
    tensors["head.fc.weight"] = _rand(classes, 16)
    tensors["head.fc.bias"] = _rand(classes)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tensors


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "vgg16"
    assert cfg.architectures == ["vgg16"]
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize("model_type", ["vgg11", "vgg13", "vgg19", "vgg16_bn", "timm_vgg"])
def test_plugin_matches_vgg_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["resnet50", "vit_base_patch16_224", "llama", ""])
def test_plugin_rejects_unrelated_model_types(model_type: str):
    assert not plugin.matches(model_type)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_vgg_image_classification"
    assert bundle_config["input_image_h"] == 224
    assert bundle_config["num_classes"] == 5
    assert bundle_config["image_mean"] == [0.485, 0.456, 0.406]
    assert bundle_config["crop_pct"] == pytest.approx(0.875)
    assert bundle_config["interpolation"] == "bilinear"


def test_layout_places_a_pool_where_the_sequential_index_jumps(tmp_path: Path):
    """A gap larger than conv+ReLU means torchvision put a MaxPool there."""
    _write_tiny_vgg(tmp_path, conv_indices=(0, 2, 5, 7, 10))
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    layout = cfg.raw["_timm_vgg_config"]

    # 0->2 is conv+ReLU (no pool); 2->5, 5->7 no, 7->10 yes; plus the closing pool.
    assert layout["layers"] == [
        ("conv", 0),
        ("conv", 2),
        ("pool", -1),
        ("conv", 5),
        ("conv", 7),
        ("pool", -1),
        ("conv", 10),
        ("pool", -1),
    ]
    assert layout["num_pools"] == 3


def test_load_weights_rejects_checkpoint_without_convolutions(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    from safetensors.numpy import load_file

    kept = {
        k: v
        for k, v in load_file(str(tmp_path / "model.safetensors")).items()
        if not k.startswith("features.")
    }
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no features"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())


def test_build_engine_rejects_input_not_divisible_by_the_pool_count(tmp_path: Path):
    _write_tiny_vgg(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)
    cfg.raw["_timm_vgg_config"]["image_size_h"] = 225

    with pytest.raises(ValueError, match="divisible by 8"):
        plugin.build_engine(cfg, weights, 0)
