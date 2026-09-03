# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm DenseNet image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_densenet import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(t: dict, prefix: str, ch: int) -> None:
    t[f"{prefix}.weight"] = _rand(ch)
    t[f"{prefix}.bias"] = _rand(ch)
    t[f"{prefix}.running_mean"] = _rand(ch)
    t[f"{prefix}.running_var"] = np.abs(_rand(ch)) + 1.0


def _write_tiny_densenet(
    tmp_path: Path, *, block_layers: tuple[int, ...] = (2, 3, 2, 2)
) -> None:
    """Write a miniature DenseNet whose key names imply the block structure."""
    classes = 5
    growth = 4
    stem = 8
    config = {
        "architecture": "densenet121",
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    t: dict[str, np.ndarray] = {}
    t["features.conv0.weight"] = _rand(stem, 3, 7, 7)
    _bn(t, "features.norm0", stem)

    ch = stem
    for block, count in enumerate(block_layers, start=1):
        for layer in range(1, count + 1):
            p = f"features.denseblock{block}.denselayer{layer}"
            mid = growth * 2
            _bn(t, f"{p}.norm1", ch)
            t[f"{p}.conv1.weight"] = _rand(mid, ch, 1, 1)
            _bn(t, f"{p}.norm2", mid)
            t[f"{p}.conv2.weight"] = _rand(growth, mid, 3, 3)
            ch += growth                      # each layer concatenates its output
        if block < len(block_layers):
            _bn(t, f"features.transition{block}.norm", ch)
            t[f"features.transition{block}.conv.weight"] = _rand(ch // 2, ch, 1, 1)
            ch //= 2

    _bn(t, "features.norm5", ch)
    t["classifier.weight"] = _rand(classes, ch)
    t["classifier.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_densenet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "densenet121"
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize(
    "model_type", ["densenet121", "densenet169", "densenet201", "timm_densenet"]
)
def test_plugin_matches_densenet_variants(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["resnet50", "vgg16", "efficientnet_b0", ""])
def test_plugin_rejects_unrelated_model_types(model_type: str):
    assert not plugin.matches(model_type)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_densenet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_densenet_image_classification"
    assert bundle_config["num_classes"] == 5
    assert bundle_config["interpolation"] == "bicubic"


def test_layout_counts_layers_per_dense_block(tmp_path: Path):
    """The whole structure is derivable; DenseNet needs no architecture table."""
    _write_tiny_densenet(tmp_path, block_layers=(2, 3, 2, 2))
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    layout = cfg.raw["_timm_densenet_config"]

    assert layout["block_layers"] == [2, 3, 2, 2]
    assert layout["num_blocks"] == 4


def test_layout_supports_a_different_block_count(tmp_path: Path):
    _write_tiny_densenet(tmp_path, block_layers=(2, 2, 2))
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)

    assert cfg.raw["_timm_densenet_config"]["block_layers"] == [2, 2, 2]


def test_load_weights_rejects_a_checkpoint_without_dense_blocks(tmp_path: Path):
    _write_tiny_densenet(tmp_path)
    from safetensors.numpy import load_file

    kept = {
        k: v
        for k, v in load_file(str(tmp_path / "model.safetensors")).items()
        if "denseblock" not in k
    }
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no features.denseblock"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write_tiny_densenet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())
